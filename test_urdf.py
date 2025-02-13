import pytest
import xml.etree.ElementTree as ET
import os
from generate_urdf import generate_stewart_platform, save_urdf, NameManager
from extract_urdf_properties import extract_joint_properties, extract_link_properties, JointProperties, LinkProperties
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Set, Optional
import re
from dataclasses import dataclass

@dataclass
class URDFTestData:
    """Container for URDF test data."""
    original: Dict[str, Dict]  # Original URDF properties
    generated: Dict[str, Dict]  # Generated URDF properties and data

class URDFTestFixture:
    """Base class for URDF test fixtures."""
    def __init__(self, urdf_data: URDFTestData, name_manager: NameManager):
        self.urdf_data = urdf_data
        self.name_manager = name_manager
        self.original_joints = urdf_data.original['joints']
        self.original_links = urdf_data.original['links']
        self.generated_joints = urdf_data.generated['joints']
        self.generated_links = urdf_data.generated['links']
        self.generated_root = urdf_data.generated['root']
    
    def map_to_original_name(self, generated_name: str) -> str:
        """Map a generated name back to its original name for property comparison."""
        # Handle special case for base link
        if generated_name.startswith('base_link'):
            return 'base_link'
        
        # For components that already have a number suffix in original URDF
        if any(pattern in generated_name for pattern in ['bottom', 'cylinder', 'rod', 'top', 'UJ']):
            # Extract the base name without the stage suffix
            match = re.match(r'([A-Za-z]+\d+)(\d+)', generated_name)
            if match:
                return match.group(1)  # Return the base name with original number
        
        # For J*T and J*B links, they already have the correct number in their name
        if any(pattern in generated_name for pattern in ['J1T1', 'J2T1', 'J3T_1', 'J4T1', 'J5T_1', 'J6T1',
                                                       'J1B_1', 'J2B1', 'J3B1', 'J4B1', 'J5B1', 'J6B1']):
            # Remove the stage suffix
            return generated_name[:-1]
        
        # For joints and other components
        if generated_name.endswith('1'):  # Stage 1
            return generated_name[:-1]
        
        return generated_name

    def verify_joint_properties(self, joint: ET.Element, expected_props: Dict, joint_name: Optional[str] = None):
        """Verify joint properties match expected values."""
        if joint_name is None:
            joint_name = joint.get('name')
        
        # Skip joints without a type (transmission joints)
        if joint.get('type') is None:
            return
        
        # Check type
        assert joint.get('type') == expected_props['type'], \
            f"Joint {joint_name} type mismatch: {joint.get('type')} != {expected_props['type']}"
    
        # Check origin
        origin = joint.find('origin')
        assert origin is not None, f"Joint {joint_name} missing origin"
        assert origin.get('xyz') == expected_props['origin']['xyz'], \
            f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {expected_props['origin']['xyz']}"
        assert origin.get('rpy') == expected_props['origin']['rpy'], \
            f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {expected_props['origin']['rpy']}"
    
        # Check axis
        axis = joint.find('axis')
        assert axis is not None, f"Joint {joint_name} missing axis"
        assert axis.get('xyz') == expected_props['axis'], \
            f"Joint {joint_name} axis mismatch: {axis.get('xyz')} != {expected_props['axis']}"
    
        # Check limits if expected
        if 'limits' in expected_props:
            limit = joint.find('limit')
            assert limit is not None, f"Joint {joint_name} missing limits"
            for prop in ['upper', 'lower', 'effort', 'velocity']:
                assert limit.get(prop) == expected_props['limits'][prop], \
                    f"Joint {joint_name} {prop} limit mismatch: {limit.get(prop)} != {expected_props['limits'][prop]}"

    def verify_link_properties(self, link_name: str):
        """Verify link properties match between original and generated URDF."""
        orig_name = self.map_to_original_name(link_name)
        assert orig_name in self.original_links, f"Original link {orig_name} should exist"
        
        gen_link = self.generated_links[link_name]
        orig_link = self.original_links[orig_name]
        
        # Compare inertial properties
        for prop in ['mass', 'ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
            assert float(orig_link.inertial[prop]) == float(gen_link.inertial[prop]), \
                f"Link {link_name} {prop} should match"
        
        # Compare visual and collision mesh references
        assert gen_link.visual['mesh'] == orig_link.visual['mesh'], \
            f"Link {link_name} visual mesh should match"
        assert gen_link.visual['scale'] == orig_link.visual['scale'], \
            f"Link {link_name} visual scale should match"
        
        # Compare origin coordinates
        orig_xyz = tuple(map(float, orig_link.visual['origin']['xyz'].split()))
        gen_xyz = tuple(map(float, gen_link.visual['origin']['xyz'].split()))
        assert gen_xyz == orig_xyz, \
            f"Link {link_name} visual origin should match"

class KinematicChainFixture(URDFTestFixture):
    """Fixture for testing kinematic chain properties."""
    def verify_chain_connection(self, parent_base: str, child_base: str, joint_type: str = 'revolute'):
        """Verify connection between two links in the kinematic chain.
        
        Args:
            parent_base: Base name of the parent link
            child_base: Base name of the child link
            joint_type: Type of joint ('revolute', 'prismatic', 'fixed', 'continuous')
        """
        parent_name = self.name_manager.get_component_name(parent_base)
        child_name = self.name_manager.get_component_name(child_base)
        
        # Find the joint connecting these links
        joint = None
        joint_types_to_check = []
        
        if joint_type == 'fixed':
            # For fixed joints, just look for fixed type
            joint_types_to_check = ['fixed']
        else:
            # For movable joints, check both specified type and continuous
            joint_types_to_check = [joint_type, 'continuous']
        
        # Search for joints of allowed types
        for joint_type_to_check in joint_types_to_check:
            for j in self.generated_root.findall(f".//joint[@type='{joint_type_to_check}']"):
                if (j.find('parent').get('link') == parent_name and 
                    j.find('child').get('link') == child_name):
                    joint = j
                    break
            if joint is not None:
                break
        
        assert joint is not None, \
            f"Should find a {' or '.join(joint_types_to_check)} joint connecting {parent_name} to {child_name}"
        
        # Verify the joint has all required elements
        assert joint.find('origin') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have origin"
        
        # Only check for axis and limits on movable joints
        if joint_type not in ['fixed']:
            assert joint.find('axis') is not None, \
                f"Joint connecting {parent_name} to {child_name} should have axis"
            
            # Only check for limits if the joint is not continuous
            if joint.get('type') != 'continuous':
                assert joint.find('limit') is not None, \
                    f"Joint connecting {parent_name} to {child_name} should have limits"

@dataclass
class ChainConfigs:
    """Configuration data for kinematic chain tests."""
    # First leg chain (through X1)
    first_leg_chain = [
        ("base_link", "X1bottom1", "revolute"),
        ("X1bottom1", "cylinder11", "revolute"),
        ("cylinder11", "rod11", "prismatic"),
        ("rod11", "piston11", "revolute"),
        ("piston11", "X1top1", "revolute"),
        ("X1top1", "UJ11", "revolute"),
        ("UJ11", "J1B_1", "fixed"),
        ("J6T1", "TOP1", "fixed"),  # J6T1 connects to TOP1 first
        ("TOP1", "J1T1", "fixed"),  # Then TOP1 connects to J1T1
        ("TOP1", "indicator1", "fixed")
    ]
    
    # J6 chain (through X6)
    j6_chain = [
        ("base_link", "X6bottom1", "revolute"),
        ("X6bottom1", "cylinder61", "revolute"),
        ("cylinder61", "rod61", "prismatic"),
        ("rod61", "piston61", "revolute"),
        ("piston61", "X6top1", "revolute"),
        ("X6top1", "UJ61", "revolute"),
        ("UJ61", "J6B1", "fixed"),
        ("J6B1", "J6T1", "fixed"),
        ("J6T1", "TOP1", "fixed")
    ]
    
    # Universal joint connections
    universal_joint_connections = [
        ("X1top1", "UJ11"),
        ("X6top1", "UJ61"),
        ("X5top1", "UJ51"),
        ("X4top1", "UJ41"),
        ("X3top1", "UJ31"),
        ("X2top1", "UJ21")
    ]
    
    # J*T connections to TOP1
    jt_connections = [
        ("TOP1", "J1T1"),
        ("TOP1", "J2T1"),
        ("TOP1", "J3T_1"),  # Note: J3T_1 has underscore
        ("TOP1", "J4T1"),
        ("TOP1", "J5T_1")   # Note: J5T_1 has underscore
    ]

    # Rigid joint configurations
    rigid_joint_configs = [
        # (joint_num, parent, child)
        (59, "UJ11", "J1B_1"),
        (60, "UJ21", "J2B1"),
        (61, "UJ31", "J3B1"),
        (62, "UJ41", "J4B1"),
        (63, "UJ51", "J5B1"),
        (64, "UJ61", "J6B1"),
        (65, "J6B1", "J6T1"),
        (66, "J6T1", "TOP1"),
        (67, "TOP1", "J1T1"),
        (68, "TOP1", "J2T1"),
        (69, "TOP1", "J3T_1"),
        (70, "TOP1", "J4T1"),
        (71, "TOP1", "J5T_1"),
        (77, "TOP1", "indicator1")
    ]

    @staticmethod
    def get_rigid_joint_name(joint_num: int, stage: int) -> str:
        """Get the name of a rigid joint with stage suffix."""
        return f"Rigid_{joint_num}{stage}"

    @staticmethod
    def get_original_rigid_joint_name(joint_num: int) -> str:
        """Get the original name of a rigid joint without stage suffix."""
        return f"Rigid_{joint_num}"

class TestKinematicChain:
    """Tests for complete kinematic chain."""
    def test_no_floating_links(self, test_fixture: URDFTestFixture):
        """Test that all links (except base) are connected by joints."""
        # Get all connected links
        connected_links = set()
        for joint in test_fixture.generated_joints.values():
            connected_links.add(joint.parent)
            connected_links.add(joint.child)
        
        # Base link should be the only unconnected link
        all_links = set(test_fixture.generated_links.keys())
        unconnected = all_links - connected_links
        assert len(unconnected) <= 1, "Only base_link should be unconnected"
        if unconnected:
            assert list(unconnected)[0] == 'base_link1', \
                "The only unconnected link should be base_link1"

    def test_universal_joint_chain(self, kinematic_chain_fixture: KinematicChainFixture):
        """Test the kinematic chain from top links through universal joints."""
        # Test connections from top links to universal joints
        for parent_base, child_base in ChainConfigs.universal_joint_connections:
            kinematic_chain_fixture.verify_chain_connection(parent_base, child_base)

    def test_complete_chain(self, kinematic_chain_fixture: KinematicChainFixture):
        """Test the complete kinematic chain from base to indicator."""
        # Test chain from base to indicator through first leg (X1)
        for parent_base, child_base, joint_type in ChainConfigs.first_leg_chain:
            kinematic_chain_fixture.verify_chain_connection(parent_base, child_base, joint_type)

        # Test chain through J6T1 to TOP1
        for parent_base, child_base, joint_type in ChainConfigs.j6_chain:
            kinematic_chain_fixture.verify_chain_connection(parent_base, child_base, joint_type)

        # Test connections from TOP1 to all J*T links
        for parent_base, child_base in ChainConfigs.jt_connections:
            kinematic_chain_fixture.verify_chain_connection(parent_base, child_base, "fixed")

@pytest.fixture(scope="module")
def urdf_data() -> URDFTestData:
    """Fixture providing properties from both original and generated URDF."""
    # Extract properties from original URDF
    original_revolute_props = extract_joint_properties('Stewart.urdf', "Revolute_")
    original_slider_props = extract_joint_properties('Stewart.urdf', "Slider_")
    original_rigid_props = extract_joint_properties('Stewart.urdf', "Rigid_")
    original_joint_props = {
        **original_revolute_props,
        **original_slider_props,
        **original_rigid_props
    }
    original_link_props = extract_link_properties('Stewart.urdf')
    
    # Generate our URDF
    generated_urdf = generate_stewart_platform(stage=1)
    generated_root = ET.fromstring(generated_urdf)
    
    # Extract properties from generated URDF
    generated_revolute_props = extract_joint_properties(generated_urdf, "Revolute_")
    generated_slider_props = extract_joint_properties(generated_urdf, "Slider_")
    generated_rigid_props = extract_joint_properties(generated_urdf, "Rigid_")
    generated_joint_props = {
        **generated_revolute_props,
        **generated_slider_props,
        **generated_rigid_props
    }
    generated_link_props = extract_link_properties(generated_urdf)
    
    return URDFTestData(
        original={
            'joints': original_joint_props,
            'links': original_link_props
        },
        generated={
            'joints': generated_joint_props,
            'links': generated_link_props,
            'root': generated_root,
            'urdf': generated_urdf
        }
    )

@pytest.fixture(scope="module")
def name_manager() -> NameManager:
    """Fixture providing a NameManager instance for stage 1."""
    return NameManager(stage=1)

@pytest.fixture(scope="module")
def test_fixture(urdf_data: URDFTestData, name_manager: NameManager) -> URDFTestFixture:
    """Fixture providing the base test fixture."""
    return URDFTestFixture(urdf_data, name_manager)

@pytest.fixture(scope="module")
def kinematic_chain_fixture(urdf_data: URDFTestData, name_manager: NameManager) -> KinematicChainFixture:
    """Fixture providing the kinematic chain test fixture."""
    return KinematicChainFixture(urdf_data, name_manager)

class TestBasicProperties:
    """Tests for basic URDF properties."""
    def test_robot_name(self, urdf_data: URDFTestData):
        """Test if robot name structure is correct."""
        generated_root = urdf_data.generated['root']
        assert generated_root.get('name') == "Stewart_1", "Robot name should be 'Stewart_1' for stage 1"

    def test_material_properties(self, urdf_data: URDFTestData):
        """Test if material properties match."""
        generated_root = urdf_data.generated['root']
        material = generated_root.find('material')
        color = material.find('color').get('rgba')
        assert color == "0.700 0.700 0.700 1.000", "Material color should match"

    def test_stage_naming(self):
        """Test if stage-based naming is working correctly."""
        stage2_urdf = generate_stewart_platform(stage=2, base_prefix="upper_")
        root = ET.fromstring(stage2_urdf)
        base_link = root.find(f".//link[@name='upper_base_link2']")
        assert base_link is not None, "Stage 2 base link should have correct name"

class TestBaseLink:
    """Tests for base link properties."""
    def test_base_link_structure(self, test_fixture: URDFTestFixture):
        """
        Test the structure of the base link.

        Example:
            >>> base_link_name = test_fixture.name_manager.get_base_link_name()
            >>> test_fixture.verify_link_properties(base_link_name)
        """
        base_link_name = test_fixture.name_manager.get_base_link_name()
        test_fixture.verify_link_properties(base_link_name)

    def test_base_connections(self, test_fixture: URDFTestFixture):
        """
        Test if base link connections match the original structure.

        Example:
            >>> # Verify connections from base link match expected original connections
            >>> actual_connections = {(j, p) for j, p in [(joint_name, joint_props.child) 
            ...             for joint_name, joint_props in test_fixture.generated_joints.items() 
            ...             if joint_props.parent == test_fixture.name_manager.get_base_link_name()]}
            >>> expected_connections = {...}  # Expected set of connections
            >>> actual_connections == expected_connections
            True
        """
        base_link_name = test_fixture.name_manager.get_base_link_name()

        # Get actual connections from joint properties
        actual_connections = set()
        for joint_name, joint_props in test_fixture.generated_joints.items():
            if joint_props.parent == base_link_name:
                actual_connections.add((joint_name, joint_props.child))

        # Get expected connections
        expected_base_connections = set()
        for joint_num, bottom_link in [
            (1, "X1bottom1"),  # Use exact names from mesh files
            (2, "X6bottom1"),
            (3, "X5bottom1"),
            (4, "X2bottom1"),
            (5, "X4bottom1"),
            (6, "X3bottom1")
        ]:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            link_name = test_fixture.name_manager.get_component_name(bottom_link)
            expected_base_connections.add((joint_name, link_name))

        assert actual_connections == expected_base_connections, \
            f"""Base link connections mismatch:
            Generated: {sorted(actual_connections)}
            Expected: {sorted(expected_base_connections)}
            Missing: {sorted(expected_base_connections - actual_connections)}
            Extra: {sorted(actual_connections - expected_base_connections)}"""

class TestCylinderLinks:
    """Tests for cylinder links and their connections."""
    def test_cylinder_links(self, test_fixture: URDFTestFixture):
        """Test if cylinder links are correctly generated."""
        cylinder_configs = [
            ("cylinder61", "X6bottom1"),
            ("cylinder51", "X5bottom1"),
            ("cylinder11", "X1bottom1"),
            ("cylinder21", "X2bottom1"),
            ("cylinder31", "X3bottom1"),
            ("cylinder41", "X4bottom1")
        ]
        
        for base_name, _ in cylinder_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_cylinder_joints(self, test_fixture: URDFTestFixture):
        """Test if cylinder joints are correctly connected."""
        joint_configs = [
            (7, "X6bottom1", "cylinder61"),
            (8, "X5bottom1", "cylinder51"),
            (9, "X1bottom1", "cylinder11"),
            (10, "X2bottom1", "cylinder21"),
            (11, "X3bottom1", "cylinder31"),
            (12, "X4bottom1", "cylinder41")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            joint = test_fixture.generated_joints[joint_name]
            orig_joint = test_fixture.original_joints[f"Revolute_{joint_num}"]
            
            assert joint.parent == parent_name, \
                f"Joint {joint_name} should have parent {parent_name}"
            assert joint.child == child_name, \
                f"Joint {joint_name} should have child {child_name}"
            assert joint.joint_type == orig_joint.joint_type, \
                f"Joint {joint_name} should be {orig_joint.joint_type}"
            assert joint.axis == orig_joint.axis, \
                f"Joint {joint_name} should have correct axis"
        
class TestRodLinks:
    """Tests for rod links and their connections."""
    def test_rod_links(self, test_fixture: URDFTestFixture):
        """Test if rod links have correct properties."""
        rod_names = ["rod11", "rod21", "rod31", "rod41", "rod51", "rod61"]
        for base_name in rod_names:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_rod_joints(self, test_fixture: URDFTestFixture):
        """Test if rod joints are correctly connected."""
        joint_configs = [
            (13, "cylinder11", "rod11"),
            (14, "cylinder21", "rod21"),
            (15, "cylinder31", "rod31"),
            (16, "cylinder41", "rod41"),
            (17, "cylinder51", "rod51"),
            (18, "cylinder61", "rod61")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            joint = test_fixture.generated_joints[joint_name]
            orig_joint = test_fixture.original_joints[f"Slider_{joint_num}"]
            
            assert joint.parent == parent_name, \
                f"Joint {joint_name} should have parent {parent_name}"
            assert joint.child == child_name, \
                f"Joint {joint_name} should have child {child_name}"
            assert joint.joint_type == orig_joint.joint_type, \
                f"Joint {joint_name} should be {orig_joint.joint_type}"
            assert joint.axis == orig_joint.axis, \
                f"Joint {joint_name} should have correct axis"
        
class TestPistonLinks:
    """Tests for piston links and their connections."""
    def test_piston_links(self, test_fixture: URDFTestFixture):
        """Test if piston links are correctly generated."""
        piston_configs = [
            "piston11", "piston21", "piston31",
            "piston41", "piston51", "piston61"
        ]
        for base_name in piston_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_piston_joints(self, test_fixture: URDFTestFixture):
        """Test if piston joints are correctly connected."""
        joint_configs = [
            (19, "rod11", "piston11"),
            (20, "rod21", "piston21"),
            (21, "rod31", "piston31"),
            (22, "rod61", "piston61"),
            (23, "rod51", "piston51"),
            (24, "rod41", "piston41")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            joint = test_fixture.generated_joints[joint_name]
            orig_joint = test_fixture.original_joints[f"Revolute_{joint_num}"]
            
            assert joint.parent == parent_name, \
                f"Joint {joint_name} should have parent {parent_name}"
            assert joint.child == child_name, \
                f"Joint {joint_name} should have child {child_name}"
            assert joint.joint_type == orig_joint.joint_type, \
                f"Joint {joint_name} should be {orig_joint.joint_type}"
            assert joint.axis == orig_joint.axis, \
                f"Joint {joint_name} should have correct axis"
        
class TestTopLinks:
    """Tests for top links and their connections."""
    def test_top_links(self, test_fixture: URDFTestFixture):
        """Test if top links are correctly generated."""
        top_configs = [
            "X1top1", "X2top1", "X3top1",
            "X4top1", "X5top1", "X6top1"
        ]
        for base_name in top_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_top_joints(self, test_fixture: URDFTestFixture):
        """Test if top joints are correctly connected."""
        joint_configs = [
            (26, "piston11", "X1top1"),
            (35, "piston21", "X2top1"),
            (33, "piston31", "X3top1"),
            (27, "piston41", "X4top1"),
            (28, "piston51", "X5top1"),
            (25, "piston61", "X6top1")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            joint = test_fixture.generated_joints[joint_name]
            orig_joint = test_fixture.original_joints[f"Revolute_{joint_num}"]
            
            assert joint.parent == parent_name, \
                f"Joint {joint_name} should have parent {parent_name}"
            assert joint.child == child_name, \
                f"Joint {joint_name} should have child {child_name}"
            assert joint.joint_type == orig_joint.joint_type, \
                f"Joint {joint_name} should be {orig_joint.joint_type}"
            assert joint.axis == orig_joint.axis, \
                f"Joint {joint_name} should have correct axis"

class TestUniversalJoints:
    """Tests for universal joints and their connections."""
    def test_universal_joint_links(self, test_fixture: URDFTestFixture):
        """Test if universal joint links are correctly generated."""
        uj_configs = [
            "UJ11", "UJ21", "UJ31",
            "UJ41", "UJ51", "UJ61"
        ]
        for base_name in uj_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_universal_joint_connections(self, test_fixture: URDFTestFixture):
        """Test if universal joint connections are correctly configured."""
        joint_configs = [
            (29, "X1top1", "UJ11"),
            (30, "X6top1", "UJ61"),
            (31, "X5top1", "UJ51"),
            (32, "X4top1", "UJ41"),
            (34, "X3top1", "UJ31"),
            (36, "X2top1", "UJ21")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            joint_name = test_fixture.name_manager.get_joint_name(joint_num)
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            joint = test_fixture.generated_joints[joint_name]
            orig_joint = test_fixture.original_joints[f"Revolute_{joint_num}"]
            
            assert joint.parent == parent_name, \
                f"Joint {joint_name} should have parent {parent_name}"
            assert joint.child == child_name, \
                f"Joint {joint_name} should have child {child_name}"
            assert joint.joint_type == orig_joint.joint_type, \
                f"Joint {joint_name} should be {orig_joint.joint_type}"
            assert joint.axis == orig_joint.axis, \
                f"Joint {joint_name} should have correct axis"
        
class TestJBLinks:
    """Tests for J*B links and their rigid connections."""
    def test_jb_links(self, test_fixture: URDFTestFixture):
        """Test if J*B links are correctly generated."""
        jb_configs = [
            "J1B_1", "J2B1", "J3B1",
            "J4B1", "J5B1", "J6B1"
        ]
        for base_name in jb_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_jb_rigid_connections(self, test_fixture: URDFTestFixture):
        """Test if J*B links are correctly connected to universal joints with rigid joints."""
        joint_configs = [
            (59, "UJ11", "J1B_1"),
            (60, "UJ21", "J2B1"),
            (61, "UJ31", "J3B1"),
            (62, "UJ41", "J4B1"),
            (63, "UJ51", "J5B1"),
            (64, "UJ61", "J6B1")
        ]
        
        for joint_num, parent_base, child_base in joint_configs:
            # Generated joint name includes stage suffix
            joint_name = f"Rigid_{joint_num}{test_fixture.name_manager.stage}"
            # Original joint name doesn't have stage suffix
            orig_joint_name = f"Rigid_{joint_num}"
            
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            # Find the joint in the XML
            joint = None
            for j in test_fixture.generated_root.findall(".//joint[@type='fixed']"):
                if (j.get('name') == joint_name and
                    j.find('parent').get('link') == parent_name and
                    j.find('child').get('link') == child_name):
                    joint = j
                    break
            
            assert joint is not None, \
                f"Should find rigid joint {joint_name} connecting {parent_name} to {child_name}"
            
            # Verify joint properties using original joint name
            orig_props = test_fixture.original_joints[orig_joint_name]
            
            # Check origin
            origin = joint.find('origin')
            assert origin is not None, f"Joint {joint_name} missing origin"
            assert origin.get('xyz') == orig_props.origin['xyz'], \
                f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {orig_props.origin['xyz']}"
            assert origin.get('rpy') == orig_props.origin['rpy'], \
                f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {orig_props.origin['rpy']}"

class TestJTLinks:
    """Tests for J*T links and their rigid connections."""
    def test_jt_links(self, test_fixture: URDFTestFixture):
        """Test if J*T links are correctly generated."""
        jt_configs = [
            "J6T1",  # First one connects to J6B1
            "J1T1", "J2T1", "J3T_1",  # Note: J3T_1 has underscore
            "J4T1", "J5T_1"  # Note: J5T_1 has underscore
        ]
        for base_name in jt_configs:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_jt_rigid_connections(self, test_fixture: URDFTestFixture):
        """Test if J*T links are correctly connected with rigid joints."""
        # First test J6T1 connection to J6B1
        j6t_config = (65, "J6B1", "J6T1")
        joint_name = f"Rigid_{j6t_config[0]}{test_fixture.name_manager.stage}"
        orig_joint_name = f"Rigid_{j6t_config[0]}"
        parent_name = test_fixture.name_manager.get_component_name(j6t_config[1])
        child_name = test_fixture.name_manager.get_component_name(j6t_config[2])
        
        # Find the joint in the XML
        joint = None
        for j in test_fixture.generated_root.findall(".//joint[@type='fixed']"):
            if (j.get('name') == joint_name and
                j.find('parent').get('link') == parent_name and
                j.find('child').get('link') == child_name):
                joint = j
                break
        
        assert joint is not None, \
            f"Should find rigid joint {joint_name} connecting {parent_name} to {child_name}"
        
        # Verify joint properties using original joint name
        orig_props = test_fixture.original_joints[orig_joint_name]
        
        # Check origin
        origin = joint.find('origin')
        assert origin is not None, f"Joint {joint_name} missing origin"
        assert origin.get('xyz') == orig_props.origin['xyz'], \
            f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {orig_props.origin['xyz']}"
        assert origin.get('rpy') == orig_props.origin['rpy'], \
            f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {orig_props.origin['rpy']}"

        # Test connections from TOP1 to other J*T links
        jt_configs = [
            (67, "TOP1", "J1T1"),
            (68, "TOP1", "J2T1"),
            (69, "TOP1", "J3T_1"),  # Note: J3T_1 has underscore
            (70, "TOP1", "J4T1"),
            (71, "TOP1", "J5T_1")   # Note: J5T_1 has underscore
        ]
        
        for joint_num, parent_base, child_base in jt_configs:
            joint_name = f"Rigid_{joint_num}{test_fixture.name_manager.stage}"
            orig_joint_name = f"Rigid_{joint_num}"
            parent_name = test_fixture.name_manager.get_component_name(parent_base)
            child_name = test_fixture.name_manager.get_component_name(child_base)
            
            # Find the joint in the XML
            joint = None
            for j in test_fixture.generated_root.findall(".//joint[@type='fixed']"):
                if (j.get('name') == joint_name and
                    j.find('parent').get('link') == parent_name and
                    j.find('child').get('link') == child_name):
                    joint = j
                    break
            
            assert joint is not None, \
                f"Should find rigid joint {joint_name} connecting {parent_name} to {child_name}"
            
            # Verify joint properties using original joint name
            orig_props = test_fixture.original_joints[orig_joint_name]
            
            # Check origin
            origin = joint.find('origin')
            assert origin is not None, f"Joint {joint_name} missing origin"
            assert origin.get('xyz') == orig_props.origin['xyz'], \
                f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {orig_props.origin['xyz']}"
            assert origin.get('rpy') == orig_props.origin['rpy'], \
                f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {orig_props.origin['rpy']}"

class TestTopPlatform:
    """Tests for TOP1 and indicator links."""
    def test_top_platform_links(self, test_fixture: URDFTestFixture):
        """Test if TOP1 and indicator links are correctly generated."""
        for base_name in ["TOP1", "indicator1"]:
            test_fixture.verify_link_properties(test_fixture.name_manager.get_component_name(base_name))

    def test_top_platform_connections(self, test_fixture: URDFTestFixture):
        """Test if TOP1 and indicator are correctly connected."""
        # Test connection from J6T1 to TOP1
        j6t_top_config = (66, "J6T1", "TOP1")
        joint_name = f"Rigid_{j6t_top_config[0]}{test_fixture.name_manager.stage}"
        orig_joint_name = f"Rigid_{j6t_top_config[0]}"
        parent_name = test_fixture.name_manager.get_component_name(j6t_top_config[1])
        child_name = test_fixture.name_manager.get_component_name(j6t_top_config[2])
        
        # Find the joint in the XML
        joint = None
        for j in test_fixture.generated_root.findall(".//joint[@type='fixed']"):
            if (j.get('name') == joint_name and
                j.find('parent').get('link') == parent_name and
                j.find('child').get('link') == child_name):
                joint = j
                break
        
        assert joint is not None, \
            f"Should find rigid joint {joint_name} connecting {parent_name} to {child_name}"
        
        # Verify joint properties using original joint name
        orig_props = test_fixture.original_joints[orig_joint_name]
        
        # Check origin
        origin = joint.find('origin')
        assert origin is not None, f"Joint {joint_name} missing origin"
        assert origin.get('xyz') == orig_props.origin['xyz'], \
            f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {orig_props.origin['xyz']}"
        assert origin.get('rpy') == orig_props.origin['rpy'], \
            f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {orig_props.origin['rpy']}"

        # Test connection from TOP1 to indicator1
        indicator_config = (77, "TOP1", "indicator1")
        joint_name = f"Rigid_{indicator_config[0]}{test_fixture.name_manager.stage}"
        orig_joint_name = f"Rigid_{indicator_config[0]}"
        parent_name = test_fixture.name_manager.get_component_name(indicator_config[1])
        child_name = test_fixture.name_manager.get_component_name(indicator_config[2])
        
        # Find the joint in the XML
        joint = None
        for j in test_fixture.generated_root.findall(".//joint[@type='fixed']"):
            if (j.get('name') == joint_name and
                j.find('parent').get('link') == parent_name and
                j.find('child').get('link') == child_name):
                joint = j
                break
        
        assert joint is not None, \
            f"Should find rigid joint {joint_name} connecting {parent_name} to {child_name}"
        
        # Verify joint properties using original joint name
        orig_props = test_fixture.original_joints[orig_joint_name]
        
        # Check origin
        origin = joint.find('origin')
        assert origin is not None, f"Joint {joint_name} missing origin"
        assert origin.get('xyz') == orig_props.origin['xyz'], \
            f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {orig_props.origin['xyz']}"
        assert origin.get('rpy') == orig_props.origin['rpy'], \
            f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {orig_props.origin['rpy']}"

class TestPyBulletIntegration:
    """Tests for PyBullet integration."""
    @pytest.fixture(scope="function")
    def pybullet_client(self):
        """
        Fixture providing a PyBullet client.

        Example:
            >>> client = p.connect(p.DIRECT)
            >>> isinstance(client, int)
            True
        """
        client = p.connect(p.DIRECT)  # Headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        yield client
        p.disconnect(client)

    def test_pybullet_loading(self, pybullet_client, test_fixture: URDFTestFixture):
        """
        Test if URDF loads correctly in PyBullet.

        Example:
            >>> test_urdf_path = "test_stewart.urdf"
            >>> save_urdf(test_fixture.urdf_data.generated['urdf'], test_urdf_path)
            >>> robot_id = p.loadURDF(test_urdf_path, flags=p.URDF_USE_INERTIA_FROM_FILE)
            >>> robot_id > -1
            True
        """
        test_urdf_path = "test_stewart.urdf"
        save_urdf(test_fixture.urdf_data.generated['urdf'], test_urdf_path)
    
        try:
            robot_id = p.loadURDF(test_urdf_path, flags=p.URDF_USE_INERTIA_FROM_FILE)
            assert robot_id > -1, "URDF should load successfully"
        
            # Count only movable joints (revolute and prismatic)
            movable_joints = 0
            num_joints = p.getNumJoints(robot_id)
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                joint_type = joint_info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    movable_joints += 1
            
            expected_joints = 36  # Based on Link_graph.txt: 30 revolute + 6 prismatic joints
            assert movable_joints == expected_joints, \
                f"Should have {expected_joints} movable joints (30 revolute + 6 prismatic), but found {movable_joints}"
        
            # Test joint properties
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                joint_name = joint_info[1].decode('utf-8')
                joint_type = joint_info[2]
                
                # Skip fixed joints
                if joint_type == p.JOINT_FIXED:
                    continue
            
                # Get corresponding joint properties
                joint_props = test_fixture.generated_joints.get(joint_name)
                assert joint_props is not None, f"Joint {joint_name} should exist in properties"
            
                # Get original joint name for comparison
                orig_name = test_fixture.map_to_original_name(joint_name)
            
                # Compare joint type
                if joint_props.joint_type == 'revolute':
                    assert joint_type == p.JOINT_REVOLUTE, \
                        f"Joint {joint_name} should be revolute"
                elif joint_props.joint_type == 'prismatic':
                    assert joint_type == p.JOINT_PRISMATIC, \
                        f"Joint {joint_name} should be prismatic"
            
                # Compare joint limits
                if joint_props.limits:
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    assert abs(float(joint_props.limits['lower']) - lower_limit) < 1e-6, \
                        f"Joint {joint_name} lower limit should match"
                    assert abs(float(joint_props.limits['upper']) - upper_limit) < 1e-6, \
                        f"Joint {joint_name} upper limit should match"
    
        finally:
            if os.path.exists(test_urdf_path):
                os.remove(test_urdf_path)