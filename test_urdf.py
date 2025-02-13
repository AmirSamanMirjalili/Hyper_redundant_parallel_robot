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
    generated: Dict[str, Dict]  # Generated URDF properties

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
        if any(pattern in generated_name for pattern in ['bottom', 'cylinder', 'rod', 'top']):
            # Extract the base name without the stage suffix
            match = re.match(r'([A-Za-z]+\d+)(\d+)', generated_name)
            if match:
                return match.group(1)  # Return the base name with original number
        
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
        """Verify connection between two links in the kinematic chain."""
        parent_name = self.name_manager.get_component_name(parent_base)
        child_name = self.name_manager.get_component_name(child_base)
        
        # Find the joint connecting these links
        joint = None
        for j in self.generated_root.findall(f".//joint[@type='{joint_type}']"):
            if (j.find('parent').get('link') == parent_name and 
                j.find('child').get('link') == child_name):
                joint = j
                break
        
        assert joint is not None, \
            f"Should find a {joint_type} joint connecting {parent_name} to {child_name}"
        
        # Verify the joint has all required elements
        assert joint.find('origin') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have origin"
        assert joint.find('axis') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have axis"
        assert joint.find('limit') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have limits"

@pytest.fixture(scope="module")
def urdf_data() -> URDFTestData:
    """Fixture providing properties from both original and generated URDF."""
    # Extract properties from original URDF
    original_revolute_props = extract_joint_properties('Stewart.urdf', "Revolute_")
    original_slider_props = extract_joint_properties('Stewart.urdf', "Slider_")
    original_joint_props = {**original_revolute_props, **original_slider_props}
    original_link_props = extract_link_properties('Stewart.urdf')
    
    # Generate our URDF
    generated_urdf = generate_stewart_platform(stage=1)
    generated_root = ET.fromstring(generated_urdf)
    
    # Extract properties from generated URDF
    generated_revolute_props = extract_joint_properties(generated_urdf, "Revolute_")
    generated_slider_props = extract_joint_properties(generated_urdf, "Slider_")
    generated_joint_props = {**generated_revolute_props, **generated_slider_props}
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
        """Test the structure of the base link."""
        base_link_name = test_fixture.name_manager.get_base_link_name()
        test_fixture.verify_link_properties(base_link_name)

    def test_base_connections(self, test_fixture: URDFTestFixture):
        """Test if base link connections match the original structure."""
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

    def test_no_cycles(self, test_fixture: URDFTestFixture):
        """Test that the joint-link structure contains no cycles."""
        def find_cycle(start_link: str, graph: Dict[str, Set[str]], visited: Set[str], path: Set[str]) -> bool:
            if start_link in path:
                return True
            if start_link in visited:
                return False
                
            visited.add(start_link)
            path.add(start_link)
            
            for next_link in graph.get(start_link, set()):
                if find_cycle(next_link, graph, visited, path):
                    return True
                    
            path.remove(start_link)
            return False
        
        # Build connection graph from joint properties
        graph: Dict[str, Set[str]] = {}
        for joint in test_fixture.generated_joints.values():
            if joint.parent not in graph:
                graph[joint.parent] = set()
            graph[joint.parent].add(joint.child)
        
        # Check for cycles from each link
        visited: Set[str] = set()
        path: Set[str] = set()
        
        for link in graph.keys():
            assert not find_cycle(link, graph, visited, path), \
                f"Found cycle in joint-link structure starting from {link}"

class TestPyBulletIntegration:
    """Tests for PyBullet integration."""
    @pytest.fixture(scope="function")
    def pybullet_client(self):
        """Fixture providing a PyBullet client."""
        client = p.connect(p.DIRECT)  # Headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        yield client
        p.disconnect(client)

    def test_pybullet_loading(self, pybullet_client, test_fixture: URDFTestFixture):
        """Test if URDF loads correctly in PyBullet."""
        test_urdf_path = "test_stewart.urdf"
        save_urdf(test_fixture.urdf_data.generated['urdf'], test_urdf_path)
        
        try:
            robot_id = p.loadURDF(test_urdf_path, flags=p.URDF_USE_INERTIA_FROM_FILE)
            assert robot_id > -1, "URDF should load successfully"
            
            # Get number of joints
            num_joints = p.getNumJoints(robot_id)
            expected_joints = len([j for j in test_fixture.generated_joints.values() 
                                 if j.joint_type in ['revolute', 'prismatic']])
            assert num_joints == expected_joints, \
                f"Should have {expected_joints} movable joints (revolute + prismatic)"
            
            # Test joint properties
            for i in range(num_joints):
                joint_info = p.getJointInfo(robot_id, i)
                joint_name = joint_info[1].decode('utf-8')
                
                # Get corresponding joint properties
                joint_props = test_fixture.generated_joints.get(joint_name)
                assert joint_props is not None, f"Joint {joint_name} should exist in properties"
                
                # Get original joint name for comparison
                orig_name = test_fixture.map_to_original_name(joint_name)
                
                # Compare joint type
                joint_type = joint_info[2]
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