import pytest
import xml.etree.ElementTree as ET
import os
from generate_urdf import generate_stewart_platform, save_urdf
from extract_urdf_properties import extract_joint_properties, extract_link_properties, JointProperties, LinkProperties
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Set
import re
from generate_urdf import NameManager

@pytest.fixture(scope="module")
def urdf_properties():
    """Fixture providing properties from both original and generated URDF"""
    # Extract properties from original URDF
    original_revolute_props = extract_joint_properties('Stewart.urdf', "Revolute_")
    original_slider_props = extract_joint_properties('Stewart.urdf', "Slider_")
    original_joint_props = {**original_revolute_props, **original_slider_props}
    original_link_props = extract_link_properties('Stewart.urdf')
    
    print("\nOriginal Link Properties:")
    print(f"Number of links: {len(original_link_props)}")
    print(f"Link names: {sorted(original_link_props.keys())}")
    
    # Generate our URDF
    generated_urdf = generate_stewart_platform(stage=1)
    
    # Parse the generated URDF directly
    generated_root = ET.fromstring(generated_urdf)
    
    # Extract properties directly from the XML string
    generated_revolute_props = extract_joint_properties(generated_urdf, "Revolute_")
    generated_slider_props = extract_joint_properties(generated_urdf, "Slider_")
    generated_joint_props = {**generated_revolute_props, **generated_slider_props}
    generated_link_props = extract_link_properties(generated_urdf)
    
    print("\nGenerated Link Properties:")
    print(f"Number of links: {len(generated_link_props)}")
    print(f"Link names: {sorted(generated_link_props.keys())}")
    
    # Check for missing links
    original_cylinders = [name for name in original_link_props.keys() if 'cylinder' in name]
    generated_cylinders = [name for name in generated_link_props.keys() if 'cylinder' in name]
    print("\nCylinder Links:")
    print(f"Original cylinders: {sorted(original_cylinders)}")
    print(f"Generated cylinders: {sorted(generated_cylinders)}")
    
    return {
        'original': {
            'joints': original_joint_props,
            'links': original_link_props
        },
        'generated': {
            'joints': generated_joint_props,
            'links': generated_link_props,
            'root': generated_root,
            'urdf': generated_urdf
        }
    }

@pytest.fixture(scope="module")
def name_manager():
    """Fixture providing a NameManager instance for stage 1"""
    return NameManager(stage=1)

def test_robot_name(urdf_properties):
    """Test if robot name structure is correct"""
    generated_root = urdf_properties['generated']['root']
    assert generated_root.get('name') == "Stewart_1", "Robot name should be 'Stewart_1' for stage 1"

def test_material_properties(urdf_properties):
    """Test if material properties match"""
    generated_root = urdf_properties['generated']['root']
    material = generated_root.find('material')
    color = material.find('color').get('rgba')
    assert color == "0.700 0.700 0.700 1.000", "Material color should match"

def test_base_link_structure(urdf_properties, name_manager):
    """Test the structure of the base link"""
    original_props = urdf_properties['original']['links']
    generated_props = urdf_properties['generated']['links']
    
    # Get base link properties using name manager
    base_link_name = name_manager.get_component_name('base_link')
    original_base = original_props.get('base_link')
    generated_base = generated_props.get(base_link_name)
    
    assert generated_base is not None, "Base link should exist"
    assert original_base is not None, "Original base link should exist"
    
    # Compare inertial properties
    for prop in ['mass', 'ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
        assert float(original_base.inertial[prop]) == float(generated_base.inertial[prop]), \
            f"Base link {prop} should match"
    
    # Compare visual and collision mesh references
    assert generated_base.visual['mesh'] == name_manager.get_mesh_filename('base_link'), \
        "Base link should reference correct mesh file"
    assert generated_base.visual['scale'] == "0.001 0.001 0.001", \
        "Mesh scale should be correct"

def test_stage_naming():
    """Test if stage-based naming is working correctly"""
    # Generate URDFs for different stages
    stage2_urdf = generate_stewart_platform(stage=2, base_prefix="upper_")
    
    # Parse the URDF string
    root = ET.fromstring(stage2_urdf)
    
    # Find the base link directly
    base_link = root.find(f".//link[@name='upper_base_link2']")
    assert base_link is not None, "Stage 2 base link should have correct name"

def test_joint_properties(urdf_properties):
    """Test if joint properties match between original and generated URDF"""
    original_joints = urdf_properties['original']['joints']
    generated_joints = urdf_properties['generated']['joints']
    
    # For each joint in the original URDF
    for orig_name, orig_props in original_joints.items():
        # Get the corresponding generated joint name (e.g., Revolute_2 -> Revolute_21)
        gen_name = f"{orig_name}1"  # Add stage number
        
        assert gen_name in generated_joints, f"Generated joint {gen_name} should exist"
        gen_props = generated_joints[gen_name]
        
        # Compare properties
        assert gen_props.joint_type == orig_props.joint_type, \
            f"Joint {gen_name} type should match"
        assert gen_props.axis == orig_props.axis, \
            f"Joint {gen_name} axis should match"
        
        # Compare origin coordinates
        orig_xyz = tuple(map(float, orig_props.origin['xyz'].split()))
        gen_xyz = tuple(map(float, gen_props.origin['xyz'].split()))
        assert gen_xyz == orig_xyz, \
            f"Joint {gen_name} origin should match"
        
        # Compare limits if they exist
        if orig_props.limits:
            assert gen_props.limits is not None, f"Joint {gen_name} should have limits"
            for limit_type in ['upper', 'lower', 'effort', 'velocity']:
                assert float(gen_props.limits[limit_type]) == float(orig_props.limits[limit_type]), \
                    f"Joint {gen_name} {limit_type} limit should match"

def map_to_original_name(generated_name: str, name_manager: NameManager) -> str:
    """Map a generated name back to its original name for property comparison.
    
    Examples:
        base_link1 -> base_link
        X1bottom11 -> X1bottom1  (stage 1 suffix removed)
        cylinder611 -> cylinder61 (stage 1 suffix removed)
        rod611 -> rod61 (stage 1 suffix removed)
        Revolute_11 -> Revolute_1
    """
    # Handle special case for base link
    if generated_name.startswith('base_link'):
        return 'base_link'
    
    # For components that already have a number suffix in original URDF
    if any(pattern in generated_name for pattern in ['bottom', 'cylinder', 'rod']):
        # Extract the base name without the stage suffix
        match = re.match(r'([A-Za-z]+\d+)(\d+)', generated_name)
        if match:
            return match.group(1)  # Return the base name with original number
    
    # For joints and other components
    if generated_name.endswith('1'):  # Stage 1
        return generated_name[:-1]
    
    return generated_name

def test_link_properties(urdf_properties):
    """Test if link properties match between original and generated URDF"""
    original_props = urdf_properties['original']['links']
    generated_props = urdf_properties['generated']['links']
    
    # For each link in the generated URDF
    for gen_name, gen_props in generated_props.items():
        # Get the corresponding original link name
        orig_name = map_to_original_name(gen_name, name_manager)
        
        assert orig_name in original_props, f"Original link {orig_name} should exist"
        orig_props = original_props[orig_name]
        
        # Compare inertial properties
        for prop in ['mass', 'ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
            assert float(orig_props.inertial[prop]) == float(gen_props.inertial[prop]), \
                f"Link {gen_name} {prop} should match"
        
        # Compare visual and collision mesh references
        # Note: Mesh filenames should use original names
        assert gen_props.visual['mesh'] == orig_props.visual['mesh'], \
            f"Link {gen_name} visual mesh should match"
        assert gen_props.visual['scale'] == orig_props.visual['scale'], \
            f"Link {gen_name} visual scale should match"
        
        # Compare origin coordinates
        orig_xyz = tuple(map(float, orig_props.visual['origin']['xyz'].split()))
        gen_xyz = tuple(map(float, gen_props.visual['origin']['xyz'].split()))
        assert gen_xyz == orig_xyz, \
            f"Link {gen_name} visual origin should match"

def test_base_connections(urdf_properties, link_graph_data, name_manager):
    """Test if base link connections match the original structure"""
    generated_joints = urdf_properties['generated']['joints']
    original_joints = urdf_properties['original']['joints']

    def verify_base_connections(stage_num):
        """Verify base connections for a specific stage"""
        base_link_name = name_manager.get_base_link_name()

        # Get actual connections from joint properties
        actual_connections = set()
        for joint_name, joint_props in generated_joints.items():
            if joint_props.parent == base_link_name:
                actual_connections.add((joint_name, joint_props.child))

        # Get expected connections from link graph and adjust for stage
        expected_base_connections = set()
        for joint_num, bottom_link in [
            (1, "X1bottom1"),  # Use exact names from mesh files
            (2, "X6bottom1"),
            (3, "X5bottom1"),
            (4, "X2bottom1"),
            (5, "X4bottom1"),
            (6, "X3bottom1")
        ]:
            joint_name = name_manager.get_joint_name(joint_num)
            link_name = name_manager.get_component_name(bottom_link)
            expected_base_connections.add((joint_name, link_name))

        # Compare connections
        assert actual_connections == expected_base_connections, \
            f"""Base link connections mismatch for stage {stage_num}:
            Generated: {sorted(actual_connections)}
            Expected: {sorted(expected_base_connections)}
            Missing: {sorted(expected_base_connections - actual_connections)}
            Extra: {sorted(actual_connections - expected_base_connections)}"""

        return actual_connections

    # Verify base connections for the current stage
    base_connections = verify_base_connections(name_manager.stage)
    
    # Verify properties of each base joint
    for joint_name, joint_props in generated_joints.items():
        # Get original joint name for comparison
        orig_name = map_to_original_name(joint_name, name_manager)
        if orig_name in original_joints:
            orig_props = original_joints[orig_name]
            
            # Compare properties
            assert joint_props.joint_type == orig_props.joint_type, \
                f"Joint {joint_name} type should match"
            assert joint_props.axis == orig_props.axis, \
                f"Joint {joint_name} axis should match"
            
            # Compare limits if they exist
            if orig_props.limits:
                assert joint_props.limits is not None, \
                    f"Joint {joint_name} should have limits"
                for prop in ['upper', 'lower', 'effort', 'velocity']:
                    assert float(joint_props.limits[prop]) == float(orig_props.limits[prop]), \
                        f"Joint {joint_name} {prop} limit should match"

@pytest.fixture(scope="function")
def pybullet_client():
    """Fixture providing a PyBullet client"""
    client = p.connect(p.DIRECT)  # Headless mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    yield client
    p.disconnect(client)

def test_pybullet_loading(pybullet_client, urdf_properties, name_manager):
    """Test if URDF loads correctly in PyBullet"""
    generated_urdf = urdf_properties['generated']['urdf']
    generated_joints = urdf_properties['generated']['joints']
    test_urdf_path = "test_stewart.urdf"
    save_urdf(generated_urdf, test_urdf_path)
    
    try:
        robot_id = p.loadURDF(test_urdf_path, flags=p.URDF_USE_INERTIA_FROM_FILE)
        assert robot_id > -1, "URDF should load successfully"
        
        # Get number of joints
        num_joints = p.getNumJoints(robot_id)
        expected_joints = len([j for j in generated_joints.values() 
                             if j.joint_type in ['revolute', 'prismatic']])
        assert num_joints == expected_joints, \
            f"Should have {expected_joints} movable joints (revolute + prismatic)"
        
        # Test joint properties
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            # Get corresponding joint properties
            joint_props = generated_joints.get(joint_name)
            assert joint_props is not None, f"Joint {joint_name} should exist in properties"
            
            # Get original joint name for comparison
            orig_name = map_to_original_name(joint_name, name_manager)
            
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

def test_transmission_properties(urdf_properties):
    """Test if transmission properties are correctly set"""
    generated_root = urdf_properties['generated']['root']
    generated_joints = urdf_properties['generated']['joints']
    
    # Check transmissions for all revolute joints
    for joint_name, joint_props in generated_joints.items():
        if joint_props.joint_type == 'revolute':
            # Find corresponding transmission
            transmission = generated_root.find(f".//transmission/joint[@name='{joint_name}']")
            assert transmission is not None, f"Transmission for joint {joint_name} should exist"
            
            # Check hardware interface
            interface = transmission.find('hardwareInterface')
            assert interface is not None, "Transmission should have hardware interface"
            assert interface.text == "PositionJointInterface", \
                "Should use PositionJointInterface"
            
            # Check actuator
            actuator = transmission.getparent().find(f"actuator[@name='{joint_name}_actr']")
            assert actuator is not None, f"Actuator for joint {joint_name} should exist"
            assert actuator.find('mechanicalReduction').text == "1", \
                "Mechanical reduction should be 1"

# Tests to be implemented later
@pytest.mark.skip(reason="Not implemented yet")
def test_link_properties():
    """Will implement when more links are added"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_joint_properties():
    """Will implement when more joints are added"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_transmission_properties():
    """Will implement when transmissions are added"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_link_joint_hierarchy():
    """Will implement when hierarchy is complete"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_kinematic_chain():
    """Will implement when chain is complete"""
    pass

@pytest.fixture(scope="module")
def link_graph_data():
    """Fixture providing parsed data from Link_graph.txt"""
    connections: Dict[str, List[Tuple[str, str]]] = {}
    
    with open('Link_graph.txt', 'r') as f:
        for line in f:
            # Parse lines like: "base_link" -> "X6bottom1" [ label = "Revolute_2" ]
            match = re.match(r'\s*"([^"]+)"\s*->\s*"([^"]+)"\s*\[\s*label\s*=\s*"([^"]+)"\s*\]', line)
            if match:
                parent, child, joint = match.groups()
                if parent not in connections:
                    connections[parent] = []
                connections[parent].append((joint, child))
    
    return connections

def verify_joint_properties(joint, expected_properties, joint_name=None):
    """
    Verify joint properties match expected values.
    
    Args:
        joint: XML element representing the joint
        expected_properties: Dictionary of expected joint properties
        joint_name: Optional joint name for error messages (defaults to joint's name attribute)
    """
    if joint_name is None:
        joint_name = joint.get('name')
        
    # Skip joints without a type (transmission joints)
    if joint.get('type') is None:
        return
        
    # Check type
    assert joint.get('type') == expected_properties['type'], \
        f"Joint {joint_name} type mismatch: {joint.get('type')} != {expected_properties['type']}"
    
    # Check origin
    origin = joint.find('origin')
    assert origin is not None, f"Joint {joint_name} missing origin"
    assert origin.get('xyz') == expected_properties['origin']['xyz'], \
        f"Joint {joint_name} origin xyz mismatch: {origin.get('xyz')} != {expected_properties['origin']['xyz']}"
    assert origin.get('rpy') == expected_properties['origin']['rpy'], \
        f"Joint {joint_name} origin rpy mismatch: {origin.get('rpy')} != {expected_properties['origin']['rpy']}"
    
    # Check axis
    axis = joint.find('axis')
    assert axis is not None, f"Joint {joint_name} missing axis"
    assert axis.get('xyz') == expected_properties['axis'], \
        f"Joint {joint_name} axis mismatch: {axis.get('xyz')} != {expected_properties['axis']}"
    
    # Check limits if expected
    if 'limits' in expected_properties:
        limit = joint.find('limit')
        assert limit is not None, f"Joint {joint_name} missing limits"
        for prop in ['upper', 'lower', 'effort', 'velocity']:
            assert limit.get(prop) == expected_properties['limits'][prop], \
                f"Joint {joint_name} {prop} limit mismatch: {limit.get(prop)} != {expected_properties['limits'][prop]}"

def get_joint_connections(root, parent_link=None):
    """
    Get all joint connections from a URDF root element.
    
    Args:
        root: XML root element of the URDF
        parent_link: Optional parent link name to filter connections
        
    Returns:
        Set of (joint_name, child_link) tuples
    """
    joints = root.findall(".//joint[@type]")  # Only get joints with a type attribute
    connections = set()
    
    for joint in joints:
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        
        if parent_elem is not None and child_elem is not None:
            if parent_link is None or parent_elem.get('link') == parent_link:
                connections.add((joint.get('name'), child_elem.get('link')))
    
    return connections

def adjust_names_for_stage(connections, stage, name_manager):
    """
    Adjust connection names for a specific stage using the name manager.
    
    Args:
        connections: Set of (joint_name, link_name) tuples
        stage: Stage number to append
        name_manager: NameManager instance to use for name handling
        
    Returns:
        Set of adjusted (joint_name, link_name) tuples
    """
    adjusted = set()
    
    for joint, link in connections:
        # Extract joint number if it's a numbered joint
        if '_' in joint:
            joint_num = int(joint.split('_')[1])
            joint_name = name_manager.get_joint_name(joint_num)
        else:
            joint_name = name_manager.get_component_name(joint)
        
        # For links, use the name manager to get the correct name
        link_name = name_manager.get_component_name(link)
        
        adjusted.add((joint_name, link_name))
    
    return adjusted

def print_debug_info(generated_root, expected_connections, actual_connections, link_graph_data=None):
    """Print debug information about URDF connections."""
    print("\nAll joints in generated URDF:")
    for j in generated_root.findall(".//joint[@type]"):  # Only get joints with a type attribute
        print(f"Joint name: {j.get('name')}")
        parent_elem = j.find('parent')
        child_elem = j.find('child')
        print(f"  Parent element: {parent_elem}")
        print(f"  Child element: {child_elem}")
        if parent_elem is not None:
            print(f"  Parent link: {parent_elem.get('link')}")
        if child_elem is not None:
            print(f"  Child link: {child_elem.get('link')}")
        print("  Full joint XML:", ET.tostring(j, encoding='unicode'))
        print("-" * 50)
    
    print("\nDebug information for connections:")
    print("Generated connections:", actual_connections)
    print("Expected connections:", expected_connections)
    if link_graph_data:
        print("\nLink graph data:", link_graph_data)

def test_joint_consistency(urdf_properties):
    """Test if joints form a consistent structure"""
    generated_root = urdf_properties['generated']['root']
    generated_joints = urdf_properties['generated']['joints']
    
    # All joints should be in the properties
    for joint in generated_root.findall(".//joint[@type]"):
        joint_name = joint.get('name')
        assert joint_name in generated_joints, f"Joint {joint_name} should be in extracted properties"
        
        joint_props = generated_joints[joint_name]
        # Verify parent and child links exist and match
        assert joint_props.parent == joint.find('parent').get('link'), \
            f"Joint {joint_name} parent link mismatch"
        assert joint_props.child == joint.find('child').get('link'), \
            f"Joint {joint_name} child link mismatch"

def test_connection_attributes(urdf_properties, link_graph_data, name_manager):
    """Test if joint connections have correct attributes"""
    generated_joints = urdf_properties['generated']['joints']
    original_joints = urdf_properties['original']['joints']
    
    # Test implemented joints
    for parent, connections in link_graph_data.items():
        for joint_name, child in connections:
            print(f"\nTesting connection: {parent} -> {child} via {joint_name}")
            
            # Extract joint number and create stage 1 joint name
            joint_num = int(joint_name.split('_')[1])
            stage1_joint = name_manager.get_joint_name(joint_num)
            print(f"  Joint mapping: {joint_name} -> {stage1_joint}")
            
            # Add stage suffix to parent and child names
            stage1_parent = name_manager.get_component_name(parent)
            
            # Handle different component types
            if "bottom1" in child:
                # For bottom links, keep both original "1" suffix and stage suffix
                stage1_child = name_manager.get_component_name(child)
            elif any(x in child for x in ["cylinder", "rod"]):
                # For cylinders and rods, keep original name and add stage suffix
                stage1_child = name_manager.get_component_name(child)
            else:
                # For other components, strip original suffix if present
                stage1_child = name_manager.get_component_name(child.rsplit('1', 1)[0])
            
            print(f"  Parent mapping: {parent} -> {stage1_parent}")
            print(f"  Child mapping: {child} -> {stage1_child}")
            
            if stage1_joint in generated_joints:
                gen_props = generated_joints[stage1_joint]
                orig_props = original_joints.get(joint_name)
                
                print(f"  Generated joint properties found: {gen_props is not None}")
                print(f"  Original joint properties found: {orig_props is not None}")
                
                if orig_props:
                    # Verify joint type
                    assert gen_props.joint_type == orig_props.joint_type, \
                        f"Joint {stage1_joint} should be {orig_props.joint_type}"
                    
                    # Verify parent-child relationship
                    assert gen_props.parent == stage1_parent, \
                        f"Joint {stage1_joint} should have parent {stage1_parent}, got {gen_props.parent}"
                    assert gen_props.child == stage1_child, \
                        f"Joint {stage1_joint} should have child {stage1_child}, got {gen_props.child}"
                    
                    # Verify axis
                    assert gen_props.axis == orig_props.axis, \
                        f"Joint {stage1_joint} should have correct axis"
                    
                    # Verify limits
                    if orig_props.limits:
                        assert gen_props.limits is not None, \
                            f"Joint {stage1_joint} should have limits"
                        for prop in ['upper', 'lower', 'effort', 'velocity']:
                            assert gen_props.limits[prop] == orig_props.limits[prop], \
                                f"Joint {stage1_joint} should have correct {prop} limit"
                    
                    print(f"  All properties verified successfully for joint {stage1_joint}")

def test_no_floating_links(urdf_properties):
    """Test that all links (except base) are connected by joints"""
    generated_joints = urdf_properties['generated']['joints']
    generated_links = urdf_properties['generated']['links']
    
    # Get all connected links
    connected_links = set()
    for joint in generated_joints.values():
        connected_links.add(joint.parent)
        connected_links.add(joint.child)
    
    # Base link should be the only unconnected link
    all_links = set(generated_links.keys())
    unconnected = all_links - connected_links
    assert len(unconnected) <= 1, "Only base_link should be unconnected"
    if unconnected:
        assert list(unconnected)[0] == 'base_link1', \
            "The only unconnected link should be base_link1"

def test_no_cycles(urdf_properties):
    """Test that the joint-link structure contains no cycles"""
    generated_joints = urdf_properties['generated']['joints']
    
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
    for joint in generated_joints.values():
        if joint.parent not in graph:
            graph[joint.parent] = set()
        graph[joint.parent].add(joint.child)
    
    # Check for cycles from each link
    visited: Set[str] = set()
    path: Set[str] = set()
    
    for link in graph.keys():
        assert not find_cycle(link, graph, visited, path), \
            f"Found cycle in joint-link structure starting from {link}"

def test_cylinder_links(urdf_properties, name_manager):
    """Test if cylinder links are correctly generated"""
    original_props = urdf_properties['original']['links']
    generated_props = urdf_properties['generated']['links']
    
    # Test each cylinder link using exact names from mesh files
    cylinder_configs = [
        ("cylinder61", "X6bottom1"),  # Use exact name from cylinder61.stl
        ("cylinder51", "X5bottom1"),  # Use exact name from cylinder51.stl
        ("cylinder11", "X1bottom1"),  # Use exact name from cylinder11.stl
        ("cylinder21", "X2bottom1"),  # Use exact name from cylinder21.stl
        ("cylinder31", "X3bottom1"),  # Use exact name from cylinder31.stl
        ("cylinder41", "X4bottom1")   # Use exact name from cylinder41.stl
    ]
    
    for base_name, parent_base in cylinder_configs:
        # Generate names using NameManager
        link_name = name_manager.get_component_name(base_name)
        parent_name = name_manager.get_component_name(parent_base)
        
        print(f"\nTesting cylinder link: {link_name}")
        print(f"Original name: {base_name}")
        print(f"Parent link: {parent_name}")
        print(f"Generated properties keys: {list(generated_props.keys())}")  # Debug output
        print(f"Original properties keys: {list(original_props.keys())}")    # Debug output
        
        assert link_name in generated_props, f"Cylinder link {link_name} should exist"
        assert base_name in original_props, f"Original cylinder link {base_name} should exist"
        
        gen_link = generated_props[link_name]
        orig_link = original_props[base_name]  # Use exact mesh file name
        
        # Compare inertial properties
        for prop in ['mass', 'ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
            assert float(orig_link.inertial[prop]) == float(gen_link.inertial[prop]), \
                f"Cylinder {link_name} {prop} should match"
        
        # Compare visual and collision properties
        assert gen_link.visual['mesh'] == name_manager.get_mesh_filename(base_name), \
            f"Cylinder {link_name} should reference correct mesh file"
        assert gen_link.visual['scale'] == "0.001 0.001 0.001", \
            f"Cylinder {link_name} mesh scale should be correct"

def test_cylinder_joints(urdf_properties, name_manager):
    """Test if cylinder joints are correctly connected"""
    generated_joints = urdf_properties['generated']['joints']
    original_joints = urdf_properties['original']['joints']
    
    # Test revolute joints connecting bottom links to cylinders (Revolute_7-12)
    joint_configs = [
        (7, "X6bottom1", "cylinder61"),  # Use exact names from mesh files
        (8, "X5bottom1", "cylinder51"),
        (9, "X1bottom1", "cylinder11"),
        (10, "X2bottom1", "cylinder21"),
        (11, "X3bottom1", "cylinder31"),
        (12, "X4bottom1", "cylinder41")
    ]
    
    for joint_num, parent_base, child_base in joint_configs:
        # Generate names using NameManager
        joint_name = name_manager.get_joint_name(joint_num)
        parent_name = name_manager.get_component_name(parent_base)
        child_name = name_manager.get_component_name(child_base)
        base_joint_name = f"Revolute_{joint_num}"
        
        assert joint_name in generated_joints, f"Joint {joint_name} should exist"
        
        joint = generated_joints[joint_name]
        orig_joint = original_joints[base_joint_name]
        
        # Verify parent-child relationship
        assert joint.parent == parent_name, \
            f"Joint {joint_name} should have parent {parent_name}"
        assert joint.child == child_name, \
            f"Joint {joint_name} should have child {child_name}"
        
        # Verify joint properties
        assert joint.joint_type == orig_joint.joint_type, \
            f"Joint {joint_name} should be {orig_joint.joint_type}"
        assert joint.axis == orig_joint.axis, \
            f"Joint {joint_name} should have correct axis"
        
        # Verify limits if they exist
        if orig_joint.limits:
            assert joint.limits is not None, \
                f"Joint {joint_name} should have limits"
            for prop in ['upper', 'lower', 'effort', 'velocity']:
                assert float(joint.limits[prop]) == float(orig_joint.limits[prop]), \
                    f"Joint {joint_name} {prop} limit should match"

def test_cylinder_kinematic_chain(urdf_properties, name_manager):
    """Test if the kinematic chain from bottom links to cylinders is correct"""
    generated_root = urdf_properties['generated']['root']
    
    # Check each bottom link to cylinder connection
    connections = [
        ("X6bottom", "cylinder6"),
        ("X5bottom", "cylinder5"),
        ("X1bottom", "cylinder1"),
        ("X2bottom", "cylinder2"),
        ("X3bottom", "cylinder3"),
        ("X4bottom", "cylinder4")
    ]
    
    for parent_base, child_base in connections:
        parent_name = name_manager.get_component_name(parent_base)
        child_name = name_manager.get_component_name(child_base)
        
        # Find the joint connecting these links
        joint = None
        for j in generated_root.findall(".//joint[@type='revolute']"):
            if (j.find('parent').get('link') == parent_name and 
                j.find('child').get('link') == child_name):
                joint = j
                break
        
        assert joint is not None, \
            f"Should find a revolute joint connecting {parent_name} to {child_name}"
        
        # Verify the joint has all required elements
        assert joint.find('origin') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have origin"
        assert joint.find('axis') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have axis"
        assert joint.find('limit') is not None, \
            f"Joint connecting {parent_name} to {child_name} should have limits"

def test_rod_joints(urdf_properties, name_manager):
    """Test if rod links and slider joints are correctly connected"""
    generated_joints = urdf_properties['generated']['joints']
    original_joints = urdf_properties['original']['joints']
    
    # Test slider joints connecting cylinders to rods (Slider_13-18)
    joint_configs = [
        (13, "cylinder11", "rod11"),  # Use exact names from mesh files
        (14, "cylinder21", "rod21"),
        (15, "cylinder31", "rod31"),
        (16, "cylinder41", "rod41"),
        (17, "cylinder51", "rod51"),
        (18, "cylinder61", "rod61")
    ]
    
    for joint_num, parent_base, child_base in joint_configs:
        # Generate names using NameManager
        joint_name = name_manager.get_joint_name(joint_num)
        parent_name = name_manager.get_component_name(parent_base)
        child_name = name_manager.get_component_name(child_base)
        base_joint_name = f"Slider_{joint_num}"
        
        assert joint_name in generated_joints, f"Joint {joint_name} should exist"
        
        joint = generated_joints[joint_name]
        orig_joint = original_joints[base_joint_name]
        
        # Verify parent-child relationship
        assert joint.parent == parent_name, \
            f"Joint {joint_name} should have parent {parent_name}"
        assert joint.child == child_name, \
            f"Joint {joint_name} should have child {child_name}"
        
        # Verify joint properties
        assert joint.joint_type == orig_joint.joint_type, \
            f"Joint {joint_name} should be {orig_joint.joint_type}"
        assert joint.axis == orig_joint.axis, \
            f"Joint {joint_name} should have correct axis"
        
        # Verify limits if they exist
        if orig_joint.limits:
            assert joint.limits is not None, \
                f"Joint {joint_name} should have limits"
            for prop in ['upper', 'lower', 'effort', 'velocity']:
                assert float(joint.limits[prop]) == float(orig_joint.limits[prop]), \
                    f"Joint {joint_name} {prop} limit should match"

def test_rod_links(urdf_properties, name_manager):
    """Test if rod links have correct properties"""
    generated_links = urdf_properties['generated']['links']
    original_links = urdf_properties['original']['links']
    
    # Test rod links using exact names from mesh files
    rod_names = ["rod11", "rod21", "rod31", "rod41", "rod51", "rod61"]
    
    for base_name in rod_names:
        # Generate names using NameManager
        link_name = name_manager.get_component_name(base_name)
        
        assert link_name in generated_links, f"Link {link_name} should exist"
        
        link = generated_links[link_name]
        orig_link = original_links[base_name]  # Use exact mesh file name
        
        # Verify inertial properties
        assert link.inertial is not None, f"Link {link_name} should have inertial properties"
        assert orig_link.inertial is not None, f"Original link {base_name} should have inertial properties"
        
        # Compare inertial values
        assert float(link.inertial['mass']) == float(orig_link.inertial['mass']), \
            f"Link {link_name} should have correct mass"
        
        for prop in ['ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
            assert float(link.inertial[prop]) == float(orig_link.inertial[prop]), \
                f"Link {link_name} should have correct {prop}"
        
        # Verify visual properties
        assert link.visual is not None, f"Link {link_name} should have visual properties"
        assert link.visual['mesh'] == f"meshes/{base_name}.stl", \
            f"Link {link_name} should have correct mesh file"
        
        # Verify collision properties
        assert link.collision is not None, f"Link {link_name} should have collision properties"
        assert link.collision['mesh'] == f"meshes/{base_name}.stl", \
            f"Link {link_name} should have correct collision mesh file" 