import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import os

@dataclass
class JointProperties:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Dict[str, str]  # {'xyz': '0 0 0', 'rpy': '0 0 0'}
    axis: str  # '1 0 0'
    limits: Optional[Dict[str, str]] = None  # {'upper': '0.785', 'lower': '-0.785', ...}

@dataclass
class LinkProperties:
    name: str
    inertial: Dict[str, str]  # {'origin': {'xyz': '0 0 0', 'rpy': '0 0 0'}, 'mass': '1.0', ...}
    visual: Dict[str, str]  # {'origin': {'xyz': '0 0 0', 'rpy': '0 0 0'}, 'mesh': 'path.stl', ...}
    collision: Dict[str, str]  # Similar to visual

def clean_xml_string(xml_str: str) -> str:
    """Clean up XML string by removing excessive whitespace."""
    # Remove empty lines and normalize whitespace
    lines = [line.strip() for line in xml_str.splitlines() if line.strip()]
    return '\n'.join(lines)

def extract_properties_from_xml_string(xml_str: str) -> ET.Element:
    """Parse XML string and return the root element."""
    try:
        return ET.fromstring(clean_xml_string(xml_str))
    except ET.ParseError as e:
        print(f"Error parsing XML string: {str(e)}")
        print("XML content:")
        print(xml_str)
        raise

def extract_joint_properties_from_string(xml_str: str, joint_prefix: str = "") -> Dict[str, JointProperties]:
    """Extract joint properties from an XML string."""
    try:
        root = extract_properties_from_xml_string(xml_str)
        joints = {}
        
        # First, collect all links to validate against
        valid_links = {link.get('name') for link in root.findall(".//link")}
        print(f"\nDebug: Found valid links: {valid_links}")
        
        # Find all joints but exclude those in transmissions
        for joint in root.findall(".//joint[@type]"):  # Only get joints with a type attribute
            try:
                name = joint.get('name')
                if joint_prefix and not name.startswith(joint_prefix):
                    continue
                
                # Debug output for joint XML
                print(f"\nDebug: Processing joint {name}")
                print(f"Joint XML: {ET.tostring(joint, encoding='unicode')}")
                
                # Get joint type (default to fixed if not specified)
                joint_type = joint.get('type', 'fixed')
                
                # Get origin properties
                origin = joint.find('origin')
                origin_props = {
                    'xyz': origin.get('xyz', '0 0 0'),
                    'rpy': origin.get('rpy', '0 0 0')
                } if origin is not None else {'xyz': '0 0 0', 'rpy': '0 0 0'}
                
                # Get axis (default to x-axis if not specified)
                axis = joint.find('axis')
                axis_xyz = axis.get('xyz', '1 0 0') if axis is not None else '1 0 0'
                
                # Get limits for non-fixed joints
                limit = joint.find('limit')
                limit_props = None
                if limit is not None and joint_type != 'fixed':
                    limit_props = {
                        'upper': limit.get('upper'),
                        'lower': limit.get('lower'),
                        'effort': limit.get('effort', '100'),  # Default values from Stewart.urdf
                        'velocity': limit.get('velocity', '100')
                    }
                
                # Get parent and child links with direct XML path
                parent = joint.find('./parent')
                child = joint.find('./child')
                
                print(f"Debug: Found parent element: {ET.tostring(parent, encoding='unicode') if parent is not None else None}")
                print(f"Debug: Found child element: {ET.tostring(child, encoding='unicode') if child is not None else None}")
                
                # Get parent and child link names, with proper XML element validation
                parent_link = parent.get('link') if parent is not None else None
                child_link = child.get('link') if child is not None else None
                
                print(f"Debug: Parent link: {parent_link}")
                print(f"Debug: Child link: {child_link}")
                
                # Validate that both parent and child links exist in the URDF
                if parent_link is None or child_link is None:
                    if joint_prefix and name.startswith(joint_prefix):
                        print(f"Warning: Joint {name} has missing link reference: parent={parent_link}, child={child_link}")
                    continue
                
                if parent_link not in valid_links or child_link not in valid_links:
                    if joint_prefix and name.startswith(joint_prefix):
                        print(f"Warning: Joint {name} references non-existent link(s): parent={parent_link}, child={child_link}")
                    continue
                
                # Create JointProperties object
                joints[name] = JointProperties(
                    name=name,
                    joint_type=joint_type,
                    parent=parent_link,
                    child=child_link,
                    origin=origin_props,
                    axis=axis_xyz,
                    limits=limit_props
                )
            except Exception as e:
                if joint_prefix and name.startswith(joint_prefix):
                    print(f"Warning: Failed to process joint {name}: {str(e)}")
                continue
        
        return joints
    except Exception as e:
        print(f"Error extracting joint properties from XML string: {str(e)}")
        return {}

def extract_link_properties_from_string(xml_str: str, link_prefix: str = "") -> Dict[str, LinkProperties]:
    """Extract link properties from an XML string."""
    try:
        root = extract_properties_from_xml_string(xml_str)
        links = {}
        
        for link in root.findall(".//link"):
            name = link.get('name')
            if link_prefix and not name.startswith(link_prefix):
                continue
                
            # Extract inertial properties
            inertial = link.find('inertial')
            inertial_props = {}
            if inertial is not None:
                origin = inertial.find('origin')
                if origin is not None:
                    inertial_props['origin'] = {
                        'xyz': origin.get('xyz', '0 0 0'),
                        'rpy': origin.get('rpy', '0 0 0')
                    }
                
                mass = inertial.find('mass')
                if mass is not None:
                    inertial_props['mass'] = mass.get('value')
                
                inertia = inertial.find('inertia')
                if inertia is not None:
                    for prop in ['ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
                        inertial_props[prop] = inertia.get(prop, '0')
            
            # Extract visual properties
            visual = link.find('visual')
            visual_props = {}
            if visual is not None:
                origin = visual.find('origin')
                if origin is not None:
                    visual_props['origin'] = {
                        'xyz': origin.get('xyz', '0 0 0'),
                        'rpy': origin.get('rpy', '0 0 0')
                    }
                
                geometry = visual.find('.//mesh')
                if geometry is not None:
                    visual_props['mesh'] = geometry.get('filename')
                    visual_props['scale'] = geometry.get('scale', '0.001 0.001 0.001')
            
            # Extract collision properties
            collision = link.find('collision')
            collision_props = {}
            if collision is not None:
                origin = collision.find('origin')
                if origin is not None:
                    collision_props['origin'] = {
                        'xyz': origin.get('xyz', '0 0 0'),
                        'rpy': origin.get('rpy', '0 0 0')
                    }
                
                geometry = collision.find('.//mesh')
                if geometry is not None:
                    collision_props['mesh'] = geometry.get('filename')
                    collision_props['scale'] = geometry.get('scale', '0.001 0.001 0.001')
            
            # Create LinkProperties object
            links[name] = LinkProperties(
                name=name,
                inertial=inertial_props,
                visual=visual_props,
                collision=collision_props
            )
        
        return links
    except Exception as e:
        print(f"Error extracting link properties from XML string: {str(e)}")
        return {}

def extract_joint_properties(urdf_path_or_str: str, joint_prefix: str = "") -> Dict[str, JointProperties]:
    """
    Extract joint properties from a URDF file or XML string.
    
    Args:
        urdf_path_or_str: Path to the URDF file or XML string
        joint_prefix: Optional prefix to filter joints (e.g., "Revolute_")
        
    Returns:
        Dictionary mapping joint names to their properties
    """
    try:
        # Check if input is a file path or XML string
        if os.path.exists(urdf_path_or_str):
            with open(urdf_path_or_str, 'r') as f:
                xml_str = f.read()
        else:
            xml_str = urdf_path_or_str
        return extract_joint_properties_from_string(xml_str, joint_prefix)
    except Exception as e:
        print(f"Error reading URDF: {str(e)}")
        return {}

def extract_link_properties(urdf_path_or_str: str, link_prefix: str = "") -> Dict[str, LinkProperties]:
    """
    Extract link properties from a URDF file or XML string.
    
    Args:
        urdf_path_or_str: Path to the URDF file or XML string
        link_prefix: Optional prefix to filter links
        
    Returns:
        Dictionary mapping link names to their properties
    """
    try:
        # Check if input is a file path or XML string
        if os.path.exists(urdf_path_or_str):
            with open(urdf_path_or_str, 'r') as f:
                xml_str = f.read()
        else:
            xml_str = urdf_path_or_str
        return extract_link_properties_from_string(xml_str, link_prefix)
    except Exception as e:
        print(f"Error reading URDF: {str(e)}")
        return {}

def save_properties_to_json(properties: Dict, output_file: str):
    """Save extracted properties to a JSON file."""
    # Convert dataclass objects to dictionaries
    json_data = {k: vars(v) for k, v in properties.items()}
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

def main():
    # Example usage
    urdf_file = "Stewart.urdf"  # Note the capital S
    
    # Extract joint properties - get both Revolute_ and Slider_ joints
    print(f"Extracting joint properties from {urdf_file}...")
    joint_properties = {}
    
    # Extract revolute joints
    revolute_joints = extract_joint_properties(urdf_file, "Revolute_")
    joint_properties.update(revolute_joints)
    
    # Extract slider joints
    slider_joints = extract_joint_properties(urdf_file, "Slider_")
    joint_properties.update(slider_joints)
    
    if joint_properties:
        save_properties_to_json(joint_properties, "joint_properties.json")
        print(f"Saved {len(joint_properties)} joint properties to joint_properties.json")
    
        # Print some example properties
        print("\nExample Joint Properties:")
        for name, props in list(joint_properties.items())[:2]:  # First two joints
            print(f"\n{name}:")
            print(f"  Type: {props.joint_type}")
            print(f"  Parent: {props.parent}")
            print(f"  Child: {props.child}")
            print(f"  Origin: {props.origin}")
            print(f"  Axis: {props.axis}")
            if props.limits:
                print(f"  Limits: {props.limits}")
    
    # Extract link properties
    print(f"\nExtracting link properties from {urdf_file}...")
    link_properties = extract_link_properties(urdf_file)
    
    if link_properties:
        save_properties_to_json(link_properties, "link_properties.json")
        print(f"Saved {len(link_properties)} link properties to link_properties.json")
    
        print("\nExample Link Properties:")
        for name, props in list(link_properties.items())[:2]:  # First two links
            print(f"\n{name}:")
            print(f"  Inertial: {props.inertial.get('mass', 'N/A')} kg")
            print(f"  Visual mesh: {props.visual.get('mesh', 'N/A')}")

if __name__ == "__main__":
    main() 