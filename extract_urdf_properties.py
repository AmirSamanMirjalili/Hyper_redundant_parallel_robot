import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from abc import ABC, abstractmethod

@dataclass
class JointProperties:
    """Properties of a URDF joint."""
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Dict[str, str]  # {'xyz': '0 0 0', 'rpy': '0 0 0'}
    axis: str  # '1 0 0'
    limits: Optional[Dict[str, str]] = None  # {'upper': '0.785', 'lower': '-0.785', ...}

@dataclass
class LinkProperties:
    """Properties of a URDF link."""
    name: str
    inertial: Dict[str, str]  # {'origin': {'xyz': '0 0 0', 'rpy': '0 0 0'}, 'mass': '1.0', ...}
    visual: Dict[str, str]  # {'origin': {'xyz': '0 0 0', 'rpy': '0 0 0'}, 'mesh': 'path.stl', ...}
    collision: Dict[str, str]  # Similar to visual

class XMLParser:
    """Class for handling XML parsing and validation."""
    @staticmethod
    def clean_xml_string(xml_str: str) -> str:
        """Clean up XML string by removing excessive whitespace."""
        lines = [line.strip() for line in xml_str.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    @staticmethod
    def parse_xml_string(xml_str: str) -> ET.Element:
        """Parse XML string and return the root element."""
        try:
            return ET.fromstring(XMLParser.clean_xml_string(xml_str))
        except ET.ParseError as e:
            print(f"Error parsing XML string: {str(e)}")
            print("XML content:")
            print(xml_str)
            raise

    @staticmethod
    def get_origin_properties(element: ET.Element) -> Dict[str, str]:
        """Extract origin properties from an XML element."""
        origin = element.find('origin')
        if origin is not None:
            return {
                'xyz': origin.get('xyz', '0 0 0'),
                'rpy': origin.get('rpy', '0 0 0')
            }
        return {'xyz': '0 0 0', 'rpy': '0 0 0'}

    @staticmethod
    def get_geometry_properties(element: ET.Element) -> Dict[str, str]:
        """Extract geometry properties from an XML element."""
        geometry = element.find('.//mesh')
        if geometry is not None:
            return {
                'mesh': geometry.get('filename'),
                'scale': geometry.get('scale', '0.001 0.001 0.001')
            }
        return {}

class PropertyExtractor(ABC):
    """Abstract base class for property extractors."""
    def __init__(self, xml_root: ET.Element):
        self.root = xml_root
        self.valid_links = {link.get('name') for link in self.root.findall(".//link")}
    
    @abstractmethod
    def extract(self, prefix: str = "") -> Dict[str, Any]:
        """Extract properties from XML."""
        pass

class JointPropertyExtractor(PropertyExtractor):
    """Class for extracting joint properties."""
    def extract(self, prefix: str = "") -> Dict[str, JointProperties]:
        """Extract joint properties from XML."""
        joints = {}
        
        for joint in self.root.findall(".//joint[@type]"):
            try:
                name = joint.get('name')
                if prefix and not name.startswith(prefix):
                    continue
                
                # Get joint type
                joint_type = joint.get('type', 'fixed')
                
                # Get origin and axis properties
                origin_props = XMLParser.get_origin_properties(joint)
                axis = joint.find('axis')
                axis_xyz = axis.get('xyz', '1 0 0') if axis is not None else '1 0 0'
                
                # Get limits
                limit = joint.find('limit')
                limit_props = None
                if limit is not None and joint_type != 'fixed':
                    limit_props = {
                        'upper': limit.get('upper'),
                        'lower': limit.get('lower'),
                        'effort': limit.get('effort', '100'),
                        'velocity': limit.get('velocity', '100')
                    }
                
                # Get parent and child links
                parent = joint.find('./parent')
                child = joint.find('./child')
                parent_link = parent.get('link') if parent is not None else None
                child_link = child.get('link') if child is not None else None
                
                # Validate links
                if not self._validate_links(name, parent_link, child_link, prefix):
                    continue
                
                # Create JointProperties
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
                if prefix and name.startswith(prefix):
                    print(f"Warning: Failed to process joint {name}: {str(e)}")
                continue
        
        return joints
    
    def _validate_links(self, joint_name: str, parent_link: Optional[str], child_link: Optional[str], prefix: str) -> bool:
        """Validate parent and child links exist."""
        if parent_link is None or child_link is None:
            if prefix and joint_name.startswith(prefix):
                print(f"Warning: Joint {joint_name} has missing link reference: parent={parent_link}, child={child_link}")
            return False
        
        if parent_link not in self.valid_links or child_link not in self.valid_links:
            if prefix and joint_name.startswith(prefix):
                print(f"Warning: Joint {joint_name} references non-existent link(s): parent={parent_link}, child={child_link}")
            return False
        
        return True

class LinkPropertyExtractor(PropertyExtractor):
    """Class for extracting link properties."""
    def extract(self, prefix: str = "") -> Dict[str, LinkProperties]:
        """Extract link properties from XML."""
        links = {}
        
        for link in self.root.findall(".//link"):
            try:
                name = link.get('name')
                if prefix and not name.startswith(prefix):
                    continue
                
                # Extract inertial properties
                inertial_props = self._extract_inertial_properties(link)
                
                # Extract visual properties
                visual_props = self._extract_visual_properties(link)
                
                # Extract collision properties
                collision_props = self._extract_collision_properties(link)
                
                # Create LinkProperties
                links[name] = LinkProperties(
                    name=name,
                    inertial=inertial_props,
                    visual=visual_props,
                    collision=collision_props
                )
            except Exception as e:
                print(f"Warning: Failed to process link {name}: {str(e)}")
                continue
        
        return links
    
    def _extract_inertial_properties(self, link: ET.Element) -> Dict[str, str]:
        """Extract inertial properties from a link element."""
        inertial = link.find('inertial')
        if inertial is None:
            return {}
        
        props = {}
        props['origin'] = XMLParser.get_origin_properties(inertial)
        
        mass = inertial.find('mass')
        if mass is not None:
            props['mass'] = mass.get('value')
        
        inertia = inertial.find('inertia')
        if inertia is not None:
            for prop in ['ixx', 'iyy', 'izz', 'ixy', 'iyz', 'ixz']:
                props[prop] = inertia.get(prop, '0')
        
        return props
    
    def _extract_visual_properties(self, link: ET.Element) -> Dict[str, str]:
        """Extract visual properties from a link element."""
        visual = link.find('visual')
        if visual is None:
            return {}
        
        props = {}
        props['origin'] = XMLParser.get_origin_properties(visual)
        props.update(XMLParser.get_geometry_properties(visual))
        return props
    
    def _extract_collision_properties(self, link: ET.Element) -> Dict[str, str]:
        """Extract collision properties from a link element."""
        collision = link.find('collision')
        if collision is None:
            return {}
        
        props = {}
        props['origin'] = XMLParser.get_origin_properties(collision)
        props.update(XMLParser.get_geometry_properties(collision))
        return props

class URDFPropertyExtractor:
    """Main class for extracting properties from URDF files."""
    def __init__(self, urdf_path_or_str: str):
        """Initialize with URDF file path or XML string."""
        self.xml_str = self._load_urdf(urdf_path_or_str)
        self.root = XMLParser.parse_xml_string(self.xml_str)
        self.joint_extractor = JointPropertyExtractor(self.root)
        self.link_extractor = LinkPropertyExtractor(self.root)
    
    def _load_urdf(self, urdf_path_or_str: str) -> str:
        """Load URDF from file path or return XML string."""
        if os.path.exists(urdf_path_or_str):
            with open(urdf_path_or_str, 'r') as f:
                return f.read()
        return urdf_path_or_str
    
    def extract_joint_properties(self, joint_prefix: str = "") -> Dict[str, JointProperties]:
        """Extract joint properties with optional prefix filter."""
        return self.joint_extractor.extract(joint_prefix)
    
    def extract_link_properties(self, link_prefix: str = "") -> Dict[str, LinkProperties]:
        """Extract link properties with optional prefix filter."""
        return self.link_extractor.extract(link_prefix)
    
    def save_properties_to_json(self, properties: Dict, output_file: str):
        """Save extracted properties to a JSON file."""
        json_data = {k: vars(v) for k, v in properties.items()}
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

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
        extractor = URDFPropertyExtractor(urdf_path_or_str)
        return extractor.extract_joint_properties(joint_prefix)
    except Exception as e:
        print(f"Error extracting joint properties: {str(e)}")
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
        extractor = URDFPropertyExtractor(urdf_path_or_str)
        return extractor.extract_link_properties(link_prefix)
    except Exception as e:
        print(f"Error extracting link properties: {str(e)}")
        return {}

def main():
    """Example usage of the property extractors."""
    urdf_file = "Stewart.urdf"
    
    try:
        # Create URDF property extractor
        extractor = URDFPropertyExtractor(urdf_file)
        
        # Extract joint properties
        print(f"Extracting joint properties from {urdf_file}...")
        joint_properties = {}
        
        # Extract revolute joints
        revolute_joints = extractor.extract_joint_properties("Revolute_")
        joint_properties.update(revolute_joints)
        
        # Extract slider joints
        slider_joints = extractor.extract_joint_properties("Slider_")
        joint_properties.update(slider_joints)
        
        if joint_properties:
            extractor.save_properties_to_json(joint_properties, "joint_properties.json")
            print(f"Saved {len(joint_properties)} joint properties to joint_properties.json")
            
            # Print example properties
            print("\nExample Joint Properties:")
            for name, props in list(joint_properties.items())[:2]:
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
        link_properties = extractor.extract_link_properties()
        
        if link_properties:
            extractor.save_properties_to_json(link_properties, "link_properties.json")
            print(f"Saved {len(link_properties)} link properties to link_properties.json")
            
            print("\nExample Link Properties:")
            for name, props in list(link_properties.items())[:2]:
                print(f"\n{name}:")
                print(f"  Inertial: {props.inertial.get('mass', 'N/A')} kg")
                print(f"  Visual mesh: {props.visual.get('mesh', 'N/A')}")
    
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 