from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
from extract_urdf_properties import extract_joint_properties, extract_link_properties

@dataclass
class Origin:
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float] = (0, 0, 0)

    def _format_value(self, value: float) -> str:
        """Format a value, using integer format for zero values."""
        return '0' if value == 0 else str(value)

    def to_xml(self) -> str:
        xyz_str = ' '.join(self._format_value(v) for v in self.xyz)
        rpy_str = ' '.join(self._format_value(v) for v in self.rpy)
        return f'xyz="{xyz_str}" rpy="{rpy_str}"'

@dataclass
class Inertial:
    origin: Origin
    mass: float
    ixx: float
    iyy: float
    izz: float
    ixy: float
    iyz: float
    ixz: float

    def to_xml(self) -> str:
        return f"""
            <inertial>
                <origin {self.origin.to_xml()}/>
                <mass value="{self.mass}"/>
                <inertia ixx="{self.ixx}" iyy="{self.iyy}" izz="{self.izz}" 
                         ixy="{self.ixy}" iyz="{self.iyz}" ixz="{self.ixz}"/>
            </inertial>"""

@dataclass
class Geometry:
    mesh_filename: str
    scale: Tuple[float, float, float] = (0.001, 0.001, 0.001)

    def to_xml(self) -> str:
        return f"""
                <geometry>
                    <mesh filename="{self.mesh_filename}" scale="{self.scale[0]} {self.scale[1]} {self.scale[2]}"/>
                </geometry>"""

@dataclass
class Visual:
    origin: Origin
    geometry: Geometry
    material_name: str = "silver"

    def to_xml(self) -> str:
        return f"""
            <visual>
                <origin {self.origin.to_xml()}/>
                {self.geometry.to_xml()}
                <material name="{self.material_name}"/>
            </visual>"""

@dataclass
class Collision:
    origin: Origin
    geometry: Geometry

    def to_xml(self) -> str:
        return f"""
            <collision>
                <origin {self.origin.to_xml()}/>
                {self.geometry.to_xml()}
            </collision>"""

@dataclass
class Link:
    name: str
    inertial: Inertial
    visual: Visual
    collision: Collision

    def to_xml(self) -> str:
        return f"""
        <link name="{self.name}">
            {self.inertial.to_xml()}
            {self.visual.to_xml()}
            {self.collision.to_xml()}
        </link>"""

@dataclass
class Transmission:
    name: str
    joint_name: str
    
    def to_xml(self) -> str:
        return f"""
        <transmission name="{self.name}">
            <type>transmission_interface/SimpleTransmission</type>
            <joint name="{self.joint_name}">
                <hardwareInterface>PositionJointInterface</hardwareInterface>
            </joint>
            <actuator name="{self.joint_name}_actr">
                <hardwareInterface>PositionJointInterface</hardwareInterface>
                <mechanicalReduction>1</mechanicalReduction>
            </actuator>
        </transmission>"""

@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Origin
    axis: Tuple[float, float, float]
    limits: Optional[Dict[str, float]] = None
    transmission: Optional[Transmission] = None

    def to_xml(self) -> str:
        """Generate XML representation of the joint."""
        # Build limits string if present
        limits_str = ""
        if self.limits:
            limits_str = (f'<limit upper="{self.limits["upper"]}" lower="{self.limits["lower"]}" '
                         f'effort="{self.limits["effort"]}" velocity="{self.limits["velocity"]}"/>')

        # Build joint XML with explicit parent and child elements
        joint_xml = f"""
        <joint name="{self.name}" type="{self.joint_type}">
            <origin {self.origin.to_xml()}/>
            <parent link="{self.parent}"/>
            <child link="{self.child}"/>
            <axis xyz="{self.axis[0]} {self.axis[1]} {self.axis[2]}"/>
            {limits_str}
        </joint>"""

        # Add transmission if present
        if self.transmission:
            joint_xml += "\n" + self.transmission.to_xml()

        return joint_xml

class NameManager:
    """Manages naming conventions for URDF components."""
    def __init__(self, stage: int, base_prefix: str = ""):
        self.stage = stage
        self.base_prefix = base_prefix
        self.stage_suffix = str(stage)

    def get_base_link_name(self) -> str:
        """Get the name of the base link."""
        return f"{self.base_prefix}base_link{self.stage_suffix}"

    def get_component_name(self, base_name: str) -> str:
        """Generate a unique name for a component."""
        # For bottom links, keep the original "1" suffix and add stage number
        if "bottom" in base_name:
            if base_name.endswith('1'):
                # If it already has a suffix, just add stage number
                return f"{base_name}{self.stage_suffix}"
            else:
                # If no suffix, add both original suffix and stage number
                return f"{base_name}1{self.stage_suffix}"
        # For other components, just append stage number
        return f"{base_name}{self.stage_suffix}"

    def get_joint_name(self, joint_num: int) -> str:
        """Get the name of a joint."""
        # All joints use Revolute_ prefix for PyBullet compatibility
        return f"Revolute_{joint_num}{self.stage_suffix}"

    def get_transmission_name(self, joint_name: str) -> str:
        """Get the name of a transmission."""
        return f"{joint_name}_tran"

    def strip_stage_suffix(self, name: str) -> str:
        """Remove stage suffix from a name."""
        return name.rsplit(self.stage_suffix, 1)[0]

    def get_original_bottom_link_name(self, link_name: str) -> str:
        """Get the original name for a bottom link."""
        if "bottom" in link_name:
            x_num = link_name.split('X')[1][0]  # Get the number after X
            return f"X{x_num}bottom1"
        return link_name

    def get_mesh_filename(self, base_name: str) -> str:
        """Get the mesh filename for a component."""
        if "bottom" in base_name:
            return f"meshes/{base_name}1.stl"  # Use original name with 1 suffix for mesh
        if "base_link" in base_name:
            return "meshes/base_link.stl"
        if "cylinder" in base_name or "rod" in base_name:  # Add rod links to this condition
            return f"meshes/{base_name}1.stl"  # Add 1 suffix for cylinder and rod meshes
        return f"meshes/{base_name}.stl"

class StewartPlatformURDF:
    def __init__(self, stage: int = 1, base_prefix: str = "", base_z: float = 0):
        """
        Initialize a Stewart Platform URDF generator.
        
        Args:
            stage: The stage number of this platform (1, 2, 3, etc.)
            base_prefix: Prefix for the base link name (e.g., "upper_", "lower_")
            base_z: Z-offset for the entire platform
        """
        self.stage = stage
        self.base_z = base_z
        self.name_mgr = NameManager(stage, base_prefix)
        
        # Load properties from original URDF
        self.original_joint_props = extract_joint_properties("Stewart.urdf")  # Remove Revolute_ prefix to load all joints
        self.original_link_props = extract_link_properties("Stewart.urdf")
        
        self.links: List[Link] = []
        self.joints: List[Joint] = []
        
        self._init_base_link()
        self._init_bottom_links()
        self._init_cylinder_links()
        self._init_rod_links()

    def _apply_z_offset(self, origin: Origin) -> Origin:
        """Apply the base_z offset to an origin."""
        return Origin(
            (origin.xyz[0], origin.xyz[1], origin.xyz[2] + self.base_z),
            origin.rpy
        )

    def _get_original_joint_props(self, joint_name: str) -> Optional[Dict]:
        """Get properties of a joint from the original URDF."""
        try:
            # Extract the joint number and create the original name
            base_name = self.name_mgr.strip_stage_suffix(joint_name)
            
            # Handle both Revolute_ and Slider_ prefixes
            if base_name.startswith('Slider_'):
                # For explicit Slider_ lookups, use as is
                original_name = base_name
            elif base_name.startswith('Revolute_'):
                # For Revolute_ joints, check if it's actually a slider joint (13-18)
                joint_num = int(base_name.split('_')[1])
                if 13 <= joint_num <= 18:
                    original_name = f"Slider_{joint_num}"
                else:
                    original_name = base_name
            else:
                return None

            # Get the original properties
            if original_name in self.original_joint_props:
                props = self.original_joint_props[original_name]
                
                # Update parent/child links to use stage-appropriate names
                if props.parent == "base_link":
                    props.parent = self.name_mgr.get_base_link_name()
                else:
                    props.parent = self.name_mgr.get_component_name(props.parent)
                    
                if "bottom" in props.child:
                    props.child = self.name_mgr.get_component_name(props.child.rsplit('1', 1)[0])
                else:
                    props.child = self.name_mgr.get_component_name(props.child)
                
                # For slider joints (13-18), keep the joint type as prismatic
                joint_num = int(original_name.split('_')[1])
                if 13 <= joint_num <= 18:
                    props.joint_type = "prismatic"
                
                return props
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to process joint name {joint_name}: {str(e)}")
            return None
        
        return None

    def _get_original_link_props(self, link_name: str) -> Optional[Dict]:
        """Get properties of a link from the original URDF."""
        # Get original name
        if "base_link" in link_name:
            base_name = "base_link"
        elif "bottom" in link_name:
            base_name = self.name_mgr.get_original_bottom_link_name(link_name)
        elif "rod" in link_name:
            # For rod links, add "1" suffix to match original URDF
            base_name = f"{link_name.rsplit('1', 1)[0]}1"
        else:
            base_name = link_name
        
        if base_name in self.original_link_props:
            return self.original_link_props[base_name]
        
        print(f"Warning: No properties found for link {base_name} (original name for {link_name})")
        return None

    def _init_base_link(self):
        """Initialize the base link of the Stewart platform."""
        base_link_name = self.name_mgr.get_base_link_name()
        original_props = self._get_original_link_props("base_link")
        
        if original_props:
            inertial_props = original_props.inertial
            base_origin = Origin(
                tuple(map(float, inertial_props['origin']['xyz'].split())),
                tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
            )
            if self.base_z:
                base_origin = self._apply_z_offset(base_origin)

            base_link = Link(
                name=base_link_name,
                inertial=Inertial(
                    origin=base_origin,
                    mass=float(inertial_props['mass']),
                    ixx=float(inertial_props['ixx']),
                    iyy=float(inertial_props['iyy']),
                    izz=float(inertial_props['izz']),
                    ixy=float(inertial_props['ixy']),
                    iyz=float(inertial_props['iyz']),
                    ixz=float(inertial_props['ixz'])
                ),
                visual=Visual(
                    origin=self._apply_z_offset(Origin((0, 0, 0))),
                    geometry=Geometry(self.name_mgr.get_mesh_filename("base_link"))
                ),
                collision=Collision(
                    origin=self._apply_z_offset(Origin((0, 0, 0))),
                    geometry=Geometry(self.name_mgr.get_mesh_filename("base_link"))
                )
            )
            self.links.append(base_link)

    def _init_bottom_links(self):
        """Initialize the bottom links and their joints."""
        # Bottom link configurations based on Link_graph.txt and Stewart.urdf
        bottom_configs = [
            # (name, joint_num)  # Joint numbers from Link_graph.txt
            ("X1bottom", 1),  # base_link -> X1bottom1 [ label = "Revolute_1" ]
            ("X6bottom", 2),  # base_link -> X6bottom1 [ label = "Revolute_2" ]
            ("X5bottom", 3),  # base_link -> X5bottom1 [ label = "Revolute_3" ]
            ("X2bottom", 4),  # base_link -> X2bottom1 [ label = "Revolute_4" ]
            ("X4bottom", 5),  # base_link -> X4bottom1 [ label = "Revolute_5" ]
            ("X3bottom", 6)   # base_link -> X3bottom1 [ label = "Revolute_6" ]
        ]

        for base_name, joint_num in bottom_configs:
            # Generate names for this stage
            link_name = self.name_mgr.get_component_name(base_name)
            joint_name = self.name_mgr.get_joint_name(joint_num)
            
            # Get original properties
            original_link_props = self._get_original_link_props(link_name)
            original_joint_props = self._get_original_joint_props(joint_name)
            
            if original_link_props and original_joint_props:
                # Create link
                inertial_props = original_link_props.inertial
                link_origin = Origin(
                    tuple(map(float, inertial_props['origin']['xyz'].split())),
                    tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
                )
                
                # Create the link
                bottom_link = Link(
                    name=link_name,
                    inertial=Inertial(
                        origin=link_origin,
                        mass=float(inertial_props['mass']),
                        ixx=float(inertial_props['ixx']),
                        iyy=float(inertial_props['iyy']),
                        izz=float(inertial_props['izz']),
                        ixy=float(inertial_props['ixy']),
                        iyz=float(inertial_props['iyz']),
                        ixz=float(inertial_props['ixz'])
                    ),
                    visual=Visual(
                        origin=Origin(
                            tuple(map(float, original_link_props.visual['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.visual['origin']['rpy'].split())) if 'rpy' in original_link_props.visual['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    ),
                    collision=Collision(
                        origin=Origin(
                            tuple(map(float, original_link_props.collision['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.collision['origin']['rpy'].split())) if 'rpy' in original_link_props.collision['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    )
                )
                self.links.append(bottom_link)

                # Create joint with explicit parent and child links
                joint_origin = Origin(
                    tuple(map(float, original_joint_props.origin['xyz'].split())),
                    tuple(map(float, original_joint_props.origin['rpy'].split())) if 'rpy' in original_joint_props.origin else (0, 0, 0)
                )
                
                # Set parent and child links explicitly
                parent_link = self.name_mgr.get_base_link_name()  # All bottom links connect to base_link
                child_link = link_name  # The bottom link we just created
                
                joint = Joint(
                    name=joint_name,
                    joint_type=original_joint_props.joint_type,
                    parent=parent_link,
                    child=child_link,
                    origin=joint_origin,
                    axis=tuple(map(float, original_joint_props.axis.split())),
                    limits=original_joint_props.limits,
                    transmission=Transmission(
                        self.name_mgr.get_transmission_name(joint_name),
                        joint_name
                    )
                )
                self.joints.append(joint)

    def _init_cylinder_links(self):
        """Initialize the cylinder links and their revolute joints with bottom links."""
        # Cylinder configurations based on Link_graph.txt
        cylinder_configs = [
            # (cylinder_name, parent_bottom_link, joint_num)
            ("cylinder6", "X6bottom", 7),
            ("cylinder5", "X5bottom", 8),
            ("cylinder1", "X1bottom", 9),
            ("cylinder2", "X2bottom", 10),
            ("cylinder3", "X3bottom", 11),
            ("cylinder4", "X4bottom", 12)
        ]

        for base_name, parent_base, joint_num in cylinder_configs:
            # Generate names for this stage
            link_name = self.name_mgr.get_component_name(base_name)
            joint_name = self.name_mgr.get_joint_name(joint_num)
            parent_name = self.name_mgr.get_component_name(parent_base)
            
            # Get original properties
            original_link_props = self._get_original_link_props(link_name)
            original_joint_props = self._get_original_joint_props(joint_name)
            
            if original_link_props and original_joint_props:
                # Create cylinder link
                inertial_props = original_link_props.inertial
                link_origin = Origin(
                    tuple(map(float, inertial_props['origin']['xyz'].split())),
                    tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
                )
                
                cylinder_link = Link(
                    name=link_name,
                    inertial=Inertial(
                        origin=link_origin,
                        mass=float(inertial_props['mass']),
                        ixx=float(inertial_props['ixx']),
                        iyy=float(inertial_props['iyy']),
                        izz=float(inertial_props['izz']),
                        ixy=float(inertial_props['ixy']),
                        iyz=float(inertial_props['iyz']),
                        ixz=float(inertial_props['ixz'])
                    ),
                    visual=Visual(
                        origin=Origin(
                            tuple(map(float, original_link_props.visual['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.visual['origin']['rpy'].split())) if 'rpy' in original_link_props.visual['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    ),
                    collision=Collision(
                        origin=Origin(
                            tuple(map(float, original_link_props.collision['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.collision['origin']['rpy'].split())) if 'rpy' in original_link_props.collision['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    )
                )
                self.links.append(cylinder_link)

                # Create revolute joint connecting bottom link to cylinder
                joint_origin = Origin(
                    tuple(map(float, original_joint_props.origin['xyz'].split())),
                    tuple(map(float, original_joint_props.origin['rpy'].split())) if 'rpy' in original_joint_props.origin else (0, 0, 0)
                )
                
                joint = Joint(
                    name=joint_name,
                    joint_type=original_joint_props.joint_type,
                    parent=parent_name,
                    child=link_name,
                    origin=joint_origin,
                    axis=tuple(map(float, original_joint_props.axis.split())),
                    limits=original_joint_props.limits,
                    transmission=Transmission(
                        self.name_mgr.get_transmission_name(joint_name),
                        joint_name
                    )
                )
                self.joints.append(joint)
                
                print(f"\nDebug: Generated cylinder link and joint:")
                print(f"Link name: {link_name}")
                print(f"Joint name: {joint_name}")
                print(f"Parent link: {parent_name}")
                print(f"Joint type: {joint.joint_type}")
                print(f"Joint axis: {joint.axis}")
                if joint.limits:
                    print(f"Joint limits: {joint.limits}")

    def _init_rod_links(self):
        """Initialize the rod links and their slider joints with cylinders."""
        # Rod configurations based on Link_graph.txt
        rod_configs = [
            # (rod_name, parent_cylinder, joint_num)
            ("rod1", "cylinder1", 13),
            ("rod2", "cylinder2", 14),
            ("rod3", "cylinder3", 15),
            ("rod4", "cylinder4", 16),
            ("rod5", "cylinder5", 17),
            ("rod6", "cylinder6", 18)
        ]

        for base_name, parent_base, joint_num in rod_configs:
            # Generate names for this stage
            link_name = self.name_mgr.get_component_name(base_name)
            joint_name = self.name_mgr.get_joint_name(joint_num)  # This will use Revolute_ prefix
            parent_name = self.name_mgr.get_component_name(parent_base)
            
            # Get original properties using Slider_ prefix to match original URDF
            original_link_props = self._get_original_link_props(link_name)
            original_joint_props = self._get_original_joint_props(joint_name)  # Will handle Slider_ internally
            
            if original_link_props and original_joint_props:
                # Create rod link
                inertial_props = original_link_props.inertial
                link_origin = Origin(
                    tuple(map(float, inertial_props['origin']['xyz'].split())),
                    tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
                )
                
                rod_link = Link(
                    name=link_name,
                    inertial=Inertial(
                        origin=link_origin,
                        mass=float(inertial_props['mass']),
                        ixx=float(inertial_props['ixx']),
                        iyy=float(inertial_props['iyy']),
                        izz=float(inertial_props['izz']),
                        ixy=float(inertial_props['ixy']),
                        iyz=float(inertial_props['iyz']),
                        ixz=float(inertial_props['ixz'])
                    ),
                    visual=Visual(
                        origin=Origin(
                            tuple(map(float, original_link_props.visual['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.visual['origin']['rpy'].split())) if 'rpy' in original_link_props.visual['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    ),
                    collision=Collision(
                        origin=Origin(
                            tuple(map(float, original_link_props.collision['origin']['xyz'].split())),
                            tuple(map(float, original_link_props.collision['origin']['rpy'].split())) if 'rpy' in original_link_props.collision['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    )
                )
                self.links.append(rod_link)

                # Create slider joint connecting cylinder to rod
                joint_origin = Origin(
                    tuple(map(float, original_joint_props.origin['xyz'].split())),
                    tuple(map(float, original_joint_props.origin['rpy'].split())) if 'rpy' in original_joint_props.origin else (0, 0, 0)
                )
                
                joint = Joint(
                    name=joint_name,  # Use Revolute_ prefix for PyBullet compatibility
                    joint_type=original_joint_props.joint_type,  # Will be "prismatic" from _get_original_joint_props
                    parent=parent_name,
                    child=link_name,
                    origin=joint_origin,
                    axis=tuple(map(float, original_joint_props.axis.split())),
                    limits=original_joint_props.limits,
                    transmission=Transmission(
                        self.name_mgr.get_transmission_name(joint_name),
                        joint_name
                    )
                )
                self.joints.append(joint)
                
                print(f"\nDebug: Generated rod link and joint:")
                print(f"Link name: {link_name}")
                print(f"Joint name: {joint_name}")
                print(f"Parent link: {parent_name}")
                print(f"Joint type: {joint.joint_type}")
                print(f"Joint axis: {joint.axis}")
                if joint.limits:
                    print(f"Joint limits: {joint.limits}")

    def generate(self) -> str:
        """Generate the URDF XML string."""
        xml_parts = [
            '<?xml version="1.0" ?>',
            f'<robot name="Stewart_{self.stage}">',
            '<material name="silver">',
            '    <color rgba="0.700 0.700 0.700 1.000"/>',
            '</material>'
        ]
        
        # Add all links
        for link in self.links:
            xml_parts.append(link.to_xml())
            
        # Add all joints with debug output
        print("\nDebug: Generating joints:")
        for joint in self.joints:
            joint_xml = joint.to_xml()
            print(f"\nJoint {joint.name}:")
            print(f"Parent: {joint.parent}")
            print(f"Child: {joint.child}")
            print("Generated XML:")
            print(joint_xml)
            xml_parts.append(joint_xml)
            
        xml_parts.append('</robot>')
        urdf_str = '\n'.join(xml_parts)
        
        # Validate and clean up XML
        try:
            # First parse to validate
            root = ET.fromstring(urdf_str)
            
            # Clean up the XML by re-serializing without extra whitespace
            clean_xml = ET.tostring(root, encoding='unicode', method='xml')
            
            # Parse again and pretty print for debugging
            root = ET.fromstring(clean_xml)
            from xml.dom import minidom
            pretty_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            
            # Debug: Print final XML
            print("\nDebug: Complete generated URDF (pretty printed):")
            print(pretty_xml)
            
            # Return the clean XML, not the pretty-printed version
            return clean_xml
            
        except ET.ParseError as e:
            print(f"\nError: Failed to parse generated XML: {e}")
            print("Original XML:")
            print(urdf_str)
            raise

def generate_stewart_platform(stage: int = 1, base_prefix: str = "", base_z: float = 0) -> str:
    """
    Generate URDF string for a Stewart platform.
    
    Args:
        stage: The stage number (1, 2, 3, etc.)
        base_prefix: Prefix for the base link name
        base_z: Z-offset for the entire platform
    """
    platform = StewartPlatformURDF(stage, base_prefix, base_z)
    return platform.generate()

def save_urdf(urdf_string: str, filename: str = "stewart.urdf") -> None:
    """Save URDF string to a file."""
    with open(filename, "w") as f:
        f.write(urdf_string)

if __name__ == "__main__":
    # Example: Generate two stacked platforms
    platform1 = generate_stewart_platform(stage=1, base_prefix="lower_", base_z=0)
    # platform2 = generate_stewart_platform(stage=2, base_prefix="upper_", base_z=0.5)  # 0.5m above first platform
    
    save_urdf(platform1, "stewart_lower.urdf")
    # save_urdf(platform2, "stewart_upper.urdf") 