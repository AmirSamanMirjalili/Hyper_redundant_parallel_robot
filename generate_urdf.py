from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import xml.etree.ElementTree as ET
from extract_urdf_properties import extract_joint_properties, extract_link_properties, JointProperties, LinkProperties
import re

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
    """Manages naming conventions for URDF components.

    This class provides methods to generate consistent and unique names for
    links, joints, and other components in a URDF file, especially when
    creating multi-stage Stewart platforms. It handles the addition of stage-specific
    suffixes to names and ensures that mesh filenames are correctly referenced.

    For example, if you have a base name "bottom_link" and the stage is 2,
    the `get_component_name` method will return "bottom_link2".  For "X1bottom1" it will return "X1bottom12".
    """
    def __init__(self, stage: int, base_prefix: str = ""):
        """Initializes the NameManager.

        Args:
            stage (int): The stage number of the platform. This is used to generate
                         unique names for each stage. For example, stage 1, stage 2, etc.
            base_prefix (str, optional): A prefix to add to the base link name.
                                         Defaults to "".  For example, "upper_" or "lower_".
        """
        self.stage = stage
        self.base_prefix = base_prefix
        self.stage_suffix = str(stage)

    def get_base_link_name(self) -> str:
        """Get the name of the base link.

        Returns:
            str: The name of the base link, including the base prefix and stage suffix.
                 For example, if base_prefix is "upper_" and stage is 2, the result
                 would be "upper_base_link2".
        """
        return f"{self.base_prefix}base_link{self.stage_suffix}"

    def get_component_name(self, base_name: str) -> str:
        """Generate a unique name for a component.

        This method handles special naming conventions for components, ensuring
        that original names from mesh files are preserved while adding the stage number.

        Args:
            base_name (str): The base name of the component (e.g., "X1bottom1", "cylinder11", "rod11").

        Returns:
            str: A unique name for the component, including the stage suffix.
                 - If base_name is "X1bottom1", result is "X1bottom11" (for stage 1)
                 - If base_name is "cylinder11", result is "cylinder111" (for stage 1)
                 - If base_name is "rod11", result is "rod111" (for stage 1)
                 - If base_name is "base_link", result is "base_link1" (for stage 1)
        """
        # For components that already have a number suffix in original URDF
        if base_name.endswith('1'):
            # Just add stage number
            return f"{base_name}{self.stage_suffix}"
        # For base_link and other components without number suffix
        return f"{base_name}{self.stage_suffix}"

    def get_joint_name(self, joint_num: int) -> str:
        """Get the name of a joint.

        Args:
            joint_num (int): The joint number.

        Returns:
            str: The name of the joint with appropriate prefix and stage suffix.
                 For slider joints (13-18), uses "Slider_" prefix.
                 For other joints, uses "Revolute_" prefix.
                 For example:
                 - If joint_num is 3, result is "Revolute_31" (for stage 1)
                 - If joint_num is 13, result is "Slider_131" (for stage 1)
        """
        # Use Slider_ prefix for joints 13-18, Revolute_ for others
        prefix = "Slider_" if 13 <= joint_num <= 18 else "Revolute_"
        return f"{prefix}{joint_num}{self.stage_suffix}"

    def get_transmission_name(self, joint_name: str) -> str:
        """Get the name of a transmission.

        Args:
            joint_name (str): The name of the joint associated with the transmission.

        Returns:
            str: The name of the transmission, which is the joint name with "_tran" appended.
                 For example:
                 - If joint_name is "Revolute_32", result is "Revolute_32_tran"
                 - If joint_name is "Slider_132", result is "Slider_132_tran"
        """
        return f"{joint_name}_tran"

    def strip_stage_suffix(self, name: str) -> str:
        """Remove stage suffix from a name.

        Args:
            name (str): The name from which to remove the stage suffix.

        Returns:
            str: The name without the stage suffix.
                 For example:
                 - If name is "cylinder611" and stage is 1, returns "cylinder61"
                 - If name is "rod611" and stage is 1, returns "rod61"
                 - If name is "X1bottom11" and stage is 1, returns "X1bottom1"
        """
        # Handle special cases for components that already have a number suffix
        if any(pattern in name for pattern in ['cylinder', 'rod', 'bottom']):
            # Extract the base name without the stage suffix
            match = re.match(r'([A-Za-z]+\d+)(\d+)', name)
            if match:
                return match.group(1)  # Return the base name with original number
        
        # For other components, just remove the stage suffix
        return name.rsplit(self.stage_suffix, 1)[0]

    def get_original_bottom_link_name(self, link_name: str) -> str:
        """Get the original name for a bottom link.

        This is used to retrieve properties from the original URDF.

        Args:
            link_name (str): The name of the bottom link with the stage suffix.

        Returns:
            str: The original name of the bottom link without the stage suffix.
                 For example, if link_name is "X1bottom12", the result is "X1bottom1".
        """
        if "bottom" in link_name:
            x_num = link_name.split('X')[1][0]  # Get the number after X
            return f"X{x_num}bottom1"
        return link_name

    def get_mesh_filename(self, base_name: str) -> str:
        """Get the mesh filename for a component.

        This method ensures that the correct mesh file is referenced based on the
        component's base name. The mesh filenames should be exactly the same as the
        file name without any stage suffix.

        Args:
            base_name (str): The base name of the component (e.g., "X1bottom1", "cylinder11", "rod11").

        Returns:
            str: The filename of the mesh.
                 - If base_name is "X1bottom1", result is "meshes/X1bottom1.stl"
                 - If base_name is "cylinder11", result is "meshes/cylinder11.stl"
                 - If base_name is "rod11", result is "meshes/rod11.stl"
                 - If base_name is "base_link", result is "meshes/base_link.stl"
        """
        return f"meshes/{base_name}.stl"

@dataclass
class LinkFactory:
    """Factory class for creating links with consistent properties."""
    name_manager: NameManager
    original_props: Dict[str, LinkProperties]
    base_z: float = 0

    def create_link(self, base_name: str, link_name: str = None) -> Link:
        """Create a link with properties from original URDF.
        
        Args:
            base_name: Original name from mesh files (e.g., "X1bottom1")
            link_name: Optional override for the generated link name
        """
        if link_name is None:
            link_name = self.name_manager.get_component_name(base_name)
            
        orig_props = self.original_props[base_name]
        inertial_props = orig_props.inertial
        
        # Create inertial origin
        link_origin = Origin(
            xyz=tuple(map(float, inertial_props['origin']['xyz'].split())),
            rpy=tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
        )
        
        if self.base_z:
            link_origin = self._apply_z_offset(link_origin)
        
        return Link(
            name=link_name,
            inertial=self._create_inertial(inertial_props, link_origin),
            visual=self._create_visual(orig_props.visual, base_name),
            collision=self._create_collision(orig_props.collision, base_name)
        )
    
    def _create_inertial(self, props: Dict, origin: Origin) -> Inertial:
        """Create inertial properties from original data."""
        return Inertial(
            origin=origin,
            mass=float(props['mass']),
            ixx=float(props['ixx']),
            iyy=float(props['iyy']),
            izz=float(props['izz']),
            ixy=float(props['ixy']),
            iyz=float(props['iyz']),
            ixz=float(props['ixz'])
        )
    
    def _create_visual(self, props: Dict, base_name: str) -> Visual:
        """Create visual properties from original data."""
        origin = Origin(
            xyz=tuple(map(float, props['origin']['xyz'].split())),
            rpy=tuple(map(float, props['origin']['rpy'].split())) if 'rpy' in props['origin'] else (0, 0, 0)
        )
        if self.base_z:
            origin = self._apply_z_offset(origin)
        return Visual(
            origin=origin,
            geometry=Geometry(self.name_manager.get_mesh_filename(base_name))
        )
    
    def _create_collision(self, props: Dict, base_name: str) -> Collision:
        """Create collision properties from original data."""
        origin = Origin(
            xyz=tuple(map(float, props['origin']['xyz'].split())),
            rpy=tuple(map(float, props['origin']['rpy'].split())) if 'rpy' in props['origin'] else (0, 0, 0)
        )
        if self.base_z:
            origin = self._apply_z_offset(origin)
        return Collision(
            origin=origin,
            geometry=Geometry(self.name_manager.get_mesh_filename(base_name))
        )

    def _apply_z_offset(self, origin: Origin) -> Origin:
        """Apply the base_z offset to an origin."""
        return Origin(
            (origin.xyz[0], origin.xyz[1], origin.xyz[2] + self.base_z),
            origin.rpy
        )

@dataclass
class JointFactory:
    """Factory class for creating joints with consistent properties."""
    name_manager: NameManager
    original_props: Dict[str, JointProperties]
    
    def create_joint(self, joint_num: int, parent: str, child: str) -> Joint:
        """Create a joint with properties from original URDF.
        
        Args:
            joint_num: Joint number from Link_graph.txt
            parent: Parent link name
            child: Child link name
        """
        joint_name = self.name_manager.get_joint_name(joint_num)
        base_name = self.name_manager.strip_stage_suffix(joint_name)
            
        # Get original properties
        if base_name.startswith('Slider_'):
            orig_props = self.original_props[base_name]
        else:
            orig_props = self.original_props[f"Revolute_{joint_num}"]
        
        # Create origin
        joint_origin = Origin(
            xyz=tuple(map(float, orig_props.origin['xyz'].split())),
            rpy=tuple(map(float, orig_props.origin['rpy'].split())) if 'rpy' in orig_props.origin else (0, 0, 0)
        )
        
        return Joint(
            name=joint_name,
            joint_type=orig_props.joint_type,
            parent=parent,
            child=child,
            origin=joint_origin,
            axis=tuple(map(float, orig_props.axis.split())),
            limits=orig_props.limits,
            transmission=Transmission(
                self.name_manager.get_transmission_name(joint_name),
                joint_name
            )
        )

@dataclass
class StewartPlatformConfig:
    """Configuration data for Stewart platform components."""
        # Bottom link configurations based on Link_graph.txt and mesh files
    bottom_configs = [
        # (name, joint_num)  # Joint numbers from Link_graph.txt
        ("X1bottom1", 1),  # Use exact name from X1bottom1.stl
        ("X6bottom1", 2),  # Use exact name from X6bottom1.stl
        ("X5bottom1", 3),  # Use exact name from X5bottom1.stl
        ("X2bottom1", 4),  # Use exact name from X2bottom1.stl
        ("X4bottom1", 5),  # Use exact name from X4bottom1.stl
        ("X3bottom1", 6)   # Use exact name from X3bottom1.stl
    ]
    
    # Cylinder configurations based on Link_graph.txt and mesh files
    cylinder_configs = [
        # (cylinder_name, parent_bottom_link, joint_num)
        ("cylinder61", "X6bottom1", 7),  # Use exact name from cylinder61.stl
        ("cylinder51", "X5bottom1", 8),  # Use exact name from cylinder51.stl
        ("cylinder11", "X1bottom1", 9),  # Use exact name from cylinder11.stl
        ("cylinder21", "X2bottom1", 10), # Use exact name from cylinder21.stl
        ("cylinder31", "X3bottom1", 11), # Use exact name from cylinder31.stl
        ("cylinder41", "X4bottom1", 12)  # Use exact name from cylinder41.stl
    ]

    # Rod configurations based on Link_graph.txt and mesh files
    rod_configs = [
        # (rod_name, parent_cylinder, joint_num)
        ("rod11", "cylinder11", 13),  # Use exact name from rod11.stl
        ("rod21", "cylinder21", 14),  # Use exact name from rod21.stl
        ("rod31", "cylinder31", 15),  # Use exact name from rod31.stl
        ("rod41", "cylinder41", 16),  # Use exact name from rod41.stl
        ("rod51", "cylinder51", 17),  # Use exact name from rod51.stl
        ("rod61", "cylinder61", 18)   # Use exact name from rod61.stl
    ]

# Piston configurations based on Link_graph.txt and mesh files
    piston_configs = [
    # (piston_name, parent_rod, joint_num)
    ("piston11", "rod11", 19),  # Use exact name from piston11.stl
    ("piston21", "rod21", 20),  # Use exact name from piston21.stl
    ("piston31", "rod31", 21),  # Use exact name from piston31.stl
    ("piston61", "rod61", 22),  # Use exact name from piston61.stl
    ("piston51", "rod51", 23),  # Use exact name from piston51.stl
    ("piston41", "rod41", 24)   # Use exact name from piston41.stl
]
    
    # Top link configurations based on Link_graph.txt and mesh files
    top_configs = [
    # (top_name, parent_piston, joint_num)
        ("X1top1", "piston11", 26),  # Use exact name from X1top1.stl
        ("X2top1", "piston21", 35),  # Use exact name from X2top1.stl
        ("X3top1", "piston31", 33),  # Use exact name from X3top1.stl
        ("X4top1", "piston41", 27),  # Use exact name from X4top1.stl
        ("X5top1", "piston51", 28),  # Use exact name from X5top1.stl
        ("X6top1", "piston61", 25)   # Use exact name from X6top1.stl
    ]

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
        self.name_manager = NameManager(stage, base_prefix)
        
        # Load properties from original URDF
        self.original_joint_props = extract_joint_properties("Stewart.urdf")
        self.original_link_props = extract_link_properties("Stewart.urdf")
        
        # Create factories
        self.link_factory = LinkFactory(self.name_manager, self.original_link_props, self.base_z)
        self.joint_factory = JointFactory(self.name_manager, self.original_joint_props)
        
        # Initialize component lists
        self.links: List[Link] = []
        self.joints: List[Joint] = []
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all components in the correct order."""
        self._init_base_link()
        self._init_bottom_links()
        self._init_cylinder_links()
        self._init_rod_links()
        self._init_piston_links()
        self._init_top_links()
    
    def _init_base_link(self):
        """Initialize the base link."""
        base_link = self.link_factory.create_link("base_link", self.name_manager.get_base_link_name())
        self.links.append(base_link)
    
    def _init_bottom_links(self):
        """Initialize bottom links and their joints with base."""
        for base_name, joint_num in StewartPlatformConfig.bottom_configs:
            # Create link
            link = self.link_factory.create_link(base_name)
            self.links.append(link)
            
            # Create joint connecting to base
            joint = self.joint_factory.create_joint(
                joint_num,
                self.name_manager.get_base_link_name(),
                self.name_manager.get_component_name(base_name)
            )
            self.joints.append(joint)
    
    def _init_cylinder_links(self):
        """Initialize cylinder links and their joints with bottom links."""
        for base_name, parent_base, joint_num in StewartPlatformConfig.cylinder_configs:
            # Create link
            link = self.link_factory.create_link(base_name)
            self.links.append(link)
            
            # Create joint
            joint = self.joint_factory.create_joint(
                joint_num,
                self.name_manager.get_component_name(parent_base),
                self.name_manager.get_component_name(base_name)
            )
            self.joints.append(joint)
    
    def _init_rod_links(self):
        """Initialize rod links and their slider joints with cylinders."""
        for base_name, parent_base, joint_num in StewartPlatformConfig.rod_configs:
            # Create link
            link = self.link_factory.create_link(base_name)
            self.links.append(link)
            
            # Create joint
            joint = self.joint_factory.create_joint(
                joint_num,
                self.name_manager.get_component_name(parent_base),
                self.name_manager.get_component_name(base_name)
            )
            self.joints.append(joint)
    
    def _init_piston_links(self):
        """Initialize piston links and their joints with rods."""
        for base_name, parent_base, joint_num in StewartPlatformConfig.piston_configs:
            # Create link
            link = self.link_factory.create_link(base_name)
            self.links.append(link)
            
            # Create joint
            joint = self.joint_factory.create_joint(
                joint_num,
                self.name_manager.get_component_name(parent_base),
                self.name_manager.get_component_name(base_name)
            )
            self.joints.append(joint)
    
    def _init_top_links(self):
        """Initialize top links and their joints with pistons."""
        for base_name, parent_base, joint_num in StewartPlatformConfig.top_configs:
            # Create link
            link = self.link_factory.create_link(base_name)
            self.links.append(link)
            
            # Create joint
            joint = self.joint_factory.create_joint(
                joint_num,
                self.name_manager.get_component_name(parent_base),
                self.name_manager.get_component_name(base_name)
                )
            self.joints.append(joint)

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
            
        # Add all joints
        for joint in self.joints:
            xml_parts.append(joint.to_xml())
            
        xml_parts.append('</robot>')
        urdf_str = '\n'.join(xml_parts)
        
        # Validate and clean up XML
        try:
            root = ET.fromstring(urdf_str)
            clean_xml = ET.tostring(root, encoding='unicode', method='xml')
            return clean_xml
        except ET.ParseError as e:
            print(f"[XML Error] Failed to parse generated XML: {e}")
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