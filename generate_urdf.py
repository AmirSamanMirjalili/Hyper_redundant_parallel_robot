from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
from extract_urdf_properties import extract_joint_properties, extract_link_properties
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
                joint_num = int(base_name.split('_')[1])
            elif base_name.startswith('Revolute_'):
                # For Revolute_ joints, check if it's actually a slider joint (13-18)
                joint_num = int(base_name.split('_')[1])
                if 13 <= joint_num <= 18:
                    original_name = f"Slider_{joint_num}"
                else:
                    original_name = base_name
            else:
                return None

            print(f"[Joint Lookup] {joint_name} -> {original_name}")
            
            # Get the original properties
            if original_name in self.original_joint_props:
                props = self.original_joint_props[original_name]
                print(f"[Joint Props] Found properties for {original_name}: type={props.joint_type}")
                
                # Update parent/child links to use stage-appropriate names
                if props.parent == "base_link":
                    props.parent = self.name_mgr.get_base_link_name()
                else:
                    props.parent = self.name_mgr.get_component_name(props.parent)
                    
                if "bottom" in props.child:
                    props.child = self.name_mgr.get_component_name(props.child.rsplit('1', 1)[0])
                else:
                    props.child = self.name_mgr.get_component_name(props.child)
                
                return props
            
            print(f"[Joint Props] No properties found for {original_name}")
            return None
        except (ValueError, IndexError) as e:
            print(f"[Joint Error] Failed to process joint name {joint_name}: {str(e)}")
            return None

    def _get_original_link_props(self, link_name: str) -> Optional[Dict]:
        """Get properties of a link from the original URDF."""
        # Get original name
        if "base_link" in link_name:
            base_name = "base_link"
        elif "bottom" in link_name:
            base_name = self.name_mgr.get_original_bottom_link_name(link_name)
        elif "rod" in link_name or "cylinder" in link_name:
            # For rod and cylinder links, strip stage suffix to get original name
            base_name = self.name_mgr.strip_stage_suffix(link_name)
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
        print("\n[Cylinder Links] Starting cylinder link initialization")
        
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

        for base_name, parent_base, joint_num in cylinder_configs:
            # Generate names for this stage using NameManager
            link_name = self.name_mgr.get_component_name(base_name)
            joint_name = self.name_mgr.get_joint_name(joint_num)
            parent_name = self.name_mgr.get_component_name(parent_base)
            
            print(f"\n[Cylinder Link] Processing cylinder {base_name}:")
            print(f"  Generated names:")
            print(f"    Link: {link_name}")
            print(f"    Joint: {joint_name}")
            print(f"    Parent: {parent_name}")
            
            # Get original properties
            original_link_props = self._get_original_link_props(link_name)
            original_joint_props = self._get_original_joint_props(joint_name)
            
            print(f"  Original properties:")
            print(f"    Link props found: {original_link_props is not None}")
            print(f"    Joint props found: {original_joint_props is not None}")
            
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
                print(f"  Added cylinder link: {link_name}")

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
                print(f"  Added joint: {joint_name} ({joint.joint_type})")
                print(f"    Parent: {joint.parent}")
                print(f"    Child: {joint.child}")
                print(f"    Axis: {joint.axis}")
                if joint.limits:
                    print(f"    Limits: {joint.limits}")
            else:
                print(f"  ERROR: Missing properties for cylinder {base_name}")
                if not original_link_props:
                    print(f"    Missing link properties")
                if not original_joint_props:
                    print(f"    Missing joint properties")

    def _init_rod_links(self):
        """Initialize the rod links and their slider joints with cylinders."""
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

        for base_name, parent_base, joint_num in rod_configs:
            # Generate names for this stage using NameManager
            link_name = self.name_mgr.get_component_name(base_name)
            joint_name = self.name_mgr.get_joint_name(joint_num)  # Will return Slider_XX for these joints
            parent_name = self.name_mgr.get_component_name(parent_base)
            
            print(f"\n[Rod Joint] Processing joint {joint_num}:")
            print(f"[Rod Joint] Generated names: joint={joint_name}, link={link_name}, parent={parent_name}")
            
            # Get original properties
            original_link_props = self._get_original_link_props(link_name)
            original_joint_props = self._get_original_joint_props(joint_name)  # Will handle Slider_ prefix internally
            
            if original_link_props and original_joint_props:
                # Create rod link
                inertial_props = original_link_props.inertial
                link_origin = Origin(
                    xyz=tuple(map(float, inertial_props['origin']['xyz'].split())),
                    rpy=tuple(map(float, inertial_props['origin']['rpy'].split())) if 'rpy' in inertial_props['origin'] else (0, 0, 0)
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
                            xyz=tuple(map(float, original_link_props.visual['origin']['xyz'].split())),
                            rpy=tuple(map(float, original_link_props.visual['origin']['rpy'].split())) if 'rpy' in original_link_props.visual['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    ),
                    collision=Collision(
                        origin=Origin(
                            xyz=tuple(map(float, original_link_props.collision['origin']['xyz'].split())),
                            rpy=tuple(map(float, original_link_props.collision['origin']['rpy'].split())) if 'rpy' in original_link_props.collision['origin'] else (0, 0, 0)
                        ),
                        geometry=Geometry(self.name_mgr.get_mesh_filename(base_name))
                    )
                )
                self.links.append(rod_link)

                # Create slider joint connecting cylinder to rod
                joint_origin = Origin(
                    xyz=tuple(map(float, original_joint_props.origin['xyz'].split())),
                    rpy=tuple(map(float, original_joint_props.origin['rpy'].split())) if 'rpy' in original_joint_props.origin else (0, 0, 0)
                )
                # Create the slider joint with explicit type and properties
                joint = Joint(
                    name=joint_name,  # Using NameManager's joint name (Slider_XX)
                    joint_type="prismatic",  # Explicitly set to prismatic for slider joints
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
                print(f"[Rod Joint] Created joint: name={joint.name}, type={joint.joint_type}, parent={joint.parent}, child={joint.child}")
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
            
        # Add all joints with debug output
        print("\n[XML Generation] Adding joints:")
        for joint in self.joints:
            joint_xml = joint.to_xml()
            print(f"[XML Generation] Joint {joint.name} ({joint.joint_type})")
            xml_parts.append(joint_xml)
            
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