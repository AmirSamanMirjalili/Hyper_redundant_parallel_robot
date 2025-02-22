"""
Platform controller module for Stewart Platform.

This module provides high-level control of the Stewart Platform by integrating
kinematics, geometry, and actuator control.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import pybullet as p
from config import StewartConfig
from kinematics import KinematicsSolver
from actuator import ActuatorController
from generate_urdf import NameManager

class StewartPlatformController:
    """High-level controller for Stewart Platform."""
    
    def __init__(
        self,
        config: StewartConfig,
        urdf_path: str,
        actuator_indices: List[int],
        robot_id: Optional[int] = None,
        base_position: List[float] = [0, 0, 0.1],
        base_orientation: List[float] = [0, 0, 0, 1],
        fixed_base: bool = True,
        stage: int = 1,
        base_prefix: str = ""
    ):
        """Initialize Stewart Platform controller.
        
        Args:
            config: Complete platform configuration
            urdf_path: Path to platform URDF file
            actuator_indices: List of joint indices for actuators
            robot_id: Optional existing PyBullet body ID. If None, will load URDF
            base_position: Initial base position [x, y, z]
            base_orientation: Initial base orientation [x, y, z, w]
            fixed_base: Whether to fix the base
            stage: Stage number for this controller
            base_prefix: Prefix for base link name
        """
        self.config = config
        self.kinematics = KinematicsSolver(config)
        self.constraints = []  # Store constraint IDs
        self.joint_pairs = []  # Store joint pairs
        self.actuator_indices = actuator_indices
        
        # Initialize name manager
        self.name_manager = NameManager(stage, base_prefix)
        
        # Use existing robot ID or load new one
        if robot_id is not None:
            self.robot_id = robot_id
        else:
            # Load robot in PyBullet
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=base_position,
                baseOrientation=base_orientation,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                useFixedBase=fixed_base
            )
        
        # Initialize actuator controller
        self.actuator_controller = ActuatorController(
            self.robot_id,
            actuator_indices,
            config.actuator
        )
        
        # Initialize state
        self._current_translation = np.zeros(3)
        self._current_rotation = np.zeros(3)
        
        # Create mapping of joint names to indices
        self.joint_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            self.joint_name_to_id[info[1].decode('utf-8')] = i
    
    def set_joint_pairs(self, joint_pairs: List[Tuple[int, int]]) -> None:
        """Set the joint pairs for constraints.
        
        Args:
            joint_pairs: List of tuples containing (parent_joint, child_joint) indices
        """
        # Convert joint numbers to actual joint names and then to PyBullet indices
        converted_pairs = []
        for parent_num, child_num in joint_pairs:
            parent_name = f"Rigid_{parent_num}"
            child_name = f"Rigid_{child_num}"
            
            if parent_name in self.joint_name_to_id and child_name in self.joint_name_to_id:
                parent_id = self.joint_name_to_id[parent_name]
                child_id = self.joint_name_to_id[child_name]
                converted_pairs.append((parent_id, child_id))
            else:
                print(f"Warning: Could not find rigid joint pair {parent_name} -> {child_name}")
                print(f"Available joints: {sorted(self.joint_name_to_id.keys())}")
        
        self.joint_pairs = converted_pairs
    
    def set_constraints(self) -> None:
        """Create fixed constraints between joint pairs and disable motors."""
        # Remove any existing constraints for this controller
        self.cleanup()
        
        # Create a fixed constraint for each pair of linked joints
        for parent_id, child_id in self.joint_pairs:
            try:
                # Get joint positions
                parent_state = p.getJointState(self.robot_id, parent_id)
                child_state = p.getJointState(self.robot_id, child_id)
                parent_info = p.getJointInfo(self.robot_id, parent_id)
                child_info = p.getJointInfo(self.robot_id, child_id)
                
                # Create constraint at current positions
                constraint_id = p.createConstraint(
                    self.robot_id, parent_id,
                    self.robot_id, child_id,
                    p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0]
                )
                
                # Set constraint parameters
                p.changeConstraint(constraint_id, maxForce=1e20)
                self.constraints.append(constraint_id)
                
                print(f"Created constraint between {parent_info[1].decode('utf-8')} and {child_info[1].decode('utf-8')}")
                
            except p.error as e:
                print(f"Failed to create constraint between joints {parent_id} and {child_id}")
                print(f"Parent joint info: {p.getJointInfo(self.robot_id, parent_id)}")
                print(f"Child joint info: {p.getJointInfo(self.robot_id, child_id)}")
                raise e

        # Disable the motors for constrained joints
        for parent_id, child_id in self.joint_pairs:
            p.setJointMotorControl2(
                self.robot_id,
                parent_id,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )
            p.setJointMotorControl2(
                self.robot_id,
                child_id,
                controlMode=p.VELOCITY_CONTROL,
                force=0
            )
        
        # Enable motors for actuators
        for idx in self.actuator_indices:
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                force=self.config.actuator.max_force
            )
        
    def move_to_pose(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        duration: float,
        use_pwm: bool = True,
        check_limits: bool = True
    ) -> bool:
        """Move platform to desired pose.
        
        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation angles [roll, pitch, yaw] in degrees
            duration: Movement duration in seconds
            use_pwm: Whether to use PWM control
            check_limits: Whether to check motion and actuator limits
            
        Returns:
            True if movement successful, False otherwise
        """
        try:
            # Solve inverse kinematics
            leg_lengths, _ = self.kinematics.solve_inverse(
                translation,
                rotation,
                check_limits=check_limits
            )
            
            # Move actuators
            self.actuator_controller.move_to_lengths(
                leg_lengths,
                duration,
                use_pwm=use_pwm
            )
            
            # Update current pose
            self._current_translation = translation
            self._current_rotation = rotation
            
            return True
            
        except ValueError as e:
            print(f"Movement failed: {e}")
            return False
            
    def execute_trajectory(
        self,
        poses: List[Tuple[np.ndarray, np.ndarray, float]],
        use_pwm: bool = True,
        check_limits: bool = True
    ) -> bool:
        """Execute a sequence of poses.
        
        Args:
            poses: List of (translation, rotation, duration) tuples
            use_pwm: Whether to use PWM control
            check_limits: Whether to check motion and actuator limits
            
        Returns:
            True if all movements successful, False otherwise
        """
        for translation, rotation, duration in poses:
            if not self.move_to_pose(
                translation,
                rotation,
                duration,
                use_pwm=use_pwm,
                check_limits=check_limits
            ):
                return False
        return True
    
    def reset_position(self, duration: float = 2.0) -> None:
        """Reset platform to home position.
        
        Args:
            duration: Movement duration in seconds
        """
        self.move_to_pose(
            translation=np.zeros(3),
            rotation=np.zeros(3),
            duration=duration,
            use_pwm=False,
            check_limits=False
        )
        
    @property
    def current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current platform pose.
        
        Returns:
            Tuple of current translation and rotation
        """
        return self._current_translation, self._current_rotation
    
    def get_platform_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current platform position and orientation from PyBullet.
        
        Returns:
            Tuple of position and orientation (quaternion)
        """
        # Find the TOP1 link name for this stage
        top_link_name = self.name_manager.get_component_name("TOP1")
        print(f"\nLooking for TOP1 link: {top_link_name}")
        
        # Get all link indices and names
        link_indices = {}
        print("\nAvailable links:")
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode('utf-8')  # Link name is at index 12
            link_indices[link_name] = i
            print(f"  {i}: {link_name}")
        
        # Find TOP1 link
        if top_link_name not in link_indices:
            raise ValueError(f"Could not find TOP1 link: {top_link_name}\nAvailable links: {sorted(link_indices.keys())}")
        
        # Get the state of the TOP1 link
        link_state = p.getLinkState(self.robot_id, link_indices[top_link_name])
        return np.array(link_state[0]), np.array(link_state[1])  # position, orientation (quaternion)
    
    def verify_kinematics(self, desired_translation: np.ndarray, desired_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Verify inverse kinematics by comparing desired pose with actual platform pose.
        
        This method:
        1. Gets the current platform pose from PyBullet
        2. Converts the quaternion orientation to Euler angles
        3. Compares with desired pose
        4. Returns the position and orientation errors
        
        Args:
            desired_translation: Desired translation vector [x, y, z]
            desired_rotation: Desired rotation angles [roll, pitch, yaw] in degrees
            
        Returns:
            Tuple of:
            - Translation error vector [x, y, z] in meters
            - Rotation error vector [roll, pitch, yaw] in degrees
        """
        # Get current platform pose
        actual_pos, actual_quat = self.get_platform_state()
        
        # Convert quaternion to Euler angles (in degrees)
        actual_euler = np.array(p.getEulerFromQuaternion(actual_quat)) * 180.0 / np.pi
        
        # Calculate errors
        translation_error = desired_translation - actual_pos
        rotation_error = desired_rotation - actual_euler
        
        # Print verification results
        print("\nKinematics Verification Results:")
        print(f"Translation (meters):")
        print(f"  Desired:  {desired_translation}")
        print(f"  Actual:   {actual_pos}")
        print(f"  Error:    {translation_error}")
        print(f"  Abs Error: {np.abs(translation_error)}")
        print(f"\nRotation (degrees):")
        print(f"  Desired:  {desired_rotation}")
        print(f"  Actual:   {actual_euler}")
        print(f"  Error:    {rotation_error}")
        print(f"  Abs Error: {np.abs(rotation_error)}")
        
        return translation_error, rotation_error
    
    def cleanup(self) -> None:
        """Clean up constraints and connections."""
        for constraint_id in self.constraints:
            p.removeConstraint(constraint_id)
        self.constraints = [] 