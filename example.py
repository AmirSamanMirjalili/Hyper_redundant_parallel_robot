"""
Example usage of Stewart Platform implementation.

This script demonstrates how to use the Stewart Platform controller
to perform various movements and trajectories with multiple stages.
"""

import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from dataclasses import dataclass
from config import StewartConfig, GeometryConfig, ActuatorConfig, MotionConfig
from platform_controller import StewartPlatformController
from generate_urdf import generate_stewart_platform, NameManager
from typing import List, Dict, Tuple, Optional

@dataclass
class StageConfig:
    """Configuration for a single stage of the Stewart Platform."""
    stage_num: int
    prefix: str
    z_offset: float
    base_position: List[float]
    is_fixed: bool

class MultiStageStewartPlatform:
    """Manages multiple stages of Stewart Platform."""
    
    def __init__(self, num_stages: int = 2, stage_spacing: float = 0.5):
        """Initialize multi-stage Stewart Platform.
        
        Args:
            num_stages: Number of Stewart Platform stages
            stage_spacing: Vertical spacing between stages in meters
        """
        self.num_stages = num_stages
        self.stage_spacing = stage_spacing
        self.controllers: Dict[int, StewartPlatformController] = {}
        self.urdf_path: Optional[str] = None
        self.robot_id: Optional[int] = None
        
        # Initialize simulation
        self._init_simulation()
        self._setup_robot()
        self._create_controllers()
    
    def _init_simulation(self) -> None:
        """Initialize PyBullet simulation environment."""
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Configure camera view based on number of stages
        self._setup_camera()
        
        # Load ground plane
        p.loadURDF("plane.urdf")
    
    def _setup_camera(self) -> None:
        """Configure camera view based on number of stages."""
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0 + 0.5 * self.num_stages,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.25 * self.num_stages]
        )
    
    def _get_stage_configs(self) -> List[StageConfig]:
        """Generate configuration for each stage.
        
        Returns:
            List of stage configurations
        """
        configs = []
        for i in range(self.num_stages):
            stage_num = i + 1
            prefix = "lower_" if i == 0 else f"upper_{i}_"
            z_offset = i * self.stage_spacing
            
            configs.append(StageConfig(
                stage_num=stage_num,
                prefix=prefix,
                z_offset=self.stage_spacing,
                base_position=[0, 0, 0.1 + z_offset],
                is_fixed=(i == 0)
            ))
        return configs
    
    def _setup_robot(self) -> None:
        """Generate and load URDF for all stages."""
        # Generate URDF stages configuration
        stages = [(cfg.stage_num, cfg.prefix, cfg.z_offset) 
                 for cfg in self._get_stage_configs()]
        
        # Generate and save URDF
        urdf_string, self.all_joint_pairs = generate_stewart_platform(stages)
        self.urdf_path = "stewart_combined.urdf"
        
        with open(self.urdf_path, "w") as f:
            f.write(urdf_string)
            
        # Load the robot
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 0],
            useFixedBase=1
        )
    
    def _find_stage_actuator_indices(self, stage_config: StageConfig) -> List[int]:
        """Find actuator indices for a specific stage by analyzing joint connections.
        
        This method finds slider joints (actuators) by:
        1. Collecting all joint information for the stage
        2. Identifying slider (prismatic) joints
        3. Finding their connected revolute joints
        4. Validating the connections match expected stage configuration
        
        Args:
            stage_config: Configuration for the stage
            
        Returns:
            List of actuator indices for the stage
        """
        name_manager = NameManager(stage_config.stage_num, stage_config.prefix)
        stage_joints = {}
        stage_slider_joints = []
        joint_pairs = []
        
        print(f"\nAnalyzing joints for stage {stage_config.stage_num}...")
        
        # First pass: Collect joint information for this stage
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Handle parent/child link names
            parent_link = (joint_info[12].decode('utf-8') 
                         if isinstance(joint_info[12], bytes) 
                         else str(joint_info[12]))
            child_link = (joint_info[16].decode('utf-8') 
                         if isinstance(joint_info[16], bytes) 
                         else str(joint_info[16]))
            
            # Check if this joint belongs to our stage
            if name_manager.stage_suffix in joint_name:
                stage_joints[i] = {
                    'name': joint_name,
                    'type': joint_type,
                    'parent': parent_link,
                    'child': child_link
                }
                print(f"Found stage joint: {joint_name} (Type: {joint_type})")
                
                # Collect slider joints
                if joint_type == p.JOINT_PRISMATIC:
                    stage_slider_joints.append(i)
                    print(f"Found actuator: {joint_name}")
        
        # Second pass: Find connected revolute joints for each slider
        for slider_idx in stage_slider_joints:
            slider_info = stage_joints[slider_idx]
            parent_link = slider_info['parent']
            child_link = slider_info['child']
            
            # Find connected revolute joints
            for i, info in stage_joints.items():
                if info['type'] == p.JOINT_REVOLUTE:
                    # Check if this revolute joint connects to our slider
                    if info['child'] == parent_link:
                        joint_pairs.append((i, slider_idx))
                        break
        
        # Sort joint pairs based on slider indices
        joint_pairs.sort(key=lambda x: x[1])
        stage_slider_joints.sort()
        
        print(f"\nJoint pairs for stage {stage_config.stage_num}:")
        for revolute, slider in joint_pairs:
            print(f"Revolute {revolute} ({stage_joints[revolute]['name']}) -> "
                  f"Slider {slider} ({stage_joints[slider]['name']})")
        
        print(f"Found {len(stage_slider_joints)} actuators: {stage_slider_joints}")
        
        if not stage_slider_joints:
            print(f"Warning: No slider joints found for stage {stage_config.stage_num}")
        
        return stage_slider_joints
    
    def _get_base_config(self) -> StewartConfig:
        """Get base configuration for Stewart Platform.
        
        Returns:
            Base configuration for all stages
        """
        return StewartConfig(
            geometry=GeometryConfig(
                radius_base=0.2,
                radius_platform=0.15,
                gamma_base=30.0,
                gamma_platform=25.0
            ),
            actuator=ActuatorConfig(
                max_force=250.0,
                pwm_frequency=50.0,
                duty_cycle=0.5,
                min_length=0.15,
                max_length=0.35
            ),
            motion=MotionConfig(
                max_translation=(0.05, 0.05, 0.05),
                max_rotation=(15.0, 15.0, 15.0),
                max_velocity=0.1,
                max_acceleration=0.2
            )
        )
    
    def _create_controllers(self) -> None:
        """Create controllers for each stage."""
        base_config = self._get_base_config()
        print("\nCreating controllers for each stage...")
        
        for stage_config in self._get_stage_configs():
            print(f"\nSetting up stage {stage_config.stage_num}...")
            
            # Find actuator indices
            actuator_indices = self._find_stage_actuator_indices(stage_config)
            if not actuator_indices:
                print(f"Skipping stage {stage_config.stage_num} due to missing actuators")
                continue
            
            # Create controller
            print(f"Creating controller with actuators: {actuator_indices}")
            controller = StewartPlatformController(
                config=base_config,
                urdf_path=self.urdf_path,
                robot_id=self.robot_id,
                actuator_indices=actuator_indices,
                base_position=stage_config.base_position,
                fixed_base=stage_config.is_fixed,
                stage=stage_config.stage_num,
                base_prefix=stage_config.prefix
            )
            
            # Setup constraints
            print(f"Setting up constraints for stage {stage_config.stage_num}...")
            controller.set_joint_pairs(self.all_joint_pairs[stage_config.stage_num - 1])
            controller.set_constraints()
            
            self.controllers[stage_config.stage_num] = controller
            print(f"Stage {stage_config.stage_num} setup complete")
    
    def move_stage(
        self,
        stage: int,
        translation: np.ndarray,
        rotation: np.ndarray,
        duration: float,
        use_pwm: bool = False
    ) -> bool:
        """Move a specific stage to desired pose."""
        if stage not in self.controllers:
            print(f"Stage {stage} does not exist")
            return False
        
        return self.controllers[stage].move_to_pose(
            translation,
            rotation,
            duration,
            use_pwm=use_pwm
        )
    
    def move_all_stages(
        self,
        translations: List[np.ndarray],
        rotations: List[np.ndarray],
        durations: List[float],
        use_pwm: bool = False
    ) -> bool:
        """Move all stages to desired poses."""
        if not self._validate_motion_inputs(translations, rotations, durations):
            return False
        
        success = True
        for stage in range(1, self.num_stages + 1):
            idx = stage - 1
            if not self.move_stage(
                stage,
                translations[idx],
                rotations[idx],
                durations[idx],
                use_pwm=use_pwm
            ):
                success = False
                break
        
        return success
    
    def _validate_motion_inputs(
        self,
        translations: List[np.ndarray],
        rotations: List[np.ndarray],
        durations: List[float]
    ) -> bool:
        """Validate motion inputs match number of stages."""
        if (len(translations) != self.num_stages or
            len(rotations) != self.num_stages or
            len(durations) != self.num_stages):
            print("Number of poses must match number of stages")
            return False
        return True
    
    def reset_all_stages(self, duration: float = 2.0) -> None:
        """Reset all stages to home position."""
        for controller in self.controllers.values():
            controller.reset_position(duration)
    
    def cleanup(self) -> None:
        """Clean up all controllers and resources."""
        for controller in self.controllers.values():
            controller.cleanup()
        p.disconnect()
        if self.urdf_path and os.path.exists(self.urdf_path):
            os.remove(self.urdf_path)

def run_example_movements(platform: MultiStageStewartPlatform) -> None:
    """Run example movements demonstrating platform capabilities."""
    print("\nWaiting for simulation to stabilize...")
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Run verification test first
    run_verification_test(platform)
    
    # Example 1: Individual stage movements
    print("\nExample 1: Moving individual stages")
    _run_individual_movements(platform)
    
    # Reset stages
    print("\nResetting all stages...")
    platform.reset_all_stages(duration=3.0)
    time.sleep(3)
    
    # Example 2: Coordinated movements
    print("\nExample 2: Coordinated movement")
    _run_coordinated_movements(platform)
    
    # Final reset
    print("\nResetting all stages...")
    platform.reset_all_stages(duration=3.0)

def _run_individual_movements(platform: MultiStageStewartPlatform) -> None:
    """Run individual stage movement examples."""
    for stage in range(1, platform.num_stages + 1):
        if stage not in platform.controllers:
            print(f"Stage {stage} controller not available, skipping...")
            continue
        
        print(f"\nMoving stage {stage}...")
        
        # Move up (smaller movement)
        print("Step 1: Moving up")
        translation = np.array([0, 0, 0.02])  # Reduced height
        rotation = np.zeros(3)
        success = platform.move_stage(
            stage,
            translation=translation,
            rotation=rotation,
            duration=2.0
        )
        if not success:
            print(f"Failed to move stage {stage} up")
        time.sleep(2)
        
        # Verify kinematics
        print(f"\nVerifying kinematics for stage {stage} (up position)...")
        platform.controllers[stage].verify_kinematics(translation, rotation)
        
        # Rotate (smaller angle)
        print(f"\nStep 2: Rotating stage {stage}")
        translation = np.array([0, 0, 0.02])  # Keep height
        rotation = np.array([10, 0, 0])       # Reduced angle
        success = platform.move_stage(
            stage,
            translation=translation,
            rotation=rotation,
            duration=2.0
        )
        if not success:
            print(f"Failed to rotate stage {stage}")
        time.sleep(2)
        
        # Verify kinematics
        print(f"\nVerifying kinematics for stage {stage} (rotated position)...")
        platform.controllers[stage].verify_kinematics(translation, rotation)

def _run_coordinated_movements(platform: MultiStageStewartPlatform) -> None:
    """Run coordinated movement examples."""
    # Skip coordinated movements for single stage
    if platform.num_stages < 2:
        print("Skipping coordinated movements (requires multiple stages)")
        return
        
    durations = [2.0] * platform.num_stages
    
    # Wave motion
    translations = [
        np.array([0, 0, 0.02]),  # First stage up
        np.array([0, 0, 0.01])   # Second stage at base
    ]
    rotations = [
        np.array([0, 0, 0]),     # First stage level
        np.array([10, 0, 0])     # Second stage tilted
    ]
    
    print("Executing wave motion...")
    success = platform.move_all_stages(translations, rotations, durations)
    if not success:
        print("Failed to execute wave motion")
    time.sleep(3)
    
    # Verify kinematics for each stage
    print("\nVerifying kinematics for wave motion...")
    for stage in range(1, platform.num_stages + 1):
        print(f"\nStage {stage}:")
        platform.controllers[stage].verify_kinematics(
            translations[stage-1],
            rotations[stage-1]
        )
    
    # Inverse motion
    translations = [
        np.array([0, 0, 0.01]),  # First stage at base
        np.array([0, 0, 0.02])   # Second stage up
    ]
    rotations = [
        np.array([10, 0, 0]),    # First stage tilted
        np.array([0, 0, 0])      # Second stage level
    ]
    
    print("\nExecuting inverse motion...")
    success = platform.move_all_stages(translations, rotations, durations)
    if not success:
        print("Failed to execute inverse motion")
    time.sleep(3)
    
    # Verify kinematics for each stage
    print("\nVerifying kinematics for inverse motion...")
    for stage in range(1, platform.num_stages + 1):
        print(f"\nStage {stage}:")
        platform.controllers[stage].verify_kinematics(
            translations[stage-1],
            rotations[stage-1]
        )

def run_verification_test(platform: MultiStageStewartPlatform) -> None:
    """Run a comprehensive verification test of the inverse kinematics.
    
    This test moves each stage through a series of poses and verifies
    the actual platform pose matches the desired pose.
    """
    print("\nRunning comprehensive kinematics verification test...")
    
    # Test poses to verify
    test_poses = [
        # (translation, rotation, duration)
        (np.array([0, 0, 0.02]), np.zeros(3), 2.0),                  # Pure Z translation
        (np.array([0.01, 0, 0.02]), np.zeros(3), 2.0),              # X + Z translation
        (np.array([0, 0.01, 0.02]), np.zeros(3), 2.0),              # Y + Z translation
        (np.array([0, 0, 0.02]), np.array([10, 0, 0]), 2.0),        # Z + Roll
        (np.array([0, 0, 0.02]), np.array([0, 10, 0]), 2.0),        # Z + Pitch
        (np.array([0, 0, 0.02]), np.array([0, 0, 10]), 2.0),        # Z + Yaw
        (np.array([0.01, 0.01, 0.02]), np.array([5, 5, 5]), 2.0),   # Combined motion
    ]
    
    for stage in range(1, platform.num_stages + 1):
        if stage not in platform.controllers:
            print(f"Stage {stage} controller not available, skipping...")
            continue
            
        print(f"\nTesting stage {stage}...")
        
        for i, (translation, rotation, duration) in enumerate(test_poses, 1):
            print(f"\nTest pose {i}:")
            print(f"Translation: {translation}")
            print(f"Rotation: {rotation}")
            
            # Move to pose
            success = platform.move_stage(
                stage,
                translation,
                rotation,
                duration
            )
            
            if not success:
                print(f"Failed to move to test pose {i}")
                continue
                
            # Let the platform settle
            time.sleep(duration)
            
            # Verify kinematics
            platform.controllers[stage].verify_kinematics(translation, rotation)
            
        # Reset stage position
        platform.controllers[stage].reset_position()
        time.sleep(2)

def main():
    """Main function demonstrating multi-stage Stewart Platform usage."""
    print("\nInitializing Stewart Platform...")
    # Initialize with proper spacing even for single stage
    platform = MultiStageStewartPlatform(num_stages=1, stage_spacing=0.5)
    
    try:
        run_example_movements(platform)
        
        print("\nSimulation running. Press Ctrl+C to exit.")
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        platform.cleanup()

if __name__ == "__main__":
    main() 