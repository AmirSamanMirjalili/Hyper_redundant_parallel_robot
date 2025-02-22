import pybullet as p
import pybullet_data
import time
import os
from generate_urdf import generate_stewart_platform, save_urdf

class StewartPlatform:
    def __init__(self, urdf_path, base_position=[0, 0, 0.1], base_orientation=[0, 0, 0, 1], fixed_base=1):
        """Initialize the Stewart Platform.
        
        Args:
            urdf_path (str): Path to the URDF file
            base_position (list): Initial base position [x, y, z]
            base_orientation (list): Initial base orientation as quaternion [x, y, z, w]
            fixed_base (int): Whether to fix the base (1) or not (0)
        """
        self.urdf_path = urdf_path
        self.robotId = p.loadURDF(urdf_path, 
                                basePosition=base_position,
                                baseOrientation=base_orientation,
                                flags=p.URDF_USE_INERTIA_FROM_FILE,
                                useFixedBase=fixed_base)
        self.n = p.getNumJoints(self.robotId)
        self.joint_indices = []  # Will store joint pair indices
        self.constraints = []  # Will store constraint IDs
        
        # Create a mapping of joint names to their indices
        self.joint_name_to_id = {}
        for i in range(self.n):
            info = p.getJointInfo(self.robotId, i)
            self.joint_name_to_id[info[1].decode('utf-8')] = i
        
    def set_joint_pairs(self, joint_pairs):
        """Set the joint pairs for constraints.
        
        Args:
            joint_pairs (list): List of tuples containing (parent_joint, child_joint) indices
                              These are the rigid joint numbers that form closed kinematic chains
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
        
        self.joint_indices = converted_pairs
        
    def set_constraints(self):
        """Create fixed constraints between joint pairs and disable motors."""
        # Create a fixed constraint for each pair of linked joints
        for parent_id, child_id in self.joint_indices:
            # Create the constraint
            try:
                parent_info = p.getJointInfo(self.robotId, parent_id)
                child_info = p.getJointInfo(self.robotId, child_id)
                parent_name = parent_info[1].decode('utf-8')
                child_name = child_info[1].decode('utf-8')

                constraint_id = p.createConstraint(self.robotId, parent_id, 
                                                self.robotId, child_id, 
                                                p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                # Store the constraint ID
                self.constraints.append(constraint_id)
                # Set constraint parameters
                p.changeConstraint(constraint_id, maxForce=1e20)

                print(f"Creating fixed constraint between {parent_name} and {child_name}")

            except p.error as e:
                print(f"Failed to create constraint between joints {parent_id} and {child_id}")
                print(f"Parent joint info: {p.getJointInfo(self.robotId, parent_id)}")
                print(f"Child joint info: {p.getJointInfo(self.robotId, child_id)}")
                raise e

        # Disable the motors for all joints
        for i in range(self.n):
            maxForce = 0
            mode = p.VELOCITY_CONTROL
            p.setJointMotorControl2(self.robotId, i,
                                  controlMode=mode, force=maxForce)

def main():
    # Connect to PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,  # Distance to view both platforms
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.25]  # Adjusted to center both platforms
    )

    # Generate and save combined URDF
    # stages = [
    #     (1, "lower_", 0),      # Stage 1 at z=0
    #     (2, "upper_", 0),   # Stage 2 with 8cm spacing from Stage 1
    #     (3, "upper2_", 0),   # Stage 3 with 8cm spacing from Stage 2
    #     (4, "upper3_", 0),   # Stage 4 with 8cm spacing from Stage 3
    #     (5, "upper4_", 0),   # Stage 5 with 8cm spacing from Stage 4
    #     (6, "upper5_", 0),   # Stage 6 with 8cm spacing from Stage 5
    # ]
    stages = [(1, "lower_", 0)]
    urdf_path = "stewart_combined.urdf"
    urdf_string, all_joint_pairs = generate_stewart_platform(stages)
    save_urdf(urdf_string, urdf_path)

    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Create and initialize Stewart Platform
    platform = StewartPlatform(urdf_path)
    
    # Set up constraints for each stage
    for stage_pairs in all_joint_pairs:
        platform.set_joint_pairs(stage_pairs)
        platform.set_constraints()

    # Run simulation
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    p.disconnect()
    os.remove(urdf_path)

if __name__ == "__main__":
    main() 