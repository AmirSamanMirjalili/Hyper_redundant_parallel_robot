import pybullet as p
import pybullet_data
import time
import os
from generate_urdf import generate_stewart_platform, save_urdf

def main():
    # Connect to PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,  # Increased to better see the platform
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )

    # Generate and save URDF
    urdf_path = "stewart_lower.urdf"
    urdf_string = generate_stewart_platform(stage=1)  # Using the new function
    save_urdf(urdf_string, urdf_path)

    # Load ground plane and URDF
    planeId = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, [0, 0, 0.1], [0, 0, 0, 1],
                         flags=p.URDF_USE_INERTIA_FROM_FILE)

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