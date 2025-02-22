"""
Actuator control module for Stewart Platform.

This module handles the control of linear actuators using PWM signals
and position control through PyBullet.
"""

import numpy as np
import pybullet as p
import time
from typing import List, Optional
from config import ActuatorConfig

class PWMController:
    """Handles PWM signal generation for actuator control."""
    
    def __init__(self, config: ActuatorConfig):
        """Initialize PWM controller.
        
        Args:
            config: Actuator configuration
        """
        self.config = config
        self.period = 1.0 / config.pwm_frequency
        self.high_time = self.period * config.duty_cycle
        
    def get_pwm_state(self, current_time: float) -> bool:
        """Get PWM signal state at given time.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            True if signal should be high, False if low
        """
        return (current_time % self.period) < self.high_time

class ActuatorController:
    """Controls linear actuators through PyBullet."""
    
    def __init__(
        self,
        robot_id: int,
        actuator_indices: List[int],
        config: ActuatorConfig
    ):
        """Initialize actuator controller.
        
        Args:
            robot_id: PyBullet body ID
            actuator_indices: List of joint indices for actuators
            config: Actuator configuration
        """
        self.robot_id = robot_id
        self.actuator_indices = actuator_indices
        self.config = config
        self._prev_target = np.zeros(len(actuator_indices))
        
    def get_current_lengths(self) -> np.ndarray:
        """Get current actuator lengths.
        
        Returns:
            Array of current actuator lengths
        """
        return np.array([
            p.getJointState(self.robot_id, idx)[0]
            for idx in self.actuator_indices
        ])
    
    def _generate_quintic_trajectory(
        self,
        start_positions: np.ndarray,
        end_positions: np.ndarray,
        steps: int
    ) -> List[np.ndarray]:
        """Generate a smooth quintic polynomial trajectory between positions.
        
        Uses the formula:
            q(τ) = q_i + (q_f - q_i) * (10 * τ^3 - 15 * τ^4 + 6 * τ^5)
        where τ is the normalized time (0 to 1).
        
        Args:
            start_positions: Initial actuator positions
            end_positions: Target actuator positions
            steps: Number of interpolation steps
            
        Returns:
            List of actuator position arrays for each timestep
        """
        trajectory = []
        for i in range(len(start_positions)):  # For each actuator
            q_i = start_positions[i]
            q_f = end_positions[i]
            actuator_trajectory = []
            
            for t in range(steps + 1):
                tau = t / steps  # Normalize time to [0, 1]
                q_t = q_i + (q_f - q_i) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
                actuator_trajectory.append(q_t)
            
            trajectory.append(actuator_trajectory)
        
        # Transpose trajectory so each element is a timestep with all actuator positions
        return [np.array(pos) for pos in zip(*trajectory)]
        
    def move_to_lengths(
        self,
        target_lengths: np.ndarray,
        duration: float,
        use_pwm: bool = False  # PWM is optional now since we have smooth trajectories
    ) -> None:
        """Move actuators to target lengths using quintic polynomial trajectory.
        
        Args:
            target_lengths: Array of target actuator lengths
            duration: Movement duration in seconds
            use_pwm: Whether to use PWM control (optional)
        """
        if len(target_lengths) != len(self.actuator_indices):
            raise ValueError("Number of target lengths must match number of actuators")
            
        # Check actuator limits
        if np.any(target_lengths < self.config.min_length) or \
           np.any(target_lengths > self.config.max_length):
            raise ValueError(
                f"Target lengths must be between {self.config.min_length} and "
                f"{self.config.max_length}"
            )
        
        # Get current positions as starting point
        current_positions = self.get_current_lengths()
        
        # Calculate number of steps based on duration and simulation frequency
        steps = int(duration * 240)  # Using 240 Hz simulation frequency
        
        # Generate smooth trajectory
        trajectory = self._generate_quintic_trajectory(
            current_positions,
            target_lengths,
            steps
        )
        
        # Execute trajectory
        time_step = 1.0 / 240.0
        for positions in trajectory:
            # Apply position control
            p.setJointMotorControlArray(
                self.robot_id,
                self.actuator_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=positions,
                forces=[self.config.max_force] * len(self.actuator_indices)
            )
            
            # Step simulation
            p.stepSimulation()
            time.sleep(time_step)
        
        # Update previous target
        self._prev_target = target_lengths
        
    def reset_position(self, duration: float = 2.0) -> None:
        """Reset actuators to home position.
        
        Args:
            duration: Movement duration in seconds
        """
        # Move to middle position between min and max length
        home_position = np.full(
            len(self.actuator_indices),
            (self.config.min_length + self.config.max_length) / 2
        )
        self.move_to_lengths(
            home_position,
            duration=duration,
            use_pwm=False
        ) 