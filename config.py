"""
Configuration module for Stewart Platform.

This module contains all configuration parameters for the Stewart Platform,
including geometric parameters, control parameters, and physical constraints.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class GeometryConfig:
    """Geometric parameters of the Stewart Platform."""
    radius_base: float  # Radius of the base platform
    radius_platform: float  # Radius of the top platform
    gamma_base: float  # Half angle between anchors on base (degrees)
    gamma_platform: float  # Half angle between anchors on platform (degrees)
    home_position: np.ndarray = np.array([0, 0, 0.257547])  # Default home position
    
    def __post_init__(self):
        """Convert angles to radians and validate parameters."""
        if not all(v > 0 for v in [self.radius_base, self.radius_platform]):
            raise ValueError("Radii must be positive")
        if not all(0 < v < 180 for v in [self.gamma_base, self.gamma_platform]):
            raise ValueError("Gamma angles must be between 0 and 180 degrees")

@dataclass
class ActuatorConfig:
    """Configuration for the actuators."""
    max_force: float = 250.0  # Maximum force in Newtons
    pwm_frequency: float = 50.0  # PWM frequency in Hz
    duty_cycle: float = 0.5  # PWM duty cycle (0-1)
    min_length: float = 0.2  # Minimum actuator length in meters
    max_length: float = 0.4  # Maximum actuator length in meters
    
    def __post_init__(self):
        """Validate actuator parameters."""
        if not 0 < self.duty_cycle < 1:
            raise ValueError("Duty cycle must be between 0 and 1")
        if self.min_length >= self.max_length:
            raise ValueError("Minimum length must be less than maximum length")

@dataclass
class MotionConfig:
    """Configuration for platform motion limits."""
    max_translation: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # Max translation in meters
    max_rotation: Tuple[float, float, float] = (30.0, 30.0, 30.0)  # Max rotation in degrees
    max_velocity: float = 0.1  # Maximum velocity in m/s
    max_acceleration: float = 0.2  # Maximum acceleration in m/s²
    
    def __post_init__(self):
        """Validate motion parameters."""
        if not all(v > 0 for v in self.max_translation + self.max_rotation):
            raise ValueError("Motion limits must be positive")

@dataclass
class StewartConfig:
    """Complete configuration for the Stewart Platform."""
    geometry: GeometryConfig
    actuator: ActuatorConfig
    motion: MotionConfig
    
    @classmethod
    def default_config(cls) -> 'StewartConfig':
        """Create a default configuration."""
        return cls(
            geometry=GeometryConfig(
                radius_base=0.2,
                radius_platform=0.15,
                gamma_base=30.0,
                gamma_platform=25.0
            ),
            actuator=ActuatorConfig(),
            motion=MotionConfig()
        )
    
    def validate(self) -> bool:
        """Validate the complete configuration."""
        try:
            # The dataclass post_init validations will run automatically
            return True
        except ValueError as e:
            print(f"Configuration validation failed: {e}")
            return False

# Default configuration instance
default_config = StewartConfig.default_config() 