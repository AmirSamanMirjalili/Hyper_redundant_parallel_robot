"""
Geometry module for Stewart Platform.

This module handles all geometric calculations including:
- Base and platform anchor point calculations
- Rotation matrices
- Frame transformations
"""

import numpy as np
from typing import Tuple, List
from config import GeometryConfig

class RotationMatrix:
    """Handles rotation matrix calculations."""
    
    @staticmethod
    def rot_x(theta_deg: float) -> np.ndarray:
        """Create rotation matrix for rotation around X axis.
        
        Args:
            theta_deg: Rotation angle in degrees
            
        Returns:
            3x3 rotation matrix
        """
        theta = np.deg2rad(theta_deg)
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    @staticmethod
    def rot_y(theta_deg: float) -> np.ndarray:
        """Create rotation matrix for rotation around Y axis.
        
        Args:
            theta_deg: Rotation angle in degrees
            
        Returns:
            3x3 rotation matrix
        """
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    @staticmethod
    def rot_z(theta_deg: float) -> np.ndarray:
        """Create rotation matrix for rotation around Z axis.
        
        Args:
            theta_deg: Rotation angle in degrees
            
        Returns:
            3x3 rotation matrix
        """
        theta = np.deg2rad(theta_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    @classmethod
    def combined_rotation(cls, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Create combined rotation matrix from roll, pitch, yaw angles.
        
        Args:
            roll: Rotation around X axis in degrees
            pitch: Rotation around Y axis in degrees
            yaw: Rotation around Z axis in degrees
            
        Returns:
            3x3 combined rotation matrix
        """
        # Order: Roll (X) -> Pitch (Y) -> Yaw (Z)
        Rx = cls.rot_x(roll)
        Ry = cls.rot_y(pitch)
        Rz = cls.rot_z(yaw)
        return np.matmul(np.matmul(Rz, Ry), Rx)

class StewartGeometry:
    """Handles Stewart Platform geometry calculations."""
    
    def __init__(self, config: GeometryConfig):
        """Initialize with geometric configuration.
        
        Args:
            config: Platform geometry configuration
        """
        self.config = config
        self._base_anchors = None
        self._platform_anchors = None
        self.update_anchors()
    
    def update_anchors(self) -> None:
        """Update base and platform anchor points."""
        # Base anchor angles (in radians)
        phi_base = np.array([
            7*np.pi/6 + np.deg2rad(self.config.gamma_base),
            7*np.pi/6 - np.deg2rad(self.config.gamma_base),
            np.pi/2 + np.deg2rad(self.config.gamma_base),
            np.pi/2 - np.deg2rad(self.config.gamma_base),
            11*np.pi/6 + np.deg2rad(self.config.gamma_base),
            11*np.pi/6 - np.deg2rad(self.config.gamma_base)
        ])
        
        # Platform anchor angles (in radians)
        phi_platform = np.array([
            3*np.pi/2 - np.deg2rad(self.config.gamma_platform),
            5*np.pi/6 + np.deg2rad(self.config.gamma_platform),
            5*np.pi/6 - np.deg2rad(self.config.gamma_platform),
            np.pi/6 + np.deg2rad(self.config.gamma_platform),
            np.pi/6 - np.deg2rad(self.config.gamma_platform),
            3*np.pi/2 + np.deg2rad(self.config.gamma_platform)
        ])
        
        # Calculate base anchor points
        self._base_anchors = np.array([
            [self.config.radius_base * np.cos(phi) for phi in phi_base],
            [self.config.radius_base * np.sin(phi) for phi in phi_base],
            [0.0] * 6
        ])
        
        # Calculate platform anchor points
        self._platform_anchors = np.array([
            [self.config.radius_platform * np.cos(phi) for phi in phi_platform],
            [self.config.radius_platform * np.sin(phi) for phi in phi_platform],
            [0.0] * 6
        ])
    
    @property
    def base_anchors(self) -> np.ndarray:
        """Get base anchor points in base frame."""
        return self._base_anchors
    
    @property
    def platform_anchors(self) -> np.ndarray:
        """Get platform anchor points in platform frame."""
        return self._platform_anchors
    
    def get_platform_points(self, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Get platform anchor points in base frame after transformation.
        
        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation angles [roll, pitch, yaw] in degrees
            
        Returns:
            3x6 array of platform anchor points in base frame
        """
        # Create rotation matrix
        R = RotationMatrix.combined_rotation(*rotation)
        
        # Transform platform points to base frame
        transformed_points = (
            np.repeat(translation[:, np.newaxis], 6, axis=1) +  # Translation
            np.repeat(self.config.home_position[:, np.newaxis], 6, axis=1) +  # Home position
            np.matmul(R, self._platform_anchors)  # Rotation
        )
        
        return transformed_points 