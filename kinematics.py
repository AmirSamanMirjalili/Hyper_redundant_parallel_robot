"""
Kinematics module for Stewart Platform.

This module handles inverse kinematics calculations to determine required
actuator lengths for desired platform position and orientation.
"""

import numpy as np
from typing import Tuple, List, Optional
from config import StewartConfig
from geometry import StewartGeometry

class KinematicsSolver:
    """Solves inverse kinematics for Stewart Platform."""
    
    def __init__(self, config: StewartConfig):
        """Initialize kinematics solver.
        
        Args:
            config: Complete Stewart Platform configuration
        """
        self.config = config
        self.geometry = StewartGeometry(config.geometry)
        self._prev_leg_lengths = None
        
    def check_motion_limits(self, translation: np.ndarray, rotation: np.ndarray) -> bool:
        """Check if desired motion is within platform limits.
        
        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation angles [roll, pitch, yaw] in degrees
            
        Returns:
            True if motion is within limits, False otherwise
        """
        # Check translation limits
        if not all(abs(t) <= max_t for t, max_t in zip(translation, self.config.motion.max_translation)):
            return False
            
        # Check rotation limits
        if not all(abs(r) <= max_r for r, max_r in zip(rotation, self.config.motion.max_rotation)):
            return False
            
        return True
    
    def check_actuator_limits(self, leg_lengths: np.ndarray) -> bool:
        """Check if required leg lengths are within actuator limits.
        
        Args:
            leg_lengths: Array of 6 leg lengths
            
        Returns:
            True if lengths are within limits, False otherwise
        """
        return all(
            self.config.actuator.min_length <= length <= self.config.actuator.max_length
            for length in leg_lengths
        )
    
    def solve_inverse(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        check_limits: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve inverse kinematics to get required leg lengths and leg vectors.
        
        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation angles [roll, pitch, yaw] in degrees
            check_limits: Whether to check motion and actuator limits
            
        Returns:
            Tuple of:
            - Array of 6 leg lengths
            - 3x6 array of leg vectors
            
        Raises:
            ValueError: If motion is outside platform limits
        """
        # Check motion limits if required
        if check_limits and not self.check_motion_limits(translation, rotation):
            raise ValueError("Desired motion exceeds platform limits")
        
        # Get transformed platform points
        platform_points = self.geometry.get_platform_points(translation, rotation)
        
        # Calculate leg vectors (from base to platform anchor points)
        leg_vectors = platform_points - self.geometry.base_anchors
        
        # Calculate leg lengths
        leg_lengths = np.linalg.norm(leg_vectors, axis=0)
        
        # Check actuator limits if required
        if check_limits and not self.check_actuator_limits(leg_lengths):
            raise ValueError("Required leg lengths exceed actuator limits")
        
        # Store leg lengths for future reference
        self._prev_leg_lengths = leg_lengths
        
        return leg_lengths, leg_vectors
    
    def get_leg_length_changes(self, new_lengths: np.ndarray) -> Optional[np.ndarray]:
        """Calculate changes in leg lengths from previous position.
        
        Args:
            new_lengths: Array of 6 new leg lengths
            
        Returns:
            Array of 6 leg length changes, or None if no previous lengths
        """
        if self._prev_leg_lengths is None:
            return None
        return new_lengths - self._prev_leg_lengths
    
    def reset_state(self) -> None:
        """Reset solver state (previous leg lengths)."""
        self._prev_leg_lengths = None 