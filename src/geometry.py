"""
Geometry and 3D Ray-Tracing Utilities for 6G Hybrid RF/VLC Networks.
Provides Euclidean distance calculations, irradiance/incidence angle determination,
and 3D bounding-box ray intersection for optical line-of-sight (LOS) blockage.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Obstacle3D:
    """
    3D Rectangular Obstacle (e.g., human blocker, indoor partition, pillar).
    Defined by axis-aligned bounding box (AABB): [x_min, x_max], [y_min, y_max], [z_min, z_max].
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    name: str = "Blocker"

    @classmethod
    def from_center_and_size(cls, center: Tuple[float, float, float],
                             size: Tuple[float, float, float], name: str = "Blocker"):
        """Construct from center (x, y, z) and dimension (dx, dy, dz)."""
        cx, cy, cz = center
        dx, dy, dz = size
        return cls(
            x_min=cx - dx / 2.0, x_max=cx + dx / 2.0,
            y_min=cy - dy / 2.0, y_max=cy + dy / 2.0,
            z_min=cz - dz / 2.0, z_max=cz + dz / 2.0,
            name=name
        )


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate 3D Euclidean distance between two points in meters."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def calculate_vlc_angles(led_pos: np.ndarray, user_pos: np.ndarray,
                         led_normal: np.ndarray = np.array([0.0, 0.0, -1.0]),
                         pd_normal: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Tuple[float, float]:
    """
    Calculate irradiance angle (phi) at transmitter LED and incidence angle (psi) at receiver photodiode.
    
    Parameters:
        led_pos: (x, y, z) of LED transmitter (ceiling)
        user_pos: (x, y, z) of user receiver (desk level)
        led_normal: Normal vector of LED surface (default pointing straight down: [0, 0, -1])
        pd_normal: Normal vector of PD surface (default pointing straight up: [0, 0, 1])
        
    Returns:
        (phi_rad, psi_rad): Irradiance angle and incidence angle in radians.
    """
    led_p = np.asarray(led_pos, dtype=float)
    user_p = np.asarray(user_pos, dtype=float)
    
    # Vector pointing from LED to user
    vec_tx_to_rx = user_p - led_p
    dist = np.linalg.norm(vec_tx_to_rx)
    if dist < 1e-9:
        return 0.0, 0.0
    
    unit_tx_to_rx = vec_tx_to_rx / dist
    unit_rx_to_tx = -unit_tx_to_rx
    
    # cos(phi) = (unit_tx_to_rx . led_normal) / (|unit| * |normal|)
    cos_phi = np.dot(unit_tx_to_rx, led_normal) / (np.linalg.norm(led_normal))
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi_rad = float(np.arccos(cos_phi))
    
    # cos(psi) = (unit_rx_to_tx . pd_normal) / (|unit| * |normal|)
    cos_psi = np.dot(unit_rx_to_tx, pd_normal) / (np.linalg.norm(pd_normal))
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    psi_rad = float(np.arccos(cos_psi))
    
    return phi_rad, psi_rad


def check_segment_aabb_intersection(p1: np.ndarray, p2: np.ndarray, obstacle: Obstacle3D) -> bool:
    """
    Slab Method for 3D Line Segment vs Axis-Aligned Bounding Box (AABB) intersection.
    Tests if line segment p1 -> p2 intersects the obstacle volume.
    
    Parameters:
        p1: Start point (3,) [LED position]
        p2: End point (3,) [User position]
        obstacle: Obstacle3D bounding box
        
    Returns:
        True if the line-of-sight segment is blocked by the obstacle, False otherwise.
    """
    d = p2 - p1
    t_min = 0.0
    t_max = 1.0
    
    bounds = [
        (obstacle.x_min, obstacle.x_max),
        (obstacle.y_min, obstacle.y_max),
        (obstacle.z_min, obstacle.z_max)
    ]
    
    for i in range(3):
        if abs(d[i]) < 1e-9:
            # Ray is parallel to slab. Check if origin is outside slab.
            if p1[i] < bounds[i][0] or p1[i] > bounds[i][1]:
                return False
        else:
            inv_d = 1.0 / d[i]
            t1 = (bounds[i][0] - p1[i]) * inv_d
            t2 = (bounds[i][1] - p1[i]) * inv_d
            
            if t1 > t2:
                t1, t2 = t2, t1
                
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            
            if t_min > t_max:
                return False
                
    return bool(t_min <= t_max and t_max >= 0.0 and t_min <= 1.0)
