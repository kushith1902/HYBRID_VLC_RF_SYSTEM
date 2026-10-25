"""
Indoor Environment Management and Mobility Simulation.
Manages spatial configuration of 140 GHz RF AP, 4 VLC LEDs, mobile users,
3D obstacles, and line-of-sight blockage matrices over time.
Supports high user densities and variable physical obstacle counts.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from config.default_config import SystemConfig, RoomConfig
from src.geometry import Obstacle3D, euclidean_distance, calculate_vlc_angles, check_segment_aabb_intersection


class IndoorEnvironment:
    """
    Simulated 3D Indoor Network Environment for 6G Hybrid RF/VLC Networks.
    """
    def __init__(self, config: Optional[SystemConfig] = None, seed: Optional[int] = None):
        self.config = config or SystemConfig()
        self.rng = np.random.default_rng(seed if seed is not None else self.config.random_seed)
        
        # Room dimensions
        self.room: RoomConfig = self.config.room
        
        # Access Point position (3,)
        self.rf_ap_pos = np.array(self.config.rf.ap_position, dtype=float)
        
        # VLC LED positions (num_leds, 3)
        self.vlc_led_positions = np.array(self.config.vlc.led_positions, dtype=float)
        self.num_leds = len(self.vlc_led_positions)
        
        # Users: positions (K, 3) and velocities (K, 3)
        self.num_users = self.config.num_users
        self.user_positions = np.zeros((self.num_users, 3), dtype=float)
        self.user_velocities = np.zeros((self.num_users, 3), dtype=float)
        
        # Obstacles list
        self.obstacles: List[Obstacle3D] = []
        
        # Initialize default layout with variable obstacles
        self.reset_users()
        self.set_variable_obstacles(count=3)

    def reset_users(self, user_positions: Optional[np.ndarray] = None):
        """
        Initialize or reset user positions inside the room at desk height (0.85m).
        Ensures a minimum margin of 0.3m from walls.
        Supports high user density scenarios.
        """
        margin = 0.3
        if user_positions is not None:
            self.user_positions = np.array(user_positions, dtype=float)
            self.num_users = len(self.user_positions)
        else:
            xs = self.rng.uniform(margin, self.room.length - margin, size=self.num_users)
            ys = self.rng.uniform(margin, self.room.width - margin, size=self.num_users)
            zs = np.full(self.num_users, self.room.user_height)
            self.user_positions = np.column_stack([xs, ys, zs])
            
        # Initialize 2D velocity vectors in range [0.1, 1.2] m/s (pedestrian indoor mobility)
        speeds = self.rng.uniform(0.2, 1.0, size=self.num_users)
        angles = self.rng.uniform(0, 2 * np.pi, size=self.num_users)
        vx = speeds * np.cos(angles)
        vy = speeds * np.sin(angles)
        vz = np.zeros(self.num_users)
        self.user_velocities = np.column_stack([vx, vy, vz])

    def init_default_obstacles(self):
        """Default setup with 3 realistic obstacles."""
        self.set_variable_obstacles(count=3)

    def set_variable_obstacles(self, count: int = 3):
        """
        Configure a variable number of realistic indoor obstacles (1 to 6).
        Includes architectural columns (ceiling-height), office cubicle partitions,
        and human blockers to guarantee realistic line-of-sight blockage for some users.
        """
        catalog = [
            # 1. Architectural Concrete Column (Floor to ceiling - blocks both VLC and shadows RF)
            Obstacle3D.from_center_and_size(
                center=(2.0, 2.5, 1.5),
                size=(0.55, 0.55, 3.0),
                name="Architectural Column"
            ),
            # 2. Office Partition Wall (Room divider)
            Obstacle3D.from_center_and_size(
                center=(3.4, 2.6, 1.0),
                size=(0.2, 1.8, 2.0),
                name="Office Partition Wall"
            ),
            # 3. Human Blocker 1 (Standing person)
            Obstacle3D.from_center_and_size(
                center=(1.5, 1.8, 0.85),
                size=(0.5, 0.5, 1.75),
                name="Human Blocker A"
            ),
            # 4. Human Blocker 2 (Walking person)
            Obstacle3D.from_center_and_size(
                center=(3.2, 3.8, 0.85),
                size=(0.5, 0.5, 1.75),
                name="Human Blocker B"
            ),
            # 5. Tall Equipment Rack / Cabinet
            Obstacle3D.from_center_and_size(
                center=(1.2, 3.4, 1.1),
                size=(0.7, 0.7, 2.2),
                name="Server Equipment Rack"
            ),
            # 6. Presentation Board / Display Panel
            Obstacle3D.from_center_and_size(
                center=(2.8, 1.3, 1.0),
                size=(1.5, 0.2, 2.0),
                name="Display Screen Wall"
            )
        ]
        
        valid_count = max(0, min(count, len(catalog)))
        self.obstacles = catalog[:valid_count]

    def add_obstacle(self, obstacle: Obstacle3D):
        """Add a custom 3D obstacle."""
        self.obstacles.append(obstacle)

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []

    def step_mobility(self, dt: float = 0.5):
        """
        Update user positions using linear motion with boundary reflection.
        Keeps users within room boundaries.
        
        Parameters:
            dt: Time step in seconds.
        """
        margin = 0.2
        for k in range(self.num_users):
            new_pos = self.user_positions[k] + self.user_velocities[k] * dt
            
            # X boundary reflection
            if new_pos[0] < margin:
                new_pos[0] = margin + (margin - new_pos[0])
                self.user_velocities[k, 0] *= -1.0
            elif new_pos[0] > self.room.length - margin:
                new_pos[0] = (self.room.length - margin) - (new_pos[0] - (self.room.length - margin))
                self.user_velocities[k, 0] *= -1.0
                
            # Y boundary reflection
            if new_pos[1] < margin:
                new_pos[1] = margin + (margin - new_pos[1])
                self.user_velocities[k, 1] *= -1.0
            elif new_pos[1] > self.room.width - margin:
                new_pos[1] = (self.room.width - margin) - (new_pos[1] - (self.room.width - margin))
                self.user_velocities[k, 1] *= -1.0
                
            new_pos[2] = self.room.user_height # Maintain desk height
            self.user_positions[k] = new_pos

    def get_rf_distances(self) -> np.ndarray:
        """Calculate 3D distances from RF AP to each user (K,)."""
        return np.linalg.norm(self.user_positions - self.rf_ap_pos, axis=1)

    def compute_rf_los_status(self) -> np.ndarray:
        """
        Determine whether the 140 GHz RF path from AP to User k has direct LOS (True)
        or is shadowed/blocked by an obstacle (False = NLOS).
        """
        is_los = np.ones(self.num_users, dtype=bool)
        for k in range(self.num_users):
            user_pos = self.user_positions[k]
            for obs in self.obstacles:
                if check_segment_aabb_intersection(self.rf_ap_pos, user_pos, obs):
                    is_los[k] = False
                    break
        return is_los

    def get_vlc_distances(self) -> np.ndarray:
        """Calculate 3D distances from each VLC LED to each user (num_leds, K)."""
        dists = np.zeros((self.num_leds, self.num_users), dtype=float)
        for m in range(self.num_leds):
            dists[m] = np.linalg.norm(self.user_positions - self.vlc_led_positions[m], axis=1)
        return dists

    def compute_optical_blockage_matrix(self) -> np.ndarray:
        """
        Compute optical line-of-sight blockage matrix B of shape (num_leds, K).
        B[m, k] = 1 if the direct LOS ray from LED m to User k is blocked by any obstacle, else 0.
        """
        blockage_matrix = np.zeros((self.num_leds, self.num_users), dtype=int)
        for m in range(self.num_leds):
            led_pos = self.vlc_led_positions[m]
            for k in range(self.num_users):
                user_pos = self.user_positions[k]
                for obs in self.obstacles:
                    if check_segment_aabb_intersection(led_pos, user_pos, obs):
                        blockage_matrix[m, k] = 1
                        break
        return blockage_matrix

    def get_vlc_angles_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute irradiance (phi) and incidence (psi) angles for all (LED, user) pairs.
        
        Returns:
            phi_mat: (num_leds, K) in radians
            psi_mat: (num_leds, K) in radians
        """
        phi_mat = np.zeros((self.num_leds, self.num_users), dtype=float)
        psi_mat = np.zeros((self.num_leds, self.num_users), dtype=float)
        for m in range(self.num_leds):
            for k in range(self.num_users):
                phi, psi = calculate_vlc_angles(self.vlc_led_positions[m], self.user_positions[k])
                phi_mat[m, k] = phi
                psi_mat[m, k] = psi
        return phi_mat, psi_mat
