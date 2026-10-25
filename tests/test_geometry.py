"""
Unit tests for 3D Geometry, Angles, and Optical Blockage Intersection.
"""

import numpy as np
import pytest
from src.geometry import Obstacle3D, euclidean_distance, calculate_vlc_angles, check_segment_aabb_intersection


def test_euclidean_distance():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    assert np.isclose(euclidean_distance(p1, p2), 5.0)

    p3 = np.array([1.0, 2.0, 3.0])
    p4 = np.array([1.0, 2.0, 3.0])
    assert np.isclose(euclidean_distance(p3, p4), 0.0)


def test_vlc_angles_directly_below():
    # LED at ceiling (2.5, 2.5, 3.0), user directly underneath at (2.5, 2.5, 0.85)
    led_pos = np.array([2.5, 2.5, 3.0])
    user_pos = np.array([2.5, 2.5, 0.85])
    phi, psi = calculate_vlc_angles(led_pos, user_pos)
    # Irradiance and incidence angles should both be 0 radians (normal incidence)
    assert np.isclose(phi, 0.0, atol=1e-5)
    assert np.isclose(psi, 0.0, atol=1e-5)


def test_vlc_angles_offset():
    # LED at (0.0, 0.0, 3.0), user at (3.0, 0.0, 0.0)
    # dx = 3, dz = -3 -> 45 degrees = pi/4 radians
    led_pos = np.array([0.0, 0.0, 3.0])
    user_pos = np.array([3.0, 0.0, 0.0])
    phi, psi = calculate_vlc_angles(led_pos, user_pos)
    expected_rad = np.pi / 4.0
    assert np.isclose(phi, expected_rad, atol=1e-4)
    assert np.isclose(psi, expected_rad, atol=1e-4)


def test_obstacle_intersection_blocked():
    # Obstacle centered between LED and user
    obs = Obstacle3D.from_center_and_size(center=(2.0, 2.0, 1.5), size=(1.0, 1.0, 1.0))
    led_pos = np.array([2.0, 2.0, 3.0])
    user_pos = np.array([2.0, 2.0, 0.0])
    assert check_segment_aabb_intersection(led_pos, user_pos, obs)


def test_obstacle_intersection_clear():
    # Obstacle placed far to the side
    obs = Obstacle3D.from_center_and_size(center=(4.0, 4.0, 1.5), size=(0.5, 0.5, 1.0))
    led_pos = np.array([1.0, 1.0, 3.0])
    user_pos = np.array([1.0, 1.0, 0.0])
    assert not check_segment_aabb_intersection(led_pos, user_pos, obs)


def test_variable_obstacles_and_rf_los():
    from src.environment import IndoorEnvironment
    from config.default_config import SystemConfig
    
    cfg = SystemConfig(num_users=8)
    env = IndoorEnvironment(cfg, seed=42)
    
    # Test setting variable obstacle counts
    env.set_variable_obstacles(count=4)
    assert len(env.obstacles) == 4
    
    env.set_variable_obstacles(count=1)
    assert len(env.obstacles) == 1
    
    # Test RF LOS computation
    env.set_variable_obstacles(count=6)
    rf_los = env.compute_rf_los_status()
    assert len(rf_los) == 8
    assert isinstance(rf_los[0], (bool, np.bool_))
