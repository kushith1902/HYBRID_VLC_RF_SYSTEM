"""
Unit tests for Optimization Teacher and Feasibility Checking.
"""

import numpy as np
import pytest
from src.optimizer import OptimizationTeacher
from config.default_config import SystemConfig


def test_optimizer_produces_valid_result():
    config = SystemConfig(num_users=4)
    teacher = OptimizationTeacher(config, alpha_grid_points=3, beta_grid_points=3)
    
    rf_gains = np.array([2e-7, 1.5e-7, 1e-7, 3e-7])
    vlc_gains = np.array([1.2e-5, 0.8e-5, 0.0, 1.5e-5]) # User 2 is blocked
    is_blocked = np.array([False, False, True, False])
    
    res = teacher.optimize(rf_gains, vlc_gains, is_blocked, min_rate_bps=1e6)
    
    assert 'modes' in res
    assert 'alpha' in res
    assert 'beta' in res
    assert 'energy_efficiency' in res
    assert 'is_feasible' in res
    assert len(res['modes']) == 4
    assert 0.0 <= res['alpha'] <= 1.0
    assert 0.0 <= res['beta'] <= 1.0
    assert res['energy_efficiency'] > 0.0
    assert res['optimization_runtime_sec'] > 0.0


def test_optimizer_marks_infeasible_when_min_rate_unachievable():
    config = SystemConfig(num_users=4)
    teacher = OptimizationTeacher(config, alpha_grid_points=3, beta_grid_points=3)
    
    # Very poor channels
    rf_gains = np.full(4, 1e-15)
    vlc_gains = np.full(4, 0.0)
    is_blocked = np.full(4, True)
    
    # Require impossible 100 Gbps
    res = teacher.optimize(rf_gains, vlc_gains, is_blocked, min_rate_bps=1e11)
    assert res['is_feasible'] is False
    assert res['solution_type'] == "Best_Effort_Infeasible"


def test_optimizer_high_user_density():
    # Test high user density with K=12 users and mixed blockage
    K = 12
    config = SystemConfig(num_users=K)
    teacher = OptimizationTeacher(config, alpha_grid_points=3, beta_grid_points=3)
    
    rf_gains = np.full(K, 2e-7)
    vlc_gains = np.full(K, 1e-5)
    # Block half the users
    is_blocked = np.array([True if i % 2 == 0 else False for i in range(K)])
    vlc_gains[is_blocked] = 0.0
    
    res = teacher.optimize(rf_gains, vlc_gains, is_blocked, min_rate_bps=1e6)
    assert len(res['modes']) == K
    assert res['energy_efficiency'] > 0.0
    assert res['optimization_runtime_sec'] < 2.0 # Must be fast and scalable
