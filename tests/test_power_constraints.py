"""
Unit tests for Power Allocation, Constraints, and Energy Efficiency.
"""

import numpy as np
import pytest
from src.power_allocation import PowerAllocationModel
from config.default_config import PowerConsumptionConfig, RFConfig, VLCConfig


def test_power_allocation_alpha_bounds():
    power_model = PowerAllocationModel()
    
    # Test alpha = 0 (100% VLC)
    p_rf, p_vlc = power_model.allocate_transmit_powers(0.0)
    assert np.isclose(p_rf, 0.0)
    assert p_vlc > 0.0
    
    # Test alpha = 1 (100% RF)
    p_rf, p_vlc = power_model.allocate_transmit_powers(1.0)
    assert p_rf > 0.0
    assert np.isclose(p_vlc, 0.0)
    
    # Test out of bounds clamping
    p_rf_neg, _ = power_model.allocate_transmit_powers(-0.5)
    assert np.isclose(p_rf_neg, 0.0)
    
    p_rf_high, _ = power_model.allocate_transmit_powers(1.5)
    assert np.isclose(p_rf_high, power_model.rf_cfg.tx_power_max_w)


def test_total_consumed_power_and_ee():
    power_model = PowerAllocationModel()
    p_rf_tx = 0.1 # 100 mW
    p_vlc_opt = 10.0 # 10 W
    total_power = power_model.calculate_total_consumed_power(p_rf_tx, p_vlc_opt, num_users=4)
    
    # Total power must include static DSP, PA, driver, circuits
    assert total_power > (0.1 + 10.0)
    
    # EE calculation
    sum_rate = 500e6 # 500 Mbps
    ee = power_model.calculate_energy_efficiency(sum_rate, total_power)
    assert ee > 0.0
    assert np.isclose(ee, sum_rate / total_power)
