"""
Unit tests for 140 GHz Sub-THz RF Channel, Path Loss, and Thermal Noise.
"""

import numpy as np
import pytest
from src.rf_channel import RFChannelModel
from config.default_config import RFConfig


def test_free_space_path_loss_1m():
    # FSPL at 140 GHz, 1m reference distance:
    # FSPL = 20 * log10(4 * pi * 1 * 140e9 / 3e8) approx 75.36 dB
    rf_model = RFChannelModel()
    dists = np.array([1.0])
    pl = rf_model.calculate_path_loss_db(dists, enable_shadowing=False)
    assert np.isclose(pl[0], 75.36, atol=0.2)


def test_path_loss_increases_with_distance():
    rf_model = RFChannelModel()
    dists = np.array([1.5, 3.0, 5.0])
    pl = rf_model.calculate_path_loss_db(dists, enable_shadowing=False)
    assert pl[0] < pl[1] < pl[2]


def test_channel_gain_positive_and_finite():
    rf_model = RFChannelModel(seed=42)
    dists = np.array([2.0, 3.5, 4.0])
    gains = rf_model.calculate_channel_gains(dists)
    assert np.all(gains > 0.0)
    assert np.all(np.isfinite(gains))
    assert np.all(gains < 1.0) # Channel gain should be much less than 1 (path loss is > 70 dB)


def test_noise_power_calculation():
    rf_model = RFChannelModel()
    # k_b * T * B * NF
    # k_b = 1.38e-23, T = 295, B = 1e9, NF = 7 dB (5.01)
    # sigma^2 approx 2.04e-11 W (-76.9 dBm)
    assert 1e-12 < rf_model.noise_power_w < 1e-10
    assert np.isclose(rf_model.noise_power_dbm, -76.9, atol=1.0)
