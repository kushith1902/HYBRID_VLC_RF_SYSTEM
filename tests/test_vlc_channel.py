"""
Unit tests for VLC IM/DD Optical Channel, Lambertian Gain, and Blockage.
"""

import numpy as np
import pytest
from src.vlc_channel import VLCChannelModel
from config.default_config import VLCConfig


def test_lambertian_order_60_degrees():
    # For semi-angle of 60 deg, cos(60) = 0.5. m = -ln(2)/ln(0.5) = 1.0
    cfg = VLCConfig(semi_angle_half_power_deg=60.0)
    vlc = VLCChannelModel(cfg)
    assert np.isclose(vlc.m, 1.0, atol=1e-5)


def test_optical_concentrator_gain():
    cfg = VLCConfig(pd_fov_deg=60.0, refractive_index=1.5)
    vlc = VLCChannelModel(cfg)
    # g(psi) = n^2 / sin^2(FOV) = 2.25 / (sqrt(3)/2)^2 = 2.25 / 0.75 = 3.0
    assert np.isclose(vlc.g_conc, 3.0, atol=1e-5)


def test_dc_gain_zero_outside_fov():
    vlc = VLCChannelModel()
    dists = np.array([[2.0]])
    phi = np.array([[0.1]])
    psi_outside = np.array([[np.radians(75.0)]]) # 75 deg > FOV (60 deg)
    h = vlc.calculate_dc_gain_matrix(dists, phi, psi_outside)
    assert h[0, 0] == 0.0


def test_dc_gain_zero_when_blocked():
    vlc = VLCChannelModel()
    dists = np.array([[2.0]])
    phi = np.array([[0.1]])
    psi = np.array([[0.1]])
    blockage = np.array([[1]]) # Blocked
    h = vlc.calculate_dc_gain_matrix(dists, phi, psi, blockage_matrix=blockage)
    assert h[0, 0] == 0.0


def test_electrical_snr_zero_for_blocked_link():
    vlc = VLCChannelModel()
    h_blocked = np.array([0.0])
    snr = vlc.calculate_electrical_snr_linear(tx_optical_power_w=5.0, effective_h=h_blocked)
    assert snr[0] == 0.0
