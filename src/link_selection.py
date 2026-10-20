"""
Link Selection Algorithms and Feasibility Handlers for Hybrid RF/VLC Networks.
Supports RF-only, VLC-only, and Hybrid dual-connectivity modes.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np


LINK_MODES = {
    0: "RF",
    1: "VLC",
    2: "Hybrid"
}


def rule_based_link_selection(vlc_effective_h: np.ndarray,
                              rf_snr_db: np.ndarray,
                              is_blocked: np.ndarray,
                              vlc_gain_threshold: float = 1.0e-5,
                              rf_snr_threshold_db: float = 15.0) -> np.ndarray:
    """
    Rule-Based Link Selection Heuristic:
    - If VLC is blocked (or H_VLC < threshold): Select RF-only (0).
    - If VLC is unblocked, has high gain (> threshold), and RF SNR is moderate: Select VLC-only (1)
      to offload RF traffic.
    - If user requires high capacity and both links have excellent quality: Select Hybrid (2).
    
    Parameters:
        vlc_effective_h: (K,) optical channel gain
        rf_snr_db: (K,) RF SNR in dB
        is_blocked: (K,) boolean array (True if optical link is blocked)
        vlc_gain_threshold: Minimum optical gain for reliable VLC
        rf_snr_threshold_db: RF threshold for high quality
        
    Returns:
        modes: (K,) array of integers in {0, 1, 2}
    """
    K = len(vlc_effective_h)
    modes = np.zeros(K, dtype=int)
    
    for k in range(K):
        blocked = bool(is_blocked[k])
        h_vlc = float(vlc_effective_h[k])
        snr_rf = float(rf_snr_db[k])
        
        if blocked or h_vlc < vlc_gain_threshold:
            # Optical path blocked -> Must use RF
            modes[k] = 0 # RF-only
        elif h_vlc >= vlc_gain_threshold and snr_rf >= rf_snr_threshold_db:
            # Both links strong -> Hybrid aggregation
            modes[k] = 2 # Hybrid
        elif h_vlc >= vlc_gain_threshold and snr_rf < rf_snr_threshold_db:
            # VLC is strong, RF is weaker -> Offload to VLC
            modes[k] = 1 # VLC-only
        else:
            modes[k] = 0
            
    return modes


def validate_and_project_link_selection(modes: np.ndarray, is_blocked: np.ndarray) -> np.ndarray:
    """
    Sanity projection for link modes:
    If a user is assigned VLC-only (1) but their optical path is completely blocked,
    project them to RF-only (0) to prevent guaranteed outage.
    If assigned Hybrid (2) and optical path is blocked, they gracefully degrade to RF.
    """
    projected = np.array(modes, dtype=int)
    for k in range(len(projected)):
        if is_blocked[k] and projected[k] == 1:
            projected[k] = 0 # Fall back to RF
    return projected
