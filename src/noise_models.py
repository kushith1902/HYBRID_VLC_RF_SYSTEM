"""
Noise Calculation, SNR/SINR, and Achievable Rate Computation Layer.
Correctly distinguishes RF complex-baseband Shannon capacity from
VLC Intensity-Modulation / Direct-Detection (IM/DD) optical capacity lower bounds.
"""

from typing import Tuple, Optional, Union
import numpy as np


def calculate_rf_rate(sinr_linear: Union[float, np.ndarray], bandwidth_hz: float = 1.0e9) -> Union[float, np.ndarray]:
    """
    Calculate achievable data rate for RF complex-baseband channel using Shannon capacity formula:
    R_RF = B_RF * log2(1 + SINR_RF)   [bits/second]
    
    Parameters:
        sinr_linear: Signal-to-Interference-plus-Noise Ratio in linear scale
        bandwidth_hz: RF channel bandwidth in Hz (default: 1 GHz)
        
    Returns:
        Achievable rate in bits/second.
    """
    sinr = np.maximum(np.asarray(sinr_linear, dtype=float), 0.0)
    rate = bandwidth_hz * np.log2(1.0 + sinr)
    return float(rate) if np.isscalar(sinr_linear) else rate


def calculate_vlc_rate(sinr_elec_linear: Union[float, np.ndarray], bandwidth_hz: float = 20.0e6) -> Union[float, np.ndarray]:
    """
    Calculate achievable data rate for VLC IM/DD optical channel.
    Uses the rigorously derived Lapidoth / Hranilovic / Haas IM/DD capacity lower bound
    for non-negative real optical signals with average optical power constraint:
    R_VLC = (B_VLC / 2) * log2(1 + (e / (2*pi)) * SINR_elec)   [bits/second]
    
    Parameters:
        sinr_elec_linear: Electrical SINR in linear scale (>= 0)
        bandwidth_hz: Optical modulation bandwidth in Hz (default: 20 MHz)
        
    Returns:
        Achievable rate in bits/second.
    """
    sinr = np.maximum(np.asarray(sinr_elec_linear, dtype=float), 0.0)
    # Factor e / (2 * pi) approx 0.4328
    factor = np.e / (2.0 * np.pi)
    rate = (bandwidth_hz / 2.0) * np.log2(1.0 + factor * sinr)
    return float(rate) if np.isscalar(sinr_elec_linear) else rate


def calculate_hybrid_rate(rf_rate: Union[float, np.ndarray], vlc_rate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate aggregate achievable rate for hybrid RF/VLC dual-connectivity:
    In parallel multi-homing transmission, non-overlapping frequency bands
    (140 GHz sub-THz and 400-700 THz optical) provide mutually orthogonal communication channels,
    allowing independent parallel stream multiplexing:
    R_hybrid = R_RF + R_VLC   [bits/second]
    
    Parameters:
        rf_rate: RF link rate in bps
        vlc_rate: VLC link rate in bps
        
    Returns:
        Total aggregated rate in bps.
    """
    return np.asarray(rf_rate) + np.asarray(vlc_rate)
