"""
140 GHz Sub-THz Radio Frequency (RF) Channel Model.
Implements the Close-In (CI) free-space reference distance path-loss model,
atmospheric absorption, shadow fading, Rician/Rayleigh small-scale fading,
directional antenna gains, and receiver thermal noise.
Reference: Rappaport et al., IEEE Trans. Antennas Propag. (sub-THz indoor measurements).
"""

from typing import Tuple, Optional
import numpy as np
from config.default_config import RFConfig


class RFChannelModel:
    """
    Sub-THz 140 GHz RF Propagation and Channel Gain Model.
    """
    def __init__(self, config: Optional[RFConfig] = None, seed: Optional[int] = None):
        self.config = config or RFConfig()
        self.rng = np.random.default_rng(seed)
        
        # Speed of light
        self.c = 3.0e8
        
        # Free-space path loss at d0 = 1.0 m in dB
        # FSPL(f_c, 1m) = 20*log10(4*pi*d0*f_c / c)
        self.d0 = 1.0
        self.fspl_1m_db = float(20.0 * np.log10(4.0 * np.pi * self.d0 * self.config.carrier_frequency / self.c))
        
        # Thermal noise power in Watts: sigma^2 = k_B * T * B * 10^(NF/10)
        k_b = 1.380649e-23 # Boltzmann constant J/K
        noise_figure_linear = 10.0 ** (self.config.noise_figure_db / 10.0)
        self.noise_power_w = float(k_b * self.config.temperature_k * self.config.bandwidth * noise_figure_linear)
        self.noise_power_dbm = float(10.0 * np.log10(self.noise_power_w) + 30.0)

    def calculate_path_loss_db(self, distances: np.ndarray, is_los: Optional[np.ndarray] = None,
                               enable_shadowing: bool = True) -> np.ndarray:
        """
        Calculate Close-In (CI) path loss in dB:
        PL(d) = FSPL(d0) + 10 * n * log10(d / d0) + alpha_abs * d + X_sigma
        
        Parameters:
            distances: 1D array of distances in meters (must be > 0)
            is_los: Boolean array indicating LOS status (True=LOS, False=NLOS)
            enable_shadowing: Whether to include log-normal shadow fading
            
        Returns:
            Path loss in dB (1D array)
        """
        d = np.maximum(np.asarray(distances, dtype=float), self.d0)
        num_users = len(d)
        
        if is_los is None:
            # Default to LOS for unobstructed indoor
            is_los = np.ones(num_users, dtype=bool)
        else:
            is_los = np.asarray(is_los, dtype=bool)
            
        path_loss_exp = np.where(is_los, self.config.path_loss_exp_los, self.config.path_loss_exp_nlos)
        shadow_std = np.where(is_los, self.config.shadow_std_los_db, self.config.shadow_std_nlos_db)
        
        # Distance-dependent CI path loss
        pl_ci = self.fspl_1m_db + 10.0 * path_loss_exp * np.log10(d / self.d0)
        
        # Molecular / atmospheric absorption at 140 GHz
        pl_abs = self.config.molecular_absorption_db_m * d
        
        # Shadow fading
        if enable_shadowing:
            shadowing = self.rng.normal(0.0, shadow_std, size=num_users)
        else:
            shadowing = np.zeros(num_users)
            
        return pl_ci + pl_abs + shadowing

    def calculate_channel_gains(self, distances: np.ndarray, is_los: Optional[np.ndarray] = None,
                                enable_fading: bool = True) -> np.ndarray:
        """
        Calculate linear channel power gains |h_RF,k|^2.
        |h_RF,k|^2 = 10^(- (PL - G_tx - G_rx) / 10) * |g_small|^2
        
        Parameters:
            distances: (K,) user distances to RF AP in meters
            is_los: (K,) boolean array of LOS states
            enable_fading: Whether to include Rician / Rayleigh small scale fading
            
        Returns:
            channel_gains: (K,) linear power gain
        """
        dists = np.asarray(distances, dtype=float)
        num_users = len(dists)
        
        pl_db = self.calculate_path_loss_db(dists, is_los=is_los, enable_shadowing=True)
        total_gain_dbi = self.config.tx_antenna_gain_dbi + self.config.rx_antenna_gain_dbi
        
        # Large scale channel gain
        large_scale_gain_linear = 10.0 ** (-(pl_db - total_gain_dbi) / 10.0)
        
        if enable_fading:
            # Rician fading for LOS (K_factor = 8 dB), Rayleigh for NLOS (K_factor = 0)
            if is_los is None:
                is_los = np.ones(num_users, dtype=bool)
                
            k_factor_db = np.where(is_los, 8.0, -100.0)
            k_linear = 10.0 ** (k_factor_db / 10.0)
            
            # Non-central chi-squared with 2 DOF, normalized mean = 1
            # Signal envelope |g|^2
            small_scale_fading = np.zeros(num_users)
            for i in range(num_users):
                k_val = k_linear[i]
                if k_val > 1e-3:
                    mu = np.sqrt(k_val / (k_val + 1.0))
                    sigma = np.sqrt(0.5 / (k_val + 1.0))
                    re = self.rng.normal(mu, sigma)
                    im = self.rng.normal(0.0, sigma)
                    small_scale_fading[i] = re**2 + im**2
                else:
                    # Rayleigh: exponential with mean 1
                    small_scale_fading[i] = self.rng.exponential(1.0)
        else:
            small_scale_fading = np.ones(num_users)
            
        channel_gains = large_scale_gain_linear * small_scale_fading
        return np.maximum(channel_gains, 1e-25)

    def calculate_snr_linear(self, tx_power_w: float, channel_gains: np.ndarray) -> np.ndarray:
        """
        Calculate RF Signal-to-Noise Ratio (SNR) in linear scale:
        SNR_k = (P_tx * |h_k|^2) / sigma_RF^2
        """
        rx_power = tx_power_w * np.asarray(channel_gains, dtype=float)
        return rx_power / self.noise_power_w

    def calculate_snr_db(self, tx_power_w: float, channel_gains: np.ndarray) -> np.ndarray:
        """Calculate RF SNR in dB."""
        snr_lin = self.calculate_snr_linear(tx_power_w, channel_gains)
        return 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
