"""
Visible Light Communication (VLC) Intensity-Modulation / Direct-Detection (IM/DD) Channel Model.
Implements Lambertian optical radiation, receiver optical concentrator and filter gains,
incidence/irradiance geometry, line-of-sight (LOS) optical blockage,
photodiode responsivity, and shot/thermal electrical noise.
Reference: Kahn and Barry, Proc. IEEE 1997; Komine and Nakagawa, IEEE Trans. Consum. Electron. 2004.
"""

from typing import Tuple, Optional
import numpy as np
from config.default_config import VLCConfig


class VLCChannelModel:
    """
    Indoor VLC IM/DD Channel Model.
    """
    def __init__(self, config: Optional[VLCConfig] = None):
        self.config = config or VLCConfig()
        
        # Precompute constants
        self.m = self.config.lambertian_order
        self.fov_rad = np.radians(self.config.pd_fov_deg)
        self.a_pd = self.config.pd_area_m2
        self.g_conc = self.config.optical_concentrator_gain
        self.t_s = self.config.optical_filter_gain
        self.r_pd = self.config.pd_responsivity
        self.b_vlc = self.config.bandwidth
        
        # Physical constants
        self.q_electron = 1.602176634e-19 # Coulombs
        self.i2_noise_bandwidth_factor = 0.562 # Noise bandwidth factor for rectangular pulse
        
        # Thermal noise variance in A^2
        self.sigma_thermal_sq = self.config.thermal_noise_density * self.b_vlc

    def calculate_dc_gain_matrix(self, distances: np.ndarray, phi_rad: np.ndarray,
                                psi_rad: np.ndarray, blockage_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate optical DC channel gain H_VLC(0) for all (LED, user) pairs.
        
        H = [(m+1) * A_PD / (2*pi*d^2)] * cos^m(phi) * cos(psi) * T_s * g(psi)   for 0 <= psi <= FOV
        H = 0                                                                   for psi > FOV or Blocked
        
        Parameters:
            distances: (num_leds, K) distances in meters
            phi_rad: (num_leds, K) irradiance angles in radians
            psi_rad: (num_leds, K) incidence angles in radians
            blockage_matrix: (num_leds, K) 1 if blocked by obstacle, 0 if clear LOS
            
        Returns:
            h_matrix: (num_leds, K) linear optical DC channel gains
        """
        d = np.asarray(distances, dtype=float)
        phi = np.asarray(phi_rad, dtype=float)
        psi = np.asarray(psi_rad, dtype=float)
        
        # Check FOV constraint: psi must be <= FOV
        in_fov = (psi >= 0.0) & (psi <= self.fov_rad)
        
        # Cosines (only positive values, non-negative optical radiation)
        cos_phi = np.maximum(np.cos(phi), 0.0)
        cos_psi = np.maximum(np.cos(psi), 0.0)
        
        # Geometric Lambertian gain
        h_geom = ((self.m + 1.0) * self.a_pd / (2.0 * np.pi * (d ** 2))) * (cos_phi ** self.m) * cos_psi
        h_matrix = h_geom * self.t_s * self.g_conc * in_fov.astype(float)
        
        # Apply physical blockage: if blocked, DC gain is 0 (LOS-dominated indoor model)
        if blockage_matrix is not None:
            blocked = np.asarray(blockage_matrix, dtype=bool)
            h_matrix = np.where(blocked, 0.0, h_matrix)
            
        return np.maximum(h_matrix, 0.0)

    def get_effective_user_channel_gains(self, h_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each user, select the strongest unblocked LED transmitter.
        
        Parameters:
            h_matrix: (num_leds, K) DC channel gains
            
        Returns:
            best_h: (K,) effective optical channel gain for each user
            best_led_indices: (K,) index of serving LED for each user
        """
        best_led_indices = np.argmax(h_matrix, axis=0)
        best_h = np.max(h_matrix, axis=0)
        return best_h, best_led_indices

    def calculate_noise_variance(self, rx_optical_power: np.ndarray) -> np.ndarray:
        """
        Calculate total electrical noise variance sigma_VLC^2 = sigma_shot^2 + sigma_thermal^2.
        
        sigma_shot^2 = 2 * q * R_pd * P_rx,opt * B + 2 * q * I_bg * I2 * B
        sigma_thermal^2 = N_thermal * B
        
        Parameters:
            rx_optical_power: (K,) received optical power in Watts
            
        Returns:
            sigma_vlc_sq: (K,) total noise variance in A^2
        """
        p_opt = np.maximum(np.asarray(rx_optical_power, dtype=float), 0.0)
        
        # Shot noise caused by received signal and background solar/ambient light
        shot_signal = 2.0 * self.q_electron * self.r_pd * p_opt * self.b_vlc
        shot_ambient = 2.0 * self.q_electron * self.config.bg_current * self.i2_noise_bandwidth_factor * self.b_vlc
        sigma_shot_sq = (shot_signal + shot_ambient) * self.config.shot_noise_factor
        
        return sigma_shot_sq + self.sigma_thermal_sq

    def calculate_electrical_snr_linear(self, tx_optical_power_w: float, effective_h: np.ndarray) -> np.ndarray:
        """
        Calculate electrical SNR in linear scale for IM/DD VLC:
        SNR_elec = (R_pd * H * P_tx,opt)^2 / sigma_VLC^2
        
        Parameters:
            tx_optical_power_w: Radiated optical power per transmitter in Watts
            effective_h: (K,) optical DC channel gain
            
        Returns:
            snr_elec: (K,) linear electrical SNR
        """
        h = np.maximum(np.asarray(effective_h, dtype=float), 0.0)
        rx_optical_power = tx_optical_power_w * h
        noise_sq = self.calculate_noise_variance(rx_optical_power)
        
        rx_electrical_power = (self.r_pd * rx_optical_power) ** 2
        
        snr_elec = np.where(h > 1e-15, rx_electrical_power / noise_sq, 0.0)
        return snr_elec

    def calculate_electrical_snr_db(self, tx_optical_power_w: float, effective_h: np.ndarray) -> np.ndarray:
        """Calculate electrical SNR in dB."""
        snr_lin = self.calculate_electrical_snr_linear(tx_optical_power_w, effective_h)
        return 10.0 * np.log10(np.maximum(snr_lin, 1e-12))
