"""
Power Allocation and Hardware Electrical Consumption Models.
Computes RF/VLC power mapping via parameter alpha, RSMA power split via beta,
and calculates total consumed electrical power for Energy Efficiency (EE) analysis.
Reference: Auer et al., EARTH Project; Chen et al., IEEE Trans. Green Commun. Netw.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from config.default_config import PowerConsumptionConfig, RFConfig, VLCConfig


class PowerAllocationModel:
    """
    Manages physical power distribution and electrical consumption.
    """
    def __init__(self, power_cfg: Optional[PowerConsumptionConfig] = None,
                 rf_cfg: Optional[RFConfig] = None,
                 vlc_cfg: Optional[VLCConfig] = None):
        self.power_cfg = power_cfg or PowerConsumptionConfig()
        self.rf_cfg = rf_cfg or RFConfig()
        self.vlc_cfg = vlc_cfg or VLCConfig()

    def allocate_transmit_powers(self, alpha: float) -> Tuple[float, float]:
        """
        Map alpha in [0, 1] to physical RF and VLC transmit powers:
        P_RF,tx = alpha * P_RF,max
        P_VLC,tx = (1 - alpha) * P_VLC,max
        
        Parameters:
            alpha: RF power allocation ratio in [0.0, 1.0]
            
        Returns:
            p_rf_tx_w: RF radiated power in Watts
            p_vlc_opt_w: Total VLC radiated optical power in Watts
        """
        alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
        p_rf_tx_w = alpha_clamped * self.rf_cfg.tx_power_max_w
        p_vlc_opt_w = (1.0 - alpha_clamped) * self.vlc_cfg.total_optical_power_max_w
        return p_rf_tx_w, p_vlc_opt_w

    def calculate_total_consumed_power(self, p_rf_tx_w: float, p_vlc_opt_w: float,
                                       num_users: int = 4) -> float:
        """
        Calculate total electrical power drawn from the grid (in Watts):
        P_total = P_PA,RF + P_RFIC + P_driver,VLC + P_circuit,VLC + P_static + P_user,Rx
        
        Where:
            P_PA,RF = P_RF,tx / eta_PA (RF Power Amplifier consumption)
            P_driver,VLC = P_VLC,opt / eta_LED (LED driver and electrical consumption)
        """
        p_rf_tx = max(0.0, float(p_rf_tx_w))
        p_vlc_opt = max(0.0, float(p_vlc_opt_w))
        
        # RF power amplifier electrical consumption
        p_pa_rf = p_rf_tx / self.power_cfg.rf_pa_efficiency
        p_rfic = self.power_cfg.rf_rfic_power_w
        
        # VLC driver electrical consumption
        p_driver_vlc = p_vlc_opt / self.power_cfg.vlc_led_efficiency
        p_vlc_circuit = self.vlc_cfg.num_leds * self.power_cfg.vlc_circuit_power_per_led_w
        
        # Static central controller + baseband DSP power
        p_static = self.power_cfg.static_dsp_power_w
        
        # Mobile receivers' circuit power
        p_rx_users = num_users * self.power_cfg.rx_circuit_power_per_user_w
        
        total_elec_w = p_pa_rf + p_rfic + p_driver_vlc + p_vlc_circuit + p_static + p_rx_users
        return float(total_elec_w)

    def calculate_energy_efficiency(self, sum_rate_bps: float, total_elec_power_w: float) -> float:
        """
        Calculate Energy Efficiency (EE) in bits / Joule:
        EE = Total Delivered Sum Rate (bps) / Total Consumed Electrical Power (Watts)
        """
        if total_elec_power_w <= 1e-9:
            return 0.0
        return float(sum_rate_bps / total_elec_power_w)
