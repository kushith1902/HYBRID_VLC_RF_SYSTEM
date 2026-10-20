"""
Network Transmission Evaluation and Optimization Objective Function.
Computes per-user achievable rates, total electrical power consumption,
energy efficiency (EE in bits/J), and verifies physical constraints.
"""

from typing import Dict, Any, Optional
import numpy as np
from config.default_config import SystemConfig
from src.rsma import compute_rsma_rf_rates, compute_rsma_vlc_rates
from src.power_allocation import PowerAllocationModel


class SystemEvaluator:
    """
    Evaluates communication performance metrics for given channel realizations,
    link selection decisions, and power allocation parameters (alpha, beta).
    """
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.power_model = PowerAllocationModel(
            power_cfg=self.config.power,
            rf_cfg=self.config.rf,
            vlc_cfg=self.config.vlc
        )

    def evaluate_state(self, rf_channel_gains: np.ndarray,
                       vlc_effective_h: np.ndarray,
                       is_blocked: np.ndarray,
                       modes: np.ndarray,
                       alpha: float,
                       beta: float,
                       min_rate_bps: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate full network state for candidate (modes, alpha, beta).
        
        Parameters:
            rf_channel_gains: (K,) |h_RF,k|^2 linear RF power gains
            vlc_effective_h: (K,) H_VLC,k optical DC channel gains
            is_blocked: (K,) boolean optical blockage flags
            modes: (K,) link mode per user: 0=RF, 1=VLC, 2=Hybrid
            alpha: RF power allocation factor in [0, 1]
            beta: RSMA common stream power fraction in [0, 1]
            min_rate_bps: Minimum required rate constraint per user (default from config)
            
        Returns:
            Dictionary containing:
                'user_rates': (K,) achievable rates in bps
                'rf_rates': (K,) RF component rates in bps
                'vlc_rates': (K,) VLC component rates in bps
                'sum_rate': float sum rate in bps
                'p_rf_tx_w': float RF transmit power in Watts
                'p_vlc_opt_w': float VLC optical power in Watts
                'total_elec_power_w': float total electrical power in Watts
                'energy_efficiency': float EE in bits/Joule
                'is_feasible': bool True if all QoS constraints are met
                'min_rate_satisfied': (K,) bool per-user rate feasibility
                'outage_users': int count of users with R_k < min_rate
        """
        K = len(rf_channel_gains)
        min_rate = min_rate_bps if min_rate_bps is not None else self.config.min_user_rate_bps
        
        # 1. Allocate transmit powers from alpha
        p_rf_tx_w, p_vlc_opt_w = self.power_model.allocate_transmit_powers(alpha)
        
        # 2. Identify active users on each link
        # Mode 0: RF only, Mode 1: VLC only, Mode 2: Hybrid (both)
        rf_active_mask = (modes == 0) | (modes == 2)
        vlc_active_mask = (modes == 1) | (modes == 2)
        
        rf_user_rates = np.zeros(K)
        vlc_user_rates = np.zeros(K)
        
        # 3. Evaluate RF RSMA Transmission for RF-active users
        if np.any(rf_active_mask) and p_rf_tx_w > 1e-6:
            # Active RF channels
            active_rf_indices = np.where(rf_active_mask)[0]
            sub_rf_gains = rf_channel_gains[active_rf_indices]
            
            # Sub-RF RSMA
            k_active = len(active_rf_indices)
            p_sub_weights = np.full(k_active, 1.0 / k_active)
            c_sub_weights = np.full(k_active, 1.0 / k_active)
            
            # Thermal noise from RF config
            noise_w = 10.0 ** ((self.config.rf.noise_figure_db - 174.0 + 10.0 * np.log10(self.config.rf.bandwidth) - 30.0) / 10.0)
            
            rf_res = compute_rsma_rf_rates(
                channel_gains=sub_rf_gains,
                noise_power_w=noise_w,
                tx_power_w=p_rf_tx_w,
                beta=beta,
                bandwidth_hz=self.config.rf.bandwidth,
                private_weights=p_sub_weights,
                common_rate_weights=c_sub_weights
            )
            rf_user_rates[active_rf_indices] = rf_res['total_user_rates']
            
        # 4. Evaluate VLC RSMA Transmission for VLC-active users
        if np.any(vlc_active_mask) and p_vlc_opt_w > 1e-3:
            active_vlc_indices = np.where(vlc_active_mask)[0]
            sub_vlc_gains = vlc_effective_h[active_vlc_indices].copy()
            
            # Blocked users have 0 VLC gain
            for idx, user_idx in enumerate(active_vlc_indices):
                if is_blocked[user_idx]:
                    sub_vlc_gains[idx] = 0.0
                    
            k_vlc = len(active_vlc_indices)
            p_vlc_weights = np.full(k_vlc, 1.0 / k_vlc)
            c_vlc_weights = np.full(k_vlc, 1.0 / k_vlc)
            
            noise_vlc = self.config.vlc.thermal_noise_density * self.config.vlc.bandwidth
            
            vlc_res = compute_rsma_vlc_rates(
                effective_h=sub_vlc_gains,
                tx_optical_power_w=p_vlc_opt_w,
                beta=beta,
                bandwidth_hz=self.config.vlc.bandwidth,
                r_pd=self.config.vlc.pd_responsivity,
                noise_var_sq=noise_vlc,
                private_weights=p_vlc_weights,
                common_rate_weights=c_vlc_weights
            )
            vlc_user_rates[active_vlc_indices] = vlc_res['total_user_rates']
            
        # 5. Total User Rates: R_k = R_RF,k + R_VLC,k
        total_user_rates = rf_user_rates + vlc_user_rates
        sum_rate = float(np.sum(total_user_rates))
        
        # 6. Electrical Power & Energy Efficiency
        total_elec_power_w = self.power_model.calculate_total_consumed_power(
            p_rf_tx_w=p_rf_tx_w,
            p_vlc_opt_w=p_vlc_opt_w,
            num_users=K
        )
        ee = self.power_model.calculate_energy_efficiency(sum_rate, total_elec_power_w)
        
        # 7. Constraint and Outage Checking
        min_rate_satisfied = total_user_rates >= (min_rate - 1e-6)
        is_feasible = bool(np.all(min_rate_satisfied))
        outage_count = int(np.sum(~min_rate_satisfied))
        
        return {
            'user_rates': total_user_rates,
            'rf_rates': rf_user_rates,
            'vlc_rates': vlc_user_rates,
            'sum_rate': sum_rate,
            'p_rf_tx_w': p_rf_tx_w,
            'p_vlc_opt_w': p_vlc_opt_w,
            'total_elec_power_w': total_elec_power_w,
            'energy_efficiency': ee,
            'is_feasible': is_feasible,
            'min_rate_satisfied': min_rate_satisfied,
            'outage_users': outage_count
        }
