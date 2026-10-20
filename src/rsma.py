"""
Rate-Splitting Multiple Access (RSMA), SDMA, and NOMA Transmission Models.
Implements 1-layer downlink RSMA with common/private stream splitting,
Successive Interference Cancellation (SIC), common rate partitioning,
and exact comparisons with SDMA (beta=0) and Power-Domain NOMA.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from src.noise_models import calculate_rf_rate, calculate_vlc_rate


def compute_rsma_rf_rates(channel_gains: np.ndarray, noise_power_w: float,
                          tx_power_w: float, beta: float,
                          bandwidth_hz: float = 1.0e9,
                          private_weights: Optional[np.ndarray] = None,
                          common_rate_weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute RSMA rates for 140 GHz RF downlink.
    
    Parameters:
        channel_gains: (K,) linear channel power gains |h_RF,k|^2
        noise_power_w: Thermal noise power in Watts
        tx_power_w: Total RF transmit power in Watts
        beta: RSMA common stream power allocation fraction in [0.0, 1.0]
        bandwidth_hz: RF bandwidth in Hz
        private_weights: (K,) power fractions for private streams, sum to 1.0
        common_rate_weights: (K,) fractions to split common rate among users, sum to 1.0
        
    Returns:
        Dictionary containing:
            'common_sinr': (K,) SINR of common stream at each user
            'private_sinr': (K,) SINR of private stream after SIC
            'common_rate_max': float, bottleneck common rate capacity
            'common_user_rates': (K,) common rate portion allocated to each user
            'private_user_rates': (K,) private rate for each user
            'total_user_rates': (K,) R_k = c_k + R_p,k
            'sum_rate': float
            'sic_feasible': (K,) boolean feasibility flags
    """
    g = np.maximum(np.asarray(channel_gains, dtype=float), 1e-25)
    num_users = len(g)
    beta = float(np.clip(beta, 0.0, 1.0))
    
    if private_weights is None:
        p_weights = np.full(num_users, 1.0 / num_users)
    else:
        p_weights = np.asarray(private_weights, dtype=float)
        p_weights = p_weights / np.sum(p_weights)
        
    if common_rate_weights is None:
        c_weights = np.full(num_users, 1.0 / num_users)
    else:
        c_weights = np.asarray(common_rate_weights, dtype=float)
        c_weights = c_weights / np.sum(c_weights)
        
    # Power allocation:
    # P_common = beta * P_tx
    # P_private_k = (1 - beta) * P_tx * w_k
    p_common = beta * tx_power_w
    p_private = (1.0 - beta) * tx_power_w * p_weights
    p_private_sum = np.sum(p_private)
    
    # 1. Common Stream Decoding at each user k:
    # SINR_c,k = (|h_k|^2 * P_common) / (|h_k|^2 * sum_j P_private_j + sigma_k^2)
    common_interference = g * p_private_sum + noise_power_w
    common_sinr = np.where(p_common > 1e-15, (g * p_common) / common_interference, 0.0)
    
    # Common stream rates decodable by user k
    common_rates_per_user = calculate_rf_rate(common_sinr, bandwidth_hz)
    
    # The common stream must be decodable by all users -> bottleneck rate
    common_rate_max = float(np.min(common_rates_per_user)) if p_common > 1e-15 else 0.0
    common_user_rates = common_rate_max * c_weights
    
    # 2. Private Stream Decoding at each user k after ideal SIC of common stream:
    # SINR_p,k = (|h_k|^2 * P_p,k) / (|h_k|^2 * sum_{j != k} P_p,j + sigma_k^2)
    private_sinr = np.zeros(num_users)
    for k in range(num_users):
        other_private_power = p_private_sum - p_private[k]
        denom = g[k] * other_private_power + noise_power_w
        private_sinr[k] = (g[k] * p_private[k]) / denom if p_private[k] > 1e-15 else 0.0
        
    private_user_rates = calculate_rf_rate(private_sinr, bandwidth_hz)
    total_user_rates = common_user_rates + private_user_rates
    
    # SIC feasibility: User can decode common stream if common_sinr > min threshold (-3 dB)
    sic_feasible = common_sinr >= 10.0 ** (-3.0 / 10.0)
    
    return {
        'common_sinr': common_sinr,
        'private_sinr': private_sinr,
        'common_rate_max': common_rate_max,
        'common_user_rates': common_user_rates,
        'private_user_rates': private_user_rates,
        'total_user_rates': total_user_rates,
        'sum_rate': float(np.sum(total_user_rates)),
        'sic_feasible': sic_feasible
    }


def compute_rsma_vlc_rates(effective_h: np.ndarray, tx_optical_power_w: float,
                           beta: float, bandwidth_hz: float = 20.0e6,
                           r_pd: float = 0.53, noise_var_sq: float = 2.0e-15,
                           private_weights: Optional[np.ndarray] = None,
                           common_rate_weights: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute RSMA rates for VLC IM/DD optical channel.
    Uses electrical SNR and IM/DD optical capacity lower bound.
    """
    h = np.maximum(np.asarray(effective_h, dtype=float), 0.0)
    num_users = len(h)
    beta = float(np.clip(beta, 0.0, 1.0))
    
    if private_weights is None:
        p_weights = np.full(num_users, 1.0 / num_users)
    else:
        p_weights = np.asarray(private_weights, dtype=float)
        p_weights = p_weights / np.sum(p_weights)
        
    if common_rate_weights is None:
        c_weights = np.full(num_users, 1.0 / num_users)
    else:
        c_weights = np.asarray(common_rate_weights, dtype=float)
        c_weights = c_weights / np.sum(c_weights)
        
    # Optical powers
    p_opt_c = beta * tx_optical_power_w
    p_opt_p = (1.0 - beta) * tx_optical_power_w * p_weights
    p_opt_p_sum = np.sum(p_opt_p)
    
    # Electrical channel gains: (R_pd * H)^2
    g_elec = (r_pd * h) ** 2
    
    # Common stream electrical SINR
    common_interf_elec = g_elec * (p_opt_p_sum ** 2) + noise_var_sq
    common_signal_elec = g_elec * (p_opt_c ** 2)
    common_sinr_elec = np.where((h > 1e-12) & (p_opt_c > 1e-15), common_signal_elec / common_interf_elec, 0.0)
    
    common_rates_per_user = calculate_vlc_rate(common_sinr_elec, bandwidth_hz)
    # Bottleneck for all active users that have non-zero channel
    active_mask = h > 1e-12
    if np.any(active_mask) and p_opt_c > 1e-15:
        common_rate_max = float(np.min(common_rates_per_user[active_mask]))
    else:
        common_rate_max = 0.0
        
    common_user_rates = np.where(active_mask, common_rate_max * c_weights, 0.0)
    
    # Private stream electrical SINR after SIC
    private_sinr_elec = np.zeros(num_users)
    for k in range(num_users):
        if not active_mask[k]:
            continue
        other_opt_p = p_opt_p_sum - p_opt_p[k]
        denom = g_elec[k] * (other_opt_p ** 2) + noise_var_sq
        sig = g_elec[k] * (p_opt_p[k] ** 2)
        private_sinr_elec[k] = sig / denom if p_opt_p[k] > 1e-15 else 0.0
        
    private_user_rates = calculate_vlc_rate(private_sinr_elec, bandwidth_hz)
    total_user_rates = common_user_rates + private_user_rates
    
    return {
        'common_sinr': common_sinr_elec,
        'private_sinr': private_sinr_elec,
        'common_rate_max': common_rate_max,
        'common_user_rates': common_user_rates,
        'private_user_rates': private_user_rates,
        'total_user_rates': total_user_rates,
        'sum_rate': float(np.sum(total_user_rates))
    }


def compute_sdma_rates(channel_gains: np.ndarray, noise_power_w: float,
                       tx_power_w: float, bandwidth_hz: float = 1.0e9,
                       is_vlc: bool = False, r_pd: float = 0.53) -> np.ndarray:
    """
    Spatial Division Multiple Access (SDMA) Baseline:
    Purely private streams (beta = 0). No rate-splitting.
    Users decode private streams directly, treating all co-channel streams as interference.
    """
    if is_vlc:
        res = compute_rsma_vlc_rates(
            effective_h=channel_gains, tx_optical_power_w=tx_power_w,
            beta=0.0, bandwidth_hz=bandwidth_hz, r_pd=r_pd, noise_var_sq=noise_power_w
        )
        return res['total_user_rates']
    else:
        res = compute_rsma_rf_rates(
            channel_gains=channel_gains, noise_power_w=noise_power_w,
            tx_power_w=tx_power_w, beta=0.0, bandwidth_hz=bandwidth_hz
        )
        return res['total_user_rates']


def compute_noma_rates(channel_gains: np.ndarray, noise_power_w: float,
                       tx_power_w: float, bandwidth_hz: float = 1.0e9) -> np.ndarray:
    """
    Power-Domain NOMA Baseline with Successive Interference Cancellation (SIC):
    Users ordered by channel gain: g_(1) <= g_(2) <= ... <= g_(K).
    Weaker users receive more power. Stronger users perform SIC on weaker users.
    """
    g = np.maximum(np.asarray(channel_gains, dtype=float), 1e-25)
    K = len(g)
    
    # Sort users by channel gain
    sort_idx = np.argsort(g) # ascending order (weakest to strongest)
    inv_sort_idx = np.argsort(sort_idx)
    
    # Power allocation inversely proportional to channel rank (standard NOMA)
    # alpha_k = (K - k + 1) / sum(1..K)
    ranks = np.arange(1, K + 1)
    rev_ranks = (K - ranks + 1).astype(float)
    p_fractions = rev_ranks / np.sum(rev_ranks)
    p_alloc = p_fractions * tx_power_w # p_alloc[0] is for weakest user
    
    # User at sorted index i decodes users 0..i-1 via SIC, cancels them,
    # and treats users i+1..K-1 as interference.
    sorted_rates = np.zeros(K)
    for i in range(K):
        user_gain = g[sort_idx[i]]
        # Interference from stronger users:
        interf_power = np.sum(p_alloc[i+1:]) if i < K - 1 else 0.0
        sinr = (user_gain * p_alloc[i]) / (user_gain * interf_power + noise_power_w)
        sorted_rates[i] = calculate_rf_rate(sinr, bandwidth_hz)
        
    # Re-order rates back to original user ordering
    return sorted_rates[inv_sort_idx]
