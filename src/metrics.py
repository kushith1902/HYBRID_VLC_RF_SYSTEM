"""
Performance Metrics Calculation Library for 6G Wireless Networks.
Computes Jain's Fairness Index, Outage/Coverage Probabilities,
Energy Efficiency, Spectral Efficiency, and Latency Statistics.
All calculated directly from simulation and model outputs.
"""

from typing import Dict, Any, List, Union
import numpy as np


def calculate_jains_fairness_index(rates: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Jain's Fairness Index:
    J = (sum(R_k))^2 / (K * sum(R_k^2))
    
    Range: [1/K, 1.0]. A value of 1.0 denotes perfect equality.
    """
    r = np.asarray(rates, dtype=float)
    K = len(r)
    if K == 0 or np.sum(r) <= 1e-12:
        return 0.0
    sum_r = np.sum(r)
    sum_r_sq = np.sum(r ** 2)
    if sum_r_sq <= 1e-15:
        return 0.0
    fairness = (sum_r ** 2) / (K * sum_r_sq)
    return float(np.clip(fairness, 1.0 / K, 1.0))


def calculate_outage_probability(user_rates: np.ndarray, min_rate_bps: float) -> float:
    """
    Calculate outage probability:
    P_out = (Count of users with R_k < R_min) / (Total evaluated users)
    """
    r = np.asarray(user_rates, dtype=float)
    if len(r) == 0:
        return 0.0
    outage_count = np.sum(r < min_rate_bps)
    return float(outage_count / len(r))


def calculate_coverage_probability(user_rates: np.ndarray, min_rate_bps: float) -> float:
    """Coverage probability = 1 - Outage probability."""
    return float(1.0 - calculate_outage_probability(user_rates, min_rate_bps))


def calculate_spectral_efficiency(sum_rate_bps: float, bandwidth_hz: float) -> float:
    """
    Calculate Spectral Efficiency in bps/Hz:
    SE = Sum Rate / Bandwidth
    """
    if bandwidth_hz <= 0.0:
        return 0.0
    return float(sum_rate_bps / bandwidth_hz)


def calculate_latency_percentiles(latencies_ms: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Compute mean, median (p50), and 95th percentile latency."""
    l_arr = np.asarray(latencies_ms, dtype=float)
    if len(l_arr) == 0:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0}
    return {
        'mean': float(np.mean(l_arr)),
        'p50': float(np.median(l_arr)),
        'p95': float(np.percentile(l_arr, 95))
    }
