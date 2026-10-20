"""
Baseline Transmission and Allocation Algorithms for 6G Hybrid RF/VLC Networks.
Implements:
1. RF-only transmission
2. VLC-only transmission
3. Fixed Hybrid (50/50 power allocation)
4. Rule-Based Hybrid RSMA
5. AI Link Selection with Fixed Alpha/Beta
6. Proposed Joint AI (evaluated via model)
7. Multiple access baselines (SDMA, NOMA)
All evaluated under identical channel states and power budgets.
"""

from typing import Dict, Any, Optional
import numpy as np
from config.default_config import SystemConfig
from src.link_selection import rule_based_link_selection, validate_and_project_link_selection
from src.objective import SystemEvaluator


class BaselineManager:
    """
    Executes and records standardized baseline benchmarks.
    """
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.evaluator = SystemEvaluator(self.config)

    def run_rf_only(self, rf_channel_gains: np.ndarray, vlc_effective_h: np.ndarray,
                    is_blocked: np.ndarray, beta: float = 0.3) -> Dict[str, Any]:
        """Baseline 1: All users served via 140 GHz sub-THz RF only."""
        K = len(rf_channel_gains)
        modes = np.zeros(K, dtype=int) # All RF
        alpha = 1.0 # 100% RF power
        res = self.evaluator.evaluate_state(rf_channel_gains, vlc_effective_h, is_blocked, modes, alpha, beta)
        res['method_name'] = "RF-only"
        res['modes'] = modes
        res['alpha'] = alpha
        res['beta'] = beta
        return res

    def run_vlc_only(self, rf_channel_gains: np.ndarray, vlc_effective_h: np.ndarray,
                     is_blocked: np.ndarray, beta: float = 0.3) -> Dict[str, Any]:
        """Baseline 2: All users served via VLC optical only."""
        K = len(rf_channel_gains)
        modes = np.ones(K, dtype=int) # All VLC
        alpha = 0.0 # 100% VLC power
        res = self.evaluator.evaluate_state(rf_channel_gains, vlc_effective_h, is_blocked, modes, alpha, beta)
        res['method_name'] = "VLC-only"
        res['modes'] = modes
        res['alpha'] = alpha
        res['beta'] = beta
        return res

    def run_fixed_hybrid(self, rf_channel_gains: np.ndarray, vlc_effective_h: np.ndarray,
                         is_blocked: np.ndarray, alpha: float = 0.5, beta: float = 0.3) -> Dict[str, Any]:
        """Baseline 3: Fixed Hybrid dual-connectivity with nominal 50/50 power allocation."""
        K = len(rf_channel_gains)
        modes = np.full(K, 2, dtype=int) # All Hybrid
        res = self.evaluator.evaluate_state(rf_channel_gains, vlc_effective_h, is_blocked, modes, alpha, beta)
        res['method_name'] = "Fixed Hybrid (50/50)"
        res['modes'] = modes
        res['alpha'] = alpha
        res['beta'] = beta
        return res

    def run_rule_based_hybrid(self, rf_channel_gains: np.ndarray, vlc_effective_h: np.ndarray,
                              is_blocked: np.ndarray, rf_snr_db: np.ndarray,
                              beta: float = 0.3) -> Dict[str, Any]:
        """Baseline 4: Rule-based link selection with dynamic alpha based on link loading."""
        modes = rule_based_link_selection(vlc_effective_h, rf_snr_db, is_blocked)
        # Adapt alpha: ratio of RF active links to total active links
        num_rf = np.sum((modes == 0) | (modes == 2))
        num_vlc = np.sum((modes == 1) | (modes == 2))
        total_links = max(1, num_rf + num_vlc)
        alpha = float(np.clip(num_rf / total_links, 0.2, 0.8))
        
        res = self.evaluator.evaluate_state(rf_channel_gains, vlc_effective_h, is_blocked, modes, alpha, beta)
        res['method_name'] = "Rule-Based Hybrid RSMA"
        res['modes'] = modes
        res['alpha'] = alpha
        res['beta'] = beta
        return res
