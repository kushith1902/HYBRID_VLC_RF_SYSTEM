"""
Optimization Teacher for Joint Link Selection and RSMA Power Allocation.
Performs structured search over link assignments and (alpha, beta) power allocations
to maximize Energy Efficiency (EE) [bits/Joule] subject to per-user QoS rate constraints.
Generates genuine physical target labels for AI neural-network training.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from itertools import product
from config.default_config import SystemConfig
from src.objective import SystemEvaluator
from src.link_selection import rule_based_link_selection


class OptimizationTeacher:
    """
    Teacher Optimizer that explores the feasible action space to find
    optimal or near-optimal joint link selection and (alpha, beta) power allocations.
    """
    def __init__(self, config: Optional[SystemConfig] = None,
                 alpha_grid_points: int = 9,
                 beta_grid_points: int = 7):
        self.config = config or SystemConfig()
        self.evaluator = SystemEvaluator(self.config)
        
        # Power allocation search grids
        self.alpha_grid = np.linspace(0.1, 0.9, alpha_grid_points)
        self.beta_grid = np.linspace(0.05, 0.50, beta_grid_points)

    def optimize(self, rf_channel_gains: np.ndarray,
                 vlc_effective_h: np.ndarray,
                 is_blocked: np.ndarray,
                 rf_snr_db: Optional[np.ndarray] = None,
                 min_rate_bps: Optional[float] = None) -> Dict[str, Any]:
        """
        Run joint optimization to find best (modes, alpha, beta) maximizing Energy Efficiency.
        
        Parameters:
            rf_channel_gains: (K,) |h_RF,k|^2 linear channel gains
            vlc_effective_h: (K,) H_VLC,k optical DC channel gains
            is_blocked: (K,) optical blockage flags
            rf_snr_db: (K,) RF SNR in dB (optional, for heuristic generation)
            min_rate_bps: QoS minimum rate constraint per user
            
        Returns:
            Dictionary containing best allocation, objective value, feasibility, and runtime.
        """
        start_time = time.perf_counter()
        K = len(rf_channel_gains)
        min_rate = min_rate_bps if min_rate_bps is not None else self.config.min_user_rate_bps
        
        # Candidate mode vectors to evaluate
        candidate_modes = []
        
        # 1. Structural baselines
        candidate_modes.append(np.zeros(K, dtype=int)) # All RF
        if not np.all(is_blocked):
            candidate_modes.append(np.ones(K, dtype=int)) # All VLC
        candidate_modes.append(np.full(K, 2, dtype=int)) # All Hybrid
        
        # 2. Rule-based mode vector
        if rf_snr_db is None:
            # Estimate SNR from gains
            noise_w = 10.0 ** ((self.config.rf.noise_figure_db - 174.0 + 10.0 * np.log10(self.config.rf.bandwidth) - 30.0) / 10.0)
            rf_snr_db = 10.0 * np.log10(np.maximum(self.config.rf.tx_power_max_w * rf_channel_gains / noise_w, 1e-12))
            
        rule_mode = rule_based_link_selection(vlc_effective_h, rf_snr_db, is_blocked)
        candidate_modes.append(rule_mode)
        
        # 3. Targeted mode exploration:
        # For each user, if blocked -> mode 0 (RF)
        # If unblocked -> test mode 1 (VLC) and mode 2 (Hybrid)
        unblocked_indices = [k for k in range(K) if not is_blocked[k]]
        if len(unblocked_indices) <= 4:
            # Enumerate modes for unblocked users
            for choices in product([0, 1, 2], repeat=len(unblocked_indices)):
                m = np.zeros(K, dtype=int)
                for idx, u_idx in enumerate(unblocked_indices):
                    m[u_idx] = choices[idx]
                candidate_modes.append(m)
                
        # Remove duplicate mode arrays
        unique_modes = []
        seen = set()
        for m in candidate_modes:
            key = tuple(m.tolist())
            if key not in seen:
                seen.add(key)
                unique_modes.append(m)
                
        best_result = None
        best_ee = -1.0
        best_feasible_result = None
        best_feasible_ee = -1.0
        
        # Evaluate combinations of (modes, alpha, beta)
        for m in unique_modes:
            for alpha in self.alpha_grid:
                for beta in self.beta_grid:
                    res = self.evaluator.evaluate_state(
                        rf_channel_gains=rf_channel_gains,
                        vlc_effective_h=vlc_effective_h,
                        is_blocked=is_blocked,
                        modes=m,
                        alpha=alpha,
                        beta=beta,
                        min_rate_bps=min_rate
                    )
                    
                    ee = res['energy_efficiency']
                    
                    if res['is_feasible']:
                        if ee > best_feasible_ee:
                            best_feasible_ee = ee
                            best_feasible_result = {
                                **res,
                                'modes': m.copy(),
                                'alpha': float(alpha),
                                'beta': float(beta)
                            }
                    else:
                        if ee > best_ee:
                            best_ee = ee
                            best_result = {
                                **res,
                                'modes': m.copy(),
                                'alpha': float(alpha),
                                'beta': float(beta)
                            }
                            
        runtime = time.perf_counter() - start_time
        
        # Select best feasible if available, otherwise best candidate marked infeasible
        if best_feasible_result is not None:
            final_res = best_feasible_result
            final_res['solution_type'] = "Optimal_Feasible"
        elif best_result is not None:
            final_res = best_result
            final_res['solution_type'] = "Best_Effort_Infeasible"
        else:
            # Fallback
            final_res = self.evaluator.evaluate_state(
                rf_channel_gains, vlc_effective_h, is_blocked,
                modes=np.zeros(K, dtype=int), alpha=0.5, beta=0.3, min_rate_bps=min_rate
            )
            final_res['modes'] = np.zeros(K, dtype=int)
            final_res['alpha'] = 0.5
            final_res['beta'] = 0.3
            final_res['solution_type'] = "Fallback"
            
        final_res['optimization_runtime_sec'] = runtime
        return final_res
