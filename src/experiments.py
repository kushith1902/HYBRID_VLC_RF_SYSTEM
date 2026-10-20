"""
Reproducible Experimental Evaluation and Scientific Study Suite.
Runs configurable experiments:
- Experiment A: Impact of User Count (K = 2, 4, 6, 8)
- Experiment B: Impact of Optical Blockage / Obstacles (0 to 4 obstacles)
- Experiment C: Multi-Access Comparison (RSMA vs SDMA vs NOMA)
- Experiment D: Power Allocation Sensitivity (Sweeping Alpha in [0, 1])
- Experiment E: RSMA Common Stream Sensitivity (Sweeping Beta in [0, 1])
- Experiment F: Comprehensive Baseline Benchmark Comparison
- Experiment G: Ablation Study (Joint Link vs Fixed, Adaptive Alpha/Beta vs Fixed, RSMA vs SDMA)
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from config.default_config import SystemConfig
from src.environment import IndoorEnvironment
from src.rf_channel import RFChannelModel
from src.vlc_channel import VLCChannelModel
from src.geometry import Obstacle3D
from src.objective import SystemEvaluator
from src.baselines import BaselineManager
from src.optimizer import OptimizationTeacher
from src.evaluate import ModelEvaluator
from src.metrics import calculate_jains_fairness_index, calculate_outage_probability, calculate_coverage_probability
from src.rsma import compute_sdma_rates, compute_noma_rates


class ExperimentRunner:
    """
    Manages and runs scientific experiment batches with reproducible seeds.
    """
    def __init__(self, config: Optional[SystemConfig] = None, seed: int = 42):
        self.config = config or SystemConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.rf_model = RFChannelModel(self.config.rf, seed=seed)
        self.vlc_model = VLCChannelModel(self.config.vlc)
        self.evaluator = SystemEvaluator(self.config)
        self.baseline_mgr = BaselineManager(self.config)
        self.teacher = OptimizationTeacher(self.config, alpha_grid_points=7, beta_grid_points=5)

    def run_multi_access_comparison(self, num_trials: int = 20) -> pd.DataFrame:
        """
        Experiment C: Systematic comparison between RSMA, SDMA, and NOMA
        under identical RF channel realizations and transmit powers.
        """
        records = []
        env = IndoorEnvironment(self.config, seed=self.seed)
        noise_w = 10.0 ** ((self.config.rf.noise_figure_db - 174.0 + 10.0 * np.log10(self.config.rf.bandwidth) - 30.0) / 10.0)
        p_tx = self.config.rf.tx_power_max_w
        
        for t in range(num_trials):
            env.reset_users()
            dists = env.get_rf_distances()
            rf_gains = self.rf_model.calculate_channel_gains(dists, enable_fading=True)
            
            # 1. RSMA with adaptive beta = 0.3
            from src.rsma import compute_rsma_rf_rates
            rsma_res = compute_rsma_rf_rates(rf_gains, noise_w, p_tx, beta=0.3, bandwidth_hz=self.config.rf.bandwidth)
            rsma_sum_rate = rsma_res['sum_rate']
            rsma_fairness = calculate_jains_fairness_index(rsma_res['total_user_rates'])
            
            # 2. SDMA (beta = 0.0)
            sdma_rates = compute_sdma_rates(rf_gains, noise_w, p_tx, bandwidth_hz=self.config.rf.bandwidth)
            sdma_sum_rate = float(np.sum(sdma_rates))
            sdma_fairness = calculate_jains_fairness_index(sdma_rates)
            
            # 3. NOMA
            noma_rates = compute_noma_rates(rf_gains, noise_w, p_tx, bandwidth_hz=self.config.rf.bandwidth)
            noma_sum_rate = float(np.sum(noma_rates))
            noma_fairness = calculate_jains_fairness_index(noma_rates)
            
            records.append({
                'trial': t,
                'RSMA_Sum_Rate_Mbps': rsma_sum_rate / 1e6,
                'SDMA_Sum_Rate_Mbps': sdma_sum_rate / 1e6,
                'NOMA_Sum_Rate_Mbps': noma_sum_rate / 1e6,
                'RSMA_Fairness': rsma_fairness,
                'SDMA_Fairness': sdma_fairness,
                'NOMA_Fairness': noma_fairness,
            })
            
        return pd.DataFrame(records)

    def run_alpha_sweep(self, num_points: int = 11, num_trials: int = 15) -> pd.DataFrame:
        """
        Experiment D: Sweep alpha from 0.0 (pure VLC) to 1.0 (pure RF)
        to evaluate the rate-power-EE tradeoff.
        """
        alphas = np.linspace(0.0, 1.0, num_points)
        records = []
        env = IndoorEnvironment(self.config, seed=self.seed)
        
        for alpha in alphas:
            rates_list, ee_list, powers_list = [], [], []
            for _ in range(num_trials):
                env.reset_users()
                dists_rf = env.get_rf_distances()
                rf_gains = self.rf_model.calculate_channel_gains(dists_rf)
                
                dists_vlc = env.get_vlc_distances()
                phi, psi = env.get_vlc_angles_matrix()
                blk = env.compute_optical_blockage_matrix()
                h_mat = self.vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
                vlc_eff, _ = self.vlc_model.get_effective_user_channel_gains(h_mat)
                is_blk = (vlc_eff < 1e-12)
                
                # Hybrid mode for all users
                modes = np.full(self.config.num_users, 2, dtype=int)
                res = self.evaluator.evaluate_state(rf_gains, vlc_eff, is_blk, modes, alpha=alpha, beta=0.3)
                rates_list.append(res['sum_rate'] / 1e6)
                ee_list.append(res['energy_efficiency'])
                powers_list.append(res['total_elec_power_w'])
                
            records.append({
                'alpha': float(alpha),
                'mean_sum_rate_mbps': float(np.mean(rates_list)),
                'std_sum_rate_mbps': float(np.std(rates_list)),
                'mean_ee_bits_per_j': float(np.mean(ee_list)),
                'mean_elec_power_w': float(np.mean(powers_list))
            })
            
        return pd.DataFrame(records)

    def run_beta_sweep(self, num_points: int = 11, num_trials: int = 15) -> pd.DataFrame:
        """
        Experiment E: Sweep RSMA beta from 0.0 to 0.7 to evaluate common stream impact.
        """
        betas = np.linspace(0.0, 0.7, num_points)
        records = []
        env = IndoorEnvironment(self.config, seed=self.seed)
        
        for beta in betas:
            rates_list, ee_list, fairness_list = [], [], []
            for _ in range(num_trials):
                env.reset_users()
                dists_rf = env.get_rf_distances()
                rf_gains = self.rf_model.calculate_channel_gains(dists_rf)
                
                dists_vlc = env.get_vlc_distances()
                phi, psi = env.get_vlc_angles_matrix()
                blk = env.compute_optical_blockage_matrix()
                h_mat = self.vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
                vlc_eff, _ = self.vlc_model.get_effective_user_channel_gains(h_mat)
                is_blk = (vlc_eff < 1e-12)
                
                modes = np.full(self.config.num_users, 2, dtype=int)
                res = self.evaluator.evaluate_state(rf_gains, vlc_eff, is_blk, modes, alpha=0.5, beta=beta)
                rates_list.append(res['sum_rate'] / 1e6)
                ee_list.append(res['energy_efficiency'])
                fairness_list.append(calculate_jains_fairness_index(res['user_rates']))
                
            records.append({
                'beta': float(beta),
                'mean_sum_rate_mbps': float(np.mean(rates_list)),
                'mean_ee_bits_per_j': float(np.mean(ee_list)),
                'mean_fairness': float(np.mean(fairness_list))
            })
            
        return pd.DataFrame(records)

    def run_blockage_experiment(self, max_obstacles: int = 4, num_trials: int = 15) -> pd.DataFrame:
        """
        Experiment B: Impact of increasing optical obstacle count on throughput and EE.
        """
        records = []
        env = IndoorEnvironment(self.config, seed=self.seed)
        
        for num_obs in range(max_obstacles + 1):
            env.clear_obstacles()
            for o in range(num_obs):
                ox = 1.0 + o * 0.8
                oy = 1.0 + o * 0.8
                env.add_obstacle(Obstacle3D.from_center_and_size((ox, oy, 0.85), (0.5, 0.5, 1.6), f"Blocker_{o}"))
                
            vlc_only_rates, hybrid_rates, opt_ee_list = [], [], []
            for _ in range(num_trials):
                env.reset_users()
                rf_gains = self.rf_model.calculate_channel_gains(env.get_rf_distances())
                dists_vlc = env.get_vlc_distances()
                phi, psi = env.get_vlc_angles_matrix()
                blk = env.compute_optical_blockage_matrix()
                h_mat = self.vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
                vlc_eff, _ = self.vlc_model.get_effective_user_channel_gains(h_mat)
                is_blk = (vlc_eff < 1e-12)
                
                # VLC-only
                vlc_res = self.baseline_mgr.run_vlc_only(rf_gains, vlc_eff, is_blk)
                vlc_only_rates.append(vlc_res['sum_rate'] / 1e6)
                
                # Hybrid RSMA
                hyb_res = self.baseline_mgr.run_fixed_hybrid(rf_gains, vlc_eff, is_blk)
                hybrid_rates.append(hyb_res['sum_rate'] / 1e6)
                
                # Optimizer
                opt_res = self.teacher.optimize(rf_gains, vlc_eff, is_blk)
                opt_ee_list.append(opt_res['energy_efficiency'])
                
            records.append({
                'num_obstacles': num_obs,
                'vlc_only_rate_mbps': float(np.mean(vlc_only_rates)),
                'hybrid_rate_mbps': float(np.mean(hybrid_rates)),
                'optimized_ee': float(np.mean(opt_ee_list))
            })
            
        return pd.DataFrame(records)

    def run_ablation_study(self, num_trials: int = 20) -> pd.DataFrame:
        """
        Experiment G: Ablation Study to quantify individual contributions of:
        1. Full Proposed (Adaptive Links + Adaptive Alpha + Adaptive Beta + RSMA)
        2. Ablation 1: Fixed Link Mode (All Hybrid) + Adaptive Alpha/Beta
        3. Ablation 2: Adaptive Links + Fixed Alpha (0.5) + Adaptive Beta
        4. Ablation 3: Adaptive Links + Adaptive Alpha + Fixed Beta (0.3)
        5. Ablation 4: Adaptive Links + Adaptive Alpha + SDMA (Beta = 0.0)
        """
        env = IndoorEnvironment(self.config, seed=self.seed)
        results = {
            'Full_Proposed': [],
            'Fixed_Link_Mode': [],
            'Fixed_Alpha': [],
            'Fixed_Beta': [],
            'SDMA_No_RSMA': []
        }
        
        for _ in range(num_trials):
            env.reset_users()
            rf_gains = self.rf_model.calculate_channel_gains(env.get_rf_distances())
            dists_vlc = env.get_vlc_distances()
            phi, psi = env.get_vlc_angles_matrix()
            blk = env.compute_optical_blockage_matrix()
            h_mat = self.vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
            vlc_eff, _ = self.vlc_model.get_effective_user_channel_gains(h_mat)
            is_blk = (vlc_eff < 1e-12)
            
            # Full Proposed (Optimizer teacher)
            full = self.teacher.optimize(rf_gains, vlc_eff, is_blk)
            results['Full_Proposed'].append(full['energy_efficiency'])
            
            # Fixed Link Mode (All Hybrid)
            hyb_res = self.evaluator.evaluate_state(
                rf_gains, vlc_eff, is_blk, modes=np.full(4, 2, dtype=int),
                alpha=full['alpha'], beta=full['beta']
            )
            results['Fixed_Link_Mode'].append(hyb_res['energy_efficiency'])
            
            # Fixed Alpha (0.5)
            alpha_fixed = self.evaluator.evaluate_state(
                rf_gains, vlc_eff, is_blk, modes=full['modes'],
                alpha=0.5, beta=full['beta']
            )
            results['Fixed_Alpha'].append(alpha_fixed['energy_efficiency'])
            
            # Fixed Beta (0.3)
            beta_fixed = self.evaluator.evaluate_state(
                rf_gains, vlc_eff, is_blk, modes=full['modes'],
                alpha=full['alpha'], beta=0.3
            )
            results['Fixed_Beta'].append(beta_fixed['energy_efficiency'])
            
            # SDMA (Beta = 0.0)
            sdma_res = self.evaluator.evaluate_state(
                rf_gains, vlc_eff, is_blk, modes=full['modes'],
                alpha=full['alpha'], beta=0.0
            )
            results['SDMA_No_RSMA'].append(sdma_res['energy_efficiency'])
            
        summary_rows = []
        for scheme, ees in results.items():
            summary_rows.append({
                'Ablation_Variant': scheme,
                'Mean_EE_bits_per_J': float(np.mean(ees)),
                'Std_EE': float(np.std(ees)),
                'Relative_to_Full_pct': float(100.0 * np.mean(ees) / max(1e-6, np.mean(results['Full_Proposed'])))
            })
            
        return pd.DataFrame(summary_rows)
