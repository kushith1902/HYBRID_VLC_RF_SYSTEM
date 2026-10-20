"""
Physics-Based Dataset Generator for AI-Driven Joint Link Selection and Power Allocation.
Simulates realistic indoor propagation, obstacles, optical blockage, and queries
the Optimization Teacher to generate genuine supervised training labels.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.default_config import SystemConfig
from src.environment import IndoorEnvironment
from src.rf_channel import RFChannelModel
from src.vlc_channel import VLCChannelModel
from src.geometry import Obstacle3D
from src.optimizer import OptimizationTeacher


class DatasetGenerator:
    """
    Generates reproducible simulation datasets with physical propagation features
    and optimization-teacher target decisions.
    """
    def __init__(self, config: Optional[SystemConfig] = None, seed: int = 42):
        self.config = config or SystemConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.rf_model = RFChannelModel(self.config.rf, seed=seed)
        self.vlc_model = VLCChannelModel(self.config.vlc)
        self.teacher = OptimizationTeacher(self.config, alpha_grid_points=7, beta_grid_points=5)

    def generate_scenario(self, scenario_id: int, env: IndoorEnvironment) -> Dict[str, Any]:
        """
        Generate a single physics-based scenario and compute optimizer target decisions.
        """
        K = env.num_users
        
        # 1. Environment geometry and blockage
        rf_distances = env.get_rf_distances()
        vlc_distances = env.get_vlc_distances()
        phi_mat, psi_mat = env.get_vlc_angles_matrix()
        blockage_matrix = env.compute_optical_blockage_matrix()
        
        # 2. RF channel computation
        rf_gains = self.rf_model.calculate_channel_gains(rf_distances, enable_fading=True)
        rf_snr_lin = self.rf_model.calculate_snr_linear(self.config.rf.tx_power_max_w, rf_gains)
        rf_snr_db = 10.0 * np.log10(np.maximum(rf_snr_lin, 1e-12))
        
        # 3. VLC channel computation
        h_matrix = self.vlc_model.calculate_dc_gain_matrix(
            distances=vlc_distances,
            phi_rad=phi_mat,
            psi_rad=psi_mat,
            blockage_matrix=blockage_matrix
        )
        vlc_effective_h, serving_leds = self.vlc_model.get_effective_user_channel_gains(h_matrix)
        vlc_snr_lin = self.vlc_model.calculate_electrical_snr_linear(
            tx_optical_power_w=self.config.vlc.optical_power_per_led_w,
            effective_h=vlc_effective_h
        )
        vlc_snr_db = 10.0 * np.log10(np.maximum(vlc_snr_lin, 1e-12))
        
        # Blockage indicator per user (True if all LEDs blocked or effective gain is zero)
        is_blocked = (vlc_effective_h < 1e-12)
        
        # 4. Optimization Teacher target decision
        opt_res = self.teacher.optimize(
            rf_channel_gains=rf_gains,
            vlc_effective_h=vlc_effective_h,
            is_blocked=is_blocked,
            rf_snr_db=rf_snr_db,
            min_rate_bps=self.config.min_user_rate_bps
        )
        
        # Build tabular feature dictionary
        record: Dict[str, Any] = {
            'scenario_id': scenario_id,
            'num_users': K,
            'num_obstacles': len(env.obstacles),
            'min_rate_bps': self.config.min_user_rate_bps,
        }
        
        # Per-user spatial & channel features
        for k in range(K):
            record[f'u{k}_x'] = float(env.user_positions[k, 0])
            record[f'u{k}_y'] = float(env.user_positions[k, 1])
            record[f'u{k}_z'] = float(env.user_positions[k, 2])
            record[f'u{k}_rf_dist'] = float(rf_distances[k])
            record[f'u{k}_rf_gain_db'] = float(10.0 * np.log10(np.maximum(rf_gains[k], 1e-25)))
            record[f'u{k}_rf_snr_db'] = float(rf_snr_db[k])
            record[f'u{k}_vlc_gain_db'] = float(10.0 * np.log10(np.maximum(vlc_effective_h[k], 1e-25)))
            record[f'u{k}_vlc_snr_db'] = float(vlc_snr_db[k])
            record[f'u{k}_is_blocked'] = int(is_blocked[k])
            record[f'u{k}_serving_led'] = int(serving_leds[k])
            
        # Target labels from Optimization Teacher
        for k in range(K):
            record[f'target_mode_u{k}'] = int(opt_res['modes'][k])
            record[f'target_rate_u{k}'] = float(opt_res['user_rates'][k])
            
        record['target_alpha'] = float(opt_res['alpha'])
        record['target_beta'] = float(opt_res['beta'])
        record['target_sum_rate'] = float(opt_res['sum_rate'])
        record['target_elec_power_w'] = float(opt_res['total_elec_power_w'])
        record['target_ee_bits_per_j'] = float(opt_res['energy_efficiency'])
        record['is_feasible'] = int(opt_res['is_feasible'])
        record['solution_type'] = str(opt_res['solution_type'])
        record['optimizer_runtime_sec'] = float(opt_res['optimization_runtime_sec'])
        
        return record

    def generate_dataset(self, num_scenarios: int = 200,
                          randomize_obstacles: bool = True,
                          progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Generate multiple reproducible simulation scenarios.
        """
        records = []
        env = IndoorEnvironment(self.config, seed=self.seed)
        
        for i in range(num_scenarios):
            # Randomize user positions
            env.reset_users()
            
            # Optionally randomize obstacle position
            if randomize_obstacles:
                env.clear_obstacles()
                num_obs = self.rng.integers(1, 4) # 1 to 3 obstacles
                for o_idx in range(num_obs):
                    ox = self.rng.uniform(1.0, 4.0)
                    oy = self.rng.uniform(1.0, 4.0)
                    oz = 0.85
                    sx = self.rng.uniform(0.3, 0.6)
                    sy = self.rng.uniform(0.3, 0.6)
                    sz = self.rng.uniform(1.2, 1.8) # human/obstacle height
                    env.add_obstacle(Obstacle3D.from_center_and_size((ox, oy, oz), (sx, sy, sz), f"Obs_{o_idx}"))
                    
            record = self.generate_scenario(i, env)
            records.append(record)
            
            if progress_callback is not None:
                progress_callback(i + 1, num_scenarios)
                
        df = pd.DataFrame(records)
        return df
