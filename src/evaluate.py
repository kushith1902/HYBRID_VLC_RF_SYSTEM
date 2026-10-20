"""
Evaluation Framework for Trained AI Controller on Unseen Test Scenarios.
Loads model checkpoint, performs inference with timing measurements,
applies physical feasibility projection, and evaluates against optimization teacher.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from config.default_config import SystemConfig
from src.model import JointLinkPowerNN
from src.objective import SystemEvaluator
from src.train import FEATURE_COLS, extract_features_and_targets


class ModelEvaluator:
    """
    Evaluates AI controller performance and compares against the optimizer teacher.
    """
    def __init__(self, checkpoint_path: str = "models/best_joint_model.pt",
                 config: Optional[SystemConfig] = None):
        self.checkpoint_path = checkpoint_path
        self.config = config or SystemConfig()
        self.evaluator = SystemEvaluator(self.config)
        
        self.model: Optional[JointLinkPowerNN] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.history: Optional[Dict[str, List[float]]] = None
        
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load trained PyTorch checkpoint if available."""
        if not os.path.exists(self.checkpoint_path):
            return
            
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        in_features = checkpoint.get('in_features', 36)
        num_users = checkpoint.get('num_users', 4)
        
        self.model = JointLinkPowerNN(in_features=in_features, num_users=num_users)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        self.history = checkpoint.get('history', None)

    @property
    def is_trained(self) -> bool:
        """Check if model has been successfully loaded."""
        return self.model is not None and self.scaler_mean is not None

    def evaluate_scenario(self, row: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a single unseen test scenario using the trained AI model.
        """
        if not self.is_trained:
            raise RuntimeError("Model checkpoint is not loaded. Train the model first.")
            
        K = self.config.num_users
        
        # 1. Extract raw features
        raw_x = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)
        norm_x = (raw_x - self.scaler_mean) / self.scaler_std
        
        # Blockage states
        is_blocked = np.array([bool(row[f'u{k}_is_blocked']) for k in range(K)], dtype=bool)
        
        # Channel gains
        rf_gains = np.array([10.0 ** (row[f'u{k}_rf_gain_db'] / 10.0) for k in range(K)], dtype=float)
        vlc_gains = np.array([10.0 ** (row[f'u{k}_vlc_gain_db'] / 10.0) for k in range(K)], dtype=float)
        
        # 2. Measure AI inference latency
        t0 = time.perf_counter()
        pred_res = self.model.predict_allocation(norm_x, is_blocked)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # 3. Physical system evaluation using AI decisions
        proj_modes = pred_res['projected_modes']
        pred_alpha = pred_res['predicted_alpha']
        pred_beta = pred_res['predicted_beta']
        
        ai_perf = self.evaluator.evaluate_state(
            rf_channel_gains=rf_gains,
            vlc_effective_h=vlc_gains,
            is_blocked=is_blocked,
            modes=proj_modes,
            alpha=pred_alpha,
            beta=pred_beta,
            min_rate_bps=self.config.min_user_rate_bps
        )
        
        # 4. Teacher targets
        teacher_modes = np.array([int(row[f'target_mode_u{k}']) for k in range(K)])
        teacher_alpha = float(row['target_alpha'])
        teacher_beta = float(row['target_beta'])
        teacher_ee = float(row['target_ee_bits_per_j'])
        teacher_sum_rate = float(row['target_sum_rate'])
        teacher_runtime_ms = float(row.get('optimizer_runtime_sec', 0.1)) * 1000.0
        
        # Agreement and error metrics
        mode_agreement = float(np.mean(proj_modes == teacher_modes))
        alpha_abs_error = abs(pred_alpha - teacher_alpha)
        beta_abs_error = abs(pred_beta - teacher_beta)
        ee_gap_pct = float(100.0 * (teacher_ee - ai_perf['energy_efficiency']) / max(1e-6, teacher_ee))
        
        return {
            'ai_modes': proj_modes,
            'raw_pred_modes': pred_res['predicted_modes'],
            'ai_alpha': pred_alpha,
            'ai_beta': pred_beta,
            'ai_sum_rate': ai_perf['sum_rate'],
            'ai_total_power_w': ai_perf['total_elec_power_w'],
            'ai_ee': ai_perf['energy_efficiency'],
            'ai_feasible': ai_perf['is_feasible'],
            'ai_outage_count': ai_perf['outage_users'],
            'ai_latency_ms': latency_ms,
            'teacher_modes': teacher_modes,
            'teacher_alpha': teacher_alpha,
            'teacher_beta': teacher_beta,
            'teacher_ee': teacher_ee,
            'teacher_sum_rate': teacher_sum_rate,
            'teacher_runtime_ms': teacher_runtime_ms,
            'mode_agreement': mode_agreement,
            'alpha_abs_error': alpha_abs_error,
            'beta_abs_error': beta_abs_error,
            'ee_gap_pct': ee_gap_pct
        }

    def evaluate_test_dataset(self, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate full test dataset and return detailed records and aggregated metrics.
        """
        records = []
        for _, row in test_df.iterrows():
            rec = self.evaluate_scenario(row)
            records.append(rec)
            
        results_df = pd.DataFrame(records)
        
        summary = {
            'total_scenarios': len(results_df),
            'mean_link_agreement': float(results_df['mode_agreement'].mean() * 100.0),
            'mean_alpha_mae': float(results_df['alpha_abs_error'].mean()),
            'mean_beta_mae': float(results_df['beta_abs_error'].mean()),
            'mean_ai_ee': float(results_df['ai_ee'].mean()),
            'mean_teacher_ee': float(results_df['teacher_ee'].mean()),
            'mean_ai_sum_rate_mbps': float(results_df['ai_sum_rate'].mean() / 1e6),
            'mean_teacher_sum_rate_mbps': float(results_df['teacher_sum_rate'].mean() / 1e6),
            'mean_ai_latency_ms': float(results_df['ai_latency_ms'].mean()),
            'p95_ai_latency_ms': float(results_df['ai_latency_ms'].quantile(0.95)),
            'mean_teacher_runtime_ms': float(results_df['teacher_runtime_ms'].mean()),
            'speedup_factor': float(results_df['teacher_runtime_ms'].mean() / max(1e-4, results_df['ai_latency_ms'].mean())),
            'ai_feasibility_rate': float(results_df['ai_feasible'].mean() * 100.0),
            'mean_ee_gap_pct': float(results_df['ee_gap_pct'].mean())
        }
        
        return results_df, summary
