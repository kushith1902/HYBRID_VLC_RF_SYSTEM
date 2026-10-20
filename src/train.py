"""
Training Pipeline for Optimization-Guided Multi-Output Neural Network.
Implements multi-task loss, feature standardization, validation monitoring,
early stopping, and checkpointing.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.model import JointLinkPowerNN


FEATURE_COLS = [
    'u0_x', 'u0_y', 'u0_z', 'u0_rf_dist', 'u0_rf_gain_db', 'u0_rf_snr_db', 'u0_vlc_gain_db', 'u0_vlc_snr_db', 'u0_is_blocked',
    'u1_x', 'u1_y', 'u1_z', 'u1_rf_dist', 'u1_rf_gain_db', 'u1_rf_snr_db', 'u1_vlc_gain_db', 'u1_vlc_snr_db', 'u1_is_blocked',
    'u2_x', 'u2_y', 'u2_z', 'u2_rf_dist', 'u2_rf_gain_db', 'u2_rf_snr_db', 'u2_vlc_gain_db', 'u2_vlc_snr_db', 'u2_is_blocked',
    'u3_x', 'u3_y', 'u3_z', 'u3_rf_dist', 'u3_rf_gain_db', 'u3_rf_snr_db', 'u3_vlc_gain_db', 'u3_vlc_snr_db', 'u3_is_blocked'
]


def extract_features_and_targets(df: pd.DataFrame, num_users: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature matrices and supervised targets from scenario DataFrame.
    """
    # Feature matrix X: (N, in_features)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    
    # Target link modes: (N, num_users)
    mode_cols = [f'target_mode_u{k}' for k in range(num_users)]
    y_modes = df[mode_cols].to_numpy(dtype=np.int64)
    
    # Target alpha: (N, 1)
    y_alpha = df[['target_alpha']].to_numpy(dtype=np.float32)
    
    # Target beta: (N, 1)
    y_beta = df[['target_beta']].to_numpy(dtype=np.float32)
    
    return X, y_modes, y_alpha, y_beta


class ModelTrainer:
    """
    Supervised Training Manager for the Multi-Output Neural Network.
    """
    def __init__(self, in_features: int = 36, num_users: int = 4,
                 lambda_link: float = 1.0, lambda_alpha: float = 2.0, lambda_beta: float = 2.0):
        self.in_features = in_features
        self.num_users = num_users
        self.lambda_link = lambda_link
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        
        self.model = JointLinkPowerNN(in_features=in_features, num_users=num_users)
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
            checkpoint_dir: str = "models",
            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train the multi-output network using Adam optimizer and multi-task loss.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_joint_model.pt")
        
        # 1. Extract data
        X_train, y_modes_train, y_alpha_train, y_beta_train = extract_features_and_targets(train_df, self.num_users)
        X_val, y_modes_val, y_alpha_val, y_beta_val = extract_features_and_targets(val_df, self.num_users)
        
        # 2. Standardization
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-7 # Prevent division by zero
        
        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        X_val_norm = (X_val - self.scaler_mean) / self.scaler_std
        
        # 3. Create PyTorch Datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(y_modes_train, dtype=torch.long),
            torch.tensor(y_alpha_train, dtype=torch.float32),
            torch.tensor(y_beta_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(y_modes_val, dtype=torch.long),
            torch.tensor(y_alpha_val, dtype=torch.float32),
            torch.tensor(y_beta_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_link_acc': [],
            'val_link_acc': [],
            'val_alpha_mse': [],
            'val_beta_mse': []
        }
        
        best_val_loss = float('inf')
        patience = 12
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training Phase
            self.model.train()
            total_train_loss = 0.0
            correct_links_train = 0
            total_links_train = 0
            
            for b_x, b_modes, b_alpha, b_beta in train_loader:
                optimizer.zero_grad()
                pred_link_logits, pred_alpha, pred_beta = self.model(b_x)
                
                # Multi-class link loss: flatten batch and user dimensions
                loss_link = ce_loss_fn(pred_link_logits.view(-1, 3), b_modes.view(-1))
                loss_alpha = mse_loss_fn(pred_alpha, b_alpha)
                loss_beta = mse_loss_fn(pred_beta, b_beta)
                
                total_loss = (self.lambda_link * loss_link +
                              self.lambda_alpha * loss_alpha +
                              self.lambda_beta * loss_beta)
                
                total_loss.backward()
                optimizer.step()
                
                total_train_loss += total_loss.item() * len(b_x)
                
                # Accuracy calculation
                pred_classes = torch.argmax(pred_link_logits, dim=2)
                correct_links_train += (pred_classes == b_modes).sum().item()
                total_links_train += b_modes.numel()
                
            train_epoch_loss = total_train_loss / len(train_dataset)
            train_link_acc = correct_links_train / total_links_train
            
            # Validation Phase
            self.model.eval()
            total_val_loss = 0.0
            correct_links_val = 0
            total_links_val = 0
            val_alpha_mse_accum = 0.0
            val_beta_mse_accum = 0.0
            
            with torch.no_grad():
                for b_x, b_modes, b_alpha, b_beta in val_loader:
                    pred_link_logits, pred_alpha, pred_beta = self.model(b_x)
                    
                    loss_link = ce_loss_fn(pred_link_logits.view(-1, 3), b_modes.view(-1))
                    loss_alpha = mse_loss_fn(pred_alpha, b_alpha)
                    loss_beta = mse_loss_fn(pred_beta, b_beta)
                    
                    batch_val_loss = (self.lambda_link * loss_link +
                                      self.lambda_alpha * loss_alpha +
                                      self.lambda_beta * loss_beta)
                    
                    total_val_loss += batch_val_loss.item() * len(b_x)
                    
                    pred_classes = torch.argmax(pred_link_logits, dim=2)
                    correct_links_val += (pred_classes == b_modes).sum().item()
                    total_links_val += b_modes.numel()
                    
                    val_alpha_mse_accum += loss_alpha.item() * len(b_x)
                    val_beta_mse_accum += loss_beta.item() * len(b_x)
                    
            val_epoch_loss = total_val_loss / len(val_dataset)
            val_link_acc = correct_links_val / total_links_val
            val_alpha_mse = val_alpha_mse_accum / len(val_dataset)
            val_beta_mse = val_beta_mse_accum / len(val_dataset)
            
            history['train_loss'].append(float(train_epoch_loss))
            history['val_loss'].append(float(val_epoch_loss))
            history['train_link_acc'].append(float(train_link_acc))
            history['val_link_acc'].append(float(val_link_acc))
            history['val_alpha_mse'].append(float(val_alpha_mse))
            history['val_beta_mse'].append(float(val_beta_mse))
            
            # Checkpoint saving
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler_mean': self.scaler_mean,
                    'scaler_std': self.scaler_std,
                    'history': history,
                    'in_features': self.in_features,
                    'num_users': self.num_users,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
            if progress_callback is not None:
                progress_callback(epoch, epochs, train_epoch_loss, val_epoch_loss)
                
        return {
            'checkpoint_path': checkpoint_path,
            'best_val_loss': float(best_val_loss),
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
