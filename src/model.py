"""
PyTorch Multi-Output Neural Network for Joint Link Selection and Power Allocation.
Shared representation backbone with multi-task heads for:
1. User Link Mode Classification (RF, VLC, Hybrid)
2. RF/VLC Power Allocation Parameter (alpha in [0, 1])
3. RSMA Common Stream Power Allocation Parameter (beta in [0, 1])
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLinkPowerNN(nn.Module):
    """
    Multi-Output Neural Network for 6G Hybrid RF/VLC RSMA.
    """
    def __init__(self, in_features: int = 36, num_users: int = 4, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.in_features = in_features
        self.num_users = num_users
        
        # Shared Feature Representation Backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU()
        )
        
        # Head 1: Link Selection Classification (3 classes per user: RF=0, VLC=1, Hybrid=2)
        self.link_head = nn.Linear(hidden_dim2, num_users * 3)
        
        # Head 2: Alpha Regression Head (RF vs VLC power fraction in [0, 1])
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Bounded strictly in [0, 1]
        )
        
        # Head 3: Beta Regression Head (RSMA common power fraction in [0, 1])
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Bounded strictly in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
            x: Tensor of shape (batch_size, in_features)
            
        Returns:
            link_logits: Tensor of shape (batch_size, num_users, 3)
            alpha: Tensor of shape (batch_size, 1) in [0, 1]
            beta: Tensor of shape (batch_size, 1) in [0, 1]
        """
        # Handle 1D input by unsqueezing to (1, in_features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.shared_backbone(x)
        
        # Reshape link logits to (batch_size, num_users, 3)
        raw_links = self.link_head(features)
        link_logits = raw_links.view(-1, self.num_users, 3)
        
        alpha = self.alpha_head(features)
        beta = self.beta_head(features)
        
        return link_logits, alpha, beta

    def predict_allocation(self, x_np: np.ndarray, is_blocked: np.ndarray) -> Dict[str, Any]:
        """
        Perform inference and apply post-prediction feasibility projection.
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
                
            link_logits, alpha_tensor, beta_tensor = self.forward(x_tensor)
            
            # Predict modes via argmax over classes
            predicted_modes = torch.argmax(link_logits[0], dim=1).cpu().numpy()
            alpha_val = float(alpha_tensor[0, 0].cpu().numpy())
            beta_val = float(beta_tensor[0, 0].cpu().numpy())
            
            # Feasibility Projection:
            # If a user is predicted VLC-only (1) but is physically blocked, fall back to RF (0)
            projected_modes = predicted_modes.copy()
            for k in range(self.num_users):
                if is_blocked[k] and projected_modes[k] == 1:
                    projected_modes[k] = 0 # Fall back to RF
                    
        return {
            'predicted_modes': predicted_modes,
            'projected_modes': projected_modes,
            'predicted_alpha': alpha_val,
            'predicted_beta': beta_val
        }
