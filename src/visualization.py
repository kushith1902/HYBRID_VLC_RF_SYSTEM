"""
Plotly and Matplotlib Scientific Visualizations for 6G Hybrid RF/VLC Networks.
Generates:
1. Interactive 3D indoor room layout with AP, LEDs, obstacles, users, and optical blockage rays.
2. 2D scientific charts for path loss, VLC spatial gain, throughput, EE, RSMA comparison, and loss curves.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config.default_config import SystemConfig
from src.environment import IndoorEnvironment
from src.geometry import Obstacle3D, check_segment_aabb_intersection


def create_3d_indoor_room_plot(env: IndoorEnvironment,
                               assigned_modes: Optional[np.ndarray] = None,
                               show_blocked_rays: bool = True) -> go.Figure:
    """
    Generate an interactive 3D room visualization with AP, LEDs, obstacles,
    users, and ray-traced communication links.
    """
    fig = go.Figure()
    
    # 1. Room Boundary Wireframe (5x5x3 m)
    lx, ly, lz = env.room.length, env.room.width, env.room.height
    box_x = [0, lx, lx, 0, 0, 0, lx, lx, 0, 0, 0, 0, lx, lx, lx, lx]
    box_y = [0, 0, ly, ly, 0, 0, 0, ly, ly, 0, 0, ly, ly, 0, 0, ly]
    box_z = [0, 0, 0, 0, 0, lz, lz, lz, lz, lz, 0, 0, 0, 0, lz, lz]
    
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        line=dict(color='#334155', width=3),
        name='Room Boundary',
        hoverinfo='none'
    ))
    
    # 2. RF Access Point (Sub-THz 140 GHz)
    ap = env.rf_ap_pos
    fig.add_trace(go.Scatter3d(
        x=[ap[0]], y=[ap[1]], z=[ap[2]],
        mode='markers+text',
        marker=dict(size=10, color='#00f2fe', symbol='diamond', line=dict(color='#ffffff', width=2)),
        text=['140 GHz AP'],
        textposition='bottom center',
        textfont=dict(color='#00f2fe', size=11),
        name='Sub-THz RF AP'
    ))
    
    # 3. VLC LED Transmitters (Ceiling)
    leds = env.vlc_led_positions
    fig.add_trace(go.Scatter3d(
        x=leds[:, 0], y=leds[:, 1], z=leds[:, 2],
        mode='markers+text',
        marker=dict(size=9, color='#fbbf24', symbol='circle', line=dict(color='#ffffff', width=2)),
        text=[f'LED {i}' for i in range(len(leds))],
        textposition='top center',
        textfont=dict(color='#fbbf24', size=11),
        name='VLC LEDs'
    ))
    
    # 4. 3D Obstacles (e.g. human blockers)
    for idx, obs in enumerate(env.obstacles):
        # Draw cuboid mesh / edges
        ox = [obs.x_min, obs.x_max, obs.x_max, obs.x_min, obs.x_min, obs.x_min, obs.x_max, obs.x_max, obs.x_min, obs.x_min]
        oy = [obs.y_min, obs.y_min, obs.y_max, obs.y_max, obs.y_min, obs.y_min, obs.y_min, obs.y_max, obs.y_max, obs.y_min]
        oz = [obs.z_min, obs.z_min, obs.z_min, obs.z_min, obs.z_min, obs.z_max, obs.z_max, obs.z_max, obs.z_max, obs.z_max]
        
        fig.add_trace(go.Scatter3d(
            x=ox, y=oy, z=oz,
            mode='lines',
            line=dict(color='#ef4444', width=4),
            name=f'Obstacle: {obs.name}'
        ))
        
    # 5. Mobile Users
    users = env.user_positions
    K = len(users)
    
    # Color mapping for modes: RF=cyan (#38bdf8), VLC=gold (#fbbf24), Hybrid=purple (#a855f7)
    mode_colors = {0: '#38bdf8', 1: '#fbbf24', 2: '#a855f7'}
    mode_names = {0: 'RF', 1: 'VLC', 2: 'Hybrid'}
    
    user_colors = []
    user_texts = []
    for k in range(K):
        m = assigned_modes[k] if assigned_modes is not None else 0
        c = mode_colors.get(m, '#38bdf8')
        user_colors.append(c)
        user_texts.append(f'User {k} ({mode_names.get(m, "RF")})')
        
    fig.add_trace(go.Scatter3d(
        x=users[:, 0], y=users[:, 1], z=users[:, 2],
        mode='markers+text',
        marker=dict(size=8, color=user_colors, line=dict(color='#ffffff', width=1.5)),
        text=user_texts,
        textposition='top right',
        textfont=dict(color='#ffffff', size=11),
        name='Mobile Users'
    ))
    
    # 6. Communication Rays & Blockage
    for k in range(K):
        u_pos = users[k]
        
        # RF Link (AP -> User)
        fig.add_trace(go.Scatter3d(
            x=[ap[0], u_pos[0]], y=[ap[1], u_pos[1]], z=[ap[2], u_pos[2]],
            mode='lines',
            line=dict(color='rgba(0, 242, 254, 0.4)', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # VLC Links (Each LED -> User)
        for m_idx, led_pos in enumerate(leds):
            is_blocked = False
            for obs in env.obstacles:
                if check_segment_aabb_intersection(led_pos, u_pos, obs):
                    is_blocked = True
                    break
                    
            if is_blocked and show_blocked_rays:
                # Blocked ray in red dashed
                fig.add_trace(go.Scatter3d(
                    x=[led_pos[0], u_pos[0]], y=[led_pos[1], u_pos[1]], z=[led_pos[2], u_pos[2]],
                    mode='lines',
                    line=dict(color='rgba(239, 68, 68, 0.6)', width=2, dash='dash'),
                    showlegend=False,
                    name='Blocked Optical Ray'
                ))
            elif not is_blocked:
                # Clear ray in subtle gold
                fig.add_trace(go.Scatter3d(
                    x=[led_pos[0], u_pos[0]], y=[led_pos[1], u_pos[1]], z=[led_pos[2], u_pos[2]],
                    mode='lines',
                    line=dict(color='rgba(251, 191, 36, 0.3)', width=1.5),
                    showlegend=False,
                    hoverinfo='none'
                ))
                
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length (m)', range=[0, lx], backgroundcolor='#0a0f1d', gridcolor='#1e293b'),
            yaxis=dict(title='Width (m)', range=[0, ly], backgroundcolor='#0a0f1d', gridcolor='#1e293b'),
            zaxis=dict(title='Height (m)', range=[0, lz], backgroundcolor='#0a0f1d', gridcolor='#1e293b'),
            aspectmode='data'
        ),
        paper_bgcolor='#0b1120',
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(font=dict(color='#e2e8f0'), bgcolor='rgba(15, 23, 42, 0.8)')
    )
    
    return fig


def plot_rf_channel_curves(rf_distances: np.ndarray, rf_path_losses: np.ndarray,
                           rf_gains_db: np.ndarray) -> go.Figure:
    """Plot RF Path Loss and Channel Gain vs Distance."""
    sorted_idx = np.argsort(rf_distances)
    d = rf_distances[sorted_idx]
    pl = rf_path_losses[sorted_idx]
    gain = rf_gains_db[sorted_idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d, y=pl, mode='lines+markers', name='Path Loss (dB)',
        line=dict(color='#ef4444', width=2.5), marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=d, y=gain, mode='lines+markers', name='Channel Power Gain (dB)',
        line=dict(color='#00f2fe', width=2.5), marker=dict(size=6), yaxis='y2'
    ))
    
    fig.update_layout(
        title='140 GHz Sub-THz Path Loss & Gain vs Distance',
        xaxis=dict(title='Distance (m)', gridcolor='#1e293b'),
        yaxis=dict(title='Path Loss (dB)', gridcolor='#1e293b', titlefont=dict(color='#ef4444')),
        yaxis2=dict(title='Channel Gain (dB)', overlaying='y', side='right', titlefont=dict(color='#00f2fe')),
        paper_bgcolor='#0b1120', plot_bgcolor='#0b1120',
        font=dict(color='#e2e8f0')
    )
    return fig


def plot_training_history(history: Dict[str, List[float]]) -> go.Figure:
    """Plot AI Training and Validation Loss Curves."""
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=history['train_loss'], mode='lines', name='Train Loss',
        line=dict(color='#38bdf8', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'], mode='lines', name='Validation Loss',
        line=dict(color='#f43f5e', width=2.5, dash='dash')
    ))
    
    fig.update_layout(
        title='Neural Network Multi-Task Training and Validation Loss',
        xaxis=dict(title='Epoch', gridcolor='#1e293b'),
        yaxis=dict(title='Combined Multi-Task Loss', gridcolor='#1e293b'),
        paper_bgcolor='#0b1120', plot_bgcolor='#0b1120',
        font=dict(color='#e2e8f0'),
        legend=dict(bgcolor='rgba(15, 23, 42, 0.8)')
    )
    return fig
