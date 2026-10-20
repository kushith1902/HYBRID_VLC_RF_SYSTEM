"""
6G Hybrid RF/VLC RSMA Simulation and AI Research Dashboard.
Interactive Streamlit UI with 3D Room Visualization, Live Optimization,
PyTorch AI Model Training, Baseline Comparisons, and Scientific Report Export.
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Internal modules
from config.default_config import SystemConfig, RoomConfig, RFConfig, VLCConfig
from src.environment import IndoorEnvironment
from src.rf_channel import RFChannelModel
from src.vlc_channel import VLCChannelModel
from src.geometry import Obstacle3D
from src.rsma import compute_rsma_rf_rates, compute_rsma_vlc_rates, compute_sdma_rates, compute_noma_rates
from src.power_allocation import PowerAllocationModel
from src.link_selection import rule_based_link_selection, LINK_MODES
from src.objective import SystemEvaluator
from src.optimizer import OptimizationTeacher
from src.baselines import BaselineManager
from src.dataset_generator import DatasetGenerator
from src.model import JointLinkPowerNN
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.metrics import calculate_jains_fairness_index, calculate_outage_probability, calculate_coverage_probability
from src.experiments import ExperimentRunner
from src.visualization import create_3d_indoor_room_plot, plot_rf_channel_curves, plot_training_history
from src.export_report import generate_research_report


# Page Setup
st.set_page_config(
    page_title="6G Hybrid RF/VLC RSMA Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Navy & Cyan accents)
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #00f2fe;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-subtitle {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #00f2fe 100%);
        color: #0f172a;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if 'config' not in st.session_state:
    st.session_state.config = SystemConfig()
if 'env' not in st.session_state:
    st.session_state.env = IndoorEnvironment(st.session_state.config, seed=42)
if 'last_sim_res' not in st.session_state:
    st.session_state.last_sim_res = None
if 'model_evaluator' not in st.session_state:
    st.session_state.model_evaluator = ModelEvaluator(checkpoint_path="models/best_joint_model.pt", config=st.session_state.config)
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'val_df' not in st.session_state:
    st.session_state.val_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/000000/satellite-sending-signal.png", width=64)
st.sidebar.title("6G Hybrid RF/VLC")
st.sidebar.markdown("**AI-Driven RSMA Resource Controller**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "1. Project Overview",
        "2. Network Configuration",
        "3. Indoor 3D Visualization",
        "4. RF Channel Analysis",
        "5. VLC Channel Analysis",
        "6. RSMA Analysis",
        "7. Optimization Teacher",
        "8. AI Training",
        "9. AI Allocation",
        "10. Baseline Comparison",
        "11. Experiments",
        "12. Results & Report Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Users:** {st.session_state.env.num_users} | **Obstacles:** {len(st.session_state.env.obstacles)}")
model_status = "Trained & Ready" if st.session_state.model_evaluator.is_trained else "Not Trained"
status_color = "#10b981" if st.session_state.model_evaluator.is_trained else "#f59e0b"
st.sidebar.markdown(f"**AI Model Status:** <span style='color:{status_color}; font-weight:bold;'>{model_status}</span>", unsafe_allow_html=True)


# ---------------- PAGE 1: OVERVIEW ----------------
if menu == "1. Project Overview":
    st.title("AI-Driven Joint Link Selection and Adaptive Power Allocation")
    st.markdown("### For Energy-Efficient RSMA in 6G Hybrid Sub-THz RF / Optical VLC Indoor Networks")
    
    st.markdown("""
    This platform provides an end-to-end physics-based simulation of a 6G hybrid indoor communication cell combining:
    - **140 GHz Sub-THz RF Access Point** with directional beamforming.
    - **4 Ceiling-Mounted VLC LED Transmitters** operating via Intensity-Modulation / Direct-Detection (IM/DD).
    - **Rate-Splitting Multiple Access (RSMA)** with Successive Interference Cancellation (SIC).
    - **Optimization Teacher & Multi-Output Deep Neural Network** predicting joint link assignment and $(\\alpha, \\beta)$ power allocations.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-title">Carrier Frequency</div>
        <div class="metric-value">140 GHz</div><div class="metric-subtitle">Sub-THz 6G Bandwidth: 1 GHz</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-title">Optical VLC LEDs</div>
        <div class="metric-value">4 Transmitters</div><div class="metric-subtitle">Lambertian Order m = 1.0</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-title">Multiple Access</div>
        <div class="metric-value">1-Layer RSMA</div><div class="metric-subtitle">Common + Private SIC</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-title">Objective Metric</div>
        <div class="metric-value">Energy Efficiency</div><div class="metric-subtitle">Unit: bits / Joule</div></div>""", unsafe_allow_html=True)

    st.subheader("System Architecture Pipeline")
    st.markdown("""
    ```
    Indoor 3D Room Geometry (5m x 5m x 3m) & User Coordinates
                             ↓
    140 GHz RF Channel Model  &  VLC IM/DD Channel & 3D Ray-Blockage Model
                             ↓
    RSMA Transmission Model (Common Stream Degradation & SIC Private Streams)
                             ↓
    Optimization Teacher (Global EE Maximization subject to QoS Min-Rate)
                             ↓
    PyTorch Multi-Output Neural Network (Real-time Sub-Millisecond AI Allocation)
                             ↓
    Baseline Benchmarks (RF-only, VLC-only, Fixed Hybrid, SDMA, NOMA)
    ```
    """)


# ---------------- PAGE 2: CONFIGURATION ----------------
elif menu == "2. Network Configuration":
    st.title("Network Parameters & Geometry Configuration")
    
    with st.form("config_form"):
        st.subheader("1. Indoor Room Dimensions")
        c1, c2, c3, c4 = st.columns(4)
        length = c1.number_input("Length (m)", min_value=3.0, max_value=15.0, value=5.0, step=0.5)
        width = c2.number_input("Width (m)", min_value=3.0, max_value=15.0, value=5.0, step=0.5)
        height = c3.number_input("Height (m)", min_value=2.5, max_value=6.0, value=3.0, step=0.5)
        num_users = c4.number_input("Number of Users (K)", min_value=2, max_value=8, value=4, step=1)
        
        st.subheader("2. 140 GHz Sub-THz RF Parameters")
        r1, r2, r3, r4 = st.columns(4)
        rf_freq_ghz = r1.number_input("RF Carrier (GHz)", min_value=100.0, max_value=300.0, value=140.0)
        rf_bw_mhz = r2.number_input("RF Bandwidth (MHz)", min_value=100.0, max_value=2000.0, value=1000.0)
        rf_p_max_dbm = r3.number_input("Max RF Power (dBm)", min_value=10.0, max_value=30.0, value=23.0)
        rf_tx_gain = r4.number_input("Tx Antenna Gain (dBi)", min_value=5.0, max_value=25.0, value=15.0)
        
        st.subheader("3. Optical VLC Parameters")
        v1, v2, v3, v4 = st.columns(4)
        vlc_p_opt = v1.number_input("Optical Power / LED (W)", min_value=1.0, max_value=20.0, value=5.0)
        vlc_semi_angle = v2.number_input("Semi-Angle (deg)", min_value=30.0, max_value=85.0, value=60.0)
        vlc_fov = v3.number_input("Receiver FOV (deg)", min_value=30.0, max_value=85.0, value=60.0)
        vlc_bw_mhz = v4.number_input("VLC Bandwidth (MHz)", min_value=5.0, max_value=50.0, value=20.0)
        
        st.subheader("4. Simulation Control")
        s1, s2 = st.columns(2)
        min_rate_mbps = s1.number_input("QoS Minimum Rate Constraint (Mbps)", min_value=1.0, max_value=50.0, value=10.0)
        sim_seed = s2.number_input("Random Seed", min_value=1, max_value=99999, value=42)
        
        submitted = st.form_submit_button("Apply Configuration & Re-initialize")
        if submitted:
            st.session_state.config.room.length = length
            st.session_state.config.room.width = width
            st.session_state.config.room.height = height
            st.session_state.config.num_users = int(num_users)
            st.session_state.config.rf.carrier_frequency = rf_freq_ghz * 1e9
            st.session_state.config.rf.bandwidth = rf_bw_mhz * 1e6
            st.session_state.config.rf.tx_power_max_dbm = rf_p_max_dbm
            st.session_state.config.rf.tx_antenna_gain_dbi = rf_tx_gain
            st.session_state.config.vlc.optical_power_per_led_w = vlc_p_opt
            st.session_state.config.vlc.semi_angle_half_power_deg = vlc_semi_angle
            st.session_state.config.vlc.pd_fov_deg = vlc_fov
            st.session_state.config.vlc.bandwidth = vlc_bw_mhz * 1e6
            st.session_state.config.min_user_rate_bps = min_rate_mbps * 1e6
            st.session_state.config.random_seed = int(sim_seed)
            
            st.session_state.env = IndoorEnvironment(st.session_state.config, seed=int(sim_seed))
            st.success("Configuration successfully updated and environment re-initialized!")


# ---------------- PAGE 3: 3D VISUALIZATION ----------------
elif menu == "3. Indoor 3D Visualization":
    st.title("Interactive 3D Indoor Network Topology")
    st.markdown("Visualizes the 140 GHz AP (cyan diamond), 4 VLC LEDs (yellow circles), 3D obstacles (red outlines), mobile users, and ray-traced optical blockage paths.")
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        if st.button("Step User Mobility (dt = 0.5s)"):
            st.session_state.env.step_mobility(dt=0.5)
            st.rerun()
    with col_ctrl2:
        if st.button("Randomize User Locations"):
            st.session_state.env.reset_users()
            st.rerun()
    with col_ctrl3:
        if st.button("Toggle Add/Remove Obstacle"):
            if len(st.session_state.env.obstacles) > 0:
                st.session_state.env.clear_obstacles()
            else:
                st.session_state.env.init_default_obstacles()
            st.rerun()
            
    fig_3d = create_3d_indoor_room_plot(st.session_state.env)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Table of User Positions and Blockage
    st.subheader("Current User Spatial States")
    blk_mat = st.session_state.env.compute_optical_blockage_matrix()
    rf_dists = st.session_state.env.get_rf_distances()
    
    user_rows = []
    for k in range(st.session_state.env.num_users):
        u_pos = st.session_state.env.user_positions[k]
        blocked_leds = [f"LED {m}" for m in range(st.session_state.env.num_leds) if blk_mat[m, k]]
        user_rows.append({
            'User': f"User {k}",
            'X (m)': f"{u_pos[0]:.2f}",
            'Y (m)': f"{u_pos[1]:.2f}",
            'Z (m)': f"{u_pos[2]:.2f}",
            'Distance to RF AP (m)': f"{rf_dists[k]:.2f}",
            'Blocked Optical Links': ", ".join(blocked_leds) if blocked_leds else "None (Clear LOS)"
        })
    st.dataframe(pd.DataFrame(user_rows), use_container_width=True)


# ---------------- PAGE 4: RF CHANNEL ANALYSIS ----------------
elif menu == "4. RF Channel Analysis":
    st.title("140 GHz Sub-THz RF Propagation Analysis")
    
    rf_model = RFChannelModel(st.session_state.config.rf, seed=st.session_state.config.random_seed)
    dists = st.session_state.env.get_rf_distances()
    pl_db = rf_model.calculate_path_loss_db(dists, enable_shadowing=False)
    gains = rf_model.calculate_channel_gains(dists, enable_fading=True)
    gains_db = 10.0 * np.log10(gains)
    snr_db = rf_model.calculate_snr_db(st.session_state.config.rf.tx_power_max_w, gains)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Carrier Frequency", f"{st.session_state.config.rf.carrier_frequency / 1e9:.1f} GHz")
    with c2:
        st.metric("1m Reference FSPL", f"{rf_model.fspl_1m_db:.2f} dB")
    with c3:
        st.metric("Thermal Noise Power", f"{rf_model.noise_power_dbm:.2f} dBm")
    with c4:
        st.metric("Mean Received SNR", f"{np.mean(snr_db):.2f} dB")
        
    st.plotly_chart(plot_rf_channel_curves(dists, pl_db, gains_db), use_container_width=True)
    
    # User RF table
    rf_table = []
    for k in range(st.session_state.env.num_users):
        rf_table.append({
            'User': f"User {k}",
            'Distance (m)': f"{dists[k]:.2f}",
            'CI Path Loss (dB)': f"{pl_db[k]:.2f}",
            'Channel Power Gain (dB)': f"{gains_db[k]:.2f}",
            'Received SNR (dB)': f"{snr_db[k]:.2f}"
        })
    st.dataframe(pd.DataFrame(rf_table), use_container_width=True)


# ---------------- PAGE 5: VLC CHANNEL ANALYSIS ----------------
elif menu == "5. VLC Channel Analysis":
    st.title("VLC IM/DD Optical Channel Analysis")
    
    vlc_model = VLCChannelModel(st.session_state.config.vlc)
    vlc_dists = st.session_state.env.get_vlc_distances()
    phi_mat, psi_mat = st.session_state.env.get_vlc_angles_matrix()
    blk_mat = st.session_state.env.compute_optical_blockage_matrix()
    
    h_matrix = vlc_model.calculate_dc_gain_matrix(vlc_dists, phi_mat, psi_mat, blk_mat)
    eff_h, serving_leds = vlc_model.get_effective_user_channel_gains(h_matrix)
    snr_elec_db = vlc_model.calculate_electrical_snr_db(st.session_state.config.vlc.optical_power_per_led_w, eff_h)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lambertian Order m", f"{vlc_model.m:.2f}")
    with c2:
        st.metric("Concentrator Gain", f"{vlc_model.g_conc:.2f}")
    with c3:
        st.metric("PD Responsivity", f"{vlc_model.r_pd} A/W")
    with c4:
        st.metric("Mean Optical Gain", f"{np.mean(eff_h):.2e}")
        
    # Heatmap of H_matrix (LEDs x Users)
    fig_heat = px.imshow(
        h_matrix,
        labels=dict(x="User Index", y="LED Index", color="Optical DC Gain"),
        x=[f"User {k}" for k in range(st.session_state.env.num_users)],
        y=[f"LED {m}" for m in range(st.session_state.env.num_leds)],
        color_continuous_scale="Viridis",
        title="Optical Channel Gain Matrix (H_VLC)"
    )
    fig_heat.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------- PAGE 6: RSMA ANALYSIS ----------------
elif menu == "6. RSMA Analysis":
    st.title("Rate-Splitting Multiple Access (RSMA) Analysis")
    st.markdown("Evaluates 1-layer downlink RSMA common and private stream powers, SIC decoding, and compares against SDMA and NOMA.")
    
    beta = st.slider("RSMA Common Stream Power Fraction (beta)", min_value=0.0, max_value=0.8, value=0.3, step=0.05)
    
    rf_model = RFChannelModel(st.session_state.config.rf)
    dists = st.session_state.env.get_rf_distances()
    gains = rf_model.calculate_channel_gains(dists)
    
    noise_w = 10.0 ** ((st.session_state.config.rf.noise_figure_db - 174.0 + 10.0 * np.log10(st.session_state.config.rf.bandwidth) - 30.0) / 10.0)
    p_tx = st.session_state.config.rf.tx_power_max_w
    
    # 1. RSMA
    rsma_res = compute_rsma_rf_rates(gains, noise_w, p_tx, beta=beta, bandwidth_hz=st.session_state.config.rf.bandwidth)
    
    # 2. SDMA
    sdma_rates = compute_sdma_rates(gains, noise_w, p_tx, bandwidth_hz=st.session_state.config.rf.bandwidth)
    
    # 3. NOMA
    noma_rates = compute_noma_rates(gains, noise_w, p_tx, bandwidth_hz=st.session_state.config.rf.bandwidth)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("RSMA Sum Rate", f"{rsma_res['sum_rate'] / 1e6:.2f} Mbps")
    with c2:
        st.metric("SDMA Sum Rate", f"{np.sum(sdma_rates) / 1e6:.2f} Mbps")
    with c3:
        st.metric("NOMA Sum Rate", f"{np.sum(noma_rates) / 1e6:.2f} Mbps")
        
    # Rate comparison bar chart
    fig_comp = go.Figure()
    users_x = [f"User {k}" for k in range(len(gains))]
    fig_comp.add_trace(go.Bar(x=users_x, y=rsma_res['total_user_rates'] / 1e6, name="RSMA (Proposed)", marker_color='#00f2fe'))
    fig_comp.add_trace(go.Bar(x=users_x, y=sdma_rates / 1e6, name="SDMA (beta=0)", marker_color='#64748b'))
    fig_comp.add_trace(go.Bar(x=users_x, y=noma_rates / 1e6, name="NOMA", marker_color='#f59e0b'))
    fig_comp.update_layout(
        title="Achievable Rate Comparison Across Multiple Access Schemes",
        yaxis_title="Achievable Rate (Mbps)",
        barmode='group',
        paper_bgcolor='#0b1120', plot_bgcolor='#0b1120',
        font=dict(color='#e2e8f0'),
        legend=dict(bgcolor='rgba(15, 23, 42, 0.8)')
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ---------------- PAGE 7: OPTIMIZATION TEACHER ----------------
elif menu == "7. Optimization Teacher":
    st.title("Optimization Teacher for Energy-Efficient Allocation")
    st.markdown("Performs structured search over supported link assignments and $(\\alpha, \\beta)$ power allocations to maximize Energy Efficiency (EE) [bits/Joule] subject to minimum rate QoS constraints.")
    
    if st.button("Run Optimization Teacher on Current State"):
        with st.spinner("Executing optimization teacher..."):
            rf_model = RFChannelModel(st.session_state.config.rf)
            vlc_model = VLCChannelModel(st.session_state.config.vlc)
            
            dists_rf = st.session_state.env.get_rf_distances()
            rf_gains = rf_model.calculate_channel_gains(dists_rf)
            
            dists_vlc = st.session_state.env.get_vlc_distances()
            phi, psi = st.session_state.env.get_vlc_angles_matrix()
            blk = st.session_state.env.compute_optical_blockage_matrix()
            h_mat = vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
            vlc_eff, _ = vlc_model.get_effective_user_channel_gains(h_mat)
            is_blk = (vlc_eff < 1e-12)
            
            teacher = OptimizationTeacher(st.session_state.config, alpha_grid_points=9, beta_grid_points=7)
            opt_res = teacher.optimize(rf_gains, vlc_eff, is_blk, min_rate_bps=st.session_state.config.min_user_rate_bps)
            st.session_state.last_sim_res = opt_res
            st.success("Optimization completed successfully!")
            
    if st.session_state.last_sim_res is not None:
        res = st.session_state.last_sim_res
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Energy Efficiency", f"{res['energy_efficiency']:.2f} bits/J")
        with c2:
            st.metric("Sum Throughput", f"{res['sum_rate'] / 1e6:.2f} Mbps")
        with c3:
            st.metric("Optimal Alpha", f"{res['alpha']:.3f}")
        with c4:
            st.metric("Optimal Beta", f"{res['beta']:.3f}")
            
        c5, c6 = st.columns(2)
        with c5:
            st.metric("Feasibility Status", "FEASIBLE" if res['is_feasible'] else "INFEASIBLE")
        with c6:
            st.metric("Runtime", f"{res['optimization_runtime_sec'] * 1000.0:.2f} ms")
            
        # Modes breakdown
        modes_summary = []
        for k in range(len(res['modes'])):
            modes_summary.append({
                'User': f"User {k}",
                'Selected Link Mode': LINK_MODES[res['modes'][k]],
                'Delivered Rate (Mbps)': f"{res['user_rates'][k] / 1e6:.2f}",
                'QoS Satisfied': "Yes" if res['min_rate_satisfied'][k] else "No"
            })
        st.dataframe(pd.DataFrame(modes_summary), use_container_width=True)


# ---------------- PAGE 8: AI TRAINING ----------------
elif menu == "8. AI Training":
    st.title("Optimization-Guided Multi-Output Neural Network Training")
    
    st.subheader("1. Generate Training & Validation Data")
    num_samples = st.number_input("Number of Physical Scenarios to Generate", min_value=50, max_value=500, value=100, step=25)
    
    if st.button("Generate Training Dataset"):
        with st.spinner("Simulating propagation scenarios and querying optimizer teacher..."):
            gen = DatasetGenerator(st.session_state.config, seed=st.session_state.config.random_seed)
            prog_bar = st.progress(0.0)
            
            def cb(curr, total):
                prog_bar.progress(float(curr / total))
                
            full_df = gen.generate_dataset(num_scenarios=int(num_samples), progress_callback=cb)
            
            # Split train (80%), val (20%)
            n_train = int(0.8 * len(full_df))
            st.session_state.train_df = full_df.iloc[:n_train].reset_index(drop=True)
            st.session_state.val_df = full_df.iloc[n_train:].reset_index(drop=True)
            st.session_state.train_df.to_csv("data/training_dataset.csv", index=False)
            st.session_state.val_df.to_csv("data/validation_dataset.csv", index=False)
            st.success(f"Generated {len(full_df)} scenarios! (Train: {len(st.session_state.train_df)}, Val: {len(st.session_state.val_df)})")
            
    st.subheader("2. Train PyTorch Neural Network")
    epochs = st.number_input("Training Epochs", min_value=10, max_value=150, value=40, step=5)
    lr = st.selectbox("Learning Rate", [1e-2, 5e-3, 1e-3, 5e-4], index=2)
    
    if st.button("Train Model Now"):
        if st.session_state.train_df is None or st.session_state.val_df is None:
            # Check if saved files exist on disk
            if os.path.exists("data/training_dataset.csv") and os.path.exists("data/validation_dataset.csv"):
                st.session_state.train_df = pd.read_csv("data/training_dataset.csv")
                st.session_state.val_df = pd.read_csv("data/validation_dataset.csv")
            else:
                st.error("Please generate the dataset first using the button above.")
                st.stop()
                
        with st.spinner("Training multi-output neural network..."):
            trainer = ModelTrainer(in_features=36, num_users=st.session_state.config.num_users)
            prog = st.progress(0.0)
            status_text = st.empty()
            
            def train_cb(ep, tot, t_loss, v_loss):
                prog.progress(float(ep / tot))
                status_text.text(f"Epoch {ep}/{tot} - Train Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f}")
                
            train_res = trainer.fit(
                train_df=st.session_state.train_df,
                val_df=st.session_state.val_df,
                epochs=int(epochs),
                lr=float(lr),
                progress_callback=train_cb
            )
            
            st.session_state.model_evaluator = ModelEvaluator(config=st.session_state.config)
            st.success("Neural Network training completed and checkpoint saved!")
            
            # Plot training history
            st.plotly_chart(plot_training_history(train_res['history']), use_container_width=True)


# ---------------- PAGE 9: AI ALLOCATION ----------------
elif menu == "9. AI Allocation":
    st.title("Real-Time AI Joint Link Selection and Power Allocation")
    
    if not st.session_state.model_evaluator.is_trained:
        st.warning("No trained AI model checkpoint found. Please go to '8. AI Training' to train the model first.")
    else:
        st.markdown("Runs sub-millisecond neural network inference on the current live network state, predicts joint link modes and $(\\alpha, \\beta)$, and compares execution time with the optimizer teacher.")
        
        if st.button("Execute AI Controller"):
            with st.spinner("Running PyTorch neural network inference..."):
                rf_model = RFChannelModel(st.session_state.config.rf)
                vlc_model = VLCChannelModel(st.session_state.config.vlc)
                
                dists_rf = st.session_state.env.get_rf_distances()
                rf_gains = rf_model.calculate_channel_gains(dists_rf)
                
                dists_vlc = st.session_state.env.get_vlc_distances()
                phi, psi = st.session_state.env.get_vlc_angles_matrix()
                blk = st.session_state.env.compute_optical_blockage_matrix()
                h_mat = vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
                vlc_eff, serving = vlc_model.get_effective_user_channel_gains(h_mat)
                is_blk = (vlc_eff < 1e-12)
                
                # Single scenario dictionary
                gen = DatasetGenerator(st.session_state.config)
                rec = gen.generate_scenario(0, st.session_state.env)
                row = pd.Series(rec)
                
                eval_res = st.session_state.model_evaluator.evaluate_scenario(row)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("AI Inference Time", f"{eval_res['ai_latency_ms']:.3f} ms")
                with c2:
                    st.metric("Teacher Runtime", f"{eval_res['teacher_runtime_ms']:.1f} ms")
                with c3:
                    speedup = eval_res['teacher_runtime_ms'] / max(1e-4, eval_res['ai_latency_ms'])
                    st.metric("Speedup Factor", f"{speedup:.1f}x")
                with c4:
                    st.metric("Link Agreement", f"{eval_res['mode_agreement'] * 100:.0f}%")
                    
                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    st.metric("AI Sum Rate", f"{eval_res['ai_sum_rate'] / 1e6:.2f} Mbps")
                with c6:
                    st.metric("Teacher Sum Rate", f"{eval_res['teacher_sum_rate'] / 1e6:.2f} Mbps")
                with c7:
                    st.metric("AI Energy Efficiency", f"{eval_res['ai_ee']:.2f} b/J")
                with c8:
                    st.metric("Teacher EE", f"{eval_res['teacher_ee']:.2f} b/J")
                    
                # Per-user comparison table
                comp_rows = []
                for k in range(st.session_state.config.num_users):
                    comp_rows.append({
                        'User': f"User {k}",
                        'AI Assigned Link': LINK_MODES[eval_res['ai_modes'][k]],
                        'Teacher Link': LINK_MODES[eval_res['teacher_modes'][k]],
                        'Physical Blockage': "Blocked" if is_blk[k] else "Clear LOS"
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


# ---------------- PAGE 10: BASELINE COMPARISON ----------------
elif menu == "10. Baseline Comparison":
    st.title("Baseline Algorithm Benchmark Comparison")
    st.markdown("Compares **RF-only**, **VLC-only**, **Fixed Hybrid (50/50)**, and **Rule-Based Hybrid RSMA** on identical channel realizations.")
    
    if st.button("Run Full Baseline Evaluation"):
        with st.spinner("Benchmarking communication baselines..."):
            rf_model = RFChannelModel(st.session_state.config.rf)
            vlc_model = VLCChannelModel(st.session_state.config.vlc)
            
            dists_rf = st.session_state.env.get_rf_distances()
            rf_gains = rf_model.calculate_channel_gains(dists_rf)
            
            dists_vlc = st.session_state.env.get_vlc_distances()
            phi, psi = st.session_state.env.get_vlc_angles_matrix()
            blk = st.session_state.env.compute_optical_blockage_matrix()
            h_mat = vlc_model.calculate_dc_gain_matrix(dists_vlc, phi, psi, blk)
            vlc_eff, _ = vlc_model.get_effective_user_channel_gains(h_mat)
            is_blk = (vlc_eff < 1e-12)
            
            snr_db = rf_model.calculate_snr_db(st.session_state.config.rf.tx_power_max_w, rf_gains)
            
            bm = BaselineManager(st.session_state.config)
            r_rf = bm.run_rf_only(rf_gains, vlc_eff, is_blk)
            r_vlc = bm.run_vlc_only(rf_gains, vlc_eff, is_blk)
            r_fix = bm.run_fixed_hybrid(rf_gains, vlc_eff, is_blk)
            r_rule = bm.run_rule_based_hybrid(rf_gains, vlc_eff, is_blk, snr_db)
            
            teacher = OptimizationTeacher(st.session_state.config, alpha_grid_points=7, beta_grid_points=5)
            r_opt = teacher.optimize(rf_gains, vlc_eff, is_blk)
            r_opt['method_name'] = "Optimizer Teacher (Target)"
            
            schemes = [r_rf, r_vlc, r_fix, r_rule, r_opt]
            table_data = []
            for s in schemes:
                table_data.append({
                    'Scheme': s['method_name'],
                    'Sum Rate (Mbps)': f"{s['sum_rate'] / 1e6:.2f}",
                    'Elec Power (W)': f"{s['total_elec_power_w']:.2f}",
                    'Energy Efficiency (bits/J)': f"{s['energy_efficiency']:.2f}",
                    'Fairness Index': f"{calculate_jains_fairness_index(s['user_rates']):.3f}",
                    'Feasible': "Yes" if s['is_feasible'] else "No"
                })
            df_comp = pd.DataFrame(table_data)
            st.dataframe(df_comp, use_container_width=True)
            
            # Plot comparison bar chart
            fig_bar = px.bar(
                df_comp,
                x='Scheme',
                y=[float(x['Energy Efficiency (bits/J)']) for x in table_data],
                labels=dict(x="Transmission Scheme", y="Energy Efficiency (bits/Joule)"),
                title="Energy Efficiency Comparison Across Baselines",
                color='Scheme',
                color_discrete_sequence=['#38bdf8', '#fbbf24', '#a855f7', '#34d399', '#00f2fe']
            )
            fig_bar.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
            st.plotly_chart(fig_bar, use_container_width=True)


# ---------------- PAGE 11: EXPERIMENTS ----------------
elif menu == "11. Experiments":
    st.title("Configurable Scientific Experiments Suite")
    
    exp_choice = st.selectbox(
        "Select Experiment",
        [
            "Experiment C: Multi-Access Comparison (RSMA vs SDMA vs NOMA)",
            "Experiment D: Power Allocation Alpha Sweep [0, 1]",
            "Experiment E: RSMA Beta Common Power Sweep [0, 0.7]",
            "Experiment B: Optical Blockage & Obstacle Count Scaling",
            "Experiment G: Ablation Study (Quantifying System Components)"
        ]
    )
    
    runner = ExperimentRunner(st.session_state.config, seed=st.session_state.config.random_seed)
    
    if st.button("Run Selected Experiment"):
        with st.spinner("Executing simulation experiment batches..."):
            if "Multi-Access" in exp_choice:
                res_df = runner.run_multi_access_comparison(num_trials=25)
                st.dataframe(res_df.describe(), use_container_width=True)
                
                # Plot
                fig = px.box(
                    res_df[['RSMA_Sum_Rate_Mbps', 'SDMA_Sum_Rate_Mbps', 'NOMA_Sum_Rate_Mbps']],
                    labels=dict(value="Throughput (Mbps)", variable="Multiple Access Scheme"),
                    title="Throughput Distribution: RSMA vs SDMA vs NOMA"
                )
                fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)
                
            elif "Alpha Sweep" in exp_choice:
                res_df = runner.run_alpha_sweep(num_points=11, num_trials=15)
                st.dataframe(res_df, use_container_width=True)
                fig = px.line(
                    res_df, x='alpha', y=['mean_sum_rate_mbps', 'mean_ee_bits_per_j'],
                    labels=dict(alpha="Alpha (RF Power Fraction)", value="Value"),
                    title="Throughput and Energy Efficiency Tradeoff vs Alpha"
                )
                fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)
                
            elif "Beta Common" in exp_choice:
                res_df = runner.run_beta_sweep(num_points=11, num_trials=15)
                st.dataframe(res_df, use_container_width=True)
                fig = px.line(
                    res_df, x='beta', y='mean_fairness',
                    labels=dict(beta="Beta (Common Stream Power Fraction)", mean_fairness="Jain's Fairness Index"),
                    title="Jain's Fairness Index vs RSMA Common Power Fraction (Beta)"
                )
                fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)
                
            elif "Optical Blockage" in exp_choice:
                res_df = runner.run_blockage_experiment(max_obstacles=4, num_trials=15)
                st.dataframe(res_df, use_container_width=True)
                fig = px.line(
                    res_df, x='num_obstacles', y=['vlc_only_rate_mbps', 'hybrid_rate_mbps'],
                    labels=dict(num_obstacles="Number of Optical Blockers", value="Throughput (Mbps)"),
                    title="VLC-only Degradation vs Hybrid Blockage Resilience"
                )
                fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)
                
            elif "Ablation" in exp_choice:
                res_df = runner.run_ablation_study(num_trials=20)
                st.dataframe(res_df, use_container_width=True)
                fig = px.bar(
                    res_df, x='Ablation_Variant', y='Mean_EE_bits_per_J',
                    labels=dict(Ablation_Variant="Variant", Mean_EE_bits_per_J="Mean EE (bits/J)"),
                    title="Ablation Analysis: Contribution of Joint Link, Alpha, Beta, and RSMA"
                )
                fig.update_layout(paper_bgcolor='#0b1120', plot_bgcolor='#0b1120', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)


# ---------------- PAGE 12: RESULTS & EXPORT ----------------
elif menu == "12. Results & Report Export":
    st.title("Results & Technical Report Export")
    
    st.markdown("Generate and download publication-ready markdown technical reports and CSV metrics tables.")
    
    if st.button("Generate Comprehensive Technical Report"):
        rep_path = generate_research_report(output_path="results/RESEARCH_REPORT.md")
        with open(rep_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.success(f"Report generated successfully at `{rep_path}`!")
        st.download_button(
            label="Download Technical Report (Markdown)",
            data=report_text,
            file_name="RESEARCH_REPORT.md",
            mime="text/markdown"
        )
        st.markdown(report_text)
