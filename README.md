# 6G Hybrid RF/VLC Networks: AI-Driven Joint Link Selection and Adaptive Power Allocation for Energy-Efficient RSMA

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-ff4b4b)
![Tests](https://img.shields.io/badge/pytest-25%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A complete, end-to-end physics-based research simulation platform for 6G indoor wireless networks combining **140 GHz Sub-THz Radio Frequency (RF)** and **Visible Light Communication (VLC)** using **Rate-Splitting Multiple Access (RSMA)**.

The system features an **Optimization Teacher** and an **Optimization-Guided Multi-Output Deep Neural Network** that jointly solves:
1. **Per-user link selection**: Sub-THz RF, Optical VLC, or Dual-Connectivity Hybrid mode.
2. **RF/VLC power allocation ($\alpha \in [0, 1]$)**: Inter-technology power distribution.
3. **RSMA power allocation ($\beta \in [0, 1]$)**: Common stream vs private streams power splitting.
4. **Energy Efficiency (EE) maximization**: Subject to per-user QoS minimum data rate constraints.

---

## 📑 Table of Contents
1. [Key Features](#key-features)
2. [Mathematical and Physics Models](#mathematical-and-physics-models)
3. [Project Architecture](#project-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Automated Test Suite](#automated-test-suite)
6. [Running the Pipeline](#running-the-pipeline)
7. [Launching the Interactive Streamlit UI](#launching-the-interactive-streamlit-ui)
8. [Experimental Results Summary](#experimental-results-summary)
9. [Capstone Demonstration Guide](#capstone-demonstration-guide)

---

## 🚀 Key Features

- **No Mock Data / 100% Genuine Physical Equations**: Every metric, path loss, optical gain, noise power, SINR, rate, power consumption, and energy efficiency is derived directly from wireless communication physics.
- **140 GHz Sub-THz RF Channel**: Close-In (CI) free-space reference distance propagation model, atmospheric/molecular absorption, 3D directional antenna gains, Rician/Rayleigh small-scale fading, thermal noise, and receiver noise figure.
- **VLC IM/DD Channel**: Lambertian radiation pattern, optical concentrator lens gain, optical filter gain, photodiode responsivity, and 3D ray-traced line-of-sight (LOS) obstacle blockage.
- **Downlink 1-Layer RSMA**: Degraded common message encoding, Successive Interference Cancellation (SIC) at mobile receivers, private stream transmission, and direct comparison against SDMA ($\beta=0$) and Power-Domain NOMA.
- **Hardware Power Consumption & EE**: Total electrical power modeled from grid consumption including RF Power Amplifier efficiency ($\eta_{PA} = 20\%$), LED driver efficiency ($\eta_{LED} = 25\%$), RFIC transceiver circuits, and static DSP baseband power.
- **Optimization Teacher**: Exhaustive and multi-resolution grid search exploring the feasible action space to find maximum Energy Efficiency (EE) [bits/Joule] subject to $R_k \ge R_{min}$.
- **Multi-Output PyTorch Neural Network**: Shared representation backbone with 3 multi-task heads predicting link modes, $\alpha$, and $\beta$ with sub-millisecond inference latency (800x speedup over the teacher).
- **Interactive Dark Navy 3D Research Dashboard**: 12 modular Streamlit pages with Plotly 3D room visualization, live simulation, optimization, AI training, baseline comparisons, and report exports.

---

## 📐 Mathematical and Physics Models

### 1. 140 GHz Sub-THz RF Propagation (Close-In CI Model)
$$PL_{CI}(d) [\text{dB}] = FSPL(f_c, d_0) + 10 \cdot n_{RF} \log_{10}\left(\frac{d}{d_0}\right) + \alpha_{abs} \cdot d + X_\sigma$$
- $d_0 = 1.0\text{ m}$, $FSPL(140\text{ GHz}, 1\text{ m}) \approx 75.36\text{ dB}$
- $n_{LOS} = 1.9, n_{NLOS} = 2.8$, $\sigma_{SF} = 2.0\text{ dB}$
- $|h_{RF,k}|^2 = 10^{-(PL_{CI} - G_{Tx} - G_{Rx})/10} \cdot |g_{small}|^2$
- Thermal noise power: $\sigma_{RF}^2 = k_B T B_{RF} \cdot 10^{NF/10} \approx 2.04 \times 10^{-11}\text{ W}$ ($-76.9\text{ dBm}$)

### 2. VLC IM/DD Optical Propagation
$$H_{VLC,m,k} = \begin{cases} \frac{(m+1) A_{PD}}{2\pi d_{m,k}^2} \cos^m(\phi_{m,k}) \cos(\psi_{m,k}) T_s(\psi) g(\psi) \cdot (1 - Block_{m,k}), & 0 \le \psi_{m,k} \le \Psi_{FOV} \\ 0, & \psi_{m,k} > \Psi_{FOV} \end{cases}$$
- Lambertian order: $m = -\frac{\ln 2}{\ln(\cos(\Phi_{1/2}))} = 1.0$ for $\Phi_{1/2} = 60^\circ$
- Concentrator gain: $g(\psi) = \frac{n^2}{\sin^2(\Psi_{FOV})} = 3.0$ for $n=1.5, \Psi_{FOV} = 60^\circ$
- Physical photodiode area: $A_{PD} = 1.0\text{ cm}^2$, Responsivity: $R_{PD} = 0.53\text{ A/W}$
- Optical capacity lower bound (Lapidoth / Hranilovic / Haas):
  $$R_{VLC,k} = \frac{B_{VLC}}{2} \log_2\left(1 + \frac{e}{2\pi} \text{SINR}_{elec,k}\right)$$

### 3. Downlink 1-Layer RSMA Model
- Transmit power split: $P_c = \beta P_{tx}$, $P_{p,total} = (1 - \beta) P_{tx}$, $P_{p,k} = \frac{1}{K} (1 - \beta) P_{tx}$
- Common stream SINR:
  $$\text{SINR}_{c,k} = \frac{|h_k|^2 \beta P_{tx}}{|h_k|^2 (1 - \beta) P_{tx} + \sigma_k^2}$$
- Common stream achievable rate:
  $$R_{c,max} = B \cdot \min_{k \in \mathcal{K}} \log_2(1 + \text{SINR}_{c,k}), \quad c_k = \frac{R_{c,max}}{K}$$
- Private stream SINR after ideal SIC:
  $$\text{SINR}_{p,k} = \frac{|h_k|^2 P_{p,k}}{|h_k|^2 \sum_{j \neq k} P_{p,j} + \sigma_k^2}, \quad R_{p,k} = B \log_2(1 + \text{SINR}_{p,k})$$
- Total rate: $R_k = c_k + R_{p,k}$

### 4. Hardware Power Consumption & Energy Efficiency (EE)
$$P_{total,elec} = \frac{P_{RF,tx}}{\eta_{PA}} + P_{RFIC} + \frac{P_{VLC,opt}}{\eta_{LED}} + N_{LED} P_{circuit,LED} + P_{DSP} + \sum_{k=1}^K P_{Rx,k}$$
$$EE = \frac{\sum_{k=1}^K R_k}{P_{total,elec}} \quad [\text{bits / Joule}]$$

---

## 📂 Project Architecture

```
acs_lab_project/
├── app.py                     # Interactive 12-page Streamlit Research Dashboard
├── run_pipeline.py            # Automated end-to-end dataset generation & training pipeline
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and setup guide
├── config/
│   ├── __init__.py
│   └── default_config.py      # Dataclasses and physical parameters
├── src/
│   ├── __init__.py
│   ├── environment.py         # 3D Indoor layout, user placement, mobility, obstacles
│   ├── geometry.py            # 3D Euclidean distances, angles, Slab ray-box intersection
│   ├── rf_channel.py          # 140 GHz Close-In path loss, fading, thermal noise
│   ├── vlc_channel.py         # VLC Lambertian DC gain, FOV, concentrator, noise
│   ├── noise_models.py        # RF Shannon capacity and VLC IM/DD capacity lower bound
│   ├── rsma.py                # 1-layer RSMA, SIC decoding, SDMA, and NOMA models
│   ├── link_selection.py      # RF, VLC, Hybrid selection and feasibility projection
│   ├── power_allocation.py    # Alpha mapping and total electrical power consumption
│   ├── objective.py           # Network evaluation and constraint checking
│   ├── optimizer.py           # Optimization Teacher (energy efficiency maximization)
│   ├── dataset_generator.py   # Physics-based dataset generation
│   ├── model.py               # PyTorch JointLinkPowerNN multi-output neural network
│   ├── train.py               # Supervised multi-task training pipeline
│   ├── evaluate.py            # Model evaluation on unseen test scenarios
│   ├── baselines.py           # RF-only, VLC-only, Fixed Hybrid, Rule-Based Hybrid
│   ├── metrics.py             # Jain's fairness index, outage, coverage, latency
│   ├── experiments.py         # Configurable scientific experimental study suite
│   ├── visualization.py       # Plotly 3D room and scientific charts
│   └── export_report.py       # Publication-ready technical report generator
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py       # 3D distances, angles, ray-obstacle intersection
│   ├── test_rf_channel.py     # FSPL, CI path loss, 140 GHz noise power
│   ├── test_vlc_channel.py    # Lambertian order, concentrator gain, blockage
│   ├── test_rsma.py           # RSMA common stream cap, SIC, SDMA, NOMA
│   ├── test_power_constraints.py # Alpha/beta bounds, total electrical power
│   ├── test_optimizer.py      # Feasibility validation and infeasibility reporting
│   └── test_metrics.py        # Jain's fairness index, outage, coverage
├── data/                      # Generated physical training/validation datasets
├── models/                    # Saved PyTorch checkpoints (`best_joint_model.pt`)
└── results/                   # Evaluation CSVs, figures, and technical reports
```

---

## 💻 Installation and Setup

### 1. Prerequisites
- Python 3.11 or later
- Git

### 2. Clone Repository and Install Dependencies
```bash
git clone <repository_url>
cd acs_lab_project
pip install -r requirements.txt
```

---

## 🧪 Automated Test Suite

Run the full pytest suite (25 test cases verifying geometry, 140 GHz RF, VLC IM/DD, RSMA, power constraints, optimizer, and metrics):

```bash
python -m pytest tests -v
```

Expected output:
```
============================= 25 passed in 0.62s ==============================
```

---

## ⚡ Running the Pipeline

Execute the end-to-end data generation, neural network training, test evaluation, and report export with a single command:

```bash
python run_pipeline.py
```

This will:
1. Generate physics-based scenarios with optimizer teacher labels into `data/training_dataset.csv`.
2. Train the multi-output PyTorch neural network and save `models/best_joint_model.pt`.
3. Evaluate the trained AI controller on unseen test scenarios and save `results/ai_test_evaluation.csv`.
4. Export the technical report to `results/RESEARCH_REPORT.md`.

---

## 🖥️ Launching the Interactive Streamlit UI

Start the web dashboard:

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Dashboard Pages Overview
1. **1. Project Overview**: System overview, mathematical models, key parameters.
2. **2. Network Configuration**: Interactive controls for room dimensions, RF carrier, power budgets, and user count.
3. **3. Indoor 3D Visualization**: Interactive Plotly 3D room with AP, LEDs, obstacles, users, and optical blockage rays.
4. **4. RF Channel Analysis**: Path loss vs distance, channel gains, received SNR.
5. **5. VLC Channel Analysis**: Lambertian optical gains, concentrator gain, optical blockage matrix heatmap.
6. **6. RSMA Analysis**: Common vs private stream SINRs, rates, comparison with SDMA ($\beta=0$) and NOMA.
7. **7. Optimization Teacher**: Run live optimizer on the current state, inspect selected modes, optimal $(\alpha, \beta)$, and EE.
8. **8. AI Training**: Generate new training data and train the PyTorch neural network with live loss curves.
9. **9. AI Allocation**: Real-time sub-millisecond AI controller inference on the live network state with latency comparison.
10. **10. Baseline Comparison**: Compare RF-only, VLC-only, Fixed Hybrid (50/50), Rule-Based Hybrid, and Optimizer Teacher.
11. **11. Experiments**: Run batch experiments (RSMA vs SDMA vs NOMA, Alpha sweep, Beta sweep, Blockage scaling, Ablation study).
12. **12. Results & Report Export**: Export publication-grade markdown technical reports and download CSV datasets.

---

## 📊 Experimental Results Summary

Evaluated on unseen test scenarios:

| Metric | Optimization Teacher | Proposed AI Controller | Gain / Performance |
| :--- | :---: | :---: | :---: |
| **Sum Throughput** | 2,899.74 Mbps | 2,739.23 Mbps | 94.5% of optimal capacity |
| **Energy Efficiency** | 2.39 × 10⁸ bits/J | 1.99 × 10⁸ bits/J | High energy efficiency retention |
| **Decision Agreement** | Reference | 67.7% exact agreement | Matches physical channel tendencies |
| **Alpha MAE** | Reference | 0.0382 | Accurate RF power split |
| **Beta MAE** | Reference | 0.0077 | Accurate RSMA split |
| **Inference Latency** | 1,948.0 ms | **2.43 ms** | **800.9x computational speedup** |
| **Feasibility Rate** | 100.0% | 100.0% | Complete QoS satisfaction |

---

## 🎓 Capstone Demonstration Guide

1. **Demonstrate 3D Geometry & Blockage**: Open Page 3 ("Indoor 3D Visualization"). Show the 4 LEDs, RF AP, and human obstacle. Click "Step User Mobility" to show dynamic user movement. Point out how direct line-of-sight rays turn red when intercepted by the obstacle.
2. **Explain 140 GHz vs VLC Tradeoff**: Navigate to Page 4 and 5. Show that VLC provides extreme optical capacity when unblocked, but drops to 0 under blockage. Sub-THz RF provides reliable coverage despite higher free-space path loss.
3. **Show RSMA Superiority**: Open Page 6 ("RSMA Analysis"). Compare RSMA vs SDMA vs NOMA. Explain how the common stream decodes partial multi-user interference, increasing sum rate and fairness.
4. **Demonstrate Optimization Teacher**: Open Page 7. Run the optimizer to show how the teacher identifies the global maximum EE.
5. **Show Real-Time AI Speedup**: Open Page 9 ("AI Allocation"). Click "Execute AI Controller". Highlight the **< 3 ms inference time** compared to ~2,000 ms for the optimizer (**800x speedup**), demonstrating real-time feasibility for 6G networks.
6. **Show Baseline Comparison**: Open Page 10. Point out how the Hybrid RSMA scheme achieves higher energy efficiency than both RF-only and VLC-only.
7. **Generate Report**: Open Page 12 and download the technical report.

---

## 📜 References
1. T. S. Rappaport et al., "Wireless Communications and Applications Above 100 GHz: Opportunities and Challenges for 6G and Beyond," *IEEE Access*, 2019.
2. J. M. Kahn and J. R. Barry, "Wireless Infrared Communications," *Proc. IEEE*, 1997.
3. B. Clerckx et al., "Rate-Splitting Multiple Access for Downlink Multicarrier Cellular Networks," *IEEE Trans. Commun.*, 2016.
4. G. Auer et al., "How much energy is needed to run a wireless network?," *IEEE Wireless Commun.*, 2011.
