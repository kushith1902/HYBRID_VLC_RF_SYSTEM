"""
Default Configuration for 6G Hybrid RF/VLC RSMA Simulation.
All physical parameters, room geometry, channel parameters, and thresholds.
Documented with units, physical equations, and scientific references.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class RoomConfig:
    """Indoor room dimensions and spatial configuration."""
    length: float = 5.0      # meters (x-axis)
    width: float = 5.0       # meters (y-axis)
    height: float = 3.0      # meters (z-axis)
    user_height: float = 0.85 # meters (desk / device height)


@dataclass
class RFConfig:
    """
    Sub-THz 140 GHz Radio Frequency Channel Configuration.
    Propagation Model: Close-In (CI) Free-Space Reference Distance Model
    Reference: Rappaport et al., IEEE Trans. Antennas Propag. (sub-THz indoor measurements)
    """
    carrier_frequency: float = 140e9  # 140 GHz
    bandwidth: float = 1.0e9          # 1.0 GHz
    tx_power_max_dbm: float = 23.0    # 23 dBm = 200 mW max transmit power
    tx_antenna_gain_dbi: float = 15.0 # Directional horn / antenna array (15 dBi)
    rx_antenna_gain_dbi: float = 5.0  # Receiver mobile antenna gain (5 dBi)
    path_loss_exp_los: float = 1.9    # CI Path loss exponent (LOS indoor)
    path_loss_exp_nlos: float = 2.8   # CI Path loss exponent (NLOS indoor)
    shadow_std_los_db: float = 2.0    # Shadow fading standard deviation LOS (dB)
    shadow_std_nlos_db: float = 4.0   # Shadow fading standard deviation NLOS (dB)
    molecular_absorption_db_m: float = 0.0005 # Molecular absorption at 140 GHz (dB/m)
    noise_figure_db: float = 7.0      # Sub-THz receiver noise figure (dB)
    temperature_k: float = 295.0      # Standard room temperature (295 K = 22 C)
    ap_position: Tuple[float, float, float] = (2.5, 2.5, 3.0) # Ceiling center

    @property
    def tx_power_max_w(self) -> float:
        """Transmit power in Watts."""
        return 10.0 ** ((self.tx_power_max_dbm - 30.0) / 10.0)

    @property
    def wavelength(self) -> float:
        """RF wavelength in meters."""
        return 3e8 / self.carrier_frequency


@dataclass
class VLCConfig:
    """
    Visible Light Communication (VLC) IM/DD Channel Configuration.
    Lambertian Radiation Model with Photodiode Receiver.
    Reference: Kahn and Barry, Proc. IEEE, 1997; Komine and Nakagawa, IEEE Trans. Consum. Electron., 2004.
    """
    num_leds: int = 4
    led_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (1.25, 1.25, 3.0),
        (1.25, 3.75, 3.0),
        (3.75, 1.25, 3.0),
        (3.75, 3.75, 3.0)
    ])
    optical_power_per_led_w: float = 5.0 # Radiated optical power per LED (5 W)
    semi_angle_half_power_deg: float = 60.0 # Semi-angle at half power (deg)
    pd_area_cm2: float = 1.0             # Photodiode physical area (1 cm^2)
    pd_fov_deg: float = 60.0             # Receiver Field of View semi-angle (deg)
    pd_responsivity: float = 0.53        # Responsivity R_pd (A/W) at 650-800 nm
    optical_filter_gain: float = 1.0     # Ts(psi) optical filter transmission
    refractive_index: float = 1.5        # Lens refractive index n
    bandwidth: float = 20.0e6            # Optical modulation bandwidth (20 MHz)
    bg_current: float = 5100e-6          # Ambient light background current (A)
    thermal_noise_density: float = 1e-22 # Receiver thermal noise current variance density (A^2/Hz)
    shot_noise_factor: float = 1.0       # Electronic charge multiplier

    @property
    def pd_area_m2(self) -> float:
        """PD area in square meters."""
        return self.pd_area_cm2 * 1e-4

    @property
    def lambertian_order(self) -> float:
        """Lambertian radiation order m = -ln(2) / ln(cos(Phi_1/2))."""
        phi_rad = np.radians(self.semi_angle_half_power_deg)
        return float(-np.log(2.0) / np.log(np.cos(phi_rad)))

    @property
    def optical_concentrator_gain(self) -> float:
        """Optical concentrator gain g(psi) = n^2 / sin^2(FOV)."""
        fov_rad = np.radians(self.pd_fov_deg)
        return float((self.refractive_index ** 2) / (np.sin(fov_rad) ** 2))

    @property
    def total_optical_power_max_w(self) -> float:
        """Total optical power budget across all LEDs in Watts."""
        return self.num_leds * self.optical_power_per_led_w


@dataclass
class RSMAConfig:
    """
    Rate-Splitting Multiple Access (RSMA) Model Parameters.
    Reference: Clerckx et al., IEEE JSAC, 2016; Mao et al., IEEE TCOM, 2018.
    """
    default_beta: float = 0.3        # Fraction of power allocated to common stream (beta in [0, 1])
    min_sinr_common_db: float = -3.0 # Minimum SINR required to decode common stream (dB)
    common_rate_weights: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])


@dataclass
class PowerConsumptionConfig:
    """
    Hardware Power Consumption Model for Energy Efficiency (EE) Evaluation.
    Reference: Auer et al., EARTH Project; Chen et al., IEEE Trans. Green Commun. Netw.
    """
    rf_pa_efficiency: float = 0.20       # Sub-THz RF Power Amplifier efficiency (20%)
    rf_rfic_power_w: float = 0.80        # Sub-THz RFIC / transceiver circuit power (Watts)
    vlc_led_efficiency: float = 0.25     # LED electrical-to-optical conversion efficiency (25%)
    vlc_circuit_power_per_led_w: float = 0.20 # VLC driver circuit power per LED (Watts)
    static_dsp_power_w: float = 1.0      # Baseband processing and central controller static power (Watts)
    rx_circuit_power_per_user_w: float = 0.15 # User receiver circuit power (Watts)


@dataclass
class SystemConfig:
    """Master simulation configuration container."""
    room: RoomConfig = field(default_factory=RoomConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    vlc: VLCConfig = field(default_factory=VLCConfig)
    rsma: RSMAConfig = field(default_factory=RSMAConfig)
    power: PowerConsumptionConfig = field(default_factory=PowerConsumptionConfig)
    
    num_users: int = 8                    # High density indoor scenario (default 8 users)
    default_alpha: float = 0.5            # RF vs VLC power allocation factor in [0, 1]
    min_user_rate_bps: float = 10.0e6     # 10 Mbps QoS minimum rate constraint per user
    random_seed: int = 42
