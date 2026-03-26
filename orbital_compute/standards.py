"""ECSS standards compliance and link budget calculations.

Implements space industry standard formats and calculations:
- Power budget per ECSS-E-ST-20C
- Thermal budget per ECSS-E-ST-31C
- Link budget per ECSS-E-ST-50C (Communications)
- Link budget calculator for S/X/Ka/optical bands
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

BOLTZMANN_K = 1.380649e-23   # J/K  (Boltzmann constant)
SPEED_OF_LIGHT = 2.998e8     # m/s
EARTH_RADIUS_KM = 6371.0


# ============================================================================
# ECSS-E-ST-20C: Power Budget
# ============================================================================

@dataclass
class PowerBudgetItem:
    """A single item in the power budget."""
    subsystem: str
    mode: str           # e.g., "nominal", "peak", "eclipse", "safe"
    power_w: float
    duty_cycle: float = 1.0    # fraction of time active (0-1)
    notes: str = ""

    @property
    def average_power_w(self) -> float:
        return self.power_w * self.duty_cycle


@dataclass
class PowerBudget:
    """ECSS-E-ST-20C compliant power budget.

    Organizes power consumption by subsystem and operating mode.
    Includes margins per ECSS standard (5% component, 10% system, 20% EOL).
    """
    satellite_name: str
    items: List[PowerBudgetItem] = field(default_factory=list)
    solar_panel_eol_watts: float = 0.0
    battery_capacity_wh: float = 0.0
    component_margin_pct: float = 5.0     # Per ECSS
    system_margin_pct: float = 10.0       # Per ECSS
    eol_degradation_pct: float = 20.0     # End-of-life solar panel degradation

    def add_item(self, subsystem: str, mode: str, power_w: float,
                 duty_cycle: float = 1.0, notes: str = "") -> None:
        self.items.append(PowerBudgetItem(subsystem, mode, power_w, duty_cycle, notes))

    def total_power_w(self, mode: str = "nominal") -> float:
        """Total power for a given mode."""
        return sum(i.average_power_w for i in self.items if i.mode == mode)

    def total_with_margins_w(self, mode: str = "nominal") -> float:
        """Total power including ECSS margins."""
        base = self.total_power_w(mode)
        with_component = base * (1 + self.component_margin_pct / 100)
        with_system = with_component * (1 + self.system_margin_pct / 100)
        return with_system

    def solar_margin_pct(self, mode: str = "nominal") -> float:
        """Power margin: (available - required) / required * 100."""
        required = self.total_with_margins_w(mode)
        if required == 0:
            return float('inf')
        eol_power = self.solar_panel_eol_watts * (1 - self.eol_degradation_pct / 100)
        return (eol_power - required) / required * 100

    def eclipse_duration_hours(self) -> float:
        """Maximum eclipse duration the battery can support."""
        mode_power = self.total_with_margins_w("eclipse")
        if mode_power == 0:
            return float('inf')
        # Usable battery = 80% of capacity (DOD limit per ECSS)
        usable_wh = self.battery_capacity_wh * 0.8
        return usable_wh / mode_power

    def modes(self) -> List[str]:
        """All unique modes in the budget."""
        return sorted(set(i.mode for i in self.items))

    def subsystems(self) -> List[str]:
        """All unique subsystems."""
        return sorted(set(i.subsystem for i in self.items))

    def format_ecss(self) -> str:
        """Format as ECSS-E-ST-20C power budget table."""
        lines = [
            f"ECSS-E-ST-20C POWER BUDGET: {self.satellite_name}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')} UTC",
            "=" * 80,
            "",
        ]

        for mode in self.modes():
            mode_items = [i for i in self.items if i.mode == mode]
            total = self.total_power_w(mode)
            total_margin = self.total_with_margins_w(mode)

            lines.append(f"MODE: {mode.upper()}")
            lines.append("-" * 80)
            lines.append(f"{'Subsystem':<25s} {'Power(W)':>10s} {'Duty':>6s} {'Avg(W)':>10s} {'Notes'}")
            lines.append("-" * 80)

            for item in mode_items:
                lines.append(
                    f"{item.subsystem:<25s} {item.power_w:>10.1f} {item.duty_cycle:>5.0%}"
                    f" {item.average_power_w:>10.1f} {item.notes}"
                )

            lines.append("-" * 80)
            lines.append(f"{'Subtotal':<25s} {'':>10s} {'':>6s} {total:>10.1f}")
            lines.append(f"{'+ Component margin (' + str(self.component_margin_pct) + '%)':<25s}"
                         f" {'':>10s} {'':>6s}"
                         f" {total * (1 + self.component_margin_pct / 100):>10.1f}")
            lines.append(f"{'+ System margin (' + str(self.system_margin_pct) + '%)':<25s}"
                         f" {'':>10s} {'':>6s} {total_margin:>10.1f}")
            lines.append("")

        # Summary
        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append(f"  Solar panel EOL power:  {self.solar_panel_eol_watts:.1f} W")
        eol_degraded = self.solar_panel_eol_watts * (1 - self.eol_degradation_pct / 100)
        lines.append(f"  After {self.eol_degradation_pct:.0f}% degradation: {eol_degraded:.1f} W")
        lines.append(f"  Battery capacity:       {self.battery_capacity_wh:.1f} Wh")

        for mode in self.modes():
            margin = self.solar_margin_pct(mode)
            status = "OK" if margin > 0 else "NEGATIVE"
            lines.append(f"  {mode:15s} margin:  {margin:+.1f}% [{status}]")

        eclipse_hrs = self.eclipse_duration_hours()
        lines.append(f"  Max eclipse duration:   {eclipse_hrs:.2f} hours")
        lines.append("=" * 80)

        return '\n'.join(lines)


def create_compute_sat_power_budget(
    name: str = "COMPUTE-SAT-1",
    solar_w: float = 2000.0,
    battery_wh: float = 5000.0,
) -> PowerBudget:
    """Create a standard power budget for a compute satellite."""
    budget = PowerBudget(
        satellite_name=name,
        solar_panel_eol_watts=solar_w,
        battery_capacity_wh=battery_wh,
    )

    # Nominal mode
    budget.add_item("ADCS", "nominal", 45.0, 1.0, "Reaction wheels + star tracker")
    budget.add_item("OBC", "nominal", 25.0, 1.0, "Main computer")
    budget.add_item("Comms-TT&C", "nominal", 15.0, 1.0, "Telemetry & telecommand")
    budget.add_item("Comms-Payload", "nominal", 80.0, 0.15, "Data downlink (during passes)")
    budget.add_item("Thermal", "nominal", 30.0, 0.5, "Heaters (duty-cycled)")
    budget.add_item("Power-Housekeeping", "nominal", 20.0, 1.0, "PDU, converters")
    budget.add_item("GPU-Compute", "nominal", 600.0, 0.7, "AI inference payload")
    budget.add_item("Storage", "nominal", 15.0, 1.0, "NVMe storage")

    # Eclipse mode
    budget.add_item("ADCS", "eclipse", 45.0, 1.0, "Reaction wheels + star tracker")
    budget.add_item("OBC", "eclipse", 25.0, 1.0, "Main computer")
    budget.add_item("Comms-TT&C", "eclipse", 15.0, 1.0, "Telemetry & telecommand")
    budget.add_item("Thermal", "eclipse", 60.0, 0.8, "Heaters (higher duty in eclipse)")
    budget.add_item("Power-Housekeeping", "eclipse", 20.0, 1.0, "PDU, converters")
    budget.add_item("GPU-Compute", "eclipse", 400.0, 0.4, "Reduced compute in eclipse")
    budget.add_item("Storage", "eclipse", 15.0, 1.0, "NVMe storage")

    # Peak mode (full compute + downlink)
    budget.add_item("ADCS", "peak", 55.0, 1.0, "Reaction wheels + star tracker (slewing)")
    budget.add_item("OBC", "peak", 25.0, 1.0, "Main computer")
    budget.add_item("Comms-TT&C", "peak", 15.0, 1.0, "Telemetry & telecommand")
    budget.add_item("Comms-Payload", "peak", 80.0, 1.0, "Data downlink (active pass)")
    budget.add_item("Thermal", "peak", 30.0, 0.5, "Heaters")
    budget.add_item("Power-Housekeeping", "peak", 20.0, 1.0, "PDU, converters")
    budget.add_item("GPU-Compute", "peak", 800.0, 1.0, "Full GPU load")
    budget.add_item("Storage", "peak", 20.0, 1.0, "NVMe storage (high I/O)")

    # Safe mode
    budget.add_item("ADCS", "safe", 20.0, 1.0, "Detumble only")
    budget.add_item("OBC", "safe", 25.0, 1.0, "Main computer")
    budget.add_item("Comms-TT&C", "safe", 15.0, 1.0, "Telemetry & telecommand")
    budget.add_item("Thermal", "safe", 60.0, 1.0, "Survival heaters")
    budget.add_item("Power-Housekeeping", "safe", 20.0, 1.0, "PDU, converters")

    return budget


# ============================================================================
# ECSS-E-ST-31C: Thermal Budget
# ============================================================================

@dataclass
class ThermalBudgetItem:
    """A component in the thermal budget."""
    component: str
    min_op_temp_c: float     # Minimum operating temperature
    max_op_temp_c: float     # Maximum operating temperature
    min_nonop_temp_c: float  # Minimum non-operating (survival) temperature
    max_nonop_temp_c: float  # Maximum non-operating temperature
    predicted_min_c: float   # Predicted minimum in sim
    predicted_max_c: float   # Predicted maximum in sim
    heat_dissipation_w: float = 0.0
    notes: str = ""

    @property
    def min_margin_c(self) -> float:
        """Margin above minimum operating temperature."""
        return self.predicted_min_c - self.min_op_temp_c

    @property
    def max_margin_c(self) -> float:
        """Margin below maximum operating temperature."""
        return self.max_op_temp_c - self.predicted_max_c

    @property
    def within_limits(self) -> bool:
        return self.min_margin_c >= 0 and self.max_margin_c >= 0


@dataclass
class ThermalBudget:
    """ECSS-E-ST-31C compliant thermal budget."""
    satellite_name: str
    items: List[ThermalBudgetItem] = field(default_factory=list)
    qualification_margin_c: float = 11.0   # ECSS qualification margin
    acceptance_margin_c: float = 5.0       # ECSS acceptance margin

    def add_item(self, **kwargs) -> None:
        self.items.append(ThermalBudgetItem(**kwargs))

    def all_within_limits(self) -> bool:
        return all(item.within_limits for item in self.items)

    def format_ecss(self) -> str:
        """Format as ECSS-E-ST-31C thermal budget table."""
        lines = [
            f"ECSS-E-ST-31C THERMAL BUDGET: {self.satellite_name}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')} UTC",
            f"Qualification margin: {self.qualification_margin_c} C | "
            f"Acceptance margin: {self.acceptance_margin_c} C",
            "=" * 100,
            "",
            f"{'Component':<20s} {'Op Range(C)':>14s} {'Predicted(C)':>14s}"
            f" {'MinMarg':>8s} {'MaxMarg':>8s} {'Status':>8s} {'Heat(W)':>8s}",
            "-" * 100,
        ]

        for item in self.items:
            status = "OK" if item.within_limits else "FAIL"
            lines.append(
                f"{item.component:<20s}"
                f" [{item.min_op_temp_c:+6.1f},{item.max_op_temp_c:+6.1f}]"
                f" [{item.predicted_min_c:+6.1f},{item.predicted_max_c:+6.1f}]"
                f" {item.min_margin_c:>+7.1f} {item.max_margin_c:>+7.1f}"
                f" {status:>8s} {item.heat_dissipation_w:>7.1f}"
            )

        lines.append("-" * 100)
        all_ok = self.all_within_limits()
        lines.append(f"\nOverall status: {'PASS' if all_ok else 'FAIL'}")
        lines.append("=" * 100)
        return '\n'.join(lines)


def create_compute_sat_thermal_budget(
    name: str = "COMPUTE-SAT-1",
    gpu_predicted_max_c: float = 72.0,
) -> ThermalBudget:
    """Create a standard thermal budget for a compute satellite."""
    budget = ThermalBudget(satellite_name=name)

    budget.add_item(
        component="OBC", min_op_temp_c=-20, max_op_temp_c=60,
        min_nonop_temp_c=-40, max_nonop_temp_c=70,
        predicted_min_c=5, predicted_max_c=45, heat_dissipation_w=25,
    )
    budget.add_item(
        component="GPU-Module", min_op_temp_c=-10, max_op_temp_c=85,
        min_nonop_temp_c=-30, max_nonop_temp_c=100,
        predicted_min_c=10, predicted_max_c=gpu_predicted_max_c,
        heat_dissipation_w=600, notes="Primary compute payload",
    )
    budget.add_item(
        component="Battery-Pack", min_op_temp_c=0, max_op_temp_c=45,
        min_nonop_temp_c=-10, max_nonop_temp_c=55,
        predicted_min_c=10, predicted_max_c=35, heat_dissipation_w=30,
    )
    budget.add_item(
        component="Star-Tracker", min_op_temp_c=-30, max_op_temp_c=50,
        min_nonop_temp_c=-40, max_nonop_temp_c=60,
        predicted_min_c=-5, predicted_max_c=30, heat_dissipation_w=5,
    )
    budget.add_item(
        component="S-Band-Radio", min_op_temp_c=-20, max_op_temp_c=60,
        min_nonop_temp_c=-30, max_nonop_temp_c=70,
        predicted_min_c=0, predicted_max_c=40, heat_dissipation_w=15,
    )
    budget.add_item(
        component="X-Band-Tx", min_op_temp_c=-10, max_op_temp_c=70,
        min_nonop_temp_c=-20, max_nonop_temp_c=80,
        predicted_min_c=5, predicted_max_c=55, heat_dissipation_w=80,
    )
    budget.add_item(
        component="Solar-Panel", min_op_temp_c=-150, max_op_temp_c=120,
        min_nonop_temp_c=-170, max_nonop_temp_c=130,
        predicted_min_c=-80, predicted_max_c=95, heat_dissipation_w=0,
    )
    budget.add_item(
        component="Reaction-Wheel", min_op_temp_c=-20, max_op_temp_c=60,
        min_nonop_temp_c=-30, max_nonop_temp_c=70,
        predicted_min_c=0, predicted_max_c=40, heat_dissipation_w=10,
    )

    return budget


# ============================================================================
# ECSS-E-ST-50C: Link Budget Calculator
# ============================================================================

@dataclass
class FrequencyBand:
    """Radio frequency band definition."""
    name: str
    frequency_hz: float
    typical_tx_power_dbw: float
    typical_antenna_gain_dbi: float   # Satellite antenna
    typical_gs_gain_dbi: float        # Ground station antenna
    atmospheric_loss_db: float = 1.0
    rain_loss_db: float = 0.0
    notes: str = ""


# Standard frequency bands
FREQUENCY_BANDS = {
    "s_band": FrequencyBand(
        "S-band", 2.2e9, 3.0, 6.0, 35.0,
        atmospheric_loss_db=0.5, rain_loss_db=0.5,
        notes="Telemetry & telecommand, low data rate",
    ),
    "x_band": FrequencyBand(
        "X-band", 8.2e9, 10.0, 12.0, 45.0,
        atmospheric_loss_db=0.8, rain_loss_db=2.0,
        notes="Science data downlink, medium data rate",
    ),
    "ka_band": FrequencyBand(
        "Ka-band", 26.5e9, 10.0, 20.0, 55.0,
        atmospheric_loss_db=2.0, rain_loss_db=6.0,
        notes="High-rate data downlink",
    ),
    "optical": FrequencyBand(
        "Optical", 2.82e14, 0.0, 100.0, 110.0,
        atmospheric_loss_db=3.0, rain_loss_db=20.0,
        notes="Laser comm, highest data rate, weather-dependent",
    ),
}


@dataclass
class LinkBudgetResult:
    """Result of a link budget calculation."""
    # Configuration
    direction: str          # "uplink" or "downlink"
    band_name: str
    frequency_hz: float
    distance_km: float
    data_rate_bps: float

    # Budget components (dB)
    tx_power_dbw: float
    tx_antenna_gain_dbi: float
    eirp_dbw: float
    free_space_loss_db: float
    atmospheric_loss_db: float
    rain_loss_db: float
    pointing_loss_db: float
    rx_antenna_gain_dbi: float
    system_noise_temp_k: float
    noise_spectral_density_dbw_hz: float

    # Results
    received_power_dbw: float
    cn0_dbhz: float         # Carrier-to-noise-density ratio
    eb_n0_db: float          # Energy per bit to noise density
    eb_n0_required_db: float # Required Eb/N0 for target BER
    link_margin_db: float    # Available - Required

    # Derived
    achievable_data_rate_bps: float   # Max rate for 3 dB margin
    link_closed: bool

    def format_report(self) -> str:
        """Format as a standard link budget report."""
        lines = [
            f"LINK BUDGET ANALYSIS ({self.direction.upper()})",
            f"Band: {self.band_name} ({self.frequency_hz / 1e9:.1f} GHz)",
            "=" * 60,
            "",
            "TRANSMITTER",
            f"  TX Power:              {self.tx_power_dbw:>8.1f} dBW",
            f"  TX Antenna Gain:       {self.tx_antenna_gain_dbi:>8.1f} dBi",
            f"  EIRP:                  {self.eirp_dbw:>8.1f} dBW",
            "",
            "PATH LOSSES",
            f"  Distance:              {self.distance_km:>8.1f} km",
            f"  Free Space Loss:       {self.free_space_loss_db:>8.1f} dB",
            f"  Atmospheric Loss:      {self.atmospheric_loss_db:>8.1f} dB",
            f"  Rain Loss:             {self.rain_loss_db:>8.1f} dB",
            f"  Pointing Loss:         {self.pointing_loss_db:>8.1f} dB",
            "",
            "RECEIVER",
            f"  RX Antenna Gain:       {self.rx_antenna_gain_dbi:>8.1f} dBi",
            f"  System Noise Temp:     {self.system_noise_temp_k:>8.1f} K",
            f"  N0:                    {self.noise_spectral_density_dbw_hz:>8.1f} dBW/Hz",
            "",
            "RESULTS",
            f"  Received Power:        {self.received_power_dbw:>8.1f} dBW",
            f"  C/N0:                  {self.cn0_dbhz:>8.1f} dB-Hz",
            f"  Data Rate:             {self.data_rate_bps:>8.0f} bps ({self.data_rate_bps / 1e6:.1f} Mbps)",
            f"  Eb/N0:                 {self.eb_n0_db:>8.1f} dB",
            f"  Eb/N0 Required:        {self.eb_n0_required_db:>8.1f} dB",
            f"  LINK MARGIN:           {self.link_margin_db:>8.1f} dB",
            f"  Link Status:           {'CLOSED' if self.link_closed else 'NOT CLOSED'}",
            "",
            f"  Achievable Rate (3dB): {self.achievable_data_rate_bps:>8.0f} bps"
            f" ({self.achievable_data_rate_bps / 1e6:.1f} Mbps)",
            "=" * 60,
        ]
        return '\n'.join(lines)


def _db(x: float) -> float:
    """Convert linear to dB."""
    return 10 * math.log10(max(x, 1e-30))


def _dbw_to_w(dbw: float) -> float:
    """Convert dBW to Watts."""
    return 10 ** (dbw / 10)


def free_space_loss_db(frequency_hz: float, distance_km: float) -> float:
    """Calculate free space path loss in dB.

    FSPL = (4 * pi * d / lambda)^2  =>  dB = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    """
    distance_m = distance_km * 1000
    wavelength = SPEED_OF_LIGHT / frequency_hz
    fspl = (4 * math.pi * distance_m / wavelength) ** 2
    return _db(fspl)


def slant_range_km(altitude_km: float, elevation_deg: float) -> float:
    """Calculate slant range from ground station to satellite.

    Uses spherical Earth geometry.
    """
    Re = EARTH_RADIUS_KM
    h = altitude_km
    el = math.radians(elevation_deg)

    # From geometry: d = -Re*sin(el) + sqrt((Re*sin(el))^2 + 2*Re*h + h^2)
    d = -Re * math.sin(el) + math.sqrt(
        (Re * math.sin(el)) ** 2 + 2 * Re * h + h ** 2
    )
    return d


def calculate_link_budget(
    direction: str = "downlink",
    band: str = "x_band",
    altitude_km: float = 550.0,
    elevation_deg: float = 10.0,
    data_rate_bps: float = 100e6,
    tx_power_dbw: Optional[float] = None,
    tx_antenna_gain_dbi: Optional[float] = None,
    rx_antenna_gain_dbi: Optional[float] = None,
    system_noise_temp_k: float = 300.0,
    pointing_loss_db: float = 1.0,
    eb_n0_required_db: float = 4.0,
    include_rain: bool = True,
) -> LinkBudgetResult:
    """Calculate uplink or downlink link budget.

    Per ECSS-E-ST-50C communications standard.

    Args:
        direction: "uplink" or "downlink"
        band: Frequency band key ("s_band", "x_band", "ka_band", "optical")
        altitude_km: Satellite orbital altitude
        elevation_deg: Ground station elevation angle
        data_rate_bps: Target data rate in bits/second
        tx_power_dbw: Transmitter power (dBW), uses band default if None
        tx_antenna_gain_dbi: TX antenna gain (dBi), uses band default if None
        rx_antenna_gain_dbi: RX antenna gain (dBi), uses band default if None
        system_noise_temp_k: System noise temperature (K)
        pointing_loss_db: Pointing loss (dB)
        eb_n0_required_db: Required Eb/N0 for target BER (dB)
        include_rain: Include rain attenuation
    """
    if band not in FREQUENCY_BANDS:
        raise ValueError(f"Unknown band: {band}. Available: {list(FREQUENCY_BANDS.keys())}")

    fb = FREQUENCY_BANDS[band]
    distance = slant_range_km(altitude_km, elevation_deg)

    # Set defaults based on direction
    if direction == "downlink":
        _tx_power = tx_power_dbw if tx_power_dbw is not None else fb.typical_tx_power_dbw
        _tx_gain = tx_antenna_gain_dbi if tx_antenna_gain_dbi is not None else fb.typical_antenna_gain_dbi
        _rx_gain = rx_antenna_gain_dbi if rx_antenna_gain_dbi is not None else fb.typical_gs_gain_dbi
    elif direction == "uplink":
        _tx_power = tx_power_dbw if tx_power_dbw is not None else 20.0  # GS typically 100 W
        _tx_gain = tx_antenna_gain_dbi if tx_antenna_gain_dbi is not None else fb.typical_gs_gain_dbi
        _rx_gain = rx_antenna_gain_dbi if rx_antenna_gain_dbi is not None else fb.typical_antenna_gain_dbi
    else:
        raise ValueError(f"direction must be 'uplink' or 'downlink', got '{direction}'")

    # EIRP
    eirp = _tx_power + _tx_gain

    # Path losses
    fspl = free_space_loss_db(fb.frequency_hz, distance)
    atm_loss = fb.atmospheric_loss_db
    rain_loss = fb.rain_loss_db if include_rain else 0.0

    # Received power
    received_power = eirp - fspl - atm_loss - rain_loss - pointing_loss_db + _rx_gain

    # Noise
    n0 = _db(BOLTZMANN_K * system_noise_temp_k)  # dBW/Hz

    # C/N0
    cn0 = received_power - n0

    # Eb/N0
    eb_n0 = cn0 - _db(data_rate_bps)

    # Link margin
    margin = eb_n0 - eb_n0_required_db

    # Achievable data rate (at 3 dB margin)
    excess_db = cn0 - eb_n0_required_db - 3.0  # 3 dB target margin
    achievable_rate = 10 ** (excess_db / 10)

    return LinkBudgetResult(
        direction=direction,
        band_name=fb.name,
        frequency_hz=fb.frequency_hz,
        distance_km=round(distance, 1),
        data_rate_bps=data_rate_bps,
        tx_power_dbw=_tx_power,
        tx_antenna_gain_dbi=_tx_gain,
        eirp_dbw=round(eirp, 1),
        free_space_loss_db=round(fspl, 1),
        atmospheric_loss_db=round(atm_loss, 1),
        rain_loss_db=round(rain_loss, 1),
        pointing_loss_db=round(pointing_loss_db, 1),
        rx_antenna_gain_dbi=_rx_gain,
        system_noise_temp_k=system_noise_temp_k,
        noise_spectral_density_dbw_hz=round(n0, 1),
        received_power_dbw=round(received_power, 1),
        cn0_dbhz=round(cn0, 1),
        eb_n0_db=round(eb_n0, 1),
        eb_n0_required_db=eb_n0_required_db,
        link_margin_db=round(margin, 1),
        achievable_data_rate_bps=round(max(0, achievable_rate), 0),
        link_closed=margin >= 0,
    )


def compare_bands(
    altitude_km: float = 550.0,
    elevation_deg: float = 10.0,
    target_margin_db: float = 3.0,
) -> Dict[str, LinkBudgetResult]:
    """Compare link budgets across all frequency bands.

    Uses default parameters for each band to show relative performance.
    """
    results = {}
    for band_key in FREQUENCY_BANDS:
        fb = FREQUENCY_BANDS[band_key]
        # Use a reasonable data rate for each band
        rate_map = {
            "s_band": 1e6,      # 1 Mbps
            "x_band": 100e6,    # 100 Mbps
            "ka_band": 500e6,   # 500 Mbps
            "optical": 1e9,     # 1 Gbps
        }
        rate = rate_map.get(band_key, 100e6)
        result = calculate_link_budget(
            direction="downlink", band=band_key,
            altitude_km=altitude_km, elevation_deg=elevation_deg,
            data_rate_bps=rate,
        )
        results[band_key] = result
    return results


# ============================================================================
# Self-test
# ============================================================================

def _self_test():
    """Test all standards compliance functions."""
    print("=" * 60)
    print("  ORBITAL COMPUTE -- STANDARDS COMPLIANCE SELF-TEST")
    print("=" * 60)

    errors = []

    # --- Power Budget Test ---
    print("\n  [ECSS-20C] Power budget test...")

    budget = create_compute_sat_power_budget("TEST-SAT-1", solar_w=2000, battery_wh=5000)

    assert len(budget.items) > 0, "No items in power budget"
    assert len(budget.modes()) >= 4, f"Expected >= 4 modes, got {budget.modes()}"

    for mode in budget.modes():
        total = budget.total_power_w(mode)
        assert total > 0, f"Zero power in mode {mode}"
        total_m = budget.total_with_margins_w(mode)
        assert total_m > total, f"Margins not applied in mode {mode}"

    ecss_report = budget.format_ecss()
    assert "ECSS-E-ST-20C" in ecss_report
    assert "nominal" in ecss_report.lower() or "NOMINAL" in ecss_report

    print(f"    Modes:                {budget.modes()}")
    for mode in budget.modes():
        total = budget.total_power_w(mode)
        with_margin = budget.total_with_margins_w(mode)
        margin_pct = budget.solar_margin_pct(mode)
        print(f"    {mode:15s}  {total:7.1f}W -> {with_margin:7.1f}W (margin: {margin_pct:+.1f}%)")
    print(f"    Eclipse duration:     {budget.eclipse_duration_hours():.2f} hours")
    print(f"    ECSS report:          OK ({len(ecss_report)} chars)")

    # --- Thermal Budget Test ---
    print("\n  [ECSS-31C] Thermal budget test...")

    thermal = create_compute_sat_thermal_budget("TEST-SAT-1")
    assert len(thermal.items) > 0
    all_ok = thermal.all_within_limits()

    ecss_thermal = thermal.format_ecss()
    assert "ECSS-E-ST-31C" in ecss_thermal

    print(f"    Components:           {len(thermal.items)}")
    print(f"    All within limits:    {all_ok}")
    for item in thermal.items:
        status = "OK" if item.within_limits else "FAIL"
        print(f"    {item.component:20s}  [{item.predicted_min_c:+.0f},{item.predicted_max_c:+.0f}]C"
              f"  margin: [{item.min_margin_c:+.0f},{item.max_margin_c:+.0f}]  {status}")
    print(f"    ECSS report:          OK ({len(ecss_thermal)} chars)")

    # --- Link Budget Test ---
    print("\n  [ECSS-50C] Link budget test...")

    # Test slant range
    dist_10 = slant_range_km(550, 10)
    dist_90 = slant_range_km(550, 90)
    assert dist_90 < dist_10, "Zenith should be closer than horizon"
    assert abs(dist_90 - 550) < 1, f"Zenith distance should be ~altitude, got {dist_90}"
    print(f"    Slant range (10 deg): {dist_10:.1f} km")
    print(f"    Slant range (90 deg): {dist_90:.1f} km")

    # Test FSPL
    fspl_s = free_space_loss_db(2.2e9, 1500)
    fspl_x = free_space_loss_db(8.2e9, 1500)
    assert fspl_x > fspl_s, "Higher frequency should have more FSPL"
    print(f"    FSPL S-band 1500km:   {fspl_s:.1f} dB")
    print(f"    FSPL X-band 1500km:   {fspl_x:.1f} dB")

    # Test full link budgets per band
    print(f"\n    Band comparison (550 km, 10 deg elevation):")
    print(f"    {'Band':<12s} {'Rate':>10s} {'FSPL':>8s} {'Margin':>8s} {'MaxRate':>12s} {'Status':>8s}")
    print(f"    {'-'*60}")

    comparison = compare_bands(altitude_km=550, elevation_deg=10.0)
    for band_key, result in comparison.items():
        rate_str = f"{result.data_rate_bps / 1e6:.0f} Mbps"
        max_rate_str = f"{result.achievable_data_rate_bps / 1e6:.1f} Mbps"
        status = "CLOSED" if result.link_closed else "OPEN"
        print(f"    {result.band_name:<12s} {rate_str:>10s}"
              f" {result.free_space_loss_db:>7.1f} {result.link_margin_db:>+7.1f}"
              f" {max_rate_str:>12s} {status:>8s}")

    # Detailed X-band test
    xband = calculate_link_budget(
        direction="downlink", band="x_band",
        altitude_km=550, elevation_deg=30,
        data_rate_bps=150e6,
    )
    assert xband.eirp_dbw > 0, "EIRP should be positive"
    assert xband.free_space_loss_db > 150, "FSPL should be > 150 dB at X-band"

    report = xband.format_report()
    assert "LINK BUDGET" in report
    print(f"\n    Detailed X-band report ({len(report)} chars): OK")

    # Uplink test
    uplink = calculate_link_budget(
        direction="uplink", band="s_band",
        altitude_km=550, elevation_deg=20,
        data_rate_bps=1e6,
    )
    assert uplink.direction == "uplink"
    print(f"    S-band uplink:        margin={uplink.link_margin_db:+.1f} dB")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    if errors:
        print(f"  FAIL -- {len(errors)} error(s)")
        for e in errors:
            print(f"    {e}")
    else:
        print("  PASS -- all standards tests passed")

        # Print sample ECSS report
        print(f"\n  Sample ECSS-20C Power Budget (first 15 lines):")
        for line in ecss_report.split('\n')[:15]:
            print(f"    {line}")

        print(f"\n  Sample Link Budget Report (first 15 lines):")
        for line in report.split('\n')[:15]:
            print(f"    {line}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
