"""Power subsystem model — solar panels, battery, and power-aware scheduling.

Models the power budget of a compute satellite:
- Solar panels generate power when in sunlight
- Battery stores energy for eclipse periods
- Compute loads draw power
- Scheduler must respect power constraints
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerConfig:
    """Power subsystem configuration for a compute satellite."""
    solar_panel_watts: float = 2000.0     # Peak solar panel output (W)
    battery_capacity_wh: float = 5000.0   # Battery capacity (Wh)
    battery_initial_pct: float = 0.80     # Initial charge (0-1)
    charge_efficiency: float = 0.90       # Solar → battery efficiency
    discharge_efficiency: float = 0.95    # Battery → load efficiency
    min_battery_pct: float = 0.20         # Never discharge below this (safety margin)
    housekeeping_watts: float = 150.0     # Always-on systems (comms, ADCS, thermal)


@dataclass
class PowerState:
    """Current power state of a satellite."""
    battery_wh: float           # Current energy stored
    battery_pct: float          # Current charge percentage
    solar_output_w: float       # Current solar panel output (0 if eclipse)
    load_w: float               # Current power draw
    net_power_w: float          # Solar - load (negative = draining battery)
    available_for_compute_w: float  # How much power can be allocated to compute
    time_to_empty_hours: float  # Hours until battery hits min (at current draw)
    can_compute: bool           # Whether enough power exists for any compute


class PowerModel:
    """Simulates satellite power subsystem over time."""

    def __init__(self, config: Optional[PowerConfig] = None):
        self.config = config or PowerConfig()
        self.battery_wh = self.config.battery_capacity_wh * self.config.battery_initial_pct

    def step(self, dt_seconds: float, in_eclipse: bool, compute_load_w: float = 0.0) -> PowerState:
        """Advance power model by dt_seconds.

        Args:
            dt_seconds: Time step in seconds
            in_eclipse: Whether satellite is in Earth's shadow
            compute_load_w: Power drawn by compute workload
        """
        dt_hours = dt_seconds / 3600.0

        # Solar input
        solar_w = 0.0 if in_eclipse else self.config.solar_panel_watts

        # Total load
        total_load_w = self.config.housekeeping_watts + compute_load_w

        # Net power
        net_w = solar_w - total_load_w

        if net_w > 0:
            # Charging battery
            charge_wh = net_w * dt_hours * self.config.charge_efficiency
            self.battery_wh = min(self.battery_wh + charge_wh, self.config.battery_capacity_wh)
        else:
            # Draining battery
            drain_wh = abs(net_w) * dt_hours / self.config.discharge_efficiency
            self.battery_wh = max(self.battery_wh - drain_wh, 0.0)

        battery_pct = self.battery_wh / self.config.battery_capacity_wh
        min_wh = self.config.min_battery_pct * self.config.battery_capacity_wh

        # Available compute power
        available_w = solar_w - self.config.housekeeping_watts
        if available_w < 0:
            # In eclipse, compute runs from battery (if above minimum)
            if self.battery_wh > min_wh:
                # Allow compute from battery, but limited
                usable_wh = self.battery_wh - min_wh
                available_w = min(usable_wh / dt_hours * self.config.discharge_efficiency,
                                  self.config.solar_panel_watts - self.config.housekeeping_watts)
            else:
                available_w = 0.0

        # Time to empty
        if net_w < 0 and self.battery_wh > min_wh:
            time_to_empty = (self.battery_wh - min_wh) / (abs(net_w) / self.config.discharge_efficiency)
        else:
            time_to_empty = float('inf')

        can_compute = available_w > 50.0 and battery_pct > self.config.min_battery_pct

        return PowerState(
            battery_wh=self.battery_wh,
            battery_pct=battery_pct,
            solar_output_w=solar_w,
            load_w=total_load_w,
            net_power_w=net_w,
            available_for_compute_w=max(0.0, available_w),
            time_to_empty_hours=time_to_empty,
            can_compute=can_compute,
        )

    def can_sustain_load(self, load_w: float, duration_hours: float, in_eclipse: bool) -> bool:
        """Check if battery can sustain a given load for a duration."""
        total_load = self.config.housekeeping_watts + load_w
        solar = 0.0 if in_eclipse else self.config.solar_panel_watts
        net = solar - total_load

        if net >= 0:
            return True  # Solar covers it

        # Need battery
        drain_wh = abs(net) * duration_hours / self.config.discharge_efficiency
        min_wh = self.config.min_battery_pct * self.config.battery_capacity_wh
        return self.battery_wh - drain_wh >= min_wh


@dataclass
class BatteryAgingModel:
    """Physics-driven battery degradation model.

    Based on arXiv:2603.04372 "Unseen Cost of Space Computing" (2026).
    Models capacity fade from cycle depth, C-rate, and temperature.
    """
    initial_capacity_wh: float = 5000.0
    cycle_count: int = 0
    cumulative_dod: float = 0.0          # Sum of depth-of-discharge per cycle
    capacity_wh: float = 5000.0
    # Degradation parameters (Li-ion, space-grade)
    calendar_fade_pct_per_year: float = 2.0   # Calendar aging
    cycle_fade_pct_per_cycle: float = 0.005   # Per full equivalent cycle
    temp_acceleration_factor: float = 1.0     # 1.0 at 25°C, 2x per 10°C above
    high_c_rate_penalty: float = 1.5          # Extra degradation at high discharge rates

    def cycle(self, dod_pct: float, c_rate: float = 0.5, temp_c: float = 25.0):
        """Record one charge/discharge cycle and update capacity.

        Args:
            dod_pct: Depth of discharge (0-100%)
            c_rate: Discharge rate (C), e.g., 0.5C, 1C, 2C
            temp_c: Battery temperature during cycle
        """
        self.cycle_count += 1
        self.cumulative_dod += dod_pct / 100.0

        # Temperature acceleration (Arrhenius-like)
        temp_factor = 2.0 ** ((temp_c - 25.0) / 10.0) if temp_c > 25 else 1.0

        # C-rate penalty (high discharge = faster degradation)
        c_factor = 1.0 + max(0, c_rate - 0.5) * (self.high_c_rate_penalty - 1.0)

        # Capacity fade this cycle
        equiv_cycles = dod_pct / 100.0  # Partial cycles count proportionally
        fade = self.cycle_fade_pct_per_cycle * equiv_cycles * temp_factor * c_factor / 100.0
        self.capacity_wh *= (1.0 - fade)

    def calendar_age(self, hours: float):
        """Apply calendar aging (independent of cycling)."""
        years = hours / 8760.0
        fade = self.calendar_fade_pct_per_year * years / 100.0
        self.capacity_wh *= (1.0 - fade)

    @property
    def soh_pct(self) -> float:
        """State of Health — remaining capacity as % of initial."""
        return (self.capacity_wh / self.initial_capacity_wh) * 100.0

    @property
    def capacity_fade_pct(self) -> float:
        """Total capacity lost as %."""
        return 100.0 - self.soh_pct

    def predict_eol(self, cycles_per_day: float, avg_dod: float = 30.0,
                     eol_threshold_pct: float = 80.0) -> float:
        """Predict days until battery reaches end-of-life threshold.

        Returns estimated days until SoH drops below threshold.
        """
        # Simple linear extrapolation from current fade rate
        if self.cycle_count < 10:
            fade_per_cycle = self.cycle_fade_pct_per_cycle * (avg_dod / 100.0)
        else:
            fade_per_cycle = self.capacity_fade_pct / max(self.cycle_count, 1)

        remaining_fade = self.soh_pct - eol_threshold_pct
        if remaining_fade <= 0:
            return 0.0
        if fade_per_cycle <= 0:
            return float('inf')

        cycles_to_eol = remaining_fade / fade_per_cycle
        return cycles_to_eol / cycles_per_day


if __name__ == "__main__":
    print("=" * 60)
    print("  POWER SUBSYSTEM DEMO")
    print("=" * 60)

    # Simulate 3 orbits
    pm = PowerModel(PowerConfig(solar_panel_watts=2000, battery_capacity_wh=5000))
    print(f"\n  Solar: {pm.config.solar_panel_watts}W, Battery: {pm.config.battery_capacity_wh}Wh")
    print(f"\n  Simulating 3 LEO orbits (95 min each)...")
    for orbit in range(3):
        # Sunlit phase (60 min)
        for _ in range(60):
            s = pm.step(60, in_eclipse=False, compute_load_w=500)
        sun_batt = s.battery_pct
        # Eclipse phase (35 min)
        for _ in range(35):
            s = pm.step(60, in_eclipse=True, compute_load_w=500)
        ecl_batt = s.battery_pct
        print(f"  Orbit {orbit+1}: sunlit→{sun_batt:.0%} eclipse→{ecl_batt:.0%} "
              f"solar={s.solar_output_w:.0f}W load={s.load_w:.0f}W")

    # Battery aging
    print(f"\n  Battery Aging (5 years, 15 cycles/day, 30% DoD):")
    ba = BatteryAgingModel(initial_capacity_wh=5000)
    for year in range(1, 6):
        for _ in range(365 * 15):
            ba.cycle(dod_pct=30, c_rate=0.5, temp_c=28)
        ba.calendar_age(8760)
        print(f"  Year {year}: SoH={ba.soh_pct:.1f}% capacity={ba.capacity_wh:.0f}Wh")
    print(f"\n{'=' * 60}")
