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
