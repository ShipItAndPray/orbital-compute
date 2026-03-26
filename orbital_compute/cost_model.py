"""Cost model for orbital compute vs terrestrial data centers.

Models the full economics of deploying and operating GPU compute in orbit,
including launch, hardware, operations, and revenue. Provides comparison
against equivalent terrestrial infrastructure.

Cost assumptions sourced from publicly available industry data (2024-2026).
All costs in USD unless otherwise noted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cost parameter defaults
# ---------------------------------------------------------------------------

@dataclass
class LaunchCosts:
    """Launch vehicle cost parameters."""
    falcon9_rideshare_per_kg: float = 5_000.0   # SpaceX Falcon 9 rideshare to LEO
    starship_per_kg: float = 500.0              # SpaceX Starship (projected)
    electron_per_kg: float = 25_000.0           # Rocket Lab Electron (dedicated small)
    default_vehicle: str = "falcon9"

    def cost_per_kg(self, vehicle: Optional[str] = None) -> float:
        v = (vehicle or self.default_vehicle).lower()
        if v in ("falcon9", "falcon 9"):
            return self.falcon9_rideshare_per_kg
        elif v == "starship":
            return self.starship_per_kg
        elif v == "electron":
            return self.electron_per_kg
        else:
            raise ValueError(f"Unknown launch vehicle: {v}")

    def launch_cost(self, mass_kg: float, vehicle: Optional[str] = None) -> float:
        return mass_kg * self.cost_per_kg(vehicle)


@dataclass
class HardwareCosts:
    """Satellite hardware cost parameters."""
    gpu_unit_cost: float = 30_000.0          # NVIDIA H100 equivalent
    n_gpus: int = 1                          # GPUs per satellite
    solar_panel_per_watt: float = 500.0      # $/W for space-grade panels
    battery_per_kwh: float = 500.0           # $/kWh for space-grade batteries
    radiator_per_m2: float = 10_000.0        # $/m^2 for radiator panels
    bus_base_cost: float = 500_000.0         # Satellite bus / structure
    radiation_hardening_pct: float = 0.30    # +30% on electronics for rad-hard

    # Typical satellite specs (override from SimulationConfig if available)
    solar_panel_watts: float = 2000.0
    battery_capacity_kwh: float = 5.0
    radiator_area_m2: float = 4.0
    satellite_dry_mass_kg: float = 120.0     # Total mass including all above

    def electronics_cost(self) -> float:
        """Cost of electronics before radiation hardening."""
        return self.gpu_unit_cost * self.n_gpus

    def electronics_cost_radhard(self) -> float:
        """Cost of electronics after radiation hardening surcharge."""
        return self.electronics_cost() * (1.0 + self.radiation_hardening_pct)

    def solar_cost(self) -> float:
        return self.solar_panel_watts * self.solar_panel_per_watt

    def battery_cost(self) -> float:
        return self.battery_capacity_kwh * self.battery_per_kwh

    def radiator_cost(self) -> float:
        return self.radiator_area_m2 * self.radiator_per_m2

    def total_hardware_cost(self) -> float:
        return (
            self.electronics_cost_radhard()
            + self.solar_cost()
            + self.battery_cost()
            + self.radiator_cost()
            + self.bus_base_cost
        )


@dataclass
class OperatingCosts:
    """Annual operating cost parameters."""
    ground_station_per_minute: float = 5.0     # Per-minute ground contact
    ground_contact_minutes_per_day: float = 30.0  # Avg daily ground contact
    tracking_telemetry_annual: float = 100_000.0
    insurance_pct_of_hardware: float = 0.05    # 5% of hardware cost / year
    mission_operations_annual: float = 500_000.0

    def ground_station_annual(self) -> float:
        return self.ground_station_per_minute * self.ground_contact_minutes_per_day * 365.0

    def insurance_annual(self, hardware_cost: float) -> float:
        return hardware_cost * self.insurance_pct_of_hardware

    def total_annual(self, hardware_cost: float) -> float:
        return (
            self.ground_station_annual()
            + self.tracking_telemetry_annual
            + self.insurance_annual(hardware_cost)
            + self.mission_operations_annual
        )


@dataclass
class RevenueModel:
    """Revenue pricing parameters."""
    aws_p4d_hourly: float = 32.0             # AWS p4d.24xlarge (8x H100) ~ $32/hr per GPU equiv
    orbit_premium_no_downlink: float = 3.0   # Multiplier for in-orbit processing
    earth_observation_premium: float = 3.0   # Premium for EO data processing
    standard_premium: float = 1.5            # General orbital compute premium

    def revenue_per_gpu_hour(self, tier: str = "standard") -> float:
        t = tier.lower()
        if t == "standard":
            return self.aws_p4d_hourly * self.standard_premium
        elif t in ("no_downlink", "in_orbit"):
            return self.aws_p4d_hourly * self.orbit_premium_no_downlink
        elif t in ("earth_observation", "eo"):
            return self.aws_p4d_hourly * self.earth_observation_premium
        elif t == "aws_equivalent":
            return self.aws_p4d_hourly
        else:
            raise ValueError(f"Unknown pricing tier: {t}")


@dataclass
class TerrestrialComparison:
    """Terrestrial data center cost parameters for comparison."""
    gpu_cost: float = 30_000.0               # Same H100
    server_overhead: float = 15_000.0        # Motherboard, RAM, NVLink, etc.
    rack_cost_per_gpu: float = 2_000.0       # Racking, networking
    power_cost_per_kwh: float = 0.08         # Industrial electricity
    gpu_power_draw_kw: float = 0.7           # H100 TDP
    pue: float = 1.3                         # Power Usage Effectiveness
    cooling_overhead_pct: float = 0.10       # 10% for cooling infra
    facility_cost_per_gpu_annual: float = 5_000.0  # Space, maintenance, staff
    lifetime_years: float = 5.0              # GPU refresh cycle

    def capex_per_gpu(self) -> float:
        return self.gpu_cost + self.server_overhead + self.rack_cost_per_gpu

    def electricity_annual_per_gpu(self) -> float:
        return self.gpu_power_draw_kw * self.pue * 8760.0 * self.power_cost_per_kwh

    def opex_annual_per_gpu(self) -> float:
        return (
            self.electricity_annual_per_gpu()
            + self.facility_cost_per_gpu_annual
        )

    def total_cost_per_gpu_year(self) -> float:
        """Annualized total cost (CAPEX amortized + OPEX)."""
        return self.capex_per_gpu() / self.lifetime_years + self.opex_annual_per_gpu()

    def cost_per_compute_hour(self, utilization_pct: float = 100.0) -> float:
        """Cost per GPU-hour at a given utilization."""
        annual_hours = 8760.0 * (utilization_pct / 100.0)
        if annual_hours == 0:
            return float("inf")
        return self.total_cost_per_gpu_year() / annual_hours


# ---------------------------------------------------------------------------
# Constellation cost calculator
# ---------------------------------------------------------------------------

@dataclass
class ConstellationCostConfig:
    """Full configuration for a constellation cost analysis."""
    n_satellites: int = 6
    mission_lifetime_years: float = 5.0
    launch: LaunchCosts = field(default_factory=LaunchCosts)
    hardware: HardwareCosts = field(default_factory=HardwareCosts)
    operating: OperatingCosts = field(default_factory=OperatingCosts)
    revenue: RevenueModel = field(default_factory=RevenueModel)
    terrestrial: TerrestrialComparison = field(default_factory=TerrestrialComparison)


def calculate_constellation_costs(
    config: Optional[ConstellationCostConfig] = None,
    utilization_pct: float = 50.0,
) -> Dict:
    """Calculate full cost analysis for an orbital compute constellation.

    Args:
        config: Cost configuration (uses defaults if None).
        utilization_pct: Expected compute utilization (0-100).

    Returns:
        Dictionary with CAPEX, OPEX, revenue projections, breakeven, and
        comparison against terrestrial equivalent.
    """
    cfg = config or ConstellationCostConfig()
    n = cfg.n_satellites
    hw = cfg.hardware
    ops = cfg.operating
    rev = cfg.revenue
    lv = cfg.launch

    # --- CAPEX ---
    per_sat_hardware = hw.total_hardware_cost()
    per_sat_launch = lv.launch_cost(hw.satellite_dry_mass_kg)
    per_sat_total = per_sat_hardware + per_sat_launch
    total_capex = per_sat_total * n

    # --- Annual OPEX ---
    # Operating costs scale sub-linearly (shared mission ops)
    per_sat_opex = ops.total_annual(per_sat_hardware)
    # Mission ops are shared across constellation, not per-sat
    shared_ops = ops.mission_operations_annual
    per_sat_variable_opex = per_sat_opex - shared_ops
    total_annual_opex = per_sat_variable_opex * n + shared_ops

    # --- Revenue ---
    gpus_total = n * hw.n_gpus
    annual_gpu_hours = gpus_total * 8760.0 * (utilization_pct / 100.0)

    revenue_by_tier = {}
    for tier in ["standard", "no_downlink", "earth_observation", "aws_equivalent"]:
        rate = rev.revenue_per_gpu_hour(tier)
        revenue_by_tier[tier] = {
            "rate_per_hour": rate,
            "annual_revenue": rate * annual_gpu_hours,
        }

    # --- Cost per compute hour ---
    annual_total_cost = total_capex / cfg.mission_lifetime_years + total_annual_opex
    if annual_gpu_hours > 0:
        cost_per_hour = annual_total_cost / annual_gpu_hours
    else:
        cost_per_hour = float("inf")

    # --- Breakeven utilization ---
    # Find utilization where annual revenue = annual cost at standard pricing
    std_rate = rev.revenue_per_gpu_hour("standard")
    annual_fixed_cost = total_capex / cfg.mission_lifetime_years + total_annual_opex
    # revenue = std_rate * gpus_total * 8760 * (util/100)
    # breakeven: revenue = cost
    if std_rate * gpus_total * 8760.0 > 0:
        breakeven_util_pct = (annual_fixed_cost / (std_rate * gpus_total * 8760.0)) * 100.0
    else:
        breakeven_util_pct = float("inf")

    # --- ROI timeline ---
    std_annual_revenue = revenue_by_tier["standard"]["annual_revenue"]
    annual_profit = std_annual_revenue - total_annual_opex
    if annual_profit > 0:
        payback_years = total_capex / annual_profit
    else:
        payback_years = float("inf")

    # --- Utilization sensitivity ---
    utilization_sweep = {}
    for u in [10, 25, 50, 75, 90, 100]:
        hrs = gpus_total * 8760.0 * (u / 100.0)
        if hrs > 0:
            cph = annual_total_cost / (gpus_total * 8760.0 * (u / 100.0))
        else:
            cph = float("inf")
        rev_std = std_rate * hrs
        profit = rev_std - total_annual_opex - total_capex / cfg.mission_lifetime_years
        utilization_sweep[f"{u}%"] = {
            "cost_per_hour": round(cph, 2),
            "annual_revenue": round(rev_std, 0),
            "annual_profit": round(profit, 0),
            "profitable": profit > 0,
        }

    # --- Terrestrial comparison ---
    terr = cfg.terrestrial
    terr_capex = terr.capex_per_gpu() * gpus_total
    terr_annual_opex = terr.opex_annual_per_gpu() * gpus_total
    terr_cost_per_hour = terr.cost_per_compute_hour(utilization_pct)
    terr_annual_total = terr.total_cost_per_gpu_year() * gpus_total

    return {
        "constellation": {
            "n_satellites": n,
            "gpus_per_satellite": hw.n_gpus,
            "total_gpus": gpus_total,
            "mission_lifetime_years": cfg.mission_lifetime_years,
        },
        "capex": {
            "per_satellite": {
                "hardware": round(per_sat_hardware, 0),
                "launch": round(per_sat_launch, 0),
                "total": round(per_sat_total, 0),
            },
            "hardware_breakdown": {
                "electronics_radhard": round(hw.electronics_cost_radhard(), 0),
                "solar_panels": round(hw.solar_cost(), 0),
                "battery": round(hw.battery_cost(), 0),
                "radiator": round(hw.radiator_cost(), 0),
                "bus_structure": round(hw.bus_base_cost, 0),
            },
            "total_constellation": round(total_capex, 0),
        },
        "opex_annual": {
            "ground_stations": round(ops.ground_station_annual() * n, 0),
            "tracking_telemetry": round(ops.tracking_telemetry_annual, 0),
            "insurance": round(ops.insurance_annual(per_sat_hardware) * n, 0),
            "mission_operations": round(shared_ops, 0),
            "total": round(total_annual_opex, 0),
        },
        "revenue": {
            "utilization_pct": utilization_pct,
            "annual_gpu_hours": round(annual_gpu_hours, 0),
            "tiers": revenue_by_tier,
        },
        "economics": {
            "cost_per_compute_hour": round(cost_per_hour, 2),
            "breakeven_utilization_pct": round(breakeven_util_pct, 1),
            "payback_years_standard_tier": round(payback_years, 2),
            "utilization_sensitivity": utilization_sweep,
        },
        "terrestrial_comparison": {
            "equivalent_gpus": gpus_total,
            "capex": round(terr_capex, 0),
            "annual_opex": round(terr_annual_opex, 0),
            "annual_total_cost": round(terr_annual_total, 0),
            "cost_per_compute_hour": round(terr_cost_per_hour, 2),
            "orbital_cost_multiple": round(cost_per_hour / terr_cost_per_hour, 2) if terr_cost_per_hour > 0 else float("inf"),
        },
    }


# ---------------------------------------------------------------------------
# Integration with simulation results
# ---------------------------------------------------------------------------

def cost_from_sim_results(
    sim_results: Dict,
    config: Optional[ConstellationCostConfig] = None,
) -> Dict:
    """Given simulation output JSON, calculate actual cost metrics.

    Args:
        sim_results: Output from Simulation.run() or loaded JSON.
        config: Cost configuration (uses defaults if None).

    Returns:
        Cost analysis enriched with actual simulation utilization data.
    """
    cfg = config or ConstellationCostConfig()

    # Extract utilization from sim results
    fleet_util = sim_results.get("fleet_utilization_pct", 0.0)
    n_sats = sim_results.get("config", {}).get("n_satellites", cfg.n_satellites)
    compute_hours = sim_results.get("total_compute_hours", 0.0)
    sim_hours = sim_results.get("config", {}).get("sim_hours", 6.0)

    cfg.n_satellites = n_sats
    base_costs = calculate_constellation_costs(cfg, utilization_pct=fleet_util)

    # Revenue at different tiers using actual compute hours delivered
    # Annualize from simulation window
    if sim_hours > 0:
        annualization_factor = 8760.0 / sim_hours
    else:
        annualization_factor = 1.0

    annual_compute_hours = compute_hours * annualization_factor
    rev = cfg.revenue

    actual_revenue = {}
    for tier in ["standard", "no_downlink", "earth_observation", "aws_equivalent"]:
        rate = rev.revenue_per_gpu_hour(tier)
        actual_revenue[tier] = {
            "rate_per_hour": rate,
            "sim_period_revenue": round(rate * compute_hours, 2),
            "annualized_revenue": round(rate * annual_compute_hours, 0),
        }

    base_costs["sim_integration"] = {
        "fleet_utilization_pct": fleet_util,
        "compute_hours_simulated": compute_hours,
        "sim_duration_hours": sim_hours,
        "annualized_compute_hours": round(annual_compute_hours, 0),
        "revenue_by_tier": actual_revenue,
        "note": "Annualized by extrapolating sim period to 8760 hours/year",
    }

    # Per-satellite breakdown if details available
    sat_details = sim_results.get("satellite_details", {})
    if sat_details:
        per_sat_econ = {}
        for name, details in sat_details.items():
            util = details.get("compute_pct", 0.0)
            annual_hrs = 8760.0 * (util / 100.0)
            per_sat_econ[name] = {
                "utilization_pct": util,
                "annual_compute_hours": round(annual_hrs, 0),
                "annual_revenue_standard": round(annual_hrs * rev.revenue_per_gpu_hour("standard"), 0),
            }
        base_costs["sim_integration"]["per_satellite"] = per_sat_econ

    return base_costs


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_cost_report(analysis: Dict) -> None:
    """Print a formatted cost analysis report."""
    c = analysis["constellation"]
    capex = analysis["capex"]
    opex = analysis["opex_annual"]
    econ = analysis["economics"]
    terr = analysis["terrestrial_comparison"]

    print("=" * 72)
    print("  ORBITAL COMPUTE COST MODEL")
    print("=" * 72)

    print(f"\n  Constellation: {c['n_satellites']} satellites, "
          f"{c['gpus_per_satellite']} GPU(s) each = {c['total_gpus']} GPUs total")
    print(f"  Mission lifetime: {c['mission_lifetime_years']} years")

    print(f"\n  CAPEX (per satellite):")
    ps = capex["per_satellite"]
    print(f"    Hardware:  ${ps['hardware']:>12,.0f}")
    print(f"    Launch:    ${ps['launch']:>12,.0f}")
    print(f"    Total:     ${ps['total']:>12,.0f}")

    print(f"\n  Hardware breakdown:")
    hb = capex["hardware_breakdown"]
    for k, v in hb.items():
        label = k.replace("_", " ").title()
        print(f"    {label:<25s} ${v:>12,.0f}")

    print(f"\n  Total constellation CAPEX: ${capex['total_constellation']:>14,.0f}")

    print(f"\n  Annual OPEX:")
    for k, v in opex.items():
        if k == "total":
            continue
        label = k.replace("_", " ").title()
        print(f"    {label:<25s} ${v:>12,.0f}")
    print(f"    {'':─<25s}{'':─>13s}")
    print(f"    {'Total':<25s} ${opex['total']:>12,.0f}")

    print(f"\n  Economics:")
    print(f"    Cost per compute-hour:      ${econ['cost_per_compute_hour']:>10,.2f}")
    print(f"    Breakeven utilization:       {econ['breakeven_utilization_pct']:>9.1f}%")
    payback = econ["payback_years_standard_tier"]
    if payback < 100:
        print(f"    Payback (standard tier):     {payback:>9.1f} years")
    else:
        print(f"    Payback (standard tier):     never (at current utilization)")

    print(f"\n  Utilization Sensitivity:")
    print(f"    {'Util':>6s}  {'Cost/hr':>10s}  {'Annual Rev':>12s}  {'Profit':>12s}  {'OK?':>4s}")
    print(f"    {'─'*50}")
    for pct_label, data in econ["utilization_sensitivity"].items():
        ok = "yes" if data["profitable"] else "NO"
        print(f"    {pct_label:>6s}  ${data['cost_per_hour']:>9,.2f}  "
              f"${data['annual_revenue']:>11,.0f}  ${data['annual_profit']:>11,.0f}  {ok:>4s}")

    print(f"\n  Terrestrial Comparison ({terr['equivalent_gpus']} GPUs):")
    print(f"    Terrestrial CAPEX:          ${terr['capex']:>12,.0f}")
    print(f"    Terrestrial annual OPEX:    ${terr['annual_opex']:>12,.0f}")
    print(f"    Terrestrial cost/hr:        ${terr['cost_per_compute_hour']:>12,.2f}")
    print(f"    Orbital cost multiple:       {terr['orbital_cost_multiple']:>11.1f}x")

    # Simulation integration section
    sim = analysis.get("sim_integration")
    if sim:
        print(f"\n  Simulation-Based Metrics:")
        print(f"    Fleet utilization (sim):     {sim['fleet_utilization_pct']:>9.1f}%")
        print(f"    Compute hours (sim period):  {sim['compute_hours_simulated']:>9.2f}")
        print(f"    Annualized compute hours:    {sim['annualized_compute_hours']:>9,.0f}")
        print(f"\n    Revenue by tier (annualized):")
        for tier, data in sim["revenue_by_tier"].items():
            label = tier.replace("_", " ").title()
            print(f"      {label:<22s} ${data['annualized_revenue']:>12,.0f}/yr  "
                  f"(@ ${data['rate_per_hour']:.0f}/hr)")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# __main__ — sample cost analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    print("\n" + "=" * 72)
    print("  ORBITAL COMPUTE COST MODEL — Sample Analysis")
    print("=" * 72)

    # --- Scenario 1: Falcon 9 launch, 6-satellite constellation ---
    print("\n>>> Scenario 1: Falcon 9 rideshare, 6 satellites, 1 GPU each")
    cfg1 = ConstellationCostConfig(
        n_satellites=6,
        mission_lifetime_years=5.0,
    )
    analysis1 = calculate_constellation_costs(cfg1, utilization_pct=50.0)
    print_cost_report(analysis1)

    # --- Scenario 2: Starship launch, 24-satellite constellation ---
    print("\n\n>>> Scenario 2: Starship launch, 24 satellites, 4 GPUs each")
    cfg2 = ConstellationCostConfig(
        n_satellites=24,
        mission_lifetime_years=5.0,
        launch=LaunchCosts(default_vehicle="starship"),
        hardware=HardwareCosts(
            n_gpus=4,
            solar_panel_watts=8000.0,
            battery_capacity_kwh=20.0,
            radiator_area_m2=16.0,
            satellite_dry_mass_kg=400.0,
        ),
    )
    analysis2 = calculate_constellation_costs(cfg2, utilization_pct=60.0)
    print_cost_report(analysis2)

    # --- Scenario 3: Integration with sim results (if available) ---
    sim_path = os.path.join(os.path.dirname(__file__), "..", "sim_results.json")
    if os.path.exists(sim_path):
        print("\n\n>>> Scenario 3: Cost analysis from simulation results")
        with open(sim_path) as f:
            sim_results = json.load(f)
        analysis3 = cost_from_sim_results(sim_results)
        print_cost_report(analysis3)
    else:
        # Use a synthetic sim result for demo
        print("\n\n>>> Scenario 3: Cost analysis from synthetic sim results")
        synthetic_sim = {
            "config": {"n_satellites": 6, "sim_hours": 6.0, "n_jobs": 20},
            "fleet_utilization_pct": 32.5,
            "total_compute_hours": 5.85,
            "satellite_details": {
                f"SAT-{i}": {
                    "compute_pct": 25.0 + i * 5.0,
                    "eclipse_pct": 35.0,
                }
                for i in range(6)
            },
        }
        analysis3 = cost_from_sim_results(synthetic_sim)
        print_cost_report(analysis3)

    # --- Summary comparison table ---
    print("\n\n" + "=" * 72)
    print("  SCENARIO COMPARISON SUMMARY")
    print("=" * 72)
    scenarios = [
        ("Falcon9 / 6 sat / 1 GPU", analysis1),
        ("Starship / 24 sat / 4 GPU", analysis2),
        ("Sim-based (6 sat)", analysis3),
    ]
    print(f"\n  {'Scenario':<28s} {'CAPEX':>14s} {'OPEX/yr':>14s} {'$/hr':>10s} {'Breakeven':>10s} {'vs Terr':>8s}")
    print(f"  {'─'*84}")
    for label, a in scenarios:
        print(f"  {label:<28s} "
              f"${a['capex']['total_constellation']:>13,.0f} "
              f"${a['opex_annual']['total']:>13,.0f} "
              f"${a['economics']['cost_per_compute_hour']:>9,.2f} "
              f"{a['economics']['breakeven_utilization_pct']:>9.1f}% "
              f"{a['terrestrial_comparison']['orbital_cost_multiple']:>7.1f}x")

    print(f"\n  Note: Terrestrial H100 cost/hr at 50% util: "
          f"${TerrestrialComparison().cost_per_compute_hour(50.0):.2f}")
    print(f"  Note: All orbital costs assume standard pricing tier "
          f"(1.5x AWS p4d rate = ${RevenueModel().revenue_per_gpu_hour('standard'):.0f}/hr)")
    print()
