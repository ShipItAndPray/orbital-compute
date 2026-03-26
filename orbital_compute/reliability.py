from __future__ import annotations

"""Reliability and availability analysis for compute constellations.

Models component failure rates, constellation degradation, and system availability.
Critical for space datacenter companies making SLA commitments.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ComponentReliability:
    """Failure rate data for satellite components."""
    # Mean Time Between Failures (hours)
    gpu_mtbf_hours: float = 50000          # ~5.7 years (space-derated)
    solar_panel_mtbf_hours: float = 200000  # ~23 years (high reliability)
    battery_mtbf_hours: float = 80000       # ~9 years
    reaction_wheel_mtbf_hours: float = 100000
    star_tracker_mtbf_hours: float = 150000
    transponder_mtbf_hours: float = 120000
    onboard_computer_mtbf_hours: float = 100000
    power_regulator_mtbf_hours: float = 150000

    @property
    def satellite_mtbf_hours(self) -> float:
        """Satellite MTBF (series model — any component failure = sat failure)."""
        failure_rates = [
            1 / self.gpu_mtbf_hours,
            1 / self.solar_panel_mtbf_hours,
            1 / self.battery_mtbf_hours,
            1 / self.reaction_wheel_mtbf_hours,
            1 / self.star_tracker_mtbf_hours,
            1 / self.transponder_mtbf_hours,
            1 / self.onboard_computer_mtbf_hours,
            1 / self.power_regulator_mtbf_hours,
        ]
        total_rate = sum(failure_rates)
        return 1 / total_rate if total_rate > 0 else float('inf')

    @property
    def satellite_mtbf_years(self) -> float:
        return self.satellite_mtbf_hours / 8760


@dataclass
class AvailabilityResult:
    """Results of availability analysis."""
    n_satellites: int
    min_operational: int          # Minimum sats needed for service
    single_sat_availability: float  # Probability one sat is working
    constellation_availability: float  # Probability >= min_operational sats working
    expected_operational: float    # Expected number of working sats
    nines: float                  # Number of 9s (e.g., 99.99% = 4 nines)
    annual_downtime_hours: float
    mttr_hours: float             # Mean Time To Restore (launch replacement)
    five_year_survival_pct: float  # % of constellation surviving after 5 years


class ReliabilityAnalyzer:
    """Analyze constellation reliability and availability."""

    def __init__(self, n_satellites: int,
                 component_reliability: Optional[ComponentReliability] = None,
                 design_life_years: float = 5.0,
                 mttr_hours: float = 8760.0):  # 1 year to launch replacement
        self.n_sats = n_satellites
        self.rel = component_reliability or ComponentReliability()
        self.design_life = design_life_years
        self.mttr_hours = mttr_hours

    def single_satellite_availability(self) -> float:
        """Steady-state availability of one satellite (A = MTBF / (MTBF + MTTR))."""
        mtbf = self.rel.satellite_mtbf_hours
        return mtbf / (mtbf + self.mttr_hours)

    def constellation_availability(self, min_operational: int) -> AvailabilityResult:
        """Probability that at least min_operational satellites are working.

        Uses binomial distribution.
        """
        p = self.single_satellite_availability()
        n = self.n_sats
        k_min = min_operational

        # P(X >= k_min) = 1 - P(X < k_min) = 1 - sum(C(n,k) * p^k * (1-p)^(n-k), k=0..k_min-1)
        prob_insufficient = 0.0
        for k in range(k_min):
            binom_coeff = math.comb(n, k)
            prob_insufficient += binom_coeff * (p ** k) * ((1 - p) ** (n - k))

        availability = 1 - prob_insufficient
        expected = n * p

        nines = -math.log10(1 - availability) if availability < 1 else float('inf')
        annual_downtime = (1 - availability) * 8760

        # 5-year survival (exponential model)
        design_hours = self.design_life * 8760
        single_survival = math.exp(-design_hours / self.rel.satellite_mtbf_hours)
        five_year_survival = single_survival * 100  # As percentage

        return AvailabilityResult(
            n_satellites=n,
            min_operational=k_min,
            single_sat_availability=p,
            constellation_availability=availability,
            expected_operational=expected,
            nines=nines,
            annual_downtime_hours=annual_downtime,
            mttr_hours=self.mttr_hours,
            five_year_survival_pct=five_year_survival,
        )

    def degradation_curve(self) -> List[dict]:
        """How service degrades as satellites fail."""
        results = []
        for operational in range(self.n_sats, 0, -1):
            failed = self.n_sats - operational
            capacity_pct = (operational / self.n_sats) * 100

            # Probability of exactly this many failures
            p = self.single_satellite_availability()
            prob = math.comb(self.n_sats, operational) * \
                   (p ** operational) * ((1 - p) ** failed)

            results.append({
                "operational": operational,
                "failed": failed,
                "capacity_pct": round(capacity_pct, 1),
                "probability": prob,
                "cumulative_availability": sum(
                    math.comb(self.n_sats, k) * (p ** k) * ((1 - p) ** (self.n_sats - k))
                    for k in range(operational, self.n_sats + 1)
                ),
            })

        return results

    def sla_analysis(self) -> dict:
        """What SLA can the constellation support?"""
        tiers = {
            "99%": 0.99,
            "99.9%": 0.999,
            "99.99%": 0.9999,
            "99.999%": 0.99999,
        }

        results = {}
        for tier_name, target in tiers.items():
            # Find minimum satellites needed for this SLA
            min_sats = 0
            for k in range(1, self.n_sats + 1):
                avail = self.constellation_availability(k)
                if avail.constellation_availability >= target:
                    min_sats = k
                    break

            if min_sats > 0:
                avail = self.constellation_availability(min_sats)
                spare = self.n_sats - min_sats
                results[tier_name] = {
                    "achievable": True,
                    "min_operational": min_sats,
                    "spare_satellites": spare,
                    "actual_availability": avail.constellation_availability,
                    "annual_downtime_hours": avail.annual_downtime_hours,
                }
            else:
                results[tier_name] = {
                    "achievable": False,
                    "min_operational": self.n_sats,
                    "spare_satellites": 0,
                    "actual_availability": self.constellation_availability(self.n_sats).constellation_availability,
                    "annual_downtime_hours": (1 - self.constellation_availability(self.n_sats).constellation_availability) * 8760,
                }

        return results

    def print_report(self):
        """Print full reliability report."""
        print("=" * 65)
        print("  CONSTELLATION RELIABILITY & AVAILABILITY ANALYSIS")
        print("=" * 65)

        print(f"\n  Fleet: {self.n_sats} satellites")
        print(f"  Design life: {self.design_life} years")
        print(f"  MTTR: {self.mttr_hours/8760:.1f} years (time to launch replacement)")
        print(f"  Satellite MTBF: {self.rel.satellite_mtbf_years:.1f} years")

        # Component breakdown
        print(f"\n  Component MTBF:")
        components = [
            ("GPU", self.rel.gpu_mtbf_hours),
            ("Solar Panel", self.rel.solar_panel_mtbf_hours),
            ("Battery", self.rel.battery_mtbf_hours),
            ("Reaction Wheel", self.rel.reaction_wheel_mtbf_hours),
            ("Star Tracker", self.rel.star_tracker_mtbf_hours),
            ("Transponder", self.rel.transponder_mtbf_hours),
            ("OBC", self.rel.onboard_computer_mtbf_hours),
            ("Power Regulator", self.rel.power_regulator_mtbf_hours),
        ]
        for name, mtbf in components:
            print(f"    {name:<20} {mtbf/8760:.1f} years")

        # Availability at different minimums
        print(f"\n  Constellation Availability:")
        print(f"  {'Min Operational':>16} {'Availability':>13} {'Nines':>7} {'Downtime/yr':>13}")
        print(f"  {'-'*49}")
        for min_op in [1, self.n_sats // 2, self.n_sats - 2, self.n_sats - 1, self.n_sats]:
            if min_op < 1 or min_op > self.n_sats:
                continue
            avail = self.constellation_availability(min_op)
            nines_str = f"{avail.nines:.1f}" if avail.nines < 20 else "~inf"
            dt_str = f"{avail.annual_downtime_hours:.2f}h" if avail.annual_downtime_hours > 0.01 else "<0.01h"
            print(f"  {min_op:>16} {avail.constellation_availability:>12.6%} {nines_str:>7} {dt_str:>13}")

        # SLA analysis
        print(f"\n  SLA Feasibility:")
        sla = self.sla_analysis()
        for tier, data in sla.items():
            status = "YES" if data["achievable"] else "NO"
            spare = data["spare_satellites"]
            print(f"    {tier}: {status} (need {data['min_operational']} operational, {spare} spare)")

        # Degradation
        print(f"\n  Graceful Degradation:")
        curve = self.degradation_curve()
        for entry in curve[:5]:
            print(f"    {entry['operational']}/{self.n_sats} operational → "
                  f"{entry['capacity_pct']}% capacity (P={entry['probability']:.4%})")

        # 5-year survival
        avail_full = self.constellation_availability(1)
        print(f"\n  5-Year Survival: {avail_full.five_year_survival_pct:.1f}% per satellite")
        expected_alive = self.n_sats * avail_full.five_year_survival_pct / 100
        print(f"  Expected satellites alive after 5 years: {expected_alive:.1f}/{self.n_sats}")

        print(f"\n{'=' * 65}")


if __name__ == "__main__":
    # Small constellation
    print("\n--- 12-Satellite Constellation ---")
    analyzer = ReliabilityAnalyzer(n_satellites=12)
    analyzer.print_report()

    # Large constellation with faster replacement
    print("\n\n--- 48-Satellite Constellation (6-month MTTR) ---")
    analyzer2 = ReliabilityAnalyzer(n_satellites=48, mttr_hours=4380)
    analyzer2.print_report()
