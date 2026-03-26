#!/usr/bin/env python3
"""Benchmark: Compare v1 (greedy) vs v2 (look-ahead) scheduler.

Runs identical workloads with both schedulers and compares:
- Job completion rate
- Fleet utilization
- Preemption count
- Time to complete all jobs
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from copy import deepcopy

from orbital_compute.simulator import Simulation, SimulationConfig
from orbital_compute.scheduler import OrbitalScheduler, ComputeJob, JobType
from orbital_compute.scheduler_v2 import LookAheadScheduler
from orbital_compute.orbit import predict_eclipse_windows


def run_benchmark(n_sats=8, hours=12, n_jobs=60, solar_watts=800, battery_wh=2000):
    """Run both schedulers on the same workload."""
    print("=" * 70)
    print("  SCHEDULER BENCHMARK: v1 (Greedy) vs v2 (Look-Ahead)")
    print("=" * 70)
    print(f"  Config: {n_sats} sats, {hours}h, {n_jobs} jobs")
    print(f"  Power: {solar_watts}W solar, {battery_wh}Wh battery")
    print(f"  (Constrained power to stress-test scheduling)\n")

    # Run v1 (greedy)
    print("--- v1: Greedy Scheduler ---")
    config_v1 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours,
        n_jobs=n_jobs, solar_panel_watts=solar_watts,
        battery_capacity_wh=battery_wh,
    )
    sim_v1 = Simulation(config_v1)
    sim_v1.setup()
    results_v1 = sim_v1.run()
    sim_v1.print_report()

    # Run v2 (look-ahead)
    print("\n--- v2: Look-Ahead Scheduler ---")
    config_v2 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours,
        n_jobs=n_jobs, solar_panel_watts=solar_watts,
        battery_capacity_wh=battery_wh,
    )
    sim_v2 = Simulation(config_v2)

    # Replace scheduler with v2 before setup
    v2_scheduler = LookAheadScheduler()
    sim_v2.scheduler = v2_scheduler
    sim_v2.setup()

    # Pre-compute eclipse forecasts for v2
    print("  Pre-computing eclipse forecasts for look-ahead...")
    for node in sim_v2.nodes:
        windows = predict_eclipse_windows(
            node.satellite, config_v2.start_time, config_v2.sim_duration_hours
        )
        v2_scheduler.set_eclipse_forecast(node.name, windows)
    print(f"  Eclipse forecasts ready for {len(sim_v2.nodes)} satellites")

    results_v2 = sim_v2.run()
    sim_v2.print_report()

    # Compare
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    metrics = [
        ("Jobs Completed", results_v1["scheduler"]["completed"], results_v2["scheduler"]["completed"]),
        ("Fleet Utilization %", results_v1["fleet_utilization_pct"], results_v2["fleet_utilization_pct"]),
        ("Compute Hours", results_v1["total_compute_hours"], results_v2["total_compute_hours"]),
        ("Preemptions", results_v1["preemption_events"], results_v2["preemption_events"]),
    ]

    print(f"\n  {'Metric':<25} {'v1 (Greedy)':>12} {'v2 (LookAhead)':>14} {'Delta':>10}")
    print(f"  {'-'*61}")
    for name, v1, v2 in metrics:
        if isinstance(v1, float):
            delta = v2 - v1
            sign = "+" if delta > 0 else ""
            print(f"  {name:<25} {v1:>12.1f} {v2:>14.1f} {sign}{delta:>9.1f}")
        else:
            delta = v2 - v1
            sign = "+" if delta > 0 else ""
            print(f"  {name:<25} {v1:>12} {v2:>14} {sign}{delta:>9}")

    # Per-satellite comparison
    print(f"\n  Per-Satellite Compute Utilization:")
    print(f"  {'Satellite':<10} {'v1':>8} {'v2':>8} {'Delta':>8}")
    print(f"  {'-'*34}")
    for name in results_v1["satellite_details"]:
        v1_pct = results_v1["satellite_details"][name]["compute_pct"]
        v2_pct = results_v2["satellite_details"].get(name, {}).get("compute_pct", 0)
        delta = v2_pct - v1_pct
        sign = "+" if delta > 0 else ""
        print(f"  {name:<10} {v1_pct:>7.1f}% {v2_pct:>7.1f}% {sign}{delta:>6.1f}%")

    print(f"\n{'=' * 70}")

    # Save results
    comparison = {
        "config": {"n_sats": n_sats, "hours": hours, "n_jobs": n_jobs,
                    "solar_watts": solar_watts, "battery_wh": battery_wh},
        "v1_greedy": results_v1,
        "v2_lookahead": results_v2,
        "comparison": {m[0]: {"v1": m[1], "v2": m[2], "delta": m[2] - m[1]} for m in metrics},
    }
    with open("examples/scheduler_benchmark.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("  Results saved to examples/scheduler_benchmark.json")


if __name__ == "__main__":
    run_benchmark()
