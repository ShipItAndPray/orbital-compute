#!/usr/bin/env python3
"""Real Starlink TLE Demo — fetch live TLEs and run orbital compute simulation.

Fetches real Starlink satellite TLEs from CelesTrak, runs a 6-hour simulation
with 30 jobs, and compares eclipse patterns between real and synthetic TLEs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone

from orbital_compute.constellations import fetch_real_tles, generate_constellation, CONSTELLATIONS
from orbital_compute.orbit import predict_eclipse_windows
from orbital_compute.power import PowerModel, PowerConfig
from orbital_compute.thermal import ThermalModel, ThermalConfig
from orbital_compute.scheduler import OrbitalScheduler
from orbital_compute.workloads import WorkloadGenerator
from orbital_compute.ground_stations import find_contact_windows, DEFAULT_GROUND_STATIONS


T0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
SIM_HOURS = 6.0
N_JOBS = 30
DT = 60.0  # seconds


def run_eclipse_analysis(satellites, label: str):
    """Analyze eclipse patterns for a set of satellites."""
    print(f"\n{'─' * 60}")
    print(f"  Eclipse Analysis: {label}")
    print(f"{'─' * 60}")

    results = []
    for sat in satellites:
        windows = predict_eclipse_windows(sat, T0, SIM_HOURS, step_seconds=60)
        total_eclipse_min = sum((e - s).total_seconds() / 60.0 for s, e in windows)
        results.append({
            "name": sat.name,
            "n_eclipses": len(windows),
            "total_eclipse_min": total_eclipse_min,
            "windows": windows,
        })

    print(f"\n  {'Satellite':<30} {'Eclipses':>9} {'Total min':>10} {'Avg min':>9}")
    print(f"  {'─' * 60}")
    for r in results:
        avg = r["total_eclipse_min"] / r["n_eclipses"] if r["n_eclipses"] > 0 else 0
        print(f"  {r['name']:<30} {r['n_eclipses']:>9} {r['total_eclipse_min']:>9.1f} {avg:>9.1f}")

    total = sum(r["total_eclipse_min"] for r in results)
    n = len(results)
    print(f"\n  Fleet average: {total / n:.1f} min eclipse per sat over {SIM_HOURS}h")
    return results


def run_simulation(satellites, jobs, label: str):
    """Run a simulation and return results."""
    print(f"\n{'═' * 60}")
    print(f"  Simulation: {label}")
    print(f"  {len(satellites)} satellites, {len(jobs)} jobs, {SIM_HOURS}h")
    print(f"{'═' * 60}")

    scheduler = OrbitalScheduler()
    scheduler.submit_jobs(list(jobs))  # copy to avoid mutation issues

    # Set up per-satellite models
    power_cfg = PowerConfig()
    thermal_cfg = ThermalConfig()
    nodes = []
    for sat in satellites:
        nodes.append({
            "sat": sat,
            "power": PowerModel(power_cfg),
            "thermal": ThermalModel(thermal_cfg),
            "compute_steps": 0,
            "eclipse_steps": 0,
            "total_steps": 0,
        })

    total_steps = int(SIM_HOURS * 3600 / DT)
    report_every = max(1, total_steps // 10)

    for step in range(total_steps):
        t = T0 + timedelta(seconds=step * DT)

        for node in nodes:
            sat = node["sat"]
            pos = sat.position_at(t)

            current_job = scheduler.running_jobs.get(sat.name)
            compute_w = current_job.power_watts if current_job else 0.0
            heat_w = current_job.heat_output_watts if current_job else 0.0

            pstate = node["power"].step(DT, pos.in_eclipse, compute_w)
            total_heat = node["power"].config.housekeeping_watts * 0.8 + heat_w
            tstate = node["thermal"].step(DT, total_heat, pos.in_eclipse)

            decision = scheduler.decide(
                satellite_name=sat.name,
                timestamp=t,
                power_available_w=pstate.available_for_compute_w,
                battery_pct=pstate.battery_pct,
                thermal_can_compute=tstate.can_compute,
                thermal_throttle=tstate.throttle_pct,
                in_eclipse=pos.in_eclipse,
            )

            if decision.action == "run" and decision.job:
                scheduler.advance_job(sat.name, DT, tstate.throttle_pct, t)
                node["compute_steps"] += 1
            if pos.in_eclipse:
                node["eclipse_steps"] += 1
            node["total_steps"] += 1

        if step % report_every == 0:
            stats = scheduler.stats()
            pct = step / total_steps * 100
            print(f"  [{pct:5.1f}%] t={t.strftime('%H:%M')} | "
                  f"done={stats['completed']}/{stats['total_jobs']}, "
                  f"running={stats['running']}, queued={stats['queued']}")

    # Final report
    stats = scheduler.stats()
    print(f"\n  Final Results:")
    print(f"    Jobs completed: {stats['completed']} / {stats['total_jobs']}")
    print(f"    Preemptions:    {stats['preempted']}")
    print(f"    Charge steps:   {stats['charge_steps']}")

    print(f"\n  Per-Satellite:")
    print(f"  {'Satellite':<30} {'Compute%':>9} {'Eclipse%':>9}")
    print(f"  {'─' * 50}")
    for node in nodes:
        ts = max(node["total_steps"], 1)
        cpct = node["compute_steps"] / ts * 100
        epct = node["eclipse_steps"] / ts * 100
        print(f"  {node['sat'].name:<30} {cpct:>8.1f}% {epct:>8.1f}%")

    return stats


def main():
    print("=" * 60)
    print("  ORBITAL COMPUTE — REAL STARLINK TLE DEMO")
    print("=" * 60)

    # ── 1. Fetch real Starlink TLEs ──
    print("\n[1] Fetching real Starlink TLEs from CelesTrak...")
    try:
        real_sats = fetch_real_tles("starlink", max_sats=20)
    except Exception as e:
        print(f"\n  ERROR: Could not fetch real TLEs: {e}")
        print("  Falling back to synthetic constellation.")
        real_sats = None

    # ── 2. Generate synthetic for comparison ──
    print("\n[2] Generating synthetic Starlink-like constellation...")
    synth_cfg = CONSTELLATIONS["starlink-mini"]
    synth_sats = generate_constellation(synth_cfg, max_sats=20)
    print(f"  Generated {len(synth_sats)} synthetic satellites")

    # ── 3. Show which real satellites we got ──
    if real_sats:
        print(f"\n[3] Real Starlink satellites loaded:")
        for i, sat in enumerate(real_sats):
            pos = sat.position_at(T0)
            print(f"  {i+1:>3}. {sat.name:<35} alt={pos.altitude_km:.0f}km "
                  f"lat={pos.lat_deg:+6.1f} lon={pos.lon_deg:+7.1f} "
                  f"{'ECLIPSE' if pos.in_eclipse else 'sunlit'}")

    # ── 4. Eclipse analysis ──
    print("\n[4] Comparing eclipse patterns...")
    if real_sats:
        real_eclipses = run_eclipse_analysis(real_sats[:10], "Real Starlink (first 10)")
    synth_eclipses = run_eclipse_analysis(synth_sats[:10], "Synthetic Starlink-mini (first 10)")

    if real_sats:
        real_avg = sum(r["total_eclipse_min"] for r in real_eclipses) / len(real_eclipses)
        synth_avg = sum(r["total_eclipse_min"] for r in synth_eclipses) / len(synth_eclipses)
        print(f"\n  Comparison:")
        print(f"    Real avg eclipse:      {real_avg:.1f} min / {SIM_HOURS}h")
        print(f"    Synthetic avg eclipse:  {synth_avg:.1f} min / {SIM_HOURS}h")
        diff_pct = abs(real_avg - synth_avg) / max(real_avg, 1) * 100
        print(f"    Difference:            {diff_pct:.1f}%")

    # ── 5. Run simulation with real TLEs ──
    print(f"\n[5] Generating {N_JOBS} mixed workloads...")
    gen = WorkloadGenerator(seed=42)
    jobs = gen.generate_batch(N_JOBS, T0, duration_hours=SIM_HOURS)
    summary = gen.summary(jobs)
    print(f"  Compute: {summary['total_compute_hours']} hours")
    print(f"  Energy:  {summary['total_energy_kwh']} kWh")
    print(f"  Workload mix: {summary['by_workload']}")

    if real_sats:
        real_stats = run_simulation(real_sats, jobs, "Real Starlink TLEs")

    # Re-generate jobs (IDs are sequential, need fresh copies for second run)
    jobs2 = gen.generate_batch(N_JOBS, T0, duration_hours=SIM_HOURS)
    synth_stats = run_simulation(synth_sats, jobs2, "Synthetic TLEs")

    # ── 6. Summary comparison ──
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    if real_sats:
        print(f"  {'Metric':<25} {'Real':>12} {'Synthetic':>12}")
        print(f"  {'─' * 50}")
        print(f"  {'Jobs completed':<25} {real_stats['completed']:>12} {synth_stats['completed']:>12}")
        print(f"  {'Preemptions':<25} {real_stats['preempted']:>12} {synth_stats['preempted']:>12}")
        print(f"  {'Charge steps':<25} {real_stats['charge_steps']:>12} {synth_stats['charge_steps']:>12}")
    else:
        print(f"  (Real TLE fetch failed — showing synthetic only)")
        print(f"  Jobs completed: {synth_stats['completed']}")
        print(f"  Preemptions:    {synth_stats['preempted']}")

    print(f"\n{'=' * 60}")
    print(f"  DEMO COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
