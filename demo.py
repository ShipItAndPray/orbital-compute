#!/usr/bin/env python3
"""Full-stack demo: Constellation + Realistic Workloads + ISL + Radiation + Scheduling.

This is the complete orbital compute simulation with all subsystems active:
- 12 satellites in a Starcloud-like constellation
- Realistic mixed workloads (Earth obs, AI inference, defense)
- Inter-satellite link mesh for data routing
- Radiation fault injection with checkpoint-restart
- Look-ahead scheduler with eclipse forecasting
- Ground station contact windows
- Web dashboard output
"""

import json
import sys
from datetime import datetime, timedelta, timezone

from orbital_compute.orbit import predict_eclipse_windows
from orbital_compute.power import PowerModel, PowerConfig
from orbital_compute.thermal import ThermalModel, ThermalConfig
from orbital_compute.scheduler_v2 import LookAheadScheduler
from orbital_compute.simulator import Simulation, SimulationConfig, SatelliteNode
from orbital_compute.constellations import generate_constellation, CONSTELLATIONS
from orbital_compute.workloads import WorkloadGenerator, WORKLOAD_CATALOG
from orbital_compute.isl import InterSatelliteNetwork
from orbital_compute.radiation import RadiationModel, RecoveryStrategy


def main():
    print("=" * 70)
    print("  ORBITAL COMPUTE — FULL STACK DEMO")
    print("  All subsystems active: orbit + power + thermal + ISL + radiation")
    print("=" * 70)

    # Configuration
    start_time = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    sim_hours = 12.0
    dt = 60.0  # seconds
    n_jobs = 80

    # 1. Create constellation
    print("\n[1/7] Creating constellation...")
    config = CONSTELLATIONS["starcloud"]
    sats = generate_constellation(config, max_sats=12)
    print(f"  {len(sats)} satellites in {config.name} configuration")
    print(f"  Altitude: {config.altitude_km} km, Inclination: {config.inclination_deg}°")

    # 2. Setup power and thermal for each satellite
    print("\n[2/7] Initializing power and thermal subsystems...")
    power_cfg = PowerConfig(
        solar_panel_watts=1500,  # Starcloud-2 class
        battery_capacity_wh=4000,
        housekeeping_watts=200,
    )
    thermal_cfg = ThermalConfig(radiator_area_m2=5.0)

    nodes = []
    for sat in sats:
        node = SatelliteNode(
            satellite=sat,
            power=PowerModel(power_cfg),
            thermal=ThermalModel(thermal_cfg),
            name=sat.name,
        )
        nodes.append(node)
    print(f"  Solar: {power_cfg.solar_panel_watts}W, Battery: {power_cfg.battery_capacity_wh}Wh")

    # 3. Generate realistic workloads
    print("\n[3/7] Generating workloads...")
    gen = WorkloadGenerator(seed=42)
    jobs = gen.generate_batch(n_jobs, start_time)
    total_compute_h = sum(j.duration_seconds for j in jobs) / 3600
    total_energy_kwh = sum(j.power_watts * j.duration_seconds / 3600000 for j in jobs)
    print(f"  {n_jobs} jobs: {total_compute_h:.1f} compute-hours, {total_energy_kwh:.1f} kWh")

    by_type = {}
    for j in jobs:
        by_type[j.name.split("-")[0]] = by_type.get(j.name.split("-")[0], 0) + 1
    for wtype, count in sorted(by_type.items()):
        print(f"    {wtype}: {count}")

    # 4. Pre-compute eclipse forecasts
    print("\n[4/7] Computing eclipse forecasts...")
    scheduler = LookAheadScheduler()
    for node in nodes:
        windows = predict_eclipse_windows(node.satellite, start_time, sim_hours)
        scheduler.set_eclipse_forecast(node.name, windows)
    print(f"  Eclipse windows computed for {len(nodes)} satellites")

    # Submit jobs
    scheduler.submit_jobs(jobs)

    # 5. Setup ISL network
    print("\n[5/7] Initializing inter-satellite link network...")
    isl = InterSatelliteNetwork(sats)
    isl.update(start_time)
    print(f"  Active links: {isl.total_links()}")
    print(f"  Avg neighbors: {isl.average_neighbors():.1f}")

    # 6. Setup radiation model
    print("\n[6/7] Initializing radiation fault model...")
    rad = RadiationModel(strategy=RecoveryStrategy.CHECKPOINT_RESTART)
    print(f"  Strategy: {rad.strategy.value}")
    print(f"  Overhead: {rad.overhead_factor():.0%}")

    # 7. Run simulation
    print(f"\n[7/7] Running {sim_hours}h simulation...")
    print("=" * 70)

    total_steps = int(sim_hours * 3600 / dt)
    report_interval = max(1, total_steps // 20)
    isl_update_interval = 300  # Update ISL every 5 minutes

    # Radiation stats
    total_seu = 0
    total_recovered = 0
    total_failed = 0

    for step in range(total_steps):
        current_time = start_time + timedelta(seconds=step * dt)

        # Update ISL periodically
        if step % max(1, isl_update_interval // int(dt)) == 0:
            isl.update(current_time)

        for node in nodes:
            pos = node.satellite.position_at(current_time)

            # Check radiation
            current_job = scheduler.running_jobs.get(node.name)
            if current_job:
                memory_mb = 512  # Typical GPU workload
                upset = rad.check_for_upset(pos.lat_deg, pos.lon_deg, dt, memory_mb)
                if upset:
                    total_seu += 1
                    result = rad.handle_upset(current_job, current_job.checkpointable)
                    if result == "recovered":
                        total_recovered += 1
                    elif result == "failed":
                        total_failed += 1

            # Power
            compute_w = current_job.power_watts if current_job else 0.0
            heat_w = current_job.heat_output_watts if current_job else 0.0
            power_state = node.power.step(dt, pos.in_eclipse, compute_w)

            # Thermal
            total_heat = node.power.config.housekeeping_watts * 0.8 + heat_w
            thermal_state = node.thermal.step(dt, total_heat, pos.in_eclipse)

            # Schedule
            decision = scheduler.decide(
                satellite_name=node.name, timestamp=current_time,
                power_available_w=power_state.available_for_compute_w,
                battery_pct=power_state.battery_pct,
                thermal_can_compute=thermal_state.can_compute,
                thermal_throttle=thermal_state.throttle_pct,
                in_eclipse=pos.in_eclipse,
            )

            # Advance job (with radiation overhead)
            is_computing = False
            if decision.action == "run" and decision.job:
                effective_dt = dt / rad.overhead_factor()
                scheduler.advance_job(node.name, effective_dt,
                                       thermal_state.throttle_pct, current_time)
                is_computing = True

            # Counters
            node.total_steps += 1
            if is_computing:
                node.compute_steps += 1
            if pos.in_eclipse:
                node.eclipse_steps += 1

            # Telemetry
            if node.total_steps % 5 == 0:
                node.power_history.append({
                    "time": current_time.isoformat(),
                    "battery_pct": round(power_state.battery_pct, 3),
                    "solar_w": round(power_state.solar_output_w, 1),
                    "load_w": round(power_state.load_w, 1),
                    "computing": is_computing,
                    "in_eclipse": pos.in_eclipse,
                })
                node.thermal_history.append({
                    "time": current_time.isoformat(),
                    "temp_c": round(thermal_state.temp_c, 1),
                    "heat_w": round(thermal_state.heat_generated_w, 1),
                    "radiated_w": round(thermal_state.heat_radiated_w, 1),
                    "throttle": round(thermal_state.throttle_pct, 2),
                })

        # Progress report
        if step % report_interval == 0:
            stats = scheduler.stats()
            eclipse_count = sum(1 for n in nodes
                                if n.power_history and n.power_history[-1].get("in_eclipse", False))
            pct = step / total_steps * 100
            isl.update(current_time)
            print(f"  [{pct:5.1f}%] t={current_time.strftime('%H:%M')} | "
                  f"jobs: {stats['completed']}/{stats['total_jobs']} done, "
                  f"{stats['running']} running | "
                  f"eclipse: {eclipse_count}/{len(nodes)} | "
                  f"ISL: {isl.total_links()} links | "
                  f"SEU: {total_seu}")

    print("=" * 70)

    # Final report
    stats = scheduler.stats()
    total_compute_delivered = sum(j.duration_seconds for j in scheduler.completed_jobs) / 3600
    fleet_util = sum(n.compute_steps for n in nodes) / max(sum(n.total_steps for n in nodes), 1) * 100

    print(f"\n{'=' * 70}")
    print(f"  FULL STACK SIMULATION REPORT")
    print(f"{'=' * 70}")
    print(f"\n  Constellation: {len(nodes)} sats ({config.name})")
    print(f"  Duration: {sim_hours}h")
    print(f"  Jobs: {stats['completed']}/{stats['total_jobs']} completed")
    print(f"  Fleet utilization: {fleet_util:.1f}%")
    print(f"  Compute delivered: {total_compute_delivered:.1f}h")
    print(f"  Preemptions: {stats['preempted']}")

    print(f"\n  Radiation ({rad.strategy.value}):")
    print(f"    Total SEU events: {total_seu}")
    print(f"    Recovered: {total_recovered}")
    print(f"    Failed: {total_failed}")
    print(f"    ECC caught: {rad.stats.caught_by_ecc}")

    isl.update(start_time + timedelta(hours=sim_hours))
    print(f"\n  ISL Network (final state):")
    print(f"    Active links: {isl.total_links()}")
    print(f"    Avg neighbors: {isl.average_neighbors():.1f}")

    print(f"\n  Per-Satellite:")
    print(f"  {'Sat':<12} {'Compute%':>9} {'Eclipse%':>9} {'AvgBatt%':>9} {'MaxTemp':>8}")
    print(f"  {'-'*47}")
    for node in nodes:
        ts = max(node.total_steps, 1)
        comp_pct = node.compute_steps / ts * 100
        ecl_pct = node.eclipse_steps / ts * 100
        avg_batt = sum(h["battery_pct"] for h in node.power_history) / max(len(node.power_history), 1) * 100
        max_temp = max((h["temp_c"] for h in node.thermal_history), default=0)
        print(f"  {node.name:<12} {comp_pct:>8.1f}% {ecl_pct:>8.1f}% {avg_batt:>8.1f}% {max_temp:>7.1f}°C")

    print(f"\n{'=' * 70}")

    # Save results
    results = {
        "config": {
            "constellation": config.name,
            "n_satellites": len(nodes),
            "sim_hours": sim_hours,
            "n_jobs": n_jobs,
            "solar_watts": power_cfg.solar_panel_watts,
            "battery_wh": power_cfg.battery_capacity_wh,
            "radiation_strategy": rad.strategy.value,
        },
        "results": {
            "jobs_completed": stats["completed"],
            "jobs_total": stats["total_jobs"],
            "fleet_utilization_pct": round(fleet_util, 1),
            "compute_hours": round(total_compute_delivered, 1),
            "preemptions": stats["preempted"],
            "seu_events": total_seu,
            "seu_recovered": total_recovered,
            "seu_failed": total_failed,
        },
    }
    with open("examples/full_stack_demo.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to examples/full_stack_demo.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
