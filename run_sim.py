#!/usr/bin/env python3
"""Run the Orbital Compute Simulator.

Simulates a constellation of compute satellites processing jobs
while respecting orbital mechanics, power, and thermal constraints.

Usage:
    python run_sim.py                    # Default 6-sat, 6-hour sim
    python run_sim.py --sats 12 --hours 24 --jobs 50
"""

import argparse
import sys
from datetime import datetime, timezone

from orbital_compute.simulator import Simulation, SimulationConfig
from orbital_compute.scheduler import ComputeJob, JobType


def main():
    parser = argparse.ArgumentParser(description="Orbital Compute Simulator")
    parser.add_argument("--sats", type=int, default=6, help="Number of satellites")
    parser.add_argument("--hours", type=float, default=6.0, help="Simulation duration (hours)")
    parser.add_argument("--jobs", type=int, default=20, help="Number of compute jobs")
    parser.add_argument("--step", type=float, default=60.0, help="Time step (seconds)")
    parser.add_argument("--output", type=str, default="sim_results.json", help="Output file")
    parser.add_argument("--solar-watts", type=float, default=2000.0, help="Solar panel output (W)")
    parser.add_argument("--battery-wh", type=float, default=5000.0, help="Battery capacity (Wh)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ORBITAL COMPUTE SIMULATOR v0.1")
    print("  Schedule compute jobs across satellite constellations")
    print("=" * 70)

    config = SimulationConfig(
        n_satellites=args.sats,
        sim_duration_hours=args.hours,
        time_step_seconds=args.step,
        n_jobs=args.jobs,
        solar_panel_watts=args.solar_watts,
        battery_capacity_wh=args.battery_wh,
    )

    sim = Simulation(config)
    sim.setup()

    # Run it
    results = sim.run()
    sim.print_report()
    sim.save_results(args.output)

    # Eclipse analysis
    print("\n  Eclipse Windows (first satellite, first 3 orbits):")
    from orbital_compute.orbit import predict_eclipse_windows
    node = sim.nodes[0]
    windows = predict_eclipse_windows(node.satellite, config.start_time, config.sim_duration_hours)
    for i, (start, end) in enumerate(windows[:6]):
        duration = (end - start).total_seconds() / 60
        print(f"    Eclipse {i+1}: {start.strftime('%H:%M:%S')} → {end.strftime('%H:%M:%S')} ({duration:.1f} min)")

    # Ground station contacts
    total_contacts = sum(len(n.contact_windows) for n in sim.nodes)
    total_contact_min = sum(
        cw.duration_seconds / 60 for n in sim.nodes for cw in n.contact_windows
    )
    print(f"\n  Ground Station Contacts: {total_contacts} passes, {total_contact_min:.0f} min total")

    # Thermal analysis
    print(f"\n  Thermal Summary:")
    from orbital_compute.thermal import ThermalModel, ThermalConfig
    tm = ThermalModel(ThermalConfig(radiator_area_m2=args.sats))
    max_heat = tm.max_sustainable_heat_w(70.0)
    print(f"    Max sustainable compute heat: {max_heat:.0f} W at 70°C equilibrium")
    print(f"    With {config.radiator_area_m2} m² radiator per satellite")

    print(f"\n  Done. Results in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
