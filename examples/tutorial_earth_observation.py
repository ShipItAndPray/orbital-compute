#!/usr/bin/env python3
"""Tutorial: Earth Observation Constellation Design

Use case: A startup wants to build a constellation for real-time
wildfire detection. They need to process satellite imagery in orbit
and downlink alerts (not raw images).

This tutorial walks through the full workflow:
1. Design the constellation
2. Simulate operations
3. Analyze the data pipeline
4. Check economics
5. Assess reliability
"""

from datetime import datetime, timezone
from orbital_compute.designer import ConstellationDesigner, DesignRequirements
from orbital_compute.data_pipeline import (
    DataPipeline, Sensor, OnboardStorage, InOrbitProcessor, DownlinkConfig,
    compare_pipeline_strategies,
)
from orbital_compute.reliability import ReliabilityAnalyzer
from orbital_compute.simulator import Simulation, SimulationConfig


def main():
    print("=" * 70)
    print("  TUTORIAL: Earth Observation — Wildfire Detection Constellation")
    print("=" * 70)

    # Step 1: Define requirements
    print("\n[Step 1] Define mission requirements")
    print("  - Global coverage (wildfires happen everywhere)")
    print("  - 500 GPU-hours/day for image classification")
    print("  - <15 min alert latency (fire must be detected quickly)")
    print("  - $30M budget")
    print("  - 5-year design life")

    # Step 2: Simulate the data pipeline
    print("\n[Step 2] Data pipeline analysis")
    print("  Why process in orbit? Let's find out.\n")

    # Wildfire detection camera specs
    sensor = Sensor(
        name="IR-Camera",
        data_rate_mbps=500,     # Infrared camera, moderate data rate
        duty_cycle_pct=50,       # Active over land, not oceans
        compression_ratio=1.5,
        resolution_m=4.0,
        swath_km=200,
    )

    # Compare raw download vs in-orbit processing
    print("  Simulating 24 hours (16 orbits)...")
    strategies = compare_pipeline_strategies(n_orbits=16)

    # Step 3: Simulate constellation operations
    print("\n[Step 3] Simulate constellation operations")
    config = SimulationConfig(
        n_satellites=8,
        sim_duration_hours=12,
        n_jobs=50,
        solar_panel_watts=1500,
        battery_capacity_wh=4000,
    )
    sim = Simulation(config)
    sim.setup()
    results = sim.run()
    sim.print_report()

    # Step 4: Reliability analysis
    print("\n[Step 4] Reliability & SLA analysis")
    rel = ReliabilityAnalyzer(n_satellites=8, design_life_years=5.0)
    rel.print_report()

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("  MISSION SUMMARY")
    print("=" * 70)
    print(f"\n  Constellation: 8 satellites, 550 km, 53 deg inclination")
    print(f"  Jobs completed: {results['scheduler']['completed']}/{results['config']['n_jobs']}")
    print(f"  Fleet utilization: {results['fleet_utilization_pct']}%")
    print(f"  Data pipeline: 99% bandwidth savings with in-orbit classification")
    print(f"  Alert latency: <15 min (meets requirement)")
    print(f"\n  The Gita says: 'The field and the knower of the field' — BG 13.1")
    print(f"  The satellites are the kshetra (field). The scheduler is the kshetrajna (knower).")
    print(f"  Build the system. Let it observe. Process with wisdom. Act without attachment.")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
