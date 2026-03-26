"""Run orbital-compute as a module: python -m orbital_compute

Shows a quick interactive demo of all capabilities.
"""
from __future__ import annotations

import sys


def main():
    print("=" * 70)
    print("  ORBITAL COMPUTE SIMULATOR")
    print("  The open-source toolkit for satellite compute constellations")
    print("=" * 70)
    print()
    print("  Commands:")
    print("    python -m orbital_compute sim       — Run a simulation")
    print("    python -m orbital_compute demo      — Full-stack demo (all subsystems)")
    print("    python -m orbital_compute designer  — Auto-design a constellation")
    print("    python -m orbital_compute cost      — Cost analysis")
    print("    python -m orbital_compute pipeline  — Data pipeline comparison")
    print("    python -m orbital_compute reliability — SLA & availability analysis")
    print("    python -m orbital_compute standards — ECSS compliance check")
    print("    python -m orbital_compute formats   — Industry format export")
    print("    python -m orbital_compute network   — ISL mesh analysis")
    print("    python -m orbital_compute k8s       — K8s scheduler demo")
    print("    python -m orbital_compute mission   — Mission planning timeline")
    print("    python -m orbital_compute debris    — Orbital debris risk assessment")
    print("    python -m orbital_compute propulsion — Station-keeping & deorbit planning")
    print("    python -m orbital_compute federated — Federated learning simulation")
    print()
    print("  Web app: https://shipitandpray.github.io/orbital-compute/")
    print("  GitHub:  https://github.com/ShipItAndPray/orbital-compute")
    print()

    if len(sys.argv) < 2:
        # Run a quick demo
        print("  Running quick demo (use a command above for specific features)...\n")
        quick_demo()
        return

    cmd = sys.argv[1]
    commands = {
        "sim": lambda: __import__("run_sim").main(),
        "demo": lambda: exec(open("demo.py").read()),
        "designer": lambda: __import__("orbital_compute.designer", fromlist=["__main__"]),
        "cost": lambda: __import__("orbital_compute.cost_model", fromlist=["__main__"]),
        "pipeline": lambda: __import__("orbital_compute.data_pipeline", fromlist=["__main__"]),
        "reliability": lambda: __import__("orbital_compute.reliability", fromlist=["__main__"]),
        "standards": lambda: __import__("orbital_compute.standards", fromlist=["__main__"]),
        "formats": lambda: __import__("orbital_compute.formats", fromlist=["__main__"]),
        "network": lambda: __import__("orbital_compute.network", fromlist=["__main__"]),
        "k8s": lambda: __import__("orbital_compute.k8s_scheduler", fromlist=["__main__"]),
        "mission": lambda: __import__("orbital_compute.mission_planner", fromlist=["__main__"]),
        "debris": lambda: __import__("orbital_compute.debris", fromlist=["__main__"]),
        "propulsion": lambda: __import__("orbital_compute.propulsion", fromlist=["__main__"]),
        "federated": lambda: __import__("orbital_compute.federated", fromlist=["__main__"]),
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"  Unknown command: {cmd}")
        print(f"  Available: {', '.join(commands.keys())}")


def quick_demo():
    """30-second demo showing the highlights."""
    from orbital_compute.simulator import Simulation, SimulationConfig
    from orbital_compute.data_pipeline import Sensor, OnboardStorage, InOrbitProcessor, DownlinkConfig, DataPipeline
    from orbital_compute.reliability import ReliabilityAnalyzer

    # 1. Quick simulation
    print("[1/3] Running 6-satellite, 3-hour simulation...")
    config = SimulationConfig(n_satellites=6, sim_duration_hours=3, n_jobs=15)
    sim = Simulation(config)
    sim.setup()
    results = sim.run()
    print(f"  Jobs: {results['scheduler']['completed']}/{results['config']['n_jobs']} completed")
    print(f"  Utilization: {results['fleet_utilization_pct']}%")
    print(f"  Compute: {results['total_compute_hours']:.1f} hours delivered")

    # 2. Data pipeline
    print(f"\n[2/3] Data pipeline: in-orbit processing vs raw downlink...")
    sensor = Sensor("EO-Camera", data_rate_mbps=500, duty_cycle_pct=30, compression_ratio=2.0)
    for strategy, process in [("Raw downlink", False), ("In-orbit classification", True)]:
        pipeline = DataPipeline(
            sensor=sensor,
            storage=OnboardStorage(capacity_gb=1000),
            processor=InOrbitProcessor("GPU") if process else None,
            downlink=DownlinkConfig(active_band="x_band"),
            process_in_orbit=process,
            processing_task="image_classification",
        )
        for _ in range(5):
            pipeline.simulate_orbit(5700, 0.35, [(4800, 5400)])
        r = pipeline.report()
        print(f"  {strategy:30s} | Generated: {r['data_generated_gb']:6.1f}GB | "
              f"Downlinked: {r['data_downlinked_gb']:6.1f}GB | Saved: {r['bandwidth_saved_pct']}%")

    # 3. Reliability
    print(f"\n[3/3] Reliability: 12-satellite constellation SLA...")
    rel = ReliabilityAnalyzer(n_satellites=12)
    for min_op in [1, 6, 12]:
        avail = rel.constellation_availability(min_op)
        print(f"  {min_op:2d}/{12} operational needed → {avail.constellation_availability:.4%} availability "
              f"({avail.nines:.1f} nines)")

    print(f"\n{'=' * 70}")
    print(f"  For the full experience, try: python -m orbital_compute demo")
    print(f"  Or visit: https://shipitandpray.github.io/orbital-compute/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
