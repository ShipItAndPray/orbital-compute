from __future__ import annotations

"""Unified CLI for orbital-compute.

Usage:
    orbital-compute sim --sats 12 --hours 24 --jobs 100
    orbital-compute demo
    orbital-compute benchmark
    orbital-compute dashboard --sats 8 --hours 12
    orbital-compute cost --sats 12 --utilization 0.3
    orbital-compute api --port 8080
    orbital-compute fetch-tles starlink --max 20
    orbital-compute visualize --sats 12 --hours 2
"""

import argparse
import sys


def cmd_sim(args):
    """Run a simulation."""
    from .simulator import Simulation, SimulationConfig
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
    sim.run()
    sim.print_report()
    if args.output:
        sim.save_results(args.output)


def cmd_demo(args):
    """Run full-stack demo."""
    import subprocess
    subprocess.run([sys.executable, "demo.py"], check=True)


def cmd_benchmark(args):
    """Run scheduler benchmark."""
    import subprocess
    subprocess.run([sys.executable, "benchmark.py"], check=True)


def cmd_dashboard(args):
    """Launch web dashboard."""
    # Import here to avoid circular imports
    import importlib
    dashboard = importlib.import_module("dashboard")
    if args.no_serve:
        from .simulator import Simulation, SimulationConfig
        config = SimulationConfig(
            n_satellites=args.sats,
            sim_duration_hours=args.hours,
            n_jobs=args.jobs,
        )
        sim = Simulation(config)
        sim.setup()
        sim.run()
        sim.print_report()
        data = dashboard.build_dashboard_data(sim)
        import json
        html = dashboard.DASHBOARD_HTML.replace("__DATA__", json.dumps(data))
        with open("dashboard.html", "w") as f:
            f.write(html)
        print("  Dashboard saved to dashboard.html")
    else:
        dashboard.main()


def cmd_cost(args):
    """Run cost analysis."""
    from .cost_model import CostAnalyzer, SatelliteSpec, ConstellationCost
    spec = SatelliteSpec()
    analyzer = CostAnalyzer(
        n_satellites=args.sats,
        spec=spec,
    )
    analyzer.full_analysis(utilization_pct=args.utilization * 100)


def cmd_api(args):
    """Start REST API server."""
    from .api import SimulationServer
    from .simulator import Simulation, SimulationConfig
    config = SimulationConfig(
        n_satellites=args.sats,
        sim_duration_hours=args.hours,
        n_jobs=args.jobs,
    )
    sim = Simulation(config)
    sim.setup()
    sim.run()
    server = SimulationServer(sim, port=args.port)
    server.start(blocking=True)


def cmd_fetch_tles(args):
    """Fetch real TLE data."""
    from .constellations import fetch_real_tles
    sats = fetch_real_tles(args.source, max_sats=args.max)
    for sat in sats:
        print(f"  {sat.name}")


def main():
    parser = argparse.ArgumentParser(
        prog="orbital-compute",
        description="Simulate and schedule compute jobs across satellite constellations",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # sim
    p = sub.add_parser("sim", help="Run simulation")
    p.add_argument("--sats", type=int, default=6)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--step", type=float, default=60.0)
    p.add_argument("--solar-watts", type=float, default=2000.0)
    p.add_argument("--battery-wh", type=float, default=5000.0)
    p.add_argument("--output", type=str, default=None)

    # demo
    sub.add_parser("demo", help="Run full-stack demo")

    # benchmark
    sub.add_parser("benchmark", help="Compare v1 vs v2 scheduler")

    # dashboard
    p = sub.add_parser("dashboard", help="Launch web dashboard")
    p.add_argument("--sats", type=int, default=6)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--no-serve", action="store_true")

    # cost
    p = sub.add_parser("cost", help="Run cost analysis")
    p.add_argument("--sats", type=int, default=12)
    p.add_argument("--utilization", type=float, default=0.3)

    # api
    p = sub.add_parser("api", help="Start REST API")
    p.add_argument("--sats", type=int, default=6)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--port", type=int, default=8080)

    # fetch-tles
    p = sub.add_parser("fetch-tles", help="Fetch real TLE data")
    p.add_argument("source", choices=["starlink", "oneweb", "planet", "spire", "active"])
    p.add_argument("--max", type=int, default=20)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "sim": cmd_sim,
        "demo": cmd_demo,
        "benchmark": cmd_benchmark,
        "dashboard": cmd_dashboard,
        "cost": cmd_cost,
        "api": cmd_api,
        "fetch-tles": cmd_fetch_tles,
    }

    try:
        return commands[args.command](args) or 0
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return 130
    except Exception as e:
        print(f"\n  Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
