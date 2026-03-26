from __future__ import annotations

"""Main simulation engine — ties orbit, power, thermal, and scheduling together.

Runs a discrete-time simulation of a satellite constellation processing compute jobs.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from .orbit import Satellite, predict_eclipse_windows, starlink_shell_1_sample
from .power import PowerModel, PowerConfig, PowerState
from .thermal import ThermalModel, ThermalConfig, ThermalState
from .scheduler import OrbitalScheduler, ComputeJob, ScheduleDecision, JobType
from .ground_stations import (GroundStation, ContactWindow, find_contact_windows,
                               DEFAULT_GROUND_STATIONS)


@dataclass
class SatelliteNode:
    """A satellite with all subsystems."""
    satellite: Satellite
    power: PowerModel
    thermal: ThermalModel
    name: str

    # Telemetry history
    power_history: list = field(default_factory=list)
    thermal_history: list = field(default_factory=list)
    schedule_history: list = field(default_factory=list)
    contact_windows: list = field(default_factory=list)

    # Counters (not sampled — accurate)
    compute_steps: int = 0
    eclipse_steps: int = 0
    total_steps: int = 0


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    n_satellites: int = 6
    sim_duration_hours: float = 6.0       # How long to simulate
    time_step_seconds: float = 60.0       # Simulation timestep
    start_time: datetime = field(
        default_factory=lambda: datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    )

    # Satellite specs (Starcloud-2 inspired)
    solar_panel_watts: float = 2000.0
    battery_capacity_wh: float = 5000.0
    radiator_area_m2: float = 4.0
    housekeeping_watts: float = 150.0

    # Job generation
    n_jobs: int = 20
    job_power_range: tuple = (200.0, 800.0)   # W per job
    job_duration_range: tuple = (300.0, 3600.0)  # seconds per job


class Simulation:
    """Run a full orbital compute simulation."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.scheduler = OrbitalScheduler()
        self.nodes: list[SatelliteNode] = []
        self.current_time = self.config.start_time
        self.results: dict = {}

    def setup(self):
        """Initialize constellation and job queue."""
        print(f"Setting up {self.config.n_satellites}-satellite constellation...")

        # Create satellites
        satellites = starlink_shell_1_sample(self.config.n_satellites)

        power_cfg = PowerConfig(
            solar_panel_watts=self.config.solar_panel_watts,
            battery_capacity_wh=self.config.battery_capacity_wh,
            housekeeping_watts=self.config.housekeeping_watts,
        )
        thermal_cfg = ThermalConfig(
            radiator_area_m2=self.config.radiator_area_m2,
        )

        for sat in satellites:
            node = SatelliteNode(
                satellite=sat,
                power=PowerModel(power_cfg),
                thermal=ThermalModel(thermal_cfg),
                name=sat.name,
            )
            self.nodes.append(node)

        # Generate jobs
        import random
        random.seed(42)
        for i in range(self.config.n_jobs):
            power = random.uniform(*self.config.job_power_range)
            duration = random.uniform(*self.config.job_duration_range)
            job_type = random.choice([JobType.BATCH, JobType.BATCH, JobType.CHECKPOINT])
            priority = random.randint(1, 8)

            job = ComputeJob(
                job_id=f"JOB-{i:04d}",
                name=f"inference-batch-{i}",
                power_watts=power,
                duration_seconds=duration,
                job_type=job_type,
                priority=priority,
                checkpointable=(job_type == JobType.CHECKPOINT),
            )
            self.scheduler.submit_job(job)

        # Compute ground station contacts
        print("  Computing ground station contacts...")
        for node in self.nodes:
            node.contact_windows = find_contact_windows(
                node.satellite, DEFAULT_GROUND_STATIONS,
                self.config.start_time, self.config.sim_duration_hours,
                step_seconds=60.0,
            )

        total_contacts = sum(len(n.contact_windows) for n in self.nodes)
        print(f"  {len(self.nodes)} satellites initialized")
        print(f"  {self.config.n_jobs} jobs queued")
        print(f"  {len(DEFAULT_GROUND_STATIONS)} ground stations")
        print(f"  {total_contacts} contact windows found")
        print(f"  Simulation: {self.config.sim_duration_hours}h at {self.config.time_step_seconds}s steps")

    def run(self) -> dict:
        """Run the simulation."""
        print(f"\nSimulating {self.config.sim_duration_hours} hours...")
        print("=" * 70)

        dt = self.config.time_step_seconds
        total_steps = int(self.config.sim_duration_hours * 3600 / dt)
        report_interval = max(1, total_steps // 20)

        for step in range(total_steps):
            self.current_time = self.config.start_time + timedelta(seconds=step * dt)

            for node in self.nodes:
                self._step_node(node, dt)

            if step % report_interval == 0:
                stats = self.scheduler.stats()
                eclipse_count = sum(
                    1 for n in self.nodes
                    if n.power_history and n.power_history[-1]["solar_w"] == 0
                )
                pct = step / total_steps * 100
                print(f"  [{pct:5.1f}%] t={self.current_time.strftime('%H:%M')} | "
                      f"jobs: {stats['completed']}/{stats['total_jobs']} done, "
                      f"{stats['running']} running | "
                      f"eclipsed: {eclipse_count}/{len(self.nodes)} sats")

        print("=" * 70)
        self.results = self._compile_results()
        return self.results

    def _step_node(self, node: SatelliteNode, dt: float):
        """Advance one satellite by one timestep."""
        # Get orbital position
        pos = node.satellite.position_at(self.current_time)

        # Get current compute load
        current_job = self.scheduler.running_jobs.get(node.name)
        compute_w = current_job.power_watts if current_job else 0.0
        heat_w = current_job.heat_output_watts if current_job else 0.0

        # Update power
        power_state = node.power.step(dt, pos.in_eclipse, compute_w)

        # Update thermal (housekeeping heat + compute heat)
        total_heat = node.power.config.housekeeping_watts * 0.8 + heat_w
        thermal_state = node.thermal.step(dt, total_heat, pos.in_eclipse)

        # Scheduling decision
        decision = self.scheduler.decide(
            satellite_name=node.name,
            timestamp=self.current_time,
            power_available_w=power_state.available_for_compute_w,
            battery_pct=power_state.battery_pct,
            thermal_can_compute=thermal_state.can_compute,
            thermal_throttle=thermal_state.throttle_pct,
            in_eclipse=pos.in_eclipse,
        )

        # Advance running job
        is_computing = False
        if decision.action == "run" and decision.job:
            self.scheduler.advance_job(node.name, dt, thermal_state.throttle_pct,
                                        self.current_time)
            is_computing = True

        # Accurate counters (every step, no sampling)
        node.total_steps += 1
        if is_computing:
            node.compute_steps += 1
        if pos.in_eclipse:
            node.eclipse_steps += 1

        # Record telemetry (sample every 5 steps to save memory)
        if node.total_steps % 5 == 0:
            node.power_history.append({
                "time": self.current_time.isoformat(),
                "battery_pct": round(power_state.battery_pct, 3),
                "solar_w": round(power_state.solar_output_w, 1),
                "load_w": round(power_state.load_w, 1),
                "computing": is_computing,
                "in_eclipse": pos.in_eclipse,
            })
            node.thermal_history.append({
                "time": self.current_time.isoformat(),
                "temp_c": round(thermal_state.temp_c, 1),
                "heat_w": round(thermal_state.heat_generated_w, 1),
                "radiated_w": round(thermal_state.heat_radiated_w, 1),
                "throttle": round(thermal_state.throttle_pct, 2),
            })

    def _compile_results(self) -> dict:
        """Compile simulation results."""
        stats = self.scheduler.stats()

        # Compute utilization per satellite (using accurate counters)
        sat_utilization = {}
        for node in self.nodes:
            ts = max(node.total_steps, 1)
            sat_utilization[node.name] = {
                "compute_pct": round(node.compute_steps / ts * 100, 1),
                "eclipse_pct": round(node.eclipse_steps / ts * 100, 1),
                "avg_battery_pct": round(sum(h["battery_pct"] for h in node.power_history) / max(len(node.power_history), 1) * 100, 1),
                "avg_temp_c": round(sum(h["temp_c"] for h in node.thermal_history) / max(len(node.thermal_history), 1), 1),
                "max_temp_c": round(max((h["temp_c"] for h in node.thermal_history), default=0), 1),
                "min_battery_pct": round(min((h["battery_pct"] for h in node.power_history), default=0) * 100, 1),
                "contact_windows": len(node.contact_windows),
                "total_contact_minutes": round(sum(w.duration_seconds for w in node.contact_windows) / 60, 1),
            }

        # Job completion stats
        completed_jobs = []
        for job in self.scheduler.completed_jobs:
            completed_jobs.append({
                "job_id": job.job_id,
                "satellite": job.assigned_satellite,
                "duration_s": round(job.duration_seconds, 1),
                "power_w": round(job.power_watts, 1),
            })

        total_compute_seconds = sum(j.duration_seconds for j in self.scheduler.completed_jobs)
        total_possible_seconds = self.config.sim_duration_hours * 3600 * self.config.n_satellites

        return {
            "config": {
                "n_satellites": self.config.n_satellites,
                "sim_hours": self.config.sim_duration_hours,
                "n_jobs": self.config.n_jobs,
            },
            "scheduler": stats,
            "fleet_utilization_pct": round(total_compute_seconds / total_possible_seconds * 100, 1),
            "total_compute_hours": round(total_compute_seconds / 3600, 2),
            "satellite_details": sat_utilization,
            "completed_jobs": completed_jobs,
            "preemption_events": stats["preempted"],
        }

    def print_report(self):
        """Print a human-readable report."""
        r = self.results

        print(f"\n{'=' * 70}")
        print(f"  ORBITAL COMPUTE SIMULATION REPORT")
        print(f"{'=' * 70}")
        print(f"\n  Constellation: {r['config']['n_satellites']} satellites")
        print(f"  Duration: {r['config']['sim_hours']} hours")
        print(f"  Jobs submitted: {r['config']['n_jobs']}")
        print(f"\n  Jobs completed: {r['scheduler']['completed']}")
        print(f"  Jobs still running: {r['scheduler']['running']}")
        print(f"  Jobs queued: {r['scheduler']['queued']}")
        print(f"  Preemption events: {r['preemption_events']}")
        print(f"\n  Fleet utilization: {r['fleet_utilization_pct']}%")
        print(f"  Total compute delivered: {r['total_compute_hours']} hours")

        print(f"\n  Per-Satellite Breakdown:")
        print(f"  {'Satellite':<10} {'Compute%':>9} {'Eclipse%':>9} {'AvgBatt%':>9} {'AvgTemp':>8} {'MaxTemp':>8}")
        print(f"  {'-'*54}")
        for name, details in r['satellite_details'].items():
            print(f"  {name:<10} {details['compute_pct']:>8.1f}% {details['eclipse_pct']:>8.1f}% "
                  f"{details['avg_battery_pct']:>8.1f}% {details['avg_temp_c']:>7.1f}°C {details['max_temp_c']:>7.1f}°C")

        print(f"\n{'=' * 70}")

    def save_results(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Results saved to {path}")
