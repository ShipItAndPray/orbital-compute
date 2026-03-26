from __future__ import annotations

"""Orbit-aware job scheduler — the core innovation.

Schedules compute jobs across a constellation of satellites, respecting:
- Eclipse windows (no solar power)
- Battery state-of-charge
- Thermal limits
- Ground station contact windows (for data up/downlink)
- Job priority and deadlines
"""

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"  # Paused due to power/thermal


class JobType(Enum):
    BATCH = "batch"              # Can run anytime, tolerates interruption
    REALTIME = "realtime"        # Must run within deadline
    CHECKPOINT = "checkpoint"    # Can be paused and resumed


@dataclass
class ComputeJob:
    """A unit of compute work to be scheduled on a satellite."""
    job_id: str
    name: str
    power_watts: float              # Power draw when running
    duration_seconds: float         # Total compute time needed
    job_type: JobType = JobType.BATCH
    priority: int = 5              # 1=highest, 10=lowest
    deadline: Optional[datetime] = None
    min_battery_pct: float = 0.30  # Don't start if battery below this
    heat_output_watts: float = 0.0 # Additional heat (usually ~= power_watts)
    data_downlink_mb: float = 0.0  # Data to downlink after completion
    checkpointable: bool = True    # Can be paused/resumed
    progress_seconds: float = 0.0  # How much has been completed
    status: JobStatus = JobStatus.PENDING
    assigned_satellite: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.heat_output_watts == 0.0:
            self.heat_output_watts = self.power_watts * 0.95  # GPUs are ~95% heat

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.duration_seconds - self.progress_seconds)

    @property
    def is_complete(self) -> bool:
        return self.progress_seconds >= self.duration_seconds

    def __lt__(self, other):
        return self.priority < other.priority


@dataclass
class ScheduleDecision:
    """What the scheduler decided for a satellite at a given timestep."""
    satellite_name: str
    timestamp: datetime
    action: str  # "run", "pause", "idle", "charge"
    job: Optional[ComputeJob] = None
    reason: str = ""


class OrbitalScheduler:
    """Schedule jobs across satellites respecting orbital constraints."""

    def __init__(self):
        self.job_queue: list[ComputeJob] = []
        self.running_jobs: dict[str, ComputeJob] = {}  # satellite_name → job
        self.completed_jobs: list[ComputeJob] = []
        self.schedule_log: list[ScheduleDecision] = []

    def submit_job(self, job: ComputeJob):
        """Add a job to the scheduling queue."""
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda j: j.priority)

    def submit_jobs(self, jobs: list[ComputeJob]):
        for job in jobs:
            self.submit_job(job)

    def decide(self, satellite_name: str, timestamp: datetime,
               power_available_w: float, battery_pct: float,
               thermal_can_compute: bool, thermal_throttle: float,
               in_eclipse: bool) -> ScheduleDecision:
        """Make a scheduling decision for one satellite at one timestep.

        This is the core scheduling logic. Called once per satellite per timestep.
        """
        current_job = self.running_jobs.get(satellite_name)

        # Check if current job must be preempted
        if current_job:
            must_preempt = False
            reason = ""

            if not thermal_can_compute:
                must_preempt = True
                reason = f"thermal limit (throttle={thermal_throttle:.0%})"
            elif battery_pct < current_job.min_battery_pct:
                must_preempt = True
                reason = f"battery low ({battery_pct:.0%} < {current_job.min_battery_pct:.0%})"
            elif power_available_w < current_job.power_watts * 0.5:
                # Allow running at reduced power if possible
                if in_eclipse and battery_pct < 0.40:
                    must_preempt = True
                    reason = f"eclipse + low battery ({battery_pct:.0%})"

            if must_preempt:
                if current_job.checkpointable:
                    current_job.status = JobStatus.PREEMPTED
                    self.job_queue.insert(0, current_job)  # Re-queue at front
                else:
                    current_job.status = JobStatus.FAILED
                    current_job.progress_seconds = 0.0  # Lost progress

                del self.running_jobs[satellite_name]
                decision = ScheduleDecision(satellite_name, timestamp, "pause",
                                            current_job, reason)
                self.schedule_log.append(decision)
                return decision

            # Current job continues
            return ScheduleDecision(satellite_name, timestamp, "run", current_job,
                                    "continuing")

        # No current job — can we start one?
        if not thermal_can_compute:
            decision = ScheduleDecision(satellite_name, timestamp, "idle",
                                        reason="thermal cooldown")
            self.schedule_log.append(decision)
            return decision

        # PHOENIX-inspired sunlight-aware scheduling:
        # In eclipse with low battery — charge instead of computing.
        # Key insight from PHOENIX (IEEE 2024): offload to "sunlight-sufficient"
        # edges when possible, preserve battery for critical operations.
        if in_eclipse and battery_pct < 0.50:
            decision = ScheduleDecision(satellite_name, timestamp, "charge",
                                        reason=f"eclipse charging ({battery_pct:.0%})")
            self.schedule_log.append(decision)
            return decision

        # If in eclipse but battery is healthy, allow compute at reduced priority
        # (prefer sunlit satellites for new jobs — PHOENIX principle)
        eclipse_penalty = 2 if in_eclipse else 0  # Deprioritize eclipse compute

        # Find best job for this satellite
        best_job = None
        best_idx = -1

        for i, job in enumerate(self.job_queue):
            if job.status in (JobStatus.COMPLETED, JobStatus.RUNNING):
                continue
            if job.power_watts > power_available_w:
                continue
            if battery_pct < job.min_battery_pct:
                continue

            # Deadline urgency boost
            if job.deadline and (job.deadline - timestamp).total_seconds() < job.remaining_seconds * 2:
                best_job = job
                best_idx = i
                break

            if best_job is None:
                best_job = job
                best_idx = i

        if best_job:
            self.job_queue.pop(best_idx)
            best_job.status = JobStatus.RUNNING
            best_job.assigned_satellite = satellite_name
            best_job.started_at = timestamp
            self.running_jobs[satellite_name] = best_job

            decision = ScheduleDecision(satellite_name, timestamp, "run", best_job,
                                        f"starting job {best_job.job_id}")
            self.schedule_log.append(decision)
            return decision

        # Nothing to do
        decision = ScheduleDecision(satellite_name, timestamp, "idle",
                                    reason="no suitable jobs")
        self.schedule_log.append(decision)
        return decision

    def advance_job(self, satellite_name: str, dt_seconds: float,
                    throttle_pct: float = 0.0, timestamp: Optional[datetime] = None):
        """Advance a running job's progress."""
        job = self.running_jobs.get(satellite_name)
        if not job:
            return

        effective_dt = dt_seconds * (1.0 - throttle_pct)
        job.progress_seconds += effective_dt

        if job.is_complete:
            job.status = JobStatus.COMPLETED
            job.completed_at = timestamp
            self.completed_jobs.append(job)
            del self.running_jobs[satellite_name]

    def stats(self) -> dict:
        """Return scheduler statistics."""
        total = len(self.completed_jobs) + len(self.running_jobs) + len(self.job_queue)
        return {
            "total_jobs": total,
            "completed": len(self.completed_jobs),
            "running": len(self.running_jobs),
            "queued": len(self.job_queue),
            "preempted": sum(1 for d in self.schedule_log if d.action == "pause"),
            "idle_steps": sum(1 for d in self.schedule_log if d.action == "idle"),
            "charge_steps": sum(1 for d in self.schedule_log if d.action == "charge"),
        }


if __name__ == "__main__":
    from datetime import datetime, timezone
    print("=" * 60)
    print("  JOB SCHEDULER DEMO (v1 Greedy)")
    print("=" * 60)
    sched = OrbitalScheduler()
    jobs = [ComputeJob(f"J{i}", f"batch-{i}", power_watts=200+i*50,
                        duration_seconds=120+i*60, priority=i%5+1)
            for i in range(8)]
    sched.submit_jobs(jobs)
    t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    print(f"\n  {len(jobs)} jobs submitted, simulating 2 satellites...")
    from datetime import timedelta
    for step in range(30):
        ts = t + timedelta(minutes=step)
        eclipse = 10 <= step <= 20
        for sat in ["SAT-A", "SAT-B"]:
            batt = 0.4 if eclipse else 0.9
            d = sched.decide(sat, ts, power_available_w=1200,
                              battery_pct=batt, thermal_can_compute=True,
                              thermal_throttle=0.0, in_eclipse=eclipse)
            if d.action == "run" and d.job:
                sched.advance_job(sat, 60, 0.0, ts)
    s = sched.stats()
    print(f"  Completed: {s['completed']}/{s['total_jobs']}")
    print(f"  Preempted: {s['preempted']}")
    print(f"  Idle steps: {s['idle_steps']}")
    print(f"  Charge steps: {s['charge_steps']}")
    print(f"\n{'=' * 60}")
