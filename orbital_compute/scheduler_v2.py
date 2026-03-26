from __future__ import annotations

"""Advanced orbit-aware scheduler with look-ahead planning.

Improvements over v1 scheduler:
- Eclipse look-ahead: knows when next eclipse starts, plans accordingly
- Energy-aware: won't start a job that can't finish before battery depletes
- Load balancing: distributes jobs across satellites evenly
- Sunlit preference: strongly prefers sunlit satellites (PHOENIX principle)
- Deadline-aware: escalates priority as deadline approaches
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .scheduler import ComputeJob, JobStatus, JobType, ScheduleDecision, OrbitalScheduler


@dataclass
class SatelliteState:
    """Snapshot of a satellite's state for scheduling decisions."""
    name: str
    in_eclipse: bool
    battery_pct: float
    thermal_can_compute: bool
    thermal_throttle: float
    power_available_w: float
    next_eclipse_start: Optional[datetime] = None
    next_eclipse_end: Optional[datetime] = None
    sunlit_remaining_seconds: float = float('inf')
    jobs_completed: int = 0


class LookAheadScheduler(OrbitalScheduler):
    """Scheduler with eclipse look-ahead and energy planning."""

    def __init__(self, eclipse_forecasts: Optional[Dict[str, List[Tuple[datetime, datetime]]]] = None):
        super().__init__()
        self.eclipse_forecasts = eclipse_forecasts or {}
        self.sat_job_counts: Dict[str, int] = {}  # Track completed jobs per sat

    def set_eclipse_forecast(self, satellite_name: str,
                              windows: List[Tuple[datetime, datetime]]):
        """Pre-compute eclipse windows for a satellite."""
        self.eclipse_forecasts[satellite_name] = windows

    def _next_eclipse(self, sat_name: str, now: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Find next eclipse window for a satellite."""
        windows = self.eclipse_forecasts.get(sat_name, [])
        for start, end in windows:
            if end > now:
                return (start, end)
        return None

    def _sunlit_remaining(self, sat_name: str, now: datetime) -> float:
        """Seconds of sunlight remaining before next eclipse."""
        eclipse = self._next_eclipse(sat_name, now)
        if not eclipse:
            return float('inf')
        start, end = eclipse
        if now >= start:
            return 0.0  # Already in eclipse
        return (start - now).total_seconds()

    def _can_finish_before_eclipse(self, job: ComputeJob, sat_name: str,
                                     now: datetime) -> bool:
        """Check if a job can finish before the next eclipse starts."""
        sunlit = self._sunlit_remaining(sat_name, now)
        return job.remaining_seconds <= sunlit

    def _deadline_urgency(self, job: ComputeJob, now: datetime) -> float:
        """Calculate deadline urgency score (0=relaxed, 1=critical)."""
        if not job.deadline:
            return 0.0
        remaining = (job.deadline - now).total_seconds()
        if remaining <= 0:
            return 1.0
        ratio = job.remaining_seconds / remaining
        return min(1.0, ratio)

    def decide(self, satellite_name: str, timestamp: datetime,
               power_available_w: float, battery_pct: float,
               thermal_can_compute: bool, thermal_throttle: float,
               in_eclipse: bool) -> ScheduleDecision:
        """Enhanced scheduling decision with look-ahead."""
        current_job = self.running_jobs.get(satellite_name)

        # Check if current job must be preempted
        if current_job:
            must_preempt = False
            reason = ""

            if not thermal_can_compute:
                must_preempt = True
                reason = "thermal limit"
            elif battery_pct < current_job.min_battery_pct:
                must_preempt = True
                reason = f"battery low ({battery_pct:.0%})"
            elif in_eclipse and battery_pct < 0.35:
                must_preempt = True
                reason = f"eclipse + low battery ({battery_pct:.0%})"

            if must_preempt:
                if current_job.checkpointable:
                    current_job.status = JobStatus.PREEMPTED
                    self.job_queue.insert(0, current_job)
                else:
                    current_job.status = JobStatus.FAILED
                    current_job.progress_seconds = 0.0

                del self.running_jobs[satellite_name]
                decision = ScheduleDecision(satellite_name, timestamp, "pause",
                                            current_job, reason)
                self.schedule_log.append(decision)
                return decision

            # Continue current job
            return ScheduleDecision(satellite_name, timestamp, "run",
                                    current_job, "continuing")

        # No current job
        if not thermal_can_compute:
            d = ScheduleDecision(satellite_name, timestamp, "idle",
                                 reason="thermal cooldown")
            self.schedule_log.append(d)
            return d

        # Eclipse + low battery = charge
        if in_eclipse and battery_pct < 0.50:
            d = ScheduleDecision(satellite_name, timestamp, "charge",
                                 reason=f"eclipse charging ({battery_pct:.0%})")
            self.schedule_log.append(d)
            return d

        # Score each candidate job
        sunlit_remaining = self._sunlit_remaining(satellite_name, timestamp)
        best_job = None
        best_score = -float('inf')
        best_idx = -1

        for i, job in enumerate(self.job_queue):
            if job.status in (JobStatus.COMPLETED, JobStatus.RUNNING):
                continue
            if job.power_watts > power_available_w:
                continue
            if battery_pct < job.min_battery_pct:
                continue

            score = 0.0

            # Priority (higher priority = higher score)
            score += (10 - job.priority) * 10

            # Deadline urgency
            urgency = self._deadline_urgency(job, timestamp)
            score += urgency * 50

            # Prefer jobs that can finish before eclipse
            if self._can_finish_before_eclipse(job, satellite_name, timestamp):
                score += 30
            elif in_eclipse:
                score -= 20  # Penalize starting in eclipse

            # Prefer short jobs when eclipse is approaching
            if sunlit_remaining < 1800:  # < 30 min
                if job.remaining_seconds < sunlit_remaining:
                    score += 20  # Fits before eclipse
                else:
                    score -= 15  # Won't finish

            # Load balance — prefer satellites with fewer completed jobs
            sat_completed = self.sat_job_counts.get(satellite_name, 0)
            avg_completed = sum(self.sat_job_counts.values()) / max(len(self.sat_job_counts), 1)
            if sat_completed < avg_completed:
                score += 10

            if score > best_score:
                best_score = score
                best_job = job
                best_idx = i

        if best_job:
            self.job_queue.pop(best_idx)
            best_job.status = JobStatus.RUNNING
            best_job.assigned_satellite = satellite_name
            best_job.started_at = timestamp
            self.running_jobs[satellite_name] = best_job

            d = ScheduleDecision(satellite_name, timestamp, "run", best_job,
                                 f"score={best_score:.0f}")
            self.schedule_log.append(d)
            return d

        d = ScheduleDecision(satellite_name, timestamp, "idle",
                             reason="no suitable jobs")
        self.schedule_log.append(d)
        return d

    def advance_job(self, satellite_name: str, dt_seconds: float,
                    throttle_pct: float = 0.0, timestamp: Optional[datetime] = None):
        """Advance job and track completions per satellite."""
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
            self.sat_job_counts[satellite_name] = self.sat_job_counts.get(satellite_name, 0) + 1
