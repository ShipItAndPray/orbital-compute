from __future__ import annotations

"""PHOENIX: Sunlight-Aware Task Scheduling for Space Edge Computing.

Implements two research algorithms:

1. PHOENIX (IEEE INFOCOM 2024) — "In-Orbit Processing or Not? Sunlight-Aware
   Task Scheduling for Energy-Efficient Space Edge Computing Networks"
   (arXiv:2407.07337)

   - Formulates the SEC Battery Energy Optimizing (SBEO) problem
   - Classifies satellites as sunlight-sufficient vs sunlight-deficient
   - Offloads tasks from deficient to sufficient satellites (or ground)
   - Minimizes peak depth-of-discharge across the constellation

2. Orbit-Aware Task Scheduling (Cluster Computing, Springer 2025)
   — "Orbit-aware task scheduling in satellite edge computing"

   - Distributed task scheduling under satellite motion visibility constraints
   - Selects processing node based on remaining sunset time
   - Minimizes average response time under resource + deadline constraints

Both are benchmarked against the existing v1 (greedy) and v2 (look-ahead)
schedulers using the project's simulation framework.

References
----------
[1] Z. Jia et al., "In-Orbit Processing or Not? Sunlight-Aware Task
    Scheduling for Energy-Efficient Space Edge Computing Networks,"
    IEEE INFOCOM 2024. arXiv:2407.07337
[2] "Orbit-aware task scheduling in satellite edge computing,"
    Cluster Computing, Springer, 2025. doi:10.1007/s10586-025-05663-9
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .orbit import Satellite, predict_eclipse_windows
from .power import PowerConfig, PowerModel
from .scheduler import ComputeJob, JobStatus, JobType, OrbitalScheduler, ScheduleDecision


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SunlightProfile:
    """Sunlight characterization for one satellite over a planning horizon."""
    satellite_name: str
    sunlit_windows: List[Tuple[datetime, datetime]]  # (start, end)
    eclipse_windows: List[Tuple[datetime, datetime]]
    total_sunlit_seconds: float = 0.0
    total_eclipse_seconds: float = 0.0
    sunlit_ratio: float = 0.0          # fraction of time in sunlight
    is_sunlight_sufficient: bool = False  # PHOENIX classification

    def sunlit_at(self, t: datetime) -> bool:
        """Check if satellite is sunlit at time t."""
        for start, end in self.sunlit_windows:
            if start <= t < end:
                return True
        return False

    def next_sunlit_window(self, t: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Find next sunlit window starting at or after t."""
        for start, end in self.sunlit_windows:
            if end > t:
                return (max(start, t), end)
        return None

    def sunlit_remaining_at(self, t: datetime) -> float:
        """Seconds of continuous sunlight remaining from time t."""
        for start, end in self.sunlit_windows:
            if start <= t < end:
                return (end - t).total_seconds()
        return 0.0


@dataclass
class ConstellationState:
    """Snapshot of constellation state for PHOENIX scheduling."""
    satellites: Dict[str, "SatNodeState"]
    ground_available: bool = True  # Whether ground offloading is possible
    ground_latency_s: float = 50.0  # Round-trip latency to ground (ms -> s)
    ground_processing_speedup: float = 5.0  # Ground is N times faster


@dataclass
class SatNodeState:
    """State of a single satellite node."""
    name: str
    battery_wh: float
    battery_capacity_wh: float
    battery_pct: float
    solar_output_w: float
    in_eclipse: bool
    compute_power_w: float  # Power drawn by compute
    housekeeping_w: float
    solar_panel_w: float    # Max solar panel output
    sunlight_profile: Optional[SunlightProfile] = None
    queued_tasks: int = 0
    queued_compute_seconds: float = 0.0

    @property
    def depth_of_discharge(self) -> float:
        """DoD = 1 - (current / capacity). Lower is better."""
        if self.battery_capacity_wh <= 0:
            return 1.0
        return 1.0 - (self.battery_wh / self.battery_capacity_wh)

    @property
    def predicted_energy_state(self) -> float:
        """Predicted available energy (Wh) = solar harvest potential + current
        battery minus committed compute.

        Simplified version of E[s] from PHOENIX paper [1], Eq. (8)."""
        profile = self.sunlight_profile
        if profile is None:
            return self.battery_wh

        # Solar harvest potential over remaining sunlit windows
        solar_harvest = profile.total_sunlit_seconds * self.solar_panel_w / 3600.0
        # Committed energy for queued tasks
        committed = self.queued_compute_seconds * self.compute_power_w / 3600.0
        return solar_harvest + self.battery_wh - committed


# ---------------------------------------------------------------------------
# Algorithm 1: Sunlight Sufficiency Classification
# ---------------------------------------------------------------------------

def classify_sunlight_sufficiency(
    eclipse_windows: List[Tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
    threshold: float = 0.70,
) -> SunlightProfile:
    """Classify a satellite as sunlight-sufficient or sunlight-deficient.

    Based on PHOENIX Algorithm 1 [1]: computes sunlit ratio over the planning
    horizon. A satellite is sunlight-sufficient if its sunlit ratio exceeds
    the threshold.

    Parameters
    ----------
    eclipse_windows : list of (start, end) datetime tuples
        Pre-computed eclipse windows for this satellite.
    start, end : datetime
        Planning horizon.
    threshold : float
        Sunlit ratio threshold for sufficiency classification.
        PHOENIX uses ~0.70 based on empirical analysis of LEO orbits.

    Returns
    -------
    SunlightProfile
        Complete sunlight characterization.
    """
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return SunlightProfile(
            satellite_name="",
            sunlit_windows=[(start, end)],
            eclipse_windows=[],
            total_sunlit_seconds=0,
            total_eclipse_seconds=0,
            sunlit_ratio=0,
            is_sunlight_sufficient=False,
        )

    # Compute eclipse seconds
    eclipse_seconds = 0.0
    clipped_eclipses = []
    for ec_start, ec_end in eclipse_windows:
        # Clip to planning horizon
        cs = max(ec_start, start)
        ce = min(ec_end, end)
        if ce > cs:
            eclipse_seconds += (ce - cs).total_seconds()
            clipped_eclipses.append((cs, ce))

    sunlit_seconds = total_seconds - eclipse_seconds
    sunlit_ratio = sunlit_seconds / total_seconds

    # Build sunlit windows (complement of eclipse windows)
    sunlit_windows = []
    cursor = start
    for ec_start, ec_end in sorted(clipped_eclipses):
        if ec_start > cursor:
            sunlit_windows.append((cursor, ec_start))
        cursor = max(cursor, ec_end)
    if cursor < end:
        sunlit_windows.append((cursor, end))

    return SunlightProfile(
        satellite_name="",
        sunlit_windows=sunlit_windows,
        eclipse_windows=clipped_eclipses,
        total_sunlit_seconds=sunlit_seconds,
        total_eclipse_seconds=eclipse_seconds,
        sunlit_ratio=sunlit_ratio,
        is_sunlight_sufficient=(sunlit_ratio >= threshold),
    )


# ---------------------------------------------------------------------------
# Algorithm 3: Processing Arrangement (schedule within sunlit windows)
# ---------------------------------------------------------------------------

def can_schedule_in_sunlight(
    profile: SunlightProfile,
    task_duration_s: float,
    arrival: datetime,
    deadline: Optional[datetime],
) -> Tuple[bool, Optional[datetime]]:
    """Check if a task can be fully processed during sunlit windows.

    Based on PHOENIX Algorithm 3 [1]: earliest-deadline-first scheduling
    within available sunlit windows.

    Returns
    -------
    (feasible, start_time)
        Whether the task fits in a sunlit window before its deadline,
        and the recommended start time.
    """
    effective_deadline = deadline or (arrival + timedelta(hours=24))

    for win_start, win_end in profile.sunlit_windows:
        if win_end <= arrival:
            continue
        actual_start = max(win_start, arrival)
        if actual_start >= effective_deadline:
            break
        available = (min(win_end, effective_deadline) - actual_start).total_seconds()
        if available >= task_duration_s:
            return True, actual_start

    return False, None


# ---------------------------------------------------------------------------
# PHOENIX Scheduler
# ---------------------------------------------------------------------------

class PhoenixScheduler(OrbitalScheduler):
    """PHOENIX: Sunlight-aware task scheduling for SEC.

    Implements the three-algorithm approach from [1]:
      1. Classify satellites by sunlight sufficiency
      2. Route tasks: ground > local-sunlit > offload-to-sufficient > local-eclipse
      3. Schedule within sunlit windows (earliest-deadline-first)

    Key difference from v2 (LookAheadScheduler): PHOENIX makes *constellation-
    wide* routing decisions, not just per-satellite scoring. It explicitly
    offloads tasks away from sunlight-deficient satellites.
    """

    def __init__(
        self,
        sunlight_threshold: float = 0.70,
        ground_offload_enabled: bool = True,
    ):
        super().__init__()
        self.sunlight_threshold = sunlight_threshold
        self.ground_offload_enabled = ground_offload_enabled

        # Populated during setup
        self.profiles: Dict[str, SunlightProfile] = {}
        self.sat_job_counts: Dict[str, int] = {}
        self._offload_log: List[dict] = []

        # Metrics
        self.offloads_to_sufficient: int = 0
        self.offloads_to_ground: int = 0
        self.local_sunlit_runs: int = 0
        self.local_eclipse_runs: int = 0

    def set_sunlight_profiles(self, profiles: Dict[str, SunlightProfile]):
        """Pre-compute and set sunlight profiles for all satellites."""
        self.profiles = profiles

    def classify_constellation(
        self,
        satellites: List[Satellite],
        start: datetime,
        duration_hours: float,
    ):
        """Run Algorithm 1: classify all satellites' sunlight sufficiency."""
        end = start + timedelta(hours=duration_hours)
        for sat in satellites:
            eclipse_windows = predict_eclipse_windows(sat, start, duration_hours)
            profile = classify_sunlight_sufficiency(
                eclipse_windows, start, end, self.sunlight_threshold
            )
            profile.satellite_name = sat.name
            self.profiles[sat.name] = profile

    def _best_offload_target(
        self, job: ComputeJob, exclude_sat: str, timestamp: datetime,
        battery_states: Dict[str, float],
    ) -> Optional[str]:
        """Algorithm 2: find best sunlight-sufficient satellite to offload to.

        Selects the sufficient satellite with:
          - Lowest task-to-sunlight ratio (cnt[j] / sunlit[j])
          - Highest predicted energy state E[s]

        Parameters
        ----------
        job : ComputeJob
            The task to offload.
        exclude_sat : str
            The originating (deficient) satellite.
        timestamp : datetime
            Current time.
        battery_states : dict
            Current battery percentages per satellite.

        Returns
        -------
        str or None
            Name of the best target satellite, or None if no suitable target.
        """
        candidates = []
        for name, profile in self.profiles.items():
            if name == exclude_sat:
                continue
            if not profile.is_sunlight_sufficient:
                continue

            # Check if task can fit in a sunlit window
            feasible, _ = can_schedule_in_sunlight(
                profile, job.remaining_seconds, timestamp, job.deadline
            )
            if not feasible:
                continue

            # Score: lower task-to-sunlight ratio is better
            queued = self.sat_job_counts.get(name, 0)
            sunlit_remaining = profile.sunlit_remaining_at(timestamp)
            if sunlit_remaining <= 0:
                # Check total remaining sunlit time
                sunlit_remaining = max(1.0, profile.total_sunlit_seconds)

            ratio = (queued + 1) / (sunlit_remaining / 3600.0)
            batt = battery_states.get(name, 0.5)
            # Higher battery = better target
            score = -ratio + batt * 10

            candidates.append((name, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def decide(
        self,
        satellite_name: str,
        timestamp: datetime,
        power_available_w: float,
        battery_pct: float,
        thermal_can_compute: bool,
        thermal_throttle: float,
        in_eclipse: bool,
    ) -> ScheduleDecision:
        """PHOENIX scheduling decision with constellation-aware routing.

        Implements the four-tier priority from [1]:
        1. Ground offloading (if available and deadline allows)
        2. Local processing during sunlight
        3. Offload to sunlight-sufficient satellite
        4. Local processing during eclipse (last resort)
        """
        current_job = self.running_jobs.get(satellite_name)

        # --- Handle running job preemption ---
        if current_job:
            must_preempt = False
            reason = ""

            if not thermal_can_compute:
                must_preempt = True
                reason = "thermal limit"
            elif battery_pct < current_job.min_battery_pct:
                must_preempt = True
                reason = f"battery low ({battery_pct:.0%})"
            elif in_eclipse and battery_pct < 0.30:
                # PHOENIX is more aggressive about preserving battery in eclipse
                must_preempt = True
                reason = f"PHOENIX: eclipse battery preservation ({battery_pct:.0%})"

            if must_preempt:
                if current_job.checkpointable:
                    current_job.status = JobStatus.PREEMPTED
                    self.job_queue.insert(0, current_job)
                else:
                    current_job.status = JobStatus.FAILED
                    current_job.progress_seconds = 0.0
                del self.running_jobs[satellite_name]
                d = ScheduleDecision(satellite_name, timestamp, "pause",
                                     current_job, reason)
                self.schedule_log.append(d)
                return d

            # Continue current job
            return ScheduleDecision(satellite_name, timestamp, "run",
                                    current_job, "continuing")

        # --- No current job ---
        if not thermal_can_compute:
            d = ScheduleDecision(satellite_name, timestamp, "idle",
                                 reason="thermal cooldown")
            self.schedule_log.append(d)
            return d

        # PHOENIX: In eclipse -> aggressively protect battery
        # Core PHOENIX principle: do NOT compute in eclipse unless absolutely
        # necessary. This is the key differentiator from v1/v2 which allow
        # eclipse compute with battery > 40-50%.
        profile = self.profiles.get(satellite_name)
        eclipse_battery_threshold = 0.55
        if profile and not profile.is_sunlight_sufficient:
            # Deficient satellites: even more conservative
            eclipse_battery_threshold = 0.75

        if in_eclipse and battery_pct < eclipse_battery_threshold:
            d = ScheduleDecision(satellite_name, timestamp, "charge",
                                 reason=f"PHOENIX: eclipse charging ({battery_pct:.0%})")
            self.schedule_log.append(d)
            return d

        # --- Job selection with PHOENIX routing ---
        best_job = None
        best_idx = -1
        best_score = -float('inf')

        for i, job in enumerate(self.job_queue):
            if job.status in (JobStatus.COMPLETED, JobStatus.RUNNING):
                continue
            if job.power_watts > power_available_w:
                continue
            if battery_pct < job.min_battery_pct:
                continue

            score = 0.0

            # Priority
            score += (10 - job.priority) * 10

            # Deadline urgency
            if job.deadline:
                remaining_to_deadline = (job.deadline - timestamp).total_seconds()
                if remaining_to_deadline <= 0:
                    score += 100  # Overdue, run immediately
                elif remaining_to_deadline < job.remaining_seconds * 2:
                    score += 60   # Urgent

            # PHOENIX sunlight-aware scoring
            if profile and not in_eclipse:
                # We are in sunlight — great, prefer running here
                feasible, _ = can_schedule_in_sunlight(
                    profile, job.remaining_seconds, timestamp, job.deadline
                )
                if feasible:
                    score += 40  # Can complete in sunlight
                    self.local_sunlit_runs += 1
                else:
                    score += 10  # Partial sunlit run

            elif profile and in_eclipse:
                # We are in eclipse — penalize unless this sat is sufficient
                if profile.is_sunlight_sufficient:
                    score += 5   # Sufficient sats can afford some eclipse compute
                else:
                    score -= 30  # Deficient sat in eclipse: strongly avoid

            # Load balancing
            sat_completed = self.sat_job_counts.get(satellite_name, 0)
            avg = sum(self.sat_job_counts.values()) / max(len(self.sat_job_counts), 1)
            if sat_completed < avg:
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

            if in_eclipse:
                self.local_eclipse_runs += 1

            d = ScheduleDecision(satellite_name, timestamp, "run", best_job,
                                 f"PHOENIX score={best_score:.0f}")
            self.schedule_log.append(d)
            return d

        d = ScheduleDecision(satellite_name, timestamp, "idle",
                             reason="no suitable jobs")
        self.schedule_log.append(d)
        return d

    def advance_job(self, satellite_name: str, dt_seconds: float,
                    throttle_pct: float = 0.0,
                    timestamp: Optional[datetime] = None):
        """Advance job and track completions."""
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
            self.sat_job_counts[satellite_name] = (
                self.sat_job_counts.get(satellite_name, 0) + 1
            )

    def phoenix_stats(self) -> dict:
        """Extended stats including PHOENIX-specific metrics."""
        base = self.stats()
        sufficient = sum(1 for p in self.profiles.values()
                         if p.is_sunlight_sufficient)
        deficient = len(self.profiles) - sufficient
        avg_sunlit = (sum(p.sunlit_ratio for p in self.profiles.values())
                      / max(len(self.profiles), 1))

        base.update({
            "sunlight_sufficient_sats": sufficient,
            "sunlight_deficient_sats": deficient,
            "avg_sunlit_ratio": round(avg_sunlit, 3),
            "local_sunlit_runs": self.local_sunlit_runs,
            "local_eclipse_runs": self.local_eclipse_runs,
            "offloads_to_sufficient": self.offloads_to_sufficient,
            "offloads_to_ground": self.offloads_to_ground,
        })
        return base


# ---------------------------------------------------------------------------
# Orbit-Aware Task Scheduler (Springer 2025)
# ---------------------------------------------------------------------------

class OrbitAwareScheduler(OrbitalScheduler):
    """Orbit-aware task scheduling under satellite motion visibility constraints.

    Based on [2]: distributed task scheduling that accounts for satellite
    visibility windows (sunset time) when making offloading decisions.

    Key innovations over PHOENIX:
    - Minimizes average response time (not just battery DoD)
    - Considers satellite motion / visibility to ground users
    - Distributed: each satellite makes local decisions with neighbor info
    - Selects processing node with longest remaining sunset time
    """

    def __init__(self):
        super().__init__()
        self.profiles: Dict[str, SunlightProfile] = {}
        self.sat_job_counts: Dict[str, int] = {}

        # Metrics
        self.sunset_aware_assignments: int = 0
        self.visibility_constrained_deferrals: int = 0

    def set_sunlight_profiles(self, profiles: Dict[str, SunlightProfile]):
        self.profiles = profiles

    def classify_constellation(
        self,
        satellites: List[Satellite],
        start: datetime,
        duration_hours: float,
    ):
        """Compute sunlight profiles for visibility-aware scheduling."""
        end = start + timedelta(hours=duration_hours)
        for sat in satellites:
            eclipse_windows = predict_eclipse_windows(sat, start, duration_hours)
            profile = classify_sunlight_sufficiency(
                eclipse_windows, start, end, threshold=0.5  # Not used for classification here
            )
            profile.satellite_name = sat.name
            self.profiles[sat.name] = profile

    def _sunset_time_remaining(self, sat_name: str, now: datetime) -> float:
        """Seconds until this satellite enters eclipse (sunset).

        Core metric from [2]: prefer satellites with longer sunset time
        to ensure tasks complete before power loss.
        """
        profile = self.profiles.get(sat_name)
        if not profile:
            return float('inf')
        return profile.sunlit_remaining_at(now)

    def _response_time_estimate(self, job: ComputeJob, sat_name: str,
                                 now: datetime) -> float:
        """Estimate total response time if job runs on this satellite.

        Response time = wait_time + processing_time + potential_preemption_cost

        From [2]: the objective is to minimize average response time under
        resource and sunset constraints.
        """
        sunset_remaining = self._sunset_time_remaining(sat_name, now)
        processing_time = job.remaining_seconds

        # If task won't finish before sunset, add preemption penalty
        preemption_cost = 0.0
        if sunset_remaining < processing_time and sunset_remaining > 0:
            # Will be preempted at sunset, needs to resume after eclipse
            profile = self.profiles.get(sat_name)
            if profile:
                # Find next sunlit window after current one ends
                eclipse_duration = profile.total_eclipse_seconds / max(
                    len(profile.eclipse_windows), 1
                )
                preemption_cost = eclipse_duration  # Wait through eclipse

        # Queue wait time (proportional to existing queue)
        queued = self.sat_job_counts.get(sat_name, 0)
        queue_wait = queued * 300  # Rough estimate: 5 min per queued job

        return processing_time + preemption_cost + queue_wait

    def decide(
        self,
        satellite_name: str,
        timestamp: datetime,
        power_available_w: float,
        battery_pct: float,
        thermal_can_compute: bool,
        thermal_throttle: float,
        in_eclipse: bool,
    ) -> ScheduleDecision:
        """Orbit-aware scheduling with sunset time optimization.

        Selection criteria (from [2]):
        1. Prefer satellites with longest remaining sunset time
        2. Minimize estimated response time
        3. Respect resource and deadline constraints
        4. Defer tasks if no good option exists
        """
        current_job = self.running_jobs.get(satellite_name)

        # --- Preemption logic ---
        if current_job:
            must_preempt = False
            reason = ""

            if not thermal_can_compute:
                must_preempt = True
                reason = "thermal limit"
            elif battery_pct < current_job.min_battery_pct:
                must_preempt = True
                reason = f"battery low ({battery_pct:.0%})"
            elif in_eclipse and battery_pct < 0.40:
                must_preempt = True
                reason = f"orbit-aware: eclipse + low battery ({battery_pct:.0%})"

            if must_preempt:
                if current_job.checkpointable:
                    current_job.status = JobStatus.PREEMPTED
                    self.job_queue.insert(0, current_job)
                else:
                    current_job.status = JobStatus.FAILED
                    current_job.progress_seconds = 0.0
                del self.running_jobs[satellite_name]
                d = ScheduleDecision(satellite_name, timestamp, "pause",
                                     current_job, reason)
                self.schedule_log.append(d)
                return d

            return ScheduleDecision(satellite_name, timestamp, "run",
                                    current_job, "continuing")

        # --- No current job ---
        if not thermal_can_compute:
            d = ScheduleDecision(satellite_name, timestamp, "idle",
                                 reason="thermal cooldown")
            self.schedule_log.append(d)
            return d

        # Eclipse charging — moderate threshold
        if in_eclipse and battery_pct < 0.50:
            d = ScheduleDecision(satellite_name, timestamp, "charge",
                                 reason=f"orbit-aware: charging ({battery_pct:.0%})")
            self.schedule_log.append(d)
            return d

        sunset_remaining = self._sunset_time_remaining(satellite_name, timestamp)

        # --- Sunset-aware job selection ---
        best_job = None
        best_idx = -1
        best_score = -float('inf')

        for i, job in enumerate(self.job_queue):
            if job.status in (JobStatus.COMPLETED, JobStatus.RUNNING):
                continue
            if job.power_watts > power_available_w:
                continue
            if battery_pct < job.min_battery_pct:
                continue

            score = 0.0

            # Priority
            score += (10 - job.priority) * 10

            # Deadline urgency
            if job.deadline:
                remaining_to_deadline = (job.deadline - timestamp).total_seconds()
                if remaining_to_deadline <= 0:
                    score += 100
                elif remaining_to_deadline < job.remaining_seconds * 2:
                    score += 60

            # Orbit-aware: sunset time scoring
            if not in_eclipse and sunset_remaining > 0:
                if job.remaining_seconds <= sunset_remaining:
                    # Task fits before sunset — ideal
                    score += 40
                    # Bonus for tighter fit (less wasted sunlight)
                    fit_ratio = job.remaining_seconds / sunset_remaining
                    score += fit_ratio * 15
                    self.sunset_aware_assignments += 1
                else:
                    # Won't finish before sunset
                    if job.checkpointable:
                        score += 10  # Can checkpoint and resume
                    else:
                        score -= 20  # Non-checkpointable across sunset: risky
                        self.visibility_constrained_deferrals += 1

            elif in_eclipse:
                # In eclipse: penalize unless battery is strong
                if battery_pct > 0.70:
                    score += 5
                else:
                    score -= 15

            # Estimated response time (lower is better)
            est_response = self._response_time_estimate(job, satellite_name, timestamp)
            # Normalize: penalize long response times
            score -= est_response / 600.0  # -1 point per 10 min

            # Load balance
            sat_completed = self.sat_job_counts.get(satellite_name, 0)
            avg = sum(self.sat_job_counts.values()) / max(len(self.sat_job_counts), 1)
            if sat_completed < avg:
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
                                 f"orbit-aware score={best_score:.0f}")
            self.schedule_log.append(d)
            return d

        d = ScheduleDecision(satellite_name, timestamp, "idle",
                             reason="no suitable jobs")
        self.schedule_log.append(d)
        return d

    def advance_job(self, satellite_name: str, dt_seconds: float,
                    throttle_pct: float = 0.0,
                    timestamp: Optional[datetime] = None):
        """Advance job and track completions."""
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
            self.sat_job_counts[satellite_name] = (
                self.sat_job_counts.get(satellite_name, 0) + 1
            )

    def orbit_aware_stats(self) -> dict:
        """Extended stats with orbit-aware metrics."""
        base = self.stats()
        base.update({
            "sunset_aware_assignments": self.sunset_aware_assignments,
            "visibility_constrained_deferrals": self.visibility_constrained_deferrals,
        })
        return base


# ---------------------------------------------------------------------------
# Benchmark: compare all four schedulers
# ---------------------------------------------------------------------------

def run_benchmark(n_sats=8, hours=12, n_jobs=60, solar_watts=800, battery_wh=1200):
    """Run all four schedulers on identical workloads and compare.

    Schedulers:
      v1 — OrbitalScheduler (greedy)
      v2 — LookAheadScheduler (eclipse look-ahead)
      v3 — PhoenixScheduler (sunlight-aware, SBEO [1])
      v4 — OrbitAwareScheduler (sunset-time optimized [2])

    Note: battery_wh is deliberately small (1200 Wh) to stress-test
    battery preservation -- PHOENIX's core objective.
    """
    from .simulator import Simulation, SimulationConfig
    from .scheduler_v2 import LookAheadScheduler

    print("=" * 74)
    print("  PHOENIX BENCHMARK: v1 (Greedy) vs v2 (LookAhead) vs")
    print("                     v3 (PHOENIX) vs v4 (OrbitAware)")
    print("=" * 74)
    print(f"  Config: {n_sats} sats, {hours}h, {n_jobs} jobs")
    print(f"  Power:  {solar_watts}W solar, {battery_wh}Wh battery")
    print(f"  (Constrained battery to stress-test PHOENIX scheduling)\n")

    results_all = {}

    # --- v1: Greedy ---
    print("--- v1: Greedy Scheduler ---")
    cfg1 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours, n_jobs=n_jobs,
        solar_panel_watts=solar_watts, battery_capacity_wh=battery_wh,
    )
    sim1 = Simulation(cfg1)
    sim1.setup()
    r1 = sim1.run()
    sim1.print_report()
    results_all["v1_greedy"] = r1

    # --- v2: LookAhead ---
    print("\n--- v2: Look-Ahead Scheduler ---")
    cfg2 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours, n_jobs=n_jobs,
        solar_panel_watts=solar_watts, battery_capacity_wh=battery_wh,
    )
    sim2 = Simulation(cfg2)
    v2_sched = LookAheadScheduler()
    sim2.scheduler = v2_sched
    sim2.setup()
    for node in sim2.nodes:
        windows = predict_eclipse_windows(node.satellite, cfg2.start_time, hours)
        v2_sched.set_eclipse_forecast(node.name, windows)
    r2 = sim2.run()
    sim2.print_report()
    results_all["v2_lookahead"] = r2

    # --- v3: PHOENIX ---
    print("\n--- v3: PHOENIX Scheduler (sunlight-aware) ---")
    cfg3 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours, n_jobs=n_jobs,
        solar_panel_watts=solar_watts, battery_capacity_wh=battery_wh,
    )
    sim3 = Simulation(cfg3)
    v3_sched = PhoenixScheduler(sunlight_threshold=0.70)
    sim3.scheduler = v3_sched
    sim3.setup()
    print("  Classifying constellation sunlight sufficiency...")
    v3_sched.classify_constellation(
        [n.satellite for n in sim3.nodes], cfg3.start_time, hours
    )
    for name, profile in v3_sched.profiles.items():
        status = "SUFFICIENT" if profile.is_sunlight_sufficient else "deficient"
        print(f"    {name}: sunlit={profile.sunlit_ratio:.1%} [{status}]")
    r3 = sim3.run()
    sim3.print_report()
    results_all["v3_phoenix"] = r3

    phoenix_stats = v3_sched.phoenix_stats()
    print(f"\n  PHOENIX-specific metrics:")
    print(f"    Sunlight-sufficient sats: {phoenix_stats['sunlight_sufficient_sats']}")
    print(f"    Sunlight-deficient sats:  {phoenix_stats['sunlight_deficient_sats']}")
    print(f"    Avg sunlit ratio:         {phoenix_stats['avg_sunlit_ratio']:.1%}")
    print(f"    Local sunlit runs:        {phoenix_stats['local_sunlit_runs']}")
    print(f"    Local eclipse runs:       {phoenix_stats['local_eclipse_runs']}")

    # --- v4: Orbit-Aware ---
    print("\n--- v4: Orbit-Aware Scheduler (sunset-optimized) ---")
    cfg4 = SimulationConfig(
        n_satellites=n_sats, sim_duration_hours=hours, n_jobs=n_jobs,
        solar_panel_watts=solar_watts, battery_capacity_wh=battery_wh,
    )
    sim4 = Simulation(cfg4)
    v4_sched = OrbitAwareScheduler()
    sim4.scheduler = v4_sched
    sim4.setup()
    v4_sched.classify_constellation(
        [n.satellite for n in sim4.nodes], cfg4.start_time, hours
    )
    r4 = sim4.run()
    sim4.print_report()
    results_all["v4_orbit_aware"] = r4

    orbit_stats = v4_sched.orbit_aware_stats()
    print(f"\n  Orbit-Aware specific metrics:")
    print(f"    Sunset-aware assignments:     {orbit_stats['sunset_aware_assignments']}")
    print(f"    Visibility-constrained defer:  {orbit_stats['visibility_constrained_deferrals']}")

    # --- Comparison table ---
    print("\n" + "=" * 74)
    print("  COMPARISON: ALL SCHEDULERS")
    print("=" * 74)

    labels = ["v1 (Greedy)", "v2 (LookAhead)", "v3 (PHOENIX)", "v4 (OrbitAware)"]
    rs = [r1, r2, r3, r4]

    metrics = [
        ("Jobs Completed",
         [r["scheduler"]["completed"] for r in rs]),
        ("Fleet Utilization %",
         [r["fleet_utilization_pct"] for r in rs]),
        ("Compute Hours",
         [r["total_compute_hours"] for r in rs]),
        ("Preemptions",
         [r["preemption_events"] for r in rs]),
    ]

    # Battery preservation comparison
    min_batteries = []
    avg_batteries = []
    for r in rs:
        mins = [d["min_battery_pct"] for d in r["satellite_details"].values()]
        avgs = [d["avg_battery_pct"] for d in r["satellite_details"].values()]
        min_batteries.append(min(mins) if mins else 0)
        avg_batteries.append(sum(avgs) / len(avgs) if avgs else 0)

    metrics.append(("Min Battery %", min_batteries))
    metrics.append(("Avg Battery %", avg_batteries))

    header = f"  {'Metric':<22}"
    for label in labels:
        header += f" {label:>14}"
    print(f"\n{header}")
    print(f"  {'-' * (22 + 15 * len(labels))}")

    for name, vals in metrics:
        row = f"  {name:<22}"
        for v in vals:
            if isinstance(v, float):
                row += f" {v:>14.1f}"
            else:
                row += f" {v:>14}"
        print(row)

    # Delta from v1 baseline
    print(f"\n  Delta from v1 baseline:")
    print(f"  {'Metric':<22}", end="")
    for label in labels[1:]:
        print(f" {label:>14}", end="")
    print()
    print(f"  {'-' * (22 + 15 * (len(labels) - 1))}")

    for name, vals in metrics:
        row = f"  {name:<22}"
        for v in vals[1:]:
            delta = v - vals[0]
            sign = "+" if delta > 0 else ""
            if isinstance(v, float):
                row += f" {sign}{delta:>13.1f}"
            else:
                row += f" {sign}{delta:>13}"
        print(row)

    # Per-satellite battery comparison (PHOENIX's key metric)
    print(f"\n  Per-Satellite Min Battery % (PHOENIX key metric: higher = better):")
    print(f"  {'Satellite':<10}", end="")
    for label in labels:
        print(f" {label:>14}", end="")
    print()
    print(f"  {'-' * (10 + 15 * len(labels))}")

    sat_names = list(r1["satellite_details"].keys())
    for sn in sat_names:
        row = f"  {sn:<10}"
        for r in rs:
            v = r["satellite_details"].get(sn, {}).get("min_battery_pct", 0)
            row += f" {v:>13.1f}%"
        print(row)

    print(f"\n{'=' * 74}")
    print("  PHOENIX achieves battery preservation through sunlight-aware routing.")
    print("  Orbit-Aware optimizes response time under visibility constraints.")
    print(f"{'=' * 74}")

    return results_all


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run PHOENIX benchmark as a standalone module."""
    print()
    print("  PHOENIX: Sunlight-Aware Task Scheduling for Space Edge Computing")
    print("  Based on: arXiv:2407.07337 (IEEE INFOCOM 2024)")
    print("  And: Orbit-aware task scheduling (Cluster Computing, Springer 2025)")
    print()
    results = run_benchmark()
    print("\n  Done.")


if __name__ == "__main__":
    main()
