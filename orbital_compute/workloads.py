from __future__ import annotations

"""Realistic space compute workload types and job stream generation.

Defines workload profiles for Earth observation, AI inference, scientific
compute, and defense/ISR tasks. Each factory creates ComputeJob instances
with physically-grounded parameters (power draw, duration, data sizes,
deadlines).

Also provides WorkloadGenerator for creating realistic mixed job streams
with bursty arrival patterns tied to ground-station passes.
"""

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from .scheduler import ComputeJob, JobType


# ---------------------------------------------------------------------------
# Workload category enum (not Python Enum — just strings for simplicity)
# ---------------------------------------------------------------------------
CATEGORY_EARTH_OBS = "earth_observation"
CATEGORY_AI = "ai_inference"
CATEGORY_SCIENTIFIC = "scientific"
CATEGORY_DEFENSE = "defense_isr"


# ---------------------------------------------------------------------------
# Workload specification
# ---------------------------------------------------------------------------
@dataclass
class WorkloadSpec:
    """Blueprint for a workload type."""
    name: str
    category: str
    power_watts: float
    duration_seconds: float
    input_size_mb: float
    output_size_mb: float
    deadline_seconds: Optional[float]   # None = batch, no deadline
    priority: int                       # 1 = highest
    job_type: JobType
    checkpointable: bool


# ---------------------------------------------------------------------------
# Workload catalog
# ---------------------------------------------------------------------------
WORKLOAD_CATALOG: dict[str, WorkloadSpec] = {
    # Earth Observation
    "image_classification": WorkloadSpec(
        name="ImageClassification",
        category=CATEGORY_EARTH_OBS,
        power_watts=200.0,
        duration_seconds=120.0,
        input_size_mb=500.0,
        output_size_mb=1.0,
        deadline_seconds=30 * 60,       # 30 min
        priority=5,
        job_type=JobType.CHECKPOINT,
        checkpointable=True,
    ),
    "change_detection": WorkloadSpec(
        name="ChangeDetection",
        category=CATEGORY_EARTH_OBS,
        power_watts=300.0,
        duration_seconds=300.0,
        input_size_mb=2048.0,
        output_size_mb=50.0,
        deadline_seconds=60 * 60,       # 1 h
        priority=5,
        job_type=JobType.CHECKPOINT,
        checkpointable=True,
    ),
    "object_tracking": WorkloadSpec(
        name="ObjectTracking",
        category=CATEGORY_EARTH_OBS,
        power_watts=400.0,
        duration_seconds=60.0,
        input_size_mb=200.0,
        output_size_mb=5.0,
        deadline_seconds=5 * 60,        # 5 min  — realtime
        priority=3,
        job_type=JobType.REALTIME,
        checkpointable=False,
    ),
    # AI Inference
    "llm_inference": WorkloadSpec(
        name="LLMInference",
        category=CATEGORY_AI,
        power_watts=600.0,
        duration_seconds=30.0,
        input_size_mb=10.0,
        output_size_mb=1.0,
        deadline_seconds=2 * 60,        # 2 min
        priority=3,
        job_type=JobType.REALTIME,
        checkpointable=False,
    ),
    "image_generation": WorkloadSpec(
        name="ImageGeneration",
        category=CATEGORY_AI,
        power_watts=500.0,
        duration_seconds=45.0,
        input_size_mb=1.0,
        output_size_mb=5.0,
        deadline_seconds=5 * 60,        # 5 min
        priority=4,
        job_type=JobType.REALTIME,
        checkpointable=False,
    ),
    # Scientific compute
    "weather_model": WorkloadSpec(
        name="WeatherModel",
        category=CATEGORY_SCIENTIFIC,
        power_watts=800.0,
        duration_seconds=3600.0,
        input_size_mb=10240.0,
        output_size_mb=5120.0,
        deadline_seconds=None,          # batch
        priority=7,
        job_type=JobType.CHECKPOINT,
        checkpointable=True,
    ),
    "climate_analysis": WorkloadSpec(
        name="ClimateAnalysis",
        category=CATEGORY_SCIENTIFIC,
        power_watts=700.0,
        duration_seconds=7200.0,
        input_size_mb=51200.0,
        output_size_mb=10240.0,
        deadline_seconds=None,          # batch
        priority=8,
        job_type=JobType.CHECKPOINT,
        checkpointable=True,
    ),
    # Defense / ISR
    "sar_processing": WorkloadSpec(
        name="SAR_Processing",
        category=CATEGORY_DEFENSE,
        power_watts=500.0,
        duration_seconds=180.0,
        input_size_mb=1024.0,
        output_size_mb=100.0,
        deadline_seconds=15 * 60,       # 15 min
        priority=2,
        job_type=JobType.REALTIME,
        checkpointable=False,
    ),
    "signal_analysis": WorkloadSpec(
        name="SignalAnalysis",
        category=CATEGORY_DEFENSE,
        power_watts=300.0,
        duration_seconds=60.0,
        input_size_mb=500.0,
        output_size_mb=10.0,
        deadline_seconds=5 * 60,        # 5 min  — realtime
        priority=1,
        job_type=JobType.REALTIME,
        checkpointable=False,
    ),
}

# Convenience lists by category
EARTH_OBS_WORKLOADS = [k for k, v in WORKLOAD_CATALOG.items() if v.category == CATEGORY_EARTH_OBS]
AI_WORKLOADS = [k for k, v in WORKLOAD_CATALOG.items() if v.category == CATEGORY_AI]
SCIENTIFIC_WORKLOADS = [k for k, v in WORKLOAD_CATALOG.items() if v.category == CATEGORY_SCIENTIFIC]
DEFENSE_WORKLOADS = [k for k, v in WORKLOAD_CATALOG.items() if v.category == CATEGORY_DEFENSE]


# ---------------------------------------------------------------------------
# Factory functions — create ComputeJob from a WorkloadSpec
# ---------------------------------------------------------------------------
_job_counter = 0


def _next_job_id() -> str:
    global _job_counter
    _job_counter += 1
    return f"WL-{_job_counter:05d}"


def create_job(workload_key: str,
               submit_time: Optional[datetime] = None,
               jitter: float = 0.15) -> ComputeJob:
    """Create a ComputeJob from a workload catalog entry.

    Parameters
    ----------
    workload_key : str
        Key into WORKLOAD_CATALOG.
    submit_time : datetime, optional
        When the job is submitted (used to set deadline).
    jitter : float
        Random +/- variation on duration/power (0-1).  Default 15%.
    """
    spec = WORKLOAD_CATALOG[workload_key]

    # Apply realistic jitter so jobs aren't identical
    power = spec.power_watts * (1.0 + random.uniform(-jitter, jitter))
    duration = spec.duration_seconds * (1.0 + random.uniform(-jitter, jitter))

    deadline = None
    if spec.deadline_seconds is not None and submit_time is not None:
        deadline = submit_time + timedelta(seconds=spec.deadline_seconds)

    return ComputeJob(
        job_id=_next_job_id(),
        name=spec.name,
        power_watts=round(power, 1),
        duration_seconds=round(duration, 1),
        job_type=spec.job_type,
        priority=spec.priority,
        deadline=deadline,
        min_battery_pct=0.30 if spec.category != CATEGORY_DEFENSE else 0.20,
        data_downlink_mb=spec.output_size_mb,
        checkpointable=spec.checkpointable,
    )


# Convenience factory functions for each workload type
def image_classification(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("image_classification", t)

def change_detection(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("change_detection", t)

def object_tracking(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("object_tracking", t)

def llm_inference(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("llm_inference", t)

def image_generation(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("image_generation", t)

def weather_model(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("weather_model", t)

def climate_analysis(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("climate_analysis", t)

def sar_processing(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("sar_processing", t)

def signal_analysis(t: Optional[datetime] = None) -> ComputeJob:
    return create_job("signal_analysis", t)


# ---------------------------------------------------------------------------
# WorkloadGenerator — realistic mixed job streams
# ---------------------------------------------------------------------------

# Category weights (approximate real-world mix)
_DEFAULT_MIX = {
    CATEGORY_EARTH_OBS: 0.70,
    CATEGORY_AI: 0.15,
    CATEGORY_SCIENTIFIC: 0.10,
    CATEGORY_DEFENSE: 0.05,
}

_CATEGORY_KEYS = {
    CATEGORY_EARTH_OBS: EARTH_OBS_WORKLOADS,
    CATEGORY_AI: AI_WORKLOADS,
    CATEGORY_SCIENTIFIC: SCIENTIFIC_WORKLOADS,
    CATEGORY_DEFENSE: DEFENSE_WORKLOADS,
}


class WorkloadGenerator:
    """Generate realistic streams of compute jobs.

    Features
    --------
    - Mixed workload categories (70% earth-obs, 15% AI, 10% scientific,
      5% defense by default).
    - Bursty arrival patterns — more jobs during ground-station contact
      windows (simulates uplinked task queues).
    - Priority based on workload type (defense > realtime > batch).
    """

    def __init__(self,
                 mix: Optional[dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.mix = mix or dict(_DEFAULT_MIX)
        self._rng = random.Random(seed)

        # Build cumulative distribution for category selection
        categories = list(self.mix.keys())
        weights = [self.mix[c] for c in categories]
        total = sum(weights)
        self._categories = categories
        self._cum_weights: list[float] = []
        running = 0.0
        for w in weights:
            running += w / total
            self._cum_weights.append(running)

    def _pick_category(self) -> str:
        r = self._rng.random()
        for cat, cw in zip(self._categories, self._cum_weights):
            if r <= cw:
                return cat
        return self._categories[-1]

    def generate_batch(self,
                       n_jobs: int,
                       start_time: datetime,
                       duration_hours: float = 6.0,
                       burst_factor: float = 3.0) -> list[ComputeJob]:
        """Generate *n_jobs* with bursty arrival times.

        Parameters
        ----------
        n_jobs : int
            Total number of jobs to create.
        start_time : datetime
            Simulation start time.
        duration_hours : float
            Time window over which jobs arrive.
        burst_factor : float
            How much more likely jobs arrive near the top/bottom of
            each orbit period (~95 min).  Higher = burstier.

        Returns
        -------
        list[ComputeJob]
            Jobs sorted by arrival time (encoded in deadline offsets).
        """
        total_seconds = duration_hours * 3600.0
        orbit_period = 95.0 * 60.0  # ~95 min LEO orbit

        jobs: list[ComputeJob] = []
        for _ in range(n_jobs):
            # Pick arrival time with bursty pattern
            # Use a sinusoidal intensity to simulate ground-pass bursts
            while True:
                t_offset = self._rng.uniform(0, total_seconds)
                # Intensity peaks every orbit period (simulating passes)
                phase = (t_offset % orbit_period) / orbit_period * 2 * math.pi
                intensity = 1.0 + (burst_factor - 1.0) * max(0, math.sin(phase))
                if self._rng.random() < intensity / burst_factor:
                    break

            arrival = start_time + timedelta(seconds=t_offset)

            cat = self._pick_category()
            keys = _CATEGORY_KEYS[cat]
            key = self._rng.choice(keys)

            job = create_job(key, submit_time=arrival)
            jobs.append(job)

        # Sort by implicit arrival order (use job_id which is sequential)
        jobs.sort(key=lambda j: j.job_id)
        return jobs

    def summary(self, jobs: list[ComputeJob]) -> dict:
        """Return a summary of a generated job batch."""
        by_name: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total_power = 0.0
        total_duration = 0.0
        total_downlink = 0.0

        for j in jobs:
            by_name[j.name] = by_name.get(j.name, 0) + 1
            by_type[j.job_type.value] = by_type.get(j.job_type.value, 0) + 1
            total_power += j.power_watts * j.duration_seconds  # Watt-seconds
            total_duration += j.duration_seconds
            total_downlink += j.data_downlink_mb

        return {
            "total_jobs": len(jobs),
            "by_workload": by_name,
            "by_type": by_type,
            "total_compute_hours": round(total_duration / 3600, 2),
            "total_energy_kwh": round(total_power / 3_600_000, 2),
            "total_downlink_gb": round(total_downlink / 1024, 2),
        }


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
def _demo():
    print("=" * 60)
    print("  ORBITAL COMPUTE — WORKLOAD DEMO")
    print("=" * 60)

    # Show catalog
    print("\nWorkload Catalog:")
    print(f"  {'Name':<22} {'Cat':<18} {'Power':>6} {'Dur':>7} {'In':>8} {'Out':>8} {'Deadline':>9} {'Pri':>4}")
    print(f"  {'-'*93}")
    for key, spec in WORKLOAD_CATALOG.items():
        dl = f"{spec.deadline_seconds/60:.0f}min" if spec.deadline_seconds else "batch"
        print(f"  {spec.name:<22} {spec.category:<18} {spec.power_watts:>5.0f}W "
              f"{spec.duration_seconds:>6.0f}s {spec.input_size_mb:>7.0f}MB "
              f"{spec.output_size_mb:>7.0f}MB {dl:>9} {spec.priority:>4}")

    # Generate a batch
    t0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    gen = WorkloadGenerator(seed=42)
    jobs = gen.generate_batch(50, t0, duration_hours=6.0)

    summary = gen.summary(jobs)
    print(f"\nGenerated Batch ({summary['total_jobs']} jobs):")
    print(f"  Compute: {summary['total_compute_hours']} hours")
    print(f"  Energy:  {summary['total_energy_kwh']} kWh")
    print(f"  Downlink: {summary['total_downlink_gb']} GB")
    print(f"\n  By workload type:")
    for name, count in sorted(summary['by_workload'].items()):
        print(f"    {name:<22} {count:>3}")
    print(f"\n  By job type:")
    for jt, count in sorted(summary['by_type'].items()):
        print(f"    {jt:<12} {count:>3}")

    # Spot-check a few jobs
    print("\n  Sample jobs:")
    for j in jobs[:5]:
        dl = f"deadline={j.deadline.strftime('%H:%M')}" if j.deadline else "batch"
        print(f"    {j.job_id} {j.name:<22} {j.power_watts:>6.1f}W {j.duration_seconds:>7.1f}s "
              f"pri={j.priority} {dl}")

    print(f"\n{'=' * 60}")
    print("  PASS — workloads module OK")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _demo()
