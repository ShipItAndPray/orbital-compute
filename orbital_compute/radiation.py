"""Radiation fault injection for orbital compute hardware in LEO.

Models Single Event Upsets (SEUs) caused by trapped protons and heavy ions,
with emphasis on the South Atlantic Anomaly (SAA). Provides configurable
recovery strategies: checkpoint/restart, dual execution, and TMR.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base SEU rate in LEO outside SAA (upsets / bit / day)
BASE_SEU_RATE: float = 1e-7

# Multiplier while transiting the South Atlantic Anomaly
SAA_MULTIPLIER: float = 10.0

# SAA bounding box (coarse rectangle)
SAA_LAT_MIN: float = -45.0
SAA_LAT_MAX: float = -15.0
SAA_LON_MIN: float = -90.0
SAA_LON_MAX: float = 40.0

# Fraction of SEUs that are multi-bit (uncorrectable by ECC)
MULTI_BIT_FRACTION: float = 0.01

BITS_PER_MB: int = 8 * 1024 * 1024
SECONDS_PER_DAY: float = 86_400.0


# ---------------------------------------------------------------------------
# Recovery strategy enum
# ---------------------------------------------------------------------------

class RecoveryStrategy(Enum):
    NONE = "NONE"
    CHECKPOINT_RESTART = "CHECKPOINT_RESTART"
    DUAL_EXECUTION = "DUAL_EXECUTION"
    TMR = "TMR"


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------

@dataclass
class RadiationStats:
    total_upsets: int = 0
    single_bit: int = 0
    multi_bit: int = 0
    caught_by_ecc: int = 0
    caught_by_strategy: int = 0
    uncaught: int = 0
    recovered: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Simple job stub (for handle_upset)
# ---------------------------------------------------------------------------

@dataclass
class ComputeJob:
    name: str
    memory_mb: float
    duration_s: float
    elapsed_s: float = 0.0
    failed: bool = False


# ---------------------------------------------------------------------------
# RadiationModel
# ---------------------------------------------------------------------------

class RadiationModel:
    """Monte Carlo radiation fault model for LEO compute payloads."""

    def __init__(
        self,
        strategy: RecoveryStrategy = RecoveryStrategy.NONE,
        ecc_coverage: float = 0.99,
        seed: Optional[int] = None,
    ) -> None:
        self.strategy = strategy
        self.ecc_coverage = ecc_coverage
        self.stats = RadiationStats()
        self._rng = random.Random(seed)

    # ----- helpers -----

    @staticmethod
    def in_saa(lat_deg: float, lon_deg: float) -> bool:
        """Return True if the satellite ground-track is inside the SAA box."""
        return (
            SAA_LAT_MIN <= lat_deg <= SAA_LAT_MAX
            and SAA_LON_MIN <= lon_deg <= SAA_LON_MAX
        )

    def _seu_rate(self, lat_deg: float, lon_deg: float) -> float:
        """Return SEU rate (upsets/bit/day) for the given position."""
        rate = BASE_SEU_RATE
        if self.in_saa(lat_deg, lon_deg):
            rate *= SAA_MULTIPLIER
        return rate

    # ----- public API -----

    def check_for_upset(
        self,
        lat_deg: float,
        lon_deg: float,
        dt_seconds: float,
        memory_mb: float,
    ) -> bool:
        """Check whether an SEU occurs during *dt_seconds* at the given position.

        The probability is:
            P = 1 - (1 - rate_per_bit_per_day)^(bits * dt/day)
        For small rates this simplifies via the Poisson approximation:
            P ~ rate * bits * dt/day
        which avoids floating-point underflow.
        """
        rate = self._seu_rate(lat_deg, lon_deg)
        bits = memory_mb * BITS_PER_MB
        lam = rate * bits * (dt_seconds / SECONDS_PER_DAY)  # expected upsets

        # Poisson: P(>=1 upset) = 1 - exp(-lambda)
        prob = 1.0 - math.exp(-lam)
        return self._rng.random() < prob

    def handle_upset(
        self,
        job: ComputeJob,
        has_checkpoint: bool = False,
    ) -> str:
        """Process an SEU that hit *job*. Returns one of:
        ``"recovered"`` | ``"failed"`` | ``"undetected"``
        """
        self.stats.total_upsets += 1

        # Determine single-bit vs multi-bit
        is_multi = self._rng.random() < MULTI_BIT_FRACTION
        if is_multi:
            self.stats.multi_bit += 1
        else:
            self.stats.single_bit += 1

        # ECC handles single-bit errors transparently
        if not is_multi and self._rng.random() < self.ecc_coverage:
            self.stats.caught_by_ecc += 1
            return "recovered"

        # If ECC didn't catch it, strategy decides
        if self.strategy == RecoveryStrategy.NONE:
            self.stats.failed += 1
            job.failed = True
            return "failed"

        if self.strategy == RecoveryStrategy.CHECKPOINT_RESTART:
            if has_checkpoint:
                self.stats.recovered += 1
                return "recovered"
            else:
                # No checkpoint available — job must restart from scratch
                self.stats.failed += 1
                job.failed = True
                return "failed"

        if self.strategy == RecoveryStrategy.DUAL_EXECUTION:
            # Dual execution detects the error (mismatch). Recovery succeeds
            # because the uncorrupted copy is available.
            self.stats.caught_by_strategy += 1
            self.stats.recovered += 1
            return "recovered"

        if self.strategy == RecoveryStrategy.TMR:
            # Majority vote — tolerates any single-copy corruption
            self.stats.caught_by_strategy += 1
            self.stats.recovered += 1
            return "recovered"

        # Fallback (should not be reached)
        self.stats.uncaught += 1
        return "undetected"

    def overhead_factor(self) -> float:
        """Return compute overhead multiplier for the chosen strategy."""
        return {
            RecoveryStrategy.NONE: 1.0,
            RecoveryStrategy.CHECKPOINT_RESTART: 1.05,
            RecoveryStrategy.DUAL_EXECUTION: 2.0,
            RecoveryStrategy.TMR: 3.0,
        }[self.strategy]


# ---------------------------------------------------------------------------
# 24-hour simulation demo
# ---------------------------------------------------------------------------

def _simulate_24h() -> None:
    """Simulate 24 h of LEO radiation exposure with a satellite that
    periodically transits the SAA, comparing all recovery strategies."""

    print("=" * 70)
    print("  Radiation Fault Injection — 24 h LEO Simulation")
    print("=" * 70)

    sim_seconds = 24 * 3600
    dt = 60.0  # 1-minute time steps
    steps = int(sim_seconds / dt)
    memory_mb = 4096.0  # 4 GB payload RAM

    # Orbital period ~95 min for a 500 km LEO orbit
    orbital_period = 95 * 60.0  # seconds

    # Simple sinusoidal ground-track model
    # Latitude oscillates +/- inclination (51.6 deg, ISS-like)
    # Longitude drifts westward ~22.5 deg per orbit
    inclination = 51.6
    lon_drift_per_s = -360.0 / sim_seconds  # roughly

    seed = 42

    for strategy in RecoveryStrategy:
        model = RadiationModel(strategy=strategy, seed=seed)
        job = ComputeJob(name="ml-training", memory_mb=memory_mb, duration_s=sim_seconds)

        saa_time = 0.0
        upset_times: list[float] = []

        for step in range(steps):
            t = step * dt

            # Ground-track approximation
            orbit_phase = 2.0 * math.pi * t / orbital_period
            lat = inclination * math.sin(orbit_phase)
            lon = ((t * lon_drift_per_s + 180.0) % 360.0) - 180.0

            if RadiationModel.in_saa(lat, lon):
                saa_time += dt

            if model.check_for_upset(lat, lon, dt, memory_mb):
                has_ckpt = strategy == RecoveryStrategy.CHECKPOINT_RESTART
                result = model.handle_upset(job, has_checkpoint=has_ckpt)
                upset_times.append(t)

        stats = model.stats
        print(f"\nStrategy: {strategy.value}")
        print(f"  Overhead factor:     {model.overhead_factor():.2f}x")
        print(f"  SAA transit time:    {saa_time / 60:.1f} min ({100 * saa_time / sim_seconds:.1f}%)")
        print(f"  Total SEU events:    {stats.total_upsets}")
        print(f"    Single-bit:        {stats.single_bit}")
        print(f"    Multi-bit:         {stats.multi_bit}")
        print(f"    Caught by ECC:     {stats.caught_by_ecc}")
        print(f"    Caught by strat:   {stats.caught_by_strategy}")
        print(f"    Recovered:         {stats.recovered}")
        print(f"    Failed:            {stats.failed}")
        print(f"    Uncaught:          {stats.uncaught}")

    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print("=" * 70)


if __name__ == "__main__":
    _simulate_24h()
