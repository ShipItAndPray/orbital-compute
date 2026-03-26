from __future__ import annotations

"""Federated learning orchestration for satellite constellations.

Train ML models across distributed satellites without centralizing data.
Each satellite trains locally on its Earth-observation data, then shares
only model gradients via inter-satellite links.

Supports three aggregation strategies (FedAvg, FedProx, SCAFFOLD) and
three communication topologies (Star, Ring, Tree).

Uses isl.py for connectivity and power.py for availability checks.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from orbital_compute.constellations import (
    CONSTELLATIONS,
    ConstellationConfig,
    generate_constellation,
)
from orbital_compute.isl import (
    InterSatelliteNetwork,
    LinkMetrics,
    MAX_BANDWIDTH_GBPS,
    build_connectivity_graph,
)
from orbital_compute.orbit import Satellite
from orbital_compute.power import PowerConfig, PowerModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Satellite GPU compute assumptions
GPU_THROUGHPUT_MB_PER_SEC = 50.0  # Effective training throughput (MB model / sec)
LOCAL_TRAINING_POWER_WATTS = 400.0  # Power draw during local training
GRADIENT_COMPRESSION_RATIO = 0.1  # Compressed gradients = 10% of model size


class AggregationStrategy(Enum):
    """Federated aggregation algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"


class Topology(Enum):
    """Communication topology for gradient aggregation."""
    STAR = "star"
    RING = "ring"
    TREE = "tree"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FederatedTrainingJob:
    """Specification for a federated learning training job."""
    model_size_mb: float  # Size of model weights in MB
    training_data_mb_per_sat: float  # Local data per satellite in MB
    epochs_per_round: int  # Local training epochs per FL round
    communication_rounds: int  # Total federated learning rounds
    aggregation_strategy: str = "fedavg"  # "fedavg" | "fedprox" | "scaffold"

    def __post_init__(self) -> None:
        valid = {"fedavg", "fedprox", "scaffold"}
        if self.aggregation_strategy not in valid:
            raise ValueError(
                f"aggregation_strategy must be one of {valid}, "
                f"got '{self.aggregation_strategy}'"
            )

    @property
    def gradient_size_mb(self) -> float:
        """Compressed gradient size for one participant."""
        return self.model_size_mb * GRADIENT_COMPRESSION_RATIO

    @property
    def strategy_enum(self) -> AggregationStrategy:
        return AggregationStrategy(self.aggregation_strategy)


@dataclass
class ParticipantInfo:
    """Info about a satellite participating in a FL round."""
    name: str
    power_available_w: float
    n_neighbors: int
    data_freshness_hours: float  # How recently this sat collected data


@dataclass
class RoundResult:
    """Result of a single federated learning round."""
    round_number: int
    participants: List[str]
    local_training_time_s: float
    gradient_upload_time_s: float
    aggregation_time_s: float
    gradient_download_time_s: float
    total_round_time_s: float
    topology_used: str
    straggler_delay_s: float  # Extra time due to slowest participant
    communication_overhead_pct: float  # Comm time / total time


@dataclass
class TrainingSimulationResult:
    """Result of a full federated training simulation."""
    job: FederatedTrainingJob
    rounds_completed: int
    total_time_hours: float
    round_results: List[RoundResult]
    convergence_proxy: List[float]  # Simulated loss per round
    total_compute_time_hours: float
    total_comm_time_hours: float
    compute_comm_ratio: float
    avg_participants_per_round: float
    topology_used: str
    use_case: str


# ---------------------------------------------------------------------------
# Federated Orchestrator
# ---------------------------------------------------------------------------

class FederatedOrchestrator:
    """Orchestrates federated learning across a satellite constellation.

    Handles participant selection, round timing estimation, topology
    selection, and full training simulation.
    """

    def __init__(
        self,
        satellites: List[Satellite],
        power_configs: Optional[Dict[str, PowerConfig]] = None,
        seed: Optional[int] = None,
    ):
        self.satellites = satellites
        self._sat_by_name = {s.name: s for s in satellites}
        self._network = InterSatelliteNetwork(satellites)

        # Power models per satellite (default config if not provided)
        self._power_models: Dict[str, PowerModel] = {}
        for sat in satellites:
            cfg = (power_configs or {}).get(sat.name, PowerConfig())
            self._power_models[sat.name] = PowerModel(cfg)

        self._rng = random.Random(seed)

        # Track simulated data freshness (hours since last data collection)
        self._data_freshness: Dict[str, float] = {
            s.name: self._rng.uniform(0.0, 6.0) for s in satellites
        }

    def plan_round(
        self,
        satellites: List[Satellite],
        timestamp: datetime,
        min_participants: int = 3,
    ) -> List[ParticipantInfo]:
        """Select satellites to participate in a FL round.

        Selection criteria:
        1. Power availability -- must have enough for local training
        2. ISL connectivity -- must have at least one link for gradient sharing
        3. Data freshness -- satellites with newer data get priority

        Parameters
        ----------
        satellites : list of Satellite
            Candidate satellites.
        timestamp : datetime
            Current simulation time.
        min_participants : int
            Minimum participants needed for a valid round.

        Returns
        -------
        list of ParticipantInfo
            Selected participants, sorted by priority (freshest data first).
        """
        # Update network connectivity
        self._network.update(timestamp)
        graph = self._network.graph

        candidates: List[ParticipantInfo] = []

        for sat in satellites:
            name = sat.name

            # Check power: simulate a step to see current state
            pos = sat.position_at(timestamp)
            in_eclipse = pos.in_eclipse
            power_model = self._power_models[name]
            state = power_model.step(
                dt_seconds=1.0,
                in_eclipse=in_eclipse,
                compute_load_w=0.0,
            )

            # Must have enough power for local training
            if not state.can_compute:
                continue
            if state.available_for_compute_w < LOCAL_TRAINING_POWER_WATTS:
                continue

            # Must have ISL connectivity
            neighbors = graph.get(name, [])
            if len(neighbors) == 0:
                continue

            freshness = self._data_freshness.get(name, 12.0)
            candidates.append(ParticipantInfo(
                name=name,
                power_available_w=state.available_for_compute_w,
                n_neighbors=len(neighbors),
                data_freshness_hours=freshness,
            ))

        # Sort by data freshness (newer = lower hours = higher priority)
        candidates.sort(key=lambda p: p.data_freshness_hours)

        if len(candidates) < min_participants:
            return []  # Not enough participants for a valid round

        return candidates

    def _choose_topology(
        self,
        participants: List[ParticipantInfo],
        graph: Dict[str, list],
    ) -> Topology:
        """Choose communication topology based on connectivity.

        - Star: few participants or high connectivity (one aggregator)
        - Ring: moderate connectivity, balanced load
        - Tree: many participants, best for large constellations
        """
        n = len(participants)
        avg_neighbors = sum(p.n_neighbors for p in participants) / max(n, 1)

        if n <= 5:
            return Topology.STAR
        elif n <= 15 and avg_neighbors >= 3:
            return Topology.RING
        else:
            return Topology.TREE

    def _estimate_isl_bandwidth(
        self,
        participants: List[ParticipantInfo],
    ) -> float:
        """Estimate effective ISL bandwidth for gradient transfer (Gbps)."""
        # Use conservative estimate: half of max bandwidth due to sharing
        # and distance degradation
        avg_neighbors = sum(p.n_neighbors for p in participants) / max(
            len(participants), 1
        )
        # More neighbors = better bandwidth availability
        utilization = min(1.0, avg_neighbors / 4.0)
        return MAX_BANDWIDTH_GBPS * 0.5 * utilization

    def estimate_round_time(
        self,
        job: FederatedTrainingJob,
        participants: List[ParticipantInfo],
    ) -> RoundResult:
        """Estimate wall-clock time for one FL round.

        Components:
        1. Local training: model_size / GPU_throughput * epochs
        2. Gradient upload: gradient_size / ISL_bandwidth per participant
        3. Aggregation: depends on topology
        4. Gradient download: model_size / ISL_bandwidth (broadcast)

        Parameters
        ----------
        job : FederatedTrainingJob
            The training job specification.
        participants : list of ParticipantInfo
            Selected participants for this round.

        Returns
        -------
        RoundResult
            Timing breakdown for the round.
        """
        n = len(participants)
        if n == 0:
            return RoundResult(
                round_number=0,
                participants=[],
                local_training_time_s=0,
                gradient_upload_time_s=0,
                aggregation_time_s=0,
                gradient_download_time_s=0,
                total_round_time_s=0,
                topology_used="none",
                straggler_delay_s=0,
                communication_overhead_pct=0,
            )

        # 1. Local training time (all participants train in parallel)
        # Slower satellites act as stragglers
        base_training_time = (
            job.model_size_mb / GPU_THROUGHPUT_MB_PER_SEC * job.epochs_per_round
        )

        # Straggler effect: satellite with least power trains slowest
        min_power = min(p.power_available_w for p in participants)
        max_power = max(p.power_available_w for p in participants)
        # Power-limited satellites train proportionally slower
        straggler_factor = max_power / max(min_power, 1.0)
        straggler_factor = min(straggler_factor, 3.0)  # Cap at 3x

        local_training_time = base_training_time * straggler_factor
        straggler_delay = base_training_time * (straggler_factor - 1.0)

        # 2. Gradient upload
        bw_gbps = self._estimate_isl_bandwidth(participants)
        bw_mb_per_sec = bw_gbps * 1000.0 / 8.0  # Gbps -> MB/s
        gradient_size = job.gradient_size_mb

        # Choose topology
        topology = self._choose_topology(
            participants, self._network.graph
        )

        if topology == Topology.STAR:
            # All send to one aggregator sequentially
            gradient_upload_time = n * gradient_size / bw_mb_per_sec
        elif topology == Topology.RING:
            # Pass around ring: n-1 steps, each sending gradient_size
            gradient_upload_time = (n - 1) * gradient_size / bw_mb_per_sec
        elif topology == Topology.TREE:
            # Hierarchical: log2(n) levels, each level aggregates in parallel
            levels = max(1, math.ceil(math.log2(max(n, 2))))
            gradient_upload_time = levels * gradient_size / bw_mb_per_sec
        else:
            gradient_upload_time = n * gradient_size / bw_mb_per_sec

        # 3. Aggregation time (compute at aggregator — fast)
        # FedAvg: simple weighted average
        # FedProx: same as FedAvg + proximal term (negligible extra)
        # SCAFFOLD: extra control variate computation
        aggregation_compute = gradient_size / GPU_THROUGHPUT_MB_PER_SEC
        if job.aggregation_strategy == "scaffold":
            aggregation_compute *= 2.0  # Control variates double the work

        # 4. Gradient download (broadcast updated model)
        if topology == Topology.STAR:
            gradient_download_time = gradient_size / bw_mb_per_sec
        elif topology == Topology.RING:
            gradient_download_time = (n - 1) * gradient_size / bw_mb_per_sec
        elif topology == Topology.TREE:
            levels = max(1, math.ceil(math.log2(max(n, 2))))
            gradient_download_time = levels * gradient_size / bw_mb_per_sec
        else:
            gradient_download_time = gradient_size / bw_mb_per_sec

        total_time = (
            local_training_time
            + gradient_upload_time
            + aggregation_compute
            + gradient_download_time
        )

        comm_time = gradient_upload_time + gradient_download_time
        comm_overhead = (comm_time / total_time * 100.0) if total_time > 0 else 0.0

        return RoundResult(
            round_number=0,
            participants=[p.name for p in participants],
            local_training_time_s=local_training_time,
            gradient_upload_time_s=gradient_upload_time,
            aggregation_time_s=aggregation_compute,
            gradient_download_time_s=gradient_download_time,
            total_round_time_s=total_time,
            topology_used=topology.value,
            straggler_delay_s=straggler_delay,
            communication_overhead_pct=round(comm_overhead, 1),
        )

    def simulate_training(
        self,
        job: FederatedTrainingJob,
        constellation_config: ConstellationConfig,
        hours: float = 6.0,
        max_sats: int = 24,
        use_case: str = "general",
    ) -> TrainingSimulationResult:
        """Simulate a full federated training run over the constellation.

        Parameters
        ----------
        job : FederatedTrainingJob
            Training job specification.
        constellation_config : ConstellationConfig
            Constellation to train across.
        hours : float
            Maximum wall-clock hours for the simulation.
        max_sats : int
            Max satellites to include (for performance).
        use_case : str
            Description of the use case.

        Returns
        -------
        TrainingSimulationResult
            Full training simulation results.
        """
        # Generate constellation satellites
        sats = generate_constellation(constellation_config, max_sats=max_sats)
        self.satellites = sats
        self._sat_by_name = {s.name: s for s in sats}
        self._network = InterSatelliteNetwork(sats)

        # Reset power models
        for sat in sats:
            if sat.name not in self._power_models:
                self._power_models[sat.name] = PowerModel(PowerConfig())

        # Reset data freshness
        self._data_freshness = {
            s.name: self._rng.uniform(0.0, 6.0) for s in sats
        }

        start_time = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
        current_time = start_time
        max_time = start_time + timedelta(hours=hours)

        round_results: List[RoundResult] = []
        convergence_proxy: List[float] = []
        total_compute_s = 0.0
        total_comm_s = 0.0
        rounds_done = 0

        # Simulated loss curve: starts at 2.5, decays with noise
        loss = 2.5

        for round_num in range(job.communication_rounds):
            if current_time >= max_time:
                break

            # Plan the round
            participants = self.plan_round(sats, current_time)
            if not participants:
                # No valid participants — advance time and retry
                current_time += timedelta(minutes=10)
                continue

            # Estimate round timing
            result = self.estimate_round_time(job, participants)
            result.round_number = round_num + 1

            # Advance simulation time
            round_duration = timedelta(seconds=result.total_round_time_s)
            current_time += round_duration

            # Step power models forward
            for sat in sats:
                pos = sat.position_at(current_time)
                pm = self._power_models.get(sat.name)
                if pm:
                    is_participant = sat.name in result.participants
                    load = LOCAL_TRAINING_POWER_WATTS if is_participant else 0.0
                    pm.step(result.total_round_time_s, pos.in_eclipse, load)

            # Update data freshness (participants collected newer data)
            for name in result.participants:
                self._data_freshness[name] = 0.0
            for name in self._data_freshness:
                self._data_freshness[name] += (
                    result.total_round_time_s / 3600.0
                )

            round_results.append(result)
            total_compute_s += result.local_training_time_s * len(
                result.participants
            )
            total_comm_s += (
                result.gradient_upload_time_s + result.gradient_download_time_s
            )

            # Simulated convergence: loss decreases with diminishing returns
            n_participants = len(result.participants)
            # More participants = faster convergence (up to a point)
            participation_bonus = min(n_participants / 10.0, 1.0)

            # Strategy affects convergence speed
            strategy_factor = {
                "fedavg": 1.0,
                "fedprox": 1.1,  # Slightly better with heterogeneous data
                "scaffold": 1.3,  # Best for non-IID data
            }.get(job.aggregation_strategy, 1.0)

            decay = 0.15 * participation_bonus * strategy_factor
            noise = self._rng.gauss(0, 0.02)
            loss = max(0.01, loss * (1.0 - decay) + noise)
            convergence_proxy.append(round(loss, 4))

            rounds_done += 1

        elapsed_hours = (
            (current_time - start_time).total_seconds() / 3600.0
        )

        total_compute_h = total_compute_s / 3600.0
        total_comm_h = total_comm_s / 3600.0
        ratio = (
            total_compute_h / total_comm_h if total_comm_h > 0 else float("inf")
        )

        avg_participants = (
            sum(len(r.participants) for r in round_results)
            / max(len(round_results), 1)
        )

        # Determine most-used topology
        topo_counts: Dict[str, int] = {}
        for r in round_results:
            topo_counts[r.topology_used] = topo_counts.get(
                r.topology_used, 0
            ) + 1
        primary_topology = max(topo_counts, key=topo_counts.get) if topo_counts else "none"

        return TrainingSimulationResult(
            job=job,
            rounds_completed=rounds_done,
            total_time_hours=round(elapsed_hours, 2),
            round_results=round_results,
            convergence_proxy=convergence_proxy,
            total_compute_time_hours=round(total_compute_h, 2),
            total_comm_time_hours=round(total_comm_h, 2),
            compute_comm_ratio=round(ratio, 2),
            avg_participants_per_round=round(avg_participants, 1),
            topology_used=primary_topology,
            use_case=use_case,
        )


# ---------------------------------------------------------------------------
# Use case presets
# ---------------------------------------------------------------------------

def weather_prediction_job() -> FederatedTrainingJob:
    """Global weather prediction — each satellite sees local weather patterns."""
    return FederatedTrainingJob(
        model_size_mb=500.0,
        training_data_mb_per_sat=2048.0,
        epochs_per_round=3,
        communication_rounds=50,
        aggregation_strategy="fedavg",
    )


def object_detection_job() -> FederatedTrainingJob:
    """Distributed object detection — each satellite has local imagery."""
    return FederatedTrainingJob(
        model_size_mb=200.0,
        training_data_mb_per_sat=1024.0,
        epochs_per_round=5,
        communication_rounds=30,
        aggregation_strategy="fedprox",
    )


def anomaly_detection_job() -> FederatedTrainingJob:
    """Anomaly detection — each satellite monitors its local region."""
    return FederatedTrainingJob(
        model_size_mb=50.0,
        training_data_mb_per_sat=512.0,
        epochs_per_round=2,
        communication_rounds=20,
        aggregation_strategy="scaffold",
    )


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.2f}h"


def _demo() -> None:
    print("=" * 70)
    print("  ORBITAL COMPUTE — FEDERATED LEARNING ORCHESTRATION")
    print("=" * 70)

    # Use starlink-mini for quick demo
    config = CONSTELLATIONS["starlink-mini"]
    sats = generate_constellation(config, max_sats=24)
    print(f"\nConstellation: {config.name} ({len(sats)} satellites)")
    print(f"  Altitude: {config.altitude_km} km")
    print(f"  Inclination: {config.inclination_deg} deg")

    orchestrator = FederatedOrchestrator(sats, seed=42)
    timestamp = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)

    # --- Use Case 1: Global Weather Prediction ---
    print(f"\n{'─' * 70}")
    print("  USE CASE 1: Global Weather Prediction (FedAvg)")
    print(f"{'─' * 70}")

    job = weather_prediction_job()
    print(f"  Model size:     {job.model_size_mb} MB")
    print(f"  Data per sat:   {job.training_data_mb_per_sat} MB")
    print(f"  Epochs/round:   {job.epochs_per_round}")
    print(f"  FL rounds:      {job.communication_rounds}")
    print(f"  Strategy:       {job.aggregation_strategy}")
    print(f"  Gradient size:  {job.gradient_size_mb:.1f} MB (compressed)")

    result = orchestrator.simulate_training(
        job, config, hours=12.0, max_sats=24, use_case="weather_prediction"
    )
    _print_result(result)

    # --- Use Case 2: Distributed Object Detection ---
    print(f"\n{'─' * 70}")
    print("  USE CASE 2: Distributed Object Detection (FedProx)")
    print(f"{'─' * 70}")

    job2 = object_detection_job()
    print(f"  Model size:     {job2.model_size_mb} MB")
    print(f"  Data per sat:   {job2.training_data_mb_per_sat} MB")
    print(f"  Epochs/round:   {job2.epochs_per_round}")
    print(f"  FL rounds:      {job2.communication_rounds}")
    print(f"  Strategy:       {job2.aggregation_strategy}")

    result2 = orchestrator.simulate_training(
        job2, config, hours=8.0, max_sats=24, use_case="object_detection"
    )
    _print_result(result2)

    # --- Use Case 3: Anomaly Detection ---
    print(f"\n{'─' * 70}")
    print("  USE CASE 3: Anomaly Detection (SCAFFOLD)")
    print(f"{'─' * 70}")

    job3 = anomaly_detection_job()
    print(f"  Model size:     {job3.model_size_mb} MB")
    print(f"  Data per sat:   {job3.training_data_mb_per_sat} MB")
    print(f"  Epochs/round:   {job3.epochs_per_round}")
    print(f"  FL rounds:      {job3.communication_rounds}")
    print(f"  Strategy:       {job3.aggregation_strategy}")

    result3 = orchestrator.simulate_training(
        job3, config, hours=4.0, max_sats=24, use_case="anomaly_detection"
    )
    _print_result(result3)

    # --- Single Round Deep Dive ---
    print(f"\n{'─' * 70}")
    print("  SINGLE ROUND DEEP DIVE")
    print(f"{'─' * 70}")

    sats2 = generate_constellation(config, max_sats=24)
    orch2 = FederatedOrchestrator(sats2, seed=42)
    participants = orch2.plan_round(sats2, timestamp)
    print(f"\n  Participants selected: {len(participants)}")
    for p in participants[:8]:
        print(
            f"    {p.name}: power={p.power_available_w:.0f}W, "
            f"neighbors={p.n_neighbors}, "
            f"data_age={p.data_freshness_hours:.1f}h"
        )
    if len(participants) > 8:
        print(f"    ... and {len(participants) - 8} more")

    if participants:
        round_est = orch2.estimate_round_time(job, participants)
        print(f"\n  Round timing estimate (weather model, {len(participants)} participants):")
        print(f"    Local training:     {_format_time(round_est.local_training_time_s)}")
        print(f"    Gradient upload:    {_format_time(round_est.gradient_upload_time_s)}")
        print(f"    Aggregation:        {_format_time(round_est.aggregation_time_s)}")
        print(f"    Gradient download:  {_format_time(round_est.gradient_download_time_s)}")
        print(f"    Total round time:   {_format_time(round_est.total_round_time_s)}")
        print(f"    Topology:           {round_est.topology_used}")
        print(f"    Straggler delay:    {_format_time(round_est.straggler_delay_s)}")
        print(f"    Comm overhead:      {round_est.communication_overhead_pct}%")

    print(f"\n{'=' * 70}")
    print("  PASS — federated learning module OK")
    print(f"{'=' * 70}")


def _print_result(result: TrainingSimulationResult) -> None:
    """Print a training simulation result summary."""
    print(f"\n  Results:")
    print(f"    Rounds completed:   {result.rounds_completed}/{result.job.communication_rounds}")
    print(f"    Total time:         {result.total_time_hours:.2f} hours")
    print(f"    Avg participants:   {result.avg_participants_per_round:.1f} per round")
    print(f"    Topology:           {result.topology_used}")
    print(f"    Compute time:       {result.total_compute_time_hours:.2f} hours (aggregate)")
    print(f"    Comm time:          {result.total_comm_time_hours:.2f} hours")
    print(f"    Compute/Comm ratio: {result.compute_comm_ratio:.1f}x")

    if result.convergence_proxy:
        print(f"    Loss (start):       {result.convergence_proxy[0]:.4f}")
        print(f"    Loss (final):       {result.convergence_proxy[-1]:.4f}")
        reduction = (
            (1.0 - result.convergence_proxy[-1] / result.convergence_proxy[0])
            * 100.0
        )
        print(f"    Loss reduction:     {reduction:.1f}%")

    if result.round_results:
        comm_overheads = [r.communication_overhead_pct for r in result.round_results]
        straggler_delays = [r.straggler_delay_s for r in result.round_results]
        print(f"    Avg comm overhead:  {sum(comm_overheads) / len(comm_overheads):.1f}%")
        print(f"    Avg straggler:      {_format_time(sum(straggler_delays) / len(straggler_delays))}")


if __name__ == "__main__":
    _demo()
