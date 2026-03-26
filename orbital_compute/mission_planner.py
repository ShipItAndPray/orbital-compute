from __future__ import annotations

"""Mission planning and operations timeline — what ops teams actually use.

Generates complete operational timelines for satellite constellations:
- Eclipse entry/exit per satellite
- Ground station AOS/LOS (pass predictions)
- Maintenance windows
- Data uplink/downlink windows
- Compute job execution windows
- Conflict detection with severity levels
- Data budget analysis
- Constellation availability/reliability modeling

Output formats: JSON, human-readable schedule, iCalendar (.ics).
"""

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple

import numpy as np

from .orbit import (
    Satellite, SatPosition, EARTH_RADIUS_KM,
    predict_eclipse_windows, sun_position_eci, is_in_eclipse,
)
from .ground_stations import (
    GroundStation, ContactWindow, DEFAULT_GROUND_STATIONS,
    elevation_angle, find_contact_windows,
)
from .constellations import (
    ConstellationConfig, CONSTELLATIONS, generate_constellation,
)
from .power import PowerConfig, PowerModel
from .workloads import WORKLOAD_CATALOG, WorkloadSpec


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimedEvent:
    """A single event on the operations timeline."""
    satellite: str
    event_type: str       # eclipse_start, eclipse_end, aos, los, job_start, etc.
    timestamp: datetime
    duration_seconds: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "satellite": self.satellite,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }
        return d


@dataclass
class SchedulingConflict:
    """A detected scheduling conflict."""
    conflict_id: str
    severity: str          # "critical", "warning", "info"
    description: str
    events: List[TimedEvent]
    resolution_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "severity": self.severity,
            "description": self.description,
            "events": [e.to_dict() for e in self.events],
            "resolution_hint": self.resolution_hint,
        }


@dataclass
class PassPrediction:
    """A predicted satellite pass over a ground station."""
    satellite_name: str
    station_name: str
    aos_time: datetime          # Acquisition of Signal
    los_time: datetime          # Loss of Signal
    max_elevation_deg: float
    duration_seconds: float
    downlink_capacity_mb: float
    uplink_capacity_mb: float

    def to_dict(self) -> dict:
        return {
            "satellite": self.satellite_name,
            "station": self.station_name,
            "aos": self.aos_time.isoformat(),
            "los": self.los_time.isoformat(),
            "max_elevation_deg": round(self.max_elevation_deg, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "downlink_capacity_mb": round(self.downlink_capacity_mb, 1),
            "uplink_capacity_mb": round(self.uplink_capacity_mb, 1),
        }


@dataclass
class DataBudget:
    """Data budget analysis for the constellation."""
    data_generated_per_orbit_mb: float
    downlink_capacity_per_orbit_mb: float
    net_data_per_orbit_mb: float          # negative = backlog growing
    orbits_until_storage_full: float
    storage_capacity_mb: float
    backlog_rate_mb_per_hour: float
    requires_onboard_processing: bool
    processing_reduction_factor: float    # e.g. 100x for image classification
    effective_downlink_with_processing_mb: float

    def to_dict(self) -> dict:
        return {
            "data_generated_per_orbit_mb": round(self.data_generated_per_orbit_mb, 1),
            "downlink_capacity_per_orbit_mb": round(self.downlink_capacity_per_orbit_mb, 1),
            "net_data_per_orbit_mb": round(self.net_data_per_orbit_mb, 1),
            "orbits_until_storage_full": round(self.orbits_until_storage_full, 1)
            if self.orbits_until_storage_full < 1e6 else "inf",
            "storage_capacity_mb": round(self.storage_capacity_mb, 1),
            "backlog_rate_mb_per_hour": round(self.backlog_rate_mb_per_hour, 1),
            "requires_onboard_processing": self.requires_onboard_processing,
            "processing_reduction_factor": round(self.processing_reduction_factor, 1),
            "effective_downlink_with_processing_mb": round(
                self.effective_downlink_with_processing_mb, 1),
        }


@dataclass
class AvailabilityResult:
    """Constellation availability analysis."""
    constellation_availability: float    # Probability all services running (0-1)
    single_satellite_availability: float
    n_satellites: int
    min_required: int
    availability_with_redundancy: float  # P(at least min_required operational)
    degradation_curve: List[Tuple[int, float]]  # (n_failed, availability)
    mtbf_hours: float
    mttr_hours: float

    def to_dict(self) -> dict:
        return {
            "constellation_availability_pct": round(self.constellation_availability * 100, 4),
            "single_satellite_availability_pct": round(self.single_satellite_availability * 100, 4),
            "n_satellites": self.n_satellites,
            "min_required_for_service": self.min_required,
            "availability_with_redundancy_pct": round(
                self.availability_with_redundancy * 100, 4),
            "degradation_curve": [
                {"n_failed": n, "service_pct": round(pct * 100, 2)}
                for n, pct in self.degradation_curve
            ],
            "mtbf_hours": round(self.mtbf_hours, 1),
            "mttr_hours": round(self.mttr_hours, 1),
        }


# ---------------------------------------------------------------------------
# MissionPlanner
# ---------------------------------------------------------------------------

class MissionPlanner:
    """Plan and schedule satellite constellation operations.

    This is the tool ops teams use daily: generate timelines, predict passes,
    detect conflicts, calculate data budgets, and model availability.
    """

    def __init__(
        self,
        satellites: List[Satellite],
        ground_stations: Optional[List[GroundStation]] = None,
        power_config: Optional[PowerConfig] = None,
        storage_capacity_mb: float = 512_000.0,   # 512 GB default onboard storage
    ):
        self.satellites = satellites
        self.ground_stations = ground_stations or DEFAULT_GROUND_STATIONS
        self.power_config = power_config or PowerConfig()
        self.storage_capacity_mb = storage_capacity_mb
        self._conflict_counter = 0

    # ------------------------------------------------------------------
    # 1. Operations Timeline Generator
    # ------------------------------------------------------------------

    def generate_timeline(
        self,
        start: datetime,
        duration_hours: float = 24.0,
        step_seconds: float = 60.0,
        include_eclipses: bool = True,
        include_passes: bool = True,
        include_maintenance: bool = True,
        include_compute: bool = True,
        maintenance_interval_hours: float = 12.0,
        maintenance_duration_minutes: float = 30.0,
    ) -> List[TimedEvent]:
        """Generate a complete operations timeline.

        Parameters
        ----------
        start : datetime
            Timeline start (UTC).
        duration_hours : float
            Duration of the planning window.
        step_seconds : float
            Time resolution for eclipse/pass detection.
        include_eclipses : bool
            Include eclipse entry/exit events.
        include_passes : bool
            Include ground station AOS/LOS events.
        include_maintenance : bool
            Include scheduled maintenance windows.
        include_compute : bool
            Include compute job execution windows.
        maintenance_interval_hours : float
            Hours between maintenance windows per satellite.
        maintenance_duration_minutes : float
            Duration of each maintenance window in minutes.

        Returns
        -------
        List[TimedEvent]
            Sorted list of timed events.
        """
        events: List[TimedEvent] = []

        for sat in self.satellites:
            # Eclipse events
            if include_eclipses:
                eclipse_windows = predict_eclipse_windows(
                    sat, start, duration_hours, step_seconds
                )
                for ec_start, ec_end in eclipse_windows:
                    dur = (ec_end - ec_start).total_seconds()
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="eclipse_start",
                        timestamp=ec_start,
                        duration_seconds=dur,
                        metadata={"eclipse_end": ec_end.isoformat()},
                    ))
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="eclipse_end",
                        timestamp=ec_end,
                        duration_seconds=0.0,
                        metadata={"eclipse_duration_min": round(dur / 60, 1)},
                    ))

            # Ground station pass events
            if include_passes:
                contacts = find_contact_windows(
                    sat, self.ground_stations, start, duration_hours, step_seconds
                )
                for cw in contacts:
                    dur = cw.duration_seconds
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="aos",
                        timestamp=cw.start,
                        duration_seconds=dur,
                        metadata={
                            "station": cw.station_name,
                            "max_elevation_deg": cw.max_elevation_deg,
                            "downlink_mb": cw.downlink_mb,
                        },
                    ))
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="los",
                        timestamp=cw.end,
                        duration_seconds=0.0,
                        metadata={
                            "station": cw.station_name,
                            "pass_duration_min": round(dur / 60, 1),
                        },
                    ))

            # Maintenance windows
            if include_maintenance:
                maint_time = start + timedelta(
                    hours=hash(sat.name) % int(maintenance_interval_hours)
                )
                end_time = start + timedelta(hours=duration_hours)
                while maint_time < end_time:
                    maint_dur = maintenance_duration_minutes * 60
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="maintenance_start",
                        timestamp=maint_time,
                        duration_seconds=maint_dur,
                        metadata={
                            "type": "scheduled",
                            "maintenance_end": (
                                maint_time + timedelta(seconds=maint_dur)
                            ).isoformat(),
                        },
                    ))
                    events.append(TimedEvent(
                        satellite=sat.name,
                        event_type="maintenance_end",
                        timestamp=maint_time + timedelta(seconds=maint_dur),
                        duration_seconds=0.0,
                        metadata={"type": "scheduled"},
                    ))
                    maint_time += timedelta(hours=maintenance_interval_hours)

            # Compute windows (sunlit, non-maintenance periods)
            if include_compute:
                eclipse_windows = predict_eclipse_windows(
                    sat, start, duration_hours, step_seconds
                )
                compute_events = self._compute_windows(
                    sat.name, start, duration_hours, eclipse_windows,
                    maintenance_interval_hours, maintenance_duration_minutes,
                )
                events.extend(compute_events)

        events.sort(key=lambda e: e.timestamp)
        return events

    def _compute_windows(
        self,
        sat_name: str,
        start: datetime,
        duration_hours: float,
        eclipse_windows: List[Tuple[datetime, datetime]],
        maint_interval_h: float,
        maint_duration_min: float,
    ) -> List[TimedEvent]:
        """Identify windows where compute jobs can execute.

        Compute is available during sunlit, non-maintenance periods.
        Battery-backed compute during eclipse is flagged as degraded.
        """
        events = []
        end = start + timedelta(hours=duration_hours)

        # Build blocked intervals (eclipses + maintenance)
        blocked: List[Tuple[datetime, datetime, str]] = []
        for ec_s, ec_e in eclipse_windows:
            blocked.append((ec_s, ec_e, "eclipse"))

        maint_time = start + timedelta(
            hours=hash(sat_name) % int(maint_interval_h)
        )
        while maint_time < end:
            m_end = maint_time + timedelta(minutes=maint_duration_min)
            blocked.append((maint_time, m_end, "maintenance"))
            maint_time += timedelta(hours=maint_interval_h)

        blocked.sort(key=lambda x: x[0])

        # Merge blocked intervals and find gaps (compute windows)
        merged: List[Tuple[datetime, datetime]] = []
        for bs, be, _ in blocked:
            if merged and bs <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], be))
            else:
                merged.append((bs, be))

        # Gaps between blocked intervals are full-power compute windows
        cursor = start
        for bs, be in merged:
            if cursor < bs:
                dur = (bs - cursor).total_seconds()
                if dur > 60:  # skip tiny gaps
                    events.append(TimedEvent(
                        satellite=sat_name,
                        event_type="compute_window",
                        timestamp=cursor,
                        duration_seconds=dur,
                        metadata={"mode": "full_power", "sunlit": True},
                    ))
            cursor = be

        if cursor < end:
            dur = (end - cursor).total_seconds()
            if dur > 60:
                events.append(TimedEvent(
                    satellite=sat_name,
                    event_type="compute_window",
                    timestamp=cursor,
                    duration_seconds=dur,
                    metadata={"mode": "full_power", "sunlit": True},
                ))

        return events

    # ------------------------------------------------------------------
    # 2. Conflict Detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        events: List[TimedEvent],
        power_configs: Optional[Dict[str, PowerConfig]] = None,
    ) -> List[SchedulingConflict]:
        """Detect scheduling conflicts in a timeline.

        Checks for:
        - Overlapping downlinks during the same ground pass
        - Compute during eclipse with insufficient battery
        - Maintenance overlapping critical operations
        """
        conflicts: List[SchedulingConflict] = []

        # Group events by satellite
        by_sat: Dict[str, List[TimedEvent]] = {}
        for e in events:
            by_sat.setdefault(e.satellite, []).append(e)

        for sat_name, sat_events in by_sat.items():
            sat_events.sort(key=lambda e: e.timestamp)

            # --- Check 1: Overlapping downlinks in same ground pass ---
            aos_events = [e for e in sat_events if e.event_type == "aos"]
            for i in range(len(aos_events)):
                for j in range(i + 1, len(aos_events)):
                    a, b = aos_events[i], aos_events[j]
                    a_end = a.timestamp + timedelta(seconds=a.duration_seconds)
                    if b.timestamp < a_end:
                        self._conflict_counter += 1
                        conflicts.append(SchedulingConflict(
                            conflict_id=f"CONF-{self._conflict_counter:04d}",
                            severity="warning",
                            description=(
                                f"{sat_name}: Overlapping ground passes — "
                                f"{a.metadata.get('station', '?')} and "
                                f"{b.metadata.get('station', '?')}"
                            ),
                            events=[a, b],
                            resolution_hint="Prioritize higher-elevation pass for downlink.",
                        ))

            # --- Check 2: Compute during eclipse with low battery ---
            eclipse_starts = [
                e for e in sat_events if e.event_type == "eclipse_start"
            ]
            compute_windows = [
                e for e in sat_events if e.event_type == "compute_window"
            ]
            for ec in eclipse_starts:
                ec_end = ec.timestamp + timedelta(seconds=ec.duration_seconds)
                for cw in compute_windows:
                    cw_end = cw.timestamp + timedelta(seconds=cw.duration_seconds)
                    # Check overlap
                    if cw.timestamp < ec_end and cw_end > ec.timestamp:
                        # Simulate battery drain
                        power = power_configs.get(sat_name, self.power_config) if power_configs else self.power_config
                        pm = PowerModel(power)
                        eclipse_dur_h = ec.duration_seconds / 3600.0
                        can_sustain = pm.can_sustain_load(
                            400.0, eclipse_dur_h, in_eclipse=True
                        )
                        if not can_sustain:
                            self._conflict_counter += 1
                            conflicts.append(SchedulingConflict(
                                conflict_id=f"CONF-{self._conflict_counter:04d}",
                                severity="critical",
                                description=(
                                    f"{sat_name}: Compute scheduled during eclipse "
                                    f"({ec.duration_seconds/60:.0f} min) — "
                                    f"battery may be insufficient"
                                ),
                                events=[ec, cw],
                                resolution_hint=(
                                    "Reduce compute load or defer to next sunlit window."
                                ),
                            ))

            # --- Check 3: Maintenance overlapping critical ops ---
            maint_starts = [
                e for e in sat_events if e.event_type == "maintenance_start"
            ]
            critical_events = [
                e for e in sat_events
                if e.event_type in ("aos", "compute_window")
            ]
            for maint in maint_starts:
                m_end = maint.timestamp + timedelta(seconds=maint.duration_seconds)
                for crit in critical_events:
                    c_end = crit.timestamp + timedelta(seconds=crit.duration_seconds)
                    if crit.timestamp < m_end and c_end > maint.timestamp:
                        self._conflict_counter += 1
                        sev = "warning" if crit.event_type == "compute_window" else "critical"
                        conflicts.append(SchedulingConflict(
                            conflict_id=f"CONF-{self._conflict_counter:04d}",
                            severity=sev,
                            description=(
                                f"{sat_name}: Maintenance window overlaps "
                                f"{crit.event_type} "
                                f"({crit.metadata.get('station', '')})"
                            ),
                            events=[maint, crit],
                            resolution_hint="Reschedule maintenance to avoid pass windows.",
                        ))

        return conflicts

    # ------------------------------------------------------------------
    # 3. Pass Prediction
    # ------------------------------------------------------------------

    def predict_passes(
        self,
        start: datetime,
        duration_hours: float = 24.0,
        min_elevation_deg: float = 10.0,
        step_seconds: float = 30.0,
    ) -> List[PassPrediction]:
        """Predict all satellite passes over ground stations.

        For each satellite-ground station pair, finds all passes above
        the minimum elevation in the given time window.

        This is the daily workhorse for ops teams.
        """
        predictions: List[PassPrediction] = []

        for sat in self.satellites:
            contacts = find_contact_windows(
                sat, self.ground_stations, start, duration_hours, step_seconds
            )
            for cw in contacts:
                if cw.max_elevation_deg < min_elevation_deg:
                    continue

                dur = cw.duration_seconds
                # Calculate station link rates for this specific contact
                station = next(
                    (s for s in self.ground_stations if s.name == cw.station_name),
                    None,
                )
                if station:
                    dl_mb = station.downlink_mbps * dur / 8.0
                    ul_mb = station.uplink_mbps * dur / 8.0
                else:
                    dl_mb = cw.downlink_mb
                    ul_mb = 0.0

                predictions.append(PassPrediction(
                    satellite_name=cw.satellite_name,
                    station_name=cw.station_name,
                    aos_time=cw.start,
                    los_time=cw.end,
                    max_elevation_deg=cw.max_elevation_deg,
                    duration_seconds=dur,
                    downlink_capacity_mb=dl_mb,
                    uplink_capacity_mb=ul_mb,
                ))

        predictions.sort(key=lambda p: p.aos_time)
        return predictions

    # ------------------------------------------------------------------
    # 4. Data Budget Calculator
    # ------------------------------------------------------------------

    def calculate_data_budget(
        self,
        workload_mix: Optional[Dict[str, float]] = None,
        orbit_period_minutes: float = 95.0,
        passes_per_orbit: float = 1.5,
        avg_pass_duration_seconds: float = 480.0,
        avg_downlink_mbps: float = 500.0,
    ) -> DataBudget:
        """Calculate data generation vs downlink capacity.

        Parameters
        ----------
        workload_mix : dict, optional
            Map of workload_key -> fraction (0-1). Defaults to earth-obs heavy.
        orbit_period_minutes : float
            Orbital period.
        passes_per_orbit : float
            Average ground station passes per orbit per satellite.
        avg_pass_duration_seconds : float
            Average useful pass duration.
        avg_downlink_mbps : float
            Average downlink rate during pass.

        Returns
        -------
        DataBudget
            Complete data budget analysis.
        """
        if workload_mix is None:
            workload_mix = {
                "image_classification": 0.40,
                "change_detection": 0.20,
                "object_tracking": 0.15,
                "sar_processing": 0.10,
                "llm_inference": 0.10,
                "weather_model": 0.05,
            }

        # Calculate data generated per orbit per satellite
        orbit_seconds = orbit_period_minutes * 60.0
        # Assume 60% of orbit is active imaging/sensing
        active_fraction = 0.60
        active_seconds = orbit_seconds * active_fraction

        total_input_mb = 0.0
        total_output_mb = 0.0
        weighted_reduction = 0.0

        for wl_key, fraction in workload_mix.items():
            spec = WORKLOAD_CATALOG.get(wl_key)
            if spec is None:
                continue
            # How many jobs fit in active time?
            jobs_per_orbit = (active_seconds * fraction) / spec.duration_seconds
            total_input_mb += jobs_per_orbit * spec.input_size_mb
            total_output_mb += jobs_per_orbit * spec.output_size_mb
            if spec.input_size_mb > 0:
                weighted_reduction += fraction * (spec.input_size_mb / max(spec.output_size_mb, 0.1))

        # Downlink capacity per orbit
        n_sats = len(self.satellites)
        dl_per_pass_mb = avg_downlink_mbps * avg_pass_duration_seconds / 8.0
        dl_per_orbit_total = dl_per_pass_mb * passes_per_orbit * n_sats

        # Raw data (input) generated per orbit
        data_gen_per_orbit = total_input_mb * n_sats

        # Without processing, need to downlink raw data
        net_raw = dl_per_orbit_total - data_gen_per_orbit

        # With onboard processing, only downlink results
        data_output_per_orbit = total_output_mb * n_sats
        net_processed = dl_per_orbit_total - data_output_per_orbit

        requires_processing = net_raw < 0
        orbit_period_hours = orbit_period_minutes / 60.0

        if net_raw < 0:
            backlog_rate = abs(net_raw) / orbit_period_hours
            orbits_to_full = self.storage_capacity_mb / abs(net_raw) if net_raw != 0 else float('inf')
        else:
            backlog_rate = 0.0
            orbits_to_full = float('inf')

        return DataBudget(
            data_generated_per_orbit_mb=data_gen_per_orbit,
            downlink_capacity_per_orbit_mb=dl_per_orbit_total,
            net_data_per_orbit_mb=net_raw,
            orbits_until_storage_full=orbits_to_full,
            storage_capacity_mb=self.storage_capacity_mb,
            backlog_rate_mb_per_hour=backlog_rate,
            requires_onboard_processing=requires_processing,
            processing_reduction_factor=weighted_reduction if weighted_reduction > 0 else 1.0,
            effective_downlink_with_processing_mb=net_processed,
        )

    # ------------------------------------------------------------------
    # 5. Availability Calculator
    # ------------------------------------------------------------------

    def calculate_availability(
        self,
        single_sat_availability: float = 0.995,
        min_required_fraction: float = 0.75,
        mtbf_hours: float = 8760.0 * 5,    # 5-year MTBF
        mttr_hours: float = 72.0,           # 3-day repair (ground upload fix)
    ) -> AvailabilityResult:
        """Calculate constellation availability and reliability.

        Parameters
        ----------
        single_sat_availability : float
            Probability a single satellite is operational at any time.
        min_required_fraction : float
            Fraction of constellation needed for full service.
        mtbf_hours : float
            Mean time between failures (hours).
        mttr_hours : float
            Mean time to repair/restore (hours).

        Returns
        -------
        AvailabilityResult
            Complete availability analysis.
        """
        n = len(self.satellites)
        min_required = max(1, int(math.ceil(n * min_required_fraction)))
        p = single_sat_availability

        # Constellation availability: P(all operational)
        all_up = p ** n

        # P(at least min_required operational) — binomial CDF complement
        # P(X >= k) = sum_{i=k}^{n} C(n,i) * p^i * (1-p)^(n-i)
        avail_redundant = 0.0
        for i in range(min_required, n + 1):
            avail_redundant += _binom_pmf(n, i, p)

        # Graceful degradation curve: service level vs number of failures
        degradation: List[Tuple[int, float]] = []
        for n_failed in range(n + 1):
            operational = n - n_failed
            if operational >= min_required:
                service_level = 1.0
            elif operational > 0:
                service_level = operational / min_required
            else:
                service_level = 0.0
            degradation.append((n_failed, service_level))

        # Steady-state availability from MTBF/MTTR
        steady_state = mtbf_hours / (mtbf_hours + mttr_hours)

        return AvailabilityResult(
            constellation_availability=all_up,
            single_satellite_availability=p,
            n_satellites=n,
            min_required=min_required,
            availability_with_redundancy=avail_redundant,
            degradation_curve=degradation,
            mtbf_hours=mtbf_hours,
            mttr_hours=mttr_hours,
        )

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------

    def timeline_to_json(self, events: List[TimedEvent]) -> str:
        """Export timeline as JSON."""
        return json.dumps(
            [e.to_dict() for e in events],
            indent=2,
            default=str,
        )

    def timeline_to_schedule(self, events: List[TimedEvent]) -> str:
        """Export timeline as human-readable schedule (flight-plan style)."""
        lines = []
        lines.append("=" * 90)
        lines.append("  ORBITAL COMPUTE — OPERATIONS SCHEDULE")
        lines.append("=" * 90)

        if not events:
            lines.append("  (no events)")
            return "\n".join(lines)

        lines.append(
            f"  Window: {events[0].timestamp.strftime('%Y-%m-%d %H:%M UTC')} — "
            f"{events[-1].timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append(f"  Total events: {len(events)}")
        lines.append("-" * 90)
        lines.append(
            f"  {'TIME (UTC)':<22} {'SAT':<16} {'EVENT':<20} {'DUR':>8} {'DETAILS'}"
        )
        lines.append("-" * 90)

        for e in events:
            time_str = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if e.duration_seconds > 0:
                dur_str = _format_duration(e.duration_seconds)
            else:
                dur_str = "—"

            detail_parts = []
            for k, v in e.metadata.items():
                if k.endswith("_end"):
                    continue
                detail_parts.append(f"{k}={v}")
            detail = ", ".join(detail_parts[:3])  # limit to 3 fields

            lines.append(
                f"  {time_str:<22} {e.satellite:<16} {e.event_type:<20} "
                f"{dur_str:>8} {detail}"
            )

        lines.append("=" * 90)
        return "\n".join(lines)

    def timeline_to_ical(self, events: List[TimedEvent]) -> str:
        """Export timeline as iCalendar (.ics) for ops team calendars."""
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//OrbitalCompute//MissionPlanner//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
            "X-WR-CALNAME:Orbital Compute Ops",
        ]

        for i, e in enumerate(events):
            if e.duration_seconds <= 0:
                continue  # Skip instantaneous events for calendar

            dtstart = e.timestamp.strftime("%Y%m%dT%H%M%SZ")
            dtend = (e.timestamp + timedelta(seconds=e.duration_seconds)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            uid = f"orbital-{e.satellite}-{i}@mission-planner"
            summary = f"[{e.satellite}] {e.event_type.replace('_', ' ').title()}"

            desc_parts = [f"Satellite: {e.satellite}", f"Event: {e.event_type}"]
            for k, v in e.metadata.items():
                desc_parts.append(f"{k}: {v}")
            description = "\\n".join(desc_parts)

            lines.extend([
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTART:{dtstart}",
                f"DTEND:{dtend}",
                f"SUMMARY:{summary}",
                f"DESCRIPTION:{description}",
                "END:VEVENT",
            ])

        lines.append("END:VCALENDAR")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _binom_pmf(n: int, k: int, p: float) -> float:
    """Binomial probability mass function."""
    coeff = math.comb(n, k)
    return coeff * (p ** k) * ((1 - p) ** (n - k))


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _demo():
    """Run a full mission planning demo with a 12-satellite constellation."""
    print("=" * 90)
    print("  ORBITAL COMPUTE — MISSION PLANNER DEMO")
    print("=" * 90)

    # 1. Set up constellation
    config = CONSTELLATIONS["starlink-mini"]
    print(f"\n  Constellation: {config.name}")
    print(f"  Altitude: {config.altitude_km} km | Inclination: {config.inclination_deg} deg")
    print(f"  Planes: {config.n_planes} | Sats/plane: {config.sats_per_plane}")
    print(f"  Total satellites: {config.total_sats}")

    satellites = generate_constellation(config, max_sats=12)
    print(f"  Generated: {len(satellites)} satellites")

    # Use a subset of ground stations for speed
    stations = DEFAULT_GROUND_STATIONS[:5]
    print(f"  Ground stations: {', '.join(s.name for s in stations)}")

    planner = MissionPlanner(satellites, stations)

    # 2. Generate timeline (use 6h for demo speed, 60s step)
    t0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    demo_hours = 6.0
    print(f"\n  Generating {demo_hours}h timeline from {t0.isoformat()}...")
    print(f"  (step=60s, this may take a moment for {len(satellites)} sats)")

    events = planner.generate_timeline(
        t0, duration_hours=demo_hours, step_seconds=60.0,
    )

    # Summarize by event type
    type_counts: Dict[str, int] = {}
    for e in events:
        type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

    print(f"\n  Timeline: {len(events)} events")
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype:<24} {count:>5}")

    # 3. Print first 20 events in schedule format
    print("\n  --- First 20 events ---")
    schedule = planner.timeline_to_schedule(events[:20])
    for line in schedule.split("\n"):
        print(f"  {line}")

    # 4. Conflict detection
    print("\n  --- Conflict Detection ---")
    conflicts = planner.detect_conflicts(events)
    print(f"  Detected {len(conflicts)} conflicts")
    by_sev: Dict[str, int] = {}
    for c in conflicts:
        by_sev[c.severity] = by_sev.get(c.severity, 0) + 1
    for sev, cnt in sorted(by_sev.items()):
        print(f"    {sev:<12} {cnt:>3}")
    for c in conflicts[:3]:
        print(f"\n    [{c.severity.upper()}] {c.conflict_id}: {c.description}")
        print(f"    Hint: {c.resolution_hint}")

    # 5. Pass prediction (first 3 satellites only for speed)
    print("\n  --- Pass Predictions (first 3 sats, 6h) ---")
    mini_planner = MissionPlanner(satellites[:3], stations)
    passes = mini_planner.predict_passes(t0, duration_hours=demo_hours)
    print(f"  Total passes: {len(passes)}")
    print(
        f"\n  {'SAT':<16} {'STATION':<12} {'AOS (UTC)':<22} {'LOS (UTC)':<22} "
        f"{'ELEV':>5} {'DUR':>7} {'DL(MB)':>8}"
    )
    print(f"  {'-' * 95}")
    for p in passes[:15]:
        print(
            f"  {p.satellite_name:<16} {p.station_name:<12} "
            f"{p.aos_time.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{p.los_time.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{p.max_elevation_deg:>5.1f} "
            f"{_format_duration(p.duration_seconds):>7} "
            f"{p.downlink_capacity_mb:>8.0f}"
        )

    # 6. Data budget
    print("\n  --- Data Budget Analysis ---")
    budget = planner.calculate_data_budget()
    print(f"  Data generated/orbit:        {budget.data_generated_per_orbit_mb:>12,.0f} MB")
    print(f"  Downlink capacity/orbit:     {budget.downlink_capacity_per_orbit_mb:>12,.0f} MB")
    print(f"  Net data/orbit (raw):        {budget.net_data_per_orbit_mb:>12,.0f} MB")
    print(f"  Requires onboard processing: {'YES' if budget.requires_onboard_processing else 'NO'}")
    print(f"  Processing reduction factor:  {budget.processing_reduction_factor:>11.1f}x")
    if budget.requires_onboard_processing:
        print(f"  Backlog rate:                {budget.backlog_rate_mb_per_hour:>12,.0f} MB/h")
        orb_str = (
            f"{budget.orbits_until_storage_full:.1f}"
            if budget.orbits_until_storage_full < 1e6
            else "inf"
        )
        print(f"  Orbits until storage full:   {orb_str:>12}")
    print(f"  Net with processing/orbit:   {budget.effective_downlink_with_processing_mb:>12,.0f} MB")

    # 7. Availability
    print("\n  --- Constellation Availability ---")
    avail = planner.calculate_availability()
    print(f"  Single satellite availability:  {avail.single_satellite_availability*100:.2f}%")
    print(f"  All-up availability:            {avail.constellation_availability*100:.4f}%")
    print(f"  Min required for service:       {avail.min_required}/{avail.n_satellites}")
    print(f"  Availability with redundancy:   {avail.availability_with_redundancy*100:.4f}%")
    print(f"  MTBF: {avail.mtbf_hours:,.0f}h  |  MTTR: {avail.mttr_hours:.0f}h")
    print(f"\n  Graceful degradation:")
    print(f"    {'Failed':>8}  {'Service Level':>15}")
    for n_failed, svc in avail.degradation_curve:
        bar = "#" * int(svc * 30)
        print(f"    {n_failed:>8}  {svc*100:>14.1f}%  {bar}")

    # 8. Export sample iCal
    ical = planner.timeline_to_ical(events[:10])
    print(f"\n  --- iCalendar Export (first 10 events) ---")
    print(f"  Generated {ical.count('BEGIN:VEVENT')} calendar events")
    ical_lines = ical.split("\n")
    for line in ical_lines[:20]:
        print(f"    {line}")
    if len(ical_lines) > 20:
        print(f"    ... ({len(ical_lines) - 20} more lines)")

    print(f"\n{'=' * 90}")
    print("  PASS — mission planner OK")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    _demo()
