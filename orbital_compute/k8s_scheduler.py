"""Kubernetes scheduler extender for orbit-aware pod placement.

Implements the K8s scheduler extender protocol:
- POST /filter  — remove nodes that can't run a pod (power, thermal, connectivity)
- POST /prioritize — score remaining nodes by orbital favorability
- GET /health — liveness check

Runs as a standalone HTTP server using only stdlib. No Flask dependency.

How it works:
1. K8s scheduler sends a JSON payload with a pod spec and candidate nodes
2. We map K8s node names to satellites in our constellation model
3. We evaluate orbital state (eclipse, battery, thermal, ground contact)
4. We return filtered/scored nodes back to K8s

Pod annotations control orbital constraints:
    orbital-compute/min-battery: "30"          # Min battery % to schedule
    orbital-compute/prefer-sunlit: "true"      # Prefer sunlit satellites
    orbital-compute/max-eclipse-start: "30m"   # Don't start if eclipse in <30min
    orbital-compute/needs-ground-contact: "true"  # Needs downlink capability
    orbital-compute/power-watts: "500"         # GPU power requirement

Compatible with: Starcloud (Crusoe Cloud K8s), Axiom (MicroShift), Kepler (cloud-native).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orbital state snapshot (decoupled from sgp4 so tests work without numpy)
# ---------------------------------------------------------------------------

@dataclass
class SatelliteState:
    """Snapshot of a satellite's orbital and subsystem state."""
    name: str
    in_eclipse: bool = False
    battery_pct: float = 80.0          # 0-100
    temp_c: float = 35.0               # Current temperature
    max_temp_c: float = 75.0           # Thermal throttle limit
    solar_output_w: float = 2000.0     # Current solar panel output
    available_power_w: float = 1500.0  # Power available for compute
    time_to_eclipse_min: float = 45.0  # Minutes until next eclipse (inf if N/A)
    time_to_sunlight_min: float = 0.0  # Minutes until next sunlight (0 if sunlit)
    ground_contact: bool = False       # Currently in ground station window
    ground_contact_in_min: float = 30.0  # Minutes until next ground contact
    running_pods: int = 0              # Number of pods already scheduled


@dataclass
class ConstellationState:
    """Current state of the entire constellation, refreshed periodically."""
    satellites: Dict[str, SatelliteState] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def get(self, node_name: str) -> Optional[SatelliteState]:
        """Look up satellite state by K8s node name."""
        return self.satellites.get(node_name)


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r'^(\d+)(s|m|h)$')


def _parse_duration_minutes(s: str) -> float:
    """Parse a duration string like '30m', '2h', '120s' into minutes."""
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise ValueError(f"Invalid duration: {s!r}  (expected e.g. '30m', '2h', '120s')")
    val = int(m.group(1))
    unit = m.group(2)
    if unit == 's':
        return val / 60.0
    elif unit == 'm':
        return float(val)
    elif unit == 'h':
        return val * 60.0
    return float(val)


@dataclass
class PodOrbitalConstraints:
    """Orbital constraints extracted from pod annotations."""
    min_battery_pct: float = 20.0
    prefer_sunlit: bool = True
    max_eclipse_start_min: Optional[float] = None  # Don't schedule if eclipse in < N min
    needs_ground_contact: bool = False
    power_watts: float = 0.0  # Required compute power

    @classmethod
    def from_annotations(cls, annotations: Dict[str, str]) -> "PodOrbitalConstraints":
        """Parse pod annotations into orbital constraints."""
        c = cls()
        prefix = "orbital-compute/"

        raw = annotations.get(prefix + "min-battery", "")
        if raw:
            c.min_battery_pct = float(raw)

        raw = annotations.get(prefix + "prefer-sunlit", "")
        if raw:
            c.prefer_sunlit = raw.lower() in ("true", "1", "yes")

        raw = annotations.get(prefix + "max-eclipse-start", "")
        if raw:
            c.max_eclipse_start_min = _parse_duration_minutes(raw)

        raw = annotations.get(prefix + "needs-ground-contact", "")
        if raw:
            c.needs_ground_contact = raw.lower() in ("true", "1", "yes")

        raw = annotations.get(prefix + "power-watts", "")
        if raw:
            c.power_watts = float(raw)

        return c


# ---------------------------------------------------------------------------
# Scheduler extender logic
# ---------------------------------------------------------------------------

class OrbitalSchedulerExtender:
    """K8s scheduler extender that scores nodes based on orbital state.

    Implements the two-phase K8s extender protocol:
      filter     — hard constraints: remove nodes that CAN'T run the pod
      prioritize — soft constraints: score remaining nodes for best placement
    """

    # Scoring weights (sum to ~100 for readability)
    W_SUNLIT = 25        # Prefer sunlit satellites
    W_BATTERY = 20       # Higher battery = better
    W_THERMAL = 15       # More thermal headroom = better
    W_ECLIPSE_MARGIN = 15  # More time until eclipse = better
    W_GROUND = 10        # Ground contact soon = better (for data jobs)
    W_LOAD_BALANCE = 15  # Fewer running pods = better

    MAX_SCORE = 100      # K8s extender priority scores are 0-100

    def __init__(self, state: Optional[ConstellationState] = None):
        self.state = state or ConstellationState()
        self._lock = threading.Lock()

    def update_state(self, state: ConstellationState) -> None:
        """Thread-safe state update (called by background refresh loop)."""
        with self._lock:
            self.state = state

    def _get_state(self) -> ConstellationState:
        with self._lock:
            return self.state

    # ------------------------------------------------------------------
    # Filter phase — hard constraints
    # ------------------------------------------------------------------

    def filter(self, pod: Dict[str, Any], node_names: List[str]) -> Dict[str, Any]:
        """Remove nodes that cannot run this pod.

        Args:
            pod: K8s pod spec (from scheduler extender request)
            node_names: List of candidate node names

        Returns:
            K8s ExtenderFilterResult: {
                "Nodes": { "items": [...] },
                "FailedNodes": { "node-name": "reason", ... }
            }
        """
        annotations = (pod.get("metadata", {}).get("annotations") or {})
        constraints = PodOrbitalConstraints.from_annotations(annotations)
        state = self._get_state()

        passed = []
        failed = {}

        for node_name in node_names:
            sat = state.get(node_name)
            if sat is None:
                # Unknown node — let it through (might be a ground node)
                passed.append(node_name)
                continue

            reason = self._check_hard_constraints(sat, constraints)
            if reason:
                failed[node_name] = reason
            else:
                passed.append(node_name)

        return {
            "Nodes": {"items": [{"metadata": {"name": n}} for n in passed]},
            "FailedNodes": failed,
        }

    def _check_hard_constraints(self, sat: SatelliteState,
                                 constraints: PodOrbitalConstraints) -> Optional[str]:
        """Return failure reason string, or None if node passes all checks."""

        # 1. Battery too low
        if sat.battery_pct < constraints.min_battery_pct:
            return (f"battery {sat.battery_pct:.1f}% below minimum "
                    f"{constraints.min_battery_pct:.1f}%")

        # 2. Over thermal limit
        if sat.temp_c >= sat.max_temp_c:
            return (f"temperature {sat.temp_c:.1f}C exceeds limit "
                    f"{sat.max_temp_c:.1f}C")

        # 3. Not enough power
        if constraints.power_watts > 0 and sat.available_power_w < constraints.power_watts:
            return (f"available power {sat.available_power_w:.0f}W insufficient "
                    f"for {constraints.power_watts:.0f}W requirement")

        # 4. Eclipse imminent and pod doesn't want that
        if (constraints.max_eclipse_start_min is not None
                and sat.time_to_eclipse_min < constraints.max_eclipse_start_min
                and not sat.in_eclipse):
            return (f"eclipse in {sat.time_to_eclipse_min:.0f}min, pod requires "
                    f">={constraints.max_eclipse_start_min:.0f}min margin")

        # 5. Needs ground contact but node is eclipsed with no contact window
        if constraints.needs_ground_contact and not sat.ground_contact:
            # If there's no contact for a long time, filter it out
            if sat.ground_contact_in_min > 60:
                return (f"no ground contact for {sat.ground_contact_in_min:.0f}min, "
                        f"pod requires ground contact")

        return None

    # ------------------------------------------------------------------
    # Prioritize phase — soft scoring
    # ------------------------------------------------------------------

    def prioritize(self, pod: Dict[str, Any],
                   node_names: List[str]) -> List[Dict[str, Any]]:
        """Score nodes that passed filtering.

        Args:
            pod: K8s pod spec
            node_names: Nodes that passed the filter phase

        Returns:
            K8s HostPriorityList: [{"Host": "node-name", "Score": 0-100}, ...]
        """
        annotations = (pod.get("metadata", {}).get("annotations") or {})
        constraints = PodOrbitalConstraints.from_annotations(annotations)
        state = self._get_state()

        results = []
        for node_name in node_names:
            sat = state.get(node_name)
            if sat is None:
                # Unknown node gets a neutral score
                results.append({"Host": node_name, "Score": 50})
                continue

            score = self._score_node(sat, constraints)
            results.append({"Host": node_name, "Score": score})

        return results

    def _score_node(self, sat: SatelliteState,
                    constraints: PodOrbitalConstraints) -> int:
        """Compute a 0-100 priority score for a node."""
        score = 0.0

        # 1. Sunlit vs eclipse (PHOENIX principle: sunlit is free power)
        if constraints.prefer_sunlit:
            if not sat.in_eclipse:
                score += self.W_SUNLIT
            else:
                # Partial credit if near end of eclipse
                if sat.time_to_sunlight_min < 10:
                    score += self.W_SUNLIT * 0.5
        else:
            score += self.W_SUNLIT * 0.5  # Neutral if not preferred

        # 2. Battery level (linear scale)
        battery_score = min(sat.battery_pct / 100.0, 1.0)
        score += self.W_BATTERY * battery_score

        # 3. Thermal headroom (how far below thermal limit)
        thermal_headroom = max(0.0, sat.max_temp_c - sat.temp_c)
        thermal_range = sat.max_temp_c - (-20.0)  # Min to max operating range
        thermal_score = min(thermal_headroom / thermal_range, 1.0) if thermal_range > 0 else 0.0
        score += self.W_THERMAL * thermal_score

        # 4. Time until eclipse (more sunlight remaining = better)
        if not sat.in_eclipse:
            # LEO orbital period ~95 min, max sunlit ~60 min
            eclipse_margin_score = min(sat.time_to_eclipse_min / 60.0, 1.0)
            score += self.W_ECLIPSE_MARGIN * eclipse_margin_score
        else:
            # In eclipse: score by how soon sunlight returns
            if sat.time_to_sunlight_min < 35:  # Max eclipse ~35 min
                score += self.W_ECLIPSE_MARGIN * (1.0 - sat.time_to_sunlight_min / 35.0) * 0.3

        # 5. Ground station contact (relevant for data-dependent jobs)
        if constraints.needs_ground_contact:
            if sat.ground_contact:
                score += self.W_GROUND
            else:
                # Closer contact = better
                contact_score = max(0.0, 1.0 - sat.ground_contact_in_min / 60.0)
                score += self.W_GROUND * contact_score

        # 6. Load balance (fewer running pods = better)
        # Assume max ~8 GPU pods per node
        load_score = max(0.0, 1.0 - sat.running_pods / 8.0)
        score += self.W_LOAD_BALANCE * load_score

        return max(0, min(self.MAX_SCORE, int(round(score))))


# ---------------------------------------------------------------------------
# HTTP server (K8s extender protocol)
# ---------------------------------------------------------------------------

class ExtenderHandler(BaseHTTPRequestHandler):
    """HTTP handler implementing the K8s scheduler extender protocol."""

    extender: OrbitalSchedulerExtender  # Set by server factory

    def do_GET(self) -> None:
        if self.path == "/health" or self.path == "/healthz":
            self._respond_json(200, {"status": "ok"})
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._respond_json(400, {"error": f"invalid JSON: {e}"})
            return

        if self.path == "/filter":
            self._handle_filter(data)
        elif self.path == "/prioritize":
            self._handle_prioritize(data)
        else:
            self._respond_json(404, {"error": f"unknown endpoint: {self.path}"})

    def _handle_filter(self, data: Dict[str, Any]) -> None:
        """Handle POST /filter — K8s ExtenderArgs -> ExtenderFilterResult."""
        pod = data.get("Pod", {})
        nodes = data.get("Nodes", {}).get("items", [])
        node_names = [n.get("metadata", {}).get("name", "") for n in nodes]

        result = self.extender.filter(pod, node_names)
        self._respond_json(200, result)

    def _handle_prioritize(self, data: Dict[str, Any]) -> None:
        """Handle POST /prioritize — K8s ExtenderArgs -> HostPriorityList."""
        pod = data.get("Pod", {})
        nodes = data.get("Nodes", {}).get("items", [])
        node_names = [n.get("metadata", {}).get("name", "") for n in nodes]

        result = self.extender.prioritize(pod, node_names)
        self._respond_json(200, result)

    def _respond_json(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(fmt, *args)


def create_server(extender: OrbitalSchedulerExtender,
                  host: str = "0.0.0.0", port: int = 8888) -> HTTPServer:
    """Create the extender HTTP server."""

    class Handler(ExtenderHandler):
        pass

    Handler.extender = extender  # type: ignore[attr-defined]

    server = HTTPServer((host, port), Handler)
    logger.info("Orbital scheduler extender listening on %s:%d", host, port)
    return server


# ---------------------------------------------------------------------------
# Integration with orbital_compute models (optional — uses sgp4/numpy)
# ---------------------------------------------------------------------------

def build_state_from_simulator(satellites, power_models, thermal_models,
                                ground_network=None,
                                now=None) -> ConstellationState:
    """Build ConstellationState from orbital_compute simulator objects.

    This bridges the K8s extender with the orbital_compute simulation engine.
    Only call this if sgp4 and numpy are available.

    Args:
        satellites: list of orbital_compute.orbit.Satellite
        power_models: dict mapping sat name -> PowerModel
        thermal_models: dict mapping sat name -> ThermalModel
        ground_network: optional GroundStationNetwork
        now: datetime (defaults to UTC now)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    state = ConstellationState(timestamp=now)

    for sat in satellites:
        pos = sat.position_at(now)
        pm = power_models.get(sat.name)
        tm = thermal_models.get(sat.name)

        ss = SatelliteState(
            name=sat.name,
            in_eclipse=pos.in_eclipse,
            battery_pct=(pm.battery_wh / pm.config.battery_capacity_wh * 100.0) if pm else 80.0,
            temp_c=(tm.temp_k - 273.15) if tm else 35.0,
            max_temp_c=tm.config.compute_temp_limit_c if tm else 75.0,
            solar_output_w=0.0 if pos.in_eclipse else (pm.config.solar_panel_watts if pm else 2000.0),
            available_power_w=0.0 if pos.in_eclipse else (
                pm.config.solar_panel_watts - pm.config.housekeeping_watts if pm else 1500.0
            ),
        )
        state.satellites[sat.name] = ss

    return state


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _build_demo_state() -> ConstellationState:
    """Build a mock constellation state for demo/testing."""
    state = ConstellationState(timestamp=datetime.now(timezone.utc))

    # Simulate 6 satellites in various orbital states
    sats = [
        SatelliteState(
            name="sat-alpha-01",
            in_eclipse=False, battery_pct=92.0, temp_c=35.0,
            max_temp_c=75.0, solar_output_w=2000.0,
            available_power_w=1500.0, time_to_eclipse_min=42.0,
            time_to_sunlight_min=0.0, ground_contact=True,
            ground_contact_in_min=0.0, running_pods=1,
        ),
        SatelliteState(
            name="sat-alpha-02",
            in_eclipse=False, battery_pct=78.0, temp_c=45.0,
            max_temp_c=75.0, solar_output_w=2000.0,
            available_power_w=1400.0, time_to_eclipse_min=12.0,
            time_to_sunlight_min=0.0, ground_contact=False,
            ground_contact_in_min=15.0, running_pods=3,
        ),
        SatelliteState(
            name="sat-beta-01",
            in_eclipse=True, battery_pct=55.0, temp_c=28.0,
            max_temp_c=75.0, solar_output_w=0.0,
            available_power_w=800.0, time_to_eclipse_min=0.0,
            time_to_sunlight_min=18.0, ground_contact=False,
            ground_contact_in_min=45.0, running_pods=0,
        ),
        SatelliteState(
            name="sat-beta-02",
            in_eclipse=True, battery_pct=15.0, temp_c=22.0,
            max_temp_c=75.0, solar_output_w=0.0,
            available_power_w=300.0, time_to_eclipse_min=0.0,
            time_to_sunlight_min=25.0, ground_contact=False,
            ground_contact_in_min=90.0, running_pods=0,
        ),
        SatelliteState(
            name="sat-gamma-01",
            in_eclipse=False, battery_pct=85.0, temp_c=72.0,
            max_temp_c=75.0, solar_output_w=2000.0,
            available_power_w=1500.0, time_to_eclipse_min=55.0,
            time_to_sunlight_min=0.0, ground_contact=True,
            ground_contact_in_min=0.0, running_pods=6,
        ),
        SatelliteState(
            name="sat-gamma-02",
            in_eclipse=False, battery_pct=88.0, temp_c=40.0,
            max_temp_c=75.0, solar_output_w=2000.0,
            available_power_w=1500.0, time_to_eclipse_min=50.0,
            time_to_sunlight_min=0.0, ground_contact=False,
            ground_contact_in_min=8.0, running_pods=2,
        ),
    ]

    for s in sats:
        state.satellites[s.name] = s

    return state


def _run_demo() -> None:
    """Run a self-test demonstrating filter and prioritize."""
    print("=" * 70)
    print("Orbital Compute — K8s Scheduler Extender Demo")
    print("=" * 70)

    state = _build_demo_state()
    extender = OrbitalSchedulerExtender(state)

    # --- Describe constellation state ---
    print("\nConstellation state:")
    print(f"  {'Satellite':<18} {'Eclipse':>7} {'Batt%':>6} {'Temp':>6} "
          f"{'Power':>7} {'Eclipse In':>10} {'GS Contact':>10} {'Pods':>5}")
    print("  " + "-" * 80)
    for name, sat in sorted(state.satellites.items()):
        eclipse_str = "YES" if sat.in_eclipse else "no"
        gs_str = "NOW" if sat.ground_contact else f"{sat.ground_contact_in_min:.0f}min"
        ecl_in = "in eclipse" if sat.in_eclipse else f"{sat.time_to_eclipse_min:.0f}min"
        print(f"  {name:<18} {eclipse_str:>7} {sat.battery_pct:>5.1f}% {sat.temp_c:>5.1f}C "
              f"{sat.available_power_w:>6.0f}W {ecl_in:>10} {gs_str:>10} {sat.running_pods:>5}")

    all_nodes = list(state.satellites.keys())

    # --- Test 1: Basic GPU job ---
    print("\n" + "-" * 70)
    print("Test 1: GPU training job (500W, min 30% battery, prefer sunlit)")
    pod1 = {
        "metadata": {
            "name": "gpu-training-job",
            "annotations": {
                "orbital-compute/min-battery": "30",
                "orbital-compute/prefer-sunlit": "true",
                "orbital-compute/power-watts": "500",
            }
        }
    }

    filter_result = extender.filter(pod1, all_nodes)
    passed = [n["metadata"]["name"] for n in filter_result["Nodes"]["items"]]
    failed = filter_result["FailedNodes"]

    print(f"  Passed filter: {passed}")
    for node, reason in failed.items():
        print(f"  FILTERED OUT: {node} — {reason}")

    scores = extender.prioritize(pod1, passed)
    scores.sort(key=lambda x: x["Score"], reverse=True)
    print("  Priority scores (higher = better):")
    for s in scores:
        print(f"    {s['Host']:<18} score={s['Score']}")

    # Verify sunlit nodes score higher
    sunlit_scores = [s["Score"] for s in scores
                     if not state.satellites[s["Host"]].in_eclipse]
    eclipse_scores = [s["Score"] for s in scores
                      if state.satellites[s["Host"]].in_eclipse]

    if sunlit_scores and eclipse_scores:
        assert min(sunlit_scores) > max(eclipse_scores), \
            "FAIL: sunlit nodes should score higher than eclipsed nodes"
        print("  PASS: All sunlit nodes scored higher than eclipsed nodes")

    # --- Test 2: Data downlink job ---
    print("\n" + "-" * 70)
    print("Test 2: Data downlink job (needs ground contact, min 20% battery)")
    pod2 = {
        "metadata": {
            "name": "data-downlink-job",
            "annotations": {
                "orbital-compute/min-battery": "20",
                "orbital-compute/needs-ground-contact": "true",
                "orbital-compute/prefer-sunlit": "false",
            }
        }
    }

    filter_result = extender.filter(pod2, all_nodes)
    passed2 = [n["metadata"]["name"] for n in filter_result["Nodes"]["items"]]
    failed2 = filter_result["FailedNodes"]

    print(f"  Passed filter: {passed2}")
    for node, reason in failed2.items():
        print(f"  FILTERED OUT: {node} — {reason}")

    scores2 = extender.prioritize(pod2, passed2)
    scores2.sort(key=lambda x: x["Score"], reverse=True)
    print("  Priority scores:")
    for s in scores2:
        print(f"    {s['Host']:<18} score={s['Score']}")

    # --- Test 3: Eclipse-sensitive job ---
    print("\n" + "-" * 70)
    print("Test 3: Long-running job (needs 30min+ before eclipse)")
    pod3 = {
        "metadata": {
            "name": "long-running-inference",
            "annotations": {
                "orbital-compute/min-battery": "25",
                "orbital-compute/max-eclipse-start": "30m",
                "orbital-compute/prefer-sunlit": "true",
                "orbital-compute/power-watts": "400",
            }
        }
    }

    filter_result = extender.filter(pod3, all_nodes)
    passed3 = [n["metadata"]["name"] for n in filter_result["Nodes"]["items"]]
    failed3 = filter_result["FailedNodes"]

    print(f"  Passed filter: {passed3}")
    for node, reason in failed3.items():
        print(f"  FILTERED OUT: {node} — {reason}")

    scores3 = extender.prioritize(pod3, passed3)
    scores3.sort(key=lambda x: x["Score"], reverse=True)
    print("  Priority scores:")
    for s in scores3:
        print(f"    {s['Host']:<18} score={s['Score']}")

    # --- Test 4: Simulate K8s HTTP request format ---
    print("\n" + "-" * 70)
    print("Test 4: Simulated K8s extender HTTP payload (filter)")
    k8s_payload = {
        "Pod": pod1,
        "Nodes": {
            "items": [{"metadata": {"name": n}} for n in all_nodes]
        }
    }
    print(f"  Request payload keys: {list(k8s_payload.keys())}")
    print(f"  Candidate nodes: {len(k8s_payload['Nodes']['items'])}")

    # Parse like the HTTP handler does
    pod = k8s_payload["Pod"]
    nodes = k8s_payload["Nodes"]["items"]
    node_names = [n["metadata"]["name"] for n in nodes]
    result = extender.filter(pod, node_names)
    print(f"  Filter result: {len(result['Nodes']['items'])} passed, "
          f"{len(result['FailedNodes'])} filtered")

    # --- Verify low-battery filtering ---
    print("\n" + "-" * 70)
    print("Test 5: Low-battery node filtering")
    assert "sat-beta-02" in failed, \
        "FAIL: sat-beta-02 (15% battery) should be filtered with 30% min"
    print("  PASS: sat-beta-02 (15% battery) correctly filtered out")

    print("\n" + "=" * 70)
    print("All tests passed.")
    print(f"\nTo start the HTTP server: python -m orbital_compute.k8s_scheduler --serve")
    print(f"  POST http://localhost:8888/filter")
    print(f"  POST http://localhost:8888/prioritize")
    print(f"  GET  http://localhost:8888/health")
    print("=" * 70)


def main() -> None:
    """Entry point: run demo or start HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if "--serve" in sys.argv:
        port = 8888
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        state = _build_demo_state()
        extender = OrbitalSchedulerExtender(state)
        server = create_server(extender, port=port)
        try:
            print(f"Orbital scheduler extender serving on port {port}")
            print("Endpoints: POST /filter, POST /prioritize, GET /health")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            server.shutdown()
    else:
        _run_demo()


if __name__ == "__main__":
    main()
