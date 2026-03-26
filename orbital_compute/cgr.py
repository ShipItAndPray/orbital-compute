from __future__ import annotations

"""Contact Graph Routing (CGR) — DTN routing for satellite constellations.

Implements NASA's CGR algorithm used in delay-tolerant networking (DTN).
Instead of routing on instantaneous topology snapshots (like network.py does),
CGR pre-computes a *contact plan* of all future communication opportunities
and finds time-optimal routes through the time-expanded contact graph.

References:
- S. Burleigh, "Contact Graph Routing", Internet-Draft, 2010
- G. Araniti et al., "Contact graph routing in DTN space networks", IEEE, 2015

Three layers:
1. ContactPlanGenerator — compute all contacts for a time window
2. CGRRouter — Yen's k-shortest paths on the contact graph
3. BundleProtocolSimulator — simulate DTN bundle forwarding
"""

import heapq
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .orbit import Satellite, EARTH_RADIUS_KM
from .isl import (
    SPEED_OF_LIGHT_KM_S,
    MAX_LINK_RANGE_KM,
    has_line_of_sight,
    link_metrics,
    LinkMetrics,
    InterSatelliteNetwork,
)
from .ground_stations import (
    GroundStation,
    DEFAULT_GROUND_STATIONS,
    elevation_angle,
    _lla_to_ecef,
    _ecef_to_eci,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """A single communication opportunity between two nodes.

    This is the fundamental unit of CGR. A contact has a known start/end time,
    a data rate, and a one-way light time (propagation delay).
    """
    contact_id: int
    from_node: str
    to_node: str
    start_time: datetime
    end_time: datetime
    data_rate_bps: float       # bits per second
    owlt_s: float              # one-way light time (seconds)
    residual_capacity_bits: float = 0.0  # remaining capacity for routing

    def __post_init__(self):
        duration_s = (self.end_time - self.start_time).total_seconds()
        if self.residual_capacity_bits == 0.0:
            self.residual_capacity_bits = self.data_rate_bps * duration_s

    @property
    def duration_s(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def capacity_bits(self) -> float:
        return self.data_rate_bps * self.duration_s

    def __lt__(self, other):
        return self.start_time < other.start_time


@dataclass
class ContactPlan:
    """The full set of contacts for a time window — input to CGR."""
    contacts: List[Contact]
    start_time: datetime
    end_time: datetime
    nodes: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if not self.nodes:
            for c in self.contacts:
                self.nodes.add(c.from_node)
                self.nodes.add(c.to_node)

    def contacts_from(self, node: str) -> List[Contact]:
        """All contacts originating from a node, sorted by start time."""
        result = [c for c in self.contacts if c.from_node == node]
        result.sort(key=lambda c: c.start_time)
        return result

    def contacts_between(self, a: str, b: str) -> List[Contact]:
        """All contacts between two specific nodes (either direction)."""
        return [c for c in self.contacts
                if (c.from_node == a and c.to_node == b)
                or (c.from_node == b and c.to_node == a)]


@dataclass
class CGRRoute:
    """A route through the contact graph."""
    hops: List[Contact]          # ordered contacts to traverse
    delivery_time: datetime      # when data arrives at destination
    total_latency_s: float       # end-to-end latency from dispatch
    bottleneck_rate_bps: float   # minimum data rate along the path
    volume_bits: float           # max data that can traverse this route
    path_nodes: List[str]        # ordered list of node names


@dataclass
class Bundle:
    """A DTN bundle — unit of data being forwarded."""
    bundle_id: int
    source: str
    destination: str
    size_bits: float
    created_at: datetime
    dispatch_time: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    custody_holder: Optional[str] = None
    route: Optional[CGRRoute] = None
    hop_log: List[Tuple[str, datetime]] = field(default_factory=list)
    retransmissions: int = 0

    @property
    def is_delivered(self) -> bool:
        return self.delivered_at is not None

    @property
    def delivery_latency_s(self) -> Optional[float]:
        if self.delivered_at and self.dispatch_time:
            return (self.delivered_at - self.dispatch_time).total_seconds()
        return None


# ---------------------------------------------------------------------------
# 1. Contact Plan Generator
# ---------------------------------------------------------------------------

class ContactPlanGenerator:
    """Compute all future contacts for a constellation and ground stations.

    Scans the time window at a configurable step, detecting when links
    come up and go down. Produces Contact objects for both ISLs and
    satellite-to-ground links.
    """

    def __init__(
        self,
        satellites: List[Satellite],
        ground_stations: Optional[List[GroundStation]] = None,
        step_seconds: float = 30.0,
    ):
        self.satellites = satellites
        self.ground_stations = ground_stations or []
        self.step_seconds = step_seconds
        self._sat_by_name: Dict[str, Satellite] = {s.name: s for s in satellites}

    def generate(
        self,
        start: datetime,
        duration_hours: float,
    ) -> ContactPlan:
        """Generate the full contact plan for the given time window.

        Returns a ContactPlan with all ISL and ground contacts.
        """
        end = start + timedelta(hours=duration_hours)
        step = timedelta(seconds=self.step_seconds)
        contacts: List[Contact] = []
        contact_id = 0

        sat_names = [s.name for s in self.satellites]
        n_sats = len(sat_names)

        # --- Track link state for ISLs ---
        # Key: (sat_i_name, sat_j_name) with i < j
        isl_active: Dict[Tuple[str, str], dict] = {}

        # --- Track link state for ground contacts ---
        # Key: (sat_name, station_name)
        gs_active: Dict[Tuple[str, str], dict] = {}

        t = start
        while t <= end:
            # Pre-compute all satellite ECI positions at this timestep
            positions: Dict[str, np.ndarray] = {}
            for sat in self.satellites:
                pos = sat.position_at(t)
                positions[sat.name] = np.array([pos.x_km, pos.y_km, pos.z_km])

            # --- ISL contacts ---
            for i in range(n_sats):
                for j in range(i + 1, n_sats):
                    ni, nj = sat_names[i], sat_names[j]
                    key = (ni, nj)
                    p1, p2 = positions[ni], positions[nj]
                    dist = float(np.linalg.norm(p2 - p1))

                    in_range = (
                        dist <= MAX_LINK_RANGE_KM
                        and has_line_of_sight(p1, p2)
                    )

                    if in_range and key not in isl_active:
                        # Link just came up
                        metrics = link_metrics(p1, p2)
                        isl_active[key] = {
                            "start": t,
                            "rates": [metrics.bandwidth_gbps * 1e9],
                            "owlts": [metrics.latency_s],
                        }
                    elif in_range and key in isl_active:
                        # Link still up — accumulate metrics
                        metrics = link_metrics(p1, p2)
                        isl_active[key]["rates"].append(metrics.bandwidth_gbps * 1e9)
                        isl_active[key]["owlts"].append(metrics.latency_s)
                    elif not in_range and key in isl_active:
                        # Link just went down — emit contacts (bidirectional)
                        state = isl_active.pop(key)
                        avg_rate = sum(state["rates"]) / len(state["rates"])
                        avg_owlt = sum(state["owlts"]) / len(state["owlts"])
                        for fn, tn in [(ni, nj), (nj, ni)]:
                            contacts.append(Contact(
                                contact_id=contact_id,
                                from_node=fn, to_node=tn,
                                start_time=state["start"], end_time=t,
                                data_rate_bps=avg_rate,
                                owlt_s=avg_owlt,
                            ))
                            contact_id += 1

            # --- Ground station contacts ---
            for sat in self.satellites:
                sat_eci = positions[sat.name]
                for station in self.ground_stations:
                    key = (sat.name, station.name)
                    elev = elevation_angle(sat_eci, station, t)
                    in_contact = elev >= station.min_elevation_deg

                    if in_contact and key not in gs_active:
                        gs_active[key] = {
                            "start": t,
                            "downlink_bps": station.downlink_mbps * 1e6,
                            "elevations": [elev],
                        }
                    elif in_contact and key in gs_active:
                        gs_active[key]["elevations"].append(elev)
                    elif not in_contact and key in gs_active:
                        state = gs_active.pop(key)
                        # Satellite -> ground
                        station_ecef = _lla_to_ecef(station.lat_deg, station.lon_deg)
                        station_eci = _ecef_to_eci(station_ecef, t)
                        dist_km = float(np.linalg.norm(sat_eci - station_eci))
                        owlt = dist_km / SPEED_OF_LIGHT_KM_S
                        contacts.append(Contact(
                            contact_id=contact_id,
                            from_node=sat.name,
                            to_node=f"GS:{station.name}",
                            start_time=state["start"], end_time=t,
                            data_rate_bps=state["downlink_bps"],
                            owlt_s=owlt,
                        ))
                        contact_id += 1
                        # Ground -> satellite (uplink)
                        contacts.append(Contact(
                            contact_id=contact_id,
                            from_node=f"GS:{station.name}",
                            to_node=sat.name,
                            start_time=state["start"], end_time=t,
                            data_rate_bps=station.uplink_mbps * 1e6,
                            owlt_s=owlt,
                        ))
                        contact_id += 1

            t += step

        # Close any contacts still open at end of window
        for key, state in isl_active.items():
            ni, nj = key
            avg_rate = sum(state["rates"]) / len(state["rates"])
            avg_owlt = sum(state["owlts"]) / len(state["owlts"])
            for fn, tn in [(ni, nj), (nj, ni)]:
                contacts.append(Contact(
                    contact_id=contact_id,
                    from_node=fn, to_node=tn,
                    start_time=state["start"], end_time=end,
                    data_rate_bps=avg_rate, owlt_s=avg_owlt,
                ))
                contact_id += 1

        for key, state in gs_active.items():
            sat_name, station_name = key
            contacts.append(Contact(
                contact_id=contact_id,
                from_node=sat_name, to_node=f"GS:{station_name}",
                start_time=state["start"], end_time=end,
                data_rate_bps=state["downlink_bps"], owlt_s=0.01,
            ))
            contact_id += 1
            # Find the station object for uplink rate
            for gs in self.ground_stations:
                if gs.name == station_name:
                    contacts.append(Contact(
                        contact_id=contact_id,
                        from_node=f"GS:{station_name}", to_node=sat_name,
                        start_time=state["start"], end_time=end,
                        data_rate_bps=gs.uplink_mbps * 1e6, owlt_s=0.01,
                    ))
                    contact_id += 1
                    break

        contacts.sort(key=lambda c: c.start_time)
        return ContactPlan(contacts=contacts, start_time=start, end_time=end)


# ---------------------------------------------------------------------------
# 2. CGR Route Computation
# ---------------------------------------------------------------------------

class CGRRouter:
    """Contact Graph Router — finds optimal routes through a contact plan.

    Uses a modified Dijkstra on the time-expanded contact graph, then
    Yen's algorithm for k-shortest paths.

    Key insight: in CGR, "distance" is *earliest arrival time*, not
    spatial distance. A route is a sequence of contacts where each
    contact's start time >= the arrival time at the preceding node.
    """

    def __init__(
        self,
        contact_plan: ContactPlan,
        buffer_capacity_bits: Optional[Dict[str, float]] = None,
    ):
        self.plan = contact_plan
        # Default 1 Gbit buffer per node
        self.buffer_capacity = buffer_capacity_bits or {}
        self._default_buffer = 1e9  # 1 Gbit

        # Pre-index contacts by from_node for fast lookup
        self._contacts_from: Dict[str, List[Contact]] = {}
        for c in self.plan.contacts:
            self._contacts_from.setdefault(c.from_node, []).append(c)
        for key in self._contacts_from:
            self._contacts_from[key].sort(key=lambda c: c.start_time)

    def _get_buffer(self, node: str) -> float:
        return self.buffer_capacity.get(node, self._default_buffer)

    def find_best_route(
        self,
        source: str,
        destination: str,
        dispatch_time: datetime,
        data_size_bits: float = 0,
    ) -> Optional[CGRRoute]:
        """Find the earliest-arrival route from source to destination.

        Uses modified Dijkstra where edge weight is arrival time at the
        next node, and we can only use a contact if we arrive at its
        from_node before the contact ends.

        Parameters
        ----------
        source : str
            Originating node name.
        destination : str
            Target node name.
        dispatch_time : datetime
            Earliest time data is available to send.
        data_size_bits : float
            Size of data to route. If > 0, checks that each contact has
            enough residual capacity.

        Returns
        -------
        CGRRoute or None
        """
        if source == destination:
            return CGRRoute(
                hops=[], delivery_time=dispatch_time,
                total_latency_s=0.0, bottleneck_rate_bps=float("inf"),
                volume_bits=float("inf"), path_nodes=[source],
            )

        # Dijkstra: priority queue of (arrival_time_ts, node, path_contacts)
        # arrival_time_ts is float timestamp for comparison
        INF = float("inf")
        best_arrival: Dict[str, float] = {n: INF for n in self.plan.nodes}
        dispatch_ts = dispatch_time.timestamp()
        best_arrival[source] = dispatch_ts

        # (arrival_ts, node_name, [contact_hops])
        pq: List[Tuple[float, int, str, List[Contact]]] = [
            (dispatch_ts, 0, source, [])
        ]
        counter = 1  # tiebreaker
        visited: Set[str] = set()

        while pq:
            arr_ts, _, node, hops = heapq.heappop(pq)

            if node == destination:
                arrival_dt = datetime.fromtimestamp(arr_ts, tz=timezone.utc)
                bottleneck = min((c.data_rate_bps for c in hops), default=INF)
                volume = min((c.residual_capacity_bits for c in hops), default=INF)
                return CGRRoute(
                    hops=hops,
                    delivery_time=arrival_dt,
                    total_latency_s=arr_ts - dispatch_ts,
                    bottleneck_rate_bps=bottleneck,
                    volume_bits=volume,
                    path_nodes=[source] + [c.to_node for c in hops],
                )

            if node in visited:
                continue
            visited.add(node)

            for contact in self._contacts_from.get(node, []):
                # Can we use this contact?
                # We must arrive at from_node before the contact ends
                contact_end_ts = contact.end_time.timestamp()
                contact_start_ts = contact.start_time.timestamp()

                if arr_ts > contact_end_ts:
                    continue  # Contact already over

                # Effective send start = max(our arrival, contact start)
                send_start_ts = max(arr_ts, contact_start_ts)

                # Check capacity
                if data_size_bits > 0:
                    remaining_time = contact_end_ts - send_start_ts
                    available_bits = contact.data_rate_bps * remaining_time
                    if available_bits < data_size_bits:
                        continue
                    if contact.residual_capacity_bits < data_size_bits:
                        continue

                # Arrival at the other end of this contact
                neighbor = contact.to_node
                neighbor_arrival_ts = send_start_ts + contact.owlt_s

                # Add data transfer time if we have a known size
                if data_size_bits > 0:
                    transfer_time = data_size_bits / contact.data_rate_bps
                    neighbor_arrival_ts = send_start_ts + transfer_time + contact.owlt_s

                if neighbor_arrival_ts < best_arrival.get(neighbor, INF):
                    best_arrival[neighbor] = neighbor_arrival_ts
                    heapq.heappush(pq, (
                        neighbor_arrival_ts, counter, neighbor, hops + [contact]
                    ))
                    counter += 1

        return None  # No route found

    def find_k_routes(
        self,
        source: str,
        destination: str,
        dispatch_time: datetime,
        k: int = 3,
        data_size_bits: float = 0,
    ) -> List[CGRRoute]:
        """Find k-shortest (earliest-arrival) routes using Yen's algorithm.

        Returns up to k routes, ordered by delivery time.
        """
        best = self.find_best_route(source, destination, dispatch_time, data_size_bits)
        if best is None:
            return []

        A: List[CGRRoute] = [best]  # confirmed k-shortest
        B: List[CGRRoute] = []      # candidates

        for i in range(1, k):
            prev_route = A[i - 1]

            for j in range(len(prev_route.hops)):
                spur_node = prev_route.path_nodes[j]

                # Determine spur dispatch time
                if j == 0:
                    spur_time = dispatch_time
                else:
                    spur_time = prev_route.hops[j - 1].end_time

                # Exclude contacts used by existing routes at this spur point
                excluded_contacts: Set[int] = set()
                for route in A:
                    if (route.path_nodes[:j + 1] == prev_route.path_nodes[:j + 1]
                            and j < len(route.hops)):
                        excluded_contacts.add(route.hops[j].contact_id)

                # Exclude nodes in the root path (except spur node)
                excluded_nodes = set(prev_route.path_nodes[:j])

                # Find spur path with exclusions
                spur_route = self._find_route_with_exclusions(
                    spur_node, destination, spur_time,
                    excluded_contacts, excluded_nodes, data_size_bits,
                )

                if spur_route is not None:
                    # Concatenate root path + spur path
                    root_hops = prev_route.hops[:j]
                    full_hops = root_hops + spur_route.hops
                    full_path = prev_route.path_nodes[:j] + spur_route.path_nodes

                    total_latency = (
                        spur_route.delivery_time - dispatch_time
                    ).total_seconds()
                    bottleneck = min(
                        (c.data_rate_bps for c in full_hops), default=float("inf")
                    )
                    volume = min(
                        (c.residual_capacity_bits for c in full_hops),
                        default=float("inf"),
                    )

                    candidate = CGRRoute(
                        hops=full_hops,
                        delivery_time=spur_route.delivery_time,
                        total_latency_s=total_latency,
                        bottleneck_rate_bps=bottleneck,
                        volume_bits=volume,
                        path_nodes=full_path,
                    )

                    # Avoid duplicates
                    dup = False
                    for existing in A + B:
                        if ([c.contact_id for c in existing.hops]
                                == [c.contact_id for c in candidate.hops]):
                            dup = True
                            break
                    if not dup:
                        B.append(candidate)

            if not B:
                break

            B.sort(key=lambda r: r.delivery_time)
            A.append(B.pop(0))

        return A

    def _find_route_with_exclusions(
        self,
        source: str,
        destination: str,
        dispatch_time: datetime,
        excluded_contacts: Set[int],
        excluded_nodes: Set[str],
        data_size_bits: float,
    ) -> Optional[CGRRoute]:
        """Dijkstra with contact and node exclusions (for Yen's algorithm)."""
        if source == destination:
            return CGRRoute(
                hops=[], delivery_time=dispatch_time,
                total_latency_s=0.0, bottleneck_rate_bps=float("inf"),
                volume_bits=float("inf"), path_nodes=[source],
            )

        INF = float("inf")
        dispatch_ts = dispatch_time.timestamp()
        best_arrival: Dict[str, float] = {}
        best_arrival[source] = dispatch_ts

        pq: List[Tuple[float, int, str, List[Contact]]] = [
            (dispatch_ts, 0, source, [])
        ]
        counter = 1
        visited: Set[str] = set()

        while pq:
            arr_ts, _, node, hops = heapq.heappop(pq)

            if node == destination:
                arrival_dt = datetime.fromtimestamp(arr_ts, tz=timezone.utc)
                bottleneck = min((c.data_rate_bps for c in hops), default=INF)
                volume = min((c.residual_capacity_bits for c in hops), default=INF)
                return CGRRoute(
                    hops=hops,
                    delivery_time=arrival_dt,
                    total_latency_s=arr_ts - dispatch_ts,
                    bottleneck_rate_bps=bottleneck,
                    volume_bits=volume,
                    path_nodes=[source] + [c.to_node for c in hops],
                )

            if node in visited:
                continue
            visited.add(node)

            for contact in self._contacts_from.get(node, []):
                if contact.contact_id in excluded_contacts:
                    continue
                if contact.to_node in excluded_nodes:
                    continue

                contact_end_ts = contact.end_time.timestamp()
                if arr_ts > contact_end_ts:
                    continue

                send_start_ts = max(arr_ts, contact.start_time.timestamp())

                if data_size_bits > 0:
                    remaining_time = contact_end_ts - send_start_ts
                    available_bits = contact.data_rate_bps * remaining_time
                    if available_bits < data_size_bits:
                        continue
                    if contact.residual_capacity_bits < data_size_bits:
                        continue

                neighbor = contact.to_node
                neighbor_arrival_ts = send_start_ts + contact.owlt_s
                if data_size_bits > 0:
                    transfer_time = data_size_bits / contact.data_rate_bps
                    neighbor_arrival_ts = send_start_ts + transfer_time + contact.owlt_s

                if neighbor_arrival_ts < best_arrival.get(neighbor, INF):
                    best_arrival[neighbor] = neighbor_arrival_ts
                    heapq.heappush(pq, (
                        neighbor_arrival_ts, counter, neighbor, hops + [contact]
                    ))
                    counter += 1

        return None


# ---------------------------------------------------------------------------
# 3. Bundle Protocol Simulator
# ---------------------------------------------------------------------------

class BundleProtocolSimulator:
    """Simulate DTN bundle forwarding across a constellation.

    Tracks custody transfer, delivery confirmation, and retransmission.
    Uses CGR to route bundles.
    """

    def __init__(self, router: CGRRouter):
        self.router = router
        self.bundles: List[Bundle] = []
        self._next_id = 0

    def create_bundle(
        self,
        source: str,
        destination: str,
        size_bits: float,
        created_at: datetime,
    ) -> Bundle:
        """Create a new bundle for delivery."""
        b = Bundle(
            bundle_id=self._next_id,
            source=source,
            destination=destination,
            size_bits=size_bits,
            created_at=created_at,
            custody_holder=source,
        )
        self._next_id += 1
        self.bundles.append(b)
        return b

    def route_and_forward(self, bundle: Bundle) -> bool:
        """Compute a CGR route for the bundle and simulate forwarding.

        Returns True if the bundle was successfully delivered.
        """
        route = self.router.find_best_route(
            source=bundle.source,
            destination=bundle.destination,
            dispatch_time=bundle.created_at,
            data_size_bits=bundle.size_bits,
        )

        if route is None:
            return False

        bundle.route = route
        bundle.dispatch_time = bundle.created_at
        bundle.hop_log.append((bundle.source, bundle.created_at))

        # Simulate hop-by-hop forwarding
        current_time = bundle.created_at
        for contact in route.hops:
            # Wait for contact to start if needed
            if current_time < contact.start_time:
                current_time = contact.start_time

            # Transfer time
            transfer_s = bundle.size_bits / contact.data_rate_bps
            arrival_time = current_time + timedelta(seconds=transfer_s + contact.owlt_s)

            # Deduct capacity
            contact.residual_capacity_bits -= bundle.size_bits

            # Custody transfer
            bundle.custody_holder = contact.to_node
            bundle.hop_log.append((contact.to_node, arrival_time))
            current_time = arrival_time

        bundle.delivered_at = current_time
        return True

    def run_all(self) -> Dict[str, object]:
        """Route and forward all pending bundles. Return summary stats."""
        delivered = 0
        failed = 0
        latencies = []

        for bundle in self.bundles:
            if bundle.is_delivered:
                continue
            if self.route_and_forward(bundle):
                delivered += 1
                lat = bundle.delivery_latency_s
                if lat is not None:
                    latencies.append(lat)
            else:
                failed += 1

        return {
            "total_bundles": len(self.bundles),
            "delivered": delivered,
            "failed": failed,
            "delivery_rate": delivered / max(delivered + failed, 1),
            "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_s": min(latencies) if latencies else 0,
            "max_latency_s": max(latencies) if latencies else 0,
        }


# ---------------------------------------------------------------------------
# 4. Nearest-Neighbor Router (baseline for comparison)
# ---------------------------------------------------------------------------

class NearestNeighborRouter:
    """Simple greedy nearest-neighbor routing — our baseline.

    At each hop, picks the neighbor closest to the destination.
    Re-evaluates the topology at each step (no contact plan awareness).
    This is essentially what network.py does with Dijkstra on a single
    snapshot.
    """

    def __init__(self, satellites: List[Satellite]):
        self.satellites = satellites
        self.network = InterSatelliteNetwork(satellites)
        self._sat_by_name = {s.name: s for s in satellites}

    def route(
        self,
        source: str,
        destination: str,
        dispatch_time: datetime,
        data_size_bits: float = 0,
    ) -> Optional[Tuple[float, List[str]]]:
        """Route using instantaneous Dijkstra at dispatch time.

        Returns (total_latency_s, path) or None.
        """
        self.network.update(dispatch_time)
        result = self.network.route(source, destination, weight="latency")
        if result is None:
            return None
        path, total_latency_s = result

        # Add transfer time if data size is given
        if data_size_bits > 0 and len(path) > 1:
            graph = self.network.graph
            for i in range(len(path) - 1):
                for neighbor, metrics in graph.get(path[i], []):
                    if neighbor == path[i + 1]:
                        transfer_s = data_size_bits / (metrics.bandwidth_gbps * 1e9)
                        total_latency_s += transfer_s
                        break

        return total_latency_s, path


# ---------------------------------------------------------------------------
# 5. Comparison: CGR vs Nearest-Neighbor
# ---------------------------------------------------------------------------

def run_comparison(
    satellites: List[Satellite],
    ground_stations: Optional[List[GroundStation]] = None,
    start_time: Optional[datetime] = None,
    duration_hours: float = 2.0,
    data_sizes_mb: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Compare CGR vs nearest-neighbor routing.

    Runs both algorithms on the same constellation and data transfer
    scenarios, reports delivery time improvements.
    """
    if start_time is None:
        start_time = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    if data_sizes_mb is None:
        data_sizes_mb = [1.0, 10.0, 100.0]

    gs_list = ground_stations or []

    print("=" * 70)
    print("  CONTACT GRAPH ROUTING (CGR) vs NEAREST-NEIGHBOR ROUTING")
    print("=" * 70)
    print(f"  Constellation: {len(satellites)} satellites")
    if gs_list:
        print(f"  Ground stations: {len(gs_list)}")
    print(f"  Time window: {start_time.isoformat()} + {duration_hours}h")
    print(f"  Step: 30s")
    print()

    # --- Generate contact plan ---
    print("  [1/4] Generating contact plan...")
    gen = ContactPlanGenerator(satellites, gs_list, step_seconds=30.0)
    plan = gen.generate(start_time, duration_hours)
    print(f"         {len(plan.contacts)} contacts found across {len(plan.nodes)} nodes")

    # Contact stats
    isl_contacts = [c for c in plan.contacts
                    if not c.from_node.startswith("GS:") and not c.to_node.startswith("GS:")]
    gs_contacts = [c for c in plan.contacts
                   if c.from_node.startswith("GS:") or c.to_node.startswith("GS:")]
    print(f"         ISL contacts: {len(isl_contacts)}")
    print(f"         Ground contacts: {len(gs_contacts)}")

    if plan.contacts:
        durations = [c.duration_s for c in plan.contacts]
        print(f"         Contact duration: {min(durations):.0f}s - {max(durations):.0f}s "
              f"(avg {sum(durations)/len(durations):.0f}s)")
    print()

    # --- Setup routers ---
    print("  [2/4] Initializing routers...")
    cgr_router = CGRRouter(plan)
    nn_router = NearestNeighborRouter(satellites)
    print("         CGR router ready")
    print("         Nearest-neighbor router ready")
    print()

    # --- Pick test pairs ---
    sat_names = [s.name for s in satellites]
    # Use first and last satellite, plus a mid-point pair
    pairs = []
    if len(sat_names) >= 2:
        pairs.append((sat_names[0], sat_names[-1]))
    if len(sat_names) >= 4:
        mid = len(sat_names) // 2
        pairs.append((sat_names[0], sat_names[mid]))
    if len(sat_names) >= 6:
        pairs.append((sat_names[1], sat_names[-2]))

    # --- Run comparisons ---
    print("  [3/4] Running routing comparisons...")
    results = []

    for src, dst in pairs:
        for size_mb in data_sizes_mb:
            size_bits = size_mb * 8 * 1e6  # MB to bits

            # CGR route
            cgr_route = cgr_router.find_best_route(
                src, dst, start_time, data_size_bits=size_bits
            )
            cgr_latency = cgr_route.total_latency_s if cgr_route else None
            cgr_hops = len(cgr_route.hops) if cgr_route else None

            # Nearest-neighbor route
            nn_result = nn_router.route(src, dst, start_time, data_size_bits=size_bits)
            nn_latency = nn_result[0] if nn_result else None
            nn_hops = len(nn_result[1]) - 1 if nn_result else None

            improvement = None
            if cgr_latency is not None and nn_latency is not None and nn_latency > 0:
                improvement = (nn_latency - cgr_latency) / nn_latency * 100

            results.append({
                "source": src,
                "destination": dst,
                "data_size_mb": size_mb,
                "cgr_latency_s": cgr_latency,
                "cgr_hops": cgr_hops,
                "nn_latency_s": nn_latency,
                "nn_hops": nn_hops,
                "improvement_pct": improvement,
            })
    print()

    # --- Print results ---
    print("  [4/4] Results")
    print()
    print("  {:<14} {:<14} {:>8} {:>12} {:>12} {:>10}".format(
        "Source", "Dest", "Size(MB)", "CGR(s)", "NN(s)", "Improv(%)"))
    print("  " + "-" * 70)

    for r in results:
        cgr_str = f"{r['cgr_latency_s']:.3f}" if r["cgr_latency_s"] is not None else "NO ROUTE"
        nn_str = f"{r['nn_latency_s']:.3f}" if r["nn_latency_s"] is not None else "NO ROUTE"
        imp_str = f"{r['improvement_pct']:.1f}%" if r["improvement_pct"] is not None else "N/A"
        print("  {:<14} {:<14} {:>8.1f} {:>12} {:>12} {:>10}".format(
            r["source"][:14], r["destination"][:14],
            r["data_size_mb"], cgr_str, nn_str, imp_str))

    print()

    # --- Bundle protocol simulation ---
    if pairs:
        print("  --- Bundle Protocol Simulation ---")
        bps = BundleProtocolSimulator(cgr_router)
        for src, dst in pairs:
            for size_mb in data_sizes_mb[:2]:  # Just first two sizes
                bps.create_bundle(
                    src, dst, size_mb * 8 * 1e6, start_time
                )

        stats = bps.run_all()
        print(f"  Bundles: {stats['total_bundles']}")
        print(f"  Delivered: {stats['delivered']} ({stats['delivery_rate']*100:.0f}%)")
        print(f"  Failed: {stats['failed']}")
        if stats['avg_latency_s'] > 0:
            print(f"  Avg latency: {stats['avg_latency_s']:.3f}s")
            print(f"  Min latency: {stats['min_latency_s']:.3f}s")
            print(f"  Max latency: {stats['max_latency_s']:.3f}s")
        print()

    # --- k-shortest paths demo ---
    if pairs:
        src, dst = pairs[0]
        print(f"  --- Yen's k=3 shortest paths: {src} -> {dst} ---")
        k_routes = cgr_router.find_k_routes(src, dst, start_time, k=3)
        for i, route in enumerate(k_routes):
            print(f"  Route {i+1}: {' -> '.join(route.path_nodes)}")
            print(f"           Latency: {route.total_latency_s:.3f}s, "
                  f"Hops: {len(route.hops)}, "
                  f"Bottleneck: {route.bottleneck_rate_bps/1e9:.1f} Gbps")
        print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    cgr_wins = sum(1 for r in results if r["improvement_pct"] is not None and r["improvement_pct"] > 0)
    nn_wins = sum(1 for r in results if r["improvement_pct"] is not None and r["improvement_pct"] < 0)
    ties = sum(1 for r in results if r["improvement_pct"] is not None and r["improvement_pct"] == 0)
    no_route = sum(1 for r in results if r["improvement_pct"] is None)

    print(f"  CGR wins: {cgr_wins}/{len(results)} scenarios")
    print(f"  NN wins:  {nn_wins}/{len(results)} scenarios")
    print(f"  Ties:     {ties}/{len(results)} scenarios")
    if no_route:
        print(f"  No route: {no_route}/{len(results)} scenarios")

    improvements = [r["improvement_pct"] for r in results if r["improvement_pct"] is not None]
    if improvements:
        avg_imp = sum(improvements) / len(improvements)
        print(f"  Avg improvement: {avg_imp:.1f}%")
    print()

    # Honest assessment
    print("  NOTE: For instantaneous single-snapshot routing, CGR and NN Dijkstra")
    print("  may give similar results. CGR's advantage grows with:")
    print("    - Longer time horizons (data waiting for future contacts)")
    print("    - Store-and-forward scenarios (buffered at intermediate nodes)")
    print("    - Ground station contacts (sparse and time-dependent)")
    print("    - Large data transfers (capacity-aware routing)")
    print("=" * 70)

    return {
        "contact_plan": plan,
        "results": results,
        "n_contacts": len(plan.contacts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .orbit import starlink_shell_1_sample
    from .constellations import CONSTELLATIONS, generate_constellation

    print()
    print("Contact Graph Routing (CGR) for Orbital Compute")
    print("================================================")
    print()

    # Use starlink-mini for a manageable demo
    config = CONSTELLATIONS["starlink-mini"]
    sats = generate_constellation(config)
    print(f"Constellation: {config.name} ({len(sats)} satellites)")
    print(f"  Altitude: {config.altitude_km} km, Inclination: {config.inclination_deg} deg")
    print(f"  Planes: {config.n_planes}, Sats/plane: {config.sats_per_plane}")
    print()

    # Use a subset of ground stations for the demo
    gs_subset = DEFAULT_GROUND_STATIONS[:4]

    run_comparison(
        satellites=sats,
        ground_stations=gs_subset,
        duration_hours=2.0,
        data_sizes_mb=[1.0, 10.0, 100.0],
    )
