from __future__ import annotations

"""Inter-Satellite Link (ISL) simulation — optical links between constellation satellites.

Models line-of-sight, link distance/latency/bandwidth, connectivity graphs,
and shortest-path routing between satellite pairs.

Link capacity model:
- 10 Gbps at distances < 1000 km
- Linear degradation from 1000 km to 5000 km (down to 0)
- No link beyond 5000 km
"""

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from orbital_compute.orbit import EARTH_RADIUS_KM, Satellite, SatPosition


# ISL constants
SPEED_OF_LIGHT_KM_S = 299792.458  # km/s
MAX_LINK_RANGE_KM = 5000.0
FULL_BW_RANGE_KM = 1000.0
MAX_BANDWIDTH_GBPS = 10.0


@dataclass
class LinkMetrics:
    """Metrics for a single inter-satellite link."""
    distance_km: float
    latency_s: float
    bandwidth_gbps: float
    in_range: bool


def has_line_of_sight(sat1_pos: np.ndarray, sat2_pos: np.ndarray) -> bool:
    """Check whether two satellites have unobstructed line of sight.

    Tests if the line segment between sat1 and sat2 passes through Earth.
    Uses closest-point-on-segment to Earth center (origin) approach.

    Parameters
    ----------
    sat1_pos : np.ndarray
        ECI position [x, y, z] in km for satellite 1.
    sat2_pos : np.ndarray
        ECI position [x, y, z] in km for satellite 2.

    Returns
    -------
    bool
        True if the two satellites can see each other (Earth does not block).
    """
    # Direction vector from sat1 to sat2
    d = sat2_pos - sat1_pos
    d_dot_d = np.dot(d, d)

    if d_dot_d == 0:
        # Same position — trivially visible
        return True

    # Parameter t for closest point on segment to origin (Earth center)
    t = -np.dot(sat1_pos, d) / d_dot_d
    t = np.clip(t, 0.0, 1.0)

    # Closest point on the segment to Earth center
    closest = sat1_pos + t * d
    min_dist = np.linalg.norm(closest)

    # Add a small margin (~50 km) to account for atmosphere
    return min_dist > (EARTH_RADIUS_KM + 50.0)


def link_metrics(sat1_pos: np.ndarray, sat2_pos: np.ndarray) -> LinkMetrics:
    """Calculate link distance, latency, and available bandwidth between two satellites.

    Parameters
    ----------
    sat1_pos : np.ndarray
        ECI position [x, y, z] in km for satellite 1.
    sat2_pos : np.ndarray
        ECI position [x, y, z] in km for satellite 2.

    Returns
    -------
    LinkMetrics
        Distance, one-way latency, bandwidth, and whether link is feasible.
    """
    distance = float(np.linalg.norm(sat2_pos - sat1_pos))
    latency = distance / SPEED_OF_LIGHT_KM_S

    if distance > MAX_LINK_RANGE_KM:
        return LinkMetrics(
            distance_km=distance,
            latency_s=latency,
            bandwidth_gbps=0.0,
            in_range=False,
        )

    # Bandwidth model: full 10 Gbps under 1000 km, linear degradation to 5000 km
    if distance <= FULL_BW_RANGE_KM:
        bw = MAX_BANDWIDTH_GBPS
    else:
        fraction = 1.0 - (distance - FULL_BW_RANGE_KM) / (MAX_LINK_RANGE_KM - FULL_BW_RANGE_KM)
        bw = MAX_BANDWIDTH_GBPS * fraction

    return LinkMetrics(
        distance_km=distance,
        latency_s=latency,
        bandwidth_gbps=bw,
        in_range=True,
    )


@dataclass
class ConnectivityEdge:
    """An edge in the connectivity graph."""
    source: str
    target: str
    metrics: LinkMetrics


def build_connectivity_graph(
    satellites: List[Satellite],
    timestamp: datetime,
) -> Dict[str, List[Tuple[str, LinkMetrics]]]:
    """Build an adjacency-list connectivity graph for the constellation at a given time.

    Checks every pair of satellites for line-of-sight and range, then adds
    bidirectional edges for feasible links.

    Parameters
    ----------
    satellites : list of Satellite
        The constellation satellites.
    timestamp : datetime
        The time at which to evaluate connectivity.

    Returns
    -------
    dict
        Adjacency list: {sat_name: [(neighbor_name, LinkMetrics), ...]}.
    """
    # Pre-compute all positions
    positions: Dict[str, np.ndarray] = {}
    for sat in satellites:
        pos = sat.position_at(timestamp)
        positions[sat.name] = np.array([pos.x_km, pos.y_km, pos.z_km])

    # Initialize adjacency list
    graph: Dict[str, List[Tuple[str, LinkMetrics]]] = {sat.name: [] for sat in satellites}

    names = [sat.name for sat in satellites]
    n = len(names)

    for i in range(n):
        for j in range(i + 1, n):
            p1 = positions[names[i]]
            p2 = positions[names[j]]

            # Quick distance check before expensive LOS
            dist = float(np.linalg.norm(p2 - p1))
            if dist > MAX_LINK_RANGE_KM:
                continue

            if not has_line_of_sight(p1, p2):
                continue

            metrics = link_metrics(p1, p2)
            if metrics.in_range:
                graph[names[i]].append((names[j], metrics))
                graph[names[j]].append((names[i], metrics))

    return graph


def find_route(
    graph: Dict[str, List[Tuple[str, LinkMetrics]]],
    source: str,
    destination: str,
    weight: str = "latency",
) -> Optional[Tuple[List[str], float]]:
    """Find the shortest path between two satellites using Dijkstra's algorithm.

    Parameters
    ----------
    graph : dict
        Adjacency list from build_connectivity_graph.
    source : str
        Name of the source satellite.
    destination : str
        Name of the destination satellite.
    weight : str
        Edge weight to minimize: "latency" (seconds) or "distance" (km).
        Default is "latency".

    Returns
    -------
    tuple or None
        (path, total_cost) where path is list of satellite names,
        or None if no route exists.
    """
    if source not in graph or destination not in graph:
        return None

    # Dijkstra
    dist: Dict[str, float] = {name: float("inf") for name in graph}
    dist[source] = 0.0
    prev: Dict[str, Optional[str]] = {name: None for name in graph}
    visited: Set[str] = set()

    # Priority queue: (cost, node)
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == destination:
            break

        for neighbor, metrics in graph[node]:
            if neighbor in visited:
                continue

            if weight == "distance":
                edge_cost = metrics.distance_km
            else:
                edge_cost = metrics.latency_s

            new_cost = cost + edge_cost
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev[neighbor] = node
                heapq.heappush(pq, (new_cost, neighbor))

    # Reconstruct path
    if dist[destination] == float("inf"):
        return None

    path = []
    current: Optional[str] = destination
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    return path, dist[destination]


class InterSatelliteNetwork:
    """Manages the inter-satellite link network for a constellation.

    Wraps connectivity graph construction and routing into a convenient interface.
    """

    def __init__(self, satellites: List[Satellite]):
        self.satellites = satellites
        self._sat_by_name: Dict[str, Satellite] = {s.name: s for s in satellites}
        self._graph: Optional[Dict[str, List[Tuple[str, LinkMetrics]]]] = None
        self._timestamp: Optional[datetime] = None

    def update(self, timestamp: datetime) -> None:
        """Rebuild the connectivity graph for the given timestamp."""
        self._graph = build_connectivity_graph(self.satellites, timestamp)
        self._timestamp = timestamp

    @property
    def graph(self) -> Dict[str, List[Tuple[str, LinkMetrics]]]:
        """Current connectivity graph. Call update() first."""
        if self._graph is None:
            raise RuntimeError("Call update(timestamp) before accessing the graph.")
        return self._graph

    @property
    def timestamp(self) -> Optional[datetime]:
        return self._timestamp

    def neighbors(self, sat_name: str) -> List[Tuple[str, LinkMetrics]]:
        """Get all satellites that can communicate with the given satellite."""
        return self.graph.get(sat_name, [])

    def route(
        self, source: str, destination: str, weight: str = "latency"
    ) -> Optional[Tuple[List[str], float]]:
        """Find shortest route between two satellites."""
        return find_route(self.graph, source, destination, weight=weight)

    def total_links(self) -> int:
        """Count total active links (each bidirectional link counted once)."""
        count = sum(len(neighbors) for neighbors in self.graph.values())
        return count // 2  # Each link appears twice (bidirectional)

    def average_neighbors(self) -> float:
        """Average number of neighbors per satellite."""
        if not self.graph:
            return 0.0
        return sum(len(n) for n in self.graph.values()) / len(self.graph)

    def summary(self) -> str:
        """Human-readable summary of current network state."""
        if self._graph is None:
            return "Network not initialized. Call update(timestamp)."

        n_sats = len(self._graph)
        n_links = self.total_links()
        avg_neighbors = self.average_neighbors()

        # Gather bandwidth stats
        all_bw = []
        all_latency = []
        for neighbors in self._graph.values():
            for _, m in neighbors:
                all_bw.append(m.bandwidth_gbps)
                all_latency.append(m.latency_s * 1000)  # ms

        lines = [
            f"ISL Network @ {self._timestamp}",
            f"  Satellites:      {n_sats}",
            f"  Active links:    {n_links}",
            f"  Avg neighbors:   {avg_neighbors:.1f}",
        ]
        if all_bw:
            lines.extend([
                f"  Bandwidth range: {min(all_bw):.1f} - {max(all_bw):.1f} Gbps",
                f"  Latency range:   {min(all_latency):.2f} - {max(all_latency):.2f} ms",
            ])
        else:
            lines.append("  No active links.")

        return "\n".join(lines)


if __name__ == "__main__":
    from datetime import timezone

    from orbital_compute.orbit import starlink_shell_1_sample

    print("=== Inter-Satellite Link Simulation ===\n")

    # Create 12 sample satellites
    sats = starlink_shell_1_sample(n_sats=12)
    print(f"Created {len(sats)} satellites\n")

    # Build the network
    now = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    network = InterSatelliteNetwork(sats)
    network.update(now)

    # Print summary
    print(network.summary())
    print()

    # Show neighbors for first satellite
    first = sats[0].name
    neighbors = network.neighbors(first)
    print(f"Neighbors of {first}: {len(neighbors)}")
    for name, m in sorted(neighbors, key=lambda x: x[1].distance_km):
        print(f"  -> {name}: {m.distance_km:.0f} km, "
              f"{m.latency_s*1000:.2f} ms, {m.bandwidth_gbps:.1f} Gbps")

    # Test routing between first and last satellite
    last = sats[-1].name
    print(f"\nRouting from {first} to {last}:")
    result = network.route(first, last)
    if result:
        path, total_latency = result
        print(f"  Path: {' -> '.join(path)}")
        print(f"  Total latency: {total_latency*1000:.2f} ms")
        print(f"  Hops: {len(path) - 1}")
    else:
        print("  No route found!")

    # Also test distance-based routing
    result_dist = network.route(first, last, weight="distance")
    if result_dist:
        path_d, total_dist = result_dist
        print(f"\nShortest distance route: {' -> '.join(path_d)}")
        print(f"  Total distance: {total_dist:.0f} km")

    # Test line-of-sight directly
    print("\n--- Line-of-sight tests ---")
    p1 = np.array([EARTH_RADIUS_KM + 550, 0, 0])
    p2 = np.array([-(EARTH_RADIUS_KM + 550), 0, 0])  # Opposite side of Earth
    print(f"Opposite sides of Earth: LOS={has_line_of_sight(p1, p2)}")  # Should be False

    p3 = np.array([EARTH_RADIUS_KM + 550, 100, 0])
    print(f"Nearby satellites: LOS={has_line_of_sight(p1, p3)}")  # Should be True

    print("\nDone.")
