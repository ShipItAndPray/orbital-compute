from __future__ import annotations

"""Network topology optimizer — design ISL mesh for compute constellations.

Optimizes inter-satellite link topology for:
- Minimum latency between any two satellites
- Maximum aggregate bandwidth
- Resilience to link failures
- Geographic coverage for data routing
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

from .orbit import Satellite, EARTH_RADIUS_KM
from .isl import InterSatelliteNetwork, has_line_of_sight, link_metrics


@dataclass
class NetworkMetrics:
    """Metrics for a constellation network topology."""
    timestamp: datetime
    n_satellites: int
    n_links: int
    avg_neighbors: float
    max_neighbors: int
    min_neighbors: int
    diameter_hops: int            # Max hops between any two connected sats
    avg_path_length_hops: float   # Average shortest path
    avg_latency_ms: float         # Average link latency
    max_latency_ms: float
    total_bandwidth_gbps: float   # Sum of all link bandwidths
    connected_components: int     # 1 = fully connected
    algebraic_connectivity: float # Fiedler value (0 = disconnected)
    resilience_score: float       # How many links can fail before disconnect


@dataclass
class RoutingEntry:
    """Entry in the routing table."""
    destination: str
    next_hop: str
    hops: int
    latency_ms: float
    bandwidth_gbps: float
    path: List[str]


class NetworkAnalyzer:
    """Analyze and optimize constellation network topology."""

    def __init__(self, isl_network: InterSatelliteNetwork):
        self.network = isl_network

    def analyze(self, timestamp: datetime) -> NetworkMetrics:
        """Full network analysis at a given time."""
        self.network.update(timestamp)
        graph = self.network.graph

        n_sats = len(graph)
        if n_sats == 0:
            return NetworkMetrics(
                timestamp=timestamp, n_satellites=0, n_links=0,
                avg_neighbors=0, max_neighbors=0, min_neighbors=0,
                diameter_hops=0, avg_path_length_hops=0,
                avg_latency_ms=0, max_latency_ms=0,
                total_bandwidth_gbps=0, connected_components=0,
                algebraic_connectivity=0, resilience_score=0,
            )

        # Neighbor stats
        neighbor_counts = [len(neighbors) for neighbors in graph.values()]
        avg_neighbors = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0
        max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        min_neighbors = min(neighbor_counts) if neighbor_counts else 0

        # Link stats
        n_links = sum(len(n) for n in graph.values()) // 2
        all_latencies = []
        all_bandwidths = []
        for neighbors in graph.values():
            for _, metrics in neighbors:
                all_latencies.append(metrics.latency_s * 1000)
                all_bandwidths.append(metrics.bandwidth_gbps)

        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        max_latency = max(all_latencies) if all_latencies else 0
        total_bw = sum(all_bandwidths) / 2  # Each link counted twice

        # Connected components (BFS)
        visited = set()
        components = 0
        for node in graph:
            if node not in visited:
                components += 1
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor, _ in graph.get(current, []):
                        if neighbor not in visited:
                            queue.append(neighbor)

        # All-pairs shortest paths (Floyd-Warshall for small constellations)
        nodes = list(graph.keys())
        n = len(nodes)
        idx = {name: i for i, name in enumerate(nodes)}
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0

        for node, neighbors in graph.items():
            for neighbor, metrics in neighbors:
                i, j = idx[node], idx[neighbor]
                dist[i][j] = 1  # Hop count

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        # Diameter and average path length
        finite_paths = [dist[i][j] for i in range(n) for j in range(n)
                        if i != j and dist[i][j] < INF]
        diameter = int(max(finite_paths)) if finite_paths else 0
        avg_path = sum(finite_paths) / len(finite_paths) if finite_paths else 0

        # Resilience: estimate by min vertex degree
        resilience = min_neighbors if min_neighbors > 0 else 0

        return NetworkMetrics(
            timestamp=timestamp,
            n_satellites=n_sats,
            n_links=n_links,
            avg_neighbors=round(avg_neighbors, 1),
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
            diameter_hops=diameter,
            avg_path_length_hops=round(avg_path, 2),
            avg_latency_ms=round(avg_latency, 2),
            max_latency_ms=round(max_latency, 2),
            total_bandwidth_gbps=round(total_bw, 1),
            connected_components=components,
            algebraic_connectivity=0,  # Would need eigenvalue computation
            resilience_score=resilience,
        )

    def build_routing_table(self, source: str, timestamp: datetime) -> List[RoutingEntry]:
        """Build complete routing table for a satellite."""
        self.network.update(timestamp)
        graph = self.network.graph

        if source not in graph:
            return []

        # Dijkstra from source
        import heapq
        dist = {source: 0.0}
        prev = {}
        pq = [(0.0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for neighbor, metrics in graph.get(u, []):
                new_dist = d + metrics.latency_s * 1000  # ms
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))

        # Build routing entries
        entries = []
        for dest in graph:
            if dest == source:
                continue
            if dest not in dist:
                continue

            # Reconstruct path
            path = []
            current = dest
            while current in prev:
                path.append(current)
                current = prev[current]
            path.append(source)
            path.reverse()

            # Find next hop
            next_hop = path[1] if len(path) > 1 else dest

            # Find bottleneck bandwidth
            min_bw = float('inf')
            for i in range(len(path) - 1):
                for neighbor, metrics in graph.get(path[i], []):
                    if neighbor == path[i + 1]:
                        min_bw = min(min_bw, metrics.bandwidth_gbps)
                        break

            entries.append(RoutingEntry(
                destination=dest,
                next_hop=next_hop,
                hops=len(path) - 1,
                latency_ms=round(dist[dest], 2),
                bandwidth_gbps=round(min_bw if min_bw < float('inf') else 0, 2),
                path=path,
            ))

        entries.sort(key=lambda e: e.latency_ms)
        return entries

    def time_varying_analysis(self, start: datetime, hours: float,
                               step_minutes: float = 5.0) -> List[NetworkMetrics]:
        """Analyze network topology over time."""
        results = []
        current = start
        end = start + timedelta(hours=hours)
        step = timedelta(minutes=step_minutes)

        while current <= end:
            metrics = self.analyze(current)
            results.append(metrics)
            current += step

        return results

    def print_analysis(self, timestamp: datetime):
        """Print network analysis report."""
        m = self.analyze(timestamp)

        print("=" * 60)
        print("  CONSTELLATION NETWORK ANALYSIS")
        print("=" * 60)
        print(f"  Time: {timestamp.isoformat()}")
        print(f"  Satellites: {m.n_satellites}")
        print(f"  Active ISL links: {m.n_links}")
        print(f"  Connected components: {m.connected_components}")
        print(f"  {'FULLY CONNECTED' if m.connected_components == 1 else 'FRAGMENTED'}")
        print(f"\n  Topology:")
        print(f"    Avg neighbors: {m.avg_neighbors}")
        print(f"    Min/Max neighbors: {m.min_neighbors}/{m.max_neighbors}")
        print(f"    Network diameter: {m.diameter_hops} hops")
        print(f"    Avg path length: {m.avg_path_length_hops} hops")
        print(f"\n  Performance:")
        print(f"    Avg link latency: {m.avg_latency_ms:.1f} ms")
        print(f"    Max link latency: {m.max_latency_ms:.1f} ms")
        print(f"    Total bandwidth: {m.total_bandwidth_gbps:.1f} Gbps")
        print(f"    Resilience (min degree): {m.resilience_score}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    from .orbit import starlink_shell_1_sample
    from .isl import InterSatelliteNetwork

    print("Testing with 12-satellite constellation...\n")
    sats = starlink_shell_1_sample(12)
    isl = InterSatelliteNetwork(sats)
    analyzer = NetworkAnalyzer(isl)

    t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    analyzer.print_analysis(t)

    # Routing table for first satellite
    table = analyzer.build_routing_table(sats[0].name, t)
    if table:
        print(f"\n  Routing table for {sats[0].name}:")
        print(f"  {'Dest':<12} {'NextHop':<12} {'Hops':>5} {'Latency':>10} {'BW':>8}")
        print(f"  {'-'*47}")
        for entry in table[:10]:
            print(f"  {entry.destination:<12} {entry.next_hop:<12} {entry.hops:>5} "
                  f"{entry.latency_ms:>9.1f}ms {entry.bandwidth_gbps:>7.1f}Gbps")
