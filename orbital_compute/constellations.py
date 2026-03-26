from __future__ import annotations

"""Pre-defined constellation configurations and real TLE fetching.

Provides both synthetic constellations (for testing) and real TLE data
from CelesTrak for actual satellite tracking.
"""

import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .orbit import Satellite, _fix_tle_checksum


@dataclass
class ConstellationConfig:
    """Defines a satellite constellation."""
    name: str
    altitude_km: float
    inclination_deg: float
    n_planes: int
    sats_per_plane: int
    description: str = ""

    @property
    def total_sats(self) -> int:
        return self.n_planes * self.sats_per_plane


# Reference constellations
CONSTELLATIONS = {
    "starlink-shell1": ConstellationConfig(
        name="Starlink Shell 1",
        altitude_km=550, inclination_deg=53.0,
        n_planes=72, sats_per_plane=22,
        description="SpaceX Starlink first shell (operational)",
    ),
    "starlink-mini": ConstellationConfig(
        name="Starlink Mini",
        altitude_km=550, inclination_deg=53.0,
        n_planes=4, sats_per_plane=6,
        description="Small Starlink-like constellation for testing",
    ),
    "oneweb": ConstellationConfig(
        name="OneWeb",
        altitude_km=1200, inclination_deg=87.9,
        n_planes=18, sats_per_plane=36,
        description="OneWeb LEO broadband constellation",
    ),
    "polar-compute": ConstellationConfig(
        name="Polar Compute",
        altitude_km=600, inclination_deg=97.6,
        n_planes=6, sats_per_plane=8,
        description="Sun-synchronous orbit compute constellation (minimal eclipses)",
    ),
    "equatorial-compute": ConstellationConfig(
        name="Equatorial Compute",
        altitude_km=500, inclination_deg=0.0,
        n_planes=1, sats_per_plane=12,
        description="Equatorial orbit for low-latency tropical coverage",
    ),
    "starcloud": ConstellationConfig(
        name="Starcloud",
        altitude_km=550, inclination_deg=53.0,
        n_planes=3, sats_per_plane=4,
        description="Starcloud-like compute constellation (GPU-equipped)",
    ),
}


def generate_constellation(config: ConstellationConfig,
                           max_sats: Optional[int] = None) -> List[Satellite]:
    """Generate synthetic satellites for a constellation config."""
    satellites = []
    total = config.total_sats
    if max_sats:
        total = min(total, max_sats)

    # Mean motion from altitude (circular orbit approximation)
    # T = 2*pi*sqrt(a^3/mu), a = R_earth + alt, mu = 398600.4418 km^3/s^2
    import math
    a = 6371.0 + config.altitude_km
    mu = 398600.4418
    period_s = 2 * math.pi * math.sqrt(a**3 / mu)
    mean_motion = 86400.0 / period_s  # rev/day

    sat_idx = 0
    for plane in range(config.n_planes):
        raan = (360.0 / config.n_planes) * plane

        for sat_in_plane in range(config.sats_per_plane):
            if sat_idx >= total:
                break

            # Phase offset within plane
            mean_anomaly = (360.0 / config.sats_per_plane) * sat_in_plane

            # Build TLE
            cat_num = 70000 + sat_idx
            line1 = "1 {:05d}U 26001A   26085.00000000  .00000000  00000-0  00000-0 0  9990".format(cat_num)
            line2 = "2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} {:11.8f}    00".format(
                cat_num, config.inclination_deg, raan, mean_anomaly, mean_motion
            )

            line1 = _fix_tle_checksum(line1)
            line2 = _fix_tle_checksum(line2)

            name = "{}-P{:02d}S{:02d}".format(config.name[:8], plane, sat_in_plane)
            satellites.append(Satellite(name, line1, line2))
            sat_idx += 1

    return satellites


def fetch_real_tles(source: str = "starlink", max_sats: int = 20,
                    cache_dir: str = ".tle_cache") -> List[Satellite]:
    """Fetch real TLE data from CelesTrak.

    Sources: 'starlink', 'oneweb', 'planet', 'spire', 'active'
    """
    urls = {
        "starlink": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        "oneweb": "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle",
        "planet": "https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=tle",
        "spire": "https://celestrak.org/NORAD/elements/gp.php?GROUP=spire&FORMAT=tle",
        "active": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    }

    if source not in urls:
        raise ValueError(f"Unknown source: {source}. Available: {list(urls.keys())}")

    # Check cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{source}_tles.txt")
    cache_meta = os.path.join(cache_dir, f"{source}_meta.json")

    use_cache = False
    if os.path.exists(cache_file) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            meta = json.load(f)
        cached_time = datetime.fromisoformat(meta["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600
        if age_hours < 24:
            use_cache = True

    if use_cache:
        with open(cache_file) as f:
            tle_text = f.read()
        print(f"  Using cached TLEs for {source} (age: {age_hours:.1f}h)")
    else:
        print(f"  Fetching TLEs for {source} from CelesTrak...")
        try:
            req = urllib.request.Request(urls[source],
                                         headers={"User-Agent": "orbital-compute/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                tle_text = resp.read().decode()

            with open(cache_file, "w") as f:
                f.write(tle_text)
            with open(cache_meta, "w") as f:
                json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(),
                           "source": source}, f)
        except Exception as e:
            if os.path.exists(cache_file):
                print(f"  Fetch failed ({e}), using stale cache")
                with open(cache_file) as f:
                    tle_text = f.read()
            else:
                raise RuntimeError(f"Failed to fetch TLEs: {e}")

    # Parse TLEs
    lines = [l.strip() for l in tle_text.strip().split("\n") if l.strip()]
    satellites = []

    i = 0
    while i < len(lines) - 2 and len(satellites) < max_sats:
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                satellites.append(Satellite(name, line1, line2))
            except Exception:
                pass  # Skip malformed TLEs
            i += 3
        else:
            i += 1

    print(f"  Loaded {len(satellites)} real satellites from {source}")
    return satellites


if __name__ == "__main__":
    print("=" * 60)
    print("  CONSTELLATION CONFIGURATIONS")
    print("=" * 60)
    print(f"\n  {'Name':<20} {'Alt km':>7} {'Inc°':>5} {'Planes':>7} {'Sats':>5} {'Total':>6}")
    print(f"  {'-'*50}")
    for name, cfg in CONSTELLATIONS.items():
        print(f"  {name:<20} {cfg.altitude_km:>7.0f} {cfg.inclination_deg:>5.1f} "
              f"{cfg.n_planes:>7} {cfg.sats_per_plane:>5} {cfg.total_sats:>6}")
    # Generate and test one
    cfg = CONSTELLATIONS["starcloud"]
    sats = generate_constellation(cfg, max_sats=4)
    from datetime import datetime, timezone
    t = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)
    print(f"\n  Generated {len(sats)} sats for '{cfg.name}':")
    for sat in sats:
        pos = sat.position_at(t)
        print(f"    {sat.name}: {pos.altitude_km:.0f}km, {pos.lat_deg:.1f}°N "
              f"{'[ECLIPSE]' if pos.in_eclipse else ''}")
    print(f"\n{'=' * 60}")
