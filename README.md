# Orbital Compute Simulator

Simulate and schedule compute jobs across satellite constellations. Models orbital mechanics, power systems, thermal dynamics, and ground station contacts to answer: **when and where can a satellite constellation process your workloads?**

No space hardware needed. No domain expertise required. Just `python run_sim.py`.

## Why This Exists

Companies like Starcloud, Axiom Space, and OrbitsEdge are putting GPUs in orbit. But there's no open-source software for scheduling compute across a satellite constellation. The academic papers (KubeSpace, PHOENIX, Krios) are research-only — no runnable code.

This simulator fills that gap.

## What It Models

| Subsystem | Model | Based On |
|-----------|-------|----------|
| **Orbital Mechanics** | SGP4 propagation from TLE data | Standard NORAD model |
| **Eclipse Detection** | Cylindrical Earth shadow model | Sun position + geometry |
| **Solar Power** | Panel output, battery charge/discharge, eclipse cycles | Typical LEO power budget |
| **Thermal** | Stefan-Boltzmann radiative cooling, solar/Earth IR heating | Vacuum thermal dynamics |
| **Job Scheduling** | Priority queue with power/thermal/eclipse constraints | PHOENIX sunlight-aware scheduling |
| **Ground Stations** | Elevation-angle contact windows, 10 global stations | Real station locations |

## Quick Start

```bash
# Install
git clone https://github.com/your-repo/orbital-compute.git
cd orbital-compute
pip install -r requirements.txt

# Run simulation
python run_sim.py

# Run with custom parameters
python run_sim.py --sats 12 --hours 24 --jobs 100

# Launch web dashboard
python dashboard.py
python dashboard.py --sats 8 --hours 12 --jobs 40
python dashboard.py --no-serve  # Generate HTML file only
```

## Architecture

```
orbital_compute/
├── orbit.py           # SGP4 propagation, eclipse prediction, sun position
├── power.py           # Solar panels, battery, charge/discharge model
├── thermal.py         # Radiative cooling (Stefan-Boltzmann), thermal limits
├── scheduler.py       # Orbit-aware job scheduler (priority, preemption, PHOENIX)
├── ground_stations.py # Contact windows, 10 global ground stations
└── simulator.py       # Full simulation engine
```

### Scheduler Features

- **Eclipse-aware**: Pauses compute during eclipse when battery is low
- **Sunlight-preferring**: Routes new jobs to sunlit satellites (PHOENIX principle)
- **Thermal-throttled**: Reduces compute when temperature approaches limits
- **Preemption**: Checkpointable jobs are paused and resumed, not lost
- **Priority queue**: Jobs scheduled by priority with deadline awareness

### Ground Station Network

10 stations providing global coverage:
Svalbard, Fairbanks, Wallops, Canberra, Bangalore, Hartebeest, Santiago, Tromsø, McMurdo, Singapore

## Example Output

```
Constellation: 8 satellites
Duration: 12.0 hours
Jobs completed: 40/40
Fleet utilization: 23.5%
Total compute delivered: 22.54 hours
Preemption events: 0

Per-Satellite Breakdown:
Satellite   Compute%  Eclipse%  AvgBatt%  AvgTemp  MaxTemp
SAT-000        26.2%     36.9%     97.9%  -11.5°C   21.9°C
SAT-001        23.6%     34.3%     98.1%  -13.5°C   22.6°C
...
```

## Example Scenarios

Pre-computed results are in `examples/`. Run them yourself or inspect the JSON.

### 1. Small Constellation (4 sats, 3h, 10 jobs)
```bash
python run_sim.py --sats 4 --hours 3 --jobs 10
```
| Metric | Value |
|--------|-------|
| Utilization | 42.3% |
| Compute delivered | 5.08h |
| Preemptions | 0 |

All 10 jobs complete. High utilization because few sats are kept busy.

### 2. Medium Constellation (12 sats, 24h, 80 jobs)
```bash
python run_sim.py --sats 12 --hours 24 --jobs 80
```
| Metric | Value |
|--------|-------|
| Utilization | 16.1% |
| Compute delivered | 46.36h |
| Ground contacts | 367 |

Jobs finish in ~5h, then fleet idles. Shows constellation is oversized for this workload.

### 3. Large Constellation (24 sats, 12h, 200 jobs)
```bash
python run_sim.py --sats 24 --hours 12 --jobs 200
```
| Metric | Value |
|--------|-------|
| Utilization | 38.5% |
| Compute delivered | 111.0h |
| Ground contacts | 372 |

200 jobs across 24 satellites. Fleet stays busy longer with balanced load.

### 4. Power-Constrained (6 sats, 500W solar, 1500Wh battery)
```bash
python run_sim.py --sats 6 --hours 6 --jobs 60 --solar-watts 500 --battery-wh 1500
```
| Metric | Value |
|--------|-------|
| Jobs completed | 19/60 |
| Min battery | 75.7% |
| Utilization | 30.0% |

Only 19 of 60 jobs complete — limited solar power can't sustain full compute during eclipse. Battery drops significantly. This is the scenario where scheduling intelligence matters most.

### Web Dashboard
```bash
python dashboard.py --sats 8 --hours 12 --jobs 40
```
Opens a browser dashboard showing fleet status, satellite timelines (eclipse/compute/contacts), ground station contacts, and power budgets.

## Research References

- **PHOENIX** (IEEE 2024) — Sunlight-aware task scheduling for energy-efficient space edge computing
- **KubeSpace** (arXiv 2601.21383, 2026) — Low-latency K8s control plane for LEO container orchestration
- **Krios** (USENIX HotEdge 2020) — Loosely-coupled orchestration for the LEO edge
- **Comprehensive Survey of Orbital Edge Computing** (Chinese Journal of Aeronautics, 2025)
- **Axiom Space AxDCU-1** — First orbital data center on ISS (2025), uses Red Hat Device Edge

## Limitations

- Synthetic TLEs (not real satellite orbits) — works for simulation, not operations
- Simplified thermal model (no multi-node conduction)
- No inter-satellite link simulation (yet)
- No radiation/SEU fault injection (yet)
- Python 3.9+ (not optimized for real-time)

## License

MIT
