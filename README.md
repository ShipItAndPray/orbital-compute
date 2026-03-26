# Orbital Compute Simulator

Simulate and schedule compute jobs across satellite constellations. Models orbital mechanics, power systems, thermal dynamics, inter-satellite links, radiation effects, and ground station contacts.

**No space hardware needed. No domain expertise required.**

```bash
pip install -r requirements.txt
python run_sim.py --sats 12 --hours 24 --jobs 100
```

## Why This Exists

Companies like [Starcloud](https://www.ycombinator.com/companies/starcloud), [Axiom Space](https://www.axiomspace.com/orbital-data-center), and [OrbitsEdge](https://orbitsedge.com/) are putting GPUs in orbit. But there's no open-source software for scheduling compute across a satellite constellation. Academic papers exist (KubeSpace, PHOENIX, Krios) — but none ship runnable code.

This simulator fills that gap.

## What It Models

| Subsystem | Model | Based On |
|-----------|-------|----------|
| **Orbital Mechanics** | SGP4 propagation from TLE data | Standard NORAD model |
| **Eclipse Detection** | Cylindrical Earth shadow model | Sun position + geometry |
| **Solar Power** | Panel output, battery charge/discharge, eclipse cycles | Typical LEO power budget |
| **Thermal** | Stefan-Boltzmann radiative cooling, solar/Earth IR | Vacuum thermal dynamics |
| **Job Scheduling** | Look-ahead with eclipse forecasting + load balancing | PHOENIX sunlight-aware scheduling |
| **Ground Stations** | Elevation-angle contact windows, 10 global stations | Real station locations |
| **Inter-Satellite Links** | Line-of-sight, distance/latency/bandwidth, routing | Optical ISL model |
| **Radiation** | SEU fault injection, South Atlantic Anomaly, recovery | Poisson upset model |
| **Workloads** | 9 realistic types (Earth obs, AI, defense, scientific) | Mission profiles |
| **Cost Model** | CAPEX, OPEX, break-even, ROI, terrestrial comparison | Industry pricing |

## Quick Start

```bash
git clone https://github.com/ShipItAndPray/orbital-compute.git
cd orbital-compute
pip install -r requirements.txt

# Basic simulation
python run_sim.py

# Custom constellation
python run_sim.py --sats 12 --hours 24 --jobs 100

# Full-stack demo (all subsystems)
python demo.py

# Web dashboard
python dashboard.py

# Scheduler benchmark (v1 greedy vs v2 look-ahead)
python benchmark.py

# REST API
python -m orbital_compute.api
```

## Architecture

```
orbital_compute/
├── orbit.py           # SGP4 propagation, eclipse prediction, sun position
├── power.py           # Solar panels, battery, charge/discharge model
├── thermal.py         # Radiative cooling (Stefan-Boltzmann), thermal limits
├── scheduler.py       # v1: Greedy orbit-aware scheduler
├── scheduler_v2.py    # v2: Look-ahead with eclipse forecasting + load balancing
├── ground_stations.py # Contact windows, 10 global ground stations
├── isl.py             # Inter-satellite links: LOS, routing, bandwidth
├── radiation.py       # SEU fault injection, SAA detection, recovery strategies
├── workloads.py       # 9 workload types + mixed traffic generator
├── constellations.py  # Pre-defined configs + real TLE fetching (CelesTrak)
├── cost_model.py      # Economics: CAPEX, OPEX, break-even, ROI
├── visualize.py       # 3D orbit visualization (Three.js)
├── api.py             # REST API (status, satellites, jobs, contacts, metrics)
├── simulator.py       # Main simulation engine
└── cli.py             # Unified CLI
```

## Features

### Scheduling
- **Eclipse-aware**: Pauses compute during eclipse when battery is low
- **Sunlight-preferring**: Routes new jobs to sunlit satellites (PHOENIX principle)
- **Look-ahead**: Pre-computes eclipse windows, plans jobs that finish before darkness
- **Thermal-throttled**: Reduces compute when temperature approaches limits
- **Preemption**: Checkpointable jobs are paused and resumed, not lost
- **Load balancing**: Distributes jobs evenly across the constellation
- **Deadline-aware**: Escalates priority as deadlines approach

### Workloads
| Type | Power | Duration | Category |
|------|-------|----------|----------|
| ImageClassification | 200W | 120s | Earth Observation |
| ChangeDetection | 300W | 300s | Earth Observation |
| ObjectTracking | 400W | 60s | Earth Observation (realtime) |
| LLMInference | 600W | 30s | AI Inference (realtime) |
| ImageGeneration | 500W | 45s | AI Inference |
| WeatherModel | 800W | 3600s | Scientific (batch) |
| ClimateAnalysis | 700W | 7200s | Scientific (batch) |
| SAR_Processing | 500W | 180s | Defense/ISR |
| SignalAnalysis | 300W | 60s | Defense/ISR (realtime) |

### Radiation Protection
| Strategy | Overhead | Coverage |
|----------|----------|----------|
| None | 1.0x | ECC only (catches 99% single-bit) |
| Checkpoint-Restart | 1.05x | Recovers from all caught errors |
| Dual Execution | 2.0x | Detects + recovers all errors |
| Triple Modular Redundancy | 3.0x | Masks errors via majority vote |

### Ground Station Network
10 stations providing global coverage:
Svalbard, Fairbanks, Wallops, Canberra, Bangalore, Hartebeest, Santiago, Tromso, McMurdo, Singapore

## Example Scenarios

Pre-computed results in `examples/`.

### Small Constellation (4 sats, 3h)
```bash
python run_sim.py --sats 4 --hours 3 --jobs 10
# Utilization: 42.3% | Compute: 5.08h | All jobs complete
```

### Power-Constrained (500W solar, 1500Wh battery)
```bash
python run_sim.py --sats 6 --hours 6 --jobs 60 --solar-watts 500 --battery-wh 1500
# Jobs completed: 19/60 | Battery drops to 75.7%
# This is where scheduling intelligence matters most
```

### Full-Stack Demo
```bash
python demo.py
# 12 Starcloud sats | 80 mixed workloads | ISL mesh | Radiation model
# 303 SEU events, all recovered | 80/80 jobs complete
```

## REST API

```bash
python -m orbital_compute.api --port 8080
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Constellation overview |
| `/satellites` | GET | Per-satellite detail |
| `/jobs` | GET | All jobs (pending/running/completed) |
| `/jobs` | POST | Submit new job `{"workload": "llm_inference"}` |
| `/contacts` | GET | Ground station contact windows |
| `/metrics` | GET | Fleet utilization, throughput, power stats |

## Research References

- **PHOENIX** (IEEE 2024) — Sunlight-aware task scheduling for energy-efficient space edge computing
- **KubeSpace** (arXiv 2601.21383, 2026) — Low-latency K8s control plane for LEO container orchestration
- **Krios** (USENIX HotEdge 2020) — Loosely-coupled orchestration for the LEO edge
- **Comprehensive Survey of Orbital Edge Computing** (Chinese Journal of Aeronautics, 2025)
- **Axiom Space AxDCU-1** — First orbital data center on ISS (2025), uses Red Hat Device Edge
- **Starcloud** — First H100 GPU in orbit (Nov 2025), NVIDIA partnership

## License

MIT — see [LICENSE](LICENSE)
