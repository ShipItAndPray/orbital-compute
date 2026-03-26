# Orbital Compute Simulator

**The open-source toolkit for designing, simulating, and operating satellite compute constellations.**

[![Tests](https://github.com/ShipItAndPray/orbital-compute/actions/workflows/test.yml/badge.svg)](https://github.com/ShipItAndPray/orbital-compute/actions)
[![210+ Tests](https://img.shields.io/badge/tests-210%2B%20passing-brightgreen)](https://github.com/ShipItAndPray/orbital-compute/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![21K+ Lines](https://img.shields.io/badge/code-21K%2B%20lines-informational)](https://github.com/ShipItAndPray/orbital-compute)

**[Live Demo](https://shipitandpray.github.io/orbital-compute/)** | **[3D Globe](https://shipitandpray.github.io/orbital-compute/globe.html)** | **[Cost Calculator](https://shipitandpray.github.io/orbital-compute/cost.html)** | **[Constellation Designer](https://shipitandpray.github.io/orbital-compute/designer.html)** | **[Data Pipeline](https://shipitandpray.github.io/orbital-compute/pipeline.html)**

---

Companies like [Starcloud](https://www.ycombinator.com/companies/starcloud), [Axiom Space](https://www.axiomspace.com/orbital-data-center), and [OrbitsEdge](https://orbitsedge.com/) are putting GPUs in orbit. This simulator models the complete space datacenter stack — from orbital mechanics to job scheduling to business economics.

No space hardware needed. No domain expertise required.

```bash
pip install -r requirements.txt
python demo.py
```

## Why In-Orbit Processing?

The data pipeline simulator proves the core value proposition:

```
Strategy                   Generated   Downlinked   Saved    Backlog
raw_downlink                 1042 GB      900 GB    13.6%    142 GB  (growing!)
image_classification         1042 GB       10 GB    99.0%      0 GB
object_detection             1042 GB        5 GB    99.5%      0 GB
```

**Raw downlink can't keep up** — 142 GB backlog growing every day. In-orbit processing saves 99% of bandwidth. This is why orbital compute exists.

## What It Does

### 26 Python Modules

| Category | Modules | What It Models |
|----------|---------|----------------|
| **Orbital Mechanics** | `orbit.py`, `constellations.py` | SGP4 propagation, TLE data, eclipse detection, real Starlink TLEs from CelesTrak |
| **Power & Thermal** | `power.py`, `thermal.py` | Solar panels, battery charge/discharge, Stefan-Boltzmann radiative cooling |
| **Scheduling** | `scheduler.py`, `scheduler_v2.py` | Greedy + look-ahead schedulers with eclipse forecasting (PHOENIX-inspired) |
| **Networking** | `isl.py`, `network.py` | Inter-satellite optical links, routing, topology analysis, Floyd-Warshall |
| **Radiation** | `radiation.py` | SEU fault injection, South Atlantic Anomaly, 4 recovery strategies |
| **Workloads** | `workloads.py` | 9 realistic types: Earth obs, AI inference, defense/ISR, scientific |
| **Ground Stations** | `ground_stations.py` | 10 global stations, elevation-angle contact windows, pass prediction |
| **Data Pipeline** | `data_pipeline.py` | Sensor → storage → in-orbit processing → downlink. Proves 99% bandwidth savings |
| **Mission Planning** | `mission_planner.py` | Ops timeline, conflict detection, data budget calculator, availability |
| **Design** | `designer.py` | Auto-design optimal constellation from requirements + budget |
| **Economics** | `cost_model.py` | CAPEX/OPEX, break-even, ROI, terrestrial comparison. Honest: 24-50x more expensive |
| **Reliability** | `reliability.py` | MTBF, constellation availability (binomial), SLA analysis, degradation curves |
| **Standards** | `formats.py`, `standards.py` | CCSDS OEM, STK ephemeris, ECSS power/thermal/link budgets |
| **Interface** | `api.py`, `cli.py`, `simulator.py` | REST API, unified CLI, core simulation engine |

### Interactive Web App

Live at **[shipitandpray.github.io/orbital-compute](https://shipitandpray.github.io/orbital-compute/)**

- **Dashboard** — Configure constellation, run simulation in-browser, see fleet stats + timelines
- **3D Globe** — Three.js visualization with orbits, ground stations, ISL links, play/pause
- **Cost Calculator** — Drag sliders, see CAPEX/OPEX/break-even update in real-time
- **Constellation Designer** — Input requirements, get optimal design with Walker geometry
- **Data Pipeline** — Canvas chart proving why in-orbit processing is necessary
- **API Docs** — REST endpoint documentation with curl examples

All runs in the browser. No server. No login. No cold starts.

## Quick Start

```bash
git clone https://github.com/ShipItAndPray/orbital-compute.git
cd orbital-compute
pip install -r requirements.txt

# 30-second overview of everything
python -m orbital_compute

# Full demo (all subsystems: orbit + power + thermal + ISL + radiation + scheduling)
python demo.py

# Custom simulation
python run_sim.py --sats 12 --hours 24 --jobs 100

# Constellation designer (auto-design from requirements)
python -m orbital_compute.designer

# Cost analysis
python -m orbital_compute.cost_model

# Data pipeline comparison
python -m orbital_compute.data_pipeline

# Reliability/SLA analysis
python -m orbital_compute.reliability

# Mission planning timeline
python -m orbital_compute.mission_planner

# ECSS standards compliance
python -m orbital_compute.standards

# Industry format export (CCSDS OEM, STK)
python -m orbital_compute.formats

# Web dashboard
python dashboard.py

# REST API
python -m orbital_compute.api --port 8080

# Scheduler benchmark (greedy vs look-ahead)
python benchmark.py

# Fetch real Starlink TLEs
python -m orbital_compute.constellations

# Or use make shortcuts
make test        # Run 200+ tests
make demo        # Full demo
make cost        # Cost analysis
make pipeline    # Data pipeline proof
make serve       # Local web app
```

## Key Results

### Constellation Designer
```
Input:  1000 GPU-hours/day, global coverage, $50M budget
Output: 6 satellites, 8 GPUs each, 2000 km altitude
        $45.9M total (under budget), 1017 GPU-hours/day
```

### Reliability Analysis
```
12-satellite constellation:
  99%    SLA: YES (1 operational needed, 11 spare)
  99.99% SLA: YES (1 operational needed)
  99.999% SLA: NO (need all 12 — not feasible with 1.4yr MTBF)
```

### Cost Model (Honest Assessment)
```
Orbital compute: $86-172/GPU-hour (12 sats, Falcon 9)
AWS equivalent:  $4.10/GPU-hour
Ratio: 24-50x more expensive

But: processing in orbit saves 99% downlink bandwidth.
The value prop is NOT raw $/hr — it's avoiding the downlink bottleneck.
```

### Full Simulation (12 sats, 12h, 80 jobs)
```
Jobs completed:     80/80
Fleet utilization:  32%
Eclipse fraction:   26-37% per satellite
Radiation SEUs:     303 events, all recovered via checkpoint-restart
ISL links:          6 active (mesh topology)
Temperature range:  -19°C to 24°C (within limits)
```

## Architecture

```
orbital_compute/          # 26 Python modules
├── orbit.py              # SGP4 propagation, eclipse detection
├── power.py              # Solar + battery model
├── thermal.py            # Stefan-Boltzmann radiative cooling
├── scheduler.py          # v1: Greedy orbit-aware scheduler
├── scheduler_v2.py       # v2: Look-ahead with eclipse forecasting
├── ground_stations.py    # 10 stations, contact windows
├── isl.py                # Inter-satellite links, LOS, routing
├── network.py            # Topology analysis, Floyd-Warshall routing
├── radiation.py          # SEU injection, SAA, recovery strategies
├── workloads.py          # 9 workload types + mixed traffic generator
├── data_pipeline.py      # Sensor → processing → downlink pipeline
├── mission_planner.py    # Ops timeline, pass prediction, data budgets
├── designer.py           # Auto-design constellation from requirements
├── cost_model.py         # Full economics + terrestrial comparison
├── reliability.py        # MTBF, SLA, degradation curves
├── constellations.py     # Pre-defined configs + real TLE fetch
├── formats.py            # CCSDS OEM, STK ephemeris, JSON Schema
├── standards.py          # ECSS power/thermal/link budgets
├── debris.py             # Collision probability, CAM planning, Kessler risk
├── propulsion.py         # Atmospheric drag, station-keeping, deorbit planning
├── federated.py          # Federated learning across constellation (FedAvg/FedProx)
├── k8s_scheduler.py      # Kubernetes scheduler extender for orbit-aware pods
├── api.py                # REST API
├── simulator.py          # Core simulation engine
├── visualize.py          # 3D Three.js globe generator
└── cli.py                # Unified CLI

docs/                     # Web app (GitHub Pages)
├── index.html            # Landing page
├── dashboard.html        # Interactive simulator
├── globe.html            # 3D orbit visualization
├── cost.html             # Cost calculator
├── designer.html         # Constellation designer
├── pipeline.html         # Data pipeline analysis
└── api-docs.html         # API documentation

tests/                    # 107 tests, all passing
```

## Research References

- **PHOENIX** (IEEE 2024) — Sunlight-aware task scheduling for energy-efficient space edge computing
- **KubeSpace** (arXiv 2601.21383, 2026) — Low-latency K8s control plane for LEO container orchestration
- **Krios** (USENIX HotEdge 2020) — Loosely-coupled orchestration for the LEO edge
- **Comprehensive Survey of Orbital Edge Computing** (Chinese Journal of Aeronautics, 2025)
- **Axiom Space AxDCU-1** — First orbital data center on ISS (2025), Red Hat Device Edge
- **Starcloud** — First H100 GPU in orbit (Nov 2025), NVIDIA partnership
- **ESA ASCEND** — European orbital compute initiative, €300M through 2027

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). High-impact areas:
- Kubernetes scheduler plugin
- Real satellite hardware integration
- Multi-objective constellation optimization
- Federated learning orchestrator

## License

MIT — see [LICENSE](LICENSE)
