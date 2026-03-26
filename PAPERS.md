# Academic Papers — Implementation Tracker

Papers surveyed for algorithms to implement in orbital-compute. Prioritized by impact.

## HIGH Priority — Implement Next

| # | Paper | Year | Key Algorithm | Module | Status |
|---|-------|------|---------------|--------|--------|
| 1 | [Phoenix: Sunlight-Aware Scheduling](https://arxiv.org/abs/2407.07337) | INFOCOM 2024 | SBEO optimization + two-level decomposition | `phoenix.py` | **DONE** |
| 2 | [Battery Aging Model](https://arxiv.org/html/2603.04372v1) | arXiv 2026 | Physics-driven degradation (C-rate, temp, cycle depth) | `power.py` | **DONE** |
| 3 | [RATA: Resource-Aware Task Allocator](https://arxiv.org/abs/2601.06706) | arXiv 2026 | Constellation-wide allocation with blocking analysis | `scheduler_v2.py` | Not started |
| 4 | [RedNet: App-Aware Radiation Tolerance](https://arxiv.org/abs/2407.11853) | arXiv 2024 | DNN layer rearrangement + error-tolerant activations | `radiation.py` | Not started |
| 5 | [Multi-Port ISL Communication](https://arxiv.org/abs/2601.01031) | ASPLOS 2026 | Concurrent multi-link data dissemination | `isl.py` | Not started |

## MEDIUM Priority — Enhance Existing

| # | Paper | Year | Key Algorithm | Module | Status |
|---|-------|------|---------------|--------|--------|
| 6 | [Google Suncatcher](https://arxiv.org/abs/2511.19468) | arXiv 2025 | 81-sat clusters, TPU, thermal/radiator tradeoffs | `designer.py` | Reference |
| 7 | [Energy-Aware FL](https://arxiv.org/abs/2409.14832) | GLOBECOM 2024 | FL training scheduled around eclipse periods | `federated.py` | Not started |
| 8 | [AutoFLSat](https://arxiv.org/abs/2411.00263) | arXiv 2024 | Hierarchical autonomous FL, 12-37% training time reduction | `federated.py` | Not started |
| 9 | [ALANINE: Personalized FL](https://arxiv.org/abs/2411.07752) | IEEE 2024 | Decentralized personalized FL for heterogeneous data | `federated.py` | Not started |
| 10 | [Orbit-Aware Scheduling](https://link.springer.com/article/10.1007/s10586-025-05663-9) | Springer 2025 | Formal orbit-aware optimization | `scheduler_v2.py` | Partial |
| 11 | [AGCPM ISL Routing](https://www.sciencedirect.com/science/article/abs/pii/S1389128625009235) | Elsevier 2025 | Great circle path mapping for 7K+ sats | `network.py` | Not started |
| 12 | [Q-Learning ISL Routing](https://www.sciencedirect.com/science/article/pii/S1110982325000213) | ScienceDirect 2025 | RL-based adaptive routing | `isl.py` | Not started |
| 13 | [SatFlow: Mega-Constellation Planning](https://arxiv.org/abs/2412.20475) | arXiv 2024 | Hierarchical network planning | `network.py` | Not started |
| 14 | [NAS for On-Orbit ML](https://www.nature.com/articles/s41598-025-21467-8) | Nature 2025 | 93% model compression, improved accuracy | `workloads.py` | Not started |

## Reference / Surveys

| Paper | Year | Value |
|-------|------|-------|
| [OEC Comprehensive Survey](https://www.sciencedirect.com/science/article/pii/S1000936124004709) | 2025 | Taxonomy of all OEC algorithms |
| [Computing Over Space Survey](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.06.005) | 2025 | RAD5500 (0.9 GFlops) vs A100 (156 TFlops) gap |
| [NASA HDTN](https://www.nasa.gov/communicating-with-missions/delay-disruption-tolerant-networking/) | 2024 | 900 Mbps DTN on ISS via laser comms |

## Already Implemented

| Capability | Based On | Module |
|-----------|----------|--------|
| **PHOENIX SBEO scheduling** | **INFOCOM 2024 (full implementation)** | **`phoenix.py`** |
| Eclipse-aware look-ahead | Phoenix (simplified) | `scheduler_v2.py` |
| Contact Graph Routing | NASA DTN/CGR standard | `cgr.py` |
| Federated learning (FedAvg/FedProx/SCAFFOLD) | Multiple FL papers | `federated.py` |
| SEU fault injection + SAA | Radiation environment models | `radiation.py` |
| CCSDS/ECSS standards | Space industry standards | `formats.py`, `standards.py` |
| Kessler syndrome model | NASA ORDEM (simplified) | `debris.py` |
| Stefan-Boltzmann thermal | Spacecraft thermal textbooks | `thermal.py` |
