# Changelog

## [0.2.0] - 2026-03-26

### Added
- **Constellation Designer** — auto-design optimal constellation from requirements + budget
  - Walker Delta generator, coverage analysis per latitude band
  - Evaluates 10K+ candidate designs, picks cheapest feasible
- **Data Pipeline Simulator** — proves 99% bandwidth savings with in-orbit processing
  - Sensor → storage → GPU processing → downlink flow model
  - Side-by-side comparison: raw downlink vs in-orbit processing
- **Reliability Analyzer** — MTBF, SLA analysis, degradation curves
  - Binomial availability model, component-level MTBF
  - SLA feasibility checker (99% through 99.999%)
- **Mission Planner** — operations timeline, conflict detection, pass prediction
  - iCalendar export, data budget calculator, availability analysis
- **Industry Formats** — CCSDS OEM, STK ephemeris, TLE validation
  - Ground station databases (NASA NEN, ESA ESTRACK, KSAT)
  - JSON Schema for simulation outputs
- **ECSS Standards** — power, thermal, and link budget compliance
  - S/X/Ka/Optical band link budget calculator
- **Network Analyzer** — ISL mesh topology analysis
  - Floyd-Warshall routing, connectivity analysis, resilience scoring
- **Cost Model** — full CAPEX/OPEX economics with terrestrial comparison
  - Honest result: 24-50x more expensive, but 99% bandwidth savings justify it
- **Interactive Web App** (GitHub Pages)
  - Dashboard, 3D Globe, Cost Calculator, Constellation Designer, Data Pipeline
  - All runs in-browser, no server needed
- **163 tests** across 8 test files
- **GitHub Actions CI** — tests on Python 3.9-3.12
- **CONTRIBUTING.md** — guide for open-source contributors

## [0.1.0] - 2026-03-26

### Added
- Core simulation engine with SGP4 orbital propagation
- Eclipse detection (cylindrical Earth shadow model)
- Solar power + battery model
- Stefan-Boltzmann thermal model
- Orbit-aware job scheduler (v1 greedy + v2 look-ahead)
- 10 global ground stations with contact windows
- Inter-satellite link simulation
- Radiation fault injection (SEU, SAA, 4 recovery strategies)
- 9 realistic workload types
- Pre-defined constellation configs + real TLE fetching
- REST API, CLI, web dashboard
- Scheduler benchmark (greedy vs look-ahead)
