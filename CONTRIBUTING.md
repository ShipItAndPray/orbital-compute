# Contributing to Orbital Compute Simulator

Thanks for considering contributing! This project models a real gap in the space computing industry — no open-source orbital compute scheduler existed before this.

## Where Help Is Needed

### High Impact
- **Kubernetes scheduler plugin** — The real product. A K8s scheduler extender that takes TLE data and power/thermal constraints as scheduling inputs.
- **Real satellite hardware integration** — Adapt for actual flight computers (NVIDIA Jetson, BAE RAD750).
- **Multi-objective constellation optimization** — Pareto-optimal designs trading cost vs coverage vs latency.
- **Federated learning orchestrator** — Train ML models across distributed satellites without centralizing data.

### Medium Impact
- **Improve thermal model** — Multi-node conduction, phase-change materials, heat pipes.
- **Add atmospheric drag model** — Orbit decay at low altitudes.
- **Propulsion model** — Station-keeping, orbit raising, deorbiting.
- **Space weather integration** — Real-time solar activity affecting radiation model.

### Always Welcome
- Bug fixes
- Documentation improvements
- Test coverage expansion
- Performance optimization
- New workload types

## Getting Started

```bash
git clone https://github.com/ShipItAndPray/orbital-compute.git
cd orbital-compute
pip install -r requirements.txt

# Run tests
python -m unittest discover tests -v

# Run the demo
python demo.py

# Run individual modules
python -m orbital_compute.data_pipeline
python -m orbital_compute.reliability
python -m orbital_compute.designer
```

## Code Style

- Python 3.9+ compatible (use `from __future__ import annotations`)
- Type hints on all public functions
- Docstrings on all modules and classes
- Every new module should have a `__main__` test block
- Add unit tests in `tests/`

## Commit Messages

Format: `<action> <what>`

Examples:
- `Add constellation designer with Walker optimization`
- `Fix eclipse detection for high-inclination orbits`
- `Improve thermal model with multi-node conduction`

## Architecture

Each module is self-contained with minimal coupling:
- `orbit.py` depends on: `sgp4`, `numpy`
- Everything else depends on: `orbit.py` + `numpy`
- Web app: pure HTML/CSS/JS, no build step
- No frameworks, no React, no heavy dependencies

## Research References

If you're adding physics models, cite your sources:
- Link to the paper or textbook
- Note any simplifications made
- Include validation data if possible

## License

MIT. Your contributions will be under the same license.
