# Reproducibility Statement

Every claim in this project can be verified by running a command.

| Claim | Command | Expected Output |
|-------|---------|-----------------|
| 99% bandwidth savings | `python -m orbital_compute.data_pipeline` | "image_classification: 99.0% saved" |
| 24-50x more expensive than AWS | `python -m orbital_compute.cost_model` | "Cost multiple: 50.1x" |
| Battery fades to 66% in 3 years | `python -c "from orbital_compute.power import BatteryAgingModel; b=BatteryAgingModel(); [b.cycle(30) for _ in range(16425)]; print(f'SoH: {b.soh_pct:.1f}%')"` | "SoH: ~66%" |
| PHOENIX improves battery by 23pp | `python -m orbital_compute.phoenix` | "SAT-003: 37.1% → 60.1%" |
| CGR finds routes where Dijkstra fails | `python -m orbital_compute.cgr` | "NN: NO ROUTE, CGR: delivered" |
| 0.80% fleet collision probability | `python -m orbital_compute.debris` | "Fleet collision prob: 0.0080" |
| FCC 5-year deorbit compliant at 550km | `python -m orbital_compute.propulsion` | "5-year compliant: YES" |
| 222+ tests pass on Python 3.9-3.12 | `python -m unittest discover tests` | "Ran 222 tests... OK" |
| Federated learning 97.6% loss reduction | `python -m orbital_compute.federated` | "Loss reduction: 97.6%" |

## How to reproduce everything

```bash
git clone https://github.com/ShipItAndPray/orbital-compute.git
cd orbital-compute
pip install -r requirements.txt
python -m orbital_compute    # 30-second overview
python -m unittest discover tests -v   # All tests
```

Every number has a source. Every source has a test. If it doesn't reproduce, file an issue.
