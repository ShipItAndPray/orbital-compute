# Examples

## Tutorials

| Script | Use Case | Key Finding |
|--------|----------|-------------|
| `tutorial_earth_observation.py` | Wildfire detection constellation | 99% bandwidth savings with in-orbit classification |
| `tutorial_defense_isr.py` | SAR persistent surveillance (South China Sea) | 3 min alert latency vs 48 min ground-based |
| `tutorial_ai_inference.py` | Starcloud-like H100 constellation | 63K tok/s fleet, 23.7x AWS cost, data locality wins |
| `real_starlink_demo.py` | Fetch real Starlink TLEs, compare with synthetic | Eclipse patterns match within 0.8% |

Run any tutorial:
```bash
cd orbital-compute
python examples/tutorial_earth_observation.py
```

## Pre-computed Results (JSON)

| File | Config | Key Result |
|------|--------|------------|
| `small_constellation_4sat_3h.json` | 4 sats, 3h, 10 jobs | 42.3% utilization |
| `default_6sat_6h.json` | 6 sats, 6h, 20 jobs | Baseline run |
| `constrained_power_6sat_6h.json` | 500W solar, 1500Wh battery | Only 19/60 jobs complete |
| `medium_constellation_12sat_24h.json` | 12 sats, 24h, 80 jobs | 16.1% utilization |
| `large_constellation_24sat_12h.json` | 24 sats, 12h, 200 jobs | 38.5% utilization, 111h compute |
| `full_stack_demo.json` | 12 sats, all subsystems | 303 SEU events, all recovered |
| `scheduler_benchmark.json` | Greedy vs look-ahead | v2 better load balancing |
