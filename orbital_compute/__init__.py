"""Orbital Compute Simulator — Schedule compute jobs across satellite constellations.

Quick start:
    from orbital_compute.simulator import Simulation, SimulationConfig
    sim = Simulation(SimulationConfig(n_satellites=6, sim_duration_hours=6, n_jobs=20))
    sim.setup()
    results = sim.run()
"""
__version__ = "0.3.0"
