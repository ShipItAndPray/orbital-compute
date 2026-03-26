#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Tutorial: AI Inference in Space — The Starcloud Use Case

Demonstrates running LLM inference and image generation on orbital GPUs.

The pitch: if your data is already in orbit (Earth observation imagery,
sensor feeds, satellite-to-satellite comms), why downlink terabytes to
a ground data center? Process it where it lives.

This tutorial walks through:
1. Configure a Starcloud-like constellation (12 sats, 4 GPUs each, H100-class)
2. Generate an AI-heavy workload mix (LLM inference, image classification, image gen)
3. Simulate the scheduler routing inference to sunlit satellites
4. Thermal management — 700W GPUs need serious radiators in vacuum
5. Fleet throughput: tokens/second, images/hour
6. Latency comparison: in-orbit vs ground (upload + process + download)
7. Cost per inference vs AWS pricing
8. Scheduler benchmark: greedy vs look-ahead
9. Break-even analysis: at what fleet size does orbital beat AWS?
"""

import math
import random
from datetime import datetime, timedelta, timezone

from orbital_compute.simulator import Simulation, SimulationConfig
from orbital_compute.thermal import ThermalModel, ThermalConfig, STEFAN_BOLTZMANN
from orbital_compute.power import PowerModel, PowerConfig
from orbital_compute.scheduler import OrbitalScheduler, ComputeJob, JobType
from orbital_compute.scheduler_v2 import LookAheadScheduler
from orbital_compute.workloads import (
    WorkloadGenerator, WorkloadSpec, WORKLOAD_CATALOG,
    create_job, CATEGORY_AI,
)
from orbital_compute.cost_model import (
    ConstellationCostConfig, LaunchCosts, HardwareCosts, OperatingCosts,
    RevenueModel, TerrestrialComparison,
    calculate_constellation_costs, print_cost_report,
)


# ---------------------------------------------------------------------------
# Constants — realistic hardware specs
# ---------------------------------------------------------------------------

# H100 SXM inference performance (publicly available NVIDIA benchmarks)
H100_TDP_WATTS = 700.0
H100_TOKENS_PER_SEC_LLAMA70B = 2400   # Llama-2 70B, FP8, batch=64
H100_TOKENS_PER_SEC_LLAMA7B = 12000   # Llama-2 7B, FP8, batch=128
H100_IMAGES_PER_SEC_RESNET = 15000    # ResNet-50 classification
H100_IMAGES_PER_MIN_SDXL = 8          # Stable Diffusion XL, 1024x1024

# Orbital data link speeds (realistic for LEO Ka-band)
UPLINK_MBPS = 50       # Ground-to-sat uplink
DOWNLINK_MBPS = 200    # Sat-to-ground downlink
INTER_SAT_MBPS = 10_000  # Optical inter-satellite link (Starlink-class)

# LEO orbit parameters
ORBIT_PERIOD_MIN = 95.0
ECLIPSE_FRACTION = 0.36  # ~34 min eclipse per 95 min orbit at 550 km / 53 deg


def section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}\n")


# =========================================================================
# Section 1: Constellation Configuration
# =========================================================================

def configure_constellation():
    """Configure a Starcloud-like GPU constellation."""
    section("1. CONSTELLATION: Starcloud-class GPU Fleet")

    n_sats = 12
    gpus_per_sat = 4
    total_gpus = n_sats * gpus_per_sat

    # Each satellite needs enough solar power for 4x H100 GPUs + housekeeping
    gpu_power = gpus_per_sat * H100_TDP_WATTS  # 2800W compute
    housekeeping = 200.0                        # ADCS, comms, OBC
    total_power = gpu_power + housekeeping      # 3000W

    # Solar panels must cover compute + charging during sunlit period
    # Sunlit ~64% of orbit, need to charge battery for ~36% eclipse
    # During sunlit: generate power for compute AND store energy for eclipse
    solar_needed = total_power / 0.64 * 1.15  # 15% margin
    battery_kwh = total_power * (ECLIPSE_FRACTION * ORBIT_PERIOD_MIN * 60 / 3600) / 0.95
    # That's energy needed for one eclipse crossing at full compute

    print(f"  Constellation: {n_sats} satellites, 550 km, 53 deg inclination")
    print(f"  GPUs: {gpus_per_sat}x H100-class per satellite ({total_gpus} total)")
    print(f"  Per-satellite power budget:")
    print(f"    GPU compute:   {gpu_power:,.0f} W  ({gpus_per_sat}x {H100_TDP_WATTS:.0f}W)")
    print(f"    Housekeeping:  {housekeeping:,.0f} W")
    print(f"    Total load:    {total_power:,.0f} W")
    print(f"    Solar panels:  {solar_needed:,.0f} W (sized for sunlit charging)")
    print(f"    Battery:       {battery_kwh:,.1f} kWh (one eclipse at full compute)")
    print(f"  Fleet total:     {total_gpus} GPUs = {total_gpus * H100_TDP_WATTS / 1000:.0f} kW compute")

    print(f"\n  [NOTE] For context, the ISS has ~120 kW of solar panels.")
    print(f"  This constellation needs {solar_needed * n_sats / 1000:.0f} kW total")
    print(f"  across {n_sats} satellites — ambitious but not impossible with")
    print(f"  next-gen deployable solar arrays (e.g., Maxar ROSA, 200 W/kg).")

    return n_sats, gpus_per_sat, total_power, solar_needed, battery_kwh


# =========================================================================
# Section 2: Workload Generation
# =========================================================================

def generate_ai_workload():
    """Generate an AI-heavy workload mix."""
    section("2. WORKLOAD: AI Inference Mix (60/30/10)")

    # Custom AI-heavy mix: 60% LLM, 30% image classification, 10% image gen
    ai_mix = {
        "ai_inference": 0.60,      # Mapped to LLM + image gen below
        "earth_observation": 0.30,  # Image classification
        "scientific": 0.05,
        "defense_isr": 0.05,
    }

    t0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    gen = WorkloadGenerator(mix=ai_mix, seed=42)
    jobs = gen.generate_batch(200, t0, duration_hours=6.0)

    summary = gen.summary(jobs)
    print(f"  Generated {summary['total_jobs']} jobs over 6 hours:")
    print(f"    Total compute: {summary['total_compute_hours']:.1f} hours")
    print(f"    Total energy:  {summary['total_energy_kwh']:.1f} kWh")
    print(f"    Downlink data: {summary['total_downlink_gb']:.1f} GB")

    print(f"\n  Workload breakdown:")
    for name, count in sorted(summary['by_workload'].items(), key=lambda x: -x[1]):
        pct = count / summary['total_jobs'] * 100
        print(f"    {name:<22s} {count:>4d} jobs ({pct:>5.1f}%)")

    print(f"\n  By scheduling type:")
    for jtype, count in sorted(summary['by_type'].items()):
        print(f"    {jtype:<12s} {count:>4d} jobs")

    return jobs, t0


# =========================================================================
# Section 3: Scheduler Simulation — Sunlit Routing
# =========================================================================

def run_scheduler_sim(n_sats, total_power_w, solar_w, battery_kwh):
    """Run the full simulation showing sunlit-preference scheduling."""
    section("3. SCHEDULER: Sunlit-Preference Routing")

    print("  The PHOENIX scheduling principle: prefer sunlit satellites.")
    print("  GPUs in sunlight have unlimited solar power.")
    print("  GPUs in eclipse drain battery — save those for emergencies.\n")

    # Use the simulation engine with our Starcloud config
    config = SimulationConfig(
        n_satellites=n_sats,
        sim_duration_hours=6.0,
        time_step_seconds=60.0,
        n_jobs=120,  # Heavy GPU load
        solar_panel_watts=solar_w,
        battery_capacity_wh=battery_kwh * 1000,  # Convert kWh to Wh
        radiator_area_m2=16.0,  # Large radiators for GPU heat
        housekeeping_watts=200.0,
        job_power_range=(500.0, 700.0),     # GPU-class jobs
        job_duration_range=(30.0, 300.0),    # Quick inference tasks
    )

    sim = Simulation(config)
    sim.setup()
    results = sim.run()
    sim.print_report()

    # Analyze sunlit vs eclipse compute
    sunlit_compute = 0
    eclipse_compute = 0
    total_compute = 0
    for node in sim.nodes:
        for h in node.power_history:
            if h["computing"]:
                total_compute += 1
                if h["in_eclipse"]:
                    eclipse_compute += 1
                else:
                    sunlit_compute += 1

    if total_compute > 0:
        print(f"\n  Sunlit scheduling analysis:")
        print(f"    Compute in sunlight:  {sunlit_compute:>5d} steps ({sunlit_compute/total_compute*100:.1f}%)")
        print(f"    Compute in eclipse:   {eclipse_compute:>5d} steps ({eclipse_compute/total_compute*100:.1f}%)")
        print(f"    --> Scheduler routes {sunlit_compute/total_compute*100:.0f}% of work to sunlit sats")
    else:
        print(f"\n  [No compute steps recorded in sampled telemetry]")

    return results


# =========================================================================
# Section 4: Thermal Management
# =========================================================================

def thermal_analysis(gpus_per_sat):
    """GPU thermal management in vacuum."""
    section("4. THERMAL: Managing 2.8 kW of GPU Heat in Vacuum")

    print("  In space, heat can ONLY be shed by radiation (Stefan-Boltzmann).")
    print("  No air. No convection. No fans. Just radiator panels glowing infrared.\n")

    gpu_heat = gpus_per_sat * H100_TDP_WATTS * 0.95  # 95% of power becomes heat
    housekeeping_heat = 200.0 * 0.8
    total_heat = gpu_heat + housekeeping_heat

    print(f"  Heat budget:")
    print(f"    {gpus_per_sat} GPUs @ {H100_TDP_WATTS}W * 95% efficiency = {gpu_heat:,.0f} W heat")
    print(f"    Housekeeping:                             {housekeeping_heat:,.0f} W heat")
    print(f"    Total dissipation needed:                 {total_heat:,.0f} W")

    # Calculate required radiator area for steady-state at 70C
    target_temp_c = 70.0
    target_k = target_temp_c + 273.15
    emissivity = 0.90
    space_k = 3.0

    # Radiative power = emissivity * sigma * A * (T^4 - T_space^4)
    # Solve for A: A = Q / (emissivity * sigma * (T^4 - T_space^4))
    # But we must also account for solar and Earth IR heating
    solar_absorbed_per_m2 = 0.20 * 1361.0 * 0.25  # Absorptivity * solar flux * geometry
    earth_ir_per_m2 = 240.0 * 0.5                   # Earth IR * view factor

    q_radiated_per_m2 = emissivity * STEFAN_BOLTZMANN * (target_k**4 - space_k**4)
    q_env_per_m2 = solar_absorbed_per_m2 + earth_ir_per_m2
    q_net_per_m2 = q_radiated_per_m2 - q_env_per_m2

    if q_net_per_m2 > 0:
        required_area = total_heat / q_net_per_m2
    else:
        required_area = float('inf')

    print(f"\n  Radiator sizing (steady-state at {target_temp_c}C):")
    print(f"    Radiative power per m^2: {q_radiated_per_m2:,.0f} W/m^2")
    print(f"    Environmental heating:   {q_env_per_m2:,.0f} W/m^2 (solar + Earth IR)")
    print(f"    Net cooling per m^2:     {q_net_per_m2:,.0f} W/m^2")
    print(f"    Required radiator area:  {required_area:,.1f} m^2")
    print(f"    That's a {math.sqrt(required_area):.1f}m x {math.sqrt(required_area):.1f}m panel (each side)")

    # Simulate thermal transient: sunlit -> eclipse with full GPU load
    print(f"\n  Thermal transient simulation (one orbit, full GPU load):")
    thermal_cfg = ThermalConfig(
        radiator_area_m2=max(required_area, 16.0),  # Use sized radiator
        radiator_emissivity=emissivity,
        thermal_mass_j_per_k=80000,  # Heavier satellite
        max_temp_c=85.0,
        compute_temp_limit_c=75.0,
    )
    thermal = ThermalModel(thermal_cfg, initial_temp_c=30.0)

    orbit_seconds = int(ORBIT_PERIOD_MIN * 60)
    eclipse_start = int(orbit_seconds * 0.64)  # Eclipse starts at 64%
    dt = 60  # 1 min steps

    temps = []
    throttles = []
    for t in range(0, orbit_seconds, dt):
        in_eclipse = t >= eclipse_start
        state = thermal.step(dt, total_heat, in_eclipse)
        temps.append(state.temp_c)
        throttles.append(state.throttle_pct)

    max_temp = max(temps)
    min_temp = min(temps)
    throttle_steps = sum(1 for t in throttles if t > 0)

    print(f"    Radiator area used: {thermal_cfg.radiator_area_m2:.1f} m^2")
    print(f"    Temperature range: {min_temp:.1f}C to {max_temp:.1f}C")
    print(f"    Throttle events: {throttle_steps}/{len(temps)} timesteps")
    if max_temp > thermal_cfg.compute_temp_limit_c:
        print(f"    WARNING: Peak temp {max_temp:.1f}C exceeds compute limit "
              f"({thermal_cfg.compute_temp_limit_c}C)")
        print(f"    --> Need larger radiators or duty-cycle the GPUs")
    else:
        print(f"    Thermal margin: {thermal_cfg.compute_temp_limit_c - max_temp:.1f}C below limit")

    # What's the max sustainable GPU count?
    max_sustainable = thermal.max_sustainable_heat_w(target_temp_c)
    max_gpus = int(max_sustainable / (H100_TDP_WATTS * 0.95))
    print(f"\n  Max sustainable heat at {target_temp_c}C: {max_sustainable:,.0f} W")
    print(f"  Max sustainable GPUs: {max_gpus} (at {thermal_cfg.radiator_area_m2:.0f} m^2 radiator)")

    return required_area


# =========================================================================
# Section 5: Throughput Calculation
# =========================================================================

def throughput_analysis(n_sats, gpus_per_sat):
    """Calculate fleet-wide inference throughput."""
    section("5. THROUGHPUT: Fleet-Wide Inference Capacity")

    total_gpus = n_sats * gpus_per_sat

    # Compute availability: ~64% sunlit (full power), ~36% eclipse
    # During eclipse: ~50% of sats can still compute (battery permitting)
    # Realistic duty cycle accounting for thermal throttling, scheduling gaps
    effective_duty_cycle = 0.55  # Conservative: 55% of time actually computing

    active_gpus_avg = total_gpus * effective_duty_cycle

    # LLM inference (Llama-2 70B — the realistic choice for space edge)
    llm_tokens_per_sec = active_gpus_avg * H100_TOKENS_PER_SEC_LLAMA70B
    llm_tokens_per_hour = llm_tokens_per_sec * 3600
    llm_requests_per_sec = llm_tokens_per_sec / 500  # ~500 tokens per response

    # Image classification
    img_class_per_sec = active_gpus_avg * H100_IMAGES_PER_SEC_RESNET
    img_class_per_hour = img_class_per_sec * 3600

    # Image generation (SDXL)
    img_gen_per_min = active_gpus_avg * H100_IMAGES_PER_MIN_SDXL
    img_gen_per_hour = img_gen_per_min * 60

    print(f"  Fleet: {total_gpus} GPUs, ~{effective_duty_cycle*100:.0f}% effective duty cycle")
    print(f"  Average active GPUs: {active_gpus_avg:.0f}")

    print(f"\n  LLM Inference (Llama-2 70B, FP8):")
    print(f"    {llm_tokens_per_sec:,.0f} tokens/sec fleet-wide")
    print(f"    {llm_tokens_per_hour:,.0f} tokens/hour")
    print(f"    ~{llm_requests_per_sec:,.0f} requests/sec (500 tok/response)")

    print(f"\n  Image Classification (ResNet-50):")
    print(f"    {img_class_per_sec:,.0f} images/sec")
    print(f"    {img_class_per_hour/1e6:,.1f}M images/hour")

    print(f"\n  Image Generation (SDXL 1024x1024):")
    print(f"    {img_gen_per_min:,.0f} images/min")
    print(f"    {img_gen_per_hour:,.0f} images/hour")

    # The real value proposition
    print(f"\n  [KEY INSIGHT]")
    print(f"  Raw throughput is NOT the value proposition.")
    print(f"  AWS has millions of GPUs. We have {total_gpus}.")
    print(f"  The value is DATA LOCALITY — processing data where it's captured.")
    print(f"  A 2 TB satellite image takes ~22 hours to downlink at {DOWNLINK_MBPS} Mbps.")
    print(f"  In-orbit classification takes ~2 minutes and downlinks 1 KB of results.")

    return {
        "llm_tok_per_sec": llm_tokens_per_sec,
        "img_class_per_hour": img_class_per_hour,
        "img_gen_per_hour": img_gen_per_hour,
        "active_gpus": active_gpus_avg,
    }


# =========================================================================
# Section 6: Latency Comparison
# =========================================================================

def latency_comparison():
    """Compare in-orbit vs ground latency for different workloads."""
    section("6. LATENCY: In-Orbit vs Ground Processing")

    print("  For data-locality workloads, orbit wins on latency.\n")

    # Scenario: classify a 2 GB satellite image
    image_size_gb = 2.0
    image_size_mb = image_size_gb * 1024
    result_size_kb = 10.0  # Classification result is tiny

    # Ground path: downlink -> process -> uplink result
    downlink_time_s = (image_size_mb * 8) / DOWNLINK_MBPS
    ground_process_s = 120.0  # 2 min on H100
    uplink_result_s = (result_size_kb / 1024 * 8) / UPLINK_MBPS
    ground_total_s = downlink_time_s + ground_process_s + uplink_result_s

    # Add ground-station wait time (average time until next contact)
    # At 550 km, typical satellite contacts a station 4-6 times per day
    avg_contacts_per_day = 5
    avg_wait_for_contact_s = (24 * 3600 / avg_contacts_per_day) / 2  # Half the interval
    ground_with_wait_s = ground_total_s + avg_wait_for_contact_s

    # Orbit path: process locally, downlink result only
    orbit_process_s = 120.0  # Same compute time
    orbit_downlink_result_s = (result_size_kb / 1024 * 8) / DOWNLINK_MBPS
    orbit_total_s = orbit_process_s + orbit_downlink_result_s
    orbit_with_wait_s = orbit_total_s + avg_wait_for_contact_s

    print(f"  Scenario: Classify a {image_size_gb:.0f} GB satellite image")
    print(f"  {'':─<65}")

    print(f"\n  GROUND processing path:")
    print(f"    Wait for ground contact:   {avg_wait_for_contact_s:>10,.0f}s ({avg_wait_for_contact_s/3600:.1f} hrs)")
    print(f"    Downlink {image_size_gb}GB image:       {downlink_time_s:>10,.0f}s ({downlink_time_s/60:.1f} min)")
    print(f"    Process on ground H100:    {ground_process_s:>10,.0f}s ({ground_process_s/60:.1f} min)")
    print(f"    Uplink result ({result_size_kb:.0f} KB):    {uplink_result_s:>10.1f}s")
    print(f"    TOTAL (process only):      {ground_total_s:>10,.0f}s ({ground_total_s/60:.1f} min)")
    print(f"    TOTAL (with contact wait): {ground_with_wait_s:>10,.0f}s ({ground_with_wait_s/3600:.1f} hrs)")

    print(f"\n  IN-ORBIT processing path:")
    print(f"    Process on orbital H100:   {orbit_process_s:>10,.0f}s ({orbit_process_s/60:.1f} min)")
    print(f"    Wait for ground contact:   {avg_wait_for_contact_s:>10,.0f}s ({avg_wait_for_contact_s/3600:.1f} hrs)")
    print(f"    Downlink result ({result_size_kb:.0f} KB):  {orbit_downlink_result_s:>10.1f}s")
    print(f"    TOTAL:                     {orbit_with_wait_s:>10,.0f}s ({orbit_with_wait_s/3600:.1f} hrs)")

    speedup = ground_with_wait_s / orbit_with_wait_s
    bandwidth_saved = (1 - result_size_kb / 1024 / image_size_gb / 1024) * 100

    print(f"\n  Result:")
    print(f"    Latency advantage:    {speedup:.1f}x faster in orbit")
    print(f"    Bandwidth saved:      {bandwidth_saved:.1f}%")
    print(f"    Ground contact time saved: {downlink_time_s/60:.0f} min/image")

    # Now show where ground wins: general-purpose LLM queries
    print(f"\n  {'':─<65}")
    print(f"\n  Scenario: General LLM query (no orbital data needed)")
    print(f"    Uplink prompt (2 KB):  {2*8/UPLINK_MBPS*1000:.1f} ms")
    print(f"    Process on AWS H100:   ~1 s")
    print(f"    Downlink response:     ~0.1 ms")
    print(f"    TOTAL ground:          ~1 s")
    print(f"    In-orbit:              Same ~1 s compute")
    print(f"    --> No advantage for general queries. Orbit only wins")
    print(f"        when the DATA is already in space.")


# =========================================================================
# Section 7: Cost Per Inference vs AWS
# =========================================================================

def cost_analysis(n_sats, gpus_per_sat):
    """Cost per inference vs AWS."""
    section("7. COST: Orbital Inference vs AWS Pricing")

    total_gpus = n_sats * gpus_per_sat

    # Configure orbital constellation cost
    cfg = ConstellationCostConfig(
        n_satellites=n_sats,
        mission_lifetime_years=5.0,
        launch=LaunchCosts(default_vehicle="falcon9"),
        hardware=HardwareCosts(
            n_gpus=gpus_per_sat,
            gpu_unit_cost=30_000.0,
            solar_panel_watts=5400,      # Sized from Section 1
            battery_capacity_kwh=20.0,
            radiator_area_m2=16.0,
            satellite_dry_mass_kg=350.0,  # Heavy with 4 GPUs
        ),
    )

    # Assume 50% utilization (realistic for orbital constraints)
    analysis = calculate_constellation_costs(cfg, utilization_pct=50.0)
    print_cost_report(analysis)

    # Per-inference cost comparison
    orbital_cost_per_hour = analysis["economics"]["cost_per_compute_hour"]
    aws_cost_per_hour = analysis["terrestrial_comparison"]["cost_per_compute_hour"]

    # LLM inference: ~2400 tok/s per GPU on Llama-70B
    tokens_per_gpu_hour = H100_TOKENS_PER_SEC_LLAMA70B * 3600
    orbital_cost_per_1k_tokens = orbital_cost_per_hour / tokens_per_gpu_hour * 1000
    aws_cost_per_1k_tokens = aws_cost_per_hour / tokens_per_gpu_hour * 1000

    # Image classification: 15000 img/s per GPU
    images_per_gpu_hour = H100_IMAGES_PER_SEC_RESNET * 3600
    orbital_cost_per_1k_images = orbital_cost_per_hour / images_per_gpu_hour * 1000
    aws_cost_per_1k_images = aws_cost_per_hour / images_per_gpu_hour * 1000

    print(f"\n  Per-Inference Cost Comparison:")
    print(f"  {'':─<60}")
    print(f"  {'Metric':<35s} {'Orbital':>12s} {'AWS':>12s}")
    print(f"  {'':─<60}")
    print(f"  {'GPU-hour cost':<35s} ${orbital_cost_per_hour:>11,.2f} ${aws_cost_per_hour:>11,.2f}")
    print(f"  {'LLM (per 1K tokens, Llama-70B)':<35s} ${orbital_cost_per_1k_tokens:>11,.6f} ${aws_cost_per_1k_tokens:>11,.6f}")
    print(f"  {'Classification (per 1K images)':<35s} ${orbital_cost_per_1k_images:>11,.6f} ${aws_cost_per_1k_images:>11,.6f}")
    print(f"  {'Cost multiple (orbital/AWS)':<35s} {orbital_cost_per_hour/aws_cost_per_hour:>11.1f}x {'1.0x':>12s}")

    print(f"\n  [HONEST ASSESSMENT]")
    print(f"  Orbital compute is {orbital_cost_per_hour/aws_cost_per_hour:.0f}x more expensive per GPU-hour.")
    print(f"  For general workloads, AWS wins on cost every time.")
    print(f"  But add downlink costs for data-locality workloads:")

    # Data-locality adjusted cost
    # If you need to downlink 2GB per job to process on ground:
    downlink_gb_per_job = 2.0
    ground_station_cost_per_gb = 5.0 * 60 / (DOWNLINK_MBPS / 8 / 1024) / 1024  # $/GB from contact time
    # Actually let's be more direct:
    # At $5/min ground station time and 200 Mbps downlink:
    gb_per_min = DOWNLINK_MBPS / 8 / 1024 * 60  # GB per minute
    downlink_cost_per_gb = 5.0 / gb_per_min

    downlink_cost_per_job = downlink_gb_per_job * downlink_cost_per_gb
    ground_process_time_h = 120.0 / 3600  # 2 min = 0.033 hrs
    ground_total_per_job = aws_cost_per_hour * ground_process_time_h + downlink_cost_per_job
    orbit_total_per_job = orbital_cost_per_hour * ground_process_time_h

    print(f"\n  Data-locality scenario (classify 2 GB satellite image):")
    print(f"    Ground: ${aws_cost_per_hour * ground_process_time_h:.4f} compute"
          f" + ${downlink_cost_per_job:.2f} downlink = ${ground_total_per_job:.2f}/job")
    print(f"    Orbit:  ${orbit_total_per_job:.4f} compute + $0.00 downlink"
          f" = ${orbit_total_per_job:.4f}/job")

    if orbit_total_per_job < ground_total_per_job:
        print(f"    --> ORBIT WINS by ${ground_total_per_job - orbit_total_per_job:.2f}/job for data-locality workloads")
    else:
        print(f"    --> Ground still wins by ${orbit_total_per_job - ground_total_per_job:.2f}/job")
        print(f"        (orbit needs cheaper launch or higher utilization)")

    return analysis


# =========================================================================
# Section 8: Scheduler Benchmark — Greedy vs Look-Ahead
# =========================================================================

def scheduler_benchmark():
    """Benchmark greedy vs look-ahead scheduler with GPU workloads."""
    section("8. SCHEDULER BENCHMARK: Greedy vs Look-Ahead")

    print("  Testing both schedulers with heavy GPU inference workloads.")
    print("  Look-ahead knows when eclipses start — can it plan better?\n")

    random.seed(42)
    t0 = datetime(2026, 3, 26, 0, 0, 0, tzinfo=timezone.utc)
    n_sats = 12
    dt = 60.0  # 1 min steps
    sim_hours = 6.0
    total_steps = int(sim_hours * 3600 / dt)

    # Generate GPU-heavy jobs
    n_jobs = 150
    jobs_template = []
    for i in range(n_jobs):
        power = random.uniform(500, 700)  # GPU-class
        duration = random.uniform(30, 300)  # Inference tasks
        jtype = random.choice([JobType.REALTIME, JobType.REALTIME, JobType.CHECKPOINT])
        priority = random.randint(2, 6)
        deadline = t0 + timedelta(hours=random.uniform(0.5, 6.0)) if jtype == JobType.REALTIME else None

        jobs_template.append({
            "power": power, "duration": duration,
            "jtype": jtype, "priority": priority, "deadline": deadline,
        })

    results = {}

    for sched_name, SchedulerClass in [("Greedy", OrbitalScheduler),
                                         ("Look-Ahead", LookAheadScheduler)]:
        scheduler = SchedulerClass()

        # Create fresh jobs for each scheduler
        for i, tmpl in enumerate(jobs_template):
            job = ComputeJob(
                job_id=f"{sched_name[0]}-{i:04d}",
                name=f"gpu-inference-{i}",
                power_watts=tmpl["power"],
                duration_seconds=tmpl["duration"],
                job_type=tmpl["jtype"],
                priority=tmpl["priority"],
                deadline=tmpl["deadline"],
                checkpointable=(tmpl["jtype"] == JobType.CHECKPOINT),
            )
            scheduler.submit_job(job)

        # Create power/thermal models for each satellite
        power_models = {}
        thermal_models = {}
        for s in range(n_sats):
            name = f"SAT-{s:02d}"
            power_models[name] = PowerModel(PowerConfig(
                solar_panel_watts=5400,
                battery_capacity_wh=20000,
                housekeeping_watts=200,
            ))
            thermal_models[name] = ThermalModel(ThermalConfig(
                radiator_area_m2=16.0,
                thermal_mass_j_per_k=80000,
            ))

        # Simulate
        compute_steps = {f"SAT-{s:02d}": 0 for s in range(n_sats)}
        for step in range(total_steps):
            current_time = t0 + timedelta(seconds=step * dt)

            for s in range(n_sats):
                name = f"SAT-{s:02d}"

                # Simple eclipse model: each sat has offset eclipse pattern
                orbit_pos = (step * dt + s * ORBIT_PERIOD_MIN * 60 / n_sats) % (ORBIT_PERIOD_MIN * 60)
                in_eclipse = orbit_pos > (ORBIT_PERIOD_MIN * 60 * 0.64)

                current_job = scheduler.running_jobs.get(name)
                compute_w = current_job.power_watts if current_job else 0.0
                heat_w = current_job.heat_output_watts if current_job else 0.0

                power_state = power_models[name].step(dt, in_eclipse, compute_w)
                thermal_state = thermal_models[name].step(dt, 200 * 0.8 + heat_w, in_eclipse)

                decision = scheduler.decide(
                    name, current_time,
                    power_state.available_for_compute_w,
                    power_state.battery_pct,
                    thermal_state.can_compute,
                    thermal_state.throttle_pct,
                    in_eclipse,
                )

                if decision.action == "run" and decision.job:
                    scheduler.advance_job(name, dt, thermal_state.throttle_pct, current_time)
                    compute_steps[name] += 1

        stats = scheduler.stats()
        total_compute_s = sum(j.duration_seconds for j in scheduler.completed_jobs)
        failed_jobs = sum(1 for j in scheduler.job_queue if j.status.value == "failed")
        # Count missed deadlines
        missed = 0
        for j in scheduler.completed_jobs:
            if j.deadline and j.completed_at and j.completed_at > j.deadline:
                missed += 1
        for j in scheduler.job_queue:
            if j.deadline and j.deadline < t0 + timedelta(hours=sim_hours):
                missed += 1

        fleet_util = sum(compute_steps.values()) / (n_sats * total_steps) * 100

        results[sched_name] = {
            "completed": stats["completed"],
            "preempted": stats["preempted"],
            "compute_hours": total_compute_s / 3600,
            "fleet_util": fleet_util,
            "missed_deadlines": missed,
        }

    # Print comparison
    print(f"  {'Metric':<30s} {'Greedy':>12s} {'Look-Ahead':>12s} {'Winner':>10s}")
    print(f"  {'':─<65}")

    for metric, key, higher_better in [
        ("Jobs completed", "completed", True),
        ("Preemption events", "preempted", False),
        ("Compute hours delivered", "compute_hours", True),
        ("Fleet utilization %", "fleet_util", True),
        ("Missed deadlines", "missed_deadlines", False),
    ]:
        g = results["Greedy"][key]
        l = results["Look-Ahead"][key]

        if isinstance(g, float):
            g_str = f"{g:.1f}"
            l_str = f"{l:.1f}"
        else:
            g_str = f"{g}"
            l_str = f"{l}"

        if g == l:
            winner = "tie"
        elif (g > l) == higher_better:
            winner = "Greedy"
        else:
            winner = "Look-Ahead"

        print(f"  {metric:<30s} {g_str:>12s} {l_str:>12s} {winner:>10s}")


# =========================================================================
# Section 9: Break-Even Fleet Size Analysis
# =========================================================================

def breakeven_analysis():
    """At what fleet size does orbital inference become cost-competitive?"""
    section("9. BREAK-EVEN: When Does Orbital Beat AWS?")

    print("  Scanning fleet sizes from 6 to 500 satellites...")
    print("  Assumption: Starship launch ($500/kg), 4 GPUs/sat, 50% utilization\n")

    fleet_sizes = [6, 12, 24, 48, 96, 192, 384, 500]
    aws_cost = TerrestrialComparison().cost_per_compute_hour(50.0)

    print(f"  AWS baseline: ${aws_cost:.2f}/GPU-hour (H100, 50% utilization)")
    print(f"\n  {'Fleet':>6s} {'GPUs':>6s} {'$/GPU-hr':>10s} {'vs AWS':>8s} {'Breakeven?':>12s}")
    print(f"  {'':─<50}")

    crossover = None
    for n in fleet_sizes:
        cfg = ConstellationCostConfig(
            n_satellites=n,
            mission_lifetime_years=5.0,
            launch=LaunchCosts(default_vehicle="starship"),  # Starship for scale
            hardware=HardwareCosts(
                n_gpus=4,
                solar_panel_watts=5400,
                battery_capacity_kwh=20.0,
                radiator_area_m2=16.0,
                satellite_dry_mass_kg=350.0,
            ),
        )
        analysis = calculate_constellation_costs(cfg, utilization_pct=50.0)
        cost = analysis["economics"]["cost_per_compute_hour"]
        multiple = cost / aws_cost

        competitive = "YES" if multiple <= 1.0 else "no"
        if multiple <= 1.0 and crossover is None:
            crossover = n

        print(f"  {n:>6d} {n*4:>6d} ${cost:>9,.2f} {multiple:>7.1f}x {competitive:>12s}")

    print(f"\n  {'':─<65}")
    if crossover:
        print(f"\n  Cost-competitive at {crossover} satellites ({crossover * 4} GPUs)")
    else:
        print(f"\n  Never cost-competitive for general compute (even at 500 sats)")

    print(f"\n  [HONEST CONCLUSION]")
    print(f"  For GENERAL workloads: orbital compute does not beat AWS on cost.")
    print(f"  The economics of Earth-based data centers (cheap power, cheap cooling,")
    print(f"  no launch costs, easy maintenance) are simply too strong.")
    print(f"")
    print(f"  For DATA-LOCALITY workloads (the Starcloud thesis):")
    print(f"  - Earth observation: process imagery in orbit, downlink alerts only")
    print(f"  - Satellite-to-satellite: process ISL data without touching ground")
    print(f"  - Time-critical: wildfire/disaster detection in <5 min")
    print(f"  - Bandwidth-constrained: when you can't afford to downlink TB/day")
    print(f"")
    print(f"  In these cases, the comparison isn't $/GPU-hour.")
    print(f"  It's $/insight-delivered-in-time. And orbit wins.")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 72)
    print("  TUTORIAL: AI Inference in Space — The Starcloud Use Case")
    print("  Running LLM inference and image generation on orbital GPUs")
    print("=" * 72)

    # 1. Configure constellation
    n_sats, gpus_per_sat, total_power, solar_w, battery_kwh = configure_constellation()

    # 2. Generate workload
    jobs, t0 = generate_ai_workload()

    # 3. Run scheduler simulation
    results = run_scheduler_sim(n_sats, total_power, solar_w, battery_kwh)

    # 4. Thermal analysis
    radiator_area = thermal_analysis(gpus_per_sat)

    # 5. Throughput
    throughput = throughput_analysis(n_sats, gpus_per_sat)

    # 6. Latency comparison
    latency_comparison()

    # 7. Cost analysis
    cost = cost_analysis(n_sats, gpus_per_sat)

    # 8. Scheduler benchmark
    scheduler_benchmark()

    # 9. Break-even
    breakeven_analysis()

    # Final summary
    section("FINAL SUMMARY")
    print(f"  Constellation: {n_sats} sats x {gpus_per_sat} GPUs = {n_sats * gpus_per_sat} H100-class GPUs")
    print(f"  Radiator per sat: {radiator_area:.1f} m^2 for {gpus_per_sat}x H100 thermal load")
    print(f"  Fleet throughput: {throughput['llm_tok_per_sec']:,.0f} tok/s (Llama-70B)")
    print(f"                    {throughput['img_class_per_hour']/1e6:.0f}M img/hr (classification)")
    print(f"  Cost multiple vs AWS: {cost['terrestrial_comparison']['orbital_cost_multiple']:.1f}x")
    print(f"  Value proposition: data locality, not raw $/GPU-hr")
    print(f"\n  'The wise see knowledge and action as one' — BG 5.4")
    print(f"  Process data where it is born. That is the orbital dharma.")
    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
