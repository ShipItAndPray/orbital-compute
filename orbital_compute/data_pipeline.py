from __future__ import annotations

"""Data pipeline simulator — model data flow from sensors to ground.

The #1 value prop of orbital compute: process data in orbit to avoid
the downlink bottleneck. This module models the full data pipeline:
  sensor → onboard storage → in-orbit processing → downlink → ground

Key question it answers: "How much downlink bandwidth do I save by
processing data in orbit vs downloading raw?"
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


@dataclass
class Sensor:
    """An imaging/data sensor on a satellite."""
    name: str
    data_rate_mbps: float          # Raw data generation rate
    duty_cycle_pct: float = 100.0  # % of orbit sensor is active
    compression_ratio: float = 2.0 # Lossless compression on raw data
    resolution_m: float = 1.0      # Ground sample distance (for EO)
    swath_km: float = 100.0        # Imaging swath width


@dataclass
class OnboardStorage:
    """Satellite onboard storage."""
    capacity_gb: float = 1000.0    # Total storage capacity
    write_speed_mbps: float = 5000 # Write speed
    read_speed_mbps: float = 5000  # Read speed
    used_gb: float = 0.0

    @property
    def available_gb(self) -> float:
        return max(0, self.capacity_gb - self.used_gb)

    @property
    def fill_pct(self) -> float:
        return (self.used_gb / self.capacity_gb) * 100 if self.capacity_gb > 0 else 100

    def write(self, gb: float) -> float:
        """Write data. Returns amount actually written (may be less if full)."""
        writable = min(gb, self.available_gb)
        self.used_gb += writable
        return writable

    def read(self, gb: float) -> float:
        """Read (and delete) data. Returns amount actually read."""
        readable = min(gb, self.used_gb)
        self.used_gb -= readable
        return readable


@dataclass
class ProcessingResult:
    """Result of in-orbit processing."""
    input_gb: float
    output_gb: float              # Processed output (much smaller than input)
    processing_time_s: float
    power_watts: float
    reduction_ratio: float        # input/output — the key metric

    @property
    def bandwidth_saved_pct(self) -> float:
        """How much downlink bandwidth was saved."""
        if self.input_gb == 0:
            return 0.0
        return (1 - self.output_gb / self.input_gb) * 100


@dataclass
class InOrbitProcessor:
    """In-orbit data processing capability."""
    name: str
    # Processing specs
    throughput_gbps: float = 0.1   # GB/s processing throughput
    power_watts: float = 500.0     # Power draw during processing
    # Data reduction by processing type
    reduction_ratios: dict = field(default_factory=lambda: {
        "raw_to_compressed": 4.0,       # Lossless compression
        "image_classification": 100.0,   # Raw image → classification labels
        "change_detection": 50.0,        # Two images → change mask
        "object_detection": 200.0,       # Raw image → bounding boxes
        "spectral_analysis": 20.0,       # Hyperspectral → analysis results
        "sar_processing": 10.0,          # Raw SAR → processed image
        "video_analytics": 500.0,        # Video → event summaries
        "weather_features": 30.0,        # Raw obs → feature extraction
    })

    def process(self, input_gb: float, task_type: str) -> ProcessingResult:
        """Process data and return result."""
        ratio = self.reduction_ratios.get(task_type, 10.0)
        output_gb = input_gb / ratio
        time_s = input_gb / self.throughput_gbps
        return ProcessingResult(
            input_gb=input_gb,
            output_gb=output_gb,
            processing_time_s=time_s,
            power_watts=self.power_watts,
            reduction_ratio=ratio,
        )


@dataclass
class DownlinkConfig:
    """Ground station downlink capabilities."""
    x_band_mbps: float = 800.0    # X-band downlink speed
    ka_band_mbps: float = 2000.0  # Ka-band downlink speed
    optical_mbps: float = 10000.0 # Optical downlink speed
    active_band: str = "x_band"   # Which band to use

    @property
    def rate_mbps(self) -> float:
        rates = {
            "x_band": self.x_band_mbps,
            "ka_band": self.ka_band_mbps,
            "optical": self.optical_mbps,
        }
        return rates.get(self.active_band, self.x_band_mbps)


@dataclass
class PipelineMetrics:
    """Metrics for a data pipeline simulation."""
    total_data_generated_gb: float = 0.0
    total_data_processed_gb: float = 0.0
    total_data_downlinked_gb: float = 0.0
    total_processing_results_gb: float = 0.0
    total_contact_time_s: float = 0.0
    total_downlink_time_s: float = 0.0
    storage_high_water_gb: float = 0.0
    storage_overflow_gb: float = 0.0     # Data lost due to full storage
    downlink_backlog_gb: float = 0.0     # Data waiting to be downlinked
    orbits_simulated: int = 0

    @property
    def bandwidth_saved_pct(self) -> float:
        if self.total_data_generated_gb == 0:
            return 0.0
        return (1 - self.total_data_downlinked_gb / self.total_data_generated_gb) * 100

    @property
    def effective_downlink_utilization(self) -> float:
        if self.total_contact_time_s == 0:
            return 0.0
        return (self.total_downlink_time_s / self.total_contact_time_s) * 100


class DataPipeline:
    """Simulate the full data pipeline for one satellite."""

    def __init__(self, sensor: Sensor, storage: OnboardStorage,
                 processor: Optional[InOrbitProcessor],
                 downlink: DownlinkConfig,
                 process_in_orbit: bool = True,
                 processing_task: str = "image_classification"):
        self.sensor = sensor
        self.storage = storage
        self.processor = processor
        self.downlink = downlink
        self.process_in_orbit = process_in_orbit
        self.processing_task = processing_task
        self.metrics = PipelineMetrics()

        # Queues
        self.raw_queue_gb: float = 0.0
        self.processed_queue_gb: float = 0.0

    def simulate_orbit(self, orbital_period_s: float, eclipse_fraction: float,
                       contact_windows: List[Tuple[float, float]]):
        """Simulate one complete orbit.

        Args:
            orbital_period_s: Orbital period in seconds
            eclipse_fraction: Fraction of orbit in eclipse [0,1]
            contact_windows: List of (start_s, end_s) within the orbit
        """
        self.metrics.orbits_simulated += 1
        dt = 10.0  # 10-second timesteps

        sunlit_time = orbital_period_s * (1 - eclipse_fraction)
        sensor_time = sunlit_time * (self.sensor.duty_cycle_pct / 100)

        # Phase 1: Data collection (during sunlit, sensor-active time)
        data_collected_gb = (self.sensor.data_rate_mbps * sensor_time / 8) / 1000  # Mbps → GB
        data_collected_gb /= self.sensor.compression_ratio  # Apply compression

        written = self.storage.write(data_collected_gb)
        overflow = data_collected_gb - written
        self.raw_queue_gb += written
        self.metrics.total_data_generated_gb += data_collected_gb
        self.metrics.storage_overflow_gb += overflow
        self.metrics.storage_high_water_gb = max(
            self.metrics.storage_high_water_gb, self.storage.used_gb
        )

        # Phase 2: In-orbit processing (if enabled)
        if self.process_in_orbit and self.processor and self.raw_queue_gb > 0:
            # Process as much as we can in the available time
            available_time = orbital_period_s * 0.7  # 70% of orbit for processing
            max_processable = self.processor.throughput_gbps * available_time
            to_process = min(self.raw_queue_gb, max_processable)

            result = self.processor.process(to_process, self.processing_task)
            self.raw_queue_gb -= to_process
            self.storage.read(to_process)  # Remove raw data
            self.processed_queue_gb += result.output_gb
            self.storage.write(result.output_gb)  # Store processed results
            self.metrics.total_data_processed_gb += to_process
            self.metrics.total_processing_results_gb += result.output_gb

        # Phase 3: Downlink during ground station contacts
        for contact_start, contact_end in contact_windows:
            contact_duration = contact_end - contact_start
            self.metrics.total_contact_time_s += contact_duration

            downlink_rate_gbps = self.downlink.rate_mbps / 8 / 1000  # Mbps → GB/s

            if self.process_in_orbit:
                # Downlink processed results (much smaller)
                to_downlink = min(self.processed_queue_gb,
                                   downlink_rate_gbps * contact_duration)
                self.processed_queue_gb -= to_downlink
                self.storage.read(to_downlink)
            else:
                # Downlink raw data (huge)
                to_downlink = min(self.raw_queue_gb + self.processed_queue_gb,
                                   downlink_rate_gbps * contact_duration)
                # Drain raw first, then processed
                raw_dl = min(self.raw_queue_gb, to_downlink)
                self.raw_queue_gb -= raw_dl
                self.storage.read(raw_dl)
                remaining = to_downlink - raw_dl
                if remaining > 0:
                    proc_dl = min(self.processed_queue_gb, remaining)
                    self.processed_queue_gb -= proc_dl
                    self.storage.read(proc_dl)

            self.metrics.total_data_downlinked_gb += to_downlink
            if to_downlink > 0:
                self.metrics.total_downlink_time_s += min(
                    contact_duration, to_downlink / downlink_rate_gbps
                )

        self.metrics.downlink_backlog_gb = self.raw_queue_gb + self.processed_queue_gb

    def report(self) -> dict:
        m = self.metrics
        return {
            "orbits": m.orbits_simulated,
            "data_generated_gb": round(m.total_data_generated_gb, 2),
            "data_processed_gb": round(m.total_data_processed_gb, 2),
            "data_downlinked_gb": round(m.total_data_downlinked_gb, 2),
            "bandwidth_saved_pct": round(m.bandwidth_saved_pct, 1),
            "storage_high_water_gb": round(m.storage_high_water_gb, 2),
            "storage_overflow_gb": round(m.storage_overflow_gb, 2),
            "downlink_backlog_gb": round(m.downlink_backlog_gb, 2),
            "downlink_utilization_pct": round(m.effective_downlink_utilization, 1),
            "process_in_orbit": self.process_in_orbit,
        }


def compare_pipeline_strategies(n_orbits: int = 15,
                                 orbital_period_s: float = 5700,
                                 eclipse_fraction: float = 0.35):
    """Compare in-orbit processing vs raw downlink."""
    # Typical EO satellite: 1 Gbps sensor, 1 TB storage
    sensor = Sensor("EO-Camera", data_rate_mbps=1000, duty_cycle_pct=30,
                     compression_ratio=2.0, resolution_m=0.5, swath_km=150)
    downlink = DownlinkConfig(active_band="x_band")

    # Contact: ~10 min per orbit (optimistic with global ground network)
    contact_windows = [(4800, 5400)]  # 10 min contact near end of orbit

    print("=" * 70)
    print("  DATA PIPELINE COMPARISON: In-Orbit Processing vs Raw Downlink")
    print("=" * 70)
    print(f"\n  Sensor: {sensor.data_rate_mbps} Mbps, {sensor.duty_cycle_pct}% duty cycle")
    print(f"  Downlink: {downlink.rate_mbps} Mbps ({downlink.active_band})")
    print(f"  Orbits: {n_orbits} ({n_orbits * orbital_period_s / 3600:.1f} hours)")
    print(f"  Contact: {(contact_windows[0][1]-contact_windows[0][0])/60:.0f} min/orbit")

    strategies = {}

    for task_name in ["raw_downlink", "image_classification", "change_detection",
                       "object_detection", "sar_processing"]:
        process_in_orbit = task_name != "raw_downlink"
        processor = InOrbitProcessor("GPU-1") if process_in_orbit else None

        pipeline = DataPipeline(
            sensor=sensor,
            storage=OnboardStorage(capacity_gb=2000),
            processor=processor,
            downlink=downlink,
            process_in_orbit=process_in_orbit,
            processing_task=task_name if process_in_orbit else "",
        )

        for _ in range(n_orbits):
            pipeline.simulate_orbit(orbital_period_s, eclipse_fraction, contact_windows)

        strategies[task_name] = pipeline.report()

    # Print comparison
    print(f"\n  {'Strategy':<25} {'Generated':>10} {'Downlinked':>12} {'Saved':>8} {'Backlog':>10} {'Overflow':>10}")
    print(f"  {'-'*75}")
    for name, r in strategies.items():
        print(f"  {name:<25} {r['data_generated_gb']:>9.1f}GB {r['data_downlinked_gb']:>11.1f}GB "
              f"{r['bandwidth_saved_pct']:>6.1f}% {r['downlink_backlog_gb']:>9.1f}GB "
              f"{r['storage_overflow_gb']:>9.1f}GB")

    # Key insight
    raw = strategies["raw_downlink"]
    best = min(strategies.items(), key=lambda x: x[1]["downlink_backlog_gb"] if x[0] != "raw_downlink" else float('inf'))

    print(f"\n  KEY INSIGHT:")
    print(f"  Raw downlink backlog after {n_orbits} orbits: {raw['downlink_backlog_gb']:.1f} GB")
    print(f"  Best in-orbit strategy ({best[0]}): {best[1]['downlink_backlog_gb']:.1f} GB backlog")
    print(f"  Bandwidth savings: {best[1]['bandwidth_saved_pct']:.0f}%")

    if raw['storage_overflow_gb'] > 0:
        print(f"\n  WARNING: Raw downlink causes {raw['storage_overflow_gb']:.1f} GB data loss!")
        print(f"  Storage fills up because downlink can't keep up with sensor data rate.")
        print(f"  This is why in-orbit processing exists.")

    print(f"\n{'=' * 70}")
    return strategies


if __name__ == "__main__":
    compare_pipeline_strategies()
