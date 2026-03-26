"""Tests for data_pipeline module — in-orbit processing vs raw downlink."""

import unittest

from orbital_compute.data_pipeline import (
    DataPipeline,
    DownlinkConfig,
    InOrbitProcessor,
    OnboardStorage,
    PipelineMetrics,
    ProcessingResult,
    Sensor,
)


class TestOnboardStorage(unittest.TestCase):
    """Tests for OnboardStorage."""

    def test_write_within_capacity(self):
        """Writing within capacity stores all data."""
        storage = OnboardStorage(capacity_gb=100.0, used_gb=0.0)
        written = storage.write(50.0)
        self.assertAlmostEqual(written, 50.0)
        self.assertAlmostEqual(storage.used_gb, 50.0)

    def test_write_overflow(self):
        """Writing beyond capacity returns only what fits."""
        storage = OnboardStorage(capacity_gb=100.0, used_gb=80.0)
        written = storage.write(30.0)
        self.assertAlmostEqual(written, 20.0)
        self.assertAlmostEqual(storage.used_gb, 100.0)

    def test_read_within_available(self):
        """Reading available data returns correct amount."""
        storage = OnboardStorage(capacity_gb=100.0, used_gb=50.0)
        read = storage.read(30.0)
        self.assertAlmostEqual(read, 30.0)
        self.assertAlmostEqual(storage.used_gb, 20.0)

    def test_read_more_than_available(self):
        """Reading more than available returns only what exists."""
        storage = OnboardStorage(capacity_gb=100.0, used_gb=10.0)
        read = storage.read(50.0)
        self.assertAlmostEqual(read, 10.0)
        self.assertAlmostEqual(storage.used_gb, 0.0)

    def test_fill_pct(self):
        """Fill percentage is correct."""
        storage = OnboardStorage(capacity_gb=200.0, used_gb=50.0)
        self.assertAlmostEqual(storage.fill_pct, 25.0)

    def test_available_gb(self):
        """Available GB is capacity minus used."""
        storage = OnboardStorage(capacity_gb=200.0, used_gb=50.0)
        self.assertAlmostEqual(storage.available_gb, 150.0)


class TestInOrbitProcessor(unittest.TestCase):
    """Tests for InOrbitProcessor."""

    def test_process_reduces_data(self):
        """Processing always produces less output than input."""
        proc = InOrbitProcessor("GPU-1")
        result = proc.process(10.0, "image_classification")
        self.assertLess(result.output_gb, result.input_gb)

    def test_different_tasks_different_reductions(self):
        """Different processing types give different reduction ratios."""
        proc = InOrbitProcessor("GPU-1")
        r_class = proc.process(10.0, "image_classification")
        r_compress = proc.process(10.0, "raw_to_compressed")
        r_video = proc.process(10.0, "video_analytics")
        # image_classification: 100x, raw_to_compressed: 4x, video_analytics: 500x
        self.assertNotAlmostEqual(r_class.output_gb, r_compress.output_gb)
        self.assertNotAlmostEqual(r_class.output_gb, r_video.output_gb)
        # video_analytics has highest reduction (500x), so smallest output
        self.assertLess(r_video.output_gb, r_class.output_gb)
        self.assertLess(r_class.output_gb, r_compress.output_gb)

    def test_bandwidth_saved_pct(self):
        """Bandwidth saved percentage is correct."""
        proc = InOrbitProcessor("GPU-1")
        result = proc.process(100.0, "image_classification")  # 100x reduction
        # output = 100/100 = 1.0 GB, saved = (1 - 1/100)*100 = 99%
        self.assertAlmostEqual(result.bandwidth_saved_pct, 99.0)

    def test_unknown_task_uses_default_ratio(self):
        """Unknown task type uses default reduction ratio of 10x."""
        proc = InOrbitProcessor("GPU-1")
        result = proc.process(10.0, "unknown_task_type")
        self.assertAlmostEqual(result.output_gb, 1.0)  # 10/10 = 1.0

    def test_processing_time_proportional_to_input(self):
        """Processing time scales with input size."""
        proc = InOrbitProcessor("GPU-1", throughput_gbps=1.0)
        r1 = proc.process(10.0, "image_classification")
        r2 = proc.process(20.0, "image_classification")
        self.assertAlmostEqual(r2.processing_time_s, r1.processing_time_s * 2.0)


class TestDataPipeline(unittest.TestCase):
    """Tests for DataPipeline end-to-end."""

    def _make_pipeline(self, process_in_orbit=True, storage_gb=2000.0,
                       task="image_classification"):
        sensor = Sensor("TestCam", data_rate_mbps=1000, duty_cycle_pct=30,
                        compression_ratio=2.0)
        storage = OnboardStorage(capacity_gb=storage_gb)
        processor = InOrbitProcessor("GPU-1") if process_in_orbit else None
        downlink = DownlinkConfig(active_band="x_band")
        return DataPipeline(
            sensor=sensor, storage=storage, processor=processor,
            downlink=downlink, process_in_orbit=process_in_orbit,
            processing_task=task,
        )

    def test_raw_downlink_accumulates_backlog(self):
        """Without in-orbit processing, backlog grows because downlink is slower."""
        pipeline = self._make_pipeline(process_in_orbit=False)
        for _ in range(5):
            pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        # With 1Gbps sensor, 30% duty cycle, data exceeds 800 Mbps X-band downlink
        self.assertGreater(pipeline.metrics.downlink_backlog_gb, 0.0,
                           "Raw downlink should accumulate backlog")

    def test_in_orbit_processing_reduces_backlog(self):
        """In-orbit processing reduces backlog compared to raw downlink."""
        raw = self._make_pipeline(process_in_orbit=False)
        processed = self._make_pipeline(process_in_orbit=True)

        for _ in range(5):
            raw.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
            processed.simulate_orbit(5700.0, 0.35, [(4800, 5400)])

        self.assertLess(processed.metrics.downlink_backlog_gb,
                        raw.metrics.downlink_backlog_gb,
                        "In-orbit processing should reduce backlog vs raw")

    def test_storage_overflow_when_capacity_exceeded(self):
        """Tiny storage overflows when sensor generates more data than fits."""
        pipeline = self._make_pipeline(process_in_orbit=False, storage_gb=0.01)
        pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        self.assertGreater(pipeline.metrics.storage_overflow_gb, 0.0,
                           "Tiny storage should overflow")

    def test_bandwidth_saved_pct_correct(self):
        """Bandwidth saved percentage reflects data not downlinked."""
        pipeline = self._make_pipeline(process_in_orbit=True,
                                       task="image_classification")
        for _ in range(10):
            pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        # With 100x reduction, savings should be substantial
        self.assertGreater(pipeline.metrics.bandwidth_saved_pct, 50.0)

    def test_different_processing_types_different_reductions(self):
        """Different processing tasks produce different bandwidth savings."""
        p_class = self._make_pipeline(task="image_classification")
        p_compress = self._make_pipeline(task="raw_to_compressed")
        for _ in range(5):
            p_class.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
            p_compress.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        # Classification (100x) saves more than compression (4x)
        self.assertGreater(p_class.metrics.bandwidth_saved_pct,
                           p_compress.metrics.bandwidth_saved_pct)

    def test_report_has_expected_keys(self):
        """Pipeline report dict has all expected fields."""
        pipeline = self._make_pipeline()
        pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        report = pipeline.report()
        expected_keys = [
            "orbits", "data_generated_gb", "data_processed_gb",
            "data_downlinked_gb", "bandwidth_saved_pct",
            "storage_high_water_gb", "storage_overflow_gb",
            "downlink_backlog_gb", "downlink_utilization_pct",
            "process_in_orbit",
        ]
        for key in expected_keys:
            self.assertIn(key, report)

    def test_data_generated_increases_each_orbit(self):
        """Total data generated increases with each orbit simulated."""
        pipeline = self._make_pipeline()
        pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        gen1 = pipeline.metrics.total_data_generated_gb
        pipeline.simulate_orbit(5700.0, 0.35, [(4800, 5400)])
        gen2 = pipeline.metrics.total_data_generated_gb
        self.assertGreater(gen2, gen1)


class TestProcessingResult(unittest.TestCase):
    """Tests for ProcessingResult properties."""

    def test_bandwidth_saved_zero_input(self):
        """Zero input returns 0% saved."""
        result = ProcessingResult(
            input_gb=0.0, output_gb=0.0,
            processing_time_s=0.0, power_watts=0.0,
            reduction_ratio=1.0,
        )
        self.assertAlmostEqual(result.bandwidth_saved_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
