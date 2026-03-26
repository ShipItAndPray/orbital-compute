"""Tests for debris, propulsion, and federated learning modules."""
import unittest
from orbital_compute.debris import (
    DebrisEnvironment, assess_constellation, kessler_critical_size,
)
from orbital_compute.propulsion import (
    compute_drag, station_keeping_delta_v, plan_deorbit,
    DragParameters, HALL_THRUSTER,
)
from orbital_compute.federated import (
    FederatedTrainingJob, weather_prediction_job, object_detection_job,
)


class TestDebris(unittest.TestCase):
    def test_assess_constellation(self):
        r = assess_constellation(n_satellites=10, altitude_km=550)
        self.assertGreater(r.collision_prob_per_sat_per_year, 0)

    def test_more_sats_higher_risk(self):
        r10 = assess_constellation(n_satellites=10, altitude_km=550)
        r100 = assess_constellation(n_satellites=100, altitude_km=550)
        self.assertGreater(r100.fleet_collision_prob_per_year,
                           r10.fleet_collision_prob_per_year)

    def test_kessler_critical_positive(self):
        self.assertGreater(kessler_critical_size(550), 0)


class TestPropulsion(unittest.TestCase):
    def test_drag_positive(self):
        r = compute_drag(550, DragParameters())
        self.assertGreater(r.drag_force_n, 0)

    def test_station_keeping_dv_positive(self):
        dv = station_keeping_delta_v(550)
        self.assertGreater(dv, 0)

    def test_lower_alt_more_drag(self):
        p = DragParameters()
        r400 = compute_drag(400, p)
        r800 = compute_drag(800, p)
        self.assertGreater(r400.drag_force_n, r800.drag_force_n)

    def test_deorbit_plan(self):
        plan = plan_deorbit(550, HALL_THRUSTER)
        self.assertGreater(plan.delta_v_ms, 0)


class TestFederated(unittest.TestCase):
    def test_job_gradient_size(self):
        job = FederatedTrainingJob(model_size_mb=500, training_data_mb_per_sat=1000,
                                    epochs_per_round=3, communication_rounds=10)
        self.assertGreater(job.gradient_size_mb, 0)
        self.assertLess(job.gradient_size_mb, job.model_size_mb)

    def test_weather_preset(self):
        job = weather_prediction_job()
        self.assertGreater(job.model_size_mb, 0)

    def test_object_detection_preset(self):
        job = object_detection_job()
        self.assertGreater(job.communication_rounds, 0)


class TestRedNet(unittest.TestCase):
    def test_rednet_speedup(self):
        from orbital_compute.radiation import RadiationModel, RecoveryStrategy, RedNetConfig
        r = RadiationModel(strategy=RecoveryStrategy.REDNET, rednet_config=RedNetConfig())
        self.assertLess(r.overhead_factor(), 1.0)

    def test_rednet_protection(self):
        from orbital_compute.radiation import RedNetConfig
        c = RedNetConfig()
        self.assertGreater(c.effective_protection, 0.99)

    def test_rednet_recovers_upsets(self):
        from orbital_compute.radiation import RadiationModel, RecoveryStrategy, RedNetConfig
        r = RadiationModel(strategy=RecoveryStrategy.REDNET, rednet_config=RedNetConfig(), seed=42)
        job = type('Job', (), {'failed': False})()
        result = r.handle_upset(job, True)
        self.assertEqual(result, "recovered")


if __name__ == "__main__":
    unittest.main()
