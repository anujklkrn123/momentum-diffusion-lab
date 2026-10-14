"""Physics checks; not a substitute for the common external evaluator."""

import unittest

import numpy as np

from physics import (
    _solve_couette_with_metadata,
    analytical_couette,
    solve_couette_transient,
)


class CouettePhysicsTests(unittest.TestCase):
    rho = 1000.0
    mu = 0.1
    H = 0.005
    U = 0.02
    nu = mu / rho
    t_char = H**2 / nu

    def solve(self, times, **kwargs):
        return solve_couette_transient(self.rho, self.mu, self.H, self.U, times, **kwargs)

    def test_initial_condition_boundaries_and_shapes(self):
        y, velocity = self.solve([0.0, 0.01, 0.1])
        self.assertEqual(y.shape, (51,))
        self.assertEqual(velocity.shape, (3, 51))
        self.assertEqual(y[0], 0.0)
        self.assertEqual(y[-1], self.H)
        np.testing.assert_array_equal(velocity[0, :-1], 0.0)
        np.testing.assert_array_equal(velocity[:, 0], 0.0)
        np.testing.assert_array_equal(velocity[:, -1], self.U)
        self.assertTrue(np.all(velocity >= 0))
        self.assertTrue(np.all(velocity <= self.U))

    def test_approaches_linear_steady_profile(self):
        y, velocity = self.solve(np.array([0.05, 0.2, 0.6]) * self.t_char)
        error = np.max(np.abs(velocity - self.U * y[None, :] / self.H), axis=1) / self.U
        self.assertTrue(np.all(np.diff(error) < 0))
        self.assertLess(error[-1], 0.01)

    def test_analytical_accuracy_at_prescribed_times(self):
        times = np.array([0.01, 0.05, 0.2, 0.6]) * self.t_char
        y, numerical = self.solve(times)
        reference = analytical_couette(self.nu, self.H, self.U, y, times)
        errors = np.max(np.abs(numerical - reference), axis=1) / self.U
        for dimensionless_time, error in zip([0.01, 0.05, 0.2, 0.6], errors):
            with self.subTest(dimensionless_time=dimensionless_time):
                self.assertLess(error, 0.01)

    def test_refining_grid_reduces_reference_error(self):
        errors = []
        times = [0.05 * self.t_char]
        for nodes in (26, 51, 101):
            y, numerical = self.solve(times, n_nodes=nodes, fourier_number=0.4)
            reference = analytical_couette(self.nu, self.H, self.U, y, times)
            errors.append(float(np.max(np.abs(numerical - reference)) / self.U))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_equal_kinematic_viscosity_same_history(self):
        times = np.linspace(0.0, 0.6 * self.t_char, 121)
        y, baseline = self.solve(times)
        other_y, other = solve_couette_transient(
            2 * self.rho, 2 * self.mu, self.H, self.U, times
        )
        np.testing.assert_array_equal(y, other_y)
        np.testing.assert_array_equal(baseline, other)

    def test_settling_scales_with_diffusivity_and_gap(self):
        def settling(mu, gap):
            t_char = gap**2 / (mu / self.rho)
            _, _, metadata = _solve_couette_with_metadata(
                self.rho, mu, gap, self.U, [0.0, 0.6 * t_char]
            )
            self.assertIsNotNone(metadata["settling_time"])
            return metadata["settling_time"]

        base = settling(self.mu, self.H)
        self.assertAlmostEqual(settling(2 * self.mu, self.H) / base, 0.5, delta=0.01)
        self.assertAlmostEqual(settling(self.mu, 2 * self.H) / base, 4.0, delta=0.08)

    def test_measured_settling_uses_internal_steps(self):
        _, _, metadata = _solve_couette_with_metadata(
            self.rho, self.mu, self.H, self.U, [0.0, 0.6 * self.t_char]
        )
        settled = metadata["settling_time"]
        self.assertIsNotNone(settled)
        self.assertLess(settled, 0.6 * self.t_char)
        self.assertAlmostEqual(settled / self.t_char, 0.421, delta=0.002)
        # The reported first crossing is bracketed by a real integration
        # step. It cannot simply be the final requested display frame.
        before = settled - metadata["max_dt"]
        y, profiles = self.solve([before, settled])
        errors = np.max(np.abs(profiles - self.U * y[None, :] / self.H), axis=1) / self.U
        self.assertGreaterEqual(errors[0], 0.01)
        self.assertLess(errors[1], 0.01)

    def test_no_settling_before_requested_duration(self):
        _, _, metadata = _solve_couette_with_metadata(
            self.rho, self.mu, self.H, self.U, [0.0, 0.05 * self.t_char]
        )
        self.assertIsNone(metadata["settling_time"])

    def test_exact_output_step_includes_partial_step(self):
        # A three-node grid has one interior value. For a partial step with
        # Fo=0.1 the FTCS update is exactly 0.1*U, independently of plotting.
        time = 0.1 * (self.H / 2) ** 2 / self.nu
        _, velocity = self.solve([time], n_nodes=3)
        self.assertAlmostEqual(velocity[0, 1], 0.1 * self.U, places=14)

    def test_stability_endpoint_is_supported(self):
        y, velocity = self.solve([0.0, 0.1 * self.t_char], fourier_number=0.5)
        self.assertEqual(y.size, 51)
        self.assertTrue(np.all(velocity >= 0))
        self.assertTrue(np.all(velocity <= self.U))

    def test_invalid_physical_inputs(self):
        baseline = dict(rho=self.rho, mu=self.mu, H=self.H, U=self.U, times=[0.0])
        for name in ("rho", "mu", "H", "U"):
            for invalid in (0, -1, np.nan, np.inf, True, "bad"):
                with self.subTest(name=name, invalid=invalid):
                    arguments = dict(baseline, **{name: invalid})
                    with self.assertRaises(ValueError):
                        solve_couette_transient(**arguments)

    def test_invalid_time_arrays(self):
        for invalid in ([], [[0, 1]], [0, 0], [1, 0], [-1], [np.nan], [np.inf], "bad"):
            with self.subTest(times=invalid):
                with self.assertRaises(ValueError):
                    self.solve(invalid)

    def test_invalid_grid_or_fourier_number(self):
        for invalid in (1, 2, 51.0, True, "51"):
            with self.subTest(n_nodes=invalid):
                with self.assertRaises(ValueError):
                    self.solve([0], n_nodes=invalid)
        for invalid in (0, -0.1, 0.51, np.inf, np.nan):
            with self.subTest(fourier_number=invalid):
                with self.assertRaises(ValueError):
                    self.solve([0], fourier_number=invalid)

    def test_excessive_workload_fails_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "25,000"):
            self.solve([0.0, 5.0 * self.t_char])
        # Tiny intervals still require one update apiece, even though their
        # total duration is short. The cap therefore includes output steps.
        with self.assertRaisesRegex(ValueError, "25,000"):
            self.solve(np.linspace(0, 0.001 * self.t_char, 25_002))

    def test_derived_scale_overflow_underflow(self):
        cases = [
            dict(rho=1e-300, mu=1e300),
            dict(rho=1e300, mu=1e-300),
            dict(H=1e300),
            dict(H=1e-300),
        ]
        baseline = dict(rho=self.rho, mu=self.mu, H=self.H, U=self.U, times=[0])
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    solve_couette_transient(**dict(baseline, **case))

    def test_analytical_initial_boundaries_and_invalid_inputs(self):
        y = np.array([0, self.H / 2, self.H])
        reference = analytical_couette(self.nu, self.H, self.U, y, [0, 0.1])
        np.testing.assert_array_equal(reference[0], [0, 0, self.U])
        np.testing.assert_array_equal(reference[:, 0], 0)
        np.testing.assert_array_equal(reference[:, -1], self.U)
        for bad_y in ([], [[0]], [-1e-6], [self.H + 1e-6], [np.nan]):
            with self.subTest(y=bad_y):
                with self.assertRaises(ValueError):
                    analytical_couette(self.nu, self.H, self.U, bad_y, [0.1])
        for bad_terms in (0, 2.5, True):
            with self.subTest(terms=bad_terms):
                with self.assertRaises(ValueError):
                    analytical_couette(self.nu, self.H, self.U, y, [0.1], terms=bad_terms)


if __name__ == "__main__":
    unittest.main()
