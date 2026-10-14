"""
Unit test suite for physics.py using Python's standard unittest framework.

Covers all 7 validation criteria required by the specification:
1. Initial interior velocity and both boundary velocities.
2. Convergence toward the steady linear profile.
3. Maximum analytical-reference error below 1% of U at dimensionless times 0.01, 0.05, 0.2, 0.6.
4. Reduced error with 26, 51, and 101 nodes at fixed Fo = 0.4 and dimensionless time 0.05.
5. Identical velocity histories when density and viscosity change proportionally.
6. Approximately half the settling time when nu doubles, and four times when H doubles, within 2%.
7. Rejection of invalid inputs and excessive computational workloads.
"""

import unittest
import numpy as np

from physics import (
     analytical_couette,
     compute_numerical_settling_time,
     solve_couette_transient,
)


class TestMomentumDiffusionPhysics(unittest.TestCase):
    def setUp(self):
        self.rho = 1000.0
        self.mu = 0.1
        self.nu = self.mu / self.rho
        self.H = 0.005
        self.U = 0.02
        self.t_char = (self.H ** 2) / self.nu

    def test_1_initial_and_boundary_velocities(self):
        """Test initial interior stationary condition and stationary/moving boundary conditions."""
        times = np.array([0.0, 0.01 * self.t_char, 0.1 * self.t_char, 0.5 * self.t_char])
        y, vel = solve_couette_transient(
            self.rho, self.mu, self.H, self.U, times, n_nodes=51, fourier_number=0.4
        )
        self.assertEqual(y.shape, (51,))
        self.assertEqual(vel.shape, (len(times), 51))

        # At t = 0: lower plate and interior are 0, upper plate is U
        np.testing.assert_allclose(vel[0, :-1], 0.0, atol=1e-15)
        self.assertAlmostEqual(vel[0, -1], self.U, places=14)

        # At all times: lower plate (y=0) is 0, upper plate (y=H) is U
        np.testing.assert_allclose(vel[:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(vel[:, -1], self.U, atol=1e-15)

        # Analytical function at t = 0 matches the same boundary/interior contract
        vel_ref = analytical_couette(self.nu, self.H, self.U, y, times)
        np.testing.assert_allclose(vel_ref[0, :-1], 0.0, atol=1e-15)
        self.assertAlmostEqual(vel_ref[0, -1], self.U, places=14)

    def test_2_convergence_toward_linear_steady_profile(self):
        """Test that the numerical profile converges toward u_steady = U * y / H."""
        times = np.array([0.0, 0.05 * self.t_char, 0.2 * self.t_char, 0.6 * self.t_char])
        y, vel = solve_couette_transient(
            self.rho, self.mu, self.H, self.U, times, n_nodes=51, fourier_number=0.4
        )
        u_steady = self.U * (y / self.H)

        errors = [np.max(np.abs(vel[i] - u_steady)) / self.U for i in range(len(times))]
        # Error strictly decreases over time and is well below 1% by t = 0.6 * t_char
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertGreater(errors[2], errors[3])
        self.assertLess(errors[3], 0.01)

    def test_3_agreement_with_analytical_reference_at_positive_times(self):
        """Test max velocity error vs analytical series is < 1% of U at t* = 0.01, 0.05, 0.2, 0.6."""
        dimensionless_times = np.array([0.01, 0.05, 0.2, 0.6])
        times = dimensionless_times * self.t_char

        y, vel_num = solve_couette_transient(
            self.rho, self.mu, self.H, self.U, times, n_nodes=51, fourier_number=0.4
        )
        vel_ref = analytical_couette(self.nu, self.H, self.U, y, times, terms=200)

        for idx, t_star in enumerate(dimensionless_times):
            max_rel_error = np.max(np.abs(vel_num[idx] - vel_ref[idx])) / self.U
            self.assertLess(
                max_rel_error,
                0.01,
                msg=f"Max relative error {max_rel_error:.5f} exceeded 0.01 at t* = {t_star}",
            )

    def test_4_spatial_and_temporal_refinement_reduces_error(self):
        """Test reduced error with 26, 51, and 101 nodes at fixed Fo = 0.4 and t* = 0.05."""
        t_eval = np.array([0.05 * self.t_char])
        node_counts = [26, 51, 101]
        errors = []

        for n_pts in node_counts:
            y, vel_num = solve_couette_transient(
                self.rho,
                self.mu,
                self.H,
                self.U,
                t_eval,
                n_nodes=n_pts,
                fourier_number=0.4,
            )
            vel_ref = analytical_couette(self.nu, self.H, self.U, y, t_eval, terms=200)
            rel_err = float(np.max(np.abs(vel_num[0] - vel_ref[0])) / self.U)
            errors.append(rel_err)

        self.assertLess(
            errors[1],
            errors[0],
            msg=f"Error with 51 nodes ({errors[1]}) was not less than with 26 nodes ({errors[0]}).",
        )
        self.assertLess(
            errors[2],
            errors[1],
            msg=f"Error with 101 nodes ({errors[2]}) was not less than with 51 nodes ({errors[1]}).",
        )

    def test_5_identical_velocity_for_proportional_density_and_viscosity(self):
        """Test identical velocity evolution when rho and mu double together (equal nu)."""
        times = np.linspace(0.0, 0.6 * self.t_char, 61)
        y1, vel1 = solve_couette_transient(
            self.rho, self.mu, self.H, self.U, times, n_nodes=51, fourier_number=0.4
        )
        y2, vel2 = solve_couette_transient(
            2.0 * self.rho,
            2.0 * self.mu,
            self.H,
            self.U,
            times,
            n_nodes=51,
            fourier_number=0.4,
        )
        np.testing.assert_allclose(y1, y2, atol=1e-15)
        np.testing.assert_allclose(vel1, vel2, rtol=1e-13, atol=1e-15)

    def test_6_settling_time_scaling_with_nu_and_gap(self):
        """Test settling time halves when nu doubles and quadruples when H doubles (within 2%)."""
        t_base = compute_numerical_settling_time(
            self.rho, self.mu, self.H, self.U, n_nodes=51, fourier_number=0.4
        )
        # Double kinematic viscosity by doubling dynamic viscosity at fixed density
        t_double_nu = compute_numerical_settling_time(
            self.rho, 2.0 * self.mu, self.H, self.U, n_nodes=51, fourier_number=0.4
        )
        # Double plate gap H at fixed fluid properties
        t_double_H = compute_numerical_settling_time(
            self.rho, self.mu, 2.0 * self.H, self.U, n_nodes=51, fourier_number=0.4
        )

        ratio_nu = t_double_nu / t_base
        ratio_H = t_double_H / t_base

        self.assertLess(
            abs(ratio_nu - 0.5) / 0.5,
            0.02,
            msg=f"Expected settling time ratio ~0.5 when nu doubles, got {ratio_nu:.5f}",
        )
        self.assertLess(
            abs(ratio_H - 4.0) / 4.0,
            0.02,
            msg=f"Expected settling time ratio ~4.0 when H doubles, got {ratio_H:.5f}",
        )

    def test_7_invalid_inputs_and_excessive_workload_rejected(self):
        """Test that invalid physical inputs, invalid grids/Fo, and >25,000 steps raise ValueError."""
        valid_times = np.array([0.0, 0.05])

        # Nonpositive physical inputs
        with self.assertRaises(ValueError):
            solve_couette_transient(0.0, self.mu, self.H, self.U, valid_times)
        with self.assertRaises(ValueError):
            solve_couette_transient(-1000.0, self.mu, self.H, self.U, valid_times)
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, 0.0, self.H, self.U, valid_times)
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, -0.005, self.U, valid_times)
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, 0.0, valid_times)

        # Invalid n_nodes or fourier_number
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, valid_times, n_nodes=2)
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, valid_times, n_nodes=True)
        with self.assertRaises(ValueError):
            solve_couette_transient(
                self.rho, self.mu, self.H, self.U, valid_times, fourier_number=0.0
            )
        with self.assertRaises(ValueError):
            solve_couette_transient(
                self.rho, self.mu, self.H, self.U, valid_times, fourier_number=0.51
            )

        # Invalid times arrays
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, np.array([]))
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, np.array([-0.01, 0.1]))
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, np.array([0.1, 0.1]))
        with self.assertRaises(ValueError):
            solve_couette_transient(self.rho, self.mu, self.H, self.U, np.array([0.2, 0.1]))

        # Invalid analytical_couette inputs (y outside [0, H] or invalid terms)
        with self.assertRaises(ValueError):
            analytical_couette(
                self.nu, self.H, self.U, np.array([-0.001, 0.002]), valid_times
            )
        with self.assertRaises(ValueError):
            analytical_couette(
                self.nu, self.H, self.U, np.array([0.0, self.H * 1.01]), valid_times
            )
        with self.assertRaises(ValueError):
            analytical_couette(
                self.nu, self.H, self.U, np.array([0.0, self.H]), valid_times, terms=0
            )

        # Excessive workload (> 25,000 integration steps)
        huge_times = np.array([0.0, 20.0 * self.t_char])
        with self.assertRaises(ValueError):
            solve_couette_transient(
                self.rho, self.mu, self.H, self.U, huge_times, n_nodes=51, fourier_number=0.4
            )


if __name__ == "__main__":
    unittest.main()
