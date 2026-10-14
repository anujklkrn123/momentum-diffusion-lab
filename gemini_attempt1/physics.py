"""
Core numerical solver and analytical reference evaluator for startup plane Couette flow.

This module is strictly independent of Streamlit so that automated validation
suites can import and test both functions directly.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

MAX_INTEGRATION_STEPS = 25_000


def _validate_positive_scalar(value: float, name: str) -> float:
    """Validate that a physical parameter is a finite positive real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError(f"{name} must be a finite positive real number.")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be strictly positive and finite (got {value}).")
    return val


def _validate_times(times: np.ndarray) -> np.ndarray:
    """Validate that times is a nonempty 1D array of finite, nonnegative, strictly increasing values."""
    if isinstance(times, (int, float, bool, np.number, np.bool_)):
        raise ValueError("times must be a 1D array of time values, not a scalar.")
    try:
        t_arr = np.asarray(times, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be convertible to a 1D float array.") from exc

    if t_arr.ndim != 1 or t_arr.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("times must contain only finite values.")
    if np.any(t_arr < 0.0):
        raise ValueError("times must contain only nonnegative values (>= 0).")
    if t_arr.size > 1 and np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    return t_arr


def solve_couette_transient(
    rho, mu, H, U, times, n_nodes=51, fourier_number=0.4
):
    """Return (y, velocity)."""
    rho_val = _validate_positive_scalar(rho, "rho")
    mu_val = _validate_positive_scalar(mu, "mu")
    H_val = _validate_positive_scalar(H, "H")
    U_val = _validate_positive_scalar(U, "U")
    t_arr = _validate_times(times)

    if (
        isinstance(n_nodes, (bool, np.bool_))
        or not isinstance(n_nodes, (int, np.integer))
        or int(n_nodes) < 3
    ):
        raise ValueError("n_nodes must be an integer greater than or equal to 3.")
    n_pts = int(n_nodes)

    if isinstance(fourier_number, (bool, np.bool_)) or not isinstance(
        fourier_number, (int, float, np.number)
    ):
        raise ValueError("fourier_number must be a real number in (0, 0.5].")
    fo_val = float(fourier_number)
    if not math.isfinite(fo_val) or fo_val <= 0.0 or fo_val > 0.5:
        raise ValueError(
            f"fourier_number must satisfy 0 < Fo <= 0.5 for explicit FTCS stability (got {fourier_number})."
        )

    nu = mu_val / rho_val
    y = np.linspace(0.0, H_val, n_pts, dtype=float)
    dy = H_val / float(n_pts - 1)
    dt_base = fo_val * (dy * dy) / nu

    # Estimate minimum required steps before running to fail fast if clearly excessive
    t_final = float(t_arr[-1])
    if t_final > 0.0:
        min_estimated_steps = math.ceil(t_final / dt_base)
        if min_estimated_steps > MAX_INTEGRATION_STEPS:
            raise ValueError(
                f"Requested simulation requires at least {min_estimated_steps} integration steps, "
                f"exceeding the cap of {MAX_INTEGRATION_STEPS} steps."
            )

    # Initial condition at t = 0: stationary interior and lower plate, moving upper plate
    u_curr = np.zeros(n_pts, dtype=float)
    u_curr[-1] = U_val

    velocity = np.zeros((t_arr.size, n_pts), dtype=float)
    t_curr = 0.0
    step_count = 0

    for idx, t_target in enumerate(t_arr):
        target = float(t_target)
        while t_curr < target:
            rem = target - t_curr
            # If the remaining time is within or at dt_base, land on target exactly
            if rem <= dt_base:
                dt_step = rem
                t_next = target
            # Avoid creating a microscopic floating-point remainder step when rem is just above dt_base
            elif rem <= dt_base * (1.0 + 1e-9):
                dt_step = 0.5 * rem
                t_next = t_curr + dt_step
            else:
                dt_step = dt_base
                t_next = t_curr + dt_step

            step_count += 1
            if step_count > MAX_INTEGRATION_STEPS:
                raise ValueError(
                    f"Solver exceeded the maximum cap of {MAX_INTEGRATION_STEPS} integration steps."
                )

            fo_step = nu * dt_step / (dy * dy)
            if fo_step > 0.5 + 1e-12:
                raise ValueError(
                    f"Stability violation detected: step Fourier number {fo_step:.6f} > 0.5."
                )

            u_curr[1:-1] = u_curr[1:-1] + fo_step * (
                u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]
            )
            u_curr[0] = 0.0
            u_curr[-1] = U_val
            t_curr = t_next

        velocity[idx, :] = u_curr

    return y, velocity


def analytical_couette(
    nu, H, U, y, times, terms=200
):
    """Return reference velocity array."""
    nu_val = _validate_positive_scalar(nu, "nu")
    H_val = _validate_positive_scalar(H, "H")
    U_val = _validate_positive_scalar(U, "U")
    t_arr = _validate_times(times)

    if (
        isinstance(terms, (bool, np.bool_))
        or not isinstance(terms, (int, np.integer))
        or int(terms) < 1
    ):
        raise ValueError("terms must be a positive integer (>= 1).")
    n_terms = int(terms)

    if isinstance(y, (int, float, bool, np.number, np.bool_)):
        raise ValueError("y must be a 1D array of positions in [0, H], not a scalar.")
    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must be convertible to a 1D float array.") from exc

    if y_arr.ndim != 1 or y_arr.size == 0:
        raise ValueError("y must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values.")
    if np.any(y_arr < 0.0) or np.any(y_arr > H_val):
        raise ValueError("All positions in y must lie within the closed interval [0, H].")

    u_steady = U_val * (y_arr / H_val)
    result = np.zeros((t_arr.size, y_arr.size), dtype=float)

    n = np.arange(1, n_terms + 1, dtype=float)
    signs = np.where((n.astype(int) % 2) == 1, 1.0, -1.0)
    coeffs = signs / n  # shape: (n_terms,)
    sin_matrix = np.sin(np.outer(n * math.pi / H_val, y_arr))  # shape: (n_terms, len(y))

    at_lower = np.isclose(y_arr, 0.0, rtol=0.0, atol=1e-15 * H_val)
    at_upper = np.isclose(y_arr, H_val, rtol=0.0, atol=1e-15 * H_val)

    for idx, t_val in enumerate(t_arr):
        t_float = float(t_val)
        if t_float == 0.0:
            row = np.zeros_like(y_arr)
            row[at_upper] = U_val
            result[idx, :] = row
        else:
            decay = np.exp(-(n * n) * (math.pi * math.pi) * nu_val * t_float / (H_val * H_val))
            transient_sum = np.dot(coeffs * decay, sin_matrix)
            row = u_steady - (2.0 * U_val / math.pi) * transient_sum
            row[at_lower] = 0.0
            row[at_upper] = U_val
            result[idx, :] = row

    return result


def compute_numerical_settling_time(
    rho: float,
    mu: float,
    H: float,
    U: float,
    n_nodes: int = 51,
    fourier_number: float = 0.4,
    threshold: float = 0.01,
) -> float:
    """
    Compute the numerical settling time directly from individual FTCS integration steps.

    Settled is defined as max_y |u(y, t) - u_steady(y)| / U < threshold.
    """
    rho_val = _validate_positive_scalar(rho, "rho")
    mu_val = _validate_positive_scalar(mu, "mu")
    H_val = _validate_positive_scalar(H, "H")
    U_val = _validate_positive_scalar(U, "U")
    threshold_val = _validate_positive_scalar(threshold, "threshold")

    if (
        isinstance(n_nodes, (bool, np.bool_))
        or not isinstance(n_nodes, (int, np.integer))
        or int(n_nodes) < 3
    ):
        raise ValueError("n_nodes must be an integer >= 3.")
    n_pts = int(n_nodes)

    if (
        isinstance(fourier_number, (bool, np.bool_))
        or not isinstance(fourier_number, (int, float, np.number))
        or float(fourier_number) <= 0.0
        or float(fourier_number) > 0.5
    ):
        raise ValueError("fourier_number must satisfy 0 < Fo <= 0.5.")
    fo_val = float(fourier_number)

    nu = mu_val / rho_val
    y = np.linspace(0.0, H_val, n_pts, dtype=float)
    dy = H_val / float(n_pts - 1)
    dt = fo_val * (dy * dy) / nu

    u_curr = np.zeros(n_pts, dtype=float)
    u_curr[-1] = U_val
    u_steady = U_val * (y / H_val)

    for step in range(1, MAX_INTEGRATION_STEPS + 1):
        u_curr[1:-1] = u_curr[1:-1] + fo_val * (
            u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]
        )
        u_curr[0] = 0.0
        u_curr[-1] = U_val
        max_rel_err = float(np.max(np.abs(u_curr - u_steady)) / U_val)
        if max_rel_err < threshold_val:
            return step * dt

    raise ValueError(
        f"Flow did not settle below {threshold_val * 100:.2f}% within {MAX_INTEGRATION_STEPS} integration steps."
    )


def simulate_display_trajectory(
    rho: float,
    mu: float,
    H: float,
    U: float,
    duration: float | None = None,
    n_frames: int = 121,
    n_nodes: int = 51,
    fourier_number: float = 0.4,
) -> Dict[str, object]:
    """
    Precompute display frames and integration-step diagnostics for interactive playback.

    Returns a dictionary with:
    - y: spatial grid (n_nodes,)
    - times: display time array (n_frames,)
    - velocity: velocity profiles (n_frames, n_nodes)
    - u_steady: steady linear profile (n_nodes,)
    - nu: kinematic viscosity (m^2/s)
    - t_char: characteristic diffusion time H^2 / nu (s)
    - t_settle_est: analytical late-time 1% settling estimate ~0.421 * H^2 / nu (s)
    - t_settle_num: numerical settling time measured directly from integration steps (s)
    - dt_frame: physical time increment between consecutive display frames (s)
    """
    rho_val = _validate_positive_scalar(rho, "rho")
    mu_val = _validate_positive_scalar(mu, "mu")
    H_val = _validate_positive_scalar(H, "H")
    U_val = _validate_positive_scalar(U, "U")

    nu = mu_val / rho_val
    t_char = (H_val * H_val) / nu
    t_settle_est = 0.421 * t_char

    sim_duration = 0.6 * t_char if duration is None else float(duration)
    times = np.linspace(0.0, sim_duration, int(n_frames))
    y, velocity = solve_couette_transient(
        rho=rho_val,
        mu=mu_val,
        H=H_val,
        U=U_val,
        times=times,
        n_nodes=n_nodes,
        fourier_number=fourier_number,
    )
    t_settle_num = compute_numerical_settling_time(
        rho=rho_val,
        mu=mu_val,
        H=H_val,
        U=U_val,
        n_nodes=n_nodes,
        fourier_number=fourier_number,
        threshold=0.01,
    )
    u_steady = U_val * (y / H_val)
    dt_frame = float(times[1] - times[0]) if times.size > 1 else 0.0

    return {
        "y": y,
        "times": times,
        "velocity": velocity,
        "u_steady": u_steady,
        "nu": nu,
        "t_char": t_char,
        "t_settle_est": t_settle_est,
        "t_settle_num": t_settle_num,
        "dt_frame": dt_frame,
    }
