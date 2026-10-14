"""Startup plane Couette flow, independent of the teaching interface.

The solver uses FTCS on dimensionless velocity u/U and time nu*t/H**2.
This is the same physical scheme, but avoids unnecessary dependence of
roundoff on the chosen SI scales. The two public comparison interfaces are
listed in ``__all__``; the app uses an internal metadata wrapper to read the
first settling-threshold crossing without rerunning the calculation.
"""

from numbers import Integral

import numpy as np

__all__ = ["solve_couette_transient", "analytical_couette"]

_MAX_STEPS = 25_000
_SETTLING_TOLERANCE = 0.01


def _positive_scalar(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return result


def _times_array(times):
    try:
        result = np.asarray(times, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times must be a one-dimensional numeric array.") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("times must contain finite, nonnegative values.")
    if np.any(np.diff(result) <= 0):
        raise ValueError("times must be strictly increasing.")
    return result


def _physical_scales(nu, H):
    # Explicitly reject underflow/overflow rather than passing a zero or
    # infinite diffusivity/time scale into an apparently successful solver.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        h_squared = float(np.float64(H) * np.float64(H))
        characteristic_time = float(np.float64(h_squared) / np.float64(nu))
    if not np.isfinite(h_squared) or h_squared <= 0:
        raise ValueError("H squared must be finite and positive at floating-point precision.")
    if not np.isfinite(characteristic_time) or characteristic_time <= 0:
        raise ValueError("H squared / nu must be finite and positive.")
    return characteristic_time


def _normalised_times(times, characteristic_time):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = times / characteristic_time
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Requested times overflow the diffusion time scale.")
    if np.any((times > 0) & (scaled <= 0)) or np.any(np.diff(scaled) <= 0):
        raise ValueError("Requested times cannot be resolved at floating-point precision.")
    return scaled


def _step_plan(scaled_times, max_scaled_step):
    """Plan all output-aligned steps before allocating or evolving a grid."""
    plan = []
    previous = 0.0
    total = 0
    for target in scaled_times:
        interval = float(target - previous)
        if interval == 0:
            count = 0
        else:
            ratio = interval / max_scaled_step
            if not np.isfinite(ratio) or ratio > _MAX_STEPS:
                raise ValueError(f"Requested trajectory exceeds the {_MAX_STEPS:,} integration-step limit.")
            count = max(1, int(np.ceil(ratio)))
        total += count
        if total > _MAX_STEPS:
            raise ValueError(f"Requested trajectory exceeds the {_MAX_STEPS:,} integration-step limit.")
        plan.append((float(target), count))
        previous = float(target)
    return plan


def _solve_couette_with_metadata(
    rho, mu, H, U, times, n_nodes=51, fourier_number=0.4
):
    """Return ``(y, velocity, metadata)`` for the app's cached trajectory.

    ``settling_time`` is the physical time at the first *integration-step*
    crossing below max(abs(u-u_steady))/U = 0.01, or ``None`` if the requested
    trajectory never settles. It is a grid measurement, not interpolation
    or the analytical late-time estimate. Other metadata records the actual
    number of updates and the maximum allowed physical step.
    """
    rho = _positive_scalar("rho", rho)
    mu = _positive_scalar("mu", mu)
    H = _positive_scalar("H", H)
    U = _positive_scalar("U", U)
    n_nodes = _integer("n_nodes", n_nodes, 3)
    fourier_number = _positive_scalar("fourier_number", fourier_number)
    if fourier_number > 0.5:
        raise ValueError("fourier_number must satisfy 0 < Fo <= 0.5 for explicit stability.")
    times = _times_array(times)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nu = float(np.float64(mu) / np.float64(rho))
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("mu / rho must give a finite positive kinematic viscosity.")
    characteristic_time = _physical_scales(nu, H)
    try:
        inverse_intervals = 1.0 / (n_nodes - 1)
        max_scaled_step = fourier_number * inverse_intervals**2
        dy = H * inverse_intervals
        max_dt = max_scaled_step * characteristic_time
    except OverflowError as exc:
        raise ValueError("n_nodes is too large to resolve a finite grid spacing.") from exc
    if any(not np.isfinite(x) or x <= 0 for x in (dy, max_dt, max_scaled_step)):
        raise ValueError("Grid spacing and integration step must be finite and positive.")
    scaled_times = _normalised_times(times, characteristic_time)
    plan = _step_plan(scaled_times, max_scaled_step)

    position = np.linspace(0.0, 1.0, n_nodes)
    y = position * H
    y[-1] = H
    profile = np.zeros(n_nodes, dtype=float)
    profile[-1] = 1.0
    velocities = np.empty((times.size, n_nodes), dtype=float)
    current = 0.0
    integration_steps = 0
    settling_time = None
    for output_index, (target, count) in enumerate(plan):
        for step_index in range(count):
            # The last step lands on the requested time. A tiny last step
            # caused by roundoff is still counted and never violates Fo.
            step = min(max_scaled_step, target - current)
            if step <= 0:
                # A planned extra step can be swallowed by float rounding.
                # No update happened, so it does not count as integration.
                break
            actual_fo = min(fourier_number, step / inverse_intervals**2)
            profile[1:-1] = profile[1:-1] + actual_fo * (
                profile[2:] - 2.0 * profile[1:-1] + profile[:-2]
            )
            current = target if step_index == count - 1 else min(target, current + step)
            integration_steps += 1
            if settling_time is None and np.max(np.abs(profile - position)) < _SETTLING_TOLERANCE:
                settling_time = current * characteristic_time
        current = target
        velocities[output_index] = profile * U
        velocities[output_index, 0] = 0.0
        velocities[output_index, -1] = U

    metadata = {
        "settling_time": settling_time,
        "integration_steps": integration_steps,
        "max_dt": max_dt,
        "dy": dy,
        "fourier_number": fourier_number,
        "characteristic_time": characteristic_time,
    }
    return y, velocities, metadata


def solve_couette_transient(
    rho, mu, H, U, times, n_nodes=51, fourier_number=0.4
):
    """Return ``(y, velocity)`` using stable explicit thin-layer diffusion.

    Times are finite, nonnegative, strictly increasing SI seconds; velocity
    has shape ``(len(times), n_nodes)``. At t=0 the interior is stationary
    and the upper boundary is assigned its imposed post-start speed U.
    Each call is limited to 25,000 integration steps, including shortened
    steps used to reach requested output times exactly.
    """
    y, velocity, _ = _solve_couette_with_metadata(
        rho, mu, H, U, times, n_nodes, fourier_number
    )
    return y, velocity


def analytical_couette(nu, H, U, y, times, terms=200):
    """Evaluate the truncated analytical series, with an explicit t=0 case.

    Positive times use the sine series in the specification. Very early
    times require more modes than a fixed 200-term truncation may resolve;
    validation therefore uses nu*t/H**2 >= 0.01. This evaluator is generated
    alongside this solver, so it is not an independent external benchmark.
    """
    nu = _positive_scalar("nu", nu)
    H = _positive_scalar("H", H)
    U = _positive_scalar("U", U)
    terms = _integer("terms", terms, 1)
    times = _times_array(times)
    try:
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("y must be a one-dimensional numeric array.") from exc
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y > H):
        raise ValueError("y must contain finite positions within [0, H].")
    characteristic_time = _physical_scales(nu, H)
    scaled_times = _normalised_times(times, characteristic_time)
    position = y / H
    result = np.zeros((times.size, y.size), dtype=float)
    modes = np.arange(1, terms + 1, dtype=float)
    coefficients = np.where((modes.astype(int) % 2) == 1, 1.0, -1.0) / modes
    sine_terms = np.sin(np.pi * modes[:, None] * position[None, :])
    for index, scaled_time in enumerate(scaled_times):
        if scaled_time > 0:
            with np.errstate(over="ignore", under="ignore"):
                weights = coefficients * np.exp(-(np.pi * modes) ** 2 * scaled_time)
            result[index] = U * (position - (2.0 / np.pi) * (weights @ sine_terms))
    result[:, y == 0.0] = 0.0
    result[:, y == H] = U
    return result
