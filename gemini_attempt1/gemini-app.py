"""
Momentum Diffusion Lab — Interactive Streamlit Teaching Website.

Addresses: "How does motion spread through a fluid, and why do we need two definitions of viscosity?"
Built for beginning Chemical Engineering students familiar with shear stress and velocity gradients.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from physics import simulate_display_trajectory

st.set_page_config(
    page_title="Momentum Diffusion Lab",
    page_icon="🌊",
    layout="wide",
)

CONTENT_DIR = Path(__file__).parent / "content"

BASELINE_PARAMS = {
    "rho": 1000.0,
    "mu": 0.1,
    "H": 0.005,
    "U": 0.02,
}

GUIDED_EXPERIMENTS = {
    "exp1": {
        "title": "Experiment 1: Double Density (ρ × 2), Other Inputs Fixed",
        "params": {"rho": 2000.0, "mu": 0.1, "H": 0.005, "U": 0.02},
        "question": (
            "If we double the fluid density (from 1000 to 2000 kg/m³) while keeping dynamic "
            "viscosity (μ = 0.1 Pa·s), plate gap (H = 0.005 m), and plate speed (U = 0.02 m/s) fixed, "
            "how will motion spread across the gap compared to the baseline fluid?"
        ),
        "options": [
            "Select your prediction...",
            "Motion spreads faster (settles in about half the time)",
            "Motion spreads slower (takes about twice as long to settle)",
            "Velocity evolution is unchanged (settles in the same time)",
        ],
        "correct_option": "Motion spreads slower (takes about twice as long to settle)",
        "explanation": (
            "**Why motion spreads slower:** Doubling density $\\rho$ gives each thin fluid layer twice as "
            "much mass (inertia) per unit volume, while fixed dynamic viscosity $\\mu$ provides the same "
            "shear force for a given velocity gradient. By Newton's second law ($a = F/m$), a heavier layer "
            "accelerates more slowly under the same shear force imbalance. Quantitatively, kinematic viscosity "
            "$\\nu = \\mu / \\rho$ is cut in half (from $1.0 \\times 10^{-4}$ to $0.5 \\times 10^{-4}\\ \\mathrm{m^2/s}$), "
            "so the characteristic diffusion time $t_{\\mathrm{char}} = H^2/\\nu$ and settling time **double**."
        ),
    },
    "exp2": {
        "title": "Experiment 2: Double Dynamic Viscosity (μ × 2), Other Inputs Fixed",
        "params": {"rho": 1000.0, "mu": 0.2, "H": 0.005, "U": 0.02},
        "question": (
            "If we double the dynamic viscosity (from 0.1 to 0.2 Pa·s) while keeping density "
            "(ρ = 1000 kg/m³), plate gap (H = 0.005 m), and plate speed (U = 0.02 m/s) fixed, "
            "how will motion spread across the gap compared to the baseline fluid?"
        ),
        "options": [
            "Select your prediction...",
            "Motion spreads faster (settles in about half the time)",
            "Motion spreads slower (takes about twice as long to settle)",
            "Velocity evolution is unchanged (settles in the same time)",
        ],
        "correct_option": "Motion spreads faster (settles in about half the time)",
        "explanation": (
            "**Why motion spreads faster:** Beginners often associate higher viscosity with 'slower flow,' "
            "which is true when a fixed pressure drop pushes fluid through a pipe. Here, however, the upper plate's "
            "speed $U$ is fixed and viscosity is the *coupling mechanism* that transmits horizontal momentum from layer "
            "to layer. Doubling $\\mu$ at fixed $\\rho$ doubles the shear traction between adjacent layers for the same "
            "velocity difference, accelerating interior layers twice as fast. Kinematic viscosity $\\nu = \\mu / \\rho$ "
            "**doubles**, cutting the diffusion time $t_{\\mathrm{char}} = H^2/\\nu$ in **half**."
        ),
    },
    "exp3": {
        "title": "Experiment 3: Double Both Density and Dynamic Viscosity (ρ × 2 and μ × 2)",
        "params": {"rho": 2000.0, "mu": 0.2, "H": 0.005, "U": 0.02},
        "question": (
            "If we double BOTH density (1000 → 2000 kg/m³) and dynamic viscosity (0.1 → 0.2 Pa·s) "
            "simultaneously while keeping H and U fixed, how will the velocity profile u(y, t) evolve?"
        ),
        "options": [
            "Select your prediction...",
            "Motion spreads faster because viscosity is higher",
            "Motion spreads slower because the fluid is heavier",
            "Velocity evolution is unchanged (identical u(y, t) at every instant)",
        ],
        "correct_option": "Velocity evolution is unchanged (identical u(y, t) at every instant)",
        "explanation": (
            "**Why velocity evolution is unchanged — and why we still need both $\\mu$ and $\\rho$:** "
            "When $\\mu$ and $\\rho$ both double, the shear force transmitted between layers doubles *and* the "
            "mass of each layer doubles. The two effects cancel in $a = F/m$, leaving kinematic viscosity "
            "$\\nu = \\mu / \\rho = 1.0 \\times 10^{-4}\\ \\mathrm{m^2/s}$ unchanged and producing the **exact same "
            "velocity evolution $u(y, t)$**. However, because $\\tau = \\mu (\\partial u/\\partial y)$, thefluid with "
            "twice the dynamic viscosity $\\mu$ requires **twice the shear stress** at the upper plate to drive "
            "that identical velocity gradient."
        ),
    },
    "exp4": {
        "title": "Experiment 4: Double Plate Gap (H × 2), Other Inputs Fixed",
        "params": {"rho": 1000.0, "mu": 0.1, "H": 0.010, "U": 0.02},
        "question": (
            "If we double the gap between the plates (from H = 0.005 m to H = 0.010 m) while keeping "
            "density, dynamic viscosity, and plate speed fixed, how does the characteristic diffusion time change?"
        ),
        "options": [
            "Select your prediction...",
            "Characteristic time doubles (2× longer)",
            "Characteristic time becomes four times longer (4× longer)",
            "Characteristic time is unchanged",
        ],
        "correct_option": "Characteristic time becomes four times longer (4× longer)",
        "explanation": (
            "**Why doubling the gap quadruples the diffusion time:** Momentum spreads by diffusion rather than "
            "at a constant wave speed. Doubling the gap $H$ both doubles the distance momentum must travel *and* "
            "halves the velocity gradient $\\partial u/\\partial y$ (and thus the spatial curvature $\\partial^2 u/\\partial y^2$) "
            "driving acceleration. Together, $t_{\\mathrm{char}} = H^2 / \\nu$ scales with the **square of the gap**, "
            "making the characteristic diffusion time and settling time **four times longer**."
        ),
    },
}


@st.cache_data(show_spinner=False)
def cached_trajectory(
    rho: float,
    mu: float,
    H: float,
    U: float,
    duration: float | None = None,
    n_frames: int = 121,
) -> Dict[str, object]:
    """Cached wrapper around physics.simulate_display_trajectory."""
    return simulate_display_trajectory(
        rho=rho,
        mu=mu,
        H=H,
        U=U,
        duration=duration,
        n_frames=n_frames,
        n_nodes=51,
        fourier_number=0.4,
    )


def load_json_content(filename: str) -> dict:
    path = CONTENT_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_text_content(filename: str) -> str:
    path = CONTENT_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "Pending"


def build_fluid_figure(
    title: str,
    y: np.ndarray,
    u_curr: np.ndarray,
    u_steady: np.ndarray,
    H: float,
    H_axis_max: float,
    U_axis_max: float,
    line_color: str = "#1f77b4",
) -> go.Figure:
    """
    Create a two-panel Plotly figure for one fluid:
    - Left panel: horizontal velocity arrows at fixed vertical fluid-layer positions.
    - Right panel: velocity profile u(y, t) with dashed steady-state reference u_steady(y) = U y / H.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Fixed Fluid Layers (Velocity Arrows)",
            "Velocity Profile u(y, t)",
        ),
    )

    y_mm = y * 1000.0
    H_mm = H * 1000.0
    H_axis_mm = H_axis_max * 1000.0
    u_mm_s = u_curr * 1000.0
    u_steady_mm_s = u_steady * 1000.0
    U_axis_mm_s = U_axis_max * 1000.0

    # Shaded inactive region if this fluid's gap H is smaller than the shared vertical axis max
    if H_mm < H_axis_mm - 1e-6:
        for col_idx in (1, 2):
            fig.add_hrect(
                y0=H_mm,
                y1=H_axis_mm,
                fillcolor="rgba(160, 160, 160, 0.18)",
                line_width=0,
                annotation_text="Above Upper Plate" if col_idx == 2 else "",
                annotation_position="top right",
                row=1,
                col=col_idx,
            )

    # Plate boundary lines on both subplots
    for col_idx in (1, 2):
        fig.add_hline(
            y=0.0,
            line_dash="solid",
            line_color="#333333",
            line_width=3,
            row=1,
            col=col_idx,
        )
        fig.add_hline(
            y=H_mm,
            line_dash="solid",
            line_color=line_color,
            line_width=3,
            row=1,
            col=col_idx,
        )

    # Left subplot: Fixed vertical positions with horizontal velocity arrows
    arrow_indices = np.linspace(0, len(y) - 1, 9, dtype=int)
    y_arrows = y_mm[arrow_indices]
    u_arrows = u_mm_s[arrow_indices]

    # Starting dots at x = 0 showing fixed vertical layer positions
    fig.add_trace(
        go.Scatter(
            x=np.zeros_like(y_arrows),
            y=y_arrows,
            mode="markers",
            marker=dict(size=6, color="#555555"),
            name="Layer Reference (x=0)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Horizontal bars + arrowheads representing horizontal velocity magnitude at each fixed y
    for y_pos, u_val in zip(y_arrows, u_arrows):
        if u_val > 0.015 * U_axis_mm_s:
            fig.add_annotation(
                x=u_val,
                y=y_pos,
                ax=0.0,
                ay=y_pos,
                xref="x1",
                yref="y1",
                axref="x1",
                ayref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=2.2,
                arrowcolor=line_color,
            )

    # Invisible trace in col 1 to set x-axis scale cleanly
    fig.add_trace(
        go.Scatter(
            x=[0.0, U_axis_mm_s],
            y=[0.0, H_mm],
            mode="markers",
            marker=dict(opacity=0.0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Right subplot: Steady-state dashed line and current transient profile
    fig.add_trace(
        go.Scatter(
            x=u_steady_mm_s,
            y=y_mm,
            mode="lines",
            line=dict(color="#666666", width=2, dash="dash"),
            name="Steady profile (u = Uy/H)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=u_mm_s,
            y=y_mm,
            mode="lines",
            line=dict(color=line_color, width=3.5),
            name="Current u(y, t)",
        ),
        row=1,
        col=2,
    )

    pad_y = 0.04 * H_axis_mm
    fig.update_xaxes(
        title_text="Layer Velocity u (mm/s)",
        range=[-0.05 * U_axis_mm_s, 1.08 * U_axis_mm_s],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Horizontal Velocity u (mm/s)",
        range=[-0.05 * U_axis_mm_s, 1.08 * U_axis_mm_s],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Vertical Position y across Gap (mm)",
        range=[-pad_y, H_axis_mm + pad_y],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[-pad_y, H_axis_mm + pad_y],
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)),
        height=420,
        margin=dict(l=55, r=25, t=65, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig


def render_fluid_metrics(label: str, traj: Dict[str, object], frame_idx: int, U: float) -> None:
    """Display kinematic viscosity, characteristic time, analytical estimate, and numerical settling time."""
    nu = float(traj["nu"])
    nu_stokes = nu * 1e4
    t_char = float(traj["t_char"])
    t_est = float(traj["t_settle_est"])
    t_num = float(traj["t_settle_num"])
    u_curr = np.asarray(traj["velocity"])[frame_idx]
    u_steady = np.asarray(traj["u_steady"])
    t_curr = float(np.asarray(traj["times"])[frame_idx])

    max_rel_diff = float(np.max(np.abs(u_curr - u_steady)) / U)
    is_settled = t_curr >= t_num or max_rel_diff < 0.01

    st.markdown(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Kinematic viscosity ν",
        f"{nu:.2e} m²/s",
        f"{nu_stokes:.2f} St",
        delta_color="off",
    )
    c2.metric(
        "Characteristic time H²/ν",
        f"{t_char:.4f} s",
        "Scale reference",
        delta_color="off",
    )
    c3.metric(
        "Analytical 1% estimate",
        f"≈ {t_est:.4f} s",
        "0.421 H²/ν (estimate)",
        delta_color="off",
    )
    c4.metric(
        "Measured settling time",
        f"{t_num:.4f} s",
        "Settled (<1%)" if is_settled else f"Max diff: {max_rel_diff * 100:.1f}%",
        delta_color="normal" if is_settled else "off",
    )


def render_simulation_viewer(
    viewer_key: str,
    fluid_a: Dict[str, float],
    label_a: str,
    fluid_b: Dict[str, float] | None = None,
    label_b: str | None = None,
) -> None:
    """
    Shared interactive playback and visualization component used by both
    Section B (Predict and observe) and Section D (Free exploration).
    """
    # Detect parameter or experiment changes to automatically pause and reset time to t = 0
    param_signature = (
        viewer_key,
        round(fluid_a["rho"], 6),
        round(fluid_a["mu"], 6),
        round(fluid_a["H"], 6),
        round(fluid_a["U"], 6),
        None
        if fluid_b is None
        else (
            round(fluid_b["rho"], 6),
            round(fluid_b["mu"], 6),
            round(fluid_b["H"], 6),
            round(fluid_b["U"], 6),
        ),
    )

    sig_state_key = f"{viewer_key}_last_signature"
    frame_state_key = f"{viewer_key}_frame_idx"
    play_state_key = f"{viewer_key}_playing"
    slider_widget_key = f"{viewer_key}_time_slider"

    if st.session_state.get(sig_state_key) != param_signature:
        st.session_state[sig_state_key] = param_signature
        st.session_state[frame_state_key] = 0
        st.session_state[play_state_key] = False
        st.session_state[slider_widget_key] = 0.0

    if frame_state_key not in st.session_state:
        st.session_state[frame_state_key] = 0
    if play_state_key not in st.session_state:
        st.session_state[play_state_key] = False

    # Compute shared physical duration: 0.6 * H^2 / nu for single fluid,
    # or 0.6 * max(H1^2/nu1, H2^2/nu2) for two-fluid comparisons
    nu_a = fluid_a["mu"] / fluid_a["rho"]
    t_char_a = (fluid_a["H"] ** 2) / nu_a
    if fluid_b is not None:
        nu_b = fluid_b["mu"] / fluid_b["rho"]
        t_char_b = (fluid_b["H"] ** 2) / nu_b
        shared_duration = 0.6 * max(t_char_a, t_char_b)
    else:
        shared_duration = 0.6 * t_char_a

    n_frames = 121
    traj_a = cached_trajectory(
        fluid_a["rho"],
        fluid_a["mu"],
        fluid_a["H"],
        fluid_a["U"],
        duration=shared_duration,
        n_frames=n_frames,
    )
    traj_b = (
        cached_trajectory(
            fluid_b["rho"],
            fluid_b["mu"],
            fluid_b["H"],
            fluid_b["U"],
            duration=shared_duration,
            n_frames=n_frames,
        )
        if fluid_b is not None
        else None
    )

    times = np.asarray(traj_a["times"], dtype=float)
    dt_frame = float(traj_a["dt_frame"])

    # Playback controls row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 5.6])

    with ctrl_col1:
        play_label = "⏸ Pause" if st.session_state[play_state_key] else "▶ Play"
        if st.button(play_label, key=f"{viewer_key}_play_btn", use_container_width=True):
            if not st.session_state[play_state_key] and st.session_state[frame_state_key] >= n_frames - 1:
                st.session_state[frame_state_key] = 0
                st.session_state[slider_widget_key] = 0.0
            st.session_state[play_state_key] = not st.session_state[play_state_key]
            st.rerun()

    with ctrl_col2:
        if st.button("⏮ Reset", key=f"{viewer_key}_reset_btn", use_container_width=True):
            st.session_state[play_state_key] = False
            st.session_state[frame_state_key] = 0
            st.session_state[slider_widget_key] = 0.0
            st.rerun()

    current_idx = int(np.clip(st.session_state[frame_state_key], 0, n_frames - 1))
    if slider_widget_key not in st.session_state:
        st.session_state[slider_widget_key] = float(times[current_idx])

    def _on_slider_change():
        selected_t = float(st.session_state[slider_widget_key])
        nearest_idx = int(np.argmin(np.abs(times - selected_t)))
        st.session_state[frame_state_key] = nearest_idx
        st.session_state[play_state_key] = False

    with ctrl_col3:
        st.slider(
            "Elapsed physical time t (s)",
            min_value=0.0,
            max_value=float(times[-1]),
            step=float(dt_frame),
            format="%.4f s",
            key=slider_widget_key,
            on_change=_on_slider_change,
        )

    current_idx = int(np.clip(st.session_state[frame_state_key], 0, n_frames - 1))
    current_time = float(times[current_idx])

    # Accelerated / scaled playback notice & physical time increment
    frame_wall_time_s = 0.06
    playback_ratio = dt_frame / frame_wall_time_s if frame_wall_time_s > 0 else 1.0
    playback_desc = (
        f" Slow-motion playback ({playback_ratio:.2f}× real time)"
        if playback_ratio < 0.95
        else (
            f"Accelerated playback ({playback_ratio:.2f}× real time)"
            if playback_ratio > 1.05
            else "Near real-time playback (1.0×)"
        )
    )
    st.caption(
        f"**Elapsed physical time:** `{current_time:.4f} s` (Frame {current_idx + 1} of {n_frames}) | "
        f"**Physical-time increment per frame:** `Δt_frame = {dt_frame:.5f} s` (`{dt_frame * 1000.0:.2f} ms`) | "
        f"**Playback rate:** {playback_desc}. "
        f"*Note: Horizontal arrows represent fluid velocity at fixed vertical positions, not vertical fluid movement or particle trajectories.*"
    )

    if current_idx == 0:
        st.info(
            "**Initial Condition (t = 0 s):** The upper plate at $y = H$ has just started moving rightward at speed $U$, "
            "while the interior fluid ($0 \\le y < H$) is still stationary. An instantaneous plate start creates an "
            "idealized wall-stress singularity and an initially unresolved sub-grid boundary layer at $t = 0^+$."
        )

    # Render metrics and plots
    if traj_b is None:
        render_fluid_metrics(label_a, traj_a, current_idx, fluid_a["U"])
        fig_a = build_fluid_figure(
            title=label_a,
            y=np.asarray(traj_a["y"]),
            u_curr=np.asarray(traj_a["velocity"])[current_idx],
            u_steady=np.asarray(traj_a["u_steady"]),
            H=fluid_a["H"],
            H_axis_max=fluid_a["H"],
            U_axis_max=fluid_a["U"],
            line_color="#1f77b4",
        )
        st.plotly_chart(fig_a, use_container_width=True)
    else:
        assert fluid_b is not None and label_b is not None
        H_common = max(fluid_a["H"], fluid_b["H"])
        U_common = max(fluid_a["U"], fluid_b["U"])

        render_fluid_metrics(label_a, traj_a, current_idx, fluid_a["U"])
        render_fluid_metrics(label_b, traj_b, current_idx, fluid_b["U"])

        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig_a = build_fluid_figure(
                title=label_a,
                y=np.asarray(traj_a["y"]),
                u_curr=np.asarray(traj_a["velocity"])[current_idx],
                u_steady=np.asarray(traj_a["u_steady"]),
                H=fluid_a["H"],
                H_axis_max=H_common,
                U_axis_max=U_common,
                line_color="#1f77b4",
            )
            st.plotly_chart(fig_a, use_container_width=True)

        with plot_col2:
            fig_b = build_fluid_figure(
                title=label_b,
                y=np.asarray(traj_b["y"]),
                u_curr=np.asarray(traj_b["velocity"])[current_idx],
                u_steady=np.asarray(traj_b["u_steady"]),
                H=fluid_b["H"],
                H_axis_max=H_common,
                U_axis_max=U_common,
                line_color="#d95f02",
            )
            st.plotly_chart(fig_b, use_container_width=True)

    # Advance playback frame if currently playing
    if st.session_state[play_state_key]:
        if current_idx < n_frames - 1:
            time.sleep(frame_wall_time_s)
            next_idx = current_idx + 1
            st.session_state[frame_state_key] = next_idx
            st.session_state[slider_widget_key] = float(times[next_idx])
            st.rerun()
        else:
            st.session_state[play_state_key] = False
            st.rerun()


def build_thin_layer_force_diagram() -> go.Figure:
    """Create a labeled thin-layer force balance diagram for Section E (Optional Derivation)."""
    fig = go.Figure()

    # Fluid control volume box from y to y + dy
    fig.add_shape(
        type="rect",
        x0=1.5,
        x1=6.5,
        y0=2.0,
        y1=4.0,
        line=dict(color="#1f77b4", width=2.5),
        fillcolor="rgba(31, 119, 180, 0.15)",
    )

    # Top shear traction arrow (faster fluid above pulls layer rightward)
    fig.add_annotation(
        x=6.2,
        y=4.35,
        ax=2.2,
        ay=4.35,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=3,
        arrowcolor="#2ca02c",
    )
    fig.add_annotation(
        x=4.2,
        y=4.75,
        text="Top shear force: +A · τ(y + Δy)  [faster fluid above pulls rightward]",
        showarrow=False,
        font=dict(size=13, color="#1b691b"),
    )

    # Bottom shear traction arrow (slower fluid below pulls layer leftward)
    fig.add_annotation(
        x=2.2,
        y=1.65,
        ax=5.2,
        ay=1.65,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=3,
        arrowcolor="#d62728",
    )
    fig.add_annotation(
        x=3.7,
        y=1.2,
        text="Bottom shear force: −A · τ(y)  [slower fluid below resists rightward motion]",
        showarrow=False,
        font=dict(size=13, color="#a11d1d"),
    )

    # Labels inside and beside the thin fluid layer
    fig.add_annotation(
        x=4.0,
        y=3.0,
        text=(
            "<b>Thin Fluid Layer</b> (mass = ρ A Δy)<br>"
            "Net rightward force = A [τ(y + Δy) − τ(y)]"
        ),
        showarrow=False,
        font=dict(size=13, color="#0d3b66"),
    )
    fig.add_annotation(
        x=1.05,
        y=2.0,
        text="y",
        showarrow=False,
        font=dict(size=13),
    )
    fig.add_annotation(
        x=0.85,
        y=4.0,
        text="y + Δy",
        showarrow=False,
        font=dict(size=13),
    )

    fig.update_xaxes(visible=False, range=[0.0, 8.0])
    fig.update_yaxes(visible=False, range=[0.5, 5.4])
    fig.update_layout(
        title="Thin Fluid Element Horizontal Force Balance",
        height=290,
        margin=dict(l=20, r=20, t=45, b=15),
        template="plotly_white",
    )
    return fig


def build_coaxial_viscometer_diagram() -> go.Figure:
    """Create an original top-down coaxial-cylinder viscometer diagram with equal axis scaling."""
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    r_inner = 1.0
    r_outer = 1.35

    fig = go.Figure()

    # Outer rotating cup boundary
    fig.add_trace(
        go.Scatter(
            x=r_outer * np.cos(theta),
            y=r_outer * np.sin(theta),
            mode="lines",
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.16)",
            line=dict(color="#1f77b4", width=3.5),
            name="Rotating Outer Cup (Radius R_o, angular speed ω)",
        )
    )

    # Inner stationary bob
    fig.add_trace(
        go.Scatter(
            x=r_inner * np.cos(theta),
            y=r_inner * np.sin(theta),
            mode="lines",
            fill="toself",
            fillcolor="rgba(220, 220, 220, 0.95)",
            line=dict(color="#333333", width=3),
            name="Stationary Inner Cylinder / Bob (Radius R_i)",
        )
    )

    # Radius R_i arrow (horizontal)
    fig.add_annotation(
        x=r_inner,
        y=0.0,
        ax=0.0,
        ay=0.0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor="#333333",
    )
    fig.add_annotation(
        x=0.5,
        y=0.12,
        text="R<sub>i</sub>",
        showarrow=False,
        font=dict(size=13, color="#222222"),
    )

    # Radius R_o arrow (angled)
    ang = np.deg2rad(42.0)
    fig.add_annotation(
        x=r_outer * np.cos(ang),
        y=r_outer * np.sin(ang),
        ax=0.0,
        ay=0.0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor="#1f77b4",
    )
    fig.add_annotation(
        x=0.72 * np.cos(ang) - 0.1,
        y=0.72 * np.sin(ang) + 0.12,
        text="R<sub>o</sub>",
        showarrow=False,
        font=dict(size=13, color="#1f77b4"),
    )

    # Rotation direction arrow near the top of the outer cup
    fig.add_annotation(
        x=-0.45,
        y=1.48,
        ax=0.45,
        ay=1.48,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=2.5,
        arrowcolor="#d95f02",
    )
    fig.add_annotation(
        x=0.0,
        y=1.66,
        text="Outer Cup Rotation (U = ω R<sub>o</sub>)",
        showarrow=False,
        font=dict(size=13, color="#d95f02"),
    )

    # Labels for stationary bob and annular fluid gap
    fig.add_annotation(
        x=0.0,
        y=-0.28,
        text="<b>Stationary Inner Cylinder</b><br>(Torque sensor)",
        showarrow=False,
        font=dict(size=12, color="#222222"),
    )
    fig.add_annotation(
        x=0.0,
        y=-1.18,
        text="Fluid in narrow annular gap (H = R<sub>o</sub> − R<sub>i</sub>)",
        showarrow=False,
        font=dict(size=12, color="#0d3b66"),
    )

    fig.update_xaxes(visible=False, range=[-1.9, 1.9])
    fig.update_yaxes(visible=False, range=[-1.8, 1.9], scaleanchor="x", scaleratio=1.0)
    fig.update_layout(
        title="Top-Down Schematic: Coaxial-Cylinder (Couette) Viscometer",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        template="plotly_white",
    )
    return fig


# ==============================================================================
# Header & Navigation
# ==============================================================================
st.title("🌊 Momentum Diffusion Lab")
st.subheader(
    "“How does motion spread through a fluid, and why do we need two definitions of viscosity?”"
)

tabs = st.tabs(
    [
        "A. Start Here",
        "B. Predict & Observe",
        "C & D. Free Exploration",
        "E. Optional Derivation",
        "F. Applications & Analogy",
        "AI Evidence & Comparison",
        "Presentation & References",
    ]
)

# ==============================================================================
# Section A: Start Here
# ==============================================================================
with tabs[0]:
    st.header("A. Start Here: Physical Intuition & Project Overview")

    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.markdown(
            """
            ### How Does Motion Spread Through a Fluid?
            Imagine an initially stationary fluid trapped between two horizontal parallel plates separated by a gap $H$.
            At time $t = 0$, the lower plate at $y = 0$ stays still while the upper plate at $y = H$ suddenly begins sliding
            rightward at constant speed $U$.

            **How does the fluid in the middle start moving?**
            1. **Viscosity transmits momentum between layers:** Because the fluid does not slip at the walls, the top fluid layer
               immediately moves with the upper plate at speed $U$. Viscosity resists relative sliding between neighboring horizontal
               layers, which means the faster layer above pulls the slower layer below it forward (rightward), while the slower layer
               below pulls the faster layer backward (leftward).
            2. **Unequal shear forces cause acceleration:** Consider a thin horizontal fluid layer inside the gap. The faster fluid
               above pulls its top face rightward, while the slower fluid below holds its bottom face back. Whenever the rightward
               pull on the top face is stronger than the leftward resistance on the bottom face, the layer experiences a **net horizontal
               force** and accelerates rightward ($F_{\\mathrm{net}} = m a$).
            3. **Equal opposing forces mean zero acceleration — not zero velocity:** Eventually, the velocity profile becomes a straight line
               from $0$ at the bottom plate to $U$ at the top plate:
               $$u_{\\mathrm{steady}}(y) = U\\frac{y}{H}$$
               At steady state, the velocity slope $\\partial u / \\partial y = U/H$ is uniform everywhere across the gap. The rightward pull
               on the top of every fluid layer exactly balances the leftward pull on its bottom. With zero net force, acceleration stops and
               each layer continues gliding rightward at its steady speed.
            """
        )

    with col_intro2:
        st.info(
            """
            ### About the AI Investigation
            Beyond teaching startup Couette flow, this project investigates how two AI systems
            (**Astra** and **Gemini**) independently implement and explain the same Chemical Engineering
            problem from a frozen specification, and how each responds to **one consolidated correction round**.

            - **Audience:** Beginning ChemE students familiar with shear stress and velocity gradients, prior to studying PDEs.
            - **Assumptions:** Incompressible Newtonian fluid, constant density $\\rho$ and viscosity $\\mu$, laminar unidirectional horizontal motion, no slip at the plates, no imposed horizontal pressure gradient or body force, and negligible edge effects.
            """
        )

    st.markdown("---")
    st.subheader("Governing Equations, Sign Convention, and the Two Viscosities")

    st.latex(
        r"\tau = \mu \frac{\partial u}{\partial y}, \qquad "
        r"\nu = \frac{\mu}{\rho}, \qquad "
        r"\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial y^2}"
    )

    st.markdown(
        """
        - **Why we need two definitions of viscosity:**
          - **Dynamic viscosity ($\\mu$)** relates **shear stress** ($\\tau$, force per unit area) to the **velocity gradient** ($\\partial u/\\partial y$).
            It tells you how much force per unit area is required to slide one fluid layer past another.
          - **Kinematic viscosity ($\\nu = \\mu / \\rho$)** divides dynamic viscosity by fluid density (mass per unit volume). Because acceleration depends
            on force divided by mass, $\\nu$ determines the **rate of momentum diffusion** — how quickly velocity changes spread across the gap.
          - **Comparing fluids with equal $\\nu$:** Equal kinematic viscosity $\\nu$ gives **identical velocity evolution $u(y,t)$** under identical geometry
            and plate motion, even though a fluid with greater dynamic viscosity $\\mu$ (and proportionally greater density $\\rho$) requires **greater shear stress**
            for the same velocity gradient.
        - **Shear stress sign convention:** Here, positive $\\tau = \\mu \\, \\partial u / \\partial y$ denotes the rightward traction on a fluid element's face
          whose outward normal points in the positive $y$ (upward) direction; the opposite face (outward normal in $-y$) has the opposite traction sign.
        """
    )

    st.markdown(
        """
        | Symbol | Meaning | SI Unit |
        | :--- | :--- | :--- |
        | $u$ | Horizontal fluid velocity | $\\mathrm{m/s}$ |
        | $y$ | Position across the gap | $\\mathrm{m}$ |
        | $t$ | Elapsed physical time | $\\mathrm{s}$ |
        | $U$ | Upper-plate speed | $\\mathrm{m/s}$ |
        | $H$ | Plate gap | $\\mathrm{m}$ |
        | $\\rho$ | Density | $\\mathrm{kg/m^3}$ |
        | $\\mu$ | Dynamic viscosity | $\\mathrm{Pa\\cdot s}$ |
        | $\\nu$ | Kinematic viscosity | $\\mathrm{m^2/s}$ |
        | $\\tau$ | Shear stress | $\\mathrm{Pa}$ |
        """
    )

# ==============================================================================
# Section B: Predict and Observe
# ==============================================================================
with tabs[1]:
    st.header("B. Predict and Observe: Controlled Comparisons")
    st.markdown(
        """
        All four guided experiments compare a modified hypothetical fluid against the **Baseline Fluid**:
        $\\rho = 1000\\ \\mathrm{kg/m^3}$, $\\mu = 0.1\\ \\mathrm{Pa\\cdot s}$ ($\\nu = 1.0 \\times 10^{-4}\\ \\mathrm{m^2/s} = 1.0\\ \\mathrm{St}$),
        $H = 0.005\\ \\mathrm{m}$, and $U = 0.02\\ \\mathrm{m/s}$.
        """
    )

    selected_exp_key = st.radio(
        "Choose a guided comparison experiment:",
        options=list(GUIDED_EXPERIMENTS.keys()),
        format_func=lambda k: GUIDED_EXPERIMENTS[k]["title"],
    )

    exp_data = GUIDED_EXPERIMENTS[selected_exp_key]

    st.markdown(f"### {exp_data['title']}")
    st.markdown(f"**Prediction Prompt:** {exp_data['question']}")

    pred_choice = st.selectbox(
        "Record your prediction before revealing the explanation:",
        options=exp_data["options"],
        key=f"pred_select_{selected_exp_key}",
    )

    if pred_choice == "Select your prediction...":
        st.warning(
            "👆 Select a prediction above to unlock the physical explanation, then use the simulation below to test your intuition."
        )
    else:
        if pred_choice == exp_data["correct_option"]:
            st.success(f"**Your prediction matches the physics!** ({pred_choice})")
        else:
            st.info(
                f"**Let's look closely at what happens:** You predicted *“{pred_choice}”*, "
                f"whereas in the simulation *{exp_data['correct_option'].lower()}*. Here is why:"
            )
        st.markdown(exp_data["explanation"])

    st.markdown("---")
    render_simulation_viewer(
        viewer_key=f"guided_{selected_exp_key}",
        fluid_a=BASELINE_PARAMS,
        label_a=(
            f"Baseline Fluid (ρ={BASELINE_PARAMS['rho']:.0f} kg/m³, "
            f"μ={BASELINE_PARAMS['mu']:.2f} Pa·s, H={BASELINE_PARAMS['H']*1000:.1f} mm)"
        ),
        fluid_b=exp_data["params"],
        label_b=(
            f"Modified Fluid (ρ={exp_data['params']['rho']:.0f} kg/m³, "
            f"μ={exp_data['params']['mu']:.2f} Pa·s, H={exp_data['params']['H']*1000:.1f} mm)"
        ),
    )

# ==============================================================================
# Sections C & D: Free Exploration & Visualizations
# ==============================================================================
with tabs[2]:
    st.header("C & D. Free Exploration")
    st.markdown(
        """
        Explore how density $\\rho$, dynamic viscosity $\\mu$, plate gap $H$, and upper-plate speed $U$ shape
        momentum diffusion. These bounded inputs represent **conservative laminar teaching cases** (keeping $UH/\\nu \\le 15$),
        not a universal laminar-flow threshold.
        """
    )

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        free_rho = st.slider(
            "Density ρ (kg/m³)",
            min_value=500.0,
            max_value=1500.0,
            value=1000.0,
            step=50.0,
        )
    with s_col2:
        free_mu = st.slider(
            "Dynamic viscosity μ (Pa·s)",
            min_value=0.05,
            max_value=1.00,
            value=0.10,
            step=0.05,
        )
    with s_col3:
        free_H = st.slider(
            "Plate gap H (m)",
            min_value=0.002,
            max_value=0.010,
            value=0.005,
            step=0.001,
            format="%.3f",
        )
    with s_col4:
        free_U = st.slider(
            "Upper-plate speed U (m/s)",
            min_value=0.005,
            max_value=0.050,
            value=0.020,
            step=0.005,
            format="%.3f",
        )

    st.markdown(
        """
        **Definitions of Time Scales Shown Below:**
        - **Characteristic diffusion time:** $t_{\\mathrm{char}} = \\frac{H^2}{\\nu}$
        - **Settled criterion:** $\\frac{\\max_y \vert{}u(y, t) - u_{\\mathrm{steady}}(y)\vert{}}{U} < 0.01$
        - **Analytical late-time settling estimate:** $\\approx 0.421\\frac{H^2}{\\nu}$ (an analytical leading-mode estimate, distinct from the measured numerical settling time).
        """
    )

    free_fluid = {"rho": free_rho, "mu": free_mu, "H": free_H, "U": free_U}
    render_simulation_viewer(
        viewer_key="free_exploration",
        fluid_a=free_fluid,
        label_a=(
            f"Custom Hypothetical Fluid (ρ={free_rho:.0f} kg/m³, "
            f"μ={free_mu:.2f} Pa·s, H={free_H*1000:.1f} mm, U={free_U*1000:.1f} mm/s)"
        ),
    )

# ==============================================================================
# Section E: Optional Derivation
# ==============================================================================
with tabs[3]:
    st.header("E. Optional Derivation: Thin-Layer Force Balance")
    st.markdown(
        """
        ### Physical Meaning Before Calculus
        - A **velocity gradient** ($\\partial u / \\partial y$) across the fluid creates **shear stress** ($\\tau = \\mu\\,\\partial u / \\partial y$).
        - A **difference in shear stress** between the top and bottom faces of a fluid layer creates a **net force**, which causes **acceleration** ($\\partial u / \\partial t$).
        - When the velocity profile is curved ($\\partial^2 u / \\partial y^2 \\ne 0$), the slope $\\partial u / \\partial y$ is different at the top and bottom of the layer,
          so the shear stresses do not cancel and the layer accelerates.
        """
    )

    st.plotly_chart(build_thin_layer_force_diagram(), use_container_width=True)

    with st.expander("📐 Expand Step-by-Step Calculus Derivation of the Momentum Diffusion Equation"):
        st.markdown(
            """
            #### Step 1: Newton's Second Law on a Thin Fluid Layer
            Consider a horizontal fluid layer of thickness $\\Delta y$ and horizontal plate area $A$ located between vertical positions $y$ and $y + \\Delta y$:
            - **Mass of the layer:** $m = \\rho A \\Delta y$
            - **Rightward shear force on the top face** at $y + \\Delta y$ (outward normal in $+y$): $+A\\,\\tau(y + \\Delta y)$
            - **Leftward shear force on the bottom face** at $y$ (outward normal in $-y$): $-A\\,\\tau(y)$

            Writing Newton's second law ($m \\, \\partial u / \\partial t = F_{\\mathrm{net}}$) in the horizontal direction:
            """
        )
        st.latex(r"\rho A \Delta y \frac{\partial u}{\partial t} = A\left[\tau(y + \Delta y) - \tau(y)\right]")

        st.markdown(
            """
            #### Step 2: Divide by Volume $A \\Delta y$ and Take the Thin-Layer Limit
            Dividing both sides by $A \\Delta y$ and taking the limit as $\\Delta y \\to 0$ gives the differential momentum balance:
            """
        )
        st.latex(r"\rho \frac{\partial u}{\partial t} = \frac{\partial \tau}{\partial y}")

        st.markdown(
            """
            #### Step 3: Substitute Newton's Law of Viscosity
            For a Newtonian fluid with constant dynamic viscosity $\\mu$, $\\tau = \\mu \\frac{\\partial u}{\\partial y}$. Substituting this into the balance:
            """
        )
        st.latex(r"\rho \frac{\partial u}{\partial t} = \frac{\partial \tau}{\partial y} = \mu \frac{\partial^2 u}{\partial y^2}")

        st.markdown(
            """
            Dividing both sides by density $\\rho$ and defining kinematic viscosity $\\nu = \\mu / \\rho$ yields the 1D momentum diffusion equation:
            """
        )
        st.latex(r"\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial y^2}")

        st.markdown(
            """
            #### Plain-English Translation of the Final Equation
            $$\\underbrace{\\frac{\\partial u}{\\partial t}}_{\\substack{\\text{Rate of change of} \\\\ \\text{layer velocity with time}}} = \\underbrace{\\nu}_{\\substack{\\text{Kinematic viscosity} \\\\ (\\mu / \\rho)}} \\times \\underbrace{\\frac{\\partial^2 u}{\\partial y^2}}_{\\substack{\\text{Spatial curvature of the} \\\\ \\text{velocity profile across the gap}}}$$
            """
        )

# ==============================================================================
# Section F: Applications and Transport Analogy
# ==============================================================================
with tabs[4]:
    st.header("F. Applications and Transport Analogy")

    col_app1, col_app2 = st.columns([1.15, 1.0])
    with col_app1:
        st.plotly_chart(build_coaxial_viscometer_diagram(), use_container_width=True)

    with col_app2:
        st.subheader("How a Coaxial-Cylinder Viscometer Uses Couette Flow")
        st.markdown(
            """
            Infinite flat plates are inconvenient in a laboratory because fluid would spill out the ends.
            Instead, engineers bend the parallel plates into a circle to build a **coaxial-cylinder (Couette) viscometer**:

            1. **Measuring viscosity via torque:** The outer cup of radius $R_o$ rotates at a known angular speed $\\omega$
               (giving wall speed $U = \\omega R_o$), shearing the fluid in the annular gap. The fluid exerts a viscous shear traction
               on the stationary inner cylinder (radius $R_i$), and an instrument measures the **torque** required to hold the inner
               cylinder stationary. From that torque and the rotation speed, the fluid's dynamic viscosity $\\mu$ is determined.
            2. **Narrow-gap parallel-plate approximation:** When the annular gap is very narrow compared to the cylinder radius
               ($H = R_o - R_i \\ll R_i$), the wall curvature is slight and our flat parallel-plate model is an accurate local approximation.
            3. **Effect of curvature:** When the gap is wider, cylindrical curvature changes the full equations of motion because the surface
               area $2\\pi r L$ increases with radius $r$, so shear stress is no longer strictly constant across the gap even at steady state.
            """
        )

    st.markdown("---")
    st.subheader("Transport Analogy: Momentum, Heat, and Mass Diffusion")
    st.markdown(
        """
        Under constant-property, one-dimensional diffusion assumptions, **momentum**, **heat**, and **mass** transport
        all obey mathematically analogous diffusion equations:

        | Transport Process | Diffusivity Symbol & Definition | 1D Diffusion Equation | SI Unit |
        | :--- | :--- | :--- | :--- |
        | **Momentum diffusion** | Kinematic viscosity: $\\nu = \\dfrac{\\mu}{\\rho}$ | $\\dfrac{\\partial u}{\\partial t} = \\nu \\dfrac{\\partial^2 u}{\\partial y^2}$ | $\\mathrm{m^2/s}$ |
        | **Thermal (heat) diffusion** | Thermal diffusivity: $\\alpha = \\dfrac{k}{\\rho c_p}$ | $\\dfrac{\\partial T}{\\partial t} = \\alpha \\dfrac{\\partial^2 T}{\\partial y^2}$ | $\\mathrm{m^2/s}$ |
        | **Mass (species) diffusion** | Mass diffusivity: $D$ | $\\dfrac{\\partial C_A}{\\partial t} = D \\dfrac{\\partial^2 C_A}{\\partial y^2}$ | $\\mathrm{m^2/s}$ |

        **Additional Symbols Defined:**
        - $k$: Thermal conductivity ($\\mathrm{W/(m\\cdot K)}$)
        - $c_p$: Specific heat capacity at constant pressure ($\\mathrm{J/(kg\\cdot K)}$)
        - $T$: Temperature ($\\mathrm{K}$)
        - $C_A$: Molar concentration of species $A$ ($\\mathrm{mol/m^3}$)
        - $D$: Binary mass diffusion coefficient ($\\mathrm{m^2/s}$)

        ### Shared Dimensions and the Stokes ($\mathrm{St}$) Unit
        All three diffusivities ($\\nu$, $\\alpha$, and $D$) have dimensions of **$\\mathrm{length^2 / time}$** ($\\mathrm{m^2/s}$ in SI).
        In CGS units, $1\\ \\mathrm{cm^2/s}$ is called one **stokes ($\\mathrm{St}$)**:
        $$1\\ \\mathrm{St} = 1\\ \\mathrm{cm^2/s} = 10^{-4}\\ \\mathrm{m^2/s}$$
        Because $\\nu$, $\\alpha$, and $D$ share identical dimensions of $\\mathrm{length^2/time}$, expressing *any* of the three diffusivities
        in stokes (or centistokes, $1\\ \\mathrm{cSt} = 10^{-6}\\ \\mathrm{m^2/s}$) is **dimensionally valid**, although by convention the unit stokes
        is used almost exclusively for kinematic viscosity $\\nu$.
        """
    )

# ==============================================================================
# Section 6: AI Evidence and Attribution
# ==============================================================================
with tabs[5]:
    st.header("AI Evidence, Critique Decisions, and Two-Model Comparison")
    evidence = load_json_content("ai_evidence.json")

    crit_meta = evidence.get("critique_metadata", {})
    st.subheader("1. Initial Plan Critique Metadata & Confirmed Student Decisions")
    st.markdown(
        f"""
        - **Supplied Critique Model Label:** `{crit_meta.get('critique_model_label', '4 Argon High Effort')}`
        - **Provider / Interface:** `{crit_meta.get('provider_interface', 'Pending confirmation')}` *(Not silently relabeled)*
        - **Critique Date:** `{crit_meta.get('critique_date', 'Pending')}`
        - **Tool-Access Conditions:** `{crit_meta.get('tool_access', 'Pending')}`
        """
    )

    decisions = evidence.get("critique_decisions", [])
    if decisions:
        st.table(decisions)

    st.markdown("---")
    st.subheader("2. Two-Model Comparison (Astra vs. Gemini — One Correction Round Each)")
    comp = evidence.get("model_comparison", {})
    st.info(
        f"**Selection Status:** {comp.get('selection_status', 'Pending')}\n\n"
        f"**External Verification Note:** {comp.get('external_validation_suite_status', 'Pending')} "
        "Agreement between this app's numerical solver and its own `analytical_couette` function is an internal consistency check, "
        "not independent verification."
    )

    models_list = comp.get("models", [])
    if models_list:
        st.table(models_list)

    st.markdown(
        """
        **Correction Protocol Rule:** Each model's single consolidated correction prompt contains only observed test failures,
        usability problems, and violated specification requirements — never a student-written code patch or unequal diagnostic help.
        Original and corrected submissions are preserved separately in the repository.
        """
    )

    st.markdown("---")
    st.subheader("3. Attribution: Student Contributions vs. AI Contributions")
    contribs = evidence.get("contributions", {})
    c_stud, c_ai = st.columns(2)
    with c_stud:
        st.markdown("#### Student Contributions")
        for item in contribs.get("student_contributions", []):
            st.markdown(f"- {item}")
    with c_ai:
        st.markdown("#### AI Contributions")
        for item in contribs.get("ai_contributions", []):
            st.markdown(f"- {item}")

    st.markdown("#### Independently Verified Findings")
    for item in contribs.get("independently_verified_findings", []):
        st.markdown(f"- {item}")

    st.markdown("---")
    st.subheader("4. Student Reflections & Future Learning Questions")
    reflections = evidence.get("student_reflections_and_future_questions", {})
    st.markdown(f"- **What I Learned:** {reflections.get('what_i_learned', 'Pending')}")
    st.markdown(f"- **Future Questions:** {reflections.get('future_questions', 'Pending')}")
    st.caption(
        "Edit `content/ai_evidence.json` to update model comparison results and personal reflections after completing the external evaluation."
    )

# ==============================================================================
# Section 7: Presentation, Transcript, and References
# ==============================================================================
with tabs[6]:
    st.header("Presentation, Verbatim YouTube Transcript, and References")
    pres_meta = load_json_content("presentation_metadata.json")
    transcript_raw = load_text_content("transcript.txt")
    errata_raw = load_text_content("transcript_errata.md")

    st.subheader("1. YouTube Presentation Video")
    yt_url = pres_meta.get("youtube_url", "Pending")
    if yt_url == "Pending" or not yt_url.startswith("http"):
        st.warning(
            "**YouTube URL:** `Pending` — Record the presentation after validation and update `youtube_url` in `content/presentation_metadata.json`."
        )
    else:
        st.markdown(f"**Public / Unlisted YouTube Presentation Link:** [{yt_url}]({yt_url})")
        st.video(yt_url)

    st.subheader("2. Presentation Outline")
    for item in pres_meta.get("presentation_outline", []):
        st.markdown(f"- {item}")

    st.markdown("---")
    st.subheader("3. Complete, Unedited YouTube Auto-Generated Transcript")
    st.caption(
        "Displayed verbatim from `content/transcript.txt` with original timestamps and line breaks preserved."
    )
    st.code(transcript_raw, language="text")

    st.subheader("4. Separate Transcript Errata / Technical Clarifications")
    st.caption(
        "Stored separately in `content/transcript_errata.md` so the auto-generated transcript above remains completely unedited."
    )
    st.markdown(errata_raw)

    st.markdown("---")
    st.subheader("5. References (Physics, Numerical Methods, and Software)")
    st.markdown(
        """
        - **Transport Phenomena & Startup Plane Couette Flow Analytical Series:**
          - Bird, R. B., Stewart, W. E., & Lightfoot, E. N. (2002). *Transport Phenomena* (2nd ed.). John Wiley & Sons. (Chapter 1: Viscosity and the Mechanisms of Momentum Transport; Chapter 4: Shell Momentum Balances and Velocity Distributions in Laminar Flow, Unsteady Viscous Flow between Parallel Plates).
          - Batchelor, G. K. (1967). *An Introduction to Fluid Dynamics*. Cambridge University Press. (Section 4.3: Unsteady unidirectional flow — Plane Couette flow started from rest).
        - **Explicit Finite-Difference Method (FTCS) & Von Neumann Stability ($Fo \\le 0.5$):**
          - Chapra, S. C., & Canale, R. P. (2015). *Numerical Methods for Engineers* (7th ed.). McGraw-Hill Education. (Parabolic Partial Differential Equations: Explicit Forward-Time Centered-Space method and stability criterion).
        - **Software & Scientific Computing Libraries:**
          - Python Software Foundation. *Python Language Reference* (Version 3.x). https://www.python.org/
          - Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://numpy.org/
          - Plotly Technologies Inc. *Collaborative data science / Plotly Python Open Source Graphing Library*. https://plotly.com/python/
          - Streamlit Inc. *Streamlit — Faster way to build and share data apps*. https://streamlit.io/
        """
    )
