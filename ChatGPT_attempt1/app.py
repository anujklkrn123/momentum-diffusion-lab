"""Momentum Diffusion Lab. Run with: streamlit run app.py."""
import html
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import streamlit as st

from cases import BASELINE, EXPERIMENTS, Fluid, shared_duration
from charts import arrows_figure, cylinder_diagram, force_diagram, profile_figure
from physics import _solve_couette_with_metadata

ROOT = Path(__file__).resolve().parent
TICK_SECONDS = 0.25
FRAMES = 121

st.set_page_config(page_title="Momentum Diffusion Lab", page_icon="🌊", layout="wide")


def read_text(relative):
    path = ROOT / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(relative):
    raw = read_text(relative)
    return json.loads(raw) if raw.strip() else {}


@st.cache_data(show_spinner=False, max_entries=32)
def calculate_trajectories(fluids):
    times = np.linspace(0, shared_duration(fluids), FRAMES)
    trajectories = []
    for fluid in fluids:
        trajectories.append(_solve_couette_with_metadata(
            fluid.rho, fluid.mu, fluid.H, fluid.U, times,
            n_nodes=51, fourier_number=0.4,
        ))
    return times, trajectories


def reset_player():
    st.session_state.playing = False
    st.session_state.time_frame = 0
    st.session_state.display_frame = 0
    st.session_state.sync_scrubber = False
    st.session_state.last_tick = time.monotonic()


def scrub_time():
    # Manual time changes pause the timer rather than racing it.
    st.session_state.playing = False
    st.session_state.display_frame = st.session_state.time_frame
    st.session_state.last_tick = time.monotonic()


def sync_player(signature):
    if st.session_state.get("player_signature") != signature or "display_frame" not in st.session_state:
        st.session_state.player_signature = signature
        reset_player()


def show_player(fluids):
    try:
        times, trajectories = calculate_trajectories(tuple(fluids))
    except ValueError as exc:
        st.error(f"This case cannot be simulated: {exc}")
        return

    metric_columns = st.columns(len(fluids))
    for column, fluid, trajectory in zip(metric_columns, fluids, trajectories):
        with column:
            st.markdown(f"**{fluid.label}**")
            st.caption(f"ρ = {fluid.rho:g} kg/m³ · μ = {fluid.mu:g} Pa·s · H = {fluid.H*1000:g} mm · U = {fluid.U:g} m/s")
            a, b = st.columns(2)
            a.metric("Kinematic viscosity ν", f"{fluid.nu:.3g} m²/s")
            b.metric("Characteristic time H²/ν", f"{fluid.characteristic_time:.4g} s")
            settled = trajectory[2]["settling_time"]
            st.caption(f"Measured 1% settling time: {settled:.5g} s" if settled is not None else "1% settling was not reached in this duration.")
            st.caption(f"Late-time analytical estimate: {0.421 * fluid.characteristic_time:.5g} s (approximate).")

    # Keep the input widget outside the timer fragment. Programmatically moving
    # a live widget can arrive back from the frontend as a user-change event.
    if st.session_state.get("sync_scrubber", False):
        st.session_state.time_frame = st.session_state.display_frame
        st.session_state.sync_scrubber = False
    buttons = st.columns([1, 1, 3])
    if buttons[0].button("Pause" if st.session_state.playing else "Play", key="play_pause", width="stretch"):
        if st.session_state.display_frame == FRAMES - 1:
            st.session_state.display_frame = 0
        st.session_state.playing = not st.session_state.playing
        st.session_state.time_frame = st.session_state.display_frame
        st.session_state.last_tick = time.monotonic()
        st.rerun()
    if buttons[1].button("Reset", key="reset_time", width="stretch"):
        reset_player()
        st.rerun()
    buttons[2].caption("Both cases share one physical clock." if len(fluids) == 2 else "Scrub to any displayed time; changing time pauses playback.")
    st.select_slider(
        "Elapsed physical time (s)", options=list(range(FRAMES)),
        key="time_frame", format_func=lambda i: f"{times[i]:.5g} s", on_change=scrub_time,
    )
    st.caption("Drag the time slider to pause and inspect a moment. Its thumb updates to the current time when you pause playback.")
    increment = times[1] - times[0]
    rate = increment / TICK_SECONDS
    st.caption(f"Playback: +{increment:.5g} physical seconds every nominal {TICK_SECONDS:g} wall-clock seconds ({rate:.3g}× real time). Timer delays can slow playback; both cases always advance together.")
    running = st.session_state.get("playing", False)

    @st.fragment(run_every=TICK_SECONDS if running else None)
    def player_panel():
        now = time.monotonic()
        if st.session_state.playing and now - st.session_state.last_tick >= TICK_SECONDS * 0.95:
            st.session_state.display_frame = min(FRAMES - 1, st.session_state.display_frame + 1)
            st.session_state.last_tick = now
            if st.session_state.display_frame == FRAMES - 1:
                st.session_state.playing = False
                st.session_state.sync_scrubber = True
                st.rerun()
        frame = st.session_state.display_frame
        st.markdown(f"**Elapsed time: {times[frame]:.5g} s**")
        # Stack plots to stay readable on phones; comparisons use identical scales.
        st.plotly_chart(arrows_figure(fluids, trajectories, frame), width="stretch", config={"displayModeBar": False}, key="motion_arrows")
        st.caption("Arrows show horizontal velocity. Layers stay at fixed y positions; no vertical motion or particle trajectories are implied.")
        st.plotly_chart(profile_figure(fluids, trajectories, frame), width="stretch", config={"displayModeBar": False}, key="velocity_profile")
        errors = []
        for fluid, (y, velocity, _) in zip(fluids, trajectories):
            deviation = np.max(np.abs(velocity[frame] - fluid.U * y / fluid.H)) / fluid.U
            errors.append(f"{fluid.label}: {100*deviation:.2f}% from steady" + (" · settled" if deviation < .01 else ""))
        st.caption(" | ".join(errors))
        st.caption("‘Settled’ means the largest velocity difference from the steady profile is less than 1% of the moving plate speed, at the grid nodes.")
        if frame > 0:
            early = [f.label for f in fluids if f.nu * times[frame] / f.H**2 < .01]
            if early:
                st.info("Early-time approximation: the thin layer near the moving wall may not yet be well resolved for " + ", ".join(early) + ". The numerical accuracy target applies after νt/H² ≥ 0.01.")

    player_panel()


def intro():
    st.title("Momentum Diffusion Lab")
    st.markdown("### How does motion spread through a fluid?")
    st.write("Start a plate moving and the fluid next to it moves first. Viscous forces then transmit motion to neighboring layers. This lab explores why dynamic viscosity and kinematic viscosity answer different questions.")
    st.info("This is an AI-assisted ChemE project: the same specification will be implemented by two models, checked for accuracy, and compared after one correction round each. Comparison results are pending.")
    st.subheader("Start with a thin layer")
    st.write("The faster fluid above pulls a layer rightward; the slower fluid below pulls it leftward. If those forces differ, the layer accelerates. Equal opposing forces give zero acceleration, even if the layer is already moving.")
    st.plotly_chart(force_diagram(), width="stretch", config={"displayModeBar": False})
    st.write("Dynamic viscosity measures the coupling between neighboring layers. Density measures the inertia per unit volume. Their ratio, kinematic viscosity, sets how quickly that coupling changes the velocity.")
    st.latex(r"\nu = \frac{\mu}{\rho}")
    st.write("Viscosity resists relative motion and spreads momentum. At fixed density, a greater dynamic viscosity transmits motion faster while requiring more shear stress for a given velocity gradient.")
    st.markdown("**Try Predict and observe first, then Free exploration.**")
    st.caption("All cases use hypothetical fluids with constant properties. The model assumes laminar horizontal motion, no slip, no pressure gradient, and negligible plate-edge effects.")


def guided():
    st.title("Predict and observe")
    st.write("Compare a baseline hypothetical fluid with one deliberate change. Predict which reaches its own steady profile sooner.")
    selected = st.selectbox("Choose an experiment", list(EXPERIMENTS), key="experiment")
    experiment = EXPERIMENTS[selected]
    sync_player(("guided", selected))
    prediction = st.radio("The changed case will approach steady flow…", ["Faster", "Slower", "Same rate"], index=None, key=f"prediction_{selected}", horizontal=True)
    reveal_key = f"revealed_{selected}"
    if st.button("Reveal and run experiment", disabled=prediction is None):
        st.session_state[reveal_key] = True
    if not st.session_state.get(reveal_key, False):
        st.info("Make a prediction, then reveal the experiment. Think about ν = μ/ρ and the distance across the gap.")
        return
    if prediction == experiment["answer"]:
        st.success("Your prediction matches this model.")
    else:
        st.info(f"The model predicts: {experiment['answer'].lower()}. Compare the curves to explore why.")
    st.write(experiment["explanation"])
    show_player([BASELINE, experiment["fluid"]])


def explore():
    st.title("Free exploration")
    st.write("Change one property at a time to identify its effect. These combinations are bounded teaching cases, not a universal turbulence threshold.")
    a, b = st.columns(2)
    with a:
        rho = st.slider("Density ρ (kg/m³)", 500.0, 1500.0, 1000.0, 50.0, key="density")
        mu = st.slider("Dynamic viscosity μ (Pa·s)", 0.05, 1.0, 0.1, 0.01, format="%.2f", key="viscosity")
    with b:
        gap_mm = st.slider("Plate gap H (mm)", 2.0, 10.0, 5.0, 0.5, key="gap")
        speed = st.slider("Upper-plate speed U (m/s)", 0.005, 0.050, 0.020, 0.001, format="%.3f", key="speed")
    fluid = Fluid("Your hypothetical fluid", rho, mu, gap_mm/1000, speed)
    sync_player(("explore", rho, mu, gap_mm, speed))
    show_player([fluid])


def physics_notes():
    st.title("Why two viscosities?")
    st.write("The plate sets the boundary velocity. Shear forces between layers transmit momentum. Density determines how much inertia those forces must accelerate.")
    st.latex(r"\tau=\mu\frac{\partial u}{\partial y}, \qquad \nu=\frac{\mu}{\rho}")
    st.write("Dynamic viscosity μ relates shear stress to the velocity gradient. Kinematic viscosity ν determines how quickly velocity changes spread through the gap. Equal ν gives identical velocity evolution for identical geometry and plate motion, while greater μ requires greater shear stress for the same velocity gradient.")
    st.markdown("| Symbol | Meaning | SI unit |\n|---|---|---|\n| u | Horizontal fluid velocity | m/s |\n| y | Position across the gap | m |\n| t | Elapsed time | s |\n| U | Upper-plate speed | m/s |\n| H | Gap between plates | m |\n| ρ | Density | kg/m³ |\n| μ | Dynamic viscosity | Pa·s |\n| ν | Kinematic viscosity | m²/s |\n| τ | Shear stress | Pa |")
    st.write("The final velocity profile is linear because the shear stress becomes uniform across the gap: each thin layer has balanced horizontal forces.")
    st.latex(r"u_{\mathrm{steady}}(y)=U\frac{y}{H},\qquad t_{\mathrm{char}}=\frac{H^2}{\nu}")
    st.write("Doubling H gives four times the characteristic response time. This is a scaling time, not an exact settling time. The numerical lab measures the first integration step when every grid-node velocity is within 1% of U of the steady profile.")
    with st.expander("Derive the equation using Newton's second law"):
        st.plotly_chart(force_diagram(), width="stretch", config={"displayModeBar": False})
        st.write("A is the layer's face area (m²), and Δy is its thickness (m). Its mass is ρAΔy (kg). The opposite shear tractions on its faces give a force difference.")
        st.latex(r"\rho A\Delta y\frac{\partial u}{\partial t}=A[\tau(y+\Delta y)-\tau(y)]")
        st.write("Divide by AΔy and let the layer become very thin. With constant μ:")
        st.latex(r"\rho\frac{\partial u}{\partial t}=\frac{\partial\tau}{\partial y}=\mu\frac{\partial^2u}{\partial y^2}")
        st.latex(r"\boxed{\frac{\partial u}{\partial t}=\nu\frac{\partial^2u}{\partial y^2}}")
        st.write("Rate of velocity change = kinematic viscosity × curvature of the velocity profile. Curvature describes how the velocity gradient changes from one layer to the next.")
        st.write("A velocity gradient creates shear stress. A difference in shear stress creates net acceleration. A straight velocity profile can have nonzero stress and zero acceleration.")
        st.caption("Sign convention: τ = μ∂u/∂y is the rightward traction on a face with outward normal +y; the opposite face has opposite traction. Momentum-flux conventions in some textbooks use the opposite sign.")
    with st.expander("Assumptions and numerical method"):
        st.write("The fluid is incompressible and Newtonian, with constant μ and ρ. Motion is laminar and horizontal, with no slip, no imposed horizontal pressure gradient or body force, and negligible edge effects. Temperature change and turbulence are outside this model.")
        st.latex(r"u(0,t)=0,\quad u(H,t)=U\ (t\ge0),\quad u(y,0)=0\ (0\le y<H)")
        st.write("The upper boundary uses its post-start value at time zero; the interior starts at rest. An instantaneous start creates an idealized wall-stress singularity. Very early boundary layers may be thinner than a grid cell.")
        st.write("The numerical method divides the gap into 51 nodes and advances the force balance with explicit forward-time, centered-space differences. Steps are shortened to hit each requested output time exactly.")
        st.latex(r"u_j^{n+1}=u_j^n+Fo(u_{j+1}^n-2u_j^n+u_{j-1}^n),\qquad Fo=\frac{\nu\Delta t}{(\Delta y)^2}\le0.5")
        st.write("This app uses Fo = 0.4, a 25,000-step limit per solver call, cached trajectories, and 121 display frames. Settling is measured on integration steps, not just the display frames.")
        st.write("The analytical-series accuracy checks apply at νt/H² ≥ 0.01. The late-time estimate 0.421 H²/ν follows from the slowest decaying sine mode and differs from the measured numerical crossing.")


def applications():
    st.title("Connections to ChemE")
    st.subheader("A coaxial-cylinder viscometer")
    st.plotly_chart(cylinder_diagram(), width="stretch", config={"displayModeBar": False})
    st.write("A rotating outer cup drags the fluid around a stationary inner cylinder. Measuring the torque on the inner cylinder, together with known geometry and rotation rate, can reveal a Newtonian fluid's viscosity. Curvature changes the full equations. If H = Rₒ − Rᵢ is much smaller than Rᵢ, a small part of the gap resembles parallel plates. This diagram is an application, not a cylinder simulation.")
    st.subheader("The same diffusion pattern, different quantities")
    st.markdown("| Process | What spreads? | Diffusivity (m²/s) | Constant-property, 1D equation |\n|---|---|---|---|\n| Momentum | Horizontal motion | ν = μ/ρ | ∂u/∂t = ν ∂²u/∂y² |\n| Heat | Temperature differences | α = k/(ρ cₚ) | ∂T/∂t = α ∂²T/∂y² |\n| Mass | Concentration differences | D | ∂c/∂t = D ∂²c/∂y² |")
    st.write("Here T is temperature (K), c is concentration (e.g. mol/m³), k is thermal conductivity (W/(m·K)), cₚ is specific heat capacity (J/(kg·K)), α is thermal diffusivity, and D is mass diffusivity. The analogy assumes appropriate diffusion-only conditions, with constant properties and no added heat sources or reactions.")
    st.write("All three diffusivities have dimensions length²/time. Stokes are conventionally used for kinematic viscosity, but the unit is dimensionally valid for thermal or mass diffusivity too.")
    st.latex(r"1\ \mathrm{St}=1\ \mathrm{cm^2/s}=10^{-4}\ \mathrm{m^2/s}")


def evidence_page():
    st.title("The AI investigation")
    data = read_json("content/evidence.json")
    st.info(data.get("project_status", "Model comparison pending."))
    st.write("The two models should receive this same specification in fresh sessions. Preserve the original outputs, run one shared external validation suite, then provide one correction prompt per model using observed failures and violated requirements.")
    st.subheader("Critique and student decisions")
    critique = data.get("critique", {})
    st.caption(f"Supplied critique label: {critique.get('model_label', 'Pending')} · Interface/provider: {critique.get('provider_interface', 'Pending')}")
    decisions = data.get("student_decisions", [])
    if decisions:
        st.dataframe(decisions, hide_index=True, width="stretch")
    st.subheader("Model comparison")
    submissions = data.get("submissions", [])
    if submissions:
        st.dataframe(submissions, hide_index=True, width="stretch")
    criteria = ["Initial and boundary conditions", "Steady profile", "Reference accuracy", "Grid refinement", "Equal-ν histories", "Settling-time scaling", "Invalid inputs and workload", "Teaching clarity", "Controls and responsive layout"]
    matrix = data.get("evaluation_matrix", [{"Criterion": c, "Astra first": "Pending", "Astra corrected": "Pending", "Gemini first": "Pending", "Gemini corrected": "Pending"} for c in criteria])
    st.dataframe(matrix, hide_index=True, width="stretch")
    selection = data.get("selection", {})
    st.write(f"Selection: {selection.get('status', 'Pending')}. {selection.get('reason', '')}")
    st.subheader("Contributions and provenance")
    for name, contributions in data.get("contributions", {}).items():
        st.markdown(f"**{name.title()}**")
        for item in contributions:
            st.write("• " + item)
    st.caption("Do not infer the exact model label from the interface name. The available metadata identifies Codex as GPT-6-based; the specific displayed model label must be confirmed by the student.")
    st.subheader("Validation evidence")
    report = read_text("docs/verification.md")
    st.markdown(report if report else "Pending: validation results.")
    st.caption("Local checks for this submission are separate from the unchanged external suite used for the formal two-model comparison.")
    with st.expander("Implementation specification"):
        st.markdown(read_text("docs/specification.md"))
    with st.expander("Original critique, verbatim"):
        st.text(read_text("docs/critique-original.txt") or "Pending")
    with st.expander("Full evidence record"):
        st.json(data)


def presentation_page():
    st.title("Presentation and learning")
    data = read_json("content/presentation.json")
    url = data.get("youtube_url", "").strip()
    if url:
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.hostname in {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}:
            st.link_button("Watch the YouTube presentation", url)
            st.video(url)
        else:
            st.warning("Set a valid HTTPS YouTube URL in content/presentation.json.")
    else:
        st.info("Pending: the student's YouTube presentation. Upload as Public or Unlisted, then add its URL to the content file.")
    st.subheader("Verbatim YouTube auto-generated transcript")
    raw_path = ROOT / data.get("transcript_path", "content/transcript.txt")
    # Reading bytes avoids Python's newline conversion. Display plain text, not Markdown.
    transcript = raw_path.read_bytes().decode("utf-8") if raw_path.exists() else ""
    if transcript.strip():
        with st.expander("Read the complete transcript", expanded=True):
            st.html('<pre style="white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit"><span>' + html.escape(transcript) + '</span></pre>')
        st.download_button("Download the unchanged transcript", data=raw_path.read_bytes(), file_name="youtube-auto-transcript.txt", mime="text/plain")
    else:
        st.write("Pending: paste YouTube's complete auto-generated transcript into content/transcript.txt, preserving its wording, timestamps, and line breaks. Allow time for automatic captions to process.")
    errata = read_text(data.get("errata_path", "content/errata.md"))
    if errata.strip():
        with st.expander("Transcript errata / clarifications (separate)"):
            st.markdown(errata)
    st.subheader("Presentation outline")
    for item in data.get("outline", []):
        st.markdown(f"**{item['title']}**")
        st.write(item["prompt"])
    st.subheader("What I learned and want to learn next")
    st.markdown(read_text("content/reflections.md") or "Pending: student-written reflections.")


def references_page():
    st.title("References and project files")
    st.markdown(read_text("docs/references.md") or "Pending: reference list.")
    st.link_button("GitHub repository", "https://github.com/anujklkrn123/momentum-diffusion-lab")
    st.caption("This local implementation has not yet been uploaded or selected as the final production app.")


def main():
    st.sidebar.markdown("## Momentum\nDiffusion Lab")
    st.sidebar.caption("A beginner's experiment in viscosity")
    page = st.sidebar.radio("Explore the project", ["Start here", "Predict and observe", "Free exploration", "Physics explained", "ChemE connections", "AI process and evidence", "Presentation", "References"], key="page")
    if st.session_state.get("last_page") != page:
        reset_player()
        st.session_state.last_page = page
    st.sidebar.markdown("---")
    st.sidebar.caption("Hypothetical fluids · laminar model\nPython + Streamlit + NumPy + Plotly")
    pages = {"Start here": intro, "Predict and observe": guided, "Free exploration": explore,
             "Physics explained": physics_notes, "ChemE connections": applications,
             "AI process and evidence": evidence_page, "Presentation": presentation_page, "References": references_page}
    pages[page]()


if __name__ == "__main__":
    main()
