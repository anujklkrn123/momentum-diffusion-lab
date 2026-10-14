# Momentum Diffusion Lab — Revised Implementation Specification

Build a complete, runnable Python/Streamlit project from this specification. Produce all required files, including tests and deployment instructions. Do not invent model-comparison results, student reflections, references, or presentation content that has not been supplied.

## 1. Purpose and audience

The project asks:

**“How does motion spread through a fluid, and why do we need two definitions of viscosity?”**

Teach beginning ChemE students who understand shear stress and velocity gradients but have not studied differential momentum balances or partial differential equations.

This is also an AI project comparing two independently generated implementations. The final documentation must distinguish student decisions, AI contributions, and independently verified findings.

Use Python, Streamlit, NumPy, and Plotly. No AI API calls, API keys, or visitor accounts are needed.

Target repository: https://github.com/anujklkrn123/momentum-diffusion-lab

The repository currently contains only a README. Deployment to Streamlit Community Cloud will follow implementation.

## 2. Physical model

Simulate startup plane Couette flow:

- Two horizontal parallel plates have a gap \(H\).
- The lower plate at \(y=0\) is stationary.
- The upper plate at \(y=H\) starts moving rightward at constant speed \(U\) at time zero.
- The fluid interior is initially stationary.
- Fluid motion is horizontal; vertical position identifies the layer.

Assume an incompressible Newtonian fluid, constant density and viscosity, laminar unidirectional motion, no slip, no imposed horizontal pressure gradient or body force, and negligible edge effects.

Use:

\[
\tau=\mu\frac{\partial u}{\partial y},\qquad
\nu=\frac{\mu}{\rho},\qquad
\frac{\partial u}{\partial t}
=\nu\frac{\partial^2u}{\partial y^2}.
\]

Define:

| Symbol | Meaning | SI unit |
|---|---|---|
| \(u\) | Horizontal fluid velocity | m/s |
| \(y\) | Position across the gap | m |
| \(t\) | Elapsed physical time | s |
| \(U\) | Upper-plate speed | m/s |
| \(H\) | Plate gap | m |
| \(\rho\) | Density | kg/m³ |
| \(\mu\) | Dynamic viscosity | Pa·s |
| \(\nu\) | Kinematic viscosity | m²/s |
| \(\tau\) | Shear stress | Pa |

Explain that dynamic viscosity relates shear stress to velocity gradient, while kinematic viscosity determines the rate of momentum diffusion.

Here, positive \(\tau=\mu\,\partial u/\partial y\) denotes the rightward traction on a fluid element’s face with outward normal in the positive \(y\) direction; the opposite face has the opposite traction sign.

At steady state:

\[
u_{\mathrm{steady}}(y)=U\frac{y}{H}.
\]

**Do not add transient or steady-state shear-stress readouts or stress graphs.** The student rejected them to avoid extra complexity. Retain the stress equation and a brief explanation: equal kinematic viscosity gives identical velocity evolution under identical geometry and plate motion, but greater dynamic viscosity requires greater shear stress for the same velocity gradient.

## 3. Website experience

Create a clear, responsive interface with these sections.

### A. Start here

Introduce the physics question and AI investigation.

Before showing the PDE, explain that neighboring layers exert shear forces on a thin fluid layer. Unequal forces accelerate it; equal opposing forces mean zero acceleration, not necessarily zero velocity.

Explain that viscosity resists relative motion while transmitting momentum between layers.

### B. Predict and observe

Use the baseline hypothetical fluid:

- \(\rho=1000\) kg/m³
- \(\mu=0.1\) Pa·s
- \(H=0.005\) m
- \(U=0.02\) m/s

Provide four experiments comparing the baseline with:

1. Double density, other inputs fixed: momentum diffusion is slower.
2. Double dynamic viscosity, other inputs fixed: momentum diffusion is faster.
3. Double density and dynamic viscosity together: velocity evolution is unchanged.
4. Double gap, other inputs fixed: the characteristic diffusion time becomes four times longer.

Ask viewers to predict before revealing the explanation. A wrong prediction should receive a clear explanation without judgment.

Reuse the same visualization component for guided experiments and free exploration.

### C. Visualizations and controls

Show:

- Horizontal velocity arrows at fixed vertical fluid-layer positions.
- A velocity-profile plot with velocity on the horizontal axis and position across the gap on the vertical axis.
- A continuously visible dashed steady-state profile.
- Play/pause, reset, and an elapsed-time slider labeled in seconds.
- Kinematic viscosity and characteristic diffusion time.

Arrows represent velocity, not vertical fluid movement or particle trajectories.

Both fluids in a comparison must share the same elapsed physical time, playback rate, and axis scales. Use a common vertical extent when comparing different gaps.

Label any accelerated playback, and show the physical-time increment. Reset time and pause when inputs or experiments change.

Define:

\[
t_{\mathrm{char}}=\frac{H^2}{\nu}.
\]

“Settled” means:

\[
\frac{\max_y|u-u_{\mathrm{steady}}|}{U}<0.01.
\]

Distinguish characteristic time from measured numerical settling time. The analytical late-time estimate is approximately \(0.421H^2/\nu\); label it an estimate, not the numerical measurement.

### D. Free exploration

Use hypothetical fluids with bounded inputs:

| Input | Minimum | Maximum |
|---|---:|---:|
| Density | 500 kg/m³ | 1500 kg/m³ |
| Dynamic viscosity | 0.05 Pa·s | 1 Pa·s |
| Gap | 0.002 m | 0.010 m |
| Plate speed | 0.005 m/s | 0.050 m/s |

These bounds keep \(UH/\nu\le15\). The fixed guided experiments are also conservative.

Describe these as teaching cases, not a universal laminar-flow threshold. Do not add a turbulence simulator or require students to understand Reynolds number before using the app.

### E. Optional derivation

Use a labeled thin-layer force diagram and explain:

\[
\rho A\Delta y\frac{\partial u}{\partial t}
=A[\tau(y+\Delta y)-\tau(y)].
\]

Then obtain:

\[
\rho\frac{\partial u}{\partial t}
=\frac{\partial\tau}{\partial y}
=\mu\frac{\partial^2u}{\partial y^2}.
\]

Translate the final equation into words. Explain that a velocity gradient creates shear stress, while a difference in shear stress creates net acceleration.

Keep the calculus in an expandable section.

### F. Applications and transport analogy

Include a simple, original top-down coaxial-cylinder diagram:

- Stationary inner cylinder and rotating outer cup.
- Fluid in the annular gap.
- Clearly labeled radii and rotation direction.
- Equal geometric axis scaling.

Explain viscosity measurement through torque, and that a narrow gap permits a local parallel-plate approximation. Curvature changes the full equations. Do not simulate cylinders.

Briefly compare:

- Momentum diffusivity: \(\nu=\mu/\rho\)
- Thermal diffusivity: \(\alpha=k/(\rho c_p)\)
- Mass diffusivity: \(D\)

Define the additional symbols. Under appropriate constant-property, one-dimensional diffusion assumptions, all three lead to analogous diffusion equations.

Explain that all have dimensions of length²/time. Therefore expressing any of them in stokes is dimensionally valid, although stokes are conventionally associated with kinematic viscosity:

\[
1\ \mathrm{St}=1\ \mathrm{cm^2/s}=10^{-4}\ \mathrm{m^2/s}.
\]

No additional heat or mass simulations.

## 4. Solver and shared interface

Keep computation independent of Streamlit in `physics.py`.

Expose exactly these functions:

```python
def solve_couette_transient(
    rho, mu, H, U, times, n_nodes=51, fourier_number=0.4
):
    """Return (y, velocity)."""

def analytical_couette(
    nu, H, U, y, times, terms=200
):
    """Return reference velocity array."""
```

Contract:

- Inputs use SI units.
- `times` is a nonempty, one-dimensional array of finite, nonnegative, strictly increasing times.
- `y` returned by the numerical solver includes both plates, shape `(n_nodes,)`.
- Velocity arrays have shape `(len(times), len(y))`.
- At time zero, the interior and lower plate are stationary; the upper boundary uses its imposed post-start speed \(U\).
- Reject invalid physical inputs, invalid grid sizes, and Fourier numbers outside \(0<Fo\le0.5\) with informative `ValueError` messages.
- The analytical function accepts positions within the closed interval \([0,H]\).

Use explicit forward-time, centered-space finite differences:

\[
u_j^{n+1}=u_j^n+
Fo(u_{j+1}^n-2u_j^n+u_{j-1}^n).
\]

Maintain:

\[
Fo=\frac{\nu\Delta t}{(\Delta y)^2}\le0.5.
\]

Use 51 nodes and \(Fo=0.4\) for the app. Shorten steps as needed to reach requested output times exactly.

For one fluid, use a displayed duration \(0.6H^2/\nu\). For comparisons, use \(0.6\max(H_1^2/\nu_1,H_2^2/\nu_2)\) as the shared physical duration.

Cache trajectories and store approximately 121 display frames. Playback must not recompute the entire simulation per frame. Compute settling time from integration steps rather than only these sparse display frames.

Cap each solver call at 25,000 integration steps and fail clearly if exceeded. Never silently violate stability or truncate the requested duration.

Use the analytical reference for positive times:

\[
u(y,t)=U\frac{y}{H}
-\frac{2U}{\pi}\sum_{n=1}^{200}
\frac{(-1)^{n+1}}{n}
e^{-n^2\pi^2\nu t/H^2}
\sin\left(\frac{n\pi y}{H}\right).
\]

Handle time zero explicitly. Explain that an instantaneous plate start creates an idealized wall-stress singularity and unresolved early boundary layers.

## 5. Validation

Supply runnable tests using Python’s standard `unittest` framework. Test:

1. Initial interior velocity and both boundary velocities.
2. Convergence toward the steady linear profile.
3. Maximum analytical-reference error below 1% of \(U\) at dimensionless times \(0.01,\ 0.05,\ 0.2,\ 0.6\).
4. Reduced error with 26, 51, and 101 nodes at fixed \(Fo=0.4\) and dimensionless time 0.05.
5. Identical velocity histories when density and viscosity change proportionally.
6. Approximately half the settling time when \(\nu\) doubles, and four times when \(H\) doubles, within 2%.
7. Rejection of invalid inputs and excessive computational workloads.

The formal comparison will additionally use one external, unchanged validation suite and reference evaluator against both submissions. Do not call agreement with your own analytical function independent verification.

Manually verify prediction reveal, play/pause, reset, time scrubbing, input changes, shared comparison time, readable labels, and narrow-screen layout. Report which checks actually ran.

## 6. AI evidence and attribution

Provide visible website sections for:

- Critique decisions.
- First and corrected implementation results for both models.
- Exact model labels, dates, interfaces, and tool-access conditions.
- Student contributions and AI contributions.
- References and future learning questions.

Use “Pending” for unavailable evidence.

Confirmed student decisions:

- Omit both transient and steady-state stress readouts to avoid extra complexity.
- Restrict exploration to conservative teaching cases.
- Keep heat and mass transfer as brief comparisons.
- Use one correction round per implementation.

The critique model label supplied by the student is “4 Argon High Effort.” Its provider/interface remains to be confirmed; do not silently relabel it.

Preserve each model’s original and corrected submissions separately. Do not claim that the final implementation has been selected before comparison.

Correction prompts must contain observed test failures, usability problems, and violated specification requirements. Do not give one model a code patch or substantially more diagnostic help than the other.

## 7. Presentation and publication

Include a presentation section driven by editable local content files:

- YouTube URL.
- Complete, unedited YouTube auto-generated transcript, displayed on the website with supplied timestamps and line breaks preserved.
- Separate errata, if needed.

Until supplied, show honest pending messages. Do not generate a substitute transcript.

Include a presentation outline covering the physics question, predictions, demo, AI critique decisions, comparison results, learning, and future questions. Leave personal reflections for the student to write.

Provide:

- `app.py`, `physics.py`, and necessary supporting files.
- A reproducible dependency list.
- Tests.
- README with local run and Streamlit deployment instructions.
- Editable evidence and presentation content.
- References for physics, numerical methods, and software; do not invent citations.
- A clear list of completed checks and remaining tasks.

After deployment, completion requires testing the website while signed out, exercising the demo, opening the public or unlisted YouTube video, and confirming the verbatim transcript is visible.

Implement the working project now, while leaving genuine external evidence and presentation inputs pending.