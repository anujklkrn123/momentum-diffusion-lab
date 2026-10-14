# Physics notes

## Forces first

A thin fluid layer has mass \(\rho A\Delta y\). The fluid above and below exerts shear forces on its horizontal faces. Their difference accelerates the layer horizontally:

\[
\rho A\Delta y\frac{\partial u}{\partial t}
=A[\tau(y+\Delta y)-\tau(y)].
\]

Equal opposing forces give zero acceleration. The layer can still be moving. The vertical coordinate identifies a position in the gap; the velocity arrows point horizontally.

For constant properties, dividing by layer thickness and taking a thin-layer limit gives

\[
\rho\frac{\partial u}{\partial t}=\frac{\partial\tau}{\partial y},
\qquad\tau=\mu\frac{\partial u}{\partial y},
\qquad\frac{\partial u}{\partial t}=\nu\frac{\partial^2u}{\partial y^2},
\quad\nu=\frac{\mu}{\rho}.
\]

Here \(\tau\) is mechanical traction on the face with outward normal \(+y\); a momentum-flux convention may have the opposite sign. MIT's [fluid-flow notes, §§1.3–1.6](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/a55c74dc21def01fc0054de35574d1e5_3_21.pdf) provide background on this distinction, momentum diffusivity, and startup Couette flow.

## What the experiment isolates

For a specified gap and plate speed, density and viscosity enter the velocity evolution through \(\nu\). Greater viscosity at fixed density transmits motion faster; greater density at fixed viscosity makes acceleration slower. If both change proportionally, velocity histories are identical. Dynamic viscosity still determines the shear stress associated with the same gradient. Stress readouts were omitted to keep attention on velocity.

The conditions are \(u(0,t)=0\), \(u(H,t)=U\), and an initially stationary interior. The steady profile is \(u=Uy/H\). At time zero, the upper boundary uses the imposed post-start speed; an instantaneous start is an idealization with singular wall stress. Very early thin boundary layers may be unresolved on the grid.

## Time and the analytical reference

Let \(z=y/H\), \(s=\nu t/H^2\), and \(v=u/U\). Subtract the steady profile: \(w=v-z\). The remaining problem has zero values at both ends and initial interior value \(-z\). Its sine coefficients follow directly from integration:

\[
b_n=2\int_0^1(-z)\sin(n\pi z)\,dz
=\frac{2(-1)^n}{n\pi}.
\]

Thus, for positive time,

\[
\frac{u}{U}=z+\sum_{n=1}^{\infty}
\frac{2(-1)^n}{n\pi}e^{-n^2\pi^2s}\sin(n\pi z).
\]

This derives the supplied reference for this project's initial and boundary conditions. The separation-of-variables method and exponential sine modes are described in [Gilbert Strang's heat-equation reading, pp. 5–7](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf).

The characteristic time is \(H^2/\nu\). The app measures settling at the first integration step where the maximum deviation from steady flow is below \(0.01U\). This is a numerical measurement at the specified grid, not an exact continuum event time.

For late time, the first sine mode dominates. Solving \((2/\pi)e^{-\pi^2s}=0.01\) gives

\[
s\approx\frac{\ln(200/\pi)}{\pi^2}\approx0.421.
\]

That is the source of the displayed analytical late-time estimate. Doubling \(\nu\) halves the response time; doubling \(H\) multiplies it by four.

## Numerical method

Forward time and centered space give

\[
u_j^{n+1}=u_j^n+Fo(u_{j+1}^n-2u_j^n+u_{j-1}^n),
\quad Fo=\frac{\nu\Delta t}{\Delta y^2}\le\frac12.
\]

The app uses 51 nodes and \(Fo=0.4\), shortens steps at requested output times, and limits workload. It measures settling during integration while saving about 121 frames for playback. Strang's [reading, pp. 3–5](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf) explains explicit diffusion differences, stability, and refinement.

## Cylinder connection and other diffusivities

A rotating outer cup shears fluid around a stationary inner cylinder. Torque measurements relate to viscosity and known instrument geometry; this is an application diagram, not a second simulation. See [NIST-hosted Guide to Rheological Nomenclature, rotational rheometers](https://www.nist.gov/document/b01014pdf). In a narrow annular gap, curvature is small locally; a full cylinder model still needs cylindrical equations.

Thermal diffusivity \(\alpha=k/(\rho c_p)\) uses thermal conductivity \(k\) and specific heat capacity \(c_p\). Mass diffusivity \(D\) governs concentration smoothing. Under suitable constant-property, one-dimensional, source-free assumptions, velocity, temperature, and concentration obey analogous diffusion equations. See [MIT's thermal diffusion handout](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/927ddb6b3dfc423de5570a0070614129_handout_htrans.pdf).

All three diffusivities have units \(\mathrm{m^2/s}\). Stokes are dimensionally compatible but conventionally associated with kinematic viscosity; SI units remain the default here. [NIST SI Guide §8.8](https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-8) gives \(1\ \mathrm{St}=10^{-4}\ \mathrm{m^2/s}\).
