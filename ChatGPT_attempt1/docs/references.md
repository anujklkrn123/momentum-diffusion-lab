# References and attribution

The links below were checked on 2026-10-13. Equations in the supplied specification are implemented with the assumptions stated on the website. Diagrams are drawn for this project; no photograph, external illustration, real-fluid property table, or borrowed dataset is required.

## Physics and numerical methods

- [MIT OpenCourseWare, 3.185 Transport Phenomena in Materials Engineering, fluid-flow lecture notes, Fall 2003](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/a55c74dc21def01fc0054de35574d1e5_3_21.pdf), §§1.3–1.6: mechanical stress versus momentum flux, kinematic viscosity, transport analogies, steady and startup plane Couette flow, and startup stress singularity.
- [Gilbert Strang, The Heat Equation and Convection-Diffusion, MIT 18.086 reading, Spring 2006](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf), pp. 3–7: explicit finite differences, the stability ratio, refinement, and Fourier sine modes. The Couette coefficients and 0.421 settling estimate are derived for this project's boundary conditions in `docs/physics.md`, rather than quoted as a result from this reading.
- [MIT OpenCourseWare, 1-D Thermal Diffusion Equation and Solutions, 3.185, Fall 2003](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/927ddb6b3dfc423de5570a0070614129_handout_htrans.pdf): thermal diffusivity and relation to solute diffusion.
- [NIST Guide to the SI, Chapter 8, §8.8](https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-8): viscosity units and the stokes conversion. NIST recommends SI units; this project uses SI and includes stokes as a course unit comparison.
- [NIST-hosted Guide to Rheological Nomenclature, rotational rheometer section](https://www.nist.gov/document/b01014pdf): concentric-cylinder geometry and torque-based rheometry.

## Software and publication

- [Streamlit installation documentation](https://docs.streamlit.io/get-started/installation): local installation and running an app.
- [Streamlit Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): repository, branch, entry point, dependency installation, and Python selection.
- [NumPy documentation](https://numpy.org/doc/stable/): arrays and numerical operations.
- [Plotly Python documentation](https://plotly.com/python/): interactive graphs and annotations.
- [Python unittest documentation](https://docs.python.org/3/library/unittest.html): standard-library test runner.
- [YouTube automatic-caption documentation](https://support.google.com/youtube/answer/6373554?hl=en): automatic caption processing and possible transcription errors. Preserve the original automatic transcript for the assignment and place any corrections in a separate errata section.

## AI attribution

The preserved supplied critique is labeled **4 Argon High Effort** by the student; its provider/interface and generation date are pending confirmation. The current implementation was produced in Codex using GPT-6-family assistance, including parallel agents. The exact displayed model label is pending student confirmation. It is not automatically attributed to Astra.

The other implementation, exact model labels, shared external validation results, one correction round per model, and final selection remain pending. Fill `content/evidence.json` with the actual evidence. No conclusion about relative model quality is claimed before both candidates have been evaluated.
