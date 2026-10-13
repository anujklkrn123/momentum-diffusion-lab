# momentum-diffusion-lab
# Momentum Diffusion Lab

An interactive Python/Streamlit teaching application addressing the core Chemical Engineering question:
> **“How does motion spread through a fluid, and why do we need two definitions of viscosity?”**

Designed for beginning Chemical Engineering students familiar with shear stress ($\tau = \mu\,\partial u/\partial y$) and velocity gradients, prior to studying differential momentum balances or partial differential equations.

Target repository: [https://github.com/anujklkrn123/momentum-diffusion-lab](https://github.com/anujklkrn123/momentum-diffusion-lab)

---

## Project Structure

- `physics.py` — Standalone numerical solver (`solve_couette_transient`) using explicit Forward-Time Centered-Space (FTCS) finite differences and analytical Fourier-series evaluator (`analytical_couette`), plus trajectory caching and integration-step settling-time calculation.
- `app.py` — Streamlit user interface containing Sections A–F, AI Evidence & Attribution, and Presentation & Verbatim Transcript display.
- `test_physics.py` — Standard `unittest` validation suite covering all 7 required physical, numerical, and input-validation checks.
- `requirements.txt` — Reproducible Python package dependencies (`streamlit`, `numpy`, `plotly`). No AI API keys or external accounts are required.
- `content/` — Editable local files for external comparison evidence and presentation materials:
  - `content/ai_evidence.json` — Critique decisions, model metadata, Attempt 1 & 2 comparison slots, and student reflection placeholders.
  - `content/presentation_metadata.json` — YouTube presentation URL and presentation outline.
  - `content/transcript.txt` — Verbatim YouTube auto-generated transcript (preserves timestamps and line breaks).
  - `content/transcript_errata.md` — Separate errata/clarifications file so the transcript remains unedited.

---

## Local Setup and Execution

### 1. Clone the repository and create a virtual environment
```bash
git clone [https://github.com/anujklkrn123/momentum-diffusion-lab.git](https://github.com/anujklkrn123/momentum-diffusion-lab.git)
cd momentum-diffusion-lab
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
