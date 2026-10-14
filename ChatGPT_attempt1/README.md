# Momentum Diffusion Lab

An interactive ChemE lesson about how a moving plate transmits momentum through a fluid. Predict what happens when density, dynamic viscosity, or plate gap changes, then explore velocity profiles and a thin-layer force balance.

The simulation uses Python, Streamlit, NumPy, and Plotly. The published app requires no AI account or API key. This is also an AI investigation: the student will compare two independently generated implementations using the same specification and one correction round each.

**Status:** this is the Codex implementation candidate. The second-model implementation, formal comparison, final selection, public deployment, video, and verbatim YouTube transcript are pending. Automated and manual verification are recorded separately in `docs/verification.md` when available. Do not treat a working app as completion of every class requirement.

## Run locally

Use Python 3.12. From this project directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`. Open the local address printed by Streamlit.

Run the tests:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Without a virtual environment, the equivalent commands are `python -m pip install -r requirements.txt`, `python -m streamlit run app.py`, and `python -m unittest discover -s tests -v`.

`requirements.txt` pins the three directly used packages. `requirements-lock.txt` records all resolved packages in the tested Python 3.12 environment; use `python -m pip install -r requirements-lock.txt` to reproduce those dependency versions. The app is also configured with a light theme in `.streamlit/config.toml`.

**Verification:** 29 automated checks passed. See `docs/verification.md` for numerical findings, browser checks, implementation self-check fixes, and pending external verification. No scored student correction round has been used.

## Publish on Streamlit Community Cloud

1. Preserve this entire first candidate before requesting a correction round. Keep the other model's submission separate. Record the exact displayed model labels and tools in `content/evidence.json`.
2. Add the selected app's files to [the existing GitHub repository](https://github.com/anujklkrn123/momentum-diffusion-lab). Keep `app.py`, `physics.py`, `requirements.txt`, the supporting modules, `content/`, and `docs/` together. Do not upload `.venv` or secrets.
3. Sign in to Streamlit Community Cloud and choose **Create app**. Select this repository, the branch containing the app, and main file `app.py`. Set Python **3.12** in Advanced settings. No secrets are needed. See the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
4. Open the resulting website while signed out and exercise predictions, playback, pause, reset, scrubbing, and parameter changes. Add the public URL to the GitHub README and the evidence record.
5. Publish the presentation on YouTube as public or unlisted, then add its URL and verbatim automatic transcript as described below.

## Editable project evidence

| File | What to put there |
|---|---|
| `content/evidence.json` | Exact model labels, dates, tool conditions, decisions, first/corrected results, selection |
| `content/presentation.json` | YouTube URL, presentation status, and outline |
| `content/transcript.txt` | Complete, unedited YouTube auto-generated transcript; initially empty |
| `content/errata.md` | Corrections to transcript statements, kept separate from the original |
| `content/reflections.md` | Student's own learning and future questions; initially pending |
| `docs/specification.md` | Supplied revised specification, copied verbatim |
| `docs/critique-original.txt` | Supplied model critique, copied verbatim |
| `docs/comparison-protocol.md` | Fair comparison and correction-round procedure |

Changing these content files does not require editing the physics solver. Use the word **Pending** when evidence is unavailable. The supplied critique label is **4 Argon High Effort**; its provider/interface has not been confirmed. This candidate was produced with Codex and parallel agent assistance; its exact displayed model label must be recorded by the student. It must not automatically be labeled Astra.

## Finish the presentation

Record after physics and interface verification. Show predictions and the demo, explain the critique decisions and model comparison, and discuss what you learned in your own words. Use `docs/presentation-outline.md` as a checklist.

Upload early enough to allow automatic caption processing. YouTube warns that automatic captions may not be ready immediately and may contain errors. [YouTube automatic-caption documentation](https://support.google.com/youtube/answer/6373554?hl=en)

Copy the complete YouTube **auto-generated** transcript, including supplied timestamps and line breaks, into `content/transcript.txt`. Do not correct that file or replace it with an AI-written script. Put clarifications in `content/errata.md`. Set `youtube_url` in `content/presentation.json`. Verify that both the link and the full transcript are visible on the public app.

## Physics and validation

The fluid is initially stationary inside the gap; the upper boundary takes its post-start velocity at time zero. The steady profile is linear. The numerical solver uses explicit finite differences with the diffusion stability condition and compares with a sine-series reference at resolved positive times. The app's internal tests and the later unchanged external comparison suite have different roles; agreement with this candidate's own reference function is not independent verification.

See [physics notes](docs/physics.md), [references](docs/references.md), and the preserved [implementation specification](docs/specification.md). The app uses hypothetical fluids and conservative teaching inputs. It does not measure real materials or model turbulence, non-Newtonian behavior, slip, or cylinder flow.
