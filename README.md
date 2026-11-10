# fairy-light-sens

A fairy light lamp controlled by voice. A small keyword spotting model listens for
commands such as "lamp on", "lamp off" and "lamp bright" directly on an Arduino
Nano 33 BLE Sense, using TensorFlow Lite for Microcontrollers.

The full roadmap, dataset notes and open decisions are in [docs/PLAN.md](docs/PLAN.md).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Git. uv installs Python 3.12 and
every dependency into a local `.venv`.

```bash
uv sync
```

Check that everything works (runs the tests, about 5 seconds):

```bash
uv run pytest
```

Start JupyterLab:

```bash
uv run jupyter lab
```

Then open `notebooks/00_environment_check.ipynb` and run all cells. It ends with
`Environment OK.` when TensorFlow, the audio front end and the int8 model
conversion all work.

In VS Code, pick `.venv/bin/python` as the notebook kernel instead of starting
JupyterLab.

### Adding a dependency

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock`; commit both.

### Git hooks (once per clone)

```bash
uv run pre-commit install
```

Each commit then strips notebook outputs and runs the linter and formatter. Notebook
outputs are kept out of Git so diffs stay readable; CI fails if a notebook is
committed with outputs.

## Layout

```
├── docs/PLAN.md          project plan
├── notebooks/            Jupyter notebooks, numbered in the order they are used
├── src/fairy_kws/        shared Python code (settings, feature extraction)
├── tests/                pytest tests
├── data/                 datasets (not committed)
├── models/               training outputs (not committed)
├── firmware/             Arduino sketch (added in Phase 3 of the plan)
└── .github/workflows/    CI: lint, notebook check, tests
```

`src/fairy_kws/config.py` holds the audio settings and the keyword list. The audio
settings mirror the firmware's `micro_features_micro_model_settings.h`; change them
in both places or not at all.
