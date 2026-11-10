# fairy-light-sens

A fairy light lamp controlled by voice. A small keyword spotting model listens for
commands such as "crystal on", "crystal off" and "crystal bright" directly on an Arduino
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

In VS Code, pick `.venv/bin/python` as the notebook kernel instead of starting
JupyterLab.

## Notebooks

Run them in order.

| Notebook | What it does | Time |
|---|---|---|
| `00_environment_check.ipynb` | Confirms TensorFlow, the audio front end and int8 conversion work. Ends with `Environment OK.` | under a minute |
| `01_prepare_data.ipynb` | Downloads Google Speech Commands (2.4 GB) and streams the MSWC archive (35 GB read, about 1 GB kept) to collect `crystal`, `bright`, `off` and unknown words. | 40 to 90 minutes, mostly the MSWC stream |
| `02_train_keywords.ipynb` | Trains two models, converts them to int8 TensorFlow Lite, reports accuracy, and writes `.tflite` and C array files to `models/<wake word>/`. | 20 to 30 minutes on a laptop CPU |

The data steps skip themselves once the data is on disk. The MSWC fetch can also be
run from a terminal:

```bash
uv run python -m fairy_kws.mswc
```

## Development

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
├── src/fairy_kws/        shared Python code: settings, features, datasets, models, export
├── tests/                pytest tests
├── data/                 datasets (not committed)
├── models/               training outputs, one folder per wake word (not committed)
├── firmware/             Arduino sketch (added in Phase 3 of the plan)
└── .github/workflows/    CI: lint, notebook check, tests
```

`src/fairy_kws/config.py` holds the audio settings, the wake word and the keyword list. The audio
settings mirror the firmware's `micro_features_micro_model_settings.h`; change them
in both places or not at all.
