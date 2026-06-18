# TriGuard-DF — DeepFake Detection

This repository provides TriGuard-DF: a Flask web API and a CLI for running a deepfake detection model on videos.

## Project overview

- Flask UI and API: `app.py` serves a web UI and endpoints for single and batch video inference.
- CLI inference: `main.py` runs inference from the command line for a single video.
- Model: `models/triguard_best.pt` (required) — a PyTorch checkpoint used by the inference code in `src/inference.py`.
- Code: core logic lives in `src/` (`architecture.py`, `inference.py`, `model_loader.py`, `preprocessing.py`).
- Uploads & logs: temporary uploads saved to `output/uploads/`, logs to `output/logs/`.

## Prerequisites

- OS: Windows (tested), Linux should work with appropriate Python and wheel availability.
- Python: 3.11 (recommended). The repository's `requirements.txt` pins packages that require Python 3.11 for compatible torch wheels.
- Disk: model file ~232 MB; ensure enough space.

## Recommended setup (Windows / PowerShell)

1. Install Python 3.11 if you don't have it. Example using the Microsoft package manager:

```powershell
winget install -e --id Python.Python.3.11
```

2.From the project root, create and activate a venv (we used `.venv-3.11`):

```powershell
py -3.11 -m venv .venv-3.11
& '.\.venv-3.11\Scripts\Activate.ps1'
```

3.Upgrade packaging tools and install requirements:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Notes:
-`torch==2.1.2` requires a matching Python minor version (3.11). If pip reports "No matching distribution found for torch==2.1.2", ensure you're using Python 3.11.
-The install may download large wheels (torch, opencv, mediapipe). Be patient.

## Running the Flask app (web UI + API)

1. Activate the venv (if not already):

```powershell
& '.\.venv-3.11\Scripts\Activate.ps1'
```

2.Launch the server:

```powershell
python app.py
```

3.The dev server listens on `http://127.0.0.1:5000`. Health endpoint:

```powershell
curl http://127.0.0.1:5000/health
```

- While the model loads, `/health` returns 202 with `{ "status": "loading" }`.
- Once loaded, `/health` returns `{ "status": "ready", "device": "cpu|cuda" }`.

## Dataset Link

https://drive.google.com/drive/folders/1WCR_N4n3zsyic9ZPUAJ3dM1jCOOdhY9D?usp=drive_link

```

Saurabh Kumar
