# TinyML BLDC Predictive Maintenance Backend

Production-ready, Dockerized FastAPI backend for **Low-Cost Predictive Maintenance & Fault Detection for Rotating Machinery using TinyML** (BLDC motors), engineered for farmers and small workshop owners using low-end phones over weak networks.

## Key Features
- **3-Tier Health Model**: Maps telemetry into `healthy` (`0`), `needs_attention` (`1`), and `about_to_fail` (`2`) with localized (`en` / `hi`) short headlines and actionable hints.
- **Data Recording Pipeline (`/api/v1/recordings/*`)**:
  - `POST /api/v1/recordings/ingest` extracts **28 time- and frequency-domain features** (14 vibration + 14 current: mean, std, max, RMS, peak-to-peak, skewness, kurtosis, crest factor, spectral centroid, spectrum area, high-frequency bearing energy, 1x/2x/3x rotational harmonics), runs TinyML inference, stores raw windows + features in SQLite, and pushes live SSE updates.
  - `GET /api/v1/recordings/` paginated query with filtering.
  - `GET /api/v1/recordings/export/csv` exports training-ready CSV with all 28 features and labels.
- **Failure Injection Pipeline (`/api/v1/inject/`)**:
  - `POST /api/v1/inject/` injects controlled BLDC faults (`healthy`, `imbalance`, `misalignment`, `bearing_wear`, `demagnetization`, `combined`) with configurable `severity` (`0.05–1.0`), `snr_db` (AWGN), and `drift_offset` onto a synthesized waveform or user-supplied `base_vibration_g` / `base_current_a`, immediately updating the live dashboard and optionally persisting to SQLite.
- **Low-Bandwidth Dashboard Serving (`/api/v1/dashboard/{machine_id}`)**:
  - `GET /api/v1/dashboard/{machine_id}?language=en|hi` returns ultra-lightweight JSON.
  - `GET /api/v1/dashboard/{machine_id}/stream?language=en|hi` streams real-time SSE updates.
- **Serial Bridge CLI (`scripts/serial_bridge.py`)**:
  - Reads newline-delimited JSON from ESP32/Arduino over USB Serial and posts to `/ingest`, with `--simulate` for hardware-free testing.

## Project Structure

```text
backend/
├── app/
│   ├── main.py              # FastAPI app factory and ASGI entrypoint
│   ├── schemas.py           # Pydantic v2 contracts and 28-feature definitions
│   ├── api/
│   │   └── routers/         # Versioned HTTP & SSE routes (/api/v1/*)
│   ├── core/                # Settings, en/hi localization, and live SSE state
│   ├── db/                  # SQLite WAL persistence repository
│   └── ml/                  # Feature extraction, fault injection, inference, baselines, WDCNN
├── scripts/                 # CLI tools (serial_bridge.py, baseline_eval.py, demo_reviewer.py)
├── tests/                   # Automated pytest unit & integration suite
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Quick Start (Local)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open Swagger UI at `http://localhost:8000/docs`.

### Hardware-Free Serial Simulation

```bash
python scripts/serial_bridge.py --simulate --fault cycle --count 6 --interval 1.0
```

### Run Tests

```bash
pytest -v
```

## Research Baseline Suite

### 1. DUDU-BLDC 28-Feature Tabular Baseline (Leakage-Free `recording_id`-Grouped K-Fold CV)

Run against the synthetic BLDC benchmark generator (MLP or Random Forest):

```bash
python scripts/baseline_eval.py --model mlp
python scripts/baseline_eval.py --model rf
```

Run against an exported CSV from `GET /api/v1/recordings/export/csv`:

```bash
python scripts/baseline_eval.py --csv path/to/bldc_recordings_export.csv --splits 5
```

Or invoke via API:

```bash
curl -X POST http://localhost:8000/api/v1/baseline/evaluate \
  -H "Content-Type: application/json" \
  -d '{"source":"synthetic","n_recordings":60,"windows_per_recording":3,"n_splits":5,"model_type":"mlp"}'
```

### 2. WDCNN 5-Stage 1D-CNN Reference Model (PyTorch, 64x1 Kernel + BatchNorm)

Requires PyTorch (`pip install torch`):

```bash
python -m app.ml.wdcnn --epochs 25 --window-size 256 --snr-levels 22 10 5 0
```

### Benchmark Results (`seed=42`, `recording_id`-grouped split, `leakage_verified=True`)

| Benchmark Model | Protocol | Accuracy | Balanced Acc | Int8 / Noisy Accuracy | Avg Latency (ms/win) | Footprint (float32 / int8) | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DUDU-BLDC 28-Feature (MLP)** | 5-fold `StratifiedGroupKFold` (180 win / 60 rec) | 100.00% | 100.00% | 100.00% (int8) | 0.0068 ms | 6.30 KB / 1.91 KB | 1,614 |
| **DUDU-BLDC 28-Feature (RF)** | 5-fold `StratifiedGroupKFold` (180 win / 60 rec) | 100.00% | 100.00% | — | 0.0403 ms | 7.03 KB / 2.64 KB | 450 nodes |
| **WDCNN 5-Stage 1D-CNN (64x1)** | Grouped 80/20 split (288 train / 72 test, 25 ep) | 100.00% | 100.00% | 22dB: 100.00%, 10dB: 100.00%, 5dB: 100.00%, 0dB: 88.89% | 0.0571 ms | 149.65 KB / 37.41 KB | 38,310 |

## Reviewer Demo (< 2 Minutes)

Run the automated end-to-end demo script (starts Docker Compose, checks `/health` and `/docs`, streams 5 healthy windows via `serial_bridge.py --simulate`, checks localized Hindi dashboard output, injects `imbalance` `0.45` -> `needs_attention` and `bearing_wear` `0.95` -> `about_to_fail`, evaluates the 28-feature baseline, and exports the training CSV):

```bash
python scripts/demo_reviewer.py
```

If running against an already-started local Uvicorn server (`uvicorn app.main:app --port 8000`):

```bash
python scripts/demo_reviewer.py --skip-docker
```

Pass `--teardown` to automatically stop containers (`docker compose down`) after the demo.

## Docker

```bash
docker compose up --build
```
