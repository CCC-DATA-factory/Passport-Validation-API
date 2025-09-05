# Passport Validation Service — README

**Short description**
A compact, production-oriented passport image validation API built with PassportEye + Tesseract and packaged with Docker. Includes a stress test runner (`stress_test.py`) and a quick log analyzer (`analyze_logs.py`) to measure latency, throughput and error rates.

---

## Quick start

1. Copy the repository and ensure Docker and Docker Compose are installed.
2. Place passport images for testing into `passports/` (50–200 images recommended).
3. Create a `.env` file (see *Environment variables* below).
4. Build & run the service:

```bash
# build image from Dockerfile under passport_validation_api
docker compose build
# start service in background
docker compose up -d
# view logs
docker compose logs -f passport-api
```

Health-checks are available at `http://<host>:8000/health` once the container is up.

---

## What the `docker-compose.yml` does

* Builds the image from `./passport_validation_api` (reads that folder's `Dockerfile`).
* Runs a container named `passport_min` exposing port `8000`.
* Loads environment variables from `.env` and the `environment:` block.
* Mounts two volumes:

  * `./uploads` → `/app/uploads` (read/write for incoming files)
  * `./tests` → `/app/tests` (read-only; useful for CI/test fixtures)
* Health check: `curl -f http://localhost:8000/health || exit 1`.

### Key `docker-compose` envs (examples)

```
APP_MODULE=passport_api.main:app
WORKERS=2
TESSERACT_CMD=/usr/bin/tesseract
```

Adjust `WORKERS` to match CPU and whether inference is CPU-bound.

---

## API endpoints (summary)

* `POST /validate_passport` — main endpoint.

  * Accepts multipart form-data with **two** file fields: `passport1` and `passport2`.
  * Returns JSON with detection/ocr results (faces, MRZ text, validation flags, confidence scores).

* `GET /health` — simple health check returning 200 when service is ready.

> Tip: Test locally with `curl -F "passport1=@p1.jpg" -F "passport2=@p2.jpg" http://localhost:8000/validate_passport`.

---

## Stress testing (stress\_test.py)

Purpose: realistically simulate client traffic patterns and measure service behavior.

**Place test images:** `passports/` with many passport images.

**Run a default Poisson test** against the local service:

```bash
python3 stress_test.py --mode poisson --total 1800 --duration 28800 --concurrency 100
```

**Modes supported:** `uniform`, `poisson`, `ramp`, `spike`, `burst`, `constant`.

**What it does:**

* Sends requests to `API_URL` (default `http://localhost:8000/validate_passport`) uploading two images per request.
* Controls concurrency and logs each request into `logs/` as per-request JSON files.
* Appends a summary row to `logs/summary.csv` with p50/p95/p99 and counts.

**Important**: Run the stress client from a separate machine to capture realistic network and server behavior (unless intentionally measuring local contention).

---

## Log analysis (analyze\_logs.py)

Run after a stress test to get a concise terminal report:

```bash
python3 analyze_logs.py
```

It scans `logs/req_*.json` and prints totals, error counts, status code distribution and latency stats (avg, median, p95, p99, min, max).

---

## What to measure & why (short)

* **Latency** (p50, p95, p99) — user experience and tail latency.
* **Throughput** (req/sec) — capacity of the service.
* **Error rate** — indicates overload or bugs.
* **Server resources** — CPU, memory, GPU (if used).
* **Saturation point** — where latency/errors grow non-linearly.

Monitor metrics with Prometheus + Grafana or simple sampling (`vmstat`, `top`, `nvidia-smi`).

---

## Production tips (short)

* Run FastAPI under `gunicorn` + `uvicorn.workers.UvicornWorker` and tune `--workers`.
* Place an Nginx reverse proxy for buffering and TLS termination.
* Separate heavy model inference into a dedicated service (batch or GPU-backed) to avoid blocking API workers.
* Expose Prometheus metrics for request durations & counts and build Grafana dashboards.

Example gunicorn command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app:app --workers 4 --bind 0.0.0.0:8000
```

---

## Troubleshooting

* `tesseract not found` in container: set `TESSERACT_CMD` to the tesseract binary path inside the image or install Tesseract in the Dockerfile.
* `Out of memory` under high load: reduce concurrency, add more workers/memory, or move inference to a GPU service.
* `High tail latency`: investigate blocking I/O, model hot paths, and consider batching or async inference.

---

## Files in this repo

* `passport_validation_api/` — application source and Dockerfile.
* `docker-compose.yml` — service definition.
* `stress_test.py` — stress test runner.
* `passports/` — (not checked in) images for stress testing.
* `uploads/` — runtime uploads (mounted volume).



