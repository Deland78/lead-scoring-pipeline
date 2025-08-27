# End-to-End Lead Scoring: API (FastAPI) + UI (Flask)

This repository contains a complete demo of a lead scoring pipeline with:
- A production-style API (FastAPI) for real-time predictions
- A simple web UI (Flask) to submit leads and visualize results

No database is required for the demo. Both services use pre-trained model artifacts.

## Components
- api-lead-scoring (FastAPI)
  - Endpoints: `/` (info), `/v2/health`, `/v2/predict`, `/v2/models/info`, docs at `/v2/docs`
  - Port: 5000 (default). In local dev we commonly use 5051 to avoid conflicts.
  - Models: `api-lead-scoring/models/model.joblib`, `api-lead-scoring/models/preprocessor.joblib`
- app-lead-scoring (Flask UI)
  - Routes: `/` (form + dashboard), `/health`
  - Port: 5000 (default). In App Preview we map to 3000 for convenience.
  - Models: by default searched in `./`, `../models`, `/app/models` (see MODEL_DIR below)

## Quick Start (no Docker)

### Start the API
```bash
cd api-lead-scoring
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# start on a free port (e.g. 5051)
uvicorn main:app --host 0.0.0.0 --port 5051 &
```
- Health: `curl http://127.0.0.1:5051/v2/health`
- Predict:
```bash
curl -X POST http://127.0.0.1:5051/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TotalVisits": 5,
    "Page Views Per Visit": 3.2,
    "Total Time Spent on Website": 1850,
    "Lead Origin": "API",
    "Lead Source": "Google",
    "Last Activity": "Email Opened",
    "What is your current occupation": "Working Professional"
  }'
```

### Start the UI (on 3000 for preview)
```bash
cd app-lead-scoring
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# optional: provide MODEL_DIR if artifacts are stored elsewhere
# export MODEL_DIR=/app/models
# run UI on 3000
gunicorn -b 0.0.0.0:3000 app:app &
```
Then open http://127.0.0.1:3000

## Docker (optional)

Docker is not required for local runs here. If you enable Docker, the API includes a working `Dockerfile` and `docker-compose.yml`. One fix applied: the Docker image now installs `curl` for the container `HEALTHCHECK` to work.

Build and run the API with Docker:
```bash
cd api-lead-scoring
docker compose up --build -d
# API available on http://localhost:5000
```

For the Flask UI, ensure model artifacts are available. The app now searches common locations (`./`, `../models`, `/app/models`) or set `MODEL_DIR` to the folder with the artifacts.

## Notes / Changelog
- Updated root docs to reflect FastAPI backend + Flask UI (removed old DB instructions)
- Fixed HTML template issues in the Flask UI (missing </select> tags)
- Improved Flask model loading (supports MODEL_DIR and common search paths)
- Upgraded FastAPI models to Pydantic v2 style config, removed warnings
- API Dockerfile: added `curl` for HEALTHCHECK
- Added `api-lead-scoring/start.sh` helper script