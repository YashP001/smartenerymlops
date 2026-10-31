# Docker Containerization Guide (Member 4)

This guide documents the containerization and local deployment workflow for the **AI-Assisted Smart Home Energy Advisor** inference service.

---

## 1. Prerequisites

Before running the container, ensure:
1. **Docker Desktop** is installed and running on your system.
   - For Windows, Docker Desktop should be configured to use the **WSL2** (Windows Subsystem for Linux 2) backend.
   - Verify Docker installation from PowerShell:
     ```powershell
     docker --version
     docker compose version
     ```
2. Port **8000** is available on your local host (not occupied by another process).

---

## 2. Image Architecture & Design

- **Base Image:** `python:3.13-slim` (matches the Python 3.13 training environment while minimizing container image size).
- **Working Directory:** `/app`
- **Dependencies:** Specified in `requirements-inference.txt` (only runtime inference dependencies: FastAPI, Uvicorn, scikit-learn, joblib, numpy, pandas, pydantic, mlflow).
- **Bundled Model Artifact:** `models/gradient_boosting.pkl` (137 KB) is copied into the image.
- **Security & Hygiene:** `.dockerignore` strictly excludes virtual environments (`.venv`), DVC credentials (`.dvc/config.local`), raw datasets (`data/`), host tracking databases (`mlflow.db`, `mlruns/`), and W&B files.

---

## 3. How the MLflow Registry Fallback Works Inside Docker

The containerized service relies on the registry-first loading pipeline defined in `src/predict.py`:
1. On container startup, the FastAPI `lifespan` handler attempts to query the MLflow Model Registry for `smart-home-energy-advisor:1`.
2. Because the container is completely isolated from the host's Windows filesystem and local `mlflow.db`, registry query raises a non-fatal notice:
   ```text
   Notice: MLflow registry loading unavailable (...). Falling back to local artifact.
   Loaded model from local fallback: '/app/models/gradient_boosting.pkl'
   ```
3. The service immediately loads the bundled `models/gradient_boosting.pkl` artifact via `joblib`, verifies that it accepts the 6 canonical features, and caches it in `app.state.model`.
4. The service begins serving inference requests immediately without requiring any connection to Google Drive, the host MLflow server, or external cloud networks.

---

## 4. Step-by-Step Container Execution

### Step 1: Build the Docker Image
From the repository root directory in PowerShell:
```powershell
docker build -t smart-home-energy-advisor .
```

### Step 2: Run the Docker Container
Start the container with port forwarding from host port 8000 to container port 8000:
```powershell
docker run --rm -p 8000:8000 --name smart-home-advisor-app smart-home-energy-advisor
```
*Flags explained:*
- `--rm`: Automatically removes the container when it is stopped.
- `-p 8000:8000`: Forwards host port `8000` to container port `8000`.
- `--name`: Assigns a human-readable name to the running container.

---

## 5. Verification and Testing

### 1. Health Check (`GET /health`)
From another PowerShell terminal:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
*Or using curl:*
```bash
curl http://127.0.0.1:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Next-Hour Prediction (`POST /predict`)
Send a valid 6-feature payload via PowerShell:
```powershell
$body = @{
    energy_current_hour = 0.05
    energy_previous_hour = 0.04
    energy_two_hours_ago = 0.06
    target_hour = 14
    target_day_of_week = 2
    target_is_weekend = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```
*Or using curl:*
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"energy_current_hour\": 0.05, \"energy_previous_hour\": 0.04, \"energy_two_hours_ago\": 0.06, \"target_hour\": 14, \"target_day_of_week\": 2, \"target_is_weekend\": 0}"
```
**Expected Response:**
```json
{
  "predicted_energy_kwh": 0.06106092241788783,
  "model": "GradientBoostingRegressor",
  "features_used": [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend"
  ]
}
```

### 3. Interactive Documentation (`/docs`)
Open your browser and navigate to:
```
http://localhost:8000/docs
```
You can view the interactive Swagger UI and test endpoints directly from the browser.

---

## 6. How to Stop the Container

- If running in the foreground: press `Ctrl + C` in the running terminal.
- If running in detached mode (`-d`) or from a separate terminal:
  ```powershell
  docker stop smart-home-advisor-app
  ```
