"""
api.py — FastAPI service for the Bean Leaf Disease model.

Endpoints
---------
GET  /health           uptime + model status  (powers the UI uptime widget)
GET  /metrics          latest evaluation metrics of the production model
GET  /data-stats       image counts per class (powers the UI charts)
POST /predict          single image  -> predicted class + confidence
POST /upload           bulk images or a .zip -> saved to data/train/<class>/
POST /retrain          fine-tunes the model on all current training data
GET  /retrain/status   progress of the background retraining job
"""

import os
import sys
import time
import threading
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (  # noqa: E402
    CLASS_NAMES,
    TRAIN_DIR,
    TEST_DIR,
    save_uploaded_image,
    extract_zip_uploads,
    count_images,
    count_new_uploads,
)
from src.prediction import predict, reload_model, get_model  # noqa: E402
from src import model as model_mod  # noqa: E402

START_TIME = time.time()

app = FastAPI(
    title="Bean Leaf Disease API",
    description="Classifies bean leaf images into angular_leaf_spot, bean_rust or healthy.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --------------------------------------------------------------------------- #
# Retraining job state (single background job at a time)
# --------------------------------------------------------------------------- #
_job = {"running": False, "started_at": None, "finished_at": None, "result": None, "error": None}
_job_lock = threading.Lock()


class RetrainRequest(BaseModel):
    epochs: int = 8
    force: bool = False


# --------------------------------------------------------------------------- #
# Health / monitoring
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    uptime = time.time() - START_TIME
    try:
        get_model()
        model_loaded = True
    except Exception:  # noqa: BLE001
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "uptime_seconds": round(uptime, 1),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
        "started_at": START_TIME,
        "container_id": os.getenv("HOSTNAME", "local"),  # shows which replica served you
        "classes": CLASS_NAMES,
    }


@app.get("/metrics")
def metrics():
    m = model_mod.load_metrics()
    if not m:
        raise HTTPException(404, "No metrics recorded yet. Run evaluation or retrain first.")
    return m


@app.get("/insights")
def insights():
    """Dataset insights computed in the notebook (real values, exported to models/insights.json)."""
    path = os.path.join(os.getenv("MODEL_DIR", "models"), "insights.json")
    if not os.path.exists(path):
        raise HTTPException(404, "insights.json not found. Run the notebook's export cell.")
    import json

    with open(path) as f:
        return json.load(f)


@app.get("/data-stats")
def data_stats():
    return {
        "train": count_images(TRAIN_DIR),
        "test": count_images(TEST_DIR),
        "new_uploads": count_new_uploads(TRAIN_DIR),
        "retrain_threshold": model_mod.RETRAIN_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Prediction — single datapoint
# --------------------------------------------------------------------------- #
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {file.content_type}")

    raw = await file.read()
    t0 = time.time()
    try:
        result = await run_in_threadpool(predict, raw)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not process image: {exc}")

    result["inference_ms"] = round((time.time() - t0) * 1000, 2)
    result["filename"] = file.filename
    result["served_by"] = os.getenv("HOSTNAME", "local")
    return result


# --------------------------------------------------------------------------- #
# Bulk upload of new training data
# --------------------------------------------------------------------------- #
@app.post("/upload")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    label: Optional[str] = Form(None),
):
    """
    Two modes:
      1. Multiple images + a `label` form field -> all saved under that class.
      2. A single .zip with class-named folders  -> label inferred per file.
    """
    saved, errors = 0, []

    for f in files:
        raw = await f.read()
        if f.filename.lower().endswith(".zip"):
            n, errs = extract_zip_uploads(raw)
            saved += n
            errors.extend(errs)
        else:
            if not label:
                errors.append(f"{f.filename}: no label given for a loose image")
                continue
            try:
                save_uploaded_image(raw, label)
                saved += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{f.filename}: {exc}")

    new_total = count_new_uploads(TRAIN_DIR)
    trigger, reason = model_mod.should_retrain(new_total)

    return {
        "saved": saved,
        "errors": errors,
        "new_uploads_total": new_total,
        "retrain_recommended": trigger,
        "reason": reason,
        "class_counts": count_images(TRAIN_DIR),
    }


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def _run_retrain(epochs: int):
    with _job_lock:
        _job.update(running=True, started_at=time.time(), finished_at=None, result=None, error=None)
    try:
        result = model_mod.retrain(epochs=epochs)
        reload_model()  # serve the new weights immediately
        with _job_lock:
            _job.update(running=False, finished_at=time.time(), result=result)
    except Exception as exc:  # noqa: BLE001
        with _job_lock:
            _job.update(running=False, finished_at=time.time(), error=str(exc))


@app.post("/retrain")
def retrain_endpoint(req: RetrainRequest, background_tasks: BackgroundTasks):
    """Kick off retraining in the background so the request returns immediately."""
    with _job_lock:
        if _job["running"]:
            raise HTTPException(409, "A retraining job is already running.")

    new_total = count_new_uploads(TRAIN_DIR)
    trigger, reason = model_mod.should_retrain(new_total)
    if not trigger and not req.force:
        return {
            "started": False,
            "reason": reason,
            "hint": "Upload more images or POST with force=true to override.",
        }

    background_tasks.add_task(_run_retrain, req.epochs)
    return {"started": True, "reason": reason if trigger else "forced by user", "epochs": req.epochs}


@app.get("/retrain/status")
def retrain_status():
    with _job_lock:
        job = dict(_job)
    if job["running"] and job["started_at"]:
        job["elapsed_seconds"] = round(time.time() - job["started_at"], 1)
    return job


@app.get("/")
def root():
    return {"service": "Bean Leaf Disease API", "docs": "/docs", "health": "/health"}
