"""
AQUAVISION: AI-Based Oil Spill Detection
FastAPI Production Backend (Phase 14)

Provides validated inference endpoints, Grad-CAM visualization delivery,
incident persistence, and dynamic model reporting.
"""

import os
import io
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from backend.database import (
    init_db,
    save_incident,
    get_all_incidents,
    get_incident_by_id
)
from src.inference.predict import OilSpillPredictor

app = FastAPI(
    title="AQUAVISION Maritime Intelligence API",
    description="Sentinel-1 SAR AI-Powered Oil Spill Detection & Attribution Engine",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import tempfile

def _resolve_upload_dir() -> Path:
    if os.environ.get("VERCEL") or not os.access("backend/static", os.W_OK):
        d = Path(tempfile.gettempdir()) / "aquavision_uploads"
    else:
        d = Path("backend/static/uploads")
    d.mkdir(parents=True, exist_ok=True)
    return d

UPLOAD_DIR = _resolve_upload_dir()
if UPLOAD_DIR != Path("backend/static/uploads"):
    app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Predictor instance cache
predictor: Optional[OilSpillPredictor] = None


def get_predictor() -> OilSpillPredictor:
    """Lazy loader and singleton cache for predictor with automatic serverless fallback."""
    global predictor
    if predictor is None:
        model_path = Path("models/best_model.pth")
        predictor = OilSpillPredictor.get_instance(model_path)
    return predictor


@app.on_event("startup")
async def startup_event():
    """Initializes SQLite database and preloads model if available."""
    init_db()
    model_path = Path("models/best_model.pth")
    if model_path.exists():
        try:
            get_predictor()
        except Exception as e:
            print(f"Warning: Preloading predictor encountered: {e}")


from fastapi.responses import JSONResponse, FileResponse


@app.get("/", tags=["General"], response_class=FileResponse)
async def root():
    """Serves the AQUAVISION interactive dashboard."""
    index_file = Path("backend/static/index.html")
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "project": "AQUAVISION",
        "tagline": "AI Maritime Intelligence — Oil Spill Detection and Attribution",
        "status": "online",
        "version": "1.0.0"
    })


@app.get("/api", tags=["General"])
async def api_info():
    """Returns API metadata and documentation pointers."""
    return {
        "project": "AQUAVISION",
        "tagline": "AI Maritime Intelligence — Oil Spill Detection and Attribution",
        "status": "online",
        "version": "1.0.0",
        "dataset": "Sentinel-1 SAR Oil Spill Detection Dataset",
        "docs": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    device_str = "edge-serverless"
    try:
        p = get_predictor()
        device_str = str(getattr(p, "device", "edge-serverless"))
    except Exception:
        pass

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device_str,
        "timestamp": time.time()
    }


@app.get("/model-info", tags=["Model Analytics"])
async def get_model_info():
    """Returns dynamic model evaluation report and dataset statistics."""
    report_file = Path("outputs/reports/model_report.json")
    if not report_file.exists():
        report_file = Path("backend/static/reports/model_report.json")
    if not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model report has not yet been generated. Train and evaluate the model first."
        )
    with open(report_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/samples", tags=["General"])
async def get_samples():
    """Returns preset test SAR chips for one-click demo testing."""
    manifest_file = Path("backend/static/samples/manifest.json")
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def validate_image_file(file: UploadFile) -> str:
    """Validates file presence and extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename.")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


@app.post("/predict", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """Runs fast classification without Grad-CAM image generation."""
    validate_image_file(file)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        p = get_predictor()
        res = p.predict(contents, generate_explanation=False)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-with-explanation", tags=["Inference"])
async def predict_with_explanation(file: UploadFile = File(...)):
    """
    Performs full inference:
    1. Preprocesses image
    2. Runs model inference
    3. Computes Grad-CAM attention heatmap & blended overlay
    4. Persists record to SQLite incident database
    5. Returns prediction, confidence, probabilities, and asset URLs.
    """
    ext = validate_image_file(file)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(contents)) # reopen after verify
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is corrupt or unreadable as an image.")

    try:
        p = get_predictor()
        res = p.predict(
            pil_img,
            generate_explanation=True,
            save_artifacts_dir=UPLOAD_DIR
        )

        # Convert local file paths into relative URLs for the web client
        orig_name = Path(res["image_path"]).name if res.get("image_path") else ""
        cam_name = Path(res["gradcam_path"]).name if res.get("gradcam_path") else ""
        overlay_name = Path(res["overlay_path"]).name if res.get("overlay_path") else ""

        orig_url = f"/static/uploads/{orig_name}" if orig_name else ""
        cam_url = f"/static/uploads/{cam_name}" if cam_name else ""
        overlay_url = f"/static/uploads/{overlay_name}" if overlay_name else ""

        # Persist to SQLite Incidents Table
        incident_record = {
            "incident_uuid": str(uuid.uuid4()),
            "filename": file.filename,
            "prediction": res["prediction"],
            "class_id": res["class_id"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "explanation": res["explanation"],
            "original_image_url": orig_url,
            "gradcam_image_url": cam_url,
            "overlay_image_url": overlay_url,
            "model_version": res["model_version"]
        }
        saved_row = save_incident(incident_record)

        return {
            "incident_id": saved_row["id"],
            "incident_uuid": saved_row["incident_uuid"],
            "filename": saved_row["filename"],
            "prediction": saved_row["prediction"],
            "class_id": saved_row["class_id"],
            "confidence": saved_row["confidence"],
            "probabilities": {
                "oil_spill": saved_row["oil_probability"],
                "no_oil": saved_row["no_oil_probability"]
            },
            "explanation": saved_row["explanation"],
            "original_image_url": saved_row["original_image_url"],
            "gradcam_image_url": saved_row["gradcam_image_url"],
            "overlay_image_url": saved_row["overlay_image_url"],
            "model_version": saved_row["model_version"],
            "created_at": saved_row["created_at"],
            "vessel_attribution_status": saved_row["vessel_attribution_status"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/incidents", tags=["Incidents"])
async def list_incidents(limit: int = Query(default=100, ge=1, le=500)):
    """Retrieves list of previous oil spill detection incidents."""
    return get_all_incidents(limit=limit)


@app.get("/incidents/{incident_id}", tags=["Incidents"])
async def get_incident(incident_id: int):
    """Retrieves a single incident by its ID."""
    row = get_incident_by_id(incident_id)
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return row
