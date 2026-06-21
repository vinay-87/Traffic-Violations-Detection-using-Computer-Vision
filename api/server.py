"""
FastAPI Backend — AI Traffic Violation Detection API
Flipkart Gridlock 2.0

Run: python -m api.server
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import cv2
import numpy as np
import tempfile
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import API
from src.detector import TrafficDetector
from src.tracker import VehicleTracker
from src.violation_engine import ViolationEngine
from src.ocr_engine import LicensePlateRecognizer
from src.evidence_generator import EvidenceGenerator
from src.preprocessor import ImagePreprocessor
from src.database import ViolationDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Traffic Violation Detection API",
    description="Automated traffic violation detection, classification, and evidence generation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded components
_components = {}


def _get():
    if not _components:
        logger.info("Loading AI components...")
        _components["detector"] = TrafficDetector()
        _components["tracker"] = VehicleTracker()
        _components["engine"] = ViolationEngine()
        _components["ocr"] = LicensePlateRecognizer()
        _components["evidence"] = EvidenceGenerator()
        _components["preprocessor"] = ImagePreprocessor()
        _components["db"] = ViolationDB()
        logger.info("Components ready")
    return _components


@app.get("/")
async def root():
    return {
        "service": "AI Traffic Violation Detection API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/health")
async def health():
    try:
        c = _get()
        return {
            "status": "healthy",
            "components": {k: "ready" for k in c},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        return JSONResponse(status_code=503,
                            content={"status": "unhealthy", "error": str(e)})


@app.post("/api/v1/detect")
async def detect_video(video: UploadFile = File(...)):
    """Upload video → detect all violations → return results."""
    c = _get()
    try:
        # Save to temp
        suffix = Path(video.filename or "v.mp4").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Cannot open video file")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        c["tracker"].reset()
        c["engine"].reset()
        found = []
        fidx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = c["detector"].detect(frame)
            tracks = c["tracker"].update(dets, frame)
            viols = c["engine"].detect_violations(
                frame, dets, tracks, fidx, tracker=c["tracker"]
            )

            for v in viols:
                if v.vehicle_bbox:
                    try:
                        pi, pb = c["ocr"].detect_plate_region(frame, v.vehicle_bbox)
                        if pi is not None:
                            t, cf = c["ocr"].recognize(pi)
                            v.plate_number = t
                            v.plate_confidence = cf
                    except Exception:
                        pass
                c["evidence"].add_frame_to_buffer(frame, fidx / fps)
                v = c["evidence"].generate_evidence(frame, v, dets)
                c["db"].save_violation(v)
                found.append(v)

            if not viols:
                c["evidence"].add_frame_to_buffer(frame, fidx / fps)
            fidx += 1

        cap.release()
        os.unlink(tmp_path)

        return {
            "status": "success",
            "total_frames": total,
            "violations_count": len(found),
            "violations": [v.to_dict() for v in found],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"detect error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/v1/detect/frame")
async def detect_frame(image: UploadFile = File(...)):
    """Process a single image frame."""
    c = _get()
    try:
        data = await image.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Cannot decode image")

        dets = c["detector"].detect(img)
        tracks = c["tracker"].update(dets, img)
        viols = c["engine"].detect_violations(img, dets, tracks, 0)

        for v in viols:
            if v.vehicle_bbox:
                try:
                    pi, _ = c["ocr"].detect_plate_region(img, v.vehicle_bbox)
                    if pi is not None:
                        t, cf = c["ocr"].recognize(pi)
                        v.plate_number = t
                        v.plate_confidence = cf
                except Exception:
                    pass

        return {
            "detections": len(dets),
            "violations": [v.to_dict() for v in viols],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/v1/violations")
async def list_violations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    violation_type: Optional[str] = None,
    plate_number: Optional[str] = None,
    camera_id: Optional[str] = None,
):
    """List violations from database with filters."""
    c = _get()
    results = c["db"].get_violations(
        skip=skip, limit=limit,
        violation_type=violation_type,
        plate_number=plate_number,
        camera_id=camera_id,
    )
    total = c["db"].count_violations(violation_type=violation_type)
    return {"total": total, "skip": skip, "limit": limit, "violations": results}


@app.get("/api/v1/violations/{violation_id}")
async def get_violation(violation_id: str):
    c = _get()
    result = c["db"].get_violation_by_id(violation_id)
    if not result:
        raise HTTPException(404, "Violation not found")
    return result


@app.get("/api/v1/analytics/summary")
async def analytics_summary():
    c = _get()
    return c["db"].get_summary()


if __name__ == "__main__":
    uvicorn.run("api.server:app", host=API["host"], port=API["port"],
                reload=False)
