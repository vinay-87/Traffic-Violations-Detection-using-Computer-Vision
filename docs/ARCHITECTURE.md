# System Architecture

## Overview

The AI Traffic Violation Detection System is a 4-layer pipeline that processes video input through detection, tracking, violation analysis, and evidence generation.

## Layer 1: Video Ingestion & Preprocessing

**Module**: `src/preprocessor.py`

Handles diverse input conditions:
- **Noise Reduction**: Bilateral filter (edge-preserving)
- **Contrast Enhancement**: CLAHE on LAB colour space
- **Low-Light Enhancement**: Gamma correction + brightness boost
- **Shadow Reduction**: HSV-based shadow masking with selective brightening
- **Letterbox Resize**: Aspect-ratio-preserving resize with padding

## Layer 2: Object Detection & Tracking

**Detection**: `src/detector.py` — YOLOv8n (Ultralytics)
- Pre-trained on COCO (80 classes), filtered to traffic-relevant: person, bicycle, car, motorcycle, bus, truck, traffic_light, stop_sign
- Confidence threshold: 0.40, NMS IoU: 0.50
- Auto-downloads weights on first run

**Tracking**: `src/tracker.py` — DeepSORT
- Re-identification embeddings via MobileNet
- Trajectory history (up to 60 frames per track)
- Speed estimation, direction calculation, stationarity detection

## Layer 3: Violation Analysis

**Module**: `src/violation_engine.py`

7 violation types with temporal confirmation (multi-frame validation):

| Violation | Method | Temporal Frames |
|-----------|--------|----------------|
| Helmet | IoA + head region texture/shape analysis | 4 |
| Triple Riding | Person-motorcycle IoA counting | 4 |
| Red Light | HSV traffic light analysis + stop line crossing | 3 |
| Stop Line | Centroid vs configurable Y-position | 3 |
| Wrong Side | Trajectory angle vs expected direction | 6 |
| Seatbelt | Driver region edge + diagonal line detection | 5 |
| Illegal Parking | Position variance stationarity check | 45 |

## Layer 4: Evidence & Persistence

**OCR**: `src/ocr_engine.py` — EasyOCR with Indian plate format validation
**Evidence**: `src/evidence_generator.py` — Annotated snapshots, MP4 clips, JSON metadata
**Database**: `src/database.py` — SQLAlchemy ORM on SQLite with indexed queries

## Presentation Layer

**Dashboard**: `dashboard/app.py` — Streamlit with 4 pages
**API**: `api/server.py` — FastAPI with OpenAPI docs

## Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Object Detection | YOLOv8n | SOTA speed-accuracy tradeoff, 6.2MB model |
| Tracking | DeepSORT | Proven multi-object tracker with ReID |
| OCR | EasyOCR | Multi-language support, good accuracy |
| Dashboard | Streamlit | Rapid prototyping, built-in interactivity |
| API | FastAPI | Async, auto-docs, production-ready |
| Database | SQLite + SQLAlchemy | Zero-config, portable, ORM flexibility |
| Visualization | Plotly | Interactive charts, dark theme support |
