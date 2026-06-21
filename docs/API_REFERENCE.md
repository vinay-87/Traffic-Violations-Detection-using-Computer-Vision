# API Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### GET /
Service information and health status.

**Response**:
```json
{
  "service": "AI Traffic Violation Detection API",
  "version": "1.0.0",
  "status": "online",
  "timestamp": "2026-06-20T14:00:00.000Z"
}
```

---

### GET /health
Detailed component health check.

**Response (200)**:
```json
{
  "status": "healthy",
  "components": {
    "detector": "ready",
    "tracker": "ready",
    "engine": "ready",
    "ocr": "ready",
    "evidence": "ready",
    "preprocessor": "ready",
    "db": "ready"
  }
}
```

---

### POST /api/v1/detect
Upload a video file and detect all traffic violations.

**Request**: `multipart/form-data`
- `video`: Video file (MP4, AVI, MOV)

**Response (200)**:
```json
{
  "status": "success",
  "total_frames": 360,
  "violations_count": 5,
  "violations": [
    {
      "violation_id": "VIO_A1B2C3D4E5F6",
      "violation_type": "helmet_non_compliance",
      "timestamp": "2026-06-20T14:00:00.000Z",
      "confidence": 0.85,
      "vehicle_type": "motorcycle",
      "vehicle_bbox": [100, 200, 300, 400],
      "plate_number": "KA01AB1234",
      "plate_confidence": 0.78,
      "details": { "helmet_worn": false }
    }
  ]
}
```

---

### POST /api/v1/detect/frame
Process a single image frame.

**Request**: `multipart/form-data`
- `image`: Image file (JPEG, PNG)

**Response (200)**:
```json
{
  "detections": 12,
  "violations": [...]
}
```

---

### GET /api/v1/violations
List violations with optional filters and pagination.

**Query Parameters**:
- `skip` (int, default 0): Offset for pagination
- `limit` (int, default 50, max 200): Number of results
- `violation_type` (string, optional): Filter by type
- `plate_number` (string, optional): Search by plate
- `camera_id` (string, optional): Filter by camera

**Response (200)**:
```json
{
  "total": 150,
  "skip": 0,
  "limit": 50,
  "violations": [...]
}
```

---

### GET /api/v1/violations/{violation_id}
Get full details for a specific violation.

**Response (200)**: Full violation object with evidence paths.
**Response (404)**: `{"detail": "Violation not found"}`

---

### GET /api/v1/analytics/summary
Get aggregated violation statistics.

**Response (200)**:
```json
{
  "total": 150,
  "by_type": {
    "helmet_non_compliance": 45,
    "triple_riding": 12,
    "red_light_violation": 30
  },
  "avg_confidence": 0.82
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid file, unreadable image) |
| 404 | Resource not found |
| 500 | Internal server error |
| 503 | Service unavailable (components not loaded) |
