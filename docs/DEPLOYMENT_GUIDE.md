# Deployment Guide

## Local Development (Windows)

### Prerequisites
- Python 3.10+
- NVIDIA GPU + CUDA (optional, for faster inference)

### Setup
```bash
# Option 1: One-click
setup.bat

# Option 2: Manual
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Running

**Demo mode** (synthetic video):
```bash
python demo.py --demo --output ./output --save-video
```

**Your own video**:
```bash
python demo.py --video path/to/traffic.mp4 --output ./output --save-video --display
```

**Dashboard**:
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

**API Server**:
```bash
python -m api.server
# Opens at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Docker Deployment

```bash
docker-compose up --build
```

Services:
- Dashboard: http://localhost:8501
- API: http://localhost:8000

## Cloud Deployment

### AWS / GCP

1. Use a GPU instance (e.g., AWS g4dn.xlarge, GCP n1-standard-4 + T4)
2. Install NVIDIA drivers + CUDA
3. Clone repository
4. Run setup.bat / pip install
5. Configure reverse proxy (nginx) for dashboard + API

### Scaling

- **Horizontal**: Multiple API workers via uvicorn `--workers N`
- **GPU Sharing**: Use NVIDIA Triton Inference Server for multi-model serving
- **Storage**: Switch SQLite → PostgreSQL for production
- **Queuing**: Add Redis/RabbitMQ for async video processing

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU mode) | NVIDIA GTX 1060+ |
| Storage | 5 GB | 50+ GB (for evidence) |
| Python | 3.10 | 3.10-3.12 |
