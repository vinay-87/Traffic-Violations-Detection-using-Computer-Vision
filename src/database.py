"""
Database Module — SQLAlchemy ORM for violation persistence.
Uses SQLite for zero-config deployment.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime, Text, Integer, Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.config import DATABASE

logger = logging.getLogger(__name__)

Base = declarative_base()


class ViolationRecord(Base):
    """SQLAlchemy model for violation storage."""
    __tablename__ = "violations"

    id = Column(String(36), primary_key=True)
    violation_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float, nullable=False)

    vehicle_type = Column(String(30))
    plate_number = Column(String(20), index=True)
    plate_confidence = Column(Float)

    camera_id = Column(String(50), index=True)
    latitude = Column(Float)
    longitude = Column(Float)

    snapshot_path = Column(String(500))
    video_clip_path = Column(String(500))

    status = Column(String(20), default="pending", index=True)
    details_json = Column(Text)

    inference_time_ms = Column(Float)
    model_version = Column(String(20))

    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_type_ts", "violation_type", "timestamp"),
        Index("ix_cam_ts", "camera_id", "timestamp"),
    )

    def to_dict(self) -> Dict:
        import json
        return {
            "id": self.id,
            "violation_type": self.violation_type,
            "timestamp": self.timestamp.isoformat() + "Z" if self.timestamp else None,
            "confidence": self.confidence,
            "vehicle_type": self.vehicle_type,
            "plate_number": self.plate_number,
            "plate_confidence": self.plate_confidence,
            "camera_id": self.camera_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "snapshot_path": self.snapshot_path,
            "video_clip_path": self.video_clip_path,
            "status": self.status,
            "details": json.loads(self.details_json) if self.details_json else {},
            "inference_time_ms": self.inference_time_ms,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ─── Database Manager ────────────────────────────────────────

class ViolationDB:
    """CRUD operations for violation records."""

    def __init__(self, db_url: Optional[str] = None):
        url = db_url or DATABASE["url"]
        self.engine = create_engine(url, echo=DATABASE["echo"])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database ready: {url}")

    def _session(self) -> Session:
        return self.Session()

    def save_violation(self, violation) -> str:
        """Save a Violation dataclass to the database."""
        import json
        session = self._session()
        try:
            record = ViolationRecord(
                id=violation.violation_id,
                violation_type=violation.violation_type,
                timestamp=datetime.utcnow(),
                confidence=violation.confidence,
                vehicle_type=violation.vehicle_type,
                plate_number=violation.plate_number,
                plate_confidence=violation.plate_confidence,
                camera_id=violation.camera_id,
                latitude=violation.location.get("lat") if violation.location else 12.9716,
                longitude=violation.location.get("lng") if violation.location else 77.5946,
                snapshot_path=violation.snapshot_path,
                video_clip_path=violation.video_clip_path,
                status="pending",
                details_json=json.dumps(violation.details, default=str),
                inference_time_ms=violation.inference_time_ms,
                model_version=violation.model_version,
            )
            session.add(record)
            session.commit()
            return record.id
        finally:
            session.close()

    def get_violations(
        self,
        skip: int = 0,
        limit: int = 50,
        violation_type: Optional[str] = None,
        plate_number: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> List[Dict]:
        """Query violations with optional filters."""
        session = self._session()
        try:
            q = session.query(ViolationRecord)
            if violation_type:
                q = q.filter(ViolationRecord.violation_type == violation_type)
            if plate_number:
                q = q.filter(ViolationRecord.plate_number.contains(plate_number))
            if camera_id:
                q = q.filter(ViolationRecord.camera_id == camera_id)
            q = q.order_by(ViolationRecord.timestamp.desc())
            results = q.offset(skip).limit(limit).all()
            return [r.to_dict() for r in results]
        finally:
            session.close()

    def get_violation_by_id(self, vid: str) -> Optional[Dict]:
        session = self._session()
        try:
            r = session.query(ViolationRecord).filter_by(id=vid).first()
            return r.to_dict() if r else None
        finally:
            session.close()

    def count_violations(self, violation_type: Optional[str] = None) -> int:
        session = self._session()
        try:
            q = session.query(ViolationRecord)
            if violation_type:
                q = q.filter(ViolationRecord.violation_type == violation_type)
            return q.count()
        finally:
            session.close()

    def get_summary(self) -> Dict:
        """Get aggregated violation summary."""
        session = self._session()
        try:
            from sqlalchemy import func
            total = session.query(ViolationRecord).count()
            by_type = dict(
                session.query(
                    ViolationRecord.violation_type,
                    func.count(ViolationRecord.id),
                ).group_by(ViolationRecord.violation_type).all()
            )
            avg_conf = session.query(
                func.avg(ViolationRecord.confidence)
            ).scalar()
            return {
                "total": total,
                "by_type": by_type,
                "avg_confidence": round(float(avg_conf or 0), 4),
            }
        finally:
            session.close()
