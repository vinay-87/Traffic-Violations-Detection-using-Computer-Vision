"""
Streamlit Dashboard — AI Traffic Violation Detection System
Flipkart Gridlock 2.0

Four pages: Live Monitoring, Violation Log, Analytics, Settings
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import tempfile
import plotly.express as px
import plotly.graph_objects as go

from src.config import DASHBOARD, EVIDENCE
from src.violation_engine import Violation, ViolationEngine
from src.detector import TrafficDetector
from src.tracker import VehicleTracker
from src.ocr_engine import LicensePlateRecognizer
from src.evidence_generator import EvidenceGenerator
from src.preprocessor import ImagePreprocessor
from src.database import ViolationDB

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD["title"],
    page_icon=DASHBOARD["page_icon"],
    layout=DASHBOARD["layout"],
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #FF6B35 0%, #F7C948 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 0.8rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.metric-card h3 { margin: 0; font-size: 2rem; }
.metric-card p { margin: 0; font-size: 0.85rem; opacity: 0.9; }
.violation-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.badge-helmet { background: #e74c3c; }
.badge-triple { background: #e67e22; }
.badge-redlight { background: #c0392b; }
.badge-stopline { background: #f39c12; }
.badge-wrongside { background: #8e44ad; }
.badge-seatbelt { background: #2980b9; }
.badge-parking { background: #27ae60; }
.alert-card {
    background: rgba(255, 107, 53, 0.08);
    border-left: 4px solid #FF6B35;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 0.5rem 0.5rem 0;
}
.stButton>button {
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
}
.footer {
    text-align: center;
    color: #666;
    padding: 1rem 0;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────
if "violations" not in st.session_state:
    st.session_state.violations = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "components_loaded" not in st.session_state:
    st.session_state.components_loaded = False


# ─── Lazy Component Loading ──────────────────────────────────
@st.cache_resource
def load_components():
    """Load AI components once, cache across reruns."""
    detector = TrafficDetector()
    tracker = VehicleTracker()
    engine = ViolationEngine()
    ocr = LicensePlateRecognizer()
    evidence = EvidenceGenerator()
    preprocessor = ImagePreprocessor()
    db = ViolationDB()
    return detector, tracker, engine, ocr, evidence, preprocessor, db


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Control Panel")
    st.markdown("---")

    page = st.radio("Navigation", [
        "📹 Live Monitoring",
        "📋 Violation Log",
        "📊 Analytics",
        "⚙️ Settings",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### System Status")
    c1, c2 = st.columns(2)
    c1.markdown("🟢 **AI Model**")
    c2.markdown("🟢 **Database**")

    st.markdown("---")
    st.markdown("### Quick Stats")
    total_v = len(st.session_state.violations)
    st.metric("Total Violations", total_v)

    if total_v > 0:
        df_side = pd.DataFrame([v.to_dict() for v in st.session_state.violations])
        for vt, cnt in df_side["violation_type"].value_counts().items():
            st.caption(f"• {vt.replace('_', ' ').title()}: **{cnt}**")


# ─── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🚦 AI Traffic Violation Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Flipkart Gridlock 2.0 — Automated Enforcement Prototype</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: LIVE MONITORING
# ═══════════════════════════════════════════════════════════════
if page == "📹 Live Monitoring":
    st.markdown("### 📹 Live Monitoring")

    col_video, col_stats = st.columns([2, 1])

    with col_video:
        source = st.radio("Video Source", ["Upload Video", "Use Demo"],
                          horizontal=True)

        video_file = None
        if source == "Upload Video":
            video_file = st.file_uploader(
                "Upload a traffic video",
                type=["mp4", "avi", "mov", "mkv"],
            )
        else:
            st.info("📽️ Click **Start Processing** to generate & analyse a synthetic traffic demo.")

        # Controls
        bc1, bc2, bc3 = st.columns(3)
        start_btn = bc1.button("▶ Start Processing", use_container_width=True)
        stop_btn = bc2.button("⏹ Stop", use_container_width=True)
        clear_btn = bc3.button("🗑 Clear Results", use_container_width=True)

        if clear_btn:
            st.session_state.violations = []
            st.rerun()

        if stop_btn:
            st.session_state.processing = False

        # Video display placeholder
        video_ph = st.empty()
        progress_ph = st.empty()
        status_ph = st.empty()

        if start_btn:
            st.session_state.processing = True

            try:
                detector, tracker, engine, ocr, evidence, preprocessor, db = load_components()
            except Exception as e:
                st.error(f"Failed to load AI components: {e}")
                st.session_state.processing = False
                st.stop()

            # Get video path
            if source == "Upload Video" and video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                video_path = tfile.name
                tfile.close()
            elif source == "Use Demo":
                from demo import create_demo_video
                video_path = str(Path(EVIDENCE["output_dir"]).parent / "demo_traffic.mp4")
                create_demo_video(video_path)
            else:
                st.warning("Please upload a video or select Demo mode.")
                st.session_state.processing = False
                st.stop()

            # Process video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            frame_idx = 0

            tracker.reset()
            engine.reset()

            pbar = progress_ph.progress(0, text="Processing...")

            while cap.isOpened() and st.session_state.processing:
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess
                processed, _ = preprocessor.preprocess_frame(frame)

                # Detect + Track
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

                # Violations
                violations = engine.detect_violations(
                    frame, detections, tracks, frame_idx, tracker=tracker
                )

                # OCR
                for v in violations:
                    if v.vehicle_bbox:
                        try:
                            pimg, pbbox = ocr.detect_plate_region(frame, v.vehicle_bbox)
                            if pimg is not None:
                                text, conf = ocr.recognize(pimg)
                                v.plate_number = text
                                v.plate_confidence = conf
                        except Exception:
                            pass

                # Save
                for v in violations:
                    evidence.add_frame_to_buffer(frame, frame_idx / fps)
                    v = evidence.generate_evidence(frame, v, detections)
                    st.session_state.violations.append(v)
                    db.save_violation(v)

                if not violations:
                    evidence.add_frame_to_buffer(frame, frame_idx / fps)

                # Display annotated frame
                annotated = evidence.annotate_frame(frame, violations, detections)
                display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_ph.image(display, use_container_width=True)

                progress = min((frame_idx + 1) / max(total_frames, 1), 1.0)
                pbar.progress(progress,
                              text=f"Frame {frame_idx+1}/{total_frames} | "
                                   f"Violations: {len(st.session_state.violations)}")

                frame_idx += 1

            cap.release()
            st.session_state.processing = False
            progress_ph.empty()
            status_ph.success(
                f"✅ Done! Processed {frame_idx} frames, "
                f"found {len(st.session_state.violations)} violations."
            )

    with col_stats:
        st.markdown("### 📊 Live Stats")

        v_list = st.session_state.violations
        st.metric("Total Violations", len(v_list))

        if v_list:
            df_stats = pd.DataFrame([v.to_dict() for v in v_list])
            type_counts = df_stats["violation_type"].value_counts()

            # Mini pie chart
            fig = px.pie(
                values=type_counts.values,
                names=[n.replace("_", " ").title() for n in type_counts.index],
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_layout(
                height=250, showlegend=True,
                legend=dict(font=dict(size=10)),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Violation type icons
            icons = {
                "helmet_non_compliance": "🪖",
                "triple_riding": "🏍️",
                "red_light_violation": "🚦",
                "stop_line_violation": "⚠️",
                "wrong_side_driving": "🔄",
                "seatbelt_non_compliance": "🪢",
                "illegal_parking": "🅿️",
            }
            for vt, cnt in type_counts.items():
                icon = icons.get(vt, "⚡")
                st.markdown(f"{icon} **{vt.replace('_', ' ').title()}**: {cnt}")

        st.markdown("---")
        st.markdown("### 🚨 Recent Alerts")
        for v in reversed(v_list[-5:]):
            st.markdown(f"""
<div class="alert-card">
<b>{v.violation_type.replace('_', ' ').upper()}</b><br>
<small>Vehicle: {v.vehicle_type} | Conf: {v.confidence:.0%}</small><br>
<small>Plate: {v.plate_number or 'N/A'}</small>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: VIOLATION LOG
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Violation Log":
    st.markdown("### 📋 Violation Log")

    v_list = st.session_state.violations
    if not v_list:
        # Try loading from database
        try:
            db = ViolationDB()
            db_records = db.get_violations(limit=200)
            if db_records:
                st.info(f"Loaded {len(db_records)} violations from database.")
                df = pd.DataFrame(db_records)
            else:
                st.info("No violations recorded yet. Process a video first!")
                st.stop()
        except Exception:
            st.info("No violations recorded yet. Process a video first!")
            st.stop()
    else:
        df = pd.DataFrame([v.to_dict() for v in v_list])

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        types_available = df["violation_type"].unique().tolist()
        filter_types = st.multiselect("Filter by Type", types_available)
    with fc2:
        filter_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    with fc3:
        search_plate = st.text_input("Search Plate")

    # Apply filters
    filtered = df.copy()
    if filter_types:
        filtered = filtered[filtered["violation_type"].isin(filter_types)]
    filtered = filtered[filtered["confidence"] >= filter_conf]
    if search_plate:
        filtered = filtered[
            filtered["plate_number"].fillna("").str.contains(search_plate.upper())
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} violations**")

    # Display table
    display_cols = ["violation_id", "violation_type", "timestamp",
                    "confidence", "vehicle_type", "plate_number"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[available_cols].copy()
    if "confidence" in show_df.columns:
        show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.0%}")
    show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    # Export
    ec1, ec2 = st.columns(2)
    with ec1:
        csv = show_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export CSV", csv, "violations.csv", "text/csv",
                          use_container_width=True)
    with ec2:
        json_str = filtered.to_json(orient="records", indent=2, default_handler=str)
        st.download_button("📥 Export JSON", json_str, "violations.json",
                          "application/json", use_container_width=True)

    # Detail view
    if len(filtered) > 0:
        st.markdown("---")
        st.markdown("### Violation Detail")
        sel = st.selectbox(
            "Select violation",
            range(len(filtered)),
            format_func=lambda i: (
                f"{filtered.iloc[i]['violation_type'].replace('_', ' ').title()} — "
                f"{str(filtered.iloc[i].get('timestamp', ''))[:19]}"
            ),
        )
        if sel is not None:
            row = filtered.iloc[sel]
            dc1, dc2 = st.columns(2)
            with dc1:
                st.json(row.to_dict())
            with dc2:
                snap = row.get("snapshot_path")
                if snap and Path(str(snap)).exists():
                    st.image(str(snap), caption="Evidence Snapshot")
                else:
                    st.info("Snapshot not available")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("### 📊 Analytics & Reporting")

    v_list = st.session_state.violations
    if not v_list:
        try:
            db = ViolationDB()
            db_records = db.get_violations(limit=500)
            if db_records:
                df = pd.DataFrame(db_records)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                st.info("No data available. Process some videos first!")
                st.stop()
        except Exception:
            st.info("No data available.")
            st.stop()
    else:
        df = pd.DataFrame([v.to_dict() for v in v_list])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Violations", len(df))
    mc2.metric("Violation Types", df["violation_type"].nunique())
    mc3.metric("Avg Confidence", f"{df['confidence'].mean():.0%}")
    plates = df["plate_number"].dropna().nunique()
    mc4.metric("Unique Plates", plates)

    st.markdown("---")

    # Violation type distribution
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Violations by Type")
        tc = df["violation_type"].value_counts()
        fig1 = px.bar(
            x=[n.replace("_", " ").title() for n in tc.index],
            y=tc.values,
            color=tc.values,
            color_continuous_scale="OrRd",
            labels={"x": "Type", "y": "Count"},
        )
        fig1.update_layout(height=350, showlegend=False,
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col_a2:
        st.markdown("#### Vehicle Type Distribution")
        vc = df["vehicle_type"].value_counts()
        fig2 = px.pie(
            values=vc.values,
            names=vc.index,
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # Confidence distribution
    st.markdown("#### Confidence Score Distribution")
    fig3 = px.histogram(
        df, x="confidence", nbins=20,
        color_discrete_sequence=["#FF6B35"],
        labels={"confidence": "Confidence Score"},
    )
    fig3.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Stacked violation timeline
    if len(df) > 1:
        st.markdown("#### Violation Timeline")
        df["minute"] = df["timestamp"].dt.floor("T")
        timeline = df.groupby(["minute", "violation_type"]).size().reset_index(name="count")
        timeline["violation_type"] = timeline["violation_type"].str.replace("_", " ").str.title()
        fig4 = px.area(
            timeline, x="minute", y="count", color="violation_type",
            labels={"minute": "Time", "count": "Violations", "violation_type": "Type"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig4.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4: SETTINGS
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("### ⚙️ System Settings")

    st.markdown("#### Detection Thresholds")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.slider("Object Detection Confidence", 0.1, 1.0, 0.40, 0.05,
                  key="s_det_conf")
        st.slider("Helmet IoU Threshold", 0.1, 1.0, 0.30, 0.05,
                  key="s_helmet_iou")
        st.slider("Triple Riding Overlap", 0.1, 1.0, 0.25, 0.05,
                  key="s_triple_overlap")
    with sc2:
        st.slider("Stop Line Cross Ratio", 0.1, 1.0, 0.30, 0.05,
                  key="s_stop_cross")
        st.slider("Wrong Side Angle Tolerance", 30, 150, 90, 5,
                  key="s_wrong_angle")
        st.slider("Parking Stationary Frames", 10, 150, 45, 5,
                  key="s_park_frames")

    st.markdown("---")
    st.markdown("#### Camera Configuration")
    st.text_input("Camera ID", value="CAM_001", key="s_cam_id")
    st.text_input("Location Name", value="MG Road, Bengaluru", key="s_loc")
    lc1, lc2 = st.columns(2)
    lc1.number_input("Latitude", value=12.9716, format="%.4f", key="s_lat")
    lc2.number_input("Longitude", value=77.5946, format="%.4f", key="s_lng")

    st.markdown("---")
    st.markdown("#### Model Settings")
    st.selectbox("YOLO Model", ["yolov8n.pt (Nano - Fast)",
                                "yolov8s.pt (Small - Balanced)",
                                "yolov8m.pt (Medium - Accurate)"],
                key="s_model")
    st.selectbox("Device", ["Auto (GPU if available)", "GPU (CUDA)",
                             "CPU"], key="s_device")

    st.markdown("---")
    bc1, bc2 = st.columns(2)
    if bc1.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved!")
    if bc2.button("🔄 Reset Defaults", use_container_width=True):
        st.warning("Settings reset to defaults.")


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
🚦 AI Traffic Violation Detection System | Flipkart Gridlock 2.0 | Built for Safer Roads
</div>
""", unsafe_allow_html=True)
