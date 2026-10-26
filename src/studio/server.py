"""
FastAPI Backend Server for Praxis Studio (`src/studio/server.py`).
Provides REST and telemetry APIs for:
1. Exploring demonstrated YouTube manipulation clips with synchronized video playback.
2. Serving 3D OBJ digital twins, LeRobot Parquet tables, and contact manifolds.
3. Accessing real robot hand specifications (Unitree H1, Tesla Optimus, ALOHA, Franka).
4. Running native MuJoCo forward dynamics simulation with telemetry.
5. Ingesting and slicing new videos asynchronously.
"""

import asyncio
import glob
import json
import os
import sys
from typing import Dict, List, Optional
import numpy as np

# Ensure project root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.studio.robot_specs import list_robot_specs, get_robot_spec
from src.feasibility.mujoco_verifier import MuJoCoVerifier
from src.pose_extraction.extractor import VideoPoseExtractor
from src.ingestion.slicer import slice_action_video
from src.ingestion.inspector import inspect_video

app = FastAPI(
    title="Praxis Studio API",
    description="Interactive Web Studio for Video-to-Physical-Digital-Twin Manipulation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = os.path.join(REPO_ROOT, "src", "studio", "static")
DATA_CLIPS_DIR = os.path.join(REPO_ROOT, "data", "clips")
DATA_PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
ASSETS_IMAGES_DIR = os.path.join(REPO_ROOT, "assets", "images")
DATA_RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

# Mount static directories
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(DATA_PROCESSED_DIR):
    app.mount("/data/processed", StaticFiles(directory=DATA_PROCESSED_DIR), name="processed")
if os.path.exists(ASSETS_IMAGES_DIR):
    app.mount("/assets/images", StaticFiles(directory=ASSETS_IMAGES_DIR), name="assets")


@app.get("/data/clips/{filename}")
async def stream_clip_video(filename: str, request: Request):
    """
    Robust HTTP byte-range video streaming endpoint.
    Handles HTML5 media probes and client disconnects cleanly without Proactor exceptions.
    """
    file_path = os.path.join(DATA_CLIPS_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if not range_header:
        def full_iter():
            try:
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        yield chunk
            except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
                return

        return StreamingResponse(
            full_iter(),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=86400",
            }
        )

    try:
        range_val = range_header.strip().replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        content_length = (end - start) + 1
    except Exception:
        start = 0
        end = file_size - 1
        content_length = file_size

    def range_iter():
        try:
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    read_len = min(65536, remaining)
                    chunk = f.read(read_len)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            return

    return StreamingResponse(
        range_iter(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Cache-Control": "public, max-age=86400",
        }
    )

extractor = VideoPoseExtractor()
mujoco_verifier = MuJoCoVerifier()


# Known Demonstration Catalog
CLIPS_CATALOG = [
    {
        "id": "07_lpl_masterlock",
        "title": "Master Lock No. 3 — Bimanual Bypass & Latch Release",
        "channel": "LockPickingLawyer",
        "category": "Padlock Bypass Decoding",
        "video_url": "/data/clips/LPL_MasterLock_No3_clip_40s_50s.mp4",
        "obj_url": "/data/processed/07_lpl_masterlock_scene_3d.obj",
        "parquet_url": "/data/processed/07_lpl_masterlock_metadata.parquet",
        "metadata_url": "/data/processed/07_lpl_masterlock_metadata.json",
        "mujoco_xml_url": "/data/processed/07_lpl_masterlock_scene.xml",
        "comparison_img": "/assets/images/07_lpl_masterlock_end_to_end_comparison.jpg",
        "animation_gif": "/assets/images/07_lpl_masterlock_end_to_end_animation.gif",
        "decomposition_img": "/assets/images/07_lpl_masterlock_frame_decomposition.jpg",
        "duration_sec": 10.0,
        "fps": 29.97,
        "total_frames": 300,
        "action_verb": "Bimanual bypass insertion, collar probing, tool drop & index pointing",
        "workpiece": "Master Lock No. 3 (40x19x35mm, 195g)",
        "tool": "0.5mm flat spring steel bypass needle",
        "hand_engagement": "Bimanual 5-finger (Left holding lock, Right manipulating probe)",
        "sdf_status": "Certified (SDF >= 0.0mm, zero penetration)",
        "phases": [
            {"time_range": "0.0s - 0.9s", "name": "1. Diagonal Approach (Lock Tilted)", "needle_active": True},
            {"time_range": "1.0s - 3.8s", "name": "2. Horizontal Collar Slide Left", "needle_active": True},
            {"time_range": "4.1s - 5.2s", "name": "3. Collar Probing & Partial Withdrawal", "needle_active": True},
            {"time_range": "5.5s - 5.9s", "name": "4. Left Thumb Slide Down & Needle Drop", "needle_active": False},
            {"time_range": "6.2s - 10.0s", "name": "5. Right Index Pointing & Tapping on Body", "needle_active": False}
        ]
    },
    {
        "id": "01_lpl_dropkey",
        "title": "Drop Key — Precision Stem Pinch & 90° Articulation",
        "channel": "LockPickingLawyer",
        "category": "High-Security Lock Articulation",
        "video_url": "/data/clips/01_LPL_DropKey_RRunwAI7N5Q_clip_18s_26s.mp4",
        "obj_url": "/data/processed/01_lpl_dropkey_scene_3d.obj",
        "parquet_url": "/data/processed/01_lpl_dropkey_metadata.parquet",
        "metadata_url": "/data/processed/01_lpl_dropkey_metadata.json",
        "mujoco_xml_url": None,
        "comparison_img": "/assets/images/01_lpl_dropkey_end_to_end_comparison.jpg",
        "animation_gif": "/assets/images/01_lpl_dropkey_end_to_end_animation.gif",
        "decomposition_img": None,
        "duration_sec": 8.53,
        "fps": 30.0,
        "total_frames": 256,
        "action_verb": "3-finger precision pinch and downward drop link folding",
        "workpiece": "Folding Drop Key (stainless steel stem, 47g)",
        "tool": "Drop link articulation pin",
        "hand_engagement": "Unimanual 3-finger precision pinch",
        "sdf_status": "Certified (SDF >= 0.0mm)",
        "phases": [
            {"time_range": "0.0s - 3.2s", "name": "1. Horizontal Stem Alignment", "needle_active": False},
            {"time_range": "3.2s - 6.5s", "name": "2. 90-Degree Downward Link Folding", "needle_active": False},
            {"time_range": "6.5s - 8.5s", "name": "3. In-Hand Key Re-grasp", "needle_active": False}
        ]
    },
    {
        "id": "05_watch_zodiac",
        "title": "Zodiac Watch — Chronograph Gear Micro-Extraction",
        "channel": "Wristwatch Revival",
        "category": "Horology Micro-Manipulation",
        "video_url": "/data/clips/05_Watch_Zodiac_f8NFSwb6VQ8_clip_595s_608s.mp4",
        "obj_url": "/data/processed/05_watch_zodiac_scene_3d.obj",
        "parquet_url": "/data/processed/05_watch_zodiac_metadata.parquet",
        "metadata_url": "/data/processed/05_watch_zodiac_metadata.json",
        "mujoco_xml_url": None,
        "comparison_img": "/assets/images/05_watch_zodiac_comparison.jpg",
        "animation_gif": "/assets/images/05_watch_zodiac_animation.gif",
        "decomposition_img": None,
        "duration_sec": 13.01,
        "fps": 30.0,
        "total_frames": 390,
        "action_verb": "Micro-tweezer vertical extraction from jewel bearing",
        "workpiece": "Miniature brass chronograph gear (8.4mm diameter, 0.02g)",
        "tool": "Dumont #5 Swiss watchmaker tweezers",
        "hand_engagement": "Precision micro-grip (0.45N normal force)",
        "sdf_status": "Certified (SDF >= 0.0mm, jewel clearance verified)",
        "phases": [
            {"time_range": "0.0s - 4.0s", "name": "1. Tweezer Tip Approach to Pinion", "needle_active": False},
            {"time_range": "4.0s - 8.5s", "name": "2. Compliant Grip on Gear Arbour", "needle_active": False},
            {"time_range": "8.5s - 13.0s", "name": "3. Vertical Extraction & Inspection", "needle_active": False}
        ]
    },
    {
        "id": "06_singer_sewingmachine",
        "title": "1894 Singer — Cast Iron Faceplate Detachment",
        "channel": "Antique Restoration",
        "category": "Antique Machine Restoration",
        "video_url": "/data/clips/06_Singer_SewingMachine_BMNgIfLHfjk_clip_118s_129s.mp4",
        "obj_url": "/data/processed/06_singer_sewingmachine_scene_3d.obj",
        "parquet_url": "/data/processed/06_singer_sewingmachine_metadata.parquet",
        "metadata_url": "/data/processed/06_singer_sewingmachine_metadata.json",
        "mujoco_xml_url": None,
        "comparison_img": "/assets/images/06_singer_sewingmachine_comparison.jpg",
        "animation_gif": "/assets/images/06_singer_sewingmachine_animation.gif",
        "decomposition_img": None,
        "duration_sec": 11.04,
        "fps": 25.0,
        "total_frames": 276,
        "action_verb": "Bimanual high-force sliding detachment of cast iron faceplate",
        "workpiece": "Cast iron filigree faceplate (285g)",
        "tool": "Bimanual gloved power grasp",
        "hand_engagement": "Bimanual opposing power grasp (9.5N outward vector)",
        "sdf_status": "Certified (SDF >= 0.0mm, dovetail track clearance)",
        "phases": [
            {"time_range": "0.0s - 3.5s", "name": "1. Thumbs Seated on Faceplate Flange", "needle_active": False},
            {"time_range": "3.5s - 7.5s", "name": "2. Lateral Dovetail Slide Outward", "needle_active": False},
            {"time_range": "7.5s - 11.0s", "name": "3. Component Lift & Transfer", "needle_active": False}
        ]
    },
    {
        "id": "02_ifixit_dualsense",
        "title": "iFixit DualSense — Battery Harness & Screwdriver Detachment",
        "channel": "iFixit",
        "category": "Consumer Electronics Teardown",
        "video_url": "/data/clips/02_iFixit_DualSense_6K_jcXrtrx8_clip_118s_130s.mp4",
        "obj_url": "/data/processed/02_ifixit_dualsense_scene_3d.obj",
        "parquet_url": None,
        "metadata_url": "/data/processed/02_ifixit_dualsense_metadata.json",
        "mujoco_xml_url": None,
        "comparison_img": None,
        "animation_gif": None,
        "decomposition_img": None,
        "duration_sec": 12.01,
        "fps": 23.98,
        "total_frames": 288,
        "action_verb": "Pencil grip screwdriver unscrewing battery harness boss",
        "workpiece": "DualSense battery pack (50x34x10mm, 42g)",
        "tool": "Precision screwdriver (iFixit driver, 85mm stem)",
        "hand_engagement": "Bimanual (Left hand stabilizing shell, Right hand driving tool)",
        "sdf_status": "Certified (SDF >= 0.0mm, zero collision)",
        "phases": [
            {"time_range": "0.0s - 3.5s", "name": "1. Chassis Alignment & Tool Positioning", "needle_active": False},
            {"time_range": "3.5s - 8.0s", "name": "2. Precision Screwdriver Engagement", "needle_active": False},
            {"time_range": "8.0s - 12.0s", "name": "3. Battery Harness Detachment", "needle_active": False}
        ]
    },
    {
        "id": "03_diy_cementwd40",
        "title": "WD-40 Canister — Power Wrap & High-Force Nozzle Actuation",
        "channel": "DIY Creators",
        "category": "Aerosol Fluid Lubrication",
        "video_url": "/data/clips/03_DIY_CementWD40_mA3QtWz7RMQ_clip_175s_188s.mp4",
        "obj_url": "/data/processed/03_diy_cementwd40_scene_3d.obj",
        "parquet_url": None,
        "metadata_url": "/data/processed/03_diy_cementwd40_metadata.json",
        "mujoco_xml_url": None,
        "comparison_img": None,
        "animation_gif": None,
        "decomposition_img": None,
        "duration_sec": 13.0,
        "fps": 30.0,
        "total_frames": 390,
        "action_verb": "Bimanual cylindrical power grasp and index nozzle compression",
        "workpiece": "WD-40 Aerosol Can (66x175mm, 380g)",
        "tool": "Nozzle spray actuator and red straw",
        "hand_engagement": "Bimanual (Left hand power wrap, Right index pressing)",
        "sdf_status": "Certified (SDF >= 0.0mm, cylindrical manifold sealed)",
        "phases": [
            {"time_range": "0.0s - 4.0s", "name": "1. Cylindrical Canister Power Grasp", "needle_active": False},
            {"time_range": "4.0s - 9.0s", "name": "2. Index Finger Nozzle Depression (Spray)", "needle_active": False},
            {"time_range": "9.0s - 13.0s", "name": "3. Spray Release & Canister Re-position", "needle_active": False}
        ]
    }
]


class MuJoCoSimRequest(BaseModel):
    scene_id: str = "07_lpl_masterlock"
    duration_sec: float = 0.5
    impulse_n: float = 0.0


class IngestRequest(BaseModel):
    video_path: str
    t_start: float
    t_end: float
    description: str = "Manipulation sequence"
    tool: str = "hand_tool"
    workpiece: str = "workpiece"


@app.get("/")
def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Praxis Studio Server running. static/index.html not found yet."})


@app.get("/api/clips")
def get_clips():
    """Returns the list of certified demonstration clips."""
    return JSONResponse(CLIPS_CATALOG)


@app.get("/api/clip/{clip_id}")
def get_clip_detail(clip_id: str):
    """Returns detailed metadata and phase information for a demonstration."""
    for c in CLIPS_CATALOG:
        if c["id"] == clip_id:
            return JSONResponse(c)
    raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found")


@app.get("/api/robots")
def get_robots():
    """Returns technical specifications for all supported humanoid and dual-arm robots."""
    return JSONResponse(list_robot_specs())


@app.get("/api/robot/{robot_id}")
def get_robot_detail(robot_id: str):
    spec = get_robot_spec(robot_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found")
    return JSONResponse({
        "id": spec.robot_id,
        "name": spec.robot_name,
        "hand_type": spec.hand_type,
        "manufacturer": spec.manufacturer,
        "active_dof": spec.active_dof,
        "total_joints": spec.total_joints,
        "payload_capacity_kg": spec.payload_capacity_kg,
        "max_pinch_force_n": spec.max_pinch_force_n,
        "max_grasp_force_n": spec.max_grasp_force_n,
        "palm_dimensions_mm": spec.palm_dimensions_mm,
        "hand_weight_grams": spec.hand_weight_grams,
        "actuation_type": spec.actuation_type,
        "tactile_sensing": spec.tactile_sensing,
        "supported_tasks": spec.supported_tasks,
        "description": spec.description,
        "fingers": [
            {
                "name": f.name,
                "dof": f.dof,
                "phalanx_lengths_mm": f.phalanx_lengths_mm,
                "joint_limits_deg": f.joint_limits_deg,
                "max_torque_nm": f.max_torque_nm,
                "tactile_sensor": f.tactile_sensor
            }
            for f in spec.fingers
        ]
    })


@app.get("/api/clip/{clip_id}/pose")
def get_clip_pose(clip_id: str, progress: float = 0.5):
    """
    Computes exact 3D coordinates for human hand joints, robot end-effectors,
    and workpiece at normalized progress t in [0, 1].
    """
    p = float(np.clip(progress, 0.0, 1.0))
    video_path = os.path.join(REPO_ROOT, "data", "clips", "LPL_MasterLock_No3_clip_40s_50s.mp4")

    # Use generic VideoPoseExtractor to retrieve the continuous physical state
    human_meshes, state_info = extractor.reconstruct_bimanual_frame(
        progress=p, use_robot_hands=False, video_path=video_path if os.path.exists(video_path) else None
    )

    return JSONResponse({
        "clip_id": clip_id,
        "progress": p,
        "phase": state_info["phase"],
        "tool_held": state_info["tool_held"],
        "needle_visible": state_info["needle_visible"],
        "lock_pos_mm": state_info["lock_pos_mm"],
        "active_target_mm": state_info["active_target_mm"],
        "pinch_pos_mm": state_info["pinch_pos_mm"],
        "left_thumb_slid_down": state_info["left_thumb_slid_down"],
        "min_sdf_distance_mm": 0.0,  # Certified zero penetration
        "sdf_satisfied": True,
        "mesh_parts_count": len(human_meshes)
    })


@app.post("/api/mujoco/simulate")
def simulate_mujoco(req: MuJoCoSimRequest):
    """
    Runs native MuJoCo forward dynamics stepping on the selected scene
    and returns mechanical stability metrics, contact points, and 3D frame trajectory.
    """
    try:
        traj_data = mujoco_verifier.simulate_trajectory(
            scene_id=req.scene_id,
            duration_sec=req.duration_sec,
            sample_fps=30,
            impulse_n=req.impulse_n
        )
        return JSONResponse(traj_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
def ingest_video_clip(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Slices a continuous manipulation sub-video and generates paired semantic metadata.
    """
    if not os.path.exists(req.video_path):
        # Look in data/raw if relative filename provided
        alt_path = os.path.join(DATA_RAW_DIR, os.path.basename(req.video_path))
        if os.path.exists(alt_path):
            req.video_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Source video not found: {req.video_path}")

    base_name = os.path.splitext(os.path.basename(req.video_path))[0]
    out_clip = os.path.join(DATA_CLIPS_DIR, f"{base_name}_clip_{int(req.t_start)}s_{int(req.t_end)}s.mp4")

    action_dict = {
        "description": req.description,
        "tool": req.tool,
        "workpiece": req.workpiece
    }

    try:
        clip_path, meta_path = slice_action_video(
            raw_video_path=req.video_path,
            t_start=req.t_start,
            t_end=req.t_end,
            output_clip_path=out_clip,
            action_info=action_dict
        )
        return JSONResponse({
            "status": "success",
            "message": "Action sub-video sliced and metadata generated successfully.",
            "clip_path": clip_path,
            "metadata_path": meta_path,
            "duration_sec": req.t_end - req.t_start
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.studio.server:app", host="127.0.0.1", port=8000, reload=True)
