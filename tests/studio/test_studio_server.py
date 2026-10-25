"""
Unit tests for Praxis Studio FastAPI backend (`tests/studio/test_studio_server.py`).
Verifies:
1. Static web page serving (index.html).
2. Demonstration clips catalog and phase retrieval.
3. Hardware robot specifications (Unitree H1, Tesla Optimus, ALOHA, Franka).
4. Continuous 3D pose extraction endpoints.
5. Native MuJoCo forward dynamic stability execution.
"""

import pytest
from fastapi.testclient import TestClient
from src.studio.server import app

client = TestClient(app)


def test_get_index_html():
    """Verify that the root endpoint serves the Praxis Studio single-page application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "PRAXIS STUDIO" in response.text
    assert "Three.js" in response.text or "three.min.js" in response.text


def test_get_clips_catalog():
    """Verify demonstration clips catalog returns 4 certified sequences."""
    response = client.get("/api/clips")
    assert response.status_code == 200
    clips = response.json()
    assert isinstance(clips, list)
    assert len(clips) >= 4

    clip_ids = [c["id"] for c in clips]
    assert "07_lpl_masterlock" in clip_ids
    assert "01_lpl_dropkey" in clip_ids
    assert "05_watch_zodiac" in clip_ids
    assert "06_singer_sewingmachine" in clip_ids


def test_get_clip_detail():
    """Verify detailed metadata and phase task graph for Master Lock."""
    response = client.get("/api/clip/07_lpl_masterlock")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "07_lpl_masterlock"
    assert "Master Lock" in data["workpiece"]
    assert len(data["phases"]) == 5
    assert data["duration_sec"] == 10.0


def test_get_clip_not_found():
    """Verify 404 response for unknown clip identifier."""
    response = client.get("/api/clip/non_existent_clip_123")
    assert response.status_code == 404


def test_get_robots_list():
    """Verify list of supported robot embodiments."""
    response = client.get("/api/robots")
    assert response.status_code == 200
    robots = response.json()
    assert len(robots) >= 4
    ids = [r["id"] for r in robots]
    assert "unitree_h1" in ids
    assert "tesla_optimus" in ids
    assert "aloha_viperx" in ids
    assert "franka_panda" in ids


def test_get_unitree_h1_specs():
    """Verify realistic hardware parameters for Unitree H1 Inspire DexHand."""
    response = client.get("/api/robot/unitree_h1")
    assert response.status_code == 200
    data = response.json()
    assert data["active_dof"] == 6
    assert data["total_joints"] == 12
    assert data["max_pinch_force_n"] == 10.0
    assert len(data["fingers"]) == 5
    # Verify thumb has 2 active DoF
    thumb = next(f for f in data["fingers"] if f["name"] == "Thumb")
    assert thumb["dof"] == 2


def test_get_tesla_optimus_specs():
    """Verify authentic hardware parameters for Tesla Optimus Hand Gen-2."""
    response = client.get("/api/robot/tesla_optimus")
    assert response.status_code == 200
    data = response.json()
    assert data["active_dof"] == 11
    assert data["total_joints"] == 16
    assert data["payload_capacity_kg"] == 10.0
    assert "tendon" in data["actuation_type"].lower()
    assert "tactile" in data["tactile_sensing"].lower()
    # Verify fingers have distributed pressure sensing
    index_f = next(f for f in data["fingers"] if f["name"] == "Index")
    assert "pressure" in index_f["tactile_sensor"].lower()


def test_get_unknown_robot():
    """Verify 404 for unknown robot ID."""
    response = client.get("/api/robot/mars_rover_gripper")
    assert response.status_code == 404


def test_get_clip_pose_extraction():
    """Verify 3D pose extraction endpoint along the normalized trajectory."""
    response = client.get("/api/clip/07_lpl_masterlock/pose?progress=0.45")
    assert response.status_code == 200
    pose = response.json()
    assert pose["progress"] == 0.45
    assert pose["min_sdf_distance_mm"] >= 0.0
    assert pose["mesh_parts_count"] > 0
    assert "lock_pos_mm" in pose


def test_mujoco_sim_endpoint():
    """Verify native MuJoCo physics forward dynamic simulation via HTTP API."""
    payload = {
        "scene_id": "07_lpl_masterlock",
        "duration_sec": 0.2,
        "impulse_n": 0.0
    }
    response = client.post("/api/mujoco/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == "07_lpl_masterlock"
    assert data["is_dynamically_stable"] is True
    assert data["numerical_explosion_detected"] is False
    assert data["max_velocity_m_s"] < 10.0
    assert "MuJoCo 3" in data["physics_engine"]


def test_dynamic_port_finder():
    """Verify that find_available_port detects conflict-free ports."""
    from src.studio.run import find_available_port
    port = find_available_port(start_port=8055)
    assert isinstance(port, int)
    assert 8055 <= port <= 8155
