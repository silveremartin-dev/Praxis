"""
Unit tests for 3D digital twin scene reconstructions and generic VideoPoseExtractor.
"""

import json
import os
import tempfile
import trimesh
import pytest

from tests.pose_extraction.reconstruct_watch_scene import build_watch_scene
from tests.pose_extraction.reconstruct_singer_scene import build_singer_scene
from tests.pose_extraction.reconstruct_masterlock_scene import build_masterlock_scene
from src.pose_extraction.extractor import VideoPoseExtractor


def test_build_watch_scene_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path, meta_path = build_watch_scene(output_dir=tmpdir)
        assert os.path.exists(obj_path)
        assert os.path.exists(meta_path)
        mesh = trimesh.load(obj_path)
        assert len(mesh.vertices) > 500
        assert len(mesh.faces) > 1000
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset_name"] == "Praxis_V2PDT_Horology_05"
        assert len(data["rigid_objects_3d"]) >= 3
        assert data["lerobot_export_compatibility"]["verified_zero_penetration"] is True


def test_build_singer_scene_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path, meta_path = build_singer_scene(output_dir=tmpdir)
        assert os.path.exists(obj_path)
        assert os.path.exists(meta_path)
        mesh = trimesh.load(obj_path)
        assert len(mesh.vertices) > 500
        assert len(mesh.faces) > 1000
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset_name"] == "Praxis_V2PDT_AntiqueRestoration_06"
        assert "singer_cast_iron_faceplate" in [o["id"] for o in data["rigid_objects_3d"]]
        assert data["lerobot_export_compatibility"]["urdf_retargetable"] is True


def test_build_masterlock_scene_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path, meta_path = build_masterlock_scene(output_dir=tmpdir)
        assert os.path.exists(obj_path)
        assert os.path.exists(meta_path)
        mesh = trimesh.load(obj_path)
        assert len(mesh.vertices) > 500
        assert len(mesh.faces) > 1000
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset_name"] == "Praxis_V2PDT_LockBypass_07"
        assert "master_lock_no3_body" in [o["id"] for o in data["rigid_objects_3d"]]
        assert data["lerobot_export_compatibility"]["verified_zero_penetration"] is True


def test_generic_video_pose_extractor_phases_and_robot_hands():
    extractor = VideoPoseExtractor()
    # Phase 2: Horizontal needle slide (needle visible & held)
    meshes_h, state_h = extractor.reconstruct_bimanual_frame(progress=0.25, use_robot_hands=False)
    assert state_h["needle_visible"] is True
    assert state_h["tool_held"] is True
    assert state_h["left_thumb_slid_down"] is False
    assert len(meshes_h) > 40

    # Phase 5: Needle dropped, left thumb slid down, right index pointing, with 5-finger robot hands
    meshes_r, state_r = extractor.reconstruct_bimanual_frame(progress=0.80, use_robot_hands=True)
    assert state_r["needle_visible"] is False
    assert state_r["tool_held"] is False
    assert state_r["left_thumb_slid_down"] is True
    assert len(meshes_r) > 40
