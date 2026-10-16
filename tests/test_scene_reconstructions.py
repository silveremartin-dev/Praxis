"""
Unit tests for 3D digital twin scene reconstructions (Watch Zodiac and Singer Sewing Machine).
"""

import json
import os
import tempfile
import trimesh
import pytest

from src.pose_extraction.reconstruct_watch_scene import build_watch_scene
from src.pose_extraction.reconstruct_singer_scene import build_singer_scene
from src.pose_extraction.reconstruct_masterlock_scene import build_masterlock_scene

def test_build_watch_scene_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path, meta_path = build_watch_scene(output_dir=tmpdir)
        
        assert os.path.exists(obj_path)
        assert os.path.exists(meta_path)
        
        # Validate 3D mesh
        mesh = trimesh.load(obj_path)
        assert len(mesh.vertices) > 500
        assert len(mesh.faces) > 1000
        
        # Validate JSON metadata
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

