import os
import pytest
import numpy as np
import trimesh

from src.pose_extraction.reconstruct_dropkey_scene import build_dropkey_scene, create_cylinder, create_box

def test_primitive_creation():
    cyl = create_cylinder(radius=5.0, height=50.0)
    assert isinstance(cyl, trimesh.Trimesh)
    assert len(cyl.vertices) > 0
    assert len(cyl.faces) > 0
    # Expected volume: pi * r^2 * h = 3.14159 * 25 * 50 ~ 3927
    assert 3800 < cyl.volume < 4100
    
    box = create_box(extents=[10, 20, 30])
    assert isinstance(box, trimesh.Trimesh)
    assert abs(box.volume - 6000.0) < 1.0

def test_build_dropkey_scene_generation(tmp_path):
    out_dir = str(tmp_path / "processed_test")
    obj_path, meta_path = build_dropkey_scene(output_dir=out_dir)
    
    assert os.path.exists(obj_path)
    assert os.path.exists(meta_path)
    assert obj_path.endswith(".obj")
    assert meta_path.endswith(".json")
    
    loaded_scene = trimesh.load(obj_path)
    assert len(loaded_scene.vertices) > 100
    assert len(loaded_scene.faces) > 100

from src.physics_sanitizer.sdf import sdf_cylinder, sdf_box, sdf_sphere

def test_sdf_non_penetration_constraint():
    """
    Verifies that the Signed Distance Field (SDF) of a solid rigid cylinder
    accurately detects whether external points are strictly outside or inside.
    SDF(point) > 0 for outside, < 0 for inside penetration, = 0 on boundary.
    """
    radius = 10.0
    half_height = 50.0
    
    # Point outside the cylinder along X (radius 15 > 10)
    outside_pt = np.array([[15.0, 0.0, 0.0]])
    dist_outside = sdf_cylinder(outside_pt, radius, half_height)
    assert abs(dist_outside[0] - 5.0) < 1e-4
    
    # Point inside the cylinder (radius 5 < 10)
    inside_pt = np.array([[5.0, 0.0, 0.0]])
    dist_inside = sdf_cylinder(inside_pt, radius, half_height)
    assert dist_inside[0] < 0.0
    assert abs(dist_inside[0] - (-5.0)) < 1e-4
    
    # Point exactly on surface (radius 10)
    surface_pt = np.array([[10.0, 0.0, 20.0]])
    dist_surface = sdf_cylinder(surface_pt, radius, half_height)
    assert abs(dist_surface[0]) < 1e-4
    
    # Box SDF
    box_he = [10.0, 20.0, 30.0]
    assert sdf_box([[15.0, 0.0, 0.0]], box_he)[0] > 0
    assert sdf_box([[5.0, 5.0, 5.0]], box_he)[0] < 0
