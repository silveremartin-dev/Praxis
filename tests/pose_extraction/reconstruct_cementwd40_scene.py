"""
3D Digital Twin Reconstruction for DIY Cement & WD-40 Manipulation (`tests/pose_extraction/reconstruct_cementwd40_scene.py`).
"""

import json
import os
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pose_extraction.cad_primitives import create_box, create_cylinder
from src.pose_extraction.hand_model import AnatomicalHandModel
from src.physics_sanitizer import PhysicsSanitizer, sdf_cylinder
from src.pose_extraction.metric_anchoring import MetricScaleEstimator


def build_cementwd40_scene(output_dir: str = "data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    meshes = []

    # 1. Metric Hand-Anchored Scale: Can is ~66mm diameter, 175mm height, 380g
    can_radius = 33.0
    can_half_h = 87.5

    # 2. Metal Can Body (Dark Blue with Yellow banner)
    can_body = create_cylinder(radius=can_radius, height=can_half_h * 2.0)
    can_body.apply_translation([0.0, 0.0, can_half_h])
    can_body.visual.face_colors = [20, 60, 160, 255] # WD-40 Blue
    meshes.append((can_body, "#143ca0", 0.98))

    # Yellow brand collar
    collar = create_cylinder(radius=can_radius + 0.3, height=45.0)
    collar.apply_translation([0.0, 0.0, can_half_h + 20.0])
    collar.visual.face_colors = [245, 205, 30, 255] # WD-40 Yellow
    meshes.append((collar, "#f5cd1e", 0.99))

    # 3. Spray Actuator Nozzle (Red top valve)
    nozzle = create_cylinder(radius=12.0, height=22.0)
    nozzle.apply_translation([0.0, 0.0, can_half_h * 2.0 + 11.0])
    nozzle.visual.face_colors = [220, 35, 35, 255] # Red Actuator
    meshes.append((nozzle, "#dc2323", 0.99))

    # Red extension straw
    straw = create_cylinder(radius=1.2, height=120.0)
    R_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    straw.apply_transform(R_x)
    straw.apply_translation([0.0, -60.0, can_half_h * 2.0 + 16.0])
    straw.visual.face_colors = [230, 30, 30, 255]
    meshes.append((straw, "#e61e1e", 0.99))

    # 4. Workbench table
    table = create_box(extents=[300.0, 250.0, 10.0])
    table.apply_translation([0.0, 0.0, -5.0])
    table.visual.face_colors = [75, 80, 85, 255]
    meshes.append((table, "#4b5055", 0.95))

    # 5. Left Hand (Full cylindrical power grasp wrapping around can body)
    lh = AnatomicalHandModel(is_left_hand=True, scale=0.90)
    lh_parts = lh.build_articulated_parts(
        palm_center=np.array([-42.0, 12.0, 90.0]),
        palm_rot=trimesh.transformations.euler_matrix(0, 0.3, 0.8)[:3, :3],
        finger_curls={"index": (0.85, 0.95, 0.6), "middle": (0.9, 1.0, 0.65), "ring": (0.95, 1.05, 0.7), "pinky": (1.0, 1.1, 0.75)},
    )
    raw_hand_meshes = []
    for m, ptype in lh_parts:
        color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#dc9676"
        raw_hand_meshes.append((m, color, 0.99))

    # 6. Right Hand (Index finger pressing actuator down)
    rh = AnatomicalHandModel(is_left_hand=False, scale=0.88)
    rh_parts = rh.build_articulated_parts(
        palm_center=np.array([30.0, 25.0, 210.0]),
        palm_rot=trimesh.transformations.euler_matrix(-0.5, 0.2, -1.8)[:3, :3],
        finger_curls={"index": (0.35, 0.45, 0.25), "middle": (0.9, 1.0, 0.6), "ring": (1.0, 1.1, 0.65), "pinky": (1.05, 1.15, 0.7)},
        index_tip_target_world=np.array([0.0, 0.0, can_half_h * 2.0 + 20.0])
    )
    for m, ptype in rh_parts:
        color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#d68f70"
        raw_hand_meshes.append((m, color, 0.99))

    # 7. Automated Physics Sanitization against cylindrical can SDF
    def can_sdf(pts):
        p = np.atleast_2d(pts) - np.array([0.0, 0.0, can_half_h])
        return sdf_cylinder(p, radius=can_radius, half_height=can_half_h)

    sanitizer = PhysicsSanitizer(default_contact_margin_mm=0.0)
    sanitized_hands, audit = sanitizer.sanitize_scene_meshes(
        raw_hand_meshes, can_sdf, margin_mm=0.0, hand_length_mm=184.0 * 0.90
    )
    meshes.extend(sanitized_hands)

    # Export OBJ
    trimesh_list = []
    for m, hex_color, _ in meshes:
        mesh_copy = m.copy()
        h = hex_color.lstrip("#")
        rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [255]
        mesh_copy.visual.face_colors = rgb
        trimesh_list.append(mesh_copy)

    combined = trimesh.util.concatenate(trimesh_list)
    obj_path = os.path.join(output_dir, "03_diy_cementwd40_scene_3d.obj")
    combined.export(obj_path)
    print(f"[+] Cement & WD-40 3D scene exported: {obj_path} ({len(combined.vertices)} verts, {len(combined.faces)} faces)")

    # Metadata
    metadata = {
        "dataset_name": "Praxis_V2PDT_CementWD40_03",
        "video_source": {
            "title": "DIY Fast-Setting Cement & WD-40 Spray Lubrication",
            "clip_file": "03_DIY_CementWD40_mA3QtWz7RMQ_clip_175s_188s.mp4",
            "duration_seconds": 13.0,
            "fps": 30.0
        },
        "rigid_objects_3d": [
            {
                "id": "wd40_spray_canister",
                "category": "aerosol_canister",
                "mass_grams": 380.0,
                "dimensions_mm": {"diameter": 66.0, "height": 175.0}
            }
        ],
        "physics_audit": {
            "is_physically_valid": audit.is_physically_valid,
            "initial_max_penetration_mm": audit.initial_max_penetration_mm,
            "final_max_penetration_mm": audit.final_max_penetration_mm
        }
    }
    meta_path = os.path.join(output_dir, "03_diy_cementwd40_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return obj_path, meta_path


if __name__ == "__main__":
    build_cementwd40_scene()
