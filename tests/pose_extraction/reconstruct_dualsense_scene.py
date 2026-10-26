"""
3D Digital Twin Reconstruction & Metric Anchoring for Sony DualSense Teardown (`tests/pose_extraction/reconstruct_dualsense_scene.py`).
"""

import json
import os
import sys
import numpy as np
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pose_extraction.cad_primitives import create_box, create_cylinder
from src.pose_extraction.hand_model import AnatomicalHandModel
from src.physics_sanitizer import PhysicsSanitizer, sdf_box
from src.pose_extraction.metric_anchoring import MetricScaleEstimator


def build_dualsense_scene(output_dir: str = "data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    meshes = []

    # 1. Metric hand-anchored scale
    # Hand length in 1080p close-up: ~480 px -> scale ~0.383 mm/px
    est = MetricScaleEstimator.estimate_object_from_hand_anchor(
        hand_length_px=480.0,
        object_bbox_px=(135.0, 95.0), # Battery module ~52x36 mm
        material="abs_polymer"
    )

    # 2. DualSense Controller Chassis (White matte shell)
    # 160mm wide, 105mm depth, 45mm height
    chassis = create_box(extents=[155.0, 100.0, 42.0])
    chassis.apply_translation([0.0, 0.0, 21.0])
    chassis.visual.face_colors = [240, 242, 245, 255] # DualSense White
    meshes.append((chassis, "#f0f2f5", 0.95))

    # 3. Battery Tray (Black compartment inside chassis)
    tray = create_box(extents=[55.0, 38.0, 14.0])
    tray.apply_translation([0.0, 5.0, 32.0])
    tray.visual.face_colors = [30, 35, 40, 255] # Matte Dark Grey
    meshes.append((tray, "#1e2328", 0.98))

    # 4. Lithium-Ion Battery Pack (Silver pouch cell, 42g)
    batt = create_box(extents=[50.0, 34.0, 10.0])
    batt.apply_translation([0.0, 5.0, 37.0])
    batt.visual.face_colors = [190, 195, 200, 255] # Silver foil
    meshes.append((batt, "#bec3c8", 0.99))

    # 5. Precision Screwdriver Tool (held by Right Hand)
    # Stem along Z, tip pointing down into tray screw boss
    driver_stem = create_cylinder(radius=2.0, height=85.0)
    driver_stem.apply_translation([15.0, 20.0, 75.0])
    driver_stem.visual.face_colors = [160, 165, 170, 255]
    meshes.append((driver_stem, "#a0a5aa", 0.99))

    driver_handle = create_cylinder(radius=6.5, height=50.0)
    driver_handle.apply_translation([15.0, 20.0, 125.0])
    driver_handle.visual.face_colors = [40, 120, 220, 255] # iFixit Blue handle
    meshes.append((driver_handle, "#2878dc", 0.99))

    # 6. Left Hand (stabilizing chassis left grip)
    lh = AnatomicalHandModel(is_left_hand=True, scale=0.88)
    lh_parts = lh.build_articulated_parts(
        palm_center=np.array([-85.0, -10.0, 15.0]),
        palm_rot=trimesh.transformations.euler_matrix(0, 0.4, 0.2)[:3, :3],
        finger_curls={"index": (0.5, 0.6, 0.3), "middle": (0.6, 0.7, 0.4), "ring": (0.7, 0.8, 0.5), "pinky": (0.8, 0.8, 0.5)},
    )
    raw_hand_meshes = []
    for m, ptype in lh_parts:
        color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#dc9676"
        raw_hand_meshes.append((m, color, 0.99))

    # 7. Right Hand (precision pencil grip on iFixit screwdriver handle)
    rh = AnatomicalHandModel(is_left_hand=False, scale=0.86)
    rh_parts = rh.build_articulated_parts(
        palm_center=np.array([45.0, 32.0, 120.0]),
        palm_rot=trimesh.transformations.euler_matrix(0.3, -0.4, -0.5)[:3, :3],
        finger_curls={"index": (0.4, 0.45, 0.3), "middle": (0.6, 0.7, 0.4), "ring": (0.9, 0.9, 0.6), "pinky": (1.0, 1.0, 0.6)},
    )
    for m, ptype in rh_parts:
        color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#d68f70"
        raw_hand_meshes.append((m, color, 0.99))

    # 8. Automated Physics Sanitization (Zero Penetration against chassis)
    chassis_he = np.array([77.5, 50.0, 21.0])
    def chassis_sdf(pts):
        p = np.atleast_2d(pts) - np.array([0.0, 0.0, 21.0])
        return sdf_box(p, chassis_he)

    sanitizer = PhysicsSanitizer(default_contact_margin_mm=0.0)
    sanitized_hands, audit = sanitizer.sanitize_scene_meshes(
        raw_hand_meshes, chassis_sdf, margin_mm=0.0, hand_length_mm=184.0 * 0.88
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
    obj_path = os.path.join(output_dir, "02_ifixit_dualsense_scene_3d.obj")
    combined.export(obj_path)
    print(f"[+] DualSense 3D scene exported: {obj_path} ({len(combined.vertices)} verts, {len(combined.faces)} faces)")

    # Metadata
    metadata = {
        "dataset_name": "Praxis_V2PDT_DualSense_02",
        "video_source": {
            "title": "iFixit DualSense Controller Teardown",
            "clip_file": "02_iFixit_DualSense_6K_jcXrtrx8_clip_118s_130s.mp4",
            "duration_seconds": 12.01,
            "fps": 23.98
        },
        "metric_anchoring": {
            "scale_mm_per_pixel": est.scale_mm_per_pixel,
            "hand_length_mm": est.hand_length_mm,
            "object_dimensions_mm": est.object_dimensions_mm,
            "estimated_mass_grams": est.estimated_mass_grams
        },
        "physics_audit": {
            "is_physically_valid": audit.is_physically_valid,
            "initial_max_penetration_mm": audit.initial_max_penetration_mm,
            "final_max_penetration_mm": audit.final_max_penetration_mm
        }
    }
    meta_path = os.path.join(output_dir, "02_ifixit_dualsense_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return obj_path, meta_path


if __name__ == "__main__":
    build_dualsense_scene()
