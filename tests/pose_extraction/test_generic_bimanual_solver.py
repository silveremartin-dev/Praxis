"""
Unit Tests for Generic Automated Bimanual Solver (`tests/pose_extraction/test_generic_bimanual_solver.py`).
"""

import numpy as np
import pytest

from src.pose_extraction.generic_bimanual_solver import GenericBimanualSolver, GenericSceneSolution


def test_generic_bimanual_solver_derives_scale_and_dimensions():
    solver = GenericBimanualSolver(hand_prior_length_mm=184.0)

    hand_detections = [
        {"chirality": "left", "bbox_px": (100.0, 200.0, 250.0, 420.0)},
        {"chirality": "right", "bbox_px": (500.0, 180.0, 680.0, 400.0)},
    ]
    # Workpiece in center of image
    workpiece_bbox_px = (320.0, 250.0, 420.0, 350.0)

    sol = solver.solve_scene_from_observations(
        hand_detections=hand_detections,
        workpiece_bbox_px=workpiece_bbox_px,
        workpiece_geometry_type="box",
        target_embodiment="human",
    )

    assert sol.is_valid is True
    assert sol.left_hand_present is True
    assert sol.right_hand_present is True
    assert sol.scale_mm_per_px > 0.4
    assert sol.workpiece_dimensions_mm["width"] > 20.0
    assert sol.workpiece_dimensions_mm["height"] > 20.0
    assert sol.max_penetration_mm <= 0.05
    assert len(sol.left_hand_joints) == 21
    assert len(sol.right_hand_joints) == 21
    assert len(sol.reconstructed_meshes) > 0


def test_generic_bimanual_solver_rejects_duplicate_chirality():
    solver = GenericBimanualSolver()

    # Two right hands (unphysical duplicate hallucination)
    hand_detections = [
        {"chirality": "right", "bbox_px": (100.0, 200.0, 250.0, 420.0)},
        {"chirality": "right", "bbox_px": (500.0, 180.0, 680.0, 400.0)},
    ]
    workpiece_bbox_px = (320.0, 250.0, 420.0, 350.0)

    sol = solver.solve_scene_from_observations(
        hand_detections=hand_detections,
        workpiece_bbox_px=workpiece_bbox_px,
    )

    assert sol.is_valid is False
    assert sol.max_penetration_mm == 999.0
