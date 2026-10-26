"""
Unit Tests for Hand Chirality and Convergence Auditor (`tests/pose_extraction/test_chirality_auditor.py`).
"""

import numpy as np
import pytest

from src.pose_extraction.chirality_auditor import (
    HandChiralityAuditor,
    ChiralityAuditResult,
    ConvergenceAuditResult,
    InterHandClearanceResult,
)


def test_chirality_counts_valid_bimanual():
    res = HandChiralityAuditor.audit_chirality_counts(["left", "right"])
    assert res.is_valid is True
    assert res.num_left_hands == 1
    assert res.num_right_hands == 1


def test_chirality_counts_valid_unimanual():
    res_lh = HandChiralityAuditor.audit_chirality_counts(["left"])
    assert res_lh.is_valid is True
    assert res_lh.num_left_hands == 1
    assert res_lh.num_right_hands == 0

    res_rh = HandChiralityAuditor.audit_chirality_counts(["right_hand"])
    assert res_rh.is_valid is True
    assert res_rh.num_left_hands == 0
    assert res_rh.num_right_hands == 1


def test_chirality_counts_duplicate_left_hands_rejected():
    res = HandChiralityAuditor.audit_chirality_counts(["left", "left"])
    assert res.is_valid is False
    assert res.num_left_hands == 2
    assert "Duplicate Left Hand" in res.error_message


def test_chirality_counts_duplicate_right_hands_rejected():
    res = HandChiralityAuditor.audit_chirality_counts(["right", "right", "left"])
    assert res.is_valid is False
    assert res.num_right_hands == 2
    assert "Duplicate Right Hand" in res.error_message


def test_inward_convergence_towards_workpiece():
    wp_pos = np.array([0.0, 0.0, 20.0])
    lh_pos = np.array([-40.0, -10.0, 15.0])
    rh_pos = np.array([45.0, 5.0, 18.0])

    # Left hand moving rightward (+X) toward workpiece
    lh_vel = np.array([0.2, 0.05, 0.02])
    # Right hand moving leftward (-X) toward workpiece
    rh_vel = np.array([-0.25, -0.02, 0.01])

    res = HandChiralityAuditor.audit_inward_convergence(
        lh_position=lh_pos,
        rh_position=rh_pos,
        workpiece_position=wp_pos,
        lh_velocity=lh_vel,
        rh_velocity=rh_vel,
    )
    assert res.is_convergent is True
    assert res.lh_convergence_score > 0.0
    assert res.rh_convergence_score > 0.0


def test_inter_hand_clearance_disjoint_boxes():
    lh_verts = np.array([[-50.0, 0.0, 0.0], [-20.0, 10.0, 10.0]])
    rh_verts = np.array([[15.0, 0.0, 0.0], [45.0, 10.0, 10.0]])

    res = HandChiralityAuditor.audit_inter_hand_clearance(lh_verts, rh_verts)
    assert res.is_physically_separated is True
    assert res.min_distance_mm >= 30.0
    assert res.penetration_depth_mm == 0.0


def test_inter_hand_clearance_overlapping_boxes_detected():
    lh_verts = np.array([[-30.0, 0.0, 0.0], [5.0, 10.0, 10.0]])
    rh_verts = np.array([[-2.0, 2.0, 2.0], [40.0, 12.0, 12.0]])

    res = HandChiralityAuditor.audit_inter_hand_clearance(lh_verts, rh_verts, tolerance_mm=0.5)
    assert res.is_physically_separated is False
    assert res.penetration_depth_mm > 0.5
