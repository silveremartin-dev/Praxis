"""
Unit Tests for Shot Continuity, Active Hand Presence & Goal-Directed Irreversibility (`tests/ingestion/test_shot_detector.py`).
"""

import numpy as np
import pytest
import cv2

from src.ingestion.shot_detector import (
    ShotBoundaryDetector,
    HandManipulationDetector,
    GoalDirectedActionValidator,
    GoalActionVerb,
)


@pytest.fixture
def single_shot_video(tmp_path):
    """Creates a continuous synthetic video without any camera cut."""
    v_path = str(tmp_path / "continuous_shot.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    w, h = 320, 240
    out = cv2.VideoWriter(v_path, fourcc, fps, (w, h))

    # 60 frames of smoothly varying gradient video
    for i in range(60):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(50, 150, w, dtype=np.uint8)
        frame[:, :, 1] = int(100 + 10 * np.sin(i / 10.0))
        frame[:, :, 2] = 80
        out.write(frame)
    out.release()
    return v_path


@pytest.fixture
def cut_shot_video(tmp_path):
    """Creates a synthetic video containing an abrupt camera cut at frame 30."""
    v_path = str(tmp_path / "camera_cut_shot.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    w, h = 320, 240
    out = cv2.VideoWriter(v_path, fourcc, fps, (w, h))

    # First shot: dark blue frames
    for i in range(30):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 200 # Blue
        out.write(frame)

    # Abrupt cut to bright yellow frames
    for i in range(30):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 1] = 255 # Green
        frame[:, :, 2] = 255 # Red -> Yellow
        out.write(frame)

    out.release()
    return v_path


def test_single_continuous_shot_auditor(single_shot_video):
    audit = ShotBoundaryDetector.audit_shot_continuity(single_shot_video, t_start=0.0, t_end=2.0)
    assert audit.is_single_continuous_shot is True
    assert audit.num_cuts_detected == 0
    assert audit.rejection_reason is None


def test_camera_cut_detection_rejection(cut_shot_video):
    audit = ShotBoundaryDetector.audit_shot_continuity(cut_shot_video, t_start=0.0, t_end=2.0)
    assert audit.is_single_continuous_shot is False
    assert audit.num_cuts_detected >= 1
    assert audit.rejection_reason is not None
    assert "Camera framing cut detected" in audit.rejection_reason


def test_goal_directed_action_validation():
    # 1. Valid goal-directed manipulation (open padlock)
    audit_open = GoalDirectedActionValidator.validate_action(
        action_verb="open",
        pre_condition="padlock_locked_closed",
        post_condition="padlock_unlatched_open"
    )
    assert audit_open.is_goal_directed is True
    assert audit_open.is_state_irreversible is True
    assert audit_open.rejection_reason is None

    # 2. Valid goal-directed manipulation (screw fastener)
    audit_screw = GoalDirectedActionValidator.validate_action(
        action_verb="screw",
        pre_condition="screw_loose",
        post_condition="screw_tightened_into_thread"
    )
    assert audit_screw.is_goal_directed is True
    assert audit_screw.is_state_irreversible is True

    # 3. Reversible idle action (hand waving / tapping) -> must be rejected
    audit_wave = GoalDirectedActionValidator.validate_action(
        action_verb="wave",
        pre_condition="hand_in_air",
        post_condition="hand_in_air"
    )
    assert audit_wave.is_goal_directed is False
    assert "reversible/idle movement" in audit_wave.rejection_reason

    # 4. Action with identical pre/post condition -> zero net physical work -> rejected
    audit_stagnant = GoalDirectedActionValidator.validate_action(
        action_verb="turn",
        pre_condition="dial_at_position_0",
        post_condition="dial_at_position_0"
    )
    assert audit_stagnant.is_state_irreversible is False
    assert "identical" in audit_stagnant.rejection_reason
