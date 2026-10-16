"""
Unit tests for MuJoCo physics simulation verifier.
"""

import pytest
from src.feasibility.mujoco_verifier import MuJoCoVerifier

def test_mujoco_verifier_scenes():
    verifier = MuJoCoVerifier()
    for scene_id in ["01_lpl_dropkey", "05_watch_zodiac", "06_singer_sewingmachine"]:
        res = verifier.verify_scene_stability(scene_id, duration_sec=0.2)
        assert res.numerical_explosion_detected is False
        assert res.is_dynamically_stable is True
        assert res.max_velocity_m_s < 10.0
