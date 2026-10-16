"""
Unit tests for the URDF Kinematic Feasibility Auditor.
"""

import tempfile
import numpy as np
import pytest

from src.feasibility.urdf_auditor import URDFAuditor, ROBOT_REGISTRY, RobotModelSpec

def test_robot_registry_presets():
    assert "aloha_viperx" in ROBOT_REGISTRY
    assert "unitree_h1" in ROBOT_REGISTRY
    assert "franka_panda" in ROBOT_REGISTRY
    
    aloha = ROBOT_REGISTRY["aloha_viperx"]
    assert aloha.dof == 6
    assert aloha.reach_max_m == 0.75
    
    h1 = ROBOT_REGISTRY["unitree_h1"]
    assert h1.dof == 7
    assert h1.reach_max_m == 0.85

def test_urdf_auditor_initialization():
    auditor = URDFAuditor("aloha_viperx")
    assert auditor.robot_spec.name == "aloha_viperx"
    assert auditor.acceptance_threshold == 85.0

    with pytest.raises(ValueError):
        URDFAuditor("unsupported_robot_xyz")

def test_audit_trajectory_feasible_reach():
    auditor = URDFAuditor("aloha_viperx", acceptance_threshold=85.0)
    # Feasible trajectory within ALOHA workspace (r ~ 0.45m)
    t = np.linspace(0, 1, 20)
    traj = np.column_stack([
        0.35 + 0.05 * np.cos(t),
        0.05 * np.sin(t),
        0.20 + 0.02 * t
    ])
    
    report = auditor.audit_trajectory(traj, dt=1.0/30.0)
    assert report.total_frames == 20
    assert report.feasible_frames == 20
    assert report.feasibility_score_percent == 100.0
    assert report.is_accepted is True
    assert report.mean_manipulability > 0.0

def test_audit_trajectory_exceeds_max_reach_rejection():
    auditor = URDFAuditor("aloha_viperx", acceptance_threshold=85.0)
    # Trajectory at 1.5m, well beyond 0.75m max reach
    traj = np.column_stack([
        np.ones(15) * 1.5,
        np.zeros(15),
        np.ones(15) * 0.3
    ])
    
    report = auditor.audit_trajectory(traj)
    assert report.feasible_frames == 0
    assert report.feasibility_score_percent == 0.0
    assert report.is_accepted is False
    assert "EXCEEDS_MAX_REACH" in report.violation_summary

def test_audit_trajectory_inside_min_reach():
    auditor = URDFAuditor("aloha_viperx")
    # Trajectory at 0.05m, inside 0.15m collision zone with base
    traj = np.column_stack([
        np.ones(10) * 0.05,
        np.zeros(10),
        np.ones(10) * 0.05
    ])
    
    report = auditor.audit_trajectory(traj)
    assert "INSIDE_MIN_REACH" in report.violation_summary
    assert report.is_accepted is False

def test_from_urdf_file_parser():
    dummy_urdf = """<?xml version="1.0"?>
    <robot name="test_arm">
      <link name="base_link"/>
      <link name="link1"/>
      <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="link1"/>
        <limit lower="-1.57" upper="1.57" velocity="2.0"/>
      </joint>
    </robot>
    """
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(dummy_urdf)
        tmp_name = f.name
        
    try:
        auditor = URDFAuditor.from_urdf_file(tmp_name)
        assert auditor.robot_spec.name == "test_arm"
        assert auditor.robot_spec.dof == 1
        assert auditor.robot_spec.joints[0].lower_limit == -1.57
        assert auditor.robot_spec.joints[0].max_velocity == 2.0
    finally:
        import os
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
