"""
Unit tests for the Visual Inverse Dynamics Engine.
"""

import numpy as np
import pytest

from src.dynamics.inverse_dynamics import InverseDynamicsEngine, MATERIAL_DENSITIES_G_CM3

def test_material_densities_constants():
    assert "stainless_steel" in MATERIAL_DENSITIES_G_CM3
    assert "brass" in MATERIAL_DENSITIES_G_CM3
    assert "cast_iron" in MATERIAL_DENSITIES_G_CM3
    assert MATERIAL_DENSITIES_G_CM3["brass"] > MATERIAL_DENSITIES_G_CM3["aluminum"]

def test_estimate_mass_grams():
    engine = InverseDynamicsEngine()
    # 1000 mm3 = 1 cm3 of stainless steel (~7.9g)
    mass_steel = engine.estimate_mass_grams(1000.0, "stainless_steel")
    assert pytest.approx(mass_steel, rel=1e-2) == 7.90

    # 1000 mm3 of brass (~8.5g)
    mass_brass = engine.estimate_mass_grams(1000.0, "brass")
    assert pytest.approx(mass_brass, rel=1e-2) == 8.50

def test_inertia_tensor_cylinder():
    engine = InverseDynamicsEngine()
    mass_kg = 0.05
    dims = {"radius": 4.0, "length": 120.0}
    I = engine.compute_inertia_tensor(mass_kg, "cylinder", dims)
    assert I.shape == (3, 3)
    assert I[0, 0] > 0
    assert I[1, 1] == I[0, 0] # Symmetry for cylinder along Z
    assert I[2, 2] > 0
    # Off-diagonal elements zero for principal axes
    assert I[0, 1] == 0.0
    assert I[0, 2] == 0.0

def test_inertia_tensor_box():
    engine = InverseDynamicsEngine()
    mass_kg = 0.30
    dims = {"width": 10.0, "depth": 50.0, "height": 100.0}
    I = engine.compute_inertia_tensor(mass_kg, "box", dims)
    assert I.shape == (3, 3)
    assert I[0, 0] > 0
    assert I[1, 1] > 0
    assert I[2, 2] > 0

def test_trajectory_dynamics_analysis():
    engine = InverseDynamicsEngine(gravity=9.81, friction_coefficient=0.5, safety_margin=1.2)
    t = np.linspace(0, 1, 30)
    # Simple harmonic vertical motion
    z = 0.1 * np.sin(2 * np.pi * t)
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    traj = np.column_stack([x, y, z])

    profile = engine.analyze_trajectory(traj, mass_grams=100.0, fps=30.0, num_contact_points=2)
    assert len(profile.timestamps) == 30
    assert profile.positions.shape == (30, 3)
    assert profile.velocities.shape == (30, 3)
    assert profile.accelerations.shape == (30, 3)
    assert profile.peak_acceleration_m_s2 > 0.0
    assert profile.mean_speed_m_s > 0.0
    assert profile.max_required_grip_n > 0.0
    assert len(profile.required_grip_force_newtons) == 30
    assert profile.continuity_report is not None
    assert profile.continuity_report.is_continuous is True
    assert profile.physics_tokens is not None
    assert len(profile.physics_tokens.feature_vector) == 5


from src.dynamics.inverse_dynamics import (
    PhysicalContinuityAuditor,
    GraspAffordanceAuditor,
    PhysicsTokenGenerator,
    GraspType,
    MassClass,
)


def test_interframe_continuity_auditor():
    """Verifies that smooth trajectories pass and teleportation cuts are rejected."""
    dt = 1.0 / 30.0 # 30 fps
    t = np.linspace(0, 1.0, 31)

    # 1. Smooth fine manipulation trajectory (speed ~ 0.3 m/s < 2.0 m/s)
    smooth_traj = np.column_stack([0.05 * np.cos(2 * np.pi * t), 0.05 * np.sin(2 * np.pi * t), t * 0.1])
    report_smooth = PhysicalContinuityAuditor.audit_trajectory_continuity(smooth_traj, dt=dt)
    assert report_smooth.is_continuous is True
    assert report_smooth.max_velocity_m_s < 2.0
    assert report_smooth.continuity_score > 0.8

    # 2. Teleportation jump (tracking glitch: jump of 0.25m in one frame dt=0.033s -> 7.5 m/s > 2.0 m/s)
    jump_traj = smooth_traj.copy()
    jump_traj[15] += np.array([0.25, 0.0, 0.0]) # Teleportation
    report_jump = PhysicalContinuityAuditor.audit_trajectory_continuity(jump_traj, dt=dt)
    assert report_jump.is_continuous is False
    assert report_jump.max_velocity_m_s > 2.0
    assert report_jump.rejection_reason is not None
    assert "exceeds max manipulation limit" in report_jump.rejection_reason


def test_grasp_affordance_auditor():
    """Verifies grasp coherence against Coulomb friction cone and mass."""
    # 1. Precision pinch holding 150g -> feasible
    res_pinch_light = GraspAffordanceAuditor.evaluate_grasp(GraspType.PRECISION_PINCH, mass_grams=150.0)
    assert res_pinch_light.is_physically_feasible is True

    # 2. Precision pinch attempting to hold 4000g (4 kg) -> unfeasible (exceeds human pinch grip friction cone)
    res_pinch_heavy = GraspAffordanceAuditor.evaluate_grasp(GraspType.PRECISION_PINCH, mass_grams=4000.0)
    assert res_pinch_heavy.is_physically_feasible is False
    assert "exceeds maximum payload" in res_pinch_heavy.rejection_reason

    # 3. Palmar cradle holding 4000g (4 kg) -> feasible because palm supports from underneath
    res_cradle_heavy = GraspAffordanceAuditor.evaluate_grasp(GraspType.PALMAR_CRADLE, mass_grams=4000.0)
    assert res_cradle_heavy.is_physically_feasible is True


def test_physics_token_generation():
    """Verifies that physics tokens for foundation models are correctly structured."""
    tokens = PhysicsTokenGenerator.generate_tokens(
        mass_grams=250.0,
        grasp_type=GraspType.PRECISION_PINCH,
        continuity_score=0.95,
        is_rigid=True
    )
    assert tokens.mass_class == MassClass.MEDIUM_TOOL
    assert tokens.target_stiffness_k_n_m > 0
    assert len(tokens.feature_vector) == 5
    d = tokens.to_dict()
    assert d["mass_class"] == "medium_tool"
    assert d["grasp_type"] == "precision_pinch"
    assert d["continuity_score"] == 0.95

