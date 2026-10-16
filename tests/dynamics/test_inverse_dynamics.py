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
