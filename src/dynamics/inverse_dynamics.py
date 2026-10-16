"""
Visual Inverse Dynamics Engine for Physical Digital Twin Grounding.
Infers workpiece mass, 3D inertia tensors, trajectory accelerations,
external dynamic forces, and required normal grip forces under Coulomb friction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Material densities in g/cm3 (and kg/m3 = value * 1000)
MATERIAL_DENSITIES_G_CM3: Dict[str, float] = {
    "stainless_steel": 7.90,
    "steel": 7.85,
    "brass": 8.50,
    "cast_iron": 7.20,
    "aluminum": 2.70,
    "copper": 8.96,
    "abs_polymer": 1.05,
    "polycarbonate": 1.20,
    "wood": 0.65,
    "rubber": 1.15
}

@dataclass
class DynamicsProfile:
    timestamps: np.ndarray # (N,) seconds
    positions: np.ndarray # (N, 3) meters
    velocities: np.ndarray # (N, 3) m/s
    accelerations: np.ndarray # (N, 3) m/s2
    speed: np.ndarray # (N,) m/s
    acceleration_magnitude: np.ndarray # (N,) m/s2
    dynamic_force_newtons: np.ndarray # (N, 3) N
    total_force_magnitude: np.ndarray # (N,) N
    required_grip_force_newtons: np.ndarray # (N,) N
    peak_acceleration_m_s2: float
    mean_speed_m_s: float
    max_required_grip_n: float

class InverseDynamicsEngine:
    """
    Engine to compute rigid body mass, inertia tensors, kinematic derivatives,
    and dynamic forces for manipulation trajectories.
    """
    def __init__(self, gravity: float = 9.81, friction_coefficient: float = 0.45, safety_margin: float = 1.3):
        self.g = gravity
        self.mu = friction_coefficient
        self.safety_margin = safety_margin

    @staticmethod
    def estimate_mass_grams(volume_mm3: float, material: str = "stainless_steel") -> float:
        """
        Estimates mass in grams given volume in mm3 and material name.
        """
        mat_key = material.lower()
        density_g_cm3 = MATERIAL_DENSITIES_G_CM3.get(mat_key, 7.85)
        volume_cm3 = volume_mm3 / 1000.0
        return float(volume_cm3 * density_g_cm3)

    @staticmethod
    def compute_inertia_tensor(mass_kg: float, primitive: str, dimensions_mm: Dict[str, float]) -> np.ndarray:
        """
        Computes 3x3 principal moments of inertia in kg*m^2 at center of mass.
        """
        I = np.zeros((3, 3))
        prim = primitive.lower()
        
        if "cylinder" in prim:
            r = dimensions_mm.get("radius", 5.0) / 1000.0
            h = dimensions_mm.get("height", dimensions_mm.get("length", 50.0)) / 1000.0
            # Cylinder aligned with Z axis
            I_trans = (1.0 / 12.0) * mass_kg * (3 * r**2 + h**2)
            I_axial = 0.5 * mass_kg * (r**2)
            I[0, 0] = I_trans
            I[1, 1] = I_trans
            I[2, 2] = I_axial
            
        elif "box" in prim or "plate" in prim:
            w = dimensions_mm.get("width", dimensions_mm.get("thickness", 10.0)) / 1000.0
            d = dimensions_mm.get("depth", dimensions_mm.get("width", 50.0)) / 1000.0
            h = dimensions_mm.get("height", 100.0) / 1000.0
            I[0, 0] = (1.0 / 12.0) * mass_kg * (d**2 + h**2)
            I[1, 1] = (1.0 / 12.0) * mass_kg * (w**2 + h**2)
            I[2, 2] = (1.0 / 12.0) * mass_kg * (w**2 + d**2)
            
        elif "sphere" in prim:
            r = dimensions_mm.get("radius", 10.0) / 1000.0
            I_sph = (2.0 / 5.0) * mass_kg * (r**2)
            I[0, 0] = I_sph
            I[1, 1] = I_sph
            I[2, 2] = I_sph
        else:
            # Default isotropic approximation
            dim_avg = np.mean(list(dimensions_mm.values())) / 1000.0 if dimensions_mm else 0.05
            I_iso = (1.0 / 6.0) * mass_kg * (dim_avg**2)
            I[0, 0] = I_iso
            I[1, 1] = I_iso
            I[2, 2] = I_iso
            
        return I

    def analyze_trajectory(self, trajectory_xyz_m: np.ndarray, mass_grams: float, fps: float = 30.0, num_contact_points: int = 2) -> DynamicsProfile:
        """
        Derives velocities, accelerations, and forces along a trajectory.
        trajectory_xyz_m: (N, 3) array of tool/workpiece positions in meters.
        """
        traj = np.asarray(trajectory_xyz_m, dtype=float)
        N = len(traj)
        if N < 2:
            raise ValueError("Trajectory must contain at least 2 frames")
            
        dt = 1.0 / fps
        timestamps = np.arange(N) * dt
        mass_kg = mass_grams / 1000.0

        # Numerical differentiation with central difference
        velocities = np.zeros_like(traj)
        velocities[1:-1] = (traj[2:] - traj[:-2]) / (2.0 * dt)
        velocities[0] = (traj[1] - traj[0]) / dt
        velocities[-1] = (traj[-1] - traj[-2]) / dt

        accelerations = np.zeros_like(velocities)
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2.0 * dt)
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt

        speed = np.linalg.norm(velocities, axis=1)
        acc_mag = np.linalg.norm(accelerations, axis=1)

        # Dynamic force: F_ext = m * (a - g)
        gravity_acc = np.array([0.0, 0.0, -self.g])
        effective_acc = accelerations - gravity_acc
        dynamic_forces = mass_kg * effective_acc
        total_force_mag = np.linalg.norm(dynamic_forces, axis=1)

        # Minimum grip normal force under Coulomb friction model
        # F_friction = mu * F_N * num_contacts >= F_total
        # F_N >= F_total / (mu * num_contacts) * safety_margin
        denom = max(1e-4, self.mu * num_contact_points)
        required_grip_n = (total_force_mag / denom) * self.safety_margin

        return DynamicsProfile(
            timestamps=timestamps,
            positions=traj,
            velocities=velocities,
            accelerations=accelerations,
            speed=speed,
            acceleration_magnitude=acc_mag,
            dynamic_force_newtons=dynamic_forces,
            total_force_magnitude=total_force_mag,
            required_grip_force_newtons=required_grip_n,
            peak_acceleration_m_s2=float(np.max(acc_mag)),
            mean_speed_m_s=float(np.mean(speed)),
            max_required_grip_n=float(np.max(required_grip_n))
        )
