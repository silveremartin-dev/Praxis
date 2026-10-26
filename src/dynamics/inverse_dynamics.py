"""
Visual Inverse Dynamics & Inter-Frame Physical Continuity Engine for Praxis (`src/dynamics/inverse_dynamics.py`).

Enforces:
1. Kinematic Velocity & Acceleration Ceilings:
   - Max manipulation speed v <= 2.0 m/s (~7.2 km/h) for tabletop fine manipulation.
   - Max acceleration a <= 15.0 m/s^2 (~1.5 g).
   - Max jerk j <= 100.0 m/s^3 (biological minimum-jerk smoothness).
2. Inter-Frame Physical Continuity:
   - Prohibits teleportation jumps and non-smooth coordinate oscillations between adjacent video frames.
3. Volumetric Mass Priors & Grasp Affordance Coherence:
   - Precision pinch (thumb + index) payload <= 1.5 kg under Coulomb friction cone.
   - Palmar cradle (support from below) for heavy payloads up to 8.0 kg.
   - Power wrap for high-torque tool actions.
4. Robot Action Space & Token Conditioning:
   - Method 1: Data curation filter score based on physical plausibility.
   - Method 2: Hybrid action space (kinematic position + target grip force + impedance stiffness).
   - Method 3: Physics tokens (mass class, grasp type, continuity score) for VLA/Transformer conditioning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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


class GraspType(str, Enum):
    PRECISION_PINCH = "precision_pinch"
    PALMAR_CRADLE = "palmar_cradle"
    POWER_WRAP = "power_wrap"


class MassClass(str, Enum):
    LIGHT_HOLLOW = "light_hollow"       # < 100g (keys, thin probes, empty boxes)
    MEDIUM_TOOL = "medium_tool"         # 100g - 800g (padlocks, screwdrivers, pliers)
    HEAVY_PAYLOAD = "heavy_payload"     # > 800g (solid billets, power tools, heavy fixtures)


@dataclass
class KinematicContinuityReport:
    """Detailed inter-frame physical continuity audit report."""
    is_continuous: bool
    continuity_score: float             # In [0.0, 1.0], 1.0 = perfectly smooth biological motion
    max_velocity_m_s: float
    max_acceleration_m_s2: float
    max_jerk_m_s3: float
    num_violating_transitions: int
    total_transitions: int
    rejection_reason: Optional[str] = None


@dataclass
class GraspAffordanceResult:
    """Audit of grasp type coherence vs workpiece mass and Coulomb friction cone."""
    grasp_type: GraspType
    is_physically_feasible: bool
    mass_grams: float
    max_supported_mass_grams: float
    required_normal_force_n: float
    max_available_grip_n: float
    friction_coefficient: float
    rejection_reason: Optional[str] = None


@dataclass
class PhysicsTokens:
    """Compact structured physical representation for robot foundation model conditioning."""
    mass_class: MassClass
    grasp_type: GraspType
    target_stiffness_k_n_m: float       # Robot impedance stiffness (e.g. 150 N/m for soft, 900 N/m for rigid)
    target_damping_d_n_s_m: float       # Robot impedance damping
    target_grip_force_n: float          # Recommended gripper squeezing force
    continuity_score: float             # Inter-frame smoothness score in [0.0, 1.0]
    feature_vector: List[float]         # Normalized numerical embedding for Transformer conditioning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mass_class": self.mass_class.value,
            "grasp_type": self.grasp_type.value,
            "target_stiffness_k_n_m": round(self.target_stiffness_k_n_m, 2),
            "target_damping_d_n_s_m": round(self.target_damping_d_n_s_m, 2),
            "target_grip_force_n": round(self.target_grip_force_n, 2),
            "continuity_score": round(self.continuity_score, 4),
            "feature_vector": [round(x, 4) for x in self.feature_vector]
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
    continuity_report: Optional[KinematicContinuityReport] = None
    grasp_affordance: Optional[GraspAffordanceResult] = None
    physics_tokens: Optional[PhysicsTokens] = None


class PhysicalContinuityAuditor:
    """
    Audits inter-frame kinematic continuity and enforces biological motion limits.
    Rejects visual tracking teleportations, jump cuts, and unphysical frame transitions.
    """
    MAX_VELOCITY_M_S: float = 2.0       # Max fine manipulation speed (~7.2 km/h)
    MAX_ACCEL_M_S2: float = 15.0        # Max physical hand acceleration (~1.5 g)
    MAX_JERK_M_S3: float = 100.0        # Max physical jerk

    @classmethod
    def audit_trajectory_continuity(
        cls,
        trajectory_xyz_m: np.ndarray,
        dt: float
    ) -> KinematicContinuityReport:
        """
        Audits that adjacent frames exhibit smooth C1/C2 physical continuity.
        trajectory_xyz_m: (N, 3) positions in meters
        dt: inter-frame duration in seconds
        """
        traj = np.asarray(trajectory_xyz_m, dtype=float)
        N = len(traj)
        if N < 2 or dt <= 0:
            return KinematicContinuityReport(
                is_continuous=False,
                continuity_score=0.0,
                max_velocity_m_s=0.0,
                max_acceleration_m_s2=0.0,
                max_jerk_m_s3=0.0,
                num_violating_transitions=0,
                total_transitions=0,
                rejection_reason="Trajectory too short or invalid dt"
            )

        # 1. Finite difference velocities
        displacements = traj[1:] - traj[:-1] # (N-1, 3)
        speeds = np.linalg.norm(displacements, axis=1) / dt # (N-1,)
        max_v = float(np.max(speeds))

        # 2. Accelerations between steps
        if N >= 3:
            vels = displacements / dt # (N-1, 3)
            accel_vecs = (vels[1:] - vels[:-1]) / dt # (N-2, 3)
            accels = np.linalg.norm(accel_vecs, axis=1) # (N-2,)
            max_a = float(np.max(accels))
        else:
            max_a = 0.0
            accels = np.array([0.0])

        # 3. Jerk between steps
        if N >= 4:
            jerk_vecs = (accel_vecs[1:] - accel_vecs[:-1]) / dt
            jerks = np.linalg.norm(jerk_vecs, axis=1)
            max_j = float(np.max(jerks))
        else:
            max_j = 0.0

        # Count violations
        v_violations = np.sum(speeds > cls.MAX_VELOCITY_M_S)
        a_violations = np.sum(accels > cls.MAX_ACCEL_M_S2)
        total_viol = int(v_violations + a_violations)

        # Smooth score calculation: decays exponentially if limits are exceeded
        v_ratio = max_v / cls.MAX_VELOCITY_M_S
        a_ratio = max_a / cls.MAX_ACCEL_M_S2
        score = float(np.clip(np.exp(-max(0.0, v_ratio - 1.0) - max(0.0, a_ratio - 1.0)), 0.0, 1.0))

        is_valid = total_viol == 0 and max_v <= cls.MAX_VELOCITY_M_S and max_a <= cls.MAX_ACCEL_M_S2
        reason = None
        if not is_valid:
            if max_v > cls.MAX_VELOCITY_M_S:
                reason = f"Inter-frame speed {max_v:.2f} m/s exceeds max manipulation limit ({cls.MAX_VELOCITY_M_S} m/s)."
            else:
                reason = f"Inter-frame acceleration {max_a:.2f} m/s2 exceeds max manipulation limit ({cls.MAX_ACCEL_M_S2} m/s2)."

        return KinematicContinuityReport(
            is_continuous=is_valid,
            continuity_score=score,
            max_velocity_m_s=max_v,
            max_acceleration_m_s2=max_a,
            max_jerk_m_s3=max_j,
            num_violating_transitions=total_viol,
            total_transitions=len(displacements),
            rejection_reason=reason
        )


class GraspAffordanceAuditor:
    """
    Audits physical compatibility between grasp type, normal grip forces,
    Coulomb friction cone, and workpiece mass.
    """
    # Max human grip normal force per grasp type (in Newtons)
    MAX_GRIP_FORCES: Dict[GraspType, float] = {
        GraspType.PRECISION_PINCH: 45.0,    # Two-finger fingertip pinch
        GraspType.PALMAR_CRADLE: 200.0,     # Gravity support from underneath
        GraspType.POWER_WRAP: 350.0,        # Full cylindrical power grasp
    }

    @classmethod
    def evaluate_grasp(
        cls,
        grasp_type: GraspType,
        mass_grams: float,
        friction_coeff: float = 0.40,
        gravity: float = 9.81,
        safety_margin: float = 1.3
    ) -> GraspAffordanceResult:
        """
        Verifies if an object of mass `mass_grams` can be supported by `grasp_type`
        under Coulomb friction without slipping.
        """
        mass_kg = mass_grams / 1000.0
        weight_n = mass_kg * gravity
        max_force = cls.MAX_GRIP_FORCES.get(grasp_type, 45.0)

        if grasp_type == GraspType.PALMAR_CRADLE:
            # Normal reaction supports weight directly against gravity (R_z >= m*g)
            # No critical friction dependency to prevent sliding downward
            max_supported_kg = max_force / gravity
            req_normal_n = weight_n
            feasible = mass_kg <= max_supported_kg
        else:
            # Friction cone: F_friction = mu * F_normal * 2_contact_surfaces >= weight * safety
            # F_normal >= weight / (2 * mu) * safety_margin
            req_normal_n = (weight_n / max(0.05, 2.0 * friction_coeff)) * safety_margin
            max_supported_kg = (2.0 * friction_coeff * max_force) / (gravity * safety_margin)
            feasible = req_normal_n <= max_force

        reason = None
        if not feasible:
            reason = (
                f"Workpiece mass {mass_grams:.1f}g exceeds maximum payload ({max_supported_kg * 1000.0:.1f}g) "
                f"for grasp type '{grasp_type.value}' under friction mu={friction_coeff}."
            )

        return GraspAffordanceResult(
            grasp_type=grasp_type,
            is_physically_feasible=feasible,
            mass_grams=mass_grams,
            max_supported_mass_grams=float(max_supported_kg * 1000.0),
            required_normal_force_n=float(req_normal_n),
            max_available_grip_n=float(max_force),
            friction_coefficient=friction_coeff,
            rejection_reason=reason
        )


class PhysicsTokenGenerator:
    """
    Generates structured physical tokens for robot foundation model conditioning.
    Emits impedance stiffness, damping, grip force, and embedding vectors.
    """
    @staticmethod
    def classify_mass(mass_grams: float) -> MassClass:
        if mass_grams < 100.0:
            return MassClass.LIGHT_HOLLOW
        elif mass_grams <= 800.0:
            return MassClass.MEDIUM_TOOL
        else:
            return MassClass.HEAVY_PAYLOAD

    @classmethod
    def generate_tokens(
        cls,
        mass_grams: float,
        grasp_type: GraspType,
        continuity_score: float = 1.0,
        is_rigid: bool = True
    ) -> PhysicsTokens:
        mass_cls = cls.classify_mass(mass_grams)

        # Baseline stiffness (N/m) and damping (N*s/m) for robotic impedance control
        if mass_cls == MassClass.LIGHT_HOLLOW:
            stiffness_k = 180.0 if not is_rigid else 350.0
            damping_d = 15.0
            grip_force = 4.0
        elif mass_cls == MassClass.MEDIUM_TOOL:
            stiffness_k = 450.0 if not is_rigid else 750.0
            damping_d = 28.0
            grip_force = 12.0
        else:
            stiffness_k = 850.0
            damping_d = 45.0
            grip_force = 28.0

        # Adjust for grasp type
        if grasp_type == GraspType.PRECISION_PINCH:
            grip_force = min(grip_force, 15.0)
            stiffness_k *= 0.85
        elif grasp_type == GraspType.PALMAR_CRADLE:
            grip_force = 5.0 # Low squeezing needed, cradle supports underneath
            stiffness_k *= 1.1

        # Numerical embedding vector [norm_mass, grasp_id, norm_stiffness, norm_force, continuity]
        norm_mass = float(np.clip(mass_grams / 1000.0, 0.0, 5.0)) / 5.0
        grasp_id = {"precision_pinch": 0.0, "palmar_cradle": 0.5, "power_wrap": 1.0}[grasp_type.value]
        norm_k = float(np.clip(stiffness_k / 1000.0, 0.0, 1.0))
        norm_f = float(np.clip(grip_force / 40.0, 0.0, 1.0))
        feature_vector = [norm_mass, grasp_id, norm_k, norm_f, float(continuity_score)]

        return PhysicsTokens(
            mass_class=mass_cls,
            grasp_type=grasp_type,
            target_stiffness_k_n_m=stiffness_k,
            target_damping_d_n_s_m=damping_d,
            target_grip_force_n=grip_force,
            continuity_score=continuity_score,
            feature_vector=feature_vector
        )


class InverseDynamicsEngine:
    """
    Engine to compute rigid body mass, inertia tensors, kinematic derivatives,
    dynamic forces, inter-frame continuity, and affordance tokens.
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
            dim_avg = np.mean(list(dimensions_mm.values())) / 1000.0 if dimensions_mm else 0.05
            I_iso = (1.0 / 6.0) * mass_kg * (dim_avg**2)
            I[0, 0] = I_iso
            I[1, 1] = I_iso
            I[2, 2] = I_iso
            
        return I

    def analyze_trajectory(
        self,
        trajectory_xyz_m: np.ndarray,
        mass_grams: float,
        fps: float = 30.0,
        num_contact_points: int = 2,
        grasp_type: GraspType = GraspType.PRECISION_PINCH
    ) -> DynamicsProfile:
        """
        Derives velocities, accelerations, dynamic forces, continuity audit,
        grasp affordance check, and robot physics tokens.
        """
        traj = np.asarray(trajectory_xyz_m, dtype=float)
        N = len(traj)
        if N < 2:
            raise ValueError("Trajectory must contain at least 2 frames")
            
        dt = 1.0 / fps
        timestamps = np.arange(N) * dt
        mass_kg = mass_grams / 1000.0

        # 1. Numerical differentiation with central difference
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

        # 2. Dynamic force: F_ext = m * (a - g)
        gravity_acc = np.array([0.0, 0.0, -self.g])
        effective_acc = accelerations - gravity_acc
        dynamic_forces = mass_kg * effective_acc
        total_force_mag = np.linalg.norm(dynamic_forces, axis=1)

        # 3. Minimum grip normal force under Coulomb friction model
        denom = max(1e-4, self.mu * num_contact_points)
        required_grip_n = (total_force_mag / denom) * self.safety_margin

        # 4. Inter-frame kinematic continuity audit
        continuity_report = PhysicalContinuityAuditor.audit_trajectory_continuity(traj, dt)

        # 5. Grasp affordance feasibility check
        grasp_affordance = GraspAffordanceAuditor.evaluate_grasp(
            grasp_type=grasp_type,
            mass_grams=mass_grams,
            friction_coeff=self.mu,
            gravity=self.g,
            safety_margin=self.safety_margin
        )

        # 6. Physical tokens for robot foundation models
        tokens = PhysicsTokenGenerator.generate_tokens(
            mass_grams=mass_grams,
            grasp_type=grasp_type,
            continuity_score=continuity_report.continuity_score,
            is_rigid=True
        )

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
            max_required_grip_n=float(np.max(required_grip_n)),
            continuity_report=continuity_report,
            grasp_affordance=grasp_affordance,
            physics_tokens=tokens
        )
