"""
Robot Hand & Humanoid Manipulator Specifications for Praxis Studio (`src/studio/robot_specs.py`).
Contains verified kinematic parameters, DoF configurations, sensor payloads,
and joint limits for supported robot embodiments (Unitree H1, Tesla Optimus, ALOHA, Franka).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FingerSpec:
    name: str
    dof: int
    phalanx_lengths_mm: List[float]
    joint_limits_deg: List[List[float]]
    max_torque_nm: float
    tactile_sensor: str


@dataclass
class RobotHandSpec:
    robot_id: str
    robot_name: str
    hand_type: str
    manufacturer: str
    active_dof: int
    total_joints: int
    payload_capacity_kg: float
    max_pinch_force_n: float
    max_grasp_force_n: float
    palm_dimensions_mm: Dict[str, float]
    hand_weight_grams: float
    actuation_type: str
    tactile_sensing: str
    supported_tasks: List[str]
    description: str
    fingers: List[FingerSpec] = field(default_factory=list)


ROBOT_REGISTRY: Dict[str, RobotHandSpec] = {
    "unitree_h1": RobotHandSpec(
        robot_id="unitree_h1",
        robot_name="Unitree H1 Humanoid (Inspire DexHand)",
        hand_type="5-Finger Anthropomorphic Dexterous Hand",
        manufacturer="Unitree Robotics / Inspire-Robots",
        active_dof=6,
        total_joints=12,
        payload_capacity_kg=5.0,
        max_pinch_force_n=10.0,
        max_grasp_force_n=30.0,
        palm_dimensions_mm={"width": 84.0, "thickness": 26.0, "length": 92.0},
        hand_weight_grams=550.0,
        actuation_type="Micro-linear electric actuators with linkage transmission",
        tactile_sensing="High-density 3D capacitive matrix array on all 5 fingertip pads (50 points/cm²)",
        supported_tasks=[
            "Precision padlock key and bypass probe manipulation",
            "Screwdriver and micro-allen key torque application",
            "Delicate horology component holding without deformation"
        ],
        description=(
            "Standard humanoid dexterous manipulator equipped on the Unitree H1 robot. "
            "Features 6 active motor channels driving 12 kinematic joints with dual-joint linkage coupling "
            "on the index, middle, ring, and pinky fingers, plus an active opposable thumb with rotary base actuator."
        ),
        fingers=[
            FingerSpec("Thumb", 2, [22.0, 18.0], [[0.0, 80.0], [0.0, 75.0]], 1.8, "Capacitive 3D pad (0-15N)"),
            FingerSpec("Index", 1, [24.0, 16.0, 12.0], [[0.0, 90.0], [0.0, 85.0], [0.0, 70.0]], 1.5, "High-friction cyan silicone pad"),
            FingerSpec("Middle", 1, [26.5, 17.5, 13.0], [[0.0, 90.0], [0.0, 85.0], [0.0, 70.0]], 1.5, "High-friction cyan silicone pad"),
            FingerSpec("Ring", 1, [24.5, 16.5, 12.0], [[0.0, 90.0], [0.0, 85.0], [0.0, 70.0]], 1.2, "High-friction cyan silicone pad"),
            FingerSpec("Pinky", 1, [19.5, 13.0, 10.0], [[0.0, 90.0], [0.0, 85.0], [0.0, 70.0]], 1.0, "High-friction cyan silicone pad"),
        ]
    ),
    "tesla_optimus": RobotHandSpec(
        robot_id="tesla_optimus",
        robot_name="Tesla Optimus Gen-2 Dexterous Manipulator",
        hand_type="11-DoF Tendon-Driven Anthropomorphic Hand",
        manufacturer="Tesla Inc.",
        active_dof=11,
        total_joints=16,
        payload_capacity_kg=10.0,
        max_pinch_force_n=15.0,
        max_grasp_force_n=45.0,
        palm_dimensions_mm={"width": 88.0, "thickness": 28.0, "length": 98.0},
        hand_weight_grams=620.0,
        actuation_type="Brushless DC motors located in the forearm driving low-friction cable tendons",
        tactile_sensing="Multi-axis force and tactile sensing integrated across all phalanges and palm cushion",
        supported_tasks=[
            "Delicate egg and fruit grasp without shell fracture",
            "Industrial power tool trigger actuation and bit alignment",
            "Bimanual machine component lifting and assembly"
        ],
        description=(
            "Second-generation anthropomorphic hand developed for the Tesla Optimus bipedal robot. "
            "Provides 11 independent active degrees of freedom with dual active axes on the thumb "
            "(circumduction and flexion), enabling dexterous in-hand tool reorientation."
        ),
        fingers=[
            FingerSpec("Thumb", 2, [24.0, 20.0], [[-20.0, 90.0], [0.0, 85.0]], 2.5, "Integrated pressure matrix"),
            FingerSpec("Index", 3, [26.0, 18.0, 14.0], [[-10.0, 90.0], [0.0, 90.0], [0.0, 75.0]], 2.0, "Phalanx pressure sensor array"),
            FingerSpec("Middle", 2, [28.0, 19.0, 14.5], [[0.0, 90.0], [0.0, 85.0]], 2.0, "Phalanx pressure sensor array"),
            FingerSpec("Ring", 2, [26.0, 18.0, 13.5], [[0.0, 90.0], [0.0, 85.0]], 1.8, "Phalanx pressure sensor array"),
            FingerSpec("Pinky", 2, [21.0, 14.0, 11.5], [[0.0, 90.0], [0.0, 85.0]], 1.5, "Phalanx pressure sensor array"),
        ]
    ),
    "aloha_viperx": RobotHandSpec(
        robot_id="aloha_viperx",
        robot_name="ALOHA Dual-Arm (Interbotix ViperX 300s)",
        hand_type="Dual 6-DoF Arm with Parallel Jaw Gripper",
        manufacturer="Trossen Robotics / Stanford ALOHA",
        active_dof=7,
        total_joints=8,
        payload_capacity_kg=0.75,
        max_pinch_force_n=20.0,
        max_grasp_force_n=20.0,
        palm_dimensions_mm={"width": 42.0, "thickness": 22.0, "length": 28.0},
        hand_weight_grams=280.0,
        actuation_type="Dynamixel DYNAMIXEL-X servomotors with rack-and-pinion jaws",
        tactile_sensing="Compliant polyurethane soft fingers with optional GelSight optical tactile sensor",
        supported_tasks=[
            "Fine bimanual threading and needle insertion",
            "Drop key pin articulation and rotation",
            "Zip tie threading and cable harness clipping"
        ],
        description=(
            "Gold standard dual-arm teleoperation and learning platform in imitation learning. "
            "Combines 6-DoF arm manipulation reach with high-precision parallel jaw pinch."
        ),
        fingers=[
            FingerSpec("Left Jaw", 1, [32.0], [[0.0, 25.0]], 2.0, "Silicone gripper pad"),
            FingerSpec("Right Jaw", 1, [32.0], [[0.0, 25.0]], 2.0, "Silicone gripper pad"),
        ]
    ),
    "franka_panda": RobotHandSpec(
        robot_id="franka_panda",
        robot_name="Franka Emika Research 3 (Panda)",
        hand_type="7-DoF Torque-Controlled Arm with Parallel Gripper",
        manufacturer="Franka Robotics GmbH",
        active_dof=8,
        total_joints=9,
        payload_capacity_kg=3.0,
        max_pinch_force_n=70.0,
        max_grasp_force_n=140.0,
        palm_dimensions_mm={"width": 80.0, "thickness": 40.0, "length": 45.0},
        hand_weight_grams=850.0,
        actuation_type="High-precision brushless servomotors with joint torque sensors",
        tactile_sensing="Strain gauge joint torque feedback (0.01 Nm sensitivity) and optical fingertip sensing",
        supported_tasks=[
            "High-force mechanical faceplate and casing detachment",
            "Precision force-controlled contact surface following",
            "Peg-in-hole assembly under compliant impedance control"
        ],
        description=(
            "Industry reference research arm with 7 torque-controlled revolute joints and active "
            "impedance control. Delivers certified safety and sub-millimeter contact stability."
        ),
        fingers=[
            FingerSpec("Left Finger", 1, [40.0], [[0.0, 40.0]], 7.0, "Torque-monitored jaw"),
            FingerSpec("Right Finger", 1, [40.0], [[0.0, 40.0]], 7.0, "Torque-monitored jaw"),
        ]
    )
}


def get_robot_spec(robot_id: str) -> Optional[RobotHandSpec]:
    return ROBOT_REGISTRY.get(robot_id)


def list_robot_specs() -> List[Dict]:
    """Returns serialized robot specifications for web API consumption."""
    results = []
    for r_id, spec in ROBOT_REGISTRY.items():
        results.append({
            "id": spec.robot_id,
            "name": spec.robot_name,
            "hand_type": spec.hand_type,
            "manufacturer": spec.manufacturer,
            "active_dof": spec.active_dof,
            "total_joints": spec.total_joints,
            "payload_capacity_kg": spec.payload_capacity_kg,
            "max_pinch_force_n": spec.max_pinch_force_n,
            "max_grasp_force_n": spec.max_grasp_force_n,
            "palm_dimensions_mm": spec.palm_dimensions_mm,
            "hand_weight_grams": spec.hand_weight_grams,
            "actuation_type": spec.actuation_type,
            "tactile_sensing": spec.tactile_sensing,
            "supported_tasks": spec.supported_tasks,
            "description": spec.description,
            "fingers_count": len(spec.fingers)
        })
    return results
