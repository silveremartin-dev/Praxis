"""
Praxis Kinematic Feasibility & Physics Verification Module.
"""

from src.feasibility.urdf_auditor import URDFAuditor, ROBOT_REGISTRY, RobotModelSpec, AuditReport
from src.feasibility.mujoco_verifier import MuJoCoVerifier, SimulationVerificationResult

__all__ = [
    "URDFAuditor",
    "ROBOT_REGISTRY",
    "RobotModelSpec",
    "AuditReport",
    "MuJoCoVerifier",
    "SimulationVerificationResult"
]
