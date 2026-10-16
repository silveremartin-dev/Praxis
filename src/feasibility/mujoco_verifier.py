"""
MuJoCo Physics Simulation Verifier for Praxis Digital Twins.
Generates MJCF (XML) simulation environments and executes forward dynamics
to certify mechanical stability, contact integrity, and absence of numerical explosion.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import mujoco

@dataclass
class SimulationVerificationResult:
    scene_name: str
    is_dynamically_stable: bool
    simulated_duration_seconds: float
    max_velocity_m_s: float
    max_contact_penetration_mm: float
    numerical_explosion_detected: bool
    final_object_position_m: np.ndarray

class MuJoCoVerifier:
    """
    Automated physics simulation tester running under MuJoCo forward dynamics.
    """
    def __init__(self, timestep: float = 0.002):
        self.timestep = timestep

    @staticmethod
    def generate_mjcf_scene(scene_id: str) -> str:
        """
        Synthesizes an exact MJCF XML string for the certified digital twin scenes.
        """
        if scene_id == "01_lpl_dropkey":
            return """<mujoco model="dropkey_scene">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="dropkey_stem" pos="0 0 0.05">
      <freejoint/>
      <geom name="stem_cyl" type="cylinder" size="0.004 0.06" rgba="0.7 0.7 0.8 1" mass="0.047"/>
      <body name="collar" pos="0.025 0 0">
        <geom name="collar_cyl" type="cylinder" size="0.0065 0.005" rgba="0.1 0.1 0.1 1" mass="0.010"/>
      </body>
      <body name="folding_link" pos="-0.06 0 0">
        <joint name="pin_joint" type="hinge" axis="0 1 0" range="-100 100"/>
        <geom name="link_cyl" type="cylinder" size="0.004 0.0225" pos="0 0 -0.0225" rgba="0.6 0.6 0.7 1" mass="0.018"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

        elif scene_id == "05_watch_zodiac":
            return """<mujoco model="watchmaker_scene">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="mat" type="plane" size="0.2 0.2 0.05" rgba="0.15 0.45 0.3 1"/>
    <body name="holder" pos="0 0 0.009">
      <geom name="holder_cyl" type="cylinder" size="0.028 0.009" rgba="0.8 0.8 0.8 1" mass="0.115"/>
      <body name="mainplate" pos="0 0 0.010">
        <geom name="plate_cyl" type="cylinder" size="0.015 0.0011" rgba="0.85 0.7 0.35 1" mass="0.013"/>
      </body>
    </body>
    <body name="micro_gear" pos="0.002 0.003 0.021">
      <freejoint/>
      <geom name="gear_pinion" type="cylinder" size="0.0004 0.0016" rgba="0.7 0.7 0.7 1" mass="0.000045"/>
      <geom name="gear_disk" type="cylinder" size="0.0042 0.00017" rgba="0.85 0.75 0.3 1" mass="0.00002"/>
    </body>
  </worldbody>
</mujoco>"""

        elif scene_id == "06_singer_sewingmachine":
            return """<mujoco model="singer_scene">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="bench" type="plane" size="0.4 0.4 0.05" rgba="0.45 0.3 0.2 1"/>
    <body name="singer_casting" pos="0.06 0 0.075">
      <geom name="machine_body" type="box" size="0.06 0.08 0.075" rgba="0.1 0.1 0.1 1" mass="4.85"/>
    </body>
    <body name="faceplate" pos="-0.025 0 0.085">
      <freejoint/>
      <geom name="plate_box" type="box" size="0.0019 0.031 0.0575" rgba="0.3 0.3 0.3 1" mass="0.285"/>
    </body>
  </worldbody>
</mujoco>"""


        else:
            raise ValueError(f"Unknown scene_id '{scene_id}'")

    def verify_scene_stability(self, scene_id: str, duration_sec: float = 1.0, max_allowed_velocity: float = 10.0) -> SimulationVerificationResult:
        """
        Runs MuJoCo forward stepping to verify dynamical stability.
        """
        mjcf_str = self.generate_mjcf_scene(scene_id)
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)

        total_steps = int(duration_sec / self.timestep)
        max_vel = 0.0
        max_penetration = 0.0
        explosion = False

        for _ in range(total_steps):
            mujoco.mj_step(model, data)
            
            # Check maximum generalized velocity
            if len(data.qvel) > 0:
                cur_max_v = float(np.max(np.abs(data.qvel)))
                if cur_max_v > max_vel:
                    max_vel = cur_max_v
                if cur_max_v > max_allowed_velocity or np.isnan(cur_max_v):
                    explosion = True
                    break

            # Check contact penetrations
            for i in range(data.ncon):
                contact = data.contact[i]
                penetration = -contact.dist # distance < 0 means penetration
                if penetration > max_penetration:
                    max_penetration = penetration

        max_pen_mm = float(max_penetration * 1000.0)
        # Stable if no NaN/velocity blowup and contact penetration remains compliant within restitution limits
        is_stable = (not explosion) and (max_vel < max_allowed_velocity) and (max_pen_mm < 25.0)
        final_pos = data.qpos[:3].copy() if len(data.qpos) >= 3 else np.zeros(3)

        return SimulationVerificationResult(
            scene_name=scene_id,
            is_dynamically_stable=is_stable,
            simulated_duration_seconds=duration_sec,
            max_velocity_m_s=round(max_vel, 3),
            max_contact_penetration_mm=round(max_pen_mm, 3),
            numerical_explosion_detected=explosion,
            final_object_position_m=final_pos
        )

