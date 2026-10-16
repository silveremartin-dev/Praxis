# Subagent Architecture & System Roles

This document defines the specialized autonomous modules, prompt patterns, inputs, and outputs of the AI subagents operating within the Praxis Video-to-Physical-Digital-Twin pipeline.

---

## 1. Subagent Ecosystem

| Agent Name | Engine / Technology | Primary Responsibility |
| :--- | :--- | :--- |
| **Ingestion & Slicer Agent** | VLM / OpenCV Slicer | Slices raw YouTube videos into continuous manipulation sub-clips and extracts semantic action metadata. |
| **Pose & Mesh Recovery Agent** | HaMeR / MANO Model | Extracts 3D hand joint coordinates and reconstructed 778-vertex mesh. |
| **Tool & CAD Matcher Agent** | SAM 2 + DINOv2 Retrieval | Tracks the tool/object in 2D/3D and associates it with canonical CAD models. |
| **Contact Sanitizer Agent** | SDF Optimization Solver | Resolves hand-object penetration and stabilizes physical contact manifolds. |
| **Physical Dynamics Agent** | Inverse Dynamics Solver | Infers object mass, inertia tensor, and required grip force from visual kinematics. |
| **Robot Feasibility Auditor** | Numerical IK (Pinocchio/PyBullet) | Computes kinematic reachability and joint limit compliance against target URDF models. |
| **Dataset Packager Agent** | LeRobot Serializer | Standardizes output into Apache Parquet schemas and connects to Rerun.io. |

---

## 2. Detailed Subagent Specifications

### Agent 1: Ingestion & Slicer Agent
* **Role:** High-level video inspector and temporal segmentation engine.
* **Input:** Raw video stream (`data/raw/*.mp4`).
* **Processing:** Samples frames and identifies continuous segments where hands physically manipulate rigid objects.
* **Output:**
  - Clean sub-video file: `data/clips/<video_id>_action_<clip_id>.mp4`.
  - Action metadata: JSON containing `start_time`, `end_time`, `action_verb`, `tool_name`, `workpiece_name`, and `camera_stability_score`.

### Agent 2: Pose & Mesh Recovery Agent
* **Role:** Anatomical hand reconstruction engine.
* **Input:** Action sub-video frames.
* **Output:** Wrist global transformation matrix $SE(3)$, 21 3D joint locations, and 778 surface vertices per frame.

### Agent 3: Tool & CAD Matcher Agent
* **Role:** Object identification and geometric grounding.
* **Input:** Segmented masks from SAM 2.
* **Output:** Canonical 3D CAD mesh (`.obj` / `.stl`) and continuous 6D pose estimation ($SE(3)$ position and orientation quaternion).

### Agent 4: Contact Sanitizer Agent
* **Role:** Physical collision arbitrator.
* **Loss Formulation:**
  $$\min_{\theta_{\text{hand}}, T_{\text{obj}}} L_{\text{reproj}} + \lambda_{\text{SDF}} L_{\text{pen}} + \lambda_{\text{contact}} L_{\text{contact}}$$
* **Rules:**
  - Hand vertices must never enter the interior volume of the rigid tool ($\text{SDF}(v) \ge 0$).
  - During active manipulation, contact points on the tool must adhere to force-closure heuristics.

### Agent 5: Robot Feasibility Auditor
* **Role:** Transferability gatekeeper for physical humanoid robots.
* **Input:** Sanitized 3D trajectory + target robot URDF (e.g., ALOHA, Unitree H1).
* **Metric:** Evaluates kinematic manipulability, joint position limits, and actuator velocity limits.
* **Automated Decision:** Rejects demonstrations failing to achieve $\ge 85\%$ kinematic compliance.
