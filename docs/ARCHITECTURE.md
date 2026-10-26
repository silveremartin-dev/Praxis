# Praxis System Architecture & Technical Specification

> **Praxis: Video-to-Physical-Digital-Twin (V2PDT) for Humanoid Robotics**  
> Automated engine transforming monocular in-the-wild web video (YouTube) into collision-free, contact-certified 3D physical digital twins for humanoid robot manipulation.

---

## 1. Executive Vision & Problem Formulation

### 1.1 The Humanoid Robotics Demonstration Bottleneck
Training dexterous humanoid manipulation policies requires millions of diverse, high-quality demonstration trajectories. Teleoperation (humans wearing VR headsets or body rigs) cannot scale to internet-scale volume. Conversely, learning directly from raw YouTube video pixels causes three catastrophic failure modes:
1. **The Kinematic Illusion**: Monocular 3D tracking hallucinates finger penetration through rigid tool handles or floating meshes without contact.
2. **Missing Dynamics**: RGB video contains zero tactile feedback, motor torques, or contact friction.
3. **The Embodiment Gap**: Human hands (27 DoF, compliant skin) do not match rigid 3-to-6 DoF robot grippers.

### 1.2 Boundary Principles & Scope
* **Workpiece Primacy Principle**: When extracting manipulation, **the tool and workpiece 3D trajectory is primary**; human finger joint positions are secondary proxies. Tool affordance trajectories are robot-agnostic (transferring identically to ALOHA, Unitree H1, or Franka Panda).
* **The Automated Yield Filter ($N \to \infty$)**: YouTube is an infinite reservoir. A 5% yield is completely acceptable, provided filtering is 100% automated with zero human review required:
  $$\text{Filter Score} = \text{CameraStability} \times \text{PenetrationConstraint} \times \text{URDFFeasibility} \times \text{MuJoCoStability}$$
* **Operational Design Domain (ODD)**:
  - **In-Scope (V1)**: Rigid standardized mechanical components (tools, keys, cylinders, fasteners, electronics assemblies, precision gears, machine plates).
  - **Out-of-Scope (V1)**: Deformable non-rigid topological mutations (fluids, cloth folding, food cutting).

---

## 2. End-to-End Pipeline Workflow

```mermaid
flowchart TD
    Raw["Raw YouTube Video (data/raw/*.mp4)"] --> Tier1["Tier 1: Sub-Video Mining (src/ingestion/)"]
    Tier1 --> Clips["Action Sub-Videos (data/clips/*.mp4)"]
    Tier1 --> Meta["Semantic Metadata (data/clips/*.json)"]
    
    Clips --> Pose["Module 2: 3D Scene Reconstruction (src/pose_extraction/)"]
    Pose --> SDF["Module 3: Physical Contact & SDF Sanitizer (src/physics_sanitizer/)"]
    
    SDF --> Dyn["Module 4: Inverse Dynamics Engine (src/dynamics/)"]
    Dyn --> URDF["Module 5: URDF Feasibility Auditor (src/feasibility/)"]
    URDF --> Sim["Module 6: MuJoCo Physics Verification (src/feasibility/)"]
    
    Sim --> Export["Module 7: Hugging Face LeRobot Exporter (src/exporter/)"]
    Export --> Parquet["Certified LeRobot Parquet (data/processed/*.parquet)"]
    Export --> Visuals["Visual Assets & Renders (assets/images/)"]
    
    Export --> Studio["Module 8: Praxis Web Studio (src/studio/)"]
    Sim --> Studio
    Clips --> Studio
```

---

## 3. Subsystem Technical Formulations

### Module 1: Action Slicing & Video Quality Auditor (`src/ingestion/`)
* Splits raw continuous streams into autonomous manipulation sub-videos ($5-20\text{ s}$).
* Generates structured task graph metadata (`t_start`, `t_end`, action verbs, tool ID, workpiece ID, camera stability).
* **Automated Video Quality Auditor (`quality_auditor.py`)**: Computes objective quality scores ($0-100$) and assigns certification grades (`A+`, `A`, `B`, `REJECTED`) based on 4 programmatic criteria:
  1. *Single-shot continuity*: 0 camera cuts ($\max \Delta I \le 0.14$), fixed camera plane.
  2. *Bimanual hand-to-wrist presence*: active Left and Right hands visible up to the wrist crease ($\ge 65\%$ duration).
  3. *Single compact object affordance*: compact interaction gap between hands ($< 220\text{ px}$).
  4. *Framing stability*: minimal translation jitter and steady illumination.

### Module 2: Automated Vision Tracking & Temporal Dynamics (`src/pose_extraction/`)
* **Automated Video Pose Pipeline (`visual_tracker.py`)**: 100% automated visual extraction pipeline conforming to AGENTS.md Sec 7.1:
  - Adaptive dual-space skin segmentation ($\text{YCrCb} + \text{HSV}$).
  - Palm centroid localization via Euclidean Distance Transform maxima (`cv2.distanceTransform`).
  - **Dense Optical Flow (Farneback `cv2.calcOpticalFlowFarneback`)**: Extracts continuous inter-frame pixel velocity fields $(\dot{u}, \dot{v})$ across sequential frames $F_t \to F_{t+1}$.
  - **Continuous Temporal Dynamics**: Derives metric wrist and finger velocities $\vec{v}(t)$ in $\text{m/s}$ and accelerations $\vec{a}(t)$ in $\text{m/s}^2$, strictly bounded by biological minimum-jerk constraints ($\|\vec{v}\| \le 2.0\text{ m/s}, \|\vec{a}\| \le 15.0\text{ m/s}^2$).
  - **Temporal Occlusion Persistence**: Kinematic propagation through occlusions (when fingers pass behind workpieces, their poses remain locked to the 6-DoF rigid palm transform $T_{\text{palm}}$ rather than jumping or disappearing).
  - **Dynamic Workpiece Sizing**: Workpiece metric dimensions ($D_x, D_y, D_z$) are bounded directly by the observed inter-hand physical gap ($D_x \le 0.75 \times \text{gap}_{\text{mm}}$), guaranteeing the object is never oversized.
* **Continuous Anatomical Human Hand Model (`hand_model.py`)**: 100% topologically connected kinematic tree with 21 joints per hand, physiological forward flexion bounds ($\theta_{\text{MCP}}, \theta_{\text{PIP}}, \theta_{\text{DIP}} \ge 0$, internal angle $\le 180^\circ$, zero backwards hyperextension).
* **Dexterous 5-Finger Humanoid Robot Hands (`hand_model.py`)**: Anthropomorphic robotic hands (`DexterousRobotHandModel`) with titanium palm chassis, rotary gold CMC thumb actuators, mechanical phalanges, and cyan tactile sensor pads for end-to-end dexterous robot simulation (Inspire / Unitree H1 / Optimus).
* **Parametric CAD Builder (`cad_primitives.py`)**: Declarative parametric assembly generator constructing multi-part compound assemblies with metric scale grounding.

### Module 3: Dual-Constraint Collision Arbitrator (`src/physics_sanitizer/`)
* **Hand-to-Workpiece Non-Penetration**: Enforces exact Signed Distance Field non-penetration penalty $\text{SDF}(p) \ge 0$ for rigid CAD bodies:
  - Projecting joint nodes outward along $\nabla\text{SDF}$ with skin capsule margin ($r_{\text{skin}} + 0.5\text{ mm}$), ensuring residual penetration is strictly **0.0 mm**.
* **Inter-Hand Non-Collision ($\text{LH} \cap \text{RH} = \emptyset$)**: Computes 3D segment-to-segment distance across all 20 bone segments of the Left Hand against all 20 bone segments of the Right Hand, enforcing clearance $\ge r_1 + r_2 + 1.0\text{ mm}$ ($8.0\text{ mm}$ minimum clearance; observed clearance on certified demonstrations $> 60.0\text{ mm}$).

### Module 4: Visual Inverse Dynamics Engine (`src/dynamics/`)
* **Mass Estimation**: $m = V \times \rho_{\text{material}}$ using calibrated physical density tables (brass: $8.5\text{ g/cm}^3$, steel: $7.9\text{ g/cm}^3$, cast iron: $7.2\text{ g/cm}^3$).
* **Principal Inertia Tensor**: Computes analytical $I_{3 \times 3} \in \mathbb{R}^{3 \times 3}$ at center of mass.
* **Coulomb Friction Grip Force**:
  $$F_N \ge \frac{m \|\ddot{x}(t) - g\|}{\mu \cdot n_{\text{contacts}}} \times \text{safety\_margin}$$

### Module 5: Kinematic Feasibility Auditor (`src/feasibility/`)
* Solves numerical IK and screens target trajectories against robot URDF limits (**ALOHA ViperX**, **Unitree H1**, **Franka Panda**).
* Rejection criteria:
  - Workspace reach envelope: $r \in [r_{\min}, r_{\max}]$
  - Joint position limits: $q_i \in [q_{\min}, q_{\max}]$
  - Joint velocity limits: $\dot{q}_i \le \dot{q}_{\max}$
  - Yoshikawa Manipulability measure: $w = \sqrt{\det(J J^T)} \ge 10^{-4}$ (detects singularities)
* Automated gatekeeper threshold: $\ge 85\%$ compliance.

### Module 6: MuJoCo Simulation Verification (`src/feasibility/mujoco_verifier.py`)
* Synthesizes native MJCF XML environments.
* Steps forward physics dynamics under gravity to certify dynamical stability, contact compliance, and absence of numerical explosion ($\max \|v\| < 10\text{ m/s}$).

### Module 7: Hugging Face LeRobot Parquet Serialization (`src/exporter/`)
* Standardizes digital twin demonstrations into Apache Parquet / Arrow schemas matching the official Hugging Face LeRobot standard:
  - `episode_index`: int64
  - `frame_index`: int64
  - `timestamp`: float32
  - `observation.state`: float32 list (7-DoF or 14-DoF bi-manual pose $[x, y, z, q_x, q_y, q_z, q_w]$)
  - `action`: float32 list (delta pose + gripper command)
  - `physics.contact_force_n`: float32
  - `physics.sdf_distance_mm`: float32
  - `feasibility_score`: float32
  - `task_description`: string

### Module 8: Praxis Interactive Web Studio (`src/studio/`)
* **FastAPI Telemetry Backend (`server.py`)**: High-performance asynchronous API providing:
  - Demonstration catalog endpoints (`/api/clips`, `/api/clip/{id}`) with phase breakdowns and media mounts.
  - Continuous 3D pose extraction (`/api/clip/{id}/pose?progress=t`).
  - Native MuJoCo 3.x forward dynamic verification (`/api/mujoco/simulate`).
  - Hardware specifications for humanoid robot hands (`/api/robots`, `/api/robot/{id}`).
  - Asynchronous sub-video slicing and metadata generation (`/api/ingest`).
* **Hardware Robot Specifications Database (`robot_specs.py`)**: Authentic technical kinematics and dynamics data:
  - **Unitree H1 (Inspire DexHand)**: 6 active DoF, 12 joints, 10N pinch force, 3kg payload, dual-joint linkage coupling.
  - **Tesla Optimus Gen-2 Hand**: 11 active DoF, 16 joints, 15N pinch, 45N grasp, 10kg payload, tendon-driven BLDC motors, distributed phalanx tactile sensing.
  - **ALOHA Dual-Arm (Trossen ViperX)**: 7 DoF arm + parallel gripper with compliant silicone / GelSight sensors.
  - **Franka Emika Panda**: 7 DoF torque-controlled arm + parallel jaw gripper.
* **Single-Page WebGL Studio (`static/index.html`)**:
  - Three.js WebGL canvas with OrbitControls, OBJLoader, lighting, contact manifold markers, and live SDF penetration gauge.
  - Synchronized HTML5 video player and timeline scrubber.
  - Multi-embodiment retargeting switcher.
  - One-click launch scripts: `start_web_browser.bat` (Windows) and `start_web_browser.sh` (Linux/macOS).

---

## 4. Development Status & Roadmap

| Phase | Milestone Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Foundation, raw video ingestion, sub-video slicing, semantic metadata | **Completed (100%)** |
| **Phase 2** | Metric 3D digital twins, visual asset generator, side-by-side renders | **Completed (100%)** |
| **Phase 3** | Analytical SDF solver, force-closure contacts, inverse dynamics engine | **Completed (100%)** |
| **Phase 4** | URDF kinematic auditor, automated $\ge 85\%$ filter, MuJoCo stability verification | **Completed (100%)** |
| **Phase 5** | Scaled cloud execution (Google Cloud Run / GCE L4 GPUs) & Hugging Face release | **Next (Scheduled)** |
