# AGENTS.md — Developer & AI Agent Collaboration Directives for Praxis

## 🎯 1. Core Project Identity & Scope

**Praxis is an automated data-engineering engine that converts monocular in-the-wild videos (YouTube) into physically certified, collision-free 3D digital twins for humanoid robot manipulation.**

### The Core Problem:
Humanoid robotics suffers from a crippling **demonstration bottleneck**. Teleoperation (humans wearing VR headsets or body rigs) does not scale to internet-level volume. However, training policies directly on raw, uncalibrated web video pixels leads to severe failure modes:
1. **The Kinematic Illusion**: Monocular 3D tracking hallucinates finger penetration through rigid tool handles or floating meshes without contact.
2. **Missing Dynamics**: RGB video contains zero tactile feedback, motor torques, or contact friction.
3. **The Embodiment Gap**: Human hands with 27 DoF and flexible skin do not map 1:1 to rigid 3-to-6 DoF robot grippers.

### Praxis Solution & Boundary Principles:
* **IN SCOPE**:
  1. **Sub-video Action Slicing**: Slicing raw videos into clean sub-video MP4 clips containing continuous, uninterrupted hand manipulation sequences.
  2. **Semantic & Task Graph Metadata**: Extracting high-level structured intent ("what is the person doing?", pre-conditions, post-conditions, tool identity).
  3. **Joint Hand-Object 3D Reconstruction**: Using canonical CAD priors (Objaverse/primitives) with Signed Distance Field (SDF) penalties to strictly eliminate interpenetration and enforce physical contact.
  4. **Visual Inverse Dynamics**: Inferring object mass, inertia, and applied forces from full-body posture compensation and kinematic acceleration profiles.
  5. **Kinematic Feasibility Filtering**: Auditing demonstrations against target robot URDF joint/velocity limits to reject infeasible motions automatically.
  6. **Standardized Export**: Emitting verified datasets in Hugging Face **LeRobot** format (Parquet / Arrow / Rerun).
* **OUT OF SCOPE / STRICTLY REJECTED**:
  - Non-rigid deformable topological mutations in V1 (fluid pouring, chopping/cutting deformable food, cloth folding).
  - Unconstrained pixel-level generative video-to-video hallucination (no 2D diffusion models that warp geometry).
  - Retraining proprietary foundation robot policies from scratch locally; Praxis produces the **physical training fuel (datasets)**.

---

## ⚡ 2. The Computational & Yield Filter

Praxis operates under the principle that **YouTube is a noisy, near-infinite reservoir**. A low yield (e.g. 5%) is completely acceptable, provided that **filtering is 100% automated with zero human review required**:

$$\text{Filter Score} = \text{CameraStability} \times \text{PenetrationConstraint} \times \text{URDFFeasibility}$$

* **Rule of Automated Rejection**: Any extracted clip where the hand mesh penetrates an object surface by $> 1.0\text{ mm}$, or where the inverse kinematics solver fails on the target robot URDF, MUST be discarded immediately without human intervention.
* **The "Workpiece" Primacy Principle**: When extracting manipulation, **the tool and workpiece trajectory in 3D space is primary**; human finger joint positions are secondary proxies. Tool affordance trajectories are robot-agnostic.

---

## 🔄 3. Two-Tier Workflow Architecture

All AI agents and developers implementing features in Praxis must respect the sequential two-tier pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIER 1 : Sub-Video Mining                       │
│  Raw YouTube Video (.mp4)                                              │
│         │                                                              │
│         ▼ (Slicer & Semantic Auditor)                                  │
│  1. Sub-Video Extraction (.mp4) [Only continuous hand-tool action]     │
│  2. Action Metadata (.json) ["What is the person doing?", tool, state] │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 2 : Physical 3D Digital Twin                    │
│  Sub-Video Clip + Metadata                                             │
│         │                                                              │
│         ▼ (Joint Optimization & Contact Sanitizer)                     │
│  1. 3D Object CAD Fitting (Rigid geometries: cylinder, box, lever)     │
│  2. Hand Mesh & Contact Manifold (Zero SDF penetration)                │
│  3. Inverse Dynamics (Mass, inertia, approach vectors)                 │
│  4. URDF Feasibility Audit (Target robot check: Aloha, H1, etc.)       │
│  5. Export to LeRobot (Parquet) & Rerun.io visualization               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 4. Technical Documentation Standard

* **Strict English Language Requirement**: ALL documentation (`README.md`, `docs/*.md`, `AGENTS.md`), code comments, docstrings, JSON schema definitions, and commit messages MUST be written strictly in English.
* **Traceable Grounding**: All metadata must explicitly reference the source video timestamp, frame index, and metric bounding dimensions in millimeters.
* **Standardized Formats**:
  - Sub-videos: H.264 MP4, 30 fps, preserved native aspect ratio.
  - 3D Geometries: Wavefront OBJ / GLTF with metric millimeter scale.
  - Datasets: Hugging Face LeRobot-compliant Apache Parquet / Arrow schemas.

---

## 🤖 5. Specialized Subagents & Modules

1. **`IngestionSlicer` (`src/ingestion/`)**:
   - Downloads videos and cuts physical manipulation sub-videos (`data/clips/`).
   - Produces structured metadata: action verbs, tools, workpiece, camera motion.
2. **`PoseMeshExtractor` (`src/pose_extraction/`)**:
   - Fits 3D hand poses (MANO) and tool bounding boxes across frames.
3. **`PhysicsSanitizer` (`src/physics_sanitizer/`)**:
   - Enforces $L_{\text{SDF}} \ge 0$ (no mesh collision) and snaps contact fingertips to affordance surfaces.
4. **`InverseDynamics` (`src/dynamics/`)**:
   - Evaluates acceleration curves and semantic priors to estimate object mass and required grip force.
5. **`URDFAuditor` (`src/feasibility/`)**:
   - Solves numerical IK against target robot configurations; computes feasibility score ($0-100\%$).
6. **`DatasetExporter` (`src/exporter/`)**:
   - Compiles trajectories into LeRobot-compatible Parquet files and visualizes them in Rerun.io.

---

## 🏗️ 6. Source & Test Separation Directives

* **Domain-Agnostic Generic `src/`**: All modules in `src/` (`src/ingestion/`, `src/pose_extraction/`, `src/physics_sanitizer/`, `src/dynamics/`, `src/feasibility/`, `src/exporter/`, `src/pipeline.py`) MUST remain 100% generic, reusable, and free of video-specific procedural hardcoding.
* **Mirrored `tests/` Hierarchy**: All scene-specific test fixtures, procedural benchmarks, and unit/integration tests MUST reside under `tests/` in packages mirroring `src/` (`tests/ingestion/`, `tests/pose_extraction/`, `tests/physics_sanitizer/`, `tests/dynamics/`, `tests/feasibility/`, `tests/exporter/`, `tests/integration/`).
* **Continuous Anatomical & Robotic Hands**: Hand models must enforce continuous kinematic connectivity (palmar web, thenar eminence, CMC-to-distal-phalanx chains with zero floating segments) and provide unified retargeting to 5-fingered anthropomorphic humanoid robot hands (`DexterousRobotHandModel`).

---

## 🔒 7. Automated Algorithmic Invariants & Solver Mandate (Zero Manual Cheating)

To guarantee scientific reproducibility and production dataset scaling, all physical twin extraction in Praxis MUST follow strict programmatic and physical invariants without manual scene scripting or ad-hoc constant tweaking:

### 1. The Automated Solver Mandate (No Manual Tweaking)
* 3D hand poses and workpiece states MUST be computed programmatically by solver algorithms from video visual observations (keypoints, bounding contours, optical flow, chromatic tracking) projected into metric 3D space.
* Hand positions and finger angles must NEVER be manually hand-tweaked or procedurally hardcoded with ad-hoc constant offsets in `src/`.
* Any discrepancy between vision observations and CAD models must be resolved through **principled, constrained optimization** (e.g. contact-constrained Inverse Kinematics and SDF projection), not manual cheating.

### 2. Anthropometric Invariants & Anatomical Bounds
* **Metric Scale Boundaries**: Total human hand length ($L_{\text{hand}}$, from carpal wrist crease to the distal tip of the middle finger) is an invariant physiological quantity strictly bounded between:
  $$150.0\text{ mm} \le L_{\text{hand}} \le 250.0\text{ mm} \quad (\text{Adult 50th percentile } \approx 184\text{ mm})$$
  Palm breadth across MCP knuckles must be bounded between $65.0\text{ mm}$ and $95.0\text{ mm}$.
  Any extracted hand model below $15\text{ cm}$ or above $25\text{ cm}$ is physically unfeasible and MUST be rejected automatically by `AnthropometricHandBounds`.
* **Rigid Bone Chain Kinematics**: A hand is composed of nested rigid bone cylinders articulated by anatomical joints with physiological Degrees of Freedom (DoFs). Joint angles must respect human range-of-motion limits (MCP, PIP, DIP flexion $[0, 90^\circ-105^\circ]$, spread $[-15^\circ, +15^\circ]$). Hyperextension and impossible backwards bending are strictly forbidden.

### 3. Solid Non-Penetrability Invariant ($L_{\text{SDF}} \ge 0$)
* Both the human hand and the manipulated workpiece/tool are **solid physical bodies**.
* The non-penetration condition MUST hold across all contact surfaces:
  $$\min_{x \in \mathcal{M}_{\text{hand}}} \text{SDF}_{\text{workpiece}}(x) \ge 0.0\text{ mm}$$
* Penetration of fingers or tools into solid rigid bodies is a physical impossibility. Any raw video extraction where fingers intersect solid CAD geometry must be automatically corrected by `PhysicsSanitizer` via outward gradient projection ($\nabla \text{SDF}$) onto the boundary manifold, or rejected if gross penetration exceeds the resolvable threshold.

### 4. Workpiece Rigidity & Compliance Specification
* Workpieces and tools are modeled as **rigid solids** by default ($k \to \infty$).
* When an object or portion thereof is physically compliant or semi-soft (e.g., rubber pads, foam, human skin deformation), its softness must be modeled explicitly via defined compliance parameters (elastic modulus, maximum deformation depth), while strictly preventing unrestrained mesh penetration.

---

## ⚡ 8. Dynamic Limits, Temporal Continuity & Robot Foundation Models (The 3-Method Framework)

Physical demonstrations extracted by Praxis must reflect real-world tabletop manipulation dynamics without kinematic teleportation or unphysical forces.

### 1. Kinematic Velocity, Acceleration & Smoothness Bounds
Tabletop hand manipulation recorded in fixed camera views operates under strict biological motor control bounds:
* **Velocity Bound**:
  $$\max_{t} \|\vec{v}_t\| \le 2.0\text{ m/s} \quad (\approx 7.2\text{ km/h}, \text{nominal fine manipulation } \le 1.2\text{ m/s})$$
* **Acceleration Bound**:
  $$\max_{t} \|\vec{a}_t\| \le 15.0\text{ m/s}^2 \quad (\approx 1.5\,g)$$
* **Minimum-Jerk Biological Smoothness**: Hand trajectory generation follows minimal jerk ($\|\dddot{x}\| \le 100.0\text{ m/s}^3$). High-frequency jitter or discontinuous jumps are flagged and rejected by `PhysicalContinuityAuditor`.

### 2. Inter-Frame Temporal Continuity ($C^1 / C^2$ Smoothness)
* Consecutive frames $F_t$ and $F_{t+1}$ must exhibit continuous physical transitions:
  $$\|\vec{x}_{t+1} - \vec{x}_t\| \le v_{\max} \Delta t$$
* Spatial teleportation, tracking dropouts, or instantaneous scale changes between adjacent frames are strictly forbidden. Any demonstration failing inter-frame continuity is automatically rejected.

### 3. Volumetric Mass Priors & Grasp Affordance Coherence
* **Volumetric Density Range**: Handheld workpieces typically have effective bulk densities between $0.2\text{ kg/dm}^3$ (hollow/composite) and $7.9\text{ kg/dm}^3$ (solid steel). Common handheld manipulation tools weigh between $50\text{ g}$ and $800\text{ g}$.
* **Friction Cone & Grasp Compatibility**:
  - `PRECISION_PINCH` (fingertip index-thumb): Maximum squeeze force $F_N \approx 45\text{ N}$. Under Coulomb friction ($\mu \approx 0.4$), maximum lift payload without support from underneath is strictly $\le 1.5\text{ kg}$. If a heavier mass is detected without palmar support, the sequence is physically inconsistent and rejected.
  - `PALMAR_CRADLE` (palm cupping from below): Normal reaction directly opposes gravity ($R_z \ge m g$), supporting payloads up to $8.0\text{ kg}$.
  - `POWER_WRAP`: Full cylindrical grasp with $F_N \le 350\text{ N}$, payload up to $5.0\text{ kg}$.

### 4. The 3-Method Robot Exploitation Architecture
Praxis feeds physical manipulation data into modern robot imitation learning (LeRobot, ACT, Diffusion Policy, OpenVLA) using three complementary strategies:
1. **Method 1: Automated Curation Filter Gate (Yield Filter)**:
   Discards 100% of non-physical demonstrations (speed > 2.0 m/s, penetration > 1.0 mm, unfeasible grasps), ensuring foundation models never learn from non-physical artifacts.
2. **Method 2: Hybrid Impedance & Effort Action Space**:
   Dataset action tensors serialize both kinematic targets and impedance compliance:
   $$a_t = [q_{\text{target}}, \dot{q}_{\text{target}}, F_{\text{grip}}, K_{\text{stiffness}}]$$
   allowing target robots to perform contact-rich tasks without excessive motor torques or mechanical destruction.
3. **Method 3: Physics Tokens & VLA Conditioning Embeddings**:
   Each episode exports compact physical descriptors (`mass_class`, `grasp_type`, `continuity_score`, and feature vector embeddings) into the Parquet dataset, allowing Vision-Language-Action Transformers to condition policies directly on physical dynamics.



