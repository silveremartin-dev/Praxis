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

* **Strict English Language Requirement**: ALL documentation (`README.md`, `doc/*.md`, `AGENTS.md`), code comments, docstrings, JSON schema definitions, and commit messages MUST be written strictly in English.
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
