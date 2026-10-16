# PROJECT_STATE.md — Complete Handover & Session Memory for Praxis

> **Critical Note for Future Agent / Antigravity Sessions:**
> This document preserves the entire intellectual, architectural, and technical history of the conversation from the initial project inception through repository setup, video ingestion, 3D digital twin prototyping, and test suite execution. 
> If the project directory has been renamed (e.g. from `Pose` to `Praxis`) or if a new conversation session begins, **this document serves as the absolute ground truth for continuity**.

---

## 1. Executive Identity & Repository Coordinates

* **Project Name:** **Praxis** (*Praxis: Video-to-Physical-Digital-Twin Engine for Humanoid Robotics*)
* **GitHub Remote:** `https://github.com/silveremartin-dev/Praxis.git`
* **Active Branch:** `main`
* **License:** MIT License ([LICENSE](file:///c:/Silvere/Encours/Developpement/Pose/LICENSE))
* **Language Requirement:** All code comments, docstrings, JSON schemas, commit messages, and documentation must be written strictly in **English**.
* **Directives & Rules:** Governed by [AGENTS.md](file:///c:/Silvere/Encours/Developpement/Pose/AGENTS.md).

---

## 2. Chronology of Design Decisions & Scientific Rationale

### A. The Core Robotics Problem Addressed
Humanoid robot training suffers from a crippling data collection bottleneck: manual teleoperation (humans in VR rigs) cannot scale to internet-scale volume. However, training policies directly on raw YouTube pixels fails due to:
1. **The Kinematic Illusion**: Monocular 3D tracking hallucinates finger penetration through rigid tool handles or floating meshes without contact.
2. **Missing Dynamics**: RGB video lacks tactile feedback, motor torques, and contact friction.
3. **The Embodiment Gap**: Human hands (27 DoF, compliant skin) do not match rigid 3-to-6 DoF robot grippers.

### B. Strategic Architectural Decisions
1. **Reject Pixel-to-Pixel 1st-Person Generative Hallucination**:
   Do NOT attempt to synthesize fake egocentric views via video diffusion. Generative pixel models suffer from metric scale drift and temporal jitter (a 2 mm error breaks physical manipulation).
2. **Workpiece Primacy Principle**:
   Focus on the **3D trajectory of the tool and workpiece**. A screwdriver trajectory or key insertion vector is **robot-agnostic** (it transfers identically to an ALOHA arm, Unitree H1, or Tesla Optimus).
3. **The Automated Yield Filter (Low Yield is Acceptable)**:
   YouTube is an infinite reservoir ($N \to \infty$). A 5% yield is completely acceptable, provided that **filtering is 100% automated with zero human review required**:
   $$\text{Filter Score} = \text{CameraStability} \times \text{PenetrationConstraint} \times \text{URDFFeasibility}$$
4. **Two-Tier Sequential Workflow**:
   * **Tier 1 (Sub-Video Mining)**: Slice raw videos into clean standalone `.mp4` sub-videos containing continuous manipulation, paired with structured `.json` task metadata answering *"what is the person doing?"*.
   * **Tier 2 (Physical 3D Digital Twin)**: Fit canonical CAD models (Objaverse), apply analytical Signed Distance Field (SDF) non-penetration penalties ($\text{SDF} \ge 0$), infer mass from visual inverse dynamics, audit against target robot URDFs, and export to Hugging Face **LeRobot** format (Parquet / Arrow).

---

## 3. Inventory of Completed Assets & Artifacts

### A. Downloaded Raw Videos (`data/raw/`)
All videos downloaded, verified, and multiplexed (H.264 + AAC with `+faststart`):
1. `01_LPL_DropKey_RRunwAI7N5Q.mp4` (02:26) — LockPickingLawyer Drop Key.
2. `02_iFixit_DualSense_6K_jcXrtrx8.mp4` (05:25) — iFixit PS5 DualSense Teardown.
3. `03_DIY_CementWD40_mA3QtWz7RMQ.mp4` (15:37) — DIY aluminum can cutting.
4. `04_Cuisine_Installation_KOBqgv-TpyA.mp4` (35:12) — In-situ kitchen workshop assembly.
5. `05_Watch_Zodiac_f8NFSwb6VQ8.mp4` (58:55) — Wristwatch Revival horology chronograph restoration.
6. `06_Singer_SewingMachine_BMNgIfLHfjk.mp4` (31:45) — 1894 Singer sewing machine restoration.
7. `LPL_AmericanLock_1100.mp4` (08:05) — LockPickingLawyer American Lock 1100 (re-muxed and verified).
8. `LPL_MasterLock_No3.mp4` (02:11) — Master Lock No. 3.

### B. Tier 1 Extracted Action Sub-Videos & Metadata (`data/clips/`)
Each sub-video is an autonomous MP4 clip paired with a standardized JSON task annotation:
* `01_LPL_DropKey_RRunwAI7N5Q_clip_18s_26s.mp4` + `.json` (8.53s, 256 frames): 3-finger precision pinch and 90° drop link articulation.
* `02_iFixit_DualSense_6K_jcXrtrx8_clip_118s_130s.mp4` + `.json` (12.01s, 288 frames): Controller battery pack comparison and precision driver prep.
* `03_DIY_CementWD40_mA3QtWz7RMQ_clip_175s_188s.mp4` + `.json` (13.03s, 391 frames): Bimanual circumferential slicing of can with utility knife.
* `05_Watch_Zodiac_f8NFSwb6VQ8_clip_595s_608s.mp4` + `.json` (13.01s, 390 frames): Tweezer extraction of miniature brass gear from jewel bearing.
* `06_Singer_SewingMachine_BMNgIfLHfjk_clip_118s_129s.mp4` + `.json` (11.04s, 276 frames): Bimanual detachment of heavy cast-iron faceplate.

### C. Tier 2 Physical 3D Reconstruction & Renders (`data/processed/`)
* `01_lpl_dropkey_scene_3d.obj`: Complete 3D mesh (674 vertices, 1,316 faces) featuring workbench, cylindrical drop key stem, stop collar, ring, articulated link, and finger contact spheres.
* `01_lpl_dropkey_metadata.json`: Full LeRobot-compliant state-action schema with contact vectors and metric dimensions.
* `01_lpl_dropkey_render_3d.png`: High-resolution 3D isometric render preview.

### D. Core Codebase Modules (`src/`)
* `src/ingestion/slicer.py`: Slices continuous manipulation sub-videos (`.mp4`) and generates structured JSON metadata.
* `src/ingestion/inspector.py`: Reads video stream properties (FPS, resolution, duration) and samples keyframes.
* `src/physics_sanitizer/sdf.py`: Exact analytical Signed Distance Field solvers ($O(1)$) for cylinders, boxes, and spheres enforcing zero surface penetration.
* `src/pose_extraction/reconstruct_dropkey_scene.py`: Procedural 3D scene builder and LeRobot metadata exporter.
* `src/pipeline.py`: Main CLI orchestrator.

### E. Automated Test Suite (`tests/`) — 100% Passing
12 automated unit and integration tests passing in `pytest`:
* `tests/test_slicer.py`
* `tests/test_inspector.py`
* `tests/test_geometry_and_contacts.py`
* `tests/test_lerobot_schema.py`
* `tests/test_pipeline_e2e.py`

### F. Documentation (`doc/`)
* [AGENTS.md](file:///c:/Silvere/Encours/Developpement/Pose/AGENTS.md) : Root directives and philosophy.
* [doc/objectives.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/objectives.md) : KPIs and Operational Design Domain.
* [doc/architecture.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/architecture.md) : System diagrams and data models.
* [doc/roadmap.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/roadmap.md) : 5-phase roadmap.
* [doc/subagents_architecture.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/subagents_architecture.md) : Subagent prompt and role specs.
* [doc/sources_and_provenance.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/sources_and_provenance.md) : Full video attribution and software licenses.

---

## 4. Next Steps for Immediate Execution

When the project folder is renamed to `Praxis`:
1. Re-open terminal/IDE in the renamed path `c:\Silvere\Encours\Developpement\Praxis`.
2. Run `pytest tests/ -v` to confirm environment integrity.
3. Advance Tier 2:
   - Implement automated 3D reconstruction for the watch gear extraction (`05_Watch_Zodiac`) and Singer faceplate (`06_Singer_SewingMachine`).
   - Connect the URDF kinematic feasibility checker (`src/feasibility/urdf_auditor.py`) against ALOHA / Unitree H1 URDF models.
   - Package all processed trajectories into Apache Parquet tables complying with Hugging Face LeRobot format.
