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

### C. Tier 2 Physical 3D Digital Twins, LeRobot Datasets & Visual Assets (`data/processed/` & `assets/images/`)
Each processed demonstration includes the verified 3D mesh (`.obj`), structured kinematic metadata (`.json`), official Hugging Face LeRobot dataset table (`.parquet`), side-by-side comparison JPEG, and 360° rotating animated GIF:
1. **Drop Key Demonstration** (`01_lpl_dropkey`):
   * `01_lpl_dropkey_scene_3d.obj`: Complete 3D mesh (674 vertices, 1,316 faces).
   * `01_lpl_dropkey_metadata.json`: LeRobot-compliant state-action schema with contact vectors and metric dimensions.
   * `01_lpl_dropkey_metadata.parquet`: 255 frames, 10 columns (state, action, contact force, zero-penetration SDF, feasibility).
   * `assets/images/01_lpl_dropkey_comparison.jpg`: High-resolution side-by-side video vs. 3D digital twin.
   * `assets/images/01_lpl_dropkey_animation.gif`: Smooth 360° rotating orbit GIF.
2. **Watchmaker Gear Extraction** (`05_watch_zodiac`):
   * `05_watch_zodiac_scene_3d.obj`: 912 vertices, 1,780 faces (movement holder, mainplate, jewel bearing, pinion gear, #5 Dumont tweezers, stabilizing thumb).
   * `05_watch_zodiac_metadata.json`: Micro-manipulation LeRobot schema with 0.45N tweezer grip forces.
   * `05_watch_zodiac_metadata.parquet`: 390 frames, 10 columns.
   * `assets/images/05_watch_zodiac_comparison.jpg` & `assets/images/05_watch_zodiac_animation.gif`.
3. **Singer Faceplate Detachment** (`06_singer_sewingmachine`):
   * `06_singer_sewingmachine_scene_3d.obj`: 828 vertices, 1,608 faces (cast iron machine head, filigree faceplate, needle/presser bar, bimanual gloved contacts).
   * `06_singer_sewingmachine_metadata.json`: Bimanual opposing grip with 9.5N outward detachment vector.
   * `06_singer_sewingmachine_metadata.parquet`: 276 frames, 10 columns.
4. **Master Lock No. 3 Bypass Decoding** (`07_lpl_masterlock`):
   * `07_lpl_masterlock_scene_3d.obj`: 1,262 vertices, 2,480 faces (laminated steel padlock body, red bumper, open shackle, bypass decoder needle, precision pinch grip).
   * `07_lpl_masterlock_metadata.json`: Precision insertion LeRobot schema with 1.8N insertion thrust.
   * `07_lpl_masterlock_metadata.parquet`: 300 frames, 10 columns.
   * `assets/images/07_lpl_masterlock_end_to_end_comparison.jpg`: 3-panel still comparison (Video vs. 3D Twin vs. ALOHA Robot Gripper).
   * `assets/images/07_lpl_masterlock_end_to_end_animation.gif`: Dynamic 3-panel continuous execution animation.

### D. Core Codebase Modules (`src/`)
* `src/ingestion/slicer.py`: Slices continuous manipulation sub-videos (`.mp4`) and generates structured JSON metadata.
* `src/ingestion/inspector.py`: Reads video stream properties (FPS, resolution, duration) and samples keyframes.
* `src/physics_sanitizer/sdf.py`: Exact analytical Signed Distance Field solvers ($O(1)$) for cylinders, boxes, and spheres enforcing zero surface penetration.
* `src/pose_extraction/reconstruct_dropkey_scene.py`: Procedural 3D scene builder for Drop Key.
* `src/pose_extraction/reconstruct_watch_scene.py`: Procedural 3D scene builder for horology gear micro-extraction.
* `src/pose_extraction/reconstruct_singer_scene.py`: Procedural 3D scene builder for antique restoration faceplate detachment.
* `src/pose_extraction/reconstruct_masterlock_scene.py`: Procedural 3D scene builder for Master Lock bypass decoding.
* `src/feasibility/urdf_auditor.py`: Kinematic feasibility auditor with ALOHA ViperX, Unitree H1, and Franka Panda IK verification and automated rejection threshold ($\ge 85\%$).
* `src/feasibility/mujoco_verifier.py`: Automated MuJoCo physics simulation verifier executing forward dynamics to certify stability and contact compliance without numerical explosion.
* `src/dynamics/inverse_dynamics.py`: Visual inverse dynamics engine computing mass from material density, 3D inertia tensors, trajectory acceleration, and Coulomb friction normal forces.
* `src/exporter/lerobot_exporter.py`: Hugging Face LeRobot Apache Parquet serializer with schema validation.
* `src/pipeline.py`: Main CLI orchestrator connecting all modules end-to-end.
* `scripts/render_end_to_end_robot_pipeline.py`: 3-panel dynamic animation and comparison JPEG renderer.

### E. Automated Test Suite (`tests/`) — 100% Passing
30 automated unit and integration tests passing in `pytest` (`py -m pytest tests/ -v`):
* `tests/test_slicer.py` (2 tests)
* `tests/test_inspector.py` (3 tests)
* `tests/test_geometry_and_contacts.py` (3 tests)
* `tests/test_scene_reconstructions.py` (3 tests: Watch, Singer, MasterLock)
* `tests/test_urdf_auditor.py` (6 tests)
* `tests/test_mujoco_verifier.py` (1 test)
* `tests/test_inverse_dynamics.py` (5 tests)
* `tests/test_lerobot_exporter.py` (3 tests)
* `tests/test_lerobot_schema.py` (2 tests)
* `tests/test_pipeline_e2e.py` (2 tests)

### F. Consolidated Documentation (`doc/`)
All documentation files use uppercase `.md` naming:
* [AGENTS.md](file:///c:/Silvere/Encours/Developpement/Praxis/AGENTS.md) : Root directives, automated rejection filter formulation, and coding standards.
* [doc/ARCHITECTURE.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/ARCHITECTURE.md) : Master unified technical specification, mathematical models (SDF, friction cone, manipulability, MuJoCo), data schemas, and roadmap.
* [doc/THROUGHPUT_AND_UNIT_ECONOMICS.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/THROUGHPUT_AND_UNIT_ECONOMICS.md) : 7-stage attrition funnel ("The Reality Cliff"), VLM API fees, cloud infrastructure, and comparative data sourcing benchmark.
* [doc/SOURCES_AND_PROVENANCE.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/SOURCES_AND_PROVENANCE.md) : Video attribution, timestamps, and software licenses.
* [README.md](file:///c:/Silvere/Encours/Developpement/Praxis/README.md) : Public repository showcase with end-to-end 3-panel robot retargeting visuals and financial analysis.


---

## 4. Next Steps for Immediate Execution

1. **Cloud Batch Pipeline (Phase 5)**:
   - Prepare Dockerfile with NVIDIA CUDA acceleration and GCP Cloud Run Job script (`scripts/gcp/run_batch.sh`).
2. **Automated Playlist Ingestion**:
   - Ingest additional high-yield channels (LPL, iFixit, horology) in batch.
3. **Public Dataset Release**:
   - Publish initial set of certified manipulation episodes to Hugging Face Hub under `Praxis/manipulation-digital-twins-v1`.

