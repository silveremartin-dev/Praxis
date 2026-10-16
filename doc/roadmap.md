# Development Roadmap: Praxis (V2PDT)

This roadmap outlines the 5 execution phases of Praxis, progressing from local workstation prototyping to scaled cloud processing on Google Cloud Platform (GCP).

---

## Phase 1: Foundation, Video Slicing & Pilot Data (Current)
- [x] Repository initialization, structure (`data/`, `doc/`, `scratch/`, `scripts/gcp/`, `src/`).
- [x] Core collaboration guidelines and architecture specifications (`AGENTS.md`, `doc/*.md`).
- [x] Ingestion of reference YouTube target videos into `data/raw/` (LockPickingLawyer, iFixit, horology, DIY).
- [ ] Automated sub-video extraction pipeline: slicing continuous manipulation segments into `data/clips/*.mp4`.
- [ ] Structured metadata generation ("What is the person doing?", tool IDs, action verbs).

---

## Phase 2: Kinematic Extraction & 3D Visualization
- [ ] Integration of 3D hand mesh recovery (HaMeR / MANO) in `src/pose_extraction/`.
- [ ] Tracking and 2D/3D segmentation of manipulated tools via SAM 2.
- [ ] Real-time synchronized inspection viewer via **Rerun.io** (RGB video alongside interactive 3D skeleton).
- [ ] Monocular metric scale calibration anchored against known CAD tool dimensions.

---

## Phase 3: Physical Contact & Penetration Sanitization
- [ ] Differentiable Signed Distance Field (SDF) constraint solver (`src/physics_sanitizer/sdf_optimizer.py`).
- [ ] Enforcement of force-closure contact manifolds during active tool grasp phases.
- [ ] "Before vs. After" benchmark showing elimination of mesh collision and floating grasps.
- [ ] Export of sanitized demonstrations to Hugging Face **LeRobot** format (Parquet / Arrow).

---

## Phase 4: Robot Feasibility Filtering & MuJoCo Verification
- [ ] Target robot URDF model integration (ALOHA bi-manual arm, Unitree H1 humanoid).
- [ ] Numerical Inverse Kinematics (IK) feasibility auditor (`src/feasibility/urdf_filter.py`).
- [ ] Direct simulation replay in **MuJoCo** to verify dynamic stability without numerical explosion.
- [ ] Automated rejection filter: sequences with $> 2\%$ unrecoverable errors are discarded with zero human overhead.

---

## Phase 5: Cloud Scale & Dataset Release (Google Cloud Platform)
- [ ] Docker containerization with NVIDIA CUDA acceleration.
- [ ] Batch processing pipeline for Google Compute Engine (L4/T4 GPUs) / Cloud Run Jobs (`scripts/gcp/`).
- [ ] Automated harvesting across prioritized YouTube playlists.
- [ ] First public release of certified manipulation dataset on Hugging Face Hub.
