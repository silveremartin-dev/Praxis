# Praxis Project State & Capabilities Audit

> **Updated**: October 2026  
> **Status**: **100% Passing Tests (41/41)** | **Production Ready V1.2**  
> **Core Mission**: Video-to-Physical-Digital-Twin (V2PDT) for Humanoid Robotics.

---

## 1. High-Level Summary

Praxis is an automated data-engineering engine that converts monocular in-the-wild manipulation videos into physically certified, collision-free 3D digital twins formatted for foundation robot policy learning (Hugging Face LeRobot / Rerun.io).

All key subsystems requested have been implemented, tested, and validated:
1. **Interactive Web Browser ("Praxis Studio")**: A complete single-page WebGL application with side-by-side synchronized video playback, interactive Three.js 3D digital twin viewer, contact manifold inspection, MuJoCo forward dynamics simulation, and video slicing tools.
2. **One-Click Startup Scripts**:
   - Windows: [`start_web_browser.bat`](file:///c:/Silvere/Encours/Developpement/Praxis/start_web_browser.bat)
   - Linux / macOS: [`start_web_browser.sh`](file:///c:/Silvere/Encours/Developpement/Praxis/start_web_browser.sh)
3. **Authentic Humanoid Robot Specifications**: Real kinematic and dynamic parameters for **Unitree H1** (Inspire DexHand 6 DoF), **Tesla Optimus** (Gen-2 Hand 11 DoF), **ALOHA** (ViperX dual-arm), and **Franka Emika Panda**.
4. **Native MuJoCo 3.x Simulation**: Fast forward dynamics stepping via Python C bindings (`mujoco`), verifying dynamic stability and bounding velocity/explosions under gravity.
5. **Continuous Anatomical Human Hands**: Palmar web, thenar eminence, and analytical 2-bone IK solver (`solve_two_bone_ik`) ensuring zero detached thumb or floating phalanx artifacts.
6. **Strict Separation of Generic Code & Tests**: All reusable code resides in generic domain-agnostic `src/` modules, while scene-specific fixtures and tests mirror the hierarchy under `tests/`.

---

## 2. Test Suite Status (41/41 Passing — 100%)

Run with: `py -m pytest tests/ -v`

| Test Category | File | Tests | Status |
| :--- | :--- | :---: | :---: |
| **Ingestion** | `tests/ingestion/test_inspector.py`, `test_slicer.py` | 5 | **PASS** |
| **Pose & Reconstruction** | `tests/pose_extraction/test_scene_reconstructions.py` | 4 | **PASS** |
| **Physics & SDF** | `tests/physics_sanitizer/test_geometry_and_contacts.py` | 3 | **PASS** |
| **Inverse Dynamics** | `tests/dynamics/test_inverse_dynamics.py` | 5 | **PASS** |
| **Feasibility & MuJoCo** | `tests/feasibility/test_urdf_auditor.py`, `test_mujoco_verifier.py` | 7 | **PASS** |
| **LeRobot Exporter** | `tests/exporter/test_lerobot_exporter.py`, `test_lerobot_schema.py` | 5 | **PASS** |
| **Integration E2E** | `tests/integration/test_pipeline_e2e.py` | 2 | **PASS** |
| **Praxis Studio** | `tests/studio/test_studio_server.py` | 10 | **PASS** |
| **Total** | | **41** | **100% PASS** |

---

## 3. Certified Demonstrations Catalog

| Demonstration ID | Workpiece & Tool | Action Verb | Duration | SDF Collision | MuJoCo 3.x |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `07_lpl_masterlock` | Master Lock No. 3 (195g) + 0.5mm steel probe | Bimanual bypass, probe slide, needle drop & tapping | 10.0s | Certified ($\ge 0.0\text{ mm}$) | Stable |
| `01_lpl_dropkey` | Stainless steel folding key (47g) | 3-finger precision pinch & 90° stem folding | 8.53s | Certified ($\ge 0.0\text{ mm}$) | Stable |
| `05_watch_zodiac` | Miniature brass gear (0.02g) + Dumont #5 tweezers | Micro-tweezer vertical extraction from jewel bearing | 13.01s | Certified ($\ge 0.0\text{ mm}$) | Stable |
| `06_singer_sewingmachine` | Cast iron faceplate (285g) + gloved hands | Bimanual high-force lateral slide & detachment | 11.04s | Certified ($\ge 0.0\text{ mm}$) | Stable |

---

## 4. How to Launch and Inspect

Double-click `start_web_browser.bat` (Windows) or execute `./start_web_browser.sh` (Linux/macOS).  
Alternatively:
```bash
py -m uvicorn src.studio.server:app --port 8000
```
Open **`http://localhost:8000`** in Chrome, Firefox, or Edge.
