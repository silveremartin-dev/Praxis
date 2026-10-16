# Project Objectives & Success Metrics: Praxis (V2PDT)

## 1. Vision & Value Proposition
**Praxis (Video-to-Physical-Digital-Twin)** is an automated pipeline designed to bridge the massive gap between uncurated web video data and physical humanoid robot learning.

Instead of training visuomotor policies directly on raw, uncalibrated pixels, Praxis extracts:
1. **Curated Sub-Videos**: Isolated, continuous sequences of physical hand manipulation.
2. **Semantic Action Metadata**: Structured task graph descriptions answering *"what is the person doing?"*, identifying tools, workpieces, and mechanical states.
3. **Collision-Free 3D Digital Twins**: Joint reconstruction of hands and rigid objects enforcing zero surface interpenetration via Signed Distance Fields (SDF).
4. **Physical Dynamics**: Inferred masses, inertias, and contact force profiles via visual inverse dynamics.
5. **URDF Transferability Scores**: Automated certification that target humanoid robots (ALOHA, Unitree H1, Fourier GR-1) can physically execute the motion without violating joint or velocity limits.

---

## 2. Key Objectives & Target KPIs

| Objective | Target KPI | Engineering Description |
| :--- | :--- | :--- |
| **Contact Geometric Fidelity** | Surface penetration $< 1.0\text{ mm}$ | Complete elimination of mesh artifacts where fingers pass through solid tool bodies. |
| **Automated Rejection Filter** | Silent failure rate $< 2\%$ | Physics engine verification (MuJoCo/Isaac Sim) without numerical explosion or causal confusion. |
| **Dataset Standardization** | Hugging Face LeRobot format | Native Parquet / Arrow schemas directly loadable by modern imitation learning pipelines. |
| **Robot-Agnostic Affordances** | Universal 6D Tool Trajectory | Focus on workpiece and tool trajectories so motions can be transferred to any robot morphology. |
| **Kinematic Feasibility Screening** | IK score $\ge 85\%$ | Rejection of motions requiring superhuman joint ranges or impossible velocities. |

---

## 3. Operational Design Domain (ODD)

To guarantee high physical consistency without encountering unsolvable simulation bottlenecks:
* **In-Scope Objects (V1):** Rigid, standardized mechanical components (screwdrivers, keys, padlocks, bolts, electronic assemblies, switches, rotary knobs).
* **Out-of-Scope (V1):** Highly deformable non-rigid objects (textiles, dough, liquids) and destructive transformations (sawing, grinding, food chopping).
* **Target Video Distributions:** Fixed macro and overhead perspectives with uniform lighting and deliberate hand motions (e.g., LockPickingLawyer, iFixit teardowns, precision horology, PC building).
