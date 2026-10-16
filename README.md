# Praxis: Video-to-Physical-Digital-Twin (V2PDT)

Automated engine transforming monocular in-the-wild web videos (YouTube) into collision-free, contact-certified 3D physical digital twins for humanoid robot manipulation.

---

## 📌 Repository Structure

```
Praxis/
├── AGENTS.md                       # Core project philosophy, ROI filter & agent directives
├── README.md                       # Project overview & quick start
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/                        # Ingested raw YouTube video files (.mp4)
│   ├── clips/                      # Action sub-videos (.mp4) & semantic metadata (.json)
│   └── processed/                  # Certified 3D scenes (.obj) & LeRobot datasets (.parquet)
│
├── doc/
│   ├── objectives.md               # Vision, success metrics (KPIs) & Operational Design Domain
│   ├── subagents_architecture.md   # Subagent specifications, prompt patterns & inputs/outputs
│   ├── roadmap.md                  # 5-phase execution plan (workstation to GCP)
│   ├── architecture.md             # Subsystem workflows, SDF formulation & data schemas
│   └── sources_and_provenance.md   # Video sources, timestamps & licenses
│
├── scratch/                        # Exploratory scripts, spikes, and visual renderers
├── scripts/
│   └── gcp/                        # Scaled batch execution scripts for Google Cloud Platform
└── src/
    ├── ingestion/                  # Video download, frame sampling & action sub-video slicing
    ├── pose_extraction/            # 3D hand (MANO) & tool pose recovery
    ├── physics_sanitizer/          # SDF non-penetration & contact manifold optimizer
    ├── dynamics/                   # Visual inverse dynamics (mass, inertia, grip forces)
    ├── feasibility/                # Numerical IK solver & robot URDF screening (ALOHA, H1)
    ├── exporter/                   # LeRobot Parquet serialization & dataset formatting
    └── pipeline.py                 # Central CLI pipeline orchestrator
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Extract Action Sub-Videos & Metadata
```bash
py -m src.ingestion.slicer --video data/raw/01_LPL_DropKey_RRunwAI7N5Q.mp4 --t_start 18.0 --t_end 26.5
```

### 3. Run End-to-End Pipeline
```bash
py src/pipeline.py --input_dir data/raw --output_dir data/processed
```

### 4. Run Test Suite
```bash
py -m pytest tests/ -v
```

---

## 📖 Key Documentation
* Read [AGENTS.md](file:///c:/Silvere/Encours/Developpement/Praxis/AGENTS.md) for architectural philosophy and agent rules.
* Read [doc/objectives.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/objectives.md) for project KPIs and scope boundaries.
* Read [doc/architecture.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/architecture.md) for system data flow and SDF physics formulations.
* Read [doc/roadmap.md](file:///c:/Silvere/Encours/Developpement/Praxis/doc/roadmap.md) for milestones.
* Read [PROJECT_STATE.md](file:///c:/Silvere/Encours/Developpement/Praxis/PROJECT_STATE.md) for session memory and completed assets inventory.

