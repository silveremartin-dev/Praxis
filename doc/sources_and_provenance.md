# Sources, Provenance & External Materials

This document provides complete attribution, provenance records, and licensing references for all external video streams, 3D libraries, and models utilized by Praxis.

---

## 1. Primary Video Corpora (`data/raw/`)

All video streams ingested into Praxis are recorded from public demonstrations on YouTube. They remain the intellectual property of their original creators. Praxis extracts purely kinematic trajectories, geometric bounding volumes, and physical parameters under fair research and data-engineering use.

| Reference ID | Source Title | Channel / Creator | YouTube URL | Duration | Key Manipulation Content |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `01_LPL_DropKey` | *[1619] Drop Key Locks: A Different Approach* | **LockPickingLawyer** | [youtu.be/RRunwAI7N5Q](https://www.youtube.com/watch?v=RRunwAI7N5Q) | 02:26 | Precision bimanual presentation and 90° folding of a cylindrical drop-link tool. |
| `02_iFixit_DualSense` | *PS5 DualSense Edge Controller Teardown* | **iFixit** | [youtu.be/6K_jcXrtrx8](https://www.youtube.com/watch?v=6K_jcXrtrx8) | 05:25 | Precision screwdriver engagement, prying tools, and battery connector detachment. |
| `03_DIY_CementWD40` | *Mix WD-40 with CEMENT!* | **Creative DIY** | [youtu.be/mA3QtWz7RMQ](https://www.youtube.com/watch?v=mA3QtWz7RMQ) | 15:37 | Gross manual manipulation of bottles, cans, trowels, and rigid containers. |
| `04_Cuisine_Installation`| *J'installe seul ma cuisine en 1 semaine* | **Jimmy's Workshop** | [youtu.be/KOBqgv-TpyA](https://www.youtube.com/watch?v=KOBqgv-TpyA) | 35:12 | In-situ workshop assembly: electric drill/driver, spirit levels, furniture brackets. |
| `05_Watch_Zodiac` | *Restoring an Extremely Dirty Zodiac Chronograph* | **Wristwatch Revival** (Marshall Sutcliffe) | [youtu.be/f8NFSwb6VQ8](https://www.youtube.com/watch?v=f8NFSwb6VQ8) | 58:55 | Precision horology: tweezers, fine watchmaker screwdrivers, gear train placement. |
| `LPL_AmericanLock_1100`| *[328] American Lock 1100 Picked and Gutted* | **LockPickingLawyer** | [youtu.be/StZCIDBKI-A](https://www.youtube.com/watch?v=StZCIDBKI-A) | 08:05 | Pin tumbler disassembly, tweezer extraction of springs and pins, key rotation. |
| `LPL_MasterLock_No3` | *[833] The Redesigned Master Lock No. 3* | **LockPickingLawyer** | [youtu.be/uOO8yA2eU7M](https://www.youtube.com/watch?v=uOO8yA2eU7M) | 02:11 | Padlock shackle release, direct keyway insertion and torque tensioning. |

---

## 2. 3D Meshes, Kinematic Models & Robotics Schemas

| Asset / Component | Source Project / Institution | License / Terms | Integration in Praxis |
| :--- | :--- | :--- | :--- |
| **MANO Hand Model** | Max Planck Institute for Intelligent Systems | Research / Non-commercial | 21-joint 3D kinematic structure and 778-vertex surface mesh used in `src/pose_extraction/`. |
| **Objaverse-LVIS** | Allen Institute for AI / Columbia Univ. | CC-BY 4.0 / ODC-By | Source bank of 3D CAD meshes matching everyday rigid workshop tools and hardware. |
| **LeRobot Format** | Hugging Face | Apache 2.0 | Standardized dataset schema (Apache Parquet / Arrow) for robot imitation learning. |
| **Rerun.io SDK** | Rerun.io Inc. | MIT / Apache 2.0 | Spatial 3D multimodal timeline visualization engine. |

---

## 3. Core Software Libraries & Tooling

* **`yt-dlp`**: Command-line audio/video downloader ([Unlicense](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE)).
* **`imageio-ffmpeg`**: Embedded cross-platform FFmpeg static binaries ([GPL / LGPL v3](https://github.com/imageio/imageio-ffmpeg)).
* **`opencv-python`**: Computer vision library used for frame extraction and video stream serialization ([Apache 2.0](https://github.com/opencv/opencv-python/blob/master/LICENSE.txt)).
* **`trimesh`**: Pure Python triangular mesh processing and signed distance fields ([MIT License](https://github.com/mikedh/trimesh/blob/master/LICENSE.md)).
* **`PyTorch`**: Tensor computation and automatic differentiation ([Modified BSD](https://github.com/pytorch/pytorch/blob/main/LICENSE)).

---

## 4. Software License

The Praxis codebase, tools, and transformation algorithms are released under the **MIT License**. See [LICENSE](file:///c:/Silvere/Encours/Developpement/Pose/LICENSE) for complete legal terms.
