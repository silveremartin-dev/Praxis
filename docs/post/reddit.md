# Reddit launch post — Praxis

## Suggested subreddits

- **r/robotics**: main audience. People working on imitation learning and data collection can judge the "workpiece primacy" idea and the rejection filters.
- **r/MachineLearning**: post under the `[P]` tag. Focus on the data-quality question: physics-based filtering of web video for robot learning.
- **r/LeRobot** / Hugging Face community forum: the export format is LeRobot Parquet, so feedback on schema compliance is valuable there.
- **r/computervision**: monocular 3D hand/object pose, contact plausibility, SDF constraints.
- **r/reinforcementlearning** (optional): shorter version, framed around demonstration quality rather than RL itself.

---

**Title:** [P] Praxis: an early prototype for turning YouTube manipulation videos into physically-filtered robot demonstrations (SDF non-penetration, IK feasibility, MuJoCo check, LeRobot export)

---

Hi all,

I've open-sourced **Praxis**, a research prototype exploring one question: *can we mine web videos (lockpicking, watch repair, teardowns…) for robot manipulation demonstrations, while automatically throwing away everything that isn't physically plausible?* It's early, and I'm posting for critical feedback before scaling anything.

### Motivation

Teleoperation (VR rigs, ALOHA) gives clean data but doesn't scale. Web video is effectively unlimited, but training on it directly runs into three well-known problems:

1. **Kinematic illusions**: monocular 3D tracking puts fingers *through* tool handles or leaves objects floating.
2. **Missing dynamics**: no forces, torques or friction in RGB.
3. **Embodiment gap**: a 27-DoF human hand isn't a parallel gripper.

### The approach

- **Workpiece primacy**: focus on the 3D trajectory of the *tool and workpiece* (key insertion vector, screwdriver axis), which is largely robot-agnostic, rather than on the hand itself.
- **Low yield is fine, if rejection is automatic**: the input reservoir is huge, so the pipeline aggressively discards clips and only keeps what passes every check, with no human review in the loop.

### Pipeline (what's in the repo)

1. **Ingestion**: video inspection (FPS, resolution, keyframes) and slicing into action clips with JSON task metadata.
2. **Pose / scene**: OpenCV-based hand and marker tracking, an anatomical hand model with analytic 2-bone IK, a 5-finger robot hand model, and a parametric CAD builder for tools/objects.
3. **Physics sanitizer**: analytic SDFs (box, cylinder, sphere) enforcing non-penetration (`SDF ≥ 0`).
4. **Inverse dynamics**: mass from material density, inertia tensors, accelerations from the trajectory, Coulomb friction normal forces.
5. **Feasibility**: IK audit against ALOHA ViperX, Unitree H1 and Franka Panda, with an automatic rejection threshold (≥ 85% reachable frames); forward simulation in MuJoCo to check stability.
6. **Export**: Hugging Face LeRobot-style Parquet with schema validation.

There are 4 demo twins (Master Lock No. 3, a drop-key lock, a watch gear extraction, a Singer faceplate removal), 255–390 frames each, and 31 tests. MIT licence.

### What is *not* automatic yet (important)

- **The 3D demo scenes are parametric CAD assemblies built by hand, guided by watching the videos.** They live as fixtures in `tests/pose_extraction/reconstruct_*_scene.py`. The generic `src/` modules (SDF, IK audit, inverse dynamics, MuJoCo check, export) run on them, but fully automatic video → 3D reconstruction is not there yet. The README visuals should be read in that light.
- **The throughput/cost figures are a prospective model, not a measurement.** The README's 7-stage attrition funnel (≈0.62% net yield) and the "$0.15–0.30 per demo" estimate come from assumptions in `docs/THROUGHPUT_AND_UNIT_ECONOMICS.md`. No batch job has been run at scale.
- Contact forces are *inferred* via inverse dynamics, not measured, so they're only as good as the mass/friction priors.
- **Video rights**: the source videos belong to their creators. The repo extracts trajectories and geometry, but I haven't settled how (or whether) derived datasets can be published. Advice is welcome.

### Feedback I'm looking for

- Is "workpiece trajectory first" a sound abstraction for transfer, or does it throw away too much (grasp type, approach direction)?
- Which automatic plausibility filters would you trust most? Is SDF + IK + a short MuJoCo rollout enough to reject garbage?
- Best current options for monocular hand–object reconstruction to replace the hand-built CAD step (I'm looking at HaMeR / FoundationPose-style approaches)?
- Has anyone validated LeRobot exports from synthetic/derived data in actual policy training? What went wrong?

Repo: https://github.com/silveremartin-dev/Praxis

Thanks for reading. Critical feedback is the most useful kind here.

— Silvère Martin-Michiellot
