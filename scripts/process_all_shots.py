"""
Automated Batch Processor for Multi-Shot Sub-Videos.
Runs AutomatedVideoPosePipeline on all certified sub-video shots from parent demonstrations.
Outputs verified 3D trajectory JSONs to data/processed/.
"""

import os
import sys
import json
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pose_extraction.visual_tracker import AutomatedVideoPosePipeline

CERTIFIED_SHOTS = [
    ("09_abus_shot1_key_check", "data/clips/09_abus_shot1_key_check.mp4"),
    ("09_abus_shot2_spp_picking", "data/clips/09_abus_shot2_spp_picking.mp4"),
    ("09_abus_shot3_shackle_pop", "data/clips/09_abus_shot3_shackle_pop.mp4"),
    ("09_abus_shot5_pin_extraction", "data/clips/09_abus_shot5_pin_extraction.mp4"),
    ("07_masterlock_shot1_key_check", "data/clips/07_masterlock_shot1_key_check.mp4"),
    ("07_masterlock_shot3_shackle_bypass", "data/clips/07_masterlock_shot3_shackle_bypass.mp4"),
    ("01_dropkey_shot2_cylinder_insert", "data/clips/01_dropkey_shot2_cylinder_insert.mp4"),
]

def main():
    pipeline = AutomatedVideoPosePipeline()
    out_dir = os.path.join(REPO_ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[+] Processing {len(CERTIFIED_SHOTS)} certified sub-video demonstrations with AutomatedVideoPosePipeline...")
    
    for clip_id, mp4_path in CERTIFIED_SHOTS:
        full_mp4 = os.path.join(REPO_ROOT, mp4_path)
        if not os.path.isfile(full_mp4):
            print(f"[-] Clip file missing: {full_mp4}, skipping.")
            continue
            
        t0 = time.time()
        print(f"\n--> Running automated 3D extraction on {clip_id}...")
        traj = pipeline.process_video_clip(full_mp4, clip_id)
        
        out_json = os.path.join(out_dir, f"{clip_id}_trajectory.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2)
            
        elapsed = time.time() - t0
        n_frames = len(traj.get("frames", []))
        score = traj.get("quality_score", 0.0)
        grade = traj.get("grade", "N/A")
        print(f"[+] Saved {out_json} | {n_frames} frames | Score: {score}/100 ({grade}) in {elapsed:.1f}s")
        
    print("\n[+] All certified demonstration trajectories successfully extracted and cached!")

if __name__ == "__main__":
    main()
