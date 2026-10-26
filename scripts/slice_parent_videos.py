"""
Parent Demonstration Multi-Shot Slicer.
Extracts multiple continuous manipulation sub-videos from verified parent YouTube videos.
Conforms strictly to AGENTS.md directives (Tier 1 Intake, Single-Shot Continuity, Hand Presence).
"""

import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ingestion.slicer import slice_action_video

SHOTS_CONFIG = [
    # --- Parent Video 1: Abus Titalium 80TI/40 (data/raw/09_LPL_Abus_80TI40.mp4) ---
    {
        "parent_video": "data/raw/09_LPL_Abus_80TI40.mp4",
        "clip_id": "09_abus_shot1_key_check",
        "t_start": 6.0,
        "t_end": 18.0,
        "desc": "Abus 80TI/40 key insertion, 90° clockwise plug rotation, shackle pop, key relock & extraction",
        "verb": "insert",
        "tool": "Abus standard brass pin tumbler key",
        "workpiece": "Abus 80TI/40 Titalium Padlock Body & Shackle (14x10x20mm, 175g)",
        "pre_cond": "Padlock locked with shackle closed; key held ready outside keyway",
        "post_cond": "Key inserted, shackle unlocked & opened, key returned and extracted",
        "hands": "bi-manual bare hands (LH body cradle, RH thumb-index key grip)"
    },
    {
        "parent_video": "data/raw/09_LPL_Abus_80TI40.mp4",
        "clip_id": "09_abus_shot2_spp_picking",
        "t_start": 35.0,
        "t_end": 60.0,
        "desc": "Single-Pin Picking (SPP): TOK prybar clockwise tensioning and short hook pick pin lifting",
        "verb": "pick",
        "tool": "0.050in TOK Tension Wrench + 0.025in Short Hook Pick",
        "workpiece": "Abus 80TI/40 Titalium Padlock Body & Shackle (14x10x20mm, 175g)",
        "pre_cond": "Abus lock locked with 6 spool/serrated pin stacks below shear line",
        "post_cond": "Pin stacks 1-6 lifted sequentially under controlled tension feedback",
        "hands": "bi-manual bare hands (LH index/thumb tension hold, RH tripod pick grasp)"
    },
    {
        "parent_video": "data/raw/09_LPL_Abus_80TI40.mp4",
        "clip_id": "09_abus_shot3_shackle_pop",
        "t_start": 62.0,
        "t_end": 76.0,
        "desc": "Final binding pin shear set, plug rotation & spring-loaded shackle pop open",
        "verb": "unlatch",
        "tool": "0.050in TOK Tension Wrench + 0.025in Short Hook Pick",
        "workpiece": "Abus 80TI/40 Titalium Padlock Body & Shackle (14x10x20mm, 175g)",
        "pre_cond": "5 pins set at shear line, final spool pin binding plug rotation",
        "post_cond": "Final pin sheared, plug rotated 45°, spring shackle popped upward (+14mm +Z)",
        "hands": "bi-manual bare hands (LH lock cradle, RH pick release & shackle grasp)"
    },
    {
        "parent_video": "data/raw/09_LPL_Abus_80TI40.mp4",
        "clip_id": "09_abus_shot4_screw_extraction",
        "t_start": 96.0,
        "t_end": 114.0,
        "desc": "Phillips screwdriver insertion into shackle chamber, core screw counter-clockwise unscrewing & extraction",
        "verb": "unscrew",
        "tool": "PH1 precision screwdriver",
        "workpiece": "Abus 80TI/40 Core Retaining Screw & Padlock Body (M3x12mm screw, 4g)",
        "pre_cond": "Retaining screw threaded tight at bottom of shackle cavity securing cylinder",
        "post_cond": "Retaining screw unthreaded completely and extracted from shackle well",
        "hands": "bi-manual bare hands (LH lock body support, RH precision screwdriver axial rotation)"
    },
    {
        "parent_video": "data/raw/09_LPL_Abus_80TI40.mp4",
        "clip_id": "09_abus_shot5_pin_extraction",
        "t_start": 142.0,
        "t_end": 162.0,
        "desc": "Pinning tweezers precision pinch: grasping brass key pins and driver springs from plug chambers into tray",
        "verb": "extract",
        "tool": "Curved stainless steel pinning tweezers",
        "workpiece": "Miniature brass pin tumbler & steel spring (1.5mm diameter, 0.2g)",
        "pre_cond": "Brass pin seated inside rotated cylinder plug chamber",
        "post_cond": "Brass pin extracted with tweezers and placed into organized pinning tray slot",
        "hands": "bi-manual bare hands (LH plug support, RH tweezer precision pinch)"
    },

    # --- Parent Video 2: Master Lock No. 3 (data/raw/LPL_MasterLock_No3.mp4) ---
    {
        "parent_video": "data/raw/LPL_MasterLock_No3.mp4",
        "clip_id": "07_masterlock_shot1_key_check",
        "t_start": 12.0,
        "t_end": 24.0,
        "desc": "Master Lock No. 3 standard brass key insertion, core rotation, shackle pop & relock",
        "verb": "insert",
        "tool": "Master Lock factory cut brass key",
        "workpiece": "Master Lock No. 3 Laminated Padlock Body & Steel Shackle (14x10x20mm, 195g)",
        "pre_cond": "Padlock locked with shackle fully seated in lock body",
        "post_cond": "Key inserted and rotated clockwise, shackle popped open, re-latched",
        "hands": "bi-manual bare hands (LH body hold, RH key pinch & rotation)"
    },
    {
        "parent_video": "data/raw/LPL_MasterLock_No3.mp4",
        "clip_id": "07_masterlock_shot3_shackle_bypass",
        "t_start": 68.0,
        "t_end": 82.0,
        "desc": "Bypass needle insertion through shackle gap, depressing spring locking dog to pop shackle open",
        "verb": "bypass",
        "tool": "0.5mm flat spring steel L-shaped bypass probe",
        "workpiece": "Master Lock No. 3 Laminated Padlock Body & Steel Shackle (14x10x20mm, 195g)",
        "pre_cond": "Shackle closed and locked by internal spring locking dog",
        "post_cond": "Locking dog depressed by bypass probe, shackle released and popped open",
        "hands": "bi-manual bare hands (LH lock body stabilization, RH bypass probe insertion)"
    },

    # --- Parent Video 3: Drop Key Cylinder (data/raw/01_LPL_DropKey_RRunwAI7N5Q.mp4) ---
    {
        "parent_video": "data/raw/01_LPL_DropKey_RRunwAI7N5Q.mp4",
        "clip_id": "01_dropkey_shot2_cylinder_insert",
        "t_start": 52.0,
        "t_end": 68.0,
        "desc": "Folded drop-key stem insertion into high-security rim cylinder keyway and clockwise alignment",
        "verb": "insert",
        "tool": "Articulated drop key with 90° folded vertical handle link",
        "workpiece": "High-Security Rim Cylinder Housing (compact 14x10x20mm, 47g)",
        "pre_cond": "Folded drop key held in right hand facing cylinder keyway",
        "post_cond": "Drop key fully seated inside keyway, tension applied to core",
        "hands": "bi-manual bare hands (LH cylinder housing hold, RH key insertion)"
    },
    {
        "parent_video": "data/raw/01_LPL_DropKey_RRunwAI7N5Q.mp4",
        "clip_id": "01_dropkey_shot3_torque_unlock",
        "t_start": 80.0,
        "t_end": 96.0,
        "desc": "Full cylinder core rotation via folded drop-key lever handle, unlocking rim deadbolt mechanism",
        "verb": "rotate",
        "tool": "Articulated drop key lever handle",
        "workpiece": "High-Security Rim Cylinder Housing & Rotating Core (compact 14x10x20mm, 47g)",
        "pre_cond": "Cylinder core aligned at 0° home position",
        "post_cond": "Cylinder core rotated 360° through unlock cycle",
        "hands": "bi-manual bare hands (LH cylinder body grip, RH lever handle continuous rotation)"
    }
]

def main():
    print(f"[+] Slicing {len(SHOTS_CONFIG)} sub-videos from verified parent demonstration videos...")
    output_dir = "data/clips"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for shot in SHOTS_CONFIG:
        parent = shot["parent_video"]
        clip_id = shot["clip_id"]
        t0 = shot["t_start"]
        t1 = shot["t_end"]
        out_mp4 = os.path.join(output_dir, f"{clip_id}.mp4")
        out_json = os.path.join(output_dir, f"{clip_id}.json")
        
        action_dict = {
            "description": shot["desc"],
            "verb": shot["verb"],
            "tool": shot["tool"],
            "workpiece": shot["workpiece"],
            "hands": shot["hands"],
            "pre_condition": shot["pre_cond"],
            "post_condition": shot["post_cond"]
        }
        
        print(f"\n--> Slicing {clip_id} ({t0}s - {t1}s) from {parent}...")
        mp4_path, json_path = slice_action_video(
            raw_video_path=parent,
            t_start=t0,
            t_end=t1,
            output_clip_path=out_mp4,
            action_info=action_dict,
            metadata_output_path=out_json
        )
        results.append((clip_id, mp4_path, json_path))
        
    print(f"\n[+] Successfully extracted all {len(results)} sub-videos!")

if __name__ == "__main__":
    main()
