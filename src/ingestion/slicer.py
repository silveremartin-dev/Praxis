"""
Action Sub-Video Slicer & Semantic Metadata Generator.
Extracts continuous hand-tool manipulation sub-videos (.mp4) from raw YouTube streams
and produces paired task-graph action metadata (.json).
"""

import argparse
import json
import os
import sys
import cv2

def slice_action_video(
    raw_video_path: str,
    t_start: float,
    t_end: float,
    output_clip_path: str,
    action_info: dict,
    metadata_output_path: str = None
):
    """
    Extracts an MP4 sub-video between t_start and t_end and saves action metadata.
    """
    if not os.path.exists(raw_video_path):
        raise FileNotFoundError(f"Source video not found: {raw_video_path}")
        
    os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
    
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {raw_video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)
    total_clip_frames = end_frame - start_frame
    
    # Write initial frames via OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_clip_path = output_clip_path + ".raw.mp4"
    out = cv2.VideoWriter(temp_clip_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    written_count = 0
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written_count += 1
        current_frame += 1
        
    cap.release()
    out.release()

    # Transcode to web-standard H.264 (avc1) for instant browser playback
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin, "-y", "-i", temp_clip_path,
            "-c:v", "libx264", "-g", "1", "-tune", "zerolatency", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_clip_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_clip_path):
            if os.path.exists(temp_clip_path):
                os.remove(temp_clip_path)
        else:
            # Fallback to raw if ffmpeg failed
            if os.path.exists(temp_clip_path):
                os.replace(temp_clip_path, output_clip_path)
    except Exception as e:
        print(f"[!] Warning: H.264 transcoding skipped ({e}), using OpenCV output.")
        if os.path.exists(temp_clip_path):
            os.replace(temp_clip_path, output_clip_path)
    
    actual_duration = written_count / fps if fps > 0 else 0
    print(f"[+] Sub-video successfully sliced: {output_clip_path}")
    print(f"    Frames: {written_count} | Duration: {actual_duration:.2f}s | Resolution: {width}x{height} @ {fps:.2f}fps")
    
    # Validate goal-directed action and state irreversibility
    from src.ingestion.shot_detector import GoalDirectedActionValidator
    verb = action_info.get("verb", "manipulate")
    pre_cond = action_info.get("pre_condition", "initial_unmanipulated_state")
    post_cond = action_info.get("post_condition", "target_manipulated_state")
    goal_audit = GoalDirectedActionValidator.validate_action(verb, pre_cond, post_cond)

    # Build complete semantic metadata ("what is the person doing?")
    metadata = {
        "clip_file": os.path.basename(output_clip_path),
        "source_raw_video": os.path.basename(raw_video_path),
        "timing": {
            "t_start_seconds": round(t_start, 2),
            "t_end_seconds": round(t_end, 2),
            "duration_seconds": round(actual_duration, 2),
            "fps": round(fps, 2),
            "total_frames": written_count
        },
        "action_annotation": {
            "what_is_the_person_doing": action_info.get("description", "Demonstrating physical tool manipulation"),
            "action_verb": verb,
            "target_tool": action_info.get("tool", "unspecified_tool"),
            "target_workpiece": action_info.get("workpiece", "unspecified_object"),
            "hand_engagement": action_info.get("hands", "bi-manual"),
            "camera_context": action_info.get("camera", "static_macro")
        },
        "task_teleology": {
            "is_goal_directed": goal_audit.is_goal_directed,
            "is_state_irreversible": goal_audit.is_state_irreversible,
            "pre_condition": pre_cond,
            "post_condition": post_cond,
            "action_verb": verb,
        },
        "physical_affordances": action_info.get("affordances", {}),
        "next_pipeline_stages": [
            "src.pose_extraction.hand_mesh",
            "src.physics_sanitizer.sdf_optimizer",
            "src.feasibility.urdf_auditor"
        ]
    }
    
    if metadata_output_path is None:
        metadata_output_path = os.path.splitext(output_clip_path)[0] + ".json"
        
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Semantic metadata saved: {metadata_output_path}")
    return output_clip_path, metadata_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Sub-Video Slicer")
    parser.add_argument("--video", type=str, required=True, help="Path to raw source video")
    parser.add_argument("--t_start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--t_end", type=float, required=True, help="End time in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output clip MP4 path")
    parser.add_argument("--desc", type=str, default="Hand manipulation sequence", help="What the person is doing")
    parser.add_argument("--tool", type=str, default="tool", help="Primary tool name")
    parser.add_argument("--workpiece", type=str, default="object", help="Primary workpiece name")
    
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"data/clips/{base_name}_clip_{int(args.t_start)}s_{int(args.t_end)}s.mp4"
        
    action_dict = {
        "description": args.desc,
        "tool": args.tool,
        "workpiece": args.workpiece
    }
    
    slice_action_video(args.video, args.t_start, args.t_end, args.output, action_dict)
