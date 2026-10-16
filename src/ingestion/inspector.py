"""
Video Inspection and Keyframe Extraction Module for Praxis (`src/ingestion/inspector.py`).
Analyzes physical stream properties (FPS, resolution, duration) and samples frames
for hand and object detection.
"""

import json
import os
import sys


def inspect_video(video_path: str) -> dict:
    """Returns physical stream metadata (resolution, FPS, frame count, duration) for a video file."""
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video stream: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()

    meta = {
        "file_name": os.path.basename(video_path),
        "file_path": video_path,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": frame_count,
        "duration_seconds": round(duration_sec, 2),
        "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
    }
    return meta


def extract_clip_frames(video_path: str, start_sec: float, end_sec: float, output_dir: str, sample_fps: int = 5) -> list:
    """Extracts sampled JPEG frames from a sub-video time interval."""
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    step = max(1, int(fps / sample_fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    extracted_paths = []

    idx = 0
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (current_frame - start_frame) % step == 0:
            frame_name = f"frame_{idx:04d}_t{current_frame / fps:.2f}s.jpg"
            out_file = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_file, frame)
            extracted_paths.append(out_file)
            idx += 1
        current_frame += 1

    cap.release()
    return extracted_paths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
        data = inspect_video(v_path)
        print(json.dumps(data, indent=2))
