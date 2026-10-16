import json
import os
import pytest
import numpy as np
import cv2

from src.ingestion.slicer import slice_action_video

@pytest.fixture
def source_video(tmp_path):
    video_path = str(tmp_path / "source_3s.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    width, height = 320, 240
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # 90 frames = 3 seconds
    for i in range(90):
        frame = np.full((height, width, 3), (i * 2) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path

def test_slice_action_video_creation(source_video, tmp_path):
    output_mp4 = str(tmp_path / "action_clip.mp4")
    action_info = {
        "description": "Synthetic screw insertion test",
        "verb": "screw",
        "tool": "precision_screwdriver",
        "workpiece": "fastener"
    }
    
    clip_p, meta_p = slice_action_video(
        raw_video_path=source_video,
        t_start=1.0,
        t_end=2.5,
        output_clip_path=output_mp4,
        action_info=action_info
    )
    
    assert os.path.exists(clip_p)
    assert os.path.exists(meta_p)
    
    # Check sliced video length
    cap = cv2.VideoCapture(clip_p)
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    duration = frame_count / fps
    assert 1.4 <= duration <= 1.6
    
    # Check metadata contents
    with open(meta_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["timing"]["t_start_seconds"] == 1.0
    assert data["timing"]["t_end_seconds"] == 2.5
    assert data["action_annotation"]["action_verb"] == "screw"
    assert data["action_annotation"]["target_tool"] == "precision_screwdriver"

def test_slice_action_video_missing_source():
    with pytest.raises(FileNotFoundError):
        slice_action_video("missing.mp4", 0.0, 1.0, "out.mp4", {})
