import os
import pytest
import numpy as np
import cv2

from src.ingestion.inspector import inspect_video, extract_clip_frames

@pytest.fixture
def synthetic_video(tmp_path):
    video_path = str(tmp_path / "test_synth.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    width, height = 320, 240
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # 60 frames = 2 seconds
    for i in range(60):
        frame = np.full((height, width, 3), (i * 4) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path

def test_inspect_video_properties(synthetic_video):
    meta = inspect_video(synthetic_video)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 30.0
    assert meta["total_frames"] == 60
    assert abs(meta["duration_seconds"] - 2.0) < 0.1
    assert meta["duration_formatted"] == "00:02"

def test_inspect_video_not_found():
    with pytest.raises(FileNotFoundError):
        inspect_video("non_existent_video_path.mp4")

def test_extract_clip_frames(synthetic_video, tmp_path):
    frames_dir = str(tmp_path / "extracted_frames")
    extracted = extract_clip_frames(synthetic_video, start_sec=0.5, end_sec=1.5, output_dir=frames_dir, sample_fps=5)
    assert len(extracted) > 0
    for p in extracted:
        assert os.path.exists(p)
        assert p.endswith(".jpg")
