"""
Unit tests for the Hugging Face LeRobot Parquet Exporter.
"""

import json
import os
import tempfile
import pyarrow.parquet as pq
import pytest

from src.exporter.lerobot_exporter import LeRobotExporter, LEROBOT_ARROW_SCHEMA

def test_lerobot_schema_fields():
    field_names = [f.name for f in LEROBOT_ARROW_SCHEMA]
    assert "episode_index" in field_names
    assert "frame_index" in field_names
    assert "timestamp" in field_names
    assert "observation.state" in field_names
    assert "action" in field_names
    assert "physics.contact_force_n" in field_names
    assert "physics.sdf_distance_mm" in field_names
    assert "feasibility_score" in field_names
    assert "task_description" in field_names

def test_convert_metadata_to_episode_table():
    exporter = LeRobotExporter()
    mock_meta = {
        "extracted_action_segment": {
            "duration_seconds": 2.0,
            "action_verb": "turn_screw"
        },
        "video_source": {
            "fps": 30.0,
            "clip_file": "test_clip.mp4"
        },
        "lerobot_export_compatibility": {
            "state_dim": 14,
            "action_dim": 7
        },
        "hand_contacts": {
            "right_hand": {
                "grip_normal_forces_newtons": 4.2
            }
        }
    }
    
    table = exporter.convert_metadata_to_episode(mock_meta, num_frames=60, feasibility_score=95.0)
    assert len(table) == 60
    assert table.column("episode_index")[0].as_py() == 0
    assert table.column("task_description")[0].as_py() == "turn_screw"
    assert pytest.approx(table.column("physics.contact_force_n")[0].as_py(), rel=1e-2) == 4.2
    assert table.column("feasibility_score")[0].as_py() == 95.0

def test_export_parquet_and_readback():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = LeRobotExporter(output_dir=tmpdir)
        mock_meta = {
            "extracted_action_segment": {"duration_seconds": 1.0, "action_verb": "pinch"},
            "video_source": {"fps": 10.0, "clip_file": "test.mp4"},
            "lerobot_export_compatibility": {"state_dim": 7, "action_dim": 7}
        }
        
        table = exporter.convert_metadata_to_episode(mock_meta, num_frames=10)
        pq_path = exporter.export_parquet(table, "test_episode.parquet")
        
        assert os.path.exists(pq_path)
        
        # Read back with PyArrow
        read_table = pq.read_table(pq_path)
        assert len(read_table) == 10
        assert read_table.schema.names == table.schema.names
