import json
import glob
import os
import pytest

def test_lerobot_processed_metadata_schema():
    meta_path = "data/processed/01_lpl_dropkey_metadata.json"
    if not os.path.exists(meta_path):
        pytest.skip("Processed metadata not generated yet")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    required_keys = ["dataset_name", "video_source", "extracted_action_segment", "rigid_objects_3d", "hand_contacts"]
    for key in required_keys:
        assert key in data, f"Missing required top-level key: {key}"
        
    assert "t_start_seconds" in data["extracted_action_segment"]
    assert "t_end_seconds" in data["extracted_action_segment"]
    assert "action_verb" in data["extracted_action_segment"]
    
    # Verify rigid objects have metric dimensions
    for obj in data["rigid_objects_3d"]:
        assert "id" in obj
        assert "dimensions_mm" in obj
        assert "mass_estimate_grams" in obj

def test_all_action_clips_metadata_files():
    json_files = glob.glob("data/clips/*.json")
    assert len(json_files) >= 4, "Expected at least 4 sliced action metadata files"
    
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        assert "clip_file" in meta
        assert "timing" in meta
        assert "action_annotation" in meta
        assert "what_is_the_person_doing" in meta["action_annotation"]
        assert "action_verb" in meta["action_annotation"]
        assert meta["timing"]["duration_seconds"] > 0
        assert meta["timing"]["total_frames"] > 0
