import subprocess
import sys
import pytest

def test_pipeline_help():
    cmd = [sys.executable, "-m", "src.pipeline", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Video-to-Physical-Digital-Twin" in res.stdout or "usage" in res.stdout.lower()

def test_pipeline_dry_run(tmp_path):
    empty_input = str(tmp_path / "empty_input")
    empty_output = str(tmp_path / "empty_output")
    
    cmd = [
        sys.executable, "-m", "src.pipeline",
        "--input_dir", empty_input,
        "--output_dir", empty_output
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "0" in res.stdout or "Aucune" in res.stdout
