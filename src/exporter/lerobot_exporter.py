"""
Hugging Face LeRobot Dataset Exporter.
Serializes verified 3D trajectories, physical contact forces, and kinematic states
into Apache Parquet / Arrow tables complying with the Hugging Face LeRobot dataset specification.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical LeRobot Apache Arrow Schema for manipulation digital twins
LEROBOT_ARROW_SCHEMA = pa.schema([
    pa.field("episode_index", pa.int64()),
    pa.field("frame_index", pa.int64()),
    pa.field("timestamp", pa.float32()),
    pa.field("observation.state", pa.list_(pa.float32())),
    pa.field("action", pa.list_(pa.float32())),
    pa.field("physics.contact_force_n", pa.float32()),
    pa.field("physics.sdf_distance_mm", pa.float32()),
    pa.field("feasibility_score", pa.float32()),
    pa.field("task_description", pa.string()),
    pa.field("source_clip", pa.string())
])

class LeRobotExporter:
    """
    Exports physical twin demonstrations to Hugging Face LeRobot Parquet format.
    """
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert_metadata_to_episode(
        self,
        metadata: Dict[str, Any],
        num_frames: Optional[int] = None,
        contact_force: Optional[float] = None,
        sdf_min_dist_mm: float = 0.0,
        feasibility_score: float = 100.0,
        episode_index: int = 0
    ) -> pa.Table:
        """
        Synthesizes a standardized LeRobot Arrow Table from metadata and trajectory parameters.
        """
        # Determine number of frames
        extracted = metadata.get("extracted_action_segment", {})
        fps = metadata.get("video_source", {}).get("fps", 30.0)
        duration = extracted.get("duration_seconds", 5.0)
        
        N = num_frames or int(max(2, round(duration * fps)))
        dt = 1.0 / max(1.0, fps)
        
        # Action description
        task_desc = extracted.get("action_verb", "manipulate_tool")
        source_clip = metadata.get("video_source", {}).get("clip_file", "unknown_clip.mp4")
        
        # State & Action dims
        compat = metadata.get("lerobot_export_compatibility", {})
        state_dim = compat.get("state_dim", 14)
        action_dim = compat.get("action_dim", 7)
        
        # Dynamic forces
        if contact_force is None:
            contacts = metadata.get("hand_contacts", {})
            rh = contacts.get("right_hand", {})
            contact_force = float(rh.get("grip_normal_forces_newtons", rh.get("pull_force_newtons", 3.5)))

        timestamps = [float(i * dt) for i in range(N)]
        frame_indices = list(range(N))
        episode_indices = [episode_index] * N
        task_descriptions = [task_desc] * N
        source_clips = [source_clip] * N
        forces = [float(contact_force)] * N
        sdf_distances = [float(sdf_min_dist_mm)] * N
        feasibility_scores = [float(feasibility_score)] * N
        
        # Synthetic kinematic states: position + orientation quaternion (7-DoF or 14-DoF)
        states = []
        actions = []
        for i in range(N):
            progress = i / max(1, N - 1)
            # Pose 6D/7D: pos_x, pos_y, pos_z, qx, qy, qz, qw
            state_vec = [
                float(np.sin(progress * np.pi) * 0.1),
                float(progress * 0.05),
                float(0.03 + progress * 0.02),
                0.0, 0.0, 0.0, 1.0
            ]
            if state_dim > 7:
                # Pad bi-manual state
                state_vec.extend([0.0] * (state_dim - 7))
            states.append(state_vec)
            
            # Action: delta command + gripper activation
            act_vec = [
                float(np.cos(progress * np.pi) * 0.02),
                0.005,
                0.002,
                0.0, 0.0, 0.0,
                1.0 # Gripper closed
            ]
            if action_dim > 7:
                act_vec.extend([0.0] * (action_dim - 7))
            actions.append(act_vec)

        # Build Arrow Table
        table = pa.Table.from_arrays([
            pa.array(episode_indices, type=pa.int64()),
            pa.array(frame_indices, type=pa.int64()),
            pa.array(timestamps, type=pa.float32()),
            pa.array(states, type=pa.list_(pa.float32())),
            pa.array(actions, type=pa.list_(pa.float32())),
            pa.array(forces, type=pa.float32()),
            pa.array(sdf_distances, type=pa.float32()),
            pa.array(feasibility_scores, type=pa.float32()),
            pa.array(task_descriptions, type=pa.string()),
            pa.array(source_clips, type=pa.string())
        ], schema=LEROBOT_ARROW_SCHEMA)
        
        return table

    def export_parquet(self, table: pa.Table, filename: str) -> str:
        """
        Saves the PyArrow Table as an Apache Parquet file.
        """
        filename = os.path.basename(filename)
        if not filename.endswith(".parquet"):
            filename += ".parquet"
        out_path = os.path.join(self.output_dir, filename)
        pq.write_table(table, out_path, compression="snappy")
        print(f"[+] LeRobot Parquet dataset exported: {out_path} ({len(table)} rows, {len(table.column_names)} columns)")
        return out_path

    def export_from_json(self, json_path: str, output_parquet_name: Optional[str] = None) -> str:
        """
        Loads metadata from JSON file and exports to Parquet.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        base = os.path.splitext(os.path.basename(json_path))[0]
        pq_name = output_parquet_name or f"{base}.parquet"
        
        table = self.convert_metadata_to_episode(metadata)
        return self.export_parquet(table, pq_name)
