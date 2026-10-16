"""
Scans sample frames across videos 03, 04, and 05 to pinpoint optimal manipulation segments.
"""

import os
import cv2

VIDEOS_TO_SCAN = [
    {
        "id": "03_DIY",
        "path": "data/raw/03_DIY_CementWD40_mA3QtWz7RMQ.mp4",
        "sample_times": [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0],
        "out_dir": "data/clips/scan_03_diy"
    },
    {
        "id": "04_Cuisine",
        "path": "data/raw/04_Cuisine_Installation_KOBqgv-TpyA.mp4",
        "sample_times": [120.0, 300.0, 600.0, 900.0, 1200.0, 1500.0],
        "out_dir": "data/clips/scan_04_cuisine"
    },
    {
        "id": "05_Watch",
        "path": "data/raw/05_Watch_Zodiac_f8NFSwb6VQ8.mp4",
        "sample_times": [180.0, 360.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0],
        "out_dir": "data/clips/scan_05_watch"
    }
]

def scan_video(entry):
    os.makedirs(entry["out_dir"], exist_ok=True)
    cap = cv2.VideoCapture(entry["path"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[*] Scanning {entry['id']}...")
    for t in entry["sample_times"]:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_file = os.path.join(entry["out_dir"], f"sample_t{int(t)}s.jpg")
            cv2.imwrite(out_file, frame)
            print(f"    Saved sample at t={t}s -> {out_file}")
    cap.release()

if __name__ == "__main__":
    for item in VIDEOS_TO_SCAN:
        scan_video(item)
    print("\nScan completed.")
