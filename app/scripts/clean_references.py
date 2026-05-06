import os
import json
import cv2
import torch
import numpy as np
import imageio
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import REFERENCES_DB_PATH, DEVICE, SAM2_CHECKPOINT, SAM2_CONFIG, TEMP_DIR
from app.services.ai_models import model_manager
from app.services.video_engine import extract_frames

def clean_all_references():
    if not os.path.exists(REFERENCES_DB_PATH):
        print("Reference JSON not found.")
        return

    with open(REFERENCES_DB_PATH, "r") as f:
        refs = json.load(f)

    for shot_name, info in refs.items():
        # Use the original video path if it exists, otherwise use current
        video_rel_path = info.get("video_path_original", info["video_path"])
        video_abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, video_rel_path))
        
        if not os.path.exists(video_abs_path):
            print(f"Skipping {shot_name}: File not found at {video_abs_path}")
            continue

        print(f"\n--- Cleaning Reference: {shot_name} ---")
        
        # 1. Extract Pose to find Strike Frame (Peak action)
        pose_data = model_manager.extractor.extract_from_video(video_abs_path)
        if not pose_data.get("frames"):
            print(f"Could not extract pose for {shot_name}. Skipping.")
            continue

        # Find Strike (lowest wrist position)
        wrist_y = []
        for f in pose_data["frames"]:
            lms = f.get("landmarks", {})
            ly = lms.get("left_wrist", {}).get("y", 0.5)
            ry = lms.get("right_wrist", {}).get("y", 0.5)
            wrist_y.append((ly + ry) / 2)
        
        strike_frame = int(np.argmax(wrist_y))
        print(f"Detected Strike at frame {strike_frame}")

        # Find first valid hip for SAM2
        click_coords = None
        start_tracking_frame = 0
        for frame in pose_data["frames"]:
            lms = frame.get("landmarks", {})
            if "left_hip" in lms and "right_hip" in lms:
                try:
                    l_hip, r_hip = lms["left_hip"], lms["right_hip"]
                    cap = cv2.VideoCapture(video_abs_path)
                    w, h = int(cap.get(3)), int(cap.get(4))
                    cap.release()
                    click_coords = [int((l_hip['x'] + r_hip['x']) / 2 * w), int((l_hip['y'] + r_hip['y']) / 2 * h)]
                    start_tracking_frame = frame['frame_idx']
                    break
                except: continue

        if click_coords is None:
            click_coords = [w//2, h//2]
            start_tracking_frame = 0

        # 2. Run SAM2 Propagation with Strict Cut-off
        extract_frames(video_abs_path)
        inference_state = model_manager.predictor.init_state(video_path=TEMP_DIR)
        model_manager.predictor.reset_state(inference_state)
        
        points = np.array([click_coords], dtype=np.float32)
        labels = np.array([1], np.int32)
        model_manager.predictor.add_new_points(inference_state, frame_idx=start_tracking_frame, obj_id=1, points=points, labels=labels)
        
        tracked_indices = []
        consecutive_misses = 0
        threshold = 500 # Strict person-sized mask
        
        # We cut at either:
        # a) Lost tracking for 5 frames
        # b) strike_frame + 45 frames (Typical follow-through end)
        MAX_FOLLOW_THROUGH = 45
        
        for out_frame_idx, out_obj_ids, out_mask_logits in model_manager.predictor.propagate_in_video(inference_state, start_frame_idx=start_tracking_frame):
            detected = False
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_obj_id == 1:
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    if mask.any() and mask.sum() > threshold:
                        tracked_indices.append(out_frame_idx)
                        detected = True
                        consecutive_misses = 0
            
            if not detected: consecutive_misses += 1
            
            # PHASE-BASED CUT: If we are far past the strike, stop.
            if out_frame_idx > strike_frame + MAX_FOLLOW_THROUGH:
                print(f"Reached end of follow-through (Frame {out_frame_idx}). Cutting.")
                break

            if tracked_indices and consecutive_misses > 5:
                print(f"Lost batsman at frame {out_frame_idx}. Cutting.")
                break

        if not tracked_indices:
            print(f"Failed to track batsman in {shot_name}. Skipping.")
            continue

        start_idx = min(tracked_indices)
        end_idx = max(tracked_indices)
        
        # Ensure we at least include the strike
        if end_idx < strike_frame: end_idx = min(strike_frame + 10, len(pose_data["frames"])-1)
        
        print(f"Detected range: {start_idx} to {end_idx}")

        # 3. Trim Video and JSON
        cleaned_video_rel = video_rel_path.replace(".mp4", "_cleaned.mp4")
        if "_cleaned_cleaned" in cleaned_video_rel: 
            cleaned_video_rel = cleaned_video_rel.replace("_cleaned_cleaned", "_cleaned")
        
        cleaned_abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, cleaned_video_rel))
        cap = cv2.VideoCapture(video_abs_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = imageio.get_writer(cleaned_abs_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p', macro_block_size=None)
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if start_idx <= idx <= end_idx:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        writer.close()

        # Trim JSON
        json_rel_path = info.get("angles_path_original", info["angles_path"])
        json_abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, json_rel_path))
        if os.path.exists(json_abs_path):
            with open(json_abs_path, "r") as f:
                data = json.load(f)
            
            trimmed_frames = []
            for f_data in data.get("frames", []):
                old_idx = f_data["frame_idx"]
                if start_idx <= old_idx <= end_idx:
                    f_data["frame_idx"] = old_idx - start_idx
                    trimmed_frames.append(f_data)
            
            data["frames"] = trimmed_frames
            cleaned_json_rel = json_rel_path.replace(".json", "_cleaned.json")
            if "_cleaned_cleaned" in cleaned_json_rel:
                cleaned_json_rel = cleaned_json_rel.replace("_cleaned_cleaned", "_cleaned")
                
            cleaned_json_abs = os.path.normpath(os.path.join(PROJECT_ROOT, cleaned_json_rel))
            with open(cleaned_json_abs, "w") as f:
                json.dump(data, f, indent=4)
            
            info["angles_path_original"] = json_rel_path
            info["angles_path"] = cleaned_json_rel

        # 4. Update JSON
        info["video_path_original"] = video_rel_path
        info["video_path"] = cleaned_video_rel
        print(f"Successfully cleaned: {shot_name} (Frames: {end_idx - start_idx + 1})")

    # Save updated JSON
    with open(REFERENCES_DB_PATH, "w") as f:
        json.dump(refs, f, indent=4)
    print("\n--- ALL REFERENCES CLEANED AND MANIFEST UPDATED ---")

if __name__ == "__main__":
    from app.config import setup_hardware
    setup_hardware()
    clean_all_references()
