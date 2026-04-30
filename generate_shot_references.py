import os
import json
import numpy as np
from tqdm import tqdm
import sys

# Add the project root to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.biomechanics.pose_extractor import PoseExtractor
from core.syncing.sync_engine import SyncEngine

DATASET_ROOT = r"E:\PYTHON\Cricket_Shot_Dataset\cricketshot\cricketshot\train"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "references")

def process_shot(shot_name, shot_dir, num_videos=25):
    print(f"\nProcessing shot: {shot_name}")
    
    # Get first N valid videos
    valid_exts = {".avi", ".mp4"}
    videos = [f for f in os.listdir(shot_dir) if not f.startswith("._") and os.path.splitext(f)[1].lower() in valid_exts]
    videos = sorted(videos)[:num_videos]
    
    if not videos:
        print(f"No valid videos found in {shot_dir}")
        return
        
    print(f"Selected {len(videos)} videos for {shot_name}.")
    
    extractor = PoseExtractor()
    sync_engine = SyncEngine()
    
    all_pose_data = []
    
    for i, vid in enumerate(videos):
        vid_path = os.path.join(shot_dir, vid)
        print(f"Extracting pose ({i+1}/{len(videos)}): {vid}")
        try:
            pose_data = extractor.extract_from_video(vid_path)
            if pose_data and pose_data.get("frames"):
                all_pose_data.append(pose_data)
            else:
                print(f"  Warning: No pose data extracted from {vid}")
        except Exception as e:
            print(f"  Error extracting from {vid}: {e}")
            
    if not all_pose_data:
        print(f"Failed to extract pose data for any videos in {shot_name}")
        return

    print(f"Aggregating data for {shot_name} using DTW...")
    
    # Use the first video as the template
    template_data = all_pose_data[0]
    num_template_frames = len(template_data["frames"])
    joints = template_data["metadata"]["joints"]
    
    # Initialize a list of lists to hold angles from all videos for each template frame
    # frame_angles[frame_idx][joint_name] = [angle1, angle2, ...]
    frame_angles = [ {joint: [] for joint in joints} for _ in range(num_template_frames) ]
    
    # Add template's own data
    for i, frame in enumerate(template_data["frames"]):
        for joint in joints:
            if joint in frame.get("angles", {}):
                frame_angles[i][joint].append(frame["angles"][joint])
                
    # Align other videos to the template
    for i in range(1, len(all_pose_data)):
        practice_data = all_pose_data[i]
        
        # We need vectors for DTW
        joint_weights = {
            "left_elbow": 1.5, "right_elbow": 1.5,
            "left_shoulder": 1.2, "right_shoulder": 1.2,
            "left_hip": 1.0, "right_hip": 1.0,
            "left_knee": 0.8, "right_knee": 0.8
        }
        
        def get_vector(frame):
            angles = frame.get("angles", {})
            if not angles:
                return []
            return [angles.get(j, 0.0) * joint_weights[j] for j in joints]
            
        practice_vectors = [get_vector(f) for f in practice_data["frames"]]
        template_vectors = [get_vector(f) for f in template_data["frames"]]
        
        alignment_path = sync_engine.compute_dtw(practice_vectors, template_vectors)
        
        for p_idx, t_idx in alignment_path:
            p_frame = practice_data["frames"][p_idx]
            for joint in joints:
                if joint in p_frame.get("angles", {}):
                    frame_angles[t_idx][joint].append(p_frame["angles"][joint])
                    
    print(f"Calculating statistics for {shot_name}...")
    
    result_frames = []
    
    for i in range(num_template_frames):
        frame_stat = {
            "frame_idx": i,
            "time_sec": template_data["frames"][i].get("time_sec", 0.0),
            "mean_angles": {},
            "min_angles": {},
            "max_angles": {},
            "q1_angles": {},
            "q3_angles": {}
        }
        
        for joint in joints:
            angles = frame_angles[i][joint]
            if not angles:
                continue
                
            arr = np.array(angles)
            frame_stat["mean_angles"][joint] = float(np.mean(arr))
            frame_stat["min_angles"][joint] = float(np.min(arr))
            frame_stat["max_angles"][joint] = float(np.max(arr))
            frame_stat["q1_angles"][joint] = float(np.percentile(arr, 25))
            frame_stat["q3_angles"][joint] = float(np.percentile(arr, 75))
            
        result_frames.append(frame_stat)
        
    output_data = {
        "metadata": {
            "shot_type": shot_name,
            "videos_processed": len(all_pose_data),
            "template_fps": template_data["metadata"]["fps"],
            "joints": joints
        },
        "frames": result_frames
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"{shot_name}_stats.json")
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved {shot_name} statistics to {out_file}")

def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root {DATASET_ROOT} not found.")
        return
        
    shot_types = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and not d.startswith("._")]
    
    for shot in shot_types:
        shot_dir = os.path.join(DATASET_ROOT, shot)
        process_shot(shot, shot_dir, num_videos=25)

if __name__ == "__main__":
    main()
