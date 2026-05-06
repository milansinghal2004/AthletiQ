import os
import json
import cv2
import imageio
import numpy as np
from app.config import PROJECT_ROOT, REFERENCES_DB_PATH, OUTPUTS_DIR
from app.plotting_utils import generate_biomechanic_plot

def load_references():
    if os.path.exists(REFERENCES_DB_PATH):
        with open(REFERENCES_DB_PATH, "r") as f:
            refs = json.load(f)
            for key, val in refs.items():
                # Use original path to derive stats prefix if available
                source_path = val.get("video_path_original", val["video_path"])
                base = os.path.basename(source_path)
                prefix = base.split("_reference")[0]
                val["stats_path"] = f"assets/references/{prefix}_stats.json"
                
                # Double check the stats path exists
                if not os.path.exists(os.path.join(PROJECT_ROOT, val["stats_path"])):
                    # Fallback: Try shot name as prefix
                    shot_prefix = key.lower().replace(" ", "_")
                    val["stats_path"] = f"assets/references/{shot_prefix}_stats.json"
            return refs
    return {}

def create_synced_video(practice_video, reference_video, alignment_path, progress_callback=None):
    cap_p = cv2.VideoCapture(practice_video)
    cap_r = cv2.VideoCapture(reference_video)
    fps = cap_p.get(cv2.CAP_PROP_FPS) or 30.0
    slow_fps = fps * 0.3
    
    target_h = 480
    w_p, h_p = int(cap_p.get(3)), int(cap_p.get(4))
    w_r, h_r = int(cap_r.get(3)), int(cap_r.get(4))
    scale_p, scale_r = target_h/h_p, target_h/h_r
    new_w_p, new_w_r = (int(w_p*scale_p)//2)*2, (int(w_r*scale_r)//2)*2
    
    out_path = os.path.join(OUTPUTS_DIR, "synced_comparison.mp4")
    writer = imageio.get_writer(out_path, fps=slow_fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    ref_frames = []
    while True:
        ret, f = cap_r.read()
        if not ret: break
        ref_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_r, target_h)))
    cap_r.release()
    
    p_frames = []
    while True:
        ret, f = cap_p.read()
        if not ret: break
        p_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_p, target_h)))
    cap_p.release()
    
    total_steps = len(alignment_path)
    for step_idx, (p_idx, r_idx) in enumerate(alignment_path):
        if p_idx >= len(p_frames): p_idx = len(p_frames) - 1
        if r_idx >= len(ref_frames): r_idx = len(ref_frames) - 1
        
        frame_p = p_frames[p_idx]
        frame_r = ref_frames[r_idx]
        
        combined = np.hstack((frame_p, frame_r))
        cv2.putText(combined, "PRACTICE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "REFERENCE", (new_w_p+10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        writer.append_data(combined)
        
        if progress_callback: 
            progress_callback(0.6 + (step_idx/total_steps)*0.4, "Rendering Comparison...")
    
    writer.close()
    return out_path

def run_sync_logic(practice_data, shot_type, practice_video, extractor, sync_engine, progress_callback=None):
    if not practice_video or not shot_type or not sync_engine: return None, "Setup missing", [None]*6
    
    refs = load_references()
    if shot_type not in refs: return None, "Shot not found", [None]*6
    
    ref_info = refs[shot_type]
    ref_video = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["video_path"]))
    stats_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("stats_path", "")))
    
    if not os.path.exists(ref_video): return None, "Ref video missing", [None]*6
    if not os.path.exists(stats_path): return None, "Ref stats missing", [None]*6
    
    with open(stats_path, "r") as f:
        stats_data = json.load(f)
        
    ref_angles_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("angles_path", "")))
    if os.path.exists(ref_angles_path):
        with open(ref_angles_path, "r") as f:
            reference_data = json.load(f)
    else:
        reference_data = {"frames": [{"frame_idx": fs["frame_idx"], "angles": fs.get("mean_angles", {})} for fs in stats_data["frames"]]}

    alignment_path, p_phases, r_phases = sync_engine.sync_videos(practice_data, reference_data)
    
    # Colored Terminal Output for Phases
    print(f"\033[92m\n--- Analysis for Shot Type: {shot_type} ---")
    print(f"Practice Phases (Frames): {p_phases}")
    print(f"Reference Phases (Frames): {r_phases}\n\033[0m")

    mapping = {p: r for p, r in alignment_path}

    # --- Scoring ---
    joint_weights = {"left_elbow": 1.5, "right_elbow": 1.5, "left_shoulder": 1.2, "right_shoulder": 1.2, "left_hip": 1.0, "right_hip": 1.0, "left_knee": 0.8, "right_knee": 0.8}
    total_score, total_weight = 0, 0
    for p_idx, r_idx in mapping.items():
        if p_idx < len(practice_data["frames"]) and r_idx < len(stats_data["frames"]):
            p_angles = practice_data["frames"][p_idx].get("angles", {})
            stat_frame = stats_data["frames"][r_idx]
            q1_a, q3_a = stat_frame.get("q1_angles", {}), stat_frame.get("q3_angles", {})
            min_a_ref, max_a_ref = stat_frame.get("min_angles", {}), stat_frame.get("max_angles", {})
            
            for joint, weight in joint_weights.items():
                if joint in p_angles and joint in q1_a and joint in q3_a:
                    val = p_angles[joint]
                    q1, q3 = q1_a[joint], q3_a[joint]
                    mn, mx = min_a_ref.get(joint, q1-10), max_a_ref.get(joint, q3+10)
                    s = 100 if q1 <= val <= q3 else (100*(val-mn)/(q1-mn) if val<q1 and mn!=q1 else 100*(mx-val)/(mx-q3) if val>q3 and mx!=q3 else 0)
                    total_score += max(0, s) * weight
                    total_weight += weight
                    
    final_percentage = (total_score / total_weight) if total_weight > 0 else 0
    feedback_str = f"### Overall Accuracy Score: {final_percentage:.1f}%\nBiomechanical sync complete."

    # --- Plots ---
    plots = []
    joints_to_plot = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]
    for joint in joints_to_plot:
        p_vals, q1_vals, q3_vals, mean_vals = [], [], [], []
        for p_idx, r_idx in mapping.items():
            if p_idx < len(practice_data["frames"]) and r_idx < len(stats_data["frames"]):
                p_vals.append(practice_data["frames"][p_idx].get("angles", {}).get(joint, 0))
                q1_vals.append(stats_data["frames"][r_idx].get("q1_angles", {}).get(joint, 0))
                q3_vals.append(stats_data["frames"][r_idx].get("q3_angles", {}).get(joint, 0))
                mean_vals.append(stats_data["frames"][r_idx].get("mean_angles", {}).get(joint, 0))
        plots.append(generate_biomechanic_plot(joint, p_vals, {"q1": q1_vals, "q3": q3_vals, "mean": mean_vals}, p_phases))

    out_video = create_synced_video(practice_video, ref_video, alignment_path, progress_callback=progress_callback)
    return out_video, feedback_str, plots
