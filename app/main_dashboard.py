import os
import sys
import cv2
import gradio as gr
import numpy as np
import json
import shutil
import torch
import imageio
import atexit
import multiprocessing
from datetime import datetime

# Setup Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add SAM2 to path
SAM2_PATH = os.path.join(PROJECT_ROOT, "segment-anything-2")
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

# Import custom modules
try:
    from core.biomechanics.pose_extractor import PoseExtractor
    from core.syncing.sync_engine import SyncEngine
    from core.biomechanics.angle_calculation import compute_joint_angles
    from core.shot_classifier import ShotClassifier
except ImportError as e:
    print(f"Error importing core modules: {e}")
    PoseExtractor = None
    SyncEngine = None
    compute_joint_angles = None
    ShotClassifier = None

from app.plotting_utils import generate_biomechanic_plot

try:
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.utils.misc
    from torch.nn.attention import SDPBackend
    sam2.utils.misc.get_sdp_backends = lambda dropout_p: [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    build_sam2_video_predictor = None

# --- Configurations ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam2/checkpoints/sam2_hiera_small.pt")
MODEL_CFG = "configs/sam2/sam2_hiera_s.yaml" 
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe/pose_landmarker.task")
REFERENCES_DB_PATH = os.path.join(ASSETS_DIR, "references/reference_shots.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.set_float32_matmul_precision('high')
    print("GPU Optimization: Enabled high-precision matmul for Ampere.")
else:
    torch.set_num_threads(multiprocessing.cpu_count())
    print(f"CPU Optimization: Set torch threads to {multiprocessing.cpu_count()}")

extractor = PoseExtractor(model_asset_path=POSE_MODEL_PATH) if PoseExtractor else None
sync_engine = SyncEngine() if SyncEngine else None

# ── Shot Classifier ───────────────────────────────────
shot_classifier = None
if ShotClassifier:
    try:
        shot_classifier = ShotClassifier(
            model_path  = os.path.join(MODELS_DIR, "shot_detection", "cricket_shot_r3d18_final.pth"),
            config_path = os.path.join(MODELS_DIR, "shot_detection", "cricket_shot_config.json")
        )
    except Exception as e:
        print(f"Error loading ShotClassifier: {e}")
        shot_classifier = None

predictor = None
if build_sam2_video_predictor and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading SAM2 model on {DEVICE}...")
    predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
    print("SAM2 model loaded successfully.")

# --- Shared Utilities ---
def clear_temp(dir_name="temp_video_frames"):
    temp_path = os.path.join(PROJECT_ROOT, dir_name)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

atexit.register(clear_temp)

def convert_to_mp4(input_path):
    """Transcodes videos to a browser-compatible MP4 format if needed."""
    if not input_path or input_path.lower().endswith('.mp4'):
        return input_path
    
    output_path = os.path.join(PROJECT_ROOT, "outputs", "playable_input.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Transcoding {input_path} for browser compatibility...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 25.0)
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=None)
        for frame in reader:
            writer.append_data(frame)
        writer.close()
        reader.close()
        return output_path
    except Exception as e:
        print(f"Transcoding failed ({e}). Trying original file.")
        return input_path

SHOT_LABEL_MAP = {
    "cover": "Cover Drive",
    "defense": "Defense",
    "flick": "Flick",
    "hook": "Hook",
    "late_cut": "Late Cut",
    "lofted": "Lofted Shot",
    "pull": "Pull Shot",
    "square_cut": "Square Cut",
    "straight": "Straight Drive",
    "sweep": "Sweep Shot"
}

# --- Segmentation Flow ---
def process_video_for_seg(video_path):
    if not video_path:
        return None, None, None, gr.update()
        
    # ── Auto predict shot type ────────────────────────
    predicted_shot = "None"
    if shot_classifier:
        try:
            result = shot_classifier.predict(video_path)
            raw_shot = result["shot"]
            predicted_shot = SHOT_LABEL_MAP.get(raw_shot, "None")
            conf = result["confidence"] * 100
            print(f"🏏 Auto-detected: {predicted_shot} ({conf:.1f}% confidence)")
            print(f"DEBUG → raw: {raw_shot} | mapped: {predicted_shot}")
        except Exception as e:
            print(f"Shot classifier error: {e}")
            predicted_shot = "None"

    # Ensure video is playable in browser UI
    playable_path = convert_to_mp4(video_path)
    
    frames_dir = os.path.join(PROJECT_ROOT, "temp_video_frames")
    clear_temp("temp_video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(playable_path)
    idx = 0
    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if idx == 0:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(frames_dir, f"{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return first_frame, first_frame, playable_path, gr.update(value=predicted_shot)

def segment_player(video_path, click_coords, shot_type=None, progress=gr.Progress()):
    import traceback
    try:
        if not video_path or not click_coords:
            return None, None, None
        
        if not predictor:
            return None, None, None
            
        frames_dir = os.path.join(PROJECT_ROOT, "temp_video_frames")
        
        if DEVICE == "cuda":
            autocast_ctx = torch.autocast(DEVICE, dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.inference_mode()
        
        with torch.inference_mode(), autocast_ctx:
            inference_state = predictor.init_state(video_path=frames_dir)
            predictor.reset_state(inference_state)
            
            points = np.array([[click_coords[0], click_coords[1]]], dtype=np.float32)
            labels = np.array([1], np.int32)
            
            predictor.add_new_points(inference_state, frame_idx=0, obj_id=1, points=points, labels=labels)
            
            video_segments = {}
            tracked_indices = []
            print("Running SAM2 Propagation...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                progress(out_frame_idx / 100, desc=f"Propagating SAM2 (Frame {out_frame_idx})...")
                
                # Identify if batsman (obj_id 1) is being tracked in this frame
                frame_masks = {}
                is_tracked = False
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    frame_masks[out_obj_id] = mask
                    if out_obj_id == 1 and mask.any() and mask.sum() > 100:
                        is_tracked = True
                
                video_segments[out_frame_idx] = frame_masks
                if is_tracked:
                    tracked_indices.append(out_frame_idx)

            if not tracked_indices:
                print("Warning: No batsman tracking data found.")
                return None, None, None, "Could not track the batsman. Please try clicking again."
            
            start_idx = min(tracked_indices)
            end_idx = max(tracked_indices)
            print(f"🏏 Autoclipping: Batsman tracked from frame {start_idx} to {end_idx}")
        
        out_video_path = os.path.join(PROJECT_ROOT, "outputs/output_segmented.mp4")
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = imageio.get_writer(out_video_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
        idx = 0
        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Autoclipping: Only write frames within the tracked range
            if start_idx <= idx <= end_idx:
                progress(frames_written / max(1, (end_idx - start_idx + 1)), desc=f"Rendering Isolated Player (Frame {idx}/{end_idx})...")
                
                isolated_frame = np.zeros_like(frame)
                if idx in video_segments and 1 in video_segments[idx]:
                    mask = video_segments[idx][1]
                    if mask.ndim == 3: mask = mask[0]
                    if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_indices = mask > 0
                    isolated_frame[mask_indices] = frame[mask_indices]
                
                writer.append_data(cv2.cvtColor(isolated_frame, cv2.COLOR_BGR2RGB))
                frames_written += 1
            
            idx += 1
        cap.release()
        writer.close()
    
        bio_json_path = os.path.join(PROJECT_ROOT, "outputs/segmented_biomechanics.json")
        if extractor:
            def bio_progress(curr, total, msg):
                progress(curr / max(1, total), desc=f"Biomechanics: {msg}")
            pose_data = extractor.extract_from_video(out_video_path, progress_callback=bio_progress)
            angle_results = []
            for frame in pose_data["frames"]:
                if "angles" in frame:
                    angle_results.append({"frame": frame["frame_idx"], "time": frame["time_sec"], **frame["angles"]})
            with open(bio_json_path, "w") as f:
                json.dump(angle_results, f, indent=4)
        
        sync_comparison_path = None
        score_feedback = "No shot selected for comparison."
        plots = [None] * 6 # Elbows (2), Knees (2), Hips (2)
        if shot_type and shot_type != "None" and sync_engine:
            # Use the CLIPPED video for comparison to improve DTW accuracy
            result = sync_and_compare(out_video_path, shot_type, progress=progress)
            if result:
                sync_comparison_path, score_feedback, plots = result

        if DEVICE == "cuda": torch.cuda.empty_cache()
        return out_video_path, bio_json_path, sync_comparison_path, score_feedback, *plots
    except Exception as e:
        print(f"Error in segment_player: {e}")
        traceback.print_exc()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        return None, None, None, "An error occurred during processing.", None, None, None, None

# --- Shot Sync Flow ---
def load_references():
    if os.path.exists(REFERENCES_DB_PATH):
        with open(REFERENCES_DB_PATH, "r") as f:
            refs = json.load(f)
            # Inject stats path based on video name
            for key, val in refs.items():
                base = os.path.basename(val["video_path"])
                prefix = base.split("_reference.mp4")[0]
                val["stats_path"] = f"assets/references/{prefix}_stats.json"
            return refs
    return {}

def sync_and_compare(practice_video, shot_type, progress=gr.Progress()):
    if not practice_video or not shot_type or not sync_engine: return None, "Setup missing", [None, None, None, None, None, None]
    refs = load_references()
    if shot_type not in refs: return None, "Shot not found", [None, None, None, None, None, None]
    
    ref_info = refs[shot_type]
    ref_video = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["video_path"]))
    stats_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("stats_path", "")))
    
    if not os.path.exists(ref_video): return None, "Visual reference not found", [None, None, None, None, None, None]
    if not os.path.exists(stats_path): return None, "Stats reference not found", [None, None, None, None, None, None]
    
    def progress_cb(current, total, desc): progress((current / total) * 0.4, desc=desc)
    practice_data = extractor.extract_from_video(practice_video, progress_callback=progress_cb)
    
    with open(stats_path, "r") as f:
        stats_data = json.load(f)
        
    # Load full reference pose data (including landmarks) for phase detection
    ref_angles_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("angles_path", "")))
    if os.path.exists(ref_angles_path):
        with open(ref_angles_path, "r") as f:
            reference_data = json.load(f)
    else:
        # Fallback to pseudo-reference from stats if full JSON missing
        reference_data = {"frames": []}
        for frame_stat in stats_data["frames"]:
            reference_data["frames"].append({
                "frame_idx": frame_stat["frame_idx"],
                "time_sec": frame_stat["time_sec"],
                "angles": frame_stat.get("mean_angles", {})
            })
        
    alignment_path, p_phases, r_phases = sync_engine.sync_videos(practice_data, reference_data)
    
    # Terminal Reporting with Colors
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print(f"\n{BOLD}Analysis for Shot Type: {GREEN}{shot_type}{RESET}")
    if p_phases and r_phases:
        print(f"{GREEN}Frame Generalization Data:{RESET}")
        print(f"  - Practice Strike: {GREEN}Frame {p_phases['strike']}{RESET}")
        print(f"  - Reference Strike: {GREEN}Frame {r_phases['strike']}{RESET}\n")

    # Create mapping for scoring (last-win logic is fine for statistical scoring)
    mapping = {p: r for p, r in alignment_path}
    
    # Save frame generalization (phases) metadata
    meta_path = os.path.join(PROJECT_ROOT, "outputs/sync_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "shot_type": shot_type,
            "practice_phases": p_phases,
            "reference_phases": r_phases
        }, f, indent=4)

    # Calculate Numerical Score
    joint_weights = {
        "left_elbow": 1.5, "right_elbow": 1.5,
        "left_shoulder": 1.2, "right_shoulder": 1.2,
        "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 0.8, "right_knee": 0.8
    }
    
    total_score = 0
    total_weight = 0
    
    for p_idx, r_idx in mapping.items():
        if p_idx >= len(practice_data["frames"]) or r_idx >= len(stats_data["frames"]):
            continue
            
        p_angles = practice_data["frames"][p_idx].get("angles", {})
        stat_frame = stats_data["frames"][r_idx]
        q1_a = stat_frame.get("q1_angles", {})
        q3_a = stat_frame.get("q3_angles", {})
        min_a_ref = stat_frame.get("min_angles", {})
        max_a_ref = stat_frame.get("max_angles", {})
        
        for joint, weight in joint_weights.items():
            if joint in p_angles and joint in q1_a and joint in q3_a:
                val = p_angles[joint]
                q1, q3 = q1_a[joint], q3_a[joint]
                mn, mx = min_a_ref.get(joint, q1-10), max_a_ref.get(joint, q3+10)
                
                s = 100 if q1 <= val <= q3 else (
                    100 * (val - mn) / (q1 - mn) if val < q1 and mn != q1 else
                    100 * (mx - val) / (mx - q3) if val > q3 and mx != q3 else 0
                )
                total_score += max(0, s) * weight
                total_weight += weight
                
    final_percentage = (total_score / total_weight) if total_weight > 0 else 0
    feedback_str = f"### 📊 Overall Accuracy Score: {final_percentage:.1f}%\n"
    feedback_str += "Biomechanical sync complete. Ready for external LLM evaluation."
    
        
    # ── Generate Plotly Charts & Detailed Breakdown ────────────────────────
    plots = []
    breakdown_items = []
    try:
        # Critical joints for cricket
        joints_to_plot = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]
        
        for joint in joints_to_plot:
            p_vals = []
            q1_vals = []
            q3_vals = []
            mean_vals = []
            
            for p_idx, r_idx in mapping.items():
                if p_idx < len(practice_data["frames"]) and r_idx < len(stats_data["frames"]):
                    p_angle = practice_data["frames"][p_idx].get("angles", {}).get(joint, 0)
                    p_vals.append(p_angle)
                    q1_vals.append(stats_data["frames"][r_idx].get("q1_angles", {}).get(joint, 0))
                    q3_vals.append(stats_data["frames"][r_idx].get("q3_angles", {}).get(joint, 0))
                    mean_vals.append(stats_data["frames"][r_idx].get("mean_angles", {}).get(joint, 0))
            
            ref_stats = {"q1": q1_vals, "q3": q3_vals, "mean": mean_vals}
            fig = generate_biomechanic_plot(joint, p_vals, ref_stats, p_phases)
            plots.append(fig)
            
        # Breakdown items calculation removed for external LLM evaluation.
            
    except Exception as e:
        print(f"Error generating plots/breakdown: {e}")
        if not plots: plots = [None] * 6

    out_video = create_synced_video(practice_video, ref_video, alignment_path, progress=progress)
    return out_video, feedback_str, plots

def create_synced_video(practice_video, reference_video, alignment_path, progress=None):
    cap_p = cv2.VideoCapture(practice_video)
    cap_r = cv2.VideoCapture(reference_video)
    fps = cap_p.get(cv2.CAP_PROP_FPS) or 30.0
    slow_fps = fps * 0.3
    
    target_h = 480
    w_p, h_p = int(cap_p.get(3)), int(cap_p.get(4))
    w_r, h_r = int(cap_r.get(3)), int(cap_r.get(4))
    scale_p, scale_r = target_h/h_p, target_h/h_r
    new_w_p, new_w_r = (int(w_p*scale_p)//2)*2, (int(w_r*scale_r)//2)*2
    
    out_path = os.path.join(PROJECT_ROOT, "outputs/synced_comparison.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=slow_fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    # Pre-load all frames for smooth indexed access
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
        
        if progress: 
            progress(0.6 + (step_idx/total_steps)*0.4, desc="Rendering Comparison...")
    
    writer.close()
    return out_path

# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏏 AthletiQ - Unified Performance Pipeline")
    
    with gr.Tab("Performance Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="1. Upload Practice Video")
                shot_select = gr.Dropdown(choices=["None"] + list(load_references().keys()), value="None", label="2. Select Shot Type")
                first_frame_display = gr.Image(label="3. Click on Batsman to Segment", interactive=False)
                analyze_btn = gr.Button("🚀 Run Full Shot Analysis", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                out_score = gr.Markdown("Score will appear here.")
                out_isolated = gr.Video(label="Isolated Player (Cutout)")
                out_comparison = gr.Video(label="Technical Comparison (Side-by-Side)")
                out_json = gr.File(label="Joint Angle Data (JSON)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Biomechanical Angle Trends")
                with gr.Tabs():
                    with gr.Tab("Elbows"):
                        plot_l_elbow = gr.Plot(label="Left Elbow")
                        plot_r_elbow = gr.Plot(label="Right Elbow")
                    with gr.Tab("Knees"):
                        plot_l_knee = gr.Plot(label="Left Knee")
                        plot_r_knee = gr.Plot(label="Right Knee")
                    with gr.Tab("Hips"):
                        plot_l_hip = gr.Plot(label="Left Hip")
                        plot_r_hip = gr.Plot(label="Right Hip")

        with gr.Accordion("ℹ️ How to read these charts?", open=False):
            gr.Markdown("""
            - **Blue Line**: Your technique.
            - **Green Shaded Area**: The Professional 'Ideal' Zone (IQR).
            - **Dashed Red Line**: The moment of impact (Strike).
            - **Goal**: Keep your blue line within or close to the green corridor during the strike phase.
            """)

        clean_img_state = gr.State()
        click_coord_state = gr.State()

        video_input.upload(process_video_for_seg, video_input, [first_frame_display, clean_img_state, video_input, shot_select])

        def handle_point_selection(img, evt: gr.SelectData):
            points_img = img.copy()
            cv2.circle(points_img, evt.index, 7, (0, 255, 0), -1)
            return evt.index, points_img

        first_frame_display.select(handle_point_selection, clean_img_state, [click_coord_state, first_frame_display])

        analyze_btn.click(
            segment_player, 
            inputs=[video_input, click_coord_state, shot_select], 
            outputs=[
                out_isolated, out_json, out_comparison, out_score, 
                plot_l_elbow, plot_r_elbow, plot_l_knee, plot_r_knee, plot_l_hip, plot_r_hip
            ]
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), allowed_paths=[os.path.join(PROJECT_ROOT, "outputs")])
