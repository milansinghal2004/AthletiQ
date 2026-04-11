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
# Since this file is now in the 'app/' subdirectory, we go one level up for the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    # Insert at the beginning to prioritize local modules over environment-wide ones
    sys.path.insert(0, PROJECT_ROOT)

# Add SAM2 to path (keeping submodule isolation)
SAM2_PATH = os.path.join(PROJECT_ROOT, "segment-anything-2")
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

# Import custom modules from the new 'core' package
try:
    from core.biomechanics.pose_extractor import PoseExtractor
    from core.syncing.sync_engine import SyncEngine
    from core.biomechanics.angle_calculation import compute_joint_angles
except ImportError as e:
    print(f"Error importing core modules: {e}")
    PoseExtractor = None
    SyncEngine = None
    compute_joint_angles = None

try:
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.utils.misc
    from torch.nn.attention import SDPBackend
    # Monkey-patch SAM2's SDPA backend selection
    sam2.utils.misc.get_sdp_backends = lambda dropout_p: [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    build_sam2_video_predictor = None

# --- Professional Path Configurations ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam2/checkpoints/sam2_hiera_small.pt")
MODEL_CFG = "configs/sam2/sam2_hiera_s.yaml" 
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe/pose_landmarker.task")
REFERENCES_DB_PATH = os.path.join(ASSETS_DIR, "references/reference_shots.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPU / GPU Optimizations
if DEVICE == "cuda":
    torch.set_float32_matmul_precision('high')
    print("GPU Optimization: Enabled high-precision matmul for Ampere.")
else:
    torch.set_num_threads(multiprocessing.cpu_count())
    print(f"CPU Optimization: Set torch threads to {multiprocessing.cpu_count()}")

# Global models
extractor = PoseExtractor(model_asset_path=POSE_MODEL_PATH) if PoseExtractor else None
sync_engine = SyncEngine() if SyncEngine else None

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

# --- Segmentation Flow ---
def process_video_for_seg(video_path):
    if not video_path:
        return None, None
        
    frames_dir = os.path.join(PROJECT_ROOT, "temp_video_frames")
    clear_temp("temp_video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    idx = 0
    first_frame = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- SPEED OPTIMIZATION: Downsample for GPU/CPU efficiency ---
        # 640px is plenty for posture and masks, and keeps VRAM usage low
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
    return first_frame, first_frame

def segment_player(video_path, click_coords, shot_type=None, progress=gr.Progress()):
    import traceback
    try:
        if not video_path or not click_coords:
            return None, None
        
        if not predictor:
            return None, None
            
        frames_dir = os.path.join(PROJECT_ROOT, "temp_video_frames")
        
        # Set autocast context based on device (enable bfloat16 on CPU for speed if supported)
        if DEVICE == "cuda":
            autocast_ctx = torch.autocast(DEVICE, dtype=torch.bfloat16)
        elif DEVICE == "cpu":
            try:
                autocast_ctx = torch.autocast(DEVICE, dtype=torch.bfloat16)
            except:
                autocast_ctx = torch.inference_mode()
        else:
            autocast_ctx = torch.inference_mode()
        
        with torch.inference_mode(), autocast_ctx:
            inference_state = predictor.init_state(video_path=frames_dir)
            predictor.reset_state(inference_state)
            
            points = np.array([[click_coords[0], click_coords[1]]], dtype=np.float32)
            labels = np.array([1], np.int32)
            
            predictor.add_new_points(inference_state, frame_idx=0, obj_id=1, points=points, labels=labels)
            
            video_segments = {}
            # We don't know total frames yet until we extract them, 
            # but we can get it from the cap earlier or just use a generic description
            print("Running SAM2 Propagation...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                progress(out_frame_idx / 100, desc=f"Propagating SAM2 (Frame {out_frame_idx})...")
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        
        out_video_path = os.path.join(PROJECT_ROOT, "outputs/output_segmented.mp4")
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or fps is None: fps = 30.0
        
        writer = imageio.get_writer(out_video_path, fps=fps, codec='libx264', macro_block_size=None)
    
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            progress(idx / max(1, total_frames), desc=f"Rendering Isolated Player (Frame {idx}/{total_frames})...")
            
            # Create a black frame for isolated cutout
            isolated_frame = np.zeros_like(frame)
            
            if idx in video_segments and 1 in video_segments[idx]:
                mask = video_segments[idx][1]
                if mask.ndim == 3: mask = mask[0]
                
                # Rescale mask if needed
                if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                    mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                mask_indices = mask > 0
                # Copy original pixels only where the mask is positive
                isolated_frame[mask_indices] = frame[mask_indices]
            
            writer.append_data(cv2.cvtColor(isolated_frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        writer.close()
    
        # --- NEW: Run Biomechanics on segmented video ---
        bio_json_path = None
        if extractor and compute_joint_angles:
            print("Running Biomechanics on segmented video...")
            def bio_progress(curr, total, msg):
                progress(curr / max(1, total), desc=f"Biomechanics: {msg}")
                
            pose_data = extractor.extract_from_video(out_video_path, progress_callback=bio_progress)
            angle_results = []
            for frame in pose_data["frames"]:
                if "landmarks" in frame:
                    angles = compute_joint_angles(frame["landmarks"])
                    angle_results.append({
                        "frame": frame["frame_idx"],
                        "time": frame["time_sec"],
                        **angles
                    })
        
        bio_json_path = os.path.join(PROJECT_ROOT, "outputs/segmented_biomechanics.json")
        with open(bio_json_path, "w") as f:
            json.dump(angle_results, f, indent=4)
        print(f"Biomechanics saved to {bio_json_path}")
        
        # --- NEW: If shot_type is provided, run the comparison on the isolated video ---
        sync_comparison_path = None
        if shot_type and shot_type != "None" and sync_engine:
            print(f"Running comparative analysis for {shot_type}...")
            # We use the isolated video for comparison as requested
            sync_comparison_path = sync_and_compare(out_video_path, shot_type, progress=progress)

        # Free memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        return out_video_path, bio_json_path, sync_comparison_path
    except Exception as e:
        print(f"Error in segment_player: {e}")
        traceback.print_exc()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return None, None, None

# --- Shot Sync Flow ---
def load_references():
    if os.path.exists(REFERENCES_DB_PATH):
        with open(REFERENCES_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def sync_and_compare(practice_video, shot_type, progress=gr.Progress()):
    if not practice_video or not shot_type or not sync_engine:
        print("Sync: Input missing or sync engine not initialized.")
        return None
        
    refs = load_references()
    if shot_type not in refs:
        print(f"Sync error: Shot type '{shot_type}' not found in database.")
        return None
        
    ref_info = refs[shot_type]
    # Handle both relative and absolute paths for reference video & angles
    ref_video = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["video_path"]))
    ref_angles_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["angles_path"]))
    
    if not os.path.exists(ref_video):
        print(f"Sync error: Reference video missing at {ref_video}")
        return None

    def progress_cb(current, total, desc):
        progress((current / total) * 0.4, desc=desc)
        
    practice_data = extractor.extract_from_video(practice_video, progress_callback=progress_cb)
    
    if os.path.exists(ref_angles_path):
        with open(ref_angles_path, "r") as f:
            reference_data = json.load(f)
    else:
        print(f"Reference angle data not found at {ref_angles_path}. Generating from {ref_video}...")
        reference_data = extractor.extract_from_video(ref_video, progress_callback=None)
        os.makedirs(os.path.dirname(ref_angles_path), exist_ok=True)
        with open(ref_angles_path, "w") as f:
            json.dump(reference_data, f)
        
    mapping = sync_engine.sync_videos(practice_data, reference_data)
    
    # Rendering side-by-side
    out_video = create_synced_video(practice_video, ref_video, mapping, progress=progress)
    return out_video

def create_synced_video(practice_video, reference_video, mapping, progress=None):
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
    writer = imageio.get_writer(out_path, fps=slow_fps, codec='libx264', macro_block_size=None)
    
    ref_frames = []
    while True:
        ret, f = cap_r.read()
        if not ret: break
        ref_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_r, target_h)))
    cap_r.release()
    
    p_idx = 0
    total_p = int(cap_p.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame_p = cap_p.read()
        if not ret: break
        r_idx = mapping.get(p_idx, 0)
        if r_idx >= len(ref_frames): r_idx = len(ref_frames) - 1
        frame_r = ref_frames[r_idx]
        p_resized = cv2.resize(cv2.cvtColor(frame_p, cv2.COLOR_BGR2RGB), (new_w_p, target_h))
        combined = np.hstack((p_resized, frame_r))
        cv2.putText(combined, "PRACTICE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "REFERENCE", (new_w_p+10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        writer.append_data(combined)
        p_idx += 1
        if progress: progress(0.6 + (p_idx/total_p)*0.4, desc="Rendering Comparison...")
    
    cap_p.release()
    writer.close()
    return out_path

# --- UI Definitions ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏏 AthletiQ - Unified Performance Pipeline")
    gr.Markdown("Consolidated technical analysis: Segmentation -> Biomechanics -> Comparison.")
    
    with gr.TabItem("Performance Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="1. Upload Practice Video")
                shot_select = gr.Dropdown(
                    choices=["None"] + list(load_references().keys()), 
                    value="None", 
                    label="2. Select Shot Type (Optional for Comparison)"
                )
                first_frame_display = gr.Image(label="3. Click on Batsman to Segment", interactive=False)
                analyze_btn = gr.Button("🚀 Run Full Shot Analysis", variant="primary")
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📊 Analysis Results")
                    out_isolated = gr.Video(label="Isolated Player (Cutout)")
                    out_comparison = gr.Video(label="Technical Comparison (Side-by-Side)")
                    out_json = gr.File(label="Joint Angle Data (JSON)")

        # Hidden states for tracking
        clean_img_state = gr.State()
        click_coord_state = gr.State()

        # Step 1: When video is uploaded, prepare the first frame for clicking
        video_input.upload(process_video_for_seg, video_input, [first_frame_display, clean_img_state])

        # Step 2: Handle point selection on the first frame
        def handle_point_selection(img, evt: gr.SelectData):
            points_img = img.copy()
            cv2.circle(points_img, evt.index, 7, (0, 255, 0), -1)
            return evt.index, points_img

        first_frame_display.select(handle_point_selection, clean_img_state, [click_coord_state, first_frame_display])

        # Step 3: Run the unified pipeline (segmentation -> biomechanics -> comparison)
        analyze_btn.click(
            segment_player, 
            inputs=[video_input, click_coord_state, shot_select], 
            outputs=[out_isolated, out_json, out_comparison]
        )

    gr.Markdown("---")
    gr.Markdown("Powered by SAM2, MediaPipe, and AthletiQ Engine.")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), allowed_paths=[os.path.join(PROJECT_ROOT, "outputs")])
