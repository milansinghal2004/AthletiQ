# Merged AthletiQ Dashboard
# Features from main dashboard + stick figure + interactive widget

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

extractor   = PoseExtractor(model_asset_path=POSE_MODEL_PATH) if PoseExtractor else None
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
        fps    = reader.get_meta_data().get('fps', 25.0)
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
    "cover":      "Cover Drive",
    "defense":    "Defense",
    "flick":      "Flick",
    "hook":       "Hook",
    "late_cut":   "Late Cut",
    "lofted":     "Lofted Shot",
    "pull":       "Pull Shot",
    "square_cut": "Square Cut",
    "straight":   "Straight Drive",
    "sweep":      "Sweep Shot"
}

# --- Segmentation Flow ---
def process_video_for_seg(video_path):
    if not video_path:
        return None, None, None, gr.update()

    # ── Auto predict shot type ────────────────────────
    predicted_shot = "None"
    if shot_classifier:
        try:
            result         = shot_classifier.predict(video_path)
            raw_shot       = result["shot"]
            predicted_shot = SHOT_LABEL_MAP.get(raw_shot, "None")
            conf           = result["confidence"] * 100
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
        if not ret:
            break
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


# ── NEW: Stick Figure Video Generator ─────────────────────────────────────────
def create_stick_figure_video(video_path, segmented_video_path=None, progress=None):
    """Run pose detection on segmented (isolated) video — guaranteed correct player."""
    if not extractor:
        return None

    source_path = segmented_video_path if segmented_video_path and os.path.exists(segmented_video_path) else video_path
    print(f"Stick figure source: {'segmented' if source_path == segmented_video_path else 'original'}")

    cap          = cv2.VideoCapture(source_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    BaseOptions           = mp_python.BaseOptions
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    VisionRunningMode     = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=POSE_MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    CONNECTIONS = [
        ("left_shoulder",  "right_shoulder"),
        ("left_shoulder",  "left_elbow"),
        ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow",    "right_wrist"),
        ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"),
        ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"),
        ("right_knee",     "right_ankle"),
    ]

    LANDMARK_INDICES = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow":    13, "right_elbow":    14,
        "left_wrist":    15, "right_wrist":    16,
        "left_hip":      23, "right_hip":      24,
        "left_knee":     25, "right_knee":     26,
        "left_ankle":    27, "right_ankle":    28,
    }

    out_path = os.path.join(PROJECT_ROOT, "outputs", "stick_figure.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None)

    cap  = cv2.VideoCapture(source_path)
    fidx = 0
    detected_count = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nonzero = np.nonzero(gray)

            if len(nonzero[0]) > 200:
                pad   = 30
                top   = max(0,  nonzero[0].min() - pad)
                bot   = min(H,  nonzero[0].max() + pad)
                left  = max(0,  nonzero[1].min() - pad)
                right = min(W,  nonzero[1].max() + pad)
                crop  = frame[top:bot, left:right].copy()

                # Boost brightness so MediaPipe works on isolated player
                hsv           = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2]  = np.clip(hsv[:, :, 2] * 1.8 + 30, 0, 255)
                crop          = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Add white background behind player (fixes black background problem)
                mask          = cv2.cvtColor(frame[top:bot, left:right], cv2.COLOR_BGR2GRAY)
                _, mask_bin   = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                white_bg      = np.ones_like(crop) * 200
                crop          = np.where(mask_bin[:, :, None] > 0, crop, white_bg).astype(np.uint8)

                crop_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                timestamp = int(fidx * 1000.0 / fps)
                result    = landmarker.detect_for_video(mp_image, timestamp)

                if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks  = result.pose_landmarks[0]
                    crop_h     = bot - top
                    crop_w     = right - left
                    detected_count += 1

                    lm = {}
                    for name, idx_lm in LANDMARK_INDICES.items():
                        lm[name] = {
                            "x": left + landmarks[idx_lm].x * crop_w,
                            "y": top  + landmarks[idx_lm].y * crop_h
                        }

                    for a, b in CONNECTIONS:
                        if a in lm and b in lm:
                            x1, y1 = int(lm[a]["x"]), int(lm[a]["y"])
                            x2, y2 = int(lm[b]["x"]), int(lm[b]["y"])
                            cv2.line(canvas, (x1, y1), (x2, y2), (30, 30, 30), 8)
                            cv2.line(canvas, (x1, y1), (x2, y2), (0, 200, 80), 4)

                    for name, pt in lm.items():
                        x, y = int(pt["x"]), int(pt["y"])
                        cv2.circle(canvas, (x, y), 10, (10, 10, 10), -1)
                        cv2.circle(canvas, (x, y),  8, (0, 180, 255), -1)

            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            fidx += 1

            if progress:
                progress(fidx / max(1, total_frames), desc=f"Generating Stick Figure ({fidx}/{total_frames})...")

    cap.release()
    writer.close()
    print(f"Stick figure: pose detected in {detected_count}/{fidx} frames")
    return out_path


# ── NEW: Interactive Pose Widget ───────────────────────────────────────────────
def generate_interactive_widget(pose_json_path, shot_type="None"):
    import json, os, math
    import numpy as np

    if not pose_json_path or not os.path.exists(pose_json_path):
        return "<p style='color:gray'>No pose data available.</p>"

    with open(pose_json_path, "r") as f:
        frames = json.load(f)

    if not frames:
        return "<p style='color:gray'>No pose data available.</p>"

    # ---- Average landmarks and angles across all frames ----
    all_landmarks, all_angles = {}, {}
    for frame in frames:
        for k, v in frame.get("landmarks", {}).items():
            all_landmarks.setdefault(k, []).append([v["x"], v["y"]])
        for k, v in frame.get("angles", {}).items():
            all_angles.setdefault(k, []).append(v)

    avg_landmarks = {
        k: {"x": float(np.mean([p[0] for p in pts])),
            "y": float(np.mean([p[1] for p in pts]))}
        for k, pts in all_landmarks.items()
    }
    avg_angles = {k: float(np.mean(v)) for k, v in all_angles.items() if v}

    # ---- Geometric fallback: angle at vertex B in triangle A-B-C ----
    def angle_from_pts(a, b, c):
        if not all(n in avg_landmarks for n in (a, b, c)):
            return None
        A, B, C = avg_landmarks[a], avg_landmarks[b], avg_landmarks[c]
        ba = (A["x"]-B["x"], A["y"]-B["y"])
        bc = (C["x"]-B["x"], C["y"]-B["y"])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.sqrt(ba[0]**2+ba[1]**2) * math.sqrt(bc[0]**2+bc[1]**2)
        return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag)))) if mag > 1e-9 else None

    TRIPLETS = {
        "left_shoulder":  ("left_elbow",    "left_shoulder",  "left_hip"),
        "right_shoulder": ("right_elbow",   "right_shoulder", "right_hip"),
        "left_elbow":     ("left_shoulder", "left_elbow",     "left_wrist"),
        "right_elbow":    ("right_shoulder","right_elbow",    "right_wrist"),
        "left_hip":       ("left_shoulder", "left_hip",       "left_knee"),
        "right_hip":      ("right_shoulder","right_hip",      "right_knee"),
        "left_knee":      ("left_hip",      "left_knee",      "left_ankle"),
        "right_knee":     ("right_hip",     "right_knee",     "right_ankle"),
    }

    def resolve_angle(jname):
        # Try every likely key variant the pose system might use
        for key in (f"{jname}_angle", jname, f"angle_{jname}", jname.replace("_","")):
            if key in avg_angles:
                return avg_angles[key]
        # Geometric fallback from landmark positions
        t = TRIPLETS.get(jname)
        return angle_from_pts(*t) if t else None

    # ---- Ideal ranges ----
    IDEAL_RANGES = {
        "default": {
            "left_elbow_angle":(150,180),"right_elbow_angle":(150,180),
            "left_knee_angle":(140,175),"right_knee_angle":(140,175),
            "left_hip_angle":(160,180),"right_hip_angle":(160,180),
            "left_shoulder_angle":(80,120),"right_shoulder_angle":(80,120),
        },
        "flick": {
            "left_elbow_angle":(80,140),"right_elbow_angle":(100,160),
            "left_knee_angle":(130,165),"right_knee_angle":(140,175),
            "left_hip_angle":(150,175),"right_hip_angle":(150,175),
            "left_shoulder_angle":(60,100),"right_shoulder_angle":(70,110),
        },
        "cover": {
            "left_elbow_angle":(140,175),"right_elbow_angle":(100,150),
            "left_knee_angle":(140,165),"right_knee_angle":(150,175),
            "left_hip_angle":(155,175),"right_hip_angle":(155,175),
            "left_shoulder_angle":(75,110),"right_shoulder_angle":(75,115),
        },
        "defense": {
            "left_elbow_angle":(150,180),"right_elbow_angle":(150,180),
            "left_knee_angle":(145,175),"right_knee_angle":(145,175),
            "left_hip_angle":(160,180),"right_hip_angle":(160,180),
            "left_shoulder_angle":(85,120),"right_shoulder_angle":(85,120),
        },
        "pull": {
            "left_elbow_angle":(70,130),"right_elbow_angle":(70,130),
            "left_knee_angle":(120,160),"right_knee_angle":(120,160),
            "left_hip_angle":(140,170),"right_hip_angle":(140,170),
            "left_shoulder_angle":(50,90),"right_shoulder_angle":(50,90),
        },
        "sweep": {
            "left_elbow_angle":(110,155),"right_elbow_angle":(110,155),
            "left_knee_angle":(100,145),"right_knee_angle":(110,150),
            "left_hip_angle":(130,165),"right_hip_angle":(130,165),
            "left_shoulder_angle":(60,100),"right_shoulder_angle":(60,100),
        },
    }

    JOINT_TIPS = {
        "default": {
            "left_elbow":"Keep elbows relaxed and close to the body for bat control.",
            "right_elbow":"Drive through with the right elbow leading the bat path.",
            "left_knee":"Bend the front knee to stay balanced over the ball.",
            "right_knee":"Flex the back knee slightly for a stable base.",
            "left_hip":"Stay upright at the hips for a clean, straight bat swing.",
            "right_hip":"Rotate the back hip through impact for power.",
            "left_shoulder":"Keep the front shoulder pointed at the bowler.",
            "right_shoulder":"Bring the back shoulder through the line of the ball.",
        },
        "flick": {
            "left_elbow":"Bend the front elbow early - this powers the flick.",
            "right_elbow":"The right elbow drives the wrist rotation; keep it flexible.",
            "left_knee":"A firm front leg creates the lever for the flick.",
            "right_knee":"Push off the back knee to transfer weight forward.",
            "left_hip":"Stable left hip is the pivot point for the wrist flick.",
            "right_hip":"Rotate the right hip through to add pace to the stroke.",
            "left_shoulder":"Lock the front shoulder - do not let it fly open early.",
            "right_shoulder":"Follow through with the right shoulder for full extension.",
        },
        "cover": {
            "left_elbow":"Extend the front elbow through the line of the ball.",
            "right_elbow":"Keep the right elbow tucked to avoid an open face.",
            "left_knee":"Drive off a bent front knee toward the pitch of the ball.",
            "right_knee":"Back knee low - this helps transfer weight forward.",
            "left_hip":"Lead with the front hip toward the ball line.",
            "right_hip":"Let the back hip open naturally as weight shifts forward.",
            "left_shoulder":"Lead with the front shoulder pointing cover-wards.",
            "right_shoulder":"Follow through: right shoulder should end over the left foot.",
        },
        "defense": {
            "left_elbow":"Keep elbows high and close to guide the ball down safely.",
            "right_elbow":"Right elbow up and in - prevents bat from angling away.",
            "left_knee":"Soft bend in the front knee to absorb pace.",
            "right_knee":"Slight flex in the back knee - avoid locking out.",
            "left_hip":"Stay tall at the hips; collapsing causes inside edges.",
            "right_hip":"Do not rotate the back hip - stay sideways for defense.",
            "left_shoulder":"Front shoulder stays high and closed to keep bat straight.",
            "right_shoulder":"Resist the urge to open the right shoulder early.",
        },
        "pull": {
            "left_elbow":"Front elbow bent sharply to swing across the line.",
            "right_elbow":"Right elbow drives down through the ball to keep it low.",
            "left_knee":"Stay low - bend both knees to get under the ball.",
            "right_knee":"Deep back knee flex creates the coiled power for the pull.",
            "left_hip":"Pivot the left hip early to make room for the pull swing.",
            "right_hip":"Explosive right hip rotation generates pull-shot power.",
            "left_shoulder":"Front shoulder drops slightly to get under the short ball.",
            "right_shoulder":"Roll the right shoulder over to keep the ball on the ground.",
        },
        "sweep": {
            "left_elbow":"Front elbow leads low - sweeping motion stays close to the pad.",
            "right_elbow":"Right elbow drops to guide the bat across the line.",
            "left_knee":"Front knee down - the hallmark of a good sweep.",
            "right_knee":"Back knee nearly touching the ground for a balanced sweep.",
            "left_hip":"Stay low at the hips; rising up causes mistimed sweeps.",
            "right_hip":"Rotate the back hip to sweep square or fine.",
            "left_shoulder":"Front shoulder aims at the leg side target.",
            "right_shoulder":"Full shoulder rotation completes the sweep follow-through.",
        },
    }

    GENERAL_TIPS = {
        "default":"Maintain balance and smooth swing throughout the shot.",
        "defense":"Keep bat close to pad, head over the ball for control.",
        "flick":"Use wrist rotation and maintain a firm front-leg base.",
        "cover":"Lead with the front shoulder and transfer weight forward.",
        "pull":"Stay low, roll wrists to keep the ball on the ground.",
        "sweep":"Stay balanced and use the front knee for stability.",
    }

    shot_key = shot_type.lower().replace(" ","_").replace("-","_")
    if shot_key not in IDEAL_RANGES:
        shot_key = "default"

    ranges      = IDEAL_RANGES[shot_key]
    joint_tips  = JOINT_TIPS.get(shot_key, JOINT_TIPS["default"])
    general_tip = GENERAL_TIPS.get(shot_key, GENERAL_TIPS["default"])

    def joint_status(akey, val):
        if akey not in ranges:
            return "Tracked", "#58a6ff", "ref"
        lo, hi = ranges[akey]
        margin = (hi-lo)*0.3
        if lo <= val <= hi:            return "Ideal",       "#3fb950", "ideal"
        if lo-margin <= val <= hi+margin: return "Slightly off","#d29922", "warn"
        return "Needs work", "#f85149", "bad"

    connections = [
        ("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),
        ("left_elbow","left_wrist"),("right_shoulder","right_elbow"),
        ("right_elbow","right_wrist"),("left_shoulder","left_hip"),
        ("right_shoulder","right_hip"),("left_hip","right_hip"),
        ("left_hip","left_knee"),("left_knee","left_ankle"),
        ("right_hip","right_knee"),("right_knee","right_ankle"),
    ]

    xs = [v["x"] for v in avg_landmarks.values()]
    ys = [v["y"] for v in avg_landmarks.values()]
    minx,maxx = min(xs),max(xs)
    miny,maxy = min(ys),max(ys)

    def norm(vx, vy, W=280, H=380, pad=24):
        sx = (vx-minx)/(maxx-minx+1e-6)
        sy = (vy-miny)/(maxy-miny+1e-6)
        return round(pad+sx*(W-2*pad),1), round(pad+sy*(H-2*pad),1)

    TRACKED = ["left_shoulder","right_shoulder","left_elbow","right_elbow",
               "left_hip","right_hip","left_knee","right_knee"]

    joint_data = {}
    for jname in TRACKED:
        if jname not in avg_landmarks:
            continue
        lm = avg_landmarks[jname]
        cx, cy = norm(lm["x"], lm["y"])
        val = resolve_angle(jname)
        akey = f"{jname}_angle"
        if val is not None:
            status, color, skey = joint_status(akey, val)
            lo_hi = ranges.get(akey)
            ideal_str = f"{lo_hi[0]}-{lo_hi[1]}" if lo_hi else ""
        else:
            status, color, skey = "Tracked", "#58a6ff", "ref"
            ideal_str = ""
        joint_data[jname] = {
            "cx": cx, "cy": cy, "color": color,
            "status": status, "skey": skey,
            "angle": f"{val:.1f}" if val is not None else "",
            "ideal": ideal_str,
            "tip": joint_tips.get(jname, "Focus on correct form for this joint."),
            "label": jname.replace("_"," ").title(),
        }

    # Build SVG lines
    lines_svg = ""
    for a, b in connections:
        if a in avg_landmarks and b in avg_landmarks:
            x1,y1 = norm(avg_landmarks[a]["x"], avg_landmarks[a]["y"])
            x2,y2 = norm(avg_landmarks[b]["x"], avg_landmarks[b]["y"])
            lines_svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2a3347" stroke-width="3" stroke-linecap="round"/>'

    # Build SVG circles — id uses underscores so JS lookup is simple
    circles_svg = ""
    for jname, jd in joint_data.items():
        circles_svg += (
            f'<circle id="jc_{jname}" cx="{jd["cx"]}" cy="{jd["cy"]}" r="8" '
            f'fill="{jd["color"]}" stroke="#0d1117" stroke-width="2" '
            f'style="cursor:pointer;"/>'
        )
    for lname in ["left_wrist","right_wrist","left_ankle","right_ankle"]:
        if lname in avg_landmarks and lname not in joint_data:
            lm = avg_landmarks[lname]
            cx,cy = norm(lm["x"],lm["y"])
            circles_svg += f'<circle cx="{cx}" cy="{cy}" r="5" fill="#3a4459" stroke="#0d1117" stroke-width="1.5"/>'

    # Build table rows
    table_rows = ""
    for jname in TRACKED:
        if jname not in joint_data:
            continue
        jd = joint_data[jname]
        bbg = {"ideal":"#1a2f1c","warn":"#2e2105","bad":"#2f1117"}.get(jd["skey"],"#1a1f2e")
        angle_disp = f'{jd["angle"]}&deg;' if jd["angle"] else "N/A"
        ideal_disp = f'{jd["ideal"]}&deg;' if jd["ideal"] else "N/A"
        table_rows += (
            f'<tr id="tr_{jname}" style="cursor:pointer;border-bottom:1px solid #1e2535;">'
            f'<td style="padding:8px 10px;color:#8b949e;font-size:12px;white-space:nowrap;">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{jd["color"]};margin-right:7px;vertical-align:middle;"></span>'
            f'{jd["label"]}</td>'
            f'<td style="padding:8px 6px;color:{jd["color"]};font-weight:600;font-size:13px;">{angle_disp}</td>'
            f'<td style="padding:8px 6px;font-size:11px;color:#8b949e;">{ideal_disp}</td>'
            f'<td style="padding:8px 6px;">'
            f'<span style="font-size:10px;padding:2px 7px;border-radius:4px;'
            f'background:{bbg};color:{jd["color"]};font-weight:600;">{jd["status"]}</span>'
            f'</td></tr>'
        )

    # Build JS data as a plain object literal (no json.dumps to avoid unicode escapes)
    # Each entry: jname -> {color, status, skey, angle, ideal, tip, label, cx, cy}
    js_entries = []
    for jname, jd in joint_data.items():
        # Escape single quotes in tip
        tip_safe = jd["tip"].replace("'", "\\'")
        label_safe = jd["label"].replace("'", "\\'")
        js_entries.append(
            f"'{jname}':{{"
            f"color:'{jd['color']}',"
            f"status:'{jd['status']}',"
            f"skey:'{jd['skey']}',"
            f"angle:'{jd['angle']}',"
            f"ideal:'{jd['ideal']}',"
            f"tip:'{tip_safe}',"
            f"label:'{label_safe}',"
            f"cx:'{jd['cx']}',"
            f"cy:'{jd['cy']}'"
            f"}}"
        )
    js_data = "{" + ",".join(js_entries) + "}"

    shot_display = shot_type if shot_type and shot_type.lower() != "none" else "General"
    frame_count  = len(frames)

    # ---- HTML + inline JS (no f-string braces in JS, use string concat in JS instead) ----
    html = """
<div id="cpw" style="font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#e6edf3;border-radius:12px;overflow:hidden;border:1px solid #1e2535;">

  <div style="background:#161b22;border-bottom:1px solid #1e2535;padding:12px 18px;display:flex;align-items:center;justify-content:space-between;">
    <div>
      <span style="font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#58a6ff;font-weight:600;">Cricket Pose Analysis</span>
      <span style="margin-left:10px;font-size:11px;background:#1f2d3d;color:#58a6ff;padding:2px 9px;border-radius:20px;border:1px solid #2a4060;">""" + shot_display + """</span>
    </div>
    <div style="font-size:11px;color:#484f58;">""" + str(frame_count) + """ frame""" + ("s" if frame_count != 1 else "") + """ averaged</div>
  </div>

  <div style="display:flex;">

    <div style="flex:0 0 auto;background:#0d1117;border-right:1px solid #1e2535;padding:16px;display:flex;flex-direction:column;align-items:center;">
      <svg id="cpw_svg" viewBox="0 0 328 428" width="230" height="310" style="background:#0d1117;border-radius:8px;display:block;">
        """ + lines_svg + circles_svg + """
        <circle id="cpw_ring" cx="-999" cy="-999" r="15" fill="none" stroke="#fff" stroke-width="2.5" stroke-dasharray="5 3" opacity="0"/>
      </svg>
      <p style="font-size:10px;color:#484f58;margin:8px 0 0;text-align:center;">Click any joint to inspect</p>
      <div style="display:flex;gap:12px;margin-top:8px;">
        <span style="font-size:10px;color:#3fb950;">&#9679; Ideal</span>
        <span style="font-size:10px;color:#d29922;">&#9679; Slightly off</span>
        <span style="font-size:10px;color:#f85149;">&#9679; Needs work</span>
      </div>
    </div>

    <div style="flex:1;min-width:0;display:flex;flex-direction:column;">
      <div id="cpw_detail" style="padding:16px 18px;background:#161b22;border-bottom:1px solid #1e2535;min-height:145px;display:flex;align-items:center;justify-content:center;">
        <div style="color:#484f58;font-size:13px;text-align:center;line-height:1.8;">Select a joint on the figure<br>or in the table below</div>
      </div>

      <div>
        <table id="cpw_table" style="width:100%;border-collapse:collapse;font-size:12px;">
          <thead>
            <tr style="background:#161b22;border-bottom:1px solid #1e2535;">
              <th style="padding:7px 10px;text-align:left;color:#484f58;font-weight:500;font-size:11px;">JOINT</th>
              <th style="padding:7px 6px;text-align:left;color:#484f58;font-weight:500;font-size:11px;">ANGLE</th>
              <th style="padding:7px 6px;text-align:left;color:#484f58;font-weight:500;font-size:11px;">IDEAL</th>
              <th style="padding:7px 6px;text-align:left;color:#484f58;font-weight:500;font-size:11px;">STATUS</th>
            </tr>
          </thead>
          <tbody>""" + table_rows + """</tbody>
        </table>
      </div>

      <div style="padding:12px 18px;background:#161b22;border-top:1px solid #1e2535;margin-top:auto;">
        <p style="font-size:10px;color:#58a6ff;margin:0 0 4px;letter-spacing:1px;text-transform:uppercase;font-weight:600;">General coaching tip</p>
        <p style="font-size:12px;line-height:1.6;color:#8b949e;margin:0;">""" + general_tip + """</p>
      </div>
    </div>
  </div>
</div>

<script>
var CPW_DATA = """ + js_data + """;
var CPW_CUR  = null;

function cpwCircle(n) { return document.getElementById('jc_' + n); }
function cpwRow(n)    { return document.getElementById('tr_' + n); }

function cpwDesel() {
  if (!CPW_CUR) return;
  var el = cpwCircle(CPW_CUR);
  if (el) { el.setAttribute('r','8'); el.setAttribute('stroke','#0d1117'); el.setAttribute('stroke-width','2'); }
  var row = cpwRow(CPW_CUR);
  if (row) row.style.background = '';
  var ring = document.getElementById('cpw_ring');
  if (ring) { ring.setAttribute('opacity','0'); ring.setAttribute('cx','-999'); ring.setAttribute('cy','-999'); }
  CPW_CUR = null;
}

function cpwSel(name) {
  if (CPW_CUR === name) { cpwDesel(); cpwReset(); return; }
  cpwDesel();
  CPW_CUR = name;
  var jd = CPW_DATA[name];
  if (!jd) return;

  var el = cpwCircle(name);
  if (el) {
    el.setAttribute('r','13');
    el.setAttribute('stroke', jd.color);
    el.setAttribute('stroke-width','3');
  }

  var ring = document.getElementById('cpw_ring');
  if (ring) {
    ring.setAttribute('cx', jd.cx);
    ring.setAttribute('cy', jd.cy);
    ring.setAttribute('stroke', jd.color);
    ring.setAttribute('opacity','1');
  }

  var row = cpwRow(name);
  if (row) row.style.background = '#1c2333';

  var bbg = jd.skey === 'ideal' ? '#1a2f1c' : jd.skey === 'warn' ? '#2e2105' : jd.skey === 'bad' ? '#2f1117' : '#1a1f2e';
  var aTxt = jd.angle ? (jd.angle + '&#176;') : '&mdash;';
  var iTxt = jd.ideal ? (jd.ideal + '&#176;') : '&mdash;';

  var detail = document.getElementById('cpw_detail');
  if (!detail) return;
  detail.innerHTML =
    '<div style="display:flex;align-items:flex-start;gap:14px;width:100%;">' +
      '<div style="flex:0 0 44px;">' +
        '<div style="width:44px;height:44px;border-radius:50%;background:' + bbg + ';border:2px solid ' + jd.color + ';display:flex;align-items:center;justify-content:center;">' +
          '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="' + jd.color + '" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2"/></svg>' +
        '</div>' +
      '</div>' +
      '<div style="flex:1;min-width:0;">' +
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">' +
          '<span style="font-size:14px;font-weight:600;color:#e6edf3;">' + jd.label + '</span>' +
          '<span style="font-size:10px;padding:2px 8px;border-radius:4px;background:' + bbg + ';color:' + jd.color + ';font-weight:600;">' + jd.status + '</span>' +
        '</div>' +
        '<div style="display:flex;gap:20px;margin-bottom:10px;">' +
          '<div><div style="font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px;">Detected</div><div style="font-size:26px;font-weight:700;color:' + jd.color + ';line-height:1;">' + aTxt + '</div></div>' +
          '<div style="border-left:1px solid #1e2535;padding-left:20px;"><div style="font-size:10px;color:#484f58;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px;">Ideal range</div><div style="font-size:26px;font-weight:700;color:#484f58;line-height:1;">' + iTxt + '</div></div>' +
        '</div>' +
        '<div style="background:#0d1117;border-radius:6px;padding:9px 12px;border-left:3px solid ' + jd.color + ';">' +
          '<p style="font-size:10px;color:#58a6ff;margin:0 0 3px;letter-spacing:0.8px;text-transform:uppercase;font-weight:600;">Coaching tip</p>' +
          '<p style="font-size:12px;color:#c9d1d9;margin:0;line-height:1.6;">' + jd.tip + '</p>' +
        '</div>' +
      '</div>' +
    '</div>';
}

function cpwReset() {
  var d = document.getElementById('cpw_detail');
  if (d) d.innerHTML = '<div style="color:#484f58;font-size:13px;text-align:center;line-height:1.8;">Select a joint on the figure<br>or in the table below</div>';
}

function cpwInit() {
  // Wire SVG circles individually — most robust approach for sandboxed iframes
  var jnames = Object.keys(CPW_DATA);
  for (var i = 0; i < jnames.length; i++) {
    (function(n) {
      var el = cpwCircle(n);
      if (el) el.onclick = function(e) { e.stopPropagation(); cpwSel(n); };
    })(jnames[i]);
  }

  // Wire table rows individually
  for (var j = 0; j < jnames.length; j++) {
    (function(n) {
      var row = cpwRow(n);
      if (row) row.onclick = function() { cpwSel(n); };
    })(jnames[j]);
  }

  // Click on SVG background deselects
  var svg = document.getElementById('cpw_svg');
  if (svg) svg.onclick = function(e) {
    if (e.target === svg || e.target.tagName === 'line') { cpwDesel(); cpwReset(); }
  };
}

// Run immediately AND after a short delay (handles Gradio async render)
cpwInit();
setTimeout(cpwInit, 300);
setTimeout(cpwInit, 800);
</script>
"""
    import base64
    encoded  = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    data_url = f"data:text/html;base64,{encoded}"
    return f'<iframe src="{data_url}" width="100%" height="620px" style="border:none; border-radius:12px;"></iframe>'

# ── MAIN segment_player (original features preserved + stick figure + widget added) ──
def segment_player(video_path, click_coords, shot_type=None, progress=gr.Progress()):
    import traceback
    try:
        if not video_path or not click_coords:
            return None, None, None, None, "<p style='color:gray'>No data.</p>", "No shot selected.", *([None] * 6)

        if not predictor:
            return None, None, None, None, "<p style='color:gray'>SAM2 not loaded.</p>", "SAM2 not loaded.", *([None] * 6)

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

            video_segments  = {}
            tracked_indices = []
            print("Running SAM2 Propagation...")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                progress(out_frame_idx / 100, desc=f"Propagating SAM2 (Frame {out_frame_idx})...")
                frame_masks = {}
                is_tracked  = False
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
                return None, None, None, None, "<p style='color:gray'>Could not track batsman.</p>", "Could not track the batsman.", *([None] * 6)

            start_idx = min(tracked_indices)
            end_idx   = max(tracked_indices)
            print(f"🏏 Autoclipping: Batsman tracked from frame {start_idx} to {end_idx}")

        out_video_path = os.path.join(PROJECT_ROOT, "outputs/output_segmented.mp4")
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = imageio.get_writer(out_video_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

        idx            = 0
        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_idx <= idx <= end_idx:
                progress(frames_written / max(1, (end_idx - start_idx + 1)), desc=f"Rendering Isolated Player (Frame {idx}/{end_idx})...")
                isolated_frame = np.zeros_like(frame)
                if idx in video_segments and 1 in video_segments[idx]:
                    mask = video_segments[idx][1]
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_indices = mask > 0
                    isolated_frame[mask_indices] = frame[mask_indices]
                writer.append_data(cv2.cvtColor(isolated_frame, cv2.COLOR_BGR2RGB))
                frames_written += 1
            idx += 1
        cap.release()
        writer.close()

        # ── Biomechanics on segmented video (original) ──
        bio_json_path  = os.path.join(PROJECT_ROOT, "outputs/segmented_biomechanics.json")
        pose_json_path = os.path.join(PROJECT_ROOT, "outputs/pose_data.json")

        if extractor:
            def bio_progress(curr, total, msg):
                progress(curr / max(1, total), desc=f"Biomechanics: {msg}")
            pose_data_extracted = extractor.extract_from_video(out_video_path, progress_callback=bio_progress)

            angle_results = []
            pose_results  = []
            for frame in pose_data_extracted["frames"]:
                if "angles" in frame:
                    angle_results.append({"frame": frame["frame_idx"], "time": frame["time_sec"], **frame["angles"]})
                if "landmarks" in frame and len(frame.get("landmarks", {})) > 0 and compute_joint_angles:
                    angles = compute_joint_angles(frame["landmarks"])
                    pose_results.append({
                        "frame":     frame["frame_idx"],
                        "landmarks": frame["landmarks"],
                        "angles":    angles
                    })

            with open(bio_json_path, "w") as f:
                json.dump(angle_results, f, indent=4)
            with open(pose_json_path, "w") as f:
                json.dump(pose_results, f, indent=4)

        # ── Sync & comparison (original) ──
        sync_comparison_path = None
        score_feedback       = "No shot selected for comparison."
        plots                = [None] * 6

        if shot_type and shot_type != "None" and sync_engine:
            result = sync_and_compare(out_video_path, shot_type, progress=progress)
            if result:
                sync_comparison_path, score_feedback, plots = result

        # ── NEW: Stick figure (guaranteed correct player) ──
        print("Generating stick figure video...")
        stick_figure_path = create_stick_figure_video(video_path, segmented_video_path=out_video_path, progress=progress)

        # ── NEW: Interactive widget ──
        interactive_widget = generate_interactive_widget(pose_json_path, shot_type)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return out_video_path, stick_figure_path, interactive_widget, bio_json_path, sync_comparison_path, score_feedback, *plots

    except Exception as e:
        print(f"Error in segment_player: {e}")
        traceback.print_exc()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return None, None, "<p style='color:red'>Error during analysis.</p>", None, None, "An error occurred.", *([None] * 6)


# --- Shot Sync Flow (original — unchanged) ---
def load_references():
    if os.path.exists(REFERENCES_DB_PATH):
        with open(REFERENCES_DB_PATH, "r") as f:
            refs = json.load(f)
            for key, val in refs.items():
                base = os.path.basename(val["video_path"])
                prefix = base.split("_reference.mp4")[0]
                val["stats_path"] = f"assets/references/{prefix}_stats.json"
                
            return refs
    return {}
# print(f"DEBUG sync: shot_type='{shot_type}' | refs keys={list(refs.keys())}")
def sync_and_compare(practice_video, shot_type, progress=gr.Progress()):
    if not practice_video or not shot_type or not sync_engine:
        return None, "Setup missing", [None] * 6
    refs = load_references()
    # Try exact match first, then case-insensitive match
    if shot_type not in refs:
        shot_type_lower = shot_type.lower()
        matched_key = next((k for k in refs if k.lower() == shot_type_lower), None)
        if not matched_key:
            print(f"Shot type '{shot_type}' not found in refs. Available: {list(refs.keys())}")
            return None, "Shot not found", [None] * 6
        shot_type = matched_key
        print(f"Matched shot type: {shot_type}")

    ref_info   = refs[shot_type]
    ref_video  = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["video_path"]))
    stats_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("stats_path", "")))

    if not os.path.exists(ref_video):
        return None, "Visual reference not found", [None] * 6
    if not os.path.exists(stats_path):
        return None, "Stats reference not found", [None] * 6

    def progress_cb(current, total, desc):
        progress((current / total) * 0.4, desc=desc)

    practice_data = extractor.extract_from_video(practice_video, progress_callback=progress_cb)

    with open(stats_path, "r") as f:
        stats_data = json.load(f)

    ref_angles_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("angles_path", "")))
    if os.path.exists(ref_angles_path):
        with open(ref_angles_path, "r") as f:
            reference_data = json.load(f)
    else:
        reference_data = {"frames": []}
        for frame_stat in stats_data["frames"]:
            reference_data["frames"].append({
                "frame_idx": frame_stat["frame_idx"],
                "time_sec":  frame_stat["time_sec"],
                "angles":    frame_stat.get("mean_angles", {})
            })

    alignment_path, p_phases, r_phases = sync_engine.sync_videos(practice_data, reference_data)

    GREEN = "\033[92m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{BOLD}Analysis for Shot Type: {GREEN}{shot_type}{RESET}")
    if p_phases and r_phases:
        print(f"{GREEN}Frame Generalization Data:{RESET}")
        print(f"  - Practice Strike: {GREEN}Frame {p_phases['strike']}{RESET}")
        print(f"  - Reference Strike: {GREEN}Frame {r_phases['strike']}{RESET}\n")

    mapping   = {p: r for p, r in alignment_path}
    meta_path = os.path.join(PROJECT_ROOT, "outputs/sync_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"shot_type": shot_type, "practice_phases": p_phases, "reference_phases": r_phases}, f, indent=4)

    joint_weights = {
        "left_elbow": 1.5, "right_elbow": 1.5,
        "left_shoulder": 1.2, "right_shoulder": 1.2,
        "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 0.8, "right_knee": 0.8
    }

    total_score  = 0
    total_weight = 0

    for p_idx, r_idx in mapping.items():
        if p_idx >= len(practice_data["frames"]) or r_idx >= len(stats_data["frames"]):
            continue
        p_angles   = practice_data["frames"][p_idx].get("angles", {})
        stat_frame = stats_data["frames"][r_idx]
        q1_a       = stat_frame.get("q1_angles", {})
        q3_a       = stat_frame.get("q3_angles", {})
        min_a_ref  = stat_frame.get("min_angles", {})
        max_a_ref  = stat_frame.get("max_angles", {})

        for joint, weight in joint_weights.items():
            if joint in p_angles and joint in q1_a and joint in q3_a:
                val  = p_angles[joint]
                q1, q3 = q1_a[joint], q3_a[joint]
                mn, mx = min_a_ref.get(joint, q1 - 10), max_a_ref.get(joint, q3 + 10)
                s = 100 if q1 <= val <= q3 else (
                    100 * (val - mn) / (q1 - mn) if val < q1 and mn != q1 else
                    100 * (mx - val) / (mx - q3) if val > q3 and mx != q3 else 0
                )
                total_score  += max(0, s) * weight
                total_weight += weight

    final_percentage = (total_score / total_weight) if total_weight > 0 else 0
    feedback_str     = f"### 📊 Overall Accuracy Score: {final_percentage:.1f}%\n"
    feedback_str    += "Biomechanical sync complete. Ready for external LLM evaluation."

    plots = []
    try:
        joints_to_plot = ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip"]
        for joint in joints_to_plot:
            p_vals, q1_vals, q3_vals, mean_vals = [], [], [], []
            for p_idx, r_idx in mapping.items():
                if p_idx < len(practice_data["frames"]) and r_idx < len(stats_data["frames"]):
                    p_vals.append(practice_data["frames"][p_idx].get("angles", {}).get(joint, 0))
                    q1_vals.append(stats_data["frames"][r_idx].get("q1_angles", {}).get(joint, 0))
                    q3_vals.append(stats_data["frames"][r_idx].get("q3_angles", {}).get(joint, 0))
                    mean_vals.append(stats_data["frames"][r_idx].get("mean_angles", {}).get(joint, 0))
            ref_stats = {"q1": q1_vals, "q3": q3_vals, "mean": mean_vals}
            fig = generate_biomechanic_plot(joint, p_vals, ref_stats, p_phases)
            plots.append(fig)
    except Exception as e:
        print(f"Error generating plots: {e}")
        if not plots:
            plots = [None] * 6

    out_video = create_synced_video(practice_video, ref_video, alignment_path, progress=progress)
    return out_video, feedback_str, plots


def create_synced_video(practice_video, reference_video, alignment_path, progress=None):
    cap_p    = cv2.VideoCapture(practice_video)
    cap_r    = cv2.VideoCapture(reference_video)
    fps      = cap_p.get(cv2.CAP_PROP_FPS) or 30.0
    slow_fps = fps * 0.3

    target_h = 480
    w_p, h_p = int(cap_p.get(3)), int(cap_p.get(4))
    w_r, h_r = int(cap_r.get(3)), int(cap_r.get(4))
    scale_p  = target_h / h_p
    scale_r  = target_h / h_r
    new_w_p  = (int(w_p * scale_p) // 2) * 2
    new_w_r  = (int(w_r * scale_r) // 2) * 2

    out_path = os.path.join(PROJECT_ROOT, "outputs/synced_comparison.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=slow_fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)

    ref_frames = []
    while True:
        ret, f = cap_r.read()
        if not ret:
            break
        ref_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_r, target_h)))
    cap_r.release()

    p_frames = []
    while True:
        ret, f = cap_p.read()
        if not ret:
            break
        p_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_p, target_h)))
    cap_p.release()

    total_steps = len(alignment_path)
    for step_idx, (p_idx, r_idx) in enumerate(alignment_path):
        if p_idx >= len(p_frames):   p_idx = len(p_frames) - 1
        if r_idx >= len(ref_frames): r_idx = len(ref_frames) - 1
        frame_p  = p_frames[p_idx]
        frame_r  = ref_frames[r_idx]
        combined = np.hstack((frame_p, frame_r))
        cv2.putText(combined, "PRACTICE",  (10, 30),           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "REFERENCE", (new_w_p + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        writer.append_data(combined)
        if progress:
            progress(0.6 + (step_idx / total_steps) * 0.4, desc="Rendering Comparison...")

    writer.close()
    return out_path


# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏏 AthletiQ - Unified Performance Pipeline")

    with gr.Tab("Performance Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input         = gr.Video(label="1. Upload Practice Video")
                shot_select         = gr.Dropdown(choices=["None"] + list(load_references().keys()), value="None", label="2. Select Shot Type")
                first_frame_display = gr.Image(label="3. Click on Batsman to Segment", interactive=False)
                analyze_btn         = gr.Button("🚀 Run Full Shot Analysis", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                out_score     = gr.Markdown("Score will appear here.")
                # ── NEW: side by side isolated + stick figure ──
                with gr.Row():
                    out_isolated = gr.Video(label="Isolated Player (Cutout)")
                    out_stick    = gr.Video(label="Stick Figure (Pose)")
                # ── NEW: interactive widget ──
                out_interactive = gr.HTML(label="Interactive Pose Analysis", sanitize_html=False)
                out_comparison  = gr.Video(label="Technical Comparison (Side-by-Side)")
                out_json        = gr.File(label="Joint Angle Data (JSON)")

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

        clean_img_state      = gr.State()
        click_coord_state    = gr.State()
        playable_video_state = gr.State()

        video_input.upload(
            process_video_for_seg,
            video_input,
            [first_frame_display, clean_img_state, playable_video_state, shot_select]
        )

        def handle_point_selection(img, evt: gr.SelectData):
            points_img = img.copy()
            cv2.circle(points_img, evt.index, 7, (0, 255, 0), -1)
            return evt.index, points_img

        first_frame_display.select(
            handle_point_selection,
            clean_img_state,
            [click_coord_state, first_frame_display]
        )

        analyze_btn.click(
            segment_player,
            inputs=[playable_video_state, click_coord_state, shot_select],
            outputs=[
                out_isolated, out_stick, out_interactive, out_json, out_comparison,
                out_score,
                plot_l_elbow, plot_r_elbow, plot_l_knee, plot_r_knee, plot_l_hip, plot_r_hip
            ]
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), allowed_paths=[os.path.join(PROJECT_ROOT, "outputs")])