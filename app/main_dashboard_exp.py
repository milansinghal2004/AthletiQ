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
    if not pose_json_path or not os.path.exists(pose_json_path):
        return "<p style='color:gray'>No pose data available.</p>"

    with open(pose_json_path, "r") as f:
        widget_data = json.load(f)

    if not widget_data:
        return "<p style='color:gray'>No pose data available.</p>"

    print(f"Widget: {len(widget_data)} frames loaded")

    IDEAL_RANGES = {
        "default":    {"left_elbow_angle":(150,180),"right_elbow_angle":(150,180),"left_knee_angle":(140,175),"right_knee_angle":(140,175),"left_hip_angle":(160,180),"right_hip_angle":(160,180)},
        "flick":      {"left_elbow_angle":(80,140),"right_elbow_angle":(100,160),"left_knee_angle":(130,165),"right_knee_angle":(140,175),"left_hip_angle":(150,175),"right_hip_angle":(150,175)},
        "cover":      {"left_elbow_angle":(140,175),"right_elbow_angle":(100,150),"left_knee_angle":(140,165),"right_knee_angle":(150,175),"left_hip_angle":(155,175),"right_hip_angle":(155,175)},
        "defense":    {"left_elbow_angle":(150,180),"right_elbow_angle":(150,180),"left_knee_angle":(145,175),"right_knee_angle":(145,175),"left_hip_angle":(160,180),"right_hip_angle":(160,180)},
        "pull":       {"left_elbow_angle":(70,130),"right_elbow_angle":(70,130),"left_knee_angle":(120,160),"right_knee_angle":(120,160),"left_hip_angle":(140,170),"right_hip_angle":(140,170)},
        "sweep":      {"left_elbow_angle":(80,140),"right_elbow_angle":(80,140),"left_knee_angle":(80,120),"right_knee_angle":(130,165),"left_hip_angle":(130,165),"right_hip_angle":(140,175)},
        "hook":       {"left_elbow_angle":(70,120),"right_elbow_angle":(70,120),"left_knee_angle":(120,160),"right_knee_angle":(120,160),"left_hip_angle":(140,170),"right_hip_angle":(140,170)},
        "straight":   {"left_elbow_angle":(140,180),"right_elbow_angle":(100,150),"left_knee_angle":(140,170),"right_knee_angle":(150,175),"left_hip_angle":(155,180),"right_hip_angle":(155,180)},
        "square_cut": {"left_elbow_angle":(100,150),"right_elbow_angle":(100,150),"left_knee_angle":(130,165),"right_knee_angle":(130,165),"left_hip_angle":(145,175),"right_hip_angle":(145,175)},
        "late_cut":   {"left_elbow_angle":(100,155),"right_elbow_angle":(100,155),"left_knee_angle":(130,165),"right_knee_angle":(130,165),"left_hip_angle":(145,175),"right_hip_angle":(145,175)},
        "lofted":     {"left_elbow_angle":(140,180),"right_elbow_angle":(100,150),"left_knee_angle":(130,165),"right_knee_angle":(140,175),"left_hip_angle":(150,180),"right_hip_angle":(150,180)},
    }

    shot_key        = shot_type.lower().replace(" ","_").replace("_shot","").replace("_drive","") if shot_type and shot_type != "None" else "default"
    ranges          = IDEAL_RANGES.get(shot_key, IDEAL_RANGES["default"])
    pose_json_str   = json.dumps(widget_data)
    ranges_json_str = json.dumps(ranges)

    html_page = """<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d1117; color:white; font-family:Arial,sans-serif; padding:12px; overflow-x:hidden; }
  #wrap { display:flex; gap:12px; }
  #left { flex:1; min-width:0; }
  canvas { background:#161b22; border-radius:8px; border:1px solid #30363d; cursor:crosshair; display:block; width:100%; }
  #slider { width:100%; accent-color:#58a6ff; margin-top:8px; }
  #frameInfo { display:flex; justify-content:space-between; font-size:11px; color:#8b949e; margin-top:4px; }
  #right { width:210px; flex-shrink:0; }
  .panel { background:#161b22; border-radius:8px; padding:10px; border:1px solid #30363d; margin-bottom:8px; }
  .panel-title { font-size:11px; font-weight:bold; color:#8b949e; margin-bottom:8px; letter-spacing:0.5px; }
  .legend-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; flex-shrink:0; }
  .legend-item { display:flex; align-items:center; font-size:11px; color:#8b949e; margin-bottom:4px; }
  .angle-row { margin-bottom:8px; font-size:11px; }
  .angle-name { color:#c9d1d9; text-transform:capitalize; }
  .angle-val { font-weight:bold; float:right; }
  .angle-ideal { font-size:10px; color:#8b949e; clear:both; padding-top:1px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="left">
    <canvas id="c" width="420" height="500"></canvas>
    <input type="range" id="slider" min="0" value="0">
    <div id="frameInfo">
      <span>Frame 0</span>
      <span id="fcur">Frame 0</span>
      <span id="fmax">Frame 0</span>
    </div>
  </div>
  <div id="right">
    <div class="panel" id="clickPanel">
      <p style="color:#8b949e;font-size:12px;margin:0;">👆 Click a joint to inspect</p>
    </div>
    <div class="panel">
      <div class="panel-title">JOINT ANGLES</div>
      <div id="angleList"></div>
    </div>
    <div class="panel">
      <div class="panel-title">LEGEND</div>
      <div class="legend-item"><span class="legend-dot" style="background:#3fb950"></span>Ideal range</div>
      <div class="legend-item"><span class="legend-dot" style="background:#d29922"></span>Slightly off</div>
      <div class="legend-item"><span class="legend-dot" style="background:#f85149"></span>Needs work</div>
    </div>
  </div>
</div>
<script>
var DATA   = """ + pose_json_str + """;
var RANGES = """ + ranges_json_str + """;
var CONN = [
  ['left_shoulder','right_shoulder'],
  ['left_shoulder','left_elbow'],['left_elbow','left_wrist'],
  ['right_shoulder','right_elbow'],['right_elbow','right_wrist'],
  ['left_shoulder','left_hip'],['right_shoulder','right_hip'],
  ['left_hip','right_hip'],
  ['left_hip','left_knee'],['left_knee','left_ankle'],
  ['right_hip','right_knee'],['right_knee','right_ankle']
];
var ANGLE_MAP = [
  ['left_elbow_angle','left_elbow'],
  ['right_elbow_angle','right_elbow'],
  ['left_knee_angle','left_knee'],
  ['right_knee_angle','right_knee'],
  ['left_hip_angle','left_hip'],
  ['right_hip_angle','right_hip']
];
var canvas     = document.getElementById('c');
var ctx        = canvas.getContext('2d');
var slider     = document.getElementById('slider');
var fcur       = document.getElementById('fcur');
var fmax       = document.getElementById('fmax');
var angleList  = document.getElementById('angleList');
var clickPanel = document.getElementById('clickPanel');
var selectedJoint = null;
slider.max = DATA.length - 1;
fmax.textContent = 'Frame ' + (DATA.length - 1);
function getColor(key, val) {
  if (!RANGES[key]) return '#58a6ff';
  var lo = RANGES[key][0], hi = RANGES[key][1];
  var margin = (hi - lo) * 0.3;
  if (val >= lo && val <= hi) return '#3fb950';
  if (val >= lo - margin && val <= hi + margin) return '#d29922';
  return '#f85149';
}
function getStatus(key, val) {
  if (!RANGES[key]) return 'No reference';
  var lo = RANGES[key][0], hi = RANGES[key][1];
  var margin = (hi - lo) * 0.3;
  if (val >= lo && val <= hi) return 'Ideal';
  if (val >= lo - margin && val <= hi + margin) return 'Slightly off';
  return 'Needs work';
}
function scaleLandmarks(lm, W, H) {
  var xs = [], ys = [];
  for (var k in lm) { xs.push(lm[k].x); ys.push(lm[k].y); }
  var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
  var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
  var pad = 55;
  var rangeX = maxX - minX || 0.001;
  var rangeY = maxY - minY || 0.001;
  var scaleX = (W - pad * 2) / rangeX;
  var scaleY = (H - pad * 2) / rangeY;
  var scale  = Math.min(scaleX, scaleY);
  var offX   = (W - rangeX * scale) / 2 - minX * scale;
  var offY   = (H - rangeY * scale) / 2 - minY * scale;
  var result = {};
  for (var name in lm) {
    result[name] = { x: lm[name].x * scale + offX, y: lm[name].y * scale + offY };
  }
  return result;
}
function draw(idx) {
  var W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  var frame = DATA[idx];
  if (!frame || !frame.landmarks || Object.keys(frame.landmarks).length === 0) {
    ctx.fillStyle = '#8b949e'; ctx.font = '13px Arial'; ctx.textAlign = 'center';
    ctx.fillText('No pose detected for this frame', W / 2, H / 2);
    ctx.textAlign = 'left'; return;
  }
  var rawLm  = frame.landmarks;
  var lm     = scaleLandmarks(rawLm, W, H);
  var angles = frame.angles || {};
  var jColors = {};
  for (var i = 0; i < ANGLE_MAP.length; i++) {
    var ak = ANGLE_MAP[i][0], jn = ANGLE_MAP[i][1];
    if (angles[ak] != null) jColors[jn] = getColor(ak, angles[ak]);
  }
  for (var i = 0; i < CONN.length; i++) {
    var a = CONN[i][0], b = CONN[i][1];
    if (!lm[a] || !lm[b]) continue;
    ctx.beginPath(); ctx.moveTo(lm[a].x, lm[a].y); ctx.lineTo(lm[b].x, lm[b].y);
    ctx.strokeStyle = '#1c2128'; ctx.lineWidth = 9; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lm[a].x, lm[a].y); ctx.lineTo(lm[b].x, lm[b].y);
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 5; ctx.stroke();
  }
  for (var name in lm) {
    var pt = lm[name], color = jColors[name] || '#58a6ff', sel = (name === selectedJoint), r = sel ? 12 : 8;
    ctx.beginPath(); ctx.arc(pt.x, pt.y, r + 5, 0, Math.PI * 2);
    ctx.fillStyle = color + '28'; ctx.fill();
    ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fillStyle = sel ? '#ffffff' : color; ctx.fill();
    if (sel) { ctx.beginPath(); ctx.arc(pt.x, pt.y, r+2, 0, Math.PI*2); ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.stroke(); }
  }
  for (var i = 0; i < ANGLE_MAP.length; i++) {
    var ak = ANGLE_MAP[i][0], jn = ANGLE_MAP[i][1];
    if (angles[ak] == null || !lm[jn]) continue;
    var val = angles[ak].toFixed(0) + ' deg', color = getColor(ak, angles[ak]);
    var x = lm[jn].x + 14, y = lm[jn].y - 6;
    ctx.font = 'bold 11px Arial';
    var tw = ctx.measureText(val).width + 10;
    ctx.fillStyle = 'rgba(13,17,23,0.82)'; ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x-4, y-13, tw, 18, 4); } else { ctx.rect(x-4, y-13, tw, 18); }
    ctx.fill(); ctx.fillStyle = color; ctx.fillText(val, x, y);
  }
  var html = '';
  for (var i = 0; i < ANGLE_MAP.length; i++) {
    var ak = ANGLE_MAP[i][0], val = angles[ak];
    if (val == null) continue;
    var color = getColor(ak, val);
    var range = RANGES[ak] ? RANGES[ak][0]+'-'+RANGES[ak][1]+' deg' : 'N/A';
    var label = ak.replace('_angle','').replace(/_/g,' ');
    html += '<div class="angle-row"><span class="angle-name">'+label+'</span>'
          + '<span class="angle-val" style="color:'+color+'">'+val.toFixed(0)+' deg</span>'
          + '<div class="angle-ideal">ideal: '+range+'</div></div>';
  }
  angleList.innerHTML = html;
}
slider.addEventListener('input', function() {
  var idx = parseInt(slider.value);
  fcur.textContent = 'Frame ' + idx; draw(idx);
});
canvas.addEventListener('click', function(e) {
  var rect = canvas.getBoundingClientRect();
  var scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
  var mx = (e.clientX - rect.left) * scaleX, my = (e.clientY - rect.top) * scaleY;
  var frame = DATA[parseInt(slider.value)];
  if (!frame || !frame.landmarks) return;
  var W = canvas.width, H = canvas.height;
  var lm = scaleLandmarks(frame.landmarks, W, H);
  var closest = null, minDist = 30;
  for (var name in lm) {
    var dx = lm[name].x - mx, dy = lm[name].y - my;
    var dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < minDist) { minDist = dist; closest = name; }
  }
  selectedJoint = closest;
  if (closest) {
    var match = null;
    for (var i = 0; i < ANGLE_MAP.length; i++) { if (ANGLE_MAP[i][1] === closest) { match = ANGLE_MAP[i]; break; } }
    var angles = frame.angles || {};
    if (match) {
      var ak = match[0], val = angles[ak];
      var color = val != null ? getColor(ak, val) : '#58a6ff';
      var status = val != null ? getStatus(ak, val) : 'No data';
      var range = RANGES[ak] ? RANGES[ak][0]+'-'+RANGES[ak][1]+' deg' : 'N/A';
      var sc = status === 'Ideal' ? '#3fb950' : status === 'Slightly off' ? '#d29922' : '#f85149';
      clickPanel.innerHTML = '<p style="font-size:11px;font-weight:bold;color:#58a6ff;margin:0 0 6px">'
        +closest.replace(/_/g,' ').toUpperCase()+'</p>'
        +'<p style="font-size:26px;font-weight:bold;color:'+color+';margin:0 0 2px">'
        +(val!=null?val.toFixed(1)+' deg':'N/A')+'</p>'
        +'<p style="font-size:10px;color:#8b949e;margin:0 0 6px">ideal: '+range+'</p>'
        +'<p style="font-size:13px;font-weight:bold;color:'+sc+';margin:0">'+status+'</p>';
    } else {
      clickPanel.innerHTML = '<p style="font-size:11px;font-weight:bold;color:#58a6ff;margin:0 0 4px">'
        +closest.replace(/_/g,' ').toUpperCase()+'</p>'
        +'<p style="font-size:11px;color:#8b949e;margin:0">Position tracked</p>';
    }
  }
  draw(parseInt(slider.value));
});
draw(0);
</script>
</body>
</html>"""

    import base64
    encoded  = base64.b64encode(html_page.encode('utf-8')).decode('utf-8')
    data_url = f"data:text/html;base64,{encoded}"
    return f'<iframe src="{data_url}" width="100%" height="580px" style="border:none; border-radius:12px;"></iframe>'


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
                out_interactive = gr.HTML(label="Interactive Pose Analysis")
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