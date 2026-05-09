import os
import cv2
import shutil
import imageio
import numpy as np
from app.config import PROJECT_ROOT, OUTPUTS_DIR, TEMP_DIR, POSE_MODEL_PATH

def clear_temp(dir_path=TEMP_DIR):
    """Cleans the specified temporary directory."""
    if os.path.exists(dir_path):
        import shutil
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def convert_to_mp4(input_path):
    """Transcodes videos to a browser-compatible H.264 MP4 format."""
    if not input_path:
        return input_path
    
    # Use a specific name for the playable version to avoid overwriting and enable caching
    base_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(OUTPUTS_DIR, f"playable_{base_name}.mp4")
    
    # If already transcoded, skip
    if os.path.exists(output_path):
        print(f"Using cached playable version: {output_path}")
        return output_path

    print(f"Transcoding {input_path} for browser compatibility...")
    try:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # We'll use imageio's ffmpeg writer as it's more reliable for libx264 than OpenCV on Windows
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=None)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            # OpenCV is BGR, imageio needs RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            
        cap.release()
        writer.close()
        return output_path
    except Exception as e:
        print(f"Warning: Transcoding failed ({e}). Using original.")
        return input_path

def extract_frames(video_path, max_dim=640, dir_path=TEMP_DIR):
    """Extracts frames from video into the specified directory."""
    clear_temp(dir_path)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    first_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if idx == 0:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        frame_path = os.path.join(dir_path, f"{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
        
    cap.release()
    return first_frame, idx # returns first frame and total count

def create_stick_figure_video(video_path, segmented_video_path=None, progress_callback=None):
    """Run pose detection on segmented (isolated) video — guaranteed correct player."""
    source_path = segmented_video_path if segmented_video_path and os.path.exists(segmented_video_path) else video_path
    
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    BaseOptions = mp_python.BaseOptions
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    VisionRunningMode = mp_vision.RunningMode

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
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    LANDMARK_INDICES = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    import uuid
    session_id = str(uuid.uuid4())[:8]
    out_path = os.path.join(OUTPUTS_DIR, f"stick_{session_id}.mp4")
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', macro_block_size=None)

    cap = cv2.VideoCapture(source_path)
    fidx = 0
    
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nonzero = np.nonzero(gray)

            if len(nonzero[0]) > 200:
                pad = 30
                top = max(0, nonzero[0].min() - pad)
                bot = min(H, nonzero[0].max() + pad)
                left = max(0, nonzero[1].min() - pad)
                right = min(W, nonzero[1].max() + pad)
                crop = frame[top:bot, left:right].copy()

                # Boost brightness for MediaPipe
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.8 + 30, 0, 255)
                crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Add white background behind player
                mask = cv2.cvtColor(frame[top:bot, left:right], cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                white_bg = np.ones_like(crop) * 200
                crop = np.where(mask_bin[:, :, None] > 0, crop, white_bg).astype(np.uint8)

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                timestamp = int(fidx * 1000.0 / fps)
                result = landmarker.detect_for_video(mp_image, timestamp)

                if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    crop_h, crop_w = bot - top, right - left
                    lm = {}
                    for name, idx_lm in LANDMARK_INDICES.items():
                        lm[name] = {
                            "x": left + landmarks[idx_lm].x * crop_w,
                            "y": top + landmarks[idx_lm].y * crop_h
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
                        cv2.circle(canvas, (x, y), 8, (0, 180, 255), -1)

            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            fidx += 1
            if progress_callback and fidx % 10 == 0:
                progress_callback(fidx / max(1, total_frames), f"Generating Stick Figure ({fidx}/{total_frames})...")

    cap.release()
    writer.close()
    return out_path
