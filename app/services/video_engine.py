import os
import cv2
import shutil
import imageio
from app.config import PROJECT_ROOT, OUTPUTS_DIR, TEMP_DIR

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
