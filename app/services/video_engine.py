import os
import cv2
import shutil
import imageio
from app.config import PROJECT_ROOT, OUTPUTS_DIR, TEMP_DIR

def clear_temp(dir_name=TEMP_DIR):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)

def convert_to_mp4(input_path):
    """Transcodes videos to a browser-compatible MP4 format if needed."""
    if not input_path or input_path.lower().endswith('.mp4'):
        return input_path
    
    output_path = os.path.join(OUTPUTS_DIR, "playable_input.mp4")
    
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
        print(f"⚠️ Transcoding failed ({e}). Using original.")
        return input_path

def extract_frames(video_path, max_dim=640):
    """Extracts frames from video into the temp directory."""
    clear_temp()
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
            
        frame_path = os.path.join(TEMP_DIR, f"{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
        
    cap.release()
    return first_frame, idx # returns first frame and total count
