import os
import torch
import multiprocessing

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_video_frames")

# --- Model Checkpoints ---
SAM2_CHECKPOINT = os.path.join(MODELS_DIR, "sam2/checkpoints/sam2_hiera_small.pt")
SAM2_CONFIG = "configs/sam2/sam2_hiera_s.yaml"
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "mediapipe/pose_landmarker.task")
SHOT_MODEL_PATH = os.path.join(MODELS_DIR, "shot_detection", "cricket_shot_r3d18_final.pth")
SHOT_CONFIG_PATH = os.path.join(MODELS_DIR, "shot_detection", "cricket_shot_config.json")
REFERENCES_DB_PATH = os.path.join(ASSETS_DIR, "references/reference_shots.json")

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_hardware():
    if DEVICE == "cuda":
        torch.set_float32_matmul_precision('high')
        print("GPU Optimization: Enabled high-precision matmul.")
    else:
        torch.set_num_threads(multiprocessing.cpu_count())
        print(f"CPU Optimization: Using {multiprocessing.cpu_count()} threads.")

# Ensure directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
