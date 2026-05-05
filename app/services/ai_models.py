import os
import sys
import torch
from app.config import (
    PROJECT_ROOT, DEVICE, SAM2_CHECKPOINT, SAM2_CONFIG, 
    POSE_MODEL_PATH, SHOT_MODEL_PATH, SHOT_CONFIG_PATH
)

# Add SAM2 to path
SAM2_PATH = os.path.join(PROJECT_ROOT, "segment-anything-2")
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

from core.biomechanics.pose_extractor import PoseExtractor
from core.shot_classifier import ShotClassifier
from core.syncing.sync_engine import SyncEngine

try:
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.utils.misc
    from torch.nn.attention import SDPBackend
    sam2.utils.misc.get_sdp_backends = lambda dropout_p: [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
except ImportError:
    build_sam2_video_predictor = None

class ModelManager:
    def __init__(self):
        self.predictor = None
        self.extractor = None
        self.sync_engine = None
        self.shot_classifier = None
        self._load_models()

    def _load_models(self):
        # 1. SAM2
        if build_sam2_video_predictor and os.path.exists(SAM2_CHECKPOINT):
            print(f"Loading SAM2 on {DEVICE}...")
            self.predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        
        # 2. Pose Extractor
        self.extractor = PoseExtractor(model_asset_path=POSE_MODEL_PATH)
        
        # 3. Sync Engine
        self.sync_engine = SyncEngine()
        
        # 4. Shot Classifier
        try:
            self.shot_classifier = ShotClassifier(
                model_path=SHOT_MODEL_PATH,
                config_path=SHOT_CONFIG_PATH
            )
            print("ShotClassifier Loaded.")
        except Exception as e:
            print(f"ShotClassifier load failed: {e}")

model_manager = ModelManager()
