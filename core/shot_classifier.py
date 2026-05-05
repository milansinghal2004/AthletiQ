import cv2
import json
import numpy as np
import torch
import torchvision.models.video as video_models
import torch.nn as nn
import os

class ShotClassifier:
    def __init__(self, model_path, config_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        with open(config_path, "r") as f:
            config = json.load(f)

        self.classes     = config["classes"]
        self.num_frames  = config["num_frames"]
        self.frame_size  = config["frame_size"]
        self.num_classes = config["num_classes"]

        self.model = video_models.r3d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        ))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

        print(f"ShotClassifier loaded on {self.device}")
        print(f"   Classes: {self.classes}")

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame)

        cap.release()
        return frames

    def _preprocess(self, frames):
        arr = np.array(frames, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.tensor(arr).permute(3, 0, 1, 2)
        return tensor.unsqueeze(0).float().to(self.device)

    def predict(self, video_path):
        """
        Takes a video path, returns prediction dict.
        Returns:
            {
                "shot":       "flick",
                "confidence": 0.87,
                "all_probs":  { "cover": 0.02, "flick": 0.87, ... }
            }
        """
        frames = self._sample_frames(video_path)
        if frames is None:
            return {"shot": "unknown", "confidence": 0.0, "all_probs": {}}

        tensor = self._preprocess(frames)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        shot_name  = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        all_probs  = {cls: float(p) for cls, p in zip(self.classes, probs)}

        return {
            "shot":       shot_name,
            "confidence": confidence,
            "all_probs":  all_probs
        }