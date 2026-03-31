import os
import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class PoseExtractor:
    def __init__(self, model_asset_path='pose_landmarker.task'):
        self.model_asset_path = model_asset_path
        self._ensure_model()
        
        BaseOptions = mp_python.BaseOptions
        PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
        VisionRunningMode = mp_vision.RunningMode

        # MediaPipe GPU delegate is not yet supported on Windows Python API.
        # We use CPU for posture, while SAM2 handles the GPU heavy lifting.
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False
        )
        
    def _ensure_model(self):
        if not os.path.exists(self.model_asset_path):
            import urllib.request
            print(f"Downloading {self.model_asset_path}...")
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', 
                self.model_asset_path
            )

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle between three points (a, b, c) with b as vertex."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return float(np.degrees(angle))

    def extract_from_video(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0:
            fps = 30.0
            
        output_data = {
            "metadata": {
                "fps": fps,
                "joints": [
                    "left_elbow", "right_elbow",
                    "left_shoulder", "right_shoulder",
                    "left_hip", "right_hip",
                    "left_knee", "right_knee"
                ]
            },
            "frames": []
        }
        
        frame_idx = 0
        with mp_vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                timestamp_ms = int(frame_idx * 1000.0 / fps)
                pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                frame_data = {
                    "frame_idx": frame_idx,
                    "time_sec": float(frame_idx / fps),
                    "angles": {}
                }
                
                if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                    landmarks = pose_result.pose_landmarks[0]
                    
                    def get_pt(idx):
                        return [landmarks[idx].x, landmarks[idx].y]
                    
                    # Landmarks mapping for biomechanics
                    frame_data["landmarks"] = {
                        "left_shoulder": {"x": landmarks[11].x, "y": landmarks[11].y},
                        "right_shoulder": {"x": landmarks[12].x, "y": landmarks[12].y},
                        "left_elbow": {"x": landmarks[13].x, "y": landmarks[13].y},
                        "right_elbow": {"x": landmarks[14].x, "y": landmarks[14].y},
                        "left_wrist": {"x": landmarks[15].x, "y": landmarks[15].y},
                        "right_wrist": {"x": landmarks[16].x, "y": landmarks[16].y},
                        "left_hip": {"x": landmarks[23].x, "y": landmarks[23].y},
                        "right_hip": {"x": landmarks[24].x, "y": landmarks[24].y},
                        "left_knee": {"x": landmarks[25].x, "y": landmarks[25].y},
                        "right_knee": {"x": landmarks[26].x, "y": landmarks[26].y},
                        "left_ankle": {"x": landmarks[27].x, "y": landmarks[27].y},
                        "right_ankle": {"x": landmarks[28].x, "y": landmarks[28].y}
                    }
                    
                    try:
                        # Existing angle logic remains for sync
                        p = {
                            "l_sh": get_pt(11), "r_sh": get_pt(12),
                            "l_el": get_pt(13), "r_el": get_pt(14),
                            "l_wr": get_pt(15), "r_wr": get_pt(16),
                            "l_hp": get_pt(23), "r_hp": get_pt(24),
                            "l_kn": get_pt(25), "r_kn": get_pt(26),
                            "l_ak": get_pt(27), "r_ak": get_pt(28)
                        }
                        
                        frame_data["angles"] = {
                            "left_elbow": self.calculate_angle(p["l_sh"], p["l_el"], p["l_wr"]),
                            "right_elbow": self.calculate_angle(p["r_sh"], p["r_el"], p["r_wr"]),
                            "left_shoulder": self.calculate_angle(p["l_hp"], p["l_sh"], p["l_el"]),
                            "right_shoulder": self.calculate_angle(p["r_hp"], p["r_sh"], p["r_el"]),
                            "left_hip": self.calculate_angle(p["l_sh"], p["l_hp"], p["l_kn"]),
                            "right_hip": self.calculate_angle(p["r_sh"], p["r_hp"], p["r_kn"]),
                            "left_knee": self.calculate_angle(p["l_hp"], p["l_kn"], p["l_ak"]),
                            "right_knee": self.calculate_angle(p["r_hp"], p["r_kn"], p["r_ak"])
                        }
                    except Exception as e:
                        pass
                
                output_data["frames"].append(frame_data)
                frame_idx += 1
                
                if progress_callback and total_frames > 0:
                    progress_callback(frame_idx, total_frames, "Extracting Posture...")
                
        cap.release()
        return output_data
