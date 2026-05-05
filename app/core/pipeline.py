import os
import json
import torch
import numpy as np
import imageio
import cv2
from app.config import OUTPUTS_DIR, DEVICE
from app.services.video_engine import convert_to_mp4, extract_frames
from app.services.ai_models import model_manager
from app.services.analysis_engine import run_sync_logic

class AthletiQPipeline:
    def __init__(self):
        self.mm = model_manager # Model Manager singleton

    def auto_detect_shot(self, video_path):
        if self.mm.shot_classifier:
            try:
                result = self.mm.shot_classifier.predict(video_path)
                return result["shot"], result["confidence"]
            except Exception as e:
                print(f"Shot classifier error: {e}")
        return "None", 0.0

    def process(self, video_path, click_coords, shot_type=None, progress_callback=None):
        """
        The main trigger function for the entire analysis.
        This is what your future backend will call.
        """
        try:
            if not video_path or not click_coords:
                return None, "Missing inputs"

            # 1. Prepare Video & Frames
            playable_path = convert_to_mp4(video_path)
            first_frame, _ = extract_frames(playable_path)

            # 2. SAM2 Propagation
            if DEVICE == "cuda":
                autocast_ctx = torch.autocast(DEVICE, dtype=torch.bfloat16)
            else:
                autocast_ctx = torch.inference_mode()

            from app.config import TEMP_DIR
            with torch.inference_mode(), autocast_ctx:
                inference_state = self.mm.predictor.init_state(video_path=TEMP_DIR)
                self.mm.predictor.reset_state(inference_state)
                
                points = np.array([[click_coords[0], click_coords[1]]], dtype=np.float32)
                labels = np.array([1], np.int32)
                self.mm.predictor.add_new_points(inference_state, frame_idx=0, obj_id=1, points=points, labels=labels)
                
                video_segments = {}
                tracked_indices = []
                for out_frame_idx, out_obj_ids, out_mask_logits in self.mm.predictor.propagate_in_video(inference_state):
                    is_tracked = False
                    frame_masks = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        frame_masks[out_obj_id] = mask
                        if out_obj_id == 1 and mask.any() and mask.sum() > 100:
                            is_tracked = True
                    
                    video_segments[out_frame_idx] = frame_masks
                    if is_tracked: tracked_indices.append(out_frame_idx)

                if not tracked_indices:
                    return None, "Could not track player."

                start_idx, end_idx = min(tracked_indices), max(tracked_indices)

            # 3. Isolated Player Video Generation
            out_isolated_path = os.path.join(OUTPUTS_DIR, "output_segmented.mp4")
            cap = cv2.VideoCapture(playable_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = imageio.get_writer(out_isolated_path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
            
            idx, frames_written = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if start_idx <= idx <= end_idx:
                    isolated_frame = np.zeros_like(frame)
                    if idx in video_segments and 1 in video_segments[idx]:
                        mask = video_segments[idx][1]
                        if mask.ndim == 3: mask = mask[0]
                        if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        isolated_frame[mask > 0] = frame[mask > 0]
                    writer.append_data(cv2.cvtColor(isolated_frame, cv2.COLOR_BGR2RGB))
                    frames_written += 1
                idx += 1
            cap.release()
            writer.close()

            # 4. Biomechanics Extraction
            bio_json_path = os.path.join(OUTPUTS_DIR, "segmented_biomechanics.json")
            practice_pose_data = self.mm.extractor.extract_from_video(out_isolated_path)
            angle_results = []
            for frame in practice_pose_data["frames"]:
                if "angles" in frame:
                    angle_results.append({"frame": frame["frame_idx"], "time": frame["time_sec"], **frame["angles"]})
            with open(bio_json_path, "w") as f:
                json.dump(angle_results, f, indent=4)

            # 5. Sync & Final Analytics
            sync_video, feedback, plots = run_sync_logic(
                practice_pose_data, shot_type, out_isolated_path, 
                self.mm.extractor, self.mm.sync_engine, progress_callback=progress_callback
            )

            if DEVICE == "cuda": torch.cuda.empty_cache()

            return {
                "isolated_video": out_isolated_path,
                "biomechanics_json": bio_json_path,
                "sync_video": sync_video,
                "feedback": feedback,
                "plots": plots
            }, None
        except Exception as e:
            if DEVICE == "cuda": torch.cuda.empty_cache()
            return None, str(e)

# Singleton instance
pipeline = AthletiQPipeline()
