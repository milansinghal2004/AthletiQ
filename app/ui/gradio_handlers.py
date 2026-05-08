import cv2
import gradio as gr
from app.core.pipeline import pipeline
from app.services.video_engine import convert_to_mp4, extract_frames

# Mappings for UI display
from app.config import SHOT_LABEL_MAP

def handle_video_upload(video_path):
    if not video_path:
        return None, None, None, gr.update()
    
    # Auto-detect shot
    raw_shot, conf = pipeline.auto_detect_shot(video_path)
    predicted_shot = SHOT_LABEL_MAP.get(raw_shot, "None")
    print(f"\033[92m[Shot] Auto-detected: {predicted_shot} ({conf*100:.1f}%)\033[0m")

    # Prepare for clicking
    playable_path = convert_to_mp4(video_path)
    first_frame, _ = extract_frames(playable_path)
    
    return first_frame, first_frame, playable_path, gr.update(value=predicted_shot)

def handle_point_selection(img, evt: gr.SelectData):
    points_img = img.copy()
    cv2.circle(points_img, evt.index, 7, (0, 255, 0), -1)
    return evt.index, points_img

def run_full_analysis(video_path, click_coords, shot_type, user_id, progress=gr.Progress()):
    print(f"\033[94m[Analysis] starting with User ID: {user_id} (type: {type(user_id)})\033[0m")
    def pg_callback(curr, msg): progress(curr, desc=msg)
    
    result, error = pipeline.process(video_path, click_coords, shot_type, progress_callback=pg_callback)
    
    if error:
        return [None]*3 + [f"Error: {error}"] + [None]*12

    # Save to user profile if user_id exists
    if user_id:
        print(f"\033[94m[Profile] Attempting to save results for User ID: {user_id}\033[0m")
        try:
            import requests
            import re
            
            # Extract score from feedback string (matches "Overall Score: 85" or "Score: 85.4%")
            import re
            score_match = re.search(r"(?:Overall\s+)?Score:\s*(\d+(?:\.\d+)?)", result["feedback"], re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            
            payload = {
                "user_id": int(user_id) if str(user_id).isdigit() else user_id,
                "shot_type": shot_type,
                "score": score,
                "video_path": video_path
            }
            
            save_res = requests.post("http://127.0.0.1:3000/api/save-analysis", json=payload)
            if save_res.status_code == 200:
                print(f"\033[92m[Profile] Result successfully archived to database.\033[0m")
            else:
                print(f"\033[91m[Profile] Backend rejected save: {save_res.text}\033[0m")
        except Exception as e:
            print(f"\033[91m[Profile] Save failed: {str(e)}\033[0m")
    else:
        print("\033[93m[Profile] No User ID provided. Skipping history save.\033[0m")

    return [
        result["isolated_video"],
        result["biomechanics_json"],
        result["sync_video"],
        result["feedback"]
    ]

def bind_events(components):
    components["video_input"].upload(
        handle_video_upload, 
        components["video_input"], 
        [components["first_frame_display"], components["clean_img_state"], components["video_input"], components["shot_select"]]
    )

    components["first_frame_display"].select(
        handle_point_selection, 
        components["clean_img_state"], 
        [components["click_coord_state"], components["first_frame_display"]]
    )

    components["analyze_btn"].click(
        run_full_analysis, 
        inputs=[components["video_input"], components["click_coord_state"], components["shot_select"], components["user_id_state"]], 
        outputs=[
            components["out_isolated"], components["out_json"], components["out_comparison"], 
            components["out_score"]
        ]
    )
