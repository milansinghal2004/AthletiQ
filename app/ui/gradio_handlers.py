import cv2
import gradio as gr
from app.core.pipeline import pipeline
from app.services.video_engine import convert_to_mp4, extract_frames

# Mappings for UI display
SHOT_LABEL_MAP = {
    "cover": "Cover Drive", "defense": "Defense", "flick": "Flick",
    "hook": "Hook", "late_cut": "Late Cut", "lofted": "Lofted Shot",
    "pull": "Pull Shot", "square_cut": "Square Cut",
    "straight": "Straight Drive", "sweep": "Sweep Shot"
}

def handle_video_upload(video_path):
    if not video_path:
        return None, None, None, gr.update()
    
    # Auto-detect shot
    raw_shot, conf = pipeline.auto_detect_shot(video_path)
    predicted_shot = SHOT_LABEL_MAP.get(raw_shot, "None")
    print(f"🏏 Auto-detected: {predicted_shot} ({conf*100:.1f}%)")

    # Prepare for clicking
    playable_path = convert_to_mp4(video_path)
    first_frame, _ = extract_frames(playable_path)
    
    return first_frame, first_frame, playable_path, gr.update(value=predicted_shot)

def handle_point_selection(img, evt: gr.SelectData):
    points_img = img.copy()
    cv2.circle(points_img, evt.index, 7, (0, 255, 0), -1)
    return evt.index, points_img

def run_full_analysis(video_path, click_coords, shot_type, progress=gr.Progress()):
    def pg_callback(curr, msg): progress(curr, desc=msg)
    
    result, error = pipeline.process(video_path, click_coords, shot_type, progress_callback=pg_callback)
    
    if error:
        return [None]*3 + [f"Error: {error}"] + [None]*6

    return [
        result["isolated_video"],
        result["biomechanics_json"],
        result["sync_video"],
        result["feedback"],
        *result["plots"]
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
        inputs=[components["video_input"], components["click_coord_state"], components["shot_select"]], 
        outputs=[
            components["out_isolated"], components["out_json"], components["out_comparison"], 
            components["out_score"], 
            components["plot_l_elbow"], components["plot_r_elbow"], 
            components["plot_l_knee"], components["plot_r_knee"], 
            components["plot_l_hip"], components["plot_r_hip"]
        ]
    )
