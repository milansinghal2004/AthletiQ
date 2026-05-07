import os
import sys
import atexit
import gradio as gr

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import setup_hardware, PROJECT_ROOT
from app.services.video_engine import clear_temp
from app.ui.dashboard_ui import create_ui_layout
from app.ui.gradio_handlers import bind_events
from app.ui.theme import get_athletiq_theme, get_athletiq_css

def start_app():
    # 1. Initialize Hardware
    setup_hardware()

    # 2. Cleanup on exit
    atexit.register(clear_temp)

    # 3. Create UI & Bind Logic inside the context
    print("Initializing Models...")
    with gr.Blocks() as demo:
        # Create Layout
        components = create_ui_layout()
        
        # Bind Events
        bind_events(components)

        # 4. Handle URL parameters (Auto-load video from frontend)
        def on_load(request: gr.Request):
            video_path = request.query_params.get("video")
            user_id = request.query_params.get("user_id")
            
            f1, f2, v, s = [None] * 4
            if video_path and os.path.exists(video_path):
                print(f"Auto-loading video: {video_path}")
                from app.ui.gradio_handlers import handle_video_upload
                # Get all pre-processed data
                f1, f2, v, s = handle_video_upload(video_path)
            
            return f1, f2, v, s, user_id

        demo.load(on_load, None, [
            components["first_frame_display"], 
            components["clean_img_state"], 
            components["video_input"], 
            components["shot_select"],
            components["user_id_state"]
        ])

    # 5. Launch
    print("AthletiQ is ready.")
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        theme=get_athletiq_theme(),
        css=get_athletiq_css(),
        allowed_paths=[
            os.path.join(PROJECT_ROOT, "outputs"), 
            os.path.join(PROJECT_ROOT, "frontend", "uploads"),
        ]
    )

if __name__ == "__main__":
    start_app()
