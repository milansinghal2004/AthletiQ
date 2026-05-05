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

def start_app():
    # 1. Initialize Hardware
    setup_hardware()

    # 2. Cleanup on exit
    atexit.register(clear_temp)

    # 3. Create UI & Bind Logic inside the context
    print("Initializing Models...")
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Create Layout
        components = create_ui_layout()
        
        # Bind Events (MUST be inside Blocks context)
        bind_events(components)

    # 4. Launch
    print("AthletiQ is ready.")
    demo.launch(
        allowed_paths=[os.path.join(PROJECT_ROOT, "outputs")]
    )

if __name__ == "__main__":
    start_app()
