import gradio as gr
from app.services.analysis_engine import load_references

def create_ui_layout():
    gr.Markdown("# 🏏 AthletiQ - Unified Performance Pipeline")
    
    with gr.Tab("Performance Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="1. Upload Practice Video")
                shot_select = gr.Dropdown(choices=["None"] + list(load_references().keys()), value="None", label="2. Select Shot Type")
                first_frame_display = gr.Image(label="3. Click on Batsman to Segment", interactive=False)
                analyze_btn = gr.Button("🚀 Run Full Shot Analysis", variant="primary")
            
            with gr.Column(scale=1):
                gr.HTML("<div style='margin-bottom:10px;'><span style='color:#00ff88; font-weight:800; font-family:\"Rajdhani\", sans-serif; letter-spacing:2px; text-transform:uppercase; font-size:16px;'>📊 Technical Analysis HUD</span></div>")
                out_score = gr.Markdown(
                value="*Analysis pending...*",
                label="Biomechanical Analysis Report"
            )
                out_interactive = gr.HTML(label="Interactive Pose Analysis", sanitize_html=False)
                with gr.Row():
                    out_isolated = gr.Video(label="Isolated Player (Cutout)")
                    out_stick = gr.Video(label="Stick Figure (Pose)")
                out_comparison = gr.Video(label="Technical Comparison (Side-by-Side)")
                out_json = gr.File(label="Joint Angle Data (JSON)")

        # Shared States
    clean_img_state = gr.State()
    click_coord_state = gr.State()
    user_id_state = gr.State()
    
    return {
        "video_input": video_input,
        "shot_select": shot_select,
        "first_frame_display": first_frame_display,
        "analyze_btn": analyze_btn,
        "out_score": out_score,
        "out_isolated": out_isolated,
        "out_stick": out_stick,
        "out_interactive": out_interactive,
        "out_comparison": out_comparison,
        "out_json": out_json,
        "clean_img_state": clean_img_state,
        "click_coord_state": click_coord_state,
        "user_id_state": user_id_state
    }
