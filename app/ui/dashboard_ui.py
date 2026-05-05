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
                gr.Markdown("### 📊 Analysis Results")
                out_score = gr.Markdown("Score will appear here.")
                out_isolated = gr.Video(label="Isolated Player (Cutout)")
                out_comparison = gr.Video(label="Technical Comparison (Side-by-Side)")
                out_json = gr.File(label="Joint Angle Data (JSON)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Biomechanical Angle Trends")
                with gr.Tabs():
                    with gr.Tab("Elbows"):
                        plot_l_elbow = gr.Plot(label="Left Elbow")
                        plot_r_elbow = gr.Plot(label="Right Elbow")
                    with gr.Tab("Knees"):
                        plot_l_knee = gr.Plot(label="Left Knee")
                        plot_r_knee = gr.Plot(label="Right Knee")
                    with gr.Tab("Hips"):
                        plot_l_hip = gr.Plot(label="Left Hip")
                        plot_r_hip = gr.Plot(label="Right Hip")

        with gr.Accordion("ℹ️ How to read these charts?", open=False):
            gr.Markdown("""
            - **Blue Line**: Your technique.
            - **Green Shaded Area**: The Professional 'Ideal' Zone (IQR).
            - **Dashed Red Line**: The moment of impact (Strike).
            - **Goal**: Keep your blue line within or close to the green corridor during the strike phase.
            """)

    # Shared States
    clean_img_state = gr.State()
    click_coord_state = gr.State()
    
    return {
        "video_input": video_input,
        "shot_select": shot_select,
        "first_frame_display": first_frame_display,
        "analyze_btn": analyze_btn,
        "out_score": out_score,
        "out_isolated": out_isolated,
        "out_comparison": out_comparison,
        "out_json": out_json,
        "plot_l_elbow": plot_l_elbow,
        "plot_r_elbow": plot_r_elbow,
        "plot_l_knee": plot_l_knee,
        "plot_r_knee": plot_r_knee,
        "plot_l_hip": plot_l_hip,
        "plot_r_hip": plot_r_hip,
        "clean_img_state": clean_img_state,
        "click_coord_state": click_coord_state
    }
