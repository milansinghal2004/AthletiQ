import gradio as gr

def get_athletiq_theme():
    # Define the core color palette
    primary_neon = "#00ff88"  # var(--acid)
    secondary_cyan = "#00e5ff" # var(--acid2)
    dark_bg = "#0a0a0c"
    card_bg = "#121217"
    border_color = "rgba(0, 255, 136, 0.15)"
    text_color = "#ffffff"
    text_dim = "#94a3b8"

    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Rajdhani"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Share Tech Mono"), "monospace"],
    ).set(
        # Colors
        body_background_fill=dark_bg,
        body_background_fill_dark=dark_bg,
        block_background_fill=card_bg,
        block_background_fill_dark=card_bg,
        block_border_width="1px",
        block_border_color=border_color,
        
        # Text
        body_text_color=text_color,
        body_text_color_dark=text_color,
        block_label_text_color=secondary_cyan,
        block_title_text_color=primary_neon,
        
        # Buttons
        button_primary_background_fill=primary_neon,
        button_primary_background_fill_hover=secondary_cyan,
        button_primary_text_color="#000000",
        button_secondary_background_fill="transparent",
        button_secondary_border_color=primary_neon,
        button_secondary_text_color=primary_neon,
        
        # Inputs
        input_background_fill="#1a1a23",
        input_border_color=border_color,
        input_border_color_focus=primary_neon,
        
        # Misc
        layout_gap="20px",
        block_radius="8px",
        button_large_radius="4px",
    )
    
    return theme

def get_athletiq_css():
    return """
    /* Futuristic scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0c; }
    ::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #00e5ff; }

    /* Custom Header Styles */
    h1 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        background: linear-gradient(90deg, #00ff88, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px !important;
    }

    /* Component Enhancements */
    .gradio-container {
        border: 1px solid rgba(0, 255, 136, 0.05);
        box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
    }
    
    .gr-box {
        border-radius: 8px !important;
        background: #121217 !important;
        border: 1px solid rgba(0, 255, 136, 0.1) !important;
    }

    /* Highlight Active Elements */
    .selected {
        border-color: #00ff88 !important;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.2) !important;
    }
    """
