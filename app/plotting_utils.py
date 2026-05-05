import plotly.graph_objects as go
import numpy as np

def generate_biomechanic_plot(joint_name, practice_data, reference_stats, phases):
    """
    Generates a Plotly figure for a specific joint's angle trend.
    
    Args:
        joint_name (str): Name of the joint (e.g., 'left_elbow')
        practice_data (list): List of angles for the practice video (aligned)
        reference_stats (dict): Dictionary containing 'q1', 'q3', 'mean' lists
        phases (dict): Dictionary with 'strike' frame index
    """
    fig = go.Figure()

    # Time/Frame Axis
    x = list(range(len(practice_data)))

    # 1. Reference IQR Band (Shaded Area)
    if 'q1' in reference_stats and 'q3' in reference_stats:
        q1 = reference_stats['q1']
        q3 = reference_stats['q3']
        
        # Ensure lengths match practice data (handle DTW artifacts)
        q1 = q1[:len(x)]
        q3 = q3[:len(x)]

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=q3 + q1[::-1],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f'Professional IQR'
        ))

    # 2. Reference Mean (Dashed Line)
    if 'mean' in reference_stats:
        mean_vals = reference_stats['mean'][:len(x)]
        fig.add_trace(go.Scatter(
            x=x, y=mean_vals,
            mode='lines',
            line=dict(color='rgba(0, 255, 0, 0.5)', dash='dash', width=1),
            name='Professional Mean'
        ))

    # 3. Practice Trace (Bold Solid Line)
    fig.add_trace(go.Scatter(
        x=x, y=practice_data,
        mode='lines',
        line=dict(color='#00D4FF', width=3),
        name='Your Technique'
    ))

    # 4. Strike Marker (Vertical Line)
    if phases and 'strike' in phases:
        strike_idx = phases['strike']
        fig.add_vline(
            x=strike_idx, 
            line_width=2, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Strike",
            annotation_position="top right"
        )

    # Styling
    clean_name = joint_name.replace("_", " ").title()
    fig.update_layout(
        title=dict(text=f"{clean_name} Angle Trend", x=0.5, font=dict(size=18)),
        xaxis_title="Frame (Aligned)",
        yaxis_title="Angle (Degrees)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Set Y-axis range to be reasonable for joint angles
    fig.update_yaxes(range=[0, 180])

    return fig
