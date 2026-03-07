import json
import os
import numpy as np
import cv2
import shutil
from .utils import setup_logger, simple_dtw

logger = setup_logger("DashboardGenerator")

class DashboardGenerator:
    def __init__(self, config):
        self.config = config
        self.skeleton_pairs = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

    def _generate_plotly_json(self, poses_3d, name_prefix):
        """Generates a JSON-compatible list of traces for each frame. Always returns 2 traces."""
        # For 2D view, we take X and Y and map them to a 2D Cartesian plane.
        # MediaPipe Y is down, so we use -p[1] to flip it up.
        def to_viz(p): return [p[0], -p[1]]
        
        frames_data = []
        for pose in poses_3d:
            traces = []
            
            # Base trace structures
            joint_trace = {
                "type": "scatter", "x": [], "y": [],
                "mode": "markers", "marker": {"size": 6, "color": "#ef4444" if "Ideal" in name_prefix else "#ef4444"},
                "name": f"{name_prefix}_Joints"
            }
            line_trace = {
                "type": "scatter", "x": [], "y": [],
                "mode": "lines", "line": {"color": "#38bdf8" if "Ideal" in name_prefix else "#4ade80", "width": 5},
                "name": f"{name_prefix}_Skeleton"
            }
            
            if pose is not None:
                mapped_pts = np.array([to_viz(p) for p in pose])
                
                # Update joints
                joint_trace["x"] = mapped_pts[:, 0].tolist()
                joint_trace["y"] = mapped_pts[:, 1].tolist()
                
                # Update skeleton
                sx, sy = [], []
                for p1, p2 in self.skeleton_pairs:
                    if p1 < len(mapped_pts) and p2 < len(mapped_pts):
                        sx.extend([mapped_pts[p1, 0], mapped_pts[p2, 0], None])
                        sy.extend([mapped_pts[p1, 1], mapped_pts[p2, 1], None])
                line_trace["x"] = sx
                line_trace["y"] = sy
                
            traces.append(joint_trace)
            traces.append(line_trace)
            frames_data.append(traces)
            
        return frames_data

    def _convert_to_mp4(self, input_path, output_name, target_dir):
        """Converts video to browser-friendly MP4 in the target directory."""
        output_path = os.path.join(target_dir, output_name)
        
        # If it's already an MP4, just copy it to preserve browser compatibility
        if input_path.lower().endswith('.mp4'):
            logger.info(f"Input is already MP4, copying directly to {output_path}...")
            if os.path.exists(output_path): os.remove(output_path)
            shutil.copy2(input_path, output_path)
            return output_name
            
        logger.info(f"Converting {input_path} to MP4 in {target_dir}...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            # Fallback: just copy if conversion fails
            shutil.copy2(input_path, output_path)
            return output_name
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        for codec in ['avc1', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            tmp_path = output_path + f".{codec}.mp4"
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            if out.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
                out.release()
                if os.path.exists(output_path): os.remove(output_path)
                os.rename(tmp_path, output_path)
                cap.release()
                return output_name
            out.release()
            if os.path.exists(tmp_path): os.remove(tmp_path)
            
        cap.release()
        shutil.copy2(input_path, output_path) # Last resort copy
        return output_name

    def generate_dashboard(self, orig_data_path, ideal_data_path, orig_video_path, ideal_video_path, output_path):
        """Creates a synchronized 4-pane modern dashboard."""
        with open(orig_data_path, 'r') as f:
            orig_data = json.load(f)
        with open(ideal_data_path, 'r') as f:
            ideal_data = json.load(f)
            
        orig_poses = [np.array(p) if p else None for p in orig_data['keypoints_3d']]
        ideal_poses = [np.array(p) if p else None for p in ideal_data['keypoints_3d']]
        
        # Browser-compatible MP4 conversion in the output directory
        output_dir = os.path.dirname(output_path)
        rel_orig_vid = self._convert_to_mp4(orig_video_path, "orig_clip.mp4", output_dir)
        rel_ideal_vid = self._convert_to_mp4(ideal_video_path, "ideal_clip.mp4", output_dir)
        
        # Synchronize using DTW
        logger.info("Synchronizing sequences with DTW...")
        mapping = simple_dtw(orig_poses, ideal_poses)
        
        # Prepare Plotly Data
        orig_traces = self._generate_plotly_json(orig_poses, "Current")
        ideal_traces = self._generate_plotly_json(ideal_poses, "Ideal")
        
        # Build HTML
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biomechanical Motion Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent: #38bdf8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }}
        body {{
            background-color: var(--bg);
            color: white;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 600;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 20px;
            height: 85vh;
        }}
        .pane {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        .pane-label {{
            position: absolute;
            top: 15px;
            left: 20px;
            background: rgba(15, 23, 42, 0.8);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.7rem;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            z-index: 10;
            border: 1px solid var(--glass-border);
            color: var(--accent);
            font-weight: 600;
        }}
        video {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            background: black;
        }}
        .plotly-container {{
            width: 100%;
            height: 100%;
        }}
        .controls {{
            grid-column: span 2;
            display: flex;
            align-items: center;
            gap: 25px;
            background: rgba(30, 41, 59, 0.9);
            padding: 20px 40px;
            border-radius: 60px;
            margin-top: 30px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        input[type=range] {{
            flex-grow: 1;
            cursor: pointer;
            height: 6px;
            border-radius: 5px;
            background: #475569;
            outline: none;
            -webkit-appearance: none;
        }}
        input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px var(--accent);
        }}
        button {{
            background: var(--accent);
            border: none;
            color: white;
            padding: 10px 25px;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.3s;
        }}
        button:hover {{ opacity: 0.8; transform: scale(1.05); }}
    </style>
</head>
<body>
    <div class="header">Cricket Biomechanics Dashboard</div>
    
    <div class="dashboard-grid">
        <div class="pane">
            <div class="pane-label">Current Clip</div>
            <video id="v-orig" src="{rel_orig_vid}" muted loop playsinline></video>
        </div>
        <div class="pane">
            <div class="pane-label">Current 2D View</div>
            <div id="p-orig" class="plotly-container"></div>
        </div>
        <div class="pane">
            <div class="pane-label">Ideal Reference</div>
            <video id="v-ideal" src="{rel_ideal_vid}" muted loop playsinline></video>
        </div>
        <div class="pane">
            <div class="pane-label">Ideal 2D View</div>
            <div id="p-ideal" class="plotly-container"></div>
        </div>
    </div>

    <div class="controls">
        <button id="playBtn">Play All</button>
        <input type="range" id="masterSlider" min="0" max="{len(orig_poses)-1}" value="0">
        <div id="frameCounter">Frame: 0</div>
    </div>

    <script>
        const origTraces = {json.dumps(orig_traces)};
        const idealTraces = {json.dumps(ideal_traces)};
        const mapping = {json.dumps(mapping)};
        
        const layout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{l:0, r:0, b:0, t:0}},
            xaxis: {{range: [-1, 1], showgrid: false, zeroline: false, color: '#475569', scaleanchor: 'y'}},
            yaxis: {{range: [-1.2, 1.2], showgrid: false, zeroline: false, color: '#475569'}},
            showlegend: false
        }};

        // INITIALIZE WITH PLACEHOLDERS TO ENABLE RESTYLE
        Plotly.newPlot('p-orig', origTraces[0].length ? origTraces[0] : [{{type:'scatter'}}, {{type:'scatter'}}], layout);
        Plotly.newPlot('p-ideal', idealTraces[0].length ? idealTraces[0] : [{{type:'scatter'}}, {{type:'scatter'}}], layout);

        const vOrig = document.getElementById('v-orig');
        const vIdeal = document.getElementById('v-ideal');
        const slider = document.getElementById('masterSlider');
        const playBtn = document.getElementById('playBtn');
        const counter = document.getElementById('frameCounter');

        // Set native playback rate for videos (0.25x)
        vOrig.playbackRate = 0.25;
        vIdeal.playbackRate = 0.25;

        let isPlaying = false;
        let playInterval;

        function updateFrame(i) {{
            i = parseInt(i);
            const j = mapping[i] || 0;
            
            // OPTIMIZED PLOTLY UPDATE
            // Instead of react (full re-render), use restyle for existing traces
            if (origTraces[i] && origTraces[i].length > 0) {{
                Plotly.restyle('p-orig', {{
                    x: origTraces[i].map(t => t.x),
                    y: origTraces[i].map(t => t.y)
                }});
            }}
            
            if (idealTraces[j] && idealTraces[j].length > 0) {{
                Plotly.restyle('p-ideal', {{
                    x: idealTraces[j].map(t => t.x),
                    y: idealTraces[j].map(t => t.y)
                }});
            }}
            
            // Video sync is expensive, only seek if far apart or specifically requested
            // Browsers are better at playing naturally
            const fps = 30;
            vOrig.currentTime = i / fps;
            vIdeal.currentTime = j / fps;
            
            slider.value = i;
            counter.innerText = "Frame: " + i;
        }}

        let frameIter = 0;
        function playLoop() {{
            if (!isPlaying) return;
            frameIter = (parseInt(slider.value) + 1) % {len(orig_poses)};
            updateFrame(frameIter);
            requestAnimationFrame(() => {{
                // Throttle to ~7.5fps for 0.25x speed
                setTimeout(playLoop, 133); 
            }});
        }}

        slider.oninput = (e) => {{
            isPlaying = false;
            playBtn.innerText = "Play All";
            updateFrame(e.target.value);
        }};

        playBtn.onclick = () => {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playBtn.innerText = "Pause";
                playLoop();
            }} else {{
                playBtn.innerText = "Play All";
            }}
        }};
    </script>
</body>
</html>
        """
        with open(output_path, 'w') as f:
            f.write(html_template)
        logger.info(f"Dashboard generated at {output_path}")

if __name__ == "__main__":
    # Test/Example usage
    import argparse
    gen = DashboardGenerator({{}})
    # ... logic to run if called directly
