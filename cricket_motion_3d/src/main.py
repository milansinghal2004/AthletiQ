import os
import argparse
import time
from .utils import setup_logger, load_config
from .frame_extractor import FrameExtractor
from .pose_estimator import PoseEstimator
from .pose_lifter import PoseLifter
from .smoothing import PoseSmoothing
from .biomechanics_engine import BiomechanicsEngine
from ..visualization.overlay_renderer import OverlayRenderer
from ..visualization.skeleton_3d_viewer import Skeleton3DViewer
from ..export.json_writer import save_to_json
from ..export.csv_writer import save_to_csv
from .dashboard_generator import DashboardGenerator

from .selection import SelectionModel

logger = setup_logger("MainPipeline")

def run_pipeline(video_path, ideal_video_path, config_path):
    # 1. Load Config
    config = load_config(config_path)
    output_base = config['io']['output_dir']
    if not os.path.exists(output_base):
        os.makedirs(output_base)
        
    start_time = time.time()
    
    # 2. Initialize Modules
    logger.info("Initializing modules...")
    selection_model = SelectionModel(config)
    pose_estimator = PoseEstimator(config)
    pose_lifter = PoseLifter(config)
    smoother = PoseSmoothing(config)
    bio_engine = BiomechanicsEngine(config)
    overlay_renderer = OverlayRenderer(config)
    viewer_3d = Skeleton3DViewer(config)
    
    # 3. Selection / Refinement
    video_to_analyze = video_path
    if config.get('selection', {}).get('enabled', False):
        video_to_analyze = selection_model.select_batsman(video_path, output_base)

    # 4. 3D Pose Extraction
    engine_type = config.get('processing', {}).get('3d_engine', 'mediapipe')
    
    if engine_type == 'mediapipe':
        # MediaPipe handles video directly
        results_3d = pose_lifter.process_video_mediapipe(video_to_analyze)
        keypoints_2d = [] # Optional: map 33 to 25 if needed for 2D overlay
    else:
        # 5. OpenPose 2D -> 3D Lifting
        json_output_dir = os.path.join(output_base, "json")
        logger.info("Starting 2D Pose Estimation (OpenPose)...")
        success = pose_estimator.run_inference(video_to_analyze, json_output_dir)
        if not success:
            logger.error("OpenPose inference failed. Exiting.")
            return
        
        keypoints_2d = pose_estimator.load_keypoints(json_output_dir)
        logger.info(f"Loaded 2D keypoints for {len(keypoints_2d)} frames.")
        
        logger.info("Starting 3D Pose Lifting...")
        results_3d = pose_lifter.process_video_keypoints(keypoints_2d)
    
    # 6. Temporal Smoothing
    logger.info("Smoothing 3D trajectories...")
    results_3d_smoothed = smoother.smooth_sequence(results_3d)
    
    # 7. Biomechanical Analysis
    logger.info("Computing biomechanical metrics...")
    metrics_list = bio_engine.process_sequence(results_3d_smoothed)
    
    # 8. Export Data
    logger.info("Exporting results...")
    save_to_json({
        "video": video_path,
        "keypoints_3d": results_3d_smoothed,
        "metrics": metrics_list
    }, os.path.join(output_base, config['io']['export_json']))
    
    save_to_csv(metrics_list, os.path.join(output_base, config['io']['export_csv']))
    
    # 9. Visualization
    logger.info("Generating visualizations...")
    if keypoints_2d:
        annotated_video_path = os.path.join(output_base, "annotated_video.mp4")
        overlay_renderer.render_annotated_video(video_to_analyze, keypoints_2d, metrics_list, annotated_video_path)
    
    vis_3d_path = os.path.join(output_base, "skeleton_3d_animated.html")
    viewer_3d.create_animation(results_3d_smoothed, vis_3d_path)
    
    # 10. Generate 4-Pane Dashboard
    logger.info("Generating 4-Pane Analysis Dashboard...")
    dashboard_gen = DashboardGenerator(config)
    ideal_data_path = os.path.join(output_base, "ideal_motion_data.json")
    
    # Path to the "ideal" reference clip
    ideal_video = ideal_video_path
    
    logger.info("Extracting 3D pose for ideal reference video to ensure sync...")
    if engine_type == 'mediapipe':
        ideal_results_3d = pose_lifter.process_video_mediapipe(ideal_video)
    else:
        ideal_json_dir = os.path.join(output_base, "ideal_json")
        success = pose_estimator.run_inference(ideal_video, ideal_json_dir)
        ideal_kp2d = pose_estimator.load_keypoints(ideal_json_dir)
        ideal_results_3d = pose_lifter.process_video_keypoints(ideal_kp2d)
        
    ideal_results_3d_smoothed = smoother.smooth_sequence(ideal_results_3d)
    
    save_to_json({
        "video": ideal_video,
        "keypoints_3d": ideal_results_3d_smoothed,
        "metrics": []
    }, ideal_data_path)
    
    if os.path.exists(ideal_data_path):
        dashboard_gen.generate_dashboard(
            os.path.join(output_base, config['io']['export_json']),
            ideal_data_path,
            os.path.abspath(video_path),
            os.path.abspath(ideal_video),
            os.path.join(output_base, "analysis_dashboard.html")
        )
    else:
        logger.warning(f"Ideal motion data not found at {ideal_data_path}. Dashboard skipped.")
    
    end_time = time.time()
    logger.info(f"Pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    print("\n--- Cricket 3D Motion Analysis Interactive CLI ---")
    
    video_to_process = input("Please enter the path to the original video file: ").strip()
    # Strip quotes if dragged and dropped into terminal
    video_to_process = video_to_process.strip('\"\'')
    while not os.path.exists(video_to_process):
        print(f"Error: File not found at '{video_to_process}'")
        video_to_process = input("Please enter a valid path to the original video file: ").strip()
        video_to_process = video_to_process.strip('\"\'')

    ideal_video_path = input("Please enter the path to the ideal reference video file: ").strip()
    ideal_video_path = ideal_video_path.strip('\"\'')
    while not os.path.exists(ideal_video_path):
        print(f"Error: File not found at '{ideal_video_path}'")
        ideal_video_path = input("Please enter a valid path to the ideal reference video file: ").strip()
        ideal_video_path = ideal_video_path.strip('\"\'')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "config.yaml")

    run_pipeline(video_to_process, ideal_video_path, config_path)
