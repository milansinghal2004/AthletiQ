# AthletiQ Detailed Class Diagram

This document provides a comprehensive map of the class structures and functional relationships within the AthletiQ analysis engine.

```mermaid
classDiagram
    %% Core Orchestration
    class AthletiQPipeline {
        +mm: ModelManager
        +auto_detect_shot(video_path) : (str, float)
        +process(video_path, click_coords, shot_type) : dict
    }

    class ModelManager {
        +predictor: SAM2VideoPredictor
        +extractor: PoseExtractor
        +sync_engine: SyncEngine
        +shot_classifier: ShotClassifier
        -_load_models()
    }

    %% AI & Vision Components
    class PoseExtractor {
        +model_asset_path: str
        +options: PoseLandmarkerOptions
        +extract_from_video(video_path) : dict
        +calculate_angle(a, b, c) : float$
        -_interpolate_gaps(frames) : list
        -_ensure_model()
    }

    class SyncEngine {
        +compute_dtw(p_angles, r_angles) : list
        +identify_phases(pose_data) : dict
        +sync_videos(p_data, r_data) : (list, dict, dict)
        -_euclidean_distance(v1, v2) : float
        -_sync_global(p_data, r_data) : list
    }

    class ShotClassifier {
        +model: torch.nn.Module
        +config: dict
        +predict(video_path) : dict
        -_preprocess_video(video_path) : torch.Tensor
    }

    %% Service Components
    class LLMEngine {
        +api_url: str
        +model_name: str
        +generate_feedback(p_frames, r_frames, shot_type) : str
        +generate_joint_tips(avg_angles, ranges, shot_type) : dict
    }

    class VideoEngine {
        <<Utility>>
        +convert_to_mp4(input_path) : str
        +extract_frames(video_path) : (str, list)
        +create_stick_figure_video(video_path, pose_json) : str
        +generate_isolated_video(video_path, masks) : str
    }

    class AnalysisEngine {
        <<Utility>>
        +load_references() : dict
        +run_sync_logic(p_data, shot, video, extractor, sync) : (str, str)
        +create_synced_video(p_vid, r_vid, path) : str
        +generate_interactive_widget(json_path, shot) : str
    }

    %% UI & Handling
    class GradioHandlers {
        <<Utility>>
        +handle_video_upload(video_path)
        +handle_point_selection(img, select_data)
        +run_full_analysis(video, coords, shot, user_id)
        +bind_events(components)
    }

    %% Relationships
    AthletiQPipeline *-- ModelManager : composition
    ModelManager o-- PoseExtractor : aggregation
    ModelManager o-- SyncEngine : aggregation
    ModelManager o-- ShotClassifier : aggregation
    
    AthletiQPipeline ..> VideoEngine : uses
    AthletiQPipeline ..> AnalysisEngine : uses
    
    GradioHandlers ..> AthletiQPipeline : triggers
    AnalysisEngine ..> LLMEngine : requests feedback
    AnalysisEngine ..> SyncEngine : calls sync_videos
```

## Key Architectural Relationships

1.  **Orchestration (Controller)**: `AthletiQPipeline` acts as the Facade for the entire analysis process. It manages the session flow and coordinates between vision tracking and biomechanical analysis.
2.  **Resource Management (Singleton)**: `ModelManager` ensures that heavy AI models (SAM2, MediaPipe) are loaded into memory exactly once and are accessible across different pipeline steps.
3.  **Specialized Logic (Stateless Engines)**: `PoseExtractor` and `SyncEngine` are designed as pure functional engines that process data vectors without maintaining internal session state, making them robust and testable.
4.  **UI Bridge**: `GradioHandlers` translates high-level UI events (clicks, uploads) into pipeline calls, acting as the glue between the Gradio interface and the backend processing.
