# AthletiQ System Analysis & Architecture

This document provides a technical overview of the AthletiQ biomechanical analysis platform, including architecture diagrams, class structures, and data flow charts.

---

## 1. System Architecture Diagram
The architecture follows a modular, decoupled design with a Node.js gateway and a specialized Python AI engine.

```mermaid
graph TB
    subgraph "Frontend Layer (UI/UX)"
        A[HTML5/JS Dashboard]
        B[Gradio Diagnostic HUD]
    end

    subgraph "Service Layer (Node.js)"
        C[Express.js Gateway]
        D[Auth & History API]
    end

    subgraph "Analysis Engine (Python)"
        E[AthletiQ Pipeline Controller]
        F[Video processing Engine]
        G[Analysis & Scoring Logic]
    end

    subgraph "AI & Machine Learning Layer"
        H[SAM2 - Tracking]
        I[MediaPipe - Pose]
        J[Custom CNN - Shot Classifier]
        K[Ollama - LLM Feedback]
    end

    subgraph "Data Persistence"
        L[(PostgreSQL - User Data)]
        M[Local Storage - Video/JSON]
    end

    A <--> C
    C <--> E
    E --> F
    E --> G
    F <--> H
    G <--> I
    G <--> J
    G <--> K
    C <--> L
    E <--> M
    B <--> E
```

---

## 2. Core Class Diagram
The system is built around a centralized model manager and a high-level pipeline orchestrator.

```mermaid
classDiagram
    class ModelManager {
        +predictor: SAM2Predictor
        +extractor: PoseExtractor
        +sync_engine: SyncEngine
        +shot_classifier: ShotClassifier
        -_load_models()
    }

    class AthletiQPipeline {
        +mm: ModelManager
        +process(video_path, click_coords)
        +auto_detect_shot(video_path)
    }

    class PoseExtractor {
        +options: PoseLandmarkerOptions
        +extract_from_video(video_path)
        +calculate_angle(a, b, c)
        -_interpolate_gaps(frames)
    }

    class SyncEngine {
        +compute_dtw(p_angles, r_angles)
        +identify_phases(pose_data)
        +sync_videos(practice, reference)
    }

    class ShotClassifier {
        +model: torch.nn.Module
        +predict(video_path)
    }

    class LLMEngine {
        +generate_feedback(practice, stats)
        +generate_joint_tips(avg_angles)
    }

    AthletiQPipeline *-- ModelManager
    ModelManager o-- PoseExtractor
    ModelManager o-- SyncEngine
    ModelManager o-- ShotClassifier
    AthletiQPipeline ..> LLMEngine : uses

> [!TIP]
> For a more exhaustive view of attributes and methods, see the [Dedicated Class Diagram](class_diagram.md).

```

---

## 3. Analysis Pipeline Flowchart
The AthletiQ pipeline is a multi-stage process that transforms raw video into structured biomechanical intelligence.

```mermaid
graph LR
    %% Stage 1: Preparation
    subgraph ST1 ["Stage 1: Preparation"]
        direction TB
        A([User Upload]) --> B{Auth?}
        B -- No --> C[Login/Reg]
        B -- Yes --> D[Click Tracking]
        C --> B
    end

    %% Stage 2: Vision Core
    subgraph ST2 ["Stage 2: Vision & AI Core"]
        direction LR
        E[SAM2 Tracking] --> F[Player Isolation]
        F --> G[Pose Extraction]
    end

    %% Stage 3: Synchronization
    subgraph ST3 ["Stage 3: Synchronization"]
        direction LR
        H[Shot Detection] --> I[Segmented DTW]
        I --> J[Accuracy Scoring]
    end

    %% Stage 4: Feedback
    subgraph ST4 ["Stage 4: Feedback & UI"]
        direction TB
        K[LLM Report] --> L[Interactive HUD]
        L --> M[(History DB)]
    end

    %% Connections
    D --> ST2
    ST2 --> ST3
    ST3 --> ST4

    %% Styling
    style ST1 fill:#003322,stroke:#00ff88,stroke-width:2px
    style ST2 fill:#002233,stroke:#00e5ff,stroke-width:2px
    style ST3 fill:#330033,stroke:#ff00ff,stroke-width:2px
    style ST4 fill:#222222,stroke:#ffffff,stroke-width:2px
```

---

## 4. Specialized Logic: Segmented DTW
The core differentiator of AthletiQ is the **Segmented DTW** which ensures critical frames (like the strike) are perfectly aligned despite temporal variations in user tempo.

```mermaid
graph LR
    P[Practice Video] --> S1[Phase Identification]
    R[Ref Video] --> S1
    S1 --> |Find Strike| Anchor((Strike Anchor))
    Anchor --> SegA[DTW Segment A: Takeback]
    Anchor --> SegB[DTW Segment B: Release]
    SegA --> Combined[Full Sync Path]
    SegB --> Combined
```
