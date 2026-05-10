# 🏏 AthletiQ: Advanced AI Biomechanical Analysis Pipeline

![AthletiQ Banner](https://img.shields.io/badge/Precision-Biomechanics-00ff88?style=for-the-badge&logo=cricket)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM-FF6F00?style=for-the-badge&logo=ollama&logoColor=white)

**AthletiQ** is a professional-grade performance diagnostic platform designed to provide elite-level biomechanical feedback for cricket players. By integrating cutting-edge computer vision (SAM2), temporal synchronization (Segmented DTW), and generative AI (Ollama), AthletiQ transforms standard practice videos into actionable technical insights.

---

## 🚀 Core Capabilities

### 🧠 Vision & AI Intelligence
*   **Meta SAM2 High-Fidelity Tracking**: Ultra-precise point-to-object tracking for dynamic player segmentation and high-fidelity background isolation.
*   **12-Point Biomechanical Extraction**: Specialized skeletal tracking using MediaPipe, focusing on critical joint angles (elbows, knees, hips, shoulders).
*   **R3D-18 CNN Shot Detection**: Automatic classification of 10+ cricket shot types (Cover Drive, Pull, Flick, etc.).
*   **Interactive Diagnostic Widget**: A custom-built SVG interface with real-time clickable joint analysis, ideal range overlays, and personalized coaching tips.
*   **Segmented DTW Alignment**: Proprietary temporal alignment using Dynamic Time Warping to synchronize player movement with professional benchmarks.
*   **Gemma-4 Technical Reports**: LLM-powered feedback engine providing deep biomechanical reasoning and technical improvement strategies.

### 💻 Full-Stack Architecture
*   **Frontend**: A "Cyber-Command" themed interface featuring glassmorphism and side-by-side comparative visualization.
*   **Orchestration**: Node.js (Express) backend managing user sessions, multi-step analysis triggers, and database synchronization.
*   **Diagnostic Engine**: Python-based pipeline orchestrating heavy-duty AI processing and high-fidelity video rendering.
*   **Cloud Persistence**: PostgreSQL (Neon Cloud) for historical tracking, performance analytics, and user growth profiling.

---

## 🏗️ System Architecture & Workflow

### 📋 Professional Analysis Pipeline Flowchart
A multi-dimensional mapping of the AthletiQ biomechanical processing engine.

```mermaid
graph TD
    %% --- INGESTION STAGE ---
    subgraph IN ["1. Ingestion & Initialization"]
        A([User Video Upload]) --> B[FFmpeg: Normalization to MP4]
        B --> C[OpenCV: Frame Extraction]
        B --> D[R3D-18: Automatic Shot Detection]
    end

    %% --- VISION STAGE ---
    subgraph VN ["2. Neural Vision Core"]
        direction LR
        E[UI Click Coordinates] --> F[SAM2: State Initialization]
        F --> G[SAM2: Point-to-Mask Propagation]
        G --> H[Isolator: BG Darkening & Neon Glow]
    end

    %% --- BIOMECHANICS STAGE ---
    subgraph BM ["3. Biomechanical Decoding"]
        direction TB
        H --> I[MediaPipe: 12-Point Landmark Detection]
        I --> J[Geometry Engine: 8-Joint Angle Calculation]
        J --> K[Linear Interpolation: Gap Filling & Smoothing]
    end

    %% --- ALIGNMENT STAGE ---
    subgraph SY ["4. Temporal Intelligence"]
        direction LR
        K --> L[Phase ID: Strike Point Detection]
        L --> M[Segmented DTW: Pro-Benchmarking]
        M --> N[Scoring: Weighted Euclidean Deviation]
    end

    %% --- INSIGHT STAGE ---
    subgraph RT ["5. Insights & HUD Rendering"]
        direction TB
        N --> O[Ollama: Gemma-4 Biometric Reasoning]
        O --> P[Interactive SVG Widget Generation]
        P --> Q[Side-by-Side Synced Video Render]
    end

    %% --- EXTERNAL CONNECTIONS ---
    IN --> VN
    VN --> BM
    BM --> SY
    SY --> RT
    RT --> Z[(PostgreSQL: History Archive)]

    %% --- STYLING ---
    style IN fill:#0a1a0a,stroke:#00ff88,stroke-width:2px
    style VN fill:#0a1a1a,stroke:#00e5ff,stroke-width:2px
    style BM fill:#1a0a1a,stroke:#ff00ff,stroke-width:2px
    style SY fill:#1a1a0a,stroke:#ffff00,stroke-width:2px
    style RT fill:#1a1a1a,stroke:#ffffff,stroke-width:2px
```

### 🏛️ Technical Architecture
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

> [!NOTE]
> For a more detailed breakdown of the internal algorithms, class structures, and segmented DTW logic, refer to the [Full System Analysis](docs/system_analysis.md) and the [Detailed Class Diagram](docs/class_diagram.md).

---

## 🛠️ Installation & Component Setup

### 1. Prerequisites
*   **Python**: 3.10+
*   **Node.js**: 18.x+
*   **Ollama**: Installed and running locally
*   **GPU**: NVIDIA GPU with CUDA 11.8+ (Required for SAM2 performance)

### 2. Backend Setup (AI Engine)
```bash
# Clone repo
git clone https://github.com/milansinghal2004/AthletiQ.git
cd AthletiQ

# Install dependencies
pip install -r requirements.txt

# SAM2 Sub-module (Critical)
cd segment-anything-2
pip install -e .
cd ..
```

> [!CAUTION]
> **Performance Note**: Ensure that the SAM2 C++ extensions are compiled (`_C` module). If missing, the "Propagate in Video" step will fall back to Pure Python and run 50x slower.

### 3. Generative Reasoning (Ollama)
AthletiQ utilizes the high-parameter `gemma4` model for deep technical analysis.
1. [Download Ollama](https://ollama.com/download)
2. Pull the high-fidelity model:
   ```bash
   ollama pull gemma4
   ```

### 4. Database Persistence (PostgreSQL)
1. Configure your `.env` in the `frontend/` directory with your Neon DB string:
   ```env
   DATABASE_URL=postgresql://user:password@host/neondb?sslmode=verify-full
   ```

---

## 🚀 Execution Workflow

1.  **Start Orchestration**:
    ```bash
    cd frontend && npm start
    ```
2.  **Access Hub**: Visit `http://localhost:3000`.
3.  **Analyze**: 
    *   Upload video and click on the player.
    *   Select shot type (or let AI auto-detect).
    *   View side-by-side comparative analysis and generative technical report.
4.  **Track**: View historical trends in your profile dashboard.

---
*Developed with ❤️ by the AthletiQ Team - Redefining Athletic Performance Through AI.*
