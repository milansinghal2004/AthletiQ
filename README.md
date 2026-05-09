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
*   **Meta SAM2 Integration**: High-precision point-to-object tracking for dynamic background isolation and player segmentation.
*   **12-Point Keypoint Tracking**: Specialized skeletal extraction using MediaPipe, focusing on critical joint angles (elbows, knees, hips, shoulders).
*   **R3D-18 CNN Shot Detection**: Automatic classification of 10+ cricket shot types (Cover Drive, Pull, Flick, etc.).
*   **Interactive SVG Diagnostic Widget**: Real-time interactive biomechanical report with clickable joint analysis and ideal range overlays.
*   **Segmented DTW Alignment**: Proprietary temporal alignment using Dynamic Time Warping to synchronize player movement with professional benchmarks.
*   **Generative Technical Reports**: LLM-powered feedback engine providing context-aware coaching tips based on biomechanical deltas.

### 💻 Full-Stack Architecture
*   **Frontend**: A high-fidelity, "Cyber-Command" themed interface built with Vanilla JS and CSS Glassmorphism.
*   **Orchestration**: Node.js (Express) backend managing user sessions, multi-step analysis triggers, and database synchronization.
*   **Diagnostic Engine**: Python-based pipeline orchestrating heavy-duty AI processing and Gradio-powered visualization.
*   **Cloud Persistence**: PostgreSQL (Neon Cloud) for robust historical tracking, user profile management, and performance analytics.

---

## 🏗️ System Architecture & Workflow

AthletiQ operates on a **Distributed Service Architecture**, separating the user-facing web interface from the resource-intensive AI processing engine.

### System Flowchart
```mermaid
flowchart TD
    Start([User Starts Practice]) --> Login{Authenticated?}
    Login -- No --> Register[Register / Login]
    Register --> Login
    Login -- Yes --> Upload[Upload Cricket Video]
    
    Upload --> NodeSrv[Node.js Express Server]
    NodeSrv --> DB[(Neon PostgreSQL)]
    
    NodeSrv --> PythonLaunch[Spawn Analysis Runner]
    PythonLaunch --> SAM2[SAM2: Point-to-Object Tracking]
    SAM2 --> Autoclip[Intelligent Video Auto-clipping]
    
    Autoclip --> MP[MediaPipe: 12-Point Keypoint Extraction]
    MP --> DTW[Segmented DTW: Temporal Alignment]
    
    DTW --> AI_Eng[Analysis Engine: Biomechanical Delta]
    AI_Eng --> Ollama[Ollama LLM: Generative Technical Report]
    
    Ollama --> Result[Final Analytics Dashboard]
    Result --> Save[Archive Results to DB]
    Save --> History[View Performance History]
```

### Technical Architecture
```mermaid
graph TB
    subgraph "Client Layer"
        UI["Modern Web Interface (Vanilla JS/CSS)"]
        GR_UI["Gradio Analysis Dashboard"]
    end

    subgraph "Orchestration Layer (Node.js)"
        Srv["Express Server (server.js)"]
        Auth["Session & Auth Manager"]
        Runner["Python Subprocess Runner"]
    end

    subgraph "AI & Processing Layer (Python)"
        Pipe["AthletiQ Master Pipeline"]
        subgraph "Engines"
            SAM["SAM2 Tracking Engine"]
            Pose["MediaPipe Pose Engine"]
            Shot["R3D-18 Shot Classifier"]
            Sync["DTW Alignment Engine"]
            LLM["Ollama Generative Engine"]
        end
    end

    subgraph "Persistence Layer"
        Postgres[(PostgreSQL - Neon Cloud)]
        RefDB[(Biomechanical Reference DB)]
        Media[(S3 / Local Video Storage)]
    end

    UI <--> Srv
    Srv <--> Auth
    Srv <--> Postgres
    Srv --> Runner
    Runner --> Pipe
    
    Pipe --> SAM
    Pipe --> Pose
    Pipe --> Shot
    Pipe --> Sync
    Pipe --> LLM
    
    Sync --> RefDB
    Pipe --> GR_UI
    Srv <--> GR_UI
```

---

## 🛠️ Installation & Component Setup

### 1. Prerequisites
*   **Python**: 3.10 or higher
*   **Node.js**: 18.x or higher
*   **Ollama**: Installed and running locally
*   **GPU**: NVIDIA GPU with CUDA 11.8+ (Highly recommended for SAM2)

### 2. Backend Setup (AI Engine)
```bash
# Clone and enter repo
git clone https://github.com/milansinghal2004/AthletiQ.git
cd AthletiQ

# Install core dependencies
pip install -r requirements.txt

# SAM2 Sub-module Installation
cd segment-anything-2
pip install -e .
cd ..
```

> [!IMPORTANT]
> **Model Weights**: Place `sam2_hiera_small.pt` in `models/sam2/checkpoints/` and the MediaPipe `pose_landmarker.task` in `models/mediapipe/`.

### 3. Generative Feedback (Ollama)
AthletiQ uses Ollama for local, privacy-focused LLM inference.
1. [Download Ollama](https://ollama.com/download)
2. Pull the required model (defaults to `gemma2` or custom tag):
   ```bash
   ollama pull gemma2
   ```
3. Ensure the server is running on `http://localhost:11434`.

### 4. Database Linking (PostgreSQL)
AthletiQ utilizes **Neon PostgreSQL** for serverless persistence.
1. Create a free project on [Neon.tech](https://neon.tech).
2. Obtain your `DATABASE_URL`.
3. Create a `.env` file in the `frontend/` directory:
   ```env
   DATABASE_URL=postgresql://user:password@host/neondb?sslmode=verify-full
   ```

### 5. Frontend Setup
```bash
cd frontend
npm install
```

---

## 🚀 Execution Workflow

### Running Locally

1.  **Start the Orchestration Server**:
    ```bash
    cd frontend
    npm start
    ```
    *Access the main dashboard at: `http://localhost:3000`*

2.  **Start the Analysis Dashboard** (Optional - Auto-launched by Frontend):
    ```bash
    # From project root
    python app/main.py
    ```

### Using the App
1.  **Login**: Access your personalized history and technical stats.
2.  **Upload**: Drag and drop your practice session video.
3.  **SAM2 Selection**: Click the ball-contact point to initiate tracking.
4.  **Analyze**: View your biomechanical score, joint-angle overlays, and AI technical report.
5.  **History**: Track your improvement over time via the interactive profile dashboard.

---

## 🏷️ Technical Keywords
`Biomechanics` `Cricket Analytics` `Computer Vision` `SAM2` `MediaPipe` `DTW` `Express.js` `Gradio` `Ollama` `PostgreSQL` `AI Coaching`

---
*Developed with ❤️ by the AthletiQ Team - Redefining Athletic Performance Through AI.*
