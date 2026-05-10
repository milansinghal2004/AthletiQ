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

### 📋 System Flowchart
```mermaid
flowchart TD
    Start([User Uploads Practice Clip]) --> Auth{Authenticated?}
    Auth -- No --> Reg[Register / Login]
    Reg --> Auth
    Auth -- Yes --> Select[Interactive Video Selection]
    
    Select --> |Click Player| NodeSrv[Node.js Orchestrator]
    NodeSrv --> |Session Init| DB[(Neon PostgreSQL)]
    
    NodeSrv --> |Spawn Runner| Pipe[Python Master Pipeline]
    
    subgraph "AI Processing Pipeline"
        Pipe --> SAM2[SAM2: Precise Object Tracking]
        SAM2 --> Isolator[Player Isolation & Neon Glow Rendering]
        Isolator --> Pose[MediaPipe: 12-Joint Biomechanics]
        Pose --> Shot[Shot Classifier: Automatic Detection]
        Shot --> Sync[Segmented DTW: Pro-Benchmark Alignment]
    end
    
    Sync --> |Angle Deltas| LLM[Gemma-4: Generative Technical Report]
    LLM --> |Personalized Tips| UI[Interactive Diagnostic Widget]
    
    UI --> |User Feedback| Save[Archive Result to Profile]
    Save --> History[View Improvement Trends]
```

### 🏛️ Technical Architecture
```mermaid
graph TB
    subgraph "Client Interface (The Cyber-Command)"
        UI["Landing Dashboard (Vanilla JS)"]
        GR_UI["Gradio Analysis Center"]
    end

    subgraph "Orchestration & Data Layer"
        Srv["Express.js Server"]
        Auth["Auth & Session Manager"]
        Postgres[(PostgreSQL - Neon Cloud)]
    end

    subgraph "AI Engine (The Diagnostic Core)"
        subgraph "Vision Engines"
            SAM["SAM2 Tracking"]
            POSE["MediaPipe Extractor"]
            SHOT["R3D-18 Classifier"]
        end
        subgraph "Reasoning Engines"
            SYNC["Segmented DTW Sync"]
            GEN["Gemma-4 LLM Engine"]
        end
    end

    UI <--> Srv
    Srv <--> Postgres
    Srv --> |Analysis Trigger| SAM
    
    SAM --> POSE
    POSE --> SHOT
    SHOT --> SYNC
    SYNC --> GEN
    
    GEN --> GR_UI
    Srv <--> GR_UI
```

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
