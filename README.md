# AthletiQ: Unified Performance Pipeline

**AthletiQ** is a state-of-the-art technical analysis suite designed for elite cricket performance tracking. By leveraging cutting-edge Computer Vision and Biomechanical modeling, AthletiQ transforms raw practice footage into actionable, frame-accurate insights.

--- 

## 💡 Ideation
In modern professional cricket, the difference between elite performance and average results often lies in the fine details of biomechanics and technical consistency. AthletiQ was conceived to democratize high-end sports lab analysis, providing coaches and athletes with a unified pipeline that automates player isolation, pose extraction, and standard-aligned shot comparison.

---

## 🚀 Core Capabilities

### 1. Precision Player Segmentation (Meta SAM2)
Using Meta’s **Segment Anything Model 2 (SAM2)**, AthletiQ allows users to isolate a player from complex backgrounds with a single click. This ensures that technical analysis is focused entirely on the athlete, eliminating environmental noise.

### 2. Biomechanical Extraction (MediaPipe)
The system integrates **MediaPipe** Pose Landmarking to reconstruct 3D skeletal data. It automatically calculates critical technical metrics, including:
*   Joint angles (Elbow, Knee, Shoulder)
*   Stance stability
*   Power-transfer alignment

### 3. Intelligent Shot Synchronization (SyncEngine)
Equipped with a custom **SyncEngine** utilizing Dynamic Time Warping (DTW), the platform aligns user practice videos with professional reference standards in temporal space. This allows for precise, frame-by-frame comparison of shot mechanics regardless of recording speed.

### 4. Technical Comparison Rendering
Generates high-fidelity, side-by-side comparison videos at **0.3x slow motion**, enabling granular review of technical flaws and areas for improvement.

---

## 📂 Project Structure

```text
AthletiQ/
├── app/                    # Application entry points and analytics suite
│   └── main_dashboard.py   # Unified Analytics Suite entry point
├── core/                   # Core algorithmic packages
│   ├── biomechanics/       # Pose extraction and angle calculation logic
│   └── syncing/            # DTW-based synchronization engine
├── models/                 # Pre-trained model checkpoints and configurations
│   ├── sam2/               # Meta SAM2 assets
│   └── mediapipe/          # Pose landmarker tasks
├── assets/                 # Reference database and technical standards
│   └── references/         # Shot-specific reference data (Videos/JSON)
├── data/                   # Input data storage
├── outputs/                # Processed analysis results (Segmented videos/JSON)
└── requirements.txt        # Dependency specification
```

---

## 🎯 Target Audience & Relevance
*   **Professional Coaches**: Streamline technical reviews with automated segmentation and comparison.
*   **Performance Analysts**: Quantify movement patterns with high-precision biomechanical data.
*   **Atheletes**: Direct visual feedback against professional standards for self-paced improvement.

---

## 🛠 Installation (Windows)

### Prerequisites
*   **Python 3.10+** (Recommended: [Anaconda/Miniconda](https://www.anaconda.com/))
*   **NVIDIA GPU** (Optional but recommended for SAM2 acceleration)
*   **Git**

### Step 1: Clone the Repository
```powershell
git clone <repository-url>
cd AthletiQ
```

### Step 2: Environment Setup
Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Model Assets
Ensure the following model assets are placed in the `models/` directory:
*   **SAM2 Checkpoint**: `sam2_hiera_small.pt` should be in `models/sam2/checkpoints/`.
*   **MediaPipe Task**: `pose_landmarker.task` should be in `models/mediapipe/`.

---

## 📖 Usage Guide

Currently, the analysis can be initiated through the **Unified Analytics Suite**:

1.  **Initialize the Suite**:
    ```powershell
    python app/main_dashboard.py
    ```
2.  **Upload Footage**: Provide a raw practice video (mp4/avi).
3.  **Identify Subject**: Select the athlete by clicking on them in the initial frame.
4.  **Analyze**: Run the full pipeline to generate isolated cutouts and biomechanical JSON data.
5.  **Compare**: Select a reference shot type to generate a synchronized side-by-side comparison.

### Usage Parameters
| Parameter | Description | Recommended Value |
| :--- | :--- | :--- |
| `Device` | Compute backend | `cuda` (Automatic if available) |
| `Resolution` | Internal processing width | `640px` (Normalized for efficiency) |
| `Speed` | Comparison playback speed | `0.3x` |

---

## 🛤 Roadmap
*   **[UPCOMING] Web Interface**: Transitioning to a dedicated, high-performance Web UI version for enhanced user experience and cloud integration.
*   **Multi-Angle Analysis**: Support for simultaneous processing of front-on and side-on footage.
*   **Automated Coaching Narratives**: AI-generated feedback based on angle deviations.
