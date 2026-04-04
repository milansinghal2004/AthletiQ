# 🏏 AthletiQ - Advanced Cricket Performance Analytics

AthletiQ is a high-performance, single-stage pipeline for cricket technical analysis. It leverages state-of-the-art computer vision models (SAM2 and MediaPipe) to provide automated player segmentation, biomechanical joint-angle extraction, and side-by-side technical comparison against ideal references. 

--- 
 
## 🏗️ System Architecture

The project follows a modular, production-ready architecture designed for scalability and clear separation of concerns.

```text
AthletiQ/
├── app/
│   └── main_dashboard.py      # Unified Gradio Entry Point
├── core/
│   ├── biomechanics/          # Pose extraction & Angle logic
│   └── syncing/               # Shot alignment (DTW Engine)
├── models/
│   ├── sam2/                  # SAM2 Checkpoints & Configs
│   └── mediapipe/             # Pose Landmarker Tasks
├── assets/
│   └── references/            # Professional Reference Metadata & Videos
└── outputs/                   # Analysis Results (Auto-generated)
```

---

## ⚡ Key Features

- **Unified Performance Pipeline**: A streamlined, single-screen workflow from upload to comparison.
- **Isolated Player Cutouts**: Deep background removal using Meta's SAM2 for focused technical review.
- **Biomechanical Analysis**: Automated extraction of 8 critical joint angles (elbows, shoulders, hips, knees).
- **Intelligent Shot Sync**: Dynamic Time Warping (DTW) engine to align practice shots with professional references in 30% slow-motion.
- **GPU Accelerated**: Optimized for NVIDIA RTX 30-series GPUs with mixed-precision inference and CUDA kernels.

---

## 🚀 Production Setup

### 1. Environment Initialization
Ensure you have Python 3.10+ installed. Activate your virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Prerequisites
AthletiQ requires the following model binaries (excluded from Git for repository hygiene):
- **SAM2 Hiera Small**: Place `sam2_hiera_small.pt` in `models/sam2/checkpoints/`.
- **MediaPipe Pose**: Place `pose_landmarker.task` in `models/mediapipe/`.

### 3. Execution

Launch the unified dashboard:

```bash
python app/main_dashboard.py
```

---

## 🛠️ Technical Workflow

1. **Input Stage**: User uploads a practice video and selects the shot type (e.g., Cover Drive).
2. **Segmentation**: User selects the batsman by clicking once. SAM2 propagates the mask through the video.
3. **Biomechanics**: The system isolates the player and extracts 3D-pose landmarks using MediaPipe.
4. **Alignment**: The DTW engine calculates the optimal temporal mapping between the practice and reference frames.
5. **Output**: A side-by-side, slowed-down comparison video is rendered alongside a biomechanical JSON report.

---

## 🖥️ Hardware Optimizations

- **VRAM Management**: Automatic background downsampling (640px) ensures smooth operation on 4GB VRAM cards (RTX 3050 Laptop).
- **Precision**: Uses `torch.set_float32_matmul_precision('high')` for Ampere architecture speedups.
- **Inference**: Mixed-precision (`bfloat16`) enabled for compatible CUDA devices.

---

## 🛡️ Git & Deployment
- **.gitignore**: Configured to exclude all large video binaries, temporary frames, and model checkpoints.
- **Stateless Analysis**: All intermediate frames and temporary data are cleared automatically on app exit to maintain privacy and disk space.

---

**AthletiQ - Precision Analytics for the Modern Game.**
