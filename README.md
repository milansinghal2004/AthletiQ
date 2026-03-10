# AthletiQ – AI-Powered Sports Motion Analysis

## Overview

AthletiQ is an AI-driven sports analytics project designed to analyze cricket batting technique using computer vision and machine learning. The system processes video input to extract human pose landmarks, detect batting shots, and evaluate player movements by comparing them with reference motion patterns.

The platform provides quantitative performance analysis and corrective feedback to help players improve their technique without requiring expensive motion capture systems.

---

## Key Features

* Pose estimation using MediaPipe to extract skeletal keypoints
* Automatic cricket shot detection from video
* Clip extraction around batting action
* Movement synchronization with reference shots
* Motion comparison using Dynamic Time Warping (DTW)
* Joint-level performance scoring
* AI-generated technique feedback and corrections
* UI dashboard for visualizing analysis results

---

## System Pipeline

1. Video Input
2. Pose Estimation (MediaPipe)
3. Pose Validation and Smoothing
4. Shot Detection
5. Automatic Clip Extraction
6. Motion Synchronization
7. Movement Comparison (DTW)
8. Performance Scoring
9. Feedback & Correction Generation
10. Visualization through UI

---

## Tech Stack

**Computer Vision**

* OpenCV
* MediaPipe Pose

**Machine Learning**

* NumPy
* SciPy
* DTW algorithms

**Backend**

* Python

**Frontend**

* React / Streamlit (TBD)

---

## Project Timeline

Development runs from **March 11 to April 25**, covering four phases:

1. Pose Detection
2. Shot Detection Pipeline
3. Movement Analysis
4. UI & Backend Integration

---

## Goal

The objective of AthletiQ is to create a scalable AI-powered coaching assistant capable of analyzing sports movements from simple video input and delivering actionable performance insights.

---

## Future Improvements

* 3D pose estimation for improved depth accuracy
* Multi-angle video analysis
* Real-time feedback system
* Integration with mobile applications
* Expansion to other sports such as tennis, golf, and baseball
