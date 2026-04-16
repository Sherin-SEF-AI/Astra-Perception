# Astra Perception: Advanced AI-Powered ADAS Suite

Astra Perception is a high-performance, distributed Advanced Driver Assistance System (ADAS) specifically engineered for the complexities of unstructured road environments. Built with a Cyber-Minimalist aesthetic, it combines real-time computer vision with intelligent threat analysis to provide a next-generation "Glass Cockpit" experience for any vehicle.

## Key Highlights

- **Indian Road Optimized**: Specialized Drivable Area Detection that excels where traditional lane-marking systems fail.
- **Distributed Architecture**: A modular system featuring a dedicated "AI Lens" for perception and a separate "Control ECU" for vehicle orchestration.
- **Cyber-Minimalist HUD**: A sleek, high-contrast visual interface designed for maximum information density with zero clutter.
- **Intelligence at the Edge**: Leverages YOLOv8, ONNX Runtime, and TensorRT for lightning-fast inference on GPU.

## Core Features

### 1. High-Precision Perception
- **Object Detection**: Tracks pedestrians, vehicles, animals (cows, dogs), and traffic signs with frame-by-frame persistence.
- **Drivable Area Analysis**: Uses multi-scale road modeling and texture-aware filtering to identify safe paths in real-time.
- **Surface Hazard Detection**: Adaptive Gaussian thresholding to pinpoint potholes and speed bumps even in challenging lighting.

### 2. Intelligent Threat Manager
- **Kalman Filter Tracking**: Smooths distance and velocity data to eliminate jitter and provide stable Time-to-Collision (TTC) metrics.
- **AI Intent Analysis**: Detects "Cut-In" maneuvers and crossing threats before they enter your direct path.
- **Consolidated Alerts**: A smart priority engine ensures you only hear the most critical warning at any given moment.

### 3. Level 1 Control Simulation
- **Virtual Actuators**: Integrated PID controllers for simulated Steering and Acceleration/Braking.
- **2D Radar Dashboard**: A standalone ECU dashboard with motion trails and top-down spatial visualization.

## Technical Stack

- **Core**: Python 3.12+
- **AI Inference**: YOLOv8 (Ultralytics), ONNX Runtime (CUDA/TensorRT)
- **Computer Vision**: OpenCV (LAB Color Space, Adaptive Filtering)
- **UI Framework**: PyQt6 (Deep-Space Dark Theme)
- **Communication**: Low-latency UDP Socket Bridge

## Quick Start

### Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure CUDA 12 and TensorRT libraries are in your system path.

### Launching the Suite
Astra Perception comes with three integrated launch modes accessible via your desktop menu:
- **Vision: ADAS AI Lens**: Launches the camera perception engine.
- **Drive: Control ECU**: Launches the standalone control dashboard.
- **Nexus: ADAS AI Stack**: Launches the full integrated system.

## Future Roadmap
- Integration with real CAN bus hardware for physical actuation.
- Deep Learning-based Semantic Segmentation for road boundaries.
- V2X (Vehicle-to-Everything) communication support.

## Author
**Sherin Joseph Roy**
Professional AI Engineer specializing in Autonomous Systems and Computer Vision.

## License
Distributed under the MIT License. See `LICENSE` for more information.
