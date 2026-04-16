# Astra Perception: High-Performance AI-ADAS Suite

![Version](https://img.shields.io/badge/version-2.1.0-cyan)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![GPU](https://img.shields.io/badge/GPU-RTX%20Enabled-green)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

Astra Perception is a high-performance, distributed Advanced Driver Assistance System (ADAS) specifically engineered for the complexities of unstructured road environments. Built with a Cyber-Minimalist aesthetic, it combines real-time computer vision with intelligent threat analysis to provide a "Glass Cockpit" experience for any vehicle.

---

## Technical Innovation

### Specialized Road Surface Modeling
Unlike traditional ADAS that relies on perfect lane markings, Astra uses Multi-Scale Road Surface Modeling. It excels on roads with faded markings, potholes, and varied paving by identifying the physical drivable surface rather than just lines.

### Intelligent Temporal Hallucination
To solve the issue of detection flickering, the system features a Hallucination Memory. It maintains object persistence for up to 3 frames if the AI momentarily loses visibility, ensuring rock-solid and stable safety warnings.

### Distributed ECU Architecture
The system utilizes a modular automotive architecture:
1. **Vision Engine (AI Lens)**: Manages high-frequency sensor data and heavy AI inference on the GPU.
2. **Control ECU (Cockpit)**: A standalone terminal that receives perception packets via low-latency UDP to manage vehicle logic and trajectory planning.

---

## Core Capabilities

### 1. High-Precision Perception
- **Small-Object Detection**: Powered by YOLOv8 Small for superior tracking of distant motorcycles, pedestrians, and obstacles.
- **Dynamic Digital HUD**: A 30 FPS persistent drivable area overlay featuring pulsing grid-pattern caching for zero CPU overhead.
- **Surface Hazard Analysis**: Adaptive Gaussian filtering to pinpoint potholes and speed bumps, cross-referenced with the drivable area mask.

### 2. Threat Management
- **Trajectory Smoothing**: Integrated Kalman Filters for precise distance and velocity prediction.
- **Predictive Intent**: Automatic detection of "Cut-In" maneuvers from the side before they enter the vehicle's direct path.
- **Priority Resolution**: A smart logic engine that isolates the single most critical threat to prevent alert fatigue.

### 3. Level 1 Control Simulation
- **Virtual Actuators**: Real-time PID controllers for simulated steering and acceleration/braking response.
- **Tactical Radar**: A 2D spatial visualization dashboard featuring motion trails for all tracked objects.

---

## Technical Stack

- **Inference**: Ultralytics YOLOv8 (GPU / CUDA / FP16)
- **Tracking**: Kalman-Filtered Centroid Tracking with Persistence Memory
- **Vision Logic**: OpenCV (LAB Color Space, L1 Manhattan Distance Optimization)
- **Interface**: PyQt6 with Cyber-Minimalist Glass-Panel CSS
- **Concurrency**: Asynchronous Multi-threaded Producer-Consumer Pipeline

---

## Installation and Deployment

### Prerequisites
- Python 3.12 or higher
- NVIDIA GPU with CUDA 12 support (Automatic high-performance CPU fallback)

### Installation
```bash
git clone https://github.com/Sherin-SEF-AI/Astra-Perception.git
cd Astra-Perception
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Launch Configurations
The suite provides three distinct operational modes accessible via the application menu:
- **Vision Engine**: The primary perception and AI analysis module.
- **Control ECU**: The standalone radar and cockpit telemetry dashboard.
- **Integrated Stack**: Simultaneous launch of the complete perception and control system.

---

## Author
**Sherin Joseph Roy**
Professional AI Engineer specializing in Autonomous Systems and Computer Vision.

## License
Distributed under the MIT License. See `LICENSE` for more information.
