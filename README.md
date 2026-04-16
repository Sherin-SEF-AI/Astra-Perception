# 🌠 Astra Perception: High-Performance AI-ADAS Suite

![Version](https://img.shields.io/badge/version-2.1.0-cyan)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey)
![GPU](https://img.shields.io/badge/GPU-RTX%20Enabled-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**Astra Perception** is a next-generation, distributed Advanced Driver Assistance System (ADAS) specifically engineered for the complexities of unstructured road environments. Built with a Cyber-Minimalist aesthetic, it combines real-time computer vision with intelligent threat analysis to provide a "Glass Cockpit" experience for any vehicle.

---

## 🚀 Unique Selling Points

### 🇮🇳 Optimized for Indian Road Conditions
Unlike traditional ADAS that relies on perfect lane markings, Astra uses **Multi-Scale Road Surface Modeling**. It excels on roads with faded markings, potholes, and varied paving by identifying the physical drivable surface rather than just lines.

### 🧠 AI Temporal Hallucination
Ever seen a detection box flicker? Astra eliminates this. Our **Hallucination Memory** keeps objects alive for up to 3 frames if the AI momentarily loses them, ensuring warnings are rock-solid and flicker-free.

### 📡 Distributed ECU Architecture
Mimicking real automotive hardware, the system splits perception and decision-making:
1.  **AI Lens (Vision Engine)**: Handles raw sensor data and heavy AI inference on GPU.
2.  **Control ECU (Cockpit)**: A standalone dashboard that receives data via low-latency UDP to manage vehicle logic and path planning.

---

## 🛠️ Core Features

### 1. High-Precision Perception
- **Small-Object Mastery**: Powered by **YOLOv8 Small** (upgraded from Nano) for superior detection of distant bikes, pedestrians, and hazards.
- **Pulsing Cyber-Grid**: A 30 FPS persistent drivable area overlay with dynamic grid-pattern caching for zero CPU overhead.
- **Surface Sentinel**: Detects potholes and speed bumps using Adaptive Gaussian filtering, cross-referenced with the drivable area mask.

### 2. Intelligent Threat Manager
- **Kalman Filter Smoothing**: Predicts trajectories and smooths distance/velocity data.
- **Intent Analysis**: Automatically detects "Cut-In" maneuvers from the side before they become an immediate danger.
- **Priority Engine**: A smart "Single-Voice" system that only alerts you to the #1 most critical threat, avoiding beeping fatigue.

### 3. Level 1 Control Simulation
- **Virtual Actuators**: Real-time PID controllers for simulated Steering and Acceleration.
- **Top-Down Radar**: A tactical 2D visualization with motion trails for all tracked objects.

---

## 💻 Technical Stack

- **AI Inference**: Ultralytics YOLOv8 (GPU / CUDA / FP16)
- **Tracking**: Kalman-Filtered Centroid Tracking + Hallucination Memory
- **CV Math**: OpenCV (LAB Color Space, L1 Manhattan Distance Optimization)
- **Interface**: PyQt6 with Cyber-Minimalist Glass-Panel CSS
- **Performance**: Asynchronous Producer-Consumer Pipeline (Multi-threaded)

---

## ⚡ Quick Start

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA 12 support (Fallbacks to high-performance CPU mode automatically)

### Installation
```bash
git clone https://github.com/Sherin-SEF-AI/Astra-Perception.git
cd Astra-Perception
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Launch Modes
Access three distinct launch modes from your application menu:
- **Vision: AI Lens**: The primary perception engine.
- **Drive: Control ECU**: The standalone radar and cockpit dashboard.
- **Nexus: Full Stack**: Launches the entire integrated system with one click.

---

## 🤝 Author
**Sherin Joseph Roy**
*Professional AI Engineer Specializing in Autonomous Systems*

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
