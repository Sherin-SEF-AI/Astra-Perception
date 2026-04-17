import sys
import os
import time
import cv2
import numpy as np
import threading
import pyttsx3
import json
import socket
from collections import deque
from enum import IntEnum
from queue import Queue, Empty
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QFormLayout, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

from inference import ObjectDetector, DrivableAreaDetector, MotionDetector, SurfaceHazardDetector
from utils import scan_droidcam, list_local_cameras
from controllers import LongitudinalController, LateralController

class ThreatLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class ThreatAnalyzer:
    """AI Logic layer to filter and prioritize alerts"""
    def __init__(self):
        self.last_alert_type = None
        self.last_alert_time = 0
        self.alert_in_progress = False
        
    def analyze(self, results, frame_width):
        if not results: return None, ThreatLevel.NONE, None
        
        candidates = []
        for res in results:
            box = res['box']
            dist = res['distance']
            ttc = res['ttc']
            vx = res.get('vx', 0)
            mid_x = (box[0] + box[2]) / 2
            
            # Spatial logic: Is it in my path?
            is_ahead = (frame_width * 0.3) < mid_x < (frame_width * 0.7)
            
            # Intent logic: Is it cutting in?
            # If object is on left (mid_x < 0.3) and moving right (vx > 5)
            is_cutting_in = (mid_x < frame_width * 0.3 and vx > 5) or \
                            (mid_x > frame_width * 0.7 and vx < -5)
            
            score = 0
            level = ThreatLevel.NONE
            alert_text = ""
            
            if ttc < 1.8 and (is_ahead or is_cutting_in):
                level = ThreatLevel.URGENT
                alert_text = "BRAKE NOW" if ttc < 1.0 else "Emergency: Collision danger"
                score = 100 / (ttc + 0.01)
            elif is_cutting_in and dist < 12:
                level = ThreatLevel.HIGH
                alert_text = "Vehicle cutting in"
                score = 80 - dist
            elif is_ahead and ttc < 3.5:
                level = ThreatLevel.MEDIUM
                alert_text = f"Watch {res['class_id']} ahead"
                score = 60 - ttc*10
            elif is_ahead and dist < 5:
                level = ThreatLevel.LOW
                alert_text = "Close proximity"
                score = 40 - dist
                
            if level > ThreatLevel.NONE:
                candidates.append((score, level, alert_text, res.get('id')))
        
        if not candidates: return None, ThreatLevel.NONE, None
        
        # Pick the most dangerous candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2], candidates[0][1], candidates[0][3]

class VoiceAlertThread(QThread):
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self._run_flag = True
        self.last_alert_time = {} # {alert_text: timestamp}
        
    def run(self):
        engine = pyttsx3.init()
        # Improve voice quality
        engine.setProperty('rate', 160) # Slightly faster but clear
        engine.setProperty('volume', 1.0)
        
        while self._run_flag:
            if self.queue:
                text = self.queue.popleft()
                engine.say(text)
                engine.runAndWait()
            else:
                time.sleep(0.1)

    def speak(self, text, cooldown=5, object_id=None):
        """Speaks alert with a smart cooldown to prevent spam.
        If object_id is provided, tracking is per-object.
        """
        now = time.time()
        key = f"{text}_{object_id}" if object_id is not None else text
        
        if key not in self.last_alert_time or (now - self.last_alert_time[key]) > cooldown:
            # Check if similar text is already in queue to avoid repetition
            if not any(text in item for item in self.queue):
                self.queue.append(text)
                self.last_alert_time[key] = now

    def stop(self):
        self._run_flag = False
        self.wait()

class VideoCaptureThread(threading.Thread):
    """Dedicated thread for non-blocking camera I/O"""
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.ret = False
        self.frame = None
        self.running = True
        
        print(f"DEBUG: Attempting to open camera: {source}")
        
        # Strategy: Try specific backends for better compatibility
        if isinstance(source, int):
            # Try V4L2 first on Linux, then default
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
        elif isinstance(source, str) and source.startswith("http"):
            # IP Camera
            self.cap = cv2.VideoCapture(source)
        else:
            # Fallback
            self.cap = cv2.VideoCapture(source)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"DEBUG: Camera {source} opened successfully")
        else:
            print(f"DEBUG: CRITICAL - Failed to open camera {source}")

    def run(self):
        retry_count = 0
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                retry_count += 1
                # Try to reconnect every 2 seconds if connection is lost
                if retry_count > 20:
                    self.cap.release()
                    time.sleep(2.0)
                    self.cap = cv2.VideoCapture(self.source)
                    retry_count = 0
                else:
                    time.sleep(0.1)
            else:
                retry_count = 0

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    metrics_signal = pyqtSignal(float, int, str)
    error_signal = pyqtSignal(str)
    voice_alert_signal = pyqtSignal(str, int, object) # text, cooldown, object_id (can be None)

    def __init__(self, camera_index=0, mode="ADAS", vehicle_mode="Scooter"):
        super().__init__()
        self.camera_index = camera_index
        self.mode = mode
        self.vehicle_mode = vehicle_mode
        self._run_flag = True
        
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        
        self.detector = None
        self.drivable_detector = None
        self.motion_detector = None
        self.hazard_detector = None
        self.threat_analyzer = ThreatAnalyzer()
        self.avg_fps_val = 0.0
        
        # Virtual Actuators (Level 1)
        self.lon_controller = LongitudinalController()
        self.lat_controller = LateralController()
        self.virtual_steering = 0.0
        self.virtual_accel = 0.0
        self.last_time = time.time()
        
        # UDP Broadcast for Vehicle Control ECU (Level 1)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_target = ('127.0.0.1', 5005)
        
        if mode == "ADAS":
            self.drivable_detector = DrivableAreaDetector(vehicle_mode=self.vehicle_mode)
            self.hazard_detector = SurfaceHazardDetector()
            model_path = "yolov8s.pt"
            if os.path.exists(model_path):
                try:
                    self.detector = ObjectDetector(model_path)
                except Exception as e:
                    print(f"Error loading model: {e}")
            self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                           5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign',
                           15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant'}
            self.last_results = []
            
        elif mode == "BLIND_SPOT":
            self.motion_detector = MotionDetector()

    def inference_loop(self):
        """Dedicated thread for AI inference to leverage GPU without blocking UI"""
        frame_count = 0
        while self._run_flag:
            try:
                frame = self.frame_queue.get(timeout=1)
                frame_count += 1
                
                # 1. Run Object Detection every 2nd frame
                obj_results = self.last_results
                if frame_count % 2 == 0:
                    if self.mode == "ADAS" and self.detector:
                        # Use last known mask for spatial filtering
                        mask = self.drivable_detector.last_mask if self.drivable_detector else None
                        obj_results = self.detector.detect(frame, drivable_mask=mask)
                
                # 2. Run Drivable Area Detection EVERY frame
                da_mask = None
                da_status = "Searching..."
                if self.mode == "ADAS" and self.drivable_detector:
                    obj_boxes = [r['box'] for r in obj_results]
                    da_mask, da_status = self.drivable_detector.detect(frame, obj_boxes)
                
                # 3. Run Surface Hazard Detection
                hazards = []
                if self.mode == "ADAS" and self.hazard_detector:
                    # Use internal mask from detector
                    mask = self.drivable_detector.last_mask if self.drivable_detector else None
                    hazards = self.hazard_detector.detect(frame, drivable_mask=mask)
                
                if not self.result_queue.full():
                    self.result_queue.put((obj_results, da_mask, da_status, hazards))
                
                self.frame_queue.task_done()
            except Empty:
                continue

    def draw_glass_panel(self, frame, x, y, w, h, color=(40, 40, 40), alpha=0.6):
        """Draws a semi-transparent glass panel for HUD elements"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    def draw_targeting_brackets(self, frame, box, color, label, dist, ttc):
        """Draws modern corner brackets instead of full boxes"""
        x1, y1, x2, y2 = box
        l = int(min(x2-x1, y2-y1) * 0.2) # Bracket length
        t = 2 # Thickness
        
        # Corners
        cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
        cv2.line(frame, (x2, y1), (x2-l, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1+l), color, t)
        cv2.line(frame, (x1, y2), (x1+l, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2-l), color, t)
        cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2-l), color, t)

        # Label pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{label.upper()} {dist}m"
        (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
        self.draw_glass_panel(frame, x1, y1-th-10, tw+10, th+8, color=(0,0,0), alpha=0.5)
        cv2.putText(frame, text, (x1+5, y1-7), font, 0.4, (255, 255, 255), 1)

    def draw_virtual_hud(self, frame, steering, accel):
        height, width = frame.shape[:2]
        
        # 1. Telemetry Panel (Top Left)
        self.draw_glass_panel(frame, 15, 15, 200, 95, alpha=0.4)
        gpu_status = "ON" if self.detector and self.detector.device == 'cuda' else "OFF"
        gpu_color = (0, 255, 0) if gpu_status == "ON" else (100, 100, 100)
        
        cv2.putText(frame, f"SYSTEM: AI-ACTIVE", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"MODE: {self.mode}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"GPU: {gpu_status}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gpu_color, 1)
        cv2.putText(frame, f"FPS: {self.avg_fps_val:.1f}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. Control Dial (Bottom Center)
        cx, cy = width // 2, height - 60
        r = 45
        angle = steering * 40
        import math
        
        # Steering Wheel Glow
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -150, -30, (60, 60, 60), 1)
        
        # Dynamic steering needle
        nx = int(cx + r * math.cos(math.radians(angle - 90)))
        ny = int(cy + r * math.sin(math.radians(angle - 90)))
        cv2.line(frame, (cx, cy), (nx, ny), (0, 255, 255), 2)
        cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)
        
        # Center marker
        cv2.line(frame, (cx, cy-r-5), (cx, cy-r+5), (255, 255, 255), 1)

        # 3. Acceleration / Braking Vertical Gauges
        bx = width - 45
        by = height - 150
        bw, bh = 12, 100
        
        # Background track
        self.draw_glass_panel(frame, bx-5, by-10, bw+10, bh+20, alpha=0.3)
        
        if accel > 0: # Throttle (Cyan/Green)
            h = int(accel * bh)
            cv2.rectangle(frame, (bx, by + bh - h), (bx + bw, by + bh), (255, 255, 0), -1)
            cv2.putText(frame, "PWR", (bx-30, by+bh), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        else: # Brake (Red/Orange)
            h = int(abs(accel) * bh)
            cv2.rectangle(frame, (bx, by + bh - h), (bx + bw, by + bh), (0, 0, 255), -1)
            cv2.putText(frame, "BRK", (bx-30, by+bh), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def run(self):
        source = self.camera_index
        # Try to convert to int if it's a numeric string
        if isinstance(source, str) and not source.startswith("http"):
            try:
                source = int(source)
            except:
                pass
        
        self.cap_thread = VideoCaptureThread(source)
        self.cap_thread.start()
        time.sleep(0.5) # Give it a moment to start

        if not self.cap_thread.ret:
            self.error_signal.emit(f"Failed to open camera {source}")
            self.cap_thread.stop()
            return

        # Start the inference thread
        inf_thread = threading.Thread(target=self.inference_loop, daemon=True)
        inf_thread.start()

        fps_avg = deque(maxlen=30)
        frame_idx = 0
        last_metric_time = time.time()
        
        # Persistence memory for fluid HUD
        last_da_mask = None
        last_hazards = []
        
        while self._run_flag:
            ret, frame = self.cap_thread.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame_idx += 1
            
            # Accurate FPS: Measure time for 30 frames and divide
            if frame_idx % 30 == 0:
                now = time.time()
                elapsed = now - last_metric_time
                self.avg_fps_val = 30.0 / elapsed if elapsed > 0 else 0
                last_metric_time = now

            lane_status = "N/A"
            # 1. Submit frame for object detection
            if self.mode == "ADAS":
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Try to get the latest combined results (4-element tuple)
                try:
                    self.last_results, new_mask, da_status, hazards = self.result_queue.get_nowait()
                    if new_mask is not None: last_da_mask = new_mask
                    last_hazards = hazards
                    lane_status = da_status
                except Empty:
                    pass
                
                # 2. Apply Persistent Drivable Area Overlay (Fluid 30+ FPS)
                if self.drivable_detector and last_da_mask is not None:
                    frame = self.drivable_detector.draw_overlay(frame, last_da_mask)

                # 3. Consolidated AI Threat Analysis
                alert_text, threat_level, threat_id = self.threat_analyzer.analyze(self.last_results, frame.shape[1])
                if alert_text:
                    cooldown = 15 if threat_level <= ThreatLevel.LOW else (8 if threat_level <= ThreatLevel.MEDIUM else 3)
                    self.voice_alert_signal.emit(alert_text, cooldown, threat_id)

                # 4. Draw Object Overlays
                for res in self.last_results:
                    box, dist, ttc, cls_id, obj_id = res['box'], res['distance'], res['ttc'], res['class_id'], res.get('id', '')
                    label = self.classes.get(cls_id, 'object')
                    color = (0, 255, 0)
                    if res.get('id') == threat_id:
                        color = (0, 0, 255) if threat_level >= ThreatLevel.HIGH else (0, 255, 255)
                    
                    self.draw_targeting_brackets(frame, box, color, label, dist, ttc)

                # 5. Draw Surface Hazards
                for h_type, (hx, hy, hw, hh) in last_hazards:
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)
                    cv2.putText(frame, h_type, (hx, hy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    self.voice_alert_signal.emit(f"{h_type} ahead", 20, None)

            elif self.mode == "BLIND_SPOT" and self.motion_detector:
                frame, motion = self.motion_detector.detect(frame)
                if motion:
                    self.voice_alert_signal.emit("Vehicle in blind spot", 10, None)
                lane_status = "Blind Spot Active"
            
            elif self.mode == "DRIVER":
                frame = cv2.flip(frame, 1)
                lane_status = "Mirror View"

            # (FPS metrics handled at the start of loop for accuracy)

            # 6. Virtual Actuators (Level 1 Control Simulation)
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            
            # Steering
            if self.drivable_detector and self.drivable_detector.last_path_center_x is not None:
                self.virtual_steering = self.lat_controller.calculate(frame.shape[1]//2, 
                                                                    self.drivable_detector.last_path_center_x, dt)
            
            # Speed/Brake
            threat_ttc, threat_dist = 99.0, 99.0
            if threat_id is not None:
                for res in self.last_results:
                    if res.get('id') == threat_id:
                        threat_ttc = res['ttc']; threat_dist = res['distance']
                        break
            
            self.virtual_accel = self.lon_controller.calculate(threat_ttc, threat_dist, dt)
            self.draw_virtual_hud(frame, self.virtual_steering, self.virtual_accel)

            # 7. Broadcast Data for Separate Control ECU
            try:
                packet = {
                    "mode": self.mode,
                    "path_center_x": self.drivable_detector.last_path_center_x if self.drivable_detector else None,
                    "frame_w": frame.shape[1],
                    "frame_h": frame.shape[0],
                    "threat_id": threat_id,
                    "threats": [
                        {
                            "id": r.get('id'),
                            "dist": r.get('distance'),
                            "ttc": r.get('ttc'),
                            "vx": r.get('vx', 0),
                            "cls": self.classes.get(r.get('class_id', 0), 'obj'),
                            "box": [int(v) for v in r.get('box', [])]
                        } for r in self.last_results
                    ],
                    "hazards": last_hazards,
                    "recommended_speed": self.drivable_detector.recommended_speed if self.drivable_detector else "N/A"
                }
                self.udp_sock.sendto(json.dumps(packet).encode(), self.udp_target)
            except Exception as e:
                print(f"UDP Broadcast error: {e}")

            self.change_pixmap_signal.emit(frame)
            self.metrics_signal.emit(self.avg_fps_val, len(self.last_results), lane_status)

    def stop(self):
        self._run_flag = False
        if hasattr(self, 'cap_thread'):
            self.cap_thread.stop()
        self.wait()

class ADASMainWindow(QMainWindow):
    new_ip_cam_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astra Perception: AI Lens")
        self.setMinimumSize(1200, 800)
        self.setup_theme()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.threads = {}
        self.camera_combos = {}
        
        # Voice alert system
        self.voice_thread = VoiceAlertThread()
        self.voice_thread.start()
        
        # Grid for cameras
        self.video_grid = QGridLayout()
        self.main_video = QLabel("<b>PERCEPTION LENS</b><br><small>OFFLINE</small>")
        self.side_video = QLabel("<b>SIDE RADAR</b><br><small>OFFLINE</small>")
        self.rear_video = QLabel("<b>REAR VIEW</b><br><small>OFFLINE</small>")
        
        for v in [self.main_video, self.side_video, self.rear_video]:
            v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setStyleSheet("""
                background-color: #050507; 
                color: #333; 
                border: 2px solid #1A1A1F; 
                border-radius: 12px;
                font-size: 16px;
            """)
            v.setMinimumSize(400, 300)

        self.video_grid.addWidget(self.main_video, 0, 0, 2, 2)
        self.video_grid.addWidget(self.side_video, 0, 2)
        self.video_grid.addWidget(self.rear_video, 1, 2)
        
        self.main_layout.addLayout(self.video_grid, stretch=3)
        
        self.control_layout = QVBoxLayout()
        global_group = QGroupBox("System Profile")
        global_layout = QFormLayout()
        self.vehicle_profile = QComboBox()
        self.vehicle_profile.addItems(["Scooter", "Car"])
        global_layout.addRow("Vehicle Type:", self.vehicle_profile)
        global_group.setLayout(global_layout)
        self.control_layout.addWidget(global_group)

        self.setup_camera_controls("Main ADAS", 0, "ADAS", self.main_video)
        self.setup_camera_controls("Side (Blind Spot)", 1, "BLIND_SPOT", self.side_video)
        self.setup_camera_controls("Rear (Mirror)", 2, "DRIVER", self.rear_video)
        self.control_layout.addStretch()
        self.main_layout.addLayout(self.control_layout, stretch=1)
        
        self.new_ip_cam_signal.connect(self.add_ip_camera)
        threading.Thread(target=self.run_auto_scan, daemon=True).start()

    def run_auto_scan(self):
        local_cams = list_local_cameras()
        if local_cams:
            for combo in self.camera_combos.values():
                current = combo.currentText()
                if not current.startswith("http"):
                    combo.clear()
                    combo.addItems(local_cams)
                    if current in local_cams: combo.setCurrentText(current)
        
        # Robust scanning for DroidCam and IP Cams
        scan_droidcam(lambda url: self.new_ip_cam_signal.emit(url))

    @pyqtSlot(str)
    def add_ip_camera(self, url):
        # Add to all relevant camera selectors
        for name, combo in self.camera_combos.items():
            if combo.findText(url) == -1:
                combo.addItem(url)
                # Auto-select if the corresponding view is currently offline
                label = self.main_video if "Main" in name else (self.side_video if "Side" in name else self.rear_video)
                if "OFFLINE" in label.text().upper():
                    combo.setCurrentText(url)

    def setup_camera_controls(self, name, default_idx, mode, label):
        group = QGroupBox(name)
        layout = QFormLayout()
        combo = QComboBox()
        if mode == "ADAS":
            combo.setEditable(True)
            combo.setPlaceholderText("Index or http://IP:PORT/video")
        combo.addItems([str(i) for i in range(11)])
        combo.setCurrentIndex(default_idx if default_idx < combo.count() else 0)
        self.camera_combos[name] = combo
        fps_lbl, status_lbl = QLabel("0.0"), QLabel("READY")
        status_lbl.setStyleSheet("color: orange;")
        start_btn, stop_btn = QPushButton("Enable"), QPushButton("Disable")
        stop_btn.setEnabled(False)
        start_btn.clicked.connect(lambda: self.start_cam(name, combo, mode, label, start_btn, stop_btn, fps_lbl, status_lbl))
        stop_btn.clicked.connect(lambda: self.stop_cam(name, combo, start_btn, stop_btn, label, fps_lbl, status_lbl))
        layout.addRow("Index/URL:", combo)
        layout.addRow(start_btn, stop_btn)
        layout.addRow("FPS:", fps_lbl)
        layout.addRow("Status:", status_lbl)
        group.setLayout(layout)
        self.control_layout.addWidget(group)

    def start_cam(self, name, combo, mode, label, btn_on, btn_off, fps_lbl, status_lbl):
        source = combo.currentText()
        v_profile = self.vehicle_profile.currentText()
        thread = CameraThread(source, mode, vehicle_mode=v_profile)
        thread.change_pixmap_signal.connect(lambda img: self.update_image(label, img))
        thread.metrics_signal.connect(lambda fps, objs, stat: self.update_metrics(fps, stat, fps_lbl, status_lbl))
        thread.error_signal.connect(lambda err: self.handle_single_cam_error(name, label, status_lbl, btn_on, btn_off, err))
        thread.voice_alert_signal.connect(self.voice_thread.speak)
        thread.start()
        self.threads[name] = thread
        btn_on.setEnabled(False)
        btn_off.setEnabled(True)

    def handle_single_cam_error(self, name, label, status_lbl, btn_on, btn_off, err_msg):
        self.stop_cam(name, None, btn_on, btn_off, label, None, status_lbl)
        label.setText(f"ERROR: {err_msg}")
        status_lbl.setText("Camera Unavailable")
        status_lbl.setStyleSheet("color: red;")

    def stop_cam(self, name, combo, btn_on, btn_off, label, fps_lbl, status_lbl):
        if name in self.threads:
            self.threads[name].stop()
            del self.threads[name]
        btn_on.setEnabled(True)
        btn_off.setEnabled(False)
        label.clear()
        label.setText(f"{name} Offline")
        if fps_lbl: fps_lbl.setText("0.0")
        if status_lbl:
            status_lbl.setText("Offline")
            status_lbl.setStyleSheet("color: black;")

    def update_image(self, label, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(qt_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(p)

    def update_metrics(self, fps, status, fps_lbl, status_lbl):
        fps_lbl.setText(f"{fps:.1f}")
        status_lbl.setText(status)
        if "Warning" in status or "Departure" in status or "Brake" in status:
            status_lbl.setStyleSheet("color: red; font-weight: bold;")
        elif "OK" in status:
            status_lbl.setStyleSheet("color: green;")
        else:
            status_lbl.setStyleSheet("color: orange;")

    def setup_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A0A0C;
            }
            QWidget {
                background-color: #0A0A0C;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QGroupBox {
                border: 1px solid #2A2A2E;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                color: #00FFFF;
                background-color: #121216;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                background-color: transparent;
            }
            QPushButton {
                background-color: #1F1F23;
                color: #E0E0E0;
                border: 1px solid #3A3A3F;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2A2A2E;
                border: 1px solid #00FFFF;
            }
            QPushButton:disabled {
                background-color: #0D0D0F;
                color: #444;
            }
            QComboBox {
                background-color: #1F1F23;
                border: 1px solid #3A3A3F;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
        """)

    def closeEvent(self, event):
        for t in self.threads.values(): t.stop()
        self.voice_thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ADASMainWindow()
    window.show()
    sys.exit(app.exec())
