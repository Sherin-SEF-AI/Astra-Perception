import sys
import os

# Silence OpenCV and UI warnings for a professional console experience
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import time
import cv2
import numpy as np
import threading
import pyttsx3
import json
import socket
import math
import logging
from collections import deque
from enum import IntEnum
from queue import Queue, Empty

# Silence Ultralytics/PyTorch noise
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QFormLayout, QGridLayout,
    QCheckBox, QTextEdit
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
            
            is_ahead = (frame_width * 0.3) < mid_x < (frame_width * 0.7)
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
                alert_text = f"Watch object ahead"
                score = 60 - ttc*10
            elif is_ahead and dist < 5:
                level = ThreatLevel.LOW
                alert_text = "Close proximity"
                score = 40 - dist
                
            if level > ThreatLevel.NONE:
                candidates.append((score, level, alert_text, res.get('id')))
        
        if not candidates: return None, ThreatLevel.NONE, None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2], candidates[0][1], candidates[0][3]

class VoiceAlertThread(QThread):
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self._run_flag = True
        self.last_alert_time = {}
        
    def run(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        while self._run_flag:
            if self.queue:
                text = self.queue.popleft()
                try:
                    engine.say(text)
                    engine.runAndWait()
                except:
                    pass
            else:
                time.sleep(0.1)

    def speak(self, text, cooldown=5, object_id=None):
        now = time.time()
        key = f"{text}_{object_id}" if object_id is not None else text
        if key not in self.last_alert_time or (now - self.last_alert_time[key]) > cooldown:
            if not any(text in item for item in self.queue):
                self.queue.append(text)
                self.last_alert_time[key] = now

    def stop(self):
        self._run_flag = False
        self.wait()

class VideoCaptureThread(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.ret = False
        self.frame = None
        self.running = True
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(source)
        elif isinstance(source, str) and source.startswith("http"):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self):
        retry_count = 0
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                retry_count += 1
                if retry_count > 20:
                    self.cap.release(); time.sleep(2.0)
                    self.cap = cv2.VideoCapture(self.source)
                    retry_count = 0
                else: time.sleep(0.1)
            else: retry_count = 0

    def read(self): return self.ret, self.frame
    def stop(self): self.running = False; self.cap.release()

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    metrics_signal = pyqtSignal(float, int, str)
    error_signal = pyqtSignal(str)
    voice_alert_signal = pyqtSignal(str, int, object)
    log_signal = pyqtSignal(str)

    def __init__(self, camera_index=0, mode="ADAS", vehicle_mode="Scooter"):
        super().__init__()
        self.camera_index = camera_index; self.mode = mode; self.vehicle_mode = vehicle_mode
        self._run_flag = True
        self.enable_objects = False; self.enable_drivable = True; self.enable_hazards = False
        self.frame_queue = Queue(maxsize=2); self.result_queue = Queue(maxsize=2)
        self.detector = None; self.drivable_detector = None; self.motion_detector = None
        self.hazard_detector = None; self.threat_analyzer = ThreatAnalyzer()
        self.avg_fps_val = 0.0; self.last_results = []
        self.lon_controller = LongitudinalController(); self.lat_controller = LateralController()
        self.virtual_steering = 0.0; self.virtual_accel = 0.0; self.last_time = time.time()
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.udp_target = ('127.0.0.1', 5005)
        
        if mode == "ADAS":
            self.drivable_detector = DrivableAreaDetector(vehicle_mode=self.vehicle_mode)
            self.hazard_detector = SurfaceHazardDetector()
            model_path = "yolov8s.pt"
            if os.path.exists(model_path):
                try: self.detector = ObjectDetector(model_path)
                except Exception as e: print(f"Error loading model: {e}")
            self.classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant'}
        elif mode == "BLIND_SPOT": self.motion_detector = MotionDetector()

    def inference_loop(self):
        frame_count = 0
        while self._run_flag:
            try:
                frame = self.frame_queue.get(timeout=1); frame_count += 1
                obj_results = []
                if self.enable_objects and self.detector and frame_count % 2 == 0:
                    obj_results = self.detector.detect(frame)
                    if obj_results: self.log_signal.emit(f"Vision: {len(obj_results)} entities tracked")
                elif not self.enable_objects: obj_results = []
                else: obj_results = self.last_results

                da_mask = None; da_status = "AI Standby"
                if self.enable_drivable and self.drivable_detector:
                    obj_boxes = [r['box'] for r in obj_results]
                    da_mask, da_status = self.drivable_detector.detect(frame, obj_boxes)
                
                hazards = []
                if self.enable_hazards and self.hazard_detector:
                    mask = self.drivable_detector.last_mask if self.drivable_detector else None
                    hazards = self.hazard_detector.detect(frame, drivable_mask=mask)
                    if hazards: self.log_signal.emit(f"Alert: {hazards[0][0]} detected")
                
                if not self.result_queue.full(): self.result_queue.put((obj_results, da_mask, da_status, hazards))
                self.frame_queue.task_done()
            except Empty: continue

    def draw_glass_panel(self, frame, x, y, w, h, color=(40, 40, 40), alpha=0.6):
        overlay = frame.copy(); cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    def draw_targeting_brackets(self, frame, box, color, label, dist, ttc):
        x1, y1, x2, y2 = box; l = int(min(x2-x1, y2-y1) * 0.2)
        cv2.line(frame, (x1, y1), (x1+l, y1), color, 2); cv2.line(frame, (x1, y1), (x1, y1+l), color, 2)
        cv2.line(frame, (x2, y1), (x2-l, y1), color, 2); cv2.line(frame, (x2, y1), (x2, y1+l), color, 2)
        cv2.line(frame, (x1, y2), (x1+l, y2), color, 2); cv2.line(frame, (x1, y2), (x1, y2-l), color, 2)
        cv2.line(frame, (x2, y2), (x2-l, y2), color, 2); cv2.line(frame, (x2, y2), (x2, y2-l), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX; text = f"{label.upper()} {dist}m"; (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
        self.draw_glass_panel(frame, x1, y1-th-10, tw+10, th+8, color=(0,0,0), alpha=0.5)
        cv2.putText(frame, text, (x1+5, y1-7), font, 0.4, (255, 255, 255), 1)

    def draw_virtual_hud(self, frame, steering, accel):
        h, w = frame.shape[:2]; self.draw_glass_panel(frame, 15, 15, 200, 95, alpha=0.4)
        gpu = "ON" if self.detector and self.detector.device == 'cuda' else "OFF"
        gc = (0, 255, 0) if gpu == "ON" else (100, 100, 100)
        cv2.putText(frame, f"SYSTEM: AI-ACTIVE", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"MODE: {self.mode}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"GPU: {gpu}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 1)
        cv2.putText(frame, f"FPS: {self.avg_fps_val:.1f}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cx, cy, r, angle = w // 2, h - 60, 45, steering * 40
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -150, -30, (60, 60, 60), 1)
        nx = int(cx + r * math.cos(math.radians(angle - 90)))
        ny = int(cy + r * math.sin(math.radians(angle - 90)))
        cv2.line(frame, (cx, cy), (nx, ny), (0, 255, 255), 2); cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)
        cv2.line(frame, (cx, cy-r-5), (cx, cy-r+5), (255, 255, 255), 1)
        bx, by, bw, bh = w - 45, h - 150, 12, 100; self.draw_glass_panel(frame, bx-5, by-10, bw+10, bh+20, alpha=0.3)
        if accel > 0:
            ah = int(accel * bh); cv2.rectangle(frame, (bx, by + bh - ah), (bx + bw, by + bh), (255, 255, 0), -1)
        else:
            ah = int(abs(accel) * bh); cv2.rectangle(frame, (bx, by + bh - ah), (bx + bw, by + bh), (0, 0, 255), -1)

    def run(self):
        source = self.camera_index
        if isinstance(source, str) and not source.startswith("http"):
            try: source = int(source)
            except: pass
        self.cap_thread = VideoCaptureThread(source); self.cap_thread.start(); time.sleep(0.5)
        if not self.cap_thread.ret: self.error_signal.emit(f"Failed: {source}"); self.cap_thread.stop(); return
        threading.Thread(target=self.inference_loop, daemon=True).start()
        frame_idx, last_metric_time, last_da_mask, last_hazards = 0, time.time(), None, []
        while self._run_flag:
            ret, frame = self.cap_thread.read()
            if not ret or frame is None: time.sleep(0.01); continue
            frame_idx += 1
            if frame_idx % 30 == 0:
                now = time.time(); self.avg_fps_val = 30.0 / (now - last_metric_time); last_metric_time = now
            if frame_idx % 2 != 0: self.change_pixmap_signal.emit(frame); continue
            lane_status = "N/A"; tid = None
            if self.mode == "ADAS":
                if not self.frame_queue.full(): self.frame_queue.put(frame.copy())
                try:
                    self.last_results, new_mask, da_status, hazards = self.result_queue.get_nowait()
                    if new_mask is not None: last_da_mask = new_mask
                    last_hazards, lane_status = hazards, da_status
                except Empty: pass
                if self.drivable_detector and last_da_mask is not None: frame = self.drivable_detector.draw_overlay(frame, last_da_mask)
                alert, level, tid = self.threat_analyzer.analyze(self.last_results, frame.shape[1])
                if alert: self.voice_alert_signal.emit(alert, 15 if level <= ThreatLevel.LOW else (8 if level <= ThreatLevel.MEDIUM else 3), tid)
                for r in self.last_results:
                    box, cid, oid = r['box'], r['class_id'], r.get('id', '')
                    label = self.classes.get(cid, 'object'); color = (0, 255, 0)
                    if r.get('id') == tid: color = (0, 0, 255) if level >= ThreatLevel.HIGH else (0, 255, 255)
                    self.draw_targeting_brackets(frame, box, color, label, r['distance'], r['ttc'])
                for h_type, (hx, hy, hw, hh) in last_hazards:
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)
                    self.voice_alert_signal.emit(f"{h_type} ahead", 20, None)
            elif self.mode == "BLIND_SPOT" and self.motion_detector:
                frame, motion = self.motion_detector.detect(frame)
                if motion: self.voice_alert_signal.emit("Vehicle in blind spot", 10, None)
                lane_status = "Blind Spot Active"
            elif self.mode == "DRIVER": frame = cv2.flip(frame, 1); lane_status = "Mirror View"
            now = time.time(); dt = now - self.last_time; self.last_time = now
            if self.drivable_detector and self.drivable_detector.last_path_center_x is not None:
                self.virtual_steering = self.lat_controller.calculate(frame.shape[1]//2, self.drivable_detector.last_path_center_x, dt)
            tttc, tdist = 99.0, 99.0
            if tid is not None:
                for r in self.last_results:
                    if r.get('id') == tid: tttc, tdist = r['ttc'], r['distance']; break
            self.virtual_accel = self.lon_controller.calculate(tttc, tdist, dt); self.draw_virtual_hud(frame, self.virtual_steering, self.virtual_accel)
            try:
                packet = { "mode": self.mode, "path_center_x": self.drivable_detector.last_path_center_x if self.drivable_detector else None, "frame_w": frame.shape[1], "frame_h": frame.shape[0], "threat_id": tid, "threats": [{ "id": r.get('id'), "dist": r.get('distance'), "ttc": r.get('ttc'), "vx": r.get('vx', 0), "cls": self.classes.get(r.get('class_id', 0), 'obj'), "box": [int(v) for v in r.get('box', [])] } for r in self.last_results], "hazards": last_hazards, "recommended_speed": self.drivable_detector.recommended_speed if self.drivable_detector else "N/A" }
                self.udp_sock.sendto(json.dumps(packet).encode(), self.udp_target)
            except: pass
            self.change_pixmap_signal.emit(frame); self.metrics_signal.emit(self.avg_fps_val, len(self.last_results), lane_status)

    def stop(self):
        self._run_flag = False
        if hasattr(self, 'cap_thread'): self.cap_thread.stop()
        self.wait()

class ADASMainWindow(QMainWindow):
    new_ip_cam_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astra Perception: AI Lens"); self.setMinimumSize(1200, 950); self.setup_theme()
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.upper_layout = QHBoxLayout(); self.main_layout.addLayout(self.upper_layout, stretch=6)
        self.threads = {}; self.camera_combos = {}
        self.voice_thread = VoiceAlertThread(); self.voice_thread.start()
        self.video_grid = QGridLayout()
        self.main_video = QLabel("<b>PERCEPTION LENS</b><br><small>OFFLINE</small>")
        self.side_video = QLabel("<b>SIDE RADAR</b><br><small>OFFLINE</small>")
        self.rear_video = QLabel("<b>REAR VIEW</b><br><small>OFFLINE</small>")
        for v in [self.main_video, self.side_video, self.rear_video]:
            v.setAlignment(Qt.AlignmentFlag.AlignCenter); v.setStyleSheet("background-color: #050507; color: #333; border: 2px solid #1A1A1F; border-radius: 12px; font-size: 16px;")
            v.setMinimumSize(400, 300)
        self.video_grid.addWidget(self.main_video, 0, 0, 2, 2); self.video_grid.addWidget(self.side_video, 0, 2); self.video_grid.addWidget(self.rear_video, 1, 2)
        self.upper_layout.addLayout(self.video_grid, stretch=3)
        self.control_layout = QVBoxLayout()
        global_group = QGroupBox("System Profile"); global_layout = QFormLayout()
        self.vehicle_profile = QComboBox(); self.vehicle_profile.addItems(["Scooter", "Car"])
        global_layout.addRow("Vehicle Type:", self.vehicle_profile); global_group.setLayout(global_layout); self.control_layout.addWidget(global_group)
        self.setup_camera_controls("Main ADAS", 0, "ADAS", self.main_video)
        self.setup_camera_controls("Side Radar", 1, "BLIND_SPOT", self.side_video)
        self.setup_camera_controls("Rear View", 2, "DRIVER", self.rear_video)
        self.control_layout.addStretch(); self.upper_layout.addLayout(self.control_layout, stretch=1)
        self.event_feed = QTextEdit(); self.event_feed.setReadOnly(True)
        self.event_feed.setStyleSheet("background-color: #050507; color: #00FF00; border: 1px solid #1A1A1F; font-family: 'Consolas', monospace; font-size: 11px; border-radius: 5px;")
        self.event_feed.setPlaceholderText(">> NEURAL EVENT FEED ACTIVE...")
        self.main_layout.addWidget(QLabel("SYSTEM LOGS"), stretch=0); self.main_layout.addWidget(self.event_feed, stretch=1)
        self.new_ip_cam_signal.connect(self.add_ip_camera)
        threading.Thread(target=self.run_auto_scan, daemon=True).start()

    def run_auto_scan(self):
        local_cams = list_local_cameras()
        if local_cams:
            for combo in self.camera_combos.items():
                current = combo[1].currentText()
                if not current.startswith("http"):
                    combo[1].clear(); combo[1].addItems(local_cams)
                    if current in local_cams: combo[1].setCurrentText(current)
        scan_droidcam(lambda url: self.new_ip_cam_signal.emit(url))

    @pyqtSlot(str)
    def add_ip_camera(self, url):
        for name, combo in self.camera_combos.items():
            if combo[1].findText(url) == -1:
                combo[1].addItem(url)
                label = self.main_video if "Main" in name else (self.side_video if "Side" in name else self.rear_video)
                if "OFFLINE" in label.text().upper(): combo[1].setCurrentText(url)

    def setup_camera_controls(self, name, default_idx, mode, label):
        group = QGroupBox(name); layout = QVBoxLayout(); combo = QComboBox()
        if mode == "ADAS": combo.setEditable(True); combo.setPlaceholderText("Index or URL")
        combo.addItems([str(i) for i in range(11)]); combo.setCurrentIndex(default_idx if default_idx < combo.count() else 0)
        self.camera_combos[name] = (mode, combo)
        status_row = QHBoxLayout(); status_lbl = QLabel("READY"); status_lbl.setStyleSheet("color: orange;"); fps_lbl = QLabel("0.0")
        status_row.addWidget(status_lbl); status_row.addStretch(); status_row.addWidget(QLabel("FPS:")); status_row.addWidget(fps_lbl); layout.addLayout(status_row)
        cb_obj, cb_path, cb_haz = None, None, None
        if mode == "ADAS":
            toggles = QHBoxLayout(); cb_obj = QCheckBox("Vision"); cb_path = QCheckBox("Path"); cb_path.setChecked(True); cb_haz = QCheckBox("Surface")
            toggles.addWidget(cb_obj); toggles.addWidget(cb_path); toggles.addWidget(cb_haz); layout.addLayout(toggles)
        layout.addWidget(combo); btn_row = QHBoxLayout(); btn_on = QPushButton("ENABLE"); btn_off = QPushButton("DISABLE"); btn_off.setEnabled(False)
        btn_on.clicked.connect(lambda: self.start_cam(name, combo, mode, label, btn_on, btn_off, fps_lbl, status_lbl, cb_obj, cb_path, cb_haz))
        btn_off.clicked.connect(lambda: self.stop_cam(name, combo, btn_on, btn_off, label, fps_lbl, status_lbl))
        btn_row.addWidget(btn_on); btn_row.addWidget(btn_off); layout.addLayout(btn_row); group.setLayout(layout); self.control_layout.addWidget(group)

    def start_cam(self, name, combo, mode, label, btn_on, btn_off, fps_lbl, status_lbl, cb_obj, cb_path, cb_haz):
        source = combo.currentText(); v_profile = self.vehicle_profile.currentText()
        thread = CameraThread(source, mode, vehicle_mode=v_profile)
        if cb_obj:
            cb_obj.stateChanged.connect(lambda: setattr(thread, 'enable_objects', cb_obj.isChecked()))
            thread.enable_objects = cb_obj.isChecked()
        if cb_path:
            cb_path.stateChanged.connect(lambda: setattr(thread, 'enable_drivable', cb_path.isChecked()))
            thread.enable_drivable = cb_path.isChecked()
        if cb_haz:
            cb_haz.stateChanged.connect(lambda: setattr(thread, 'enable_hazards', cb_haz.isChecked()))
            thread.enable_hazards = cb_haz.isChecked()
        thread.change_pixmap_signal.connect(lambda img: self.update_image(label, img))
        thread.metrics_signal.connect(lambda fps, objs, stat: self.update_metrics(fps, stat, fps_lbl, status_lbl))
        thread.log_signal.connect(self.event_feed.append); thread.error_signal.connect(lambda err: self.handle_single_cam_error(name, label, status_lbl, btn_on, btn_off, err))
        thread.voice_alert_signal.connect(self.voice_thread.speak); thread.start(); self.threads[name] = thread; btn_on.setEnabled(False); btn_off.setEnabled(True)

    def handle_single_cam_error(self, name, label, status_lbl, btn_on, btn_off, err):
        self.stop_cam(name, None, btn_on, btn_off, label, None, status_lbl)
        label.setText(f"ERROR: {err}"); status_lbl.setText("Unavailable"); status_lbl.setStyleSheet("color: red;")

    def stop_cam(self, name, combo, btn_on, btn_off, label, fps_lbl, status_lbl):
        if name in self.threads: self.threads[name].stop(); del self.threads[name]
        btn_on.setEnabled(True); btn_off.setEnabled(False); label.clear(); label.setText(f"<b>{name.upper()}</b><br><small>OFFLINE</small>")
        if fps_lbl: fps_lbl.setText("0.0")
        if status_lbl: status_lbl.setText("READY"); status_lbl.setStyleSheet("color: orange;")

    def update_image(self, label, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(qt_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio); label.setPixmap(p)

    def update_metrics(self, fps, status, fps_lbl, status_lbl):
        fps_lbl.setText(f"{fps:.1f}"); status_lbl.setText(status)
        if any(x in status for x in ["Warning", "Departure", "Brake"]): status_lbl.setStyleSheet("color: red; font-weight: bold;")
        elif "OK" in status: status_lbl.setStyleSheet("color: green;")
        else: status_lbl.setStyleSheet("color: orange;")

    def setup_theme(self):
        self.setStyleSheet("""QMainWindow { background-color: #0A0A0C; } QWidget { background-color: #0A0A0C; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; } QGroupBox { border: 1px solid #2A2A2E; border-radius: 8px; margin-top: 15px; font-weight: bold; color: #00FFFF; background-color: #121216; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; } QPushButton { background-color: #1F1F23; color: #E0E0E0; border: 1px solid #3A3A3F; padding: 8px; border-radius: 5px; font-weight: bold; } QPushButton:hover { background-color: #2A2A2E; border: 1px solid #00FFFF; } QPushButton:disabled { background-color: #0D0D0F; color: #444; } QComboBox { background-color: #1F1F23; border: 1px solid #3A3A3F; border-radius: 4px; padding: 5px; color: white; } QCheckBox { spacing: 5px; } QCheckBox::indicator { width: 15px; height: 15px; }""")

    def closeEvent(self, event):
        for t in self.threads.values(): t.stop()
        self.voice_thread.stop(); super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv); window = ADASMainWindow(); window.show(); sys.exit(app.exec())
