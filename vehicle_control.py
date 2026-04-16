import sys
import os
import time
import json
import socket
import numpy as np
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QPolygonF

from controllers import LongitudinalController, LateralController

class ECUReceiver(QThread):
    packet_received = pyqtSignal(dict)
    
    def __init__(self, port=5005):
        super().__init__()
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', self.port))
        self.sock.settimeout(1.0)
        self._run_flag = True

    def run(self):
        while self._run_flag:
            try:
                data, _ = self.sock.recvfrom(4096)
                packet = json.loads(data.decode())
                self.packet_received.emit(packet)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"ECU Receiver Error: {e}")

    def stop(self):
        self._run_flag = False
        self.wait()

class RadarWidget(QWidget):
    """Sleek Cyberpunk Radar for vehicle control"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 250)
        self.threats = []
        self.hazards = []
        self.threat_id = None
        self.path_x = None
        self.frame_w = 640

    def update_data(self, threats, threat_id, path_x, frame_w, hazards=[]):
        self.threats = threats
        self.threat_id = threat_id
        self.path_x = path_x
        self.frame_w = frame_w
        self.hazards = hazards
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dark Background
        painter.fillRect(self.rect(), QColor(10, 10, 15))
        
        w, h = self.width(), self.height()
        cx, cy = w // 2, h - 40
        
        # Draw scanning arcs
        painter.setPen(QPen(QColor(0, 255, 255, 40), 1))
        for d in [10, 20, 30]:
            r = int(d * 6) # Scale
            painter.drawEllipse(cx - r, cy - r, r*2, r*2)
        
        # Center Line
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.drawLine(cx, 0, cx, h)

        # Draw Target Path (Glowy line)
        if self.path_x is not None:
            offset = (self.path_x - (self.frame_w/2)) / (self.frame_w/2)
            tx = cx + int(offset * 80)
            painter.setPen(QPen(QColor(0, 255, 150, 180), 2, Qt.PenStyle.SolidLine))
            painter.drawLine(cx, cy, tx, 20)

        # Draw Vehicle (Minimalist Triangle)
        painter.setBrush(QBrush(QColor(0, 180, 255)))
        painter.setPen(Qt.PenStyle.NoPen)
        poly = QPolygonF([QPointF(cx, cy-15), QPointF(cx-10, cy+5), QPointF(cx+10, cy+5)])
        painter.drawPolygon(poly)

        # Draw Hazards (Potholes/Bumps as rectangles)
        for h_type, box in self.hazards:
            hx = cx + int(((box[0] + box[2]/2) - (self.frame_w/2)) / (self.frame_w/2) * 80)
            # Map Y coordinate roughly to radar distance
            hy = cy - int((box[1] / 480) * 100) 
            
            painter.setBrush(QBrush(QColor(255, 0, 255, 150)))
            painter.drawRect(hx-10, hy-5, 20, 10)
            painter.setPen(QPen(QColor(255, 0, 255)))
            painter.drawText(hx+12, hy+5, h_type)

        # Draw Threats (Clean Dots)
        for t in self.threats:
            box = t.get('box', [0,0,0,0])
            mid_x = (box[0] + box[2]) / 2
            offset = (mid_x - (self.frame_w/2)) / (self.frame_w/2)
            
            tx = cx + int(offset * 80)
            ty = cy - int(t.get('dist', 0) * 6)
            
            is_active = t.get('id') == self.threat_id
            color = QColor(255, 50, 50) if is_active else QColor(200, 200, 0, 150)
            
            painter.setBrush(QBrush(color))
            painter.drawEllipse(tx - 4, ty - 4, 8, 8)
            
            if is_active:
                painter.setPen(QPen(color))
                painter.drawText(tx + 10, ty + 5, f"{t.get('dist')}m")

class SteeringWheelWidget(QWidget):
    """Modern, Minimalist Steering Indicator"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(180, 180)
        self.angle = 0

    def set_angle(self, angle):
        self.angle = angle
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        
        painter.translate(cx, cy)
        
        # Outer Ring
        painter.setPen(QPen(QColor(50, 50, 60), 2))
        painter.drawEllipse(-70, -70, 140, 140)
        
        painter.rotate(self.angle)
        
        # Rotating Bar
        painter.setPen(QPen(QColor(0, 255, 255), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(-60, 0, 60, 0)
        
        # Top Indicator
        painter.setBrush(QBrush(QColor(0, 255, 255)))
        painter.drawEllipse(-4, -74, 8, 8)

class ECUDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astra Perception: Control ECU")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #121212; color: white; font-family: Segoe UI, sans-serif;")
        
        # Controllers
        self.lon_controller = LongitudinalController()
        self.lat_controller = LateralController()
        self.last_time = time.time()
        
        self.init_ui()
        
        # UDP Receiver
        self.receiver = ECUReceiver()
        self.receiver.packet_received.connect(self.process_packet)
        self.receiver.start()
        
        # Connection Watchdog
        self.last_packet_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_connection)
        self.timer.start(100)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # LEFT: Controls HUD
        left_panel = QVBoxLayout()
        
        self.status_lbl = QLabel("DISCONNECTED")
        self.status_lbl.setStyleSheet("color: red; font-size: 18px; font-weight: bold;")
        left_panel.addWidget(self.status_lbl)
        
        # Steering Section
        self.steer_widget = SteeringWheelWidget()
        left_panel.addWidget(self.steer_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.steer_lbl = QLabel("STEERING: 0.0°")
        self.steer_lbl.setStyleSheet("font-size: 20px; color: #00FFFF;")
        left_panel.addWidget(self.steer_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Accel/Brake Bars
        bars_layout = QHBoxLayout()
        
        throttle_box = QVBoxLayout()
        throttle_box.addWidget(QLabel("THROTTLE"))
        self.throttle_bar = QProgressBar()
        self.throttle_bar.setOrientation(Qt.Orientation.Vertical)
        self.throttle_bar.setRange(0, 100)
        self.throttle_bar.setStyleSheet("QProgressBar::chunk { background-color: #00FF00; }")
        throttle_box.addWidget(self.throttle_bar)
        bars_layout.addLayout(throttle_box)
        
        brake_box = QVBoxLayout()
        brake_box.addWidget(QLabel("BRAKE"))
        self.brake_bar = QProgressBar()
        self.brake_bar.setOrientation(Qt.Orientation.Vertical)
        self.brake_bar.setRange(0, 100)
        self.brake_bar.setStyleSheet("QProgressBar::chunk { background-color: #FF0000; }")
        brake_box.addWidget(self.brake_bar)
        bars_layout.addLayout(brake_box)
        
        left_panel.addLayout(bars_layout)
        layout.addLayout(left_panel, stretch=1)
        
        # RIGHT: Radar / Perception Feedback
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("PERCEPTION RADAR (2D)"))
        self.radar = RadarWidget()
        right_panel.addWidget(self.radar)
        
        self.threat_lbl = QLabel("ACTIVE THREAT: NONE")
        self.threat_lbl.setStyleSheet("color: #FF6600; font-size: 16px;")
        right_panel.addWidget(self.threat_lbl)
        
        layout.addLayout(right_panel, stretch=2)

    def process_packet(self, packet):
        self.last_packet_time = time.time()
        self.status_lbl.setText("ECU CONNECTED")
        self.status_lbl.setStyleSheet("color: #00FF00; font-size: 18px; font-weight: bold;")
        
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # 1. Update Controllers
        threat_id = packet.get('threat_id')
        threat_ttc, threat_dist = 99.0, 99.0
        threat_label = "NONE"
        
        for t in packet.get('threats', []):
            if t.get('id') == threat_id:
                threat_ttc = t.get('ttc')
                threat_dist = t.get('dist')
                threat_label = f"{t.get('cls')} (ID:{threat_id})"
                break
        
        accel = self.lon_controller.calculate(threat_ttc, threat_dist, dt)
        
        steering = 0.0
        if packet.get('path_center_x') is not None:
            steering = self.lat_controller.calculate(packet.get('frame_w')//2, 
                                                    packet.get('path_center_x'), dt)
        
        # 2. Update UI
        self.steer_widget.set_angle(steering * 45) # Visual scale
        self.steer_lbl.setText(f"STEERING: {steering*45:.1f}°")
        
        if accel > 0:
            self.throttle_bar.setValue(int(accel * 200)) # Scale for visibility
            self.brake_bar.setValue(0)
        else:
            self.throttle_bar.setValue(0)
            self.brake_bar.setValue(int(abs(accel) * 100))
            
        self.threat_lbl.setText(f"ACTIVE THREAT: {threat_label}")
        
        # Update Radar
        self.radar.update_data(packet.get('threats', []), threat_id, 
                               packet.get('path_center_x'), packet.get('frame_w'),
                               hazards=packet.get('hazards', []))

    def check_connection(self):
        if time.time() - self.last_packet_time > 1.5:
            self.status_lbl.setText("ECU DISCONNECTED - NO SENSOR DATA")
            self.status_lbl.setStyleSheet("color: red;")
            self.throttle_bar.setValue(0)
            self.brake_bar.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECUDashboard()
    window.show()
    sys.exit(app.exec())
