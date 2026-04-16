import cv2
import numpy as np
from collections import deque
import time
import math
import os
from ultralytics import YOLO
import torch

class CentroidTracker:
    def __init__(self, max_disappeared=15):
        self.next_object_id = 0
        self.objects = {} # {id: centroid}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.kf_filters = {} # {id: KalmanFilter}
        
    def _init_kf(self):
        kf = cv2.KalmanFilter(4, 2) # State: [x, y, vx, vy], Measurement: [x, y]
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        return kf

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.kf_filters[self.next_object_id] = self._init_kf()
        self.kf_filters[self.next_object_id].statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.kf_filters[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            input_centroids[i] = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

        object_velocities = {}
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                
                object_id = object_ids[row]
                self.kf_filters[object_id].correct(np.array([[np.float32(input_centroids[col][0])], 
                                                           [np.float32(input_centroids[col][1])]], np.float32))
                prediction = self.kf_filters[object_id].predict()
                
                self.objects[object_id] = (int(prediction[0]), int(prediction[1]))
                object_velocities[object_id] = (float(prediction[2]), float(prediction[3]))
                
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects, object_velocities

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt", conf_thres=0.4, iou_thres=0.45):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # Load model to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        print(f"DEBUG: Astra Perception Engine loaded on {self.device.upper()}")
        
        self.focal_length = 457 
        self.class_dims = {
            0: (0.5, 1.7), 1: (0.8, 1.2), 2: (1.8, 1.5), 3: (0.8, 1.2),
            5: (2.5, 3.5), 7: (2.5, 3.8), 9: (0.3, 0.8), 11: (0.7, 0.7)
        }
        
        self.dist_history = {}
        self.tracker = CentroidTracker(max_disappeared=15)
        self.hallucination_memory = {}

    def estimate_distance(self, box, class_id, object_id=None):
        px_w = box[2] - box[0]
        px_h = box[3] - box[1]
        if px_w <= 0 or px_h <= 0: return 0, 99.0

        real_w, real_h = self.class_dims.get(class_id, (1.8, 1.5))
        focal_len = self.focal_length 

        dist_w = (real_w * focal_len) / px_w
        dist_h = (real_h * focal_len) / px_h
        distance = round((dist_w * 0.4) + (dist_h * 0.6), 1)

        ttc = 99.0 
        now = time.time()
        key = object_id if object_id is not None else class_id
        if key not in self.dist_history:
            self.dist_history[key] = deque(maxlen=10)
        self.dist_history[key].append((now, distance))

        if len(self.dist_history[key]) >= 3:
            times, dists = zip(*self.dist_history[key])
            dt = times[-1] - times[0]
            if dt > 0:
                rel_vel = (dists[0] - dists[-1]) / dt 
                if rel_vel > 0.3: ttc = round(dists[-1] / rel_vel, 1)

        return distance, ttc

    def detect(self, image, drivable_mask=None):
        results = self.model.predict(
            source=image, conf=self.conf_threshold, iou=self.iou_threshold,
            device=self.device, half=True if self.device == 'cuda' else False, verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            return self._handle_hallucinations([])

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        orig_h, orig_w = image.shape[:2]
        detected_rects, detection_data = [], []
        
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            mid_y = (box[1] + box[3]) / 2
            if mid_y < orig_h * 0.4 and class_ids[i] not in [9, 11]: continue

            detected_rects.append([box[0], box[1], box[2], box[3]])
            detection_data.append({'class_id': class_ids[i], 'score': scores[i], 'box': box})

        tracked_objects, tracked_velocities = self.tracker.update(detected_rects)
        current_results = []
        for object_id, centroid in tracked_objects.items():
            best_match_idx = -1
            min_dist = 60
            for i, data in enumerate(detection_data):
                box = data['box']
                d_centroid = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                dist = np.linalg.norm(np.array(centroid) - np.array(d_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                data = detection_data[best_match_idx]
                box = data['box']
                dist_val, ttc = self.estimate_distance(box, data['class_id'], object_id)
                res = {
                    'id': object_id, 'box': [box[0], box[1], box[2], box[3]],
                    'class_id': data['class_id'], 'score': data['score'],
                    'distance': dist_val, 'ttc': ttc, 'vx': tracked_velocities.get(object_id, (0,0))[0]
                }
                current_results.append(res)
                self.hallucination_memory[object_id] = (res, 0)
        
        return self._handle_hallucinations(current_results)

    def _handle_hallucinations(self, current_results):
        active_ids = [r['id'] for r in current_results]
        final_results = current_results.copy()
        for obj_id in list(self.hallucination_memory.keys()):
            if obj_id not in active_ids:
                data, missed_count = self.hallucination_memory[obj_id]
                if missed_count < 3:
                    self.hallucination_memory[obj_id] = (data, missed_count + 1)
                    final_results.append(data)
                else:
                    del self.hallucination_memory[obj_id]
        return final_results

class SurfaceHazardDetector:
    def __init__(self):
        self.hazard_history = deque(maxlen=10)

    def detect(self, image, drivable_mask=None):
        height, width = image.shape[:2]
        roi_y1, roi_y2 = int(height * 0.65), int(height * 0.95)
        roi_x1, roi_x2 = int(width * 0.15), int(width * 0.85)
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0: return []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(blur, 30, 100)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hazards = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 400 < area < 8000:
                (x, y, w, h) = cv2.boundingRect(c)
                if 0.6 < w/float(h) < 2.5:
                    fx, fy = x + roi_x1, y + roi_y1
                    if drivable_mask is not None and drivable_mask[fy + h//2, fx + w//2] == 0: continue
                    roi_edges = edges[y:y+h, x:x+w]
                    if (np.sum(roi_edges == 255) / (w * h)) > 0.05:
                        hazards.append(("Pothole", (fx, fy, w, h)))

        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        yellow = cv2.inRange(hls, np.array([15, 40, 80]), np.array([35, 200, 255]))
        white = cv2.inRange(hls, np.array([0, 180, 0]), np.array([180, 255, 255]))
        s_cnts, _ = cv2.findContours(cv2.bitwise_or(yellow, white), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in s_cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w > 80 and h < 50:
                fx, fy = x + roi_x1, y + roi_y1
                if drivable_mask is None or drivable_mask[fy + h//2, fx + w//2] != 0:
                    hazards.append(("Speed Bump", (fx, fy, w, h)))
        return hazards

class MotionDetector:
    def __init__(self): self.avg = None
    def detect(self, image):
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self.avg is None: self.avg = gray.copy().astype("float"); return image, False
        cv2.accumulateWeighted(gray, self.avg, 0.5)
        thresh = cv2.dilate(cv2.threshold(cv2.absdiff(gray, cv2.convertScaleAbs(self.avg)), 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = False
        for c in cnts:
            if cv2.contourArea(c) >= 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                motion = True
        return image, motion

class DrivableAreaDetector:
    def __init__(self, vehicle_mode="Scooter"):
        self.vehicle_mode = vehicle_mode; self.history = deque(maxlen=5)
        self.last_path_center_x = None; self.last_mask = None; self.grid_cache = None
    def _create_grid(self, h, w):
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(0, h, 15): cv2.line(grid, (0, i), (w, i), (0, 255, 0), 1)
        for i in range(0, w, 45): cv2.line(grid, (i, 0), (i, h), (0, 255, 0), 1)
        return grid
    def detect(self, image, object_boxes=[]):
        height, width = image.shape[:2]; scale = 0.25
        small = cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        sh, sw = small.shape[:2]
        samples = [small[int(sh*0.75):int(sh*0.85), int(sw*0.45):int(sw*0.55)],
                   small[int(sh*0.70):int(sh*0.80), int(sw*0.30):int(sw*0.40)],
                   small[int(sh*0.70):int(sh*0.80), int(sw*0.60):int(sw*0.70)]]
        valid = [s for s in samples if s.size > 0]
        if not valid: return None, "Searching..."
        labs = [cv2.cvtColor(s, cv2.COLOR_BGR2LAB) for s in valid]
        ml, ma, mb = np.median([np.median(s[:,:,0]) for s in labs]), np.median([np.median(s[:,:,1]) for s in labs]), np.median([np.median(s[:,:,2]) for s in labs])
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB); gray = lab[:,:,0].astype(np.uint8)
        tex = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        dist = 0.2 * np.abs(lab[:,:,0]-ml) + 1.2 * np.abs(lab[:,:,1]-ma) + 1.2 * np.abs(lab[:,:,2]-mb)
        mask = np.zeros((sh, sw), dtype=np.uint8)
        mask[(dist < 25.0) & (tex < 60.0)] = 255
        for box in object_boxes:
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            mask[max(0, y1):min(sh, y2), max(0, x1):min(sw, x2)] = 0
        mask[0:int(sh*0.5), :] = 0
        cnts, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = [c for c in cnts if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= sh * 0.8]
        if valid_cnts:
            largest = max(valid_cnts, key=cv2.contourArea); dms = np.zeros((sh, sw), dtype=np.uint8)
            cv2.drawContours(dms, [largest], -1, 255, -1)
            M = cv2.moments(largest)
            if M["m00"] != 0: self.last_path_center_x = int(M["m10"]/M["m00"]) * (1/scale)
            full = cv2.resize(dms, (width, height), interpolation=cv2.INTER_LINEAR); self.history.append(full)
            if len(self.history) >= 2:
                sm = np.zeros_like(full, dtype=np.float32)
                for m in self.history: sm += m
                full = (sm / len(self.history)).astype(np.uint8)
            self.last_mask = full; return full, "Drivable Area OK"
        return None, "Searching..."
    def draw_overlay(self, image, mask):
        if mask is None: return image
        h, w = image.shape[:2]
        if self.grid_cache is None or self.grid_cache.shape[:2] != (h, w): self.grid_cache = self._create_grid(h, w)
        
        # Enhanced Pulsing logic (higher floor for visibility)
        pulse = (math.sin(time.time()*5)+1.5)/2.5 
        overlay = image.copy()
        
        # Vibrant Neon Green base for better visibility
        overlay[mask > 127] = [0, 255, 0] 
        
        # Add grid with improved contrast
        grid_part = (self.grid_cache.astype(np.float32) * pulse).astype(np.uint8)
        cv2.add(overlay, grid_part, dst=overlay, mask=mask)
        
        # Increased alpha from 0.12 to 0.30 for much better "pop"
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
