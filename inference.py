import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import time
import math

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
                # Kalman Update
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
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.45):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # Determine best available execution provider
        available_providers = ort.get_available_providers()
        providers = []
        
        # TensorRT configuration with FP16
        if 'TensorrtExecutionProvider' in available_providers:
            providers.append(('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }))
        
        if 'CUDAExecutionProvider' in available_providers:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }))
            
        providers.append('CPUExecutionProvider')
        
        # Initialize ONNX session
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Failed to initialize with GPU providers, falling back to CPU: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Refined Focal length for DroidCam (approx 70deg HFOV at 640px)
        self.focal_length = 457 
        
        # Real-world dimensions (Width, Height) in meters
        self.class_dims = {
            0: (0.5, 1.7),   # person
            1: (0.8, 1.2),   # bicycle
            2: (1.8, 1.5),   # car
            3: (0.8, 1.2),   # motorcycle
            5: (2.5, 3.5),   # bus
            7: (2.5, 3.8),   # truck
            9: (0.3, 0.8),   # traffic light
            11: (0.7, 0.7)   # stop sign
        }
        
        # Distance smoothing and tracking memory
        self.dist_history = {} # {object_id: deque of (time, distance)}
        self.tracker = CentroidTracker(max_disappeared=15)

    def estimate_distance(self, box, class_id, object_id=None):
        """Calculates distance using Width and Height from the bounding box"""
        # box is [x_min, y_min, width, height]
        px_w = box[2]
        px_h = box[3]
        if px_w <= 0 or px_h <= 0: return 0, 99.0

        real_w, real_h = self.class_dims.get(class_id, (1.8, 1.5))
        focal_len = self.focal_length 

        dist_w = (real_w * focal_len) / px_w
        dist_h = (real_h * focal_len) / px_h
        distance = round((dist_w * 0.4) + (dist_h * 0.6), 1)

        # TTC Calculation using tracked object ID
        ttc = 99.0 # Default safe
        now = time.time()

        key = object_id if object_id is not None else class_id
        if key not in self.dist_history:
            self.dist_history[key] = deque(maxlen=10)

        self.dist_history[key].append((now, distance))

        if len(self.dist_history[key]) >= 3:
            # Simple linear regression for relative velocity
            times, dists = zip(*self.dist_history[key])
            dt = times[-1] - times[0]
            if dt > 0:
                rel_vel = (dists[0] - dists[-1]) / dt # Positive means closing in
                if rel_vel > 0.3: # Only if closing at > 0.3 m/s
                    ttc = round(dists[-1] / rel_vel, 1)

        return distance, ttc

    def preprocess(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_tensor = resized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        input_tensor /= 255.0
        return input_tensor, image.shape[:2]

    def detect(self, image):
        input_tensor, (orig_h, orig_w) = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = np.squeeze(outputs[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            self.tracker.update([])
            return []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]

        x_factor = orig_w / self.input_width
        y_factor = orig_h / self.input_height

        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * x_factor
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * y_factor
        boxes[:, 2] = boxes[:, 2] * x_factor
        boxes[:, 3] = boxes[:, 3] * y_factor

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)

        detected_rects = []
        detection_data = []
        for i in indices:
            idx = int(i) if isinstance(i, np.ndarray) else i
            box = boxes[idx].astype(int)
            detected_rects.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            detection_data.append({'class_id': class_ids[idx], 'score': scores[idx], 'box': box})

        # Update tracker
        tracked_objects, tracked_velocities = self.tracker.update(detected_rects)

        results = []
        # Match tracked objects back to detection data based on centroid proximity
        for object_id, centroid in tracked_objects.items():
            best_match_idx = -1
            min_dist = 50 # Centroid must be within 50px

            for i, data in enumerate(detection_data):
                box = data['box']
                d_centroid = (box[0] + box[2]/2, box[1] + box[3]/2)
                dist = np.linalg.norm(np.array(centroid) - np.array(d_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                data = detection_data[best_match_idx]
                box = data['box']
                dist_val, ttc = self.estimate_distance(box, data['class_id'], object_id)

                # Intent analysis using Kalman velocity
                velocity = tracked_velocities.get(object_id, (0, 0))
                # vx > 0 means moving right, vx < 0 means moving left (in image space)
                lateral_velocity = velocity[0] 

                results.append({
                    'id': object_id,
                    'box': [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    'class_id': data['class_id'],
                    'score': data['score'],
                    'distance': dist_val,
                    'ttc': ttc,
                    'vx': lateral_velocity
                })

        return results

class SurfaceHazardDetector:
    """Intelligent Pothole and Speed Bump Detector tailored for Indian Roads"""
    def __init__(self):
        self.hazard_history = deque(maxlen=10) # For temporal stability

    def detect(self, image, drivable_mask=None):
        height, width = image.shape[:2]

        # Define ROI (Immediate 5-8 meters)
        roi_y1, roi_y2 = int(height * 0.65), int(height * 0.95)
        roi_x1, roi_x2 = int(width * 0.15), int(width * 0.85)

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0: return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hazards = []

        # 1. Improved Pothole Detection (Adaptive + Canny)
        # Potholes aren't just dark; they have texture and depth shadows
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding to handle varied lighting/shadows
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)

        # Edge density check - potholes have "rough" edges
        edges = cv2.Canny(blur, 30, 100)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if 400 < area < 8000:
                (x, y, w, h) = cv2.boundingRect(c)
                aspect_ratio = w / float(h)

                # Potholes are usually horizontal-ish ellipses on the road
                if 0.6 < aspect_ratio < 2.5:
                    # Coordinate mapping back to full frame
                    fx, fy = x + roi_x1, y + roi_y1

                    # VALIDATION: Check if it's inside the drivable area
                    is_in_path = True
                    if drivable_mask is not None:
                        # Sample center of the detected hazard in the mask
                        mask_sample = drivable_mask[fy + h//2, fx + w//2]
                        if mask_sample == 0: # Outside green area
                            is_in_path = False

                    if is_in_path:
                        # Edge density check: Does it have enough edges to be a hole?
                        roi_edges = edges[y:y+h, x:x+w]
                        edge_density = np.sum(roi_edges == 255) / (w * h)
                        if edge_density > 0.05: # Minimal texture required
                            hazards.append(("Pothole", (fx, fy, w, h)))

        # 2. Speed Bump Detection (Color + Geometry)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        # Yellow (Standard) and White (Faded) markers
        yellow = cv2.inRange(hls, np.array([15, 40, 80]), np.array([35, 200, 255]))
        white = cv2.inRange(hls, np.array([0, 180, 0]), np.array([180, 255, 255]))
        combined = cv2.bitwise_or(yellow, white)

        s_cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in s_cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # Speed bumps are wide horizontal strips
            if w > 80 and h < 50:
                fx, fy = x + roi_x1, y + roi_y1

                # Check path alignment
                if drivable_mask is not None:
                    if drivable_mask[fy + h//2, fx + w//2] != 0:
                        hazards.append(("Speed Bump", (fx, fy, w, h)))
                else:
                    hazards.append(("Speed Bump", (fx, fy, w, h)))

        return hazards
class MotionDetector:
    """Lightweight motion detection for Blind Spot Monitoring"""
    def __init__(self):
        self.avg = None

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.avg is None:
            self.avg = gray.copy().astype("float")
            return image, False

        cv2.accumulateWeighted(gray, self.avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        
        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            motion_detected = True
            
        return image, motion_detected

class DrivableAreaDetector:
    """
    Robust Drivable Area Detection for Indian Roads.
    Uses dynamic color/texture modeling of the road surface combined with
    obstacle masking to find free space in unstructured environments.
    """
    def __init__(self, vehicle_mode="Scooter"):
        self.vehicle_mode = vehicle_mode
        self.history = deque(maxlen=5)
        self.last_path_center_x = None
        self.last_mask = None
        
    def detect_and_draw(self, image, object_boxes=[]):
        height, width = image.shape[:2]
        
        # 0. Extreme Downsample for math layer (Fastest processing)
        scale = 0.25
        small_image = cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        s_height, s_width = small_image.shape[:2]
        
        # 1. Multi-Scale Adaptive Sampling (Robust to shadows/lighting)
        # Sample 3 regions: Center-bottom, Mid-left, Mid-right
        samples = [
            small_image[int(s_height*0.85):int(s_height*0.95), int(s_width*0.45):int(s_width*0.55)],
            small_image[int(s_height*0.80):int(s_height*0.90), int(s_width*0.35):int(s_width*0.40)],
            small_image[int(s_height*0.80):int(s_height*0.90), int(s_width*0.60):int(s_width*0.65)]
        ]
        
        valid_samples = [s for s in samples if s.size > 0]
        if not valid_samples: return image, "Searching..."
        
        lab_samples = [cv2.cvtColor(s, cv2.COLOR_BGR2LAB) for s in valid_samples]
        median_l = np.median([np.median(s[:,:,0]) for s in lab_samples])
        median_a = np.median([np.median(s[:,:,1]) for s in lab_samples])
        median_b = np.median([np.median(s[:,:,2]) for s in lab_samples])
        
        lab = cv2.cvtColor(small_image, cv2.COLOR_BGR2LAB)
        
        # 2. Fast L1 Distance
        dist = 0.2 * np.abs(lab[:, :, 0] - median_l) + \
               1.2 * np.abs(lab[:, :, 1] - median_a) + \
               1.2 * np.abs(lab[:, :, 2] - median_b)
                       
        # 3. Texture-Aware Filtering (Roads are smooth, surroundings are rough)
        gray = lab[:,:,0].astype(np.uint8)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.abs(sobelx) + np.abs(sobely)
        
        # Threshold: Low color distance AND low texture (smoothness)
        road_mask = np.zeros((s_height, s_width), dtype=np.uint8)
        road_mask[(dist < 18.0) & (texture < 40.0)] = 255
        
        # 4. Obstacle & Horizon Masking
        for box in object_boxes:
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            road_mask[max(0, y1):min(s_height, y2), max(0, x1):min(s_width, x2)] = 0
            
        road_mask[0:int(s_height*0.45), :] = 0
        
        # 5. Connectivity Analysis
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        cnts, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drivable_mask_small = np.zeros((s_height, s_width), dtype=np.uint8)
        
        # Must touch bottom center area
        valid_cnts = [c for c in cnts if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= s_height - 10]
                
        if valid_cnts:
            largest_cnt = max(valid_cnts, key=cv2.contourArea)
            cv2.drawContours(drivable_mask_small, [largest_cnt], -1, 255, -1)
            
            # Moments for path center
            M = cv2.moments(largest_cnt)
            if M["m00"] != 0:
                self.last_path_center_x = int(M["m10"] / M["m00"]) * (1/scale)
            
            # 6. Smooth Upscaling & Grid Rendering
            full_mask = cv2.resize(drivable_mask_small, (width, height), interpolation=cv2.INTER_LINEAR)
            self.history.append(full_mask)
            
            if len(self.history) >= 3:
                full_mask = np.bitwise_and.reduce(list(self.history))
            
            self.last_mask = full_mask
            
            overlay = image.copy()
            # Pulsing Grid Effect
            grid_val = int(150 + 50 * math.sin(time.time() * 4))
            
            grid_overlay = np.zeros_like(image)
            for i in range(0, height, 12):
                cv2.line(grid_overlay, (0, i), (width, i), (0, grid_val, 0), 1)
            for i in range(0, width, 40):
                cv2.line(grid_overlay, (i, 0), (i, height), (0, grid_val, 0), 1)

            overlay[full_mask > 127] = [0, 80, 0] # Base tint
            overlay[np.where((grid_overlay > 0).any(axis=2) & (full_mask > 127))] = [0, 255, 0]

            result = cv2.addWeighted(image, 0.88, overlay, 0.12, 0)
            return result, "Drivable Area OK"
            
        return image, "Searching..."

