import time
import math
from src.common import logger, get_config, is_person_in_roi, compute_iou

class AnomalyDetector:
    def __init__(self, violence_detector=None, fall_detector=None):
        """
        Initialize Anomaly Detector.

        Args:
            violence_detector (ViolenceDetector, optional): Instance of ViolenceDetector.
            fall_detector (FallDetector, optional): Instance of FallDetector.
        """
        config = get_config()
        anomaly_cfg = config.get("anomaly", {})

        # Config parameters
        self.dwell_time_threshold = anomaly_cfg.get("dwell_time_threshold", 5.0)
        self.running_speed_thresh = anomaly_cfg.get("running_speed_thresh", 200.0)
        
        # Kept for fallback or hybrid usage
        fall_cfg = anomaly_cfg.get("fall", {})
        self.fall_ratio_drop = fall_cfg.get("ratio_drop_thresh", 0.8)
        self.fall_delta_y = fall_cfg.get("delta_y_thresh", 100)
        self.fall_t_max = fall_cfg.get("t_max", 2.0)
        
        roi_list = config.get("detection", {}).get("roi", [])
        self.roi = tuple(roi_list) if len(roi_list) == 4 else None

        self.violence_detector = violence_detector
        self.fall_detector = fall_detector
        
        # State
        self._enter_time = {}   # track_id -> timestamp (intrusion/loitering)
        self._last_pos = {}     # track_id -> (timestamp, cx, cy) (running)
        self._posture_hist = {} # track_id -> list of (timestamp, ratio, cy) (fall heuristic)

    def update(self, tracks, frame, timestamp=None):
        """
        Check for anomalies.
        """
        if timestamp is None:
            timestamp = time.time()
            
        events = []
        
        # 1. Check Intrusion & Loitering
        events += self._check_intrusion_loitering(tracks, timestamp)
        
        # 2. Check Running
        events += self._check_running(tracks, timestamp)
        
        # 3. Check Violence
        if self.violence_detector:
            events += self._check_violence(tracks, frame, timestamp)
            
        # 4. Check Fall (Model + Heuristic Hybrid)
        events += self._check_fall(tracks, frame, timestamp)
        
        return events

    def _check_intrusion_loitering(self, tracks, timestamp):
        events = []
        if not self.roi:
            return events

        current_ids = set()
        
        for track in tracks:
            tid = track["id"]
            bbox = track["bbox"]
            current_ids.add(tid)
            
            in_roi = is_person_in_roi(bbox, self.roi)
            
            if in_roi:
                if tid not in self._enter_time:
                    self._enter_time[tid] = timestamp
                    # Optional: Trigger intrusion immediately
                    events.append({
                        "type": "intrusion",
                        "track_id": tid,
                        "message": f"Track {tid} entered ROI",
                        "timestamp": timestamp,
                        "extra": {"bbox": bbox}
                    })
                else:
                    # Check dwell time
                    dwell = timestamp - self._enter_time[tid]
                    if dwell >= self.dwell_time_threshold:
                        events.append({
                            "type": "loitering",
                            "track_id": tid,
                            "message": f"Track {tid} loitering ({dwell:.1f}s)",
                            "timestamp": timestamp,
                            "extra": {"dwell_time": dwell, "bbox": bbox}
                        })
                        del self._enter_time[tid] # Alert once per dwell limit
            else:
                if tid in self._enter_time:
                    del self._enter_time[tid]
        
        # Clean up lost tracks
        for tid in list(self._enter_time.keys()):
            if tid not in current_ids:
                del self._enter_time[tid]
                
        return events

    def _check_running(self, tracks, timestamp):
        events = []
        for track in tracks:
            tid = track["id"]
            bbox = track["bbox"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if tid in self._last_pos:
                last_ts, last_cx, last_cy = self._last_pos[tid]
                dt = timestamp - last_ts
                
                if dt > 0:
                    dist = math.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                    speed = dist / dt # px/sec
                    
                    if speed >= self.running_speed_thresh:
                        events.append({
                            "type": "running",
                            "track_id": tid,
                            "message": f"Track {tid} running ({speed:.1f} px/s)",
                            "timestamp": timestamp,
                            "extra": {"speed": speed, "bbox": bbox}
                        })
            
            self._last_pos[tid] = (timestamp, cx, cy)
        return events

    def _check_violence(self, tracks, frame, timestamp):
        events = []
        if not self.violence_detector:
            return events

        violence_boxes = self.violence_detector.infer(frame)
        
        for vb in violence_boxes:
            v_bbox = vb["bbox"]
            v_score = vb["score"]
            
            for track in tracks:
                t_bbox = track["bbox"]
                tid = track["id"]
                
                iou = compute_iou(v_bbox, t_bbox)
                
                if iou > 0.3: 
                     events.append({
                        "type": "violence",
                        "track_id": tid,
                        "message": f"Violence detected near track {tid}",
                        "timestamp": timestamp,
                        "extra": {
                            "bbox": t_bbox, 
                            "violence_bbox": v_bbox,
                            "score": v_score
                        }
                    })
        return events

    def _check_fall(self, tracks, frame, timestamp):
        events = []
        
        # 1. Model-based Detection (Priority)
        if self.fall_detector:
            fall_boxes = self.fall_detector.infer(frame)
            for fb in fall_boxes:
                f_bbox = fb["bbox"]
                f_score = fb["score"]
                
                for track in tracks:
                    t_bbox = track["bbox"]
                    tid = track["id"]
                    
                    iou = compute_iou(f_bbox, t_bbox)
                    
                    # If high overlap with a tracked person, assign fall to them
                    if iou > 0.3:
                        events.append({
                            "type": "fall",
                            "track_id": tid,
                            "message": f"Fall detected for track {tid}",
                            "timestamp": timestamp,
                            "extra": {
                                "bbox": t_bbox,
                                "fall_bbox": f_bbox,
                                "score": f_score,
                                "method": "model"
                            }
                        })
            # If using model, valid to skip heuristic or run both. 
            # Often model is better. 
            # We continue to heuristic only if needed, but for now let's rely on model if present.
            # But just in case, we can keep heuristic as backup? 
            # User explicitly gave a model, so the model is the source of truth.
            return events

        # 2. Heuristic Detection (Fallback)
        for track in tracks:
            tid = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                continue
                
            ratio = h / max(w, 1.0)
            cy = (y1 + y2) / 2
            
            if tid not in self._posture_hist:
                self._posture_hist[tid] = []
            
            hist = self._posture_hist[tid]
            hist.append((timestamp, ratio, cy))
            
            if len(hist) > 30:
                hist.pop(0)
            
            if len(hist) >= 5:
                start_ts, start_r, start_cy = hist[0]
                end_ts, end_r, end_cy = hist[-1]
                
                dt = end_ts - start_ts
                
                if 0 < dt <= self.fall_t_max:
                    ratio_drop = start_r - end_r
                    delta_y = end_cy - start_cy
                    
                    if ratio_drop >= self.fall_ratio_drop and delta_y >= self.fall_delta_y:
                        events.append({
                            "type": "fall",
                            "track_id": tid,
                            "message": f"Possible fall detected for track {tid}",
                            "timestamp": timestamp,
                            "extra": {
                                "ratio_drop": ratio_drop,
                                "delta_y": delta_y,
                                "bbox": track["bbox"],
                                "method": "heuristic"
                            }
                        })
                        self._posture_hist[tid] = []
                        
        current_ids = {t["id"] for t in tracks}
        for tid in list(self._posture_hist.keys()):
            if tid not in current_ids:
                del self._posture_hist[tid]
                
        return events
