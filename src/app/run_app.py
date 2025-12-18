import cv2
import time
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common import (
    load_config, get_config, setup_logger, 
    VideoStream, Notifier, EvidenceRecorder, 
    draw_roi, draw_detections, FPS, compute_iou
)
from src.detection.detector import Detector
from src.tracking.deepsort_tracker import DeepSortTracker
from src.anomaly import (
    ViolenceDetector, FallDetector, AnomalyDetector
)

logger = setup_logger()

def draw_tracks(frame, tracks, events=None):
    """
    Draw tracks and highlight anomalies.
    """
    events_map = {}
    if events:
        # Event Priority
        priority = {
            "FALL": 5,
            "VIOLENCE": 4,
            "INTRUSION": 3,
            "RUNNING": 2,
            "LOITERING": 1
        }
        
        for e in events:
            # Map track_id to event type for coloring
            tid = e.get("track_id")
            if tid is not None:
                new_type = e.get("type", "").upper()
                current_type = events_map.get(tid)
                
                # Only overwrite if new event has higher priority
                if current_type:
                    if priority.get(new_type, 0) > priority.get(current_type, 0):
                        events_map[tid] = new_type
                else:
                    events_map[tid] = new_type
                
    for track in tracks:
        tid = track["id"]
        x1, y1, x2, y2 = track["bbox"]
        score = track.get("score", 0)
        
        # Determine color based on event
        color = (0, 255, 0) # Green (Normal)
        label_prefix = ""
        
        if tid in events_map:
            etype = events_map[tid]
            # Priority Color Map
            # Fall > Violence > Intrusion > Running > Loitering
            if etype == "FALL":
                color = (0, 165, 255) # Orange
                label_prefix = "[FALL] "
            elif etype == "VIOLENCE":
                color = (0, 0, 255) # Red
                label_prefix = "[VIOLENCE] "
            elif etype == "INTRUSION":
                color = (0, 255, 255) # Yellow
                label_prefix = "[INTRUSION] "
            elif etype == "RUNNING":
                color = (255, 0, 255) # Purple
                label_prefix = "[RUN] "
            elif etype == "LOITERING":
                color = (255, 255, 0) # Cyan
                label_prefix = "[LOITER] "
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{label_prefix}ID:{tid} {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def run():
    # 1. Load Configuration
    # Determine absolute path to config file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    
    config = load_config(config_path)
    
    # 2. Initialize Components
    logger.info("Initializing components...")
    
    # Camera
    video_src = config["camera"].get("source", 0)
    width = config["camera"].get("width", 1280)
    height = config["camera"].get("height", 720)
    vs = VideoStream(src=video_src, width=width, height=height).start()
    time.sleep(2.0) # Warmup
    
    # Helper to resolve paths
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        return os.path.join(base_dir, p)

    # Detector
    det_cfg = config["detection"]
    detector = Detector(model_path=resolve_path(det_cfg["model_path"]), 
                        conf=det_cfg["conf_threshold"],
                        iou=det_cfg.get("iou_threshold", 0.45)) # Default if not in config
    
    # Tracker
    tracker = DeepSortTracker()
    
    # Anomaly Detectors
    anomaly_cfg = config["anomaly"]
    
    # Violence
    viol_cfg = anomaly_cfg.get("violence", {})
    violence_detector = ViolenceDetector(
        model_path=resolve_path(viol_cfg.get("model_path")),
        conf=viol_cfg.get("conf", 0.5),
        iou=viol_cfg.get("iou", 0.45)
    )
    
    # Fall
    fall_cfg = anomaly_cfg.get("fall", {})
    fall_detector = FallDetector(
        model_path=resolve_path(fall_cfg.get("model_path")),
        conf=fall_cfg.get("conf", 0.5),
        iou=fall_cfg.get("iou", 0.45)
    )
    
    # Main Anomaly Manager
    anomaly_detector = AnomalyDetector(
        violence_detector=violence_detector,
        fall_detector=fall_detector
    )
    
    # Utilities
    notifier = Notifier()
    evidence_recorder = EvidenceRecorder()
    fps = FPS().start()
    
    logger.info("System Ready. Press 'q' to quit.")
    
    cv2.namedWindow("Camera AI", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # 1. Read Frame
            frame = vs.read()
            if frame is None:
                logger.warning("Frame not received. Exiting or retrying...")
                break
                
            current_ts = time.time()
            
            # 2. Detection
            detections = detector.detect(frame)
            
            # 3. Tracking
            tracks = tracker.update(detections, frame, current_ts)
            
            # 4. Anomaly Detection
            events = anomaly_detector.update(tracks, frame, current_ts)
            
            # 5. Handle Events
            for event in events:
                # Save Evidence
                snapshot_path = evidence_recorder.save_evidence(event, frame)
                
                # Notify
                notifier.notify(event, snapshot_path)
            
            # 6. Visualization
            draw_tracks(frame, tracks, events)
            
            # Draw ROI
            if anomaly_detector.roi:
                draw_roi(frame, anomaly_detector.roi, color=(255, 0, 0), label="Restricted Area")
                
            # Draw FPS
            fps.update()
            cv2.putText(frame, f"FPS: {fps.fps():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera AI", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        fps.stop()
        logger.info(f"Approx. FPS: {fps.fps():.2f}")
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
