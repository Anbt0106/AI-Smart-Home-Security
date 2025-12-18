import os
import cv2
import json
import time
from datetime import datetime
from src.common import logger

class EvidenceRecorder:
    def __init__(self, base_dir="data/evidence"):
        """
        Initialize Evidence Recorder.
        
        Args:
            base_dir (str): Base directory for storing evidence.
        """
        self.base_dir = base_dir
        
        # Define event types to record
        self.event_types = ["violence", "fall", "intrusion", "loitering", "running"]
        
        # Create directories
        for et in self.event_types:
            path = os.path.join(self.base_dir, et)
            if not os.path.exists(path):
                os.makedirs(path)
                
        # Main log file
        self.log_file = os.path.join(self.base_dir, "evidence_log.json")

    def save_evidence(self, event, frame):
        """
        Save evidence for an anomaly event.
        
        Args:
            event (dict): Event dictionary containing type, track_id, timestamp, etc.
            frame (np.ndarray): The video frame.
            
        Returns:
            str: Path to the saved snapshot image.
        """
        e_type = event.get("type", "unknown")
        timestamp = event.get("timestamp", time.time())
        track_id = event.get("track_id", "unknown")
        
        # Create filename: timestamp_trackid.jpg
        # Use datetime for readable filename
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{dt_str}_{track_id}.jpg"
        
        # Determine folder
        save_dir = os.path.join(self.base_dir, e_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) # Handle unknown logic event types gracefully
            
        screenshot_path = os.path.join(save_dir, filename)
        
        try:
            # 1. Save Image
            cv2.imwrite(screenshot_path, frame)
            
            # 2. Append to Log
            record = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "event_type": e_type,
                "track_id": track_id,
                "bbox": event.get("extra", {}).get("bbox"),
                "message": event.get("message"),
                "snapshot_path": screenshot_path
            }
            
            self._append_log(record)
            
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")
            return None

    def _append_log(self, record):
        """
        Append a record to the JSON log file.
        """
        # Simple JSONL (Line-delimited JSON) for efficiency and atomicity
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log evidence: {e}")
