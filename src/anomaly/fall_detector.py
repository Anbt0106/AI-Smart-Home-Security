from ultralytics import YOLO
import torch
from src.common import logger

class FallDetector:
    def __init__(self, 
                 model_path="models/fall/Fall_Yolov11.pt", 
                 conf=0.5, 
                 iou=0.45, 
                 class_id=0,
                 infer_every_n_frames=5):
        """
        Initialize Fall Detector using YOLOv11.

        Args:
            model_path (str): Path to trained fall detection model.
            conf (float): Confidence threshold.
            iou (float): IoU threshold for NMS.
            class_id (int): Class ID for fall (usually 0).
            infer_every_n_frames (int): Run inference every N frames.
        """
        self.conf = conf
        self.iou = iou
        self.class_id = class_id # Assume 'Fall' is class 0, or user might need to adjust
        self.infer_every_n_frames = infer_every_n_frames
        self._frame_count = 0
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing FallDetector on {device} with model: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            # Warmup
            # self.model(torch.zeros(1, 3, 640, 640).to(device)) 
        except Exception as e:
            logger.error(f"Failed to load Fall model: {e}")
            self.model = None

    def infer(self, frame):
        """
        Detect falls in the frame.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            list[dict]: List of fall detections, each containing 'bbox' and 'score'.
        """
        self._frame_count += 1
        
        # Skip frames to save resources
        if self._frame_count % self.infer_every_n_frames != 0:
            return []
            
        if self.model is None:
            return []

        fall_detections = []
        try:
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    
                    # Assuming we check for specific class, or if model only has "Fall"
                    if cls_id == self.class_id:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        score = float(box.conf[0].item())
                        
                        fall_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score
                        })
                        
        except Exception as e:
            logger.error(f"Error in fall inference: {e}")
            
        return fall_detections
