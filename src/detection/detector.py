from typing import List, Dict, Optional, Union
import numpy as np
import torch
from ultralytics import YOLO
from src.common import logger

class Detector:
    def __init__(self, 
                 model_path: str = "yolo11n.pt", 
                 device: Optional[str] = None, 
                 conf: float = 0.5, 
                 iou: float = 0.45, 
                 allowed_classes: Optional[List[int]] = None):
        """
        Initialize the YOLO Detector.

        Args:
            model_path (str): Path to the YOLO model file.
            device (str, optional): Device to run inference on ('cuda' or 'cpu'). Auto-detect if None.
            conf (float): Confidence threshold.
            iou (float): IOU threshold for NMS.
            allowed_classes (List[int], optional): List of class IDs to filter detections (e.g., [0] for person).
        """
        # Determine device if not provided
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Detector on device: {self.device}")
        
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.allowed_classes = allowed_classes

    def detect(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """
        Run object detection on a frame.

        Args:
            frame (np.ndarray): Input image (BGR).

        Returns:
            List[Dict]: List of detections. Each dict contains:
                - bbox: [x1, y1, x2, y2]
                - cls_id: int
                - cls_name: str
                - score: float
        """
        # Run inference
        # We can pass allowed_classes directly to the model for efficiency,
        # but the prompt asked for filtering logic. Using the model's 'classes' argument
        # is the most optimized way to do filtering in proper YOLO usage.
        results = self.model(
            frame, 
            conf=self.conf, 
            iou=self.iou, 
            classes=self.allowed_classes, # Efficient filtering
            device=self.device, 
            verbose=False
        )
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract data
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                # If we didn't use the 'classes' arg above, we would filter here:
                # if self.allowed_classes and cls_id not in self.allowed_classes: continue

                cls_name = self.model.names[cls_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "score": score
                })
        
        return detections

    def detect_people(self, frame) -> List[tuple]:
        """
        Legacy wrapper for backward compatibility with main.py.
        Returns just the list of bounding boxes for 'person' class.
        """
        detections = self.detect(frame)
        boxes = []
        for d in detections:
            if d['cls_name'] == 'person': # or d['cls_id'] == 0
                x1, y1, x2, y2 = d['bbox']
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
