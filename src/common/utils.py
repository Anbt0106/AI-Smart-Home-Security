import cv2
import numpy as np
import time
import os

def draw_roi(frame, roi_coords, color=(255, 0, 0), label="ROI"):
    """
    Draw Region of Interest on the frame.
    
    Args:
        frame: The image frame.
        roi_coords: Tuple (x1, y1, x2, y2).
        color: BGR tuple.
        label: Text label for the ROI.
    """
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def is_person_in_roi(box, roi_coords):
    """
    Check if the center of a bounding box is inside the ROI.
    
    Args:
        box: (x1, y1, x2, y2) of the person.
        roi_coords: (x1, y1, x2, y2) of the region.
        
    Returns:
        bool: True if center is inside ROI.
    """
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi_coords
    
    # Calculate center of the person box
    center_x = (bx1 + bx2) // 2
    center_y = (by1 + by2) // 2
    
    if rx1 < center_x < rx2 and ry1 < center_y < ry2:
        return True
    return False

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        float: IoU value.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou


def draw_detections(frame, boxes, color=(0, 255, 0), label="Person"):
    """
    Draw bounding boxes for detections.
    """
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def save_snapshot(frame, filename="snapshot.jpg", folder="data/snapshots"):
    """
    Save the current frame to disk.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return path

class FPS:
    """
    Helper class to measure Frames Per Second.
    """
    def __init__(self):
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = time.time()
        self._numFrames = 0
        return self

    def update(self):
        self._numFrames += 1

    def stop(self):
        self._end = time.time()

    def elapsed(self):
        return (time.time() - self._start)

    def fps(self):
        return self._numFrames / self.elapsed()
