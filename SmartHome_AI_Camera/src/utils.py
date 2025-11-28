import cv2
import numpy as np

def draw_roi(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def is_person_in_roi(box, roi_coords):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi_coords
    
    # Calculate center of the person box
    center_x = (bx1 + bx2) // 2
    center_y = (by1 + by2) // 2
    
    if rx1 < center_x < rx2 and ry1 < center_y < ry2:
        return True
    return False

def draw_detections(frame, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
