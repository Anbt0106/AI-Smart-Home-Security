import cv2
import yaml
import time
import os
from src.camera import VideoStream
from src.detector import Detector
from src.face_rec import FaceRec
from src.notifier import Notifier
from src.utils import draw_roi, is_person_in_roi, draw_detections

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load Config
    config = load_config()
    roi_coords = config["detection"]["roi"]
    unknown_threshold = config["face_recognition"]["unknown_threshold_seconds"]
    
    # Initialize Modules
    print("[INFO] Starting Camera...")
    vs = VideoStream(src=config["camera"]["source"]).start()
    time.sleep(2.0) # Warmup
    
    print("[INFO] Loading Detector...")
    detector = Detector(model_path=config["detection"]["model_path"], 
                        conf=config["detection"]["conf_threshold"])
    
    print("[INFO] Loading Face Recognition...")
    face_rec = FaceRec(encodings_path=config["face_recognition"]["encodings_path"],
                       tolerance=config["face_recognition"]["tolerance"])
    
    print("[INFO] Initializing Notifier...")
    notifier = Notifier(config)
    
    # State variables
    unknown_timer_start = None
    last_hello_time = 0
    HELLO_COOLDOWN = 300 # 5 minutes
    
    print("[INFO] System Ready. Press 'q' to quit.")
    
    # Create a resizable window
    cv2.namedWindow("Smart Home Camera", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = vs.read()
            if frame is None:
                break
            
            # Resize for performance (optional, but good for FPS)
            frame = cv2.resize(frame, (config["camera"]["width"], config["camera"]["height"]))
            
            # 1. Detect People
            person_boxes = detector.detect_people(frame)
            
            # 2. Draw ROI
            draw_roi(frame, roi_coords)
            
            # 3. Process Detections
            person_in_roi = False
            
            for box in person_boxes:
                if is_person_in_roi(box, roi_coords):
                    person_in_roi = True
                    
                    # 4. Face Recognition
                    name = face_rec.recognize_face(frame, box)
                    
                    # Draw name
                    x1, y1, x2, y2 = box
                    label = f"{name}"
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 5. Logic
                    if name == "Unknown":
                        if unknown_timer_start is None:
                            unknown_timer_start = time.time()
                        else:
                            elapsed = time.time() - unknown_timer_start
                            if elapsed > unknown_threshold:
                                cv2.putText(frame, "INTRUDER ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                notifier.play_sound("alarm.mp3") # Ensure this file exists
                                # Capture and send alert (throttled)
                                if int(elapsed) % 10 == 0: # Send every 10s if persisting
                                    img_path = "intruder.jpg"
                                    cv2.imwrite(img_path, frame)
                                    notifier.send_telegram_alert("Intruder detected!", img_path)
                    else:
                        # Known person
                        unknown_timer_start = None # Reset timer
                        if time.time() - last_hello_time > HELLO_COOLDOWN:
                            notifier.play_sound("welcome.mp3") # Ensure this file exists
                            notifier.send_telegram_alert(f"Welcome home, {name}!")
                            last_hello_time = time.time()
                
            draw_detections(frame, person_boxes)
            
            if not person_in_roi:
                unknown_timer_start = None
            
            cv2.imshow("Smart Home Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Cleaning up...")
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
