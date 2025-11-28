from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_people(self, frame):
        results = self.model(frame, classes=0, conf=self.conf, verbose=False) # class 0 is person
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
