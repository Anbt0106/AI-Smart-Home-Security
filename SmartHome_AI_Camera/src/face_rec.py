import face_recognition
import pickle
import os
import cv2

class FaceRec:
    def __init__(self, encodings_path="assets/encodings.pickle", tolerance=0.6):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            print(f"[INFO] Loaded encodings for: {list(set(self.known_names))}")
        else:
            print("[INFO] No encodings found. Please train faces first.")

    def train_faces(self, dataset_path="assets/faces"):
        print("[INFO] Quantifying faces...")
        imagePaths = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    imagePaths.append(os.path.join(root, file))

        knownEncodings = []
        knownNames = []

        for (i, imagePath) in enumerate(imagePaths):
            print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
            name = imagePath.split(os.path.sep)[-2]
            
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)

        print("[INFO] Serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        print("[INFO] Training complete.")
        self.load_encodings()

    def recognize_face(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return "Unknown"

        face_roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Detect face in the ROI (using cnn or hog, here we assume the person box is the face area roughly, 
        # but better to re-detect face within the person box for accuracy, or just use the whole ROI if it's a close up)
        # For efficiency in this specific flow, we might just encode the whole ROI if we assume it's a face.
        # However, YOLO detects *person*, not *face*. So we need to find face *inside* the person box.
        
        boxes = face_recognition.face_locations(rgb_roi, model="hog")
        if not boxes:
            return "Unknown"
            
        encodings = face_recognition.face_encodings(rgb_roi, boxes)
        name = "Unknown"

        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=self.tolerance)
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
                break # Return the first match found
        
        return name
