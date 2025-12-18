import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis

class FaceRec:
    def __init__(self, encodings_path="data/assets/encodings.pickle", tolerance=0.5):
        self.encodings_path = encodings_path
        # InsightFace uses Cosine Similarity. 
        # Threshold usually 0.4-0.6. Higher means stricter match.
        # We use 'tolerance' from config as the similarity threshold.
        self.threshold = tolerance 
        
        self.known_encodings = []
        self.known_names = []
        
        # Initialize InsightFace
        # name='buffalo_l' is the default model pack (includes detection and recognition)
        # providers=['CPUExecutionProvider'] ensures it runs on CPU
        print("[INFO] Initializing InsightFace (this might take a while to download models on first run)...")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            print(f"[INFO] Loaded encodings for: {list(set(self.known_names))}")
        else:
            print("[INFO] No encodings found. Please register faces first.")

    def train_faces(self, dataset_path="data/assets/faces"):
        print("[INFO] Processing faces with InsightFace...")
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
            
            img = cv2.imread(imagePath)
            if img is None:
                continue
            
            # InsightFace expects BGR image (OpenCV default)
            faces = self.app.get(img)
            
            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                embedding = faces[0].embedding
                knownEncodings.append(embedding)
                knownNames.append(name)
            else:
                print(f"[WARN] No face found in {imagePath}")

        print("[INFO] Serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        print("[INFO] Registration complete.")
        self.load_encodings()

    def recognize_face(self, frame, person_box):
        # Crop the person
        x1, y1, x2, y2 = person_box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return "Unknown"
            
        person_img = frame[y1:y2, x1:x2]
        
        # Detect face in the crop
        faces = self.app.get(person_img)
        
        if len(faces) == 0:
            return "Unknown"
            
        # Get the largest face in the crop
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        embedding = face.embedding
        
        if len(self.known_encodings) == 0:
            return "Unknown"
            
        # Compute Cosine Similarity manually to avoid sklearn dependency
        # Sim = (A . B) / (||A|| * ||B||)
        # InsightFace embeddings are usually normalized, but let's be safe
        
        best_sim = -1.0
        best_name = "Unknown"
        
        for idx, known_emb in enumerate(self.known_encodings):
            # Calculate cosine similarity
            sim = np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb))
            
            if sim > best_sim:
                best_sim = sim
                best_name = self.known_names[idx]
        
        if best_sim > self.threshold:
            return best_name
        else:
            return "Unknown"
