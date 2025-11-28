import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_rec import FaceRec

if __name__ == "__main__":
    print("[INFO] Starting training...")
    # Calculate absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encodings_path = os.path.join(base_dir, "assets", "encodings.pickle")
    dataset_path = os.path.join(base_dir, "assets", "faces")
    
    print(f"[INFO] Encodings path: {encodings_path}")
    print(f"[INFO] Dataset path: {dataset_path}")

    fr = FaceRec(encodings_path=encodings_path)
    fr.train_faces(dataset_path=dataset_path)
