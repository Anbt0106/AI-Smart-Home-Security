import cv2
import os
import sys

# Add src to path to import utils if needed, or just use raw cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def capture_faces(name):
    save_dir = os.path.join("..", "assets", "faces", name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cap = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Press 's' to save a face image for user '{name}'. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            filename = os.path.join(save_dir, f"{name}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")
            count += 1
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python capture_faces.py <name>")
        name = input("Enter name: ")
    else:
        name = sys.argv[1]
    capture_faces(name)
