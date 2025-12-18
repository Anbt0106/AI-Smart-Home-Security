import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.common import logger, get_config

class DeepSortTracker:
    def __init__(self, tracking_config=None):
        """
        Initialize DeepSort Tracker.
        
        Args:
            tracking_config (dict, optional): Tracking configuration dictionary.
                                            If None, loads from global config.
        """
        if tracking_config is None:
            config = get_config()
            tracking_config = config.get("tracking", {})
        
        max_age = tracking_config.get("max_age", 30)
        n_init = tracking_config.get("n_init", 3)
        nn_budget = tracking_config.get("nn_budget", 100)
        max_cosine_distance = tracking_config.get("max_cosine_distance", 0.2)
        embedder = tracking_config.get("embedder", "mobilenet")
        embedder_gpu = tracking_config.get("embedder_gpu", True)
        
        # Override embedder_gpu if CUDA is not available
        if embedder_gpu and not torch.cuda.is_available():
            logger.warning("CUDA not available, switching DeepSort embedder to CPU.")
            embedder_gpu = False
            
        logger.info(f"Initializing DeepSort with max_age={max_age}, n_init={n_init}, embedder={embedder}")
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
            override_track_class=None
        )
        
        # History dict to store trajectories: track_id -> list of (timestamp, bbox)
        self.history = {}

    def update(self, detections, frame, timestamp=None):
        """
        Update tracks based on new detections.

        Args:
            detections (list): List of dicts from Detector (bbox, score, cls_id, cls_name).
            frame (np.ndarray): The current video frame.
            timestamp (float, optional): Current timestamp. Defaults to time.time().

        Returns:
            list: List of track dicts with standardized format.
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 1. Convert detections to DeepSORT format: ([left, top, w, h], confidence, class_id)
        bbs = []
        for det in detections:
            bbox = det["bbox"] # [x1, y1, x2, y2]
            score = det["score"]
            cls_id = det["cls_id"] # usually 0 for person, or pass actual ID
            
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # deep-sort-realtime expects [left, top, w, h]
            # We pass 'score' in 'others' dict to retrieve it later via track.get_det_supplementary()
            bbs.append(([x1, y1, w, h], score, cls_id, {'score': score}))
            
        # 2. Update tracker
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        
        tracks_output = []
        active_ids = set()
        
        # 3. Format output and update history
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            active_ids.add(track_id)
            
            # to_ltrb() returns [x1, y1, x2, y2]
            ltrb = track.to_ltrb() 
            x1, y1, x2, y2 = map(int, ltrb)
            
            cls_id = 0
            cls_name = "person"
            try:
                 if hasattr(track, 'det_class'):
                     cls_id = track.det_class
            except:
                pass
            
            # Retrieve score from supplementary info
            score = 1.0 # Default
            if hasattr(track, 'get_det_supplementary'):
                supp = track.get_det_supplementary()
                if supp and 'score' in supp:
                    score = supp['score']
            elif hasattr(track, 'det_conf'):
                 # Fallback if available
                 score = track.det_conf if track.det_conf is not None else 1.0

            # Standard output format
            track_data = {
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": cls_name,
                "score": score,
                "timestamp": timestamp
            }
            tracks_output.append(track_data)
            
            # Update history
            if track_id not in self.history:
                self.history[track_id] = []
            
            # Append current state
            self.history[track_id].append((timestamp, [x1, y1, x2, y2]))
            
            # Optional: Limit history size to prevent memory leak (e.g. keep last 1000 points)
            if len(self.history[track_id]) > 1000:
                self.history[track_id].pop(0)

        return tracks_output
