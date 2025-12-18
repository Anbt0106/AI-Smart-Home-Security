import cv2
import threading
import time
from .logger import logger

class VideoStream:
    def __init__(self, src=0, width=None, height=None, retry_delay=5):
        """
        Initialize VideoStream.
        
        Args:
            src (int, str): Camera source (0 for webcam, or RTSP/File path).
            width (int, optional): Resize width.
            height (int, optional): Resize height.
            retry_delay (int): Seconds to wait before retrying connection.
        """
        self.src = src
        self.width = width
        self.height = height
        self.retry_delay = retry_delay
        
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        self._open_camera()

    def _open_camera(self):
        """Internal method to open the camera source."""
        logger.info(f"Opening camera source: {self.src}")
        self.stream = cv2.VideoCapture(self.src)
        
        if not self.stream.isOpened():
            logger.error(f"Failed to open camera source: {self.src}")
        else:
            (self.grabbed, self.frame) = self.stream.read()
            if self.grabbed:
                logger.info(f"Camera opened successfully: {self.src}")

                self.fps = self.stream.get(cv2.CAP_PROP_FPS)
                if self.fps > 0:
                    self.delay = 1.0 / self.fps
                    logger.info(f"Source FPS: {self.fps}. Target delay: {self.delay:.4f}s")
                else:
                    self.delay = 0.005 
            else:
                logger.warning(f"Camera opened but failed to read initial frame: {self.src}")

    def start(self):
        """Start the video capture thread."""
        logger.info("Starting video stream thread.")
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Thread worker to continuously read frames."""
        while True:
            start_time = time.time()
            
            if self.stopped:
                logger.info("Video stream thread stopping.")
                return

            if self.stream is None or not self.stream.isOpened():
                logger.warning("Stream is closed. Attempting to reconnect...")
                time.sleep(self.retry_delay)
                self._open_camera()
                continue

            (grabbed, frame) = self.stream.read()
            
            if not grabbed:
                logger.warning("Failed to grab frame. Retrying...")
                # If reading from file, this might mean EOF. If stream, might mean disconnect.
                # For stream, we try to reopen.
                self.stream.release()
                time.sleep(self.retry_delay)
                self._open_camera()
                continue
            
            # Resize if needed
            if self.width and self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            elif self.width: # width only
                 h, w = frame.shape[:2]
                 aspect = self.width / w
                 self.height = int(h * aspect)
                 frame = cv2.resize(frame, (self.width, self.height))

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            
            # Sync with video FPS
            elapsed = time.time() - start_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)

    def read(self):
        """Return the most recent frame."""
        with self.lock:
            return self.frame

    def stop(self):
        """Stop the video stream."""
        self.stopped = True
        if self.stream:
            self.stream.release()
        logger.info("Video stream stopped.")
