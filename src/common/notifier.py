import pygame
import requests
import threading
import os
import time
from datetime import datetime
from .config import get_notify_config
from .logger import logger

class Notifier:
    def __init__(self):
        """
        Initialize Notifier with config loaded via common.config.
        """
        config = get_notify_config()
        self.telegram_config = config.get("telegram", {})
        self.audio_config = config.get("audio", {})
        
        self.sound_dir = self.audio_config.get("sound_dir", "assets/sounds")
        self.telegram_token = self.telegram_config.get("bot_token", "")
        self.chat_id = self.telegram_config.get("chat_id", "")
        
        # Initialize sound mixer
        try:
            pygame.mixer.init()
            logger.info("Pygame mixer initialized.")
        except Exception as e:
            logger.error(f"Failed to init pygame mixer: {e}")

        self.lock = threading.Lock()

    def play_sound(self, filename):
        """
        Play a sound file in a separate thread.
        
        Args:
            filename (str): Name of the sound file (e.g., 'alarm.mp3').
        """
        sound_path = os.path.join(self.sound_dir, filename)
        if os.path.exists(sound_path):
            threading.Thread(target=self._play, args=(sound_path,)).start()
        else:
            logger.warning(f"Sound file not found: {sound_path}")

    def _play(self, sound_path):
        with self.lock:
            if not pygame.mixer.music.get_busy():
                try:
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                    logger.debug(f"Playing sound: {sound_path}")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error playing sound {sound_path}: {e}")

    def send_telegram_alert(self, message, image_path=None):
        """
        Send a text and optional image to Telegram.
        """
        if not self.telegram_token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured.")
            return

        threading.Thread(target=self._send_telegram, args=(message, image_path)).start()

    def _send_telegram(self, message, image_path):
        try:
            url_msg = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            requests.post(url_msg, data=data, timeout=10)
            
            if image_path and os.path.exists(image_path):
                url_photo = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(image_path, "rb") as f:
                    files = {"photo": f}
                    requests.post(url_photo, data={"chat_id": self.chat_id}, files=files, timeout=20)
            
            logger.info(f"Telegram alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def notify(self, event, image_path=None):
        """
        Generic notification method for any event type.
        
        Args:
            event (dict): Event dictionary.
            image_path (str, optional): Path to the snapshot image.
        """
        e_type = event.get("type", "unknown").upper()
        message = event.get("message", "")
        timestamp = event.get("timestamp")
        
        formatted_message = f"[{e_type}] {message}"
        if timestamp:
            dt = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            formatted_message = f"[{dt}] {formatted_message}"
        
        # Log to console/file
        # Use warning level for severe events, info for others
        if e_type in ["VIOLENCE", "FALL", "INTRUSION"]:
            logger.warning(formatted_message)
            # Play sound based on type
            if e_type == "INTRUSION":
                self.play_sound("alarm.mp3")
            elif e_type == "FALL":
                self.play_sound("warning.mp3")
            elif e_type == "VIOLENCE":
                self.play_sound("alarm.mp3")
        else:
            logger.info(formatted_message)
            
        # Send Telegram
        self.send_telegram_alert(formatted_message, image_path)

    def notify_intrusion(self, image_path, message="INTRUDER ALERT!"):
        """Deprecated: Use notify() instead."""
        self.notify({"type": "intrusion", "message": message}, image_path)

    def notify_fall(self, image_path, message="FALL DETECTED!"):
        """Deprecated: Use notify() instead."""
        self.notify({"type": "fall", "message": message}, image_path)
