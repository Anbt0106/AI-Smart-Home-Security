import pygame
import requests
import threading
import os
import time

class Notifier:
    def __init__(self, config):
        self.config = config
        self.sound_dir = config.get("audio", {}).get("sound_dir", "assets/sounds")
        self.telegram_token = config.get("telegram", {}).get("bot_token", "")
        self.chat_id = config.get("telegram", {}).get("chat_id", "")
        
        pygame.mixer.init()
        self.lock = threading.Lock()

    def play_sound(self, filename):
        sound_path = os.path.join(self.sound_dir, filename)
        if os.path.exists(sound_path):
            threading.Thread(target=self._play, args=(sound_path,)).start()
        else:
            print(f"[WARN] Sound file not found: {sound_path}")

    def _play(self, sound_path):
        with self.lock:
            if not pygame.mixer.music.get_busy():
                try:
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"[ERROR] Playing sound: {e}")

    def send_telegram_alert(self, message, image_path=None):
        if not self.telegram_token or not self.chat_id:
            print("[WARN] Telegram token or chat_id not set.")
            return

        threading.Thread(target=self._send_telegram, args=(message, image_path)).start()

    def _send_telegram(self, message, image_path):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            requests.post(url, data=data)
            
            if image_path and os.path.exists(image_path):
                url_photo = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(image_path, "rb") as f:
                    files = {"photo": f}
                    requests.post(url_photo, data={"chat_id": self.chat_id}, files=files)
            print(f"[INFO] Telegram sent: {message}")
        except Exception as e:
            print(f"[ERROR] Sending Telegram: {e}")
