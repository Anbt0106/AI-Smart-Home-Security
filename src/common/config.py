import yaml
import os
import sys

_CONFIG = {}

def load_config(path="configs/config.yaml"):
    """
    Load configuration from a YAML file.
    """
    global _CONFIG
    if not os.path.exists(path):
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
        
    try:
        with open(path, "r") as f:
            _CONFIG = yaml.safe_load(f)
        return _CONFIG
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

def get_config():
    """Return the entire configuration dictionary."""
    return _CONFIG

def get_camera_config():
    """Return camera configuration."""
    return _CONFIG.get("camera", {})

def get_detection_config():
    """Return detection configuration."""
    return _CONFIG.get("detection", {})

def get_face_recognition_config():
    """Return face recognition configuration."""
    return _CONFIG.get("face_recognition", {})

def get_notify_config():
    """Return notification (telegram/audio) configuration."""
    # Merging telegram and audio into a single notify config for convenience if needed,
    # or just returning a dict containing both. 
    # Based on user request "get_notify_config", might imply generic notification settings.
    # Let's return a dict with both 'telegram' and 'audio' keys.
    return {
        "telegram": _CONFIG.get("telegram", {}),
        "audio": _CONFIG.get("audio", {})
    }
