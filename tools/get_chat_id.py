import requests
import time

TOKEN = "8246476696:AAHCYYuHAvo2O_iTz2-tuAewMXrzBosifGY"
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print(f"1. Open your bot in Telegram: https://t.me/SmartCameraAI_bot")
print("2. Send a message (e.g., 'Hello') to the bot.")
print("3. Waiting for message...")

while True:
    try:
        response = requests.get(URL).json()
        if response["ok"] and response["result"]:
            # Get the last message
            last_update = response["result"][-1]
            chat_id = last_update["message"]["chat"]["id"]
            user = last_update["message"]["from"]["first_name"]
            
            print(f"\n[SUCCESS] Found Chat ID for {user}!")
            print(f"Chat ID: {chat_id}")
            print(f"\nPlease copy '{chat_id}' and paste it into 'configs/config.yaml' replacing 'YOUR_CHAT_ID'.")
            break
        time.sleep(2)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
