#!/usr/bin/env python
"""Test Discord webhook locally."""

import os
import sys
import requests
from datetime import datetime

def test_discord_webhook(webhook_url=None):
    """Send a test message to Discord."""
    if webhook_url is None:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        webhook_url = input("Enter your Discord webhook URL: ")
    
    if not webhook_url:
        print("❌ No webhook URL provided")
        return False
    
    print(f"Sending test message to Discord...")
    
    embed = {
        "title": "🧪 Test Message from RewACT Pipeline",
        "description": "If you can see this, your Discord webhook is configured correctly!",
        "color": 0x00ff00,  # Green
        "fields": [
            {"name": "Status", "value": "✅ Working", "inline": True},
            {"name": "Timestamp", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), "inline": True},
        ],
        "footer": {"text": "RewACT Training Pipeline"},
    }
    
    payload = {
        "username": "RewACT Pipeline",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Test message sent successfully!")
        print("   Check your Discord channel to see the message.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send message: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False

if __name__ == "__main__":
    webhook_url = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_discord_webhook(webhook_url)
    sys.exit(0 if success else 1)




