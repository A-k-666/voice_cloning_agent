"""
Test script to upload recording to Speechify and download cloned voice TTS as MP3
Uses requests library for simpler multipart form handling
"""
import requests
import time
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv(".env.local")

BASE_URL = "https://api.sws.speechify.com/v1"
API_KEY = os.getenv("SPEECHIFY_API_KEY")

if not API_KEY:
    raise ValueError("SPEECHIFY_API_KEY not found in .env.local")


import asyncio
from src.speechify_client import SpeechifyClient


async def upload_and_test():
    """Use SpeechifyClient (aiohttp) to clone, then TTS MP3 via HTTP."""
    # Use most recent recording
    recordings_dir = Path("recordings")
    recordings = sorted(recordings_dir.glob("voice_sample_*.wav"), reverse=True)
    
    if not recordings:
        print("❌ No recordings found in recordings/ folder")
        return
    
    wav_file = recordings[0]
    print(f"📁 Using recording: {wav_file.name}")
    print(f"📊 File size: {wav_file.stat().st_size / 1024:.2f}KB\n")
    
    # Prepare consent JSON per docs: fullName (camelCase) + email required
    full_name_value = "Ayush Kharbujkar"
    consent_payload = {
        "fullName": full_name_value,  # camelCase as per docs
        "email": "ayush.kharbujkar@example.com",  # required per docs
        "given_by": full_name_value,
        "text": f"I, {full_name_value}, consent to Speechify using my recording to create a cloned voice for this project.",
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "confirmation": True
    }
    consent_json_string = json.dumps(consent_payload)
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    
    print("="*60)
    print("UPLOADING TO SPEECHIFY API")
    print("="*60)
    print(f"📤 POST {BASE_URL}/voices")
    print(f"📋 Fields: name, full_name, consent (JSON), language, locale, description, sample")
    print(f"📋 Consent JSON: {consent_json_string}\n")
    
    # Use SpeechifyClient for clone (handles multipart/fields)
    client = SpeechifyClient()
    voice_name = f"AyushVoice_{int(time.time())}"
    print("🧰 Using SpeechifyClient to create clone...")
    # Extract email from consent JSON if present, else use default
    try:
        consent_data = json.loads(consent_json_string)
        user_email = consent_data.get("email", f"ayush.kharbujkar@example.com")
    except:
        user_email = "ayush.kharbujkar@example.com"
    
    voice_id = await client.create_clone(
        file_path=str(wav_file),
        voice_name=voice_name,
        full_name=full_name_value,
        email=user_email,
        consent_json=consent_json_string,
        language="en",
        locale="en-US",
        gender="notSpecified",
        description=f"Voice clone for {full_name_value}",
    )
    if not voice_id:
        print("❌ Voice creation failed.")
        return None

    print(f"✅ Clone ready: {voice_id}")
    # Generate TTS as MP3
    print("="*60)
    print("GENERATING TTS WITH CLONED VOICE (MP3)")
    print("="*60)
    test_text = "Hello, this is my cloned voice speaking. The voice cloning test was successful!"
    tts_resp = requests.post(
        f"{BASE_URL}/audio/speech",
        headers={
            **headers,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        json={
            "voice_id": voice_id,
            "input": test_text,
            "format": "mp3",
        },
        timeout=30
    )
    if tts_resp.status_code == 200:
        mp3_bytes = tts_resp.content
        output_path = Path("recordings") / f"cloned_voice_test_{voice_id}.mp3"
        with open(output_path, "wb") as f:
            f.write(mp3_bytes)
        print(f"✅ TTS generated successfully!")
        print(f"💾 Saved MP3: {output_path}")
        print(f"📁 Size: {len(mp3_bytes) / 1024:.2f}KB")
        print("="*60)
        return voice_id
    else:
        print(f"❌ TTS failed: {tts_resp.status_code} - {tts_resp.text}")
        return None


if __name__ == "__main__":
    import sys
    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("\n" + "="*60)
    print("SPEECHIFY VOICE CLONING TEST")
    print("="*60 + "\n")
    
    asyncio.run(upload_and_test())
