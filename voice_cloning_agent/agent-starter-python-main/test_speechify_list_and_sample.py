import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env.local")

BASE_URL = "https://api.sws.speechify.com/v1"
API_KEY = os.getenv("SPEECHIFY_API_KEY")

if not API_KEY:
    print("SPEECHIFY_API_KEY not found in .env.local")
    sys.exit(1)


def main() -> None:
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Step 1: List voices
    print("Fetching voices...")
    voices_res = requests.get(f"{BASE_URL}/voices", headers=headers, timeout=60)
    if voices_res.status_code != 200:
        print("Failed to fetch voices:", voices_res.status_code, voices_res.text)
        sys.exit(1)

    voices = voices_res.json()
    if not isinstance(voices, list) or not voices:
        print("No voices returned")
        sys.exit(1)

    print(f"Voices fetched: {len(voices)}")

    # Prefer personal voice if available
    voice = next((v for v in voices if v.get("type") == "personal"), voices[0])
    voice_id = voice.get("id")
    display_name = voice.get("display_name") or voice.get("name") or voice_id
    print(f"Using Voice ID: {voice_id} - {display_name}")

    # Step 2: Download sample audio
    sample_url = f"{BASE_URL}/voices/{voice_id}/sample"
    print("Downloading sample audio...")
    sample_res = requests.get(sample_url, headers=headers, timeout=120)
    if sample_res.status_code != 200:
        print("Failed to download sample:", sample_res.status_code, sample_res.text)
        sys.exit(1)

    # Save under generated_audio/
    out_dir = Path(__file__).parent / "generated_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "voice_sample.wav"
    with open(out_path, "wb") as f:
        f.write(sample_res.content)

    print(f"Sample downloaded successfully -> {out_path}")


if __name__ == "__main__":
    main()


