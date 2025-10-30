"""
Voice Clone Manager - Records, validates, and creates Speechify voice clones
Handles the complete flow: recording → validation → cloning → voice_id return
"""

import os
import time
import requests
import logging
import wave
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger("voice_clone_manager")

SPEECHIFY_API = "https://api.sws.speechify.com/v1/voices"
ASSEMBLY_UPLOAD = "https://api.assemblyai.com/v2/upload"
ASSEMBLY_TRANSCRIBE = "https://api.assemblyai.com/v2/transcript"

SPEECHIFY_KEY = os.environ.get("SPEECHIFY_API_KEY")
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")


def save_wav(path: str, frames: bytes, sample_rate: int = 16000, nchannels: int = 1, sampwidth: int = 2):
    """Save raw PCM frames (int16) or existing WAV bytes to a proper WAV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # If frames already a valid WAV header, write directly
    try:
        # if frames looks like raw PCM bytes, write using wave
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(frames)
        return str(p)
    except Exception as e:
        # fallback: write raw bytes
        with open(str(p), "wb") as f:
            f.write(frames)
        return str(p)


def upload_file_to_assembly(filepath: str) -> Optional[str]:
    """Upload file to AssemblyAI and return upload_url."""
    if not ASSEMBLY_KEY:
        logger.error("ASSEMBLYAI_API_KEY not set")
        return None
    headers = {"authorization": ASSEMBLY_KEY}
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(ASSEMBLY_UPLOAD, headers=headers, files={"file": f}, timeout=30)
        if resp.status_code != 200:
            logger.error("Assembly upload failed: %s %s", resp.status_code, resp.text)
            return None
        return resp.json().get("upload_url")
    except Exception as e:
        logger.error("Assembly upload error: %s", e)
        return None


def transcribe_with_assembly(upload_url: str, timeout_sec: int = 60) -> Optional[dict]:
    """Create transcription job and poll until complete. Returns transcript json."""
    if not ASSEMBLY_KEY:
        logger.error("ASSEMBLYAI_API_KEY not set")
        return None
    
    headers = {"authorization": ASSEMBLY_KEY, "content-type": "application/json"}
    payload = {"audio_url": upload_url, "auto_chapters": False}
    try:
        resp = requests.post(ASSEMBLY_TRANSCRIBE, headers=headers, json=payload, timeout=30)
        if resp.status_code not in (200, 201):
            logger.error("Assembly create transcript failed: %s %s", resp.status_code, resp.text)
            return None
        job = resp.json()
        tid = job["id"]
        # poll
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            r = requests.get(f"{ASSEMBLY_TRANSCRIBE}/{tid}", headers=headers, timeout=10)
            if r.status_code != 200:
                logger.error("Assembly poll error: %s %s", r.status_code, r.text)
                return None
            j = r.json()
            status = j.get("status")
            if status == "completed":
                return j
            if status == "failed":
                logger.error("Assembly transcription failed: %s", j)
                return None
            time.sleep(1.5)
        logger.warning("Assembly transcription timed out")
        return None
    except Exception as e:
        logger.error("Assembly transcription error: %s", e)
        return None


def validate_sample_via_assembly(wav_path: str, min_seconds: float = 8.0) -> bool:
    """Return True if transcription exists and duration >= min_seconds."""
    upload_url = upload_file_to_assembly(wav_path)
    if not upload_url:
        logger.warning("Failed to upload to AssemblyAI")
        return False
    result = transcribe_with_assembly(upload_url, timeout_sec=90)
    if not result:
        logger.warning("Failed to transcribe with AssemblyAI")
        return False
    # Assembly result contains 'audio_duration' in ms sometimes, else derive using word timestamps in 'words'
    duration_ms = result.get("audio_duration") or result.get("audio_duration_ms")
    if duration_ms:
        duration = duration_ms / 1000.0
    else:
        # fallback: estimate from last word end
        words = result.get("words") or []
        duration = (words[-1]["end"] / 1000.0) if words else 0.0
    logger.info("Sample duration (s): %.2f", duration)
    has_speech = len(result.get("words", [])) > 0
    is_long_enough = duration >= min_seconds
    logger.info("Validation: duration=%.2fs (min=%.2fs), has_speech=%s", duration, min_seconds, has_speech)
    return is_long_enough and has_speech


def create_speechify_voice(sample_wav_path: str, name: str, gender: str = "notSpecified", locale: str = "en-US", consent_text: str = None, timeout: int = 60) -> Optional[dict]:
    """
    POST multipart/form-data to Speechify /v1/voices
    Returns the created voice JSON (contains 'id' and 'status') or None.
    """
    if SPEECHIFY_KEY is None:
        logger.error("SPEECHIFY_API_KEY not set")
        return None

    headers = {"Authorization": f"Bearer {SPEECHIFY_KEY}"}
    try:
        with open(sample_wav_path, "rb") as f:
            files = {
                "sample": (os.path.basename(sample_wav_path), f, "audio/wav"),
            }
            data = {
                "name": name,
                "gender": gender,
                "locale": locale,
            }
            if consent_text:
                data["consent"] = consent_text
            else:
                data["consent"] = "true"  # Default consent
            
            logger.info(f"📤 Uploading voice sample to Speechify... ({os.path.getsize(sample_wav_path)} bytes)")
            resp = requests.post(SPEECHIFY_API, headers=headers, files=files, data=data, timeout=timeout)
            
            if resp.status_code in (200, 201):
                j = resp.json()
                voice_id = j.get("id")
                status = j.get("status", "unknown")
                logger.info(f"✅ Voice upload complete: id={voice_id}, status={status}")
                return j
            else:
                logger.error("Speechify create voice failed: %s - %s", resp.status_code, resp.text)
                return None
    except Exception as e:
        logger.error("Speechify create voice error: %s", e)
        return None


def get_voice_status(voice_id: str) -> Optional[dict]:
    """Get current status of a voice clone from Speechify API"""
    if SPEECHIFY_KEY is None:
        logger.error("SPEECHIFY_API_KEY not set")
        return None
    
    headers = {"Authorization": f"Bearer {SPEECHIFY_KEY}"}
    try:
        resp = requests.get(f"{SPEECHIFY_API}/{voice_id}", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.warning(f"Failed to get voice status: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting voice status: {e}")
        return None


async def wait_for_cloning_ready(voice_id: str, max_wait_seconds: int = 120, poll_interval: float = 5.0) -> bool:
    """
    Poll Speechify API until voice clone is ready (async version).
    Returns True if ready, False if timeout or error.
    """
    import asyncio
    logger.info(f"⏳ Polling for voice clone status (voice_id={voice_id})...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        status_data = get_voice_status(voice_id)
        if not status_data:
            # If we can't get status, assume it's ready (some APIs return ready immediately)
            logger.warning("Could not get status, assuming voice is ready")
            return True
        
        status = status_data.get("status", "").lower()
        logger.info(f"🕐 Voice status: {status}")
        
        if status in ["ready", "completed", "active"]:
            logger.info("🟢 Voice cloning complete!")
            return True
        elif status in ["failed", "error"]:
            logger.error(f"❌ Voice cloning failed: {status_data}")
            return False
        
        # Still processing - use async sleep
        await asyncio.sleep(poll_interval)
    
    logger.warning(f"⏰ Timeout waiting for voice clone (waited {max_wait_seconds}s)")
    return False


def create_and_validate_clone(frames_bytes: bytes, out_dir: str, caller_id: str, name: Optional[str] = None, consent_text: Optional[str] = None, min_seconds: float = 10.0) -> Optional[str]:
    """
    Full convenience: save frames to WAV, validate via Assembly, create Speechify clone, return voice_id.
    frames_bytes: raw pcm frames (should be mono signed int16 pcm or already a WAV).
    Note: This only UPLOADS the voice - use wait_for_cloning_ready() to poll for completion.
    """
    name = name or f"clone_{caller_id}_{int(time.time())}"
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    wav_path = str(out_dir_path / f"{caller_id}_{int(time.time())}.wav")
    save_wav(wav_path, frames_bytes, sample_rate=22050, nchannels=1, sampwidth=2)

    # validate with assembly
    logger.info("🔍 Validating sample with AssemblyAI...")
    ok = validate_sample_via_assembly(wav_path, min_seconds=min_seconds)
    if not ok:
        logger.warning("❌ Sample did not pass validation (too short / no speech)")
        return None

    logger.info("📤 Creating Speechify voice clone...")
    created = create_speechify_voice(
        sample_wav_path=wav_path, 
        name=name, 
        gender="notSpecified", 
        locale="en-US", 
        consent_text=consent_text or "User consented via live call"
    )
    if created and created.get("id"):
        voice_id = created.get("id")
        logger.info(f"✅ Voice uploaded successfully! voice_id={voice_id} (status may be 'processing')")
        return voice_id
    return None

