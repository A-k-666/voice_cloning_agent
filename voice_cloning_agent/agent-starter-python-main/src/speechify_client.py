"""
Speechify API Client for Voice Cloning and TTS
"""

import aiohttp
import asyncio
import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger("speechify-client")

BASE_URL = "https://api.sws.speechify.com/v1"
API_KEY = os.getenv("SPEECHIFY_API_KEY")


class SpeechifyClient:
    """Client for Speechify voice cloning and TTS"""
    
    def __init__(self):
        if not API_KEY:
            raise ValueError("SPEECHIFY_API_KEY not found in environment")
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
        }
    
    async def create_clone(
        self,
        file_path: str,
        voice_name: str = "user_voice",
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        consent_json: Optional[str] = None,
        language: str = "en",
        locale: str = "en-US",
        gender: str = "notSpecified",
        description: Optional[str] = None,
    ) -> Optional[str]:
        """Create a voice clone from audio sample
        
        Per Speechify docs: https://docs.sws.speechify.com/docs/features/voice-cloning
        - POST to /v1/voices
        - Required: audio file, voice_name, consent (must be true)
        - Audio: 10-30 seconds, <5MB, clear speech
        """
        try:
            from pathlib import Path
            
            print("\n" + "="*60)
            print("🎨 VOICE CLONING - POSTING TO SPEECHIFY API")
            print("="*60)
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"❌ ERROR: Audio file not found: {file_path}")
                logger.error(f"❌ Audio file not found: {file_path}")
                return None
            
            file_size = file_path_obj.stat().st_size
            print(f"📤 Uploading: {file_path_obj.name}")
            print(f"📁 File size: {file_size / 1024:.2f}KB ({file_size / 1024 / 1024:.2f}MB)")
            print(f"🔗 API Endpoint: {BASE_URL}/voices")
            logger.info(f"🎨 Creating voice clone from: {file_path} ({file_size / 1024:.2f}KB)")
            
            # Verify file size (Speechify requirement: <5MB)
            if file_size > 5 * 1024 * 1024:
                print(f"❌ ERROR: File too large: {file_size / 1024 / 1024:.2f}MB (max 5MB)")
                logger.error(f"❌ File too large: {file_size / 1024 / 1024:.2f}MB (max 5MB)")
                return None
            
            print(f"✅ File size validation: OK")
            
            # Read audio file
            print("📖 Reading audio file...")
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            
            # Prepare form data (multipart/form-data as per Speechify API)
            print("📦 Preparing multipart/form-data...")
            import json
            from datetime import datetime
            
            # Build consent JSON per docs: must include fullName (camelCase) and email
            if consent_json:
                consent_json_string = consent_json
                try:
                    parsed = json.loads(consent_json_string)
                    full_name_value = parsed.get("fullName") or parsed.get("full_name") or full_name or "Voice Cloning User"
                except Exception:
                    full_name_value = full_name or "Voice Cloning User"
            else:
                fn = full_name or "Voice Cloning User"
                user_email = email or f"user@{fn.replace(' ', '').lower()}.example.com"
                consent_payload = {
                    "fullName": fn,  # camelCase as per docs
                    "email": user_email,  # required per docs
                    "given_by": fn,
                    "text": f"I, {fn}, consent to Speechify using my recording to create a cloned voice for this project.",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "confirmation": True,
                }
                consent_json_string = json.dumps(consent_payload)
                full_name_value = fn
            
            data = aiohttp.FormData()
            # Per docs: name, gender (required), display_name, consent (with fullName+email), language, locale, sample
            data.add_field("name", voice_name)
            data.add_field("gender", gender)  # required enum: male/female/notSpecified
            data.add_field("display_name", full_name_value)
            data.add_field("consent", consent_json_string)
            data.add_field("language", language)
            data.add_field("locale", locale)
            if description:
                data.add_field("description", description)
            
            print(f"📋 Consent JSON: {consent_json_string[:100]}...")  # Log first 100 chars
            logger.debug(f"Consent payload: {consent_json_string}")
            
            # Add audio file (Speechify requires 'sample' field name, not 'audio')
            # Use file-like object for proper multipart upload
            data.add_field(
                "sample",  # Speechify API expects 'sample' field name
                audio_bytes,
                filename=file_path_obj.name,
                content_type="audio/wav"  # WAV format
            )
            
            print("🔑 Fields added: name, gender, display_name, consent (with fullName+email), language, locale, sample")
            print(f"🔍 Field values preview:")
            print(f"   name: {voice_name}")
            print(f"   gender: {gender}")
            print(f"   display_name: {full_name_value}")
            print(f"   consent (first 100 chars): {consent_json_string[:100]}...")
            # Debug-dump multipart fields (best-effort; uses internal API)
            try:
                for idx, fld in enumerate(getattr(data, "_fields", []), start=1):
                    # fld: (name, value, content_type, headers)
                    # In newer aiohttp versions it's (name, value) and value has .filename/.content_type
                    name = None
                    filename = None
                    content_type_dbg = None
                    preview = None
                    try:
                        name = fld[0]
                        val = fld[1]
                        if hasattr(val, 'filename') and val.filename:
                            filename = val.filename
                        if hasattr(val, 'content_type') and val.content_type:
                            content_type_dbg = val.content_type
                        # Try to preview text payloads
                        if hasattr(val, 'value') and isinstance(val.value, (str, bytes)):
                            pv = val.value if isinstance(val.value, str) else val.value.decode(errors='ignore')
                            preview = (pv[:80] + '...') if len(pv) > 80 else pv
                    except Exception:
                        pass
                    print(f"🧩 Part {idx}: name={name} filename={filename} content_type={content_type_dbg} preview={preview}")
            except Exception as e:
                print(f"⚠️ Could not introspect FormData fields: {e}")
            print("📤 POST request sending...")
            logger.info(f"📤 POSTing to {BASE_URL}/voices with {len(audio_bytes)} bytes audio")
            
            # Robust networking: higher timeouts + retry
            timeout = aiohttp.ClientTimeout(total=180, connect=60, sock_connect=60, sock_read=120)
            attempts = 3
            last_err = None
            result = None
            voice_id = None

            for attempt in range(1, attempts + 1):
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            f"{BASE_URL}/voices",
                            headers=self.headers,
                            data=data,
                        ) as resp:
                            print(f"📥 Response status: {resp.status}")
                            if resp.status != 200:
                                error_text = await resp.text()
                                error_headers = dict(resp.headers)
                                print(f"❌ ERROR: Clone creation failed: {resp.status}")
                                print(f"📄 Error body (full): {error_text}")
                                print(f"📋 Response headers: {error_headers}")
                                logger.error(f"❌ Clone creation failed: {resp.status} - {error_text}")
                                return None
                            result = await resp.json()
                            voice_id = result.get("id")
                            print(f"✅ POST succeeded! Response: {result}")
                            print(f"🎤 Voice ID: {voice_id}")
                            break
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                    last_err = e
                    logger.warning(f"Network error on attempt {attempt}/{attempts}: {e}")
                    await asyncio.sleep(5 * attempt)  # simple backoff

            if result is None or voice_id is None:
                if last_err:
                    logger.error(f"❌ Voice cloning request failed after retries: {last_err}")
                return None

            # Validate voice_id
            if not voice_id:
                print(f"❌ ERROR: No voice_id in response")
                print(f"📄 Response: {result}")
                logger.error(f"❌ No voice_id in response: {result}")
                return None

            print(f"✅ Voice clone created: {voice_id}")
            print(f"🔄 Polling for readiness via /voices list (10-30s)...")
            logger.info(f"🔄 Voice clone started: {voice_id}, polling list for readiness...")

            # Poll by listing voices and checking if our voice appears (max 30 attempts, 3s each)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for attempt in range(30):
                    await asyncio.sleep(3)
                    try:
                        async with session.get(
                            f"{BASE_URL}/voices",
                            headers=self.headers,
                        ) as list_resp:
                            elapsed = (attempt + 1) * 3
                            if list_resp.status != 200:
                                print(f"⚠️ Voices list failed: HTTP {list_resp.status} (elapsed {elapsed}s)")
                                logger.warning(f"Voices list failed: {list_resp.status}")
                                continue

                            voices = await list_resp.json()
                            found = None
                            if isinstance(voices, list):
                                for v in voices:
                                    if v.get("id") == voice_id:
                                        found = v
                                        break

                            if found:
                                print(f"\n✅ VOICE CLONE READY (found in list)!")
                                print(f"🎤 Voice ID: {voice_id}")
                                print(f"⏱️ Total time: {elapsed}s")
                                print("="*60 + "\n")
                                logger.info(f"✅ Voice clone ready (listed): {voice_id}")
                                return voice_id

                            print(f"📊 Not listed yet (attempt {attempt + 1}/30, elapsed {elapsed}s)...")
                    except Exception as poll_err:
                        logger.debug(f"Polling error: {poll_err}")
                        continue

            print(f"\n❌ TIMEOUT: Voice cloning took longer than 90 seconds")
            print("="*60 + "\n")
            logger.error("❌ Voice cloning timeout after 90 seconds")
            return None
                    
        except Exception as e:
            logger.error(f"❌ Voice cloning error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def tts(self, voice_id: str, text: str, model: str = "simba-english") -> Optional[bytes]:
        """Generate speech using cloned voice"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{BASE_URL}/audio/speech",
                    headers={
                        **self.headers,
                        "Accept": "audio/pcm",
                        "Content-Type": "application/json",
                    },
                    json={
                        "voice_id": voice_id,
                        "model": model,
                        "input": text,
                        "format": "pcm_16000",  # 16kHz PCM (as before)
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.read()
                        content_type = resp.headers.get("Content-Type", "").lower()
                        
                        # Speechify returns JSON with base64-encoded audio
                        if "json" in content_type:
                            try:
                                import json
                                import base64
                                logger.debug(f"Parsing JSON response, size: {len(response_data)} bytes")
                                json_data = json.loads(response_data.decode())
                                logger.debug(f"JSON parsed successfully, keys: {list(json_data.keys())[:5]}")
                                
                                # Extract base64 audio (Speechify may use 'audio_data' or 'audio_base64')
                                audio_b64 = json_data.get("audio_base64") or json_data.get("audio_data")
                                if audio_b64:
                                    logger.debug(f"Found base64 audio key, length: {len(audio_b64)} chars")
                                    audio_bytes = base64.b64decode(audio_b64)
                                    audio_format = json_data.get('audio_format', 'unknown')
                                    logger.info(f"✅ TTS generated: {len(audio_bytes)} bytes (decoded from base64), format: {audio_format}")
                                    
                                    # Debug: Save TTS output for testing (Issue #4) - always save first one for debugging
                                    import pathlib
                                    import time
                                    debug_path = pathlib.Path("generated_audio") / f"speechify_test_{int(time.time())}.wav"
                                    debug_path.parent.mkdir(exist_ok=True)
                                    try:
                                        with open(debug_path, "wb") as f:
                                            f.write(audio_bytes)
                                        logger.info(f"🔍 Auto-saved TTS output to {debug_path} (for debugging)")
                                    except Exception as save_err:
                                        logger.warning(f"Could not save debug file: {save_err}")
                                    
                                    return audio_bytes
                                else:
                                    logger.error(f"❌ No base64 audio in JSON response. Keys: {list(json_data.keys())[:10]}")
                                    return None
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON decode error: {e}, response preview: {response_data[:200]}")
                                return None
                            except Exception as e:
                                logger.error(f"❌ Failed to parse JSON/base64 audio: {e}", exc_info=True)
                                return None
                        else:
                            # Raw PCM (fallback)
                            logger.info(f"✅ TTS generated: {len(response_data)} bytes (raw PCM), content-type: {content_type}")
                            return response_data
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ TTS failed: {resp.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            return None


