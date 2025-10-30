"""
Speechify Service for Voice Cloning and TTS
Uses Speechify API for instant voice cloning and text-to-speech
"""

import os
import requests
import json
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv(".env.local")
logger = logging.getLogger("speechify-service")


class SpeechifyService:
    """Service for voice cloning and TTS using Speechify API"""
    
    def __init__(self):
        self.api_key = os.getenv("SPEECHIFY_API_KEY")
        if not self.api_key:
            raise ValueError("SPEECHIFY_API_KEY not found in environment variables")
        
        self.base_url = "https://api.sws.speechify.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.save_responses = True  # Always save TTS responses
        self.default_voice_id = None  # Will be set after getting available voices
    
    def get_available_voices(self) -> Optional[list]:
        """Get list of available voices from Speechify API"""
        try:
            url = f"{self.base_url}/voices"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                # Response can be a list directly or wrapped in an object
                if isinstance(result, list):
                    voices = result
                else:
                    voices = result.get("voices", []) or result.get("data", []) or []
                
                # Set default voice ID if not set (use first shared voice)
                if not self.default_voice_id and voices:
                    for voice in voices:
                        # Handle both dict and object access
                        voice_type = voice.get("type") if isinstance(voice, dict) else getattr(voice, "type", None)
                        voice_id = voice.get("id") if isinstance(voice, dict) else getattr(voice, "id", None)
                        display_name = voice.get("display_name") if isinstance(voice, dict) else getattr(voice, "display_name", "")
                        
                        if voice_type == "shared" or voice_type is None:  # Accept shared or any if type is missing
                            self.default_voice_id = voice_id
                            logger.info(f"Using default voice: {display_name} (ID: {self.default_voice_id})")
                            break
                
                return voices
            else:
                print(f"Failed to get voices: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error getting voices: {str(e)}")
            return None
    
    def clone_voice(self, audio_file_path: str, name: str = "Cloned Voice", gender: str = "notSpecified", locale: str = "en-US") -> Optional[str]:
        """Create a cloned voice from audio sample (10-30 seconds recommended)
        
        Args:
            audio_file_path: Path to audio file (WAV/MP3)
            name: Display name for the cloned voice
            gender: "male", "female", or "notSpecified"
            locale: Language locale (e.g., "en-US", "en-IN")
        """
        try:
            with open(audio_file_path, "rb") as f:
                # Speechify API expects "sample" field for audio file, not "file"
                files = {
                    "sample": (os.path.basename(audio_file_path), f, "audio/wav")
                }
                data = {
                    "name": name,
                    "gender": gender,
                    "locale": locale,
                    "consent": "true"  # Required: confirm voice ownership
                }
                
                response = requests.post(
                    f"{self.base_url}/voices",
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code in [200, 201]:
                result = response.json()
                # Speechify returns voice with "id" field
                voice_id = result.get("id")
                display_name = result.get("display_name", name)
                print(f"✅ Voice cloned successfully!")
                print(f"   Voice ID: {voice_id}")
                print(f"   Display Name: {display_name}")
                return voice_id
            else:
                print(f"Voice cloning failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error cloning voice: {str(e)}")
            return None
    
    def generate_speech_stream(self, text: str, voice_id: str, save_to_file: bool = False) -> Optional[bytes]:
        """Generate speech using Speechify API (blocking REST endpoint)
        
        Returns PCM audio bytes or None if failed.
        For streaming, use WebSocket endpoint separately.
        """
        try:
            url = f"{self.base_url}/audio/speech"
            headers = {**self.headers, "Content-Type": "application/json"}
            # Speechify API requires "input" field, not "text"
            data = {
                "voice_id": voice_id,
                "input": text,  # ✅ Changed from "text" to "input"
                "format": "pcm_16000"  # Request 16kHz PCM format
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                audio_bytes = response.content
                
                # Optional: Save to file for testing
                if save_to_file or self.save_responses:
                    try:
                        import os
                        import uuid
                        import wave
                        import numpy as np
                        debug_dir = "generated_audio"
                        if not os.path.exists(debug_dir):
                            os.makedirs(debug_dir)
                        
                        # Convert PCM to WAV for easier playback
                        file_id = uuid.uuid4().hex[:12]
                        filepath = os.path.join(debug_dir, f"response_{file_id}.wav")
                        
                        # PCM is 16-bit signed int16 at 16kHz
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        with wave.open(filepath, "wb") as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(16000)  # 16kHz
                            wf.writeframes(audio_array.tobytes())
                        
                        print(f"✅ Saved audio response to: {filepath}")
                        # Note: Auto-playback disabled - LiveKit console mode already handles audio playback
                        # Playing audio via winsound causes conflicts with LiveKit's audio system
                    except Exception as save_err:
                        print(f"Could not save audio file: {save_err}")
                
                return audio_bytes
            else:
                print(f"TTS failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

