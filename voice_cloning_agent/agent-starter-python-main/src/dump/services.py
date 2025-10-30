"""
Voice Cloning Services for LiveKit Agent
ElevenLabs for TTS and Voice Cloning
AssemblyAI for STT
OpenAI for LLM
"""

import os
import requests
import time
import tempfile
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ElevenLabsService:
    """Service for voice cloning and TTS using ElevenLabs"""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {"xi-api-key": self.api_key}
        self.save_responses = False  # Set to True to save all TTS responses to generated_audio/
    
    def clone_voice(self, audio_file_path: str, name: str = "Cloned Voice") -> Optional[str]:
        """Clone voice from audio sample"""
        try:
            from pathlib import Path
            ext = Path(audio_file_path).suffix.lower()
            content_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            
            with open(audio_file_path, "rb") as f:
                files = {"files": (os.path.basename(audio_file_path), f, content_type)}
                data = {"name": name, "description": "Voice cloned from user audio sample"}
                
                response = requests.post(
                    f"{self.base_url}/voices/add",
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code in [200, 201]:
                result = response.json()
                voice_id = result.get("voice_id") or result.get("id")
                return voice_id if voice_id else None
            else:
                print(f"Voice cloning failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error cloning voice: {str(e)}")
            return None
    
    def generate_speech_stream(self, text: str, voice_id: str, save_to_file: bool = False) -> Optional[bytes]:
        """Generate speech and return audio bytes for streaming"""
        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            headers = {**self.headers, "Content-Type": "application/json"}
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.6, "similarity_boost": 0.8}
            }
            
            # ElevenLabs API: output_format as query parameter (not in body)
            # Add as query param to URL
            url_with_format = f"{url}?output_format=pcm_16000"
            
            # Increased timeout for reliability (60 seconds)
            response = requests.post(url_with_format, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                audio_bytes = response.content
                
                # Optional: Save to file for testing
                if save_to_file or self.save_responses:
                    try:
                        import os
                        import uuid
                        from datetime import datetime
                        debug_dir = "generated_audio"
                        if not os.path.exists(debug_dir):
                            os.makedirs(debug_dir)
                        
                        # Save as MP3 (convert PCM to MP3 or save raw PCM)
                        # For testing, save raw response (ElevenLabs returns MP3 if pcm_16000 not accepted)
                        if audio_bytes[:3] == b'ID3':
                            # It's MP3
                            file_ext = ".mp3"
                        else:
                            # It's PCM - save as WAV for easier playback
                            file_ext = ".wav"
                            # Convert PCM to WAV
                            import wave
                            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            temp_wav_path = temp_wav.name
                            temp_wav.close()
                            with wave.open(temp_wav_path, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                import numpy as np
                                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                                wf.writeframes(audio_array.tobytes())
                            
                            with open(temp_wav_path, "rb") as f:
                                audio_bytes = f.read()
                            os.unlink(temp_wav_path)
                        
                        file_id = uuid.uuid4().hex[:12]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Always save as WAV for consistency and easier playback
                        if audio_bytes[:3] == b'ID3':
                            # Convert MP3 to WAV using pydub
                            try:
                                from pydub import AudioSegment
                                import tempfile
                                # Save MP3 to temp file
                                tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                                tmp_mp3.write(audio_bytes)
                                tmp_mp3.close()
                                # Convert to WAV
                                audio_segment = AudioSegment.from_mp3(tmp_mp3.name)
                                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                                audio_segment.export(tmp_wav.name, format="wav")
                                tmp_wav.close()
                                # Read WAV bytes
                                with open(tmp_wav.name, "rb") as f:
                                    audio_bytes = f.read()
                                # Cleanup
                                os.unlink(tmp_mp3.name)
                                os.unlink(tmp_wav.name)
                                file_ext = ".wav"
                            except Exception as conv_err:
                                print(f"MP3 to WAV conversion failed, saving as MP3: {conv_err}")
                                file_ext = ".mp3"
                        else:
                            # Already PCM - save as WAV
                            import wave
                            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            temp_wav_path = temp_wav.name
                            temp_wav.close()
                            with wave.open(temp_wav_path, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                import numpy as np
                                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                                wf.writeframes(audio_array.tobytes())
                            
                            with open(temp_wav_path, "rb") as f:
                                audio_bytes = f.read()
                            os.unlink(temp_wav_path)
                            file_ext = ".wav"
                        
                        filename = f"response_{file_id}.wav"
                        filepath = os.path.join(debug_dir, filename)
                        
                        with open(filepath, "wb") as f:
                            f.write(audio_bytes)
                        
                        print(f"Saved audio response to: {filepath}")
                        
                        # Audio saved - print to terminal only (no external player)
                        print(f"✅ Audio saved and ready: {filepath}")
                    except Exception as save_err:
                        print(f"Could not save audio file: {save_err}")
                
                return audio_bytes
            else:
                print(f"TTS failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

