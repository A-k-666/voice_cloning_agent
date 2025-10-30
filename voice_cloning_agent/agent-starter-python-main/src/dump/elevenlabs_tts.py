"""
Custom ElevenLabs TTS Adapter for LiveKit
Streams audio directly - no file playback
"""

import asyncio
import io
import logging
from typing import AsyncIterator
from livekit import rtc
from livekit.agents import tts

from services import ElevenLabsService

logger = logging.getLogger("elevenlabs-tts")


class ElevenLabsTTSBase:
    """Base class for ElevenLabs TTS synthesis"""
    
    def __init__(self, elevenlabs: ElevenLabsService, voice_id: str):
        self.elevenlabs = elevenlabs
        self.voice_id = voice_id
    
    async def synthesize(self, text: str) -> AsyncIterator[rtc.AudioFrame]:
        """Synthesize text to speech and stream audio frames"""
        try:
            # Generate audio from ElevenLabs (save to file if enabled)
            logger.info(f"Generating speech for: {text[:50]}...")
            audio_bytes = self.elevenlabs.generate_speech_stream(text, self.voice_id, save_to_file=self.elevenlabs.save_responses)
            
            if not audio_bytes:
                logger.error("Failed to generate audio from ElevenLabs")
                return
            
            # Decode audio - try PCM first, fallback to MP3 if needed
            try:
                import numpy as np
                
                # Check if it's MP3 (starts with ID3) or PCM
                if audio_bytes[:3] == b'ID3':
                    # It's MP3 - ElevenLabs API didn't accept pcm_16000 format
                    # Try to decode MP3 using pydub (requires ffmpeg) or return error
                    logger.warning("ElevenLabs returned MP3 instead of PCM. Trying to decode...")
                    try:
                        from pydub import AudioSegment
                        import tempfile
                        import os
                        
                        # Save to temp file and decode (close file before reading)
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                                tmp.write(audio_bytes)
                                tmp_path = tmp.name
                            
                            # File is now closed, safe to read
                            audio_segment = AudioSegment.from_mp3(tmp_path)
                            raw_audio = audio_segment.raw_data
                            sample_rate = audio_segment.frame_rate
                            # Convert to numpy
                            if audio_segment.channels == 2:
                                # Stereo to mono
                                audio_array = np.frombuffer(raw_audio, dtype=np.int16).reshape(-1, 2)
                                audio_array = np.mean(audio_array, axis=1).astype(np.int16)
                            else:
                                audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                            logger.info(f"Decoded MP3: {len(audio_array)} samples at {sample_rate}Hz")
                        finally:
                            # Clean up temp file
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass  # Ignore cleanup errors
                    except Exception as mp3_error:
                        logger.error(f"MP3 decode failed (ffmpeg required): {mp3_error}")
                        logger.error("Install ffmpeg: https://ffmpeg.org/download.html")
                        return  # No frames to yield
                else:
                    # It's PCM - ensure signed 16-bit format (same as Inworld)
                    import numpy as np
                    # Ensure signed 16-bit PCM format (convert if needed)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Guarantee it's int16 signed format (same as Inworld)
                    if audio_array.dtype != np.int16:
                        audio_array = audio_array.astype(np.int16)
                    # Convert back to bytes to ensure proper format
                    pcm16_bytes = audio_array.tobytes()
                    
                    sample_rate = 16000  # PCM format from ElevenLabs is 16kHz
                    logger.info(f"Using PCM format: {len(audio_array)} samples at {sample_rate}Hz (signed int16)")
                    
                    # Re-read from proper PCM bytes
                    audio_array = np.frombuffer(pcm16_bytes, dtype=np.int16)
                
                target_sample_rate = 24000  # LiveKit needs 24kHz
                
                # Resample from source rate to 24kHz (using scipy)
                if sample_rate != target_sample_rate:
                    try:
                        from scipy import signal
                        num_samples = int(len(audio_array) * 24000 / sample_rate)
                        audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                    except ImportError:
                        # Simple resampling without scipy
                        logger.warning("scipy not available, using basic resampling")
                        indices = np.linspace(0, len(audio_array)-1, int(len(audio_array) * 24000 / sample_rate))
                        audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
                
                # Ensure int16
                if audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)
                
                # Yield audio frames in small chunks for near real-time (no delay)
                chunk_size = 480  # Smaller chunks for faster delivery (20ms at 24kHz)
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]
                    
                    # Pad if last chunk is smaller
                    if len(chunk) < chunk_size:
                        padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                        chunk = np.concatenate([chunk, padding])
                    
                    frame = rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(chunk),
                    )
                    yield frame
                    # No sleep - faster delivery for near real-time
                    
            except ImportError as e:
                logger.error(f"Required library missing: {e}")
                logger.error("Install: pip install pydub numpy scipy")
                return
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                return
                    
        except Exception as e:
            logger.error(f"Error in ElevenLabs TTS synthesis: {e}")
            import traceback
            traceback.print_exc()
            return


class ElevenLabsTTS(tts.TTS):
    """ElevenLabs TTS adapter for LiveKit with streaming"""
    
    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # Use non-streaming mode with synthesize()
            sample_rate=24000,
            num_channels=1,
        )
        self.elevenlabs = ElevenLabsService()
        self._base = ElevenLabsTTSBase(self.elevenlabs, voice_id)
        self.voice_id = voice_id
    
    def set_voice(self, voice_id: str):
        """Update voice ID (for cloned voice)"""
        self.voice_id = voice_id
        self._base = ElevenLabsTTSBase(self.elevenlabs, voice_id)
        logger.info(f"TTS voice updated to: {voice_id}")
    
    def synthesize(self, text: str, *, conn_options=None):
        """Synthesize text to speech - returns ChunkedStream (non-streaming but fast)"""
        from livekit.agents.tts.tts import ChunkedStream, DEFAULT_API_CONNECT_OPTIONS, SynthesizedAudio
        
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        
        # Create ChunkedStream that will call _run to generate audio (non-streaming but fast)
        class ElevenLabsChunkedStream(ChunkedStream):
            async def _run(self, output_emitter) -> None:
                """Run synthesis and emit audio frames - fixed for LiveKit timing"""
                try:
                    import uuid
                    import time
                    request_id = str(uuid.uuid4())
                    
                    # Initialize emitter (AudioEmitter is already initialized by ChunkedStream)
                    # But ensure it's ready - don't call initialize() if already started
                    try:
                        output_emitter.initialize(
                            request_id=request_id,
                            sample_rate=24000,
                            num_channels=1,
                            mime_type="audio/pcm",
                        )
                        logger.info("AudioEmitter initialized")
                    except RuntimeError as e:
                        if "already started" not in str(e).lower():
                            raise
                        logger.info("AudioEmitter already initialized")
                    
                    # CRITICAL FIX: Push silent frame IMMEDIATELY (before ElevenLabs API call)
                    # This satisfies LiveKit's ~2s timeout expectation
                    # ElevenLabs takes 2-4s to generate, so LiveKit would timeout without this
                    import numpy as np
                    silent_duration_ms = 100  # 100ms of silence
                    silent_samples = int(24000 * silent_duration_ms / 1000)  # 2400 samples at 24kHz
                    silent_frame = np.zeros(silent_samples, dtype=np.int16).tobytes()
                    
                    output_emitter.push(silent_frame)
                    await asyncio.sleep(0.05)  # Give LiveKit time to process silent frame
                    logger.info("Silent pre-frame pushed - LiveKit timeout prevented")
                    
                    # Stream frames AS THEY ARE GENERATED (not collect all first)
                    # This ensures LiveKit receives frames immediately
                    frames_data = []
                    first_frame_pushed = False
                    
                    # Push frames as they come from ElevenLabs (streaming approach)
                    async for frame in self._tts._base.synthesize(self._input_text):
                        frame_data = frame.data
                        
                        # Ensure frame_data is bytes
                        if not isinstance(frame_data, bytes):
                            if hasattr(frame_data, 'tobytes'):
                                frame_data = frame_data.tobytes()
                            else:
                                import numpy as np
                                frame_data = np.array(frame_data, dtype=np.int16).tobytes()
                        
                        frames_data.append(frame_data)
                        
                        # Push FIRST frame immediately (critical for LiveKit timeout)
                        if not first_frame_pushed and frame_data:
                            output_emitter.push(frame_data)
                            first_frame_pushed = True
                            logger.info("First frame pushed to LiveKit immediately - playback started")
                            await asyncio.sleep(0.01)  # Let LiveKit process first frame
                    
                    logger.info(f"Collected {len(frames_data)} audio frames from ElevenLabs")
                    
                    if not frames_data:
                        logger.error("No audio frames generated! ElevenLabs returned empty audio.")
                        raise ValueError("No audio frames to push")
                    
                    # ALWAYS Save audio to file in generated_audio/ folder (WAV format)
                    try:
                        import os
                        import wave
                        import platform
                        import subprocess
                        debug_dir = "generated_audio"
                        if not os.path.exists(debug_dir):
                            os.makedirs(debug_dir)
                        file_id = uuid.uuid4().hex[:12]
                        test_file = os.path.join(debug_dir, f"response_{file_id}.wav")
                        
                        # Combine all frame data (ensure all are bytes)
                        combined_bytes = []
                        for fd in frames_data:
                            if isinstance(fd, bytes):
                                combined_bytes.append(fd)
                            elif hasattr(fd, 'tobytes'):
                                combined_bytes.append(fd.tobytes())
                            else:
                                # Convert to bytes
                                import numpy as np
                                if isinstance(fd, np.ndarray):
                                    combined_bytes.append(fd.astype(np.int16).tobytes())
                                else:
                                    arr = np.array(fd, dtype=np.int16)
                                    combined_bytes.append(arr.tobytes())
                        
                        total_bytes = b''.join(combined_bytes)
                        total_samples = len(total_bytes) // 2  # 16-bit = 2 bytes per sample
                        
                        if len(total_bytes) == 0:
                            logger.warning("No audio data to save - skipping file save")
                        else:
                            with wave.open(test_file, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(24000)
                                wf.writeframes(total_bytes)
                            
                            logger.info(f"✅ Saved audio response to {test_file} ({total_samples} samples, {total_samples/24000:.2f}s)")
                            
                            # Play in terminal (Windows winsound, Mac/Linux subprocess)
                            try:
                                if platform.system() == "Windows":
                                    try:
                                        import winsound
                                        winsound.PlaySound(test_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                                        logger.info(f"🔊 Playing in terminal: {test_file}")
                                    except Exception as win_err:
                                        logger.info(f"✅ Audio ready: {test_file} (play manually)")
                                elif platform.system() == "Darwin":  # macOS
                                    subprocess.Popen(["afplay", test_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    logger.info(f"🔊 Playing in terminal: {test_file}")
                                else:  # Linux
                                    subprocess.Popen(["aplay", test_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    logger.info(f"🔊 Playing in terminal: {test_file}")
                            except Exception as play_err:
                                logger.info(f"✅ Audio ready: {test_file} (play manually if needed)")
                    except Exception as save_error:
                        logger.error(f"❌ Could not save audio: {save_error}")
                        import traceback
                        traceback.print_exc()
                    
                    # Push remaining frames gradually (first frame already pushed during collection)
                    # Start counting from where we left off
                    start_push = time.time()
                    pushed_count = 1 if first_frame_pushed else 0
                    
                    # Push remaining frames gradually (smooth streaming)
                    start_idx = 1 if first_frame_pushed else 0
                    for i in range(start_idx, len(frames_data)):
                        try:
                            output_emitter.push(frames_data[i])
                            pushed_count += 1
                            
                            # Smooth streaming delay (10ms = ~100fps)
                            await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            logger.warning("Frame pushing cancelled - stopping early")
                            break
                        except Exception as push_err:
                            logger.error(f"Error pushing frame {i}: {push_err}")
                            break
                    
                    push_duration = time.time() - start_push
                    logger.info(f"Pushed {pushed_count}/{len(frames_data)} frames total in {push_duration:.3f}s")
                    
                    # Flush after all frames pushed
                    output_emitter.flush()
                    logger.info("AudioEmitter flushed")
                    
                    # Small delay after flush for LiveKit to process
                    await asyncio.sleep(0.1)
                    
                    logger.info(f"✅ Stream finished: {pushed_count} frames pushed successfully")
                except asyncio.CancelledError:
                    logger.warning("TTS task was cancelled before completion")
                    # Don't re-raise - allow graceful cancellation
                    return
                except Exception as e:
                    logger.error(f"Error in ChunkedStream._run: {e}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to let LiveKit handle it
        
        return ElevenLabsChunkedStream(tts=self, input_text=text, conn_options=conn_options)
