"""
Speechify TTS Adapter for LiveKit
Real-time streaming TTS using Speechify API
"""

import asyncio
import logging
from typing import AsyncIterator
from livekit import rtc
from livekit.agents import tts

from speechify_client import SpeechifyClient

logger = logging.getLogger("speechify-tts")


class SpeechifyTTSBase:
    """Base class for Speechify TTS synthesis"""
    
    def __init__(self, speechify_client: SpeechifyClient, voice_id: str):
        self.speechify = speechify_client
        self.voice_id = voice_id
    
    async def synthesize(self, text: str) -> AsyncIterator[rtc.AudioFrame]:
        """Synthesize text to speech and stream audio frames"""
        try:
            logger.info(f"Generating speech for: {text[:50]}...")
            
            # Generate audio from Speechify (async)
            audio_bytes = await self.speechify.tts(self.voice_id, text)
            
            if not audio_bytes:
                logger.error("Failed to generate audio from Speechify")
                return
            
            import numpy as np
            import wave
            import io
            
            # Check if audio is WAV format (starts with "RIFF")
            if audio_bytes[:4] == b'RIFF' or audio_bytes[:4] == b'riff':
                # Extract PCM from WAV
                logger.info("Detected WAV format, extracting PCM...")
                wav_io = io.BytesIO(audio_bytes)
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    num_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert to mono if stereo
                    if num_channels == 2:
                        frames_array = np.frombuffer(frames, dtype=np.int16)
                        frames_array = frames_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                        frames = frames_array.tobytes()
                    
                    audio_bytes = frames
                    logger.info(f"Extracted PCM: {sample_rate}Hz, {num_channels}ch, {sample_width*8}-bit")
            else:
                # Assume raw PCM - trim odd byte if needed
                if len(audio_bytes) % 2 != 0:
                    logger.debug("PCM buffer length not multiple of 2; trimming last byte")
                    audio_bytes = audio_bytes[:-1]
                sample_rate = 16000  # Default for raw PCM

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            target_sample_rate = 24000  # LiveKit needs 24kHz
            
            logger.info(f"Using PCM format: {len(audio_array)} samples at {sample_rate}Hz")
            
            # Resample from 16kHz to 24kHz
            if sample_rate != target_sample_rate:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) * 24000 / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                    logger.info(f"Resampled to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("scipy not available, using basic resampling")
                    indices = np.linspace(0, len(audio_array)-1, int(len(audio_array) * 24000 / sample_rate))
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
            
            # Ensure int16
            if audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)
            
            # Check amplitude and normalize if needed (Issue #3)
            max_amplitude = np.max(np.abs(audio_array))
            logger.info(f"Audio amplitude range: {audio_array.min()} to {audio_array.max()} (max abs: {max_amplitude})")
            
            if max_amplitude > 0:
                # Normalize if too quiet (but avoid clipping)
                if max_amplitude < 1000:
                    logger.warning(f"Audio too quiet (max={max_amplitude}), normalizing...")
                    audio_array = (audio_array / max_amplitude * 32767 * 0.9).astype(np.int16)
                    logger.info(f"After normalization: {audio_array.min()} to {audio_array.max()}")
            
            # Clip to int16 range
            audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
            
            # Yield audio frames in exact 480-sample chunks (20ms at 24kHz) - Issue #1
            chunk_size = 480  # EXACTLY 20ms at 24kHz
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                
                # Pad if last chunk is smaller (must be exactly 480)
                if len(chunk) < chunk_size:
                    padding = np.zeros(chunk_size - len(chunk), dtype=np.int16)
                    chunk = np.concatenate([chunk, padding])
                
                # Ensure exact format - Issue #1
                frame = rtc.AudioFrame(
                    data=chunk.astype(np.int16).tobytes(),
                    sample_rate=24000,  # Must match LiveKit exactly
                    num_channels=1,     # Mono
                    samples_per_channel=480,  # EXACTLY 480 (20ms)
                )
                yield frame
                await asyncio.sleep(0.02)  # 20ms pacing for real-time streaming
                
        except Exception as e:
            logger.error(f"Error in Speechify TTS synthesis: {e}")
            import traceback
            traceback.print_exc()
            return


class SpeechifyTTS(tts.TTS):
    """Speechify TTS adapter for LiveKit"""
    
    def __init__(self, voice_id: str):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.speechify_client = SpeechifyClient()
        self._base = SpeechifyTTSBase(self.speechify_client, voice_id)
        self.voice_id = voice_id
    
    def set_voice(self, voice_id: str):
        """Update voice ID (for cloned voice)"""
        self.voice_id = voice_id
        self._base = SpeechifyTTSBase(self.speechify_client, voice_id)
        logger.info(f"TTS voice updated to: {voice_id}")
    
    def synthesize(self, text: str, *, conn_options=None):
        """Synthesize text to speech - returns ChunkedStream"""
        from livekit.agents.tts.tts import ChunkedStream, DEFAULT_API_CONNECT_OPTIONS
        
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        
        class SpeechifyChunkedStream(ChunkedStream):
            async def _run(self, output_emitter) -> None:
                """Run synthesis and emit audio frames"""
                try:
                    import uuid
                    import numpy as np
                    
                    request_id = str(uuid.uuid4())
                    
                    # Initialize emitter
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
                    
                    # Push silent frame immediately (prevents LiveKit timeout)
                    silent_duration_ms = 100
                    silent_samples = int(24000 * silent_duration_ms / 1000)
                    silent_frame = np.zeros(silent_samples, dtype=np.int16).tobytes()
                    output_emitter.push(silent_frame)
                    await asyncio.sleep(0.05)
                    logger.info("Silent pre-frame pushed")
                    
                    # Push frames IMMEDIATELY as they're generated (Issue #2 - continuous push)
                    frame_count = 0
                    has_audio = False
                    
                    async for frame in self._tts._base.synthesize(self._input_text):
                        frame_data = frame.data
                        
                        if not isinstance(frame_data, bytes):
                            if hasattr(frame_data, 'tobytes'):
                                frame_data = frame_data.tobytes()
                            else:
                                frame_data = np.array(frame_data, dtype=np.int16).tobytes()
                        
                        # Push immediately for real-time playback (Issue #2)
                        if frame_data and len(frame_data) > 0:
                            try:
                                output_emitter.push(frame_data)
                                frame_count += 1
                                has_audio = True
                                
                                # Log first few frames for debugging
                                if frame_count <= 3:
                                    logger.info(f"Pushed frame {frame_count} to LiveKit ({len(frame_data)} bytes)")
                                
                                # 20ms pacing (matches frame duration)
                                await asyncio.sleep(0.02)
                            except Exception as push_err:
                                logger.error(f"Error pushing frame {frame_count}: {push_err}")
                                break
                    
                    logger.info(f"Pushed {frame_count} audio frames total")
                    
                    if not has_audio:
                        logger.warning("No audio frames received - pushing silent frame")
                        silent_frame = np.zeros(480, dtype=np.int16).tobytes()
                        output_emitter.push(silent_frame)
                    
                    # Flush
                    output_emitter.flush()
                    logger.info("AudioEmitter flushed")
                    
                except asyncio.CancelledError:
                    logger.warning("TTS task cancelled")
                    return
                except Exception as e:
                    logger.error(f"Error in ChunkedStream._run: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        return SpeechifyChunkedStream(tts=self, input_text=text, conn_options=conn_options)
