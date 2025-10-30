"""
Custom Speechify TTS Adapter for LiveKit
Streams audio directly using Speechify API
"""

import asyncio
import logging
from typing import AsyncIterator
from livekit import rtc
from livekit.agents import tts

from speechify_service import SpeechifyService

logger = logging.getLogger("speechify-tts")


class SpeechifyTTSBase:
    """Base class for Speechify TTS synthesis"""
    
    def __init__(self, speechify: SpeechifyService, voice_id: str):
        self.speechify = speechify
        self.voice_id = voice_id
    
    async def synthesize(self, text: str) -> AsyncIterator[rtc.AudioFrame]:
        """Synthesize text to speech and stream audio frames"""
        try:
            # Generate audio from Speechify (make blocking call non-blocking)
            logger.info(f"Generating speech for: {text[:50]}...")
            # Run blocking API call in thread pool to avoid hanging
            audio_bytes = await asyncio.to_thread(
                self.speechify.generate_speech_stream,
                text,
                self.voice_id,
                self.speechify.save_responses
            )
            
            if not audio_bytes:
                logger.error("Failed to generate audio from Speechify")
                return
            
            # Decode PCM audio (16-bit signed int16 at 16kHz)
            try:
                import numpy as np
                
                # Ensure signed 16-bit PCM format
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                if audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)
                
                sample_rate = 16000  # Speechify returns 16kHz PCM
                target_sample_rate = 24000  # LiveKit needs 24kHz
                
                logger.info(f"Using PCM format: {len(audio_array)} samples at {sample_rate}Hz")
                
                # Resample from 16kHz to 24kHz (using scipy or basic interpolation)
                if sample_rate != target_sample_rate:
                    try:
                        from scipy import signal
                        num_samples = int(len(audio_array) * 24000 / sample_rate)
                        audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                        logger.info(f"Resampled to {target_sample_rate}Hz")
                    except ImportError:
                        # Simple resampling without scipy
                        logger.warning("scipy not available, using basic resampling")
                        indices = np.linspace(0, len(audio_array)-1, int(len(audio_array) * 24000 / sample_rate))
                        audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
                
                # Ensure int16
                if audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)
                
                # Yield audio frames in small chunks for real-time streaming
                # IMPORTANT: Yield first chunk immediately to start playback fast
                chunk_size = 480  # 20ms at 24kHz (smaller chunks for faster delivery)
                
                # Yield first chunk immediately (before processing rest)
                if len(audio_array) > 0:
                    first_chunk = audio_array[:chunk_size]
                    if len(first_chunk) < chunk_size:
                        padding = np.zeros(chunk_size - len(first_chunk), dtype=np.int16)
                        first_chunk = np.concatenate([first_chunk, padding])
                    
                    first_frame = rtc.AudioFrame(
                        data=first_chunk.tobytes(),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(first_chunk),
                    )
                    yield first_frame
                    await asyncio.sleep(0.001)  # Small yield to let LiveKit process
                
                # Yield remaining chunks
                for i in range(chunk_size, len(audio_array), chunk_size):
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
                    await asyncio.sleep(0.001)  # Small yield for responsiveness
                    
            except ImportError as e:
                logger.error(f"Required library missing: {e}")
                logger.error("Install: pip install numpy scipy")
                return
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                return
                
        except Exception as e:
            logger.error(f"Error in Speechify TTS synthesis: {e}")
            import traceback
            traceback.print_exc()
            return


class SpeechifyTTS(tts.TTS):
    """Speechify TTS adapter for LiveKit with streaming"""
    
    def __init__(self, voice_id: str):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),  # Use non-streaming mode with synthesize()
            sample_rate=24000,
            num_channels=1,
        )
        self.speechify = SpeechifyService()
        self._base = SpeechifyTTSBase(self.speechify, voice_id)
        self.voice_id = voice_id
    
    def set_voice(self, voice_id: str):
        """Update voice ID (for cloned voice)"""
        self.voice_id = voice_id
        self._base = SpeechifyTTSBase(self.speechify, voice_id)
        logger.info(f"TTS voice updated to: {voice_id}")
    
    def synthesize(self, text: str, *, conn_options=None):
        """Synthesize text to speech - returns ChunkedStream (non-streaming but fast)"""
        from livekit.agents.tts.tts import ChunkedStream, DEFAULT_API_CONNECT_OPTIONS
        
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        
        # Create ChunkedStream that will call _run to generate audio
        class SpeechifyChunkedStream(ChunkedStream):
            async def _run(self, output_emitter) -> None:
                """Run synthesis and emit audio frames - fixed for LiveKit timing"""
                try:
                    import uuid
                    import time
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
                        logger.info("AudioEmitter already initialized")
                    
                    # CRITICAL FIX: Push silent frame IMMEDIATELY (before Speechify API call)
                    # This satisfies LiveKit's ~2s timeout expectation
                    import numpy as np
                    silent_duration_ms = 100  # 100ms of silence
                    silent_samples = int(24000 * silent_duration_ms / 1000)  # 2400 samples at 24kHz
                    silent_frame = np.zeros(silent_samples, dtype=np.int16).tobytes()
                    
                    output_emitter.push(silent_frame)
                    await asyncio.sleep(0.05)  # Give LiveKit time to process silent frame
                    logger.info("Silent pre-frame pushed - LiveKit timeout prevented")
                    
                    # Collect frames from Speechify synthesis
                    frames_data = []
                    first_frame_pushed = False
                    
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
                    
                    logger.info(f"Collected {len(frames_data)} audio frames from Speechify")
                    
                    if not frames_data:
                        logger.error("No audio frames generated! Speechify returned empty audio.")
                        # Don't raise error - just push silent frame and log
                        # This prevents session crash
                        logger.warning("Pushing minimal silent frame to prevent session crash")
                        import numpy as np
                        silent_frame = np.zeros(480, dtype=np.int16).tobytes()  # 20ms at 24kHz
                        output_emitter.push(silent_frame)
                        output_emitter.flush()
                        logger.error("Speechify TTS failed - no audio frames available")
                        return  # Return gracefully instead of raising
                    
                    # Push remaining frames gradually (first frame already pushed)
                    start_push = time.time()
                    pushed_count = 1 if first_frame_pushed else 0
                    start_idx = 1 if first_frame_pushed else 0
                    
                    for i in range(start_idx, len(frames_data)):
                        try:
                            output_emitter.push(frames_data[i])
                            pushed_count += 1
                            await asyncio.sleep(0.01)  # Smooth streaming delay
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
                    await asyncio.sleep(0.1)
                    
                    logger.info(f"✅ Stream finished: {pushed_count} frames pushed successfully")
                except asyncio.CancelledError:
                    logger.warning("TTS task was cancelled before completion")
                    return
                except Exception as e:
                    logger.error(f"Error in ChunkedStream._run: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        return SpeechifyChunkedStream(tts=self, input_text=text, conn_options=conn_options)

