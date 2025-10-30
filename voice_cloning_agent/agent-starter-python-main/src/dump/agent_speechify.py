"""
LiveKit Voice Cloning Agent using Speechify API
Real-time telephonic voice cloning with Speechify, OpenAI, and AssemblyAI
"""

import logging
import asyncio
import os
import tempfile
import wave
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins import openai, assemblyai
from livekit import rtc
import openai as openai_sdk  # Separate import for OpenAI SDK (for direct API calls)

from speechify_service import SpeechifyService
from speechify_tts import SpeechifyTTS
from voice_clone_manager import create_and_validate_clone

logger = logging.getLogger("speechify-cloning-agent")
load_dotenv(".env.local")


class SpeechifyCloningAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a friendly, helpful voice cloning assistant. Speak naturally like a human.
            - Be conversational, warm, and brief (1-2 sentences max).
            - Explain what's happening in simple, clear terms.
            - Use casual, friendly language - no robotic phrases.
            - After cloning: Just repeat what user says naturally, don't announce it.""",
        )
        self.cloned_voice_id = None
        self.state = "initial"  # initial, waiting_yes, recording, analyzing, cloning, conversation
        self.recording_start_time = None
        self.last_speech_time = None  # Track when user last spoke (legacy - used by new recording system)


def play_audio_in_terminal(wav_file_path: str):
    """Play WAV file in terminal using simple Python audio playback"""
    try:
        import platform
        import subprocess
        
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.PlaySound(wav_file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                logger.info(f"Playing audio in terminal: {wav_file_path}")
            except Exception as e:
                logger.info(f"Audio ready: {wav_file_path} (play manually if needed)")
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["afplay", wav_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Playing audio in terminal: {wav_file_path}")
        else:  # Linux
            subprocess.Popen(["aplay", wav_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Playing audio in terminal: {wav_file_path}")
    except Exception as e:
        logger.warning(f"Could not play audio in terminal: {e}")


def prewarm(proc: JobProcess):
    """Preload models"""
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.warning(f"VAD preload failed: {e}, will load on-demand")


async def capture_audio_bytes(ctx, session: AgentSession, duration: float) -> Optional[bytes]:
    """Capture audio frames from user and return raw PCM bytes for voice cloning
    
    Uses the JobContext's room to access participant audio tracks.
    Waits robustly for user to speak and captures full duration.
    """
    try:
        import numpy as np
        
        # Get room from context (not session)
        room = ctx.room
        if not room:
            logger.error("No room available in context")
            return None
        
        # LiveKit audio processing - collect from room participants
        sample_rate = 48000  # LiveKit default input sample rate
        audio_buffer = []
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"🎙️ Starting audio capture - waiting for {duration} seconds of user speech...")
        
        # Wait for participants to join (if not already)
        if not room.remote_participants:
            logger.info("Waiting for participant to join...")
            await asyncio.sleep(1.0)
        
        participants = list(room.remote_participants.values())
        if not participants:
            logger.error("No remote participants found for recording")
            return None
        
        user = participants[0]
        audio_track = None
        
        # Find and subscribe to audio track
        for track_pub in user.track_publications.values():
            if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_track = track_pub.track
                if not track_pub.subscribed:
                    await track_pub.set_subscribed(True)
                logger.info(f"✅ Subscribed to audio track: {audio_track.sid}")
                break
        
        if not audio_track:
            logger.error("No audio track found from user")
            return None
        
        logger.info(f"🔊 Audio track found, capturing {duration}s of audio (will wait for user to speak)...")
        
        # Collect frames directly from track - wait for user to speak
        frames_collected = 0
        has_audio = False
        silence_threshold = 500  # Minimum RMS value to consider as audio
        
        async def collect_frames():
            nonlocal frames_collected, has_audio
            try:
                async for frame in audio_track:
                    # Get frame data
                    frame_data = None
                    if hasattr(frame, 'data'):
                        frame_data = frame.data
                    elif hasattr(frame, 'samples'):
                        frame_data = np.array(frame.samples, dtype=np.int16)
                    else:
                        continue
                    
                    if frame_data is None:
                        continue
                    
                    # Convert to bytes
                    if isinstance(frame_data, bytes):
                        frame_bytes = frame_data
                    elif hasattr(frame_data, 'tobytes'):
                        frame_bytes = frame_data.tobytes()
                    else:
                        arr = np.array(frame_data, dtype=np.int16)
                        frame_bytes = arr.tobytes()
                    
                    # Check if this frame has actual audio (not just silence)
                    if len(frame_bytes) > 0:
                        audio_array = np.frombuffer(frame_bytes, dtype=np.int16)
                        rms = np.sqrt(np.mean(audio_array**2))
                        if rms > silence_threshold:
                            has_audio = True
                            logger.info(f"🔊 Audio detected (RMS: {rms:.1f})")
                    
                    audio_buffer.append(frame_bytes)
                    frames_collected += 1
                    
                    # Check timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration:
                        logger.info(f"⏱️ Duration reached: {elapsed:.1f}s, collected {frames_collected} frames")
                        break
                
                logger.info(f"✅ Collected {frames_collected} audio frames (has_audio={has_audio})")
            except asyncio.CancelledError:
                logger.info("Audio collection cancelled")
            except Exception as e:
                logger.error(f"Error collecting frames: {e}")
                import traceback
                traceback.print_exc()
        
        # Record for specified duration (with extra buffer for connection)
        try:
            await asyncio.wait_for(collect_frames(), timeout=duration + 10.0)
        except asyncio.TimeoutError:
            logger.warning(f"Audio collection timed out after {duration + 10.0}s")
        
        # Combine all audio frames into raw PCM bytes
        if audio_buffer:
            total_bytes = b''.join(audio_buffer)
            total_samples = len(total_bytes) // 2  # 16-bit
            
            if total_samples == 0:
                logger.error("No audio samples collected")
                return None
            
            actual_duration = total_samples / sample_rate
            logger.info(f"✅ Total audio collected: {total_samples} samples ({actual_duration:.2f}s)")
            
            if not has_audio:
                logger.warning("⚠️ No significant audio detected (mostly silence)")
            
            # Convert to numpy array for resampling
            audio_array = np.frombuffer(total_bytes, dtype=np.int16)
            
            # Resample to 22050 Hz (Speechify recommended for cloning)
            target_sample_rate = 22050
            
            if sample_rate != target_sample_rate:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                    logger.info(f"✅ Resampled to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("scipy not available, using original sample rate")
            
            # Return raw PCM bytes (int16, mono, 22050Hz)
            return audio_array.astype(np.int16).tobytes()
        else:
            logger.error("❌ No audio frames collected")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error capturing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


async def transcribe_audio(assemblyai_stt, audio_file_path: str) -> Optional[str]:
    """Transcribe audio file using AssemblyAI to check if recording has enough words"""
    try:
        # Use AssemblyAI to transcribe the audio
        # Note: AssemblyAI plugin might need file upload - check their API
        # For now, return a placeholder that OpenAI can analyze based on file size/duration
        
        import os
        file_size = os.path.getsize(audio_file_path)
        duration_estimate = file_size / (22050 * 2)  # Rough estimate: bytes / (sample_rate * 2 bytes per sample)
        
        logger.info(f"Transcription check: file_size={file_size}, estimated_duration={duration_estimate:.2f}s")
        
        # Return metadata for OpenAI analysis
        return f"Audio file: {file_size} bytes, ~{duration_estimate:.1f} seconds"
        
    except Exception as e:
        logger.warning(f"Transcription check failed: {e}")
        return None


async def entrypoint(ctx: JobContext):
    """Main entrypoint"""
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Wait for participant
    await ctx.wait_for_participant()
    
    # Initialize services
    speechify_service = SpeechifyService()
    agent = SpeechifyCloningAgent()
    
    # Create initial TTS adapter with female voice (will be updated after cloning)
    # Speechify uses voice_id, not voice parameter
    # Female voices: "sarah", "kate", "jane", "amy", "lisa", "nancy" (need to get actual IDs from API)
    # Using "sarah" as default female voice ID
    tts_adapter = SpeechifyTTS(voice_id="sarah")
    
    # Store reference for easy access (will update after cloning)
    session_tts_storage = {"tts": tts_adapter, "voice_id": None}
    
    logger.info("Using Speechify TTS - will switch to cloned voice after cloning")
    
    # Create session
    assemblyai_stt = assemblyai.STT()
    session = AgentSession(
        stt=assemblyai_stt,
        llm=openai.LLM(
            model="gpt-3.5-turbo",
            temperature=0.8,  # More creative, natural responses
            max_completion_tokens=50,  # Allow slightly longer for natural conversation
        ),
        tts=tts_adapter,
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(),
    )
    
    # Metrics
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Audio recording configuration
    SAMPLE_RATE_LIVEKIT = 48000  # LiveKit default
    SAMPLE_RATE_TARGET = 22050   # Speechify requirement for voice cloning (22050 preferred)
    MIN_RECORD_SECONDS = 10.0
    MAX_RECORD_SECONDS = 30.0
    SILENCE_SECONDS_TO_STOP = 4.5
    SILENCE_RMS_THRESHOLD = 300.0  # Tune this: typical voice RMS 1000-4000, silence <300
    
    async def countdown_and_start(session):
        """Hardcoded countdown before recording starts"""
        await session.say("Starting recording in 3... 2... 1... Go ahead and speak.", allow_interruptions=False)
        await asyncio.sleep(0.3)  # Small delay for system to prepare
        logger.info("✅ Countdown complete, recording started")
    
    async def get_user_audio_iterator(room, session):
        """Get async iterator for user audio frames (returns PCM int16 bytes)
        
        Works in both console mode and real room mode.
        In console mode, uses session's audio input stream.
        """
        import numpy as np
        
        # Try room-based audio first (real telephony mode)
        if room and room.remote_participants:
            participants = list(room.remote_participants.values())
            if participants:
                user = participants[0]
                audio_track = None
                
                # Find and subscribe to audio track
                for track_pub in user.track_publications.values():
                    if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                        audio_track = track_pub.track
                        if not track_pub.subscribed:
                            await track_pub.set_subscribed(True)
                        break
                
                if audio_track:
                    logger.info("✅ Using room audio track (telephony mode)")
                    
                    # Yield PCM int16 bytes from frames
                    async for frame in audio_track:
                        if agent.state != "recording":
                            break
                        
                        try:
                            # Convert frame to int16 bytes
                            if hasattr(frame, 'data'):
                                frame_data = frame.data
                            elif hasattr(frame, 'samples'):
                                frame_data = np.array(frame.samples, dtype=np.int16)
                            else:
                                continue
                            
                            if frame_data is None:
                                continue
                            
                            # Convert to bytes
                            if isinstance(frame_data, bytes):
                                yield frame_data
                            elif hasattr(frame_data, 'tobytes'):
                                yield frame_data.tobytes()
                            else:
                                arr = np.array(frame_data, dtype=np.int16)
                                yield arr.tobytes()
                        
                        except Exception as e:
                            logger.debug(f"Frame conversion error: {e}")
                    return
        
        # Fallback: Console mode - try to get audio from session's recognition system
        logger.warning("⚠️ No room audio track - console mode detected")
        logger.info("Attempting to capture from session audio input...")
        
        # In console mode (ChatCLI), audio comes through session's STT system
        # The session's audio recognition component has access to raw audio
        # Try to access it through the session's internal components
        
        # Option 1: Try session._audio_recognition (LiveKit internal)
        if hasattr(session, '_audio_recognition'):
            audio_rec = session._audio_recognition
            if hasattr(audio_rec, '_audio_source'):
                logger.info("✅ Found audio source via _audio_recognition")
                audio_source = audio_rec._audio_source
                if audio_source:
                    async for frame in audio_source:
                        if agent.state != "recording":
                            break
                        # Convert AudioFrame to int16 bytes
                        if hasattr(frame, 'data'):
                            yield frame.data if isinstance(frame.data, bytes) else frame.data.tobytes()
                        elif hasattr(frame, 'samples'):
                            yield np.array(frame.samples, dtype=np.int16).tobytes()
                    return
        
        # Option 2: Try session's activity audio source
        if hasattr(session, '_activity') and hasattr(session._activity, '_audio_source'):
            logger.info("✅ Found audio source via _activity")
            audio_source = session._activity._audio_source
            if audio_source:
                async for frame in audio_source:
                    if agent.state != "recording":
                        break
                    if hasattr(frame, 'data'):
                        yield frame.data if isinstance(frame.data, bytes) else frame.data.tobytes()
                    elif hasattr(frame, 'samples'):
                        yield np.array(frame.samples, dtype=np.int16).tobytes()
                return
        
        # Option 3: Use session's room input if available (for console, this might be the local participant)
        if room and hasattr(room, 'local_participant'):
            local = room.local_participant
            for track_pub in local.track_publications.values():
                if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info("✅ Using local participant audio track")
                    async for frame in track_pub.track:
                        if agent.state != "recording":
                            break
                        if hasattr(frame, 'data'):
                            yield frame.data if isinstance(frame.data, bytes) else frame.data.tobytes()
                        elif hasattr(frame, 'samples'):
                            yield np.array(frame.samples, dtype=np.int16).tobytes()
                    return
        
        # Last resort: For console mode, use pyaudio directly to capture microphone
        logger.warning("⚠️ Console mode: No session audio source found - using direct microphone capture")
        try:
            import pyaudio
            
            CHUNK = 4800  # ~100ms at 48kHz (bigger chunks = better throughput)
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 48000
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            logger.info("✅ Using PyAudio for direct microphone capture")
            
            try:
                while agent.state == "recording":
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    yield data
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
            
            return
            
        except ImportError:
            logger.error("PyAudio not installed. Install with: pip install pyaudio")
        except Exception as e:
            logger.error(f"PyAudio capture failed: {e}")
        
        # Final fallback: Error
        error_msg = (
            "No audio source found. In console mode, audio capture may require:\n"
            "1. Real LiveKit room (for telephony)\n"
            "2. PyAudio installed (pip install pyaudio) for direct mic capture\n"
            "3. Proper microphone permissions enabled"
        )
        logger.error(error_msg)
        raise ValueError("No audio track found - console mode requires PyAudio or real LiveKit room")
    
    async def record_user_audio(audio_frame_iterator, out_wav_path, session):
        """
        Record user audio with RMS-based silence detection.
        
        Args:
            audio_frame_iterator: Async iterator yielding PCM int16 bytes
            out_wav_path: Path to save WAV file
            session: AgentSession for logging
        
        Returns:
            Path to saved WAV file or None on failure
        """
        from recording_utils import rms_from_pcm16_bytes, write_wav_pcm16, concat_pcm_chunks, resample_audio
        
        q = asyncio.Queue()
        chunks = []
        recording_start = None
        last_speech_time = None
        prod_task = None
        
        # Producer: push frames into queue
        async def producer():
            nonlocal recording_start, last_speech_time
            frame_count = 0
            try:
                recording_start = asyncio.get_event_loop().time()
                last_speech_time = recording_start
                
                logger.info(f"🎙️ Producer started, collecting audio frames...")
                
                async for frame_bytes in audio_frame_iterator:
                    if agent.state != "recording":
                        logger.info("Recording state ended, stopping producer")
                        break
                    
                    if frame_bytes and len(frame_bytes) > 0:
                        await q.put(frame_bytes)
                        frame_count += 1
                        
                        # Quick RMS check for speech detection
                        try:
                            rms = rms_from_pcm16_bytes(frame_bytes)
                            if rms > SILENCE_RMS_THRESHOLD:
                                last_speech_time = asyncio.get_event_loop().time()
                                if frame_count % 50 == 0:  # Log every 50 frames
                                    logger.debug(f"RMS={rms:.1f} (speech detected, frames={frame_count})")
                        except Exception:
                            last_speech_time = asyncio.get_event_loop().time()
                    else:
                        logger.warning(f"Empty frame received, skipping")
                    
                    # Stop if max duration reached
                    elapsed = asyncio.get_event_loop().time() - recording_start
                    if elapsed >= MAX_RECORD_SECONDS:
                        logger.info(f"Max duration reached ({elapsed:.1f}s, {frame_count} frames)")
                        break
                
                logger.info(f"✅ Producer finished: {frame_count} frames collected")
                # Signal end
                await q.put(None)
            
            except Exception as e:
                logger.error(f"Producer error: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Producer collected {frame_count} frames before error")
                await q.put(None)
        
        # Start producer
        prod_task = asyncio.create_task(producer())
        
        # Consumer + silence detection
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                
                chunks.append(item)
                
                now = asyncio.get_event_loop().time()
                elapsed = now - recording_start
                
                # Wait for minimum recording length
                if elapsed < MIN_RECORD_SECONDS:
                    continue
                
                # Check for trailing silence
                silence_elapsed = now - last_speech_time
                if silence_elapsed >= SILENCE_SECONDS_TO_STOP:
                    logger.info(f"✅ Silence detected: {silence_elapsed:.1f}s silence after {elapsed:.1f}s total")
                    # Wait a bit for trailing frames
                    await asyncio.sleep(0.05)
                    break
            
            # Ensure producer finished
            if prod_task and not prod_task.done():
                try:
                    await asyncio.wait_for(prod_task, timeout=1.0)
                except asyncio.TimeoutError:
                    prod_task.cancel()
                    await prod_task
        
        except Exception as e:
            logger.error(f"Recording error: {e}")
            if prod_task:
                prod_task.cancel()
            return None
        
        # Combine chunks and save
        if not chunks:
            logger.error("No audio chunks recorded")
            return None
        
        logger.info(f"✅ Consumer finished: {len(chunks)} chunks collected")
        
        pcm_all = concat_pcm_chunks(chunks)
        total_bytes = len(pcm_all)
        total_samples_at_48k = total_bytes // 2  # int16 = 2 bytes per sample
        duration_at_48k = total_samples_at_48k / SAMPLE_RATE_LIVEKIT
        
        logger.info(f"📊 Raw audio: {total_bytes} bytes, {total_samples_at_48k} samples ({duration_at_48k:.2f}s at {SAMPLE_RATE_LIVEKIT}Hz)")
        
        if duration_at_48k < MIN_RECORD_SECONDS:
            logger.error(f"❌ Audio too short: {duration_at_48k:.2f}s < {MIN_RECORD_SECONDS}s minimum")
            return None
        
        # Resample if needed (48000Hz -> 22050Hz)
        if SAMPLE_RATE_LIVEKIT != SAMPLE_RATE_TARGET:
            pcm_all = resample_audio(pcm_all, SAMPLE_RATE_LIVEKIT, SAMPLE_RATE_TARGET)
            total_samples_resampled = len(pcm_all) // 2
            duration_resampled = total_samples_resampled / SAMPLE_RATE_TARGET
            logger.info(f"📊 Resampled audio: {total_samples_resampled} samples ({duration_resampled:.2f}s at {SAMPLE_RATE_TARGET}Hz)")
        
        # Save WAV file
        write_wav_pcm16(out_wav_path, pcm_all, sample_rate=SAMPLE_RATE_TARGET, nchannels=1)
        
        actual_duration = len(pcm_all) / (SAMPLE_RATE_TARGET * 2)
        logger.info(f"✅ Recording saved: {out_wav_path} ({actual_duration:.2f}s, {len(chunks)} chunks)")
        
        if actual_duration < MIN_RECORD_SECONDS:
            logger.warning(f"⚠️ Saved file duration ({actual_duration:.2f}s) is less than minimum ({MIN_RECORD_SECONDS}s)")
            return None
        
        return out_wav_path
    
    # Voice cloning flow
    @session.on("agent_tts_stopped")
    def on_agent_tts_stopped():
        logger.info(f"🔊 Agent TTS stopped | current state: {agent.state}")
        # State transitions handled by OpenAI detection now
    
    cloning_task_started = {"value": False}
    
    # Define clone function (uses saved WAV file - no analysis, just check size/duration)
    async def clone_with_file(wav_path: str):
        """Clone voice from saved WAV file - simple file size/duration check"""
        try:
            agent.state = "cloning"
            
            import wave
            import os
            
            if not os.path.exists(wav_path):
                logger.error(f"WAV file not found: {wav_path}")
                await session.say("Recording file not found. Please try again.", allow_interruptions=False)
                agent.state = "waiting_yes"
                cloning_task_started["value"] = False
                return
            
            # Read file info
            file_size = os.path.getsize(wav_path)
            with wave.open(wav_path, 'rb') as wf:
                file_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                actual_duration = file_frames / sample_rate
            
            logger.info(f"✅ Loaded WAV: {wav_path} ({actual_duration:.2f}s, {file_size} bytes, {sample_rate}Hz)")
            
            # Simple validation: file size > 20KB OR duration > 10s
            MIN_FILE_SIZE_KB = 20
            MIN_DURATION_SEC = 10.0
            
            if file_size < MIN_FILE_SIZE_KB * 1024 and actual_duration < MIN_DURATION_SEC:
                logger.warning(f"⚠️ Audio too short: {actual_duration:.1f}s, {file_size} bytes")
                await session.say("Recording was too short. Please speak for at least 10 seconds and try again.", allow_interruptions=False)
                agent.state = "waiting_yes"
                agent.recording_start_time = None
                cloning_task_started["value"] = False
                return
            
            # Good to go - proceed to cloning
            await session.say("Recording looks good. Creating your voice clone now.", allow_interruptions=False)
            
            caller_id = ctx.room.name if ctx.room else f"user_{int(time.time())}"
            
            # Use saved WAV file directly (already in correct format)
            temp_path = wav_path
            
            from voice_clone_manager import create_speechify_voice
            created = create_speechify_voice(
                sample_wav_path=temp_path,
                name=f"Voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                gender="notSpecified",
                locale="en-US",
                consent_text="User consented via live call"
            )
            
            voice_id = created.get("id") if created else None
            
            if not voice_id:
                await session.say("Voice upload failed. Please try again.", allow_interruptions=False)
                agent.state = "waiting_yes"
                cloning_task_started["value"] = False
                return
            
            # Poll for cloning completion with hardcoded messages
            from voice_clone_manager import wait_for_cloning_ready
            await session.say("Processing your voice. This will take a moment.", allow_interruptions=False)
            
            # Hardcoded fun facts (no LLM calls)
            fun_facts = [
                "Did you know honey never spoils?",
                "Octopuses have three hearts.",
                "Bananas are berries, but strawberries aren't.",
                "Sharks have been around longer than trees.",
                "A group of flamingos is called a flamboyance.",
            ]
            clone_ready = False
            msg_count = 0
            
            while not clone_ready and msg_count < 5:
                clone_ready = await wait_for_cloning_ready(voice_id, max_wait_seconds=10, poll_interval=5.0)
                if not clone_ready and msg_count < len(fun_facts):
                    await session.say(fun_facts[msg_count % len(fun_facts)], allow_interruptions=False)
                    msg_count += 1
            
            if clone_ready:
                agent.cloned_voice_id = voice_id
                new_tts_adapter = speechify.TTS(voice_id=voice_id, model="simba-english")
                session._tts = new_tts_adapter
                agent.state = "conversation"
                await session.say("Your voice clone is ready! Say anything and I'll repeat it.", allow_interruptions=False)
            else:
                await session.say("Voice cloning took too long. Please try again.", allow_interruptions=False)
                agent.state = "waiting_yes"
                cloning_task_started["value"] = False
        
        except Exception as e:
            logger.error(f"Error in clone: {e}")
            import traceback
            traceback.print_exc()
            await session.say("Something went wrong. Please try again.", allow_interruptions=False)
            agent.state = "waiting_yes"
            cloning_task_started["value"] = False
    
    @session.on("user_input_transcribed")
    def on_user_speech(ev):
        """Handle user speech transcription event"""
        # Extract text from event object
        text = ev.text if hasattr(ev, 'text') else str(ev)
        current_time = asyncio.get_event_loop().time()
        logger.info(f"📝 User: '{text[:50]}...' | state={agent.state}")
        
        # Update last speech time
        agent.last_speech_time = current_time
        
        # STATE 1: Initial/Waiting for "yes" - Simple keyword detection (no LLM delay)
        if agent.state in ["initial", "waiting_yes"]:
            # Prevent duplicate recordings
            if cloning_task_started["value"] or agent.state == "recording":
                logger.info("Already recording or cloning, ignoring input")
                return
            
            # Simple positive keyword detection
            text_lower = text.lower().strip()
            positive_keywords = ["yes", "yeah", "yep", "yup", "okay", "ok", "ready", "sure", "fine", "go ahead", "let's do it", "i'm ready", "let's go"]
            
            is_positive = any(keyword in text_lower for keyword in positive_keywords)
            logger.info(f"📝 User said: '{text[:50]}...' | Positive: {is_positive} | state: {agent.state}")
            
            if is_positive:
                logger.info("✅ Positive response detected! Starting recording...")
                
                # Set state immediately to prevent duplicates
                agent.state = "recording"
                cloning_task_started["value"] = True
                
                async def start_recording_flow():
                    """Complete recording flow: countdown → record → clone"""
                    try:
                        # Step 1: Countdown (hardcoded, fast)
                        await countdown_and_start(session)
                        
                        # Step 2: Record audio with silence detection
                        agent.recording_start_time = asyncio.get_event_loop().time()
                        caller_id = ctx.room.name if ctx.room else f"user_{int(time.time())}"
                        wav_path = f"generated_audio/recording_{caller_id}_{int(time.time())}.wav"
                        
                        audio_iter = get_user_audio_iterator(ctx.room, session)
                        saved_path = await record_user_audio(audio_iter, wav_path, session)
                        
                        if saved_path:
                            # Step 3: Clone directly (no analysis)
                            logger.info(f"✅ Recording saved: {saved_path}, starting clone")
                            asyncio.create_task(clone_with_file(saved_path))
                        else:
                            logger.error("Recording failed - no file saved")
                            await session.say("Could not save audio. Please try again.", allow_interruptions=False)
                            agent.state = "waiting_yes"
                            cloning_task_started["value"] = False
                    
                    except Exception as e:
                        logger.error(f"Recording flow error: {e}")
                        import traceback
                        traceback.print_exc()
                        await session.say("Recording error. Please try again.", allow_interruptions=False)
                        agent.state = "waiting_yes"
                        cloning_task_started["value"] = False
                
                # Start recording flow
                asyncio.create_task(session.say("Perfect! Get ready.", allow_interruptions=False))
                asyncio.create_task(start_recording_flow())
            else:
                logger.info("⏳ Not a positive response, staying in waiting_yes")
                # Stay in waiting_yes state
        
        # STATE 2: Recording - Ignore input during recording
        elif agent.state == "recording":
            logger.debug("Recording in progress, ignoring input")
            # Silence detection and state transitions handled in record_user_audio
        
        elif agent.state == "conversation":
            # User spoke - agent repeats via LLM (handled automatically by session)
            logger.info(f"💬 Conversation mode: User said '{text[:50]}', agent will repeat")
            pass
        else:
            # Unknown state - log it
            logger.warning(f"⚠️ Unknown agent state: {agent.state}, user said: '{text[:50] if text else 'N/A'}...'")
    
    # Start session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    # Hardcoded greeting (fast, no LLM delay)
    greeting = "Hey! I can clone your voice. Ready to record? Just say yes, then speak for about 15 to 20 seconds."
    await session.say(greeting, allow_interruptions=False)
    agent.state = "waiting_yes"
    
    # Connect
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

