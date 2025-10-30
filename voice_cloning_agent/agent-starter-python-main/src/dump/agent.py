"""
LiveKit Voice Cloning Agent
Real-time telephonic voice cloning - minimal and simple
"""

import logging
import asyncio
import os
import tempfile
import wave
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

from services import ElevenLabsService
from elevenlabs_tts import ElevenLabsTTS

logger = logging.getLogger("voice-cloning-agent")
load_dotenv(".env.local")


class VoiceCloningAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a voice cloning agent. Keep responses extremely short.
            After voice cloning is done, simply repeat exactly what the user says.
            Do not add explanations. Just repeat their words verbatim. Maximum 100 tokens.""",
        )
        self.cloned_voice_id = None
        self.state = "greeting"  # greeting, recording, cloned, conversation
        self.recording_frames = []
        self.recording_start_time = None


def play_audio_in_terminal(wav_file_path: str):
    """Play WAV file in terminal using simple Python audio playback"""
    try:
        import platform
        import subprocess
        
        if platform.system() == "Windows":
            # Use winsound for simple beep or Python wave player
            try:
                import winsound
                winsound.PlaySound(wav_file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                logger.info(f"Playing audio in terminal: {wav_file_path}")
            except Exception as e:
                # Fallback to system default player (silent mode)
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


async def capture_audio_frames(session: AgentSession, duration: float) -> Optional[str]:
    """Capture audio frames from user for voice cloning using session's audio stream"""
    try:
        import numpy as np
        
        # Create file to save audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()
        
        # LiveKit audio processing - collect from session input
        sample_rate = 48000  # LiveKit default input sample rate
        target_duration_samples = int(duration * sample_rate)
        frames_collected = []
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Recording {duration} seconds...")
        
        # Use session's audio input stream to capture user audio
        # Create a temporary audio buffer
        audio_buffer = []
        
        # Collect audio frames from the room
        # We'll use the room's audio frame callback
        def on_audio_frame(frame: rtc.AudioFrame):
            """Callback to collect audio frames"""
            if frame.sample_rate == sample_rate:
                # Convert frame data to bytes
                frame_bytes = frame.data
                if isinstance(frame_bytes, bytes):
                    audio_buffer.append(frame_bytes)
                elif hasattr(frame_bytes, 'tobytes'):
                    audio_buffer.append(frame_bytes.tobytes())
                else:
                    # Convert numpy array to bytes
                    import numpy as np
                    arr = np.frombuffer(frame_bytes, dtype=np.int16) if not isinstance(frame_bytes, np.ndarray) else frame_bytes
                    audio_buffer.append(arr.tobytes())
            
            # Check if we have enough duration
            elapsed = asyncio.get_event_loop().time() - start_time
            return elapsed >= duration
        
        # Subscribe to user audio track properly
        participants = list(session.room.remote_participants.values())
        if not participants:
            logger.error("No remote participants found for recording")
            return None
        
        user = participants[0]
        audio_track = None
        
        # Find and subscribe to audio track
        for track_pub in user.track_publications.values():
            if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_track = track_pub.track
                # Ensure track is subscribed
                if not track_pub.subscribed:
                    await track_pub.set_subscribed(True)
                break
        
        if not audio_track:
            logger.error("No audio track found from user")
            return None
        
        logger.info(f"Audio track found, starting {duration}s recording...")
        
        # Collect frames directly from track
        collected_samples = 0
        samples_per_frame = 480  # Typical frame size at 48kHz (10ms)
        target_samples = int(duration * sample_rate)
        
        async def collect_frames():
            nonlocal collected_samples
            try:
                async for frame in audio_track:
                    if collected_samples >= target_samples:
                        break
                    
                    # Get frame data
                    if hasattr(frame, 'data'):
                        frame_data = frame.data
                    elif hasattr(frame, 'samples'):
                        import numpy as np
                        frame_data = np.array(frame.samples, dtype=np.int16).tobytes()
                    else:
                        frame_data = bytes(frame) if isinstance(frame, (bytes, bytearray)) else None
                    
                    if frame_data:
                        audio_buffer.append(frame_data)
                        collected_samples += len(frame_data) // 2  # 16-bit = 2 bytes per sample
                    
                    # Check timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration:
                        break
                
                logger.info(f"Collected {len(audio_buffer)} audio chunks, {collected_samples} samples")
            except Exception as e:
                logger.error(f"Error collecting frames: {e}")
                import traceback
                traceback.print_exc()
        
        # Record for specified duration
        await asyncio.wait_for(collect_frames(), timeout=duration + 5.0)
        
        # Combine all audio frames
        if audio_buffer:
            # Ensure we have enough audio
            total_bytes = sum(len(chunk) for chunk in audio_buffer)
            total_samples = total_bytes // 2  # 16-bit
            
            logger.info(f"Total audio collected: {total_samples} samples ({total_samples/sample_rate:.2f}s)")
            
            # Resample to 22050 Hz (ElevenLabs recommended for cloning)
            target_sample_rate = 22050
            
            # Convert to numpy array for resampling
            audio_array = np.frombuffer(b''.join(audio_buffer), dtype=np.int16)
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                    audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                    sample_rate = target_sample_rate
                    logger.info(f"Resampled to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("scipy not available, using original sample rate")
            
            # Save to WAV file in generated_audio/ folder (ElevenLabs needs WAV/MP3)
            debug_dir = "generated_audio"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Use generated_audio folder instead of temp file
            file_id = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            final_path = os.path.join(debug_dir, f"{file_id}.wav")
            
            with wave.open(final_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_array.tobytes())
            
            file_size = os.path.getsize(final_path)
            logger.info(f"✅ Saved recording: {final_path} ({file_size} bytes, {len(audio_array)/sample_rate:.2f}s)")
            
            # Clean up temp file if different
            if temp_path != final_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return final_path
        else:
            logger.error("No audio frames collected")
            return None
        
    except Exception as e:
        logger.error(f"Error capturing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


async def entrypoint(ctx: JobContext):
    """Main entrypoint"""
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Wait for participant
    await ctx.wait_for_participant()
    
    # Initialize services
    elevenlabs = ElevenLabsService()
    elevenlabs.save_responses = True  # Enable saving all TTS responses to generated_audio/ for testing
    agent = VoiceCloningAgent()
    
    # Create custom ElevenLabs TTS
    tts_adapter = ElevenLabsTTS(voice_id="21m00Tcm4TlvDq8ikWAM")
    logger.info("Audio responses will be saved to generated_audio/ folder for testing")
    
    # Create session - removed MultilingualModel() to fix Windows timeout
    session = AgentSession(
        stt=assemblyai.STT(),
        llm=openai.LLM(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_completion_tokens=100,
        ),
        tts=tts_adapter,
        # turn_detection removed - using default STT-based turn detection
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

    # Voice cloning flow - use sync callbacks with asyncio.create_task for async work
    @session.on("agent_tts_stopped")
    def on_agent_tts_stopped():
        if agent.state == "greeting":
            agent.state = "recording"
            agent.recording_start_time = asyncio.get_event_loop().time()
            logger.info("=== NOW RECORDING: Speak for 12 seconds ===")
            logger.info("Recording state - listening for 12 seconds...")
    
    # Track if cloning task already started (prevent multiple triggers)
    cloning_task_started = {"value": False}
    
    @session.on("user_input_transcribed")
    def on_user_speech(text: str):
        if agent.state == "recording":
            # Check if 12 seconds have passed
            if agent.recording_start_time and not cloning_task_started["value"]:
                elapsed = asyncio.get_event_loop().time() - agent.recording_start_time
                logger.info(f"Recording: {elapsed:.1f} seconds elapsed (need 12.0)")
                if elapsed >= 12.0:
                    cloning_task_started["value"] = True  # Mark as started
                    logger.info("=== 12 SECONDS REACHED - Starting voice cloning ===")
                    
                    # Capture audio and clone - use create_task for async work
                    async def clone_voice_task():
                        try:
                            # Step 1: Capture audio
                            logger.info("Recording complete, capturing audio...")
                            await session.say("Proper audio received. Cloning now.", allow_interruptions=False)
                            recording_file = await capture_audio_frames(session, 12.0)
                            
                            if recording_file:
                                # Step 2: Verify audio quality using OpenAI
                                try:
                                    import os
                                    file_size = os.path.getsize(recording_file)
                                    logger.info(f"Audio captured: {recording_file} ({file_size} bytes)")
                                    
                                    # Use OpenAI to analyze audio capture quality
                                    llm_client = openai_sdk.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                                    analysis_prompt = f"""Check if audio recording is proper for voice cloning:
- File size: {file_size} bytes
- Expected: 12 seconds of clear speech
- Response: 'VALID' if proper, 'INVALID' if too short/noisy/empty. One word only."""
                                    
                                    response = llm_client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": analysis_prompt}],
                                        max_tokens=10,
                                        temperature=0.1
                                    )
                                    audio_status = response.choices[0].message.content.strip().upper()
                                    logger.info(f"OpenAI audio check: {audio_status}")
                                    
                                    if "INVALID" in audio_status or file_size < 50000:  # Less than ~50KB is likely too short
                                        logger.warning("Audio quality check: May be insufficient for cloning")
                                        await session.say("Audio quality check: May need more clear speech. Trying anyway.", allow_interruptions=False)
                                    else:
                                        logger.info("Audio quality check: Valid for cloning")
                                except Exception as analysis_err:
                                    logger.warning(f"OpenAI audio analysis failed: {analysis_err}, proceeding anyway")
                                
                                # Step 3: Clone voice with ElevenLabs
                                logger.info(f"Sending audio to ElevenLabs for cloning...")
                                voice_id = elevenlabs.clone_voice(recording_file, name=f"Voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                                
                                if voice_id:
                                    agent.cloned_voice_id = voice_id
                                    tts_adapter.set_voice(voice_id)
                                    agent.state = "conversation"
                                    logger.info(f"✅ Voice cloned! ID: {voice_id}")
                                    logger.info("Now using cloned voice for all future responses")
                                    await session.say("Cloning ready. Now speak and I will repeat it in your cloned voice.", allow_interruptions=False)
                                    
                                    # Play test audio in terminal
                                    try:
                                        play_audio_in_terminal(recording_file)
                                    except Exception as play_err:
                                        logger.warning(f"Terminal audio play failed: {play_err}")
                                else:
                                    logger.error("Voice cloning failed - no voice_id returned")
                                    await session.say("Voice cloning failed. Please try again.", allow_interruptions=False)
                            else:
                                logger.error("Failed to capture audio - recording_file is None")
                                await session.say("Failed to capture audio. Please try again.", allow_interruptions=False)
                        except Exception as e:
                            logger.error(f"Error in clone_voice_task: {e}")
                            import traceback
                            traceback.print_exc()
                            await session.say("Error during cloning. Please try again.", allow_interruptions=False)
                    
                    asyncio.create_task(clone_voice_task())
        
        elif agent.state == "conversation":
            # User spoke - agent repeats via LLM (handled automatically by session)
            pass
    
    # Start session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting
    greeting = "Hi I am cloning agent, please speak something for 10 secs to generate your audio clone."
    await session.say(greeting, allow_interruptions=False)
    agent.state = "greeting"
    
    # Connect
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
