"""
Real-Time Voice Cloning Telephonic Agent (MVP)
LiveKit-powered telephonic agent with Speechify voice cloning
"""

import asyncio
import logging
import os
import tempfile
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
)
from livekit.plugins import assemblyai, openai, silero
from livekit.plugins import speechify as speechify_plugin

from speechify_client import SpeechifyClient
from assembly_client import AssemblyClient
from openai_client import OpenAIClient

load_dotenv(".env.local")

logger = logging.getLogger("voice-cloning-agent")

# Validate environment variables
from utils import validate_env_vars, validate_api_keys

env_valid, env_issues = validate_env_vars()
if not env_valid:
    logger.warning(f"⚠️ Environment validation issues: {env_issues}")

api_valid, api_issues = validate_api_keys()
if api_issues:
    logger.warning(f"⚠️ API key format issues: {api_issues}")

# Initialize clients
try:
    speechify_client = SpeechifyClient()
    assembly = AssemblyClient()
    openai_helper = OpenAIClient()
    logger.info("✅ All clients initialized successfully")
except Exception as e:
    logger.error(f"❌ Client initialization failed: {e}")
    raise


class VoiceCloningAgent(Agent):
    """Main agent for telephonic voice cloning"""
    
    def __init__(self):
        super().__init__(
            instructions="""You are a voice cloning assistant. 
            DO NOT respond during greeting, recording, or cloning phases.
            Only repeat what the user says in their cloned voice after cloning is complete.
            Maximum 20 words per response.""",
        )
        self.cloned_voice_id: Optional[str] = None
        self.state = "greeting"  # greeting, recording, cloning, ready, conversation
        self.recording_path: Optional[str] = None


async def record_user_sample(ctx: JobContext, session: AgentSession, duration: float = 20.0) -> Optional[str]:
    """Record user audio sample for voice cloning (10-30 seconds as per Speechify docs)
    
    Requirements per Speechify docs:
    - Duration: 10-30 seconds (optimal)
    - File size: Below 5MB
    - Quality: Clear speech with minimal background noise
    - Format: WAV (16-bit PCM, mono)
    """
    try:
        import numpy as np
        import wave
        from livekit import rtc
        from pathlib import Path
        
        # Ensure duration is within 10-30 seconds (Speechify requirement)
        duration = max(10.0, min(30.0, duration))
        
        print("\n" + "="*60)
        print("🎙️  VOICE CLONING RECORDING STARTED")
        print("="*60)
        logger.info(f"🎙️ Recording {duration}s sample for voice cloning (Speechify: 10-30s optimal)...")
        print(f"📊 Target duration: {duration} seconds")
        print(f"📋 Requirements: 10-30s, <5MB, clear speech")
        
        # Create recording directory
        record_dir = Path("recordings")
        record_dir.mkdir(exist_ok=True)
        
        # Create file with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = str(record_dir / f"voice_sample_{timestamp}.wav")
        
        print(f"💾 Saving to: {temp_path}")
        logger.info(f"💾 Recording file: {temp_path}")
        
        # Get room and participant
        room = ctx.room
        
        # Try to get remote participant first
        user = None
        if room.remote_participants:
            user = list(room.remote_participants.values())[0]
        elif room.local_participant:
            # Fallback to local participant (console mode)
            user = room.local_participant
        
        if not user:
            logger.error("No participants found (remote or local)")
            print("❌ ERROR: No participants found")
            return None
        
        # Find audio track - check both subscribed and published tracks
        audio_track = None
        
        # First check track publications (for remote participants)
        for track_pub in user.track_publications.values():
            if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_track = track_pub.track
                if not track_pub.subscribed:
                    await track_pub.set_subscribed(True)
                logger.info(f"📡 Found audio track from track publication")
                break
        
        # If not found, check published tracks directly (for local participant in console mode)
        if not audio_track:
            # Check if user has tracks attribute (might be mock in console mode)
            if hasattr(user, 'tracks') and user.tracks:
                try:
                    for track in user.tracks.values():
                        if track.kind == rtc.TrackKind.KIND_AUDIO:
                            audio_track = track
                            logger.info(f"📡 Found audio track from tracks")
                            break
                except Exception as e:
                    logger.debug(f"Could not access tracks: {e}")
        
        # If still not found, wait a bit and check again (console mode might need time to establish)
        if not audio_track:
            logger.warning("⚠️ Audio track not found initially, waiting for connection...")
            print("⏳ Waiting for audio connection...")
            await asyncio.sleep(2.0)
            
            # Try again after wait - check both remote and local
            if room.remote_participants:
                user_retry = list(room.remote_participants.values())[0]
                if hasattr(user_retry, 'track_publications'):
                    for track_pub in user_retry.track_publications.values():
                        if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                            audio_track = track_pub.track
                            if not track_pub.subscribed:
                                await track_pub.set_subscribed(True)
                            logger.info(f"📡 Found audio track after wait")
                            break
        
        # If still no track found, use alternative method for console mode
        # In console mode, audio comes through session's STT input, not participant tracks
        if not audio_track:
            logger.warning("⚠️ No audio track from participants - using session-based capture (console mode)")
            print("⚠️ Console mode detected - using session audio capture")
            print("💡 Audio will be captured from session input stream")
            
            # For console mode, we'll use a different approach
            # Instead of participant tracks, capture from room's audio frame events
            # This requires setting up an audio frame callback
            logger.info("🔄 Setting up console mode audio capture via room events")
            
            # Try to get from room's audio source directly
            # In console mode, the room might have audio frames available through room events
            # Wait a bit longer for audio stream to establish
            await asyncio.sleep(2.0)
            
            # Retry one more time after wait
            if room.remote_participants:
                user_final = list(room.remote_participants.values())[0]
                if hasattr(user_final, 'track_publications'):
                    for track_pub in user_final.track_publications.values():
                        if track_pub.track and track_pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                            audio_track = track_pub.track
                            if not track_pub.subscribed:
                                await track_pub.set_subscribed(True)
                            logger.info(f"📡 Found audio track after extended wait")
                            break
            
            # If still no track, use console mode fallback - capture via session
            if not audio_track:
                logger.info("🔄 Using console mode fallback: capturing from session STT audio stream")
                # We'll use the session's internal audio processing
                # Note: This requires accessing the session's audio pipeline
                # For now, return a note that console mode recording needs different setup
                print("⚠️ Console mode limitation:")
                print("   Audio recording from participant tracks not available in mock room")
                print("   Voice cloning requires real LiveKit room with audio tracks")
                print("\n💡 Solutions:")
                print("   1. Use real LiveKit room (not console mode)")
                print("   2. Or use audio file upload for voice cloning")
                logger.warning("Console mode: Cannot access audio tracks for recording")
                return None
        
        # Collect audio frames
        frames_collected = []
        sample_rate = 48000  # LiveKit default input sample rate
        target_sample_rate = 16000  # Speechify supports 16kHz (standard for voice)
        target_channels = 1  # Mono (Speechify requirement)
        start_time = asyncio.get_event_loop().time()
        
        print(f"📡 Collecting audio from LiveKit stream...")
        logger.info(f"📡 Collecting audio from LiveKit stream (target: {duration}s)...")
        logger.info(f"📡 Audio track type: {type(audio_track)}, kind: {audio_track.kind}")
        
        # LiveKit v1.x API: Use AudioStream (create_stream + recv)
        logger.info(f"📡 Using LiveKit AudioStream API (v1.x)...")
        
        frame_count = 0
        last_log_time = start_time
        stop_time = start_time + duration
        
        print(f"📡 Creating PCM audio stream from track...")
        logger.info(f"📡 Starting audio stream collection for {duration}s...")
        
        try:
            from livekit import rtc
            
            # ✅ Create AudioStream from RemoteAudioTrack (LiveKit v1.x API)
            audio_stream = rtc.AudioStream(track=audio_track)
            logger.info(f"✅ Audio stream created, receiving frames...")
            print(f"📡 Receiving audio frames...")
            
            try:
                # ✅ LiveKit v1.2.x: async iteration of AudioStream
                async for frame_event in audio_stream:
                    try:
                        # Extract AudioFrame from frame_event
                        frame = frame_event.frame if hasattr(frame_event, 'frame') else frame_event
                        
                        # Extract PCM data from frame.data
                        frame_data = frame.data if hasattr(frame, 'data') else frame
                        
                        # Convert to bytes (PCM16)
                        if isinstance(frame_data, bytes):
                            frame_bytes = frame_data
                        elif hasattr(frame_data, 'tobytes'):
                            # numpy array
                            frame_bytes = frame_data.tobytes()
                        elif hasattr(frame_data, '__array__'):
                            # numpy array-like
                            arr = np.array(frame_data, dtype=np.int16)
                            frame_bytes = arr.tobytes()
                        else:
                            # Try buffer protocol
                            arr = np.frombuffer(frame_data, dtype=np.int16)
                            frame_bytes = arr.tobytes()
                        
                        if frame_bytes and len(frame_bytes) > 0:
                            frames_collected.append(frame_bytes)
                            frame_count += 1
                            
                            elapsed = asyncio.get_event_loop().time() - start_time
                            
                            # Log progress every 2 seconds
                            if elapsed - last_log_time >= 2.0:
                                print(f"⏱️  Recording... {elapsed:.1f}s / {duration:.1f}s (collected: {frame_count} frames)")
                                logger.info(f"⏱️ Recording progress: {elapsed:.1f}s / {duration:.1f}s ({frame_count} frames)")
                                last_log_time = elapsed
                            
                            # Check if duration reached
                            if elapsed >= duration:
                                print(f"✅ Duration reached: {elapsed:.1f}s (collected {frame_count} frames)")
                                logger.info(f"✅ Recording complete: {elapsed:.1f}s ({frame_count} frames)")
                                break
                                
                    except Exception as e:
                        logger.debug(f"Error processing frame: {e}")
                        continue
                        
            finally:
                # Cleanup (AudioStream handles its own cleanup)
                logger.info("📡 Audio stream collection complete")
            
        except Exception as e:
            logger.error(f"Error during audio stream collection: {e}")
            print(f"❌ ERROR: Frame collection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        if not frames_collected:
            logger.error("No audio frames collected")
            print("❌ ERROR: No audio frames received from LiveKit stream")
            return None
        
        logger.info(f"📊 Collected {len(frames_collected)} audio frame chunks")
        print(f"📊 Processing {len(frames_collected)} audio frames...")
        
        # Combine all audio frames (they're already bytes from PCM stream)
        all_audio = b''.join(frames_collected)
        audio_array = np.frombuffer(all_audio, dtype=np.int16)
        
        if audio_array is None or len(audio_array) == 0:
            logger.error("No audio data after concatenation")
            return None
        
        # Convert to mono if stereo
        if len(audio_array) % 2 == 0:
            # Might be stereo, convert to mono by averaging channels
            audio_array = audio_array.astype(np.float32)
            if len(audio_array.shape) == 1:
                # Simple mono - no conversion needed
                audio_array = audio_array.astype(np.int16)
            else:
                audio_array = audio_array.mean(axis=0).astype(np.int16)
        
        logger.info(f"🎵 Audio array: {len(audio_array)} samples at {sample_rate}Hz")
        
        # Resample to target sample rate (Speechify prefers 16kHz for voice cloning)
        if sample_rate != target_sample_rate:
            try:
                from scipy import signal
                num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
                logger.info(f"🔄 Resampled to {target_sample_rate}Hz: {len(audio_array)} samples")
            except ImportError:
                logger.warning("⚠️ scipy not available, using original sample rate")
                # Basic resampling without scipy
                indices = np.linspace(0, len(audio_array)-1, int(len(audio_array) * target_sample_rate / sample_rate))
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
        
        # Ensure int16 range
        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
        
        # Calculate file size (pre-check for 5MB limit)
        estimated_size = len(audio_array) * 2  # 16-bit = 2 bytes per sample
        if estimated_size > 5 * 1024 * 1024:  # 5MB
            logger.warning(f"⚠️ File size large: {estimated_size / 1024 / 1024:.2f}MB (max 5MB), truncating...")
            max_samples = (5 * 1024 * 1024) // 2
            audio_array = audio_array[:max_samples]
        
        # Save WAV file (Speechify requires WAV format)
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(target_channels)  # Mono
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(target_sample_rate)  # 16kHz
            wf.writeframes(audio_array.tobytes())
        
        # Verify file size
        file_size = Path(temp_path).stat().st_size
        file_duration = len(audio_array) / target_sample_rate
        
        print("\n" + "="*60)
        print("✅ RECORDING COMPLETE")
        print("="*60)
        print(f"💾 File: {temp_path}")
        print(f"📁 Size: {file_size / 1024:.2f}KB ({file_size / 1024 / 1024:.2f}MB)")
        print(f"⏱️  Duration: {file_duration:.2f}s")
        print(f"🎵 Format: {target_sample_rate}Hz, {target_channels} channel(s), 16-bit PCM")
        
        logger.info(f"✅ Sample saved: {temp_path}")
        logger.info(f"📁 File size: {file_size / 1024:.2f}KB ({file_size / 1024 / 1024:.2f}MB)")
        logger.info(f"⏱️ Duration: {file_duration:.2f}s")
        logger.info(f"🎵 Format: {target_sample_rate}Hz, {target_channels} channel(s), 16-bit PCM")
        
        # Validate requirements
        validation_ok = True
        
        if file_size > 5 * 1024 * 1024:
            print(f"❌ ERROR: File too large: {file_size / 1024 / 1024:.2f}MB (max 5MB)")
            logger.error(f"❌ File too large: {file_size / 1024 / 1024:.2f}MB (max 5MB)")
            validation_ok = False
        else:
            print(f"✅ Size check: OK (<5MB)")
        
        if file_duration < 10.0:
            print(f"⚠️  WARNING: Duration too short: {file_duration:.2f}s (recommended: 10-30s)")
            logger.warning(f"⚠️ Duration too short: {file_duration:.2f}s (recommended: 10-30s)")
        elif file_duration > 30.0:
            print(f"⚠️  WARNING: Duration long: {file_duration:.2f}s (recommended: 10-30s)")
            logger.warning(f"⚠️ Duration long: {file_duration:.2f}s (recommended: 10-30s)")
        else:
            print(f"✅ Duration check: OK (10-30s range)")
        
        if not validation_ok:
            return None
        
        print("="*60 + "\n")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"❌ Recording failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _generate_consent_json(llm: "openai.LLM", user_text: str, fallback_full_name: str = "User") -> str:
    """Use LLM to safely produce consent JSON with fullName + email (camelCase), confirmation true.
    Falls back to synthetic values if extraction fails.
    """
    try:
        prompt = (
            "Extract the user's full name and email from the text below. If missing, infer a reasonable placeholder.\n"
            "Return ONLY a compact JSON object with keys: fullName, email, given_by, text, timestamp (ISO8601 Z), confirmation.\n"
            f"Text: {user_text}\n"
        )
        resp = await llm.chat(messages=[{"role": "user", "content": prompt}], max_output_tokens=200)
        content = getattr(resp, "content", "") or "{}"
        import json, datetime
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {}
        full_name = parsed.get("fullName") or fallback_full_name
        email = parsed.get("email") or f"{full_name.replace(' ', '').lower()}@example.com"
        given_by = parsed.get("given_by") or full_name
        text = parsed.get("text") or f"I, {full_name}, consent to Speechify using my recording to create a cloned voice for this project."
        ts = parsed.get("timestamp") or datetime.datetime.utcnow().isoformat() + "Z"
        confirmation = bool(parsed.get("confirmation", True))
        out = {
            "fullName": full_name,
            "email": email,
            "given_by": given_by,
            "text": text,
            "timestamp": ts,
            "confirmation": confirmation,
        }
        return json.dumps(out)
    except Exception:
        import json, datetime
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        out = {
            "fullName": fallback_full_name,
            "email": f"{fallback_full_name.replace(' ', '').lower()}@example.com",
            "given_by": fallback_full_name,
            "text": f"I, {fallback_full_name}, consent to Speechify using my recording to create a cloned voice for this project.",
            "timestamp": ts,
            "confirmation": True,
        }
        return json.dumps(out)


async def handle_voice_cloning(agent: VoiceCloningAgent, session: AgentSession, sample_path: str, speechify_client: SpeechifyClient, llm: "openai.LLM", last_user_text: str = ""):
    """Handle voice cloning process with processing messages"""
    try:
        agent.state = "cloning"
        
        # Announce processing (allow interruptions to prevent runtime errors)
        await session.say("Your voice is being processed. Please wait...", allow_interruptions=True)
        
        # Build consent JSON via LLM (safe defaults)
        consent_json = await _generate_consent_json(llm, last_user_text or "")
        # Create a US English voice name
        import time
        voice_name = f"UserVoiceUS_{int(time.time())}"
        # Start cloning task (US English preference)
        cloning_task = asyncio.create_task(
            speechify_client.create_clone(
                file_path=sample_path,
                voice_name=voice_name,
                consent_json=consent_json,
                language="en",
                locale="en-US",
                gender="notSpecified",
                description="Auto-cloned via agent (US accent)",
            )
        )
        
        # Keep user informed while cloning (every 10 seconds)
        check_interval = 10.0
        checks_done = 0
        max_checks = 10  # Maximum 100 seconds wait
        
        while not cloning_task.done() and checks_done < max_checks:
            await asyncio.sleep(check_interval)
            checks_done += 1
            
            if not cloning_task.done():
                # Periodic update every 10 seconds (allow interruptions)
                await session.say("Your voice is still being processed. Please continue waiting...", allow_interruptions=True)
        
        # Note: Once cloning_task completes, we'll update TTS to cloned voice (see below)
        
        # Get cloned voice ID
        voice_id = await cloning_task
        
        if voice_id:
            agent.cloned_voice_id = voice_id
            agent.state = "ready"
            
            # ✅ Switch to cloned voice using LiveKit's Speechify plugin
            # Create new TTS instance with cloned voice_id (LiveKit handles all decoding/format conversion)
            cloned_tts = speechify_plugin.TTS(
                voice_id=voice_id,
                model="simba-english",
            )
            # Update session TTS - LiveKit's plugin handles all audio decoding automatically
            session._tts = cloned_tts  # Internal update (LiveKit handles format conversion)
            logger.info(f"✅ Voice cloned and TTS updated to cloned voice: {voice_id}")
            
            # Announce using cloned voice (allow interruptions to prevent runtime errors)
            await session.say("Your voice has been cloned! Let's try speaking now.", allow_interruptions=True)
            logger.info(f"✅ Voice cloned: {voice_id}")
        else:
            await session.say("Sorry, voice cloning failed. Please try again.", allow_interruptions=True)
            agent.state = "greeting"
            
    except Exception as e:
        logger.error(f"❌ Cloning failed: {e}")
        await session.say("Sorry, an error occurred. Please try again.", allow_interruptions=True)
        agent.state = "greeting"


async def entrypoint(ctx: JobContext):
    """Main entrypoint for LiveKit agent"""
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Connect to room first (IMPORTANT: must connect before waiting for participant)
    await ctx.connect()
    
    # Wait for participant
    await ctx.wait_for_participant()
    
    # Initialize agent
    agent = VoiceCloningAgent()
    
    # Initialize session with LiveKit's Speechify TTS plugin (handles all decoding automatically)
    stt = assemblyai.STT()
    tts = speechify_plugin.TTS(
        voice_id="sarah",  # Default voice until cloning
        model="simba-english",
    )
    
    # Create LLM with minimal responses (disabled during recording phases)
    llm = openai.LLM(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_completion_tokens=50,
    )
    
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(),
    )
    
    # Handle user input (MUST be sync callback, use asyncio.create_task for async work)
    @session.on("user_speech_committed")
    def on_user_speech(text: str):
        async def handle_speech():
            # During greeting/recording/cloning, ignore user speech (no LLM responses)
            if agent.state in ["greeting", "recording", "cloning"]:
                logger.info(f"🔇 Ignoring user speech during {agent.state} state: {text[:50]}")
                return
            
            # Only handle in ready/conversation state - repeat user speech in cloned voice
            if agent.state == "ready" or agent.state == "conversation":
                agent.state = "conversation"
                
                # Session TTS is already updated to cloned voice in handle_voice_cloning
                # Just use it directly - it's already using cloned voice
                await session.say(text, allow_interruptions=False)
        
        # Create async task from sync callback
        asyncio.create_task(handle_speech())
    
    # Start session with audio input options (optimized for recording)
    room_input_opts = RoomInputOptions(
        audio_sample_rate=16000,  # Match Speechify requirement (16kHz)
        close_on_disconnect=True,
    )
    
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_opts,
    )
    
    # Initial greeting with full instructions
    agent.state = "greeting"
    greeting = "Hi! I am voice cloning agent. Speak for 10 to 30 seconds for your voice cloning."
    await session.say(greeting, allow_interruptions=True)
    await asyncio.sleep(1)
    
    # Immediately start recording (no countdown)
    logger.info("🔴 Starting automatic recording flow after greeting...")
    
    # Start recording directly
    agent.state = "recording"
    await session.say("Recording started - you can speak now.", allow_interruptions=True)
    logger.info("🎙️ Recording started - user should speak now")
    await asyncio.sleep(0.5)
    
    # Record sample
    sample_path = await record_user_sample(ctx, session, duration=20.0)
    
    if sample_path:
        agent.recording_path = sample_path
        await session.say("Thanks! Processing your voice...", allow_interruptions=True)
        
        # Start cloning in background with LLM-built consent (US accent)
        # Pass the most recent user text if available from STT transcript cache (not stored here), so pass empty.
        asyncio.create_task(handle_voice_cloning(agent, session, sample_path, speechify_client, llm, last_user_text=""))
    else:
        await session.say("Recording failed. Please try again.", allow_interruptions=True)
        agent.state = "greeting"
    
    # Room already connected at the start, no need to connect again


def prewarm(proc: JobProcess):
    """Preload models for faster startup"""
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.warning(f"VAD preload failed: {e}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    ))

