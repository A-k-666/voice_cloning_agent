"""
Audio recording utilities for voice cloning agent.
Handles robust audio buffering, silence detection, and WAV file saving.
"""
import asyncio
import wave
import time
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("recording_utils")


def rms_from_pcm16_bytes(b: bytes) -> float:
    """Calculate RMS energy from PCM int16 bytes"""
    if not b:
        return 0.0
    arr = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt((arr**2).mean()))


def write_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int = 16000, nchannels: int = 1):
    """Write PCM int16 bytes to WAV file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    logger.info("WAV written: %s (%d bytes, %.2f seconds)", path, len(pcm_bytes), len(pcm_bytes) / (sample_rate * 2))


def concat_pcm_chunks(chunks: list[bytes]) -> bytes:
    """Concatenate list of PCM byte chunks into single bytes"""
    return b"".join(chunks)


def resample_audio(audio_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample audio from src_rate to dst_rate using scipy"""
    try:
        from scipy import signal
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        num_samples = int(len(arr) * dst_rate / src_rate)
        resampled = signal.resample(arr, num_samples).astype(np.int16)
        logger.info("Resampled: %dHz -> %dHz (%d -> %d samples)", src_rate, dst_rate, len(arr), len(resampled))
        return resampled.tobytes()
    except ImportError:
        logger.warning("scipy not available, using original sample rate")
        return audio_bytes
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        return audio_bytes






