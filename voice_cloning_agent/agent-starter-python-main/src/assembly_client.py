"""
AssemblyAI Client for Speech-to-Text
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv(".env.local")

logger = logging.getLogger("assembly-client")

API_KEY = os.getenv("ASSEMBLYAI_API_KEY")


class AssemblyClient:
    """Client for AssemblyAI STT"""
    
    def __init__(self):
        if not API_KEY:
            logger.warning("ASSEMBLYAI_API_KEY not found - STT will use LiveKit plugin default")
        self.api_key = API_KEY
    
    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file (not used directly, LiveKit plugin handles it)"""
        # This is for future use if needed
        # AssemblyAI is handled by LiveKit plugin in agent.py
        logger.debug(f"Transcription request for: {audio_path}")
        return None





