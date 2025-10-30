"""
Utility functions for validation and checks
"""

import os
import logging
from typing import Tuple, List

logger = logging.getLogger("voice-cloning-agent")


def validate_env_vars() -> Tuple[bool, List[str]]:
    """Validate all required environment variables"""
    required_vars = [
        "SPEECHIFY_API_KEY",
        "OPENAI_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]
    
    missing = []
    invalid = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        elif value == f"your_{var.lower()}" or value.startswith("your_"):
            invalid.append(var)
    
    if missing:
        logger.error(f"❌ Missing env vars: {', '.join(missing)}")
    
    if invalid:
        logger.warning(f"⚠️ Invalid/placeholder env vars: {', '.join(invalid)}")
    
    is_valid = len(missing) == 0 and len(invalid) == 0
    
    return is_valid, missing + invalid


def validate_api_keys() -> Tuple[bool, List[str]]:
    """Quick validation of API key formats"""
    issues = []
    
    # Check OpenAI key format
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and not (openai_key.startswith("sk-") and len(openai_key) > 20):
        issues.append("OPENAI_API_KEY format looks invalid")
    
    # Check Speechify key
    speechify_key = os.getenv("SPEECHIFY_API_KEY", "")
    if speechify_key and len(speechify_key) < 10:
        issues.append("SPEECHIFY_API_KEY too short")
    
    # Check AssemblyAI key
    assembly_key = os.getenv("ASSEMBLYAI_API_KEY", "")
    if assembly_key and len(assembly_key) < 10:
        issues.append("ASSEMBLYAI_API_KEY too short")
    
    return len(issues) == 0, issues




