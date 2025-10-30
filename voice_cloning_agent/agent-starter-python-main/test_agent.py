"""
Test script to verify all components work before running full agent
"""

import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv(".env.local")

print("🧪 Testing Voice Cloning Agent Components...\n")

# Test 1: Environment variables
print("1️⃣  Checking environment variables...")
required_vars = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ASSEMBLYAI_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
all_present = True
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"   ✅ {var}: {'*' * 10}")
    else:
        print(f"   ❌ {var}: MISSING")
        all_present = False

if not all_present:
    print("\n❌ Missing environment variables!")
    sys.exit(1)

print("\n✅ All environment variables present\n")

# Test 2: Import services
print("2️⃣  Testing imports...")
try:
    from services import ElevenLabsService
    print("   ✅ ElevenLabsService imported")
except Exception as e:
    print(f"   ❌ ElevenLabsService import failed: {e}")
    sys.exit(1)

try:
    from elevenlabs_tts import ElevenLabsTTS
    print("   ✅ ElevenLabsTTS imported")
except Exception as e:
    print(f"   ❌ ElevenLabsTTS import failed: {e}")
    sys.exit(1)

try:
    from livekit.plugins import openai, assemblyai, silero
    print("   ✅ LiveKit plugins imported")
except Exception as e:
    print(f"   ❌ LiveKit plugins import failed: {e}")
    sys.exit(1)

# Test 3: Initialize services
print("\n3️⃣  Testing service initialization...")
try:
    elevenlabs = ElevenLabsService()
    print("   ✅ ElevenLabsService initialized")
except Exception as e:
    print(f"   ❌ ElevenLabsService init failed: {e}")
    sys.exit(1)

try:
    tts_adapter = ElevenLabsTTS(voice_id="21m00Tcm4TlvDq8ikWAM")
    print("   ✅ ElevenLabsTTS initialized")
except Exception as e:
    print(f"   ❌ ElevenLabsTTS init failed: {e}")
    sys.exit(1)

try:
    stt = assemblyai.STT()
    print("   ✅ AssemblyAI STT initialized")
except Exception as e:
    print(f"   ❌ AssemblyAI STT init failed: {e}")
    sys.exit(1)

try:
    llm = openai.LLM(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_completion_tokens=100,
    )
    print("   ✅ OpenAI LLM initialized")
except Exception as e:
    print(f"   ❌ OpenAI LLM init failed: {e}")
    sys.exit(1)

try:
    vad = silero.VAD.load()
    print("   ✅ Silero VAD initialized")
except Exception as e:
    print(f"   ❌ Silero VAD init failed: {e}")
    sys.exit(1)

print("\n✅ All tests passed! Agent should work now.")
print("\n🚀 Run: python -m src.agent console")






