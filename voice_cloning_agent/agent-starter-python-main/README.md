# Voice Cloning Agent – Python Backend

LiveKit agent that records a user's voice, clones it via Speechify, and speaks using the cloned voice.

## Prerequisites
- Python 3.10+
- LiveKit Cloud project (or self-hosted)
- Speechify API key

## Setup
1. Create env file from template:
   - Copy `ENV_TEMPLATE.txt` to `.env.local` (or `.env`)
   - Fill the values below
2. Install deps:
```bash
python -m venv venv
venv/Scripts/activate  # Windows
pip install -e .
```

## Run (dev)
```bash
python -m src.agent dev
```

## Env Vars
- `LIVEKIT_URL` (e.g. wss://your.livekit.cloud)
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `SPEECHIFY_API_KEY`
- `OPENAI_API_KEY` (for consent JSON generation)

## What it does
- Joins a LiveKit room, records a short sample
- Creates a Speechify voice clone (with consent JSON)
- Switches TTS to the cloned voice when ready

## Notes
- Audio outputs and recordings are gitignored
- Do not commit real env files; use the template
