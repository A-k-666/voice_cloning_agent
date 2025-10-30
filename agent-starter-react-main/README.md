# Voice Cloning Agent – React Frontend

Minimal Next.js UI to join a LiveKit room and talk to the Python agent.

## Prerequisites
- Node.js 18+
- pnpm (recommended) or npm/yarn
- LiveKit Cloud project (or self-hosted)

## Setup
1. Create env file from template:
   - Copy `ENV_TEMPLATE.txt` to `.env.local`
   - Fill values for `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`

2. Install deps:
```bash
pnpm install
```

## Run (dev)
```bash
pnpm dev
```
Open http://localhost:3000

## What it does
- Calls `/api/connection-details` to mint a LiveKit token
- Connects to room and streams audio to the agent

## Commands
- Dev: `pnpm dev`
- Lint: `pnpm lint`
- Build: `pnpm build`
- Start: `pnpm start`

## Env Vars
- `LIVEKIT_URL` (e.g. wss://your.livekit.cloud)
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

## Notes
- No secrets are committed. Use `.env.local` locally.
- This app has no server rendering requirements beyond the API route.
