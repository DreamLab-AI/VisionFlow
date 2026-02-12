# AFD: Voice-to-Voice Real-Time Audio Routing

**Status**: Implementation
**Date**: 2026-02-11
**Scope**: Multi-user voice control of agents + spatial voice chat via Vircadia

---

## Problem Statement

Each user must control their agents in real-time via voice, and all users must hear each other through the Vircadia spatial audio system. This requires multiplexing four distinct audio planes through a mix of local Docker services (Kokoro TTS, Turbo Whisper STT) and a WebRTC SFU (LiveKit).

## Decision: Push-to-Talk + LiveKit Sidecar + Turbo Whisper + Opus

### Audio Planes

```
┌─────────────────────────────────────────────────────────────────┐
│  PLANE 1: PRIVATE COMMAND (PTT held)                            │
│  User mic → Opus → /ws/speech/{user} → Turbo Whisper → STT     │
│  → VoiceCommand → MCP agent control                            │
│  ISOLATION: Per-user WebSocket session + AudioRouter scoping    │
├─────────────────────────────────────────────────────────────────┤
│  PLANE 2: PRIVATE RESPONSE                                      │
│  Agent result → Kokoro TTS (per-agent voice preset) → Opus     │
│  → AudioRouter.route_agent_audio() → owner's WS only           │
│  ISOLATION: AudioRouter user-scoped broadcast channels          │
├─────────────────────────────────────────────────────────────────┤
│  PLANE 3: PUBLIC VOICE CHAT (PTT released)                      │
│  User mic → LiveKit SFU room → WebRTC Opus → all users         │
│  SPATIAL: Web Audio HRTF panner driven by Vircadia positions    │
├─────────────────────────────────────────────────────────────────┤
│  PLANE 4: SPATIAL AGENT VOICE                                   │
│  Agent TTS → Opus → LiveKit virtual participant at agent pos    │
│  → all nearby users hear spatially                              │
│  OPTIONAL: Configurable public/private per agent                │
└─────────────────────────────────────────────────────────────────┘
```

### PTT Routing Logic

```
                     ┌──────────────┐
                     │  Space key   │
                     │  (PTT key)   │
                     └──────┬───────┘
                            │
               ┌────────────┴────────────┐
               │                         │
          KEY HELD                  KEY RELEASED
               │                         │
    ┌──────────▼──────────┐   ┌──────────▼──────────┐
    │   Mic → /ws/speech  │   │   Mic → LiveKit SFU │
    │   (Turbo Whisper)   │   │   (spatial chat)     │
    │                     │   │                      │
    │   LiveKit MUTED     │   │   /ws/speech STOPPED │
    └─────────────────────┘   └──────────────────────┘
```

### Infrastructure Stack

```
docker-compose.voice.yml
├── livekit (v1.7)        — WebRTC SFU, spatial voice rooms
│   Port 7880 (WS+HTTP), 7882/udp (RTC)
│   50 max participants per room
│
├── turbo-whisper          — faster-whisper-server with CUDA
│   Port 8100 → 8000 (OpenAI-compatible REST + streaming)
│   Model: Systran/faster-whisper-large-v3
│   beam_size=1 for minimum latency
│
└── kokoro-tts             — Kokoro FastAPI TTS
    Port 8880 (OpenAI-compatible /v1/audio/speech)
    Default format: Opus 48kHz
```

### Audio Format: Opus Throughout

| Segment | Format | Rationale |
|---------|--------|-----------|
| Browser mic capture | WebM/Opus 48kHz | Native browser MediaRecorder |
| Client → Server WS | Opus binary frames | No transcoding needed |
| Turbo Whisper input | Opus (auto-detected) | faster-whisper accepts Opus |
| Kokoro TTS output | Opus 48kHz | Configured via `response_format=opus` |
| Server → Client WS | Opus binary frames | Direct passthrough |
| LiveKit WebRTC | Opus 48kHz (default) | WebRTC standard codec |

### Latency Budget (Target: <500ms for agent command ACK)

```
User speaks ──────────────────────── 0ms
  ├─ MediaRecorder flush ──────────── ~50ms  (reduced from 100ms via timeslice)
  ├─ WebSocket to server ──────────── ~5ms
  ├─ Turbo Whisper streaming STT ──── ~300ms (GPU, beam_size=1, VAD filter)
  │   ↑ Replaces 200ms×30 polling = potential 6s → now ~300ms
  ├─ VoiceCommand parse ───────────── ~1ms
  ├─ MCP agent call ───────────────── ~50ms
  └─ ACK text → client ───────────── ~5ms
                              TOTAL ≈ ~410ms ✓

Agent response (delayed, async):
  ├─ Kokoro TTS synthesis ─────────── ~200ms (streaming first chunk)
  ├─ Opus frames → client WS ─────── ~10ms
  └─ Web Audio decode + play ──────── ~30ms
                              TOTAL ≈ ~240ms after agent completes
```

## New Components

### Backend (Rust)

| File | Purpose |
|------|---------|
| `src/services/audio_router.rs` | User-scoped session manager with per-user broadcast channels |
| `src/types/speech.rs` | Extended with `TurboWhisper` STT provider, user-scoped commands, `AgentSpatialInfo`, `AudioTarget` |
| `src/config/mod.rs` | `VoiceRoutingSettings`, `LiveKitSettings`, `TurboWhisperSettings`, `AgentVoicePreset` |

### Frontend (TypeScript)

| File | Purpose |
|------|---------|
| `client/src/services/PushToTalkService.ts` | Keyboard PTT with push/toggle modes, routes mic between agents and chat |
| `client/src/services/LiveKitVoiceService.ts` | WebRTC room connection, HRTF spatial panning from Vircadia positions |
| `client/src/services/VoiceOrchestrator.ts` | Wires PTT + VoiceWS + LiveKit + AudioInput together |

### Infrastructure

| File | Purpose |
|------|---------|
| `docker-compose.voice.yml` | LiveKit SFU, Turbo Whisper, Kokoro TTS containers |
| `config/livekit.yaml` | LiveKit SFU configuration (rooms, Opus, spatial) |

## Agent Voice Identity

Each agent type gets a distinct Kokoro voice preset so users can distinguish agents by ear:

| Agent Type | Voice ID | Speed | Description |
|------------|----------|-------|-------------|
| researcher | `af_sarah` | 1.0 | Female, measured |
| coder | `am_adam` | 1.1 | Male, slightly fast |
| analyst | `bf_emma` | 1.0 | British female |
| optimizer | `am_michael` | 0.95 | Male, deliberate |
| coordinator | `af_heart` | 1.0 | Default female |

Custom presets are configurable via `voice_routing.agent_voices` in settings.

## AudioRouter: User-Scoped Channel Architecture

```rust
// Each user gets isolated broadcast channels
struct UserVoiceSession {
    user_id: String,
    private_audio_tx: broadcast::Sender<Vec<u8>>,  // TTS audio → only this user
    transcription_tx: broadcast::Sender<String>,    // STT text → only this user
    owned_agents: Vec<String>,                       // Agents this user controls
    ptt_active: bool,                                // Current PTT state
    spatial_position: [f32; 3],                      // Vircadia world position
}

// Agent responses route through ownership:
//   agent.owner_user_id → sessions[owner].private_audio_tx
// NOT through the global broadcast (which was the old broken path)
```

## Integration Points

### Vircadia ↔ LiveKit Position Sync

```
Vircadia World Server (:3020)
  ↓ entity position updates
CollaborativeGraphSync
  ↓ user + agent positions
VoiceOrchestrator
  ├── updateUserPosition()     → LiveKit listener position
  └── updateRemotePosition()   → LiveKit panner node positions
                                 (HRTF spatial audio)
```

### BotsVircadiaBridge ↔ AudioRouter Agent Sync

```
Agent spawned via MCP
  ↓
BotsVircadiaBridge.syncAgentToEntity()  → Vircadia entity at 3D position
AudioRouter.register_agent()             → voice identity + owner mapping
  ↓
Agent completes task → response text
  ↓
AudioRouter.get_agent_voice(agent_id)    → { voice_id, speed, position }
SpeechService.text_to_speech(text, opts) → Kokoro TTS (agent's unique voice)
AudioRouter.route_agent_audio(agent_id)  → owner's private channel
                                           (or LiveKit spatial if public)
```
