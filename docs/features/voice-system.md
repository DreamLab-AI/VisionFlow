# Voice System Integration

**Status: 95% Complete** | **STT/TTS: Fully Functional** | **Limitation: Simulated Swarm Responses**

## Implementation Status

### ✅ **FULLY IMPLEMENTED (95%)**

**Core Voice Infrastructure:**
- ✅ **STT/TTS Services**: Whisper STT + Kokoro TTS with Docker network integration
- ✅ **WebSocket Pipeline**: Full duplex audio streaming via `/ws/speech` endpoint
- ✅ **Voice Command Parsing**: Natural language intent recognition with SwarmIntent enum
- ✅ **Audio Processing**: MediaRecorder → WebSocket → Whisper → TTS → Browser playback
- ✅ **Command Detection**: Automatic voice command recognition and processing

**Advanced Features:**
- ✅ **Voice Preamble System**: Context-aware response formatting for conversational TTS
- ✅ **Session Management**: Per-session voice command tracking and context
- ✅ **Error Handling**: Comprehensive fallback systems for audio processing
- ✅ **Network Integration**: Docker service discovery and IP-based connections

**Command Processing:**
```rust
pub enum SwarmIntent {
    SpawnAgent { agent_type: String, capabilities: Vec<String> },
    QueryStatus { target: Option<String> },
    ExecuteTask { description: String, priority: TaskPriority },
    UpdateGraph { action: GraphAction },
    ListAgents,
    StopAgent { agent_id: String },
    Help,
}
```

### ⚠️ **REMAINING WORK (5%)**

**Integration Gap:**
- 🔄 **Real Swarm Integration**: Currently returns simulated responses instead of executing actual swarm commands
- 🔄 **SupervisorActor Wiring**: Need to connect voice commands to actual swarm operations via SupervisorActor

**Current Limitation**: Voice commands are parsed correctly but return mock responses instead of controlling real agent swarms.

**Required Implementation:**
```rust
// In speech_service.rs, replace simulated response with real integration:
if let Ok(voice_cmd) = VoiceCommand::parse(&transcription_text, session_id) {
    // TODO: Wire to actual SupervisorActor
    let supervisor_addr = SupervisorActor::from_registry(); // Not implemented
    let response = supervisor_addr.send(voice_cmd).await?;

    // Generate TTS from real swarm response
    let tts_text = VoicePreamble::wrap_response(&response.text);
    // Send to Kokoro TTS...
}
```

## Overview

This document describes the sophisticated voice-to-voice interaction system integrated into VisionFlow, enabling real-time speech-to-text (STT) and text-to-speech (TTS) capabilities with advanced swarm command processing.

## Architecture Overview

The voice system consists of three main components:

1. **Client-Side (TypeScript/React)**
   - `AudioInputService`: Manages microphone capture and audio streaming
   - `AudioOutputService`: Handles audio playback with queue management
   - `VoiceWebSocketService`: WebSocket client for bidirectional audio communication
   - Voice UI Components: `VoiceButton` and `VoiceIndicator`

2. **Backend (Rust/Actix)**
   - `/ws/speech` WebSocket endpoint for audio streaming
   - `SpeechService`: Orchestrates TTS and STT operations
   - Support for multiple providers (Kokoro TTS, Whisper STT)
   - Default providers: Kokoro for TTS, Whisper for STT

3. **External Services**
   - Kokoro TTS API (containerized)
   - Whisper STT API (async task-based API at whisper-webui-backend:8000)

## Audio Flow Diagram

```
┌─────────────┐     Audio Stream    ┌──────────────┐     HTTP POST    ┌─────────────┐
│   Browser   │ ──────────────────> │   Backend    │ ──────────────> │   Whisper   │
│             │                      │   (/ws/      │  (multipart)    │   Service   │
│ Microphone  │                      │   speech)    │                 │   (Async)   │
└─────────────┘                      │              │ <───────────────┤             │
       │                             │              │   Task ID       └─────────────┘
       │                             │              │                         │
       │                             │              │ ──────────────>        │
       │                             │              │   Poll Status          │
       │                             │              │ <───────────────────────┘
       v                             │              │   Transcription
┌─────────────┐     Audio Playback   │              │                 ┌─────────────┐
│   Browser   │ <────────────────── │              │ ──────────────> │   Kokoro    │
│             │                      │              │     TTS Request │   Service   │
│   Speaker   │                      │              │                 └─────────────┘
└─────────────┘                      └──────────────┘     Audio Stream
```

## WebSocket Protocol

### Connection
- Endpoint: `ws://[host]/ws/speech`
- Heartbeat: 5-second ping/pong with 10-second timeout

### Message Types

#### Client → Server

1. **TTS Request**
```json
{
  "type": "tts",
  "text": "Hello, world!",
  "voice": "af_heart",     // optional
  "speed": 1.0,            // optional
  "stream": true           // optional
}
```

2. **STT Control**
```json
{
  "type": "stt",
  "action": "start",       // or "stop"
  "language": "en",        // optional
  "model": "whisper-1"     // optional
}
```

3. **Audio Data**
- Binary WebSocket frames containing audio chunks
- Format: `audio/webm;codecs=opus` (preferred)
- Sample rate: 48kHz, mono

#### Server → Client

1. **Connection Established**
```json
{
  "type": "connected",
  "message": "Connected to speech service"
}
```

2. **Transcription Result**
```json
{
  "type": "transcription",
  "data": {
    "text": "Hello, world!",
    "isFinal": true,
    "timestamp": 1234567890123
  }
}
```

3. **Audio Data**
- Binary WebSocket frames containing TTS audio
- Format: MP3 (default) or as configured

4. **Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Client API Usage

### Basic Voice Interaction

```typescript
import { VoiceWebSocketService } from './services/VoiceWebSocketService';

// Get service instance
const voiceService = VoiceWebSocketService.getInstance();

// Connect to voice service
await voiceService.connectToSpeech('http://localhost:3000');

// Send text for TTS
await voiceService.sendTextForTTS({
  text: "Hello, I'm your AI assistant",
  voice: "af_heart",
  speed: 1.0,
  stream: true
});

// Start voice input (STT)
await voiceService.startAudioStreaming({
  language: "en"
});

// Listen for transcriptions
voiceService.on('transcription', (result) => {
  console.log('User said:', result.text);
});

// Stop voice input
voiceService.stopAudioStreaming();
```

### UI Components

```tsx
import { VoiceButton, VoiceIndicator } from './components';

// Voice control button
<VoiceButton
  size="md"
  variant="primary"
  className="my-voice-btn"
/>

// Voice status indicator with transcription display
<VoiceIndicator
  showTranscription={true}
  showStatus={true}
/>
```

## Backend Configuration

### Settings Structure

```rust
// In settings.toml or environment variables

[kokoro]
api_url = "http://172.18.0.9:8880"  # Kokoro container IP on docker_ragflow network
default_voice = "af_heart"  # Options: af_bella, af_sky, af_heart, etc.
default_speed = 1.0
default_format = "mp3"
stream = true

[whisper]
api_url = "http://whisper-webui-backend:8000"  # Docker DNS name
default_model = "base"  # Or "large-v2" for better accuracy
default_language = "en"
timeout = 30
temperature = 0.0
return_timestamps = false
vad_filter = true  # Voice activity detection
word_timestamps = false
```

### Docker Services

The voice services run within the Docker network:

```yaml
# Docker Network Configuration
# Both services must be on the same Docker network (docker_ragflow)

services:
  kokoro:  # Container name may vary (e.g., friendly_dewdney, trusting_hugle)
    image: ghcr.io/remsky/kokoro-fastapi-gpu:latest
    ports:
      - "8880:8880"
    networks:
      - docker_ragflow  # Must be added to this network
    # Current IP: 172.18.0.9 (may change on restart)

  whisper-webui-backend:
    image: jhj0517/whisper-webui-backend:latest
    ports:
      - "8000:8000"
    networks:
      - docker_ragflow
    # Current IP: 172.18.0.5
    environment:
      - MODEL_SIZE=large-v2  # Options: tiny, base, small, medium, large, large-v2
      - DEVICE=cuda  # or cpu
```

## Implementation Status

### ✅ Completed
- TTS Backend with Kokoro integration (IP-based connection)
- STT Backend with async Whisper API integration
- WebSocket endpoint (`/ws/speech`)
- Audio streaming infrastructure
- Client-side audio services
- Voice UI components
- Full speech service architecture with provider switching
- Async task polling for Whisper transcriptions
- Spacebar push-to-talk hotkey support
- Voice status indicator in control panel
- Docker network integration for both services
- End-to-end voice pipeline testing

### 🚧 In Progress
- Full duplex audio communication optimisation
- Voice activity detection (VAD)
- Enhanced error recovery for streaming

### 📋 Planned
- Multiple language support
- Voice command processing
- Audio visualisations
- Additional hotkey configurations
- Noise gate and echo cancellation
- Real-time streaming transcription (when Whisper supports it)

## API Reference

### AudioInputService

```typescript
class AudioInputService {
  // Request microphone access
  requestMicrophoneAccess(constraints?: AudioConstraints): Promise<boolean>

  // Start/stop recording
  startRecording(mimeType?: string): Promise<void>
  stopRecording(): void

  // Audio level monitoring
  getAudioLevel(): number  // 0-1
  getFrequencyData(): Uint8Array

  // Events
  on('audioChunk', (chunk: AudioChunk) => void)
  on('audioLevel', (level: number) => void)
  on('stateChange', (state: AudioInputState) => void)
}
```

### AudioOutputService

```typescript
class AudioOutputService {
  // Queue audio for playback
  queueAudio(audioData: ArrayBuffer, id?: string): Promise<void>

  // Playback control
  stop(): void
  pause(): void
  resume(): void

  // Volume control
  setVolume(volume: number): void  // 0-1
  getVolume(): number

  // Events
  on('audioStarted', (item: AudioQueueItem) => void)
  on('audioEnded', (item: AudioQueueItem) => void)
  on('stateChange', (state: AudioOutputState) => void)
}
```

### VoiceWebSocketService

```typescript
class VoiceWebSocketService extends WebSocketService {
  // Connection
  connectToSpeech(baseUrl: string): Promise<void>

  // TTS
  sendTextForTTS(request: TTSRequest): Promise<void>

  // STT
  startAudioStreaming(options?: { language?: string }): Promise<void>
  stopAudioStreaming(): void

  // Events
  on('voiceConnected', (data: any) => void)
  on('transcription', (result: TranscriptionResult) => void)
  on('audioReceived', (buffer: ArrayBuffer) => void)
}
```

## Whisper API Integration Details

### Async Task-Based Processing

The Whisper integration uses an asynchronous task-based API that follows this workflow:

1. **Audio Submission**: Audio data is sent as multipart/form-data to `/transcription/`
2. **Task Creation**: Whisper returns a task ID immediately (not the transcription)
3. **Status Polling**: Backend polls `/task/{identifier}` endpoint until completion
4. **Result Extraction**: Once status is "completed", transcription text is extracted from result array

### Whisper API Endpoints

```bash
# Submit audio for transcription
POST http://whisper-webui-backend:8000/transcription/
Content-Type: multipart/form-data
- file: audio data (WAV/WebM/etc)
- model_size: "base" or "large-v2"
- lang: language code (optional)
- vad_filter: true/false (voice activity detection)

# Response
{
  "identifier": "uuid-task-id",
  "status": "queued",
  "message": "Transcription task has queued"
}

# Poll task status
GET http://whisper-webui-backend:8000/task/{identifier}

# Response when completed
{
  "identifier": "uuid-task-id",
  "status": "completed",
  "result": [
    {
      "text": "Transcribed text here",
      "start": 0.0,
      "end": 2.0,
      "tokens": [...],
      "temperature": 0.0
    }
  ],
  "duration": 1.5,
  "progress": 1.0
}
```

### Polling Configuration

The backend polls with these parameters:
- **Poll Interval**: 200ms between checks
- **Max Attempts**: 30 (6 seconds total timeout)
- **Status Values**: "queued", "in_progress", "completed", "failed"

### Error Handling

The system handles several error scenarios:
- **Task Timeout**: After 30 attempts, stops polling and logs timeout
- **Task Failure**: If status is "failed", extracts error message
- **Connection Errors**: Logs and retries with exponential backoff
- **Invalid Response**: Falls back gracefully if response format unexpected

## Testing

### Manual Testing

1. **Test TTS with Kokoro**:
   ```bash
   # Direct API test (using current IP)
   curl -X POST "http://172.18.0.9:8880/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"model":"kokoro","input":"Hello world","voice":"af_bella","response_format":"wav"}' \
     --output test.wav
   ```

2. **Test STT with Whisper**:
   ```bash
   # Submit audio for transcription
   curl -X POST "http://172.18.0.5:8000/transcription/" \
     -F "file=@test.wav" \
     -F "model_size=base" \
     -F "lang=en"
   # Returns task ID, then poll for result
   ```

3. **Test Complete Pipeline**:
   ```bash
   # Run the comprehensive test script
   bash //scripts/voice_pipeline_test.sh
   ```

4. **Test Audio Capture**:
   - Press spacebar (push-to-talk) in the UI
   - Check voice indicator in control center
   - Check browser console for audio level logs
   - Verify microphone permission prompt

5. **Test End-to-End**:
   - Open the application at http://localhost:3001
   - Press and hold spacebar to record
   - Speak a phrase
   - Release spacebar
   - Verify transcription appears
   - System responds with TTS audio

### Integration Tests

See `tests/voice_integration_test.rs` for backend tests.

## Troubleshooting

### Common Issues

1. **No Audio Output**
   - Check Kokoro service is running: `docker ps | grep kokoro`
   - Verify Kokoro is on the correct network: `docker network inspect docker_ragflow`
   - If not on network, add it: `docker network connect docker_ragflow <container_name>`
   - Update IP in settings.yaml after network changes
   - Verify audio format compatibility
   - Check browser audio permissions

2. **Microphone Not Working**
   - Ensure HTTPS or localhost (required for getUserMedia)
   - Check browser microphone permissions
   - Verify AudioContext is not suspended

3. **WebSocket Connection Failed**
   - Check `/ws/speech` endpoint is accessible
   - Verify CORS settings
   - Check for proxy/firewall issues

4. **Transcription Not Working**
   - Verify Whisper container is running: `docker ps | grep whisper`
   - Check if both services are on same network
   - Try with different model_size (base vs large-v2)
   - Check audio format compatibility
   - Verify audio data is being sent

5. **Network Connection Issues**
   - Both Kokoro and Whisper must be on `docker_ragflow` network
   - To add container to network: `docker network connect docker_ragflow <container_name>`
   - Find container IPs: `docker network inspect docker_ragflow | grep -A 5 "<container_name>"`
   - Update settings.yaml with correct IPs after network changes

### Debug Logging

Enable debug logs:
```bash
# Backend
RUST_LOG=debug cargo run

# Frontend
localStorage.setItem('debug', 'voice:*')
```