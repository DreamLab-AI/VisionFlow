# Voice System WebRTC Migration Plan

## Current Architecture Analysis

### VoiceWebSocketService (Current Implementation)
**File:** `ext/client/src/services/VoiceWebSocketService.ts`

**Current Pattern:**
- Uses WebSocket for both signaling AND audio transmission
- Sends audio chunks via WebSocket binary frames
- Receives TTS audio via WebSocket binary frames
- Handles STT transcription via WebSocket messages

**Issues:**
1. **High Latency**: WebSocket uses TCP, causing ~50% higher latency with packet loss
2. **Not Optimized**: WebSocket isn't designed for real-time audio streaming
3. **Choppy Playback**: UDP-based WebRTC would provide smoother audio
4. **No P2P**: All audio goes through backend server, increasing load

**Current Usage (5 files):**
- `src/components/VoiceButton.tsx`
- `src/components/VoiceIndicator.tsx`
- `src/components/VoiceStatusIndicator.tsx`
- `src/hooks/useVoiceInteraction.ts`
- `src/components/AuthGatedVoiceButton.tsx`

---

## WebRTC Research Findings (2024)

### Why WebRTC for Audio?

Based on comprehensive research:

1. **Low Latency**: WebRTC uses UDP, reducing latency by ~50% compared to WebSocket
2. **Optimized for Audio**: Built-in audio codecs (Opus, G.711) with automatic bitrate adjustment
3. **Jitter Buffer**: Handles network jitter and packet loss gracefully
4. **End-to-End Encryption**: Native DTLS/SRTP encryption
5. **Peer-to-Peer**: Direct audio streams (when possible) reduce server load

### Hybrid Architecture (Recommended)

**WebSocket Role:** Signaling only
- Exchange SDP offers/answers
- ICE candidate exchange
- Connection negotiation
- Metadata and control messages

**WebRTC Role:** Audio transmission
- Send microphone audio via RTC data channels or media tracks
- Receive TTS audio via RTC media tracks
- Low-latency, optimized transmission

---

## Migration Design

### Phase 1: Create WebRTC Service Layer

**New Files:**
```
src/services/voice/
├── VoiceSignalingService.ts      # WebSocket signaling (SDP/ICE)
├── VoiceRTCService.ts             # WebRTC audio transmission
├── VoiceConnectionManager.ts      # Coordinates signaling + RTC
└── types/voiceTypes.ts            # Shared types
```

### Phase 2: Update Backend (Rust)

**Backend Changes Required:**
```rust
// Add WebRTC signaling endpoint
POST /api/voice/offer       # Accept SDP offer
POST /api/voice/answer      # Return SDP answer
POST /api/voice/ice         # Exchange ICE candidates

// Existing WebSocket remains for signaling
WS /ws/speech               # Signaling messages only
```

**Suggested Backend Library:** `webrtc-rs` or `mediasoup` (Node.js bridge)

### Phase 3: Client Implementation

#### VoiceSignalingService.ts (WebSocket)
```typescript
export class VoiceSignalingService {
  private ws: WebSocket;

  // Send SDP offer/answer
  async sendOffer(offer: RTCSessionDescriptionInit): Promise<void>
  async sendAnswer(answer: RTCSessionDescriptionInit): Promise<void>

  // Send ICE candidates
  async sendIceCandidate(candidate: RTCIceCandidate): Promise<void>

  // Receive signaling messages
  onSignalingMessage(callback: (msg: SignalingMessage) => void): void
}
```

#### VoiceRTCService.ts (WebRTC)
```typescript
export class VoiceRTCService {
  private peerConnection: RTCPeerConnection;
  private audioStream: MediaStream;

  // Initialize RTC connection
  async createConnection(): Promise<void>

  // Audio transmission
  async startAudioStream(stream: MediaStream): Promise<void>
  async stopAudioStream(): Promise<void>

  // Receive TTS audio
  onRemoteAudio(callback: (stream: MediaStream) => void): void

  // Handle ICE candidates
  onIceCandidate(callback: (candidate: RTCIceCandidate) => void): void
}
```

#### VoiceConnectionManager.ts (Coordinator)
```typescript
export class VoiceConnectionManager {
  private signaling: VoiceSignalingService;
  private rtc: VoiceRTCService;

  async connect(baseUrl: string): Promise<void> {
    // 1. Connect WebSocket for signaling
    await this.signaling.connect(baseUrl);

    // 2. Create RTC peer connection
    await this.rtc.createConnection();

    // 3. Set up signaling handlers
    this.setupSignalingHandlers();

    // 4. Create and send SDP offer
    await this.createAndSendOffer();
  }

  private setupSignalingHandlers(): void {
    // Wire signaling to RTC
    this.signaling.onSignalingMessage(async (msg) => {
      if (msg.type === 'answer') {
        await this.rtc.setRemoteDescription(msg.answer);
      } else if (msg.type === 'ice') {
        await this.rtc.addIceCandidate(msg.candidate);
      }
    });

    // Wire RTC to signaling
    this.rtc.onIceCandidate((candidate) => {
      this.signaling.sendIceCandidate(candidate);
    });
  }
}
```

---

## Migration Steps (Detailed)

### Step 1: Backend WebRTC Integration (Week 1-2)

**Tasks:**
1. Research Rust WebRTC libraries (`webrtc-rs`, `str0m`)
2. Add SDP offer/answer endpoints to backend
3. Implement ICE candidate exchange
4. Set up STUN/TURN servers (optional, for NAT traversal)
5. Test signaling flow with simple client

**Deliverable:** Backend can negotiate WebRTC connections

### Step 2: Client WebRTC Services (Week 2-3)

**Tasks:**
1. Create `VoiceSignalingService.ts` (WebSocket wrapper)
2. Create `VoiceRTCService.ts` (RTCPeerConnection wrapper)
3. Create `VoiceConnectionManager.ts` (orchestrator)
4. Write unit tests for each service
5. Test end-to-end connection establishment

**Deliverable:** Client can establish WebRTC connection with backend

### Step 3: Audio Streaming Integration (Week 3-4)

**Tasks:**
1. Integrate `AudioInputService` with WebRTC data channels
2. Integrate `AudioOutputService` with RTC media tracks
3. Test STT via WebRTC audio streams
4. Test TTS playback via WebRTC media tracks
5. Measure latency improvements

**Deliverable:** Full voice chat working via WebRTC

### Step 4: Refactor UI Components (Week 4-5)

**Tasks:**
1. Update `VoiceButton.tsx` to use `VoiceConnectionManager`
2. Update `useVoiceInteraction.ts` hook
3. Update `VoiceIndicator.tsx` for connection status
4. Add WebRTC stats display (optional debug UI)
5. Remove deprecated `VoiceWebSocketService` imports

**Deliverable:** UI fully migrated to WebRTC architecture

### Step 5: Testing & Rollout (Week 5-6)

**Tasks:**
1. Comprehensive testing on different networks
2. Test with packet loss simulation
3. Performance benchmarks (latency, CPU, bandwidth)
4. Feature flag for gradual rollout
5. Documentation update

**Deliverable:** Production-ready WebRTC voice system

---

## Risk Assessment

### Low Risk
- ✅ Backend can keep existing `/ws/speech` endpoint for backward compatibility
- ✅ Feature flag allows gradual rollout
- ✅ WebRTC is well-supported in modern browsers

### Medium Risk
- ⚠️ STUN/TURN servers may be needed for corporate networks
- ⚠️ Increased complexity in connection management
- ⚠️ Need to handle RTC connection failures gracefully

### High Risk
- 🚨 Major architectural change requires thorough testing
- 🚨 Backend WebRTC library selection critical
- 🚨 NAT traversal may require external TURN servers

---

## Success Criteria

✅ **Performance:**
- Latency reduced by ≥30% compared to WebSocket
- Smooth audio playback even with 10% packet loss
- CPU usage comparable or lower

✅ **Reliability:**
- Connection success rate ≥95%
- Automatic reconnection on failure
- Graceful degradation to WebSocket fallback

✅ **Quality:**
- Audio quality maintained or improved
- No audio artifacts or distortion
- Consistent transcription accuracy

✅ **Compatibility:**
- Works on Chrome, Firefox, Safari, Edge
- Mobile browser support (iOS Safari, Android Chrome)
- Corporate network compatibility

---

## Alternative: Hybrid Approach (Fallback)

If full WebRTC migration is too complex:

### Keep WebSocket for Some Use Cases
- **WebRTC:** Real-time voice chat (low latency required)
- **WebSocket:** Pre-recorded TTS playback (latency acceptable)
- **WebSocket:** Transcription results (text data, not audio)

This allows incremental migration while maintaining current functionality.

---

## Next Steps

1. **Prototype Backend:** Test Rust WebRTC library integration
2. **Proof of Concept:** Simple WebRTC audio stream demo
3. **Architecture Review:** Review with team before full implementation
4. **Timeline Planning:** Allocate 6 weeks for full migration
5. **Spike Story:** Create technical spike for feasibility

---

## References

- [WebRTC vs WebSocket for Real-Time Audio (2024)](https://webrtc.ventures/2024/10/real-time-voice-ai-openai-vs-open-source-solutions/)
- [MDN WebRTC API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
- [webrtc-rs Library](https://github.com/webrtc-rs/webrtc)
- [WebRTC Best Practices (Google)](https://webrtc.github.io/samples/)

---

**Status:** Research Complete, Ready for Implementation Planning
**Estimated Effort:** 6 weeks (1 backend developer + 1 frontend developer)
**Priority:** Medium (improves UX but not blocking)
