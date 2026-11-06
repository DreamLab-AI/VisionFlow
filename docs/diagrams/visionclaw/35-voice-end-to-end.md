---
id: VC-35
title: Voice end to end — PTT, STT, intent, TTS
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/IDENTITY-authority-chain.md
adrs: [ADR-2002, ADR-2039, ADR-2075, ADR-2151, ADR-2148]
sources:
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/docker-compose.speech.yml
  - ../project/agentbox/voice/compose.web.yml
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
  - ../project/docker-compose.unified.yml
  - ../project/docs/reference/configuration.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/adr/ADR-2148-enable-local-typed-decisions.md
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/client/src/services/PushToTalkService.ts
  - ../project/client/src/features/voice/pttAgentBinding.ts
  - ../project/client/src/features/voice/usePushToTalkAgentBinding.ts
  - ../project/client/src/services/AudioContextManager.ts
  - ../project/client/src/services/AudioInputService.ts
  - ../project/client/src/services/AudioOutputService.ts
  - ../project/client/src/services/VoiceWebSocketService.ts
  - ../project/client/src/services/LiveKitVoiceService.ts
  - ../project/client/src/hooks/useVoiceInteraction.ts
  - ../project/client/src/services/WebSocketRegistry.ts
  - ../project/src/handlers/speech_socket_handler.rs
  - ../project/src/services/jev_voice.rs
  - ../project/src/services/speech_service.rs
  - ../project/src/services/audio_router.rs
  - ../project/src/services/voice_intent_client.rs
  - ../project/src/actors/voice_interface_actor.rs
  - ../project/src/actors/voice_commands.rs
  - ../project/src/actors/elevation_voice.rs
  - ../project/src/types/speech.rs
  - ../project/crates/visionclaw-domain/src/config/services.rs
  - ../project/src/bin/generate_types.rs
  - ../project/voice-stack/README.md
  - ../project/voice-stack/unmute/docker-compose.yml
  - ../project/xr-client/rust/src/webrtc_audio.rs
  - ../project/client/src/services/WebSocketEventBus.ts
  - ../project/src/actors/elevation_actor.rs
  - ../project/client/src/components/VoiceButton.tsx
verified_commit: {visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d, agentbox: 5cffdd1d926a9d10c393ec6d6c7a0f4e74ab2925, unmute: e348e567c7020d40020412024f7aa03c38a3bc3e}
---

## For developers

Voice from the push-to-talk key to the spoken reply: microphone capture, the `/ws/speech` socket, speech-to-text, the governed voice-intent path that signs a kind-31402 event (ADR-2148), streamed text-to-speech with barge-in, the voice actors, LiveKit spatial voice, and the external voice estate in agentbox (ADR-2151). Re-read at ec785afa9: the VisionClaw side moved only in generated type lines and the BASELINE activation notes; no voice mechanism changed. The agentbox and unmute stamps are unchanged.

## For the business

A user holds a key, speaks, and the selected agent hears them and answers aloud; speaking again cuts the answer off. Requests that would change the shared graph are not acted on straight from speech: they go through the same signed, reviewable path as typed decisions. Headset users do not yet get the spatial voice the browser has.

## VC-35.1 Push-to-talk state machine and the agent DID binding

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> commanding: keydown on config.key
    commanding --> chatting: keyup and hold >= minHoldDuration and voiceChatEnabled
    commanding --> idle: keyup and hold >= minHoldDuration and not voiceChatEnabled
    commanding --> chatting: keyup and hold < minHoldDuration and voiceChatEnabled
    commanding --> idle: keyup and hold < minHoldDuration and not voiceChatEnabled
    chatting --> commanding: keydown on config.key
    note right of commanding
        PTTState = idle | commanding | chatting
        client/src/services/PushToTalkService.ts:18
        DEFAULT_CONFIG key=' ' (Space), mode='push',
        minHoldDuration=150ms, voiceChatEnabled=true
        PushToTalkService.ts:31-36
        push mode records keyDownTime and enters commanding
        PushToTalkService.ts:189-191
    end note
    note left of idle
        GUARDS on keydown PushToTalkService.ts:180-185
        1. e.key !== config.key returns
        2. e.repeat returns (no key-repeat retrigger)
        3. isEditableTarget(e) returns - never steal a keystroke
           from an input or selection control PushToTalkService.ts:156-177
           walks e.composedPath()[0] so shadow-DOM inputs are caught
        then e.preventDefault()
    end note
    note right of chatting
        TOGGLE mode PushToTalkService.ts:193-199
        commanding to chatting-or-idle on press, otherwise to commanding.
        keyup is ignored entirely in toggle mode PushToTalkService.ts:203
        A sub-minHoldDuration tap is logged and treated as accidental
        PushToTalkService.ts:209-214
    end note
```

## VC-35.2 Selection to PTT binding — did:nostr threading

```mermaid
sequenceDiagram
    autonumber
    participant GR as Graph selection (node-selected event)
    participant HB as usePushToTalkAgentBinding<br/>client/src/features/voice/usePushToTalkAgentBinding.ts
    participant RS as resolveSelectedAgentDid<br/>client/src/features/voice/pttAgentBinding.ts:36
    participant CD as isCanonicalDid<br/>client/src/features/voice/pttAgentBinding.ts:15
    participant PT as PushToTalkService<br/>client/src/services/PushToTalkService.ts:40
    participant VW as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts:35
    participant AR as AudioRouter<br/>src/services/audio_router.rs:62

    GR->>HB: NodeSelectedDetail {nodeId, metadata}
    HB->>RS: resolveSelectedAgentDid(detail)
    RS->>CD: isCanonicalDid(detail.nodeId)
    Note over CD: DID_NOSTR_RE = /^did:nostr:0-9a-f{64}$/<br/>ADR-125 I1 BIP-340 x-only hex.<br/>pttAgentBinding.ts:12
    alt nodeId is itself a canonical DID
        RS-->>HB: detail.nodeId
        Note right of RS: an agent node is KEYED by its DID<br/>pttAgentBinding.ts:42
    else metadata carries a DID
        RS->>CD: isCanonicalDid(metaDid)
        alt canonical
            RS-->>HB: metaDid
        else
            RS-->>HB: null
        end
        Note right of RS: pttAgentBinding.ts:53
    end
    HB->>PT: setSelectedAgentDid(did)
    PT->>CD: isCanonicalDid(did)
    PT->>PT: selectedAgentDid = canonical ? did : null
    Note over PT: INVARIANT a non-canonical DID binds to null,<br/>never to a raw string. PushToTalkService.ts:111
    PT->>PT: setState('commanding') on next PTT press
    PT->>VW: wsNotifyCallback(pttActive = state === 'commanding', selectedAgentDid)
    Note over PT,VW: COM-15 / D6 AC1 - a PTT-start carries the selected<br/>agent's did:nostr onto the server session.<br/>PushToTalkService.ts:229-230, :49
    Note over PT,VW: the hook wires the callback: ptt.onServerNotify((active, did) => voice.setPtt(active, did))<br/>client/src/features/voice/usePushToTalkAgentBinding.ts:62-63
    VW->>AR: JSON {type 'set_ptt', active, actorDid}
    Note over VW: VoiceWebSocketService.ts:279-284. An active edge first sends cancel_tts (VC-35.13) :274-277
    AR->>AR: register_user then set_ptt_with_target(user_id, active, actor_did)<br/>src/handlers/speech_socket_handler.rs:752, :761, :764<br/>src/services/audio_router.rs:202
    Note over AR: SetPttRequest src/handlers/speech_socket_handler.rs:66-70. Some(did) is stored only if<br/>validate_target_did accepts it, while None LEAVES the existing binding<br/>src/services/audio_router.rs:198-201, :211-212
    Note over AR: DEBT the session binding is write-only in production. The only reader,<br/>selected_agent_did (src/services/audio_router.rs:238), is called from tests alone,<br/>and the governed target is the per-command actorDid on voice_command<br/>(VoiceWebSocketService.ts:295, :303 and usePushToTalkAgentBinding.ts:85), not this binding.<br/>The client's unbind sends actorDid null, which the server reads as None and keeps the old DID.
    Note over HB,PT: TENSION the governed predicate is state === 'commanding' AND a canonical DID<br/>(client/src/features/voice/pttAgentBinding.ts:61-64), checked on every FINAL transcript event<br/>(usePushToTalkAgentBinding.ts:73-85). In push mode the speaker's own utterance arrives after Space keyup:<br/>VoiceButton stops listening on keyup (client/src/components/VoiceButton.tsx:115-120), the recording is<br/>sent only on stop (VoiceWebSocketService.ts:399-411) and the same keyup has already moved PTT to chatting<br/>or idle (PushToTalkService.ts:212, :217), so that utterance is NOT dispatched. What does arrive while<br/>Space is held is the server's "Whisper STT ready" status, broadcast as a final transcription<br/>(src/services/speech_service.rs:429), and any other socket's transcript on the global broadcast.<br/>Each of those is sent as a governed voice_command for the bound DID. Source tracing only, not exercised
```

## VC-35.3 Microphone capture — AudioContext, getUserMedia, MediaRecorder

```mermaid
sequenceDiagram
    autonumber
    participant UI as useVoiceInteraction<br/>client/src/hooks/useVoiceInteraction.ts
    participant VW as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts:35
    participant AI as AudioInputService<br/>client/src/services/AudioInputService.ts:36
    participant ACM as AudioContextManager<br/>client/src/services/AudioContextManager.ts:3
    participant NAV as navigator.mediaDevices
    participant MR as MediaRecorder

    UI->>VW: startAudioStreaming({language}) useVoiceInteraction.ts:138
    VW->>AI: AudioInputService.getBrowserSupport()
    Note over VW: VoiceWebSocketService.ts:320. DRIFT the result is only logged as<br/>"DEVELOPER MODE: Browser support checks bypassed" :321 - nothing acts on it here.<br/>The gate that does act is VoiceButton's isSupported (client/src/components/VoiceButton.tsx:73-76)
    VW->>AI: requestMicrophoneAccess() with no constraints VoiceWebSocketService.ts:339
    AI->>AI: resolve getUserMedia across vendor prefixes
    Note over AI: navigator.mediaDevices.getUserMedia, then webkit/moz/ms<br/>legacy shims. AudioInputService.ts:11-13, :77-82
    alt no getUserMedia at all
        AI-->>VW: throw 'getUserMedia is not supported'
        Note right of AI: AudioInputService.ts:111 - BREAK
    else modern path
        AI->>NAV: getUserMedia(defaultConstraints)
        Note over AI,NAV: sampleRate defaults to 48000<br/>AudioInputService.ts:97
    else legacy callback shim
        AI->>NAV: getUserMedia(constraints, resolve, reject) promisified
        Note over AI: AudioInputService.ts:105-108
    end
    alt user denies permission
        NAV-->>AI: DOMException NotAllowedError
        AI-->>VW: false - caught, state 'error' AudioInputService.ts:117-121
        VW-->>UI: throw 'Microphone access denied' - BREAK
        Note over VW: VoiceWebSocketService.ts:340-341
    else granted
        NAV-->>AI: MediaStream
    end
    VW->>AI: startRecording() takes the default mimeType 'audio/webm.codecs=opus'
    Note over AI,VW: VoiceWebSocketService.ts:345, AudioInputService.ts:178
    alt stream not ready
        AI-->>VW: throw 'Microphone not ready. Call requestMicrophoneAccess first.'
        Note right of AI: AudioInputService.ts:180-181
    end
    AI->>AI: getSupportedMimeType(mimeType) negotiates a codec the browser accepts
    Note over AI: AudioInputService.ts:188
    AI->>MR: new MediaRecorder(stream, {mimeType: supportedType})
    Note over AI: AudioInputService.ts:190-191
    loop while recording, 100 ms timeslice AudioInputService.ts:220
        MR-->>AI: dataavailable Blob chunks kept in audioBlobs
        Note over AI: AudioInputService.ts:195-200
    end
    opt MediaRecorder error
        AI-->>AI: gatedConsole.voice.error('MediaRecorder error'), emit error, stopRecording
        Note right of AI: AudioInputService.ts:205-207
    end
    AI->>AI: on stop, assemble completeAudio Blob and emit recordingComplete
    Note over AI: AudioInputService.ts:210-215
    AI->>VW: recordingComplete listener reads arrayBuffer() and sends ONE binary frame
    Note over AI,VW: VoiceWebSocketService.ts:397-411. The audio leaves only when recording stops,<br/>so STT is per utterance, not streamed
    Note over ACM: AudioContextManager is a singleton wrapping one AudioContext<br/>with a webkitAudioContext fallback. AudioContextManager.ts:10-26.<br/>AudioOutputService (AudioOutputService.ts:53) and AudioInputService<br/>(AudioInputService.ts:62) take their context from it. LiveKitVoiceService is<br/>the exception: it builds its own AudioContext (LiveKitVoiceService.ts:151), see VC-35.11
```

## VC-35.4 /ws/speech transport — connect, registry, message dispatch

```mermaid
sequenceDiagram
    autonumber
    participant VW as VoiceWebSocketService.connect<br/>client/src/services/VoiceWebSocketService.ts:70
    participant REG as WebSocketRegistry<br/>client/src/services/WebSocketRegistry.ts
    participant BUS as WebSocketEventBus<br/>client/src/services/WebSocketEventBus.ts
    participant SV as speech_socket_handler<br/>src/handlers/speech_socket_handler.rs:989
    participant SS as SpeechSocket actor<br/>src/handlers/speech_socket_handler.rs:82
    participant AO as AudioOutputService<br/>client/src/services/AudioOutputService.ts:15

    VW->>VW: wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/speech'
    Note over VW: VoiceWebSocketService.ts:65
    VW->>SV: WebSocket upgrade (no header, no query token)
    rect rgb(238, 244, 252)
        SV->>SV: derive connection_url from scheme, host and path_and_query
        Note over SV: speech_socket_handler.rs:1002. The HTTP-equivalent URL the<br/>client must sign as the NIP-98 u tag now comes from the shared helper<br/>nip98_request_url(&req) - same derivation the graph socket uses at<br/>socket_flow_handler/http_handler.rs:427
        SV->>SS: SpeechSocket::new(socket_id, app_state, connection_url, dev_bypass_ok)
        Note over SV,SS: speech_socket_handler.rs:1007-1014, signature :103-108. dev_bypass_ok is<br/>dev_bypass_permitted(&req) :1005, compiled only under cfg(debug_assertions or dev-auth),<br/>absent from release builds. pubkey starts None :124
        SV-->>VW: 101 Switching Protocols
        Note over SV,VW: RESOLVED ADR-2075: NO credential is checked at upgrade. The old<br/>path accepted any non-empty Bearer value OR ?token= query parameter<br/>without verifying either, and browsers cannot set WebSocket headers,<br/>so the browser client (which sent neither) was 401d outright. Query<br/>token auth is removed. The socket opens anonymous and useless
    end
    SS->>SS: ctx.run_later(AUTH_DEADLINE) speech_socket_handler.rs:486
    Note over SS: AUTH_DEADLINE = 30s speech_socket_handler.rs:22. A socket still<br/>unauthenticated at the deadline is sent an error and ctx.stop() :487-500,<br/>so it cannot hold the subscription open beyond 30 s
    Note over SS: TENSION started() subscribes this socket to the SpeechService audio and<br/>transcription broadcasts before any authenticate frame (speech_socket_handler.rs:511-541),<br/>and the forwarders write to the socket with no pubkey check (:556-558, :571-583). For up to<br/>AUTH_DEADLINE an unauthenticated peer receives every TTS chunk and transcript broadcast<br/>to all sockets. ADR-2075 guards inbound commands only
    VW->>REG: register('voice', url, socket)
    Note over VW,REG: REGISTRY_NAME = 'voice' VoiceWebSocketService.ts:14, :89
    VW->>BUS: emit('connection:open', {name 'voice', url})
    Note over VW: VoiceWebSocketService.ts:90 - see VC-30.5
    VW->>VW: sendAuthenticate(url) VoiceWebSocketService.ts:91, :133
    alt not nostrAuth.isAuthenticated()
        VW-->>VW: warn "not authenticated - /ws/speech will refuse commands" - BREAK
    else dev mode
        VW->>SS: {"type":"authenticate","token":"dev-session-token","pubkey":...}
        Note over VW,SS: VoiceWebSocketService.ts:140-149. Server accepts ONLY when<br/>dev_bypass_ok, i.e. DEV_AUTH_LOOPBACK=1 and a loopback peer<br/>speech_socket_handler.rs:179-200. Never accepted ungated
    else NIP-98
        VW->>VW: httpUrl = wsUrl with ws scheme swapped for http
        VW->>SS: {"type":"authenticate","event":"<base64 kind-27235>"}
        Note over VW,SS: VoiceWebSocketService.ts:153-155. INVARIANT the signed u tag<br/>must equal the socket's own connection_url, so the client signs<br/>exactly the http equivalent of the URL it connected to
    end
    SS->>SS: handle_authenticate speech_socket_handler.rs:155
    alt dev_full_bypass_active() and dev build
        SS-->>VW: authenticate_success pubkey dev-mode-local-admin
        Note over SS: ADR-2039 LAN-local full bypass, compiled out of release<br/>speech_socket_handler.rs:163-175
    else NIP-98 verified
        SS->>SS: verify_nip98_auth("Nostr <b64>", connection_url, "GET", None)
        Note over SS: speech_socket_handler.rs:214-225, via NostrService - single-use replay cache ADR-2002
        SS-->>VW: authenticate_success pubkey
    else verification fails
        SS-->>VW: authenticate_error "NIP-98 WebSocket authentication failed"
        Note over SS,VW: speech_socket_handler.rs:239-246. The socket stays unauthenticated and is closed at the deadline
    end
    Note over SS: RESOLVED ADR-2075: authenticate is the ONLY frame accepted while<br/>pubkey is None. reject_unauthenticated (speech_socket_handler.rs:133, applied :627-633)<br/>refuses tts, stt, voice_command, set_ptt and cancel_tts with an error frame.<br/>Ping, pong and the plain-text "ping" stay free so heartbeats work pre-auth :605-618.<br/>Binary audio frames are NOT behind the gate: the Binary arm :956-976 forwards any<br/>frame to process_audio_chunk without a pubkey check
    loop each inbound message
        SV-->>VW: MessageEvent
        alt binary (ArrayBuffer or Blob)
            VW->>VW: handleAudioData - Blob converted via arrayBuffer()
            Note over VW: VoiceWebSocketService.ts:225
            VW->>AO: enqueue for playback
        else JSON VoiceMessage
            VW->>VW: JSON.parse(event.data)
            Note over VW: VoiceMessage.type is one of<br/>tts | stt | audio_chunk | transcription | error | connected |<br/>authenticate_success | authenticate_error VoiceWebSocketService.ts:17
            alt type 'connected'
                VW-->>VW: emit voiceConnected VoiceWebSocketService.ts:181-183
            else type 'transcription'
                VW->>VW: handleTranscription(data) then transcriptionCallback
                Note over VW: VoiceWebSocketService.ts:186, :242
            else type 'error'
                VW-->>VW: message.data or message.error or 'Unknown voice service error'
                Note over VW: VoiceWebSocketService.ts:190-191
            else voice_response, voice_clarification, voice_intent_error, stt_started
                VW-->>VW: not in the VoiceMessage union, so emitted as a generic 'message'
                Note over VW: VoiceWebSocketService.ts:213-214
            end
        end
    end
    opt socket closes
        VW->>VW: attemptReconnect(url)
        Note over VW: only on a close code other than 1000. maxReconnectAttempts=5,<br/>reconnectDelay=2000ms VoiceWebSocketService.ts:44-46, :109-110
    end
```

## VC-35.6 STT — Whisper and OpenAI backends (Turbo Whisper retired)

`ProcessAudioChunkForUser`, `TextToSpeechForUser`, `TextToSpeechSpatial`, `AgentSpatialInfo` and the `TurboWhisper` provider and its `TurboWhisperSettings` were deleted (VisionClaw `cefd9d1d7`, 2026-10-18, estate register ER-24) and are gone at this stamp: `STTProvider` is `Whisper | OpenAI` and `SpeechCommand` has no user-scoped or spatial variants (`src/types/speech.rs:71-88`).

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket Binary arm<br/>src/handlers/speech_socket_handler.rs:956
    participant SP as SpeechService<br/>src/services/speech_service.rs:32
    participant CFG as AppFullSettings whisper<br/>src/bin/generate_types.rs:454
    participant WH as whisper-webui-backend
    participant TX as SpeechService transcription_tx broadcast<br/>src/services/speech_service.rs:43

    SS->>SP: process_audio_chunk(bytes) - one frame per finished recording (VC-35.3)
    Note over SS,SP: speech_socket_handler.rs:968, sent as SpeechCommand::ProcessAudioChunk<br/>src/services/speech_service.rs:923-924. No pubkey check on this arm (VC-35.4)
    SP->>SP: read stt_provider RwLock, default Whisper
    Note over SP: STTProvider = Whisper | OpenAI src/types/speech.rs:71-74,<br/>initialised to Whisper src/services/speech_service.rs:75
    alt STTProvider::Whisper
        SP->>CFG: settings.whisper - when absent the chunk is dropped with an error
        Note right of CFG: src/services/speech_service.rs:482-487, :686
        CFG-->>SP: api_url, else default "http://whisper-webui-backend:8000"
        Note right of CFG: src/services/speech_service.rs:488-491
        SP->>WH: POST multipart {api_url}/transcription/ with model_size, lang, vad_filter...
        Note over SP,WH: url built at speech_service.rs:492-495, header sniffed webm or wav :497-509,<br/>form fields :516-546, posted at :553-555
        WH-->>SP: {identifier} - a queued task, not the transcript
        loop up to 30 polls, 200 ms apart
            SP->>WH: GET {api_url}/task/{identifier}
            Note over SP,WH: src/services/speech_service.rs:570-576, :578-579, :590-591
            WH-->>SP: status queued, in_progress, failed or completed with segment texts
        end
        SP->>TX: send(joined segment text)
        Note over SP,TX: src/services/speech_service.rs:605-617
    else STTProvider::OpenAI
        SP->>SP: POST https://api.openai.com/v1/audio/transcriptions, file sent as audio.wav
        Note over SP: src/services/speech_service.rs:698-699, :701-706. TENSION this branch leaves<br/>the LAN, which VC-35.9 records the TTS loop no longer does. The default<br/>provider is Whisper, so it runs only if SetSTTProvider selects it
        SP->>TX: send(text) src/services/speech_service.rs:744
    end
    TX-->>SS: every SpeechSocket subscribed at started() receives it
    SS-->>SS: TranscriptionMessage handler emits {type 'transcription', isFinal true}
    Note over SS: speech_socket_handler.rs:526-541, :568-584. The broadcast is GLOBAL:<br/>all sockets get every transcript, not only the speaker's
    Note over SP,TX: TENSION a Whisper transcript that contains any of 20 keywords (agent, node, graph,<br/>run, show...) is ALSO executed server-side: is_voice_command, VoiceCommand::parse and<br/>execute_voice_command_with_context (src/services/speech_service.rs:620-633, :1032-1060, :1062),<br/>which can call swarm init and agent spawn on MCP_HOST (:1066-1068, :1079, :1086). That path<br/>has no actorDid, no clarification gate and no signature - it is not the governed path of VC-35.7.<br/>Whether MCP_HOST answers is deployment state
    Note over TX: DEBT status strings share the transcript channel: "Whisper STT ready",<br/>"Transcription stopped" (src/services/speech_service.rs:429, :464) reach clients as<br/>final transcriptions and reach VoiceInterfaceActor and ElevationActor (VC-35.10)
    Note over SS: DIVERGENCE Whisper-WebUI is a BROKEN SYMLINK in this tree<br/>(Whisper-WebUI to /mnt/mldata/githubs/Whisper-WebUI, target absent),<br/>so the STT container source is not checked out here. Only the<br/>VisionClaw side of the URL contract above is verifiable from this repo.
    Note over SS,SP: A separate control frame starts and stops STT — client<br/>{"type":"stt","action":"start"|"stop", "language", "model"} deserializes<br/>to STTActionRequest speech_socket_handler.rs:36-40, matched at :668-679.<br/>"start" builds TranscriptionOptions{language,model,stream:true} and calls<br/>start_transcription :677-688, whose Whisper branch only checks config and<br/>broadcasts "Whisper STT ready" (src/services/speech_service.rs:414-429). Client sends<br/>it from startAudioStreaming VoiceWebSocketService.ts:349-357, model default 'whisper-1' :353
```

## VC-35.7 Governed voice path — clarification gate then signed kind-31402 to /v1/voice-intent

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket voice_command match arm<br/>src/handlers/speech_socket_handler.rs:776
    participant GATE as ClarificationGate.evaluate/merge<br/>src/services/voice_clarification (app_state.clarification_gate)
    participant PGV as process_governed_voice<br/>speech_socket_handler.rs:368
    participant PVI as process_voice_intent<br/>speech_socket_handler.rs:308
    participant VC as VoiceIntentClient.dispatch<br/>src/services/voice_intent_client.rs:207
    participant AB as agentbox /v1/voice-intent
    participant KO as PocketTts TTS

    SS->>SS: parse VoiceCommandRequest {text, sessionId, respondViaVoice, actorDid, confidence}<br/>speech_socket_handler.rs:44, :777-778
    alt actor_did present and canonical (is_canonical_did)<br/>speech_socket_handler.rs:787-790
        Note over SS,PGV: COM-15 / D6 (the ack is now PocketTts, ab5724422): a bound command ALWAYS takes this governed path<br/>(signed 31402 to /v1/voice-intent to a PocketTts ack), never the settings<br/>assistant - falsification clause 2. speech_socket_handler.rs:780-786. A malformed DID is filtered out<br/>and the command falls to the unbound branch below
        SS->>PGV: process_governed_voice(app_state,session_key,text,actor_did,confidence)<br/>speech_socket_handler.rs:802-808. session_key is sessionId else the socket id :797-800
        PGV->>PGV: take_pending_clarification(session_key)<br/>speech_socket_handler.rs:384 - a repair turn if one is pending, else fresh
        opt Jev option (default off, fail-open, Compose flag on: docker-compose.unified.yml:21, final transcript sent via the shared agentbox proxy :7-11) -- VC-35-S1-01
            PGV->>PGV: reply_abandons_pending(jev, pending, transcript) -- on a repair turn Jev may judge the reply abandons the pending request, which is then evaluated fresh instead of merged<br/>src/services/jev_voice.rs:68, speech_socket_handler.rs:393-399
        end
        PGV->>GATE: pending some ? merge(pending,transcript,confidence) : evaluate(transcript,confidence)<br/>speech_socket_handler.rs:397, :399, :402
        PGV->>PGV: tighten(jev, outcome) -- after the gate Jev may only HOLD a dispatch it judges unclear and can never release a held utterance<br/>OPEN: the outcome-changing threshold is unset in Compose (docker-compose.unified.yml:25),<br/>which accepts nothing automatically, so neither Jev call alters a turn yet,<br/>and activation is live (VisionClaw ADR-2148 accepted / complete / live, docs/adr/ADR-2148-enable-local-typed-decisions.md:5-7, matching docs/BASELINE-architecture.md:54-56)<br/>docs/reference/configuration.md:475<br/>src/services/jev_voice.rs:99, speech_socket_handler.rs:404
        alt GateOutcome::Clarify {prompt, pending}
            PGV->>PGV: set_pending_clarification(session_key, pending)<br/>speech_socket_handler.rs:411-414
            PGV->>KO: speak the clarification prompt over the PocketTts path<br/>speech_socket_handler.rs:416-420 - NOT dispatched
            Note right of PGV: PRD-023 WP-10. Absent confidence is NOT a block - the intent<br/>gate still applies, it is just not gated on missing telemetry.<br/>speech_socket_handler.rs:57-58 (doc), CANARY_V3_REPAIR fire :434
            PGV-->>SS: GovernedVoiceResult::Clarified{prompt,slot}
            SS-->>SS: emit voice_clarification, isFinal=false<br/>speech_socket_handler.rs:835, :839
        else GateOutcome::Dispatch {transcript}
            PGV->>PVI: process_voice_intent(app_state,transcript,actor_did)<br/>speech_socket_handler.rs:447-449
            PVI->>VC: client.dispatch(&transcript,&actor_did,200)<br/>speech_socket_handler.rs:318-319
            alt no VoiceIntentClient configured
                PVI-->>PGV: Err "governed voice loop unconfigured"<br/>speech_socket_handler.rs:316
            else configured
                VC->>AB: POST /v1/voice-intent - endpoint resolution, the panel signing key<br/>(load_panel_secret, voice_intent_client.rs:154) gating, 31402 signing and<br/>identical broker/voice signing all live inside<br/>VoiceIntentClient::dispatch (voice_intent_client.rs) - see EXTERNAL note
                alt accepted
                    AB-->>VC: VoiceIntentAccepted{event_id, intent.verb}
                    PVI->>PVI: ack = ack_sentence(accepted, actor_did)<br/>speech_socket_handler.rs:323
                    PVI->>KO: speak the acknowledgement over the PocketTts path<br/>speech_socket_handler.rs:325-328 (COM-15 AC3)
                    PVI->>PVI: CANARY_COM15_PTT observed-fire evidence<br/>speech_socket_handler.rs:344
                    PVI-->>PGV: Ok(ack)
                    PGV-->>SS: GovernedVoiceResult::Dispatched(ack)
                    SS-->>SS: emit voice_response, isFinal=true, governed=true<br/>speech_socket_handler.rs:813, :816-817
                else rejected or transport failure
                    AB-->>VC: VoiceIntentError::Rejected | Http
                    VC-->>PVI: Err(e)
                    PVI-->>PGV: Err(e.to_string())
                    PGV-->>SS: Err(e) - voice_intent_error "governed voice dispatch failed"<br/>speech_socket_handler.rs:850-858 - NEVER re-routed to the<br/>settings assistant on a bound command
                    Note over PVI: DRIFT the doc comment on process_voice_intent still says Err lets the caller<br/>fall back to the settings assistant (speech_socket_handler.rs:305-307), but the caller does not
                end
            end
        end
    else no actor_did (unbound)
        SS->>SS: is_swarm_command? then handle_swarm_voice_command, which only replies<br/>"Swarm voice commands deprecated - use API endpoints instead"<br/>speech_socket_handler.rs:455-464, :466-472, :866-867
        SS->>SS: else the ungoverned settings assistant path, process_voice_command_with_tags<br/>falling back to process_voice_command speech_socket_handler.rs:868-880, :901-902
        Note right of SS: VoiceCommand::parse(text,session_id) src/actors/voice_commands.rs:13,116<br/>maps free text to SwarmIntent (SpawnAgent|QueryStatus|ExecuteTask|<br/>UpdateGraph|ListAgents|StopAgent|Help, enum at voice_commands.rs:43).<br/>process_voice_command calls it at src/services/speech_service.rs:948-952.<br/>DEBT VoicePreamble::generate (voice_commands.rs:90-108) has NO production caller:<br/>only its own tests call it (voice_commands.rs:264, :272), so no voice hint is prefixed
    end
    Note over VC,AB: EXTERNAL: endpoint resolution (AGENTBOX_VOICE_INTENT_URL else<br/>AGENTBOX_MANAGEMENT_URL+VOICE_INTENT_PATH), the panel signing key resolved<br/>by acsp key_file load_panel_secret (voice_intent_client.rs:154-163, was<br/>the ACSP_PANEL_NOSTR_PRIVKEY env var), and 31402 signing identical to the<br/>broker path all live in VoiceIntentClient - encapsulated behind<br/>client.dispatch() instead of being inlined here.
```

## VC-35.9 TTS over PocketTts, streamed PCM and the barge-in cancel path

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket
    participant SP as SpeechService<br/>src/services/speech_service.rs:32
    participant PC as pocket_tts settings<br/>src/handlers/speech_socket_handler.rs:274
    participant PT as pocket-tts service
    participant AX as SpeechService audio_tx broadcast<br/>src/services/speech_service.rs:41
    participant VW as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts:35
    participant AO as AudioOutputService<br/>client/src/services/AudioOutputService.ts:15

    SS->>SP: text_to_speech(text, SpeechOptions) sends SpeechCommand::TextToSpeech<br/>src/services/speech_service.rs:847-852, from the tts frame speech_socket_handler.rs:645-663
    Note over SS,SP: SpeechOptions voice, speed, stream src/types/speech.rs:91.<br/>The user-scoped TextToSpeechForUser and spatial variants are deleted, SpeechCommand has only TextToSpeech(String, SpeechOptions) src/types/speech.rs:77-88
    SS->>PC: read settings.pocket_tts
    Note over PC: default_voice "alba", default_speed 1.0, stream true<br/>speech_socket_handler.rs:276-278, :279-281, :282, folded into SpeechOptions :284-289.<br/>The governed ack, clarification and actor confirmations use SpeechOptions::default()<br/>(alba, 1.0, stream) src/types/speech.rs:104-111.<br/>PocketTtsSettings api_url, default_voice, default_format,<br/>default_speed, timeout, stream<br/>crates/visionclaw-domain/src/config/services.rs:78,<br/>generated client type src/bin/generate_types.rs:443
    SP->>SP: abort the in-flight tts_task, then read tts_provider<br/>src/services/speech_service.rs:288, :292
    Note over SP: DRIFT CLOSED 2026-09-13: TTSProvider is a ONE-variant enum,<br/>PocketTts. The OpenAI branch that was the only leg of this loop<br/>leaving the LAN is deleted, and Kokoro with it.<br/>src/types/speech.rs:66, default at<br/>src/services/speech_service.rs:74
    SP->>PT: POST api_url or http://pocket-tts:8000 plus /v1/audio/speech<br/>model pocket-tts, response_format pcm, stream true<br/>src/services/speech_service.rs:303-315, :318, :320-327
    Note over SP,PT: DEBT the body hard-codes "speed": 1.0 and "stream": true (src/services/speech_service.rs:325-326)<br/>and the bytes are always streamed (:355), so SpeechOptions speed, stream and format<br/>(and the per-request tts speed) never reach pocket-tts. Only voice is forwarded :323
    loop streamed chunks, in a spawned tts_task src/services/speech_service.rs:359-381
        PT-->>SP: bytes the client decodes as signed 16-bit little-endian PCM at 24 kHz
        SP->>AX: audio_tx.send(chunk) src/services/speech_service.rs:366
    end
    AX-->>SS: every SpeechSocket subscribed at started() speech_socket_handler.rs:511-523
    SS-->>VW: binary frame via AudioChunkMessage, ctx.binary speech_socket_handler.rs:553-558
    Note over SP,AX: DRIFT AudioRouter's route_agent_audio (src/services/audio_router.rs:310), subscribe_user_audio<br/>(:370) and update_agent_position (:297) are not on this path: no production code calls them,<br/>and the speech socket uses the router only for set_ptt (VC-35.2). The broadcast is global,<br/>so every connected speech socket plays every utterance. The 24 kHz rate is the client's<br/>assumption (VoiceWebSocketService.ts:233), the server does not state it
    VW->>VW: join pcmRemainder with the frame, keep an EVEN byte count<br/>client/src/services/VoiceWebSocketService.ts:41, :228, :232
    Note over VW: INVARIANT a streamed PCM frame can split a 16-bit sample across<br/>two websocket messages. The odd trailing byte is carried to the next<br/>frame, or every chunk boundary would click.<br/>client/src/services/VoiceWebSocketService.ts:233
    VW->>AO: queuePcm(evenBytes, 24000)<br/>client/src/services/AudioOutputService.ts:29
    AO->>AO: build an AudioBuffer, schedule at max(pcmNextTime, now + 0.02)<br/>client/src/services/AudioOutputService.ts:34, :49
    Note over AO: Gapless: each chunk is scheduled at the running pcmNextTime and<br/>advances it by its own duration, rather than queued behind the older<br/>decode-then-play path. client/src/services/AudioOutputService.ts:27, :49
    AO-->>AO: AudioBufferSourceNode through gainNode to destination
    Note over AO: idle is emitted only when the LAST scheduled source ends<br/>client/src/services/AudioOutputService.ts:42, :44
```

## VC-35.10 VoiceInterfaceActor and the elevation voice ledger

```mermaid
sequenceDiagram
    autonumber
    participant SP as SpeechService transcript broadcast<br/>src/services/speech_service.rs:944
    participant VI as VoiceInterfaceActor<br/>src/actors/voice_interface_actor.rs:102
    participant PI as parse_interface_intent<br/>src/actors/voice_interface_actor.rs:91
    participant EA as ElevationActor<br/>src/actors/elevation_actor.rs:116
    participant CI as ConceptIndex<br/>src/actors/elevation_voice.rs:43
    participant HM as harvest_mentions<br/>src/actors/elevation_voice.rs:87
    participant LD as VoiceDemandLedger<br/>src/actors/elevation_voice.rs:120
    participant PE as parse_elevation_intent<br/>src/actors/elevation_voice.rs:192
    participant KO as PocketTts TTS

    Note over VI: "One assistant, two mouths" - the actor routes spoken configuration requests to the<br/>same settings assistant the Control Center drives (REST) and confirms over local PocketTts TTS.<br/>src/actors/voice_interface_actor.rs:3-7
    SP-->>VI: VoiceLine message, subscribed in started()
    Note over VI: impl Actor voice_interface_actor.rs:129, subscribe_to_transcriptions :137, do_send :140
    VI->>PI: parse_interface_intent(text) - needs a config verb AND an interface noun :93-95
    alt an interface intent matches
        PI-->>VI: Some(command)
        VI->>KO: speak "Adjusting the interface." BEFORE dispatch voice_interface_actor.rs:165
        VI->>VI: CreateTask to TaskOrchestratorActor, agent researcher, settings_assistant_task<br/>voice_interface_actor.rs:162, :168-176
        opt dispatch fails
            VI->>KO: speak "Sorry - the settings assistant is unavailable." :183-185
        end
    else no match
        PI-->>VI: None - fall through :155-157
    end
    Note over VI: Handler<VoiceLine> voice_interface_actor.rs:151
    par elevation guidance
        SP-->>EA: VoiceTranscript, subscribed when speech is present
        Note over EA: src/actors/elevation_actor.rs:1122-1125, logs "voice guidance active (local Whisper STT to<br/>demand ledger. PocketTts TTS confirmations)" :1133
        EA->>PE: parse_elevation_intent(text) FIRST - an explicit command jumps the queue
        Note over EA: Handler<VoiceTranscript> src/actors/elevation_actor.rs:1138, :1145
        alt elevation intent recognised
            PE-->>EA: Some(phrase)
            EA->>CI: lookup(phrase), else keep the phrase elevation_actor.rs:1146-1150
            Note over CI: elevation_voice.rs:77
            alt already seen or pending
                EA->>KO: speak "{label} is already in elevation review." elevation_actor.rs:1151-1153
            else new
                EA->>LD: note(label, line, "", now) elevation_actor.rs:1155
                EA->>EA: publish an ACSP action request and persist the pending case<br/>elevation_actor.rs:1166-1180 - returns silently when ACSP is absent :1156-1158
                EA->>KO: speak "Opened an elevation case for {label}..." elevation_actor.rs:1189-1191
                Note over EA,KO: fn speak - fire-and-forget, never blocks case<br/>handling. src/actors/elevation_actor.rs:229
            end
        else ambient speech
            EA->>HM: harvest_mentions(transcript, concept_index)
            HM-->>EA: Vec<label>
            loop each mention
                EA->>LD: note(label, line, "", now)
                Note over LD: elevation_actor.rs:1200-1201, note at elevation_voice.rs:135
            end
        end
    end
    Note over EA,LD: The ledger is read on the elevation CYCLE, not per transcript: RunCycle prunes stale<br/>demands (elevation_actor.rs:1214, prune at elevation_voice.rs:176) and ranks candidates by the<br/>time-decayed score (elevation_actor.rs:1235-1237, score at elevation_voice.rs:164). The<br/>ConceptIndex starts empty (elevation_actor.rs:219) and is rebuilt from the cycle's labels<br/>(elevation_actor.rs:1316-1318, build at elevation_voice.rs:53, len :68)
    Note over SP,EA: TENSION both actors consume the GLOBAL transcript broadcast, so speech from any<br/>connected speech socket - including one not yet authenticated (VC-35.4) - can open a<br/>settings-assistant task or an elevation case. Neither path carries a speaker identity
```

## VC-35.11 LiveKit spatial voice — browser path present, XR path absent

```mermaid
sequenceDiagram
    autonumber
    participant UI as caller
    participant LK as LiveKitVoiceService<br/>client/src/services/LiveKitVoiceService.ts:66
    participant DI as dynamic import shim<br/>client/src/services/LiveKitVoiceService.ts:106
    participant RM as LiveKit Room
    participant AC as AudioContext
    participant XR as SpatialVoiceRouter (Godot)<br/>xr-client/rust/src/webrtc_audio.rs:141

    UI->>LK: connect(config {serverUrl, token JWT, roomName, spatialAudio, maxDistance})
    Note over LK: LiveKitConfig LiveKitVoiceService.ts:20-31. Caller must supply a JWT - this interface<br/>does not mint it :23-24
    Note over UI,LK: DEBT no caller exists: the only other mention of LiveKitVoiceService in client/src<br/>is a doc comment (client/src/services/PushToTalkService.ts:9), so this path is present but unwired
    LK->>DI: Function('m','return import(m)')(livekitModule)
    Note over DI: The SDK is loaded through a runtime-constructed dynamic<br/>import so the bundler cannot statically hoist it -<br/>LiveKit is lazy and optional. LiveKitVoiceService.ts:106
    alt SDK import fails
        DI-->>LK: throw - voice chat unavailable, BREAK
    else loaded
        DI-->>LK: {Room, RoomEvent, Track}
        LK->>RM: new Room(opts)
        Note over LK: LiveKitVoiceService.ts:108
        Note over LK: handlers for TrackSubscribed, TrackUnsubscribed, ParticipantDisconnected,<br/>Disconnected are registered before connecting LiveKitVoiceService.ts:121-143
        LK->>RM: room.connect(config.serverUrl, config.token)
        RM-->>LK: connected, isConnected = true LiveKitVoiceService.ts:146-147
        opt config.spatialAudio
            LK->>AC: new AudioContext({sampleRate 48000}), listener at the origin
            Note over LK: LiveKitVoiceService.ts:150-158 - its own context, not AudioContextManager's
        end
        loop each remote audio track
            RM-->>LK: TrackSubscribed, kind Audio
            LK->>AC: handleRemoteAudio builds a PannerNode for the participant
            Note over LK: LiveKitVoiceService.ts:124-125, :240, :257. remoteParticipants Map :71,<br/>listenerPosition {x,y,z} :70
        end
    end
    Note over XR: DIVERGENCE the Godot XR client stores voice routing state.<br/>SpatialVoiceRouterCore (webrtc_audio.rs:43) owns the per-avatar<br/>position map and ListenerTransform (:30) / VoiceTrackState (:37),<br/>but the livekit-android AAR media transport that would consume it<br/>is not wired on any built target. Media transport remains absent. docs/XR-client.md 'Known divergences',<br/>the LiveKit bullet - see VC-36
```

The [1 November audit](../../estate-review/2026-11-01-room-admission-and-voice.md) qualifies this historical map: state tests do not prove spatial audio, ADR-2160 explicitly defers voice, and its proposed revocation argument has an unresolved dependency-refresh path. See [VC-50](50-deferred-space-voice.md).

## VC-35.12 The external voice estate (Track A) after the 2026-09-13 re-ownership

```mermaid
flowchart TB
    subgraph owned["Speech synthesis, owned by agentbox"]
        P1["pocket-tts, one warm CPU-only service on visionclaw_network<br/>started by BOTH the Agentbox and VisionClaw launchers<br/>voice-stack/README.md:3, :4"]
        P2["HTTP /v1/audio/speech serves narration and VisionClaw<br/>incremental MessagePack /api/tts_streaming serves the web backend<br/>voice-stack/README.md:5, :6"]
        P3["agentbox/docker-compose.speech.yml, container pocket-tts<br/>loopback publish on port 8898 to container port 8000<br/>agentbox/docker-compose.speech.yml:39, :43, :44"]
    end
    subgraph web["Web stack, maintained definitions"]
        W1["agentbox/voice/compose.web.yml is the maintained definition set<br/>launched by agentbox/agentbox.sh voice up<br/>voice-stack/README.md:10, :11"]
        W2["traefik v3.3.1, unmute-frontend, unmute-backend<br/>agentbox/voice/compose.web.yml:4, :20, :35"]
        W3["backend env KYUTAI_STT_URL ws to speech-to-text<br/>KYUTAI_TTS_URL ws to pocket-tts<br/>KYUTAI_LLM_URL http to agentbox tab0-bridge<br/>agentbox/voice/compose.web.yml:43, :44, :45"]
    end
    subgraph legacy["voice-stack/unmute checkout"]
        U1["source context for the web frontend, backend and the existing STT<br/>NO LONGER owns a speech-synthesis container<br/>voice-stack/README.md:8, :9"]
        U2["unmute/docker-compose.yml is the UPSTREAM file: services traefik,<br/>frontend, backend, tts, stt and llm<br/>unmute/docker-compose.yml:3, :18, :32, :55, :78, :101"]
    end
    P1 --> P2 --> P3
    W1 --> W2 --> W3
    W3 --> P1
    U1 --> W1
    U2 -.-> INV
    INV["INVARIANT: the upstream Compose file must NOT be launched<br/>independently. It is read as source context only; the maintained<br/>definitions are the agentbox ones above.<br/>voice-stack/README.md:11, :12"]
    U2 -.-> UNPIN
    UNPIN["DEBT: voice-stack/unmute is a NESTED checkout that the VisionClaw repo<br/>does not track, so a citation into it can only be pinned through its own<br/>revision. This topic carries an unmute key in verified_commit for exactly<br/>that reason, and its working tree is dirty: read the declared revision,<br/>never the files on disk.<br/>unmute/docker-compose.yml:3"]
    legacy --> SEP
    VL["TENSION: ADR-2151 (VoiceLink, voice at the client edge over the ssh attach chain) is decision_status proposed, activation inactive, and implementation partial<br/>since its 2026-11-02 reconciliation (was none; source and bake exist), agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5-7, :21, while the live manifest has the [voicelink] slot enabled = true, agentbox/agentbox.toml:2264.<br/>It is a per-connection voice-paned process, not part of Track A or the VisionClaw voice loop"]
    SEP -.-> VL
    SEP["SEPARATE SUBSYSTEM: this is the agentbox tmux voice plane (Track A),<br/>not the VisionClaw graph voice loop of VC-35.1 to VC-35.11. Its LLM is<br/>the tab0-bridge reached as KYUTAI_LLM_URL, its STT is the Kyutai<br/>service, and its TTS is now the SAME pocket-tts VisionClaw uses.<br/>agentbox/voice/compose.web.yml:45, :44. see AB-06 for the console boundary"]
    owned --> CONV
    CONV["DRIFT CLOSED 2026-09-13 (ab5724422): the two tracks used to run<br/>different synthesisers - Kokoro for the VisionClaw visualiser and<br/>Kyutai TTS for the web stack. One pocket-tts service now serves both,<br/>and voice-stack/unmute-override.yml was deleted with the split.<br/>voice-stack/README.md:3, :8"]
    DIV["Kokoros, Whisper-WebUI and xinference are UNTRACKED symlinks at the repo root,<br/>gitignored at .gitignore lines 240-242 and absent from .gitmodules - NOT submodules.<br/>All three dangle in this container. git ls-files returns nothing for any of them.<br/>Kokoros and Whisper-WebUI are now doubly stale: the Kokoro TTS branch was<br/>deleted from the server on 2026-09-13. xinference is DIFFERENT - it has live<br/>compose consumers (docker-compose.unified.yml:160, :405, agentbox/docker-compose.yml:86),<br/>so Xinference is a runtime endpoint dependency. Those URL consumers do not prove<br/>the dangling checkout link is used. see ES-01.6"]
    legacy --> DIV
    HIST["Historical engine benchmarks and their retired harness live under<br/>docs/gap-close-evidence/voice-latency-2026-09-10/ and are NOT<br/>deployment files. voice-stack/README.md:14, :15"]
    CONV --> HIST
```

## VC-35.13 Barge-in, cancelling an utterance already in flight

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator presses push-to-talk
    participant VW as VoiceWebSocketService.setPtt<br/>client/src/services/VoiceWebSocketService.ts:272
    participant AO as AudioOutputService.stop<br/>client/src/services/AudioOutputService.ts:148
    participant SS as SpeechSocket message dispatch<br/>src/handlers/speech_socket_handler.rs:603
    participant SP as SpeechService.stop_speech<br/>src/services/speech_service.rs:933
    participant CMD as SpeechCommand StopSpeech arm<br/>src/services/speech_service.rs:264

    U->>VW: setPtt(true, actorDid)
    VW->>AO: audioOutput.stop()
    AO->>AO: stop and disconnect every scheduled PCM source, reset pcmNextTime<br/>client/src/services/AudioOutputService.ts:150, :152
    VW->>VW: pcmRemainder reset to empty<br/>client/src/services/VoiceWebSocketService.ts:276
    VW->>SS: {"type":"cancel_tts"} VoiceWebSocketService.ts:277 - refused before authenticate (VC-35.4)
    SS->>SP: stop_speech() spawned on the actor context<br/>src/handlers/speech_socket_handler.rs:639
    SP->>CMD: SpeechCommand StopSpeech
    CMD->>CMD: tts_task.abort then await<br/>src/services/speech_service.rs:265
    Note over CMD: INVARIANT the abort is awaited, so the aborted task cannot still be<br/>writing chunks when the next TextToSpeech starts. The same abort runs<br/>on Close and at the head of TextToSpeech.<br/>src/services/speech_service.rs:270, :288
    Note over VW,CMD: Cancellation is THREE-SIDED: the browser stops what is already<br/>scheduled, the socket tells the server to stop producing, and the<br/>service aborts the generating task. Stopping only the browser would<br/>leave the model generating into a dropped stream.
    Note over SS,CMD: TENSION there is ONE tts_task in the SpeechService loop and its audio is broadcast<br/>to every socket (VC-35.9), so any authenticated socket's cancel_tts or new tts frame aborts<br/>the utterance every other listener is hearing. src/services/speech_service.rs:264-268, :287-291
    Note over VW: A new sendTextForTTS also stops local playback and clears the<br/>remainder before sending, so a fresh utterance never plays over<br/>the tail of the old one.<br/>client/src/services/VoiceWebSocketService.ts:255, :256
```

## VC-35.14 Session lifecycle — connect, heartbeat timeout, auth deadline, reconnect

```mermaid
sequenceDiagram
    autonumber
    participant VW as VoiceWebSocketService.connect<br/>client/src/services/VoiceWebSocketService.ts:70
    participant REG as WebSocketRegistry
    participant SS as SpeechSocket actor<br/>src/handlers/speech_socket_handler.rs:479

    VW->>SS: new WebSocket(url)
    SS->>SS: started - spawn heartbeat and auth-deadline timers<br/>speech_socket_handler.rs:482, :486
    Note over SS: HEARTBEAT_INTERVAL=5s, CLIENT_TIMEOUT=10s, AUTH_DEADLINE=30s<br/>speech_socket_handler.rs:17, :18, :22
    alt no authenticate frame within AUTH_DEADLINE
        SS-->>VW: error "authentication deadline exceeded"
        SS->>SS: ctx.stop()<br/>speech_socket_handler.rs:487-500
    else client stops answering pings
        SS->>SS: Instant::now - heartbeat > CLIENT_TIMEOUT<br/>speech_socket_handler.rs:254-257
        SS->>SS: log "heartbeat failed, disconnecting" then ctx.stop()
    else client sends a Close frame
        SS->>SS: log "Client disconnected", ctx.close, ctx.stop<br/>speech_socket_handler.rs:978-982
    end
    SS-->>VW: onclose {code, reason}
    VW->>REG: webSocketRegistry.unregister('voice')<br/>VoiceWebSocketService.ts:102
    VW->>VW: emit('disconnected', event)<br/>VoiceWebSocketService.ts:108
    alt event.code !== 1000 (abnormal closure)
        VW->>VW: attemptReconnect(url)<br/>VoiceWebSocketService.ts:110, :461-468
        Note over VW: reconnectAttempts capped at maxReconnectAttempts=5,<br/>fixed reconnectDelay=2000ms between tries - no backoff growth.<br/>VoiceWebSocketService.ts:44-46
        VW->>VW: setTimeout -> connect(url) again
    else caller-initiated disconnect()
        Note over VW: disconnect() closes with code 1000 (:509), so the onclose<br/>abnormal-closure check (:109) does not reconnect. It then sets<br/>reconnectAttempts = maxReconnectAttempts (:513), which stops further attempts, though a reconnect<br/>already scheduled by setTimeout still fires (:464-467). VoiceWebSocketService.ts:505-513
    end
```
