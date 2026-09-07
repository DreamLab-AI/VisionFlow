---
id: VC-35
title: Voice end to end — PTT, STT, intent, TTS
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/IDENTITY-authority-chain.md
adrs: [ADR-2002, ADR-2039, ADR-2075]
sources:
  - ../project/agentbox/docker-compose.yml
  - ../project/docker-compose.unified.yml
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
verified_commit: dd82a07b0
---

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
    VW->>AR: JSON {type 'set_ptt', ...}
    Note over VW: VoiceWebSocketService.ts:215
    AR->>AR: set_ptt_with_target(user_id, active, target)<br/>src/services/audio_router.rs:202
    AR->>AR: bind_selected_agent(user_id, selected_agent_did)<br/>src/services/audio_router.rs:227
    Note over AR: SetPttRequest threads the DID onto this socket's<br/>AudioRouter session so a following spoken command has<br/>a verifiable target. src/handlers/speech_socket_handler.rs:66-70
```

## VC-35.3 Microphone capture — AudioContext, getUserMedia, MediaRecorder

```mermaid
sequenceDiagram
    autonumber
    participant UI as useVoiceInteraction<br/>client/src/hooks/useVoiceInteraction.ts
    participant VW as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts:35
    participant AI as AudioInputService<br/>client/src/services/AudioInputService.ts:39
    participant ACM as AudioContextManager<br/>client/src/services/AudioContextManager.ts:3
    participant NAV as navigator.mediaDevices
    participant MR as MediaRecorder

    UI->>VW: startRecording()
    VW->>AI: AudioInputService.getBrowserSupport()
    Note over VW: VoiceWebSocketService.ts:254
    VW->>AI: requestMicrophoneAccess(constraints)
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
        AI-->>VW: false
        VW-->>UI: error surfaced - BREAK
    else granted
        NAV-->>AI: MediaStream
    end
    VW->>AI: startRecording(mimeType default 'audio/webm.codecs=opus')
    Note over AI: AudioInputService.ts:178
    alt stream not ready
        AI-->>VW: throw 'Microphone not ready. Call requestMicrophoneAccess first.'
        Note right of AI: AudioInputService.ts:180-181
    end
    AI->>AI: getSupportedMimeType(mimeType) negotiates a codec the browser accepts
    Note over AI: AudioInputService.ts:188
    AI->>MR: new MediaRecorder(stream, {mimeType: supportedType})
    Note over AI: AudioInputService.ts:190-191
    loop while recording
        MR-->>AI: dataavailable Blob chunks
    end
    opt MediaRecorder error
        AI-->>AI: gatedConsole.voice.error('MediaRecorder error')
        Note right of AI: AudioInputService.ts:205
    end
    AI->>AI: on stop, assemble completeAudio Blob
    AI->>VW: setupAudioInputListeners handler reads arrayBuffer()
    Note over AI,VW: VoiceWebSocketService.ts:386-398
    Note over ACM: AudioContextManager is a singleton wrapping one AudioContext<br/>with a webkitAudioContext fallback. AudioContextManager.ts:10-21.<br/>AudioOutputService takes its context from the same singleton<br/>AudioOutputService.ts:28 - one context for the whole app.
```

## VC-35.4 /ws/speech transport — connect, registry, message dispatch

```mermaid
sequenceDiagram
    autonumber
    participant VW as VoiceWebSocketService.connect<br/>client/src/services/VoiceWebSocketService.ts:69
    participant REG as WebSocketRegistry<br/>client/src/services/WebSocketRegistry.ts
    participant BUS as WebSocketEventBus<br/>client/src/services/WebSocketEventBus.ts
    participant SV as speech_socket_handler<br/>src/handlers/speech_socket_handler.rs:971-972
    participant SS as SpeechSocket actor<br/>src/handlers/speech_socket_handler.rs:89
    participant AO as AudioOutputService<br/>client/src/services/AudioOutputService.ts:15

    VW->>VW: wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/speech'
    Note over VW: VoiceWebSocketService.ts:63
    VW->>SV: WebSocket upgrade (no header, no query token)
    rect rgb(238, 244, 252)
        SV->>SV: derive connection_url from scheme, host and path_and_query
        Note over SV: speech_socket_handler.rs:982-995. This is the HTTP-equivalent<br/>URL the client must sign as the NIP-98 u tag - same derivation the<br/>graph socket uses at socket_flow_handler/http_handler.rs:364
        SV->>SS: SpeechSocket::new(id, app_state, None, connection_url, dev_bypass_ok)
        Note over SV,SS: speech_socket_handler.rs:1005. dev_bypass_ok is<br/>dev_bypass_permitted(&req) behind cfg(debug_assertions or dev-auth),<br/>false in release. pubkey starts None
        SV-->>VW: 101 Switching Protocols
        Note over SV,VW: RESOLVED ADR-2075: NO credential is checked at upgrade. The old<br/>path accepted any non-empty Bearer value OR ?token= query parameter<br/>without verifying either, and browsers cannot set WebSocket headers,<br/>so the browser client (which sent neither) was 401d outright. Query<br/>token auth is removed. The socket opens anonymous and useless
    end
    SS->>SS: ctx.run_later(AUTH_DEADLINE) speech_socket_handler.rs:479
    Note over SS: AUTH_DEADLINE = 30s speech_socket_handler.rs:22. A socket still<br/>unauthenticated at the deadline is sent an error and ctx.stop() - an<br/>unauthenticated peer cannot hold an audio broadcast subscription open
    VW->>REG: register('voice', url, socket)
    Note over VW,REG: REGISTRY_NAME = 'voice' VoiceWebSocketService.ts:14, :88
    VW->>BUS: emit('connection:open', {name 'voice', url})
    Note over VW: VoiceWebSocketService.ts:88 - see VC-30.5
    VW->>VW: sendAuthenticate(url) VoiceWebSocketService.ts:90, :132
    alt not nostrAuth.isAuthenticated()
        VW-->>VW: warn "not authenticated - /ws/speech will refuse commands" - BREAK
    else dev mode
        VW->>SS: {"type":"authenticate","token":"dev-session-token","pubkey":...}
        Note over VW,SS: VoiceWebSocketService.ts:139-148. Server accepts ONLY when<br/>dev_bypass_ok, i.e. DEV_AUTH_LOOPBACK=1 and a loopback peer<br/>speech_socket_handler.rs:187-213. Never accepted ungated
    else NIP-98
        VW->>VW: httpUrl = wsUrl with ws scheme swapped for http
        VW->>SS: {"type":"authenticate","event":"<base64 kind-27235>"}
        Note over VW,SS: VoiceWebSocketService.ts:150-154. INVARIANT the signed u tag<br/>must equal the socket's own connection_url, so the client signs<br/>exactly the http equivalent of the URL it connected to
    end
    SS->>SS: handle_authenticate speech_socket_handler.rs:163
    alt dev_full_bypass_active() and dev build
        SS-->>VW: authenticate_success pubkey dev-mode-local-admin
        Note over SS: ADR-2039 LAN-local full bypass, compiled out of release<br/>speech_socket_handler.rs:171-184
    else NIP-98 verified
        SS->>SS: verify_nip98_auth("Nostr <b64>", connection_url, "GET", None)
        Note over SS: via NostrService - single-use replay cache ADR-2002
        SS-->>VW: authenticate_success pubkey
    else verification fails
        SS-->>VW: authenticate_error "NIP-98 WebSocket authentication failed"
        Note over SS,VW: socket stays unauthenticated and is closed at the deadline
    end
    Note over SS: RESOLVED ADR-2075: authenticate is the ONLY frame accepted while<br/>pubkey is None. reject_unauthenticated (speech_socket_handler.rs:141)<br/>refuses tts, stt, voice_command and set_ptt with an error frame.<br/>Ping and pong stay free so heartbeats work pre-auth
    loop each inbound message
        SV-->>VW: MessageEvent
        alt binary (ArrayBuffer or Blob)
            VW->>VW: handleAudioData - Blob converted via arrayBuffer()
            Note over VW: VoiceWebSocketService.ts:221-224
            VW->>AO: enqueue for playback
        else JSON VoiceMessage
            VW->>VW: JSON.parse(event.data)
            Note over VW: VoiceMessage.type is one of<br/>tts | stt | audio_chunk | transcription | error | connected |<br/>authenticate_success | authenticate_error VoiceWebSocketService.ts:17
            alt type 'connected'
                VW-->>VW: mark ready VoiceWebSocketService.ts:180-183
            else type 'transcription'
                VW->>VW: handleTranscription(data) then transcriptionCallback
                Note over VW: VoiceWebSocketService.ts:186, :236
            else type 'error'
                VW-->>VW: message.data or message.error or 'Unknown voice service error'
                Note over VW: VoiceWebSocketService.ts:189-193
            end
        end
    end
    opt socket closes
        VW->>VW: attemptReconnect(url)
        Note over VW: maxReconnectAttempts=5, reconnectDelay=2000ms<br/>VoiceWebSocketService.ts:44-45, :109
    end
```

## VC-35.5 Outbound voice frames the client sends

```mermaid
classDiagram
    class VoiceMessage {
        <<union type>>
        +type tts | stt | audio_chunk | transcription | error | connected
        +data unknown
        +defined at VoiceWebSocketService.ts line 15
    }
    class TtsRequest {
        +type "tts"
        +sent by sendTTSRequest
        +VoiceWebSocketService.ts line 196
    }
    class SetPttRequest {
        +type "set_ptt"
        +pttActive bool
        +selectedAgentDid did:nostr or null
        +client VoiceWebSocketService.ts line 215
        +server SetPttRequest speech_socket_handler.rs line 69
    }
    class VoiceCommandRequest {
        +type "voice_command"
        +text String
        +sessionId Option
        +respondViaVoice Option
        +actorDid Option did:nostr COM-15 D6
        +confidence Option f32 PRD-023 WP-10
        +client VoiceWebSocketService.ts line 235
        +server speech_socket_handler.rs line 47
    }
    class SttRequest {
        +type "stt"
        +audio payload
        +sent at VoiceWebSocketService.ts lines 284 and 316
    }
    class TranscriptionRequest {
        +action String
        +language Option
        +model Option
        +server speech_socket_handler.rs line 40
    }
    VoiceMessage <|-- TtsRequest
    VoiceMessage <|-- SttRequest
    SetPttRequest ..> VoiceCommandRequest : binds the target DID for the next command
    VoiceCommandRequest ..> TranscriptionRequest : follows STT
```

## VC-35.6 STT — Whisper and Turbo Whisper backends

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket
    participant SP as SpeechService<br/>src/services/speech_service.rs:32
    participant CFG as AppFullSettings whisper<br/>src/bin/generate_types.rs:453
    participant WH as whisper-webui-backend
    participant TW as turbo-whisper streaming
    participant AR as AudioRouter<br/>src/services/audio_router.rs:62

    SS->>SP: SpeechCommand::ProcessAudioChunk(bytes)
    Note over SS,SP: or ProcessAudioChunkForUser(bytes, user_id) for<br/>user-scoped routing. src/types/speech.rs:93-95
    SP->>SP: read stt_provider RwLock
    Note over SP: STTProvider = Whisper | TurboWhisper | OpenAI<br/>src/types/speech.rs:72-77
    alt STTProvider::Whisper
        SP->>CFG: config.api_url
        alt unset
            CFG-->>SP: default "http://whisper-webui-backend:8000"
            Note right of CFG: src/services/speech_service.rs:485, :556
        end
        SP->>WH: POST {api_url}/v1/audio/transcriptions
        Note over SP,WH: url built at speech_service.rs:557-559
        WH-->>SP: transcript text
    else STTProvider::TurboWhisper
        SP->>TW: ws://turbo-whisper:8000/v1/audio/transcriptions
        Note over SP,TW: streaming endpoint default crates/visionclaw-domain/src/config/services.rs:192-194<br/>REST fallback http://turbo-whisper:8000/v1/audio/transcriptions :195-197<br/>model default Systran/faster-whisper-large-v3 :198-200.<br/>MOVED 2026-09-05: was src/config/services.rs, now the visionclaw-domain crate
        loop streaming partials
            TW-->>SP: partial transcript
        end
    else STTProvider::OpenAI
        SP->>SP: OpenAI transcription path
    end
    SP->>AR: route_transcription(user_id, text)
    Note over SP,AR: src/services/audio_router.rs:356. Subscribers read via<br/>subscribe_user_transcriptions :381 or the global audio<br/>broadcast :392
    AR-->>SS: transcription broadcast
    SS-->>SS: emit VoiceMessage {type 'transcription'} to the client
    Note over SS: DIVERGENCE Whisper-WebUI is a BROKEN SYMLINK in this tree<br/>(Whisper-WebUI to /mnt/mldata/githubs/Whisper-WebUI, target absent),<br/>so the STT container source is not checked out here. Only the<br/>client-side URL contract above is verifiable from this repo.
```

## VC-35.7 Governed voice path — clarification gate then signed kind-31402 to /v1/voice-intent

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket voice_command match arm<br/>src/handlers/speech_socket_handler.rs:759
    participant GATE as ClarificationGate.evaluate/merge<br/>src/services/voice_clarification (app_state.clarification_gate)
    participant PGV as process_governed_voice<br/>speech_socket_handler.rs:374
    participant PVI as process_voice_intent<br/>speech_socket_handler.rs:314
    participant VC as VoiceIntentClient.dispatch<br/>src/services/voice_intent_client.rs:325 (call site)
    participant AB as agentbox /v1/voice-intent
    participant KO as Kokoro TTS

    SS->>SS: parse VoiceCommandRequest {text, sessionId, respondViaVoice, actorDid, confidence}<br/>speech_socket_handler.rs:51-67
    alt actor_did present and canonical (is_canonical_did)<br/>speech_socket_handler.rs:770-773
        Note over SS,PGV: COM-15 / D6: a bound command ALWAYS takes this governed path<br/>(signed 31402 to /v1/voice-intent to Kokoro ack), never the settings<br/>assistant - falsification clause 2. speech_socket_handler.rs:763-769
        SS->>PGV: process_governed_voice(app_state,session_key,text,actor_did,confidence)<br/>speech_socket_handler.rs:784-792
        PGV->>PGV: take_pending_clarification(session_key)<br/>speech_socket_handler.rs:388-392 - a repair turn if one is pending, else fresh
        PGV->>GATE: pending some ? merge(pending,transcript,confidence) : evaluate(transcript,confidence)<br/>speech_socket_handler.rs:394-397
        alt GateOutcome::Clarify {prompt, pending}
            PGV->>PGV: set_pending_clarification(session_key, pending)<br/>speech_socket_handler.rs:404-407
            PGV->>KO: speak the clarification prompt over Kokoro TTS<br/>speech_socket_handler.rs:411-418 - NOT dispatched
            Note right of PGV: PRD-023 WP-10. Absent confidence is NOT a block - the intent<br/>gate still applies, it is just not gated on missing telemetry.<br/>speech_socket_handler.rs:61-66 (doc), CANARY_V3_REPAIR fire :422-436
            PGV-->>SS: GovernedVoiceResult::Clarified{prompt,slot}
            SS-->>SS: emit voice_clarification, isFinal=false<br/>speech_socket_handler.rs:810-832
        else GateOutcome::Dispatch {transcript}
            PGV->>PVI: process_voice_intent(app_state,transcript,actor_did)<br/>speech_socket_handler.rs:440-443
            PVI->>VC: client.dispatch(&transcript,&actor_did,200)<br/>speech_socket_handler.rs:319-327
            alt no VoiceIntentClient configured
                PVI-->>PGV: Err "governed voice loop unconfigured"<br/>speech_socket_handler.rs:319-322
            else configured
                VC->>AB: POST /v1/voice-intent - endpoint resolution, ACSP_PANEL_NOSTR_PRIVKEY<br/>gating, 31402 signing and identical broker/voice signing all live<br/>inside VoiceIntentClient::dispatch (voice_intent_client.rs) - see EXTERNAL note
                alt accepted
                    AB-->>VC: VoiceIntentAccepted{event_id, intent.verb}
                    PVI->>PVI: ack = ack_sentence(accepted, actor_did)<br/>speech_socket_handler.rs:329
                    PVI->>KO: speak the acknowledgement over Kokoro TTS<br/>speech_socket_handler.rs:331-339 (COM-15 AC3)
                    PVI->>PVI: CANARY_COM15_PTT observed-fire evidence<br/>speech_socket_handler.rs:343-359
                    PVI-->>PGV: Ok(ack)
                    PGV-->>SS: GovernedVoiceResult::Dispatched(ack)
                    SS-->>SS: emit voice_response, isFinal=true, governed=true<br/>speech_socket_handler.rs:794-809
                else rejected or transport failure
                    AB-->>VC: VoiceIntentError::Rejected | Http
                    VC-->>PVI: Err(e)
                    PVI-->>PGV: Err(e.to_string())
                    PGV-->>SS: Err(e) - "governed voice dispatch failed"<br/>speech_socket_handler.rs:833-845 - NEVER re-routed to the<br/>settings assistant on a bound command
                end
            end
        end
    else no actor_did (unbound)
        SS->>SS: is_swarm_command? handle_swarm_voice_command<br/>speech_socket_handler.rs:448-457,849-850<br/>else ungoverned global settings assistant path :851-889
    end
    Note over VC,AB: EXTERNAL: endpoint resolution (AGENTBOX_VOICE_INTENT_URL else<br/>AGENTBOX_MANAGEMENT_URL+VOICE_INTENT_PATH), the ACSP_PANEL_NOSTR_PRIVKEY<br/>gate, and 31402 signing identical to the broker path all live in<br/>VoiceIntentClient (voice_intent_client.rs) - unchanged by this refactor,<br/>only encapsulated behind client.dispatch() instead of being inlined here.
```

## VC-35.8 Ungoverned swarm-intent parse and the voice preamble

```mermaid
classDiagram
    class VoiceCommand {
        +parsed_intent SwarmIntent
        +session_id String
        +parse(text, session_id) Result
        +src/actors/voice_commands.rs line 13 and 116
    }
    class SwarmIntent {
        <<enumeration>>
        SpawnAgent agent_type String capabilities Vec
        QueryStatus target Option
        ExecuteTask description String priority TaskPriority
        UpdateGraph action GraphAction
        ListAgents
        StopAgent agent_id String
        Help
        +src/actors/voice_commands.rs line 43
    }
    class VoicePreamble {
        +generate(intent) String
        +SpawnAgent to " Confirm agent creation."
        +QueryStatus to " Summarize status briefly."
        +ExecuteTask to " Acknowledge task."
        +UpdateGraph to " Confirm graph change."
        +ListAgents to " List agents concisely."
        +StopAgent to " Confirm stopping."
        +Help to " Give brief help."
        +src/actors/voice_commands.rs lines 90 to 104
    }
    class SwarmVoiceResponse {
        +src/actors/voice_commands.rs line 28
    }
    class ConversationContext {
        +src/actors/voice_commands.rs line 82
    }
    VoiceCommand --> SwarmIntent
    SwarmIntent --> VoicePreamble
    VoiceCommand --> ConversationContext
    SwarmVoiceResponse --> VoicePreamble
```

## VC-35.9 TTS — Kokoro and OpenAI backends, audio return path

```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechSocket
    participant SP as SpeechService<br/>src/services/speech_service.rs:32
    participant KC as kokoro settings<br/>src/handlers/speech_socket_handler.rs:278
    participant KO as kokoro-tts-container
    participant OA as api.openai.com
    participant AR as AudioRouter<br/>src/services/audio_router.rs:310
    participant VW as VoiceWebSocketService
    participant AO as AudioOutputService<br/>client/src/services/AudioOutputService.ts:15

    SS->>SP: SpeechCommand::TextToSpeech(text, SpeechOptions)
    Note over SS,SP: SpeechOptions {voice, speed, stream} src/types/speech.rs:99-102.<br/>TextToSpeechForUser routes audio to one user only :84-88
    SS->>KC: read settings.kokoro
    Note over KC: default_voice, default_speed 1.0, stream true<br/>speech_socket_handler.rs:278-288.<br/>KokoroSettings {api_url, default_voice, default_format,<br/>default_speed, timeout, stream, return_timestamps, sample_rate}<br/>src/bin/generate_types.rs:442-451
    SP->>SP: read tts_provider RwLock
    Note over SP: TTSProvider = OpenAI | Kokoro src/types/speech.rs:66-69
    alt TTSProvider::Kokoro
        SP->>SP: api_url_base = config.api_url or "http://kokoro-tts-container:8880"
        Note over SP: src/services/speech_service.rs:363-368
        SP->>KO: POST {base}/v1/audio/speech
        Note over SP,KO: url built speech_service.rs:370-374, sent :389
        alt stream true
            loop audio chunks
                KO-->>SP: streamed audio bytes
            end
        else
            KO-->>SP: complete audio buffer
        end
    else TTSProvider::OpenAI
        SP->>OA: POST https://api.openai.com/v1/audio/speech
        Note over SP,OA: speech_service.rs:289-301.<br/>DIVERGENCE this is the only leg of the voice loop that leaves<br/>the LAN. The Kokoro branch keeps audio local.
        OA-->>SP: audio buffer
    end
    SP->>AR: route_agent_audio(...)
    Note over SP,AR: src/services/audio_router.rs:310. Per-user delivery via<br/>subscribe_user_audio :370, global via subscribe_global_audio :392.<br/>Agent spatial position updated by update_agent_position :297
    AR-->>VW: binary audio frames on /ws/speech
    VW->>AO: enqueue AudioQueueItem
    Note over AO: playbackQueue AudioOutputService.ts:18, gainNode :20,<br/>state idle/... :21, volume 1.0 :25, single currentSource :19
    AO->>AO: processQueue() serialises playback
    Note over AO: AudioOutputService.ts:58. isProcessing and stopRequested<br/>guard re-entrancy and barge-in :23-24
    AO-->>AO: AudioBufferSourceNode through gainNode to destination
```

## VC-35.10 VoiceInterfaceActor and the elevation voice ledger

```mermaid
sequenceDiagram
    autonumber
    participant SP as SpeechService transcript broadcast
    participant VI as VoiceInterfaceActor<br/>src/actors/voice_interface_actor.rs:102
    participant PI as parse_interface_intent<br/>src/actors/voice_interface_actor.rs:91
    participant EA as ElevationActor<br/>src/actors/elevation_actor.rs:116
    participant CI as ConceptIndex<br/>src/actors/elevation_voice.rs:43
    participant HM as harvest_mentions<br/>src/actors/elevation_voice.rs:87
    participant LD as VoiceDemandLedger<br/>src/actors/elevation_voice.rs:120
    participant PE as parse_elevation_intent<br/>src/actors/elevation_voice.rs:192
    participant KO as Kokoro TTS

    Note over VI: "One assistant, two mouths" - the actor speaks over the<br/>REST API and confirms over local Kokoro TTS.<br/>src/actors/voice_interface_actor.rs:6
    SP-->>VI: VoiceLine message
    VI->>PI: parse_interface_intent(text)
    alt an interface intent matches
        PI-->>VI: Some(intent)
        VI->>KO: speak confirmation
    else no match
        PI-->>VI: None - fall through
    end
    Note over VI: Handler<VoiceLine> voice_interface_actor.rs:151,<br/>impl Actor :129
    par elevation guidance
        SP-->>EA: transcript
        EA->>CI: ConceptIndex::build(labels)
        Note over CI: elevation_voice.rs:53, lookup :77, len :68
        EA->>HM: harvest_mentions(transcript, index)
        HM-->>EA: Vec<label>
        loop each mention
            EA->>LD: note(label, excerpt, speaker, now)
            Note over LD: elevation_voice.rs:135
        end
        EA->>LD: score(label, now)
        Note over LD: time-decayed demand score elevation_voice.rs:164.<br/>prune(now) expires stale demands :176
        EA->>PE: parse_elevation_intent(text)
        alt elevation intent recognised
            PE-->>EA: Some(label)
            EA->>KO: speak a short confirmation into the immersive session
            Note over EA,KO: src/actors/elevation_actor.rs:204
        end
        Note over EA: logs "voice guidance active (local Whisper STT to demand<br/>ledger. Kokoro TTS confirmations)"<br/>src/actors/elevation_actor.rs:759
    end
```

## VC-35.11 LiveKit spatial voice — browser path present, XR path absent

```mermaid
sequenceDiagram
    autonumber
    participant UI as caller
    participant LK as LiveKitVoiceService<br/>client/src/services/LiveKitVoiceService.ts:66
    participant DI as dynamic import shim<br/>client/src/services/LiveKitVoiceService.ts:107
    participant RM as LiveKit Room
    participant AC as AudioContext
    participant XR as SpatialVoiceRouter (Godot)<br/>xr-client/rust/src/webrtc_audio.rs:140

    UI->>LK: connect(config {token JWT, room name, url})
    Note over LK: LiveKitConfig token is a JWT minted server-side with the<br/>LiveKit API key and secret. LiveKitVoiceService.ts:23-25
    LK->>DI: Function('m','return import(m)')(livekitModule)
    Note over DI: The SDK is loaded through a runtime-constructed dynamic<br/>import so the bundler cannot statically hoist it -<br/>LiveKit is lazy and optional. LiveKitVoiceService.ts:107
    alt SDK import fails
        DI-->>LK: throw - voice chat unavailable, BREAK
    else loaded
        DI-->>LK: {Room, RoomEvent, Track}
        LK->>RM: new Room(opts)
        Note over LK: LiveKitVoiceService.ts:109
        LK->>RM: connect(url, token)
        RM-->>LK: connected, isConnected = true
        loop each remote participant
            RM-->>LK: track subscribed
            LK->>AC: build a spatial node for the participant
            Note over LK: remoteParticipants Map LiveKitVoiceService.ts:71,<br/>listenerPosition {x,y,z} :70
        end
    end
    Note over XR: DIVERGENCE the Godot XR client has the routing MATHS only.<br/>SpatialVoiceRouterCore (webrtc_audio.rs:39) owns the per-avatar<br/>position map and ListenerTransform (:26) / VoiceTrackState (:33),<br/>but the livekit-android AAR media transport that would consume it<br/>is not wired on any built target. Voice is design-complete,<br/>transport-absent there. docs/XR-client.md 'Known divergences'<br/>bullet 3 - see VC-36.17
```

## VC-35.12 The external voice-stack (Track A) — a separate meta-controller

```mermaid
flowchart TB
    subgraph lap["Laptop browser over tailnet, self-signed TLS"]
        L1["https://host:8444 voice-console<br/>Unmute UI in an iframe plus a live tab-0 feed<br/>voice-stack/README.md:6-8"]
        L2["https://host:8443 Unmute origin<br/>caddy to frontend:3000, /api to backend:80<br/>voice-stack/README.md:9"]
    end
    subgraph gpu["RTX A6000 CUDA device 0"]
        S1["Kyutai STT 1B - semantic VAD, streaming<br/>voice-stack/README.md:10"]
        S2["Kyutai TTS 1.6B - streaming<br/>voice-stack/README.md:11"]
    end
    subgraph compose["voice-stack/unmute/docker-compose.yml"]
        C1["traefik v3.3.1 port 80<br/>unmute/docker-compose.yml:4, :13-14"]
        C2["frontend unmute-frontend:latest<br/>unmute/docker-compose.yml:19"]
        C3["backend unmute-backend:latest<br/>unmute/docker-compose.yml:33"]
        C4["stt moshi-server worker --config configs/stt.toml<br/>unmute/docker-compose.yml:79-80"]
        C5["tts moshi-server worker --config configs/tts.toml<br/>unmute/docker-compose.yml:56-57"]
        C6["llm vllm/vllm-openai:v0.11.0<br/>unmute/docker-compose.yml:102"]
        C7["backend env KYUTAI_STT_URL=ws://stt:8080<br/>KYUTAI_TTS_URL=ws://tts:8080<br/>KYUTAI_LLM_URL=http://llm:8000<br/>unmute/docker-compose.yml:40-42"]
    end
    subgraph box["agentbox container"]
        B1["tab0-bridge port 8971 - OpenAI-compatible<br/>brain headless claude -p, tools tmux send-keys<br/>window 0 only, feed WS /feed<br/>voice-stack/README.md:14-17"]
    end
    L1 --> L2
    L2 --> C2
    L2 --> C3
    C3 --> C7
    C7 --> C4
    C7 --> C5
    C7 --> B1
    C4 --> S1
    C5 --> S2
    C6 -. "replaced by tab0-bridge as the LLM" .-> B1
    box --> SEP
    SEP["SEPARATE SUBSYSTEM: this is the agentbox tmux voice plane<br/>(Track A), not the VisionClaw graph voice loop. Its LLM is the<br/>tab0-bridge, its STT/TTS are Kyutai models, and its grammar is<br/>'tell tab zero to ...' / 'what's tab zero doing?'.<br/>voice-stack/README.md:53-56. The kokoros container serving the<br/>VisionClaw visualiser is explicitly untouched by it<br/>voice-stack/README.md:20 - see AB-06 for the console boundary."]
    compose --> DIV
    DIV["Kokoros, Whisper-WebUI and xinference are UNTRACKED symlinks at the repo root,<br/>gitignored at .gitignore:227-229 and absent from .gitmodules - NOT submodules.<br/>All three dangle in this container (targets /mnt/nvme/githubs/Kokoros,<br/>/mnt/mldata/githubs/Whisper-WebUI, /mnt/nvme/githubs/xinference). git ls-files<br/>returns nothing for any of them. Kokoros and Whisper-WebUI have no tracked<br/>.yml/.toml/.rs/.sh reference and are pure developer convenience; xinference is<br/>DIFFERENT - it has live compose consumers (docker-compose.unified.yml:315,<br/>agentbox/docker-compose.yml:89), so Xinference is a runtime endpoint dependency.<br/>These URL consumers do not prove the dangling checkout link is used. The container contracts are knowable only from the<br/>consuming Rust: kokoro-tts-container:8880 /v1/audio/speech and<br/>whisper-webui-backend:8000 /v1/audio/transcriptions - see VC-35.6 and<br/>VC-35.9. No port or protocol here was read from their own sources. see ES-01.6"]
```
