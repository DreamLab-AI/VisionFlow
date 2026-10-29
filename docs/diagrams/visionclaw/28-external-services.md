---
id: VC-28
title: External services — outbound integrations
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2066, ADR-2115, ADR-2148]
sources:
  - ../project/crates/visionclaw-domain/src/types/speech.rs
  - ../project/src/services/ragflow_service.rs
  - ../project/src/handlers/ragflow_handler.rs
  - ../project/src/services/perplexity_service.rs
  - ../project/src/handlers/image_gen_handler.rs
  - ../project/src/services/github_pr_service.rs
  - ../project/src/services/github/config.rs
  - ../project/src/services/speech_service.rs
  - ../project/crates/visionclaw-domain/src/config/app_settings.rs
  - ../project/src/app_state.rs
  - ../project/src/config/feature_access.rs
  - ../project/docs/reference/configuration.md
  - ../project/docker-compose.unified.yml
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/adr/ADR-2148-enable-local-typed-decisions.md
  - ../project/docs/how-to/agent-orchestration.md
  - ../project/src/main.rs
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
verified_commit: {visionclaw: e74bcddc772d47a45b1971156a2bab540eaa027c, agentbox: 0b0f7918961e629280faf86f88227535abf514b5}
---
## VC-28.1 ragflow_service — outbound RAGFlow agent API
```mermaid
sequenceDiagram
    autonumber
    participant H as ragflow_handler
    participant RS as RAGFlowService<br/>src/services/ragflow_service.rs:101
    participant ENV as env
    participant RF as RAGFlow API<br/>base_url/api/v1/agents

    H->>RS: RAGFlowService::new(settings)<br/>src/services/ragflow_service.rs:135
    RS->>ENV: config_from(env RAGFLOW_API_KEY, RAGFLOW_API_BASE_URL, RAGFLOW_AGENT_ID):138-142
    alt key absent, empty or blank (config_from :119-121)
        RS-->>H: Err RAGFlowError::NotConfigured - the chat service is optional, startup logs one WARN<br/>and leaves ragflow_service None
    else key set but base URL or agent id blank (:122-131)
        RS-->>H: Err RAGFlowError::ParseError - startup logs an error and leaves ragflow_service None
    else all present
        RS-->>H: Ok(RAGFlowService{client,api_key,base_url,agent_id})
    end

    rect rgb(225,230,250)
    Note over RS,RF: trust boundary - outbound HTTPS to configured RAGFLOW_API_BASE_URL
    H->>RS: create_session(user_id) - :158
    RS->>RF: POST {base_url}/api/v1/agents/{agent_id}/sessions?user_id=.. - :170<br/>header Authorization Bearer api_key
    alt status is_success - :180
        RF-->>RS: 200 {data:{id}}
        RS-->>H: Ok(session_id) via result[data][id] - :184
    else non-2xx - :193
        RF-->>RS: status + body
        RS-->>H: Err StatusError(status,body) - :199
    else JSON missing data.id - :186
        RS-->>H: Err ParseError(Failed to parse session ID) - :188
    end
    end

    rect rgb(225,230,250)
    H->>RS: send_chat_message(session_id,message,stream_preference) - :343
    RS->>RF: POST {base_url}/api/v1/agents/{agent_id}/completions - :367<br/>CompletionRequest{question,stream,session_id,sync_dsl:false}
    alt !status.is_success - :377
        RF-->>RS: status + body
        RS-->>H: Err StatusError - :383
    else stream_preference=false - :386
        RF-->>RS: 200 {data:{answer,session_id}}
        alt data.answer present - :396-400
            RS-->>H: Ok(ChatResponse::Buffered{answer,session_id}) - :414
        else data.answer missing - :401-404
            RS-->>H: Err ParseError Answer not found in non-streamed RAGFlow response - :402-404
        end
        Note right of RS: DOC-CORRECTED 2026-09-05: the missing-answer ParseError branch is now drawn<br/>code at ragflow_service.rs:396-405 was already correct - the diagram omitted it
    else stream_preference=true - :419
        RF-->>RS: SSE bytes_stream "data: {...}"
        loop each SSE line - :427
            RS->>RS: parse JSON, extract data.answer chunk - :446-456
        end
        RS-->>H: Ok(ChatResponse::Streaming(byte_stream)) - :480
    end
    end
    Note over H,RS: startup maps NotConfigured to one WARN and any other error to an error log, both leaving ragflow_service None - src/main.rs:431-444
```
## VC-28.2 ragflow_handler — routes and dead handler
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CFG as config()<br/>src/handlers/ragflow_handler.rs:623
    participant H as EnhancedRagFlowHandler<br/>src/handlers/ragflow_handler.rs:192
    participant N as NostrService
    participant RS as RAGFlowService

    Note over CFG: scope /ragflow registers 6 routes - :628-641
    C->>CFG: POST /ragflow/session -> create_session - :139,629
    C->>CFG: POST /ragflow/message -> send_message - :52,630
    C->>CFG: POST /ragflow/chat -> handler.chat_enhanced - :208,631
    C->>CFG: POST /ragflow/session/enhanced -> create_session_enhanced - :415,635
    C->>CFG: GET /ragflow/history/{session_id} -> get_session_history - :170,638
    C->>CFG: GET /ragflow/history/enhanced/{session_id} -> get_session_history_enhanced - :475,639

    rect rgb(225,230,250)
    C->>H: chat_enhanced(req,state,payload) - :208
    alt rate limit exceeded - :216
        H-->>C: 429 TooManyRequests - :221
    else payload > MAX_REQUEST_SIZE - :229
        H-->>C: 413 PayloadTooLarge - :231
    else missing X-Nostr-Pubkey - :245
        H-->>C: 401 Unauthorized - :251
    else missing Authorization Bearer - :383
        H-->>C: 401 Unauthorized - :393
    else nostr_service.validate_session fails - :275
        H->>N: validate_session(pubkey,token) - :275
        N-->>H: false
        H-->>C: 401 invalid_session - :280
    else no ragflow access and not power user - :289
        H-->>C: 403 insufficient_permissions - :294
    else state.ragflow_service is None - :342
        H-->>C: 503 RAGFlow service unavailable
    else ok
        H->>RS: create_session or send_chat_message (see VC-28.1)
        RS-->>H: ChatResponse
        H-->>C: 200 JSON or text/event-stream
    end
    end

    Note over H: RESOLVED ADR-2066: handle_ragflow_chat was defined but wired into no route<br/>(zero route references tree-wide) and has been deleted, with its now-unused<br/>RagflowChatRequest/Responder imports. RagflowChatResponse is kept - chat_enhanced uses it.
```
## VC-28.3 perplexity_service — outbound Perplexity query API
```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant PS as PerplexityService<br/>src/services/perplexity_service.rs:35
    participant SET as AppFullSettings.perplexity
    participant PX as Perplexity API<br/>settings.perplexity.api_url

    Caller->>PS: PerplexityService::new_with_settings(settings) - :53
    PS->>SET: read perplexity.timeout - :62
    PS->>PS: Client::builder().timeout(secs) - :66-68<br/>default 30s if unset - :63

    rect rgb(225,230,250)
    Caller->>PS: query(query,conversation_id) - :92
    PS->>SET: settings.perplexity - :99
    alt perplexity config is None - :101
        PS-->>Caller: Err Perplexity settings not configured - :104
    else api_url missing - :112
        PS-->>Caller: Err Perplexity API URL not configured
    else api_key missing - :116
        PS-->>Caller: Err Perplexity API Key not configured
    else model missing - :120
        PS-->>Caller: Err Perplexity model not configured
    else configured
        PS->>PX: POST api_url - :137<br/>header Authorization Bearer api_key<br/>QueryRequest{query,conversation_id,model,max_tokens,temperature,top_p,presence_penalty,frequency_penalty}
        alt !status.is_success - :144
            PX-->>PS: status + body
            PS-->>Caller: Err Perplexity API error - :150-153
        else 200
            PX-->>PS: {content,link}
            PS-->>Caller: Ok(content) - :157
        end
    end
    end
    Note over PS: DOC-CORRECTED 2026-09-05: endpoint config is read from AppFullSettings.perplexity - api_url, api_key, model at :99-120<br/>not from PERPLEXITY_* env vars - only PERPLEXITY_ENABLED_PUBKEYS gates feature access at feature_access.rs:20 and<br/>PERPLEXITY_API_KEY is read solely as a readiness expectation signal at app_state.rs:1473 - configuration.md:220 and<br/>how-to/agent-orchestration.md:449 corrected
```
## VC-28.4 image_gen_handler — ComfyUI submit path (user session)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant IG as submit_image_job<br/>src/handlers/image_gen_handler.rs:596-600
    participant ENV as env
    participant CF as ComfyUI native API<br/>COMFYUI_URL :37
    participant SOLID as Solid proxy<br/>SOLID_INTERNAL_URL :43

    IG->>ENV: comfyui_base() = COMFYUI_URL or http://comfyui:8188 - :37-38
    C->>IG: POST /image-gen/submit - configure_routes :907
    alt no Nostr session - get_user_npub :568 returns None
        IG-->>C: 401 Authentication required - :601-610
    else authed
        IG->>IG: build_workflow(model,body,seed,prefix) - :621<br/>model = ImageModel::from_env IMAGE_GEN_MODEL - :81-83
        IG->>CF: run_comfyui_job(workflow,job_id) - :630, fn :458
        IG->>CF: POST {comfyui_base}/prompt - :464-467<br/>{prompt:workflow,client_id:job_id} - client timeout 300s :460-462
        alt connection error - :468-475
            CF-->>IG: reqwest Err
            IG-->>C: 503 ComfyUI unreachable - :471-474
        else non-2xx - :477-484
            CF-->>IG: status + body
            IG-->>C: 400 ComfyUI rejected workflow - :480-484
        else no prompt_id in response - :493-502
            IG-->>C: 500 No prompt_id in ComfyUI response - :497-502
        else accepted - :505-509
            loop attempt 0..60 - sleep 5s - :511-512<br/>HISTORY_POLLS :430, HISTORY_POLL_INTERVAL :431
                IG->>CF: GET {comfyui_base}/history/{prompt_id} - :509,513
                CF-->>IG: {[prompt_id]:{outputs}} or poll error (logged, retried) - :514-525
            end
            alt still no saved image after 60 attempts (~5min) - :531-536
                IG-->>C: 504 GatewayTimeout Timed out waiting for ComfyUI - :531-536
            else found
                IG->>CF: GET {comfyui_base}/view?filename=..&subfolder=..&type=output - :538-545
                alt GET or bytes() fails - :546-554
                    IG-->>C: 500 Failed to GET image / fetch image bytes - :549-553
                else bytes ok
                    IG->>SOLID: PUT {solid_base}/api/solid/pods/{npub}/{folder}/{job}.png - :639-647<br/>header Authorization forwarded from client - :656-664, client timeout 60s :649-653
                    alt PUT succeeds (2xx or 201) - :668
                        SOLID-->>IG: stored
                        IG-->>C: 200 {job_id,pod_image_url,comfyui_filename,seed} - :682-692
                    else PUT fails or errors - :672-680
                        SOLID-->>IG: non-2xx or reqwest Err
                        IG-->>C: 200 pod_image_url:null - store skipped, warn logged - :672-680
                    end
                end
            end
        end
    end
```
## VC-28.5 image_gen_handler — ComfyUI agent path (native pipeline), status, health
```mermaid
sequenceDiagram
    autonumber
    participant A as MCP Agent
    participant AS as agent_submit_image_job<br/>src/handlers/image_gen_handler.rs:700-703
    participant ENV as env
    participant CF as ComfyUI native API<br/>COMFYUI_URL :37
    participant POD as embedded solid-pod-rs

    ENV-->>AS: agent_key_authorised() reads VISIONCLAW_AGENT_KEY, constant-time compare<br/>fails closed if unset/empty - NO changeme-agent-key default (ADR-2093) - :50-55<br/>dev or dev-auth builds bypass to always-true - :60-63
    A->>AS: POST /image-gen/agent-submit - configure_routes :908<br/>header X-Agent-Key - :707
    alt X-Agent-Key missing or wrong - :709-716
        AS-->>A: 401 Invalid or missing X-Agent-Key - :710-715
    else authed
        AS->>AS: user_npub defaults to agent - :717-719<br/>build_workflow from IMAGE_GEN_MODEL - :738-740
        Note over AS,CF: CHANGED 2026-10 (af3dff3f): the ComfyUI Salad wrapper is deleted<br/>- COMFYUI_SALAD_URL is gone from the file - and the agent path now runs<br/>the SAME native ComfyUI pipeline as /submit via run_comfyui_job - :747, fn :458<br/>(doc comment :694-698 records why: the wrapper on port 3000 was never reached)
        AS->>CF: POST {comfyui_base}/prompt, poll /history, GET /view - :747
        CF-->>AS: ComfyJob prompt_id + filename + bytes - :459-563
        opt feature solid-pod-embed - :775-776
            AS->>POD: storage.exists/create_container/put - :782-819
            POD-->>AS: Ok(url) or None on failure (warn, non-fatal) - :815-819
        end
        AS-->>A: 200 {job_id,pod_image_url,comfyui_filename,seed} - :756-767
    end

    Note over AS: DIVERGENCE: try_store_in_pod is a no-op returning None when<br/>the solid-pod-embed feature is disabled - :822-831

    participant GJ as get_job_status<br/>src/handlers/image_gen_handler.rs:834-837
    A->>GJ: GET /image-gen/status/{job_id} - configure_routes :909
    GJ->>CF: GET {comfyui_base}/history/{job_id} - :837
    alt request errors - :861-866
        GJ-->>A: 503 ComfyUI unreachable - :861-864
    else body has job_id key - :843
        GJ-->>A: 200 status completed + outputs - :843-850
    else not found yet - :852-861
        GJ-->>A: 200 status pending or unknown by HTTP status - :852-861
    end

    participant HL as health<br/>src/handlers/image_gen_handler.rs:869-873
    A->>HL: GET /image-gen/health - configure_routes :906
    HL->>CF: GET {comfyui_base}/system_stats - :875-877 (timeout 5s :870-873)
    alt 2xx - :878
        HL-->>A: 200 status ok, vram_free/total - :878-886
    else non-2xx or unreachable - :889,893
        HL-->>A: 200 status degraded (never a 5xx) - :889-898
    end
```
## VC-28.6 github_pr_service — outbound GitHub REST API (git data + PR)
```mermaid
sequenceDiagram
    autonumber
    participant Caller as ElevationActor/agent
    participant GH as GitHubPRService<br/>src/services/github_pr_service.rs:21
    participant ENV as env
    participant API as api.github.com<br/>:246

    GH->>ENV: GitHubPRService::new() - :128<br/>token via github_token_from_env(), PRIVATE_REPO_GITHUB_PAT only (see Note below)
    GH->>ENV: GITHUB_OWNER/GITHUB_REPO_OWNER - :130-134
    GH->>ENV: GITHUB_REPO/GITHUB_REPO_NAME - :136-140
    GH->>ENV: GITHUB_BRANCH/GITHUB_BASE_BRANCH default main - :142-146

    rect rgb(225,230,250)
    Caller->>GH: create_ontology_pr(file_path,content,title,body,agent_ctx) - :274
    alt token.is_empty() - :282
        GH-->>Caller: Err PRIVATE_REPO_GITHUB_PAT not configured - :283
    else configured
        GH->>API: GET repos/{owner}/{repo}/git/ref/heads/{base_branch} - get_ref_sha :325,329
        GH->>API: POST git/blobs {content,encoding:utf-8} - create_blob :349,358
        GH->>API: POST git/trees {base_tree,tree:[{path,mode:100644,type:blob,sha}]} - create_tree :379,396
        GH->>API: POST git/commits {message,tree,parents:[base_sha]} - create_commit :419,432
        GH->>API: POST git/refs {ref:refs/heads/{branch},sha} - create_ref :455,462
        alt create_ref 422 branch exists - :473
            GH->>API: PATCH git/refs/heads/{branch} {sha,force:true} - update_ref :490,498
        end
        GH->>API: POST pulls {title,body,head,base,labels:[ontology,agent-proposed]} - create_pull_request :517,532
        alt create_pull_request 422 PR exists - :544
            GH->>API: GET pulls?head={owner}:{branch}&state=open - get_existing_pr_url :564,572
        end
        alt any step non-2xx (not the handled 422s) - :365,405,441,471,541
            API-->>GH: status + body
            GH-->>Caller: Err "<step> failed (<status>): <body>"
        else json parse fails on any response
            GH-->>Caller: Err "Failed to parse <x> response: <e>"
        else success
            GH-->>Caller: Ok(pr.html_url) - :322,560
        end
    end
    end

    rect rgb(225,230,250)
    Caller->>GH: pr_state(pr_ref) - GOV-2 poll :208
    alt token empty - :209
        GH-->>Caller: Err cannot poll PR state - :211
    else no PR number parseable - :214
        GH-->>Caller: Err cannot extract a PR number - :215
    else
        GH->>API: GET pulls/{number} - :216,220
        alt non-2xx - :226
            GH-->>Caller: Err Get PR state failed - :229
        else 200
            API-->>GH: {state,merged_at,merged} - :232
            GH->>GH: classify_pr_state - :196<br/>merged_at/merged=true -> Merged, closed -> ClosedUnmerged, else Open
            GH-->>Caller: Ok(PrState)
        end
    end
    end
    Note over GH,API: every request carries Authorization Bearer token + Accept<br/>application/vnd.github+json + User-Agent VisionClaw-OntologyAgent/1.0 - headers() :251-268
    Note over GH,ENV: token via github_token_from_env(), config.rs:27-28 -<br/>PRIVATE_REPO_GITHUB_PAT only, config.rs:23 - legacy LOGSEQ_PRIVATE_REPO_GITHUB fallback deleted (ADR-2115)
```
## VC-28.7 speech_service — outbound boundary only (see VC-35 for full pipeline)
```mermaid
sequenceDiagram
    autonumber
    participant SS as SpeechService<br/>src/services/speech_service.rs:32
    participant PT as PocketTts<br/>settings.pocket_tts.api_url
    participant WHI as Whisper STT<br/>settings.whisper.api_url
    participant MCP as MCP swarm TCP<br/>MCP_HOST and MCP_TCP_PORT

    rect rgb(225,230,250)
    alt TextToSpeech, TTSProvider PocketTts - speech_service.rs:295
        SS->>SS: abort and await the in-flight tts_task before starting a new one<br/>src/services/speech_service.rs:288
        SS->>PT: POST api_url or http://pocket-tts:8000 plus /v1/audio/speech<br/>body model pocket-tts, response_format pcm, stream true<br/>src/services/speech_service.rs:309, :313, :320, :322
    else StopSpeech - speech_service.rs:264
        SS->>SS: tts_task.abort then await - barge-in cancels mid-utterance<br/>src/services/speech_service.rs:265, public entry at :937
    else STTProvider Whisper, StartTranscription - speech_service.rs:410
        SS->>WHI: uses whisper.api_url or http://whisper-webui-backend:8000<br/>ready-check only, no request here<br/>src/services/speech_service.rs:414, :426
    else STTProvider Whisper, ProcessAudioChunk - speech_service.rs:475
        SS->>WHI: POST multipart to the transcription endpoint<br/>src/services/speech_service.rs:491, :516, :555
        loop poll GET api_url plus /task/identifier, max 30 attempts
            WHI-->>SS: status queued, in_progress, completed or failed<br/>src/services/speech_service.rs:571, :578, :602
        end
        alt status failed or 30 attempts exceeded
            SS-->>SS: error logged, transcription dropped<br/>src/services/speech_service.rs:583, :640
        else completed
            SS-->>SS: broadcast transcription text<br/>src/services/speech_service.rs:603, :617
        end
    end
    end

    rect rgb(225,230,250)
    Note over SS,MCP: voice command intents relay to the multi-agent swarm over MCP TCP<br/>src/services/speech_service.rs:1062
    SS->>MCP: call_swarm_init, call_agent_spawn, call_agent_list, call_task_orchestrate<br/>host MCP_HOST default multi-agent-container, port MCP_TCP_PORT default 9500<br/>src/services/speech_service.rs:1067, :1068, :1079, :1086, :1118
    alt call fails
        MCP-->>SS: error
        SS-->>SS: "Failed to ... Error" spoken back to the user<br/>src/services/speech_service.rs:1133, :1171, :1209
    else ok
        MCP-->>SS: swarmId, agents or taskId JSON
        SS-->>SS: formatted confirmation text
    end
    end

    Note over SS: see VC-35 for the full speech pipeline - WS ingress, tag manager,<br/>voice command parsing, TTS response routing are out of scope here
    Note over SS: DRIFT CLOSED 2026-09-13 (ab5724422): TTSProvider is now a ONE-variant<br/>enum, PocketTts. The OpenAI branch and its api.openai.com call are deleted<br/>and Kokoro is gone, so this boundary no longer reaches any paid TTS API.<br/>visionclaw-domain/src/types/speech.rs:66, default at speech_service.rs:74
    Note over SS: DRIFT: the note above is true of TTS only. STTProvider declares Whisper and OpenAI (the unimplemented TurboWhisper variant<br/>and its two warnings were deleted after af3dff3f2) and the default is Whisper, but SetSTTProvider(OpenAI) is live: ProcessAudioChunk then POSTs<br/>the audio to https://api.openai.com/v1/audio/transcriptions when settings.openai holds an API key, so a third-party speech edge exists on demand.<br/>visionclaw-domain/src/types/speech.rs:71-74, speech_service.rs:75, :689-699, :722
    Note over SS: TextToSpeechForUser, TextToSpeechSpatial, ProcessAudioChunkForUser and AgentSpatialInfo were logged no-ops with no sender and are deleted<br/>(VisionClaw cefd9d1d7 under ER-24), so SpeechCommand carries no user-scoped or spatial entry point.<br/>visionclaw-domain/src/types/speech.rs:77-87
    Note over SS: TENSION (ADR-2151 vs manifest): the agentbox VoiceLink edge voice slot runs voice on the operator's client and reaches the container over an ssh stdio<br/>voice-paned process, never through SpeechService. ADR-2151 is decision_status proposed with implementation_status none and activation inactive,<br/>yet the manifest sets [voicelink] enabled = true. agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5-7, agentbox/agentbox.toml:2218-2223
```
## VC-28.9 consolidated external-dependency map
```mermaid
flowchart LR
    subgraph app["visionclaw_container"]
        RS["RAGFlowService<br/>ragflow_service.rs:101"]
        PS["PerplexityService<br/>perplexity_service.rs:35"]
        IG["image_gen_handler<br/>submit_image_job:628"]
        AG["image_gen_handler<br/>agent_submit_image_job:732"]
        GH["GitHubPRService<br/>github_pr_service.rs:21"]
        SS["SpeechService<br/>speech_service.rs:32"]
    end

    RF["RAGFlow API<br/>env RAGFLOW_API_BASE_URL and RAGFLOW_AGENT_ID - required once RAGFLOW_API_KEY is set, no key = chat disabled with one WARN, boot continues"]
    PX["Perplexity API<br/>settings.perplexity.api_url - no env default"]
    CFU["ComfyUI native<br/>env COMFYUI_URL default http://comfyui:8188"]
    SOL["Solid proxy<br/>env SOLID_INTERNAL_URL default http://127.0.0.1 port 4001, path /api/solid"]
    GHA["api.github.com<br/>env PRIVATE_REPO_GITHUB_PAT / GITHUB_OWNER / GITHUB_REPO"]
    PT["PocketTts<br/>settings.pocket_tts.api_url default http://pocket-tts:8000<br/>visionclaw-domain/src/config/app_settings.rs:99"]
    WHI["Whisper STT<br/>settings.whisper.api_url default http://whisper-webui-backend:8000"]
    MCPS["MCP swarm TCP<br/>env MCP_HOST default multi-agent-container, MCP_TCP_PORT default 9500"]

    RS -->|"POST /api/v1/agents/.. - degraded: ragflow_service None, chat answers 503, no fallback"| RF
    PS -->|"POST api_url - degraded: error surfaced to caller"| PX
    IG -->|"POST /prompt, GET /history, GET /view - fatal for that job"| CFU
    AG -->|"POST /prompt, GET /history, GET /view via run_comfyui_job - fatal for that job"| CFU
    IG -->|"PUT pods/.. - degraded: pod_image_url null, image still returned"| SOL
    GH -->|"git data + pulls API - fatal: PR/state ops fail"| GHA
    SS -->|"TTS - degraded: audio chunk skipped, chat continues"| PT
    SS -->|"STT - degraded: transcription dropped"| WHI
    SS -->|"voice-command relay - degraded: spoken error reply"| MCPS
    JV["Jev hub (opt-in)<br/>12 decision items: library default off, but the supplied Compose environment sets VISIONCLAW_JEV=1 for every item<br/>docs/reference/configuration.md:411-413, docker-compose.unified.yml:5-14<br/>OPEN: opt-in is recorded but activation is staged until the host deployment is recreated - docs/BASELINE-architecture.md:42-50, docs/adr/ADR-2148-enable-local-typed-decisions.md:7"]
    JEP["System One endpoint, first match wins:<br/>VISIONCLAW_JEV_ENDPOINT (Compose supplies the local systemone:8097 facade, model laya-typed-decisions, VISIONCLAW_JEV_LOCAL=1), else agentbox loopback proxy, else api.typesafe.ai with TYPESAFE_API_KEY, else none<br/>docs/reference/configuration.md:454-459, docker-compose.unified.yml:9-11"]
    JV -->|"POST /v1/systemone - degraded: every fallback returns to the deterministic path, no egress when nothing is set<br/>docs/reference/configuration.md:477"| JEP

    N1["RESOLVED ADR-2066 (2026-09-05) — the QuicTransportServer node and its<br/>unreachable XR QUIC 0-RTT client edge were removed from this map with the<br/>code itself. See VC-04.28 for the postcard wire types that survived."]
    SS --- N1
    N2["2026-09-13 (ab5724422): the OpenAI TTS and Kokoro TTS edges are gone with<br/>the enum variants that drove them, so every TTS edge is LAN-local.<br/>DRIFT: the STT side still has an on-demand third-party edge - SetSTTProvider(OpenAI)<br/>sends audio to api.openai.com (default stays Whisper, LAN-local).<br/>visionclaw-domain/src/types/speech.rs:66, :71-74, speech_service.rs:295, :689-699"]
    PT --- N2

    classDef fatal fill:#5a1e1e,stroke:#c0392b,color:#fff
    classDef degraded fill:#4a3a10,stroke:#d4a017,color:#fff
    class GHA,CFU fatal
    class RF,PX,SOL,PT,WHI,MCPS,JEP degraded
```
