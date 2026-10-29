---
id: AB-12
title: tab0-bridge and the interaction plane
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2009, ADR-2010, ADR-2011, ADR-2047, ADR-2088, ADR-2151, ADR-2153]
sources:
  - ../project/agentbox/config/tab0-bridge/server.mjs
  - ../project/agentbox/config/tab0-bridge/turn-sink.cjs
  - ../project/agentbox/config/tab0-bridge/start.sh
  - ../project/agentbox/config/tab0-bridge/deploy.sh
  - ../project/agentbox/config/tab0-bridge/package.json
  - ../project/agentbox/config/tab0-bridge/README.md
  - ../project/agentbox/management-api/lib/voice-intent.js
  - ../project/agentbox/management-api/routes/voice-intent.js
  - ../project/agentbox/management-api/lib/junkiejarvis-agent.js
  - ../project/agentbox/management-api/lib/agent-control-surface.js
  - ../project/agentbox/management-api/routes/approvals.js
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/voice/README.md
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/voice/console/Caddyfile
  - ../project/agentbox/flake.nix
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/config/nostr-gateway/nostr-send.cjs
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/management-api/lib/zone-keys.js
  - ../project/agentbox/management-api/lib/voice-intent-jev.js
  - ../project/agentbox/management-api/lib/approval-review-jev.js
  - ../project/agentbox/scripts/aoe-seed-sessions.mjs
  - ../project/agentbox/config/hooks/trust-seed.cjs
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
verified_commit: 1da8d21719a27394f190866223b88533b52d87f9
---

## AB-12.2 Bridge boot — reconcile, listen, coordinator resolve
```mermaid
sequenceDiagram
    autonumber
    participant Sup as Supervisor<br/>agentbox/flake.nix:2975
    participant Dep as deploy.sh<br/>agentbox/config/tab0-bridge/deploy.sh:1
    participant Node as server.mjs<br/>agentbox/config/tab0-bridge/server.mjs:45
    participant AoEd as AoE daemon port 9095

    Sup->>Dep: bash deploy.sh reconcile, one supervisor command line (flake.nix:2976)
    Dep->>Dep: copy server.mjs, turn-sink.cjs, start.sh, package.json via md5 compare (deploy.sh:27-35)
    opt node_modules/ws missing
        Dep->>Dep: npm install --omit=dev (deploy.sh:37-39)
    end
    Dep-->>Sup: exit 0, reconcile-only mode, no launch (deploy.sh:44-46)
    Sup->>Node: exec node server.mjs, foreground, autorestart, same command line (flake.nix:2976)
    Node->>Node: read BRIDGE_PORT, default 8971 (server.mjs:45)
    Node->>Node: read BRIDGE_TMUX_SESSION, default agentbox (server.mjs:47)
    Node->>Node: read BRIDGE_TOKEN, default empty string (server.mjs:49)
    Node->>Node: read BRIDGE_BIND, default 0.0.0.0 (server.mjs:55)
    Node->>Node: BIND_IS_LOOPBACK check, 127.0.0.1, ::1, localhost (server.mjs:56)
    alt TOKEN empty and BIND not loopback
        Node->>Node: console.error, refusing to start (server.mjs:60-65)
        Node->>Node: process.exit(1)
        Note over Node: INVARIANT — a non-loopback bind with no BRIDGE_TOKEN is refused at startup, server.mjs:60-66
    end
    Node->>Node: delete CHILD_ENV.ANTHROPIC_API_KEY, empty key poisons OAuth chain (server.mjs:123-126)
    Node->>Node: set CHILD_ENV.MAX_THINKING_TOKENS 0 and CLAUDE_CODE_DISABLE_CLAUDE_MDS 1 for lean headless turns, read BRIDGE_EFFORT default low (server.mjs:131-133)
    Node->>Node: http.createServer, WebSocketServer path /feed (server.mjs:739,824-828)
    Node->>Node: server.listen(PORT, BIND) (server.mjs:837)
    Node->>AoEd: GET /api/sessions?state=live, resolveCoordinatorSession (server.mjs:223-225)
    alt AoE reachable and title matches tab0
        AoEd-->>Node: 200, session list
        Node->>Node: pin aoeSessionId for process lifetime (server.mjs:234, ADR-044 D2)
    else AoE unreachable or no match
        AoEd-->>Node: error or no match
        Node->>Node: aoeSessionId stays null, fall back to tmux (server.mjs:238-241)
    end
    loop every 30000 ms while aoeSessionId is null
        Node->>AoEd: GET /api/sessions?state=live, re-resolve (server.mjs:847)
    end
```

## AB-12.3 Global HTTP auth gate
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant B as tab0-bridge<br/>server.mjs:739

    C->>B: HTTP request, any path (server.mjs:739-741)
    alt path is /health
        B-->>C: 200, ok true, no auth check (server.mjs:743-746)
    else path requires auth
        B->>B: authorised(req) (server.mjs:725)
        alt TOKEN is empty
            B->>B: return true, open gate, loopback dev only (server.mjs:726)
        else Authorization header equals Bearer TOKEN
            B->>B: return true (server.mjs:728)
        else Authorization starts with Nostr, verifyNip98Credential succeeds
            B->>B: return true, see AB-10.3 for NIP-98 verify internals (server.mjs:729,703-723)
        else query token equals TOKEN
            B->>B: return true (server.mjs:732)
        else query auth verifies as a Nostr credential
            B->>B: return true (server.mjs:733-734)
        else none of the above
            B->>B: return false (server.mjs:736)
        end
        alt authorised false
            B-->>C: 401, error unauthorised (server.mjs:749)
        else authorised true
            B->>B: dispatch to route table (server.mjs:750-810)
        end
    end
Note over B: INVARIANT — every surface except /health requires the bearer when BRIDGE_TOKEN is<br/>set, including /v1/chat/completions, /v1/models, /feed, /tab0/send, /nostr/*, /turns, /tabs*,<br/>/aoe/sessions — ADR-044 finding 1, server.mjs:640-655
Note over B: the /feed WebSocket upgrade reuses this identical gate — verifyClient(info) calls<br/>authorised(info.req) (server.mjs:824-828) — but browser WS clients cannot set an Authorization<br/>header, so ?token= or ?auth= is the only carrier there (server.mjs:644,651) — on accept the<br/>connection handler sends a snapshot of the last 50 turns (server.mjs:833-835)
```

## AB-12.5 Inject into tmux window 0 — sendToTab0
```mermaid
sequenceDiagram
    autonumber
    participant Cl as Client
    participant B as tab0-bridge<br/>server.mjs:773, POST /tab0/send
    participant S as sendToTab0()<br/>server.mjs:273
    participant A as aoeSend()<br/>server.mjs:249
    participant AoEd as AoE daemon port 9095
    participant T as tmux CLI

    Cl->>B: POST /tab0/send, body text, source (server.mjs:773-776)
    B->>S: sendToTab0(text, source) (server.mjs:774)
    S->>S: clean = strip control chars, trim (server.mjs:274)
    alt clean is empty
        S-->>B: throw Error, empty text (server.mjs:275)
    end
    S->>A: aoeSend(clean) (server.mjs:278)
    A->>A: aoeSessionId unset, resolveCoordinatorSession() (server.mjs:250-252)
    A->>AoEd: POST /api/sessions/:id/send, Authorization Bearer aoe-token, body message clean (server.mjs:254)
    alt AoE responds 200
        AoEd-->>A: 200
    else AoE responds 404, session drift
        AoEd-->>A: 404
        A->>A: aoeSessionId = null, re-resolve once (server.mjs:255-257)
        A->>AoEd: retry POST /api/sessions/:id/send (server.mjs:259)
    else AoE unreachable or non-200
        AoEd-->>A: transport error or non-200 (server.mjs:261)
        A-->>S: throw Error
    end
    alt aoeSend succeeded
        S->>S: via = aoe (server.mjs:276)
    else aoeSend threw
        S->>S: via = tmux, console.error, AoE unreachable falling back (server.mjs:279-284)
        S->>T: tmux send-keys -t agentbox:0 -l clean (server.mjs:285)
        S->>S: await 1500 ms so the Enter is not read as part of a paste (server.mjs:286-288)
        S->>T: tmux send-keys -t agentbox:0 Enter (server.mjs:289)
        Note over S: FAIL-OPEN — degrade to the legacy tmux send-keys path, races AoE input accounting, ADR-044 D3 (server.mjs:279-284)<br/>the fallback now pauses 1500 ms between the text and the Enter because a TUI such as Codex reads a text+Enter burst as a paste and keeps the Enter as a newline (server.mjs:286-288)<br/>nostr-gateway sendKeys does the same with a scheduled, non-blocking Enter (config/nostr-gateway/gateway.cjs:289-295)
    end
    S->>S: pushTurn(voice-inject or nostr-inject, clean, via) (server.mjs:291)
    S-->>B: return clean text
    B-->>Cl: 200, ok true, sent clean (server.mjs:775)
```

## AB-12.6 AoE coordinator session resolution and drift retry
```mermaid
sequenceDiagram
    autonumber
    participant Ti as 30s interval<br/>server.mjs:847
    participant R as resolveCoordinatorSession()<br/>server.mjs:223
    participant AoEd as AoE daemon port 9095

    Ti->>R: invoke when aoeSessionId is null (server.mjs:847)
    R->>AoEd: GET /api/sessions?state=live (server.mjs:225)
    alt status not 200
        AoEd-->>R: non-200
        R-->>Ti: return null (server.mjs:226)
    else status 200
        AoEd-->>R: 200, sessions array or wrapped data (server.mjs:211-215)
        R->>R: want = AOE_COORDINATOR_TITLE lowercased, default tab0 (server.mjs:117,227)
        R->>R: match session whose title, slug or name equals or includes want (server.mjs:228-231)
        alt match found with id or session_id
            R->>R: aoeSessionId = String(id), pin for process lifetime (server.mjs:234, ADR-044 D2)
            R-->>Ti: return aoeSessionId
        else no match
            R-->>Ti: return null (server.mjs:238)
        end
    end
    alt any error thrown, fetch abort, connection refused
        R-->>Ti: catch, return null, AoE not up yet (server.mjs:239-240)
    end
Note over R: INVARIANT — tab0-bridge targets exactly ONE pinned coordinator session<br/>(server.mjs:111-117). POST /tab0/send carries no per-request session id, so an<br/>arbitrary-session inject surface does not exist here — contrast the AoE daemon's own<br/>multi-session API reached via the proxy, see AB-10.x
```

## AB-12.7 Unmute voice loop through tab0-bridge
```mermaid
sequenceDiagram
    autonumber
    participant Mic as Browser mic, port 8444 cockpit
    participant Cad as Caddy port 8444<br/>voice/console/Caddyfile
    participant UFE as Unmute frontend port 3000
    participant UBE as Unmute backend port 80
    participant B as tab0-bridge<br/>server.mjs:753, POST /v1/chat/completions
    participant Cc as claude -p child<br/>server.mjs:301

    rect rgb(255,240,240)
    Note over Mic,UBE: trust boundary — LAN door 1, port 8444 published 0.0.0.0
    Note over Mic,Cad: TENSION resolved in the manifest's own terms — [voice].enabled is now TRUE<br/>(agentbox.toml:2209), descriptive sidecar state for /v1/system. The gate stays apply-class<br/>sidecar, so the voice compose overlay still has its own lifecycle and `agentbox up`<br/>never consults the flag (docker-compose.voice.yml:1) - the manifest describes the<br/>surface, the overlay governs it
    Mic->>Cad: HTTPS, mic audio via /embed and /api/* (Caddyfile handle /embed*, handle_path /api/*)
    Cad->>UFE: reverse_proxy frontend:3000 (Caddyfile handle /embed*)
    Cad->>UBE: reverse_proxy backend:80, /v1/realtime (Caddyfile handle_path /api/*)
    end
    UBE->>B: POST /v1/chat/completions, Authorization Bearer BRIDGE_TOKEN aka KYUTAI_LLM_API_KEY, stream true (server.mjs:753-756, voice/README.md:88)
    B->>B: authorised(req), global gate, see AB-12.3
    alt userText equals the silence marker or is empty
        B-->>UBE: SSE, empty content, finish_reason stop, no LLM call (server.mjs:515-528)
    else userText has content
        B->>B: pushTurn(voice-user, text) (server.mjs:533)
        B->>B: metaSystemPrompt(), recent turns, tmux windows, AoE-or-tmux job instructions (server.mjs:379-447)
        B->>B: metaAllowedTools(), Bash allowlist, tmux and AOE_CURL patterns (server.mjs:455-475)
        B->>Cc: spawn claude -p --model haiku --effort BRIDGE_EFFORT --strict-mcp-config<br/>--mcp-config empty --settings disableAllHooks --disable-slash-commands --tools Bash?, stream-json (server.mjs:308-321)
        loop stream_event, content_block_delta, text_delta
            Cc-->>B: partial text chunk (server.mjs:335-342)
            B-->>UBE: SSE data chunk, delta content (server.mjs:540,557)
        end
        Cc-->>B: result event, final text or exit code (server.mjs:344-358)
        B->>B: pushTurn(voice-reply, full text) (server.mjs:566)
        B-->>UBE: SSE data DONE (server.mjs:570)
    end
    UBE-->>UFE: synthesised speech, TTS
Note over B: DIVERGENCE — port 8444 and port 8443 are published 0.0.0.0 by<br/>docker-compose.voice.yml:38-39, while only port 9096 is the ADR-045 D2 sanctioned NIP-98-gated LAN<br/>door covered by the loopback CI gate. The Unmute voice loop itself reaches tab0-bridge only<br/>over the internal visionclaw_network hostname agentbox:8971, which is never host-published<br/>(docker-compose.yml:33,153-155)
Note over Cad,B: RESOLVED ADR-2047 — voice/README.md now routes /feed and /bridge/* to<br/>agentbox:9096 in both its route table and its ASCII map, naming the ADR-069 server-side<br/>BRIDGE_TOKEN credential exchange (voice/README.md:26,117-118). The remaining 8971 reference<br/>above is correct — it is this Unmute backend calling the bridge container-to-container, not a<br/>browser path through Caddy (voice/README.md:37)
Note over Cc: lean headless surface (measured 2.1.280/haiku: 38.4k-token/9.3s prefix → 6.8k/2.7s<br/>with MCP, hooks, skills and CLAUDE.md stripped) — CHILD_ENV.MAX_THINKING_TOKENS=0 and<br/>CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 set once at boot, see AB-12.2 (server.mjs:131-133)
```

## AB-12.8 mgmt-api voice-intent — mandate-gated ACSP dispatch
```mermaid
sequenceDiagram
    autonumber
    participant Ca as REST caller
    participant M as management-api<br/>routes/voice-intent.js:85
    participant VI as lib/voice-intent.js<br/>parseIntent:112
    participant Ma as lib/mandate.js<br/>see AB-11.10
    participant ACS as agent-control-surface.js<br/>buildActionRequest:176
    participant D as dispatchActionRequest<br/>server.js:902

Note over Ca,M: SCOPE — this route is not reached from the tab0-bridge cockpit or the Unmute<br/>voice loop, grep confirmed no reference to voice-intent.js under config/tab0-bridge. It is an<br/>independent management-api REST surface, included because the brief named it as an entry point.
    Ca->>M: POST /v1/voice-intent, transcript, actor_did, mandate (routes/voice-intent.js:85-112)
    M->>M: verifyAgentEventRequest(request), see AB-11.12 (routes/voice-intent.js:155)
    alt speaker auth not ok
        M-->>Ca: reply auth.status, error (routes/voice-intent.js:156-158)
    end
    alt mandate missing or not an object
        M-->>Ca: 403, mandate-required (routes/voice-intent.js:164-169)
    end
    M->>Ma: recordFromSignedMandate(mandate), see AB-11.10 (routes/voice-intent.js:172)
    alt recordFromSignedMandate throws
        M-->>Ca: 403, mandate-invalid (routes/voice-intent.js:174)
    end
    M->>M: verifyMandateEvent(mandate), Schnorr verifyEvent (routes/voice-intent.js:64-71,176)
    alt signature does not verify
        M-->>Ca: 403, mandate-unverified (routes/voice-intent.js:176-181)
    end
    M->>Ma: isMandateActive(mandateRecord) (routes/voice-intent.js:182)
    alt mandate revoked or expired
        M-->>Ca: 403, mandate-inactive (routes/voice-intent.js:183-186)
    end
    opt auth.did is present
        M->>Ma: reconcileSourceUrn(mandateRecord.agent, auth.did) (routes/voice-intent.js:192)
        alt grantee does not match verified speaker
            M-->>Ca: 403, mandate-speaker-mismatch (routes/voice-intent.js:193-198)
        end
    end
    M->>Ma: normalisePubkey(actor_did), validate target principal (routes/voice-intent.js:205)
    alt actor_did invalid
        M-->>Ca: 400, actor_did-invalid (routes/voice-intent.js:206-211)
    end
    M->>VI: transcriptToAction(transcript, actorRef, duration_ms) (routes/voice-intent.js:217)
    VI->>VI: parseIntent matches RULES — link, transform, delete, update, create, query, in order (lib/voice-intent.js:52-141)
    Note over VI: unrecognised utterance falls back to a read-only query, action_type QUERY, B3 fail-safe (lib/voice-intent.js:132-141)
    VI-->>M: intent — verb, action_type, subject, object, recognised
    M->>M: interpret(intent), the Jev option, resolves null unless AGENTBOX_JEV_VOICE_INTENT=1 (routes/voice-intent.js:223-227)
    alt interpretation.clarify is set
        M-->>Ca: 422, clarification-needed, nothing dispatched (routes/voice-intent.js:228-233)
    end
    M->>ACS: buildActionRequest(panelId, priority high, category voice-intent) (agent-control-surface.js:176-196, routes/voice-intent.js:244-265)
    ACS-->>M: unsigned kind-31402 event, d-tag panelId, p-tag actorPubkey
    alt no dispatchActionRequest wired
        M-->>Ca: 503, dispatch-unavailable (routes/voice-intent.js:237-242)
    end
    M->>D: dispatchActionRequest(unsigned), see AB-11.x for the authority gate (routes/voice-intent.js:269)
    alt dispatch throws
        M-->>Ca: 503, dispatch-failed (routes/voice-intent.js:270-276)
    end
    alt signedRequest has no id
        M-->>Ca: 503, dispatch-unsigned (routes/voice-intent.js:277-282)
    end
    D-->>M: signedRequest with id
    M->>M: emitAgentAction, beam-parity notification (routes/voice-intent.js:286-297)
    M-->>Ca: 200, dispatched true, speaker_did, actor_did, event_id, intent, dispatch, plus interpretation when the Jev option answered (routes/voice-intent.js:305-329)
    Note over M,VI: INVARIANT — the Jev option runs only after the speaker, mandate and target checks, and its sole power<br/>over the outcome is to abstain: it never picks a verb, subject or target, never signs or dispatches<br/>(routes/voice-intent.js:223-226, lib/voice-intent-jev.js:17-23). It is local-only and fails to null,<br/>so a miss leaves the route behaving as before (lib/voice-intent-jev.js:25-28)
Note over ACS: SCOPE — agent-control-surface.js mints unsigned NIP-33 events (kinds 31400-31405)<br/>consumed by the EXTERNAL nostr-rust-forum forum-client's GovernancePage, which<br/>dreamlab-ai-website shallow-clones at build time (agent-control-surface.js:4-19) — it is not an<br/>agentbox-native operator dashboard. PANEL_SCHEMAS, LAYOUT_HINTS and ACTION_PRIORITIES are frozen<br/>module constants (agent-control-surface.js:43,46,48) — publishPanelEvent is a thin delegate over<br/>an ALREADY-CONNECTED NostrBridge, no in-request relay I/O (agent-control-surface.js:238)
```

## AB-12.9 Inline NIP-98 approval decision
```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator, port 8444 cockpit
    participant Cad as Caddy port 8444
    participant Pr as nip98-proxy port 9096<br/>see AB-10.x
    participant M as management-api<br/>routes/approvals.js:55
    participant Az as lib/authz.js<br/>isApprover
    participant Cs as authority consumer<br/>signAndPublishDecision

Op->>Cad: GET /approvals/*, NIP-98 kind-27235 via window.nostr, or<br/>break-glass bearer (Caddyfile handle /approvals/*,<br/>voice/README.md:96-103)
Cad->>Pr: reverse_proxy agentbox:9096, Authorization forwarded<br/>(Caddyfile handle /approvals/*)
    Pr->>Pr: verify NIP-98 or session cookie, see AB-10.3, AB-10.6
Pr->>M: GET /v1/approvals, strip prefix, route table<br/>(routes/approvals.js:55)
    alt authority consumer not wired
M-->>Op: 200, approvals empty, wired false, note<br/>(routes/approvals.js:88-90)
    end
    M->>M: annotateApprovals(c.listPending()), Jev option adds an advisory review_aid only when AGENTBOX_JEV_APPROVAL_REVIEW=1 (routes/approvals.js:47,91-94)
    Note over M: INVARIANT — the review aid never reorders or hides a request and the decide route does not consult it<br/>(routes/approvals.js:91-93, lib/approval-review-jev.js:16-18)
M-->>Op: 200, approvals array, count, wired true<br/>(routes/approvals.js:95)
Op->>Op: operator reviews the list, signs a kind-27235 NIP-98 header via<br/>window.nostr for the decide POST
Op->>Cad: POST /approvals/:id/decide, body outcome or decision,<br/>reasoning
    Cad->>Pr: reverse_proxy agentbox:9096, Authorization forwarded
    Pr->>M: POST /v1/approvals/:id/decide (routes/approvals.js:99)
    alt request.auth.mode is not nip98
        M-->>Op: 401, nip98_required (routes/approvals.js:139-144)
    end
M->>Az: isApprover(request.auth.pubkey, manifest)<br/>(routes/approvals.js:150)
    alt pubkey not on the approval allowlist
        M-->>Op: 403, forbidden_not_approver (routes/approvals.js:155-159)
    end
    alt authority consumer unwired
        M-->>Op: 503, authority_consumer_unwired (routes/approvals.js:162-166)
    end
    M->>Cs: isDecided(id) (routes/approvals.js:174)
    alt already decided
        Cs-->>M: true, plus prior outcome
M-->>Op: 409, already_decided, request_event_id, outcome,<br/>response_event_id (routes/approvals.js:175-182)
    end
    M->>Cs: getPending(id) (routes/approvals.js:184)
    alt no pending request with that id
        M-->>Op: 404, unknown_request (routes/approvals.js:185-189)
    end
M->>M: normalise outcome, deny maps to reject<br/>(routes/approvals.js:193-200)
M->>Cs: signAndPublishDecision(requestId, outcome, reasoning)<br/>(routes/approvals.js:205)
    Cs-->>M: signed kind-31403 event id
M-->>Op: 200, success true, request_event_id, response_event_id,<br/>outcome, decided_by (routes/approvals.js:230-236)
Note over M: INVARIANT — the decision record is ALWAYS a Schnorr-signed<br/>kind-31403 event, never an unsigned approval, ADR-043 D4.7<br/>(routes/approvals.js:16-22)
Note over M,Cs: see AB-14.x for the governance approvals pipeline<br/>internals, signAndPublishDecision and the authority gate itself — not<br/>drawn here
```

## AB-12.10 AoE session board — cockpit proxy path and the bridge's own passthrough
```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator console
    participant Cad as Caddy port 8444
    participant Pr as nip98-proxy port 9096
    participant AoEd as AoE daemon port 9095
    participant B as tab0-bridge<br/>server.mjs:795

    rect rgb(235,245,255)
    Note over Op,AoEd: Part 1 — cockpit session board via the sole NIP-98 ingress, see AB-10.x for proxy verification internals
    Op->>Cad: GET /aoe/*, NIP-98 or nip07 session cookie (Caddyfile handle_path /aoe/*)
    Cad->>Pr: reverse_proxy agentbox:9096, Authorization forwarded
    Pr->>Pr: verifyNip98 or session cookie, X-Agentbox-Pubkey injected, see AB-10.3, AB-10.7
    Pr->>AoEd: forward, Authorization Bearer daemon token replaces browser credential, see AB-10.9 (proxy.mjs:1037-1045)
    AoEd-->>Pr: session list json
    Pr-->>Cad: response
    Cad-->>Op: session list rendered
    end
    rect rgb(255,245,235)
    Note over B,AoEd: Part 2 — tab0-bridge's own passthrough, used by the voice console feed, independent of the proxy path
    Op->>B: GET /aoe/sessions, Bearer BRIDGE_TOKEN or NIP-98, see AB-12.3 (server.mjs:795)
    B->>AoEd: aoeRequest GET /api/sessions?state=live, Authorization Bearer aoe-token from serve.url (server.mjs:799, 92-109)
    alt AoE responds non-200
        AoEd-->>B: non-200
        B-->>Op: 502, error aoe unavailable, status (server.mjs:800)
    end
    alt AoE unreachable, transport error
        B-->>Op: 502, error aoe unreachable, detail (server.mjs:802-803)
    end
    AoEd-->>B: 200, session list
    B-->>Op: 200, sessions array, coordinator aoeSessionId (server.mjs:801)
    end
```

## AB-12.11 junkiejarvis-agent.js listen, zone-key grants and reply flow
```mermaid
sequenceDiagram
    autonumber
    participant R as Nostr relay pool, via NostrBridge
    participant Ag as JunkieJarvisAgent<br/>lib/junkiejarvis-agent.js:705
    participant Llm as callLlm()<br/>lib/junkiejarvis-agent.js:442

Note over R,Ag: SCOPE — junkiejarvis-agent.js has no reference to tab0-bridge, tmux, AoE or<br/>port 8971, grep confirmed. An independent forum bot riding management-api's shared NostrBridge,<br/>included because the brief named it as an entry point.
    Ag->>R: bridge.subscribe, kinds 1059, filter p equals pubkey plus the optional zone-inbox pubkey from zoneKeys.loadInboxSk — carries gift-wrapped DMs, zone-key grants and epoch tickets (lib/junkiejarvis-agent.js:767-776)
    Ag->>R: bridge.subscribe, kinds 42, filter p equals pubkey, channel mentions (lib/junkiejarvis-agent.js:782-787)
    Ag->>Ag: _scheduleProfilePublish, setTimeout 2000 ms, then publish kind-0 profile (lib/junkiejarvis-agent.js:797)
    R-->>Ag: inbound event, kind 1059 or kind 42 (lib/junkiejarvis-agent.js:839-844)
    Ag->>Ag: _dedup(event.id), in-memory set capped at DEFAULT_DEDUP_CAP (lib/junkiejarvis-agent.js:750-759)
    alt already seen
        Ag->>Ag: drop event (lib/junkiejarvis-agent.js:842)
    end
    alt kind is 1059, gift wrap
        Ag->>Ag: zoneKeys.unwrapAny(wrap, signer.skBytes), recover rumor — this authenticates the seal author, unlike nip59.unwrapEvent (lib/junkiejarvis-agent.js:859-860)
        alt rumor.kind is KIND_ZONE_KEY_GRANT (21453) or KIND_EPOCH_TICKET (kind 1014, NIP-171)
            Ag->>Ag: _handleGrant(wrap) — key material, no dedup/backlog floor, a grant sent while down is still valid (lib/junkiejarvis-agent.js:870-871,907-919)
            Ag->>Ag: zoneKeys.unwrapAny(wrap, skBytes) re-opens the seal, so the sealer is known (lib/junkiejarvis-agent.js:902-905,914)
            Ag->>Ag: _acceptGrantWithRetry(opened, 0) — acceptGrant: a kind-21453 grant needs isAdmin(sealer), a kind-1014 ticket is validated against the one bound group pubkey with no admin check (lib/junkiejarvis-agent.js:927-931, lib/zone-keys.js:343,360,498-499)
            alt granted
                Ag->>Ag: store zone key, log zone+epoch only, never the secret
            else rejected
                Ag->>Ag: refuse, log res.error
            else retry, admin status unresolved (relay unreachable)
                Ag->>Ag: setTimeout per GRANT_RETRY_DELAYS_MS at this attempt, re-attempt, give up after the last delay
            end
            Note over Ag: never falls through to _handleDm — a grant is never treated as a DM message
        else rumor.kind is KIND_DM_RUMOR
            Ag->>Ag: _shouldIgnore(rumor.pubkey), _dedup(rumor.id), backlog floor (lib/junkiejarvis-agent.js:873-883)
            Ag->>Llm: callLlm(userText), brisk professional personality (lib/junkiejarvis-agent.js:887,442)
            Llm-->>Ag: reply text, or apology on outage, fail-open
            Ag->>R: _sendDm → sendGiftWrappedDm, nip59, to the asker (lib/junkiejarvis-agent.js:898,981-993)
        end
    else kind is 42, channel message
        Ag->>Ag: _shouldIgnore(pubkey), backlog floor (lib/junkiejarvis-agent.js:1001-1002)
        alt zoneKeys.hasZkTag(tags)
            Ag->>Ag: zoneKeys.readOutcome(event, _lookupZoneKey) (lib/junkiejarvis-agent.js:1004-1005,951-952)
            alt not decrypted, key missing or wrong
                Ag->>Ag: log outcome, skip — ciphertext never reaches the LLM (lib/junkiejarvis-agent.js:1006-1013)
            else decrypted
                Ag->>Ag: sealed original present — attribute to inner author, id, tags, not the migrator (lib/junkiejarvis-agent.js:1015-1023)
            end
        end
        Ag->>Ag: isChannelMention, p-tag or at-junkiejarvis text (lib/junkiejarvis-agent.js:146,1026)
        Ag->>Llm: callLlm(userText), brisk professional personality (lib/junkiejarvis-agent.js:1034,442)
        Llm-->>Ag: reply text, or apology on outage, fail-open
        Ag->>Ag: _replyZone(srcEvent, channelId) — zk tag first, else cached kind-40 section lookup (lib/junkiejarvis-agent.js:1068,962-978)
        Ag->>R: _sendChannelReply — e-tag root preserved, p-tag asker, zoneKeys.writePlan encrypts when the target zone is set (lib/junkiejarvis-agent.js:1039,1042-1076)
    end
    Ag->>Ag: truncateReply to maxReply, default 280 chars (lib/junkiejarvis-agent.js:267,1140-1141)
Note over Ag,R: INVARIANT ADR-2088 — sendGiftWrappedDm is the ONE gift-wrap site in the repo<br/>(lib/junkiejarvis-agent.js:654). _sendDm delegates to it so the nightly forum-suggestions<br/>tenant, which has a bridge and a signer but no agent instance, sends through the identical<br/>envelope instead of hand-rolling a second one
Note over Ag: hasSchedulingIntent may buildCalendarEvent, kind-31923 NIP-52, on behalf of forum members (lib/junkiejarvis-agent.js:371,1121)
```

## AB-12.13 turn-sink capture
```mermaid
sequenceDiagram
    autonumber
    participant Cc as Claude Code hook, Stop or UserPromptSubmit
    participant Sk as turn-sink.cjs<br/>agentbox/config/tab0-bridge/turn-sink.cjs:1
    participant B as tab0-bridge<br/>server.mjs:757, POST /hook/turn
    participant Su as summarise()<br/>server.mjs:362
    participant Fe as WebSocket /feed clients

    Cc->>Sk: hook invocation, argv[2] equals Stop or UserPromptSubmit, JSON on stdin (turn-sink.cjs:10,37-40)
    Sk->>Sk: cwd check, payload.cwd must start with /home/devuser/workspace/project (turn-sink.cjs:44-45)
    alt cwd does not match, tab0-bridge's own headless sessions excluded
        Sk-->>Cc: finish, no post (turn-sink.cjs:45,12-15)
    end
    alt event is UserPromptSubmit
        Sk->>Sk: text = payload.prompt (turn-sink.cjs:47)
    else event is Stop
        Sk->>Sk: text = lastAssistantText(transcript_path), scan transcript backwards for the last assistant text block (turn-sink.cjs:17-33,49)
    end
    alt text is null or empty
        Sk-->>Cc: finish, no post (turn-sink.cjs:53)
    end
    Sk->>B: POST /hook/turn, body event, text sliced to 20000 chars, timeout 1500 ms (turn-sink.cjs:55-60)
    B->>B: pushTurn(kind, text), kind from event name (server.mjs:757-761)
    B->>Fe: broadcast, type turn (server.mjs:154,158)
    alt kind is assistant and text length over 350 chars
        B->>Su: summarise(turn.text, kind), claude -p, one to three sentences (server.mjs:362-375,763)
        Su-->>B: summary text, or null on error, fail open
        B->>Fe: broadcast, type turn-update, turn with summary (server.mjs:764)
    end
    B-->>Sk: 200, ok true, id turn.id (server.mjs:767)
    Sk-->>Cc: Stop prints ok true json, UserPromptSubmit prints nothing, stdout would inject into context (turn-sink.cjs:2-3,12-15)
    Note over Sk: fail-open by design, any error still exits 0, 3000 ms safety timeout, unref'd (turn-sink.cjs:65)
    Note over B,Fe: boundary — the mobile bridge digests this same turn feed into a kind-30840 summary event, see AB-13.5, not drawn here
```

## AB-12.14 Read-only and outbound route family
```mermaid
sequenceDiagram
    autonumber
    participant Cl as Client
    participant B as tab0-bridge<br/>server.mjs:739
    participant T as tmux CLI
    participant Nf as ~/.claude/nostr-inbox files
    participant Sc as nostr-send.cjs

    Note over Cl,B: family of read-only and outbound surfaces sharing the global auth gate, see AB-12.3 — one sequence covers all members
    Cl->>B: GET /turns?n=50 (server.mjs:769-772)
    B-->>Cl: turns array, sliced to n, capped at MAX_TURNS 300 (server.mjs:770-771,57)
    Cl->>B: GET /tabs (server.mjs:792-793)
    B->>T: tmux list-windows -t agentbox -F index name active (server.mjs:170-176)
    T-->>B: window list
    B-->>Cl: tabs array
    Cl->>B: GET /tabs/:n?lines=60 (server.mjs:806-809)
    B->>T: tmux capture-pane -p -t agentbox:n -S -lines, capped at 200 (server.mjs:178-181,808)
    T-->>B: pane text, trailing whitespace stripped
    B-->>Cl: index, output
    Cl->>B: GET /nostr/status (server.mjs:777-778)
    B->>Nf: read gateway.lock pid, check pidAlive, check mirror-key.txt exists (server.mjs:588-600)
    B-->>Cl: gateway armed, stale-lock or off, mirrorKey bool, sendReady bool
    Cl->>B: GET /nostr/events?n=20 (server.mjs:780-783)
    B->>Nf: read commands.jsonl, tail n lines, capped at 100 (server.mjs:602-609,781)
    B-->>Cl: events array
    Cl->>B: POST /nostr/send, body text (server.mjs:784-791)
    alt text empty after trim and slice 3500
        B-->>Cl: 400, error empty text (server.mjs:787)
    end
    B->>Sc: spawn node nostr-send.cjs text, 12000 ms kill timer (server.mjs:613-622)
    Sc-->>B: exit code 0, success, or non-zero
    B->>B: pushTurn(nostr-out, clean) when ok (server.mjs:789)
    B-->>Cl: ok boolean
    Note over Sc: nostr-send.cjs is fail-open, exit 0 even on delivery failure — ok true means handed to the relay path, not delivered (server.mjs:611-612)
```

## AB-12.15 Session creation today and the proposed session plane
```mermaid
flowchart TB
    subgraph LIVE["Live at the verified commit"]
        SEED["aoe-seed-sessions.mjs<br/>reads interaction_plane session_seeds<br/>scripts/aoe-seed-sessions.mjs:5,829-835"]
        SKIP["title already exists, log and skip<br/>DEBT: never reconciled, so a manifest change<br/>never reaches a live session<br/>scripts/aoe-seed-sessions.mjs:830-832"]
        AOE["AoE daemon port 9095<br/>behind nip98-proxy port 9096"]
        TB["tab0-bridge sendToTab0<br/>aoeSend first, tmux send-keys fallback<br/>config/tab0-bridge/server.mjs:273-293"]
        GW["nostr-gateway sendKeys<br/>text now, Enter scheduled 1500 ms later<br/>config/nostr-gateway/gateway.cjs:292-295"]
        TS["trust-seed.cjs<br/>root claude.json plus each profile that already has<br/>profiles slug .claude .claude.json<br/>config/hooks/trust-seed.cjs:78-86"]
    end
    subgraph PROP["PROPOSED, not live: ADR-2153 decision proposed, implementation none, inactive"]
        PL["services/agentbox-sessions daemon<br/>unix socket, no TCP listener<br/>docs/adr/ADR-2153-session-plane-replaces-aoe.md:32,40-42"]
        RC["reconcile seeds at boot and on a timer,<br/>recreate dead or drifted panes<br/>docs/adr/ADR-2153-session-plane-replaces-aoe.md:34-37"]
    end
    SEED --> SKIP
    SEED -->|create session| AOE
    TB -->|aoeSend| AOE
    GW -->|tmux| TMUX["tmux server, window per session"]
    TB -.->|fallback| TMUX
    TS -.->|writes folder trust| STORES["claude.json stores, root and per profile"]
    PL -.->|would replace| AOE
    PL --> RC
    TB -.->|would move to the socket| PL
    GW -.->|would move to the socket| PL
```
**What it shows.** How a session comes to exist and how text reaches it today: the seeder creates AoE sessions from the manifest seeds and never touches one that already exists, tab0-bridge and the nostr-gateway inject text through AoE or tmux, and trust-seed writes folder trust into the root store and into each wrapper profile store that already exists. The dashed lower box is ADR-2153, which proposes replacing AoE; it is drawn as a proposal only.
**Why it is this way.** The seeder was written as a create-only boot step (PRD-021 WS2, `scripts/aoe-seed-sessions.mjs:3-5`); ADR-2153 records the cost of that on 2026-10-28 and proposes an owned daemon that reconciles (`docs/adr/ADR-2153-session-plane-replaces-aoe.md:25,34-37`).

**Debt:** `scripts/aoe-seed-sessions.mjs:830-832` skips any session whose title already exists ("already exists — skipping (never killed)"), which is why a changed model or command in the manifest never reaches a live session (`docs/adr/ADR-2153-session-plane-replaces-aoe.md:25`).
**Open:** ADR-2153 is proposed with implementation none and activation inactive (`docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`), so nothing in it has been accepted: the ADR names a review trigger but no acceptance trigger, and until a decision is taken the AoE edges drawn in AB-12.5, AB-12.6 and AB-12.10 remain the live path (`docs/adr/ADR-2153-session-plane-replaces-aoe.md:13`)?
**Drift:** the seeder header calls the script a boot "reconciler" (`scripts/aoe-seed-sessions.mjs:3,5`) while its loop only creates and skips existing sessions (`scripts/aoe-seed-sessions.mjs:829-832`).
**Invariant:** trust-seed offers folder trust to a wrapper profile store only when `profiles/<slug>/.claude/.claude.json` already exists, so a profile without that file is never created or trusted by it (`config/hooks/trust-seed.cjs:78-86`).
**Tension (ADR-2151 vs manifest):** ADR-2151 (its pin commit 11dafbde9 landed after d9477245d held it proposed) records the VoiceLink edge voice slot as proposed, implementation none, inactive (`docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`), while the live manifest sets `[voicelink] enabled = true` and `record_events = true` (`agentbox.toml:2225,2232`); that path reaches the container by `voice-paned --stdio` over ssh and bypasses tab0-bridge and the Unmute loop (`agentbox.toml:2215-2219`).
