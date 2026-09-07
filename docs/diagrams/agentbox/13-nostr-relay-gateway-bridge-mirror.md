---
id: AB-13
title: Nostr — relay, gateway, pod bridge, session mirror
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/SECURITY-profiles.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2012, ADR-2025, ADR-2026, ADR-2061]
sources:
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/hooks/lib/egress-policy.cjs
  - ../project/agentbox/services/nostr-pod-bridge/src/main.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/bootstrap.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/lib.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/admission.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/session_summary.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/egress_policy.rs
  - ../project/agentbox/tests/fixtures/egress-redaction.v1.json
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/mcp/nostr-bridge/relay-consumer.js
  - ../project/agentbox/mcp/nostr-bridge/default-intent-spec.js
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/lib/bc20-provenance-bridge.js
  - ../project/agentbox/schema/federation-kinds.json
  - ../project/agentbox/docs/user/nostr-control-gateway.md
  - ../project/agentbox/docs/adr/ADR-2012-relay-allowlist-only-ingress.md
  - ../project/agentbox/docs/adr/ADR-2025-cross-repo-federation-contract.md
  - ../project/agentbox/docs/adr/ADR-2026-session-mirror-egress-boundary.md
  - ../project/agentbox/management-api/lib/governance-decision-waiter.js
  - ../project/agentbox/management-api/lib/llm-marketplace.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/management-api/lib/agent-control-surface.js
verified_commit: 2c521c5bb
---

## AB-13.1 Nostr topology — relay, gateway, pod bridge, mirror, mesh

```mermaid
flowchart TB
    subgraph lan["LAN / container boundary"]
        subgraph relayslot["relay slot [program:nostr-relay] flake.nix:2209-2239"]
            PB["nostr-pod-bridge daemon<br/>services/nostr-pod-bridge/src/main.rs:109 run_daemon<br/>embedded relay :7777 loopback (podBridgeEnabled=true, default)"]
            RS["nostr-rs-relay binary<br/>flake.nix:2230 else-branch (podBridgeEnabled=false only)"]
        end
        GW["nostr-gateway daemon<br/>config/nostr-gateway/gateway.cjs:743 connect()<br/>[program:nostr-gateway] flake.nix:1896"]
        MGMT["management-api RelayConsumer<br/>management-api/server.js:1324<br/>mcp/nostr-bridge/relay-consumer.js:85 (legacy JS consumer, still wired)"]
        AOE["AoE interaction plane :9095<br/>gateway.cjs:115-146 aoeRequest()"]
        TAB0["tab0-bridge :8971<br/>gateway.cjs:104,369 chatTab0()"]
    end
    subgraph cloud["Cloud egress boundary (the ONE external Nostr hop for mirror+control)"]
        CLOUD["dreamlab cloud worker relay<br/>wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev<br/>agentbox.toml:175 forum_relay_url"]
    end
    subgraph phone["Operator phone"]
        AME["Amethyst + Amber signer<br/>reads/writes the operator self-DM thread"]
    end
    MIRROR["nostr-live-mirror.cjs hook<br/>config/hooks/nostr-live-mirror.cjs:374 main()<br/>SessionStart/UserPromptSubmit/Stop/SessionEnd"]
    DIGEST["nostr-pod-bridge session-summary<br/>services/nostr-pod-bridge/src/session_summary.rs:362 run()"]
    ZAI["Z.AI / GLM summariser<br/>session_summary.rs:59 DEFAULT_ZAI_BASE"]
    FORUM["forum-backup-cron<br/>flake.nix:2455 [program:forum-backup-cron]<br/>supercronic + dreamlab-ai-website/scripts/backup/crontab (OUT OF TREE)"]
    MESH["peer agentbox relays<br/>agentbox.toml:227-236 [mesh]"]

    MIRROR -->|"kind 1059 gift wrap"| CLOUD
    GW <-->|"REQ #p=childkey / AUTH kind 22242"| CLOUD
    CLOUD <-->|"gift-wrapped DMs"| AME
    DIGEST -->|"kind 30840 sign+publish"| PB
    DIGEST -->|"POST transcript"| ZAI
    PB <-->|"ws://127.0.0.1:7777"| MGMT
    PB -->|"pods/&lt;npub&gt;/events/inbox/&lt;id&gt;.json"| MGMT
    GW -->|"Bearer token from serve.url"| AOE
    GW -->|"POST /tab0/send"| TAB0
    PB -.->|"federated_kinds (agentbox.toml:233), standalone by default"| MESH
    FORUM -.->|"Cloudflare API (not Nostr)"| CLOUD

N1["RESOLVED ADR-2065 (2026-09-05): the Rust spawn_consumer is the sole inbox writer when the<br/>pod-bridge daemon runs — RelayConsumer takes writeInbox=false via AGENTBOX_POD_INBOX_WRITER,<br/>projected from the same podBridgeEnabled expression that gates the daemon supervisor block.<br/>The JS consumer is narrowed, not deleted: it still solely implements ACSP governance 31400-31405,<br/>agent-intent 38000+, payments 38200/38201, the outbox publisher and external fanout"]
    N2["INVARIANT ADR-2012: relay ingress is allowlist-only, no fallback, no auto-add — allowed_pubkeys baked at nix build (relayAllowedPubkeysCsv, flake.nix:1461)"]
```

## AB-13.2 Relay ingress admission — allowlist gate before store/broadcast/OK

```mermaid
sequenceDiagram
    autonumber
    participant PUB as Remote publisher
    participant WS as serve_admitting_ws<br/>services/nostr-pod-bridge/src/lib.rs:790
    participant ADM as RelayAdmission.gate<br/>services/nostr-pod-bridge/src/admission.rs:436
    participant POL as PublisherPolicy.admit<br/>admission.rs:179
    participant AUD as AdmissionAudit.record<br/>admission.rs:279
    participant REL as Relay::ingest<br/>solid_pod_rs_nostr (dispatch_message_with_limits)

    PUB->>WS: ["EVENT", ev]
    WS->>ADM: gate(text) admission.rs:436
    ADM->>ADM: inspect_frame(policy, text) admission.rs:365
    ADM->>POL: admit(author) admission.rs:179
    alt author == self_pubkey
        POL-->>ADM: Admit(SelfAuthored) admission.rs:184-186
    else author in allowed set
        POL-->>ADM: Admit(AllowListed) admission.rs:187-188
    else allowlist non-empty, author not listed
        POL-->>ADM: Reject(NotAllowListed) admission.rs:190-193
    else allowlist EMPTY
        POL-->>ADM: Reject(DenyAllEmptyAllowlist) admission.rs:190-192
        Note over POL: INVARIANT ADR-2012 empty allowlist = deny-all for every remote author (admission.rs:35-45, agentbox.toml:144-153 no fallback no auto-add)
    end
    alt Admitted
        ADM-->>WS: None (proceed) admission.rs:451
        WS->>REL: dispatch_message_with_limits(relay, subs, text, limits) lib.rs:820-822
        REL-->>PUB: ["OK", id, true, ""]
        REL-->>WS: broadcast to live subscribers
    else Rejected
        ADM->>AUD: record(RelayAdmission, id, author, kind, reason) admission.rs:465-473
        ADM-->>WS: Some(["OK", id, false, reason.ok_message()]) admission.rs:406-408
        WS-->>PUB: negative OK — blocked: ... (NIP-20)
        Note over REL: event is NEVER verified-and-stored, NEVER broadcast, NEVER positively acked (lib.rs:751-757)
    end
Note over ADM,REL: DIVERGENCE (ADR-2012 closeout 2026-09-04): this gate closes the historical<br/>gap where the relay stored/broadcast/OK'd BEFORE the inbox consumer authorised —<br/>admission.rs:1-17 records that prior state and the fix. Two boundaries remain distinct:<br/>RelayAdmission (this diagram) and InboxAuthorisation (AB-13.6) — a relay OK is a transport ack,<br/>not an authorised commit (admission.rs:19-33)
```

## AB-13.3 Allowlist projection at nix build — relay implementation selection

```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml<br/>[sovereign_mesh.relay] :131-181
    participant NIX as flake.nix evaluation<br/>flake.nix:1323-1351
    participant CSV as relayAllowedPubkeysCsv<br/>flake.nix:1461
    participant TOMLGEN as relayAllowedPubkeysToml<br/>flake.nix:1462-1473
    participant SUP as supervisord generated text<br/>flake.nix:2209-2239
    participant PB as nostr-pod-bridge process<br/>services/nostr-pod-bridge/src/lib.rs:133 BridgeConfig::from_env

    NIX->>NIX: relayEnabled = relayCfg.enabled flake.nix:1323
    NIX->>NIX: relayLocal = relayEnabled and impl in {nostr-rs-relay, rnostr} flake.nix:1325
    NIX->>NIX: podBridgeEnabled = relayLocal and relayCfg.pod_bridge flake.nix:1351
    TOML->>NIX: allowed_pubkeys[] :144-153, pod_bridge=true :160
    NIX->>CSV: relayAllowedPubkeysCsv = concatStringsSep "," allowed_pubkeys flake.nix:1461
    alt podBridgeEnabled == true (default: pod_bridge = true)
        NIX->>SUP: [program:nostr-relay] command=nostr-pod-bridge flake.nix:2217-2228
        SUP->>PB: env AGENTBOX_ALLOWED_PUBKEYS=relayAllowedPubkeysCsv flake.nix:2221
        Note over TOMLGEN: relayConfigText / relayAllowedPubkeysToml is generated but UNUSED on this path (flake.nix:1456 "Unused on the pod_bridge path — the bridge is env-configured")
        PB->>PB: allowed_pubkeys = env.split(",").filter(nonempty) lib.rs:144-150
    else podBridgeEnabled == false (implementation=nostr-rs-relay, pod_bridge=false)
        NIX->>TOMLGEN: relayAllowedPubkeysToml — empty array emits explicit pubkey_whitelist = [ ] flake.nix:1462-1473
        Note over TOMLGEN: comment explains the omission bug — an omitted pubkey_whitelist accepts EVERY author, an explicit empty array is ADR-2012 deny-all (flake.nix:1462-1469)
        NIX->>SUP: [program:nostr-relay] command=nostr-rs-relay --config /etc/agentbox/nostr-relay.toml flake.nix:2229-2238
    end
Note over TOML,PB: no runtime mutation path — no auto-add, no fallback (admission.rs:139-141).<br/>Changing allowed_pubkeys requires ./agentbox.sh rebuild (Nix build-time artefact, ADR-2012<br/>Consequences)
```

## AB-13.4 Event-kind map part 1 — transport, auth and reference kinds

```mermaid
classDiagram
    class Kind1059_GiftWrap {
        kind = 1059
        producer nostr-live-mirror.cjs:408 nip59.wrapEvent
        producer gateway.cjs:279 buildWrap
        consumer lib.rs:266 effective_message unwrap_gift
        consumer gateway.cjs:705 handleWrap
        signer mirror child key HMAC derived, nostr-live-mirror.cjs:211
    }
    class Kind14_DmRumor {
        kind = 14 NIP-17 rumor inside the gift wrap
        producer nostr-live-mirror.cjs:398 rumor
        producer gateway.cjs:278 rumor
        consumer nostr-pod-bridge unwrap_gift lib.rs:268
        signer sealed sender see AB-13.9
    }
    class Kind22242_NIP42Auth {
        kind = 22242 KIND_AUTH relay session AUTH
        producer gateway.cjs:667 authenticate finalizeEvent
        consumer cloud relay and embedded relay AUTH check
        signer operator or derived child key gateway.cjs:176-179
    }
    class Kind27235_NIP98 {
        kind = 27235 nostr-bridge.js:55 kinds.AUTH
        producer nostr-bridge.js verifyNip98 callers see AB-10.4
        consumer management-api HTTP auth middleware see AB-10.4
        note see AB-10.4 for full verification path
    }
    class Kind30078_AgentState {
        kind = 30078 nostr-bridge.js:57 AGENT_STATE
        producer see AB-17.x agent-event publishing
        consumer nostr-bridge.js:290 default subscribeKinds
    }
    class Kind30000_30001_Refs {
        kind = 30000 BRIEF_REF nostr-bridge.js:58
        kind = 30001 BEAD_REF nostr-bridge.js:59
        federated agentbox.toml:233 federated_kinds
    }
    class Kind30910_Invite {
        kind = 30910 NIP-58 invite agentbox.toml multi_user invite_kind
        federated agentbox.toml:233
    }
    Kind1059_GiftWrap --> Kind14_DmRumor : unwraps to
    note for Kind27235_NIP98 "identity minting and the DID behind every signer on this page is AB-11.2. Full NIP-98 verification is AB-10.4."
```

## AB-13.15 Event-kind map part 2 — session record and ACSP governance kinds

```mermaid
classDiagram
    class Kind30840_SessionSummary {
        kind = 30840 KIND_SESSION_SUMMARY lib.rs:95
        producer publish_session_summary lib.rs:484
        consumer process_event session_path lib.rs:386-391
        signer agent recipient_sk lib.rs:488 signing_key_from_bytes
    }
    class Kind30841_ProjectTracking {
        kind = 30841 KIND_PROJECT_TRACKING lib.rs:100
        producer publish_project_tracking lib.rs:618
        consumer process_event projects_path lib.rs:392-397
        signer agent recipient_sk lib.rs:622
    }
    class Kind31400_31405_ACSP {
        kind range 31400-31405 nostr-bridge.js:61-66
        PANEL_DEFINITION PANEL_STATE ACTION_REQUEST ACTION_RESPONSE PANEL_UPDATE PANEL_RETIRED
        producer agentbox governance publisher outbound
        consumer relay-consumer.js:78-79 GOVERNANCE_KIND_MIN_MAX _isGovernanceEvent
        consumer relay-consumer.js:603 _writeGovernanceEvent
        sink governance-decision-waiter server.js:1339
    }
    note for Kind31400_31405_ACSP "the ACSP producer/consumer split and the decision loop are AB-11.10 and AB-11.11.<br/>agent-control-surface.js builds these kinds for the external forum client — it is not an agentbox dashboard. see AB-12.12"
```

## AB-13.16 Event-kind map part 3 — federation, job and marketplace kinds

```mermaid
classDiagram
    class Kind38000_38099_AgentIntent {
        kind range 38000-38099 relay-consumer.js:66-67
        producer VisionClaw voice-origin ActionRequest default-intent-spec.js:7
        consumer relay-consumer.js:586 _isAgentIntent
        consumer relay-consumer.js:454 _writeIntentMarker
        dispatch default-intent-spec.js:71 defaultIntentSpec when AGENTBOX_INTENT_COMMAND set
    }
    class Kind38100_38199_AgentResponse {
        kind range 38100-38199 relay-consumer.js:68-69 AGENT_RESPONSE_MIN_MAX
    }
    class Kind38200_38201_Jobs {
        kind = 38200 JOB_ESTIMATE nostr-bridge.js:67
        kind = 38201 JOB_SETTLEMENT nostr-bridge.js:68
        producer nostr-bridge.js:641 publishJobEstimate
        producer nostr-bridge.js:676 publishJobSettlement
        consumer relay-consumer.js:642 _writePaymentEvent
    }
    class Kind38300_38305_LLMMarketplace {
        kind = 38300 Advertisement management-api/lib/llm-marketplace.js:25
        kind = 38301 Request :26
        kind = 38302 Grant :27
        kind = 38303 Deny :28 non-federated point-to-point
        kind = 38304 Receipt :29
        kind = 38305 Revocation :30 non-federated point-to-point
        federated agentbox.toml:286-287 38300 38301 38302 38304 only
    }
    Kind38000_38099_AgentIntent --> Kind38100_38199_AgentResponse : responder replies with
    note for Kind38200_38201_Jobs "job estimate and settlement share the ACSP decision sink in AB-13.15 — see AB-11.10 for the gate that consumes it"
    note for Kind38000_38099_AgentIntent "agent-event publishing and the BC20 provenance kind map are AB-17.1 and AB-17.4 —<br/>this class covers only the relay-consumer dispatch path"
    note for Kind38300_38305_LLMMarketplace "DIVERGENCE — 38303 Deny and 38305 Revocation are deliberately NOT federated (point-to-point only),<br/>so a peer that federates the other four kinds never learns of a denial or a revocation. see AB-15"
```

## AB-13.5 nostr-gateway command flow — relay subscribe to reply

```mermaid
sequenceDiagram
    autonumber
    participant CLOUD as cloud relay<br/>gateway.cjs:84 DEFAULT_RELAY
    participant CONN as connect<br/>config/nostr-gateway/gateway.cjs:743
    participant ONM as onMessage<br/>gateway.cjs:730
    participant HW as handleWrap<br/>gateway.cjs:702
    participant DISP as dispatch<br/>gateway.cjs:286
    participant AOE as aoeRequest<br/>gateway.cjs:431
    participant TOK as readAoeToken<br/>gateway.cjs:127
    participant TMUX as tmux fleet<br/>gateway.cjs:246-274

    CONN->>CLOUD: new WS(relayUrl) gateway.cjs:744
    CLOUD-->>CONN: AUTH challenge
    CONN->>CLOUD: ["AUTH", finalizeEvent(kind 22242)] gateway.cjs:667-668
    CONN->>CLOUD: ["REQ","ctrl",{kinds:[1059],#p:[pub],since:now-50h}] gateway.cjs:664
    CLOUD-->>ONM: EOSE
    ONM->>ONM: armed = true gateway.cjs:737
    CLOUD-->>ONM: ["EVENT", wrap]
    ONM->>HW: handleWrap(ws, wrap) gateway.cjs:735
    HW->>HW: nip59.unwrapEvent(wrap, sk) gateway.cjs:705
    alt sealed sender != commanderPub
        HW-->>HW: dropped — only operator may command gateway.cjs:708
    else not armed (cold-boot backlog)
        HW-->>HW: skipped, backlog message gateway.cjs:721
    else replay — wrap.id in executed.ids
        HW-->>HW: skipped, replayed message gateway.cjs:722
    else stale — age > CMD_FRESH_WINDOW (600s)
        HW-->>HW: skipped, stale cmd gateway.cjs:723-724
    else fresh authorised command
        HW->>HW: recordExecuted(wrap.id) gateway.cjs:725
        HW->>DISP: dispatch(ws, text) when text starts with / gateway.cjs:727
        DISP->>DISP: verb = body.split(/space/)[0] gateway.cjs:288
        alt verb is tabs/peek/help
            DISP->>TMUX: capture-pane (zero tokens) gateway.cjs:246-253
        else verb is report
            DISP->>DISP: doReport spends one Sonnet call gateway.cjs:627
        else verb is spawn/cd
            DISP->>AOE: aoeCreateSession(repoPath, tool) gateway.cjs:452
            AOE->>TOK: readAoeToken() gateway.cjs:127-145
            TOK-->>AOE: Bearer token from ~/.config/agent-of-empires/serve.url
            AOE-->>DISP: session id status gateway.cjs:456-457
        else verb is tab/say/exit/quit
            DISP->>TMUX: sendKeys(idx, text) gateway.cjs:274
        else free-form instruction
            DISP->>DISP: routeInstruction — one bounded Sonnet C2 call gateway.cjs:562-615
        end
        DISP->>CLOUD: reply(ws, text) buildWrap + nip59.wrapEvent gateway.cjs:277-281
    end
Note over HW: replay guard ordering (gateway.cjs:14-38): 1 relay AUTH, 2 sealed sender == child<br/>pubkey, 3 arm-after-EOSE, 4 durable executed.json, 5 CMD_FRESH_WINDOW=600s freshness, 6 grammar<br/>(leading slash)
    Note over CLOUD,TMUX: rect boundary — this whole sequence runs LAN-side except the cloud relay hop. See AB-13.13 for the connection lifecycle state machine
```

## AB-13.6 nostr-pod-bridge inbox write — authorise, unwrap, persist

```mermaid
sequenceDiagram
    autonumber
    participant REL as Relay broadcast channel<br/>services/nostr-pod-bridge/src/lib.rs:702 relay.subscribe
    participant CONS as spawn_consumer loop<br/>lib.rs:697-747
    participant PROC as process_event<br/>lib.rs:370
    participant AUTHZ as authorize<br/>lib.rs:227
    participant EFF as effective_message<br/>lib.rs:254
    participant ADDR as addressed_to<br/>lib.rs:282
    participant WRITE as write_json<br/>lib.rs:356
    participant ADM as RelayAdmission.note_inbox_rejection<br/>admission.rs:482

    REL-->>CONS: rx.recv() event lib.rs:706
    alt ev.pubkey == cfg.recipient_pubkey (self-authored)
        CONS-->>CONS: skip — egress already persisted lib.rs:713-716
    else remote event
        CONS->>PROC: process_event(ev, cfg) lib.rs:717
        PROC->>AUTHZ: authorize(ev, cfg) lib.rs:371
        alt author not in allowed_pubkeys
            AUTHZ-->>PROC: Err(Unauthorized) lib.rs:227-235
            PROC-->>CONS: Err(Unauthorized(reason))
            CONS->>ADM: note_inbox_rejection(id, pubkey, reason) lib.rs:729
Note over ADM: DIVERGENCE ADR-2012 closeout — the RELAY already admitted, stored, broadcast and<br/>OK'd this event (AB-13.2). The INBOX boundary refuses it independently, and this is the durable<br/>evidence that a relay OK is not an authorised commit (admission.rs:19-33, lib.rs:47-57)
        else authorised
            AUTHZ-->>PROC: Ok(Authz::Direct) lib.rs:228-229
            PROC->>EFF: effective_message(ev, cfg) lib.rs:373
            alt ev.kind == KIND_GIFT_WRAP (1059)
                EFF->>EFF: unwrap_gift(core, recipient_sk) lib.rs:267-268
            else plain event
                EFF-->>EFF: pass through unchanged lib.rs:256-263
            end
            EFF-->>PROC: EffectiveMessage{sender_pubkey,kind,tags,content}
            PROC->>ADDR: addressed_to(recipient, ev, msg.tags) lib.rs:375
            alt not addressed to this agent
                ADDR-->>PROC: false
                PROC-->>CONS: Err(NotAddressed) — skipped, debug-logged lib.rs:731-733
            else addressed
                PROC->>PROC: format_as_ldn(ev, msg) lib.rs:379
                PROC->>WRITE: write_json(inbox_path) lib.rs:380-384
                alt msg.kind == KIND_SESSION_SUMMARY (30840)
                    PROC->>WRITE: write_json(session_path) lib.rs:386-391
                else msg.kind == KIND_PROJECT_TRACKING (30841)
                    PROC->>WRITE: write_json(projects_path) lib.rs:392-397
                end
                PROC-->>CONS: Ok(())
            end
        end
    end
Note over CONS,WRITE: DIVERGENCE — a SECOND, independent JS consumer<br/>(mcp/nostr-bridge/relay-consumer.js:279 _onInbound, wired at<br/>management-api/server.js:1280-1356) subscribes to the SAME relay and writes to the SAME<br/>pods/NPUB/events/inbox/ path with its own allowlist (AGENTBOX_RELAY_ALLOWED_PUBKEYS) and its<br/>own I01-I10 invariants (relay-consumer.js:39-46), independently of BridgeConfig.allowed_pubkeys<br/>here
```

## AB-13.7 nostr-bridge / relay-consumer — in-process library, not an MCP tool server

```mermaid
sequenceDiagram
    autonumber
    participant BOOT as management-api boot<br/>management-api/server.js:1280
    participant RC as RelayConsumer.start<br/>mcp/nostr-bridge/relay-consumer.js:224
    participant NB as NostrBridge<br/>mcp/servers/nostr-bridge.js:269
    participant CONN as RelayConnection<br/>mcp/servers/nostr-bridge.js:131
    participant SPEC as buildDefaultIntentSpec<br/>mcp/nostr-bridge/default-intent-spec.js:60
    participant GDW as governance-decision-waiter<br/>management-api/lib/governance-decision-waiter.js

Note over BOOT,GDW: CORRECTION — despite the path mcp/servers/nostr-bridge.js, this file's own<br/>header (lines 1-15) declares it library-only, consumed in-process by management-api. There is<br/>NO supervisord [program:nostr-bridge] and NO MCP tool schema (no tool()/registerTool calls) in<br/>either file — this sequence draws the real in-process call chain, not an MCP tool invocation
    BOOT->>BOOT: if AGENTBOX_RELAY_ENABLED and AGENTBOX_RELAY_POD_BRIDGE server.js:1280-1281
    BOOT->>SPEC: buildDefaultIntentSpec() server.js:1299
    alt AGENTBOX_INTENT_COMMAND unset
        SPEC-->>BOOT: null — marker-only path unchanged default-intent-spec.js:62-63
    else command configured
        SPEC-->>BOOT: defaultIntentSpec(event, context) function default-intent-spec.js:71-88
    end
    BOOT->>RC: new RelayConsumer({npubs, allowedPubkeys, intentSpec, governanceDecisionSink: GDW}) server.js:1324-1341
    BOOT->>RC: await consumer.start() server.js:1342
    RC->>NB: this._bridge.connect() relay-consumer.js:225
    NB->>CONN: conn.connect() for each relay in NOSTR_RELAYS mcp/servers/nostr-bridge.js:341-344
    RC->>NB: this._bridge.subscribe({kinds: allowedKinds}, onInbound) relay-consumer.js:226-229
    RC->>RC: _ensureMailboxDirs() relay-consumer.js:230
    RC->>RC: setInterval(_flushOutbox, 500ms) relay-consumer.js:231-233 DEFAULT_OUTBOX_POLL_MS
    loop every 500ms
        RC->>RC: _flushOutbox scans pods/*/events/outbox/*.json relay-consumer.js:678-694
        RC->>NB: sign + publish pending outbox entries relay-consumer.js:697-751
    end
    NB-->>RC: onInbound(event, relayUrl) relay-consumer.js:228
    RC->>RC: _verifySig(event) I01 relay-consumer.js:281,427
    RC->>RC: _passesIngressPolicy(event) I07 relay-consumer.js:288,444
    RC->>RC: _findRecipientNpub(event) I10 relay-consumer.js:507
    alt kind in 38000-38099 (agent-intent) and intentSpec present
        RC->>SPEC: intentSpec(event, context) relay-consumer.js referencing default-intent-spec.js:71
        SPEC-->>RC: {command, args, env with AGENTBOX_INTENT_SOURCE_URN} default-intent-spec.js:74-85
    else kind in 31400-31405 (governance) and inbound is 31403
RC->>GDW: governanceDecisionSink.notify(...) server.js:1339,<br/>relay-consumer.js:603
    end
Note over NB,CONN: subscription keepalive — CloudFlare Durable Object relays<br/>stop pushing to an<br/>idle REQ after ~20s regardless of socket liveness, so subRefreshMs=15000<br/>(mcp/servers/nostr-bridge.js:326) reissues every active subscription under a<br/>FRESH wire id,<br/>independent of reconnects (junkiejarvis "answers then goes quiet" regression,<br/>mcp/servers/nostr-bridge.js:301-325)
```

## AB-13.8 Control gateway — operator command to handler mapping

```mermaid
sequenceDiagram
    autonumber
    participant DOC as nostr-control-gateway.md<br/>agentbox/docs/user/nostr-control-gateway.md
    participant DISP as dispatch<br/>config/nostr-gateway/gateway.cjs:286

Note over DOC,DISP: doc Commands table (nostr-control-gateway.md:56-73) lists<br/>tabs, report, report n, report question, peek, help, free-form instruction,<br/>tab n text, say text
    DISP->>DISP: verb == help or empty -> reply(HELP) gateway.cjs:292
    DISP->>DISP: verb == tabs -> listTabs() gateway.cjs:293
    DISP->>DISP: verb == report -> doReport(ws, after) gateway.cjs:294, 627
    DISP->>DISP: verb == peek -> capture(idx, k) gateway.cjs:295-301
DISP->>DISP: verb == tab -> doSend(ws, idx, instr, explicit /tab)<br/>gateway.cjs:302-308
DISP->>DISP: verb == say -> broadcast sendKeys to every agentWindows<br/>gateway.cjs:309-318
DISP->>DISP: free-form (no matching verb) -> routeInstruction(ws, body)<br/>gateway.cjs:339, 562
DISP->>DISP: verb == spawn or cd -> doSpawn(ws, dir, agent, rest)<br/>gateway.cjs:319-331, 486
    DISP->>DISP: verb == exit or quit -> doExit(ws, idx) gateway.cjs:333-337, 540
Note over DOC,DISP: DOC-DRIFT — nostr-control-gateway.md Commands tables (Ask<br/>:58-65, Instruct<br/>:69-73) list only tabs, report, report n, report question, peek, help,<br/>free-form instruction,<br/>tab n text, say text. The doc omits /spawn and /exit and /quit, which ARE implemented<br/>(gateway.cjs:319-337, HELP text gateway.cjs:234-237) and are even mentioned<br/>later in the doc<br/>prose under Lifecycle (nostr-control-gateway.md:16) but never tabulated as Commands
Note over DISP: gate order enforced before dispatch is reached (AB-13.5) —<br/>relay AUTH, sealed<br/>sender == commanderPub, arm-after-EOSE, durable executed.json,<br/>CMD_FRESH_WINDOW=600s, leading<br/>slash grammar (gateway.cjs:14-38)
```

## AB-13.9 Session-mirror egress — per-turn NIP-59 gift wrap to the cloud relay

```mermaid
sequenceDiagram
    autonumber
    participant HOOK as Claude Code hook event<br/>SessionStart/UserPromptSubmit/Stop/SessionEnd
    participant MAIN as main<br/>config/hooks/nostr-live-mirror.cjs:374
    participant POL as egress-policy.cjs<br/>config/hooks/lib/egress-policy.cjs:136
    participant BODY as bodyForEvent<br/>nostr-live-mirror.cjs:271
    participant URI as mintActivityUrn<br/>nostr-live-mirror.cjs:163
    participant KEY as deriveChildKey<br/>nostr-live-mirror.cjs:208
    participant WRAP as nip59.wrapEvent<br/>nostr-live-mirror.cjs:465

    rect rgb(230, 245, 230)
Note over HOOK,MAIN: LAN process boundary — the hook itself never touches the<br/>network except the final PUB step
    HOOK->>MAIN: argv[2]=event, hook JSON on stdin nostr-live-mirror.cjs:376
    MAIN->>KEY: deriveChildKey() nostr-live-mirror.cjs:383
    MAIN->>POL: egressDecision('live-mirror', {identityPresent}) nostr-live-mirror.cjs:385-387
    alt AGENTBOX_EGRESS or AGENTBOX_LIVE_MIRROR is 0, or no identity
        POL-->>MAIN: allowed false, outcome skipped, reason egress-globally-disabled\|live-mirror-disabled\|no-sender-identity egress-policy.cjs:142-156
        MAIN-->>HOOK: return 0 — logs the specific outcome+reason, not a bare exit nostr-live-mirror.cjs:388-392
    else gated conditions pass
        MAIN->>BODY: bodyForEvent(event, payload) nostr-live-mirror.cjs:401
BODY-->>MAIN: session line text<br/>(SessionStart/UserPromptSubmit/Stop/SessionEnd) nostr-live-mirror.cjs:274-290
        MAIN->>POL: redactForEgress(body) — RESOLVED ADR-2026, redaction BEFORE composition,<br/>before the wrap, before any transport nostr-live-mirror.cjs:409, egress-policy.cjs:115-124
        alt redaction throws or input not a string
            POL-->>MAIN: null — fail-closed, "nothing is sent" nostr-live-mirror.cjs:410-413
        end
        MAIN->>URI: mintActivityUrn(uris, payload) nostr-live-mirror.cjs:419
URI-->>MAIN: urn:agentbox:activity:PUBKEY:sha256-12-HASH or empty (fail-open)<br/>nostr-live-mirror.cjs:163-174
MAIN->>MAIN: composeBody(redactedBody, urn) — urn NEVER truncated, cap<br/>MAX_BODY_CHARS=4000 nostr-live-mirror.cjs:420,181-196
MAIN->>POL: egressDecision('live-mirror', {recipient, identityPresent:true}) — recipient<br/>allowlist check nostr-live-mirror.cjs:450, egress-policy.cjs:157-160
MAIN->>WRAP: nip59.wrapEvent(rumor kind 14, sk, recipient)<br/>nostr-live-mirror.cjs:465
Note over WRAP: RESOLVED ADR-2026 (config/hooks/lib/egress-policy.cjs) — the rumor content is<br/>REDACTED before composition and before wrapping (redactForEgress, regex rules cover<br/>Authorization/Bearer headers, KEY=value, --flag secrets, long hex/base64 runs). The same<br/>redaction contract is implemented for the Rust session-digest path in<br/>services/nostr-pod-bridge/src/egress_policy.rs and both are checked against one shared<br/>fixture, tests/fixtures/egress-redaction.v1.json — closing the historical DIVERGENCE
        MAIN->>MAIN: hand the wrap to publishWrap — see AB-13.17
    end
    end
Note over KEY: child_sk = HMAC-SHA256(operator_sk,<br/>AGENTBOX_MIRROR_KEY_TAG default agentbox-mirror-v1)<br/>nostr-live-mirror.cjs:200-215 — keeps the ROOT<br/>operator key off the phone
```

## AB-13.17 Session-mirror egress phase 2 — publish, deadline and fail-open

```mermaid
sequenceDiagram
    autonumber
    participant MAIN as main<br/>config/hooks/nostr-live-mirror.cjs:374
    participant PUB as publishWrap<br/>nostr-live-mirror.cjs:312
    participant CLOUD as cloud worker relay<br/>dreamlab-nostr-relay workers.dev
    participant AME as Amethyst (operator phone)

    rect rgb(255, 235, 235)
Note over MAIN,CLOUD: CLOUD EGRESS BOUNDARY — the only non-LAN<br/>hop in this domain
MAIN->>MAIN: log egress attempted, relay + wrap id — logged BEFORE transport so a network<br/>kill still leaves evidence bytes were handed over nostr-live-mirror.cjs:475
MAIN->>PUB: publishWrap(WS, mirrorRelay(), wrap,<br/>DEADLINE_MS=6000) nostr-live-mirror.cjs:312-356,477
    PUB->>CLOUD: ["EVENT", wrap]
    alt relay OK true
        CLOUD-->>PUB: ["OK", id, true]
        CLOUD-->>AME: gift-wrapped DM delivered to the<br/>child-key self-DM thread
        MAIN->>MAIN: log egress accepted nostr-live-mirror.cjs:478-479
    else relay rejects or timeout or network error
PUB-->>MAIN: resolves anyway (never rejects)<br/>nostr-live-mirror.cjs:312-356
Note over MAIN: fail-open — publish failure is logged as egress failed (a THIRD, distinct<br/>outcome from skipped/accepted, RESOLVED ADR-2026) and swallowed, hook still exits 0<br/>nostr-live-mirror.cjs:480-484
    end
    end
Note over MAIN: hard kill-switch guard — setTimeout(process.exit(0),<br/>DEADLINE_MS+1500) unref'd, so the hook process can<br/>never outlive its budget nostr-live-mirror.cjs:505-506
Note over MAIN,AME: RESOLVED ADR-2026 (config/hooks/lib/egress-policy.cjs +<br/>services/nostr-pod-bridge/src/egress_policy.rs) — the mirror and the kind-30840 digest<br/>(AB-13.10) now share ONE policy contract: a global AGENTBOX_EGRESS switch, per-path<br/>switches, mandatory redaction, a recipient allowlist and the skipped/attempted/accepted/failed<br/>outcome vocabulary, cross-checked by one fixture (tests/fixtures/egress-redaction.v1.json).<br/>The two implementations differ only in transport (WS gift-wrap here, Rust digest in AB-13.10)
```

## AB-13.10 kind-30840 session-summary digest — Z.AI distil, sign, dual-write

```mermaid
sequenceDiagram
    autonumber
    participant SE as SessionEnd hook payload
    participant RUN as session_summary::run<br/>services/nostr-pod-bridge/src/session_summary.rs:386
    participant POL as egress_policy<br/>services/nostr-pod-bridge/src/egress_policy.rs:77,428
    participant MIR as mirror<br/>session_summary.rs:348
    participant EXT as extract_transcript<br/>session_summary.rs:171
    participant ZAI as summarise_via_zai<br/>session_summary.rs:248
    participant PARSE as parse_json_object<br/>session_summary.rs:196
    participant DIG as build_digest<br/>session_summary.rs:329
    participant PUB as publish_session_summary<br/>services/nostr-pod-bridge/src/lib.rs:484

    SE->>RUN: nostr-pod-bridge session-summary, stdin JSON main.rs:83
    RUN->>RUN: identity_present = bridge_configured and has zai key<br/>session_summary.rs:397,85-98
    RUN->>POL: egress_decision(env, identity_present)<br/>session_summary.rs:398, egress_policy.rs:77-112
    alt not decision.allowed
        RUN-->>SE: log outcome+reason, return Ok(())<br/>session_summary.rs:399-411
    else allowed
        RUN->>MIR: mirror(env, session_id, path) session_summary.rs:424
        MIR->>EXT: extract_transcript(path) session_summary.rs:349,171-173
        EXT->>EXT: flatten JSONL to ROLE: text, trim MAX_TRANSCRIPT_CHARS<br/>session_summary.rs:149-168
        alt transcript empty
            MIR-->>RUN: Ok(false) session_summary.rs:350-352
        end
        MIR->>POL: redact_for_egress(transcript)<br/>session_summary.rs:360, egress_policy.rs:428
        alt redacted transcript empty
            MIR-->>RUN: log skipped, Ok(false) session_summary.rs:361-367
        end
        rect rgb(255, 235, 220)
        Note over ZAI: cloud egress boundary - the ONE external LLM hop here<br/>(session_summary.rs:34-36). Bytes are already redacted
        MIR->>ZAI: summarise_via_zai(env, redacted) session_summary.rs:374
        ZAI->>ZAI: POST z.ai messages, model glm-5.3, timeout 180s<br/>session_summary.rs:57,60,62,228-265
        ZAI-->>PARSE: anthropic_text then parse_json_object<br/>session_summary.rs:178-195,267
        PARSE-->>MIR: {summary, actions[], actionable_questions[]}
        end
        MIR->>DIG: build_digest(env, digest, session_id) session_summary.rs:375
        DIG->>DIG: mint_activity_urn - same scheme as live mirror<br/>session_summary.rs:304-322,340
        DIG-->>MIR: SessionSummary{...,activity_urn}
        MIR->>PUB: publish_session_summary lib.rs:484, session_summary.rs:378
        PUB->>PUB: sign_event kind 30840 lib.rs:488-502
        PUB->>PUB: write_json inbox+session dual write lib.rs:517-526
        PUB->>PUB: publish_to_relay best-effort, warn-only lib.rs:528-530,685
        PUB-->>MIR: Ok(())
        MIR-->>RUN: log accepted, Ok(true) session_summary.rs:379
    end
    Note over RUN: always Ok(()) - failures logged and swallowed session_summary.rs:424-434
Note over ZAI,PUB: RESOLVED ADR-2026 - see AB-13.9/AB-13.17's RESOLVED notes. This path still sends<br/>FLATTENED, now REDACTED transcript text to Z.AI before publication - a different content scope than<br/>the mirror's zero-hop seal, but both paths share one policy contract now (egress_policy.rs +<br/>egress-policy.cjs, checked against tests/fixtures/egress-redaction.v1.json)
```

## AB-13.11 forum-backup-cron — supervised schedule (script out of tree)

```mermaid
flowchart LR
    SUP["supervisord [program:forum-backup-cron]<br/>agentbox/flake.nix:2455"]
    CRON["supercronic -split-logs<br/>flake.nix:2456"]
    SCRIPT["dreamlab-ai-website/scripts/backup/crontab<br/>OUT OF TREE — mounted sibling repo, not verifiable here"]
    CF["Cloudflare API<br/>CLOUDFLARE_API_TOKEN / ACCOUNT_ID"]
    NAS["NAS backup target"]
    SUP -->|"autostart, priority 250"| CRON
    CRON -->|"reads crontab, PATH pinned to coreutils/grep/findutils/curl/jq/gzip flake.nix:2458"| SCRIPT
    SCRIPT -->|"fails loud exit 2 if token/account id absent flake.nix:2452"| CF
    CF --> NAS
N1["DIVERGENCE: forum-backup-cron is a Cloudflare forum backup job, not a Nostr relay/kind<br/>path. It is included here only because the brief scoped 'forum backup' under this topic file.<br/>The script itself lives outside the agentbox tree (dreamlab-ai-website), so no fn-level<br/>citation is possible beyond the supervisor stanza"]
```

## AB-13.12 Federation — agentbox mesh peers vs the agentbox to VisionClaw URN bridge

```mermaid
sequenceDiagram
    autonumber
    participant TOML as agentbox.toml [mesh]<br/>agentbox.toml:227-236
    participant PEER as peer agentbox relay<br/>ws://peer:7777 (tailnet/cloudflare tunnel)
    participant BC20 as bc20-provenance-bridge<br/>management-api/lib/bc20-provenance-bridge.js
    participant VCU as VisionClaw src/uri minter<br/>see ES- estate side, not drawn here

    rect rgb(220, 235, 250)
    Note over TOML,PEER: PATH A — Nostr relay-to-relay federation between agentbox instances (mode=standalone by default, agentbox.toml:228)
    TOML->>TOML: federated_kinds = [1,1059,30001,30050,30078,30910,31400-31405,38000,38100,38300,38301,38302,38304] agentbox.toml:233
    alt mesh.mode == standalone (default)
        TOML-->>PEER: relay is loopback-only, no peer_relays configured agentbox.toml:228,232
    else mesh.mode == client
        TOML->>PEER: subscribe to subscribed_kinds subset, filtered by allowed_remote_dids agentbox.toml:234-235
    end
    Note over TOML: kinds 38303 (Deny) and 38305 (Revocation) are deliberately NON-federated, point-to-point only agentbox.toml:287
    end
    rect rgb(250, 235, 220)
    Note over BC20,VCU: PATH B — the agentbox to VisionClaw cross-repo contract (ADR-2025), an HTTP/URN grammar bridge, NOT a Nostr kind subscription
    BC20->>BC20: sha12(input) content-address truncation to 12 lowercase hex bc20-provenance-bridge.js:132
    BC20->>BC20: toVisionclaw(agentboxUrn) via the closed AGENTBOX_TO_VISIONCLAW kind map,<br/>DERIVED from schema/federation-kinds.json at require time (ADR-2061)<br/>bc20-provenance-bridge.js:112-127,158
    alt kind unmapped
        BC20-->>BC20: _countDrop + onDrop, dropped and logged (B04 closed map) bc20-provenance-bridge.js:182-183
    else kind == agent
        BC20-->>VCU: urn:agentbox:agent:PUBKEY:name -> did:nostr:PUBKEY (no URN kind, identity IS the key) bc20-provenance-bridge.js:169-179
    else content-addressed kind (execution, kg)
        BC20-->>VCU: urn:visionclaw:execution:sha12(agentboxUrn) bc20-provenance-bridge.js:190,189
    end
    BC20->>BC20: toAgentbox(visionclawId) reverse direction bc20-provenance-bridge.js:239
    Note over BC20: content-addressed reverse crossings need a durable UrnMapping store to recover the source urn:agentbox identity — onDrop otherwise bc20-provenance-bridge.js:288
    end
Note over TOML,VCU: DIVERGENCE — these are TWO SEPARATE federation mechanisms sharing the name<br/>federation. Path A moves Nostr EVENTS between agentbox relay peers by KIND NUMBER. Path B<br/>translates IDENTIFIER STRINGS between urn:agentbox and urn:visionclaw over HTTP, governed<br/>separately by ADR-2025 (decision_status proposed, activation_status inactive per its 2026-09-04<br/>closeout). Neither implements the other
```

## AB-13.13 nostr-gateway relay connection — subscribe, live, reconnect lifecycle

```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Connected: ws open, gateway.cjs:749 log connected
    Connected --> Authenticating: AUTH challenge frame received, gateway.cjs:734,665
    Authenticating --> SubscribedColdBoot: AUTH sent, REQ ctrl since now-50h, gateway.cjs:667-669,664
    SubscribedColdBoot --> Armed: first EOSE received, armed=true coldBoot=false, gateway.cjs:737
    Armed --> Armed: EVENT frames dispatched via handleWrap, gateway.cjs:735,702
    Armed --> Armed: keep-warm re-REQ plus ping every 15000ms, gateway.cjs:759
    Armed --> Reconnecting: ws close event, gateway.cjs:751
    Connecting --> Reconnecting: ws error, gateway.cjs:752
    Reconnecting --> Connecting: setTimeout(connect, 5000), gateway.cjs:751
    Connecting --> SubscribedWarm: reconnect and coldBoot is false, armed stays true, gateway.cjs:749
    SubscribedWarm --> Armed: seen-set dedupes replayed history, disconnect-gap commands still dispatched, gateway.cjs:745-748
    note right of Armed
        INVARIANT do not shorten the since window or add a
        created_at freshness check here — NIP-59 randomizes
        gift-wrap created_at up to 48h into the past
        (gateway.cjs:89, nostr-control-gateway.md:103-115)
    end note
```

## AB-13.14 nostr-pod-bridge main — the four entry points of one binary

```mermaid
sequenceDiagram
    autonumber
    participant ARGV as std::env::args<br/>services/nostr-pod-bridge/src/main.rs:59
    participant BOOT as bootstrap::run<br/>services/nostr-pod-bridge/src/bootstrap.rs:299
    participant DAEMON as run_daemon<br/>main.rs:109
    participant SESS as run_session_summary<br/>main.rs:83
    participant SUM as run_summarise / run_track<br/>main.rs:92,101

    ARGV->>ARGV: match args().nth(1) main.rs:59
    alt argv1 == bootstrap
        ARGV->>BOOT: bootstrap::run(&env) main.rs:61
Note over BOOT: boot phase [2/8] — keypair, pod scaffolding, DID docs (contract.rs),<br/>gitmark/blocktrails web contract, identity.env. Runs as root, before any bridge secret exists<br/>(main.rs:27-28,60). Full identity-minting internals are AB-11.2 — not duplicated here
    else argv1 == session-summary
        ARGV->>SESS: run_session_summary(&env) main.rs:62,83-88
        SESS-->>SESS: see AB-13.10 for the full kind-30840 pipeline
    else argv1 == summarise
        ARGV->>SUM: run_summarise(&BridgeConfig::from_env) main.rs:63,92-97
        SUM->>SUM: read SessionSummary JSON from stdin, publish_session_summary main.rs:93-96
    else argv1 == track
        ARGV->>SUM: run_track(&BridgeConfig::from_env) main.rs:64,101-106
        SUM->>SUM: read ProjectTrackingDigest JSON from stdin, publish_project_tracking main.rs:103-105
    else no argv1 (daemon mode)
        ARGV->>DAEMON: run_daemon(BridgeConfig::from_env) main.rs:69,109
        DAEMON-->>DAEMON: see AB-13.1 to AB-13.3 and AB-13.6 for the embedded relay and inbox pipeline
    else unknown subcommand
        ARGV-->>ARGV: anyhow error naming the four valid forms main.rs:65-68
    end
Note over ARGV,SUM: this one binary replaced scripts/sovereign-bootstrap.py and<br/>config/hooks/nostr-session-summary.py (main.rs:1-10) — only bootstrap resolves its own roots<br/>and never touches BridgeConfig, since it is what CREATES the bridge secrets (main.rs:26-28)
```

