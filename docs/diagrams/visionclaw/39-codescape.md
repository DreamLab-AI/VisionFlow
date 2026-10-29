---
id: VC-39
title: Codescape - the separate code-graph section, its crates, wire reuse, voice and proxy trust
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/codescape/docs/BASELINE-codescape.md
  - ../project/docs/PROTOCOL-registry.md
adrs: [ADR-2143, ADR-2018, ADR-2019, ADR-2024, ADR-2141, ADR-2148]
sources:
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
  - ../project/codescape/Cargo.toml
  - ../project/codescape/crates/codescape-model/Cargo.toml
  - ../project/codescape/crates/codescape-ingest/Cargo.toml
  - ../project/codescape/crates/codescape-layout/Cargo.toml
  - ../project/codescape/crates/codescape-collab/Cargo.toml
  - ../project/codescape/crates/codescape-server/Cargo.toml
  - ../project/crates/visionclaw-wire/src/lib.rs
  - ../project/codescape/docs/BASELINE-codescape.md
  - ../project/codescape/crates/codescape-layout/src/ptx_source.rs
  - ../project/codescape/crates/codescape-server/tests/isolation.rs
  - ../project/codescape/crates/codescape-model/src/lib.rs
  - ../project/codescape/crates/codescape-ingest/src/lib.rs
  - ../project/codescape/crates/codescape-layout/src/lib.rs
  - ../project/codescape/crates/codescape-collab/src/lib.rs
  - ../project/codescape/crates/codescape-server/src/lib.rs
  - ../project/codescape/crates/codescape-model/src/wire.rs
  - ../project/codescape/crates/codescape-model/src/ids.rs
  - ../project/crates/visionclaw-protocol/src/wire_fixtures.rs
  - ../project/codescape/crates/codescape-collab/src/floor.rs
  - ../project/codescape/crates/codescape-collab/src/grammar.rs
  - ../project/codescape/crates/codescape-collab/src/llm.rs
  - ../project/codescape/crates/codescape-server/src/main.rs
  - ../project/codescape/crates/codescape-server/src/state.rs
  - ../project/codescape/crates/codescape-server/src/engine.rs
  - ../project/codescape/crates/codescape-server/src/frames.rs
  - ../project/codescape/crates/codescape-server/src/wire.rs
  - ../project/codescape/crates/codescape-server/src/proxy.rs
  - ../project/codescape/crates/codescape-server/src/config.rs
  - ../project/codescape/crates/codescape-server/src/auth.rs
  - ../project/codescape/crates/codescape-server/src/http.rs
  - ../project/codescape/crates/codescape-server/src/egress.rs
  - ../project/codescape/crates/codescape-server/src/ingest_api.rs
  - ../project/codescape/crates/codescape-server/src/ingest_sink.rs
  - ../project/codescape/crates/codescape-server/src/xr_presence.rs
  - ../project/codescape/crates/codescape-server/src/routes/mod.rs
  - ../project/codescape/crates/codescape-server/src/routes/ws.rs
  - ../project/codescape/crates/codescape-server/src/routes/voice.rs
  - ../project/codescape/crates/codescape-server/src/voice/asr.rs
  - ../project/codescape/crates/codescape-server/src/voice/tts.rs
  - ../project/codescape/crates/codescape-server/src/voice/loom.rs
  - ../project/codescape/crates/codescape-server/src/voice/jev.rs
  - ../project/codescape/README.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docker-compose.system-one.yml
  - ../project/nginx.dev.conf
  - ../project/docker-compose.unified.yml
  - ../project/codescape/client/src/net/config.ts
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/agentbox/config/hooks/activity-sink.cjs
verified_commit: {visionclaw: e74bcddc772d47a45b1971156a2bab540eaa027c, agentbox: 0b0f7918961e629280faf86f88227535abf514b5, codescape: c742770ce2a454343d1bba7e1e815c40557cafdf}
---

Codescape is a separate section: a near-real-time 3D view of a codebase that humans and agents share by voice, with its own Cargo workspace, its own compose service and a tested isolation boundary from the knowledge-work system (ADR-2143). It reuses the frozen V3/V5 position wire so the desktop and Godot clients render it unchanged, and it never mounts the corpus, never reaches RuVector and holds no signing key. The agentbox hook is on agentbox `main` (ADR-2146, activity-sink.cjs:5), and the standalone repository now exists: `gh repo view DreamLab-AI/codescape` resolves as a public repository on 2026-10-29. Since the previous stamp the voice path gained an opt-in System One (Jev) pick between grammar and Loom (codescape-server/src/voice/jev.rs:375) and the compose service serves the whole estate as one scene.

## VC-39.1 Crate DAG, licences and the isolation fence (Codescape is its own repository since ADR-2143 A5)

```mermaid
flowchart TB
    subgraph REPO["DreamLab-AI/codescape, a git submodule at codescape/ in VisionClaw, own [workspace] and own Cargo.lock, never a member of another workspace<br/>codescape/Cargo.toml:17-28"]
        direction TB
        LIC["licence split: the workspace default is AGPL-3.0-only (codescape-ingest, codescape-layout, codescape-server inherit it),<br/>codescape-model and codescape-collab override it with MIT OR Apache-2.0 and publish = true<br/>codescape/Cargo.toml:34 codescape-model/Cargo.toml:7 codescape-model/Cargo.toml:15 codescape-collab/Cargo.toml:7"]
        M["codescape-model: nodes, interned 26-bit ids, LOD, V3 mapping, I/O-free<br/>codescape-model/src/lib.rs:1-8 deps serde, serde_json, thiserror and visionclaw-wire codescape-model/Cargo.toml:17-21"]
        I["codescape-ingest: the only graphicle consumer, no graphicle type crosses its API<br/>codescape-ingest/src/lib.rs:1-9 codescape-ingest/Cargo.toml:14"]
        L["codescape-layout: CPU Barnes-Hut oracle, cuda feature drives the visionclaw-gpu PTX loaded at run time<br/>codescape-layout/src/lib.rs:1-26 codescape-layout/Cargo.toml:14"]
        C["codescape-collab: actors, heat, floor, grammar, I/O-free<br/>codescape-collab/src/lib.rs:1-5 codescape-collab/Cargo.toml:17"]
        S["codescape-server: actix, every network adapter, the binary<br/>codescape-server/src/lib.rs:1-12 codescape-server/Cargo.toml:22-24"]
        I --> M
        L --> M
        C --> M
        S --> M
        S --> I
        S --> L
        S --> C
    end
    G["graphicle, Apache-2.0, git dependency pinned to rev 003f936<br/>codescape/Cargo.toml:53"]
    PINS["exact pins graphicle forces: smallvec 1.16.1, unicode-normalization 0.1.22<br/>codescape/Cargo.toml:75-76"]
    WIRE["visionclaw-wire 0.1.0 from crates.io, dependency-free, MIT OR Apache-2.0: the frozen V3/V5 record and fixtures<br/>codescape/Cargo.toml:48 visionclaw-wire/src/lib.rs:83 codescape-model/src/wire.rs:17"]
    PTX["PTX and ptx-build-manifest.txt read at run time from CODESCAPE_PTX_DIR, no link to visionclaw-gpu<br/>codescape-layout/src/ptx_source.rs:1-25"]
    I --> G
    G --- PINS
    M --> WIRE
    L -.->|"feature cuda, cust only"| PTX
    WHY["INVARIANT I6: separate lock because graphicle turns on serde_json arbitrary_precision,<br/>which would unify into any crate sharing the resolution<br/>codescape/Cargo.toml:12-16"]
    REPO --- WHY
    BAN["INVARIANT I1 and I1a: no ontology, memory-cloud, vault, Oxigraph, Postgres, RuVector, Solid or VisionClaw-server crate in the build graph, and visionclaw-wire is the ONLY visionclaw-* crate, both feature sets<br/>codescape-server/tests/isolation.rs:121 codescape-server/tests/isolation.rs:139"]
    REPO --- BAN
    NOGIT["INVARIANT I10 and I11: no git+ source naming the VisionClaw repository, and no permissive crate depends on an AGPL crate<br/>codescape-server/tests/isolation.rs:215 codescape-server/tests/isolation.rs:324"]
    REPO --- NOGIT
```

**What it shows.** Five crates cut by I/O: model, collab and the CPU layout compile with no graphicle, network or GPU; ingest alone sees graphicle; the server alone does network I/O. The two former edges into the VisionClaw monorepo, the optional `visionclaw-gpu` path dependency and the path-attribute fixture include, are gone: the wire comes from the crates.io crate `visionclaw-wire` and the PTX arrives as a run-time artefact. **Why it is this way.** ADR-2143 A5 extracted the workspace to its own repository so it is reusable outside VisionClaw without a 72 GB git history; the licence split lets the two I/O-free crates be published permissively (`codescape/Cargo.toml:34`).

## VC-39.2 Runtime topology: compose peer, shared port, clients, hook ingest and voice egress

```mermaid
flowchart TB
    subgraph HOST["host"]
        LAN["laptop or headset on the LAN"]
        LOOP["host loopback, port 4100 published to 127.0.0.1 by default<br/>docker-compose.unified.yml:633"]
    end
    subgraph NET["visionclaw_network"]
        subgraph VC["visionclaw_container, mounts /vault, never hosts the Codescape service"]
            NGX["dev nginx port 3001, location ^~ /codescape/, upstream resolved lazily by Docker DNS<br/>nginx.dev.conf:66 nginx.dev.conf:74-77"]
        end
        subgraph CS["codescape service, profile codescape, runs the standalone submodule mounted read-only at /app/codescape through its own deploy/codescape-entrypoint.sh, plus read-only views of the estate scene and ten member repositories<br/>docker-compose.unified.yml:454 docker-compose.unified.yml:464 docker-compose.unified.yml:520 docker-compose.unified.yml:482 docker-compose.unified.yml:530-611 docker-compose.unified.yml:651-652"]
            MAIN["main listener port 4100: REST, /ws/positions, /ws/events, /wss, /ws/voice, /ws/presence<br/>codescape-server/src/routes/mod.rs:406-418"]
            ING["ingest listener port 4101, only when a token file is configured, never published<br/>codescape-server/src/main.rs:175-189 docker-compose.unified.yml:629"]
            ENG["RepoEngine per repo: ingest thread, layout thread up to 240 Hz, publish every 16 ms<br/>codescape-server/src/engine.rs:32-36 codescape-server/src/engine.rs:713 codescape-server/src/engine.rs:767"]
            MAIN --> ENG
            ING --> ENG
        end
        AB["agentbox container: activity-sink hook, fail-open, Bearer token, 1 s budget<br/>activity-sink.cjs:4-23 activity-sink.cjs:276 activity-sink.cjs:306"]
        ASR["nemotron-asr, streaming ASR websocket<br/>codescape-server/src/config.rs:129-134"]
        TTS["pocket-tts<br/>codescape-server/src/config.rs:137-142"]
        LOOM["Loom door, optional out-of-grammar fallback<br/>docker-compose.unified.yml:503"]
        JEV["System One facade systemone:8097, opt-in CODESCAPE_JEV, local-only, endpoint from compose<br/>docker-compose.unified.yml:494-500 voice/jev.rs:156-205; the facade is defined by agentbox docker-compose.system-one.yml:3 and :106"]
    end
    LAN -->|"http port 3001 /codescape/"| NGX
    LAN -.->|"direct, only with CODESCAPE_PUBLISH_ADDR set"| LOOP
    LOOP --> MAIN
    NGX -->|"rewrite strips /codescape, forwards X-Forwarded-*"| MAIN
    AB -->|"POST /ingest/v1/activity"| ING
    ENG -->|"egress allowlist checked per URL<br/>codescape-server/src/egress.rs:77"| ASR
    ENG --> TTS
    ENG -.-> LOOM
    ENG -.->|"Jev URL joins the allowlist only when enabled and resolved<br/>codescape-server/src/config.rs:281-282"| JEV
    DESK["desktop client: base from a /codescape first path segment, ?server= override<br/>codescape/client/src/net/config.ts:15-18 codescape/client/src/net/config.ts:32-57"]
    XR["Godot XR client: Backend toggle appends /codescape to XR_BACKEND_WS once<br/>xr-client/scripts/graph_scene.gd:76 xr-client/scripts/graph_scene.gd:909-920 xr-client/scripts/graph_scene.gd:1533-1542"]
    XRN["see VC-40.1 and VC-40.2 - after this stamp (ADR-2143 A6) the XR client gates every server-bound request<br/>on a per-backend capability table and abandons in-flight requests on a switch"]
    DESK --- LAN
    XR --- LAN
    XR -.- XRN
    REF["INVARIANT I8: RUVECTOR_PG_CONNINFO in the environment exits 2 before argument parsing<br/>codescape-server/src/main.rs:47-51 codescape-server/src/config.rs:15"]
    CS --- REF
```

**What it shows.** Codescape is a network peer of VisionClaw, never a co-tenant: nginx reaches it by service name and answers 502 when it is down, the hook reaches port 4101 only across the compose network, and voice egress goes only to the configured ASR, TTS and Loom URLs plus the P6 voice-intent URL and, when enabled, the Jev URL (codescape-server/src/config.rs:265-283); the compose peer now maps the estate scene and ten member repositories read-only, not just VisionClaw and agentbox (docker-compose.unified.yml:482). **Why it is this way.** `visionclaw_container` mounts `/vault` and carries the RuVector connection string, so running Codescape there would make isolation a convention (ADR-2143-codescape-section-separate-workspace-and-port.md:187-193); `/codescape/` on port 3001 is the owner's dev-only waiver of A2 gap 4 (nginx.dev.conf:63-65).

## VC-39.3 One position socket tick: admission, base records, selection, V5, ack

```mermaid
sequenceDiagram
    autonumber
    participant CL as client, desktop or XR
    participant PL as position_loop<br/>codescape-server/src/routes/ws.rs:307
    participant PS as PositionStream.tick<br/>codescape-server/src/frames.rs:106
    participant AW as AckWindow<br/>codescape-server/src/wire.rs:146
    participant EN as RepoEngine.base_records<br/>codescape-server/src/engine.rs:514
    participant FS as FrameSelector.select<br/>codescape-server/src/wire.rs:234
    CL->>PL: open /wss, XR gets initialGraphLoad text first, then a forced keyframe (codescape-server/src/routes/ws.rs:325)
    Note over PL: ticker at frame_hz, default 10, range 1 to 30 (codescape-server/src/config.rs:117)
    PL->>PS: tick(now) on each ticker beat (codescape-server/src/routes/ws.rs:402)
    PS->>AW: admit(now) (codescape-server/src/frames.rs:107)
    alt client never acked, or fewer than 2 frames in flight
        AW-->>PS: Send (codescape-server/src/wire.rs:147-149)
    else 2 in flight and oldest under 2 s
        AW-->>PS: Skip, lag recorded, no frame this tick (codescape-server/src/wire.rs:157-161)
    else oldest unacked for 2 s or more
        AW-->>PS: Recover, window cleared, keyframe forced (codescape-server/src/wire.rs:152-155)
    end
    PS->>EN: base_records(now), one shared build per key of positions version, revision, 250 ms heat bucket, actors (codescape-server/src/engine.rs:526-536)
    EN-->>PS: code records via record_for plus actor records hovering 4 units above their focus (codescape-server/src/engine.rs:552 codescape-server/src/engine.rs:569)
    Note over PS: overlay per-client SSSP hops and parent from the focus (codescape-server/src/frames.rs:134-137)
    PS->>FS: select(current, now, 16384) (codescape-server/src/frames.rs:141-143)
    Note over FS: keyframe when forced or every 10 s clears the baseline, else only movers beyond 0.01 or changed fields (codescape-server/src/wire.rs:35 codescape-server/src/wire.rs:238-243 codescape-server/src/wire.rs:45-67)
    FS-->>PS: at most 16384 records, the rest deferred to the next tick (codescape-server/src/wire.rs:254-259)
    PS->>PS: seq advance, on_sent, v5_frame(seq, records) (codescape-server/src/frames.rs:147-150)
    PS-->>PL: V5 frame bytes
    PL->>CL: binary V5, 1 + 8 + 52n bytes (visionclaw-wire/src/lib.rs:183)
    CL->>PL: 0x34 ack, 21 bytes, cumulative (codescape-model/src/wire.rs:88-91)
    PL->>AW: on_ack(seq) switches pacing on and drops every frame up to seq (codescape-server/src/wire.rs:176-182)
    Note over PL: XR only, 0x23 agent actions and a topology resend at most every 2 s after deltas (codescape-server/src/routes/ws.rs:37 codescape-server/src/routes/ws.rs:395-401 codescape-server/src/routes/ws.rs:413-418)
```

**What it shows.** The whole per-client frame pipeline: admission by the ack window, a base shared by every client of the tick, a per-client projection and SSSP overlay, a diff against what was actually sent, and the budget that spreads a large keyframe over several sequenced frames. **Why it is this way.** A desktop that never acks is never paced, so it cannot stall, and the XR freshness gate silently drops stale frames without acking, so the timeout must recover rather than wait (codescape-server/src/wire.rs:7-12); velocity is ignored in the diff because it changes every step (codescape-server/src/wire.rs:44).

## VC-39.4 The frozen V3 record and what each field means for a code graph

```mermaid
classDiagram
    class NodeRecord {
        frozen 52 bytes, little-endian : visionclaw-wire/src/lib.rs:83
        struct : visionclaw-wire/src/lib.rs:107-125
        serialiser : visionclaw-wire/src/lib.rs:147
        u32 wire_id at 0, node id masked to 26 bits
        f32x3 position at 4, velocity at 16
        f32 sssp_distance at 28, i32 sssp_parent at 32
        u32 cluster_id at 36
        f32 anomaly at 40
        u32 community_id at 44
        f32 centrality at 48
    }
    class CodeNodeMapping {
        record_for : codescape-model/src/wire.rs:28-46
        wire_id is node.id and NODE_ID_MASK, no flag bits
        cluster_id is the crate or top-level directory
        community_id is the stable-matched Leiden community
        centrality is importance clamped 0 to 1
        anomaly is activity heat clamped 0 to 1
        sssp is BFS hops and parent from the focus
    }
    class ActorMapping {
        actor_record : codescape-model/src/wire.rs:49-53
        wire_id carries AGENT_FLAG 0x80000000
        flag constant : codescape-model/src/wire.rs:25
        anomaly fixed by the caller to 1.0
    }
    class IdSpace {
        NODE_ID_MASK 0x03FFFFFF : ids.rs:16
        interned monotonic per epoch, never hashed
    }
    class V5Frame {
        opcode 0x05 then u64 sequence then n records
        builder : visionclaw-wire/src/lib.rs:183
    }
    class AckFrame {
        0x34, u64 seq, u32 nodes, u64 client ts, 21 bytes
        decode_ack : codescape-model/src/wire.rs:88
    }
    class AgentActionFrame {
        0x23, u16 count, bodies of agent, node, type, ts, duration
        builder : codescape-model/src/wire.rs:115-127
        types Query 0, Update 1, Create 2, Delete 3
    }
    CodeNodeMapping ..> NodeRecord : fills
    ActorMapping ..> NodeRecord : fills
    CodeNodeMapping ..> IdSpace : masks with
    V5Frame o-- NodeRecord : carries
    AckFrame ..> V5Frame : acknowledges
```

**What it shows.** Codescape claims no tag byte and no flag bit: it re-reads the five semantic slots of the shared record as code-graph meanings and marks actors with the existing agent flag, so both clients decode it with their own parsers. **Why it is this way.** A new wire format would fork three decoders, which ADR-2018 and ADR-2019 exist to prevent (ADR-2143-codescape-section-separate-workspace-and-port.md:101); ids are interned, not hashed, because 100k symbol keys hashed into 26 bits would collide (ADR-2143-codescape-section-separate-workspace-and-port.md:27, ADR-2143-codescape-section-separate-workspace-and-port.md:45).

## VC-39.5 Voice: one floor, ASR, grammar or Loom, executor, spoken answer

```mermaid
sequenceDiagram
    autonumber
    participant B as browser, push-to-talk
    participant V as voice_loop<br/>codescape-server/src/routes/voice.rs:192
    participant F as FloorGate over FloorArbiter<br/>codescape-server/src/state.rs:182
    participant A as AsrSession, nemotron<br/>codescape-server/src/voice/asr.rs:256
    participant R as run_command_inner<br/>codescape-server/src/routes/mod.rs:289
    participant G as parse_command<br/>grammar.rs:484
    participant JV as Jev pick, after_grammar<br/>codescape-server/src/voice/jev.rs:375
    participant LM as Loom ask<br/>codescape-server/src/voice/loom.rs:148
    participant T as pocket-tts speak<br/>codescape-server/src/voice/tts.rs:78
    B->>V: text type start, button down (codescape-server/src/routes/voice.rs:238-245)
    V->>F: request(actor, conn), delegating to FloorArbiter.request, broadcast floor state (codescape-server/src/state.rs:182 floor.rs:91)
    F-->>V: Held by this actor arrives on the event subscription (codescape-server/src/routes/voice.rs:298-300)
    V->>A: start, the egress allowlist checks the ASR URL first (codescape-server/src/voice/asr.rs:261)
    B->>V: binary f32 PCM at 24 kHz, buffered as preroll until the floor is held (codescape-server/src/routes/voice.rs:278-290)
    A-->>V: Partial captions, broadcast on /ws/events (codescape-server/src/routes/voice.rs:337-338)
    B->>V: text type stop, finish (codescape-server/src/routes/voice.rs:247-250)
    A-->>V: Final transcript, floor released (codescape-server/src/routes/voice.rs:340-343)
    Note over V: one command at a time per socket, a second is answered busy (codescape-server/src/routes/voice.rs:347-349)
    V->>R: run_command, text cut to 512 characters (codescape-server/src/routes/mod.rs:287)
    R->>G: parse_command(text), wrapped by after_grammar (codescape-server/src/routes/mod.rs:297)
    alt grammar match
        G-->>R: typed Command
    else no match and Jev enabled and it picks an offered argument-free command
        R->>JV: NoMatch only, never a grammar hit, empty or invalid<br/>(codescape-server/src/voice/jev.rs:380-385)
        JV-->>R: typed Command, validated, confidence at or above the threshold<br/>(codescape-server/src/voice/jev.rs:345-354)
        Note over JV: none, timeout, transport, non-2xx, malformed or low confidence<br/>all fall through to Loom (codescape-server/src/voice/jev.rs:326-351)
    else no match and Loom configured
        R->>LM: ask with LOOM_SYSTEM and command_schema (codescape-server/src/routes/mod.rs:308)
        LM-->>R: JSON, then command_from_llm validates it (llm.rs:209)
        Note over R: INVARIANT: a reply that fails validation is spoken as text, never executed (codescape-server/src/routes/mod.rs:320-334)
    else no match, no Loom
        R-->>V: error naming the normalised text (codescape-server/src/routes/mod.rs:303-306)
    end
    R->>R: execute(Command) on a blocking thread (codescape-server/src/routes/mod.rs:351-354)
    R-->>V: QueryResult and a Focus effect, view and layout burst follow (codescape-server/src/routes/voice.rs:359-366)
    V->>T: speak(answer), to the speaker only (codescape-server/src/routes/voice.rs:372)
    T-->>B: tts_start, s16le chunks, tts_end (codescape-server/src/routes/voice.rs:112-117)
    Note over F: claim per connection, hold cap 30 s, ticked every second by main (codescape-server/src/state.rs:139 floor.rs:13 codescape-server/src/main.rs:239-250)
```

**What it shows.** A single floor gates the single-stream ASR, the final transcript goes through the same executor as typed commands, Jev may pick one of ten argument-free commands only when the grammar fails (off by default, README.md:289), and the Loom is consulted only when both miss. **Why it is this way.** Every action must be a typed `Command` before it runs and LLM output that fails the schema is spoken, never executed; agent dispatch is deferred to a browser-signed kind-31402 relay in P6 because Codescape holds no signing key (ADR-2143-codescape-section-separate-workspace-and-port.md:65).

## VC-39.6 Proxy trust: whose forwarded headers are believed

```mermaid
flowchart TB
    REQ["request reaches port 4100, TCP peer p, headers h"]
    CFG["boot: --trust-proxy-from without --trust-proxy is a boot error<br/>codescape-server/src/config.rs:237-239"]
    EN{"trust enabled?<br/>proxy.rs:268"}
    LIST{"peer list empty?<br/>proxy.rs:271"}
    MATCH{"p inside a listed address or CIDR,<br/>or among the resolved hostnames?<br/>proxy.rs:274-278"}
    REFRESH["hostname entries resolved at boot and every 60 s, a name that stops resolving trusts no peer<br/>proxy.rs:115 proxy.rs:243-262 codescape-server/src/main.rs:198-237"]
    OFF["trust does not apply: Host for same-origin, TCP peer for rate limits, configured or bind origin for NIP-98 URLs<br/>proxy.rs:289-290 proxy.rs:307"]
    ON["trust applies"]
    HOST["same-origin host: first X-Forwarded-Host if a bare authority, else Host<br/>proxy.rs:283-291"]
    IP["rate-limit key: LAST X-Forwarded-For hop, the one nginx appended<br/>proxy.rs:295-308"]
    ORIG["public origin: explicit --public-origin wins, else proto://forwarded-host plus prefix<br/>proxy.rs:368-375 proxy.rs:313-326"]
    PFX["prefix: a well-formed X-Forwarded-Prefix, else --public-path-prefix<br/>proxy.rs:322-324 proxy.rs:78-95"]
    USE["NIP-98 URL for /wss and every verify is rebuilt from that origin, never from Host<br/>codescape-server/src/routes/mod.rs:427-430 codescape-server/src/routes/ws.rs:263-266"]
    REQ --> EN
    CFG -.-> EN
    EN -->|no| OFF
    EN -->|yes| LIST
    LIST -->|"yes, any peer"| ON
    LIST -->|no| MATCH
    REFRESH -.-> MATCH
    MATCH -->|"no, or peer unknown"| OFF
    MATCH -->|yes| ON
    ON --> HOST
    ON --> IP
    ON --> ORIG
    ORIG --> PFX
    ORIG --> USE
    OFF --> USE
    COMPOSE["compose sets trust on, narrowed to the VisionClaw service alias, prefix /codescape<br/>docker-compose.unified.yml:506 docker-compose.unified.yml:510-511"]
    NG["nginx sends Host, X-Forwarded-For appended, Proto, Host, Prefix, hides upstream COOP/COEP/CORP<br/>nginx.dev.conf:81-86 nginx.dev.conf:94-96"]
    COMPOSE -.-> EN
    NG -.-> REQ
```

**What it shows.** A sibling container on the compose network that sends the same headers as nginx is treated exactly as trust off, because trust needs both the switch and a matching peer; only the last forwarded-for hop is believed, since the leftmost entries are client-supplied. **Why it is this way.** ADR-2143 A4.4 (ADR-2143-codescape-section-separate-workspace-and-port.md:205-213); the duplicate isolation headers are hidden because two copies parse as `unsafe-none` and defeat cross-origin isolation (nginx.dev.conf:92-93).

## VC-39.7 Register: tensions, debt, drift and open questions at codescape c742770

```mermaid
flowchart TB
    subgraph SEC["OPEN: auth"]
        direction TB
        R1["OPEN until P6: the main port runs OpenAccess, reported honestly as auth open, while<br/>port 3001 /codescape/ is LAN-published in dev by owner decision<br/>codescape-server/src/auth.rs:120-124 codescape-server/src/routes/mod.rs:69 nginx.dev.conf:63-65"]
        R2["OPEN: /ws/presence shim accepts the Schnorr signature unverified, the claimed DID is only a label,<br/>bounds 64 members per room, 256 rooms<br/>codescape-server/src/xr_presence.rs:20-23 codescape-server/src/xr_presence.rs:65-67"]
        R3["TENSION: I8a egress allowlist and no-/vault are REJECTED as invariants, compose non-publication<br/>is not an egress boundary, open until P2 ships an egress test<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:82 ADR-2143-codescape-section-separate-workspace-and-port.md:144"]
        R1 --- R2 --- R3
    end
    subgraph JEVG["OPEN: Jev option and estate scene"]
        direction TB
        J1["OPEN: enabled in compose but accepts nothing: the threshold defaults to 1.0 and compose leaves it empty, README states wiring and fallback tests only, no measured uplift, and BASELINE says activation is staged until the host is recreated<br/>codescape-server/src/voice/jev.rs:166-171 docker-compose.unified.yml:499-500 codescape/README.md:296 docs/BASELINE-architecture.md:49-50"]
        J2["DEBT: the estate mounts default to an absolute host volume path, overridable only by CODESCAPE_ESTATE_ROOT<br/>docker-compose.unified.yml:556"]
        J1 --- J2
    end
    subgraph PERF["DEBT: scale targets"]
        direction TB
        P1["MISSED: GPU step at 100k nodes 7.8 ms against 5 ms, kernels only 1.7 ms<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:133"]
        P2["MISSED: cold settle at 100k 13.7 s on GPU against 10 s, CPU 56.6 s, cause host-side work per step<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:134-136"]
        P3["OPEN: I7a GPU-vs-CPU conformance is an opt-in ignored test, not an invariant, required at P2<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:80 ADR-2143-codescape-section-separate-workspace-and-port.md:140-143"]
        P1 --- P2 --- P3
    end
    subgraph XRG["OPEN: XR"]
        direction TB
        X1["OPEN: empty agent hover label in the headset, actor ids carry AGENT_FLAG but the headset<br/>names agents from its label table, else a demo role, else agent N<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:114 xr-client/scripts/graph_scene.gd:2692-2700"]
        X2["DOC-DRIFT: the P4 row says /ws/presence is unimplemented, but the route is registered<br/>and A4.5 adds the shim<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:114 codescape-server/src/routes/mod.rs:418 ADR-2143-codescape-section-separate-workspace-and-port.md:217-218"]
        X3["OPEN: /wss ignores nodePin, nodeUnpin and nodeDrag and there is no physics write, so the headset greys<br/>those controls (see VC-40.1); whether Codescape should honour drag and pin is undecided"]
        X1 --- X2 --- X3
    end
    subgraph EXT["OPEN: A5 extraction to DreamLab-AI/codescape, done in code, last step open"]
        direction TB
        E1["BLOCKER for crates.io: graphicle is a git dependency and crates.io refuses git dependencies, so codescape-ingest cannot be published until graphicle is<br/>codescape/Cargo.toml:53"]
        E2["OPEN: the last A5 step is unticked, Decision 1, Decision 7 and I2 are not yet amended and the governing docs are not updated, so the ADR text still describes crates/codescape and the build-compiled PTX<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:282"]
        E3["DONE: licence split, visionclaw-wire extracted, PTX from CODESCAPE_PTX_DIR, submodule mounted, I10 enforced, three crates published at 0.1.0<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:274-281"]
        E4["RESOLVED: the activity-sink hook is on agentbox main, now ADR-2146<br/>activity-sink.cjs:5"]
        E1 --- E2 --- E3 --- E4
    end
    subgraph DR["DOC-DRIFT: ADR-2143 text"]
        direction TB
        D1["A2 item 6 now says crates/codescape was untracked only at the time of A2 and has since been committed, then moved to the submodule<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:167"]
        D2["client import allowlist is wider than decision 8 states<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:149"]
        D3["I7 accepts a hand-authored manifest saying provenance=compiled, I3 and I4 are helper-level tests<br/>ADR-2143-codescape-section-separate-workspace-and-port.md:153-154"]
        D1 --- D2 --- D3
    end
```

**What it shows.** Every known gap at the stamp, each anchored to the code or the ADR line that records it; ADR-2143 is `implementation_status: partial` and `activation_status: inactive` (ADR-2143-codescape-section-separate-workspace-and-port.md:6-7), with P0 done and P2 to P6 open (ADR-2143-codescape-section-separate-workspace-and-port.md:296). **Verified.** Citations read at visionclaw `e74bcddc7`, agentbox `0b0f79189` and codescape `c742770` on 2026-10-29; the `/codescape/` prod path is untouched and prod's SPA fallback answers 200 there, not 404 (ADR-2143-codescape-section-separate-workspace-and-port.md:196-198).
