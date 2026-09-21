---
id: ES-01
title: Estate topology — substrates, network fabric, service ports, compose networks
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [agentbox:ADR-2023, agentbox:ADR-2013, visionclaw:ADR-2027, visionclaw:ADR-2025, agentbox:ADR-2009, agentbox:ADR-2012, agentbox:ADR-2062, agentbox:ADR-2034, agentbox:ADR-2104, agentbox:ADR-2096, agentbox:ADR-2098]
sources:
  - ../project/.gitmodules
  - ../project/.gitignore
  - ../project/Cargo.toml
  - ../project/docker-compose.unified.yml
  - ../project/docker-compose.cloudflared.yml
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/docker-compose.override.yml
  - ../project/agentbox/docker-compose.browsercontainer.yml
  - ../project/agentbox/docker-compose.gui-tools.yml
  - ../project/agentbox/docker-compose.solid-pods.yml
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/docker-compose.xr-runtime.yml
  - ../project/agentbox/docker-compose.hp.yml
  - ../project/agentbox/docker-compose.android.yml
  - ../project/agentbox/docker-compose.openmed.yml
  - ../project/nginx.conf
  - ../project/nginx.dev.conf
  - ../project/nginx.production.conf
  - ../project/supervisord.dev.conf
  - ../project/supervisord.production.conf
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/CLAUDE.md
  - ../project/agentbox/skills/email-search/SKILL.md
  - ../project/agentbox/docs/developer/hp-peer-node.md
  - ../project/agentbox/docs/developer/ecosystem.md
  - ../project/agentbox/docs/developer/native-pod-mesh.md
  - ../project/loom/README.md
  - ../project/agentbox/docs/adr/ADR-2023-loom-facade.md
  - ../project/agentbox/docs/adr/ADR-2013-loopback-publish-except-9096.md
  - ../project/docs/adr/ADR-2027-three-deployment-profiles.md
  - ../project/docs/adr/ADR-2025-cross-from-agentbox-closed-map.md
  - ../project/agentbox/lib/solid-pod-rs.nix
  - ../project/agentbox/flake.nix
  - ../project/agentbox/services/agentbox-mcp/src/main.rs
  - ../project/agentbox/services/agentbox-mcp/src/hub/mod.rs
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/developer/economy-loop.md
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - scripts/estate-health/roster.json
verified_commit: {visionclaw: f223bbd40, agentbox: b7b1ab81a, visionflow: df22182f3}
---
## ES-01.1 Substrate map — the VisionClaw checkout's neighbourhood, not the whole estate
```mermaid
flowchart TB
    subgraph ONDISK["On disk in this checkout"]
        VC["VisionClaw (this repo)<br/>root: Cargo.toml, docker-compose.unified.yml"]
        AB["agentbox<br/>git submodule at agentbox/<br/>.gitmodules: url=github.com/DreamLab-AI/agentbox.git"]
    end
    VC -->|"embeds as submodule<br/>path=agentbox, .gitmodules:1-3<br/>no branch key — the pin is a bare gitlink"| AB
    AB -.->|"gitlink at VisionClaw f223bbd40: c446783235<br/>git ls-tree HEAD agentbox<br/>working tree runs ahead on main"| ABPIN["submodule pin"]

    subgraph CRATE["Embedded as a Cargo dependency, not a checkout"]
        SPR["solid-pod-rs 0.4.0-alpha.15<br/>crates.io pin, Cargo.toml:219<br/>feature solid-pod-embed (ADR-032 M3)"]
        SPRN["solid-pod-rs v0.5.0-alpha.9<br/>agentbox Nix pin, solid-pod-rs.nix:53<br/>rev 1d9da527, solid-pod-rs.nix:56"]
    end
    VC -->|"Cargo dep: fs-backend, nip98-schnorr,<br/>did-nostr, quota, rate-limit"| SPR
    AB -->|"lib/solid-pod-rs.nix pin<br/>supervised solid-pod program on port 8484<br/>native-pod-mesh.md:3"| SPRN

    DIVSP["DIVERGENCE — the estate holds TWO solid-pod-rs versions at once.<br/>VisionClaw compiles crates.io 0.4.0-alpha.15 in-process<br/>(Cargo.toml:219); agentbox builds v0.5.0-alpha.9 from a tagged<br/>fetchFromGitHub rev (solid-pod-rs.nix:53,56) for the supervised<br/>port 8484 pod. native-pod-mesh.md:3 records the alpha.9 bump as live;<br/>the same doc's topology figure still labels the server<br/>v0.4.0-alpha.17 (native-pod-mesh.md:30) — DOC-DRIFT inside it.<br/>see ES-08.1"]
    SPRN --> DIVSP
    SPR --> DIVSP

    subgraph EXTERNAL1["EXTERNAL: not checked out at repo root"]
        VF["EXTERNAL: VisionFlow<br/>Umbrella coordination canon<br/>ecosystem.md: pure canon, does NOT sign the relay"]
        NRF["EXTERNAL: nostr-rust-forum<br/>Forum kit — peer on relay mesh<br/>ecosystem.md: receives IS-Envelope, renders ACSP panels"]
        DAW["EXTERNAL: dreamlab-ai-website<br/>Branded deployment<br/>ecosystem.md: downstream consumer of forum kit"]
    end
    VC -->|"peer on relay mesh<br/>renders embodied agent loop (GPU/XR graph)"| NRF
    AB -->|"hosts code-as-harness<br/>NOT double-counted in identity mesh"| NRF
    NRF -.->|"operator overlay"| DAW
    VF -.->|"documentation/positioning only<br/>agentbox/docs/developer/ecosystem.md table"| VC
    VF -.-> AB

    NOTE1["SCOPE — this map draws the SIX repositories agentbox's own<br/>ecosystem doc enumerates, of which 5 sign on the did:nostr<br/>identity mesh (VisionFlow is pure canon and signs nothing):<br/>agentbox/docs/developer/ecosystem.md."]
    NOTE2["DIVERGENCE — six is NOT the estate. The enumeration the tree<br/>treats as canonical is scripts/estate-health/roster.json, which<br/>carries FOURTEEN rows and is the only one walked row by row<br/>(roster.json:3). This map omits knowledgeGraph, visionGraph,<br/>vowl-wasm, loom, WasmVOWL, prose-sanitiser, diagram-ir and<br/>dream-engine. Those edges are drawn in ES-11; the full<br/>enumeration conflict is catalogued in VF-08.3. Kept at six here<br/>because this diagram answers what is ON DISK in the VisionClaw<br/>checkout, which is a different question from what the estate is."]
    NOTE1 -.-> VF
    NOTE2 -.-> VF
```

## ES-01.2 Network / compute fabric — gateway host, connected-node rail, retired-address trap
```mermaid
flowchart LR
    subgraph ML["The gateway host (on the LAN)"]
        MLHOST["gateway host<br/>agentbox/skills/email-search/SKILL.md:79"]
    end
    subgraph HP["The connected node (downstream, no LAN IP)"]
        HPHOST["connected node, user and address written as<br/>placeholders in the public repo<br/>agentbox/docs/developer/hp-peer-node.md:3"]
        LOOMFACADE["Loom façade port 8084<br/>colocated with the model on the connected node<br/>agentbox/skills/email-search/SKILL.md:80"]
        LOOMMODEL["loom-model container port 8085<br/>Qwen3.8-27B, cutover 2026-08-14<br/>agentbox/skills/email-search/SKILL.md:95"]
    end
    MLHOST -->|"point-to-point 25 G rail, hp-peer-node.md:4<br/>the gateway's NAT service DNATs the façade<br/>agentbox/skills/email-search/SKILL.md:99"| HPHOST
    HPHOST --> LOOMFACADE
    LOOMFACADE -->|"port 8085 HTTP delegates to the model<br/>agentbox/skills/email-search/SKILL.md:81"| LOOMMODEL
    MLHOST -->|"embeddings port 9997 on the gateway host<br/>bge models on xinference<br/>agentbox/skills/email-search/SKILL.md:113"| XINF["xinference port 9997"]

    DEAD["RETIRED-ADDRESS TRAP — the old model host is dead<br/>agentbox/docs/adr/ADR-2023-loom-facade.md:24<br/>agentbox/skills/email-search/SKILL.md:99,217"]
    DEAD -.->|"never target — black-holes every synthesis<br/>while GET /health still answers"| LOOMFACADE

    MESHNODE["a second full agentbox on the connected node,<br/>with its own did:nostr identity and not an annexe<br/>agentbox/docs/developer/hp-peer-node.md:4<br/>brought up by the compose overlay, hp-peer-node.md:17"]
    HPHOST --- MESHNODE
    MLHOST -->|"port 9096 NIP-98 door, signed by the ml node key 200,<br/>unsigned 401, hp-peer-node.md:46-47"| MESHNODE
    MESHNODE -->|"port 7777 embedded relay, allowlisted signer OK true,<br/>non-allowlisted logged rejected pubkey, hp-peer-node.md:49-50"| MLHOST

    GEN["EXTERNAL and DOC-DRIFT resolved — commit 2899b3b7e generalised<br/>every literal estate address out of this public repository, so the<br/>hostnames, the rail subnet and the retired IP that earlier revisions<br/>of this diagram cited are no longer stated in any cited source.<br/>The placeholders are the fact now: hp-peer-node.md:3 writes the peer<br/>as user-at-peer-ip, and ADR-2023-loom-facade.md:24 writes the dead<br/>host as a retired address. Literal values live only in the<br/>operator environment, which is not a source of this corpus."]
    GEN -.-> HPHOST
```

## ES-01.3 Service and port map — every published surface in the estate
```mermaid
flowchart TB
    subgraph vcstack["VisionClaw stack — docker-compose.unified.yml"]
        VCD["visionclaw_container<br/>profiles development, dev<br/>docker-compose.unified.yml:49,181-184"]
        VCP["visionclaw_prod_container<br/>profiles production, prod<br/>docker-compose.unified.yml:187,255-258"]
        LOOMB["loom-sidecar<br/>profile loom, docker-compose.unified.yml:305,366-367"]
        CFT["cloudflared-tunnel<br/>profiles production, prod<br/>docker-compose.unified.yml:261,279-281"]
    end
    subgraph abstack["agentbox stack — agentbox/docker-compose.yml"]
        ABC["agentbox container<br/>agentbox/docker-compose.yml:30"]
        RPG["ruvector-postgres<br/>agentbox/docker-compose.yml:10"]
    end
    subgraph sidecars["agentbox sidecar overlays"]
        BC["browsercontainer"]
        GT["gui-tools-service"]
        VCON["voice-console"]
        XRR["xr-runtime"]
        AND["agentbox-android"]
        OM["openmed"]
    end

    VCD -->|"port 3001 nginx<br/>docker-compose.unified.yml:163"| EXT1["host"]
    VCD -->|"port 4000 Rust backend<br/>docker-compose.unified.yml:164"| EXT1
    VCP -->|"port 3001 only<br/>docker-compose.unified.yml:231"| EXT1
    LOOMB -->|"host port 8090 to container port 8080<br/>docker-compose.unified.yml:351"| EXT1
    ABC -->|"port 9096 LAN — the ONLY 0.0.0.0 publish<br/>agentbox/docker-compose.yml:54"| EXT1
    ABC -->|"loopback ports 9090 9700 9091 8484 8888 5901 8080<br/>agentbox/docker-compose.yml:55-61"| LOOPBACK["loopback only"]
    RPG -->|"5432 internal"| ABC
    BC -->|"0.0.0.0 5903 VNC / 8931 MCP SSE<br/>host 9222 to container 9223 CDP"| EXT1
    GT -->|"0.0.0.0 5905 / 9876 / 9877"| EXT1
    VCON -->|"0.0.0.0 8443 / 8444"| EXT1
    XRR -->|"0.0.0.0 5904"| EXT1
    AND -->|"127.0.0.1 5555"| LOOPBACK
    OM -->|"127.0.0.1 9093"| LOOPBACK

    INV["INVARIANT ADR-2013 — every compose publish binds 127.0.0.1<br/>unless it is on the SANCTIONED list. In the main agentbox<br/>compose only port 9096 is a LAN door. see ES-10.8"]
    D1["NOT A DEFECT — browsercontainer maps host 9222 to container 9223<br/>by design (docker-compose.browsercontainer.yml:51-53, CDP proxy<br/>host:9222 to socat:9223 to Chrome:9222). The ADR-2013 sanctioned<br/>entry names 9222 (host side) and agentbox/CLAUDE.md names 9223<br/>(container side) — both correct, easy to misread as a conflict."]
    D2["EXTERNAL — Loom port 8084 façade and loom-model port 8085 run on<br/>HP-Desktop, NOT in either compose file. xinference port 9997 and<br/>email-mcp-gateway port 8765 are likewise separate services on<br/>visionclaw_network. see ES-01.2 and ES-06.1"]

    LOOPBACK --> INV
    BC --> D1
    LOOMB --> D2
```

## ES-01.4 Docker networks and volumes — one external bridge joins both stacks
```mermaid
flowchart TB
    subgraph net["visionclaw_network — external bridge, declared in BOTH stacks"]
        N1["docker-compose.unified.yml:369-372<br/>external true, name ${EXTERNAL_NETWORK:-visionclaw_network}"]
        N2["agentbox/docker-compose.override.yml:181-183<br/>alias visionclaw, external true"]
    end
    subgraph vcvol["VisionClaw volumes — docker-compose.unified.yml:374-396"]
        V1["loom-data — mirrored corpus generation<br/>docker-compose.unified.yml:376"]
        V2["visionclaw-data / visionclaw-logs<br/>docker-compose.unified.yml:380,383"]
        V3["npm-cache / cargo-cache / cargo-git-cache / cargo-target-cache<br/>docker-compose.unified.yml:387,390,393,396"]
    end
    subgraph abvol["agentbox volumes"]
        W1["ruvector-pg-data / ruvector-data<br/>agentbox/docker-compose.yml:191,193"]
        W2["solid-data / sovereign-identities / agentbox-secrets<br/>agentbox/docker-compose.yml:195,197,199"]
        W3["code-harness-data / agentbox-events / consultations-data<br/>agentbox/docker-compose.yml:201,203,213"]
        W4["hf-cache / codeserver-config / telemetry-data<br/>agentbox/docker-compose.yml:205,207,215"]
        W5["nostr-relay-data / tailscale-state, opencode-store, aoe-profiles<br/>agentbox/docker-compose.yml:217,219,209,211"]
    end
    subgraph shared["Cross-container shared volumes"]
        S1["gui-tools-exchange — declared by the override AND by<br/>browsercontainer and gui-tools overlays. This is how the<br/>browser sidecar reads files this container writes."]
        S2["mad-workspace — EXTERNAL alias to<br/>multi-agent-docker_workspace, from the deprecated MAD stack"]
    end

    N1 --- N2
    net --> vcvol
    net --> abvol
    net --> shared

    D1["DIVERGENCE — mad-workspace is a legacy external volume created<br/>by the DEPRECATED multi-agent-docker stack and reused so<br/>agentbox sees the full project tree. Migration path is<br/>agentbox.sh migrate-workspace, after which the override should<br/>reference agentbox-workspace instead (Q43)."]
    INV["INVARIANT — the cargo-target-cache and cargo-*-cache volumes are<br/>why the DEV container can compile Rust on startup rather than<br/>in the image build. see ES-09"]

    S2 --> D1
    V3 --> INV
```

## ES-01.5 Compose overlay composition — which file adds what
```mermaid
flowchart LR
    BASE["agentbox/docker-compose.yml<br/>agentbox + ruvector-postgres"]
    OV["docker-compose.override.yml<br/>auto-applied — joins visionclaw_network,<br/>group_add 965 for the docker socket,<br/>mounts mad-workspace + gui-tools-exchange"]
    HP["docker-compose.hp.yml<br/>2nd full agentbox on HP-Desktop,<br/>own did:nostr — see ES-01.2"]
    BCF["docker-compose.browsercontainer.yml<br/>GPU Chrome sidecar"]
    GTF["docker-compose.gui-tools.yml<br/>FHS GUI sidecar"]
    VF["docker-compose.voice.yml<br/>voice-console Caddy origin"]
    XF["docker-compose.xr-runtime.yml<br/>XR runtime + gdext build volumes"]
    SPF["docker-compose.solid-pods.yml<br/>cloudflared-pod tunnel"]
    ANF["docker-compose.android.yml"]
    OMF["docker-compose.openmed.yml"]

    BASE --> OV
    OV --> BCF
    OV --> GTF
    OV --> VF
    OV --> XF
    OV --> SPF
    OV --> ANF
    OV --> OMF
    BASE --> HP

    N1["Every overlay declares the same external network under the<br/>local alias visionclaw, so all sidecars share one bridge."]
    N2["group_add 965 is the docker socket gid — the container drives<br/>docker WITHOUT sudo, which no-new-privileges blocks."]
    D1["RESOLVED ADR-2013 — the voice overlay publishes port 8443 and<br/>port 8444 on 0.0.0.0 while the main compose publishes only port<br/>9096. Both voice doors sit on the CI-enforced SANCTIONED list<br/>(check-ports-loopback.mjs:95-96), beside the port 9096 ingress<br/>(:94) and the browser CDP door (:99) — decided exposures, not a<br/>breach. see ES-10.8"]
    D3["TENSION — the manifest declares the voice plane OFF while the<br/>voice stack runs. agentbox.toml:1681 sets [voice] enabled = false<br/>and calls it sidecar state with its own lifecycle, yet the two<br/>doors that stack publishes are permanently sanctioned in CI<br/>(check-ports-loopback.mjs:95-96). The gate therefore records the<br/>lifecycle owner, not whether voice is running: nothing in either<br/>file can be read as the answer to is voice up."]
    D2["TRAP — a build launched from INSIDE this container resolves bind<br/>paths against the HOST filesystem and silently bakes stale code.<br/>Build only from the host shell. see ES-09"]

    OV --> N1
    OV --> N2
    VF --> D1
    VF --> D3
    BASE --> D2
```

## ES-01.6 Repositories on disk — what is a checkout, what is a stub, what is external
```mermaid
flowchart TB
    subgraph real["Checked out WITH content at the repo root"]
        R1["JavaScriptSolidServer/ — 25 entries<br/>legacy JS Solid server. see ES-08"]
        R2["nntp-stack/ — 8 entries"]
        R3["voice-stack/ — 5 entries"]
        R4["loom/ — README.md + app/<br/>deployment notes only, NO implementation.<br/>The Rust loom-facade lives in the separate loom repo."]
        R5["vircadia-world/ — server/ only"]
    end
    subgraph stub["Gitignored symlinks — .gitignore:238-240, NOT submodules"]
        S1["Kokoros symlink, dangling here, .gitignore:238"]
        S2["Whisper-WebUI symlink, dangling here, .gitignore:239"]
        S3["xinference symlink, dangling in this container<br/>live consumers: docker-compose.unified.yml:318,<br/>agentbox/docker-compose.yml:108"]
    end
    subgraph ext["EXTERNAL — not on disk in any form"]
        E1["EXTERNAL: nostr-rust-forum"]
        E2["EXTERNAL: dreamlab-ai-website"]
        E3["EXTERNAL: VisionFlow canon"]
        E4["EXTERNAL: solid-pod-rs — consumed as a crates.io pin<br/>and as an agentbox Nix pin, never as a checkout"]
    end

    WARN["INVARIANT for this diagram tree — a claim about an EXTERNAL<br/>repo may only assert what THIS repo's code or docs state.<br/>Nothing about their internals is asserted here."]
    D1["DOC-DRIFT — these three are NOT submodules and are absent from<br/>.gitmodules. They are untracked symlinks to host paths<br/>(.gitignore:238-240), dangling in this container. git ls-files<br/>returns nothing for any of them. xinference nonetheless has live<br/>compose consumers, so its absence is a broken link, not an<br/>unused stub. see VC-35.12 for the Kokoros/Whisper half."]
    D2["DIVERGENCE — loom/README.md records that a second Python<br/>implementation (app/{loom_facade,ontology_proxy,<br/>ontology_scaffold,loom_graph}.py, 1,727 lines) was DELETED<br/>2026-09-03 as a dead twin of the Rust facade. see ES-06.6"]

    stub --> D1
    R4 --> D2
    ext --> WARN
```

## ES-01.7 The MCP hub is a boot-order dependency, and the wait is bounded
```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord program block<br/>agentbox/flake.nix:2532
    participant MAIN as Hub subcommand<br/>agentbox/services/agentbox-mcp/src/main.rs:46
    participant WAIT as wait_for_config<br/>agentbox/services/agentbox-mcp/src/hub/mod.rs:252
    participant SERVE as serve<br/>agentbox/services/agentbox-mcp/src/hub/mod.rs:275
    participant MAN as agentbox.toml resources.mcp_hub<br/>agentbox/agentbox.toml:1171

    SUP->>MAIN: agentbox-mcp hub --config /run/agentbox/mcp-hub.json --bind, flake.nix:2533
    MAIN->>SERVE: hand the wait budget over, main.rs:97-100
    Note over MAIN: The budget is a CLI argument with a default of<br/>600 seconds, main.rs:55-56. It is a BOUNDED wait,<br/>not an indefinite one.
    SERVE->>WAIT: poll for the projection, hub/mod.rs:280
    loop every 500 ms until the file exists
        WAIT->>WAIT: log progress every 15 s, hub/mod.rs:266-268
    end
    alt the projection never arrives
        WAIT-->>SERVE: bail naming the path and asking whether the<br/>bootstrap program is projecting it, hub/mod.rs:260-265
    else the projection is there
        SERVE->>MAN: load the config and read the server list
        MAN-->>SERVE: nine hub-routed servers, agentbox.toml:1175-1178
        SERVE->>SERVE: refuse any non-loopback bind, hub/mod.rs:283-284
        SERVE->>SERVE: bind the listener and register each child,<br/>hub/mod.rs:287,291
    end
    Note over SUP,MAN: INVARIANT — the hub is loopback only and refuses to<br/>start on any other bind, hub/mod.rs:284. The manifest<br/>pins 127.0.0.1 and the flake reads that value through,<br/>agentbox.toml:1173 and flake.nix:155.
    Note over SUP,WAIT: DEBT — at this revision the supervisor pairs<br/>autorestart with startsecs=2, flake.nix:2540, so the<br/>600-second bail is a restart rather than a park. A<br/>missing projection therefore reads RUNNING while no<br/>port is bound. ADR-2104 is the record that closes this.
    Note over MAN: Nine servers ride the hub, so this one program is the<br/>first check when several MCP servers refuse connections<br/>at once, agentbox.toml:1175-1178. see AB-09
```

## ES-01.8 The four governing documents and the proposed settlement sections they now carry
```mermaid
flowchart TB
    subgraph LIVE["Live compliance surface — unchanged by the settlement pack"]
        BC["BASELINE-container 0.4.0<br/>agentbox/docs/BASELINE-container.md:4"]
        IG["INGRESS-identity 0.2.0<br/>agentbox/docs/INGRESS-identity.md:4"]
        GC["GOVERNANCE-capabilities 0.6.0<br/>agentbox/docs/GOVERNANCE-capabilities.md:4"]
        PR["PROTOCOL-registry, proposed governing surface<br/>agentbox/docs/PROTOCOL-registry.md:3"]
    end

    PRD["PROPOSED — PRD-024 sovereign settlement<br/>agentbox/docs/developer/economy-loop.md:249<br/>the chain is the sole value instrument"]

    PRD -->|"sidechain manifest block, sidestr programs,<br/>rust-bitcoin accepted, three proposed invariants<br/>BASELINE-container.md:8"| BC
    PRD -->|"three domain-separated keys, kind 38110,<br/>a second Multikey in the DID document<br/>INGRESS-identity.md:9"| IG
    PRD -->|"every settlement passes payment_settlement,<br/>durable budget, fail-closed, anchor-not-seal<br/>GOVERNANCE-capabilities.md:8"| GC
    PRD -.->|"the kind table gains the chain-plane kinds"| PR

    RET["RETIRED — Lightning-first is superseded. x402 and l402<br/>classify but stay payable false permanently, and Lightning<br/>may return only as a bridge on-ramp<br/>agentbox/docs/developer/economy-loop.md:143"]
    PRD --> RET

    INV["INVARIANT — every one of these amendments is recorded in a<br/>clearly marked PROPOSED section and the Invariants compliance<br/>surface above it is unchanged. BASELINE-container.md:8,<br/>INGRESS-identity.md:9 and GOVERNANCE-capabilities.md:8 each<br/>say so in their own changelog entry. Nothing here is live."]
    BC --> INV
    IG --> INV
    GC --> INV

    LEDG["DIVERGENCE — the estate runs THREE independent did:nostr-keyed<br/>sats ledgers and none of them is synced with the others. The<br/>settlement pack is the first record to enumerate them in one<br/>place. see ES-03.9 and ES-08"]
    PRD --> LEDG
```
