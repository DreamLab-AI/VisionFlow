---
id: ES-01
title: Estate topology — substrates, network fabric, service ports, compose networks
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [agentbox:ADR-2139, agentbox:ADR-2151, agentbox:ADR-2023, agentbox:ADR-2013, visionclaw:ADR-2027, visionclaw:ADR-2119, visionclaw:ADR-2025, agentbox:ADR-2009, agentbox:ADR-2012, agentbox:ADR-2062, agentbox:ADR-2034, agentbox:ADR-2104, agentbox:ADR-2096, agentbox:ADR-2098, agentbox:ADR-2140, agentbox:ADR-2149, visionclaw:ADR-2143]
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
  - ../project/agentbox/docker-compose.isaac-sim.yml
  - ../project/agentbox/docker-compose.speech-ming.yml
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
  - ../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md
  - ../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
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
  - ../project/agentbox/config/entrypoint-unified.sh
  - scripts/estate-health/roster.json
verified_commit: {visionclaw: 8692b12ba987ecb87165566beb7c320a2446e4b4, agentbox: 6500582dfae23bf8a38a6942916d208154e43625, visionflow: 4d361b58ce2b3ee17e1570361eaf78270b630012}
---
## ES-01.1 Substrate map — the VisionClaw checkout's neighbourhood, not the whole estate
```mermaid
flowchart TB
    subgraph ONDISK["On disk in this checkout"]
        VC["VisionClaw (this repo)<br/>root: Cargo.toml, docker-compose.unified.yml"]
        AB["agentbox<br/>git submodule at agentbox/<br/>.gitmodules: url=github.com/DreamLab-AI/agentbox.git"]
        CS["codescape<br/>git submodule at codescape/ (ADR-2143 A5)<br/>.gitmodules:4-6: url=github.com/DreamLab-AI/codescape"]
    end
    VC -->|"embeds as submodule<br/>path=agentbox, .gitmodules:1-3<br/>no branch key — the pin is a bare gitlink"| AB
    VC -->|"second submodule, own repository and own Cargo workspace<br/>path=codescape, .gitmodules:4-6"| CS
    AB -.->|"VisionClaw pushed-head gitlink: e786197cc, an ancestor of the agentbox stamp 6500582df<br/>(only docs/ differs between them)<br/>git ls-tree of the pushed head, agentbox<br/>codescape gitlink: e3b76f63e"| ABPIN["submodule pin"]

    subgraph CRATE["Embedded as a Cargo dependency, not a checkout"]
        SPR["solid-pod-rs 0.5.0-alpha.17<br/>crates.io version, Cargo.toml:245<br/>feature solid-pod-embed (ADR-032 M3)"]
        SPRN["solid-pod-rs 0.5.0-alpha.18 workspace<br/>agentbox Nix pin, solid-pod-rs.nix:54<br/>rev 76ab7aab, RustCrypto 0.14 and SOLID_POD_JEV_ALLOW_CLOUD<br/>solid-pod-rs.nix:56-58"]
    end
    VC -->|"Cargo dep: fs-backend, nip98-schnorr,<br/>did-nostr, quota, rate-limit"| SPR
    AB -->|"lib/solid-pod-rs.nix pin<br/>supervised solid-pod program on port 8484<br/>native-pod-mesh.md:3"| SPRN

    DIVSP["DIVERGENCE re-opened 2026-10-30 — the estate's TWO solid-pod-rs pins no longer<br/>agree. VisionClaw still compiles crates.io 0.5.0-alpha.17 in-process<br/>(Cargo.toml:245); agentbox builds the 0.5.0-alpha.18 workspace from the<br/>fetchFromGitHub rev 76ab7aab (solid-pod-rs.nix:54,58) for the<br/>supervised port 8484 pod. The agentbox pin carries the<br/>SOLID_POD_JEV_ALLOW_CLOUD opt-in (solid-pod-rs.nix:56-58). The DOC<br/>lags too: native-pod-mesh.md:3-4 says Shipped at alpha.15 and that the Nix pin<br/>is v0.5.0-alpha.9, and its topology figure labels<br/>the server v0.4.0-alpha.17 (native-pod-mesh.md:30) — DOC-DRIFT<br/>inside it. see ES-08.1"]
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
    NOTE2["DIVERGENCE — six is NOT the estate. The enumeration the tree<br/>treats as canonical is scripts/estate-health/roster.json, which<br/>carries EIGHTEEN rows and is the only one walked row by row<br/>(roster.json:3, rows at roster.json:5-22); sidestr-rs is row nine<br/>(roster.json:13). This map omits knowledgeGraph, sidestr-rs,<br/>visionGraph, vowl-wasm, loom, WasmVOWL, ontomark, ontomark-obsidian,<br/>visiontouch-rs, prose-sanitiser, diagram-ir and dream-engine.<br/>Those edges are drawn in ES-11; the full<br/>enumeration conflict is catalogued in VF-08. Kept at six here<br/>because this diagram answers what is ON DISK in the VisionClaw<br/>checkout, which is a different question from what the estate is."]
    NOTE1 -.-> VF
    NOTE2 -.-> VF
```

## ES-01.2 Network / compute fabric — gateway host, Loom-only connected node, retired-address trap
```mermaid
flowchart LR
    subgraph ML["The gateway host (on the LAN)"]
        MLHOST["gateway host<br/>agentbox/skills/email-search/SKILL.md:79"]
        RECON["scripts/hp-loom-reconcile.sh, started in the background at every boot<br/>builds the committed vault HEAD and pushes the BUILT bundle<br/>agentbox/config/entrypoint-unified.sh:1218<br/>agentbox/agentbox.toml:1176-1184"]
    end
    subgraph HP["The connected node (downstream, no LAN IP) — a Loom host only"]
        HPHOST["connected node<br/>agentbox/skills/email-search/SKILL.md:80"]
        LOOMFACADE["Loom façade port 8084<br/>colocated with the model on the connected node<br/>agentbox/skills/email-search/SKILL.md:80"]
        LOOMMODEL["loom-model container port 8085<br/>Qwen3.8-27B, a swappable choice behind the façade<br/>agentbox/skills/email-search/SKILL.md:95"]
    end
    MLHOST -->|"point-to-point 25 G rail, the gateway's NAT service DNATs the façade<br/>agentbox/skills/email-search/SKILL.md:98"| HPHOST
    HPHOST --> LOOMFACADE
    LOOMFACADE -->|"port 8085 HTTP delegates to the model<br/>agentbox/skills/email-search/SKILL.md:81"| LOOMMODEL
    MLHOST -->|"embeddings port 9997 on the gateway host<br/>bge models on xinference<br/>agentbox/skills/email-search/SKILL.md:112"| XINF["xinference port 9997"]
    RECON -->|"ssh push of a verified generation, never the vault<br/>docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:57-65"| LOOMFACADE

    DEAD["RETIRED-ADDRESS TRAP — the old model host is dead<br/>agentbox/docs/adr/ADR-2023-loom-facade.md:55<br/>agentbox/skills/email-search/SKILL.md:98,212"]
    DEAD -.->|"never target — black-holes every synthesis<br/>while GET /health still answers"| LOOMFACADE

    GONE["RETIRED — the second agentbox that ran on the connected node, its docker-compose.hp.yml<br/>overlay and docs/developer/hp-peer-node.md were deleted by ADR-2140 D5, and its volumes<br/>and agentbox-core identity destroyed on 2026-10-13. No agentbox mesh peer remains.<br/>agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:75-81"]
    GONE -.-> HPHOST

    GEN["EXTERNAL and DOC-DRIFT resolved — commit 2899b3b7e generalised every literal estate<br/>address out of this public repository, so hostnames, the rail subnet and the retired IP are<br/>not stated in any cited source; the placeholders are the fact now, e.g. the dead host written<br/>as a retired address at ADR-2023-loom-facade.md:55. Literal values live only in the<br/>operator environment, which is not a source of this corpus."]
    GEN -.-> HPHOST
```

**What it shows:** the gateway host and the connected node after the node was cut back to a single role, serving the Loom. The model container sits behind the façade, a boot-time reconcile pushes a built vault generation to it, and the dead model address stays the trap.

**Why it is this way:** ADR-2140 keeps the vault canonical on the gateway host and lets the connected node hold only a built generation, so nothing there can drift from the vault (`../project/agentbox/docs/adr/ADR-2140-vault-canonical-here-hp-is-a-loom-host.md:57-60`). The earlier second agentbox, with its own `did:nostr` identity, is gone.

## ES-01.3 Service and port map — every published surface in the estate
```mermaid
flowchart TB
    subgraph vcstack["VisionClaw stack — docker-compose.unified.yml"]
        VCD["visionclaw_container<br/>profiles development, dev<br/>docker-compose.unified.yml:70,243-245"]
        VCP["visionclaw_prod_container<br/>profiles production, prod<br/>docker-compose.unified.yml:249,332-334"]
        LOOMB["loom-sidecar<br/>profile loom, docker-compose.unified.yml:383,453-454"]
        CFT["cloudflared-tunnel<br/>profile tunnel ONLY (ADR-2119)<br/>docker-compose.unified.yml:340,358-359"]
        CSV["codescape (ADR-2143 A5, standalone repository)<br/>profile codescape, opt-in with CODESCAPE=1<br/>docker-compose.unified.yml:464,659-660"]
    end
    subgraph abstack["agentbox stack — agentbox/docker-compose.yml"]
        ABC["agentbox container<br/>agentbox/docker-compose.yml:32"]
        RPG["ruvector-postgres<br/>agentbox/docker-compose.yml:12"]
    end
    subgraph sidecars["agentbox sidecar overlays"]
        BC["browsercontainer"]
        GT["gui-tools-service"]
        VCON["voice-console"]
        XRR["xr-runtime"]
        AND["agentbox-android"]
        OM["openmed"]
        ISA["isaac-sim (ADR-2149, staged)<br/>agentbox/docker-compose.isaac-sim.yml:10"]
        MING["ming-tts / ming-asr (experimental)<br/>agentbox/docker-compose.speech-ming.yml:41,75"]
    end

    VCD -->|"port 3001 nginx<br/>docker-compose.unified.yml:225"| EXT1["host"]
    VCD -->|"port 4000 Rust backend<br/>docker-compose.unified.yml:226"| EXT1
    VCP -->|"port 3001 only<br/>docker-compose.unified.yml:308"| EXT1
    LOOMB -->|"host port 8090 to container port 8080<br/>docker-compose.unified.yml:438"| EXT1
    CSV -->|"host loopback 127.0.0.1 port 4100 by default; port 4101 hook ingest stays on the compose network<br/>docker-compose.unified.yml:637-641"| LOOPBACK
    ABC -->|"port 9096 LAN — the ONLY 0.0.0.0 publish<br/>agentbox/docker-compose.yml:47"| EXT1
    ABC -->|"loopback ports 9090 9700 9091 8484 8888 5901 8080<br/>agentbox/docker-compose.yml:48-54"| LOOPBACK["loopback only"]
    RPG -->|"5432 internal"| ABC
    BC -->|"0.0.0.0 5903 VNC / 8931 MCP SSE<br/>host 9222 to container 9223 CDP"| EXT1
    GT -->|"0.0.0.0 5905 / 9876 / 9877, plus port 9881 DCC gateway on the container network only<br/>agentbox/docker-compose.gui-tools.yml:46-55"| EXT1
    VCON -->|"0.0.0.0 8443 / 8444"| EXT1
    XRR -->|"0.0.0.0 5904"| EXT1
    ISA -->|"no ports: block, port 9890 expose only<br/>agentbox/docker-compose.isaac-sim.yml:42-43"| LOOPBACK
    MING -->|"127.0.0.1 8911 and 127.0.0.1 8912<br/>agentbox/docker-compose.speech-ming.yml:47,81"| LOOPBACK
    AND -->|"127.0.0.1 5555"| LOOPBACK
    OM -->|"127.0.0.1 9093"| LOOPBACK

    INV["INVARIANT ADR-2013 — every compose publish binds 127.0.0.1<br/>unless it is on the SANCTIONED list. In the main agentbox<br/>compose only port 9096 is a LAN door. The System One proxy binds<br/>0.0.0.0:8094 inside the container for the docker networks and<br/>publishes nothing (agentbox.toml:893-894, sanctioned at<br/>check-ports-loopback.mjs:740-745). see ES-10.8"]
    D1["NOT A DEFECT — browsercontainer maps host 9222 to container 9223<br/>by design (docker-compose.browsercontainer.yml:51-53, CDP proxy<br/>host:9222 to socat:9223 to Chrome:9222). The ADR-2013 sanctioned<br/>entry names 9222 (host side) and agentbox/CLAUDE.md names 9223<br/>(container side) — both correct, easy to misread as a conflict."]
    D2["EXTERNAL — Loom port 8084 façade and loom-model port 8085 run on<br/>HP-Desktop, NOT in either compose file. xinference port 9997 and<br/>email-mcp-gateway port 8765 are likewise separate services on<br/>visionclaw_network, while the System One route is the agentbox proxy on port 8094, never published: the shared<br/>x-system-one-environment anchor gives VisionClaw, Loom and Codescape agentbox:8094 with<br/>JEV_ALLOW_CLOUD=1 (docker-compose.unified.yml:9,14,387,395,505) and the laya systemone:8097 pins are gone.<br/>CONFIGURED: it takes effect when the new agentbox image is activated and those services are recreated. see ES-01.2 and ES-06.1"]

    LOOPBACK --> INV
    BC --> D1
    LOOMB --> D2
```

**Open:** the System One proxy bind (`0.0.0.0:8094`, `network = true`) is configured in the manifest (`../project/agentbox/agentbox.toml:893-894`) and the host services are pointed at it (`../project/docker-compose.unified.yml:9,387,475`), but it takes effect only when the new agentbox image is activated and those services are recreated; the running container still dials loopback.

## ES-01.4 Docker networks and volumes — one external bridge joins both stacks
```mermaid
flowchart TB
    subgraph net["visionclaw_network — external bridge, declared in BOTH stacks"]
        N1["docker-compose.unified.yml:671-674<br/>external true, name ${EXTERNAL_NETWORK:-visionclaw_network}"]
        N2["agentbox/docker-compose.override.yml:166-168<br/>alias visionclaw, external true"]
    end
    subgraph vcvol["VisionClaw volumes — docker-compose.unified.yml:676-720"]
        V1["loom-data — mirrored corpus generation<br/>docker-compose.unified.yml:684-685"]
        V2["visionclaw-data / visionclaw-logs<br/>docker-compose.unified.yml:688,691"]
        V3["npm-cache / cargo-cache / cargo-git-cache / cargo-target-cache<br/>docker-compose.unified.yml:695,698,701,704<br/>codescape-target / -cargo-cache / -cargo-git / -data, kept apart, :709-718"]
    end
    subgraph abvol["agentbox volumes"]
        W1["ruvector-pg-data / ruvector-data<br/>agentbox/docker-compose.yml:176,178"]
        W2["solid-data / sovereign-identities / agentbox-secrets<br/>agentbox/docker-compose.yml:180,182,184"]
        W3["code-harness-data / agentbox-events / consultations-data<br/>agentbox/docker-compose.yml:186,188,190"]
        W4["hf-cache / codeserver-config / telemetry-data<br/>agentbox/docker-compose.yml:196,204,192"]
        W5["nostr-relay-data / tailscale-state, opencode-store, aoe-profiles<br/>agentbox/docker-compose.yml:206,208,198,194<br/>aoe-profiles is declared but unused: AoE is gone and the volume is kept for recovery (BASELINE-container.md:8)"]
        W6["codex-home (the whole ~/.codex, replacing a 512M tmpfs) and codex-packages<br/>mounted at agentbox/docker-compose.yml:147-148, declared :200-203"]
    end
    subgraph shared["Cross-container shared volumes"]
        S1["gui-tools-exchange — declared by the override AND by<br/>browsercontainer and gui-tools overlays. This is how the<br/>browser sidecar reads files this container writes."]
        S2["mad-workspace — EXTERNAL alias to<br/>multi-agent-docker_workspace, from the deprecated MAD stack"]
        S3["agent-workspace — EXTERNAL alias to the SAME<br/>multi-agent-docker_workspace volume, ADR-2114<br/>docker-compose.unified.yml:680-682. Mounted read-only at<br/>/vault in visionclaw_container, :222, so CorpusSource::<br/>LocalDirectory ingest reads the vault without GitHub creds"]
    end

    N1 --- N2
    net --> vcvol
    net --> abvol
    net --> shared

    D1["DIVERGENCE — mad-workspace and agent-workspace are TWO local<br/>aliases for the SAME external volume, multi-agent-docker_workspace<br/>(deprecated MAD stack). agentbox mounts it read-write for the full<br/>project tree; VisionClaw's own compose now mounts the identical<br/>volume read-only at /vault (ADR-2114). Migration path off the<br/>legacy name is agentbox.sh migrate-workspace, after which the<br/>override should reference agentbox-workspace instead (Q43)."]
    INV["INVARIANT — the cargo-target-cache and cargo-*-cache volumes are<br/>why the DEV container can compile Rust on startup rather than<br/>in the image build. see ES-09"]

    S2 --> D1
    S3 --> D1
    V3 --> INV
```

## ES-01.5 Compose overlay composition — which file adds what
```mermaid
flowchart LR
    BASE["agentbox/docker-compose.yml<br/>agentbox + ruvector-postgres"]
    OV["docker-compose.override.yml<br/>auto-applied — joins visionclaw_network,<br/>group_add 965 for the docker socket,<br/>mounts mad-workspace + gui-tools-exchange"]
    ISF["docker-compose.isaac-sim.yml<br/>opt-in headless physics sidecar, no ports (ADR-2149)"]
    SMF["docker-compose.speech-ming.yml<br/>experimental speech candidate, own compose project"]
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
    OV --> ISF
    OV --> SMF

    N1["Every overlay declares the same external network under the<br/>local alias visionclaw, so all sidecars share one bridge."]
    N2["group_add 965 is the docker socket gid — the container drives<br/>docker WITHOUT sudo, which no-new-privileges blocks."]
    D1["RESOLVED ADR-2013 — the voice overlay publishes port 8443 and<br/>port 8444 on 0.0.0.0 while the main compose publishes only port<br/>9096. Both voice doors sit on the CI-enforced SANCTIONED list<br/>(check-ports-loopback.mjs:95-96), beside the port 9096 ingress<br/>(:94) and the browser CDP door (:99) — decided exposures, not a<br/>breach. see ES-10.8"]
    D3["RESOLVED TENSION — the manifest and the stack now agree that the<br/>voice plane is UP. agentbox.toml:2209-2210 sets [voice] enabled = true<br/>and calls it sidecar state with its own lifecycle; the two doors that<br/>stack publishes are sanctioned in CI (check-ports-loopback.mjs:95-96).<br/>The earlier revision of this node recorded the opposite tension — the<br/>manifest then declared the plane OFF while the doors stayed sanctioned."]
    D2["TRAP — a build launched from INSIDE this container resolves bind<br/>paths against the HOST filesystem and silently bakes stale code.<br/>Build only from the host shell. see ES-09"]

    VL["NO OVERLAY — Voicelink (ADR-2151) adds no compose file, port or supervisor program:<br/>voice-paned is a per-connection process reached over the operator's ssh login<br/>agentbox/agentbox.toml:2215-2220"]

    OV --> N1
    OV --> N2
    VF --> D1
    VF --> D3
    BASE --> D2
    VF -.-> VL
```

**Tension:** ADR-2151 is `decision_status: proposed`, `implementation_status: none`, `activation_status: inactive` (`../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`), yet the running manifest sets `[voicelink] enabled = true` and `record_events = true` (`../project/agentbox/agentbox.toml:2226,2233`).

**Open:** Isaac Sim (ADR-2149) is `proposed`, `partial`, `staged`: the overlay exists but publishes no port and is gated by environment variables, not by the manifest (`../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:5-7,29`).

## ES-01.6 Repositories on disk — what is a checkout, what is a stub, what is external
```mermaid
flowchart TB
    subgraph real["Tracked WITH content at the repo root (git ls-tree at the stamp)"]
        R3["voice-stack/ — 2 tracked entries, README.md and console/<br/>git ls-tree at the stamp"]
        R4["loom/ — README.md + app/<br/>deployment notes only, NO implementation.<br/>The Rust loom-facade lives in the separate loom repo."]
        R5["vircadia-world/ — server/ only"]
        R6["codescape/ — a submodule gitlink, not tracked files<br/>.gitmodules:4-6, ADR-2143 A5"]
    end
    subgraph gone["Gitignored, not in the tracked tree at the stamp"]
        G1["JavaScriptSolidServer/ — .gitignore:22. legacy JS Solid server. see ES-08"]
        G2["nntp-stack/ — .gitignore:244"]
    end
    subgraph stub["Gitignored symlinks — .gitignore:240-242, NOT submodules"]
        S1["Kokoros symlink, dangling here, .gitignore:240"]
        S2["Whisper-WebUI symlink, dangling here, .gitignore:241"]
        S3["xinference symlink, dangling in this container<br/>live consumers: docker-compose.unified.yml:405,<br/>agentbox/docker-compose.yml:86"]
    end
    subgraph ext["EXTERNAL — not on disk in any form"]
        E1["EXTERNAL: nostr-rust-forum"]
        E2["EXTERNAL: dreamlab-ai-website"]
        E3["EXTERNAL: VisionFlow canon"]
        E4["EXTERNAL: solid-pod-rs — consumed as a crates.io pin<br/>and as an agentbox Nix pin, never as a checkout"]
    end

    WARN["INVARIANT for this diagram tree — a claim about an EXTERNAL<br/>repo may only assert what THIS repo's code or docs state.<br/>Nothing about their internals is asserted here."]
    D1["DOC-DRIFT — these three are NOT submodules and are absent from<br/>.gitmodules. They are untracked symlinks to host paths<br/>(.gitignore:240-242), dangling in this container. git ls-files<br/>returns nothing for any of them. xinference nonetheless has live<br/>compose consumers, so its absence is a broken link, not an<br/>unused stub. see VC-35.12 for the Kokoros/Whisper half."]
    D2["DIVERGENCE — loom/README.md records that a second Python<br/>implementation (app/{loom_facade,ontology_proxy,<br/>ontology_scaffold,loom_graph}.py, 1,727 lines) was DELETED<br/>2026-09-03 as a dead twin of the Rust facade. see ES-06.6"]

    stub --> D1
    R4 --> D2
    ext --> WARN
```

## ES-01.7 The MCP hub is a boot-order dependency, and ADR-2104 closed the false-RUNNING gap
```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord program block<br/>agentbox/flake.nix:3294
    participant MAIN as Hub subcommand<br/>agentbox/services/agentbox-mcp/src/main.rs:46
    participant WAIT as wait_for_config<br/>agentbox/services/agentbox-mcp/src/hub/mod.rs:252
    participant SERVE as serve<br/>agentbox/services/agentbox-mcp/src/hub/mod.rs:292
    participant MAN as agentbox.toml resources.mcp_hub<br/>agentbox/agentbox.toml:1596

    SUP->>MAIN: agentbox-mcp hub --config /run/agentbox/mcp-hub.json<br/>--bind --wait-config-secs 120, flake.nix:3295
    MAIN->>SERVE: hand the wait budget over, main.rs:101
    Note over MAIN: The budget is a CLI argument, default 120 seconds,<br/>main.rs:59-60, and the supervisor block passes 120<br/>explicitly, flake.nix:3295. A BOUNDED wait, not indefinite.
    SERVE->>WAIT: poll for the projection, hub/mod.rs:297
    loop every 500 ms until the file exists
        WAIT->>WAIT: log progress every 15 s, hub/mod.rs:283-285
    end
    alt the projection never arrives
        WAIT-->>SERVE: bail naming the path and asking whether the<br/>bootstrap program is projecting it, hub/mod.rs:275-281
    else the projection is there
        SERVE->>MAN: load the config and read the server list
        MAN-->>SERVE: eight hub-routed servers, agentbox.toml:1600-1603
        SERVE->>SERVE: refuse any non-loopback bind, hub/mod.rs:300-301
        SERVE->>SERVE: bind the listener and register each child,<br/>hub/mod.rs:308,305
    end
    Note over SUP,MAN: INVARIANT — the hub is loopback only and refuses to<br/>start on any other bind, hub/mod.rs:300-301 (ADR-2034 §2).
    Note over SUP,WAIT: RESOLVED ADR-2104 — startsecs=130, flake.nix:3301, is<br/>deliberately LONGER than the 120s wait budget, so a missing<br/>projection is now a FAILED START: autorestart=unexpected and<br/>startretries=2, flake.nix:3300,3302, park the program FATAL<br/>instead of the old autorestart=true + startsecs=2 reading<br/>RUNNING for days over an unbound port, flake.nix:3288-3293.
    Note over MAN: Eight servers ride the hub (was nine at an earlier<br/>revision), so this one program is the first check when<br/>several MCP servers refuse connections at once,<br/>agentbox.toml:1600-1603. see AB-09
```

## ES-01.8 The four governing documents and the proposed settlement sections they now carry
```mermaid
flowchart TB
    subgraph LIVE["Live compliance surface — unchanged by the settlement pack"]
        BC["BASELINE-container 0.5.13<br/>agentbox/docs/BASELINE-container.md:4"]
        IG["INGRESS-identity 0.2.4<br/>agentbox/docs/INGRESS-identity.md:4"]
        GC["GOVERNANCE-capabilities 0.6.11<br/>agentbox/docs/GOVERNANCE-capabilities.md:4"]
        PR["PROTOCOL-registry, proposed governing surface<br/>agentbox/docs/PROTOCOL-registry.md:3"]
    end

    PRD["PROPOSED — PRD-024 sovereign settlement<br/>agentbox/docs/developer/economy-loop.md:249<br/>the chain is the sole value instrument"]

    PRD -->|"sidechain manifest block, sidestr programs,<br/>rust-bitcoin accepted, three proposed invariants<br/>BASELINE-container.md:25"| BC
    PRD -->|"three domain-separated keys, kind 38420,<br/>a second Multikey in the DID document<br/>INGRESS-identity.md:13"| IG
    PRD -->|"every settlement passes payment_settlement,<br/>durable budget, fail-closed, anchor-not-seal<br/>GOVERNANCE-capabilities.md:19"| GC
    PRD -.->|"the kind table gains the chain-plane kinds"| PR

    FED["PROPOSED 2026-10-30 — ADR-2162 data-space federation section<br/>(VisionClaw ADR-2161): [mesh] sole reader, pinned peers,<br/>a node federation key and the dataspace URN crossing, not live<br/>agentbox/docs/PROTOCOL-registry.md:245"]
    FED -.->|"a second proposed section; the rows above it are unchanged,<br/>agentbox/docs/PROTOCOL-registry.md:247"| PR

    RET["RETIRED — Lightning-first is superseded. x402 and l402<br/>classify but stay payable false permanently, and Lightning<br/>may return only as a bridge on-ramp<br/>agentbox/docs/developer/economy-loop.md:143"]
    PRD --> RET

    INV["INVARIANT — every one of these amendments is recorded in a<br/>clearly marked PROPOSED section and the Invariants compliance<br/>surface above it is unchanged. BASELINE-container.md:25,<br/>INGRESS-identity.md:13 and GOVERNANCE-capabilities.md:19 each<br/>say so in their own changelog entry. The proposed sections<br/>are not live; only the interim testnet producer, mirror and<br/>faucet are now supervised under the sidechain gates,<br/>BASELINE-container.md:20,233, with no sidestr-node, bridge<br/>or port 9097 bind."]
    BC --> INV
    IG --> INV
    GC --> INV

    LEDG["DIVERGENCE — the estate runs THREE independent did:nostr-keyed<br/>sats ledgers and none of them is synced with the others. The<br/>settlement pack is the first record to enumerate them in one<br/>place. see ES-03.9 and ES-08"]
    PRD --> LEDG
```
