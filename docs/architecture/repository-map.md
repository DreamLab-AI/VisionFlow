# Repository Map

**Status:** Docs-only map
**Date:** 2026-05-20

VisionFlow is a federated ecosystem. No single repository contains the whole runtime.

| Repository | Local path | Role | Primary docs entry |
|---|---|---|---|
| VisionFlow | `../VisionFlow` | Ecosystem guide, public website, coordination architecture | `README.md`, `docs/ecosystem-map.md` |
| VisionClaw | `../project` | Knowledge engineering, OWL + SHACL reasoning, W3C PROV-O provenance, GPU graph physics, XR, MCP tools, Judgment Broker, embodied agent-loop renderer (beam + gluon over `/wss/agent-events`, ADR-059) | `README.md`, `docs/PRD-010-did-nostr-mesh-federation.md`, `docs/PRD-014-ecosystem-productionisation.md`, `docs/PRD-015-ecosystem-code-hygiene.md` |
| agentbox | `../project/agentbox` | Sovereign agent runtime, Nix container, skills/tools, Solid pod and Nostr bridge | `README.md`, `docs/developer/ecosystem.md`, `docs/developer/identity-mesh.md` |
| solid-pod-rs | `../solid-pod-rs` | Solid/JSS foundation library and server: LDP, WAC, NIP-98, DID:Nostr, git pods | `README.md`, `crates/solid-pod-rs/docs/explanation/ecosystem-integration.md`, `crates/solid-pod-rs/GAP-ANALYSIS.md` |
| nostr-rust-forum | `../nostr-rust-forum` | Forum kit, Cloudflare Workers, passkey auth, relay, governance UI | `README.md`, `docs/architecture.md`, `docs/consumer-surface-map.md` |
| dreamlab-ai-website | `../dreamlab-ai-website` | DreamLab branded deployment and operator overlay for forum kit | `README.md`, `forum-config/` docs |
| vowl-wasm | `../vowl-wasm` | Ontology visualisation WASM engine (clean-room Rust reimplementation of the VOWL notation, MIT, outside the AGPL boundary): OWL parsing, Barnes-Hut and SIMD force layout, render data; consumed by the corpus explorer; NGG1 and markdown-ontology paths feature-gated | `README.md` |
| knowledgeGraph | `../knowledgeGraph` | Published public knowledge graph corpus and build pipeline (ODbL-1.0 data, AGPL-3.0 pipeline); the OWL 2 TBox VisionClaw renders and the Loom grounds on; explorer frontend consumes vowl-wasm | `README.md` |
| visionGraph | `../visionGraph` | Authoring vault and publishing pipeline for narrativegoldmine.com; its `publish.yml` builds the corpus explorer against the published vowl-wasm bundle (the only live consumer) | `README.md`, `.github/workflows/publish.yml` |

## Dependency Direction

```mermaid
flowchart TB
    VF["VisionFlow\ncoordination docs"]
    VC["VisionClaw\nknowledge engineering"]
    AB["agentbox\nagent runtime"]
    SPR["solid-pod-rs\nSolid foundation"]
    NRF["nostr-rust-forum\nforum kit"]
    DLW["dreamlab-ai-website\nbranded deployment"]
    VW["vowl-wasm\nVOWL WASM engine (MIT)"]
    KG["knowledgeGraph\npublished corpus + pipeline"]
    VG["visionGraph\nauthoring vault + narrativegoldmine.com"]

    SPR --> VC
    SPR --> AB
    SPR --> NRF
    NRF --> DLW
    VW --> VG
    VW --> KG
    KG --> VG
    KG --> VC

    VC <-->|"Nostr relay mesh\nAgent Control Surface\nJudgment Broker"| AB
    VC <-->|"governance events\nhuman decisions"| NRF
    AB <-->|"agent events\npod inbox"| NRF

    VF -.-> VC
    VF -.-> AB
    VF -.-> SPR
    VF -.-> NRF
    VF -.-> DLW
    VF -.-> VW
    VF -.-> KG
    VF -.-> VG
```

## Ownership Rule

Protocol primitives should have one source of truth:

| Primitive | Preferred owner |
|---|---|
| Solid LDP, WAC, WebID, pod storage | solid-pod-rs |
| DID:Nostr document/resolution primitives | solid-pod-rs or a shared crate extracted from it |
| NIP-98 verification and replay contract | shared crate, using the most complete implementation as reference |
| Agent Control Surface kinds `31400-31405` | nostr-rust-forum, with schema consumed by agentbox and VisionClaw |
| Judgment Broker domain | VisionClaw |
| Operator deployment config | dreamlab-ai-website for DreamLab production; per-operator overlays elsewhere |
