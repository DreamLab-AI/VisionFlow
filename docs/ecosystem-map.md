# VisionFlow Ecosystem Map

**Status:** Docs plus targeted code/manifest spot-checks, updated with runtime research 2026-05-22; three provenance loops runtime-verified 2026-05-30 and **merged to VisionClaw main** (see Recently Resolved). Register reconciled against code 2026-07-03.
**Date:** 2026-05-22, register reconciled 2026-07-03
**Scope:** Repositories mounted under `/home/devuser/workspace`

> **Superseded by the closeout.** This Gap Register (G1–G8) is retained for history but is **superseded by [`docs/closeout/final-design.md`](closeout/final-design.md)** (2026-07-03), which re-audited all nine repo slices against code and found this register stale in both directions. Where a gap here disagrees with the closeout, the closeout wins. The 2026-07-03 pass corrected the merged-provenance, mesh-tally, beam/dispatcher and open-question entries below.

This document maps the VisionFlow ecosystem from repository documentation, with targeted implementation checks for the integration claims that affect cross-repo alignment. When a gap is listed, it means the docs or spot-checked code/manifests identify an absent, deferred, or inconsistent counterpart.

## Source Set

| Path | Interpreted repository | Documentation used |
|---|---|---|
| `../VisionFlow` | VisionFlow | README, website PRD/ADR/DDD |
| `../project` | VisionClaw mount | README, PRD-010/014/015, ADRs, DDD/integration research docs |
| `../project/agentbox` | agentbox | README, developer ecosystem/identity/sovereign mesh docs, relay/pod bridge code |
| `../solid-pod-rs` | solid-pod-rs | README, ecosystem integration, parity/gap analysis |
| `../nostr-rust-forum` | nostr-rust-forum | README, architecture, consumer surface map, ADR-086/087/089/093 |
| `../dreamlab-ai-website` | DreamLab deployment | README, deployment and forum-consumer docs |

## System Shape

VisionFlow is not a single executable. It is a coordination architecture that emerges from five product substrates and one branded deployment:

| Substrate | Primary responsibility | Boundary |
|---|---|---|
| VisionFlow | Ecosystem narrative, public positioning, repository map, shared coordination model | Documentation and website surface |
| VisionClaw | Knowledge engineering: OWL 2 EL reasoning, GPU graph physics (82 CUDA kernels / 9 `.cu` files / 5,854 LOC), XR, native MCP ontology tools, IS-Envelope spec owner (ADR-075), governance (main runs the inline decide/inbox handler + ADR-110 ElevationActor; the distributed BrokerActor lives on the unmerged `crashbug` branch), embodied agent-loop renderer (beam shipped over `/wss/agent-events`, ADR-059; gluon deferred) | GPU host / graph backend / semantic workbench / embodiment surface |
| agentbox | Reproducible sovereign agent runtime with Nix, 115 skills, 180+ MCP tools, 12-tool ontology bridge to VisionClaw SPARQL, browser setup wizard, privacy filtering, Nostr/Solid identity | Agent container and harness |
| solid-pod-rs | Solid/JSS foundation: LDP, WAC, NIP-98, DID:Nostr, WebID, OIDC, git pods | Shared protocol library and native pod server |
| nostr-rust-forum | Governance UI and relay kit: passkey auth, Cloudflare Workers, Agent Control Surface events | Human decision surface and Nostr relay edge |
| dreamlab-ai-website | Branded DreamLab deployment and operator overlay for the forum kit | Public site and Cloudflare deployment config |

The common primitive is `did:nostr:<hex-pubkey>`. Docs consistently describe it as the identity used for relay authentication, HTTP request signing, WAC ACL subjects, provenance, and cross-substrate routing.

## Core Flows

### Human-Governed Agent Action

1. An agent in agentbox publishes an Agent Control Surface event, normally one of kinds `31400-31405`.
2. The Nostr relay mesh routes that event to the forum.
3. The forum renders the event as a governance panel or action request.
4. A human signs an approval/rejection response.
5. VisionClaw's governance and related enrichment/write-back flows use the decision as the control point before mutation. The write-back endpoint `POST /api/enrichment-proposals/{id}/decide` is **merged to main** (`023c847b0`, gated `72bd6ec05`, hardened `bed156583`): on an attributed approval it performs a real fenced Oxigraph write (`append_derived_summary`) and flips `writeback_committed`. Agent-side application of forum-originated decisions is implemented (`handleGovernanceDecision`); the residual gaps are agentbox-*originated* cases (closeout GOV-3) and the missing `ConceptElevated` closing event (GOV-2).

### Sovereign Data Access

1. A user or agent signs an HTTP request using NIP-98.
2. A pod tier verifies the signature against the same secp256k1 pubkey used by `did:nostr`.
3. WAC evaluates access against that identity.
4. The resource is served from either Cloudflare Workers storage or a native `solid-pod-rs` server.

### Knowledge Ingestion and Provenance

1. VisionClaw ingests knowledge from Logseq/GitHub and pod-backed sources.
2. OWL reasoning and graph physics derive semantic structure.
3. Agents propose enrichments or actions.
4. Human decisions gate mutation.
5. Provenance beads and URNs preserve attribution.

### Embodied Agent Loop (voice → action → pod → KG → ontology → visualisation)

The flagship cross-substrate journey, formerly absent from this canon. A human speaks, an agent
forms intent and acts on a sovereign Solid pod and personal knowledge graph, the action is visibly
embodied in the VisionClaw GPU/XR graph, the KG mutates, and high-value personal concepts are
elevated into the shared ontology under governance — all federated over the private Nostr relay
mesh. Substrate roles: agentbox runs the agents; VisionClaw renders the embodiment; solid-pod-rs
stores sovereignly; the forum and website provide governance/operator surfaces; VisionFlow is the
canon.

1. **Voice → selected actor (VisionClaw).** A push-to-talk command (PTT → Whisper STT → Kokoro
   TTS) captures the currently-selected agent node and dispatches a *scoped* agent command rather
   than a generic swarm intent.
2. **Intent → agentbox.** The scoped command is published as a signed ACSP `ActionRequest`
   (kind 31402) addressed to the target agent's `did:nostr`.
3. **Action on the pod (solid-pod-rs).** The agent writes the personal KG to the user's pod **as
   itself**, under a scoped, revocable WAC `acl:agent` mandate (`urn:agentbox:mandate`), with a
   per-request signed NIP-98 header — never by holding the user's nsec.
4. **Embodiment (VisionClaw).** The agent's action crosses the federation boundary as a canonical
   `notifications/agent_action` event over the `/wss/agent-events` WebSocket and renders as a
   **beam** (transient coloured edge, agent → target). The **beam actor is shipped and wired at boot**
   (`AgentBeamActor`, ADR-059 Phase 2b). The **gluon** (the attractive spring force that same
   transient edge would exert) is **deferred** — a documented no-op today (`gluon_deferral_note()`);
   no incremental GPU transient-edge path exists yet.
5. **Elevation under governance.** A BC22 extractor reads the pod KG and proposes candidate
   concepts through VisionClaw's governed pipeline: human approval (policy) → GitHub PR. The
   documented terminus `→ merge → ConceptElevated` is **not yet implemented**: `ElevationActor`
   opens the PR at creation and emits no `ConceptElevated` event (grep returns zero across every
   repo), and the Whelk EL++ consistency gate is not invoked before the human-approval case is
   opened (closeout GOV-2/GOV-7). The ungoverned `/api/ontology/load` backdoor is closed.
6. **Continuous provenance.** Identity is preserved end to end by the BC20 anti-corruption layer
   (`urn:agentbox:activity` ⇄ `urn:visionclaw:execution`, `agent` ⇄ `did:nostr`), with
   `owner_did` constant at every hop.

Contracts: agentbox **ADR-014** + VisionClaw **ADR-059** (the `/wss/agent-events` channel),
agentbox **ADR-026** (cross-substrate seams), agentbox **PRD-014** (driving spec). As of
2026-05-29 the action-signal seam is wired and verified end-to-end (Phase 2a): canonical producer
on the agentbox side, authenticated ingest + broadcast hub on the VisionClaw side. The Phase-2b
**beam render actor is shipped and wired at boot**; the gluon force remains deferred. As of
2026-05-30 the VisionClaw-side ingest is runtime-verified (WS-probed live over `/wss/agent-events`,
with BC20 provenance stamped on the hot path), upgrading this seam from cargo-verified to
runtime-verified, and these commits are now **merged to main** (see Recently Resolved). Identity rides the JSON ingest envelope (the `0x23` binary frame
is identity-blind by design). The legacy MCP-TCP `:9500` path carries agent **state** snapshots, a
payload distinct from the agent **action** push, and is retired in favour of the one socket.

## Gap Register

### G1: The Umbrella Docs Are Sparse — RESOLVED

**Resolved (2026-07-03).** The referenced entry-point docs now exist in the checkout: `docs/architecture/repository-map.md`, `docs/protocol/identity-spine.md`, `docs/roadmap.md` and `docs/releases/README.md` are all present. The original premise (canonical technical reference missing) is stale.

**Residual:** a maintenance/cross-linking concern only — keep this document, the repository map, and the identity spine cross-linked as the top-level docs entry points.

### G2: Mesh Federation Is Scaffold — Standalone-First Is the Supported Mode (FROZEN)

**Corrected (2026-07-03).** The earlier tally was wrong. **Three of four runtime substrates default standalone** (agentbox `[mesh] mode="standalone"`, solid-pod-rs standalone with an embedded NIP-01 relay, and nostr-rust-forum — `forum.example.toml` and `wrangler.toml` both default `standalone`/`d1`). Only dreamlab-ai-website fans out. The federated-by-default tally is **1/4, not 2/4**.

The forum's `nostr-bbs-mesh` crate is **scaffold-only** (no `MeshTransport` implementation, not even a dependency of the relay-worker; lands Sprint v12+). The relay advertises `auth_required:false` and gates by pubkey whitelist — the claimed **NIP-42 gate is false**. **IS-Envelope routing (ADR-075) is unimplemented in every substrate** (only conformance vectors exist). solid-pod-rs "native mesh in alpha.15" is not a real feature.

**Closeout decision (FREEZE):** declare **standalone-first** as the supported deployment mode. Mesh federation (ADR-073/PRD-010, the forum-kit convergence cluster, IS-Envelope routing) is *designed, not shipped* and parked. The IS-Envelope spec and vectors remain canonical.

**Next action:** the compatibility matrix records mesh as "designed, not shipped". Do not list peer discovery, NIP-42 gate, or IS-Envelope routing as operational until a `MeshTransport` impl is wired into `relay_do`.

### G3: Protocol Implementations Are Duplicated

VisionClaw PRD-015 identifies cross-substrate duplication in NIP-98 auth, Solid pod clients, DID:Nostr resolution, WAC/ACL evaluation, Nostr key management, and URN minting.

**Impact:** Security fixes and protocol behavior can diverge across repos. This is especially risky for auth, replay protection, and DID verification.

**Next action:** Promote shared crates/contracts for NIP-98, DID:Nostr, WAC/pod client behavior, and ecosystem event types.

### G4: VisionClaw Has Documented Integration Debt

VisionClaw docs identify several still-important risks: URI resolver redirects to missing routes, historical multi-keypair drift, missing or incomplete NIP-42 support, auth hardening gaps, dead/stub code, and parallel service implementations.

**BC20 update (2026-05-29):** the BC20 anti-corruption layer was flagged "paper-only" in PRD-010.
That is now **resolved** — BC20 is real, owned, bidirectional code. The executable contract lives
in agentbox (`management-api/lib/bc20-provenance-bridge.js`, 20 tests) with a closed kind map
(`activity` ⇄ `execution`, `agent` ⇄ `did:nostr`, `thing` ⇄ `kg`, `memory` ⇄ `concept`) and a
durable `UrnMapping` that round-trips identity with zero loss; VisionClaw mirrors the canonical
ingest schema (`src/agent_events/schema.rs`) and consumes the pushed events over the authenticated
`/wss/agent-events` socket (Phase 2a, cargo-verified). See agentbox ADR-026 D1 and the Embodied
Agent Loop core flow above.

**Merged to main (verified 2026-05-30, merged since):** the `urn:visionclaw` minter (`src/uri/mod.rs`,
`afb072cfa`) now exists and runs — it was previously absent on main, blocking native minting of
crossed URNs; and the broker write-back endpoint (`POST /api/enrichment-proposals/{id}/decide`,
`023c847b0`) is live (HTTP 200, PROV-O provenance), closing the 404 that made the resolver redirect
to a missing route. WS ingest provenance (`src/agent_events/ingest.rs`, `2d94cd2fc`) is
runtime-verified on the hot path (see Recently Resolved). All three are on main with clean working trees.

**Resolved since (2026-07-03 reconciliation):**
- **Beam render actor — shipped.** `AgentBeamActor` (ADR-059 Phase 2b) is started at boot
  (`app_state.rs:577`) and fans out `0x23` frames. Only the **gluon** attractive-force sub-feature
  remains a deferred no-op.
- **ACSP 31402 dispatcher — shipped.** `AcspClient` signs and publishes kind-31402 requests, wired
  via `ElevationActor`. The framing that `AgentActionEnvelope` "is to be retired" was **incorrect**:
  ACSP governance panels (31400–31403) and the binary `AgentActionEnvelope` embodiment frame are
  distinct concerns and **coexist by design**.
- **Agent-side decision application — implemented.** `handleGovernanceDecision` applies
  forum-originated decisions with PROV-O URNs; the decide endpoint performs a real Oxigraph write.

**Still open in VisionClaw:** the **gluon** force + despawn reaper (ADR-059 §4/2b), **did:nostr-keying
of agent-actor nodes** (now **live-polled** from the management API — an improvement over "mock-polled"
— but keyed by `task_id`, not `did:nostr`, so a node cannot yet be addressed by ACSP 31402), the
`ConceptElevated` closing event, the personal-vs-shared (owner) node distinction (owner_did exists at
the URN/enrichment layer but not on the render-layer graph `Node`), agent-side application of
**agentbox-originated** cases (closeout GOV-3), and the Whelk EL++ consistency gate before elevation
(GOV-7). Tracked in agentbox PRD-014 §3 Seam E and the closeout register.

**Impact:** VisionClaw is the semantic center of the ecosystem, so gaps there block end-to-end provenance and governance flows.

**Next action:** Reconcile PRD-010/014/015 status against current code and update docs with completed versus open items.

### G5: Cloudflare Workers Portability Drives Duplication — Core Extraction SHIPPED

**Resolved for pure-logic surfaces (2026-07-03).** The decision is made and the extraction has happened: solid-pod-rs ships a no-Tokio `core` feature (`Cargo.toml` `core=[std, dep:js-sys, did-nostr-types]`), and nostr-rust-forum consumes the **published `solid-pod-rs 0.5.0-alpha.3`** with `default-features=false, features=[core]` (Cargo.lock pinned). The wac/webid/did pure-logic surfaces are shared, not reimplemented.

**Still open:** NIP-98 remains triplicated (solid-pod-rs, `nostr-bbs-core` which adds D1 replay, and the forum client) — the D1 replay store keeps the forum verifier edge-local, so a single shared NIP-98 verifier is not yet achieved (G3). Server-framework-bound surfaces stay two-tier.

**Next action:** move the replay-store abstraction into solid-pod-rs `core` so all tiers can share one NIP-98 verifier, or accept the documented edge-local exception.

### G6: Two-Tier Pods Add Operational Complexity

nostr-rust-forum ADR-093 defines a hybrid pod architecture: Cloudflare Workers pods for edge LDP/R2 and native agentbox-hosted `solid-pod-rs-server` pods for git/app capabilities.

**Impact:** Users can land on different pod tiers with different capabilities. Operators must manage native provisioning, Cloudflare Tunnel, PSK rotation, and tier-aware WebID routing.

**Next action:** Maintain a pod tier matrix and a migration story for users moving from CF-tier pods to native git-capable pods.

### G7: agentbox Mesh Is Real but Still Needs Runtime Status Reconciliation

agentbox docs describe the sovereign relay, pod inbox bridge, identity root, and Solid pod integration as core architecture. Spot checks found concrete implementation surfaces, including `mcp/nostr-bridge/relay-consumer.js`, `management-api/adapters/pods/local-solid-rs.js`, `management-api/routes/broker-bridge.js`, and manifest settings for governance kinds `31400-31405`. VisionClaw PRD-010 and earlier ecosystem audits still call out relay exposure, boot wiring, allowlist, and identity bootstrap issues as historical or unresolved gaps depending on deployment mode.

**Impact:** agentbox is not merely aspirational, but the docs still need a single status table that distinguishes shipped loopback/private-mesh behavior from full federated operation.

**Next action:** Add an agentbox mesh status table: embedded relay exposure, bridge boot wiring, NIP-42 support, allowed pubkey source, DID document relay endpoint, and multi-agent identity status.

### G8: DreamLab Website Is a Consumer Overlay, Not a Protocol Source

dreamlab-ai-website owns branding, static React pages, Cloudflare config, and forum operator overlay. It consumes nostr-rust-forum rather than owning forum source.

**Impact:** Deployment behavior depends on kit pinning, workflow clone behavior, Cloudflare resource mapping, and config compatibility.

**Next action:** Pin and publish a kit compatibility record for every production deployment.

## Recommended Roadmap

| Priority | Work | Why |
|---|---|---|
| P0 | Security/auth status reconciliation across PRD-014 gaps | Prevents publishing an architecture that implies stronger guarantees than deployed systems provide |
| P0 | Cross-substrate compatibility matrix | Makes mesh readiness explicit and testable; see [Compatibility Matrix](architecture/compatibility-matrix.md) |
| P1 | Shared protocol contracts for NIP-98, DID:Nostr, WAC, IS-Envelope | Reduces drift and duplicated security-sensitive logic |
| P1 | End-to-end governance smoke test: agentbox -> relay -> forum -> VisionClaw -> pod/provenance | Validates the central VisionFlow claim |
| P1 | End-to-end embodied-loop smoke test: voice (VisionClaw) -> ACSP ActionRequest -> agentbox actor -> signed NIP-98 pod write -> personal-KG node -> ACSP elevation prompt -> governed PR -> ConceptElevated, with one continuous `urn:agentbox:activity` ⇄ `urn:visionclaw` provenance chain | Validates the flagship cross-substrate journey (PRD-014 §7) |
| P1 | solid-pod-rs CF Workers portability decision | Determines whether duplication is temporary or permanent |
| P2 | Coordinated release/version policy | Lets consumers know which repo versions work together; see [Roadmap](roadmap.md) |
| P2 | Unified operations docs: health, backup, DR, pod tiers, relay status | Moves the ecosystem from impressive components to operable system |

## Recently Resolved

- **Three provenance loops merged to main (verified 2026-05-30, merged since).** Three
  previously paper/cargo-only seams were exercised against a running VisionClaw backend and are now
  **merged to VisionClaw main with clean working trees** (no longer "local/pending"):
  - **`urn:visionclaw` minter** (`src/uri/mod.rs`, `afb072cfa`, +13 tests) — was absent on main; now mints
    typed `concept`/`kg`/`bead`/`execution`/`group` URNs + `did:nostr`. Runtime mint observed:
    `urn:visionclaw:execution:sha256-12-44ec4693df02` (sha256-12 byte-equal to the agentbox minter).
  - **Broker write-back endpoint** `POST /api/enrichment-proposals/{id}/decide`
    (`src/handlers/enrichment_proposals_handler.rs`, `023c847b0`, gated `72bd6ec05`, hardened `bed156583`) —
    was **404**; now returns **HTTP 200**, and on an *attributed* approval performs a real fenced
    Oxigraph write (`append_derived_summary`) and flips `writeback_committed`, in addition to
    minting PROV-O provenance and broadcasting an `enrichment_decision` WS event.
    Unattributed (non-hex pubkey) payloads → `attributed:false` (recorded, not written).
  - **WS ingest BC20 provenance on the hot path** (`src/agent_events/ingest.rs` +
    `provenance.rs`, `2d94cd2fc`, +12 tests) — `process_frame()` now records provenance, crosses foreign
    `urn:agentbox:*` via `uri::cross_from_agentbox`, and stamps Signed/Malformed/Anonymous on
    `IngestOutcome::Published`. (Note: "Signed" is stamped from a structural pubkey-hex check, not a
    signature verification — closeout T4/security; treat as "asserted" until NIP-26 verification lands.)
- **Beam render actor + ACSP 31402 dispatcher shipped (2026-07-03 reconciliation):** `AgentBeamActor`
  (ADR-059 Phase 2b) is wired at boot and the `AcspClient` 31402 dispatcher is live via `ElevationActor`.
  Gluon force remains deferred; `AgentActionEnvelope` coexists with ACSP by design (not retired).
- **did:nostr canonicalisation answered by ADR-125:** VisionClaw ADR-125 (`a579bf353`, 2026-06-15)
  ratifies a single canonical Multikey/`publicKeyMultibase` DID-document form re-converging
  forum/agentbox/VisionClaw/solid-pod-rs — resolving Open Question 1 (see below).
- **IS-Envelope canonical ownership:** Resolved. VisionClaw owns the spec (ADR-075, JSON Schema, test vectors). Runtime consumers: agentbox, solid-pod-rs, nostr-rust-forum consume the vectors, but **envelope routing over a mesh transport is not implemented in any substrate**. Event kind registry remains unowned (agentbox federates 38300–38304; the forum relay drops everything above 38100).
- **BC20 anti-corruption layer (2026-05-29):** Resolved from "paper-only" to real, owned, bidirectional code. Agentbox holds the executable contract (`bc20-provenance-bridge.js`); VisionClaw mirrors the ingest schema and consumes pushed `agent_action` events over the authenticated `/wss/agent-events` socket (Phase 2a, cargo-verified). The beam+gluon render is the remaining Phase-2b increment. See agentbox ADR-026 D1 / PRD-014.
- **Embodied agent-loop documentation (2026-05-29):** Resolved the canon silence on voice ingress and personal→shared elevation flagged in earlier audits (PRD-014 X7) — the Embodied Agent Loop core flow is now documented above, and the BC20 namespace grammar is in agentbox's `CLAUDE.md` and ecosystem docs.

## Open Questions

1. ~~Is `did:nostr` resolution canonicalized in solid-pod-rs, forum core, or a new shared crate?~~ **Answered (ADR-125).** `did:nostr` is canonicalised on the Multikey form across all four substrates by VisionClaw ADR-125 (`a579bf353`, 2026-06-15), correcting the earlier ADR-074 2019-suite target. Residual: solid-pod-rs owns DID-document rendering, but VisionClaw and the forum still have independent handling paths — promote `solid_pod_rs::did_nostr_types` (or a `no_std` shared crate) as the single resolver/minter (closeout contract:identity).
2. Are Cloudflare Workers pods and native pods expected to converge, or remain separate tiers?
3. What is the minimum supported deployment: single operator, team, or cross-organization federation?
4. What exact repo/version set defines the current production DreamLab deployment?
