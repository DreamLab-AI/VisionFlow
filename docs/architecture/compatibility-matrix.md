# Ecosystem Compatibility Matrix

**Status:** Docs, metadata, code spot-check, and runtime research synthesis
**Date:** 2026-05-22

This matrix records the current compatibility posture verified by cross-repo runtime research (2026-05-22), not only sibling documentation.

| Area | VisionClaw | agentbox | nostr-rust-forum | solid-pod-rs | dreamlab-ai-website | Compatibility status |
|---|---|---|---|---|---|---|
| Identity | NIP-98, NIP-07, DID:Nostr, NIP-26 delegation, Solid/WAC documented; enterprise SSO remains a known future path | Bootstrap BIP-340 key becomes canonical `did:nostr`; accepted by relay, pods, WAC, URNs | Passkey PRF derives Nostr keys; NIP-98 across workers; NIP-42 relay | Foundation for DID:Nostr, NIP-98, Solid-OIDC, Tier 1/Tier 3 DID resolution | WebAuthn PRF + NIP-98; configured agent identities | Broad alignment on `did:nostr` + NIP-98; historical replay/parity concerns still need code recheck |
| Mesh | ADR-073 describes client/subscriber mode by default unless a local relay is added | Standalone default (`[mesh] mode = "standalone"`); embedded relay at `:7777` with pod-bridge; federated/client modes available | Federated NIP-05 is new default (`mode = "federated"`); `nostr-bbs-mesh`, NIP-42 gate, peer discovery, IS-Envelope routing | Standalone default with Tailscale tunneling; native mesh in alpha.15 | Federated architecture; CF Workers relay fan-out to public relays for censorship resistance | 2/4 runtime substrates (forum, website) default federated; 2/4 (agentbox, solid-pod-rs) default standalone. Mesh maturity varies by substrate. |
| Pod | Embedded `solid-pod-rs` at local `:8484`; WAC against `did:nostr` | Embedded Solid pod and native pod overlay via Cloudflare Tunnel | CF Worker pod tier mirrors Solid/JSS; native tier supports git via `solid-pod-rs-server` | Canonical LDP/WAC/WebID/auth/storage implementation | CF pod tier plus native agentbox pod tier for git | Strongest compatibility area; two-tier behavior is the main gap |
| Governance | Judgment Broker concept (doc-only, no runtime workbench); ontology governance; Agent Control Surface event kinds defined | Publishes and subscribes to kinds 31400-31405 via nostr-bridge; 10-tool ontology bridge proxies SPARQL to VisionClaw Oxigraph | Full implementation of kinds 31400-31405 in `governance.rs`; primary governance UI/API: agent registry, cases, roles, signed responses | Access-control foundation, not workflow owner | Governance dashboard at `/governance` renders kinds 31400-31405; NIP-98 signed responses; feature-gated | Agent Control Surface fully implemented in forum + website; agentbox publishes/subscribes. Judgment Broker is specification-only (no runtime). Ontology bridge is shipped cross-substrate integration. |
| Deployment | Docker Compose multi-profile and docs CI; productionisation PRDs still list hardening gaps | Nix-built multi-arch image, runtime contracts, image scan/SBOM, flake checks | Cloudflare Workers kit with CI/audit docs | Server/library deployment docs; CI scope may miss sibling crates | GitHub Pages + five Workers; kit clone/overlay deployment | Usable but split; kit pinning and cross-repo version policy remain weak |
| Ontology Bridge | Oxigraph SPARQL store; OWL 2 EL reasoning (Whelk-rs); 92 CUDA kernels; canonical knowledge graph | 10 MCP tools (`ontology-bridge.js`) proxy SPARQL queries to VisionClaw; gated via `[skills.ontology]` in `agentbox.toml` | Not applicable (forum does not query KG directly) | Not applicable (pod stores data, does not query KG) | Not applicable (website does not query KG) | Shipped cross-substrate integration: agentbox → VisionClaw. Single direction (query only); axiom submission available but governance review deferred. |
| Tests/Ops | Build/test commands and known issues documented; master cross-substrate fixtures under `docs/specs/fixtures/` (8+ JSON fixtures with JSON Schemas); IS-Envelope spec (ADR-075, 11 vectors) | 54 upstream fixture vectors under `tests/contract/upstream_vectors/` with SHA-256 checksums (`CHECKSUMS.txt`) and upstream pins (`UPSTREAM_PINS.md`); 5 adapter contract test suites (15 implementations) | Workspace tests, audit workflow, Worker build docs; fixture copies under `tests/fixtures/` | JSS parity tests/docs; fixture copies in `solid-pod-rs-didkey` and `solid-pod-rs-nostr`; IS-Envelope vectors present | README/test docs conflict; some gates advisory | Shared fixture corpus with checksum verification is the strongest cross-repo alignment signal. Release gating is partially automated via agentbox checksums. |

## Minimum Compatibility Record

Every coordinated release should publish:

| Field | Required value |
|---|---|
| Repo SHAs | VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website |
| Identity contract | DID method, public-key form, verification suite string |
| Relay contract | NIP list, write auth, read auth, peer relay list, allowed pubkeys |
| Pod contract | Pod tier, WAC mode, NIP-98 replay behavior, git/export capability |
| Governance contract | Supported Agent Control Surface kinds and schema versions |
| Test evidence | Fixture sync status, smoke test status, audit status |
| Operations | Health endpoints, backup/restore plan, DR owner |

## Release Manifest

Generate the current local repository SHA set with:

```sh
scripts/generate-release-manifest.sh > docs/releases/ecosystem-release.local.json
```

See [Release Manifests](../releases/README.md) for the schema and promotion rules.
