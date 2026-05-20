# Ecosystem Compatibility Matrix

**Status:** Docs and metadata synthesis
**Date:** 2026-05-20

This matrix records the current compatibility posture described by sibling repository docs. It is not a source-code audit.

| Area | VisionClaw | agentbox | nostr-rust-forum | solid-pod-rs | dreamlab-ai-website | Compatibility status |
|---|---|---|---|---|---|---|
| Identity | NIP-98, NIP-07, DID:Nostr, NIP-26 delegation, Solid/WAC documented; enterprise SSO remains a known future path | Bootstrap BIP-340 key becomes canonical `did:nostr`; accepted by relay, pods, WAC, URNs | Passkey PRF derives Nostr keys; NIP-98 across workers; NIP-42 relay | Foundation for DID:Nostr, NIP-98, Solid-OIDC, Tier 1/Tier 3 DID resolution | WebAuthn PRF + NIP-98; configured agent identities | Broad alignment on `did:nostr` + NIP-98; historical replay/parity concerns still need code recheck |
| Mesh | ADR-073 describes client/subscriber mode by default unless a local relay is added | Optional sovereign/federated mesh; relay exposure is configuration-sensitive | `nostr-bbs-mesh`, NIP-42 gate, peer discovery, IS-Envelope routing are documented | Provides identity/pod foundation, not relay mesh ownership | Mesh block exists, current deployment docs/config skew standalone by default | Mesh is designed, but default deployments are not fully federated end-to-end |
| Pod | Embedded `solid-pod-rs` at local `:8484`; WAC against `did:nostr` | Embedded Solid pod and native pod overlay via Cloudflare Tunnel | CF Worker pod tier mirrors Solid/JSS; native tier supports git via `solid-pod-rs-server` | Canonical LDP/WAC/WebID/auth/storage implementation | CF pod tier plus native agentbox pod tier for git | Strongest compatibility area; two-tier behavior is the main gap |
| Governance | Judgment Broker, ontology governance, Agent Control Surface events documented | Runtime/provenance focused; no primary governance UI | Primary governance UI/API: agent registry, cases, roles, signed responses | Access-control foundation, not workflow owner | Governance route and configured agent pubkeys | Governance centers on forum/website; VisionClaw and agentbox integration needs explicit smoke tests |
| Deployment | Docker Compose multi-profile and docs CI; productionisation PRDs still list hardening gaps | Nix-built multi-arch image, runtime contracts, image scan/SBOM, flake checks | Cloudflare Workers kit with CI/audit docs | Server/library deployment docs; CI scope may miss sibling crates | GitHub Pages + five Workers; kit clone/overlay deployment | Usable but split; kit pinning and cross-repo version policy remain weak |
| Tests/Ops | Build/test commands and known issues documented | Runtime contract tests, adapter contracts, shellcheck, gitleaks, image scan | Workspace tests, audit workflow, Worker build docs | JSS parity tests/docs; CTH runner and sibling crate scope still noted gaps | README/test docs conflict; some gates advisory | Test posture is improving, but claims and gating differ by repo |

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

