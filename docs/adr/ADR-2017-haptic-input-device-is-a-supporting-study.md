---
id: ADR-2017
title: Admit visiontouch-rs as a private proof-of-principle haptic input study, with node grab in VisionClaw as a future integration
date: 2026-10-17
decision_status: accepted
implementation_status: none
activation_status: inactive
supersedes: []
superseded_by: []
verified_commit: 4a8d4b5
verified_paths: []
owner: jjohare
review_trigger: any intent to publish the repository, distribute software, or start the VisionClaw integration
repo: visionflow
domain: BASELINE-visionflow.md
---

# ADR-2017 — Admit visiontouch-rs as a private proof-of-principle haptic input study, with node grab in VisionClaw as a future integration

## Context
- visiontouch-rs (private, DreamLab-AI/visiontouch-rs) is a clean-room Rust host stack (Linux, EPP parallel port via ppdev) for a lawfully owned 1990s PHANTOM 1.0A haptic arm whose vendor stack is unsupported.
- Legal posture: observation-first clean room, a private non-commercial study, product names referential only, no affiliation with 3D Systems. The outputs are an academic paper and, later, a pose and stylus-button input.
- The lab's VisionFlow installation wants a 6-DoF input device for grabbing nodes in the VisionClaw graph. Force delivery is not yet verified, and the integration is not built.
- The repository has its own ADR ledger (ADR-0001 onwards) and CI gate (`scripts/quality-gate.sh`); this canon needs a record so the estate map, roster and website describe it truthfully.

## Decision
- visiontouch-rs is a supporting repository, not a substrate. The canon records it in the repository map, health roster, README, ecosystem map and website as "private study, proof of principle, no economic use". Publishing, distributing software or selling a service is a re-review of this ADR.
- The only planned estate integration is pose plus stylus button producing node select and drag. It enters VisionClaw by the same path as other input, never a side channel:
  - web client: a SpacePilot-style controller class (`client/src/features/visualisation/controls/SpacePilotController.ts:138`) producing an R3F ray and drag, which drives `nodeDragStart`, `nodeDragUpdate` and `nodeDragEnd` on `/wss` (`client/src/features/graph/hooks/useGraphEventHandlers.ts:148-214`; server `src/handlers/socket_flow_handler/message_routing.rs:52-58`);
  - XR client: a new `Resolver` variant (`xr-client/rust/src/selection.rs:37`) feeding `PointerInput` (`:76`) into `SelectionArbiter` (`:240`).
- A local bridge on the device host streams pose to the client over loopback; it never goes through the server.
- Force-feedback return path is out of scope.
- Diagrams for the repository will live under `docs/diagrams/visiontouch-rs/` with area id HR, added when the code stabilises; not in this change.

## Consequences
- The canon now names a private repository without linking it: the website card is not a hyperlink and the README block says it is private.
- The roster gains a row; the nightly health snapshot reads it as unreadable without an estate read token, which is expected.
- Repository counts in prose ("nine", "ten") are unchanged: visiontouch-rs is supporting text, not a counted substrate.
- Any integration needs VisionClaw-side ADRs in its own ledger, and a physical force verification before any claim of delivered force.
- Cost: one more entry to keep truthful as the study evolves.

## Verification
Status is none and inactive: no integration code exists. Established at `4a8d4b5` by inspecting the edited files (`docs/architecture/repository-map.md`, `scripts/estate-health/roster.json`, `README.md`, `docs/ecosystem-map.md`, `website/static/index.html`, `docs/BASELINE-visionflow.md`) and running `node scripts/adr-index-gen.cjs docs/adr --check`. VisionClaw anchors were read from the research note against the current VisionClaw checkout.
