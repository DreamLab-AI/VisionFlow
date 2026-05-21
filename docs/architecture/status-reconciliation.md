# Status Reconciliation

**Status:** Cross-doc reconciliation note
**Date:** 2026-05-20

This note keeps older audit language from being mistaken for current runtime status. It is based on repository documentation and metadata only.

## VisionFlow Website Docs

| Claim area | Reconciled status |
|---|---|
| Tailwind Play CDN in ADR-001 | Superseded in implementation by local static CSS; ADR-001 now carries a 2026-05-20 amendment |
| GitHub Pages deployment via `gh-pages` push | Superseded in implementation by Pages artifact upload/deploy actions; ADR-001 now carries a 2026-05-20 amendment |
| Lighthouse and axe acceptance criteria | Axe and browser smoke checks now run through the external Chrome DevTools sidecar; Lighthouse's local-Chrome gate is superseded until it can attach cleanly to the sidecar |
| All ten website sections in nav | Static nav now links the primary PRD sections present in the page |
| Contact/demo form and hashed assets | Still deferred; tracked in [Site Verification Status](../site-verification.md) |

## VisionClaw PRD-010 / PRD-014 / PRD-015

| Document | Reconciled reading |
|---|---|
| PRD-010 DID:Nostr Mesh Federation | The early “current-state evidence” section intentionally records audit findings from before Phase 0. Later sections and memory indicate crypto fixes/absorption work progressed, but federation remains “implementation in progress.” Treat PRD-010 as historical audit plus target architecture, not proof that all listed gaps are still open. |
| PRD-014 Ecosystem Productionisation | Completion checklist marks the 60% -> 80% productionisation work complete, including zero critical gaps, substrate CI, runbooks, fixture sync, health aggregation, and security hardening. Remaining 80% -> 100% items are mesh/runtime/observability/accessibility/release-process work. |
| PRD-015 Ecosystem Code Hygiene | Success criteria show many hygiene items completed. Cross-substrate NIP-98 convergence remains explicitly deferred pending WASM-compatible shared surfaces. |

## Current High-Confidence Open Items

| Item | Why it remains open |
|---|---|
| Full mesh federation by default | Current docs/config still show standalone defaults and optional federation modes |
| IS-Envelope runtime ownership | Spec exists, but the canonical runtime/schema owner still needs to be pinned |
| Shared NIP-98 implementation | PRD-015 explicitly leaves this unchecked |
| CF Workers/native pod convergence | Forum docs still describe two-tier behavior and portability blockers |
| Browser verification sidecar reachability | The expected endpoint is `browsercontainer:9223` from Docker-network runtimes or `localhost:9222` from the host; sidecar CDP was verified on 2026-05-21 after resolving `browsercontainer` to its Docker-network IP |
