---
id: ADR-2021
title: DreamLab runs closed-loop agent fleets on VisionFlow, first on its own estate and then as an operated service on client hardware, with hard human gates for every regulated action
date: 2026-10-30
decision_status: proposed
implementation_status: none
activation_status: inactive
supersedes: []
superseded_by: []
verified_commit: b49a847
verified_paths: []
owner: jjohare
review_trigger: the owner accepts, amends or rejects this record; or the end of each quarter from Jan 2027, when the exit tests in docs/business/roadmap-2026-11-to-2028-04.md §4 are checked; or Phase 0's 30-day wake-on-signal test passes or fails
repo: visionflow
domain: BASELINE-visionflow.md
lineage: "Builds on VisionFlow ADR-2006 (canon owns the cross-repo view, not the implementation), ADR-2008 (estate health collected by CI, read by the dream cycle) and ADR-2011 (task properties set the human-agent boundary, not agent self-tiering); agentbox ADR-2153 (session plane), ADR-2115 (dream decisions on the forum governance panel), ADR-2139 (System One judge as a vendor slot) and ADR-2085 (colloquy units). Does not revive the precedent store retired in agentbox 70d017a3b."
---

# ADR-2021 — DreamLab runs closed-loop agent fleets on VisionFlow, first on its own estate and then as an operated service on client hardware

## Context
- DreamLab sells training, residentials, co-creation and an advisory retainer. The sovereign stack appears on the website only as technology; there is no offer for running governed agents on a client's estate.
- On 2026-10-30 the owner shared a founder's account of Spawn, a ten-person company that reports running permanent area-owning agents in a closed production loop with "taste judges" over recorded decisions. Those figures are Spawn's own claims.
- The same day's estate landing found the estate's loop open: signals existed but humans found them in sweeps, with a worst signal-to-action latency of about 114 days (roadmap §3).
- The core stack (VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, sidestr-rs) is AGPL-3.0-only; ontomark, dream-engine and the Loom code are MIT/Apache. The model currently behind the Ontology Loom is licensed non-commercially.
- Health and public-sector buyers require human review of outputs and change control (DCB0129/DCB0160, NHS England scribe guidance), which rules out landing agent changes directly in regulated client production.
- The first production-profile start on client hardware is the public-sector health residential, 3–5 Nov 2026.

## Decision
1. **The roadmap of record is [`docs/business/roadmap-2026-11-to-2028-04.md`](../business/roadmap-2026-11-to-2028-04.md)**, six quarters from Nov 2026 to Apr 2028, in three tracks (operate, offer, assure). Each milestone names a quarter, a checkable exit test and its dependencies. A change to a milestone's quarter or exit test amends this record.
2. **The loop closes on DreamLab's own estate first.** Phase 0 (Q1) wakes the owning session on a typed signal; Phase 1 (Q2–Q3) adds area-owning sessions, a decision corpus of owner-promoted colloquy units, and then a judge that decides with a citation, sends back, or holds for a human. No judge runs before its corpus exists. A change complies only if it makes a signal produce an agent action without a human sweep, or records why it cannot.
3. **The headline operating metric is signal-to-action latency (SAL)**, computed from existing journals, and **silent-loop-stopped** (a scheduled producer missing its cadence) is an incident class. Ship volume is not a target.
4. **Action classes follow VisionFlow ADR-2011.** Reversible internal actions may land and then be repaired. Client, clinical, identity, cryptographic, secret, data-egress and payment actions sit behind a hard human gate on every estate. An agent never chooses its own class. On client estates, production change is propose-only: the client's named human decides through a signed decision.
5. **The service line is an operated, governed agent fleet on the client's own hardware**, sold as diagnostic → deployment sprint → managed retainer, in three bundles (operated workstation; governed ontology engineering; evidence-gated repository care). AGPL parts are sold as build, operate and support, never under a proprietary licence; contracts state the client's source rights.
6. **No billable work runs on non-commercially licensed model weights.** A per-model licence register with a CI gate, and a commercially licensed default model for client profiles, precede the first invoice for Loom-assisted work.
7. **No client fleet runs on an image with an unauthenticated control listener, an unverifiable audit journal, or inactive role isolation.** These are blocking dependencies of Phase 2, not risks to accept.
8. **Health work stays non-clinical until Q6**, when the owner records in a VisionFlow ADR whether to build DCB0129/DTAC-assured capability or stay non-clinical.
9. **VisionFlow owns this contract only** (VisionFlow ADR-2006). Each implementation is recorded in its consumer repo: agentbox (signals, sessions, corpus, judge, fleet profile, action classes, licence register) and nostr-rust-forum (the governance panel and task-property enforcement).

## Consequences
- The owner's time moves from answering a queue to setting principles, saying no, deleting and asking for step changes; the decision corpus is the mechanism, and its quality limits the judge.
- Cost: assurance work (Cyber Essentials Plus, DSPT, ISO/IEC 27001) runs alongside product work from Q1, and a commercially licensed local model must be selected and paid for.
- Cost: AGPL means the stack cannot be sold as closed licences; revenue is services and operations, and every engagement must return field work to a substrate as a release, or forward deployment becomes permanent services cost.
- Model supply becomes a managed dependency: a vendor-slot extension to the coding agent (agentbox ADR-2139 pattern), a degraded-mode drill in Q3, and a local fallback. A US government order suspended commercial access to two frontier models in June 2026 ([IAPP](https://iapp.org/news/a/thought-for-the-week-us-government-order-forces-commercial-suspension-of-two-fronteir-ai-models)).
- Easier: every existing control (journals, receipts, ADR verification stamps, the decline floor) becomes sellable assurance evidence once it is live rather than staged.
- Forbidden: claiming "live" for any phase without runtime evidence; promoting a judge's decision without a cited unit; running a client fleet past any blocking dependency in decision 7.
- This record was written at the owner's explicit request during the pre-residential week, an exception to that week's standing rule against new decision records. It does not displace the pre-residential work order on the master board.

## Verification
None yet: the record is proposed, unimplemented and inactive (`implementation_status: none` verified at VisionFlow `b49a847`). Inputs were verified on 2026-10-30: licences from each repo's `LICENSE` and `Cargo.toml` `license` field; the session plane from `agentbox-sessions list` (eight per-vendor seats, no area owners); the retirement of the precedent store from agentbox `70d017a3b`; open-loop latencies from the master board section "Landing 2026-10-30" and the VisionFlow park register `docs/registers/park-2026-10-30.md`; the IAPP article retrieved and its headline and date confirmed. Phase 0's exit test is the first falsifiable check of this record.
