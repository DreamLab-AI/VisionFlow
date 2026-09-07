---
title: VisionFlow estate review
status: in-progress
date: 2026-09-04
type: explanation
---

# VisionFlow estate review

This review examines the VisionFlow estate as an organisational proposition, a set of software implementations, and a system whose promises depend on connections between repositories. It asks where the vision is supported by source and execution evidence, where the implementation has moved beyond the written account, and what still prevents the intended experience.

**Latest review: [2026-09-07 source audit and reconciliation](2026-09-07-estate-audit.md).** It provides current ADR/source dispositions, diagram corrections, a workspace census, validation receipts and a dependency-ordered [closeout sprint](closeout/2026-09-07-sprint.md). Local source/testing, historical evidence and deployed acceptance are explicitly separated. The older chapters below preserve the investigation history; later dated findings supersede conflicting current-state claims.

The source audit does not declare all optional features, upstream research or six end-to-end system journeys complete. Those implementation and acceptance obligations remain on the [master TODO](../../../project/docs/TODO-unified.md).

## Reading order

| Document | Question it answers |
|---|---|
| [Scope and evidence](scope-and-evidence.md) | Which repositories belong in the review, and what counts as proof? |
| [Vision and architectural argument](vision-and-architecture.md) | What is the system trying to achieve, and what would make that argument convincing? |
| [Book roadmap reconciliation](book-roadmap-reconciliation.md) | How do the eighteen published commitments map to current evidence and remaining work? |
| [Canon claim reconciliation](canon-claim-reconciliation.md) | Which book claims need qualification against current implementation evidence? |
| [Engineering governance](engineering-governance.md) | Do harness coverage and mandate proposals establish implemented controls? |
| [Canon and verification](canon-and-verification.md) | Does VisionFlow's own implementation enforce the claims it coordinates? |
| [Knowledge production](knowledge-production.md) | How do authoring, publication, semantic checks and privacy boundaries actually work? |
| [Authored vault transition](authored-vault-transition.md) | Which corpus is current, and did its consumers follow the Obsidian migration? |
| [Agent grounding and governance](agent-grounding-and-governance.md) | Do retrieval, local editing, signed approval and writeback preserve their contracts? |
| [Grounding delivery](grounding-delivery.md) | How does Loom assemble answers and identify the corpus generation it serves? |
| [VisionClaw data runtime](visionclaw-data-runtime.md) | How do sync, graph ownership, provenance and backup behave across commit boundaries? |
| [Role authority](role-authority.md) | How do role mutations, fallback permissions and enforcement modes affect actual authority? |
| [Runtime ingress](runtime-ingress.md) | What do proxy identity, daemon credentials and browser sessions actually enforce? |
| [Forum decisions](forum-decisions.md) | What does a signed human response establish at each hand-off? |
| [Storage and authority](storage-and-authority.md) | Do pod identity, ACL resolution, caching, replay and provenance preserve authority across tiers? |
| [Learning evidence](learning-evidence.md) | Do retained trajectories establish outcomes and preserve privacy? |
| [Consumed vector storage](consumed-vector-storage.md) | Does the loaded semantic artefact match its declared model and score geometry? |
| [Runtime profiles and egress](runtime-egress-and-profiles.md) | Where do settings and session content cross provider boundaries? |
| [Configuration projection](configuration-projection.md) | Do desired, built, projected and loaded states agree? |
| [Process lifecycle](process-lifecycle.md) | Do discovery, stale-state policy and shutdown bind to the same process instance? |
| [Adapter dispatch](adapter-dispatch.md) | Which lifecycle and privacy guarantees are enforced per operation? |
| [Catalog decisions](catalog-decisions.md) | Do skill catalogs, ranking and governance templates enforce their stated scope? |
| [Capability enforcement](capability-instructions-and-enforcement.md) | Which limits are executable controls and which are instructions? |
| [Federation identifiers](federation-identifiers.md) | Do bytes, address grammar and supported crossings agree across languages? |
| [Ontology explorer](ontology-explorer.md) | What do standalone tests establish, and which embedded/browser paths remain distinct? |
| [Self-improvement](self-improvement.md) | What does an accepted dream result prove, and is its candidate patch actually evaluated? |
| [Commercial surfaces](commercial-surfaces.md) | Do release pins, config mirrors and chat tiers mean what the interface promises? |
| [Rendered state](rendered-state.md) | How do authenticated actions, wire frames and current work state become visible? |
| [Mobile sensing companion](mobile-sensing.md) | Which mobile observations are measured, inferred or simulated, and what proves platform acceptance? |
| [Sensing extension](sensing-extension.md) | Which RuView paths are implemented, simulated or incompatible with their firmware? |
| [Shared memory](shared-memory.md) | What happens when embeddings fail or stored memories expire? |
| [Pocket provenance](pocket-provenance.md) | Can a mirrored session reference reconstruct the exact decision after restart or eviction? |
| [Agent disclosure](agent-disclosure.md) | What does an agent badge prove, and how do stale or historical registrations affect it? |
| [Decision history](decision-history.md) | What can readers reconstruct from signed decisions, projected rows and local UI history? |
| [Failure telemetry](failure-telemetry.md) | Which failures are classified, retained and counted, and which remain outside the census? |
| [Transaction cost accounting](transaction-cost-accounting.md) | Do captured token fields establish the promised per-DAG measurement? |
| [Joined provenance trace](joined-provenance-trace.md) | Does identity-based joining prove a complete, authorised action trace? |
| [Liveness evidence](liveness-observation.md) | What does an observed fire prove, and how do consumers interpret it? |
| [KPI outcomes](kpi-outcomes.md) | What do the metrics measure, and can their sources and freshness be audited? |
| [Historical obligations](historical-decision-reconciliation.md) | Which predecessor commitments survive narrow operative successors? |
| [Closeout execution sequence](closeout/execution-sequence.md) | In what order should the established gaps be resolved and accepted? |
| [Closeout roadmap](closeout/README.md) | Which dependencies and acceptance journeys govern complete-system closeout? |
| [ADR inventory](closeout/adr-inventory.md) | Which decision, archive and reference candidates remain to be reconciled? |
| [Upstream decision scope](upstream-decision-scope.md) | Which upstream families are consumed, copied, divergent or still awaiting semantic review? |
| [Upstream decision packs](closeout/upstream-packs.md) | How are the upstream records grouped for review without assuming adoption? |
| [RuView local decisions](closeout/sensing-decisions.md) | Which sensing decisions have dedicated evidence and which remain open? |
| [Vendored lineage](closeout/adr-lineage.md) | Which RuView ADRs copy or diverge from the standalone RuVector tree? |
| [Assessment completion requirements](closeout/assessment-requirements.md) | What remains to finish the requested review, separately from implementing its roadmap? |
| [Investigation ledger](investigation-ledger.md) | What findings are established, and what work remains at each level? |
| [Execution snapshot](evidence/snapshot.json) | Which checkouts, source hashes and command results support this pass? |
| [Review navigation audit](evidence/review-navigation.json) | Do local file targets, internal section anchors and incoming review links resolve? |
| [Navigation checker](evidence/review-navigation.py) | How can that bounded structural check be repeated? |
| [Receipt collector](evidence/collect.py) | How can the initial local inventory and canon probes be repeated? |
| [Knowledge receipts](evidence/knowledge-snapshot.json) | What did the full pipeline build, existing suites and publication probes establish? |
| [Knowledge probe collector](evidence/knowledge-probes.py) | How can the pipeline results be reproduced without publishing? |
| [Knowledge lineage](evidence/knowledge-boundaries.json) | Which source files and embedded explorer copies were inspected? |

Return to the [documentation index](../README.md). The [book](../../presentation/report/main.tex) records the wider argument and its history; the [baseline](../BASELINE-visionflow.md), [gap register](../registers/gap-register-v1.2.md) and [closeout](../closeout/final-design.md) provide earlier assessments to investigate, rather than substitute for current evidence.

## Coverage ledger

“Inspected” below means the named material was read. It does not imply that a repository builds or its service works. Capability maturity must be assessed per path; a single maturity label for an entire repository would conceal important differences.

| Area | Evidence collected across the review | Remaining depth |
|---|---|---|
| VisionFlow | Build/deploy/count/fixture/release and dream traces; book/canon commitments, engineering harness and mandate assessment; local drift and isolated failure probes | Whole documentation-corpus navigation, remaining source-claim reconciliation, browser and publication checks |
| VisionClaw | Shared storage/sync/provenance and WAL backup probe; agent-event attribution, XR codecs/render maths/query source; 218 XR library tests | Full reasoner/proposal transactions, browser/GPU pipeline, authenticated rendered journey, scene/headset/voice and recovery |
| agentbox | Grounding/authority, proxy, memory, learning and configuration traces; profile, egress, projector, privacy and lint fixtures; paired federation helpers | Nix/image activation, full tool authority and custody, historical obligation reconciliation, real recall and lifecycle recovery |
| solid-pod-rs | Native verifier/resolver/storage/provenance, consumer version map; 46 tests and resolver/replay probes | Full delegated revocation, OIDC, forge/payments, federation and recovery |
| nostr-rust-forum | Signing/relay/projection trace; 47 governance and 22 key tests; trust pagination fixture; ten operative records and 24 historical routes | End-to-end applied receipts, projection recovery, cross-tier pod journey and pinned deployed consumer |
| dreamlab-ai-website | Three-client release/config/chat trace; 97 tests; pin parity; all eight operative ADRs extended | Real browser-to-agent journey, tier authority, release failure gates, rotations and archive lineage |
| Loom | Composition/fusion/chat/generation trace, passing library/generation tests, actual RuVector artefact configuration probes; all four ADRs extended | Runtime rejection/reload probes, model adapters, research evaluation and agent consumers |
| knowledgeGraph | Full fresh build, parser/converters/validation, import staging, existing tests and publication-boundary probes | Content-quality assessment and complete explorer/runtime integration |
| Logseq (`project4` alias) | Historical authoring pipeline, closure/scaffold exports, publishing workflow, corpus census and existing tests | Six design records routed in a history companion; detailed historical claim verification and legacy publisher disposition remain |
| visionGraph | Current manifest-bound vault, namespace-aware publisher, corpus census, publication probes; 57 passing tests and one corpus-size assertion failure | Content/enrichment promotion, inclusion-policy alignment, public deployment and explorer behaviour |
| WasmVOWL | Standalone native tests (47 pass), frontend tests (19 pass/60 fail), schema/state findings and proposed consumer ADR | Renderer, WASM boundary, accessibility, performance evidence and whole-tree divergence |
| dream-engine | Toolkit and Rust service source trace; 78 Rust and 125 TypeScript tests; parser probes | Full cycle simulation, deployed receipts, restart recovery and candidate evaluation |
| RuVector | Agentbox MCP/store/model/filter and Loom local VectorDB/redb/index paths investigated; wrong-configuration probes; 176 vendored ADRs compared | Standalone consumed extension internals, build/release identity, persistence and recall under failures; unconsumed features need disposition |
| RuView | MAT/server/hardware distinction and parser mismatch; signal, learning, retrieval, security, tracking, mobile, training UI and macOS evidence; [current decision coverage](closeout/sensing-decisions.md) | Locate estate consumer; canonical codec, per-frame inference labels, physical measurement, retention and remaining upstream ADR scope |

Further source dependencies may establish additional estate-owned components; add them to this ledger as they are identified.


Historical follow-through: [surviving decision obligations](historical-decision-reconciliation.md) explains section-level reconciliation; the [VisionClaw routing map](closeout/visionclaw-history.md) lists all 137 historical candidates without inferring supersession.

[Liveness observations](liveness-observation.md) separates durable traffic records, attestations, health and acceptance evidence.

[KPI outcome evidence](kpi-outcomes.md) traces metric definitions, capture loss, snapshot lineage and dashboard freshness.

The [agentbox historical map](closeout/agentbox-history.md) adds 72 archived decision routes to the [section-level reconciliation](historical-decision-reconciliation.md#agentbox-routing-and-compound-obligations).


