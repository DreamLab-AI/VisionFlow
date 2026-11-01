---
title: knowledgeGraph retained explorer contracts and decision lineage
date: 2026-11-01
status: assessed-with-open-findings
scope: KG-04 five panels and knowledgeGraph ADR-2001 section supplement
---

# knowledgeGraph retained explorer contracts and decision lineage

The retained knowledgeGraph explorer exhausts its two-buffer pool across real JavaScript transfers. Its existing same-object tests pass because they do not exercise that boundary. The configured visionGraph publishing copy already uses a different pool and passes the same transfer control. This is a finding against retained source, not evidence of a current public-site failure.

All five panels in [KG-04](../diagrams/knowledgegraph/04-explorer-and-site-build.md) have been assessed and corrected. This closes this topic's source-review queue entry; implementation fixes, browser acceptance, the remaining knowledgeGraph topics and the wider estate audit remain open. No runtime code or ADR status was changed.

## Evidence and limits

| Evidence | Result and limit |
|---|---|
| knowledgeGraph source | `4ed9ac159daf402b4fb252ce559dbeb894a8d91e`; retained `explorer/modern`. |
| visionGraph comparison | `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`; `publishing-tools/WasmVOWL/modern` and configured publishing workflow. Four compared files differ; findings are not inherited across copies. |
| Focused probes | **12 passed**, Node v22.23.3. They establish controls, limitations and counterexamples, not twelve separate defects. Real MessageChannel transfers; mocked fetches with no external requests. |
| Existing transport tests | **18 passed** across `positionTransport.test.ts` and `protocol.test.ts`, Vitest 4.1.10, run in the existing knowledgeGraph checkout. The pool-return cases use same-object returns. |
| Build | `npm run build:check` passed: TypeScript build and Vite 6.4.3, 2,151 modules. Source came from a git archive in an isolated task directory, with existing node_modules linked in. Installed WASM package reported 0.1.1. This was not a fresh dependency installation or a dependency-integrity audit. |
| Build output | Graph chunk 1,030.53 kB minified; WASM 1,112.26 kB, 381.48 kB gzip. Vite warned about chunks above 500 kB. These are build sizes, not browser latency measurements. |
| Diagram gate | Five panels rendered and all 44 citations verified at declared revisions, without fallback or exceptions. Whole-corpus gate results are retained with the evidence. |
| Preserved edits | Existing knowledgeGraph baseline, README and licensing edits, VisionFlow review files and visionGraph journal were kept intact. The baseline's uncommitted publication-boundary note already qualifies the historical snapshot; it was not treated as bytes from the pinned commit. |

[Probe source](evidence/2026-11-01/kg-explorer/probes.mjs), [reproduction instructions](evidence/2026-11-01/kg-explorer/README.md), [source manifest](evidence/2026-11-01/kg-explorer/sources.json) and [retention receipt](evidence/2026-11-01/kg-explorer-retention.json) separate committed evidence from larger local build/render artefacts. No current GPU-browser journey, deployment check, throughput benchmark or complete frontend suite is claimed.

## Open findings

| ID | Finding and evidence | User consequence and next acceptance condition |
|---|---|---|
| KG-E1 | **High if retained viewer is activated:** `DoubleBuffer.release` requires membership in the original `owned` array (`src/workers/positionTransport.ts:76`). Transfer detaches that object; the returned buffer is a different object. Probe 2 rejects both returns and the third acquisition is undefined. Worker `publishFrame` skips a frame when acquisition fails (`physics.worker.ts:179`); the seed also uses the pool. Probe 12 reclaims four round trips in the visionGraph counterpart. | Retained simulation can keep ticking without updating displayed positions after two sends, including the seed. Before reuse, adopt a transfer-compatible ownership contract and verify repeated worker-to-renderer delivery, stale-generation refusal and cleanup. The public-site copy cannot be judged from this failure. |
| KG-E2 | **Medium, boundary validation:** `assertScope` checks only upper bounds (`src/types/scope.ts:144`); probe 4 accepts negative, NaN and fractional counts while rejecting ordinary overflow. Probe 5 publishes an overview with missing positions/metadata and invalid edge endpoints (`scopeStore.ts:512`). Edge collection stops at 4,000 and collapsed real nodes are sliced to budget (`scopeStore.ts:172,303`). | A cap is not a structural guarantee. Define which reader validates finite counts, aligned lengths and endpoint bounds. Display truncation should be visible to a user interpreting graph completeness. No corrupt production input is demonstrated. |
| KG-E3 | **Medium, retrieval honesty:** index HTTP 503 produces the same concept-exists stub as no match and does not attempt the page API (probe 9; `pageService.ts:158,219,267`). A null document title throws outside the fetch catch (probe 11; lookup at line 184). | A reader can mistake an outage for an unfinished concept, or encounter a rejected page load. Separate unavailable, unknown, malformed and known-but-unwritten states; validate response shape. This is not an XSS finding. |
| KG-E4 | **Medium, content fidelity:** primary Markdown HTTP 500 triggers the external raw-GitHub fallback (probe 7; `pageService.ts:129`), not just 404. When both Markdown paths fail, a populated API `body` is ignored in favour of a definition (probe 8; line 233). | The displayed prose can come from another publication state or omit an available body. Specify offline/fallback behaviour and which content representation is authoritative; test title/slug disagreement against the current producer in the pending KG-05 review. No confidential-data transmission was tested or observed. |
| KG-E5 | **Medium, freshness:** successful index, page JSON and parsed Markdown caches have no generation key or expiry (`pageService.ts:58,127,158`). After a mock generation change, probe 10 makes no new request and returns generation-one prose. | A long-lived session can retain old content after publication. Define session snapshot semantics or generation-aware invalidation, then test mixed-version navigation. Reloading creates a new module and is outside this counterexample. |
| KG-E6 | **Low, source-only route review:** `/ontology/:nodeId` redirects to `/graph` preserving only the query (`router.tsx:48,88`). The retry flag is removed immediately after requesting reload and on module evaluation (`router.tsx:19-32`). | Legacy path identity is not translated into focus; the code does not provide a durable one-reload budget across page loads. Verify deep-link intent and persistent chunk-failure behaviour in a browser before promising recovery. No reload loop was executed in this pass. |

The transfer result follows the [HTML Standard's transfer/deserialisation model](https://html.spec.whatwg.org/multipage/structured-data.html#structureddeserializewithtransfer), which creates a new ArrayBuffer object for transferred data. The standard and the real Node transfer probe support the mechanism; neither substitutes for a deployed browser acceptance test.

Two other probes establish boundaries rather than defects: short position copies intentionally retain the destination tail, matching existing tests; an unrecognised domain decodes to overview. The retained viewer has six domain names while the configured publishing copy has nine. That difference requires version-aware links, not an assumption that either tree accepts every other's URL.

## Five-panel disposition

| Panel | Assessed change |
|---|---|
| KG-04.1 | Correct lazy-route anchors and external WASM pin. Record path-node identity loss and distinguish historical in-tree Rust documentation from the package dependency. |
| KG-04.2 | Correct overview to baked/no worker buffer. Domain uses a loaded tier; focus supplies a minted buffer. Show actual transfer return and pool failure; retain default SharedArrayBuffer disablement. Removed unmeasured zero-allocation guarantee. |
| KG-04.3 | Replace the false never-truncate claim with builder selection/truncation followed by upper-bound checks. Record missing shape validation. |
| KG-04.4 | Correct sequential metadata/prose fetches, all-failure external fallback, stub-on-index-failure, body omission and cache lifetime. The old producer mismatch is deferred to KG-05 instead of assumed current. |
| KG-04.6 | Replace never-ships language with the inspected workflow's explicit source and destination. Identify different versions, domains and pool. A configured deployment target does not identify the live served revision. |

The historical numbering 1, 2, 3, 4, 6 is preserved. There is no new fifth-numbered panel and no implied missing-panel completion.

## knowledgeGraph ADR-2001 section dispositions

This is [knowledgeGraph's record](../../../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md), not VisionFlow's different ADR with the same number. Its complete text was reread. Historical measurements below remain attributed to their dates; they were not rerun by this pass.

| Section | Disposition at this review |
|---|---|
| Metadata and review trigger | Accepted/complete/live describes consolidation, not acceptance of every viewer path. `verified_commit` remains historical. A changed publisher and consumer tree warrant the documented boundary review without promoting any status. |
| Context | The missing ADR-NG-001 and ontology-content/engineering-record distinction remain relevant. The cited architecture reference now contains an obsolete in-tree Rust location. A real reference surface exists, but parts of it are stale. |
| Decision 1: living baseline | The committed baseline is a historical publication account. The operator's working edit explicitly adds that boundary; it was preserved. Current dependency and publisher evidence should be dated separately from the old baseline pin. |
| Decision 2: thin ledger | The four numbered operative records, template and preamble remain present. This pass assesses ADR-2001 sections, not every section of ADR-2002/3/4 or generated index freshness. |
| Decision 3: ontology pages | Preservation of class identities is still the stated obligation. No ontology pages were moved or rewritten. The historical banner claim is not re-certified by a frontend build. |
| Decision 4: absent records | Keep ADR-NG-001 classified historical-absent and the upstream LCR documents explicitly scoped. No reconstructed authority is introduced. |
| Consequences | The 8,138-class pin and stale 7,874-page figures are historical claims, not today's active-vault census. The useful lookup order needs present source locations as well as readable documents. |
| Verification: parent-tree facts | The prose cites parent `4886fbb3d`, whereas frontmatter cites `d09a476…`; those are different historical evidence layers. Corpus counts, absent-document count and old workflow references are not current measurements from this pass. |
| Verification: introduced artefacts | Creation of ledger/baseline and frozen banners is historical evidence. No claim is made that the old generator invocation was rerun here. |
| Closeout extension, 4 September | Canonical identity, absent lineage, versioned exports and an identified consumer remain useful acceptance conditions. The current workflow identifies the visionGraph consumer; that is distinct from the retained source built in this review. |
| Acceptance progress: ADR-NG-001 map | The classification is sound; the locator table needs revision. `pipeline/…` and `explorer/rust-wasm/…` no longer describe current root paths. FORMAT-NGG1 still carries its historical broken ADR link. This pass does not fabricate or silently repair absent authority. |
| Acceptance progress: repository identity | Historical narrativegoldmine identifiers and canonical repository identity remain different concepts. The configured workflow explicitly publishes to DreamLab-AI/knowledgeGraph. Old release-gate paths in the annex belong to the archived pipeline. |
| Acceptance progress: implemented modules | Census, visibility, manifest, release gate and compatibility listings describe the historical Python pipeline. Their previous implementation evidence does not establish the current upstream publisher or retained viewer's runtime contract. |
| Acceptance progress: tests/results | The 85-test result and original CI sequence remain dated evidence. The 18 existing frontend tests plus 12 probes here cover different boundaries and cannot be added to that number as one suite. |
| Acceptance progress: consumer browser build | July prebuilt SPA inspected in September, with blank canvas/no WASM or binary requests, is explicitly historical. Today's TypeScript/Vite build does not resolve EXP-01 or re-establish that old symptom. |
| Acceptance progress: governed paths | The original workflow and pipeline listings are historical locations. Publisher configuration now selects visionGraph's separate source tree. |
| Acceptance progress: remaining | Upstream LCR lineage, stale architecture counts and separation from active authoring remain qualified obligations. This pass does not close those histories. |
| Publication baseline re-verification, 7 September | The 96 Python tests and publication rollback/privacy claims concern that revision. Keep the dated evidence and status restraint; do not transpose it onto today's explorer or public deployment. |

## Comparison and user utility

Sigma.js is an adjacent graph-rendering component, not a replacement for the estate's corpus governance and publication pipeline. Its official [settings documentation](https://www.sigmajs.org/reference/settings/) exposes separate controls for hiding labels and edges while the camera moves, and describes a performance cost for edge events. That is a useful comparison point for making interaction and detail trade-offs explicit. It is not evidence that Sigma would outperform this viewer.

A useful comparison would give both renderers the same versioned graph and ask readers to find a named concept, follow a relation, open its source prose and explain what has been omitted from the visible scope. Measure task success, cold-load bytes/time, continuing position updates beyond two frames, label readability, reduced-motion behaviour and response to a new corpus generation. Include malformed/unavailable page data and a changed title. The intended utility is a trustworthy path from a visible relationship to dated evidence; a visually attractive graph with silent omissions or stale prose does not establish that utility. These are proposed evaluations, not measured benefits or market rankings.

KG-01, KG-05, KG-06, KG-07 and the broader archived/operative decision inventory retain their previous queue status. This report advances one topic and one decision supplement only.
