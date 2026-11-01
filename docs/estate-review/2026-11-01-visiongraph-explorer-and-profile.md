---
title: visionGraph publishing explorer and shared-profile decision
status: assessed-with-open-findings
date: 2026-11-01
---

# visionGraph publishing explorer and shared-profile decision

The publishing explorer has adopted the shared renderer and Rust/WASM codec, and its current test suite and production build pass. Its content path still masks some outages as missing prose, retains successful responses without a generation key and does not recover from a corrupt gzip stream by retrying valid raw content. The collision gate also grants broader exceptions than its pair-based allowlist suggests.

This review corrects all five existing [VG-06 panels](../diagrams/visiongraph/06-shipping-explorer-duplication.md), adds a sixth for the previously unmapped prose path, and assesses every section of ADR-VG-007. Across the linked reports below, all six current visionGraph topics and all eight operative ADRs now have source/section assessments. This is documentation coverage, not closure of implementation findings, intake programmes or current end-to-end acceptance. No product code, corpus, ADR status, release or deployment was changed.

## Evidence boundary

| Evidence | Scope and result |
|---|---|
| visionGraph source | `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`; committed explorer, publisher, ADRs, collision tools/records and migration receipts. |
| knowledgeGraph comparison | `4ed9ac159daf402b4fb252ce559dbeb894a8d91e`; retained explorer, compared by tracked relative paths and file bytes. |
| Ontomark profile | `0caae5d4b695cb720942b9b9212017c00a80ad41`; `profiles/estate` source describes and contains a real versioned profile. Its existence does not establish complete corpus migration or consumer parity. |
| Dependency preparation | Fresh `npm ci --ignore-scripts --include=dev` succeeds against the committed lock. The first offline attempt lacked cached `zwitch`; its failure is retained. Explicit test/build commands then build the Rust notice tool. No install script is silently assumed to have run. |
| Existing tests | **105 JavaScript tests across 18 files**, including migration/codec, host interaction/drag, transport and notice adapter tests; **six Rust notice-tool tests**. The JavaScript harness uses a real published WASM bundle and shared renderer, with jsdom, canvas stubs and mocked browser facilities. This is not a GPU/browser acceptance run. |
| Build | `npm run build:check` passes TypeScript and Vite. Graph chunk: 70.12 kB minified, 24.30 kB gzip. Generated runtime inventory: 99 package/version entries. These are local build measurements, not user latency or public-site observations. |
| New probes | **Nine JavaScript boundary tests** and **seven collision-lint scenarios** pass their explicit control/counterexample assertions. Fetch responses and temporary corpus files are injected; no live content is changed. An initial JS probe invocation used the wrong working directory and could not find the WASM fixture; the corrected run passes. |
| Dependency advisory scan | Explicit `npm audit --omit=dev --include=prod --json` reports two moderate package entries, zero high/critical. The inherited npm `include=dev` setting caused the first `--omit=dev` attempt to include tooling: 22 entries, including ten high. Both outputs are retained with their actual scope. Package counts are not distinct exploit counts or a penetration-test result. |

The [source manifest](evidence/2026-11-01/vg-explorer/sources.json), [copy comparison](evidence/2026-11-01/vg-explorer/copy-comparison.json), [existing test log](evidence/2026-11-01/vg-explorer/existing-tests.log), [Rust log](evidence/2026-11-01/vg-explorer/notices-tests.log), [build receipt](evidence/2026-11-01/vg-explorer/build-receipt.json), [JS probes](evidence/2026-11-01/vg-explorer/audit-boundaries.test.ts) and [collision scenarios/results](evidence/2026-11-01/vg-explorer/collision-probes.json) bound the findings. Source hashes and local outputs are retained under `/tmp/vf-vg-explorer-artifacts`; the committed receipt records build output identities.

## Findings and useful acceptance

Paths below are relative to `visionGraph/publishing-tools/WasmVOWL/modern` unless otherwise qualified.

| ID | Evidence and implication | Recommended acceptance |
|---|---|---|
| VG-E1 — medium, stale session content | `src/api/pageService.ts:61,113,163` caches index, page JSON and parsed Markdown with no generation key or expiry. The probe changes supplied generation after a successful read; the same module makes no new request and returns old prose. | Define whether a session is a consistent snapshot or should refresh. Bind every page/graph response to that generation and exercise navigation across an update. A full reload is outside this counterexample. |
| VG-E2 — medium, outage presented as absence | At lines 169–180 an unavailable index becomes an empty list. Lines 276–282 then return a stub asserting the concept exists, without attempting its page API. A null index title also throws at line 198 outside fetch handling. | Distinguish unavailable, malformed, unknown and genuinely unwritten content. Validate index entries and test recovery from each state. No XSS or live outage is established here. |
| VG-E3 — medium, compressed fallback boundary | `src/api/fetchArtefact.ts:29–30` returns an inflation stream before its body is consumed. A gzip-magic fixture with corrupt payload returns a Response, then rejects on `.text()`; the raw path is never requested. The valid gzip control succeeds. | Test corrupt compressed content with a valid raw counterpart and specify recovery. Buffer/validate or catch body-consumption failure at a layer that can retry. The full page loader catches the failure and may show summary/Quartz fallback; the helper does not perform the advertised raw retry in this case. |
| VG-E4 — medium, overview shape admission | `src/types/scope.ts:153` checks upper limits but accepts negative, fractional and NaN counts. `src/stores/scopeStore.ts:512–529` accepts a one-node overview with zero positions and edge endpoints 99/100. `GraphCanvas.tsx:73–74` subsequently dereferences endpoint nodes. | Validate finite integer counts, lengths, endpoints and required metadata before publishing JSON-derived scope state. The binary codec has separate validation and its passing tests do not validate arbitrary overview arrays. This pass probes the builder, not a deployed renderer crash. |
| VG-E5 — medium, alias exception expands to unrelated owners | `scripts/ontology-hygiene/collision_lint.py:73–75` suppresses an alias collision if **any** pair of owners is allowed. Alpha/Beta/Gamma sharing one alias passes when only Alpha/Beta is allowed; two unallowed owners correctly refuse. | Evaluate all conflicting owner pairs or require an explicit reviewed group. Ensure adding an unapproved third owner cannot inherit a pair's exemption. This is a linter result, not proof that the compiler and all publication gates also accept the fixture. |
| VG-E6 — medium, category guard does not establish parentage | `collision_lint.py:50–52` searches the entire frontmatter for a root-link substring. `related: "[[dc-telepresence]]"` makes a Telepresence fixture pass without the required `is-a` relation; a fixture without that link refuses. | Parse the intended relation and exact target, then check the category-root contract. A similarly named relation or incidental reference must not certify parentage. |
| VG-E7 — bounded collision coverage | `fmlib.py:4,43` scans only root-level `knowledge/*.md`; the linter hard-codes category checks there too. Working-only and cross-vault Tokeniser/Tokenizer variants pass, while the knowledge-only pair refuses. | Cover the actual published roles and intended nested paths, or explicitly state exclusions. Retain the earlier trigger-filter findings; this batch does not run the entire publisher against these fixtures. |
| VG-E8 — low, route and documentation drift | `router.tsx:48–50,108` drops a legacy node path while preserving its query; lines 19–32 clear the reload flag immediately and on module load. `CLAUDE.md:8–18` still describes React Three Fiber and broad feature completion; `CAPABILITIES.md:24,50` describes the retired Python pipeline and static focus. | Test legacy deep-link intent and persistent chunk failure in a real browser. Date historical measurements and give the current host/renderer entry points. No reload loop was executed here. |

Two earlier retained-explorer findings must **not** be inherited unchanged. The publishing copy's transfer pool counts in-flight ownership (`positionTransport.ts:49–53,82–87`), and the earlier real-transfer control already passed for these exact source bytes. Its external raw-GitHub Markdown fallback has been removed (`pageService.ts:135–139`); the new probe confirms all requests remain root-relative. When the mirror is absent, the current loader still omits an available API `body`, but now supplies a definition and a Quartz link (lines 241–249). That is an explicit representation choice to assess against reader needs, not the former cross-publication fallback.

The [production audit](evidence/2026-11-01/vg-explorer/production-only-audit.json) names React Router advisories for [unexpected external navigation](https://github.com/advisories/GHSA-wrjc-x8rr-h8h6) and [SSR error hydration](https://github.com/advisories/GHSA-337j-9hxr-rhxg). The inspected application uses a browser router; this review does not demonstrate the advisory-specific attacker-input or SSR conditions. The counts support dependency triage and regression planning, not a claim that the public site is exploitable or free of other vulnerabilities.

## Copy inventory and diagram disposition

The current committed trees contain 210 publishing-explorer files and 160 retained-explorer files, with **151 shared relative paths and 49 differing shared files**; 59 paths occur only in visionGraph and nine only in knowledgeGraph. This whole tracked-tree method includes committed public fixtures and metadata. It replaces the old unrepeatable 39-of-108 count; different denominators must not be read as a measured rate of code drift.

| Panel | Disposition |
|---|---|
| VG-06.1 | Replaced unsupported counts and never-ships claims with the actual configured source, assembly and distribution path. Kept the legacy Tomcat Dockerfile separate from the publisher. |
| VG-06.2 | Mapped the adopted shared renderer, Rust codec, host callbacks, drag and transfer ownership instead of limiting the panel to dependency documentation. |
| VG-06.3 | Qualified historical capability/FPS claims; corrected the old claim that no Rust tool is built, since runtime-notices is built locally. Current domain and focus simulation is explicit. |
| VG-06.4 | Corrected the missing-Loom assertion: publishing `MESH_REPOS` now has nine entries including Loom and sidestr-rs; retained code has seven plus a separate Loom export. Added basename/data-root and legacy route limits. |
| VG-06.5 | Rechecked the nine-versus-six ordered domains and exposed the different JSON overview validation boundary. |
| VG-06.6 — new | Added prose/index/compression/cache behaviour with the newly executed counterexamples. |

## ADR-VG-007, section by section

The record is read as an accepted design with partial implementation and staged activation, preserving its distinct historical receipts. Its statuses were not changed by this review.

| Section or distinct obligation | Assessment |
|---|---|
| Frontmatter and 31 October status mapping | Seven watched paths and a carried revision give a bounded staleness contract. They do not cover every collision/profile input or certify current deployment. The owner-confirmation note remains unresolved; this audit does not choose a new status. |
| Context | Authored Markdown, site chrome and deployment remain in visionGraph; reusable compiler and graph packages are real dependencies. Local source proves that ownership boundary, not every consuming application journey. |
| Decision: estate dependencies | ADR-2014/2015/2016 are cross-repository design dependencies, not additional visionGraph operative records. Existing estate assessments retain their scope; this review does not silently accept them through VG-007. |
| Decision: versioned corpus profile | `profiles/estate/profile.yaml` is real versioned data (`id: estate`, version 1.0.0), with namespaces, ordered taxonomy, role grammar, governance and redaction settings. Describing the profile as wholly absent would now be wrong. Complete migration, fixture parity and consumer convergence remain separate obligations. |
| Decision: release extraction | Workflow v0.7.0 CLI checksum and graph 0.2.0/WASM 0.2.2 tarball pins implement the shared-release path. The adapter delegates binary operations and graph drawing. The older opening implementation versions are superseded by dated re-verification, not current pins. |
| Decision: publication safeguards | Validation, public projection, certificate/body and final-site gates exist; their previously reproduced limits remain in the [publication review](2026-11-01-visiongraph-publication-and-identity.md). All-vault public staging differs from the downstream knowledge-only build. Complete one-note/generic plugin acceptance is not established by this explorer suite. |
| Implementation: producer and contract tests | The old 23-test and v0.2.0 receipt is historical. Current publisher evidence is the earlier 41-test batch; it was not rerun or added to the 105 explorer tests. The CLI checksum was not freshly downloaded/verified in this batch. |
| Implementation: shared graph and NGG1 | Confirmed current GraphCanvas → shared mount and nggGraph → Rust decoder/encoder/constants. Old canvas modules and local codec path are absent in the tracked-tree comparison. Seven migration tests are part of the 105-test run. Scope metadata, physics and site policy remain local, so removal of independent renderers does not mean all application logic is shared. |
| Implementation: release distribution | Exact tarball sources and lock integrity are present; fresh installation succeeds. The historical npm trusted-publisher block and crates.io claims were not rechecked and must stay dated. |
| Implementation: remaining migration | The record correctly separates renderer adoption from complete profile/plugin/migration acceptance. Publicly active historical components and staged overall decision are different claims. The present batch makes no new public activation claim. |
| Implementation: bulk migration rules | Census, reversible dry run, resource identity preservation and semantic/visibility deltas are meaningful obligations. Current no-op syntax receipt cannot prove a nontrivial vocabulary/schema migration or native/WASM/consumer agreement after one. |
| Consequences: shared ownership | Current adoption supports reduced duplicated renderer/codec ownership. It does not establish one producer across all consumers: the [preceding audit](2026-11-01-visiongraph-publication-and-identity.md) shows VisionClaw still builds its local vault producer and can skip corpus-only refreshes. |
| Verification: codec and host checks | The current suite passes real WASM codec tests and shared-renderer host tests. Canvas/DOM stubs limit conclusions; selection/drag assertions are not actual pointer, GPU, accessibility or device acceptance. |
| Verification: timing, memory and bundles | October timing, pixel, heap and task measurements remain dated. This batch measures a 70.12 kB graph chunk, not the historical 68.40 kB. Bundle size is not evidence of an equivalent latency improvement on users' devices. |
| Verification: security | Historical counts must not be presented as current. The new explicit production scan still reports two moderate package entries and no high/critical; effective all-dependency scan reports 22 entries. Neither scan tests exploit reachability, and the historical unavailable agent review is still no evidence. |
| Verification: notices | Current About/package notice versions match updated releases, as covered by migration tests. Six Rust tests and adapter tests pass; the local build emits 99 inventory entries. That count alone does not establish byte equality with the October public receipt or a legal compliance conclusion. |
| Verification: static routes | The publisher still creates the route shells, covered by the preceding publisher review. The local Vite build here is not the assembled Quartz/machine/site deployment. Root-route HTTP and service-worker behaviour require separate acceptance. |
| Public activation receipt, 11 October | Runtime commit, hosted runs, distribution commit, public routes, browser measurements and notice hashes are explicitly historical evidence. No new browser or network comparison to those bytes occurred here; the earlier November public generation sample has its own scope. |
| Assisted ingest, 12 October | Optional plan/accept capability does not authorise direct corpus writes. The text requires profile and governed proposal adoption through a separate census/dry-run decision. This pass does not execute or activate assisted ingest. |
| Estate alignment, 12 October | Its statement that VisionClaw becomes a shared-build consumer is an intended consequence, not an implemented fact at the inspected downstream revision. Public and private bundles, stable generation identity and migration artefacts remain independently testable contracts. |
| Re-verification, 31 October | v0.6.1 then v0.7.0 chronology is coherent; current graph 0.2.0/WASM 0.2.2 and drag code match the updated statement. Re-arming seven paths does not refresh old browser/security measurements. The current broader explorer tests supplement, rather than rewrite, the dated checks. |
| Collision gate: wiring and coverage | Fatal workflow lint exists. Seven fresh scenarios confirm knowledge-only coverage and add the pair-exemption/category-parent counterexamples above. Dangling links remain informational. No whole-workflow bypass is claimed. |
| Migration census and receipt limits | Committed JSON records 11,631 considered/unchanged, zero changes/refusals and ready status. It calls the operation syntax migration; delta notes say there was nothing to compile. The difference from the stated file counts is preserved as an unexplained counting boundary, not used as proof of missing content. No apply/rollback or nontrivial parity run occurred in this batch. |
| Collision “contradiction” paragraph | The stated contradiction is not logically established. Allowing Lo RA and LoRa as distinct concepts is compatible with merging Lo RA into a third spelling, Low-Rank Adaptation, while leaving radio LoRa distinct. Decision lines 130–131 explicitly describe that separation. A stale exemption or execution-order question may remain, but the two propositions do not by themselves require an owner to choose one concept meaning over the other. No corpus or decision file was changed. |
| Verification run and owner questions | Historical 41/6/ledger results remain dated. Current 105/6 results have different scope. Vocabulary warnings and collision errors are distinct predicates. Mention of a landing-branch merge conflict is historical branch context, not an instruction to merge or modify another worktree. |

## Comparison and user utility

[Sigma's interaction documentation](https://www.sigmajs.org/how-to/interactivity/interactions-events/) makes gesture ownership explicit: an embedded graph can consume gestures, share them with page scrolling, or leave scrolling to the page. That is a useful comparison for this host's drag/navigation design. It does not establish that Sigma is faster or that switching libraries would repair corpus freshness.

[Cytoscape.js](https://js.cytoscape.org/) provides graph modelling, analysis and interactive rendering. These are adjacent component capabilities. VisionFlow's proposed distinct value is the governed path from authored source to compiled relationships and usable evidence in different consumers; a renderer alone is not that product, and the current findings show where that path still loses trust.

A useful evaluation should ask a reader to locate a concept, traverse a relation, open its complete source prose, identify the generation and distinguish unavailable evidence from an unwritten concept. Repeat with keyboard navigation, reduced motion, a changed corpus, a broken compressed object and a mobile page that must scroll around the graph. Measure task success, omissions noticed, stale-content detection and recovery, alongside load time and interaction response. These are recommended user tests, not claimed market superiority or measured usability results.

## Catalogue verification

Strict full-corpus checks pass for 181 topics, 1,605 panels and 24,639 revision-verified citations: zero warnings, zero working-tree fallbacks and nine existing allowlisted generated-file references. Index/verification freshness checks pass. VG-06 has 51 verified citations and all six renders were visually inspected; two clipped edge labels were shortened and re-rendered. These checks do not establish whole-estate semantic or runtime acceptance. Local report links and `git diff --check` pass. Unicode inspection finds no suspicious characters; the style scan flags ordinary sentence openers and the technical noun “harness”, which were reviewed without treating a heuristic as proof of authorship.

The [retention manifest](evidence/2026-11-01/vg-explorer/retention.json) covers local build outputs, six SVG/Mermaid/PNG sets and execution evidence; all 407 exported source files match the separate source manifest.

## Reproduction and coverage ledger

Export the pinned publishing explorer into a separate executable workspace. Install from its lock with `npm ci --ignore-scripts --include=dev`; run `npm test`, `npm run build:check` and `cargo test --locked --offline --manifest-path tools/runtime-notices/Cargo.toml`. Copy the committed `audit-boundaries.test.ts` to its `tests/` directory and run `node_modules/.bin/vitest run tests/audit-boundaries.test.ts` **from the explorer root**, using an executable `TMPDIR`. The [collision harness](evidence/2026-11-01/vg-explorer/collision-probes.py) takes the exported `scripts/ontology-hygiene` directory and runs only temporary fixtures. Review the retained dependency installation and audit outputs with their explicit scope.

| VisionGraph coverage | Dated source/section assessment |
|---|---|
| VG-01; ADR-VG-003/004/008 | [Vault, Bases and domain contracts](2026-11-01-visiongraph-vault-and-domain-contracts.md) |
| VG-07; ADR-VG-005/006 | [Source admission and header importer](2026-11-01-visiongraph-source-admission.md) |
| VG-03/04/05; ADR-VG-001/002 | [Publication and generation identity](2026-11-01-visiongraph-publication-and-identity.md) |
| VG-06; ADR-VG-007 | This review: five original panels corrected plus one added content panel. |

All 27 current visionGraph panels are accounted for across these six topics. Numbering gaps are historical, not missing files. Other intake programmes, current public browser/consumer acceptance, nontrivial profile migration, implementation findings and the wider linked-estate review remain open. Coverage of these diagrams and eight decisions does not establish that every programme or runtime path in visionGraph has been audited.
