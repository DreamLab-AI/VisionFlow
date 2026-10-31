---
title: Obsidian plugin decision and lifecycle audit
status: in-progress
date: 2026-10-31
type: explanation
---

# Obsidian plugin decision and lifecycle audit

This pass covers the remaining four plugin ADRs at `272259edcda1fbc2fc5856ffa633f1a9a01a670b`, alongside the [ADR-OBS-005 and write-boundary review](2026-10-31-obsidian-boundaries.md). The five local decision files have now been read and assessed at the depth stated below. This does not certify engine semantics, distribution artefacts or installed-desktop acceptance. The source files inspected have no local diff; ADR-OBS-003 has a pre-existing uncommitted October 21 clarification, which is working-tree evidence rather than part of that commit. No plugin source was modified. [Probe receipt and source hashes](evidence/2026-10-31/obsidian-lifecycle-probes.json) record the bounded results.

## ADR-OBS-001: shared graph and distribution

This record has no section headings beyond its title; its paragraphs contain distinct obligations.

| Obligation | Evidence and disposition |
|---|---|
| Shared engine and graph ownership | `engine.json` pins v0.4.0, archive SHA-256 and source `cb563e8d`; `scripts/build-wasm.mjs` validates archive bytes and source receipt for the release path. `package.json` consumes the versioned graph archive. `src/views/graph-view.ts` imports `hierarchyGraph`, `mountGraph` and `neighbourhood`. Source supports composition; no archive was fetched or independently authenticated in this pass. |
| Typed graph and visible reasoning limits | `GraphView.refresh` derives graph data from the engine index, displays incomplete classification and a “classify to see inferred edges” message. The plugin has a manual Classify button. The actual dashed-edge renderer and identity rules are in the pinned shared package, outside this host trace. |
| Classification remains explicit | The graph view exposes an explicit button, but `SparqlBlock.run` automatically classifies when an inferred query returns `not_classified`. The record needs a scope qualification: a saved inferred query can trigger reasoning without pressing Classify. This is a behavioural distinction, not evidence that reasoning is incorrect. |
| Ready-to-install offline distribution | `scripts/package-release.mjs` stages runtime files, licences and notices, creates a ZIP and checksum. `manifest.json` marks desktop-only and offline by default. Source proves packaging intent; no fresh package, installed ZIP or desktop run was verified. |

## ADR-OBS-002: earlier ingest proposal and estate notes

| Section | Disposition |
|---|---|
| Product decision | The one-vault host/core split remains visible in the API adapters. The proposed model action now has implementation, so “future optional action” and “none in this plugin” are historical. Model output is still passed through shared accept and a visible confirmation path. The upstream OntoCast method/licence statements were read as attribution history; no source-lineage or licence compliance certification is made here. |
| Where the work belongs | `main.ts` owns endpoint settings/session credentials and invokes the shared plan/accept API. `api.ts` delegates plan/accept to the worker client. This supports the stated boundary. The full native/WASM equivalence and installed-vault acceptance obligations remain separate. |
| Findings from the reorganisation ADR fleet | Explicitly dated October 12. These cross-repository status notes must not override current consumer code, pins or later decisions. Each linked consumer remains in the estate queue; this pass does not revalidate their statuses. |
| Drafts to carry into the higher-level ADR set | Proposed owner actions, not adopted cross-estate obligations. ADR-OBS-003 narrows this proposal to a plugin contract. Preserve this provenance and link the operative implementation rather than counting the draft as another implemented feature. |

## ADR-OBS-003: ingest, credentials and network boundary

| Section/clause | Source assessment and outstanding gap |
|---|---|
| Context/status | The committed proposal labels say none/inactive, while source has `extractFacts`, the manifest declares optional endpoints, and the pin is engine v0.4.0. The existing dirty clarification recognises this; it is not a landed ADR reconciliation or independent proof of a shipped release. |
| Decision 1: optional command | `main.ts` registers/configures the ingest action and `extractFacts` rejects a missing endpoint or missing session key and serialises runs. A local unauthenticated endpoint cannot use this command without supplying some key; the transport itself can send without a key. Review this UX constraint against intended local-model use. |
| Decisions 2–3: only network module and offline disclosure | `ingest-transport.ts` supplies `postJson`; ingest and optional Jev share it. One primitive does not mean one destination or one request: ingest sends once per chunk, and Jev may use a separate configured endpoint. `settings.ts` discloses both options. The old “only ingest” wording no longer describes every optional network action. |
| Decision 4: credentials | Ingest reads `this.ingestKey`; `onunload` clears session key fields and marks the ingest token aborted. `postJson` sends the key in an Authorization header. “Never the credential” must refer to the model prompt, not to the HTTP request. No endpoint was contacted and no key was inspected. |
| Decision 5: user review and stale note | `extractFacts` sends responses through `ingestAccept`, then `confirmIngest` and `applyPatch`. `applyPatch` compares exact base text in the atomic vault callback. The previously reproduced post-write/index-failure gap remains. Undo through real Obsidian was not tested. |
| Decision 6: dependency and bundle gates | The engine pin and checksum loader exist. `scripts/check-bundle.mjs` counts a named network primitive and checks module placement in an unminified build. It explicitly does not establish an adversarial network sandbox. It also uses a stub worker in that locality probe; shipped-worker isolation needs the separate runtime gate. |
| Gates before shipping | Default TypeScript tests passed: six files, 66 tests. They include mocked network transport. They do not execute `.mjs` WASM harness tests or the `.cjs` installed/offline suite. No desktop, actual redirect, cancellation/undo UI or release acceptance was run. |
| Consequences | Shared semantic ownership remains supported. Optional Jev broadens the network and interaction surface beyond the original proposal. Mobile remains unsupported by the manifest. |
| Redirect amendment | `postJson` passes no redirect-control flag and cannot inspect a final URL. Rejecting an observed 3xx in `complete` cannot prove the underlying host did not already follow a redirect. The dirty clarification states this correctly. The stronger “only the endpoint typed” claim in settings is therefore unproven as an enforcement guarantee. |
| Dirty source clarification | Accurately identifies the implementation/pin and transport limitation in the inspected source. Preserve it without claiming the release or desktop redirect evidence it explicitly leaves unverified. |

**Cancellation limit:** `postJson` races the request against a timer but does not abort the underlying request. `callModel` checks its abort token between chunks; the host checks it after the model stage. Timeout/cancel should therefore mean stop waiting or stop subsequent work, not recall bytes already sent or prove the server stopped processing. After optional Jev judging, the host has no second `run.aborted` check immediately before confirmation. Test unload/cancel during that stage before making a full cancellation guarantee.

## ADR-OBS-004: in-note SPARQL

| Section/clause | Assessment |
|---|---|
| Context | The move from panel-only queries to saved fenced blocks is implemented by `registerSparqlBlock`. The upstream Dot Triples claim is attribution context, not independently re-audited source. |
| Decision 1: reactive lifecycle | Initial subscription and 400 ms debounce exist, but the block subscribes directly to `plugin.api.onChange`. Unlike GraphView's `plugin.subscribe`, it does not survive API replacement on reindex. An isolated probe confirmed zero subscribers on the new API. |
| Decision 2: references | Scanner/rewrite supports `__THIS__` and wikilink aliases/headings; host resolution uses Obsidian's link resolver and engine entity lookup. However, replacement has no token boundary: `?__THIS__` is rewritten into invalid `?<urn:note>`. Confirmed with the actual scanner. |
| Decision 3: dataset comment | Default is asserted. The scanner accepts the first matching comment anywhere, not only a leading comment. A trailing `# dataset: inferred` changed the chosen dataset in a probe. This is a bounded documentation/implementation mismatch. |
| Decision 4: literal safety | Scanner skips quoted strings, IRI tokens and comment bodies; unit tests cover rewriting. This does not repair the separate variable-token boundary defect. |
| Decision 5: unresolved references | `rewriteSparql` returns unresolved references before query execution; `run` renders the reason. This source path supports refusal of half-substituted patterns. |
| Decision 6: query contract | Blocks invoke the same API query with the configured row limit, and retry after classification on `not_classified`. Refusals for SERVICE/UPDATE/FROM belong to the pinned WASM engine, not this host source; no pinned-engine execution is claimed here. |
| Not adopted | No inline-field parser exists in the inspected scanner; broader semantic parsing is delegated. This bounded observation is consistent with the decision, not a whole-estate absence claim. |
| Attribution | Source comments identify the method and independent implementation. Legal completeness and upstream provenance require a separate distribution review. |

### Confirmed lifecycle counterexamples

The probes bundled the real `SparqlBlock`, scanner and `GenerationGuard` modules with esbuild. Minimal fake Obsidian elements and an API double exercised the host contract; no actual vault, network or browser was used.

| Trigger | Observation | Impact and acceptance |
|---|---|---|
| Two simultaneous blocks in `same.md` | Both use `sparql-block:same.md`; real guard rejects the first as `superseded` while the second succeeds | Independent blocks can invalidate each other's reads. Give block instances independent channels and prove simultaneous execution in one note. |
| Replace `plugin.api` after blocks load | Old API retains two subscriptions; new API has zero | A visible block can stop following index changes. Subscribe through the stable plugin event surface and prove refresh after exclusion-triggered reindex. |
| Unload block while its query is pending | The completed query still writes `true` to the element after unload | Unsubscription does not invalidate in-flight work. Gate post-await renders on lifetime/request identity and verify detached blocks remain untouched. |

The shim's API uses the actual guard semantics: it issues request IDs on the supplied channel and resolves after both blocks have started. This establishes a host-level counterexample; installed Obsidian rendering order and observed frequency remain unmeasured. Existing offline tests render and await blocks serially, which does not cover the simultaneous same-note case.

## Priorities and diagram correction

Plugin maintainers should address lifecycle identity/reindex recovery and persisted-write reporting first, then clarify network/cancellation promises and scanner token boundaries. Acceptance requires both isolated regression cases and the real host paths; passing pure-module tests alone is insufficient.

OM-06.4 previously asserted a host-side hash check from engine-only sources. It now says the host owns guarded persistence and labels the subsequent index upsert separately. The actual Obsidian host compares complete text. All five OM-06 diagrams rendered after wrapping the expanded message; the first attempt exceeded the 4,500 px ceiling and was corrected. No whole-topic revision stamp was advanced. The plugin still lacks dedicated topic coverage in `docs/diagrams`; its lifecycle needs a topic with its own repository revision mapping before the corpus can claim full estate coverage.
