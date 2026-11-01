---
title: Obsidian plugin — current decisions, release and host lifecycle
status: in-progress
date: 2026-11-01
type: explanation
---

# Obsidian plugin: current decisions, release and host lifecycle

All five plugin ADRs were reread section by section at `2a6c44e024121ec055329d37915e0ce5a05f75f2` (plugin 0.4.2). Their earlier assessments were correctly marked stale after the documents changed. This refresh restores bounded document-assessment credit, adds the missing [plugin diagram topic](../diagrams/ontomark/09-obsidian-host-lifecycle.md), and extends execution evidence to the current pinned engine and built plugin. It does not close the broader estate audit or certify installed-desktop behaviour.

The fresh production build verified engine v0.7.0's archive SHA-256 and source receipt `0caae5d4b695cb720942b9b9212017c00a80ad41`. Six unit files passed 66 tests; all four WASM harness tests passed; the bundle check reported no findings; the built plugin passed 69 checks both locally and under Docker `--network none`. Six isolated current-source probes still reproduce the previously reported host failures. Passing the normal suites and retaining those failures are compatible: the suites do not exercise the same concurrency and failure cases.

[Verification and hashes](evidence/2026-11-01/obsidian-refresh/verification.json), [executable probes](evidence/2026-11-01/obsidian-refresh/probes.mjs), [probe results](evidence/2026-11-01/obsidian-refresh/probes.json) and [network-isolated shim report](evidence/2026-11-01/obsidian-refresh/offline-docker-report.json) are retained. Source exports under the same evidence directory are byte-exact, including the five ADRs and relevant implementation, fixtures' harnesses and release tooling. The source checkout was clean and remained unchanged; execution used a disposable archive.

## What changed since the earlier assessments

The comparison baseline is `272259edcda1fbc2fc5856ffa633f1a9a01a670b`. The new estate-ledger frontmatter records decisions without proving implementation: all five `verified_paths` arrays are empty. ADR-OBS-001, 004 and 005 explicitly say their `verified_commit` is the ADR's last-modified commit, not a code revalidation. The pinned checker validates all five records and their generated index; this structural success does not arm source-staleness coverage.

ADR-OBS-001 now describes three loose release files instead of a ZIP, and the release workflow matches. Engine and graph pins have advanced to v0.7.0 and graph v0.2.0. The October 21 clarification in ADR-OBS-003 is now committed. The previous report's description of it as a dirty local annotation remains historical, not its current state.

The only production TypeScript diffs are a lint-comment clarification in `query-templates.ts` and wrapping an asynchronous settings event handler in a `void` async closure. The main host, API, transport, block renderer, scanner and generation guard relevant to the earlier findings are unchanged. Those findings were nevertheless rerun against the current source; they are not refreshed merely by copying old results.

## ADR-OBS-001: shared graph and distribution

| Paragraph or obligation | Current assessment |
| --- | --- |
| Status mapping and verification notes | Accepted/none/inactive is expressly a conservative migration of documentary status, not a reliable statement that no graph implementation exists. Keep owner status confirmation open. Empty verified paths provide no implementation freshness check. |
| Engine versus host ownership | `engine.json`, the checksum loader, graph dependency and host API preserve the separation: shared engine/graph supplies semantics and rendering; the plugin owns vault events, commands and guarded persistence. The actual archive and source receipt now passed the loader. This does not prove every upstream semantic obligation. |
| Typed graph, identity and classification | GraphView consumes the engine index and exposes incomplete classification. The host offers both manual classification and optional auto-classification (`main.ts:294–296`); an inferred saved query also classifies on `not_classified` (`sparql-block.ts:73–75`). “Classification remains explicit” needs that scope qualification. The host trace does not independently certify graph-package identity or dashed-edge semantics. |
| Release distribution | `.github/workflows/release.yml:14–24` builds/tests, attests and publishes `main.js`, `manifest.json`, `styles.css`. `scripts/package-release.mjs` is removed. Fresh bundling produced those files; installed users do not need the build toolchain. The new no-ZIP description matches source. Published release assets and their attestations were not fetched or verified. |
| Linked acceptance and attribution | The linked desktop-acceptance document remains separate evidence with its own dates and limits. Notices now reside in the repository; the bundler removes legal comments. This is a distribution-review question to resolve against the actual dependency obligations and delivered files, not a finding of licence infringement. |

## ADR-OBS-002: proposal and estate hand-off notes

| Section | Current assessment |
| --- | --- |
| Initial implementation/scope statement | “None in this plugin” describes the proposal's original state. It is contradicted as a current implementation summary by `extractFacts`, plan/accept adapters and optional Jev. Proposed status does not erase shipped source. |
| Product decision | One-vault operation and shared Rust/WASM semantic ownership remain evident. The plugin stays usable with no endpoint. Optional extraction has a reviewed apply path; no Python dependency is introduced by the inspected production build. The historical OntoCast method/licence statements were not turned into a new upstream lineage or legal certification. |
| Where the work belongs | Host settings, credentials, transport and confirmation surround shared plan/accept. The fresh WASM and shim runs strengthen execution evidence but do not satisfy the full installed-Obsidian evidence-span, repeat, cancellation and undo acceptance list. |
| Findings from the reorganisation ADR fleet | These are explicitly October 12 observations across other repositories. They remain historical pointers, not current deployment facts or blanket coverage for the linked ADRs. Consumer adoption belongs in the corresponding repository reviews. |
| Drafts to carry into the higher-level ADR set | These are proposed owner actions. They neither amend the other repositories' accepted decisions nor establish cross-estate implementation. Retain the boundary instead of counting the draft as another delivered feature. |

## ADR-OBS-003: optional ingest and the offline contract

| Section or clause | Current assessment |
| --- | --- |
| Initial status and context | The “not yet in a released engine” statement is historical: the inspected pin is v0.7.0 and the build verified it. The appended v0.4.0 clarification is also dated history. The current README/manifest disclose optional model requests; the original unconditional offline language is not the present source contract. |
| Decision 1: command absence | The command is always registered (`main.ts:559`) and reports missing endpoint/key at invocation. The no-endpoint shim observes that notice. This differs from the proposed command being absent. The host requires a non-empty session key even though `postJson` supports an empty key; a local unauthenticated endpoint therefore needs a UX decision. |
| Decisions 2–3: transport and disclosure | Ingest and optional Jev use `postJson`; one primitive is not one request or one destination. Ingest sends per chunk, Jev can use another configured endpoint and other actions. The fresh bundle locality check is bounded static evidence, and the no-endpoint network-isolated shim passes. Neither proves the configured-endpoint destination promise. |
| Decision 4: credentials | Session key fields are cleared on unload and are passed in an HTTP authorisation header. They are not part of the prompt. No real credential or configured model endpoint was used. |
| Decision 5: review and stale edits | Plan/accept precedes visible confirmation; `api.ts:242–246` compares complete base text inside the vault's atomic transform. The shim verifies approve, stale-source refusal and decline. A separate injected upsert failure still rejects after the text was persisted. Actual Obsidian undo remains untested. |
| Decision 6: engine and bundle gates | The current archive, source receipt, declaration consistency, production build and bundle checks passed. The network-callsite check is a guard against accidental changes, not an adversarial sandbox. |
| Gates before shipping | The 69-check isolated shim strengthens the no-endpoint gate. Configured-endpoint isolation, redirects, actual desktop cancellation/undo and the whole real-host acceptance matrix remain open. The shim loads the built plugin through host doubles; it is not Obsidian. |
| Consequences | Shared semantic ownership and desktop-only scope are supported. Optional Jev extends the interaction/network surface beyond the original single-ingest formulation. |
| Redirect amendment | Absolute HTTP(S) URL validation and rejection of an observable 3xx exist. The host API call supplies no redirect-control option and does not expose final URL inspection. The amendment's destination claim must remain an endpoint requirement, not an enforcement guarantee. |
| Source clarification | The committed clarification correctly narrows the redirect/credential statements. Its release and engine references identify the October 21 state; use the present pins for current operation. |

Timeout races do not abort `requestUrl` (`ingest-transport.ts:85–105`). Abort is checked between chunks and after the model stage. After optional Jev judging, there is no second `run.aborted` check before confirmation (`main.ts:488–501`). The latter is still a source-derived test target: this refresh does not claim to reproduce the Jev unload race.

## ADR-OBS-004: in-note SPARQL

| Section or clause | Current assessment |
| --- | --- |
| Status and verification notes | Accepted/partial/live is a documentary mapping; the record explicitly leaves owner confirmation and code-path staleness coverage open. |
| Context | The fence renderer exists. The upstream Dot Triples README was checked again: its current documented feature set is broader than the dated attribution description. The old assertion that its README lists quoted-string replacement as a known defect is not supported by the current README. Treat it as dated attribution, not a current superiority claim. |
| Decision 1: reactive lifecycle | The block subscribes and debounces at 400 ms, then unsubscribes on unload. The probes confirm that API replacement leaves it without a subscription to the new API and that a pending query can still render after unload. |
| Decision 2: reference rewriting | Note links resolve through Obsidian's resolver and engine entity lookup; aliases/headings reduce to note references. The scanner still replaces `__THIS__` inside a variable name, creating invalid `?<urn:note>`. |
| Decision 3: leading dataset directive | The actual scanner honours the first matching comment anywhere. A trailing directive selected inferred; the leading-only claim is inaccurate. |
| Decisions 4–5: skip literals and refuse unresolved references | Scanner branches skip strings, IRI tokens and comments; unresolved references fail before query execution. Existing unit and shim paths support these bounded cases. They do not establish a complete SPARQL lexer. |
| Decision 6: shared query path and refusals | The block uses the shared API, dataset and limit. The pinned WASM harness exercised inferred queries and a SERVICE refusal; the shim exercises broader query refusals. WASM was compared to the committed plugin golden, not a freshly generated native Rust transcript. |
| Not adopted and attribution | The inspected scanner does not adopt inline-field assertion parsing. Method attribution is retained; the legal/lineage claims were not independently certified. The referenced bridge is a separate upstream path, not newly audited by this host refresh. |

Two simultaneous blocks in the same note also share `sparql-block:<path>`. The actual generation guard marks the first request superseded when the second issues. Per-block independence is missing even though both queries concern the same valid index generation.

## ADR-OBS-005: host-owned exclusions

| Section or clause | Current assessment |
| --- | --- |
| Status notes and context | Partial/live describes the mapped documentary status. The template-placeholder rationale and engine/host boundary are consistent with the source; this pass does not independently re-establish native CLI policy equivalence. |
| Decision 1: top-level folders | `excludes.ts` implements case-insensitive first-segment matching with the templates default; folder-control validation refuses separators and `..`. The unit suite and built-plugin shim cover representative included/excluded paths. |
| Decision 2: native exclusions | Prefix/regex parsing and settings/layout/reindex refresh are implemented. Missing or malformed configuration removes the extra filter; other read errors retain prior filters. This is intentionally fail-open convenience filtering. Independent equivalence to every native Obsidian exclusion case remains unproven. |
| Decision 3: events and reindex | Initial listing, create/modify, deletion, rename into/out of excluded folders and settings-triggered reindex are exercised by the shim. Existing saved blocks lose their subscription after replacement, so correct index membership alone does not prove correct visible refresh. Direct API calls do not acquire an authorisation boundary from this adapter policy. |
| Decision 4: pure rules | The pure module is independent of Obsidian; fourteen exclusion tests are included in the 66-test run. Unlike the earlier mistaken selection, no absent API test file is counted as a test pass. |
| Why host-side / not adopted | Policy sits in the listing/event adapter; the engine indexes supplied files. Path rules avoid treating template-like prose as a reason to exclude a note. This agrees with the decision's rationale, within the stated host scope. |

## Findings and acceptance priorities

| Priority | Trigger and consequence | Evidence and acceptance |
| --- | --- | --- |
| First | An approved file write succeeds, then engine upsert rejects; caller gets rejection without an applied receipt | Reproduced with actual API and injected client failure. Expose persisted-but-unindexed state, prove idempotent recovery and show it in the real host. Preserve the stale-text guard. |
| First | Concurrent blocks, API replacement or renderer unload produce supersession, lost refresh or late writes | Three actual-module probes. Give each block independent request identity, use the stable subscription and invalidate post-await rendering on unload. Add a real desktop multi-block/reindex case. |
| Next | Placeholder inside a variable or a trailing dataset comment changes the query unexpectedly | Two actual-scanner probes. Define token and directive grammar; test variables, literals, links and comments together. |
| Next | Optional endpoints and cancellation are described more strongly than enforced | Source trace plus mocked unit coverage; configured-endpoint/redirect/Jev-unload acceptance remains open. State exactly what is sent, where the host can enforce destinations, and what cancellation stops. |
| Next | Ledger status looks authoritative while code-path bindings are empty | All five source records explicitly show this. Owners should set current implementation/activation values and verified paths after the relevant acceptance, retaining dated history. |

A fresh `npm audit` also reports two moderate entries: a transitive Moment advisory and its Obsidian development-dependency parent. The raw report is retained. The production bundle externalises Obsidian, so this is not evidence that the affected Moment implementation is bundled or exploitable through this plugin. Review the development dependency upgrade separately; no dependency was changed here.

## Comparison and user utility

The three-file release layout follows [Obsidian's documented installation contract](https://docs.obsidian.md/Plugins/Releasing/Submit%20your%20plugin), which downloads the matching release assets. This supports the packaging decision; it does not prove directory admission or attestation validity.

[Dot Triples](https://github.com/cristianvasquez/dot-triples-plugin) currently documents RDF/SPARQL over Markdown and Canvas, an embedded triplestore, optional shared-store queries and external endpoint access. Its wider document-to-RDF and multi-vault surface is a useful comparison. Ontomark's reviewed host instead makes explicit typed knowledge, shared reasoning and guarded note changes central. That distinction is an inference from the documented upstream offering and the inspected local source, not a benchmark or a claim that one subsumes the other.

[Dataview](https://blacksmithgu.github.io/obsidian-dataview/) documents metadata queries, inline fields and JavaScript-based views, and is primarily for display/calculation rather than note editing. For users who need tables of ordinary note metadata, that is already a direct product path. Ontomark should demonstrate where typed relationships, inference and a reviewed change earn their extra complexity.

A useful pilot is a small evidence notebook: keep two saved queries in one note, change a referenced note, change an exclusion, review a patch, inject index failure and recover. Measure correct refresh, time to understand the outcome, preserved source text and successful recovery. These observations would test the plugin's distinctive value and the highest-impact gaps together. No market-share, adoption or performance ranking is inferred from repository descriptions.

## Verification limits and retained failures

The first clean dependency install failed because `/tmp` is mounted `noexec`; its log is retained. Repeating the install in the owned workspace archive succeeded. The initial diagram gate rejected five punctuation/blank citation anchors; corrected anchors passed with revision-pinned citations and four renders. The normal source checkout and its existing build artefacts were not used as fresh runtime evidence.

The WASM suite's two replay tests are named as native comparisons, but their expected value is the committed plugin fixture. Its own comment says the enlarged plugin vault has no native golden writer. This run proves the two WASM targets match that fixture and the additional entropy/clock paths execute; it does not regenerate or independently prove full native/WASM parity. The offline shim demonstrates the built plugin under network isolation and host doubles. Actual desktop installation, real redirects, model-enabled destination controls, undo, mobile, exhaustive resource bounds, licence obligations and published provenance remain outside the verified result.
