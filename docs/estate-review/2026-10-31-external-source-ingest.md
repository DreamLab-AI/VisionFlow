---
title: External-source ingest, evidence and replay boundaries
status: in-progress
date: 2026-10-31
type: explanation
---

# External-source ingest, evidence and replay boundaries

Ontomark's external-source ingest produces reviewable proposed pages without invoking the corpus writer. Its current completion and identity signals need narrower interpretation: malformed model replies can produce a “complete” bundle, and the same request ID can name different contents after an endpoint or budget change. Quote validation grounds names and locates text; it does not establish that the text supports the proposed relation.

This pass examines ADR-OM-010 section by section at Ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41`, using clean source, a freshly built CLI and loopback fixtures. [Source and binary hashes](evidence/2026-10-31/ingest-sources.json), [Rust probe](evidence/2026-10-31/ingest-boundary-probe.rs), [results](evidence/2026-10-31/ingest-boundary-probe.json). No real source document, external model, live vault, forum or signer was used. Findings concern proposal preparation; they do not demonstrate unauthorised corpus mutation.

## Reproduced boundaries

| Case | Observed result | Implication |
|---|---|---|
| Model returns `not JSON` for the only chunk | Exit 0, `complete: true`, no pages, `dropped_chunks: [0]`; source recorded in ledger. Repeat reports unchanged but makes another model call. | A caller cannot use completion or unchanged status to distinguish successful extraction from an unparseable answer. |
| Same source, target and model name; endpoint A returns a supported relation and endpoint B returns an empty assertion list | Same request ID and default bundle directory; B removes A's manifest and replaces the evidence. B is correctly labelled `parser_changed`. | The request ID does not identify immutable review content or the endpoint policy that produced it. |
| Complete bundle, then zero-call rerun with a cached response | Zero calls, one cache hit, same proposed page. | The intended cache/budget interaction works. |
| Remove that fixture cache, then repeat with the same zero-call budget | Same request ID, `complete: false`, `budget_exhausted: true`; previous manifest removed. | The refusal is visible, but it destroys the previous review artefact before writing the incomplete replacement. |
| Ingest two source files, delete one, rerun | Summary lists one source, ledger retains both entries, completion true. | The CLI does not report or retire the removed source even though the library classifies it. |
| Model proposes `requires: Gamma` with the exact quote “The Quantum Widget does not require Gamma.” | Positive relation accepted at exact grounding tier, confidence 0.5 and method inferred; quote preserved. | Entity grounding and evidence location are not relation entailment or contradiction detection. |
| URL source with `--plan-only --budget-calls 0` | Loopback recorder receives HEAD and GET. | Planning and call budgets do not make source acquisition offline. This matches the help text's promise to read sources. |

The fixture's target page remained byte-identical throughout. These are deterministic counterexamples using controlled model replies, not a measurement of a real model's extraction error rate.

## Completion, cache and bundle identity

`ingest_cmd.rs:396` treats malformed assertion JSON as an absent response. At `:735` the host records the dropped chunk without marking the source unfinished or adding a skipped pair. Completion is calculated only from skipped pairs and stop state (`:821`), while the source ledger records entries still marked finished (`:842`). The evidence exposes dropped chunks, but the summary's completion flag does not incorporate them. A valid empty assertion list is a separate case and should remain distinguishable from malformed output.

Three identities serve different purposes. The parser profile includes endpoint and the complete selected-vault fingerprint (`:551-576`). Response cache keys include the request body, model and endpoint (`:444-484`). The bundle request ID includes source IDs/versions, selected target IDs/digests, model and options, but excludes the endpoint, budget and actual answers (`:606-619`). A repeat can therefore be a parser change while retaining the same bundle ID. The default path is derived from that ID, and any existing manifest is removed before replacement (`:765-785`). Evidence is then overwritten in place (`:834-838`); it and the manifest are not published as one atomic directory transaction.

The endpoint and cache-removal fixtures establish actual replacement. Crash/concurrent-writer outcomes were not exercised. Acceptance should distinguish an input request ID from an output-content identity, preserve a reviewer's previous bundle on incomplete runs, and define how dropped chunks affect completion. Neither a request ID nor a successful exit currently certifies a complete immutable artefact.

## Evidence and source lifecycle

The pure engine recomputes target digest, source reference/version and chunk boundaries before acceptance (`accept.rs:321-368`), then checks allowed keys, entity identity/kind, quote location and whether the quote mentions the value (`:415-535`). This is useful structural protection. The negation fixture passes those checks because they do not test the subject–predicate–object claim's meaning. It belongs in reviewer-facing evaluation alongside attribution, conditionals and contradictory sources; the fixture does not imply that semantic judgement should be replaced by a small negation-word filter.

The host emits relation keys from the vocabulary-derived profile (`ingest_profile.rs:51-110`). Its `review` list selects `ConflictingValue` and `CompetingValues` (`ingest_cmd.rs:790-799`), which the engine raises for single-valued literals. The ADR's implementation appendix correctly says those conflicts are currently unreachable in this relation-only path. Zero review items is therefore not evidence that sources agree with the corpus. Opposed relation assertions and the negation fixture need a different review signal.

`ledger.rs:219-269` returns observed classifications followed by removed IDs. The host zips those results with the observed items (`ingest_cmd.rs:223`), dropping the trailing removals. It never calls `Ledger::forget`. The shared ledger also spans source specifications from different invocations: blindly deleting every absent entry would be wrong when a caller intentionally selects only a subset. Define the listing scope before adding removal handling. Nothing currently retracts earlier accepted corpus relations when a source disappears; such a policy would require a separate governed proposal.

The bundle records quotes, spans, source versions and target digests, but not the full rendered source, original target snapshot or complete model request (`ingest_cmd.rs:802-838`). A version hash detects disagreement with bytes obtained later; it cannot recover a changed or deleted source. Consequently, the ADR's reproducibility claim depends on separately retained matching inputs and runtime/profile configuration. The response cache is also outside the bundle. Rechecking a complete source hash and replaying a run from the bundle alone are not established.

## What leaves the process and what budgets cover

The model receives instructions/schema, the target label, fillable keys and existing values, plus source chunk text (`ingest_cmd.rs:340-392`). It may receive private external-source content even though targets are restricted to the knowledge vault. Working/all selections are explicitly refused. Model endpoint policy accepts declared-local endpoints, local-looking IP/name forms or cloud opt-in (`ontomark-jev/src/config.rs:141-158,231-247`); its documented name policy performs no DNS verification. Model requests disable redirects (`ingest_cmd.rs:415-441`). This pass did not test DNS, proxies or cloud transport.

URL sources have a separate HTTP client: HEAD is best effort, GET fetches the body, redirects are capped at five, and content is bounded to 5 MiB by default with a streaming limit (`url.rs:40-48,57-62,100-165`). They are not subject to the model endpoint's local-only rule. Local file reads default to 16 MiB each. Model call/character budgets are optional, checked after source acquisition and after the response-cache lookup (`ingest_cmd.rs:444-472`). The character count covers message content, excluding the response schema and other HTTP-body fields; it is not a token or monetary cost bound. The 64-chunk limit is per source/target pair, not a total acquisition or run budget. Explicit targets are outside the discovery cap, which applies per source. No aggregate source-count, byte or whole-run deadline was found in the inspected host path.

These distinctions should be visible wherever a user chooses an ingest endpoint or budget. “Local model” does not mean “no URL fetch”, “plan only” does not mean “offline”, and “zero model calls” does not mean “no source reads”. The CLI's help is appropriately narrower than those interpretations.

## ADR-OM-010 section assessment

| Section | Assessment at this revision |
|---|---|
| Metadata | Proposed/none/inactive remains the owner's recorded status, while the appended implementation section documents landed code. Implementation-none is stale as a literal inventory claim; inactive remains consistent with no verified live activation. No status was changed. |
| Context | Pure note ingest and the new external-source boundary are supported by source. The upstream mechanism credits are recorded claims, not a fresh comparative assessment in this pass. Numeric-token ambiguity is addressed by explicit guards and tests. |
| Decision 1 | External evidence binds to source versions and the target digest; source fields are optional for note-only output. Numeric guards run below exact/shared-label, including initials (`similarity.rs:210-216,314-356`). Focused tests pass. Full visionGraph label/alias regression and plugin golden parity were not rerun here. |
| Decision 2 | Local files and feature-gated HTTP sources implement describe/list/fetch and refuse unsupported types. Library removal classification exists; host reporting/retirement is incomplete. Rendered-text hashes do not preserve source history. |
| Decision 3 | Explicit/discovered targets, local endpoint policy, cache and optional call/character budgets exist. The cap covers discovery per source. Request identity omits endpoint and output, and malformed replies violate a broad reading of successful repeat/completion. |
| Decision 4 | Host writes proposed pages/evidence, prints a separate proposal command and never invokes it. Quotes stay out of proposed page bodies in the tested path. No provenance keys are added. Conflict review currently cannot detect semantic relation disagreement. Existing corpus body text is copied into proposed pages; there is no general content sanitiser here. |
| Decision 5 | No new-concept creation, PDF/office connector, SaaS connector or signed-decision replay is implemented in this path. Their exclusion remains coherent. |
| Consequences | Source-port convergence is prospective. Bundle-only replay is not supported by the retained data. Separate proposal submission is a host convention; it does not establish that no other corpus write path exists. Plugin repinning and wire compatibility need their own evidence. |
| Verification 1–3 | Numeric guard/source-contract tests pass. Whole-corpus grounding and cross-version plugin wire comparison remain unverified in this pass. Passing small fixtures do not close those gates. |
| Verification 4–5 | Real CLI plus stub-model suite passes; the installed vault dry-run actually ran, produced an amend proposal and reported no blockers. Ordinary cached reruns pass. Malformed replies and changed cache/budget state add the exceptions above. |
| Verification 6 | No real bundle was proposed, signed and applied. Activation remains open. |
| Implementation appendix | Accurately describes most host mechanics, including relation-only conflict limitations and inactive status. Reproducibility and completed-run interpretation need the qualifications above; removed sources are omitted from its host classification list. |

## Validation and remaining work

The selected suites passed 90 unit/integration tests across ingest and source, 17 doc tests, and nine CLI integration tests. [Engine/source log](evidence/2026-10-31/ingest-source-suite.txt), [CLI log](evidence/2026-10-31/ingest-cli-suite.txt). A targeted rerun with captured output confirms the optional vault hand-off test was not silently skipped: [dry-run receipt](evidence/2026-10-31/ingest-propose-dry-run.txt). Its consequence verdict is `not_conservative_under_rule_table`, despite zero blockers; this is accepted proposal construction, not approval or application. No fixture was published.

The standalone Rust probe uses only the standard library plus the checkout's compiled `serde_json`, invokes the built binary, and removes its temporary corpus. To repeat, compile it with `rustc --edition=2021`, `--extern serde_json=<ontomark>/target/debug/deps/libserde_json-<hash>.rlib` and `-L dependency=<ontomark>/target/debug/deps`, then pass `<ontomark>/target/debug/ontomark` as its sole argument. Run the executable from an executable filesystem. Temporary paths and loopback ports change between runs, so IDs derived from source paths can change while the asserted relationships remain stable.

[OM-08](../diagrams/ontomark/08-external-source-ingest.md) adds the current host path. OM-05 now labels its older command inventory as historical and links to it. Full activation, plugin compatibility, source retention, semantic-quality evaluation, safe bundle publication and the remaining estate ADR/diagram queue remain open.

Diagram gates: OM-08 and OM-05 passed strict citation checks (59 and 138 references respectively), and all nine diagrams rendered. The [whole-corpus citation run](evidence/2026-10-31/ingest-citations.txt) passed across 161 topics and 1,534 diagrams: 23,810 references resolved at declared revisions, no fallbacks or unresolved references, and nine allowlisted generated-file citations. The verification anchor is fresh for all 37 declared revisions. Eight local file links in the new report/topic resolved. These checks establish structural validity and citation resolution; they do not establish whole-estate semantic review or deployed behaviour.
