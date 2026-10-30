---
id: ADR-2016
title: Make Obsidian a local application of the shared ontology core
date: 2026-10-11
decision_status: accepted
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
depends_on: [ADR-2014, ADR-2015]
verified_commit: 813a759
verified_paths: []
owner: jjohare
review_trigger: plugin capability, target support, model integration or persistence change; activation event — the ontomark-obsidian plugin is released for installation outside the test vault
repo: visionflow
domain: BASELINE-visionflow.md
---

# ADR-2016 — Make Obsidian a local application of the shared ontology core

## Context

The owner wants maximum useful Obsidian functionality with even broader utility for estate consumers. Users should need no knowledge of the estate. Local processing and Markdown authority are settled requirements; OCR and non-Markdown document ingestion are outside the present scope.

## Decision

Ship the self-contained plugin from [DreamLab-AI/ontomark-obsidian](https://github.com/DreamLab-AI/ontomark-obsidian), with bundled portable Rust/WASM capabilities and Obsidian adapters. Ontomark owns the core engine and shared `packages/graph`; the separate plugin repository owns installation, Obsidian APIs, vault events, commands, settings and host UI. It opens ordinary vaults without imposing estate folder names, namespaces or governance headers. Installation requires no database, account, service, model download or CLI. Existing notes remain readable and editable without the plugin.

Expose substantial functionality through progressive UI: typed relation editing, vocabulary inspection, identity/reference diagnostics, supported reasoning and its explanations/limits, graph navigation, local queries, export preview and guarded Markdown changes. Keep ordinary links and formal axioms visibly distinguishable. A normal vault with no ontology metadata starts with document navigation and explicit tools to introduce typed structure; it is not silently converted into a class hierarchy.

The same crates support optional richer capabilities for either plugin or estate consumers: vocabulary profiles, query evaluation, provenance, reviewed Markdown-assisted extraction, publication and federation adapters. The plugin does not need to expose every estate feature. Optional integrations are described after the standalone README quickstart and are disabled until configured.

Markdown plus machine-readable headers are authoritative. An in-memory graph or disposable local cache may accelerate interaction; deletion of that cache must be recoverable entirely from source documents and profile configuration. No background process mutates accepted ontology assertions merely because a model assigned high confidence.

The initial query capability supports local read-only SPARQL SELECT, ASK and CONSTRUCT over explicitly selected asserted, inferred or combined datasets. Basic triple patterns, joins, FILTER, OPTIONAL, UNION, ORDER BY and LIMIT are the minimum proposed contract; enumerate any additional supported forms. Unsupported syntax receives an explicit diagnostic. Apply time/result budgets and cancellation. Disable SERVICE and remote dataset loading; SPARQL UPDATE does not mutate the source. Public-view queries operate only on the public projection. Test native/WASM result parity and dataset selection before release.

Markdown-assisted automation follows source text → evidence-bearing candidates → identity/vocabulary resolution → reviewed patches → validation/reasoning. Candidate records retain source digest, span coordinate convention, quotations, model/rule version and rejected/uncertain dispositions. Preserve the full source note; avoid invented superclass, equivalence or causal assertions. Model-free structured-header processing is always available. Optional local inference has explicit installation, capability and resource requirements; it cannot be required for normal operation.

The application uses cancellable workers, generation-tagged results and transferable buffers by default. Obsidian file events invalidate affected inputs; full rebuild remains the correctness oracle until incremental equivalence is proved. Writes verify source revisions and preserve foreign headers and prose. Unload tears down workers, listeners and views. Public export is a separate previewed operation from private local viewing.

## Consequences and tensions

Rust is the common logic layer, not a requirement to rewrite host UI APIs or force native consumers through WASM. Mobile support is a target to prove, not an automatic consequence of compiling to WASM. Node/Electron APIs are unavailable on mobile; use the vault API and test actual supported devices. Desktop can ship first with accurate manifest limits if mobile gates fail, without introducing a required server fallback.

Borrow evidence, uncertainty and extraction-contract patterns from permissive projects such as EdgeQuake and edgextract; adoption of either runtime is not decided. EdgeQuake's database architecture does not fit the default product. edgextract's browser bridge currently requires cross-origin isolation, so it is not a drop-in Obsidian component. Formal extraction schemas do not replace OWL semantics.

## Verification

Required installed-plugin tests run with network disabled on a plain vault and the estate fixture: parse, edit, rename, delete, validate, reason, inspect evidence, navigate, query, export and unload/reload. Check malformed headers, ambiguous references, interrupted updates, stale worker results and non-ASCII source spans. Model-assisted tests are separate from deterministic core tests.

Portable Whelk execution and desktop packaging now have the bounded evidence below. Mobile support, broad desktop compatibility and resource-budget acceptance remain unproved. The [delivery gates](../../../visionGraph/planning/ontology-workspace/README.md) prevent a renderer-only stub being counted as completion. Consult [Obsidian mobile guidance](https://docs.obsidian.md/Plugins/Getting%20started/Mobile%20development) and [release packaging](https://docs.obsidian.md/plugins/releasing/submit-plugin) when implementing.

## Desktop implementation continuation — 2026-10-11

The [standalone plugin README](https://github.com/DreamLab-AI/ontomark-obsidian/blob/ebf8fd6/README.md)
and [installed acceptance receipt](https://github.com/DreamLab-AI/ontomark-obsidian/blob/ebf8fd6/docs/desktop-acceptance.md)
record Obsidian 1.14.4 on Linux with synthetic notes: the real Blob worker compiles
the embedded rc1 engine, indexes, classifies, answers local queries, applies a
reviewed edit, refuses a concurrent edit and unloads/reloads. Shared graph checks
cover search, keyboard activation, zoom, neighbourhood focus, inferred-edge
filtering, note opening and cleanup. The separate network-isolated Obsidian shim
passes 40 checks; its result is not substituted for the installed application.

Distribution comprises `main.js`, `manifest.json`, `styles.css`, an installable ZIP
and licence notices, with the engine embedded. Node and Rust are build tools, not
end-user prerequisites. The plugin consumes the immutable Ontomark graph 0.1.0
package; it does not own another renderer or codec. This is partial delivery of
the stated capability scope. Mobile is unsupported, and model-assisted extraction,
all proposed editing workflows, large-vault budgets and broad target coverage
remain separate acceptance work.

## Assisted ingest core continuation — 2026-10-12

The "Markdown-assisted automation" chain above (source text → evidence-bearing candidates → identity/vocabulary resolution → reviewed patches) now has its first shared implementation in Ontomark: `crates/ontomark-ingest` at [`582cfa0`](https://github.com/DreamLab-AI/ontomark/commit/582cfa0), specified by [ADR-OM-003](https://github.com/DreamLab-AI/ontomark/blob/main/docs/adr/ADR-OM-003-native-assisted-ingest.md). It is a pure `plan`/`accept` contract with no I/O and no model call: the host supplies the model; the core grounds proposed names against the vault vocabulary (never trusting model identifiers, after SPIRES), locates a verbatim quote for every value as byte and UTF-16 spans, checks literals against their XSD datatype, and returns one source-digest-bound patch with every refusal and its reason. Candidate records carry the digest, spans, quotes, contract version and withheld dispositions this ADR requires. Facts only, against existing vocabulary; no invented superclass, equivalence or causal assertions. Methods were adapted from OntoCast (Apache-2.0) under attribution; neither OntoCast nor edgextract is a dependency.

Status of this ADR is unchanged. The plugin has no ingest action; exposing one needs an engine prerelease that surfaces `plan`/`accept`, an `engine.json` repin, an explicit amendment of the plugin's offline contract for the host-side model call (off by default, user-configured endpoint, disclosed in README and manifest), and installed-Obsidian acceptance with the network-isolated gate unchanged while ingest is disabled. Those are recorded in Ontomark's ADR-OM-003 and the plugin's assisted-ingest ADR.

**Engine boundary landed (same day).** The portable engine now exposes the contract as `ingest_plan`/`ingest_accept`, covered by the native/WASM transcript, and the EXP-10 profile mapping this ADR's module docs required exists: each engine profile implements `ontomark_model::Profile`, with a parity test proving the engine projection and the typed model index the same entities. Ontomark's ADR-OM-004 records why OWL 2 EL remains the local profile and where non-EL constructs are refused by name. The plugin gates above are unchanged and still open.
