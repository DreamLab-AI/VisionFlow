---
title: vowl-wasm ingest semantics and release delivery
date: 2026-11-01
status: assessed-with-open-findings
scope: vowl-wasm and its declared visionGraph package consumer
---

# vowl-wasm ingest semantics and release delivery

Remaining vowl-wasm diagrams overstated JSON compatibility, semantic preservation and release delivery. Native probes reproduce lost ontology fields, incorrect hierarchy classification and order-dependent statistics. Public release evidence shows that 0.2.2 reached GitHub but failed at npm publication; the registry still exposes only 0.1.2. visionGraph uses the GitHub tarball directly.

This pass assesses all 17 panels in VW-01, VW-02 and VW-05, and corrects the package route in VW-04.6. Together with the [layout and API assessment](2026-11-01-vowl-layout-and-api-boundaries.md), all five vowl-wasm topics and all sections of its sole local ADR have a semantic disposition. Implementation fixes, installed-plugin acceptance and the wider estate review remain open. Production source, release settings and ADR status were not changed.

## Evidence and limits

Source revision: `28af4365a68d9aacfd7c04f63299cc96ad717ce3`. The visionGraph checkout is `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`; its declared consumer inputs, including package and lock files, match the retained `7b0875f2…` topic revision. The [input manifest](evidence/2026-11-01/vowl-ingest-source-manifest.json) compares every source/governing association across the five topics. Existing ignored package output and unrelated checkout changes were preserved.

| Check | Result and qualification |
|---|---|
| Native ingest probes | [11 passed](evidence/2026-11-01/vowl-ingest-probes.log), using an archive of the pinned source and an isolated Cargo target. [Assertions](evidence/2026-11-01/vowl-ingest-probes.rs) deliberately demonstrate current defects. |
| Release guard | [Six existing tests passed](evidence/2026-11-01/vowl-release-guard-baseline.log); adding [three audit probes](evidence/2026-11-01/vowl-release-guard-probes.rs) produced [nine passing tests](evidence/2026-11-01/vowl-release-guard-probes.log). These are three new cases, not nine. |
| Current inventory | [Verification passes with recorded toolchain strings](evidence/2026-11-01/vowl-inventory-current.json), establishing selected source/notice hash agreement. Supplying actual local rustc 1.99.1 fails against the recorded 1.99.0, as intended. This is not reproduction of the release toolchain. |
| Workflow control flow | [Two scenarios passed](evidence/2026-11-01/vowl-release-flow-probes.json). The [probe](evidence/2026-11-01/vowl-release-flow-probes.py) executes extracted shell blocks with local cargo/gh/npm doubles. It neither publishes nor exercises OIDC. |
| Prior library/bundle coverage | The preceding report retains 199 passing native tests, three ignored, five existing JS tests and its separate API/worker probes. Those results are reused at the same source revision, not counted as rerun here. |
| Diagram verification | [All 33 render](evidence/2026-11-01/vowl-ingest-diagram-gate.log), with strict revision citations and no fallback. Visual inspection corrected clipped class/member labels, crowded pin transitions and the release sequence. Initial anchor and rendering failures remain in the evidence directory. |

## Ingest and graph findings

All source paths in this section belong to vowl-wasm at the revision above. Each row identifies a distinct contract problem; the probes are synthetic minimal inputs, not a survey of production ontologies.

| Finding | Reproduction and source | User consequence and required acceptance |
|---|---|---|
| VW-I1 / high: split attributes are not joined | `src/ontology/parser.rs:53-114` and property parsing expect inline fields. A split `class`/`classAttribute`, `property`/`propertyAttribute` fixture fails for missing inline domain. With no properties, parsing succeeds while labels and IRIs fall back to IDs. `validate_iris=true` still accepts whitespace identifiers. | Define the accepted JSON dialect and reject unsupported shapes clearly, or implement the upstream joins. Test imported fixtures independently of the exporter. The option named IRI validation must perform validation or be removed from the contract. |
| VW-I2 / high: graph conversion loses semantics | `src/graph/builder.rs:24-31` omits parsed equivalents and individual counts. `:53-54` retains min/max but discards exact cardinality. `src/ontology/parser.rs:389-413` ignores bounds without a `cardinality` key and narrows u64 to u32; 4,294,967,296 becomes zero. | A successful load can display weaker or incorrect meaning. Preserve supported fields with explicit bounds and reject unsupported restrictions. Round-trip tests must compare semantic values, not only node counts. |
| VW-I3 / high: subclass mapping misses standard spelling | `src/graph/builder.rs:121-128` tests a lowercase substring against `rdfs:subClassOf`. The edge becomes `Special`; hierarchy depth-zero filtering leaves all nodes visible in the fixture. | Normal hierarchy navigation can silently fail. Match explicit supported vocabulary terms and test hierarchy behaviour after import. |
| VW-I4 / medium: validation and construction disagree | `src/ontology/parser.rs:485-507` permits an absent datatype range node; `src/graph/mod.rs:271-285` requires it. An `xsd:string` range validates then fails construction. A one-class limit truncates classes while leaving properties that reference omitted nodes. | A validation success does not establish constructibility. Define datatype representation and make limits reject or consistently project the graph, with an explicit diagnostic. |
| VW-I5 / medium: inverse expansion and statistics disagree | `src/graph/builder.rs:78-89` adds a reverse edge even when an explicit inverse property already exists. Two properties produce four edges with duplicate IDs. `src/graph/statistics.rs:371-399` follows outgoing neighbours: one edge `b → a` gives two components/two isolated nodes for order `[a,b]`, and one component for `[b,a]`. Density can exceed one. | Duplicate relations and misleading summaries undermine filtering and user trust. Define inverse identity, multigraph density and weak/strong connectivity, then require permutation-invariant statistics. |
| VW-I6 / medium: diagnostic SVG is incomplete and unescaped | `src/render/mod.rs:116-173` interpolates raw labels and edge comments, includes hidden nodes and unnormalised coordinates, and emits no edge lines. The probe uses harmless XML markers and a comment terminator, not executable script. | Do not present this Rust diagnostic output as a complete safe export. Escape text/attributes/comments, apply visibility and fitting, and test the generated document. No browser execution or production exploit is established here. |
| VW-I7 / low: zero log interval panics | `src/debug.rs:58-59` performs unchecked modulo. `DebugFlags { log_interval: 0, … }.should_log(0)` panics. | Reject zero or define it as disabled. This is a public Rust configuration path; DebugFlags is not a JS export. |

Crate and performance qualifications also needed correction. `parallel` enables a rayon dependency but no source call implements parallel loading. Crate-level `deny(unsafe_code)` is locally overridden for SIMD. CSR ticks allocate force/tree storage; the direct positions view removes a copy only at its boundary. Native Criterion benchmarks do not measure JavaScript marshalling. The phase-three tick benchmark includes setup, its serialisation case measures `graph.nodes()`, and its repeated WASM operation can settle into no-op ticks. Header targets and historical 0.1.1 compressed-size figures do not establish current host latency or compression.

## Release delivery and provenance

**VW-R1 / high: split delivery with no retry route.** The [0.2.2 release job](https://github.com/DreamLab-AI/vowl-wasm/actions/runs/41063531138/job/124088974822) passed inventory, build, contract, packaging and GitHub attachment steps, then failed npm publication with E404. [Retained run metadata and error excerpt](evidence/2026-11-01/vowl-release-run-observation.json) distinguish that failure from the successful CI run at the same commit. Public npm reads returned [E404 for 0.2.2](evidence/2026-11-01/vowl-npm-observation.json) and [only 0.1.2 in versions/latest](evidence/2026-11-01/vowl-npm-versions.json), observed on 1 November.

The [GitHub release](https://github.com/DreamLab-AI/vowl-wasm/releases/tag/v0.2.2) has a 161,091-byte package asset with API-reported SHA-256 `6da30d65bb39e69a4d96805fedf43d6d7b8db804bb3f5ad27332a509c7e0f663`, plus notices/checksums. The [observation](evidence/2026-11-01/vowl-release-observation.json) records metadata, not a fresh download/digest verification. Its `immutable` flag is false: preservation here is a workflow rule. visionGraph's package and lock files pin this GitHub URL; an ordinary registry install does not obtain the reviewed 0.2.2 implementation.

Release execution uploads before publishing (`.github/workflows/release.yml:145-173`). A retry encounters the existing-asset refusal at lines 86-90 before reaching npm. The local control-flow probe reproduces that ordering. A separate scenario shows that an existing registry version is skipped without checking its integrity against the newly attached package. Add a reviewed resume route using the already attached and verified bytes, check destination parity, and report delivery state independently for each destination. This audit does not attempt publication or alter trust configuration.

OIDC permission in YAML does not establish a configured publisher. [npm's trusted-publisher documentation](https://docs.npmjs.com/trusted-publishers/) requires matching registry-side configuration and a supported client/runtime. The observed E404 does not identify which external permission or trust setting failed. [npm provenance documentation](https://docs.npmjs.com/generating-provenance-statements/) describes a verifiable source/build link; that link is not semantic correctness or successful delivery to every channel.

**VW-R2 / medium: inventory assurance has a narrower boundary.** `tools/release-inventory/src/main.rs` checks the source-commit field's hex shape without resolving it, and excludes the release workflow and its own implementation from collected source hashes. Audit fixtures pass with an all-zero commit or changes to those excluded files. Its package operation accepts a wrong package name and invalid WASM payload when the checked metadata matches. The separate release bundle tests still apply; this is not evidence that the full pipeline accepts invalid WASM. Name the guard's scope explicitly, bind the control inputs where required, and test the packed candidate separately from the inventory.

**VW-R3 / low: notice URL mixes versions.** `docs/UPSTREAMS.md:3` links a 0.2.2 companion filename under the v0.2.1 release. The [observed v0.2.1 assets](evidence/2026-11-01/vowl-old-release-links.json) contain 0.2.1 filenames. Correct that route to the actual companion. Hash agreement establishes correspondence with reviewed documents, not a new assessment of licence completeness.

## ADR-001 follow-through

Our preceding report supplies the complete section-by-section reading. This pass adds evidence to those dispositions without changing the owner's accepted/implemented status for NGG1 ownership.

| Section | Additional disposition |
|---|---|
| Context and proposed decision | Separate release ownership exists, but the registry route does not deliver the current reviewed package. Preserve package identity while documenting the actual consumer route. |
| Graph contracts | NGG1 codec ownership remains evidenced. It does not establish semantic preservation in the independent JSON ingest path; VW-I1–I5 need their own supported-format contract and fixtures. |
| Host adapters and execution | Native Rust probes do not establish installed Obsidian behaviour. Optional features and diagnostic SVG must not be described as complete host implementations. |
| Alternatives and tradeoffs | Independent packages need an executable compatibility matrix and reliable candidate distribution. A single source owner does not eliminate separate-channel release failures. |
| Acceptance 1 and 2 | Retain prior native/JS NGG1 parity evidence, while adding imported JSON fixtures and tests against the actual candidate package. Neither synthetic ingest tests nor successful CI substitute for consumer acceptance. |
| Acceptance 3, 4 and 5 | Offline vault operation, installed-plugin lifecycle and reasoning parity remain unqualified by this pass. |
| Acceptance 6 | Inventory and notices provide useful bounded evidence. Delivery divergence, excluded control inputs and the wrong notice URL remain open. |
| Rollout and current effect | The explicit limitation to codec ownership still applies. Performance/mobile claims require representative measurements; the report does not activate the wider plugin decision. |

## Panel dispositions and user utility

| Panels | Assessment |
|---|---|
| VW-01.1, .4, .5, .6, .7, .8 | Module/feature ownership, start/version and error boundary checked; unsafe, parallel, SVG, allocation and debug claims qualified. |
| VW-02.1, .2, .3, .4, .8 | Parse/build ordering and graph/pin ownership checked; semantic loss, validation mismatch, inverse expansion and statistics defects added. |
| VW-05.1–.6 | CI matrix, release order, benchmark meaning, historical size evidence and inventory boundary checked; delivery and retry claims corrected. |
| VW-04.6 | Consumer route corrected to the GitHub tarball; earlier worker/API findings retained. |

[WebVOWL's upstream parser](https://raw.githubusercontent.com/VisualDataWeb/WebVOWL/master/src/webvowl/js/parser.js) joins class/property attribute tables by ID before constructing elements. That implementation supplies a concrete compatibility baseline for VW-I1; a shared visual vocabulary does not establish input compatibility. The [OWL 2 Primer](https://www.w3.org/TR/owl2-primer/#Property_Cardinality_Restrictions) distinguishes minimum, maximum and exact restrictions. Losing those distinctions limits the component's usefulness for inspecting ontology meaning.

A useful near-term proposition is a small offline viewer with explicit import support, stable identities and predictable navigation. Add a fixture-led import workflow that states unsupported constructs, preserves supported semantics and reports validation failures before graph replacement. Then demonstrate the same candidate package in an ordinary vault and the website, including malformed replacement, rename, cancellation and unload. Measure load time, copies and retained memory on representative vaults. This is a development recommendation from the observed gaps, not a claim of market uniqueness or completed user validation.

The [reproduction instructions](evidence/2026-11-01/vowl-ingest-reproduction.md) and [retention receipt](evidence/2026-11-01/vowl-ingest-retention.json) accompany the evidence. Whole-corpus regeneration and the [final check](evidence/2026-11-01/vowl-ingest-final-corpus-check.log) pass: 176 topics, 1,584 diagrams, 24,503 revision-verified citations, zero fallback/unresolved citations and nine allowlisted artefact citations. Structural checks remain separate from semantic coverage: other repositories and topics are still being reviewed.
