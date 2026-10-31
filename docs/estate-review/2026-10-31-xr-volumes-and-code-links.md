---
title: XR spatial volumes and code-to-concept evidence
status: in-progress
date: 2026-10-31
type: explanation
---

# XR spatial volumes and code-to-concept evidence

Codescape annotation production, runtime concept joining and spatial-volume presentation are implemented. This pass reproduces three shortcomings: comment-shaped text inside a Rust string becomes a real annotation; highlighting exceeds the documented link-instance budget; and a growing content extent can remain outside its fitted box for four ticks. It also verifies two information gaps: link records omit repository/revision provenance, and producer truncation is invisible to the join's unresolved count.

This continues the [presence and concurrent-connection review](2026-10-31-codescape-presence-and-xr.md), assessing ADR-2158 and ADR-2164 section by section. [Source receipts](evidence/2026-10-31/xr-links-sources.json) bind VisionClaw to `75b72416bac176ca8fde715cac3bd7c77657b3e7`, standalone Codescape to `58108c0357bd0dbf6588f8b6adf6e044f4c04556` and consumed Codescape to `e3b76f63e801c6a8460110b93e88a0b420e9b0b4`. Inspected files match those commits; the inspected Codescape files are identical between checkouts. No product code, owner ADR status or deployed service was changed.

## Annotation and provenance boundaries

The [Rust probe](evidence/2026-10-31/xr-links-probe.rs) uses the production Codescape libraries and a byte-for-byte copy of the pure production `link_index.rs` module. It exercises scanning, side-table updates, topology emission and joining. It does not instantiate the gdext wrapper or contact either server. The [runner](evidence/2026-10-31/xr-links-run.cjs) selects cached dependency artefacts through a fresh offline build, compiles the probe and module tests, and removes temporary executables.

| Case | Observed result | Assessment |
|---|---|---|
| A valid Rust constant contains a raw string with a line beginning `// @ontology urn:ngm:class:bm25` | The fixture compiles as Rust. Scanning binds line 2 to the fixture file; topology emission and the real join produce two links to two eligible target ids. | The line is string content, not a source comment. `comment_body` and `scan_text` recognise line prefixes without language lexical state (`annotations.rs:125,159`). ADR-2164's explicit-comment distinction is weaker than stated. |
| A snapshot names repository `audit-repo` and revision 23 | The resulting link has `repo: ""` and `revision: null`, while retaining source/target generations. | `initial_graph_load` omits repository/revision; `join` deliberately fills empty provenance (`link_index.rs:281-285`). Shape compliance does not supply traceable repository provenance. |
| One file has 33 valid, bound references, all with available targets | Only 32 enter topology metadata; the join emits 32 records and reports zero unresolved. | The 32-ref cap is documented and enforced. The information loss is not represented in the consumer's unresolved count. Zero unresolved cannot mean every authored annotation was represented. |
| Real comment, bare IRI, trailing inline marker and malformed IRI | The real comment links; the other three are refused. | Existing positive/negative syntax controls remain effective within the line-based grammar. |
| File and symbol both carry one annotation | Symbol wins; one record is produced for each eligible target id. | Roll-up de-duplication and multi-target resolution work. |
| Annotation is removed from the source file | Refresh removes its bound and emitted refs. | The side table does not retain that removed annotation. |
| Received refs have no targets | No records; all 32 received refs count as unresolved. | Missing-target accounting works for the data actually received. |

[Eight probe cases passed their assertions](evidence/2026-10-31/xr-links-probe.json), including the reproduced counterexample and contract gaps. Nine existing annotation tests and one topology-emission test also passed. The production pure-join module passed eight unit tests; its captured-live-data test remains ignored because no fresh server captures were supplied. These results do not establish live seed coverage or full Graphicle-to-headset acceptance.

Comments in the scanner's own tests explicitly avoid multiline annotation examples so that Codescape does not ingest them as real annotations (`annotations.rs:395-396`). That workaround does not enforce the boundary for arbitrary estate source. Acceptance should include examples inside Rust raw strings, Python multiline strings and documentation fences, using language-appropriate lexical classification. The current probe establishes the Rust case only.

For provenance, a stream-generation pair is useful for invalidating ephemeral ids but cannot substitute for a repository identity and revision. Before presenting a link as reviewable evidence, the record should identify the source revision and make omitted or truncated annotations visible. This is a proposed acceptance criterion; the current audit does not add a transport field or change the wire contract.

## Headless geometry and lifecycle evidence

Godot is absent from the agent container, but the authorised HP host supplies **Godot 4.6.1 stable**, engine hash `14d19694e0c88a3f9e82d899a0400f27a24c176e`. The audit copied unchanged production scripts into a temporary, isolated project and used a separate `XDG_DATA_HOME`. It neither opened the live XR project nor changed the operator's preferences. [Execution receipt](evidence/2026-10-31/xr-headless-receipt.json) and [copied-source hashes](evidence/2026-10-31/xr-headless-source-hashes.txt) match the inspected source. The [GDScript probe](evidence/2026-10-31/xr-layer-probe.gd) uses fake connections/content and the actual link-layer, spatial-volume and preference scripts.

| Case | Observed headless result | Limit or implication |
|---|---|---|
| 512 ordinary links | 512 instances, 6,144 triangles | Matches ADR-2164's ordinary-link budget. |
| Highlight their common source node | 513 endpoint markers, 1,025 total instances, 12,300 triangles, one draw call, zero link overflow | Markers are appended after the 512-record cap (`link_layer.gd:317,430-448`). This contradicts the stated total instance/triangle ceiling. The layer's triangle accounting includes markers, so the cost is counted. No measured frame-rate failure is claimed. |
| Disable links, then disconnect their Codescape source | Geometry remains empty and hidden | An initially suspected off-state reconstruction bug was disproved. `_refresh` explicitly handles disabled state. |
| Change a connection generation | Old records are purged | Supports the layer's invalidation control; scene callback wiring is additionally inspected. |
| Anisotropic or zero-size content extent | Minimum-axis fit and zero-extent fallback pass | The pure scale calculation is sound for these controls. |
| Expand a fitted source from extent 2 to extent 8 immediately after its first tick | The new corner lies outside the volume for four subsequent ticks; the fifth checked tick refreshes and contains it | `FIT_RECOMPUTE = 6`; between refreshes, containment checks use the cached AABB (`spatial_volume.gd:168-184`). This is a real transform-level counterexample with synthetic content, not a headset screenshot or a measured frequency in live streams. |
| Explicitly resize bounds | Immediate source reread and valid fit | Resize handling works; the delayed case concerns a changed content extent between cadence updates. |
| Separate debug and release preference values | Defaults and stored values remain independent | Confirms build-type separation using an isolated preference file. |

[All eight headless cases passed their final assertions](evidence/2026-10-31/xr-layer-probe.json). The [first probe attempt](evidence/2026-10-31/xr-layer-first-attempt.txt) is retained: it failed an audit hypothesis about off-state geometry, which was corrected to a passing control after inspecting `_refresh`. That was a harness expectation error, not a product regression.

Five existing GUT scripts also passed **76 tests and 1,345 assertions**, covering link lifecycle/highlighting, spatial volumes, arrangement and the native shim against a fake extension. [Raw output](evidence/2026-10-31/xr-gut-suite.txt), [plain output](evidence/2026-10-31/xr-gut-suite-plain.txt), [script/parse guard](evidence/2026-10-31/xr-gut-guard.txt). Plain logs remove ANSI escapes and trailing whitespace; the original logs are retained. The guard confirms 5/5 scripts ran with no parse/compile errors. Godot emitted texture/font/ObjectDB/resource warnings at shutdown; the receipt is therefore a passing assertion suite with teardown warnings, not a clean engine-lifecycle result. Their cause is not attributed to product code by this pass.

An [earlier GUT fixture run](evidence/2026-10-31/xr-gut-incomplete-fixture.txt) omitted `materials/frame_ribbon.gdshader` and produced parse-dependent failures. Adding the unchanged shader to the isolated package resolved those failures. Both attempts are retained to distinguish packaging errors from source findings.

## ADR-2164, section by section

| Section | Disposition |
|---|---|
| Metadata, Context and initial measurements | Accepted/none/inactive and “nothing built” are stale implementation descriptions. Historical corpus counts, IRI ranks, absence of annotations and live-server observations were not remeasured. Current source contains the producer, join, scene wiring and seed comments. |
| D1: explicit comment annotation | Shape and marker tests pass, but a comment-shaped raw-string line becomes a link. The claimed distinction from examples is not enforced by lexical context. |
| D2: binding, roll-up and storage | `bind` prefers the documented symbol, then containment, then file; `rebind` rebuilds refs from side-table rows. Existing tests cover binding against Graphicle spans and roll-up. The new probe verifies update/removal. A complete live watcher/delta journey was not run. |
| D3: delivery and read-only route | Topology emission adds refs to projected annotated nodes and caps each node at 32. The source route iterates bound annotations; the new probe calls the table/emitter, not the HTTP route. Scene switch state is not an input to server emission. File/symbol duplicate suppression passes. |
| D4: client resolution and isolation | Pure join tests verify eligible classes, missing targets and multiple ids per IRI. `binary_protocol.rs:1298-1299` replaces per-client tables at topology load. The actual gdext wrapper and two live streams were not exercised here. No server-to-server resolver was introduced. |
| D5: link record | Endpoint keys, ids and generation pairs exist. Repository/revision values are absent. Source provenance is therefore incomplete despite the record serialising in the required shape. |
| D6: offline/reconnect | Existing headless tests pass generation and disconnect cases; scene callbacks explicitly drop affected records (`graph_scene.gd:3922-3942`). The join signature uses generations and population counts. Same-count input changes deserve a separate integration test; this pass does not call that a reproduced stale-join defect. |
| D7: one switch | `XrPrefs` separates debug/release keys. The direct headless probe verifies defaults and independent stored values; layer disable clears geometry immediately and remains empty on disconnect. |
| D8: budget | One draw call is preserved. Highlight markers breach the stated 512-instance/6,144-triangle limit while being included in `triangle_count`. The ADR and tests need an explicit marker budget or a revised total contract. |
| D9: seed | Current source has explicit annotation comments. This pass does not establish that eight chosen seeds resolve under the current viewer's default cap; historic target ranks cannot prove that. |
| Rejected alternatives | The implementation follows authored annotations and client-side resolution. Sourcegraph-style symbol navigation is a distinct comparison below. Historical claims about absent corpus links are retained as dated evidence, not current universal facts. |
| Consequences and follow-on | Authoring remains necessary; omitted targets cannot appear as links. “One reviewable claim” needs lexical precision and repository provenance. The protocol key and submodule consumption exist, as assessed in the preceding Codescape review. |
| L1–L3 | Existing tests pass their covered cases, including binding and pure joins. The new raw-string counterexample exposes missing L1 scope; producer capping also limits what L3 can count. |
| L4–L5 | Existing fake-connection tests and the direct probe support invalidation, switch and preference behaviour. L5's total-instance promise fails under the reproduced highlight case. |
| L6 | Not freshly run: no live seed route/stream captures or default-cap resolution receipt. |
| Verification | The old “none/nothing built” text cannot describe the current revision. Fresh receipts here support narrower producer, pure-join and headless-layer claims, not activation or live L6. |

## ADR-2158, section by section

| Section | Disposition |
|---|---|
| Metadata and Context | The single-origin/single-backend preconditions are historical; current scene construction has multiple volumes and connections. The implementation note already acknowledges shipped UX while leaving metadata at none/inactive. External runtime-support claims are dated and not revalidated here; the measured engine for this audit is 4.6.1. |
| D1: volume contract | Bounds/mode/visibility/interactivity/close state and signals are implemented in `SpatialVolume`; existing GUT tests exercise grants, denial, fitting and state changes. This does not establish headset ergonomics. |
| D2: emulated and native arms | Emulated frame/handles and fit exist. The shim gates settings, Android, engine version, class/instance availability and extension state, then applies a timeout/fallback. Headless fake-extension tests pass. No real Android XR/native-container run occurred. |
| D3: space versus volume and frames | Local arrangement/content transforms are explicit. Cross-node placement and federation semantics are outside these fixtures and remain a separate ADR-2161 review. |
| D4: E3 arrangement | The current scene binds knowledge/ontology to VisionClaw and code to Codescape; memory remains separate furniture. Arrangement tests pass. These do not constitute a live three-volume session. |
| D5: fit and E7–E9 | Anisotropic fit, zero extent and explicit resize controls pass. Dynamic content growth exposes the cached-AABB gap. The previous pass inspected local Codescape drag refusal. E8's 90 Hz headset target remains unmeasured. |
| D6: switch and preferences | Defaults and independent per-build keys pass the direct probe. Disabling the layer clears its instances. Links use data already held by the client. Highlight geometry requires the budget qualification above. |
| D7: XR-only layout amendment | `VolumeArrangement` holds ring slots and user moves; the existing arrangement suite passes. This review does not revalidate desktop/server tri-layout parity. |
| D8: later persistence, colocation and native verification | Still outside the tested E3 path. Fake-extension success is not evidence of a supported device or shared spatial alignment. |
| D9: units, fourth body and yaw | Local graph-to-metre fitting and ring placement are implemented and tested in fixtures. The historical server-yaw assumption and a complete live transformed-stream journey remain unverified here. |
| Alternatives and Consequences | Emulation and a single optional native wrapper match source. Portability/runtime availability statements require dated external/device evidence. The fit's cached extent qualifies the promise of every-frame containment. |
| Activation prerequisites | Source integration exists; this pass does not reconstruct whether the historical preflight/build order was followed. The prior Rust transport build stalled on container writeback, and this headless fixture does not replace that test. E3 activation still needs its actual headset/session criteria. |
| Verification and implementation note | The dated UX note and source are stronger evidence of implementation than the stale “nothing built” sentence. The reported old HP runs are historical; this pass supplies fresh isolated 4.6.1 receipts without promoting the record to live. |

## User utility and comparison

Sourcegraph documents cross-repository definitions/references navigation and distinguishes search heuristics from compile-time precise navigation. Its precise path uses uploaded language-specific SCIP indexes. Those are useful benchmarks for ordinary code exploration; Codescape's annotation join currently represents an authored relationship to a concept graph. [Sourcegraph code navigation](https://sourcegraph.com/docs/code-navigation), [precise navigation](https://sourcegraph.com/docs/code-navigation/precise-code-navigation), checked 31 October 2026.

A reviewer should be able to answer “which code claims to implement this governed concept, and what evidence supports it?” more reliably than by code navigation alone. That advantage is not demonstrated by drawing links. A useful evaluation would compare the same tasks with Sourcegraph navigation, a searchable annotation table and the XR view: locate the implementation claim; open its exact revision; reject a fixture/example false link; distinguish omitted refs from missing targets; and review a changed claim after reconnect. Measure correct answers, time and unsupported conclusions. These are prospective tests, not measured comparative results.

These findings set the immediate utility priorities: preserve repository provenance, separate examples from assertions, expose truncation, and keep highlighting and live fitting within explicit budgets. Improvements to navigation convenience should be evaluated after those evidence distinctions are visible to users.

## Reproduction and remaining scope

Run the Rust probe with `node xr-links-run.cjs <codescape-root> <visionclaw-root> <output.json>`. It builds Codescape offline and compiles the copied pure join module; the eight-unit-test receipt is written beside the JSON. [Annotation tests](evidence/2026-10-31/codescape-annotations-suite.txt), [emission test](evidence/2026-10-31/codescape-annotation-wire-suite.txt), [join tests](evidence/2026-10-31/xr-links-probe-unit.txt).

For the direct engine fixture, put the unchanged `link_layer.gd`, `spatial_volume.gd` and `xr_prefs.gd` beside `xr-layer-probe.gd` in an isolated Godot project, use Compatibility rendering and a private `XDG_DATA_HOME`, then run `godot --headless --path <fixture> --script res://xr-layer-probe.gd`. For GUT, retain the original `scripts/`, `scenes/`, `materials/` and `tests/fixtures/` paths, copy the five named test scripts and the repository's `addons/gut`, then run `--script addons/gut/gut_cmdln.gd -gdir=res://tests/unit -gexit -glog=1`. Include `frame_ribbon.gdshader`; the first attempt proves why. Run `tests/gut_guard.sh` against the plain logs so skipped scripts or parse errors cannot masquerade as coverage.

[VC-45](../diagrams/visionclaw/45-code-to-concept-link-evidence.md) maps annotation/provenance boundaries; [VC-46](../diagrams/visionclaw/46-spatial-volume-fit-and-link-budget.md) maps fit, highlight and fallback behaviour. Remaining work includes fixes and regression acceptance for these findings, live L6, the full gdext/transport path, E7/E8 headset acceptance, actual native-container support, federation/space permissions, and the wider estate ADR and diagram queue. This pass is progress toward that full audit, not a completion claim.

## Documentation checks

The updated index contains 167 topics and 1,554 diagrams. [Strict citation checks](evidence/2026-10-31/xr-links-citations.txt) and [verification checks](evidence/2026-10-31/xr-links-verification.txt) pass. All nine diagrams in touched topics VC-44, VC-45 and VC-46 render; the six new diagrams were also visually inspected. This is a structural/revision check across the corpus and bounded semantic review of the named topics, not completion of the whole diagram audit.
