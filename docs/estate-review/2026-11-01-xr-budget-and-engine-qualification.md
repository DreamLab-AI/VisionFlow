---
title: XR visual budgets and engine qualification
status: in-progress
date: 2026-11-01
type: explanation
---

# XR visual budgets and engine qualification

VisionClaw's visual-extras allocator implements its bounded triangle allowance, but its whole-frame draw-call guarantee depends on demand that the live scene does not fully measure. The timing governor can retain tier 4 through sparse, overloaded GPU readings. A constructed nine-volume scene also exceeds the cost table's eight-frame clamp. These are distinct accounting and control limits; this pass does not establish a headset performance failure.

This review reads **ADR-2167 and ADR-2165 in full**, including amendments, review follow-ups and verification, against the relevant sections of `docs/XR-client.md`. VisionClaw is pinned at `75b72416bac176ca8fde715cac3bd7c77657b3e7`. All 44 files in the [source receipt](evidence/2026-11-01/xr-budget-source-receipt.json) match that revision. Neither ADR's declared governed paths differ from its recorded pin. That establishes source continuity, not freshness of its historical hardware evidence. Product code, engine installations, live sessions and owner decisions remain unchanged.

## Executed evidence

**151 existing Rust tests pass**, including all 30 `frame_budget` tests, using six complete, unchanged modules and their original source-parity fixtures. One added Rust audit test records call-pressure and portal-floor observations. The isolated crate excludes Godot adapters behind the modules' existing `cfg(not(test))`; it is not a build of the full native extension.

On the authorised HP host, **18 unchanged controller test bodies pass with 109 assertions on each of Godot 4.6.1 and 4.8-beta2**. Seven native-extension/full-scene tests are explicitly excluded from the isolated copy. Both runs pass the production `gut_guard.sh` with one expected script and no parse errors. Five additional observations reproduce identically on both engines. The projects use private temporary directories and data homes, the headless dummy renderer, no native extension and no backend connection. No shader rendering, stereo, real GPU timing or headset session ran.

[Runner](evidence/2026-11-01/xr-budget-run.sh), [fixture manifest](evidence/2026-11-01/xr-budget-fixture-manifest.json), [retained tests](evidence/2026-11-01/xr-budget-isolated-tests.gd), [Rust audit](evidence/2026-11-01/xr-budget-audit.rs), [Godot probe](evidence/2026-11-01/xr-budget-probe.gd), [4.6.1 results](evidence/2026-11-01/xr-budget-probe461.json), [4.8 results](evidence/2026-11-01/xr-budget-probe48.json) and [Rust results](evidence/2026-11-01/xr-budget-rust-audit-results.json) retain reproduction details. Individual commands ran; the consolidated runner was syntax-checked. Offline Cargo needs cached dependencies. Repeat against the recorded source hashes; the engine argument must point at a binary on the test machine.

Initial Rust failures were missing sibling source files used by parity tests; preserving the repository layout and clearing only the isolated crate's compilation cache resolved them. Initial GUT attempts lacked transitive script/shader dependencies. One reported passing tests despite compile errors; the guard rejected it. Final counts above come only from the corrected, guarded runs. The retained initial logs distinguish setup failures from product findings.

## Findings and acceptance criteria

| Finding | Evidence and implication | Required follow-through |
|---|---|---|
| XB1: the live draw-call input is incomplete | `graph_scene.gd:3048-3082` counts graph roles, volume overhead, agents, links and hulls. It supplies no measured remainder for other scene calls. By contrast, `benchmark.gd:230,572` calibrates and adds `_other_dc`. Calling the exact scene method in a minimal class before/after adding 20 mesh nodes leaves demand at zero. The dummy renderer provides no actual call total. | Reconcile measured scene calls with every priced layer, avoiding duplicate charges. Exercise HUD redraw, floor, controllers, avatars, focused handles, portals and extras. Compare allocator demand and actual render counters in the real scene; a passing allocator alone cannot prove the complete frame stays below 48 calls. |
| XB2: intermittent GPU samples can leave quality unmanaged | In `_feed_gpu`, missing readings neither advance the governor's time window nor trigger fallback unless consecutive. Twenty 12 ms samples interspersed with 179 zeros each over 40 seconds leave mode `gpu`, tier 4 and only 0.222 seconds accumulated. A further 181 consecutive zeros correctly switches to frame mode, tier 2. | Bound measurement freshness and valid-sample coverage using elapsed time. Test sparse, stale, zero and recovering streams independently of rendering. Require a measured overload to cause a bounded response even when the timer is unreliable. |
| XB3: frame cost clamps without a matching draw limit | `extras_cost` and the fallback table price at most eight frames; the style switch reaches every grouped frame. Nine actual constructed frames contain 216 triangles while `applied_cost()` reports 192. Current ordinary layout uses fewer frames; this is a scaling boundary, not proof of current production undercount. | Either cap/admit visible frames consistently or price all drawn frames with checked arithmetic. Test eight, nine, hidden and newly created frames. The measured `other_tris` path may absorb omitted geometry later; that does not make the immediate allowance/report exact. |
| XB4: draw-call overrun does not withdraw the allowance | With 49 supplied calls and a modest graph, Rust reports `over_budget: true` but grants tier 2 and a 72-triangle allowance. `pressure` includes triangle overrun but not call overrun; the guard permits zero-added-call tiers when the base already exceeds its limit. | Clarify D1's “minimums overrun” promise: either include call pressure or explicitly permit triangle-only extras during a pre-existing call overrun. This case adds no calls and is not evidence that extras caused the initial overrun. |
| XB5: historical wording needs qualification | `XR-client.md:1249` calls portal weighting an open owner decision, whereas ADR-2167 R11 records the choice to leave it unchanged. The prior measured benchmark table predates the allowance; the document says so, and it must remain visible. | Update the decision wording without reversing the owner's choice. Keep computed allocator rows separate from render measurements and rerun the latter before a current performance claim. |

A manual `XR_EYE_CANDY=4` request also remains 4 after automatic fallback caps the governor at 2. This is the documented manual pin, not an unexpected bypass of a security rule. Product messaging should distinguish manual quality from automatic adaptation. Low-cost comfort still overrides the pin and immediately withdraws demand; retained tests cover that path.

Native splitting gives `[12,12,11]` for 35 gems, three equal demands and weights `[1,0.15,0.15]`. It matches the unweighted result because three floors of 16 exceed the available cap. This confirms the accepted portal-weight limitation rather than reopening it as an unapproved design change. Query legibility and edge recognition at the accepted production tiers still need user evaluation.

## ADR-2167 section dispositions

| Section | Assessment |
|---|---|
| Status, evidence class and boundary | Proposed/partial/staged, source+test and mono/headless boundary remain candid. Current source matches all seven verified paths. Renderer and both-eye evidence are not refreshed by this pass. |
| Context and reserve | 95,000 triangles and 48 calls are the allocator limits. Triangle reserve is 5%; whole-call rounding reserves two of 50 calls. Fixed triangle/call budgets do not establish spare pixel time. |
| D1: controller, priority and allowance | Source serves hard triangle minimums, route, cloud, bursts and hulls before the requested extras allowance, then near geometry. Existing pressure, allowance-bound and production-shape tests pass. XB1 and XB4 qualify the call-budget and “minimums overrun” claims. Tier 0 delegates to the same allocator. |
| D2: tier costs and reporting | Costs and built meshes agree for the supported examples; ribbon frames reuse their existing calls and motes add one. The eight-frame clamp needs an explicit scene limit or expanded pricing (XB3). Controller feedback prevents charging its reported geometry twice. |
| D3: measured headroom and hysteresis | Startup majority probe, one-second sampled windows, thresholds, backoff and all-zero fallback exist and have passing tests. “At once” means after a completed sampled window and application cadence, not an instantaneous per-frame response. Sparse samples expose XB2. |
| D4: comfort and copied Environment | Retained tests verify low-cost off, still motes under reduced motion and copy-on-write. Headless resource tests cannot establish visual comfort in both eyes. Manual quality remains subordinate to low-cost mode. |
| D5: portal weights | Equal weights preserve the unweighted path; floors take precedence. Native tests and the added three-volume observation support the documented limits. Current splitter weights unmet demand after floors; “drawn × weight” is a shorthand, not the entire algorithm. |
| Consequences: triangle/cylinder trade-off | The recorded 72/200/584 extras costs for three frames agree with the table. Tier 3 can eliminate near cylinders while retaining gems; the owner accepted that trade-off. Historical measured rows are not measurements of the amended allocator. |
| Consequences: per-frame calls and stereo | Volume-frame calls are now included. That fixes their prior omission but does not cover all outside-layer calls (XB1). Sky pixels, both-eye fill rate and XR timer availability remain unverified. |
| Verification: timer, Rust, GUT and benchmark | Thirty current allocator tests rerun successfully. New GUT evidence is the narrower 18-test, no-gdext scope, not a replacement for historical 500-test receipts. Historical HP timing and benchmark reports were not independently replayed. |
| R11: governor, cost table, timer and frame-call fixes | Source contains those changes. Consecutive-zero fallback works; intermittent timing remains a gap. Cost-table parity between two clamped implementations alone cannot prove nine-frame drawing parity. |
| R11 F5: accepted portal-floor limitation | Preserve the owner's choice. The governing document's “open” wording is stale; no operator re-approval is needed merely to record the existing decision. |
| Bounded-allowance amendment | Source and fixtures implement the revised priority; earlier “caps exactly unchanged” descriptions are historical. Benchmark calibration and production demand are separate paths, so benchmark parity does not automatically qualify the live scene. |
| R12 re-verification | No governed-path changes since the stamp; all 30 tests pass here. The historical mutation claim was not rerun and is not newly certified. Sparse GPU signals, full-scene calls and extra-frame admission remain additional acceptance cases. |

## ADR-2165 section dispositions

| Section | Assessment |
|---|---|
| Status and context | Accepted decision with partial implementation/staged activation correctly separates engine selection from stereo acceptance. The API 4.3 extension and 4.3 project metadata are still present. Historical release/latest-crate statements are dated context, not a claim about today's latest versions. |
| D1: target and future bumps | Installed binaries report `4.6.1.stable.official.14d19694e` and `4.8.beta2.official.cafc17a19`. The limited controller suite and probes run on both. This is not a release-archive checksum verification or full client startup. Each future target still needs its own gates. |
| D2: hard cutover and rollback | The record supplies separate installation paths and a rollback command. No live process was inspected, stopped or launched; this pass neither certifies today's running engine nor replays the historical cutover. Both-eye operator evidence remains absent from this review. |
| D3: unchanged gdext | Cargo still requests `godot = "0.2"`; the extension minimum remains 4.3 and both governed paths match the stamp. No native extension was loaded here. Forward compatibility expectations support testing an older API, not waiving load or stereo qualification. |
| D4: separate engine changes and project metadata | The existing feature metadata remains unchanged. Updating its renderer tag would be a separate decision; an audit need not force a metadata edit to make a beta-target statement look consistent. |
| D5: desktop versus Android plugin | No plugin/export installation ran. Desktop OpenXR and the frozen Quest/Android path retain separate qualification obligations; the latter must verify its plugin version and exported device build when resumed. |
| Consequences: spatial API and fallback | API availability in an engine and extension availability in a runtime are different gates. The prior spatial-volume review covers current emulation; headless controller tests do not prove native spatial-container availability or either runtime's extension list. |
| Consequences: beta risk, GUT, Quest and cost | Preserve per-engine qualification and rollback. The same vendored GUT works for the isolated scope on both binaries. No claim is made about a full current client suite or Android export. |
| Verification: installation, load, suites and desktop timings | The ADR's archived checksum, full load, pixel comparison and timing rows remain historical evidence. This pass adds bounded version/test observations only. A left-eye desktop mirror or stable FPS is insufficient for right-eye rendering. |
| Verification: API break and headset | No new complete `XRInterfaceExtension` search or XR transition run occurred. The recorded one-frame view-count warning and unobserved right eye remain explicit qualification work, not an accepted harmless regression. Record both-eye evidence and the transition outcome before changing staged to live. |

## Comparison and utility

Godot's GPU timer measures viewport rendering even under a frame cap; its documentation also distinguishes global render counters and notes that headless mode returns dummy values. This supports the chosen signal while requiring explicit viewport scope, sample freshness and real rendering for acceptance. A steady headset frame rate alone cannot answer the headroom question. See [RenderingServer documentation](https://docs.godotengine.org/en/stable/classes/class_renderingserver.html#class-renderingserver-method-viewport-get-measured-render-time-gpu).

Godot's extension guidance expects older API targets to work on newer minor versions, while advising builds/tests against the earliest supported target. That supports keeping a rollback binary and an unchanged extension until a needed API demands migration; it does not prove this client renders stereo correctly. See [GDExtension guidance](https://docs.godotengine.org/en/stable/tutorials/scripting/cpp/gdextension_cpp_example.html).

Android's Dynamic Performance Framework is a useful comparator for the future standalone headset path: it connects application quality controls to thermal/performance signals and encourages profiling and gradual adaptation. It is not a drop-in replacement for the current Linux/OpenXR governor. The transferable product requirement is sustainable, intelligible quality: show which signal is trusted, its age, requested versus granted tier, and why detail changed. Evaluate long-session task accuracy and comfort alongside frame timing before spending additional pixels on decoration. See [ADPF best practices](https://developer.android.com/games/optimize/adpf/best-practices-adpf).

[VC-51 diagrams](../diagrams/visionclaw/51-xr-visual-budget-and-engine-gates.md) separate allocation, measurement reliability and engine qualification. These two records now have section dispositions; the wider estate and older XR corpus still have outstanding review obligations.

## Scoped refresh — 2026-11-01, incoming drag work

The [drag-contract review](2026-11-01-drag-holds-and-acceptance.md) checks the
new ADR-2167 annex at VisionClaw `531df5ef74453a2ed320bff968c203a9b91faed7`.
Across all seven governed paths since this report's `75b72416b` baseline,
only nine lines of `spatial_environment.gd` change: first-held-hand selection
and a volume-specific focus transform. Allocator, governor, cost-table, pass,
benchmark and probe source are byte-identical; the section dispositions and
XB1–XB5 findings stand. The new transform helper was traced, with a conditional
coordinate-supplier gap recorded for differing connection buffers. The existing
focus test checks transforms, not the whole marker's coordinate provenance.
No new renderer, stereo, timing or full-scene acceptance is claimed; historical
unit/headless receipts remain scoped to their original revisions.
