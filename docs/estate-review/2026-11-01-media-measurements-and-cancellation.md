---
title: Media measurements, cancellation and staged qualification
status: in-progress
date: 2026-11-01
type: explanation
---

# Media measurements, cancellation and staged qualification

Agentbox has working deterministic media tools and a clear editorial acceptance boundary. Its clients still have a cancellation race through the pinned ComfyUI server, and the post-build registration check can pass with a required image-loading node absent. An extreme seam index panics in a debug build; drift reports omit two promised measurement fields. These findings qualify the implementation without establishing a live sidecar failure or a released film defect.

This pass reads **ADR-2163 in full**, including all nine consequences, verification and the SpEdE addendum. It follows the provider ownership and editorial review clauses of ADR-2150; it does not claim a new complete review of that predecessor. Source is agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`. The [source receipt](evidence/2026-11-01/media-source-receipt.json) records byte identity against that revision. No source service, ADR status, model weight or live queue was changed. No H3STUDIOFILM source was read or copied.

## Executed evidence

**305 existing tests passed, with two additional node tests skipped:** 171 Rust tests including 14 doctests, 91 Python node tests, 32 Python client tests and 11 JavaScript MCP tests. The two skips need face-model weights. The node suite uses CPU NumPy/OpenCV and synthetic inputs; it does not establish torch adapters, real-face accuracy or GPU execution. Four added audit tests and three controlled cancellation cases reproduce the observations below.

| Probe | Result and scope |
|---|---|
| Actual Python `Client.cancel` plus pinned upstream handler AST | Own job still running: interrupts that job. Job already finished: no interrupt. Injected hand-off after the server snapshot: interrupts the other job. All three client results say `cancelled: true`. This is controlled interleaving with queue/signal test doubles, not a live concurrent sidecar run. |
| Actual Rust `seam_qa::run`, synthetic 50-frame FFV1 video | Frame 25 with declared colour yields editorial PASS; absent colour declaration refuses the untagged YUV source; frame 49 yields BLOCKED; `u64::MAX` panics on addition in the debug build. |
| Actual identity drift report, synthetic embeddings | `acceptance: editorial` and `gate: false` remain, but `provisional` and `acceptance_backing` are absent. Three stale calibration configurations are correctly refused. |
| Unchanged registration snippet from `verify-spede.sh`, synthetic `/object_info` | Missing `AgentboxLoadImageVerified` passes. Missing `AgentboxTrackMask` fails as a positive control. No full post-build or GPU smoke check ran. |

The [runner](evidence/2026-11-01/media-run.sh), [test summary](evidence/2026-11-01/media-test-summary.json), [cancellation results](evidence/2026-11-01/media-cancel-results.json), [bounds results](evidence/2026-11-01/media-bounds-results.json) and [node results](evidence/2026-11-01/media-node-probe-results.json) retain the commands and observations. Existing suites ran before the audit integration test was copied into the temporary Rust crate. Initial Rust failure came from flattening the shared retake-fixture path; the corrected layout required clearing that crate's cached compilation because `CARGO_MANIFEST_DIR` was embedded. Initial JavaScript failure was a missing `ws` dependency, resolved by the copied lockfile's `npm ci`. Python wheels needed the environment's C++ runtime library. These setup failures are not product findings. Initial Rust and JavaScript logs are retained; the final suite logs record the successful runs.

## Findings and repair criteria

**M1 — Cancel can reach the wrong running job (priority: before shared-queue use).** Python sends an ID-bearing `/interrupt` at `skills/comfyui/scripts/comfy_client.py:325`; JavaScript does the same at `skills/comfyui/mcp-server/server.js:149`. The pinned server checks a snapshot, releases the queue lock, then signals the global interrupt flag. The controlled probe changes the running job in that gap and records the successor being interrupted. Supplying an ID prevents an unconditional interrupt but does not make this route atomic. The same revision provides `/api/jobs/{job_id}/cancel` through `interrupt_if_running`, which checks and signals under the mutex. Adapt both clients to that route, retain history as terminal authority, and test the hand-off and no-op cases. See the [pinned server](https://github.com/Comfy-Org/ComfyUI/blob/faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52/server.py#L1035) and [queue implementation](https://github.com/Comfy-Org/ComfyUI/blob/faa7e494f3a0c9e45e1c3db8d20ee48fd5fe1b52/execution.py#L1377).

The Python result acknowledges a request, not confirmed cancellation; its documentation already assigns terminal authority to history. A receipt's prompt ID is local ownership bookkeeping, not a cryptographic authorisation boundary. No cross-user credentials or live queue access were tested. The Rust `comfy.rs` remains an upload/capability helper and does not take over provider scheduling, consistent with ADR-2150.

**M2 — Seam range validation can panic (priority: input robustness).** `seam_qa.rs:753` adds two before deciding whether the seam lies inside the video. The maximum unsigned index is valid JSON input but overflows in the tested build. Use checked arithmetic or an overflow-free comparison, and cover overlap arithmetic too. Require a structured refusal or BLOCKED result. This probe establishes no release-build behaviour and no false acceptance: it catches the panic instead of receiving a measurement receipt.

**M3 — Post-build readiness has an incomplete node list (priority: before qualification).** `verify-spede.sh:33-34` lists eight video nodes but omits `AgentboxLoadImageVerified`, which staging exports at `staging.py:206`. The exact registration snippet passes without it. Package import deliberately logs and skips a failing submodule (`__init__.py:17-24`), so checking an explicit complete manifest matters. Derive the required set from the intended workflow contract and test every required-node omission. Registration alone still does not execute a graph. The script's optional GPU smoke defaults to skipped, and its OpenCV checks establish GUI-wheel absence and headless-wheel presence rather than counting every OpenCV distribution.

**M4 — Drift metadata does not fulfil the universal receipt promise (priority: schema consistency).** `identity.py:424-425` returns a drift report without `provisional` or `acceptance_backing`. Its explicit editorial label and false gate prevent this observation from establishing an acceptance bypass. Use one measurement stamp and contract tests across all public report shapes; downstream readers should require explicit acceptance authority rather than infer it from missing fields. The tested calibration flags, including an explicitly null calibration field, remain refused.

**M5 — ADR details lag current source (priority: documentation).** The decode consequence says seam analysis uses ffmpeg defaults for untagged media, but `seam_qa.rs:711` resolves declared colour and the probe refuses an undeclared YUV matrix. `windows.rs:42` emits `window-plan/2`; both the review trigger and consequence still name `/1`. Historical test counts should remain dated evidence, accompanied by the newer results rather than silently rewritten. The broad claim that global-interrupt defects are covered is insufficient for the server hand-off case. These corrections are recorded here; the owner record remains proposed/partial/staged.

## ADR-2163 section dispositions

| Section | Assessment and remaining obligation |
|---|---|
| Status, context and predecessor | Proposed/partial/staged and blank acceptance pin remain appropriate. Current tests establish deterministic helpers, not the image or film workflow. ADR-2150's separation between toolkit and provider clients remains visible. |
| D1: clean-room and attribution | This audit preserved the no-H3-source boundary. Kinoloom adaptation notices and `NOTICE` are present. Source inspection cannot retrospectively prove who read which upstream source; preserve provenance review with release evidence. |
| D2: shared decode and root-relative records | `media.rs` centralises probing/decoding, limits frame allocation and restricts input protocols to files. Tests exercise real ffmpeg. The declared-colour refusal is stronger than the ADR consequence; M2 remains a malformed-index gap. |
| D2: seam metrics | Local/absolute thresholds and editorial verdicts are implemented. Synthetic cuts, ramps and audio tests pass. Threshold usefulness on real estate footage remains unmeasured. |
| D2: stitch and retake | Indexed trimming, linear blending, decoded frame/rate/canvas checks, colour tags and approved audio packet comparisons are implemented. `stitch.rs:277-315` removes rejected output through `discard`; passing fixtures do not prove arbitrary codecs or platform reproducibility. |
| D2: window planning and cache | Current `/2` keys bind geometry, model verification state and prior windows. `verify_in` checks available model bytes. Diff-based reuse checks both parent hash and final key, not merely equal length (`render_cache.rs:122-143`). Hardware/model execution remains separate. |
| D2: prompt lint | Rule tests pass; a syntactically consistent prompt plan cannot prove visual continuity, performance or editorial quality. Keep the pre-spend lint and post-render review as separate checks. |
| D3: capabilities and readiness | Both COMBO forms and capability manifests have tests. `/system_stats` before `/object_info` limits misleading registration probes. Neither a node name nor reported VRAM proves the workflow fits. |
| D3: safe free, VRAM and cancel | Idle checks are explicitly best effort; a job can appear and disappear between observations. VRAM is a warning receipt, not admission control. M1 invalidates a strong owned-only cancellation claim on the selected route. |
| D4: staging, relay, masks and identity | CPU suites cover verified staging, arithmetic relay, mask bounds and fake-embedding decisions. Model bytes are hash-checked before use. Real face-model tests are skipped; ComfyUI adapters and chosen-model latent strides remain unqualified. OpenCV wheel replacement needs the full image rebuild/import checks. |
| D5: shared retake geometry | Rust and Python tests pass using the same repository-relative fixture. This proves sampled agreement, not equivalence for every possible geometry. Preserve layout when copying the crate for tests. |
| D6: editorial acceptance | Rust stamps and principal identity scores honour the policy; stale calibration inputs are refused. M4 is a report-shape exception. Film acceptance remains the production review record; this audit does not create one. |
| Consequence 1: measurable gates and known defects | Measurement is executed. Frame/audio fixtures and COMBO cases pass; cancellation needs the additional atomic-route regression. |
| Consequence 2: conservative invalidation | Whole-plan geometry changes invalidate chained keys intentionally. More selective reuse needs a separate soundness argument and fixtures. |
| Consequence 3: no automatic film acceptance | Retain. Tuning thresholds must not create authority to accept a film. |
| Consequence 4: H3/node-pack qualification | Still NOT_RUN live. Arithmetic profile tests do not demonstrate model compatibility. |
| Consequence 5: rebuild and dependent imports | No image build or BFSnodes/scenedetect import run occurred. Keep these as explicit first-rebuild gates using the host build workflow. |
| Consequence 6: decode limits | Tagged/declared supported colour and geometry checks are present. Correct the stale seam-default statement; do not relax the new refusal to fit old prose. |
| Consequence 7: model verification and cache lineage | Verified models are checked on disk; the explicit unverified escape marks the plan. Parent plan hash and final-key checks exist. Unverified weights must remain visible in any user-facing qualification claim. |
| Consequence 8: schema migration | Source now uses `/2` and refuses `/1`. Update review-trigger/schema wording; a historical assertion that no old files exist is not an inventory of present production folders. |
| Consequence 9: follow-on evaluation | Real faces, long chains, live relay and latent-stride checks remain open. Add false-alert rates and editor time saved to qualification evidence. |
| Verification: suites and provenance | New counts above supersede no historical run. This pass does not rerun whole-crate clippy/fmt or certify historical clean-room process. The added Rust fixture is formatted. Image/GPU/model results remain NOT_RUN. |
| Addendum: separate SpEdE component | Dockerfile pins the component revision and `NOTICE` records separation. No imported H3 source or model orbit was inspected. An image build is still required to verify the resulting package composition. |
| Addendum: toolchain and optional build | CUDA base digest, torch pins, gsplat version and architecture list are source configuration. `SPEDE=0` is the stated skip path. Ahead-of-time compilation and dependency compatibility were not executed here. |
| Addendum: weights and restrictions | Downloader profiles and manifest record revisions/hashes and use restrictions. This is configuration inspection, not downloaded-weight validation or a legal clearance. Release qualification must record the actual selected models and permitted use. |
| Addendum: status and post-build script | Preserve NOT_RUN. Repair M3 before treating registration success as pack completeness; then run the optional GPU fixture and representative model workflows, recording each separately. |

## Comparison and user utility

Kinoloom presents a local story/cast/shot workflow with long-take generation inside the model. Its public description targets a single Windows PC and an 8 GB NVIDIA card; this is a project claim, not an estate benchmark. VisionFlow's inspected advantage is its explicit lineage, indexed edits and measurable review records. That advantage needs a workflow an editor can use without interpreting raw JSON. This positioning is an inference from the two implementations and [Kinoloom's own description](https://github.com/harlixay7/Kinoloom).

A useful next slice is a review queue showing each suspect cut, adjacent frames, audio playback, declared threshold provenance and the exact rerun scope, with the editor's disposition recorded separately. Measure false alarms, missed defects, review time and wasted GPU reruns on consented or synthetic evaluation footage. Keep face-reference retention and deletion explicit; this pass used synthetic embeddings and downloaded no face weights.

OpenTimelineIO already represents clips, timing, tracks, transitions, markers and external media references. It offers a plausible interchange boundary for exporting flagged cuts and provenance into established editorial tools. That is a proposed integration, not an implemented adapter or compatibility result. Start with rational-frame timing and a round-trip test for cuts and markers before promising full transition fidelity. See [OpenTimelineIO's documentation](https://opentimelineio.readthedocs.io/en/latest/).

[AB-38 diagrams](../diagrams/agentbox/38-video-measurements-and-provider-boundaries.md) distinguish executable measurements, provider races and unrun qualification. The wider estate audit remains active; this pass closes the section reading of ADR-2163, not all media, predecessor or deployment obligations.
