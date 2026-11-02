---
title: Stale ADR claim rebind
date: 2026-11-02
status: assessed
type: explanation
---

# Stale ADR claim rebind

I checked all 11 stale ADR claims listed in `STALE.md`. For each one I found the ADR revision whose bytes match the claimed `source_sha256`, diffed it against the pinned revision, and assessed every changed section myself. All 11 can be rebound to this report, and none had to be refused. In ten of them the change is metadata plus appended re-verification notes, and their claims check out at source, apart from the drifts listed below. Two ADRs change decision text. In ADR-2164, D8's endpoint refresh moves from 10 Hz to "every frame". ADR-2169 gains items 8 to 10: the held-only elastic overlay, the ×20 gain with the force cap scaled with it, the release that places the node, and constraints that are not part of the hold. The mechanism claims in both match the code at the pin. The main finding is that three ADRs (ADR-2002, ADR-2143 and ADR-2169) carry the same boilerplate `ff849f2ba` re-stamp from commit `ef6ebba13`. It misstates what changed on their governed paths and is dated a day early. Two re-verification notes also contradict themselves or the code in small ways. None of this changes a decision.

## Findings

**ST-1 — the `ff849f2ba` re-stamp misdescribes the governed delta (Drift).**
- **Trigger:** commit `ef6ebba13` appended an identical paragraph to ADR-2002, ADR-2143 and ADR-2169. The paragraph says the delta is "the ADR-sprint work recorded in ADR-2169, 2172, 2173, 2174 and the 2026-11-01 consolidation commits".
- **What the governed paths actually contain:**
  - ADR-2002: from `2107637d6` to `ff849f2ba`, the only change is the nine-line payment-authentication paragraph that `5fa0e1325` added to `docs/SECURITY-profiles.md` (`fix: bind payment authentication to body and shared replay cache`). That is unrelated to the sprint, and it is the same commit the claimed bytes come from.
  - ADR-2143: from `f1fa1862c` to `ff849f2ba`, the only change is one line of `docs/PROTOCOL-registry.md`, the VisionClaw hold rule. Codescape's gitlink is unchanged.
  - ADR-2169: from `e2afd65d6` to `ff849f2ba`, the only change is the 95 lines `15acb9219` deleted from `position_updates.rs`. The `3f25181a2` note just above it already accounts for those lines.
- **Dating:** all three paragraphs are headed "2026-10-31". `ff849f2ba` was committed at 2026-11-01T23:33Z.
- **Consequence:** the stamps are formally correct, since nothing changes any of the three decisions. But a reader cannot rebuild the delta from the text, and for ADR-2002 the text points at the wrong body of work.
- **Evidence:** `git diff --stat 2107637d6 ff849f2ba -- src/utils/nip98.rs docs/SECURITY-profiles.md` gives `SECURITY-profiles.md | 9 +`. `git diff --stat f1fa1862c ff849f2ba -- codescape docs/PROTOCOL-registry.md` gives `1 file changed`. `git diff --stat e2afd65d6 ff849f2ba -- <ADR-2169 verified_paths>` gives `position_updates.rs | 95 -`.
- **To close:** replace each paragraph with the actual per-path delta and the commit date.

**ST-2 — ADR-2166 annexes: wrong date, and one annex superseded by a later one (Drift).**
- **Superseded annex:** the `ef021a41d` annex describes `SpatialVolume.edge_radius` as "half the box diagonal". The later `633423357` annex records the change to half the smallest extent, and the code at the pin does that (`xr-client/scripts/spatial_volume.gd:120-126`: `minf(box.x, minf(box.y, box.z))`, `0.5 * smallest * world_scale`).
- **Wrong date:** the `633423357` annex is headed "2026-10-31", but the commit is 2026-11-01T20:33Z.
- **Scope:** that annex also says "agent edge placement holds". ADR-2166 does not decide agent edge placement; ADR-2109 and ADR-2135 do.
- **Consequence:** read in order, the history is consistent. Read alone, the `ef021a41d` annex states a formula the code no longer uses.
- **To close:** mark the `ef021a41d` sentence as superseded and correct the date.

**ST-3 — ADR-2172 `e99c16d31` annex says `hunt_grabs` is "unchanged" (Drift, minor).**
- **Change:** `hunt_grabs` (`xr-client/rust/src/render_store.rs:1788-1797`) now reads representatives through `raw_position_of`, not `position_of`.
- **Behaviour is preserved:** `position_of` became fold-aware (`render_store.rs:2304-2314`), and `raw_position_of` is the old behaviour under a new name. So the annex's substance holds, since the held-node path still reads raw slots, but "unchanged" is literally false.
- **Annex argument holds:** a folded member cannot start a grab because search and proximity map members to representatives. That relies on `canonical_visible` (`render_store.rs:1492`, used at `:1542` and `:1714`), which exists at the pin.
- **Related change, accurate:** `node_position` and `render_aabb_ids` now resolve through the fold-aware `position_of` (`binary_protocol.rs:2193-2196,2339-2350`). For a folded member, the fit AABB (D7 F1) now counts the representative's position. The annex states this correctly.

**ST-4 — ADR-2164 D8 "every frame" means every physics tick (Tension).**
- **Call path:** the link layer is packed from `_tick_links` (`graph_scene.gd:733-743`), which `_physics_process` calls (`:2633`, `:2708`). Physics runs at `physics_ticks_per_second=90` (`xr-client/project.godot:51`).
- **Limit:** "Rebuilt on every frame" is therefore true only when the display rate equals the physics rate (90 Hz). It is not true at other refresh rates or under physics catch-up.
- **What holds:** the Rust packing claim holds. `pack_links` (`render_store_pack.rs:667`) resolves each end through `drawn_position_of` (`render_store.rs:2304`), and `link_layer.gd` only uploads (`_refresh`, `:473-504`; the pack call is at `:498`).
- **What source cannot show:** the 0.09 ms and 2.6 ms figures are harness receipts that source tracing cannot check.
- **Earlier D8 finding:** the earlier report's breach of the instance budget still stands, with new anchors. Links are capped at 512 (`link_layer.gd:344-345`). Marker instances are appended after the links (`render_store_pack.rs:736-749`), and the instance count is links plus markers (`link_layer.gd:502-504`). Markers are now deduplicated per distinct end (`:720-726`). That does not reduce the 513-marker case, which has one common source and 512 distinct targets.
- **To close:** state the tick rate in D8, and give markers their own budget line.

**ST-5 — ADR-2169's `global_speed` floor is now inert across its whole span (carried rows D5 and D6 qualified).**
- **Code:** `step_overrides` turns on `elastic_overlay` for exactly the span of the floor (`force_compute_actor.rs:4581,4587`). While the overlay is on, `adaptive_speed` is forced to 0 (`execution.rs:1159-1161`). The Verlet branch of the integrator never reads `global_speed` (`visionclaw_unified.cu:1086-1092`). The kernel source reads it in one place only, on the adaptive path (`:1073`). The Rust step path writes it only into `sim_params` (`force_compute_actor.rs:1947`). `set_hold_overlay` and `set_hold_springs` run unconditionally right before every `execute_physics_step_with_bypass` call (`:2015-2017`).
- **Effect:** in the code at the pin, `HOLD_GLOBAL_SPEED = 2.0` (`force_compute_actor.rs:4532`) changes nothing. Items 8 and 9 say so. The earlier report's D5 and D6 dispositions ("the source constant implements it") remain literally true, but they now describe a constant with no effect.
- **Same withdrawal problem:** item 6's withdrawal recipe (setting `HOLD_GLOBAL_SPEED` to 0.0) is equally inert, as item 8 notes.
- **To close:** this is the owner's call, which the ADR already leaves open. Until then, the D6 "Amends ADR-2140" exception is nominal.

**ST-6 — the `HoldOverlay::set_active` doc comment contradicts item 9 (Debt).**
- **The comment:** `hold_overlay.rs:22-25` says a re-grab inside the settle window keeps the snapshot.
- **The code:** `set_springs` (`:33-42`) invalidates the snapshot on every transition from off to on, and the module's own test `a_regrab_inside_the_settle_window_snapshots_the_dropped_shape` asserts a re-snapshot.
- **Consequence:** the ADR (item 9 (iii)) and the code agree, and only the comment is stale.
- **Context:** item 8's "re-grab keeps the snapshot" rule is explicitly marked "superseded in part by item 9", so the ADR is internally consistent.
- **To close:** fix the comment, which is product code. That is outside this audit.

**ST-7 — ADR-2169 acceptance boundary still lists owner acceptance of item 6 as not established (Drift, carried).**
- **The conflict:** items 6, 8 and 9 record owner decisions dated 2026-11-01, but the acceptance boundary still names "whether the ADR-2140 amendment in item 6 is accepted by the owner" as unestablished.
- **Other carried text:** item 5's punctuation fragment ("the stored value., so") also remains.
- Both were raised in the earlier report and are unchanged at the pin.

**ST-8 — agentbox re-stamps are accurate.**
- **The four dream ADRs:** the `cc949bb79` note added to ADR-2071, 2081, 2114 and 2115 says the only governed change since `21769a881` is one test literal in `engine.rs`. Across each ADR's full `verified_paths` (including `exec-record.js`, its test, and `dream-inbox-surface.cjs`), the diff contains only `engine.rs:2134`, `pg: Default::default()`, inside `#[cfg(test)]` (from `:2100`).
- **ADR-2153:** the four new notes match `git diff --stat` over all 26 governed paths for each stated range: `2f15c531b..e1893e236`, `e1893e236..cc949bb79`, `cc949bb79..09067b095` and `09067b095..4f3bd5325`. The changed blocks are the trajectory-hook registration and the RuVector credential custody and scrub code. No ADR-2153 section mentions those blocks.
- **Stamp freshness:** the `verified_commit` of every ADR in this batch has an empty governed diff to the pin.
- **Not a defect:** this item is recorded so the rebind basis is explicit.

## ADR-2164 section assessment

The claimed bytes match VisionClaw `a1f48020c`. The only change is one bullet in D8. The frontmatter, including `implementation_status: none` and `verified_commit: 00669b7a2`, is unchanged. ADR-2164 has no `verified_paths`, so there is no governed-path delta to compute.

| Section | Disposition |
| --- | --- |
| D8: budget, endpoint bullet (changed) | **Implements, with one limit.** Rust `pack_links` and `drawn_position_of`, and `link_layer.gd` keeping only the records and the upload, are true at the pin (ST-4). "Every frame" means every 90 Hz physics tick. The timing figures cannot be checked from source. The earlier report's marker-budget breach still holds, with new anchors (ST-4). |
| All other sections (unchanged text) | **Carried** from [2026-10-31-xr-volumes-and-code-links](2026-10-31-xr-volumes-and-code-links.md), which assessed them at `75b72416b`. ADR-2164 has no `verified_paths`, so I used the files that report cited as the governed set. From `75b72416b` to the pin: `link_index.rs`, `xr_prefs.gd` and the `codescape` gitlink (`e3b76f63e`) are unchanged. `binary_protocol.rs` (+251), `graph_scene.gd` (+521), `link_layer.gd` (+260) and `spatial_volume.gd` (+24) changed. D5 still holds: the join still writes an empty `repo` and `revision: None` (`link_index.rs:281-284`). D6 and D7 rest on `_needs_purge` and `_refresh`'s disabled branch, both present (`link_layer.gd:460-482`). D4's `binary_protocol.rs:1298-1299` anchor was not re-located at the pin. That report's link-layer anchors (`link_layer.gd:317,430-448`) are stale, because packing moved to Rust, but the D8 conclusion they support still holds. |

## ADR-2143 section assessment

The claimed bytes match `34735e852`. The diff changes `verified_commit` (`f1fa1862c` → `ff849f2ba`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff from `ff849f2ba` to the pin is empty. |
| "Re-verification — 2026-10-31 (`ff849f2ba`)" (new) | **Conclusion true; description and date inaccurate (ST-1).** The governed delta is the one-line hold rule in `PROTOCOL-registry.md`, which does not concern Codescape. The record's claims are unaffected. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-codescape-runtime-boundary](2026-11-01-codescape-runtime-boundary.md) at `531df5ef7`. From `531df5ef7` to the pin, the governed delta is that same registry line. |

## ADR-2166 section assessment

The claimed bytes match `34735e852`. The diff changes `verified_commit` (→ `69b16867d`) and appends seven re-verification paragraphs.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| `03ddc61e1` annex: comment in `_physics_process` | **Accurate.** The far-ribbon comment change is present in `graph_scene.gd`. |
| `0df96bc52` annex: comment above `_tick_links` | **Accurate** (`graph_scene.gd:731-732`). |
| `e99c16d31` annex: `content_to_world` refactor | **Accurate.** `world_from_content` returns `content_to_world() * p`, and `content_to_world` runs `_ensure_built()` then `_content_world()` (`spatial_volume.gd:344-352`). The GUT counts are the author's receipts. |
| `ef021a41d` annex: `edge_radius` | **Superseded within the document (ST-2).** The helpers and the `set_scene_edge` call exist (`graph_scene.gd:3784-3796,3957`). The "half the box diagonal" formula was replaced. |
| `633423357` annex: inscribed sphere | **Accurate on code** (`spatial_volume.gd:120-126`). Its date and scope are wrong (ST-2). |
| `abcf36db1` annex | **Accurate.** The only governed changes are doc comments. The test counts are receipts. |
| `69b16867d` annex: `slots` argument | **Accurate** (`graph_scene.gd:740-743`). |
| All other sections (unchanged text) | **Carried** from [2026-10-31-portals-and-federation](2026-10-31-portals-and-federation.md) and its 2026-11-01 grab-amendment refresh at `531df5ef7`. From `531df5ef7` to the pin, the governed delta is `graph_scene.gd` (+29/−6) and `spatial_volume.gd` (+24/−1). `space_portals.gd`, `volume_arrangement.gd`, `emulated_volume.gd`, `hud.gd`, `xr_prefs.gd` and the membrane shader are byte-identical. The carried findings on portal hysteresis, the eight-membrane limit and the bounds-before-dim order therefore still rest on unchanged code. |

## ADR-2071 section assessment

The claimed bytes match agentbox `97b4e672f`. The diff changes `verified_commit` (`21769a881` → `cc949bb79`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-11-01 (`cc949bb79`)" (new) | **Accurate (ST-8).** Across all five governed paths, including `exec-record.js` and its test, the only change is the test literal at `services/dream-engine/src/engine.rs:2134`. `RuVectorConfig.pg` now carries the address and the password-file path (`ruvector.rs:22-32`). |
| All other sections (unchanged text) | **Carried** from [2026-11-01-dream-journal-and-api-down](2026-11-01-dream-journal-and-api-down.md) at `9dfc27b16`. From `9dfc27b16` to the pin, the governed delta is that one test line. |

## ADR-2081 section assessment

The claimed bytes match agentbox `21a2b49dc`. The diff changes `verified_commit` (`21769a881` → `cc949bb79`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-11-01 (`cc949bb79`)" (new) | **Accurate (ST-8).** `runner.rs`, `dispatch.rs` and `candidate_flow.rs` are unchanged. The one change is the `engine.rs:2134` test literal. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-dream-annexe-and-receipts](2026-11-01-dream-annexe-and-receipts.md) at `9dfc27b16`. From `9dfc27b16` to the pin, the governed delta is that one test line. |

## ADR-2114 section assessment

The claimed bytes match agentbox `75c349744`. The diff changes `verified_commit` (`21769a881` → `cc949bb79`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-11-01 (`cc949bb79`)" (new) | **Accurate (ST-8).** `candidate_flow.rs`, `prompt.rs` and `nightly.tmpl` are unchanged. The one change is the `engine.rs:2134` test literal. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-dream-execution-and-triage](2026-11-01-dream-execution-and-triage.md) at `9dfc27b16`. From `9dfc27b16` to the pin, the governed delta is that one test line. |

## ADR-2115 section assessment

The claimed bytes match agentbox `97b4e672f`. The diff changes `verified_commit` (`21769a881` → `cc949bb79`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-11-01 (`cc949bb79`)" (new) | **Accurate (ST-8).** `governance.rs`, `digest.rs`, `relay.rs`, `inbox.rs` and `config/hooks/dream-inbox-surface.cjs` are unchanged. The one change is the `engine.rs:2134` test literal. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-dream-governance-and-delivery](2026-11-01-dream-governance-and-delivery.md) at `9dfc27b16`. From `9dfc27b16` to the pin, the governed delta is that one test line. |

## ADR-2153 section assessment

The claimed bytes match `c970ca8b0`. The diff changes `verified_commit` (`2f15c531b` → `4f3bd5325`) and appends four re-verification sections.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-11-01 (ADR-2164 amendment)" | **Accurate.** `2f15c531b..e1893e236` changes only `entrypoint-unified.sh` (+10/−13), inside the trajectory-hook registration block. The `TRAJ_CONNINFO` input and the gate prefix are removed. |
| "Re-verification — 2026-11-01 (`cc949bb79`)", including the `09067b095` re-stamp | **Accurate.** `e1893e236..cc949bb79` changes `agentbox.toml`, `entrypoint-unified.sh` and `flake.nix` as listed: the password-free conninfo URLs, and the removed `claudeFlowConfigJson` template and baked `RUVECTOR_PG_CONNINFO`. `cc949bb79..09067b095` changes only the entrypoint's scrub block. |
| "Re-verification — 2026-11-01 (`4f3bd5325`)" | **Accurate.** `09067b095..4f3bd5325` changes only `entrypoint-unified.sh` (+112/−32), in the scrub block. No ADR-2153 section concerns RuVector credentials. A grep of the ADR for `ruvector`, `conninfo` and `credential` finds hits only in these annexes and in an unrelated OpenRouter key note. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-agentbox-session-lifecycle](2026-11-01-agentbox-session-lifecycle.md) at `9dfc27b16`. From `9dfc27b16` to the pin, the governed delta is the three files above (+264/−84). I found no session-plane, tmux, socket-ACL or `tab0` line in that diff. |

## ADR-2169 section assessment

The claimed bytes match `d6c091f0f`. The earlier report assessed this text at `531df5ef7`.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit`, `verified_paths` (changed) | **Accurate.** `verified_paths` adds `hold_overlay.rs`, `execution.rs` and `visionclaw_unified.cu`, which is where items 8 and 9 live. The governed diff from `ff849f2ba` to the pin is empty. |
| `acceptance_boundary` (sentence added) | **Cannot be checked from source.** The added sentence states GPU-test and calibration figures, which are runtime receipts. The old "owner acceptance not established" clause is stale (ST-7). |
| Item 8: elastic overlay (new) | **Implemented; superseded in part by item 9, as the ADR says.** The overlay term exists, with the rest lengths captured by `snapshot_edge_rest_kernel` (`visionclaw_unified.cu:955`) and read through `g_edge_rest_snapshot` (`:164`, `:629`, `:2387`; bound at `execution.rs:714-741`). The Verlet span exists: `adaptive_speed` is set to 0 (`execution.rs:1159-1161`) and `prev_force` is cleared on the Verlet path (`.cu:1093-1099`). `StepOverrides.elastic_overlay` exists (`force_compute_actor.rs:4553,4587`). Kernel arity is 30 for `force_pass_kernel` and 29 for the stability variant, counted from source; PTX was not inspected. The withdrawal recipe and "LinLog still runs" were not separately traced. The measured figures are receipts. |
| Item 9: gain, cap and placement (new) | **Implemented as written.** `HOLD_SPRING_GAIN = 20.0` (`mod.rs:145`, wired at `construction.rs:488`). `spring_k` and `max_force` are multiplied only under `springs_active()` (`execution.rs:1164-1167`), which is the line the "open conflict" note names. The overlap push `max_force * penetration` (`.cu:562`, `:2347`) and the boundary force `max_boundary_force = max_force` (`.cu:1138`) confirm the coefficient remark. Springs run during a hold only (`hold_springs: any_held`, `force_compute_actor.rs:4588`). Every off-to-on transition re-snapshots (`hold_overlay.rs:33-42`). The test `a_released_leaf_is_placed_where_it_was_dropped` exists (`execution.rs:4059`), and `LIVE_SPRING_K = 30.0` (`:3845`) supports the recommended `springK`. The measured numbers are receipts and were not rerun. ST-6 records a stale code comment. |
| Item 10: constraints not part of the hold (new) | **Implemented.** `sync_constraints` uploads only when `constraints_dirty` is set or the resident count differs (`force_compute_actor.rs:1674-1689`). `StepOverrides` has no constraint field (`:4540-4590`). |
| Consequences, tick-rate sentence (added) | **Cannot be checked from source.** The 29 → 9.8 ms figure belongs to ADR-2173, and the sentence itself says it awaits live confirmation. |
| Verification, item 8 paragraph (added) | **Partly checkable.** The named test modules exist (`hold_overlay::tests` in `hold_overlay.rs`, plus the GPU tests in `execution.rs`). The runs are author receipts. |
| `d5c46cdd8`, `8f796ad86`, `1ec762f39` and `e2afd65d6` stamps | **Accurate on the claims I checked.** `HOLD_GLOBAL_SPEED = 2.0` and `max` are at `force_compute_actor.rs:4532,4581`. `RELEASE_SETTLE_DAMPING` is absent. `g_edge_rest_snapshot` replaces the kernel parameter. The arity is 30/29. The text about the gain's effect on separation and boundary forces matches the `.cu`. The test counts and bit-identity runs are receipts. |
| `3f25181a2` stamp | **Accurate.** `position_updates.rs` lost only the swarm-telemetry handler. `may_hold`, `release_holds` (`:152`) and `parse_position` (`:104`) remain. |
| `ff849f2ba` stamp | **Conclusion true; description and date inaccurate (ST-1).** |
| Items 1–7 and the remaining Consequences (unchanged text) | **Carried** from [2026-11-01-drag-holds-and-acceptance](2026-11-01-drag-holds-and-acceptance.md), with D5 and D6 qualified by ST-5. From `531df5ef7` to the pin, the governed delta touches five files. `hold_registry.rs`, the orchestrator, `types.rs`, `layout_handler.rs` and the acceptance script are byte-identical, so DH1, DH2 and DH3's orchestrator half stand unchanged. I rechecked the anchors in changed files: `release_holds` still notifies `Released` before the GPU (`position_updates.rs:152-175`). `parse_position` still defaults missing axes to 0 (`:104-119`). Hold expiry still runs only after the `ComputeForces` early returns (`force_compute_actor.rs:1812-1868` precede `:1922`). |

## ADR-2172 section assessment

The claimed bytes match `34735e852`. The diff changes `verified_commit` (→ `69b16867d`) and appends seven re-verification paragraphs.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| `03ddc61e1` annex | **Accurate on source.** The coincidence tests and the per-frame far tier are present. The counts are receipts. |
| `0df96bc52` annex | **Accurate.** `git grep edge_frame` finds nothing under `xr-client/rust/src` at the pin, and `hunt_grabs` and `_grab_id` keep their roles. |
| `e99c16d31` annex | **Accurate in substance; "unchanged" literally false (ST-3).** |
| `15acb9219` annex | **Accurate.** The `binary_protocol.rs` and `render_store.rs` changes are the `0x23` state-frame ingest (`render_store.rs`, `apply_agent_state_frame`) and agent records. I read the whole `binary_protocol.rs` delta from `531df5ef7` to the pin (7 hunks). It covers agent-action decoding and its tests, the beam-anchor doc, a `render_store()` accessor for the link pack, and the doc comment on `node_position`. None of it is drag, grab, hold or `render_aabb_ids` logic. |
| `ef021a41d` annex | **Accurate.** The `graph_scene.gd` change is the agent-edge helpers only, with no grab code. |
| `abcf36db1` and `69b16867d` annexes | **Accurate.** `is_drawn` and `is_drawn_in_slot` (`render_store.rs:2281-2296`) exist, the beam-source fallback is removed, and `_tick_links` gets its `slots` argument. `position_of` resolves through `drawn_position_of`. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-xr-grab-boundaries](2026-11-01-xr-grab-boundaries.md) at `531df5ef7`. Its XG1–XG6 findings rest on `graph_scene.gd` grab functions, `backend_connection.gd` and `connection_volume_source.gd`. The latter two are byte-identical at the pin, and the 29-line `graph_scene.gd` delta contains no grab, hold, double-click or box-grip code. The report's `graph_scene.gd` line numbers belong to its own revision. |

## ADR-2002 section assessment

The claimed bytes match `5fa0e1325`, the payment repair the named report assessed. The diff changes `verified_commit` (`2107637d6` → `ff849f2ba`) and appends one section.

| Section | Disposition |
| --- | --- |
| Frontmatter `verified_commit` (changed) | **Accurate.** The governed diff to the pin is empty. |
| "Re-verification — 2026-10-31 (`ff849f2ba`)" (new) | **Conclusion true; description and date inaccurate (ST-1).** The real delta is the `SECURITY-profiles.md` payment paragraph from `5fa0e1325`, which the named report already covers. |
| All other sections (unchanged text) | **Carried** from [2026-11-01-visionclaw-payment-auth](2026-11-01-visionclaw-payment-auth.md). The governed diff from `5fa0e1325` (that report's revision) to the pin is empty. |

## Rebind outcome

| ADR | Claimed bytes at | Changed sections | Proposed |
| --- | --- | --- | --- |
| VisionClaw ADR-2164 | `a1f48020c` | D8 bullet | rebind |
| VisionClaw ADR-2143 | `34735e852` | stamp + 1 annex | rebind |
| VisionClaw ADR-2166 | `34735e852` | stamp + 7 annexes | rebind |
| agentbox ADR-2071 | `97b4e672f` | stamp + 1 annex | rebind |
| agentbox ADR-2081 | `21a2b49dc` | stamp + 1 annex | rebind |
| agentbox ADR-2114 | `75c349744` | stamp + 1 annex | rebind |
| agentbox ADR-2115 | `97b4e672f` | stamp + 1 annex | rebind |
| agentbox ADR-2153 | `c970ca8b0` | stamp + 4 annexes | rebind |
| VisionClaw ADR-2169 | `d6c091f0f` | frontmatter, items 8–10, consequence, verification, 6 stamps | rebind |
| VisionClaw ADR-2172 | `34735e852` | stamp + 7 annexes | rebind |
| VisionClaw ADR-2002 | `5fa0e1325` | stamp + 1 annex | rebind |

## Verification

- **Evidence class:** source tracing at the pinned revisions only. That means VisionClaw `86cdd7635582` and agentbox `c57d9c0508f4`, read with `git show` and `git grep` and never from working trees. I ran no product tests, builds, GPU runs, GUT runs or headset sessions. Every test count, timing and measured follow ratio quoted from an ADR is the author's receipt, not something reproduced here.
- **How the old bytes were matched:** for each ADR I walked `git log --format=%H <pin> -- <path>` from newest to oldest and SHA-256-hashed `git show <sha>:<path>`. Every claimed hash matched a historical blob. Every current hash in `STALE.md` matched the pin.
- **Deltas computed:** I used `git diff --word-diff` from the matched revision to the pin for each ADR. I used `git diff --stat` over each ADR's full `verified_paths` for three ranges: the named report's revision to the pin, each annex's stated range, and the ADR's own `verified_commit` to the pin.
- **Generator:** `diagram-index-gen.cjs` does not apply. I was assigned no topics and edited none. I made no ADR or product edits.
- **What I did not verify:**
  - I did not reassess unchanged sections against code. They are carried from the named reports, with the code delta from each report's revision recorded above.
  - For the per-commit notes in ADR-2166 and ADR-2172, I checked each claim against the state at the pin and against the aggregate governed delta. I did not re-run `git show --stat` for each named commit.
  - I did not inspect PTX.
  - I did not trace item 8's "LinLog and repulsion still run", or the withdrawal switches beyond the constants and flags named.
  - I did not check whether ADR-2164's link-pack 0.09 ms figure holds on a headset.
