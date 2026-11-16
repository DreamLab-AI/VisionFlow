---
title: Estate audit synthesis — what the code does for its users
date: 2026-11-02
status: audit-complete
type: explanation
---

# Estate audit synthesis

This is the closing synthesis of the VisionFlow estate audit. It answers one question: **what this estate does for the people who use it, and where it fails them.** It draws on four waves of review against pinned revisions, and on the component, contract, comparison and SideStr passes recorded under the completion obligations. The per-wave reports hold the detail; the findings index ([`2026-11-02-findings-index.md`](2026-11-02-findings-index.md)) holds every finding with its evidence class.

## What the audit covered

| Obligation | Result |
|---|---|
| Diagram panels | 1,750 of 1,750 assessed at pinned revisions |
| Numbered decision ADRs | 448 of 448 assessed section by section |
| Stale claims | 0 |
| Uncited components | 80 of 80 given a disposition (46 assessed, 12 covered elsewhere, 22 out of scope, 0 gaps) |
| Open SideStr and Hitch findings | 25 rows re-checked at the current pin: 24 still open, 1 not checkable, 0 closed |
| Cross-repository contracts | 36 assessed: 13 agree, 18 disagree, 5 unverified; 1 shared fixture |
| Comparisons and recommendations | 100 rows, each labelled by what supports it |

Exclusions, as the operator directed: RuView, Logseq history, the RuVector repository, and the consumed Codescape and Unmute copies. Standalone Codescape is in scope.

## Evidence class, stated once

Nearly all of this is **source tracing at pinned revisions**: reading code, not running it. Nothing here is a runtime, test or deployment result unless a report says so. "Implemented" means the code does what the record says; it does not mean the feature works in production.

**A caveat on panel detail.** Cross-vendor spot checks (Codex `gpt-6-astra`) found detail errors in panels that Opus workers had already marked accurate or corrected: 5 confirmed errors across the samples it could judge, all fixed. Most sampled panels could not be judged from excerpts, because their claims span several files or assert that something never happens. Treat a panel's mechanism as reliable and its finer detail (exact counts, which crate is gated, which actor answers) as usually right but not guaranteed.

## Cross-vendor coverage, by pass

| Pass | Cross-vendor check |
|---|---|
| Waves 1 to 3 (panels and ADRs) | Codex spot checks of samples; 5 detail errors found and fixed |
| Wave 4 contracts | Codex on the three high findings: F3 HOLDS; F1 and F2 settled by the lead from the source excerpts Codex said were missing |
| Wave 4 SideStr and Hitch | Codex could not judge E2 from excerpts; lead-verified only — **Claude-only (degraded)** |
| Wave 4 component coverage | No cross-vendor check — **Claude-only (degraded)** |
| Wave 4 comparisons | No cross-vendor check — **Claude-only (degraded)** |

Contract assessment is the README's obligation and is met; shared contract fixtures are a recommendation, not built here.

## What works for users

- **Solid pods** (solid-pod-rs) are the most carefully specified part of the estate. Every numbered ADR and panel is assessed, and the mirror publication leak (SP-GIT-02) is repaired and merged into local main with migration of pre-release stores.
- **The forum** (nostr-rust-forum) has a complete diagram and ADR record. About half of its checkable ADR sections are implemented as written (39 of 101 rows, with 30 more partly); 10 are contradicted.
- **VisionClaw's server core** — boot, actor supervision, request lifecycle and RBAC — matches its records closely once the ADR-2142 move of rosters to `AgentBeamActor` is accounted for.
- **agentbox's build and boot path** is fully catalogued, and its manifest gates are implemented as recorded, with the corrections noted.

## Where it fails users

These are the findings a user would notice. Each is confirmed against source by the lead unless marked otherwise.

1. **Agent actions never reach VisionClaw's embodiment view.** In a default release build the forwarder sends no credential and is refused with 401; in dev builds the socket opens but every frame is refused as malformed. Neither failure is logged. The 401 assumes the default manifest-derived URL: an operator who sets `AGENTBOX_HOST_WS_URL` with a `?token=` query could pass it. (W2-01, CONTRACTS-F1, CONTRACTS-F2.)
2. **Task listing always fails.** agentbox's `GET /v1/tasks` omits `task`, `provider` and `status`, which VisionClaw's client requires, so every `list_tasks` call fails to deserialise. (CONTRACTS-F3.)
3. **Saved API keys are never shown back.** VisionClaw saves them through `NostrService` and reads them from `ProtectedSettingsActor`, which nothing ever writes. (W3-VC-06.6.)
4. **The Solid mirror repair does not ship, and the mirror cannot work as shipped.** agentbox still pins solid-pod-rs `76ab7aa`, which predates SP-GIT-02. Separately, the server builds the mirror with `DefaultSsrfCheck`, which refuses loopback, while agentbox's default mirror relay is `ws://127.0.0.1:7777`; so even after the pin moves, the mirror refuses its own relay under shipped defaults (ADR-2010 finding).
5. **Ordinary loom searches are refused.** The read-only SPARQL guard rejects any search containing a word such as "add", "load" or "service". (COV-B-F1.)
6. **The loom façade can be crashed by an error message.** The embedder slices an error body at byte 200 without a character-boundary check, under `panic = "abort"`. (COV-B-F2.)

## Security findings

All confirmed against source:

- **agentbox's relay consumer admits every author when its allowlist is empty**, and the flake never sets that allowlist. The relay gate in front of it limits the exposure. (A4.)
- **VisionClaw's derived-write door checks only `power_user`.** Loom ADR-135 D7 requires a signed distiller envelope plus an allow-list for writes into `:summary`; `/ontology/derived` checks the role and nothing else (`ontology_derived_handler.rs:188`). Any power user can write distilled knowledge. (LOOM-A-01.)
- **The Isaac Sim GPU sidecar runs jobs with no authentication**, inheriting the runner's full environment. It is reachable only on the container network. (CD-F3.)
- **The LWS CID verifier never enforces its `iat` window.** It is not yet wired into the server, so it is a library defect rather than a live hole.
- **SideStr snapshots do not verify code bytes against their hash** (E2), so a crafted snapshot restores without error.

## Recommendations

1. **Add cross-repository contract fixtures.** Half the estate's contracts disagree (18 of 36). agentbox's own forwarder test asserts the defective frame shape (CONTRACTS-F2), so that side's test passes against the bug. One fixture per contract, run on both sides, would catch findings 1 and 2 above.
2. **Move agentbox's solid-pod-rs pin** past SP-GIT-02, then rebuild and deploy, so the repair reaches users.
3. **Fix the silent-failure paths**, not just the bugs: log refused agent-event frames, and surface deserialisation errors from `list_tasks`.
4. **Give prose-sanitiser and diagram-ir diagram areas.** Sixteen binaries on every agent's `PATH`, including an LLM egress path and a crawler that fetches attacker-chosen URLs, have no topic describing them. (COV-A-F11.)
5. **Close the SideStr integrity gaps**, starting with E2's code-hash check on restore.

## Not done

- **Commit B3** of SP-GIT-02 (operator recourse and durability) is queued and not started. It is availability-only.
- **No runtime, test or deployment checks** were run as part of this audit.
- **MSRV 1.89** for solid-pod-rs is unverified on a 1.89 toolchain.
- The coverage script under-counts components that topics cite through a submodule path, so a few components were reported uncited when VC-39 describes them (COV-C-F3).
