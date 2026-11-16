---
title: Estate profiles, publishing, Loom evaluation and governance delivery
date: 2026-11-02
status: assessed
type: explanation
---

# Estate profiles, publishing, Loom evaluation and governance delivery

I assessed all 31 panels in five estate topics, reading every cited line at the wave 2 pinned revisions: ES-10 (11), ES-11 (5), ES-12 (8), ES-13 (2) and ES-90 (5). Nine panels are accurate, fifteen are corrected, seven are accurate with limits and none are inaccurate-unfixed. Five of the fifteen corrections only renumber citations that moved (ES-10.8, 10.10, 10.11, 10.12 and 11.3); the mechanism they describe still holds. The most important finding is in ES-10.4/ES-10.5. They showed a release VisionClaw build entering RBAC report mode with a same-day acknowledgement. At visionclaw `86cdd76` any report-mode request is a boot-assertion finding, and a release artefact exits 2 before the RBAC gate is built. Two other panels were wrong on substance. ES-11.4 called `kit-pin-guard.yml` the only kit-pin check, but the same script also runs as a CI job and in the dream cycle. ES-11.1 named a superseded ontomark release (v0.6.0; v0.7.0 is pinned). The operator excluded RuView from this audit. On the lead's instruction, ES-90.4 no longer cites it. The panel keeps a plain statement that RuView exists and carries an `**Open:**` marker. All five topics now pass the strict gate.

## Findings

**EPPE-01: ES-10.4/ES-10.5 showed report mode reachable in release (corrected).** Scope: `src/main.rs:1137` is the only `RbacGate::from_env` in the tree, and `main.rs` is the only non-test file that calls `HttpServer::new`. The other matches are doc comments, so the correction covers every server entry point. Trigger: `RBAC_GATE_MODE=report` with `RBAC_REPORT_MODE_ACK` set to today's UTC date in a release build. The old panels routed this to "report ACTIVE". At visionclaw `86cdd76`, `evaluate_effective_profile` pushes `ReportModeRequested` whenever report mode is requested, acknowledged or not (`src/config/security_profile.rs:523-526`). `assert_effective_profile_or_exit` exits 2 on any finding in a non-debug artefact (`src/config/security_profile.rs:636,645-654`; `may_bind_listener` lets only `debug_assertions` builds continue). The assertion runs at `src/main.rs:952`, before `HttpServer::new` (`src/main.rs:972`) and the `RbacGate::from_env` wrap (`src/main.rs:1137`). Consequence: the dated ack still matters to `report_mode_acknowledged` (`src/config/security_profile.rs:473-480`) and is recorded in the finding, but it cannot activate report mode in a release build. `rbac_gate.rs:104-110` is defence in depth. ES-10.4 now carries the case as hard-fail item N3. ES-10.5's release branch now reads "unreachable in practice". A `**Drift:**` line records the change.

**EPPE-02: ES-10.3 named comment lines instead of flags (corrected).** The labels "line 121/122/135/129/105" were not path citations, so the citation gate never checked them. The compose file is unchanged since the old stamp, and the flags sit at `docker-compose.unified.yml:116,132,133,140,146`. They are now path citations, so the gate checks them.

**EPPE-03: ES-11.4's "only check" invariant was false (corrected).** At dreamlab-ai-website `0e15728`, `kit-pin-guard.yml` only runs `scripts/dream-kit-pin-guard.sh` (`.github/workflows/kit-pin-guard.yml:26`), which describes itself as a thin wrapper over `scripts/pin-parity.mjs` (`scripts/dream-kit-pin-guard.sh:4-7`). The same script is the `pin-check` job in `.github/workflows/ci.yml:215,225`. Three other fixes in the panel:
- The title said four files where the body said five.
- "On 2 Oct it moved three times" understated the history. `git log` on `rust-ci.yml` shows eight kit-pin moves that day.
- The forum-production claim cannot be checked from source. It is replaced by the fact that kit `9c36b60` is an ancestor of the forum's pinned `c39f41f`.
- The invariant said the website consumes the kit "by source clone, not as a published package". `forum-config/Cargo.toml:45-49` says the git-SHA pin was retired at beta.6. The four library crates come from crates.io at exact versions (`:50-53`), and `KIT_REF` pins only the source clone the client and workers are built from. The invariant and ES-11.1's edge-13 label now describe both channels.

**EPPE-04: ES-11.1/ES-11.2 named a superseded ontomark release and missed two new gates (corrected).** visionGraph `556a015` pins ontomark `v0.7.0` with a new digest (`.github/workflows/publish.yml:49-51`). It also added a collision-lint gate (`publish.yml:144-145`) and made the body-residue check fail the publish (`publish.yml:248-253`). Every deploy-step citation shifted by 7–11 lines and was re-resolved.

**EPPE-05: ES-11.5 dated the sidestr crate count to the wrong commit (corrected).** "Since c001415, the eight sidestr-rs crates" mixes two commits. The roster now lists seven `sidestr-*` crates plus `webledgers-teller` (`scripts/estate-health/roster.json:73-80`). The eighth was added later (commit `26df6c9`) and is a member of the sidestr-rs workspace (`sidestr-rs/Cargo.toml` at `814b9e5`).

**EPPE-06: Governance expiry has no effect on the Agentbox apply path (open, registered).** ES-13.1's tension is confirmed at agentbox `c57d9c0`. `_applyOntologyBranch` checks the request's `d` tag and IRI, then applies (`management-api/adapters/orchestrator/local-process-manager.js:444,449,461`). `applyOntologyDecision` and `proposalFromRequest` live in `management-api/lib/ontology-apply.js:372,536`. No non-test agentbox file reads `stale_after`; the only mentions are log strings in `skills/podcast-knowledge-ingest/submit-proposals.mjs:269,283`. A decision the forum has closed as expired (`nostr-rust-forum crates/nostr-bbs-relay-worker/src/cron.rs:834-836`) can still mutate the corpus. To close it, the apply path should refuse a decision whose request is past `stale_after`, or wait for the forum's projection receipt.

**EPPE-07: The deadline parser can panic on multibyte input (Debt, registered).** `parse_rfc3339_utc` checks only the byte length, then slices `s[..19]` and `&s[19..]` (`nostr-rust-forum crates/nostr-bbs-core/src/ontology_governance.rs:284,289,313`). A `stale_after` string whose byte 19 falls inside a multibyte character panics rather than returning `None`, which contradicts the function's doc contract (`:246-250`). The parser runs from `stale_after_from_content` (`ontology_governance.rs:257`) when the relay admits a proposal (`nostr-rust-forum crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:417`), so the panic would hit the admission path, not the cron. This is source reading only: I reproduced no panic. To close it, use `s.get(..19)?` and `s.get(19..)?`, and add a multibyte test. ES-13.1's old label "signed offsets" described correct behaviour as if it were a risk. The label now names the slice.

**EPPE-08: The relay consumer's checkpoint comment overstates crash safety (Drift, registered).** `relay-consumer.js:466-469` says a crash mid-apply leaves the decision re-deliverable. That holds only for the event whose handler is running. A queued later event, already in `seen` (`:464`), is persisted by the earlier event's `settle` (`:473`, `:395`) and is suppressed on restart (`:376-377`, `:459`). This matches ES-13.2. To close it, persist only ids whose handler completed, or keep an in-flight set out of the checkpoint.

**EPPE-09: RuView is out of scope and its ES-90.4 claims were removed (edited, unverified).** ES-90.4 cited `../RuView/v2/crates/wifi-densepose-sensing-server/src/main.rs` at three lines, and no `wt/RuView` alias exists. The operator excluded RuView from this audit, so on the lead's instruction I removed the RuView entry from `sources:`, its `verified_commit` key and the three line citations. ES-90.4 keeps only a plain statement that the RuView sensing server exists and has a simulated mode, and the topic now carries an `**Open:**` marker saying RuView is outside scope and unverified. I had read the file at `5728b90` before the exclusion was known. This report draws no conclusion from that reading.

**EPPE-10: ES-90.5 omitted the environment switch for JEV (corrected).** The generator's citation review is on when `--jev` is passed or when `VISIONFLOW_JEV` is set (`scripts/lib/jev.cjs:51-58`). This container sets `VISIONFLOW_JEV=1`, so every gate run in this wave sent cited lines to the remote reviewer and printed advisories. The panel now names the switch. Its generator citations moved by +2 at VisionFlow `5f8b1da` and were re-resolved. All JEV advisories raised on these topics (ES-10.2 `main.rs:958`, ES-11.4 `Cargo.toml:50-53`, ES-12.3/12.5 PRD-028 `:179`/`:279`, ES-12.8 ADR-140 `:640`) were checked by hand. Each cited line holds its label. The ES-11.4 advisory persists after the fix: lines 50–53 are exactly the four `=1.0.0-beta.16` pins the label names. The real error in that panel was the separate "not a published package" sentence (EPPE-03).

**Carried forward unchanged (re-checked):**
- ES-90.2: `darwin-entrypoint.sh` cites ADR-0008 (`:36,113`) while `docs/adrs/` at dream-machine `b77093e` stops at ADR-0007.
- ES-10.12: the egress register's `nostr-gateway` row says the operator key is in the environment (`config/egress-policy.json:447`). That is true only with `role_isolation` off.
- ES-10.11: ADR-2098 and the build disagree on the sidechain gating keys.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| ES-10.1 | accurate | `rbac_gate.rs:135-141` fail-closed public reads; `main.rs:809-819` owner-less refusal; `visibility_filter.rs:26-43` defaults on; `role_store.rs:195-208` Editor default, Viewer on an unknown value (`models/rbac.rs:68-69`); `security_profile.rs:54,72` six flags; `main.rs:944-952` call site; ADR-2038 frontmatter `accepted/partial/live` |
| ES-10.2 | corrected (wrong citation) | Boot order: `main.rs:209` hygiene, `:239` env validation, `:789-819` owner, `:952` assertion, `:972` server. Exit codes match `:125`, `:161`. Receipt citation moved from `:952` to `:958-962` (the `info!`) |
| ES-10.3 | corrected | EPPE-02. Compose flags at `:116,132,133,140,146`. Cloudflared standalone file `:3-8`, digest `:16`. ADR-2119 `:64-66` |
| ES-10.4 | corrected | EPPE-01. Added N3 (`security_profile.rs:523-526`, `main.rs:1137`). Launcher pre-flight `launch.sh:181-187,205-208,215-230` is accurate. A missing `.env.prod` is also refused for `up`/`restart` (`launch.sh:241-245`) |
| ES-10.5 | corrected | EPPE-01. `rbac_gate.rs:92-111,201,210,262`; `security_profile.rs:473-480` ack rule |
| ES-10.6 | accurate-with-limits | `auth.rs:116-131` dev bypass and release stub; `filter_auth.rs:28-29`; `auth_extractor.rs:24-25,149`; `auth.rs:170-189,291`. Limit: the SNAT premise and the headset path are runtime. The bypass also reaches `memory_cloud_handler.rs:66` and `speech_socket_handler.rs:165`, which the panel does not list |
| ES-10.7 | accurate-with-limits | `security_profile.rs:485,596,624,654`; ADR-2038 `:252,319` record the implicit-locked divergence. Limit: "release negative probes return exit 2" rests on an earlier execution receipt, not on this review |
| ES-10.8 | corrected | Renumber only: INGRESS `:244`, flake `:420,2852-2859,3529-3531`, toml `:892-893`, ADR-2139 `:298`. `check-ports-loopback.mjs:93-104` (10 doors, unchanged), `:740-745`, `server.rs:151-169` 403. ADR-2139 `activation_status: staged` matches "CONFIGURED". Limit: peer reachability on `visionclaw_network` is runtime |
| ES-10.10 | corrected | Renumber only: toml `:697-712`, `:712`. `role-accounts.json:22-26,46-51`; `session-plane-acl.json:8-11` uids 1000/961 (961 = ab-gateway); `g5-key-split.json:13-32`; `SECURITY-profiles.md:25` still lists the AoE token |
| ES-10.11 | corrected | Renumber only: flake `:322,3116,3127,3145,3162,3232`; toml `:1957-1964,1958,1959,1975,2248`; BASELINE `:234`; INGRESS `:255`. Ten SANCTIONED entries unchanged |
| ES-10.12 | corrected | Renumber only: entrypoint `:608-612` (o+rw when off), `:615-631` (degraded marker), `:4003` (DOCKER_HOST); flake `:4041-4046`. uids 960–972 (`role-accounts.json`); 46 egress routes; `docker-read-proxy.cjs:39,120-123,130-139`. Limit: the host socket mode is a host fact |
| ES-11.1 | corrected | EPPE-04. v0.7.0 at `publish.yml:49-51`; deploy `:526-531`; dispatch `:536-542`; certificate `:217-234`. VisionClaw `ontology-publish.yml:15-16,93,180-191,287,332` and `ontology_pull.rs:295` unchanged |
| ES-11.2 | corrected | EPPE-04. All steps re-resolved; collision lint and blocking body-residue steps added |
| ES-11.3 | corrected | Renumber only: notice file `publish.yml:481`. KG `package.json:20` v0.1.1 tarball, VG `package.json:25` v0.2.2 tarball, `vowl-wasm/Cargo.toml:5` 0.2.2, roster `:16,18,60,81` |
| ES-11.4 | corrected | EPPE-03. KIT_REF `rust-ci.yml:18,20,33`; `kit-pin-guard.yml:26,29`; `ci.yml:215,225`; `forum-config/Cargo.toml:45-53` |
| ES-11.5 | corrected | EPPE-05. 18 roster repos (`roster.json:5-22`); `ci_by_design` on knowledgeGraph and WasmVOWL (`:12,16`); `estate-health.mjs:107-112,558-562`; `drift-counter.yml:57-62` pins agentbox `b859b37` |
| ES-12.1 | accurate | PRD-025 `:180-216` axes 0.27/0.94/0.95/0.65, three findings, 2026-08-18 addendum; LOOM-POSITIONING `:68-84`; email-search `SKILL.md:92-93` |
| ES-12.2 | accurate | PRD-028 `:19,29,96,100,149-157` |
| ES-12.3 | accurate | PRD-028 `:169,177,179` (the JEV advisory on `:179` is a false positive) |
| ES-12.4 | accurate | PRD-028 `:216-230,290` |
| ES-12.5 | accurate | PRD-028 `:265-279,288,315-326` (302 curator notes below the 400-unit floor) |
| ES-12.6 | accurate-with-limits | ADR-139 `:13-41`, ADR-138 `:31`. Limit: traces the decision record. `ServedMode::Passthrough` exists in `crates/loom-client/src/client.rs:403`, but the façade's runtime behaviour is not confirmed |
| ES-12.7 | accurate | LOOM-POSITIONING `:30-42,52-64`; PRD-028 `:27,29,45` |
| ES-12.8 | accurate | ADR-140 `:283-287,449,628,638,640` (pilot figures match exactly) |
| ES-13.1 | corrected | EPPE-06 and EPPE-07. Forum `nip_handlers.rs:1063,1077,1344,1552,1634`, `receipts.rs:185-191`, `cron.rs:764,794,834`. Agentbox `relay-consumer.js:446-454,513,532` (signature check now fail-closed at `c57d9c0`, `_verifySig`), `local-process-manager.js:419,444,461`. Parser label and citations replaced |
| ES-13.2 | accurate-with-limits | EPPE-08. `relay-consumer.js:369-377,389-395,459-473,496-502`. Limit: the snapshot/restore reproduction is the earlier audit's fixture, not re-run |
| ES-90.1 | accurate | `roster.json:5-22` (18 rows), `:13,14,21,22`; diagram directories match the primary list |
| ES-90.2 | accurate-with-limits | dream-machine `b77093e`: `darwin-entrypoint.sh:26-37,56-135,157-184,197`, `cli/src/index.ts:323,347-355,369-383,405-412`, `darwinBounds.ts:61-66,104,190-197`, `compile/src/index.ts:248-249,268-270`, `dream.config.json:55`, `loomShim.ts:14` (30 s abort). Drift prose: dream-engine `59a3e21` `Cargo.toml:11`, `dream-gate/Cargo.toml:3`, `readiness.rs:228-232`; agentbox `services/dream-engine/Cargo.toml:26-27`. Limit: annexe-runner enforcement is external |
| ES-90.3 | accurate-with-limits | WasmVOWL `51a1484` `useWasmSimulation.ts:36,105`, lock `vowl-wasm 0.1.1`; probe log lines 1–3. Limit: the probe is a 2026-09-07 execution record |
| ES-90.4 | accurate-with-limits | `prose-sanitiser Cargo.toml:18` (seven members, at `b261f65`); loom `Cargo.toml:66` path dependency `../ruvector/crates/ruvector-core`; diagram-ir three `[[bin]]` at `15cba16`. Limit: RuView excluded by operator; not checked (EPPE-09) |
| ES-90.5 | corrected | EPPE-10. Generator functions at `:389,454,502,636,700,724,807,352,355` |

## Verification

Evidence class: source tracing at the pinned revisions. I read every cited line with `git show <sha>:<path>`. Old citations were mapped to new line numbers by matching identical lines between the stamped and pinned file versions, and each moved line was re-read rather than shifted. I did not run product tests, did not execute any binary, and claim no runtime or deployment evidence. Statements about runtime fixtures, release probes and the 2026-09-07 WasmVOWL probe are carried forward as limits.

Revisions read: visionclaw `86cdd76355823a51bb64d503c005fe277726e724`, agentbox `c57d9c0508f4044915f9aecea4b9aa5629cc6a35`, nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`, visionGraph `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`, knowledgeGraph `4ed9ac159daf402b4fb252ce559dbeb894a8d91e`, dreamlab-ai-website `0e1572801d03a1af6d8bc8861f928da087d69eed`, loom `ba21a689a57dc67dc68ad22e52b058da34a7fb05`, dream-machine `b77093e564ef`, WasmVOWL `51a148430190`, VisionFlow `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083`. These repositories were not wave-pinned. On the lead's instruction I read them at the topic's declared `verified_commit`, each of which exists locally, and left those stamps unchanged: vowl-wasm `28af436`, dream-engine `59a3e21`, prose-sanitiser `b261f65`, diagram-ir `15cba16`. RuView was excluded by the operator and removed from ES-90.

Restamps: ES-10 (visionclaw, agentbox), ES-11 (website, knowledgeGraph, visionclaw, visionGraph, VisionFlow), ES-12 (loom, agentbox; cited loom and agentbox bytes are identical between old and new stamps), ES-13 (agentbox; the forum was already at `c39f41f`), ES-90 (VisionFlow, agentbox, loom; RuView key removed).

Gate, from the worktree root, `node scripts/diagram-index-gen.cjs docs/diagrams --only <file> --check --cite-check --strict-citations --render`:

| Topic | Citations verified | Rendered | Exit |
| --- | --- | --- | --- |
| `10-deployment-and-security-profiles.md` | 134 | 11/11 | 0 |
| `11-publishing-axis.md` | 66 | 5/5 | 0 |
| `12-loom-evaluation-and-preregistration.md` | 71 | 8/8 | 0 |
| `13-governance-delivery-and-expiry.md` | 34 | 2/2 | 0 |
| `90-evidence-and-supporting-repositories.md` | 37 | 5/5 | 0 (after RuView removal, EPPE-09) |

The gate checks only citations inside Mermaid. I re-resolved prose citations by hand at the same revisions. JEV ran on every gate call because `VISIONFLOW_JEV=1` is set in this environment. Its advisories were checked by hand (EPPE-10).

Not verified: deployed VisionClaw, Agentbox or forum behaviour; whether `role_isolation` is on anywhere; host Docker socket mode; any Loom façade response; the forum deployment's kit; whether the parser panic in EPPE-07 is reachable from a signed proposal in production.
