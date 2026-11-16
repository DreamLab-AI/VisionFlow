---
title: Estate Solid-pod and build-deploy topics
date: 2026-11-02
status: assessed
type: explanation
---

# Estate Solid-pod and build-deploy topics

I assessed all 37 panels of the two estate topics ES-08 (Solid-pod estate, 13 panels) and ES-09 (build, deploy and CI estate, 24 panels) by source tracing at this wave's pinned revisions. Both topics were restamped and every citation was re-resolved. Of the 37 panels, 7 are accurate, 23 corrected, 7 accurate with limits and none inaccurate-unfixed. Two findings matter most. First, source tracing shows the compose production service does not use `supervisord.production.conf` at all. It builds `Dockerfile.production`, whose start script runs the backend in the background and nginx in the foreground, so nothing in that image restarts a backend that crashes after startup (ES-09.7). Second, the embedded pod's "non-destructive" PATCH starts from an empty graph whenever the stored resource is not N-Triples, so a PATCH against the JSON-LD the client writes replaces that content (ES-08.6). Both topics pass the strict gate, with render, at the new stamps.

## Findings

**F1. The compose production path has no process supervisor, and the supervised config points at a missing binary (ES-09.7, ES-09.19).**
- **Trigger:** `./scripts/launch.sh up prod` brings up `visionclaw-production`.
- **Consequence:** that service builds `Dockerfile.production` (`../project/docker-compose.unified.yml:256-258`), whose entrypoint is `/app/start.sh` (`../project/Dockerfile.production:237`, `:245`). The script backgrounds the backend and runs nginx in the foreground, with no supervisord (`../project/scripts/production-startup.sh:38-41`, `:68-70`). If the backend crashes after startup, nginx keeps the container alive, so compose's `restart: unless-stopped` (`../project/docker-compose.unified.yml:323`), which acts only on container exit, never fires. No compose path builds the unified `production` target, because the dev service hard-codes `target: development` (`../project/docker-compose.unified.yml:80`).
- **Second defect:** `supervisord.production.conf` is reached only through `Dockerfile.unified`'s `production` target (`../project/Dockerfile.unified:414`, `../project/scripts/prod-entrypoint.sh:23`). Its `rust-backend` command runs `/app/target/release/visionclaw-server` (`../project/supervisord.production.conf:29`), but that image puts the binary at `/app/visionclaw-server` (`../project/Dockerfile.unified:393`, `:406`).
- **Third defect:** the image sets `USER appuser` (`:427`), so the `[supervisord] user=root` line (`supervisord.production.conf:1-3`) cannot raise privilege. The panel's old "root PID1" framing was wrong for that image.
- **Evidence class:** source tracing; not run.
- **Closes when:** the production path has a restart policy for the backend (supervisord, or a compose `restart` on a backend-only process), and either the unified production target's command path is fixed or the target is retired.

**F2. PATCH against a non-N-Triples resource discards its content (ES-08.6).**
- **Trigger:** a PATCH (N3 or SPARQL-Update) on an existing Turtle or JSON-LD resource in the embedded pod.
- **Consequence:** `Graph::parse_ntriples` errors on the first non-N-Triples line (`../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs:817-828`, the same at the `v0.5.0-alpha.17` tag VisionClaw pins). The handler maps that error to an empty graph (`../project/src/handlers/solid_proxy_handler.rs:786-789`), then stores only the patch's triples as `application/n-triples` (`:805-812`). The client writes JSON-LD by default (`../project/client/src/services/solidPod/ldpClient.ts:153-157`) but never sends PATCH, so the exposure is to other Solid clients.
- **Evidence class:** source tracing; not reproduced.
- **Closes when:** a PATCH on an unparseable stored body is refused (409 or 415), or the stored body is parsed by its content type, with a test.

**F3. Pod initialisation reports doubled container paths (ES-08.5).**
- **Trigger:** `POST /solid/pods/init`, which the client calls through `initPod` and `getPodStructure`, or `/solid/pods/init-nip98`.
- **Consequence:** both structure builders pass `pod_base`, which already ends in the npub (`../project/src/handlers/solid_proxy_handler.rs:975`, `:1155`), to `pod_structure`, which appends the npub again (`:287-288`; called at `:1088` and `:1174`). Only `pods/check` passes the bare base (`:1264`).
- **Effect:** the `PodStructure` the client uses for `setPreference` would name `base/npub/npub/preferences/` rather than a provisioned container.
- **Evidence class:** source tracing; not exercised.
- **Closes when:** `pod_structure` is called with the bare base everywhere, with a test asserting one npub segment.

**F4. The public Type Index path is derived from a segment the preferences path never contains (ES-08.7).**
- **Trigger:** `ensurePublicTypeIndex`, which the type-index register and discover functions call.
- **Consequence:** `ensurePublicTypeIndex` cuts the path at the first `/settings/` (`../project/client/src/services/solidPod/typeIndex.ts:70-73`), but it is given `structure.preferences` (`../project/client/src/services/SolidPodService.ts:447`), which is `.../preferences/` (`../project/src/handlers/solid_proxy_handler.rs:294`). With `indexOf` returning -1, the substring is the first nine characters of the path.
- **Evidence class:** source tracing.
- **Closes when:** the Type Index is placed from the pod root or a real settings container, with a test.

**F5. ADR-2008's "landed on main, unreleased" is stale (ES-08.12).**
- **Trigger:** reading the record's status as current.
- **Consequence:** the D1 port commit `0befa5b` and the D6 commit `e62d028` are ancestors of the solid-pod-rs `v0.5.0-alpha.11`, `v0.5.0-alpha.17` and `v0.5.0-alpha.18` tags and of agentbox's Nix revision `76ab7aab` (checked with `merge-base --is-ancestor`). The hand-rolled `add_mod_n`/`neg_mod_n` exist at `v0.5.0-alpha.17` only as a mention in the ADR text. VisionClaw's alpha.17 dependency (`../project/Cargo.toml:246`) and agentbox's alpha.18 Nix pin therefore both ship the port.
- **Stale record text:** `activation_status: staged` and "unreleased" (`ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:7`, `:121`). The record belongs to the solid-pod-rs repository and was not edited.
- **Closes when:** the record's activation status and implementation note name the release that shipped D1 and D6.

**F6. The ADR ratchet's backlog rule has lapsed, and a new evidence rule replaced it as the only failing check (ES-09.12, ES-09.15).**
- **Trigger:** any push after 2026-10-20.
- **Consequence:** `ADR_RATCHET_UNTIL` defaults to 2026-10-20 (`../project/scripts/adr-ratchet.sh:45`), so an excess of new undecided records is reported and passes (`:119-123`).
- **What still fails a build:** the VisionFlow ADR-2007 evidence gate, which fails a closing record that lacks `evidence_class`/`acceptance_boundary` (`:81-113`). Both workflows run its self-test (`../project/.github/workflows/docs-ci.yml:39-43`, `../project/agentbox/.github/workflows/invariants.yml:212-213`).
- **Copies:** the host and agentbox copies are byte-identical at the two stamps (same sha256). The copy at VisionClaw's agentbox gitlink `c3746299e` predates the evidence gate and differs.
- **Closes when:** the owner extends or retires the backlog rule, and the step names stop describing it as live.

**F7. The ADR checker's "soft nudge" comment is not what the code does (ES-09.23).**
- **Trigger:** a record with no `verified_paths`.
- **Consequence:** the comment promises such records a nudge toward full SHAs (`../project/scripts/adr-index-gen.js:190-191`), but the warning is guarded on a non-empty `verified_paths` (`:204`), and those records skip the block (`:207`). They get no message.
- **Closes when:** the comment or the guard is changed to match the other.

**F8. Version and pin skew across the Solid deployments (ES-08.13, ES-09.10).**
- **VisionClaw:** embeds `solid-pod-rs` 0.5.0-alpha.17 (`../project/Cargo.toml:246`).
- **agentbox Nix:** builds the 0.5.0-alpha.18 workspace at `76ab7aab` (`../project/agentbox/lib/solid-pod-rs.nix:54`, `:58`), which is 25 commits behind the solid-pod-rs audit pin, and its `lib.rs` differs from the pin's.
- **agentbox docs:** the pod guide says alpha.13 (`../project/agentbox/docs/user/solid-pod.md:37`), and the baseline says alpha.17 and "Staged until the image is rebuilt" (`../project/agentbox/docs/BASELINE-container.md:241`, `:246-247`), while ADR-2145 records `complete`/`live`.
- **Submodule:** VisionClaw's agentbox gitlink at `86cdd76` is `c3746299e`, 51 commits behind the agentbox pin `c57d9c05` that these topics read agentbox at.
- **Closes when:** the two agentbox documents name the Nix pin, and the gitlink is bumped or the topics note which revision VisionClaw actually builds.

**F9. Resolved since the last stamp.** All four are marked RESOLVED in the topics:
- agentbox's stale `ontology-publish.yml` is deleted (agentbox `7b2811272`; absent at `c57d9c05`).
- The multi-user pod guide now matches the code: provision is implemented, and only suspend and archive return 501 (`../project/agentbox/docs/user/multi-user-pods.md:85-93`; `../project/agentbox/management-api/routes/admin-users.js:192`, `:241-243`, `:257-259`).
- ADR-2037's CI half exists as ADR-2086 (accepted, complete, live). The old ES-09.5 note that no CI assertion existed was wrong.
- The invariants comment now names `ontomark validate` as the successor contract check.

## Panel dispositions

### ES-08 Solid-pod estate (`estate/08-solid-pod-estate.md`)

| Panel | Disposition | Evidence |
|---|---|---|
| ES-08.1 | corrected | Feature lines moved: `default` at `Cargo.toml:279`, declaration at `:312`; supervisor blocks at `flake.nix:2681` and `:2698`. The topology claims hold: the solid-pods compose file is only the tunnel sidecar (`docker-compose.solid-pods.yml:1-7`), and JSS is absent from both trees. |
| ES-08.2 | corrected | `buildPodNip98` is at `pod-signer.js:70`; the old `:42` and `:50-59` were comment text. `withSigner` is at `adapters/index.js:85-88`. Added the ADR-2122 identity-port branch (`pod-signer.js:76-78`, `:132-136`) and the shipped `sign_requests = true` (`agentbox.toml:590`). The fail-closed path (`_solid-http-base.js:84-106`) holds. |
| ES-08.3 | corrected | The dev-mode branch is no longer in `ldpClient.ts`. Headers come from `computeAuthHeaders` (`authInterceptor.ts:51`, dev `:62-67`, NIP-98 `:76-77`), guarded by `isAuthenticated` (`ldpClient.ts:98`). The stale-session branch (`:110-115`) goes unsigned. Old anchors `:110`, `:112` and `:172-175` were wrong lines. No PATCH in the client: verified. |
| ES-08.4 | corrected | `evaluate_access` is imported at `solid_proxy_handler.rs:63`, not 55. The ACL lookup is against `FsBackend::authorization_path(storage_path)` (`:339-340`), and `FsBackend` is the `PublishedStorage` alias (`:48`). 401/403 and the feature-off empty router hold (`:351-364`, `:1820-1824`). |
| ES-08.5 | corrected | The client calls `/solid/pods/init` (`SolidPodService.ts:173`, `init_pod` `:1285`), not init-nip98. Both go through `ensure_pod_exists` (`:1148`). The response is 200 in both branches (`:1369-1375`), not 201. Added the ACL backfill (`:1157-1172`), F3 as Open, and the multi-user drift as resolved. |
| ES-08.6 | corrected | Every line reference was 4–5 lines off (now `:756`, `:780`, `:786`, `:792-793`, `:794-800`, `:806-812`, `:828-831`, `:836`, `:862-865`). The old `:44-48` were unrelated imports. Added F2 as a Tension; the title now states the N-Triples limit. |
| ES-08.7 | corrected | `AclEntry.modes` is typed Read, Write or Append only (`wacManager.ts:20`); Control appears only in `AclMode` and the owner grant (`:14`, `:46`). Added F4 as Debt in the Type Index note. |
| ES-08.8 | corrected | Lines restamped to `flake.nix:2679`, `:2681-2690`, `:2686`, `:2696-2707`, now in checkable form. Limit: no `startsecs`/`startretries` are set, so BACKOFF and FATAL follow supervisord defaults, which source cannot confirm. |
| ES-08.9 | corrected | Old anchors pointed at other functions: `agentMemory.ts:202-204` is `setAgentMemoryAccess`, and `ldpClient.ts:206-216` straddled DELETE and HEAD. Now `deleteAgentMemory` `agentMemory.ts:191-199`, `deleteResource` `ldpClient.ts:201-210`. The client source has no RuVector dispatch (searched for memory_delete, ruvector and tombstone). |
| ES-08.10 | accurate | JSS is absent from both trees at the pins. `_solid-http-base.js:11-12` and `solid-pod.md:73-75` say what the panel says. |
| ES-08.11 | accurate-with-limits | `main.rs:917`, `ontology_pull.rs:80-103`, `:309-337`, `:394-410` hold, and verify-then-write order is visible at `:337-360`. The sync and rename anchors are refined to `ontology_generation.rs:178-199`, `:203-204`. Limit: fsync durability and activation atomicity are runtime properties. |
| ES-08.12 | corrected | The ADR quotations hold at `9b6143c`. Added F5 as Drift: the port ships in alpha.11 onward. NOTLIVE now lists which decisions landed (D1, D6) and which were not built (D4, D5, D7, D8). |
| ES-08.13 | corrected | All `agentbox.toml` forge lines and the flake lines moved (`flake.nix:1538-1539`, `:1548`, `:1628-1629`; `agentbox.toml:603-625`, `:925`). `system_one.rs` re-resolved to `:554-558`, `:594-618`. Mirror start is at `lib.rs:4973-4994`, `:5034-5041`, rerank at `:5146-5165`. The Tension now quotes the baseline's alpha.17 and "Staged" text, and the Drift adds the baseline version. Limit: line numbers are the solid-pod-rs pin's, not the image's `76ab7aab`. |

### ES-09 Build, deploy and CI estate (`estate/09-build-deploy-and-ci-estate.md`)

| Panel | Disposition | Evidence |
|---|---|---|
| ES-09.1 | accurate-with-limits | The compose bind block (`docker-compose.unified.yml:169-206`) is as drawn. The host-socket trap is an operator-verified environment fact, not repository-committed. |
| ES-09.2 | accurate-with-limits | `launch.sh:640-651`, `supervisord.dev.conf:34`, `rust-backend-wrapper.sh:42`, `:66`, `:91-124` and `build.rs:38-39` all hold. Limit: rebuild timing and the health response are runtime. |
| ES-09.3 | corrected | `cargo fetch` is `Dockerfile.unified:183` and the build is `:184`; the panel had them one line late. The other stage anchors hold. |
| ES-09.4 | accurate | Stages, ARG/ENV lines, the four stub binaries (`sync_corpus.rs` included), fetch `:100`, PTX `:120`, `:230`, `:243` and `:245` all hold. |
| ES-09.5 | corrected | ARG/ENV promotion lines hold. The second note was wrong: ADR-2037 is `partial`/`staged` (`ADR-2037-...md:5-7`), and its CI half, ADR-2086, is accepted, complete and live (`ADR-2086-...md:5-7`). The remaining gap is that no gate inspects a built image. |
| ES-09.6 | accurate | Every program, `startretries`/`startsecs`, environment `:46` and socket `:60` match `supervisord.dev.conf`. |
| ES-09.7 | corrected | Retitled and extended with F1: the file is used only by the unified production target, its command path is wrong, and USER appuser defeats `user=root`. |
| ES-09.8 | accurate-with-limits | Profiles, ports, mounts, cloudflared digest, loom, codescape and the System One anchor (`:2-11`, `agentbox:8094`) hold. ADR-2148 is accepted/complete/live at the pin. Limit: "LIVE on 2026-10-30" and the port-3001 collision are runtime or close-out claims. |
| ES-09.9 | accurate-with-limits | Dev and production route tables, `^~` prefixes, X-Forwarded-Host lines and `auth.rs:170-189` hold, and root `nginx.conf` is referenced by no Dockerfile or compose file. The Open (LAN pod write signed for host:3001) stays untested. |
| ES-09.10 | corrected | Flake lines moved (`:481`, `:2622`, `:2653`, `:2669`, `:4008-4013`). The gate key is `agentbox.toml:2119-2120`, and the old `:2081-2082` were linked-objects lines. The agentbox gitlink at `86cdd76` is `c3746299e`, not `e786197cc`. Added the F8 skew as Drift. |
| ES-09.11 | corrected | `ci.yml:120` now also runs `--test layer_direction`; every other job and step anchor holds. |
| ES-09.12 | corrected | `docs-ci.yml` gained the evidence-gate steps, so every anchor moved, and the ratchet gained a second rule. Redrawn at the pin, with F6 as Open. |
| ES-09.13 | accurate-with-limits | Every `ontology-publish.yml` anchor holds at the pin (file unchanged). Limit: release, dispatch and pull behaviour are workflow-runtime claims. |
| ES-09.14 | accurate | All `xr-godot-ci.yml` anchors hold. |
| ES-09.15 | corrected | Added the evidence-gate self-test (`:212-213`); the ratchet is `:215-218` and licensing `:221`. The byte-identical claim was re-checked by hash and qualified (F6). The successor note now names `ontomark validate` (`:186-189`). The ADR-2153 prose moved to `flake.nix:2912`. |
| ES-09.16 | accurate | Contract and custody job anchors hold (`contract-tests.yml:116-274`), including the role-secret step `:161-163`. |
| ES-09.17 | accurate | `manifest-validate.yml` anchors hold (`:18-208`). |
| ES-09.18 | corrected | agentbox's stale `ontology-publish.yml` is gone (F9). Named the new `adr-checker.yml` and `security-vex.yml`. Aggregate gate anchors hold. |
| ES-09.19 | corrected | The production container is not supervisord-managed on the compose path (F1). |
| ES-09.20 | corrected | `include_str!` is at `src/uri/mod.rs:665`, and the old `:662` was a doc comment. The CB anchor is now the build line `rust-backend-wrapper.sh:66`. The mount and comment anchors hold. |
| ES-09.21 | accurate-with-limits | `launch.sh` anchors hold, and `multi-agent-docker/` is absent at `86cdd76` while `env.example` exists. Limit: the unreachable branch is inferred, not run. |
| ES-09.22 | corrected | Catalogue anchors moved: `vault-cli` `:406`, `sidechain` `:385-386`, `stateOf` `:456-458`. Nineteen System One consumers, sixteen boot and three rebuild, verified by count. The CLASS node wrongly said every entry was boot-class; corrected. |
| ES-09.23 | corrected | Re-anchored to `adr-index-gen.js:186-218`. The "soft nudge" claim was false (F7, Drift). The fifteen-failure Debt is kept as historical (not re-run). Added the ratchet's two rules. |
| ES-09.24 | accurate | Every workflow anchor and the prose anchors hold. No workflow at `c57d9c05` names `g5-key-split` or `sidechain-producer-baked`. |

## Verification

**Evidence class:** source tracing at the pinned revisions only, with no runtime, test or deployment evidence. I ran no product tests, no `cargo`, and no `adr-index-gen.js` against product ledgers. Files were read with `git show <sha>:<path>` and `git grep`. Where a citation moved, its line was re-derived from the symbol, never by offset, and I read the cited content for every citation the generator resolves. A scratchpad copy of the generator, patched only to dump its resolved citations, produced worksheets of pinned line content. Its first pass found 16 changed anchors in ES-08 and 45 in ES-09 against the old stamps. Several unchanged anchors were also wrong at the old stamps (ES-08.2, 08.3, 08.6, 08.9) and were corrected.

**Pinned revisions.** Both topics' `verified_commit` maps are restamped to these:

| Repository | Revision |
|---|---|
| visionclaw | `86cdd76355823a51bb64d503c005fe277726e724` |
| agentbox (read through `../project/agentbox`) | `c57d9c0508f4044915f9aecea4b9aa5629cc6a35` |
| solid-pod-rs | `9b6143ce55b242c322c92d8c6bcc0c1a5371b451` |
| nostr-rust-forum | `c39f41f104b114d6a3fa29013daaf6fa58b45080` |
| visionflow | `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083` (worktree HEAD) |

ES-09 cites only visionclaw and agentbox. The `v0.5.0-alpha.17` tag of solid-pod-rs and agentbox's `76ab7aab` Nix pin were read to check F2 and F8.

**Gate commands**, run from the worktree root after the edits:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 08-solid-pod-estate.md --check --cite-check --strict-citations --render
# exit 0 — 207 citations verified at the declared revision, 0 unverified, 0 unresolvable; 13/13 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 09-build-deploy-and-ci-estate.md --check --cite-check --strict-citations --render
# exit 0 — 414 citations verified at the declared revision, 0 unverified, 0 unresolvable; 24/24 rendered
```

Before the edits, both topics also passed at their old stamps (109 and 399 citations). That pass checked line existence, not truth.

**Not verified:**
- Prose citations in Tension, Open, Drift and Debt paragraphs are not checked by the gate. I resolved them by hand.
- No running pod or container was inspected. The ADR-2019 "mirror not running" observation is the close-out's.
- F1, F2, F3 and F4 were not reproduced.
- The fifteen-record ADR staleness count (ES-09.23) was not recomputed.
- ES-08.13's server line numbers are the solid-pod-rs pin's, not the image build's.
- The register markers for F3, F4, F5, F8 and the ES-09.5 built-image gap are also prose `**Open:**`, `**Debt:**` and `**Drift:**` lines under their panels. The gate counts only Mermaid citations, so I resolved those citations by hand, and the counts above do not change.
- No topic `sources:` entry names a generated or uncommitted file, so there is no generated-path Drift in these topics.
