---
title: Forum workers, auth, relay admission and pod worker
date: 2026-11-02
status: assessed
type: explanation
---

# Forum workers, auth, relay admission and pod worker

This review covers all 39 panels in NF-01 to NF-04 (6, 9, 14 and 10), traced against the source of nostr-rust-forum `c39f41f104b1`. 18 panels are accurate, 18 needed correcting, 3 are accurate with limits and none are inaccurate-unfixed. All four topics were restamped from `1b27f79` to `c39f41f104b114d6a3fa29013daaf6fa58b45080`. No file named in any of the four topics' `sources:` changed between the two revisions, so every correction was already wrong at `1b27f79`; none of them comes from drift. The most important source finding is new. When D1 returns no write metadata, the shared NIP-98 replay store reports a token as first-seen, so a valid token could be replayed within its freshness window on that path (`crates/nostr-bbs-rate-limit/src/replay.rs:37-39`). A second new finding is that a `.meta` sidecar's Control check is resolved for the sidecar path, not for the resource it protects (F13). The most important diagram error was NF-01.1's dependency graph, which drew three crate edges that do not exist.

## Findings

**F1. Replay store fails open when D1 returns no metadata (Open, product).**
- **Trigger:** `D1ReplayStore::seen_or_record` returns `Ok(true)` when `result.meta()` is not `Ok(Some(..))` (`crates/nostr-bbs-rate-limit/src/replay.rs:37-39`). The trait contract says `true` means first observation (`crates/nostr-bbs-core/src/nip98.rs:94-95`), and the core verifier only rejects when `first_seen` is false (`crates/nostr-bbs-core/src/nip98.rs:585-591`).
- **Consequence:** the full signature, URL, method, payload and freshness check still runs first (`crates/nostr-bbs-core/src/nip98.rs:576`). So the exposure is bounded: a valid captured token could be replayed within its freshness window, and only on a request where D1 omits the metadata.
- **Evidence limit:** whether D1 ever omits it is runtime behaviour. This review cannot establish that.
- **To close:** return `Err` (or `Ok(false)`) on missing metadata, with a unit test on the fallback arm. Recorded as an **Open** paragraph after NF-02.5 and in the panel's notes.

**F2. NF-01.1 drew dependency edges that do not exist (corrected).**
- **Wrong edges:**
  - `PREV --> CORE`: preview-worker links only rate-limit and ascii (`crates/nostr-bbs-preview-worker/Cargo.toml:24-25`).
  - `CFG --> FC`: forum-client links core and poker (`crates/nostr-bbs-forum-client/Cargo.toml:24`, `:27`).
  - `ASCII --> BBS`: bbs-client links config and core (`crates/nostr-bbs-bbs-client/Cargo.toml:26`, `:31`). The workspace comment still says ascii's HTML emit is "consumed by the retro BBS client" (`Cargo.toml:20-21`).
- **Missing edges:** RATE to POD, SEARCH and PREV; JEV to AUTH and SEARCH; ASCII to POD; CFG to BBS and MESH; MESH to CORE; RATE to CORE; poker to FC.
- **Wrong citations:** forum-client and bbs-client were cited at `Cargo.toml:32` and `:35`, which are poker comments. The correct lines are `:43` and `:46`.
- **Fix:** edges and citations corrected, plus a new N5 note citing every crate manifest.

**F3. NF-02.1 had the wrong entry order (corrected).** The panel drew the rate limit before the CORS preflight. The source order is:
1. schema bootstrap (`lib.rs:163`);
2. replay schema (`lib.rs:164`);
3. preflight (`lib.rs:167`);
4. rate limit (`lib.rs:175`).

So preflights are never rate-limited.

**F4. NF-02.8 put an invariant on the wrong mechanism (corrected).**
- **What the panel said:** it cited `invites.rs:217` for "pubkey-uniqueness on username_reservations, claimed via upsert-with-DO-NOTHING".
- **What that comment actually covers:** the ensure-then-merge zone-cohort grant on the relay whitelist (`invites.rs:213-230`).
- **Where uniqueness really comes from:** a single INSERT that relies on `UNIQUE(username)` and `UNIQUE(pubkey)` (`username.rs:231`, `schema.rs:211`). It is not an upsert.
- **Missing write:** the redeem path's write to the relay whitelist (`invites.rs:230`, `:245`, `:271`) was absent from the panel.
- **Fix:** both invariants are now stated separately and the write is added.

**F5. NF-03.1 showed the cron passing through the schema bootstrap (corrected).** The code comment says the opposite: neither `scheduled` nor the Durable Object bootstraps the schema (`crates/nostr-bbs-relay-worker/src/lib.rs:218-221`). `scheduled` goes straight to the demotion sweep (`lib.rs:1157`). The edge was removed and a note added.

**F6. NF-03.3 overstated when the NIP-42 write gate runs (corrected).** It said every EVENT passes `write_auth_ok` "before any other check". The rate limit, shape, expiry and signature checks all run first (`nip_handlers.rs:1057`, `:1063`, `:1070`, `:1077`, then `:1102`), as NF-03.4 itself shows. The kindless-filter assertion was cited as the whole test (`nip42.rs:383-393`). It is now cited at its actual lines (`nip42.rs:392-393`).

**F7. The pod-worker git guard citation pointed at a Content-Length read (corrected in NF-04.1 and NF-04.2).** `lib.rs:784` reads Content-Length. The guard is at `lib.rs:759` (`.git` is forbidden) and `lib.rs:766` (git requests are not implemented), before NIP-98 at `lib.rs:812`.

**F8. NF-04.3's sidecar coercion cited the wrong line, and the panel left out the second call path (corrected).**
- **Wrong citation:** `acl.rs:46` is the `ALL_MODES` constant. The coercion is at `acl.rs:88-89`.
- **Missing call path:** a path ending `.acl` (`lib.rs:118`) is dispatched to `handle_acl_request` (`lib.rs:992`), which calls `effective_acl_target` directly (`lib.rs:1648-1649`). The `coerce_required_mode_for_acl` call at `lib.rs:1015` is what elevates a `.meta` sidecar.
- **Fix:** both paths are now drawn. NF-04.2 marks where `.acl` leaves the standard path.

**F9. NF-04.5 overstated who can write a delegation (corrected).** The panel said only the owner can PUT a delegation. In fact any `acl:Control` holder on the container can (`lib.rs:1721-1725`; comment at `lib.rs:1755`). The worker-side calls are now cited (`lib.rs:1769`, `:1815`, `:1833`). The panel now also says that `preserves_owner_control` guards raw JSON-LD replacements as well as delegations.

**F10. Smaller citation retargets (corrected).**
- **NF-02.3:**
  - `webauthn.rs:933` → `:932`, the fmt condition;
  - `:906` → `:908`, the `pubkey = ?3` SQL;
  - `:1021` → `:1022`, the salt derivation.

  The first and third are the two lines JEV flagged as advisories in the baseline run; the second it flagged after the first edit.
- **NF-02.6:** `admin.rs:112` and `:145` are parameter lines; the functions start at `:110` and `:143`.
- **NF-02.5:**
  - the verifier is now cited in `replay.rs:77`, not at the `pub use`;
  - verify-then-record is now cited at `nip98.rs:576` and `:585`.
- **NF-03.1:** `lib.rs:257` → `lib.rs:256-267`, the JSON error arm.
- **NF-03.11:** sealed originals are exempt only from the past drift bound (`nip_handlers.rs:1723-1728`).
- **NF-04.10:** the BASELINE divergence notes now cite `docs/BASELINE-architecture.md:96-102`.
- **NF-04.11:** ADR-2012 status is now cited at `:5-6`, not `:6-7`.
- **NF-01 Open:** `BASELINE-architecture.md:271` is the push/VAPID line; "room (proposed, not implemented)" is at `:279`.
- **NF-01.6:** `README.md:72` → `:70`.

**F13. A `.meta` sidecar's Control check targets the sidecar, not the protected resource (Open and Drift, recorded after NF-04.3).**
- **Trigger:** only paths ending `.acl` reach `handle_acl_request` (`lib.rs:118-119`, `:992`), which checks Control on the protected resource (`lib.rs:1648-1650`). A `.meta` path stays on the standard path. There, `coerce_required_mode_for_acl` keeps only the mode (`acl.rs:89`), and the ACL is resolved and evaluated for the `.meta` path itself (`lib.rs:1016-1021`; its first probe is the `.meta.acl` sidecar, `acl.rs:200`).
- **Consequence:** where a resource has its own `.acl`, the Control answer for its `.meta` can differ from the answer for the resource. No test here establishes whether that matters in practice.
- **Drift:** the handler comment says it "covers `.meta` sidecars identically" (`lib.rs:1646-1647`), which is not true of the routing.
- **To close:** evaluate `.meta` against `effective_acl_target`'s protected resource on the standard path, with a test where a resource's own `.acl` and its container disagree.

**F14. NF-03.4 step 13 left out an exemption (corrected).** It said governance kinds need `agent_registry` "except a kanban approval request". A 31403 is also exempt (`nip_handlers.rs:1325`), because step 14 gates it instead.

**F11. Product comment drift (Drift, recorded in NF-02 after NF-02.10).** The compare-and-swap comment names the guard `stage = ?5` (`governance_api.rs:1202`). In the statement, `?5` is the event id and the stage guard is `?6` (`governance_api.rs:1210`). The behaviour matches the invariant; only the comment is stale.

**F12. Wording softened from runtime claims to their evidence class.**
- **NF-01.4:** the claim that "the edge still runs kit `9c36b60`" is now presented as carried, not re-checked.
- **NF-01.6:** VisionClaw is no longer called "the single live consumer". It is now the README's documented consumer, with universal use not established (`README.md:120`).
- **NF-02.2:** "O11 confirmed live" is now attributed to an earlier review.
- **NF-04.4:** "re-verified live" was removed.
- **NF-04.10, Tension paragraph:** ADR-2012 names an "exact pin at `Cargo.toml:155`"; that line is now a comment about the `core` feature, and the pin is a caret range at `Cargo.toml:185`. The quoted ADR line is no longer written as a citation.

**Tensions carried and still true at `c39f41f`.**
- **ADR-2007 and BASELINE invariant 6 against the caret pin (NF-01.4):** both still say exact pin (`ADR-2007…:30`, `BASELINE-architecture.md:200`); the manifest has a caret range (`Cargo.toml:181-185`).
- **ADR-2010 status against unconditional projections (NF-03.10):** ADR-2010 is `proposed/partial/staged` (`ADR-2010…:5-7`), yet the three projections run unconditionally (`nip_handlers.rs:1601`, `:1621`, `:1634`).
- **The inert `wasm-check-packages` metadata (NF-01.2):** `git grep` at the pinned sha finds the key only at `Cargo.toml:72`.

## Panel dispositions

| Panel | Disposition | Evidence |
|---|---|---|
| NF-01.1 | corrected | F2; member lines `Cargo.toml:5`–`:60`; edges from each crate manifest; Open line `BASELINE-architecture.md:279` |
| NF-01.2 | accurate | `rust-toolchain.toml:2-3`; `Cargo.toml:70-72`, `:203-213`; `ci.yml:185`, `:188`; all five `wrangler.toml:3` are `2025-09-01`, all five `:6` are `worker-build --release` |
| NF-01.3 | accurate | auth `lib.rs:144`, `:163-164`, `:873`; relay `lib.rs:200`, `:209-237`, `:286`, `:371-373`, `:1146`; `wrangler.toml:123`; search `wrangler.toml:44`; `admin_events.rs:5`; `profile_stats.rs:5-6` |
| NF-01.4 | accurate-with-limits | `Cargo.toml:78-105`, `:136`, `:181-200`; `Cargo.lock` resolves crypto-bigint 0.7.10 and k256 0.14.0; core `Cargo.toml:60`. Limit: the production kit revision is a deployment fact. |
| NF-01.5 | accurate | canary `lib.rs:10`, `:14`, `:17`, `:19`, `:35`, `:56`, `:89`, `:106`; canary `Cargo.toml:2` |
| NF-01.6 | corrected | F10, F12; `README.md:70`, `:120` |
| NF-02.1 | corrected | F3; auth `lib.rs:144-200` |
| NF-02.2 | accurate-with-limits | every route line `lib.rs:234`–`:656`; `did.rs:8`, `:18`, `:88-90`; `username.rs:812`, `:829`, `:844`; `crypto.rs:77`, `:97`; `rationale_check.rs:36`, `:38`, `:257`, `:303`; `wrangler.toml:22-25`, `:53-58`, `:66`; `admins.rs:372`. Limit: O11 "live" is not re-run. |
| NF-02.3 | corrected | F10; `webauthn.rs:723-1054` read in full order |
| NF-02.4 | accurate | `webauthn.rs:78`, `:1080-1427`; sign-count rule `:1348-1350` |
| NF-02.5 | corrected | F1, F10; `auth.rs:9-15`; `replay.rs:27`, `:37-39`, `:77`, `:94`; `nip98.rs:576`, `:585`; DB id `e3981999…` identical in auth `wrangler.toml:11` and the relay, pod and search `REPLAY_DB` entries |
| NF-02.6 | corrected | F10; every gate call `moderation.rs:209`, `:316`; `wot.rs:128`; `welcome.rs:148`; `admins.rs:161`; `governance_api.rs:362`, `:441`, `:561`, `:626`, `:670`, `:1130`, `:1407`; `invites.rs:291`, `:577`; `devices.rs:361`, `:452`, `:488`, `:641`; `username.rs:568`, `:689` |
| NF-02.7 | accurate | `devices.rs:100`, `:356-357` (404 before DB), `:383`, `:414-420` (upsert resets `revoked = 0`), `:511`; relay `nip_handlers.rs:2069`, `:2614-2619`, `:2216-2223`; 60 s TTL at `relay_do/mod.rs:105` |
| NF-02.8 | corrected | F4 |
| NF-02.10 | accurate | `governance_api.rs:1055`, `:1121-1232`, `:1401-1407`; Drift F11 added |
| NF-03.1 | corrected | F5, F10; relay `lib.rs:200-257`, `:1146-1157` |
| NF-03.2 | accurate | `nip42.rs:35`, `:144`, `:168-172`, `:176-216`, `:230-236`, `:332-343`; `nip_handlers.rs:2054`; `wrangler.toml:51` |
| NF-03.3 | corrected | F6; `nip42.rs:44`, `:115-116`, `:123-125`, `:248-249`, `:261`, `:282`, `:322-327`, `:392-393`; `README.md:130`; `wrangler.toml:39` |
| NF-03.4 | corrected | F14; gate spans read in full: `nip_handlers.rs:1049-1225`, `:1296-1412`, `:1414-1564`; `governance.rs:758-763`; `nip_handlers.rs:1703-1711`, `:1748-1760` |
| NF-03.5 | accurate | `nip_handlers.rs:56-75`, `:138-142`, `:1128-1150`; `ingress.rs:9-12`, `:27` |
| NF-03.6 | accurate | `nip_handlers.rs:56-75`, `:1213-1217` |
| NF-03.7 | accurate | `nip_handlers.rs:1226-1286`, `:1571-1573`, `:2457`, `:2534-2555`; `storage.rs:159-171`; `lib.rs:732`, `:827`; `migrations/0009_events.sql:44` |
| NF-03.8 | accurate | `trust.rs:28`, `:33`, `:76-85`, `:174-191`, `:233-238`, `:326-360` |
| NF-03.9 | accurate | `trust.rs:285`, `:308`, `:326`, `:406-419`; `trust_sweep.rs:280-282`, `:418-427` |
| NF-03.10 | accurate | `nip_handlers.rs:1546-1655` read in full (mirror needs kinds 30910/30911/30915/30916 and an admin, `:1587`), `:2228-2248`; auto-hide `moderation.rs:91`; `push_fanout.rs:47-48`, `:94`; `wrangler.toml:86`; `ADR-2010…:5-7` |
| NF-03.11 | corrected | F10; `nip_handlers.rs:41-45`, `:1662-1688`, `:1721-1731`, `:1796` |
| NF-03.12 | accurate | `nip_handlers.rs:1414-1526`, `:1748-1760`, `:2189-2198`; `zone_config.rs:133`, `:199-208` |
| NF-03.13 | accurate | `wrangler.toml:93-96`; `nip_handlers.rs:1173-1176`; `README.md:131`; relay `Cargo.toml:27`; `mesh.rs:25`, `:143` |
| NF-03.14 | accurate | `nip_handlers.rs:156`, `:1770-1914`, `:2114`, `:2156-2198`, `:2262-2287`; `nip42.rs:53-60`, `:85`, `:261`; `trust.rs:640-642` |
| NF-04.1 | corrected | F7; pod `lib.rs:94`, `:124`, `:129`, `:178`, `:304`, `:389`, `:463-730`, `:1045-1575`; `cors.rs:46` |
| NF-04.2 | corrected | F7, F8; `auth.rs:13`; `lib.rs:144`, `:207`, `:220`, `:370`, `:812`, `:971`, `:992`, `:1015-1029`, `:1116-1374`; `acl.rs:56`, `:285-311`; `quota.rs:110`; `wrangler.toml:25` |
| NF-04.3 | corrected | F8, F13; `acl.rs:34`, `:73-89`, `:200`; `lib.rs:118-119`, `:992`, `:1015-1021`, `:1646-1650` |
| NF-04.4 | corrected | F12; `acl.rs:93`, `:111`, `:161-167`, `:180-217`, `:253-273`, `:285-339`, `:360` |
| NF-04.5 | corrected | F9; `acl.rs:123`, `:394-423`; `lib.rs:1721-1833` |
| NF-04.7 | accurate | every function line in `content_negotiation.rs`, `conditional.rs`, `container.rs`, `patch.rs`, `notifications.rs`, `contexts.rs` and `quota.rs`; `lib.rs:1510` |
| NF-04.8 | accurate | `remote_storage.rs:10`, `:38`, `:58`, `:82`; `webid.rs:17`; `did.rs:7`, `:11`; `git.rs:44`, `:53`, `:58`, `:76`; `lib.rs:547` |
| NF-04.9 | accurate-with-limits | `payments.rs:34`, `:45`, `:157`; `pay_ledger.rs:130`, `:564`, `:618`, `:802`, `:1273`; `deposit_address.rs:120`; `wrangler.toml:31-32`. Limit: "production sets PAY_ENABLED true" rests on ADR-2012's amendment text (`:35-40`), not on source. |
| NF-04.10 | corrected | F10, F12; pod `Cargo.toml:22-25`, `:29`, `:56`; `export.rs:19`; `Cargo.toml:181-185`; `consumer-surface-map.md:52`; `wac_proptests.rs:1`; `BASELINE-architecture.md:96-102` |
| NF-04.11 | corrected | F10; `pay_ledger.rs:110`, `:130`, `:168`, `:412`, `:434`, `:564`, `:618`, `:1273`; `deposit_address.rs:120`; `ADR-2012…:5-6`, `:35-117`; `keys.rs:252-266`; `tests/pay_credit_guard.rs` exists; no `credit_atomic` or `D1PaymentStore` in `crates/`. The production claim in N2 carries the same limit as NF-04.9. |

NF-02 has no panel .9 and NF-04 has no panel .6; none was invented.

## Verification

- **Evidence class:** source tracing at nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`, read with `git show` and `git grep` at that sha. The alias `/home/devuser/workspace/wt/nostr-rust-forum` resolves to the same sha. No tests, builds, deployments or live endpoints were run.
- **Source diff:** `git diff --stat 1b27f79 c39f41f104b1 -- <sources>` is empty for every topic's `sources:` list, so the corrections above predate the restamp. As a positive control, the same command on `docs/adr/README.md` reports 1 file changed.
- **Gate commands:** run from the worktree root, with exit codes taken directly from `node`, not through a pipe:
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 01-workspace-and-worker-topology.md --check --cite-check --strict-citations --render` exits **0**: 111 citations verified, 6/6 rendered.
  - `… --only 02-auth-worker-passkey-nip98-and-membership.md …` exits **0**: 214 verified, 9/9 rendered. Two intermediate runs exited 1 on Mermaid parse errors caused by semicolons in notes, and one on an ambiguous `lib.rs:21`; all three were fixed.
  - `… --only 03-relay-worker-admission-auth-and-trust.md …` exits **0**: 251 verified, 14/14 rendered.
  - `… --only 04-pod-worker-ldp-wac-and-delegation.md …` exits **0**: 170 verified, 10/10 rendered.
  - Final runs after the last edits gave the same exits. Citations in prose register paragraphs are not counted by the generator; those lines were read by hand.
- **Baseline:** the same four commands on the untouched topics at `1b27f79` also passed. That pass caught none of F2–F9, because each wrong citation landed on a non-blank line.
- **Not verified:**
  - runtime behaviour of D1 metadata (F1), and whether the `.meta` target difference is reachable in practice (F13);
  - that the deployed kit, production `PAY_ENABLED`, and anomaly O11 match source;
  - Cloudflare batch atomicity for the demotion audit;
  - client-side passkey ceremonies;
  - any EXTERNAL cross-reference into agentbox, VisionClaw or solid-pod-rs.
