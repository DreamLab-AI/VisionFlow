---
title: Forum config, CI and invariants topics
date: 2026-11-02
status: assessed
type: explanation
---

# Forum config, CI and invariants topics

This review covers all 28 panels in three nostr-rust-forum diagram topics: NF-08 (config, zones, stores and admin authority), NF-09 (CI, deploy, canary and validation) and NF-10 (invariants and the anomaly register). Every cited line was read at nostr-rust-forum `c39f41f104b1`. Result: 11 accurate, 15 corrected, 2 accurate-with-limits, none inaccurate-unfixed. All three topics are restamped to `c39f41f104b114d6a3fa29013daaf6fa58b45080`. The most important finding is that the supply-chain gate has already drifted. `scripts/security-audit.sh` ignores two advisories that `deny.toml` does not, and the topic called the two lists "same exceptions". Also: the drift-gates job runs on every CI run, though its comment says otherwise; the JS-vs-WASM benchmark cannot resolve its WASM module as committed; and two topic citations claimed tests or code paths that the cited lines do not hold.

The topics were stamped at `1b27f79`. Every file they cite is byte-identical between `1b27f79` and `c39f41f`, except `scripts/adr-index-gen.js` (`git diff --stat 1b27f79 c39f41f104b1`), whose cited header line still holds at `c39f41f`. So the defects below were already present at the earlier stamp. The pin bump did not introduce them.

## Findings

**F1. The advisory ignore lists have diverged (Drift, NF-09.2).** `scripts/security-audit.sh:10` and `:12` ignore RUSTSEC-2026-0097 and RUSTSEC-2026-0221. `deny.toml:12-19` ignores only the other four. Both advisories are reviewed exceptions in `docs/security/advisory-exceptions.md:13-14`. The weekly `cargo audit` therefore suppresses advisories that the required `cargo deny check` job (`.github/workflows/ci.yml:258`, in `ci-pass` at `:314`) does not.
- Consequence: the two gates can disagree on the same lockfile. Source alone cannot tell whether `cargo deny` currently trips on these two; that depends on the live advisory database.
- What closes it: one ignore list, or a CI check that the two lists and the exceptions table name the same IDs.
- Topic change: the panel's "same exceptions" edge is now "meant to mirror", and N2 records the divergence.

**F2. drift-gates runs on every CI run (Drift, NF-09.1).** The comment says the job runs "only when its scripts or fixtures change" (`.github/workflows/ci.yml:272-273`). But the job has no `if:` (`:275-278`), and the workflow's path filters are workflow-wide and include `**/*.rs` (`:26`).
- Consequence: it runs on effectively every CI run. It is still outside `ci-pass` (`:306-314`), so a failure does not fail the `CI required` aggregate. Branch protection is configured outside the repository and was not checked.
- What closes it: correct the comment, or add a job-level path condition.

**F3. Config validation skips edits to the example file (Debt, NF-09.1).** The required `test` job validates `forum.example.toml` (`ci.yml:121`). That file is in neither the push nor the PR path filter (`ci.yml:23-36`, `:39-52`).
- Consequence: a commit that changes only the example config runs no CI at all.
- What closes it: add `forum.example.toml` to both filters.

**F4. `[deployment] profile` is silently dropped (Drift, NF-09.4).** The `Deployment` struct has only `name` and `hostname` (`crates/nostr-bbs-config/src/schema.rs:100-105`), so serde discards `profile`. Yet both `forum.example.toml:45` and `SETUP.md:53-54` say the profile "is projected to the relay as `CHANNEL_DIALECT`", and no crate source names that variable. ADR-2029 is proposed, with implementation `none` (`docs/adr/ADR-2029-room-profile-deployment-type.md:5-6`), and both documents say only `forum` deploys. So the gap is documented intent, not a broken feature. Still, the projection sentence describes code that does not exist.

**F5. The benchmark cannot resolve its WASM module (Debt, NF-09.7 and NF-10.4 O10).** `benchmarks/js-vs-wasm/bench.mjs:39` imports `@dreamlab/nostr-core-wasm`. Its `package.json:12` declares a different name, `@nostr-bbs/nostr-core-wasm`, at `file:../../crates/nostr-core/pkg`, a directory that does not exist (the crate is now `nostr-bbs-core`).
- This bench is the only in-repo JS consumer of the `wasm_bridge` exports (`crates/nostr-bbs-core/src/wasm_bridge.rs:228`), so O10's "no JS consumers" is refined, not confirmed.
- What closes it: fix the import name and path, or delete the bench.

**F6. The upstream-vector loader documents the wrong skip mechanism (Drift, NF-10.4 O10).** `tests/upstream_vectors/mod.rs:11` says tests are `#[ignore]` when fixtures are missing. No `#[ignore]` exists in those tests: `all_fixtures.rs:18` returns early with an `eprintln!`, so a missing fixture reports as a pass. The topic's "soft-skipped, not failed" is true in effect, but it cited the doc comment rather than the mechanism.

**F7. A device-gate test citation asserted the wrong property (NF-08.8 N2, NF-10.2 K3).** The exact-`"true"` invariant was cited to `devices.rs:974`, which asserts the binding *name*. The behaviour is asserted by `gate_variant_matrix_auth_worker` (`crates/nostr-bbs-auth-worker/src/devices.rs:912`), and the rule itself is `crates/nostr-bbs-core/src/feature_gate.rs:65`. Both topics now cite those lines.

**F8. The client and the workers parse the device-key flag differently (Tension, NF-08.8).** The forum client lower-cases the value and also accepts `"1"` (`crates/nostr-bbs-forum-client/src/utils/devices.rs:111`). The workers accept only the exact string `"true"`.
- Consequence: `DEVICE_KEYS_ENABLED = "1"` set consistently in all three places gives a client that attempts device-key registration while the auth worker returns 404 on the device routes (`devices.rs:905-907`) and the relay, which delegates to the same exact rule (`relay_do/nip_handlers.rs:971`), treats device keys as off.
- What closes it: make the client use the shared `feature_gate` rule.

**F9. SETUP credits the wrong worker with the governance tables (Drift, NF-08.6).** `SETUP.md:123-125` says the four governance tables are "also created inline by the auth-worker on startup". The auth bootstrap (`crates/nostr-bbs-auth-worker/src/schema.rs:19`) creates none of them. The inline copies are in the relay bootstrap (`crates/nostr-bbs-relay-worker/src/lib.rs:741`, `:1000`). The topic had cited SETUP as agreeing with it.

**F10. The ADR-2012 status was overstated (NF-08.4 N4).** The panel said "PROPOSED, NOT BUILT" and cited `:37-41` for the staleness bound. The frontmatter says proposed, implementation `partial`, activation `inactive` (`:5-7`). The staleness bound is at `:48-51`, and the record is parked at P3 (`:165`). The panel is corrected.

**F11. Line-level citation corrections.** No claim's substance changed:
- `zone_approval.rs:31` → `:30` (the function), `:32` → `:31` (the base cohort), `:36-41` → `:34-38`, and the members-only invariant `:45` → `:29`, tested at `:53-60`.
- relay `auth.rs:196` → `:195`; `whitelist.rs:451` → `:450`.
- `app.rs:867` (a channel-sync effect) → `:902` (`<Router base=FORUM_BASE>`); `channels.rs:444` (a comment) → `:453` (`insert_message`).
- `ci.yml:206` → `:208`; `audit.yml:17` → `:18`.
- auth `admin.rs:71` (a `return true`) → `:67`, the `ADMIN_PUBKEYS` read, in NF-08.7 and NF-10.6 DD3.
- `SETUP.md:119`, `:129`, `:138` and `:178` (code-fence lines) → `:120`, `:130`, `:139` and `:179`; `SETUP.md:275` (a comment) → `:276`.
- `dep-watchlist.sh:1` (the shebang) → `:2-3`.

**F12. Smaller over-claims, corrected:**
- NF-08.4 N1 said "every worker binds REPLAY_DB". The preview worker binds no replay store.
- NF-09.6 N2 said the journeys exercise "all five workers". They call relay, auth, pod and search (`e2e-forum-test.mjs:4-7`), never preview.
- NF-09.2 N1 still called solid-pod-rs an "exact crates.io pin". It is a caret range (`Cargo.toml:185`), and `allow-git = []` guarantees only the source.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| NF-08.1 | accurate | `forum.example.toml` section headers at :49–:377, placeholder admin key :86-91, commented `[[ingress]]` :165-185; `zone_config.rs:84` load and `:108` `from_json` deny-by-default. |
| NF-08.2 | accurate | `zone_config.rs:28-33` visibility; `:63`, `:133`, `:143`, `:152`, `:161`, `:177`; empty write set :182; unknown zone :163/:179; test :342-345. `ENCRYPTION_ENABLED` is in no `wrangler.toml` (confirmed by grep). |
| NF-08.3 | corrected | F11: `zone_approval.rs` lines; `username.rs:210`, `:307-308` hold. |
| NF-08.4 | corrected | F10, F12. All binding lines in the five `wrangler.toml` files hold; same KV id at auth :28/:44, guard test `admins.rs:372`. |
| NF-08.5 | accurate-with-limits | Every table line in `schema.rs` and relay `lib.rs` holds; latch at `lib.rs:195`, `:200`, `:222-225`. Limit: the retry-on-transient-D1-failure behaviour is a runtime property, read from source only. |
| NF-08.6 | corrected | F9; migration and constant lines, `events_schema.rs:66`, and the `let _ =` at `lib.rs:835` hold. |
| NF-08.7 | corrected | F11. The admin union at `admin.rs:57-103` and relay `query_is_admin` at `auth.rs:181-202` hold; `ADMIN_PUBKEYS` is declared only at search `wrangler.toml:41`. Footnote reworded so it no longer depends on the old stamp. |
| NF-08.8 | corrected | F7, F8; the relay delegates to the shared rule at `relay_do/nip_handlers.rs:971`; every relay `wrangler.toml` line holds (:11-14, :29, :37-39, :48-51, :54-61, :64-79, :86-90, :93); auth JEV lines :53-58. |
| NF-08.9 | accurate | `nip_handlers.rs:1450-1464` gate; `zone_config.rs:204-223` shape check; sealed drift exemption `:1727`. |
| NF-09.1 | corrected | F2, F3, F11; job lines `ci.yml:70-314` hold; the clippy allows at `transport.rs:236`, `:249`, `setup-skill lib.rs:71` and `mesh.rs:141` hold. |
| NF-09.2 | corrected | F1, F11, F12; `deny.toml` lines :12–:78 and `audit.yml` :9, :14, :59-66 hold. |
| NF-09.3 | accurate | Script header lines hold; parity runner HMAC at `identity-vector-parity.mjs:105`, absent-source pass at :199-200, substring check at :204. |
| NF-09.4 | corrected | F4; every `validate.rs` rule line (:6–:298) and `ingress.rs:94` holds. |
| NF-09.5 | corrected | F11 (SETUP lines). The step 0 "only forum deploys" claim holds at `SETUP.md:56-57`; the divergence note is sharpened (the auth worker bootstraps all twelve auth tables). |
| NF-09.6 | corrected | F12; all step lines hold. Limit: the suites target live production, which this review did not run. |
| NF-09.7 | corrected | F5; op and warm-up lines hold. Limit: N1's "performance half of ADR-2002" is the topic's interpretation, not a source fact. |
| NF-09.8 | accurate | `canary lib.rs:13-19`; `transport.rs:238`, `:251`, `:325`; the only `MeshSocket` impl is at `mock.rs:249`; relay dependency `Cargo.toml:27`; `mesh.rs:48` `allow(dead_code)`; `README.md:60`. |
| NF-09.9 | accurate | `setup-skill lib.rs` :11, :48, :75-82, :90-204, :111, :130. |
| NF-10.1 | corrected | F11 (`app.rs:902`, `channels.rs:453`); BASELINE invariant 6 at :200 against the caret range at `Cargo.toml:185` holds. |
| NF-10.2 | corrected | F7; `keys.rs:198`, `:225`, `:241`, `:254`, `:479`, `:484` and `trust.rs:334` hold; ADR-100 :78/:85 hold. |
| NF-10.3 | accurate | O1 `nip_handlers.rs:2248`; O3 `acl.rs:253`, `:311`; O5 `storage.rs:162`, `nip_handlers.rs:2548-2549`; O2 :1300/:1307; O11 lines; O8 `qr_svg` twice; settings subscription at :457/:479/:488-496. |
| NF-10.4 | corrected | F5, F6. O4, O6 (`dm/mod.rs:245`, `:935`, `:962`), O7, O9 and O11c (`governance_api.rs:619`) hold. |
| NF-10.6 | corrected | F11 (DD3 `admin.rs:67`). DD1 (README has no scaffolded or planned rows, by grep), DD4–DD7 hold; DD7: only `PanelDefinition`, `ActionRequest` and `ActionResponse` exist for the six kinds at `governance.rs:27-32`. DD8 holds and now cites `git_panel.rs:893`. |
| NF-10.7 | accurate | S1–S7 hold; preview has no timeout (grep); wallet gate reads `SIDESTR_WALLET` (`wallet/mod.rs:58-62`); `parent.rs:50-63`. |
| NF-10.8 | accurate | ADR-2006 accepted/complete/live; ADR-2010 proposed/partial/staged (:5-7) and :111; the IDENTITY doc still states the OFFSET defect at `IDENTITY-keys-and-trust.md:191` against `trust_sweep.rs:22`, `:286`, `:501`. |
| NF-10.10 | accurate-with-limits | `trust_sweep.rs:418-427` conditional `changes() = 1` audit; five exact-SQL tests in `scripts/tests/test_trust_transaction.py:34-58`. Limit: D1 batch-transaction semantics and the tests themselves were not run. |
| NF-10.11 | accurate | Every `known-findings.md` row line (:4-48) holds; KF-1 mechanism at `filter.rs:375`, `nip_handlers.rs:1070`, `cron.rs:419`; KF-3 against `lib.rs:200`, `:222` and auth `lib.rs:163`. |
| NF-10.12 | accurate | `governance.rs:1458`, `:1469`, `:1475`, `:1742-1749`, `:743-746`; `nip_handlers.rs:1354`. |

## Verification

Evidence class: source tracing at the pinned revision only. No product test, build, deployment or live endpoint was run. Every citation was read with `git -C /home/devuser/workspace/nostr-rust-forum show c39f41f104b1:<path>`. Absence claims were checked with `git grep … c39f41f104b1`.

Pinned revision: nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. No other repository's citations were re-read.

Gate commands, run from the worktree root after the edits. Exit codes were captured directly, without a pipe:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 08-config-zones-stores-and-admin-authority.md --check --cite-check --strict-citations --render   # exit 0; 225 citations verified; 9/9 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 09-ci-deploy-canary-and-validation.md --check --cite-check --strict-citations --render          # exit 0; 183 citations verified; 9/9 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 10-invariants-and-anomaly-register.md --check --cite-check --strict-citations --render           # exit 0; 153 citations verified; 10/10 rendered
```

Two JEV advisories remained on topic 10, and both were checked by hand and hold:
- `nostr-bbs-upstream-canary/src/lib.rs:17` is the Shape A verdict line that the I1 gate rests on.
- `tests/upstream_vectors/all_fixtures.rs:18` is the early-return skip.

Earlier advisories on `SETUP.md:119`, `dep-watchlist.sh:1` and `admin.rs:71` were real defects and are fixed (F11). All three topics were re-run after the final edit, with the exit codes above. The prose register markers (`**Drift:**`, `**Debt:**`, `**Tension:**`) were added alongside the corrected mermaid notes.

Not verified:
- whether `cargo deny check` currently fails on RUSTSEC-2026-0097 or RUSTSEC-2026-0221 (F1);
- D1 transaction and latch behaviour (NF-08.5, NF-10.10);
- the production e2e journeys (NF-09.6);
- the cross-repository EXTERNAL references (AB-*, VC-*, ES-*, DW-*), which point outside this repository.
