---
title: Solid Pod payments, CI and release panels
date: 2026-11-02
status: assessed
type: explanation
---

# Solid Pod payments, CI and release panels

I assessed all 19 panels of SP-07 and all nine panels of SP-09 by source tracing at solid-pod-rs `9b6143c`, agentbox `c57d9c0` and VisionClaw `86cdd76`. The five unreconciled SP-07 panels (.9, .14, .16, .17, .19) were traced in full. The other 14 were re-checked after the restamp, at the lead's request. Of the 28: 8 accurate, 18 corrected, 2 accurate-with-limits, 0 inaccurate-unfixed. Both topics are restamped and pass the strict gate.

The most important finding is in the deposit path. `verify_mrc20_anchor` walks only the `stateStrings` the depositor supplies, and checks only the last one. A one-string proof therefore reduces the alpha.12 "every link" walk to the head-only check it replaced. The head address of a one-state trail comes from the public issuer key alone, so funding it needs no issuer secret. Nothing in `POST /pay/.deposit` refuses such a proof, and the credited amount is whatever the self-authored state transfers to the pod. This is source tracing only: no request was sent and nothing was exploited.

The second finding concerns agentbox. Its Nix pin `76ab7aab` reports the same version, `0.5.0-alpha.18`, as the head, but predates the SP-GIT-02 mirror isolation and four other crate fixes, and agentbox compiles `forge-full` by default.

## Findings

### F1: a one-string `stateStrings` proof passes MRC20 deposit verification (security, open)

- **Trigger:** `POST /pay/.deposit` with an MRC20 body whose `anchor.pubkey` is the pod's issuer, a self-authored `state`/`prevState` pair, and `stateStrings` holding only `JCS(state)`.
- **Mechanism:**
  - `verify_mrc20_anchor` checks the state link and that the *last* string equals `JCS(state)` (`crates/solid-pod-rs/src/mrc20.rs:1061`). Nothing ties the earlier strings to `prevState`, to a seq-0 genesis, or to one string per state.
  - `verify_anchor_chain` never examines mark 0's input (`crates/solid-pod-rs/src/blocktrail.rs:1114`), so with k = 1 the walk only checks that the head output carries the derived key.
  - That key comes from the issuer's public key and the states alone (`mrc20.rs:477-490`, `:766`). Funding it needs no secret. Only a later mark needs the issuer's chained secret to spend its predecessor.
  - A head still in the mempool counts as intact (`blocktrail.rs:849`).
- **Deposit handler:** it adds the issuer/ticker binding, the state-hash and outpoint replay guards, and credits `total_transferred_to` (`crates/solid-pod-rs-server/src/handlers/pay.rs:549-660`, `mrc20.rs:253`). No balance or supply rule limits that sum. The only bound is `credit_by_outpoint`'s ceiling of 2.1 × 10^15 per credit (`crates/solid-pod-rs/src/payments.rs:469`, `:622-628`).
- **Network:** comes from the body and defaults to testnet4 (`pay.rs:566`). The default mempool endpoint is `https://mempool.space/testnet4` (`crates/solid-pod-rs-server/src/mempool.rs:61`). The configured endpoint, not the code, decides which chain is read.
- **Tests:** the route tests (`crates/solid-pod-rs-server/tests/pay_mrc20_routes.rs`) cover a missing UTXO, replay, a self-issued trail, a wrong ticker and a repeated outpoint. None covers a one-string or head-only proof.
- **Consequence:** the ticker balance can be credited without the issuer authoring any state after genesis. What that balance can then buy depends on configuration: `/pay/.withdraw`, the order book, or the AMM against a funded pool. That reach was not traced end to end. SP-07.17's INVARIANT ("a pod must not carry a reachable free-money oracle") holds for the deleted stand-in but is not shown for the remaining path.
- **Would close it:**
  - require at least two strings, with the second-to-last equal to `JCS(prevState)`, or bind the proof to the pod's recorded trail;
  - add a route test that refuses a one-string proof;
  - pass the pod's own chain instead of the body's `network`.
- **Recorded in:** SP-07.9 (Tension, N5), SP-07.17 (Tension), SP-07.19 (Tension).

### F2: the agentbox native pod is pinned before the head's crate fixes (debt, open)

- **Pin:** `lib/solid-pod-rs.nix:54-58` at agentbox `c57d9c0` pins `0.5.0-alpha.18` at rev `76ab7aab`.
- **Features:** `flake.nix:1548` adds `forge-full`, which includes `forge-mirror` (`crates/solid-pod-rs-server/Cargo.toml:170-172`), unless the manifest sets `forge.enable = false`.
- **The gap:** `76ab7aab` is an ancestor of `9b6143c`. The crate commits between them are SP-GIT-02 A/B1/B2 (`30135fe`, `bcaf9cf`, `ff8048b`), the hidden-object fetch refusal (`1a66c2d`), the NIP-98 empty-body binding (`8f2df86`), the OIDC redirect check (`fcc624a`) and the ACL-control fix (`daff10d`). All sit under `[Unreleased]` (`CHANGELOG.md:7`), and the workspace version is unchanged (`Cargo.toml:15`).
- **Consequence:** a native pod built from this pin lacks the publication isolation SP-14 records. ADR-2010 itself now says compiling `forge-full` does not prove held objects cannot be served (ADR-2010 `:62-72`).
- **Would close it:** cut a release past alpha.18 and re-pin agentbox to it.
- **Previous topic text:** it called `76ab7aab` "the last code commit before this topic's verified head". That was true at `09ec90f` but is false at the pin.
- **Recorded in:** SP-09.7 (AB node, Debt).

### F3: the README no longer cautions about `/pay/*`; the panel cited a sentence that does not exist (drift, corrected)

- **Old panel text:** SP-07.14 N2 said the README's status section warns against carrying value through these routes. That sentence is absent at both `09ec90f` and `9b6143c`; `git log -S "carry value"` shows it was removed in `a9ad0d2`.
- **What the README says now:** payments are added "through explicit features" (`README.md:3`), and a payments paragraph describes deposits, withdrawals, the order book and the AMM with no caution (`README.md:141`).
- **What the code does:** the routes are mounted unconditionally (`crates/solid-pod-rs-server/src/lib.rs:5282`).
- **Would close it:** a README caution matching ADR-2008's unfinished D4/D7 (SP-07.18), or a feature gate.

### F4: CI coverage claims in SP-09 were wrong or stale (corrected)

- **SP-09.10 N2:** it said the core-crate all-features rows compile forge, git and tls. The core crate has no such features (`crates/solid-pod-rs/Cargo.toml` `[features]`). Only `coverage` (`tarpaulin --all-features --workspace`, `ci.yml:227-230`) builds and tests them, and it is skipped on pull requests (`ci.yml:205`). Forge, git end-to-end and tls are first exercised after merge.
- **MSRV:** raised to 1.89 (`Cargo.toml:22`, `clippy.toml:9`, `ci.yml:108-127`). The msrv job now also runs `cargo check -p solid-pod-rs-forge --features mirror` (`ci.yml:127`), so the forge crate is now compiled on every run, which also qualifies SP-09.10 NG1.
- **SP-09.11 benches:** the panel said only the scheduled job runs the benches. In fact `build-test` runs `cargo test --all-targets` (`ci.yml:90`), and criterion 0.5.1 runs every bench once in test mode when cargo passes no `--bench` (criterion `src/lib.rs:963-967`, registry source). The scheduled smoke (`ci.yml:316`) passes no features, so it skips `dpop_replay_bench`, which requires `dpop-replay-cache` (`crates/solid-pod-rs/Cargo.toml:385`). Only the fuzz target is schedule-only.
- **SP-09.9:** the shipped-status list omitted three categories (`scripts/parity-check.sh:12-13`). Its 97.7% figure is stale: the checklist now says at most 86.4% (`PARITY-CHECKLIST.md:52`).
- **SP-09.7 CHANGELOG citations:** 352 lines were added under `[Unreleased]`, so every citation was re-derived by heading and text. REL10's `:263` never supported "re-pinned all eight crates"; that text is at `:833-834`. REL11 and REL12 claimed "all eight crates re-pinned", which their sections do not say, so I removed it.

### F5: CI and governance register items (new, open)

- **Path filter:** `ci.yml:6-17` limits push and pull-request runs to `crates/**`, `Cargo.toml`, `Cargo.lock` and `ci.yml`. A change to `deny.toml`, `.cargo/audit.toml` or `scripts/` alone runs nothing until the weekly schedule. Whether such a PR is then blocked by the required check is repository configuration.
- **Release dry run:** it packages only the core crate (`release.yml:76`, `:87`), so a sibling packaging error appears after manual approval, mid-loop. The release text is `git log` over `crates/solid-pod-rs` only (`release.yml:117`).
- **CODEOWNERS:** the security rules `/src/auth/`, `/src/oidc.rs` and `/src/wac.rs` match no path (`.github/CODEOWNERS:18-20`). The catch-all `*` covers the same team.
- **Dependabot:** both entries point at `/crates/solid-pod-rs` (`.github/dependabot.yml:6`, `:44`), and the workflows live at the root.
- **Comment drift:** the workspace-job comment lists six siblings, omitting forge (`ci.yml:248-251`). deny.toml points at `../../.cargo/audit.toml` (`deny.toml:38`).

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| SP-07.1 The two provenance tiers | accurate | All 7 citations land on their claim at `9b6143c` (`lib.rs:3917`, `:4091`, `:2277`; `provenance.rs:516-558`; `prov.rs:313`). `lib.rs` lines moved by content-identical remap only; the `lib.rs` hunks since `09ec90f` (async ACL lockout guard, forge `web::block`) do not touch marking or sidecars. |
| SP-07.2 git_mark_write — the skip ladder | accurate | All 13 citations land on their branch (`lib.rs:3917-3957`, no-op shim `:4123`; `ProvenanceSkip` variants `provenance.rs:624-639`). Remap only. |
| SP-07.3 Anchor policy resolution | accurate | All 15 hold (`prov.rs:90-212`, `lib.rs:3980`, `:4000`, `:4003`, `:4061`, `prov.rs:501`). The Open holds: every `.genesis(` call in server source is inside `mod tests` (`mempool.rs:1502` onwards). |
| SP-07.4 record_receipt — typed, per-tier outcomes | accurate | All 10 hold in `provenance.rs` (`:516-549`, `stage()` `:715-726`); file unchanged since `09ec90f`. |
| SP-07.5 The receipt on the wire | accurate | All 9 hold (`lib.rs:3917`, `:4023`, `:4063`, `:4079`, `:4091`, `:4105`; `provenance.rs:743`, `:753`, `:774`). Remap only. |
| SP-07.6 Provenance queries and explicit paid anchors | accurate | All 11 hold in `prov.rs` (`register` `:747`, resolve `:351-413`, anchor `:454-584`, `retain_refund` `:692`); file unchanged. |
| SP-07.7 Epoch batching | accurate | All 13 hold (`prov.rs:264-328`, `trail_store.rs:340`); both files unchanged. |
| SP-07.9 MRC20 state chain and Bitcoin anchoring | corrected | All 15 citations hold at `9b6143c` (`mrc20.rs:34-1034`, `blocktrail.rs:1068`). Added to N: the hash is over the *re-serialised typed* state, so undeclared fields are dropped (`mrc20.rs:218`). Added N5 and a Tension: the walk covers only the strings the caller supplies, and only the last is checked (`mrc20.rs:1061`); see F1. Limit: mempool answers are runtime. |
| SP-07.10 The rust-bitcoin transaction builder | corrected | 11 of 12 hold (`bitcoin_tx.rs:281-372`, `:1244`, `:1417`, `:53`). The Why sentence cited a blank doc line (`:229`); now `:230`, where the doc says every input is signed with one key. |
| SP-07.11 Mempool endpoint selection | accurate | All 9 hold (`lib.rs:417`, `main.rs:327`, `mempool.rs:87-456`). The startup-log Drift is a source reading; no process was run. |
| SP-07.12 The block anchorer | corrected | 11 of 13 land exactly (`mempool.rs:755-835`, `:1129`, `:1147`). "No block height" now cites `blockheight: None` (`:847`, was `:848`), and "is_intact permits unconfirmed marks" cites the `is_intact` call (`:856`, was `:855`). |
| SP-07.13 The stored web ledger | corrected | The sats-alias Invariant cited a blank line (`pay.rs:602`); the guard is `is_sats_unit` at `:591`. Crate-private `debit` is `payments.rs:579`, not `:580`. The VERIFY node now cites the credit (`pay.rs:655`), not the commit (`:672`). Added a pointer to F1: the handler verifies before crediting, but a one-string trail passes. Other citations hold (`payments.rs:372-565`, `:911`, `:913`). |
| SP-07.14 The /pay/* route family | corrected | All 11 handler citations and `register` (`pay.rs:1751-1763`) hold. N2's README claim was false (F3) and is replaced by a Drift. `lib.rs:5262` is now `:5282`. N was refined: `require_did*` returns 401 without NIP-98, and only dev-bearer builds accept a Bearer token (`lib.rs:611`). `is_valid_did_nostr` gates only the `/pay/.address` query (`pay.rs:776`). Accounts narrow through `account_of` (`payments.rs:458`, `:507`, `:601`). |
| SP-07.15 Payment-state commits and the read-failure gap | corrected | 9 of 10 hold (`lib.rs:208`, `pay.rs:169-210`, `:242`, `:1253`). The fallback reading is confirmed: `read_state_unchecked` takes the legacy path on any `get` error (`pay.rs:180`, `:184`). The INTENT node now cites the start of the debit-plus-intent comment (`:1320`, was `:1322`). |
| SP-07.16 Order book and AMM | corrected | All 17 `trading.rs` citations hold. There are no floats in `trading.rs`. The note said a balance is raised only by `credit_by_outpoint`. In fact public `swap`, `remove_liquidity` and `execute_swap` move value through the crate-private helpers, and public `refund` (`payments.rs:557`) and `reverse_payout` (`:536`) restore it. Note corrected. |
| SP-07.17 Deposits are MRC20-only | corrected | Confirmed: the 501 path (`pay.rs:470`), the `body_is_mrc20` sniff (`:483`), and stand-in deletion at `e62d028` (`git show --stat`: pay.rs, lib.rs, main.rs). The issuer note was wrong: the path uses `token.accepts_issuer`, which also admits `accepted_issuers` (`pay.rs:549`, `payments.rs:677`). `pod_issuer_pubkey` serves `/pay/.address`. "Never pay twice" is now scoped to one outpoint under the process-local lock (`pay.rs:464`). A Tension records F1. |
| SP-07.18 ADR-2008 — completed port and unfinished ledger authority | corrected | The MRC node cited `commit_state` (`pay.rs:672`); it now cites `credit_by_outpoint` (`:655`). Other citations hold, including ADR-2008 `:5`, `:55`, `:76`, `:147`. The D8 Open is confirmed at the wave pins: VisionClaw `86cdd76` `Cargo.lock` resolves `0.5.0-alpha.17`, the forum `1b27f79` resolves `0.5.0-alpha.18`. |
| SP-07.19 Whole-trail verification | corrected | All 13 citations hold (`mempool.rs:853-913`, `blocktrail.rs:647-1136`). Both anchorers call `verify_anchor_report` (`mempool.rs:853`, `:1491`). The BrokenLink note was wrong for this walk: links are `Some(true)` whenever a mark has inputs (`blocktrail.rs:1137`), so a mark that does not spend its predecessor fails as WrongKey/NotFound at the predecessor's index, and BrokenLink arises only with no inputs (`:1116`). The stricter-than-reference point belongs to `verify_blocktrail` (`:1031-1034`). `is_intact` accepts any mempool marks, not only the head. Tension added for F1. |
| SP-07.20 GitmarkAnchorer | corrected | Three citations sat beside their claim and now land on it: the trail lock `mempool.rs:1410` (was `:1409`), `gitmark_advance` `:1478` (was `:1480`), `save_pending_mark` `:1343` (was `:1340`). Others hold (`:1255-1375`, `:1011`, `trail_store.rs:354`, `bitcoin_tx.rs:963`). The Open holds (genesis callers are test-only). Limit: the altered-metadata Drift rests on an earlier audit probe I did not re-run. |
| SP-09.2 CI jobs and ci-required | corrected | Job lines re-derived (msrv `:107` now 1.89, wasm `:136`, deny `:163`, audit `:182`, coverage `:203`, workspace `:255`, diagrams `:281`, fuzz `:296`, ci-required `:321`, `if: always()` `:333`, assertion `:345`, coverage `if` `:205`, skipped accepted `:351`). The staleness tree is `crates/solid-pod-rs/docs/diagrams/src` (`check-diagram-staleness.sh:27`), not a root `docs/`. Added: path-filter note (`:6-17`), member-count Drift and an Open on branch protection. |
| SP-09.3 The build matrix | corrected | All citations unchanged and true (`ci.yml:47-81`). "Nightly-track" corrected to beta channel. Open added: whether a failed `continue-on-error` leg reports success to `needs.build-test.result` (`:345`) is runner behaviour. |
| SP-09.4 Supply-chain and quality gates | corrected | MSRV 1.89, plus the forge mirror step (`ci.yml:123`, `:127`). Lines re-derived: wasm `:152`, deny `:171`, audit `:195`, `:192-193`, tarpaulin `:227`, Codecov `:235`, `:240-244`, workspace `:255`, diagrams `:289`. deny.toml `:47,56,63,109` and audit.toml agree. README `:175`, `:178` hold. The mrc20 note holds (`Cargo.toml:213-215`). Added the MSRV lockstep note and the `../../.cargo` path Drift. |
| SP-09.6 The release pipeline | accurate-with-limits | `release.yml` is unchanged since `09ec90f`. Every step and citation holds (`:5-259`), and the crate manifests have no `^version =` line, so the workspace fallback fires (`:49-55`). Limit: manual approval depends on environment protection rules (`:183-184`). Tension and Drift added for the core-only dry run and the core-only release text. |
| SP-09.7 Version alignment and consumer pins | corrected | CHANGELOG citations re-derived at `9b6143c` (alpha.18 `:361` … alpha.2 `:1203`; alpha.6 still absent). REL10/11/12 re-pin text fixed. VisionClaw lines moved +1 (`Cargo.toml:225-226`, `:246-249`). The forum cite is now explicitly `Cargo.toml:181-184`, not `Cargo.lock` (the earlier JEV flag). agentbox: forge-full is from `flake.nix:1548`, not the comment at `:18`. The AB node is rewritten for F2. Deployment claims (running container, edge kit `9c36b60`) moved to an Open. An `[Unreleased]` node was added. |
| SP-09.9 Governance and maintenance | corrected | CODEOWNERS, dependabot, deny.toml and the scripts are cited true at `:1`/`:7`/`:28`. The parity note is corrected (F4). Added CODEOWNERS and dependabot Debt items, and a Drift for SP-10 R6, which belongs to another topic and was not edited. |
| SP-09.10 What CI does not gate | corrected | N2 was false (F4). NG1 is updated for the msrv forge check (`ci.yml:127`). NG2 and NG3 are scoped to pull requests, because coverage builds them post-merge (`:227-230`). `with-git-binary` (`solid-pod-rs-git/Cargo.toml:48`) and `tls` (`solid-pod-rs-server/Cargo.toml:136`) hold. |
| SP-09.11 Benches and the fuzz target | corrected | Bench function lines hold (`storage:48`, `wac:77`, `ldp:33`, `nip98:100`, `dpop:49`, `dpop:4`), and `[[bench]]` is at `Cargo.toml:366`. The heading and N2 were rewritten for F4: benches run on every build-test row, the smoke skips dpop, and only fuzz is schedule-only. |
| SP-09.12 SPARQL-Update fuzz contract | accurate-with-limits | `sparql_update.rs:6-26` matches. `SPARQL_UPDATE_MAX_BYTES` is 1 MiB at `ldp.rs:1270` (that file changed, but not that line). The benchmarks.md "Four" drift still holds (`:3`, `:11-14`). Limit added: with no seed corpus and no `-max_len`, the over-cap ASSERT branch is probably not reached by the 60 s run. |

How the 14 re-checked SP-07 panels were assessed:
- `git diff --stat 09ec90f 9b6143c` over all 16 frontmatter sources shows only `README.md` (lines 11 and 171, neither cited) and `lib.rs`. The `lib.rs` hunks make the ACL lockout guard async and move forge service construction onto the blocking pool. Neither touches provenance, anchoring or payment logic.
- All 161 citations in the 14 panels point to byte-identical text at `09ec90f` (the pre-restamp line) and `9b6143c` (the remapped line).
- I then printed each node's claim beside the line it cites and read the context wherever the two did not match. That pass found the eleven citations corrected above. All of them predate the restamp, and none changed what a panel claims.
- Because 07.1, .2, .3 and .5 had prior credits, the lead should rule on whether my dispositions supersede them.

## Verification

- **Evidence class:** source tracing at pinned revisions. Pins:
  - solid-pod-rs `9b6143ce55b242c322c92d8c6bcc0c1a5371b451`
  - agentbox `c57d9c0508f4044915f9aecea4b9aa5629cc6a35`
  - VisionClaw `86cdd76355823a51bb64d503c005fe277726e724`
  - nostr-rust-forum has no wave pin. I kept it at `1b27f79d08537b61b3d5a5d02cac9ee5999b79ed`; its `Cargo.toml` and `Cargo.lock` are unchanged to its HEAD `c39f41f`.
- **How sources were read:** product files with `git show <sha>:<path>` and `git grep <sha>` only. The criterion test-mode logic was read from the local cargo registry copy of criterion 0.5.1, the version `Cargo.lock` resolves.
- **Topic edits:** `docs/diagrams/solid-pod-rs/07-provenance-and-payments.md` (`verified_commit` → `9b6143c`; 21 `lib.rs` citations remapped; eleven panels corrected, eight confirmed accurate; `README.md` added to sources). `docs/diagrams/solid-pod-rs/09-ci-release-and-consumer-pins.md` (all four repository stamps set; every citation re-derived; `flake.nix`, `clippy.toml`, `PARITY-CHECKLIST.md` and ADR-2010 added to sources).
- **Gate commands** (run from the worktree root):
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 07-provenance-and-payments.md --check --cite-check --strict-citations --render`: exit 0, 232 citations verified at the declared revision, 19/19 rendered.
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 09-ci-release-and-consumer-pins.md --check --cite-check --strict-citations --render`: exit 0, 133 citations verified, 9/9 rendered, no JEV advisories.
  - An earlier pre-edit run of the 09 command failed with 5 "source path does not exist" errors and 8 "could not be read" citation errors (exit code not captured), because `../project` and `../nostr-rust-forum` were then absent beside the nested worktree. I also ran 09 through a scratch layout of symlinks, which gave exit 0 as well. The final runs above are the literal commands.
- **Not verified:** no product tests, probes or requests were run; F1 is unexploited. No GitHub Actions run, branch-protection setting, environment reviewer rule or Dependabot behaviour was observed. No deployment state was checked: agentbox image activation, the forum edge, and whether any consumer enables the forge. For the 14 re-checked SP-07 panels, claims that depend on earlier runtime probes (SP-07.20's altered-metadata recovery) were not re-run.
