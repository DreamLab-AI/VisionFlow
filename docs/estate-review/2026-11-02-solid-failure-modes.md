---
title: Solid Pod failure modes and invariants
date: 2026-11-02
status: assessed
type: explanation
---

# Solid Pod failure modes and invariants

All 12 panels of [SP-10](../diagrams/solid-pod-rs/10-failure-modes-and-invariants.md) were assessed against solid-pod-rs `9b6143ce55b2`. All 12 are **corrected**: none was accurate as written, none is left inaccurate. The topic is restamped from `09ec90f` to the pinned revision, and every citation was re-resolved by symbol. Most corrections are line rebinding: server `lib.rs` shifted by up to 200 lines, and the CHANGELOG grew by 352 lines at its head. Seven corrections change what the topic claims:

- The baseline has eight invariants, not seven.
- Ten citations named code they did not hold, even at the old stamp.
- The 5xx logger is not the outermost middleware, contrary to its own comment.
- The NIP-98 freshness window is 120 s wide, not 60 s.
- ADR-2003 has caught up with the audience check.
- The parity figure is now 86.4% over 300 rows.
- The reviewer checklist omitted invariants 7 and 8.

The most important finding is the logger order (SP-FM-01). SP-02.8 repeats the same claim.

## Findings

**SP-FM-01: 5xx logging middleware is innermost, not outermost (Tension).**
- **Trigger:** a 5xx response produced by CORS, NormalizePath, PathTraversalGuard or DotfileGuard.
- **Consequence:** the response never reaches `log_5xx`. The comment at `crates/solid-pod-rs-server/src/lib.rs:5207` says registering ErrorLoggingMiddleware first (`lib.rs:5211`) makes it outermost. actix-web documents the reverse: the last-registered `wrap` runs outermost. Today's guards mostly emit 4xx, so the practical gap is narrow, but SP-02.8 and SP-10.9 both relied on the claim.
- **Evidence:** traced in source. actix-web 4.14.1, the version in `Cargo.lock`, documents in `src/middleware/mod.rs` (lines 18 and 116, in the registry copy) that middleware runs in the opposite order of registration, with the last registered outermost. No request was run.
- **Closes when:** the middleware is registered last, or the comment and SP-02.8 are corrected, with a test that a guard-level 5xx is logged.

**SP-FM-02: ten citations pointed at other code at the old stamp (Drift, repaired).**
- **Trigger:** the previous stamp `09ec90f`.
- **Consequence:** at that revision, the lockout-guard node cited a ledger-path doc comment (`lib.rs:822`). The PATCH node cited the lockout guard itself (`lib.rs:1866`), and the raced-debit node cited a read-path comment (`lib.rs:991`). The generator's existence check passed, so the gate never caught the mismatch. Seven more were wrong at `09ec90f` in the same way:
  - `lib.rs:3915`: the sidecar failure, actually at `:4105`
  - `lib.rs:2152`, cited twice: the policy failure, actually at `:2244`
  - `lib.rs:3556`: `format_error_chain`, actually at `:3755`
  - `lib.rs:2991`: the proxy cap, actually at `:3181`
  - `lib.rs:1518`: write-then-mark, actually at `:1747`/`:1754`
  - `CHANGELOG.md:201`: cited for the rotate note, but it held the Control entry
- **Evidence:** at the pin, the sites are `proposed_acl_keeps_caller_control` (`lib.rs:875`), the `debit_ledger` failure that returns `acl_denial` (`lib.rs:1064`), and `seed_graph_from_patch_target`'s 409 (`lib.rs:2183`).
- **Order fixed:** the chain now follows the PUT write path. PATCH's stored-body refusal (`lib.rs:1941`) branches off before the lockout guard (`lib.rs:1961`). Sidecar elevation (`guarded_acl_target`, `lib.rs:826`) runs before policy resolution. The storage-layer component check (`storage/fs.rs:81`) runs last, not first.
- **Closes when:** already closed in the topic.

**SP-FM-03: the baseline has eight invariants (Drift, repaired).**
- **Evidence:** invariant 8 is at `crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md:291`: the mirror is on in shipped builds and fails closed. It was already present at the old stamp. SP-10.1 now lists it, and SP-10.11 now routes cache-policy and mirror changes to invariants 7 and 8.
- **Open:** no file in this repository builds `forge-full` (`crates/solid-pod-rs-server/Cargo.toml:172`) except a test. The cargo default is empty (`Cargo.toml:127`). "Shipped" therefore means consumer builds. SP-10.5 now says the default-off list describes a bare cargo build.

**SP-FM-04: NIP-98 window width (Drift, repaired).**
- **Evidence:** `TIMESTAMP_TOLERANCE` is 60 (`auth/nip98.rs:28`), applied as `abs_diff` (`auth/nip98.rs:151`). That is ±60 s, a 120 s window. It agrees with the server comment (`lib.rs:196`) and the 120 s replay TTL (`auth/replay.rs:78`).
- **Closes when:** already closed in the topic.

**SP-FM-05: invariant 1 holds after the empty-body fix.**
- **Evidence:** commit `8f2df86` changed the single verifier to bind any supplied body, empty included (`auth/nip98.rs:191`). SP-ID-01 from the 2026-11-01 report is therefore closed in source.
- **Not checked:** this review did not trace whether every server route passes `Some(body)`.
- **Closes when:** a route-level test shows each mutating route binds its body. That is outside SP-10.

**SP-FM-06: records lag code in two places (Drift).**
- **ADR-2003:** its 2026-11-01 refresh now records the audience check (`ADR-2003-solid-oidc-01-defer-lws10.md:166`). Its older "audience is not validated" section stays as history. D1 is reframed accordingly.
- **CHANGELOG:** the alpha.13 "Known gaps" still says the OIDC helpers do not check `aud` (`CHANGELOG.md:606`). Alpha.14 closed that (`CHANGELOG.md:504`).
- **PUT handler comment:** it still says provenance "errors swallowed" (`lib.rs:1752`), seven lines before it attaches the receipt (`lib.rs:1759`).
- **Closes when:** both comments are refreshed. Neither is a code defect.

**SP-FM-07: observability is narrower than the panel said (Debt).**
- **Evidence:** the server crate never calls `with_metrics`; a source search for `with_metrics|SecurityMetrics` under `crates/solid-pod-rs-server/src` found nothing. The counters are therefore never read in the native server. A plain `acl_denial` (`lib.rs:1109`) writes no log line; only policy-resolution failures (`lib.rs:2250`) and provenance failures log.
- **Correction:** SP-10.12's "logged but never counted" was corrected to match.
- **Closes when:** a deployment needs those rates; until then this is recorded debt.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| SP-10.1 | corrected | Seven → eight invariants. I8 added (`Cargo.toml:172`); `verify_at_with_policy` moved to `nip98.rs:120`. Invariant 6's EdDSA asymmetry confirmed at `oidc/mod.rs:184` and `:549` (`verify_dpop_proof_core`). Other invariant sites unchanged and read. |
| SP-10.2 | corrected | H, I and J rebound (SP-FM-02). C now cites the unconditional call (`nip98.rs:212`), the stub (`:381`) and the server compile-time assertion (`lib.rs:192`). F adds the condition gate (`evaluator.rs:306`). Order fixed, with PATCH's J as a branch. A, B, D, E, G and K read and true. |
| SP-10.3 | corrected | Sidecar failure path → `lib.rs:4105`; header → `lib.rs:1359`; PUT 201 → `lib.rs:1755`. The server delegates to `record_receipt` (`lib.rs:4023`), so the library branches it cites (`provenance.rs:512`, `:589`) are the ones the server runs; an Invariant marker now says so. Stale comment marked. Limit: "201 in every branch" and the non-atomic window are source reading only. |
| SP-10.4 | corrected | W1 → put at `lib.rs:1747`, mark at `:1754`. W2 → metadata renamed before the body (`fs.rs:256`). W4 → `lib.rs:208`. W5 → 120 s window. W6, W7 and W8 rebound to `lib.rs:2946/2951`, `:3057/3074` and `CHANGELOG.md:541/546/603`. Limit: crash, restart and replica windows are not exercised. |
| SP-10.5 | corrected | Forge node now names `forge-full` (`Cargo.toml:172`), with a Tension against invariant 8. Other defaults confirmed: `main.rs:96/102/343/454`, `schema.rs:223/322`, `solid-pod-rs/Cargo.toml:112/173`, admin 403 at `lib.rs:4449`. The JEV advisory on `main.rs:96` was checked by hand: the doc comment at `main.rs:93-95` supports the label. |
| SP-10.6 | corrected | D1 reframed (SP-FM-06), CHANGELOG cite → `:504`. D2 (`oidc/mod.rs:1013`), D3, D7 (`mempool.rs:60`) and D8 (`prov.rs:749`) read and true. The note that D2 and D3 are pinned by tests holds: the compatibility matrix records both deltas (`solid-oidc-compatibility-matrix.md:198`, `:202`), and `tests/oidc_compat_matrix.rs:330` asserts that discovery does not advertise EdDSA. No test that pins the missing `cnf.webid` branch was found by name. For D5 and D6, only the in-repo half was checked: the in-repo classifier (`resolver.rs:265`) and the only in-repo `impl ReplayStore` (`replay.rs:235`). The edge resolver and the second implementor live in nostr-rust-forum, which is not pinned for this wave, so they cannot be checked from this repository. |
| SP-10.7 | corrected | R6 updated to README's 86.4% over 300 rows (`README.md:171`). The revision in the note changed from `a4a86f6` to `9b6143c`. R3 and R4 now cite `ecosystem-integration.md:105` and `:47`. R1, R2 and R5 confirmed. |
| SP-10.8 | corrected | CHANGELOG cites +352 (`:550/559/580/603`). The rotate note moved to `:557` (it had cited the Control entry). The audit gate moved to `ci.yml:195`. deny.toml exceptions confirmed at `:47/56/63` and mirrored in `.cargo/audit.toml`. Open marker added: the 2026-08-19 audit and F3 carry no citation. |
| SP-10.9 | corrected | PF → `lib.rs:2244`, AD → `:1109`, LOG → `:3719`, `format_error_chain` → `:3755`. Two corrections to N2: the outermost claim (SP-FM-01), and the cause-chain description, since the function appends the Debug dump rather than walking `source()`. The 403/503 split is confirmed. |
| SP-10.10 | corrected | B7 → `lib.rs:3181`, B10 → `:1770`. B8 adds `notifications/legacy.rs:75`. B1–B6, B9 and the B3 depth-scan note confirmed (`wac/mod.rs:28/33/39`, `nip98.rs:29`, `replay.rs:82`, `lib.rs:383/540`, `ldp.rs:167`). Limit: no load test. |
| SP-10.11 | corrected | Q7/C7 added for invariants 7 and 8. C3 → `lib.rs:1359`. The change-process note matches `BASELINE-solid-pod-rs.md` "Change process". |
| SP-10.12 | corrected | H1, H2, H3, L1 and L3 rebound (`lib.rs:1359/1328/1122/1339/3719/2250`). L3 now says invalid or unreadable ACLs rather than denied ACLs. N3 narrowed and a Debt marker added (SP-FM-07). The metrics module and scope note confirmed (`metrics.rs:4/5/26/63/76/88/93`, `ssrf.rs:150`, `dotfile.rs:89`). Limit: the EXTERNAL note naming VisionClaw's webxr as the binder repeats the module doc and was not traced into VisionClaw. |

## Verification

**Evidence class.** Source tracing at the pinned revision, read with `git -C /home/devuser/workspace/solid-pod-rs show 9b6143ce55b2:<path>` and `git grep -n ... 9b6143ce55b2`, never from the working tree.

**What was compared.** Every citation line was compared between `09ec90f` and `9b6143c`, then re-resolved from its symbol or behaviour. The cited sources that changed in that range are `CHANGELOG.md`, `README.md`, `Cargo.toml`, `.github/workflows/ci.yml`, `auth/nip98.rs`, `ldp.rs` and server `lib.rs`. The other cited sources are byte-identical, and they were read for meaning rather than credited unchanged.

**Topic changes.** `verified_commit` is now `9b6143ce55b242c322c92d8c6bcc0c1a5371b451`. Four sources were added: ADR-2003, `ecosystem-integration.md`, `notifications/legacy.rs` and the BASELINE.

**Gate.**

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 10-failure-modes-and-invariants.md --check --cite-check --strict-citations --render
```

- **Before the edit:** exit 0, 117 citations, all verified at the old `09ec90f`. That run says nothing about the pin. It raised 6 JEV advisories: `lib.rs:991`, `:3915`, `:2152` (twice), `:3556` and `:2991`. All six are among the miscites listed in SP-FM-02.
- **After the edits:** exit 0, 134 citations verified at `9b6143c`, 0 unresolvable, 12/12 rendered. An intermediate run raised one JEV advisory (`main.rs:96`), which was checked by hand and holds; the final run raised none.

**What was not verified.**
- No test, build or `cargo` run.
- No running server.
- No observation of middleware order, crash windows, replica behaviour or 201 statuses.
- Consumer builds that set `forge-full`.
- Whether every route passes the request body to NIP-98.
- The nostr-rust-forum halves of D5 and D6, and VisionClaw's metrics binder.

**Format note.** SP-10 has no "What it shows" or "Why it is this way" paragraphs. Register markers were added as bullets under the panels, in SP-07's format.
