# Commit A fix round 6: report

Items 1–7 and 9 are done. Item 8 is half done: I5 is fixed, and I4 is recorded below as not done. All nine gates exit 0 on the final tree. Nothing is committed. The worktree /home/devuser/workspace/wt/solid-pod-rs-mirror-publication is at HEAD 31fe898, with rounds 1–6 uncommitted.

- **Starting tree:** `git diff HEAD` was byte-identical to `../final-a5.patch` (SHA-256 `8696d9cd…436f` for both).
- **Round 6 only:** `fix-a6-interdiff.patch` (1,382 lines; `CHANGELOG.md`, `lib.rs`, `mirror.rs`), built from a scratch index of HEAD plus `final-a5.patch`.
- **Full diff against 31fe898:** `fix-a6.patch` (SHA-256 `6dfafcf5…6e53`).
- **Red logs:** `red-raw.log` (11 tests), `red-raw-push.log` (1), `red-raw-unsafe-key.log` (1).
- **Green:** `green-forge.log` (a copy of gate 2).
- **Mutations:** `mutations.txt`.
- **Gates:** `fix-a6-gates.txt` (final) and `fix-a6-gates-run1.txt` (the run before the unsafe-key fix, also all exit 0), with logs in `gate-logs/` and the script in `run-gates.sh`.
- **Toolchain:** rustc 1.99.1 and git 2.56.0, with `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.

## How the red output was captured

1. **Behaviour-neutral refactor first,** with the suite green (326 lib and 66 doc tests). It made these changes:
   - added `StoredRecord` and `MirrorStore::load`, with `read` rebuilt on `load` and nothing calling `load` yet;
   - added the `#[cfg(test)] config_args` seam on `GitCli`;
   - renamed `MAX_PUBLISH_TIMEOUTS` to `MAX_TIMEOUTS` and `publish_timeouts` to `timeouts`;
   - added the shared `CLONE_URL_REFUSED` constant;
   - added a `resolution_failed` stub that returns `false`, the old behaviour.
2. **Red run:** 11 new tests (`red-raw.log`), then the push-guard test (`red-raw-push.log`).
3. **Fixes,** then green.

Three things happened after the first red run:

- **The prepare test's fixture changed.** As first written, it followed a repository for the first time, and counts for that case cannot persist (see Limitations). It now runs against a saved record. Mutation MP, which makes a prepare error terminal again (the round-5 behaviour), kills it.
- **The final review found a regression in my own `load`.** It mapped an unsafe key to `Unreadable`, so a pod user's push to `/forge/alice/demo.git` got 403 "is a mirror and read-only", and the record API answered 503 instead of 404. Test `a_namespace_that_cannot_hold_a_record_is_not_a_mirror` (`lib.rs:2326`) was red (`red-raw-unsafe-key.log`: `'alice/demo' is a mirror and read-only`). It is fixed: an unsafe key is `Missing`, because no record file can exist at a path that cannot be formed. That matches the gate's own path check.
- **A guard test was added for hideRefs.** Mutation MH showed that no test in the crate covered detection by `hideRefs` alone, before this round or after. `hide_refs_alone_is_a_legacy_trace` (`:8985`) now covers it; it was green before the fix, and MH kills it.

No existing assertion changed. One existing test changed its code but not its assertion: `repeated_publish_timeouts_of_one_state_become_terminal` now uses the renamed `MAX_TIMEOUTS`.

## Items

### 1. S2.2 and Codex High 1: a corrupt record fails closed

**Three-way load.** `MirrorStore::load` (`mirror.rs:1177`) returns a `StoredRecord` (`:1091`):
- `Missing`: the file is absent, or the key is unsafe;
- `Unreadable(reason)`: any other read error, or invalid JSON; the reason names no path;
- `Valid`.

`read` keeps its public signature: it returns `Some` for a valid record only.

**Engine.** `apply_locked` (`:3295`) handles an `Unreadable` record before anything else:
- it returns a record that is never saved, with `awaits_migration` set, a Skip whose reason is "the mirror record … ; … awaits migration", and no retry;
- `prepare` is never called, so the store is neither examined nor marked;
- the file is left as it was.

`on_announce` (`:3071`) returns `None` and saves nothing.

The brief said `prepare` should return `AwaitsMigration`. The engine instead stops before `prepare`; the outcome is the same, and this way no code path can reach a save.

**Readers.**
- The reader gate already stats the path, so a record that exists makes the repository a mirror.
- The smart-HTTP push guard (`lib.rs:732`) now refuses anything that is not `Missing`.
- The record API (`lib.rs:967`) answers 503 for `Unreadable`.

**Tests and red output:**

| Test | Red output |
|---|---|
| `an_unreadable_record_awaits_migration_and_is_left_in_place` (`:8904`): an announcement, then a state, beside a store holding an unreachable object | "the unreadable record was overwritten" (the bytes became a new `isolated: true` record) |
| `the_record_api_refuses_an_unreadable_record` (`:8947`) | `404 {"error":"not a mirror"}` |
| `an_unreadable_mirror_record_still_refuses_the_owners_push` (`lib.rs:2354`) | 401 from the CGI, past the guard |

Mutation MR (an unparseable file loads as `Missing`) kills all three.

### 2. S2.1: `legacy_traces` fails closed on config errors

`legacy_traces` (`:1829`) now reads the whole local config with `git config --local --list -z`. `hides_mirror_refs` (`:1484`) then matches the two keys case-insensitively. Any failure is an `Err`: a non-zero exit, a timeout, or a spawn error.

**How this squares with the brief.** The brief said only exit code 1 means "unset". `--list` has no unset outcome at all, so no exit status is ever read as "no hideRefs". The error fails closed: the store is not marked and stays gated. Under item 6 it is then retried, not recorded as `AwaitsMigration`, so the "counts as a trace" effect holds for every attempt without making the verdict permanent.

**Tests:**
- `a_failed_config_read_is_not_read_as_no_hidden_refs` (`:8963`), using the seam to make the config read exit 129. Red: "a failed config read let the store pass: Ok(Ready)".
- `hide_refs_alone_is_a_legacy_trace` (`:8985`), a guard; MH kills it.
- MC (keys compared case-sensitively) is an equivalent mutant, because `git config --list` already prints names in lower case.

### 3 and 5. S4 and Codex C3/R4: bounded timeouts, and DNS retried

**The retry decision is now three-way.** A private `Retry` (`:2676`) is `No`, `Yes` or `Counted`. `Mirror::failed` (`:3215`) is now the single place that:
- adds to `failures`;
- adds to `timeouts` for `Counted`;
- schedules the retry unless the retry is `No` or the limit has been reached.

**What counts towards `MAX_TIMEOUTS` (3, `:145`).** One shared count per state (`MirrorRecord::timeouts`, `:1051`) covers fetch timeouts, publish timeouts, timeouts examining the store (so a scan that keeps timing out is not rerun for ever), and DNS failures. When the limit stops the retry, the reason says "(N timeouts of this state; not retried)".

**DNS classification.** `resolution_failed` (`:1476`) recognises the two `SsrfError` resolution variants by their rendered text, because `SsrfCheck` reports only a string. A DNS failure is then recorded as "clone host not resolved: …" and is `Counted`. A resolved address the policy refuses is still "clone host refused: …" and terminal.

**Tests and red output:**

| Test | Red output |
|---|---|
| `repeated_fetch_timeouts_of_one_state_become_terminal` (`:9017`) | `fetch timeout 3 was retried` |
| `a_dns_failure_is_retried_within_the_limit_and_a_refusal_is_not` (`:9100`) | "DNS failure 1 was not retried" (it was "clone host refused") |
| `resolution_failures_are_told_from_policy_refusals` (`:9142`) | failed on the stub |

The last of these builds real `SsrfError` values, so a change to their text breaks it.

### 4. S5: the adapter's "clone URL refused" is terminal

`CLONE_URL_REFUSED` (`:1497`) is returned by both `fetch_inner` and the engine's own check, and both compare it as `No`.

- **Test:** `the_adapters_clone_url_refusal_is_terminal` (`:9049`), using an operator `ftp://` clone.
- **Red:** "a refused clone URL was retried".

### 6. S6: only evidence persists a verdict

Persisted `AwaitsMigration` now comes only from positive evidence:
- a legacy trace;
- an alternates file that exists;
- a scan that reports an unreachable object;
- an unreadable record (item 1).

**Changes:**
- `has_alternates` (`:2070`) returns `Result`. An error other than NotFound is an `Err`, not `true`. In `publish` that error is still a terminal refusal.
- `prepare_served` no longer maps a scan error to `AwaitsMigration`. Every `prepare` error goes to the retry path (item 7) and is not recorded.

**Tests and red output:**

| Test | Red output |
|---|---|
| `a_failed_scan_is_retried_and_not_persisted` (`:9158`): the commit's loose-object directory set to mode 000, so `fsck` exits 2; once restored, the real finding is persisted | "a failed scan became a verdict" |
| `an_alternates_check_error_is_retried_and_not_persisted` (`:9197`): `objects/info` set to mode 000 on an owner store followed for the first time; no record is saved, and once restored the store is marked | the saved record had `awaits_migration: true` |

Mutation MA kills the alternates test. Both tests print a SKIP and return when run as root.

### 7. S7: a non-scan `prepare` error is retryable

The `prepare` error branch in `apply_locked` (`:3400`) now goes through `failed`:
- a timeout is `Counted`;
- anything else is `Yes`;
- the reason reads "the served repository could not be examined: …".

As before, a first record is not saved beside an unprepared store.

- **Test:** `a_prepare_error_is_retried_and_its_timeouts_are_bounded` (`:9238`).
- **Red:** "a prepare error was not retried".
- **Mutation:** MP kills it on the final fixture.

### 8. I4 and I5

- **I5, done.** `timeouts` is no longer reset by a failure of another kind; only a newer state or a completed apply resets it.
  - **Test:** `other_failures_between_timeouts_do_not_reset_the_count` (`:9277`): a timeout, then a stale lease, repeated.
  - **Red:** "timeout 3 of the state was retried after interleaved failures".
- **I4, not done.**
  - **Why no test:** a deterministic red test needs the push to exit inside the same poll in which the watcher fires. Hooks are disabled on the served side and the measure seam can't slow the push, so I can't arrange that without a new seam inside the push.
  - **What the fix would be:** after a kill on the growth cap or the timeout, re-read served's heads with `local_heads`, and treat an atomic update that landed as success (about 25 lines, plus a seam). A `biased;` `select!` alone would not fix it, because the watcher can fire just before the push exits.

### 9. Docs

- **`PUBLICATION_MARKER` rustdoc (`lib.rs:~98`):** now says "publication allowed; legacy layout not detected". It states that the marker does not certify every object is reachable for a store first followed after this release, and that a record, readable or not, without a marker has not been cleared.
- **I3:**
  - The quarantine comment in `publish_inner` (`:2478`) now says the ceiling leaves it out and the growth limit includes it.
  - In `publication-design.md`, which is outside the repository and not listed in `SHA256SUMS` or `source-manifest.json`, I appended to decision 13: *"Superseded by decision 18 (round 5); see decision 19 for the marker's meaning. Round 6 narrows 'the verdict is persisted' in 18 to verdicts backed by evidence (a trace, an alternates file, an unreachable object, an unreadable record); a check that could not be made is retried and not recorded."* Nothing else in that file changed.
- **Also updated:**
  - `Prepared::AwaitsMigration`, the `MirrorGit::prepare` errors section, `prepare_served`, and `fetch_checked`;
  - the docs for `MirrorRecord::failures`, `retry_at` and `timeouts`, `PublishError::timed_out`, and `MAX_TIMEOUTS`;
  - the three CHANGELOG bullets for marking, the API, and retries. The API bullet names `timeouts`, `MAX_TIMEOUTS`, `MirrorStore::load` and `StoredRecord`.

## Limitations and residuals (not fixed)

- **A first-follow `prepare` error is not bounded by `MAX_TIMEOUTS`.** No record may be saved beside an unprepared store, because it would gate an owner's store at 503, so `failures` and `timeouts` are not persisted in that case. That has three effects:
  - the backoff never doubles;
  - the retry recurs every 60 s at the default 30 s floor, or every 2 s with `minIntervalMs: 0`;
  - it lasts until the store can be examined, a newer state arrives, or the process restarts.

  Each attempt runs only the quick checks, with no scan, because a first record never asks for one. This is documented in `MAX_TIMEOUTS`, `MirrorRecord::timeouts` and the CHANGELOG. **Option, not done:** an in-memory per-key counter used only when no record exists.
- **Codex Medium 4.** Once the limit is reached, an identical redelivery of the same state (for example after a relay reconnect) still costs one more attempt, and the count then rises past the limit. It is not checked before the attempt.
- **A permanent scan error is retried for ever.** For example, a corrupt object makes `fsck` exit non-zero, and that reruns at the capped backoff. Only scan timeouts are bounded.
- **I4,** as above.
- **Out of scope per the brief and recorded by the lead:** forks of legacy mirrors; verification CPU and scan memory limits; trusting the marker once written; reuse of an `isolated: true` record.
- **MSRV 1.89** is still unverified here.

## Gates (`fix-a6-gates.txt`)

| # | Command | Result |
|---|---|---|
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | `cargo test --locked --offline -p solid-pod-rs-forge --features mirror` | exit 0: 340 lib + 13 + 18 + 66 doc |
| 3 | `cargo test --locked --offline -p solid-pod-rs-forge` | exit 0: 228 + 13 + 18 + 51 |
| 4 | `cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge` | exit 0: 3 (filter `forge`) |
| 5 | `cargo clippy … -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings` | exit 0 |
| 6 | `cargo clippy … -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings` | exit 0 |
| 7 | `RUSTDOCFLAGS=-D warnings cargo doc … --no-deps -p solid-pod-rs-forge --features mirror` | exit 0 |
| 8 | `cargo test … -p solid-pod-rs-forge --features mirror,anchoring,announce` | exit 0: 401 lib + 9 + 13 + 23 + 85 doc |
| 9 | `cargo clippy … -p solid-pod-rs-forge --features mirror,anchoring,announce --all-targets -- -D warnings` | exit 0 |
