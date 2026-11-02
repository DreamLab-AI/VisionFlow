# Security review — Commit A, round 5

- Scope: `fix-a5/fix-a5-interdiff.patch` (`CHANGELOG.md` and `mirror.rs`), read against the current tree in `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`.
- Method: read-only. No cargo, no edits.
- Line numbers are in `crates/solid-pod-rs-forge/src/mirror.rs` unless stated otherwise.
- Labels: **CONFIRMED** means traced end to end in source. **PLAUSIBLE** means the path exists, but whether it triggers depends on timing, the environment or operator action.

## Outcome

- **Nothing material remains for Commit A.** I found no Critical, High or Medium issue.
- **Confidentiality.** No path lets a held or unapproved object reach served. Round 5's rescoped scan does not open one: every store the legacy code ever fetched into carries `hideRefs` (see Q2), and `legacy_traces` refuses it whatever `scan` says.
- **Every C1–C7 and R1–R5 finding is FIXED,** except R1, which is FIXED for stores first followed now and PARTLY FIXED for one narrow population (S3).
- **Seven Low findings and five informational notes follow,** mostly about availability.
- **One item matters for Commit B:** round 5 changed what the marker means. It no longer certifies "approved closure only" (S1).

## 1. Finding-by-finding status

| ID | Status | Evidence |
|---|---|---|
| C1 forged event cancels a pending retry | **FIXED** | `enqueue` verifies (`2781`) before reading or changing the slot (`2784-2800`). The only other inbox writer is the timer arm, which re-inserts an already-verified pending event (`2861-2870`). `on_event` still verifies too (`2729`). |
| C2 / R3 timer race loses the pending retry | **FIXED** | The timer arm keeps the slot's event only when the pending one does not supersede it (`2865-2870`). An older event there is replaced; the same event is kept. |
| C3 / R4 transient fetch or materialise failure not retried | **FIXED** | Fetch: `fetch_checked` returns `(reason, retryable)` (`2976-2981`), and the engine schedules a retry when retryable (`3293-3299`). Materialise: schedules a retry (`3279-3281`). Residuals are S4 and S5. |
| C4 retry floor counted from the attempt's start | **FIXED** | `schedule_retry` reads the clock at failure time (`3058-3062`). Every failure path uses it (`3281`, `3298`, `3396`). |
| C5 seen versus acted-on tie order | **FIXED** | `acted_on_at` (`2564-2570`), used at `3236`, gives the same order as the replay check (`3163-3167`) and `supersedes` (`2575-2577`). Worked through in Q5. |
| C6 / R2 growth cap retryable | **FIXED** | `push_failure` (`1414-1426`) is applied before redaction (`2370`), so the exact-string match is reliable. Fetch side: `2979`. |
| C7 ceiling overshoot undocumented | **FIXED** | Rustdoc at `121-130`, CHANGELOG at `108-113`. Both describe it as an admission threshold, with up to `MAX_REPO_BYTES` of growth measured by polling, so about 2.5 GiB. |
| R1 owner store stuck at 503, scanned on every event | **FIXED**, with **PARTLY** for one case | `scan` only for a pre-release record (`3036`); `mark_clean_store` runs the scan only when asked (`1772`). The verdict is persisted (`3030`, `3217`, `2950`). For states, the replay check runs before `prepare` (`3163-3195`). The scan is streamed (`1605-1645`, `1651-1714`). Residual: S3. |
| R2 scan errors, timeouts and `mark_clean_store` errors retryable | **FIXED** | `publish` no longer scans or marks (`2262-2281`). Scan errors fail closed (`3043-3046`). Publish timeouts are bounded by `MAX_PUBLISH_TIMEOUTS` (`3389-3397`). |
| R5 young quarantines tip the store past the ceiling | **FIXED** | Quarantines are left out of the ceiling (`2320-2338`). The growth limit is still measured from the full size (`2339`). A stale comment remains (I3). |
| I1 stale-lease retry overwrites a moved ref | **FIXED** (docs) | Rustdoc at `1183`, `1206`. |

## 2. Findings, ranked

### S1 — Low, CONFIRMED (matters for Commit B, not a new exposure): the marker no longer means "approved closure only"; a fork of a legacy mirror is marked when followed

**What changed.** Before round 5, the marker meant that every object in the store was reachable from its branches and tags. Now a store first followed under isolation is marked without a scan: `scan` is false whenever there is no pre-release record (`3036`, `3133-3134`, and `1772`).

**The case that matters.** A fork made earlier from a legacy mirror:
- `build_fork` runs `git clone --local --bare` (`forks.rs:~342`). That hardlinks the source's whole object store, including held objects.
- The fork has no mirror record, no `refs/mirror` (a bare clone copies only branches and tags) and no `hideRefs` (clone writes a fresh config).
- If the operator ever follows that fork's key and name, `legacy_traces` finds nothing, no scan runs, and the store is marked.

**Exposure.** None new: that fork was already served without any gate before round 5, because it has no record (already listed under Commit B in `review-a-security.md`).

**Why it still matters.** If Commit B trusts the marker, for example "migrate or sweep only unmarked stores", it will skip a store that holds held objects.

**Fix.**
- Commit B: sweep forks of legacy mirrors whether or not they carry a marker. Enumerate them through `forge.parent` in their config, not the marker.
- Commit A: record the new marker semantics, "may publish; legacy layout not detected", in `PUBLICATION_MARKER`'s rustdoc (`lib.rs:~98`) and in `publication-design.md`. Decision 13 at line 343 still states the overturned rule (I3).

### S2 — Low, PLAUSIBLE: two fail-open edges in the legacy detection

Legacy detection rests on two independent signals: `refs/mirror` refs, and `hideRefs` naming them. In addition, a scan runs for a pre-release record. Two edges each weaken one leg:

1. **The config read fails open** (`1734-1747`).
   - `if let Ok(v)` treats any failure of `git config --local --get-all` as "key unset". That includes a 15 s timeout or a spawn failure under load, not only the exit code 1 that an unset key returns.
2. **The pre-release signal can be erased.**
   - `MirrorStore::read` returns `None` for an unreadable or unparseable file (`1122-1125`).
   - `apply_locked` and `on_announce` then treat the store as first followed: `prepare_served(None)`, so `scan = false` (`3133-3134`, `2943-2947`).
   - They then save a new record with `isolated: true` (`1064`, `3141`, `2945`) over the old file, so the signal is gone for good.

**When a store with held objects would be marked:** only when all of these hold:
- its `refs/mirror` refs are gone. Legacy `drop_mirror_refs_and_gc` did this on a failed fetch, but the `gc --prune=now` that follows normally deletes the held objects too, so this also needs that gc to have failed;
- the config read fails transiently;
- the record is unreadable.

That is a chain of coincidences, so it is not a practical attack, but nothing is gained by failing open here.

**Fix.**
- Read config with `git config --local --list` (exit 0 when empty) and match the keys case-insensitively, or tell exit 1 apart from any other failure. Treat any other failure as `Err`, so prepare fails and nothing is marked.
- When the record file exists but does not parse, treat it as pre-release (`scan = true`) and do not overwrite it. `MirrorStore::read` would have to tell "absent" apart from "unreadable".

### S3 — Low, PLAUSIBLE: the scan has no legacy content to find, but it can still strand an owner store

**Legacy code always hid its refs.** In the introducing commit (`4e97893`: `materialise` at `:1215`, `fetch` at `:1235`) and at HEAD (`:1397`, `:1418`), `hide_mirror_refs` runs before any fetch into `refs/mirror`, and the fetch is aborted with `?` if it fails. So any store the legacy engine fetched into carries `hideRefs`, and `legacy_traces` already refuses it.

**So the scan can only find:**
- operator-altered configs;
- or the owner's own objects. That happens when a pre-release record exists but the legacy engine never fetched, for example the "no clone URL yet" case in `a_followed_repo_without_a_clone_url_still_records_why`, or a fetch that failed before `hide_mirror_refs`, and the owner then creates and force-pushes the repository.

**Effect.** In that second case the scan finds the dangling objects and returns `AwaitsMigration`, which is persisted (`3217`). The repository stays at 503 permanently: R1, for a smaller population.

**Fix.** Ask for the scan only when the pre-release record shows that the legacy engine actually fetched (`last_fetch_at.is_some()`, or `last_fetch.is_some()`). A record that never fetched cannot have left held objects.

### S4 — Low, CONFIRMED (a new amplification from the C3 fix): fetch timeouts are retried without limit and can tie up the four global fetch permits

**Where:** `fetch_checked` treats every git error except `GREW_PAST_CAP` as retryable (`2978-2981`), and there is no counter like `publish_timeouts`. The round 5 report acknowledges this.

**The amplification.** Before round 5, one event caused one fetch attempt. Now one event causes retries for ever.

**Attack path.**
- A followed owner, or whoever runs the clone host the owner's 30617 names, serves the fetch slowly (a slow drip under 512 MiB), so it runs to the full 60 s `FETCH_TIMEOUT`.
- Each attempt holds one of the `MAX_CONCURRENT_FETCHES` (4) permits for 60 s.
- With `minIntervalMs: 0`, the retry comes back every 60 s (`backoff_delay` caps at 60 s, `838-840`), so each such repository holds about half a permit indefinitely. Around eight such repositories starve every other mirror.
- With the default 30 s interval, `backoff_interval` stretches to `MAX_BACKOFF_MS` (1 h), so the effect is bounded.

**Fix.** Generalise `publish_timeouts` to count consecutive timeouts of the newest state for fetch as well as publish, with the same `MAX_PUBLISH_TIMEOUTS` rule.

### S5 — Low, CONFIRMED: `fetch_inner`'s own "clone URL refused" is classified retryable

**Where:** `fetch_inner` returns `"clone URL refused"` when `is_operator_clone` or `is_remote_clone` fails, or the URL starts with `-` (`2163-2165`). That string reaches the engine through the `git` closure, which marks it retryable (`2978-2981`). The identical refusal made in `fetch_checked` is terminal (`2977`, `3000`).

**Who can reach it.** An operator-configured clone that fails `is_operator_clone`: `fetch_checked` only `Url::parse`s operator URLs (`2989-2998`). It is not reachable from remote announcements, because `read_clones` and `is_remote_clone` filter them first.

**Effect.** A cheap retry about once a minute, for ever. There is no network traffic, because the refusal comes before git starts.

**Fix.** Use a shared constant compared like `GREW_PAST_CAP`, or a typed error, so that the refusal is terminal on both paths.

### S6 — Low, CONFIRMED: every `AwaitsMigration` verdict is persisted, including ones from transient errors

**Where:** a verdict from `prepare_served` is saved unconditionally (`3216-3231`, `2950`), and nothing clears it. The only writers of `awaits_migration` set it to `true` (`2950`, `3217`).

**Transient routes into a permanent verdict:**
- `has_alternates` returns `true` for any `symlink_metadata` error other than NotFound (`1967-1976`), for example EACCES or EIO. This applies to isolated owner stores as well.
- A scan error, such as a 120 s `fsck` timeout on a large store, counts as awaiting migration on purpose (`3043-3046`). That fails closed, but for good.

**Effect.** Availability only: a permanent 503 until Commit B or a hand edit of the record. There is no confidentiality impact, because this direction is fail-closed.

**Fix.**
- Persist only verdicts backed by positive evidence: a trace found, an alternates file that exists, an unreachable object reported.
- For errors, skip the state without persisting, so the next state re-examines the store.
- Or keep error verdicts, but stamp them and re-examine after a day.

### S7 — Low, PLAUSIBLE: a non-scan `prepare` error is terminal for the newest state

**Where:** `3196-3214`. Without the scan, `prepare` runs only quick checks (`for-each-ref --count=1`, `config --get-all`, `stat`), so a failure is more likely transient (load, EAGAIN) than permanent.

**Effect.** The newest state is stranded until a newer one arrives or a relay reconnect redelivers it.

**Background.** R2's reason for making this terminal was the repeated scan, and that no longer runs on this path.

**Fix.** Make it retryable through `schedule_retry`: it is now cheap, and the backoff bounds it.

### I1 — Informational: a first record can be saved beside an unprepared store through the far-future branch

**Where:** with no record, `early` prepare runs (`3133-3134`) before the far-future check, and that branch saves the record unconditionally (`3148-3161`).

**Effect.**
- If `early` returned `Err`, the store is gated at 503 until the next valid event re-prepares it.
- If it returned `AwaitsMigration`, the flag is not persisted, which costs only a re-examination.

**Reachability.** Not reachable through relays: `enqueue` drops far-future events with the same threshold and an earlier clock (`2775`). It is reachable through direct callers of the public `apply` and `on_event`.

**Fix.** Do the far-future check before `early`, or skip the save when `existing.is_none()` and `early` is not `Ok(Ready)`.

### I2 — Informational: the order of verification in `enqueue`

**Q3 answer.** The signature is verified before any slot or pending change (`2781` versus `2784`). The CPU cost is acceptable:
- verification runs only after `wanted` (kind, at most 512 tags, `d` tag, a followed author) and the far-future check;
- each frame is bounded by `MAX_FRAME_BYTES` (128 KiB);
- the work is one SHA-256 over at most a frame plus one BIP-340 verification, so it grows linearly with relay bandwidth, in the same order as the JSON parse the session already does.

**Two cheap improvements:**
- Compare against the slot first, then verify only an event that would take the slot. This is safe, because only verified events ever enter the slot. It removes the verification cost of replayed older events.
- Accepted events are verified twice (`2781`, `2729`). The cost is small; leave it, or pass a verified marker through.

**Runtime note.** Verification runs synchronously on the relay session task. On a multi-threaded runtime, a flooding relay therefore delays only its own session.

### I3 — Informational: stale documentation

- `2366-2369`: "until then the size baseline tolerates it and the ceiling counts it". Since R5 the ceiling does not count quarantines (`2316-2333`).
- `publication-design.md:343`, decision 13, still states the overturned rule. The round 5 report says this was left on purpose; it needs a "superseded in round 5" note, together with the marker semantics from S1.

### I4 — Informational: a successful push can be recorded as a terminal failure

**Where:** `run_in`'s `select!` (`1882-1886`) is not `biased`. If the push exits in the same poll in which the growth watcher fires (`grew`) or the deadline passes, either error can win.

**Effect.**
- With `GREW_PAST_CAP`, the refs have moved but the record says the publish failed, and no retry is armed, because that error is now terminal.
- The record stays inconsistent until the next state, which then plans as in sync.
- Only approved objects are involved, so there is no confidentiality impact.

**Fix.** Use `biased;` with `waited` first, or re-read `local_heads` after a kill and treat an update that landed as success.

### I5 — Informational: the timeout counter can be reset forever

`publish_timeouts` resets on any non-timeout failure (`3389-3393`). A sequence that alternates timeout, stale lease, timeout and so on never reaches `MAX_PUBLISH_TIMEOUTS`. That is contrived and bounded by the backoff. If wanted, count attempts of the newest state instead.

## 3. Answers to the brief

### Q2 — The rescoped scan

**Confidentiality: can a store holding legacy held objects escape the scan and be marked?** Not without the coincidences in S2.

- **Legacy signals.** `mark_clean_store` (`1761-1779`) refuses, whatever `scan` says:
  - alternates (`1768`);
  - any `refs/mirror` ref (`1720-1733`);
  - `hideRefs` naming `refs/mirror` (`1734-1747`).

  Every legacy version I checked set `hideRefs` before its first fetch (`4e97893:1215,1235`; `HEAD:1397,1418`). So a store that legacy code ever fetched into is refused through `legacy_traces` alone. The scan is extra defence (S3).
- **A record present while the store is unmarked.** A pre-release record gives `scan = true` (`3036`). An isolated record means the record was created under isolation; its store is then either one the engine materialised itself (`build_served` writes the marker) or an owner store prepared without a scan.
- **Event orders.** In every case the store is decided (marked, or verdict persisted) before any publish, and `publish` refuses an unmarked store (`2267-2281`).

| Order of events | What happens | Lines |
|---|---|---|
| Announcement before any state | `on_announce` prepares before its first save; if `prepare` fails it saves nothing | `2946-2955` |
| State with no record | `early` prepare runs before anything is saved | `3133-3134` |
| Restart | records and markers are on disk; no in-memory state is trusted | |
| Record deleted by hand | the next event prepares with `scan = false`, and legacy traces still refuse a legacy store. Until then the store is ungated (no record), which predates Commit A | |
| Candidate present or absent | does not affect marking: `prepare` looks only at served | `2426-2434` |

- **Who can trigger marking.** Only verified owner-signed events (`2729`, `2781`). A relay cannot forge one.

**Availability: can an ordinary owner store with dangling objects still be stuck at 503?** Only in these cases:
- a pre-release record with no legacy fetch (S3);
- a persisted verdict from a transient error (S6);
- the far-future first-record edge, from direct callers only (I1).

A store first followed now is not scanned. `an_owner_repo_with_a_force_pushed_away_commit_is_followed_and_served` (test `8353`) covers this.

**Saved verdict: can it go stale, or be written by a forged or replayed event?**
- Towards "clean": no. Nothing clears `awaits_migration`.
- Towards "stuck": yes, through S6, and nothing clears it.
- Forged events: no; they are verified first.
- Replayed states: no, when a record exists, because the replay check comes before `prepare` (`3163-3195`).
- Replayed announcements: they can reach `prepare` while `announce_at` is still unset (`2937-2941`). For a store already decided, that costs one `stat` of the marker or a check of the saved verdict, and it cannot change the verdict.

**Scan cost.**
- It is skipped for replayed states (test `the_store_is_examined_once_and_never_for_a_replay`, `8449`), and runs at most once per store, because the result is either a marker or a persisted verdict. It repeats only if saving the record fails.
- Memory in the forge process is bounded: each command's stdout is read line by line and stops at the first deciding line (`1680-1691`). A ref name line is at most about 64 KiB, the pkt-line limit on what legacy fetches could have written, and `fsck` lines are fixed-size. stderr goes to null (`1669`), so it cannot fill a pipe.
- The `git fsck` child itself uses memory in proportion to the store's object count. It is bounded by the 2.5 GiB store, runs once, is killed with its process group at 120 s (`1693-1707`), and then fails closed.

### Q3 — Verification in `enqueue`

See I2. The signature is checked before any slot or pending change, and a flood costs CPU in proportion to bandwidth, within the frame cap.

### Q4 — Retry classification

| Failure | Class | Lines | Verdict |
|---|---|---|---|
| fetch: git or network error, or timeout | retryable | `2978-2981`, `3297` | Right, but timeouts are unbounded (S4) |
| fetch: grew past the size cap (watcher or post-check) | terminal | `1885`, `2221`, `2979` | Right |
| fetch: SSRF refusal, DNS failure inside the check | terminal | `3002-3005` | Accepted decision. A transient DNS outage strands the state until the next event or relay reconnect. A typed `SsrfCheck` error (Refused or Unresolvable) would let resolution failures be retried |
| fetch: `fetch_inner`'s "clone URL refused" | retryable | `2163-2165` | Wrong (S5) |
| materialise | retryable | `3279-3281` | Right |
| publish: growth cap | terminal | `1415-1416` | Right |
| publish: timeout | retryable, terminal after 3 in a row | `1417-1422`, `3389-3397` | Right (I5 is a corner case) |
| publish: stale lease or receiver ref lock | retryable | `1424` | Right |
| publish: unmarked, legacy traces, an error checking them, or alternates | terminal | `2267-2288` | Right |
| scan error (`scan = true`) | awaits migration, persisted | `3043-3046` | Fail-closed; availability cost (S6) |
| `prepare` error (`scan = false`) | terminal for the state, not persisted | `3196-3214` | Should be retryable (S7) |
| `AwaitsMigration` | skip, no fetch, persisted, no retry | `3216-3231` | Right |

No hot loop remains: every retry goes through `schedule_retry` (at least 2 s, doubling to 60 s), and `drain` clamps waits to at least `MIN_REQUEUE_MS` (`2922`).

### Q5 — Tie ordering (C5)

**"Seen" and "acted on" are now consistent.** Take three states at the same second `t`:
- L, the last state acted on, with id `l`;
- N, the newest state seen, with id `n < l`;
- a third state E with id `e`.

The planner and the replay check then agree:
- N's redelivery is planned against `t-1` (`2566`), so it fetches.
- E with `n < e` loses the tie at the replay check (`3163-3166`), so it is refused.
- E with `e < n` supersedes N, becomes the newest state seen, and is planned against `t-1` because `e < l`.

`saturating_sub` at `created_at = 0` simply skips the state, which is harmless.

### Q6 — Other changes in round 5, checked and found sound

- **Lock order:** unchanged. `on_event` takes the in-process mutex (`2733-2734`) and then `lock_repo` (`2735`), once each.
- **Locks while preparing:** `on_announce` and `apply_locked` call `prepare` while holding both locks. The longest hold is about 4 min (two 120 s scan steps), and this happens at most once per store.
- **No locks while waiting:** none are held during a retry wait (`2852-2876`), and the permit is dropped after each apply (`2909`).
- **`first_line`:**
  - it uses `--git-dir` and the hardened environment, with `HOME` and `XDG_CONFIG_HOME` removed (`1659-1667`);
  - it kills the process group on a match, a timeout or a read error (`1693-1712`);
  - on "no match" it waits for git to exit successfully within the remaining budget (`1698-1707`). A non-zero `fsck` exit, for example from a corrupt object, is therefore an error and fails closed.
- **Detached `HEAD` and refs outside branches and tags:** both count as unreachable (`1607-1629`), so neither `HEAD` nor `refs/notes/*` can vouch for an object. The `FETCH_HEAD` and notes guards pass, per the report's tests and mutations M2 and M4.
- **`publish` no longer marks:** marking happens only in `prepare`, under the repository lock, after the replay check or before a first save.
- **`publish_timeouts`:** reset when a newer state is seen (`3186-3188`), so a replay cannot reset it.
- **Exact-string classification:** `GREW_PAST_CAP` and `GIT_TIMED_OUT` are compared before redaction (`2370`, `2979`), and redaction does not touch them anyway, because they contain no paths.
- **Ceiling:** quarantines are left out of the admission check only. The growth limit uses the full size (`2339`), so a planted or young quarantine cannot widen what one push may add.
