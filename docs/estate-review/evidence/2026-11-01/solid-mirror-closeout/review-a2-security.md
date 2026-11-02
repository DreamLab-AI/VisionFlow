# Security review — Commit A, round 2 production code

- Scope: `fix-a2-interdiff.patch`, read against the current tree (rounds 1–4) in `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`.
- Method: read-only. No cargo, no edits to the worktree.
- Line numbers are in `crates/solid-pod-rs-forge/src/mirror.rs` unless stated otherwise.
- Labels: **CONFIRMED** means the code path was traced end to end from source. **PLAUSIBLE** means the path exists, but whether it triggers depends on timing or on the deployment.

## Summary

- No path lets a held or unapproved object reach served.
- No path applies a state older than `newest_seen_*`.
- The two Medium findings are about availability and resource use, not confidentiality:
  - R1 comes from the round 3–4 check for objects no ref reaches, and makes a followed owner-pushed repository permanently 503;
  - R2 is a set of errors still counted as retryable that are permanent.
- One Low race can lose a pending retry (R3).
- The rustdoc and CHANGELOG overstate the ceiling (Q4).

## Findings, ranked

### R1 — Medium, CONFIRMED: an owner-pushed repository with any object no ref reaches is gated (503) permanently once followed, and rescanned on every event

**Where:**
- `mark_clean_store` returns `AwaitsMigration` when `holds_unreachable_objects` is true (`1586-1593`; the scan is at `1510-1541`);
- `apply_locked` then saves a record (`2870-2884`);
- from that point the reader gate (`lib.rs` `unpublished_mirror`) refuses every reader, and refuses the owner's own pushes too.

**Why it triggers:**
- An ordinary owner-pushed repository nearly always holds objects no ref reaches: any force push, rebase or branch deletion leaves them until `gc` prunes them, which takes two weeks.
- The only roots counted are `refs/heads` and `refs/tags`, so anything reachable only from `refs/forge/heads/*` (fetched PR heads) or `refs/notes` counts as well.
- Following such a repository, which needs only one owner-signed event, takes it offline until Commit B. The owner cannot recover by pushing, because the gate covers receive-pack.

**Cost:**
- `prepare_served` runs on every event, announcements included, before the replay check (`2817`, `2517`). For an unmarked store with no legacy traces, that means:
  - both scans run: `rev-list --objects --branches --tags`, then `cat-file --batch-all-objects`, each with a 120 s budget;
  - each scan's whole stdout is buffered with no size limit (`run` → `wait_with_output`) and then copied into a `HashSet`.
- A relay can replay old owner-signed events, which costs it nothing, and keep up to `MAX_CONCURRENT_FETCHES` workers doing this back to back. On a 2 GiB store the buffered output can reach hundreds of MiB per scan.
- Legacy stores are not affected: the `refs/mirror`/`hideRefs` check returns early and skips the scan (`1593`).

**Fix, any of:**
- Run the scan only for stores that show mirror history: a record that predates this release, a candidate under `mirror-work/`, or legacy config. For an owner-pushed store, objects no ref reaches are the owner's own pushes, not held mirror content.
- Persist the verdict, for example `awaitsMigration: true` in the record or a sentinel file, so the scan runs once rather than on every event.
- Move `prepare_served` after the replay check, so a replayed event costs a `stat`, not a scan.
- Stream the scans instead of buffering them; for example, compare sorted output, or use `git fsck --unreachable --no-reflogs --connectivity-only`, which prints only the objects no ref reaches.

**Test to add:** an owner-pushed repository with one force-pushed-away commit is followed, and stays readable, or is explicitly documented as gated.

### R2 — Medium, CONFIRMED: some permanent failures are classified as retryable

`From<String> for PublishError` defaults to retryable (`1194-1201`). Several errors that will recur on every retry fall through to that default:

| Failure | Where | Classified | Should be | Effect |
|---|---|---|---|---|
| `mark_clean_store` command error (a scan timing out after 120 s on a large store, `rev-list` failing on a corrupt object) | `?` at `2080` | retryable | terminal, or awaits migration | retries every 60 s (with `minIntervalMs: 0`), each one a full fetch plus up to 240 s of scanning, for ever |
| growth cap hit during the push ("repository grew past the size cap") | `run_in` Err → `PublishError::from` at `2160` | retryable | terminal for this state | the same push hits the same growth again; repeats each backoff, and each kill leaves a fresh quarantine that the stale sweep will not remove for 120 s |
| publish timeout that recurs (a push too large for 120 s) | same | retryable | debatable | retrying is right for a one-off stall, wrong for a push that always takes longer; consider making the retry terminal after N consecutive timeouts on the same state |

**Bound:** the retry interval is `max(backoff_interval(base, f), backoff_delay(f))` (`3036-3039`).
- With the default `minIntervalMs` (30 s) this decays to one attempt an hour.
- With `minIntervalMs: 0`, `backoff_interval` is 0 and `backoff_delay` caps at 60 s (`838-840`), so the loop runs once a minute for ever: a slower D1.

**Fix:**
- Classify the growth-cap kill and `mark_clean_store`'s command errors as terminal. Return `PublishError::terminal`, or map them to `AwaitsMigration`.
- Or cap the number of same-state retries (for example, 5) and then record a terminal reason.

### R3 — Low, CONFIRMED in code, PLAUSIBLE to trigger: a pending retry can be lost to an older state in the inbox

**Where:** `drain`, timer arm, `2617-2624`.

**Sequence:**
1. While S2's retry is pending, the inbox slot is empty.
2. `enqueue` inserts any state into an empty slot regardless of age (`2554-2563`), so an older S1.5 lands there.
3. If the timer and the notification are both ready, `select!` may take the timer arm, chosen at random.
4. The timer arm runs `pending.take()`, then `.entry(slot).or_insert(S2)`. The slot already holds S1.5, so S2 is discarded.
5. The next loop pops S1.5 with `pending == None`, so it is applied.
6. The replay check refuses it (`2848`), and `newest_seen_id != S1.5` means no retry is armed (`2671`).

**Result:** S2 is not retried until the relay redelivers it (on reconnect) or the owner publishes a newer state. A relay that keeps replaying old states widens the window. There is no rollback: S1.5 is refused.

**Fix:** in the timer arm, put the pending event back unless the slot already holds a state that supersedes it. Use the same `supersedes` comparison: replace an older one rather than calling `or_insert`.

### R4 — Low, CONFIRMED: a transient fetch failure is never retried

**Where:** fetch failure at `2944-2951`; `retry_at` was cleared at `2895`.

- A network error or fetch timeout increments `failures` but arms no retry, so the state is stranded until the relay redelivers it or a newer state arrives. With a stable relay connection, nothing redelivers.
- This was already the behaviour in round 1, but it is now inconsistent with publish failures, which do retry.
- A fetch rejected for size ("grew past the size cap", `fetch_inner`) is correctly left unretried.

**Fix:** set `retry_at` with the same floor for fetch errors other than the size cap.

### R5 — Low, PLAUSIBLE: young quarantine leftovers can push a store into a terminal ceiling refusal

**Where:** `2116-2127`, together with `sweep_stale_quarantines` (`~1850`).

- After a killed push (timeout or growth cap), the retry comes about 2 s later.
- The quarantine is younger than `PUBLISH_TIMEOUT` (120 s), so the sweep keeps it, and the baseline counts it.
- A few killed pushes in a row can lift the baseline past `MAX_SERVED_BYTES` even though the store's real content is under it.
- That refusal is terminal, so the mirror is stranded until a newer state. A later publication's stale sweep would have cleared the leftovers.

**Fix, either of:**
- Exclude quarantine directories from the ceiling measurement.
- Make the ceiling refusal retryable once when quarantines are present.

### I1 — Informational: a stale lease is retryable, so a ref moved behind the engine is overwritten about 2 s later

A stale lease shows up as a porcelain `!`, which goes through `porcelain_rejection` → `From<String>` and so is retryable (`2160-2166`). The retry re-reads `local_heads` and leases against the new value. A branch moved by `merge_pull` or `web_edit` on the mirror's served store is then overwritten on the retry, not left in place.

- This is correct for a mirror: upstream is authoritative, and the lease guarantees atomicity, not preservation.
- But `a_ref_moved_behind_the_engine_before_publish_is_not_overwritten` describes only the first attempt.
- **Fix:** say this in the `RefUpdate::expected` and `PublishError` rustdoc.

### I2 — Informational: a persistent lock failure re-arms the retry every ~1 s

When the lock fails persistently (for example, the lock directory is not writable), `apply_locked` returns the stored record (`2798-2815`). If `newest_seen_id` equals the event, `drain` re-arms with a wait of at least `MIN_REQUEUE_MS`. That is cheap (an open that fails), but it never stops. Acceptable; it is noted only because it is unbounded.

## Answers to the brief

### 1. Retry versus replay

**Does a retry reuse an earlier result?** No; it starts from scratch.
- The retry puts the event itself back in the inbox (`2618-2623`), and it then goes through the full path: `on_event` → `apply_locked`.
- That path re-runs `prepare` (`2817`), re-reads `local_heads` for a fresh lease (`2823-2827`), and re-plans against `last_event`, which is still S1 (`2886-2893`).
- It re-fetches (`2943`), re-judges every branch in the candidate (`2954-2974`), and builds new `RefUpdate`s.
- Nothing from the failed attempt is reused.

**Can anything older than `newest_seen_at` be applied?** No.
- The replay check (`2844-2866`) runs on every apply, including retries and wakes.
- A retry is armed only when `rec.newest_seen_id == ev.id` (`2671`), so only the newest state can be retried.
- The pending event's own retry passes the tie check, because it requires `ev.id != newest_seen_id` (`2846`).
- `newest_seen_*` also advances on the `AwaitsMigration` skip (`2867-2884`), so an older state cannot slip in after migration either.

**Do the drops lose a state that should be applied?** Only through the R3 race. Otherwise, a state that loses to the pending one by `supersedes` (`2348-2350`, `2648-2655`) is one the replay check would refuse anyway:
- an older one;
- the same event redelivered;
- one in the same second with a higher id. A "pending state with new content" is a new event with a new id; it supersedes only with a later `created_at` or a lower id, which matches the NIP-01 rule the replay check uses.

A newer state clears `pending` and is applied at once (`2656`). An announcement is applied and the pending retry is kept (`2663-2664`).

### 2. Locks while waiting

- **While waiting:** nothing is held. The wait is in `drain` (`2613-2628`), outside `on_event`, so neither the in-process mutex nor the file lock is held; the permit was dropped at `2662`.
- **Wake path:** it goes back through `on_event`, which takes the mutex (`2505-2506`) and then `lock_repo` (`2507`), once each.
- **Lost wakeups:** the `Notify` future is registered with `enable()` before the inbox is read (`2601-2603`), so none are lost.
- **Fan-out:** `notify_waiters()` (`2567`) wakes every waiting worker. Each finds its own slot empty and sleeps again until the same `due`. That is cheap, and bounded by `MAX_FOLLOW` (50).

### 3. Retry classification

| Failure | Where | Classified | Right? |
|---|---|---|---|
| stale lease | porcelain `!` → `From<String>`, `2160-2166` | retryable | Yes. Served moved, and a retry leases on the fresh value (I1). |
| receiver ref lock | receive-pack stderr → `From<String>` | retryable | Yes. |
| past the ceiling | `2123-2127` | terminal | Yes, but R5 can trigger it falsely. Check before fetching, to save a pointless 512 MiB fetch on every state of a store already past the ceiling. |
| publish timeout | `run_in` "git timed out" → `From<String>` | retryable | Mostly. Repeated timeouts on the same state should become terminal (R2). |
| growth cap during publish | `run_in` "grew past the size cap" → `From<String>` | retryable | No (R2). |
| oversize fetch | `fetch_inner` → fetch-failure path, `2944-2951` | not retried | Yes. |
| fetch failure (network, timeout) | `2944-2951` | not retried | No: transient but stranded (R4). |
| `AwaitsMigration` | `2870-2884`; in publish, terminal at `2080-2081` | skipped, not retried, no fetch | Yes. |
| `mark_clean_store` command error | `?` at `2080` | retryable | No (R2). |
| alternates present | `2084-2088` | terminal | Yes. |
| ref, sha or lease invalid; duplicate branch; unsafe key; non-UTF-8 path | `2059-2073`, `2089-2091` | terminal | Yes. |
| no served repository | `2075-2077` | retryable | Yes: the retry's apply materialises it (`2931`). |

### 4. Ceiling overshoot

Neither the rustdoc nor the CHANGELOG says so.
- `MAX_SERVED_BYTES` rustdoc (`121-126`): "this ceiling is what bounds its total".
- CHANGELOG, Unreleased (`CHANGELOG.md:101-104`): "a new ceiling (`MAX_SERVED_BYTES`, 2 GiB) bounds its total".

Because the check is made before the push (`2116-2127`) and the push may add up to `MAX_REPO_BYTES` (`2128`), the real bound is about `MAX_SERVED_BYTES + MAX_REPO_BYTES` (2.5 GiB) per store.

The same CHANGELOG bullet also says "Leftover receive-pack quarantine directories are removed". Since round 3, only leftovers older than `PUBLISH_TIMEOUT` are removed, and only by a later publication.

**Fix:** reword both. For example: "refuses publication once the store is past `MAX_SERVED_BYTES`; one publication may still add up to `MAX_REPO_BYTES`, so a store can reach `MAX_SERVED_BYTES + MAX_REPO_BYTES`".

### 5. Other items checked and found sound

- **Path-boundary redaction** (`1906-1947`): sound.
  - Specific paths are replaced before roots. The bare plugin directory is never replaced.
  - A match is skipped when the character before or after it is a segment character, which leaves URLs and longer names alone. A trailing `/` still redacts, so `'<served>/./refs/heads/main.lock'` is handled.
  - A match followed by `.` (for example `name.git.lock`) is skipped at the served level, but the `repo_root` pass then catches it and exposes only `<owner>/<name>`, which is public anyway.
  - Two limits remain, both noted before:
    - a canonicalised path behind a symlinked `data_root` will not match;
    - the upstream's `remote:` text is still stored verbatim.
- **`PublishError` wrapper** (`2300-2313`): redacts the reason and keeps the classification.
- **`AwaitsMigration` in publish**: fails closed (`2080-2081`), with `prepare` errors treated as `Ready` in `prepare_served` (`2747-2751`). This is safe because publish re-runs `mark_clean_store` and refuses. The cost of that re-run is R2.
- **Wake handling:** an announcement during a pending retry updates the clone URL before the retry runs, and the retry then uses it. This is correct.
- **The replayed-retry fix:** `newest_seen_id == ev.id` (`2671`) plus the `MIN_REQUEUE_MS` clamp (`2675`) removes the 50 ms spin reported as RED in the round 2 report.
- **Unchanged from earlier reviews, and still sound:** the hardened push, `RECEIVE_PACK`, the per-ref leases with no `+`, and the explicit `--git-dir`.
