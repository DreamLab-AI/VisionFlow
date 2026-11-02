# Code re-review: Commit B1, fix round 1

- **Reviewer:** code-reviewer (correctness focus). Read-only: no cargo was run, and nothing was edited except this file.
- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, `fix/mirror-migration`, uncommitted on `d6133e7`.
- **Read:** `b1/fix1/report.md`, `interdiff-vs-after-c.patch`, current `migrate.rs`, `lib.rs`, `forks.rs`, `migrate_tests.rs`, `CHANGELOG.md`, and my earlier `review-b1-code.md`.
- **Line anchors:** `crates/solid-pod-rs-forge/src/` unless stated otherwise.
- **Labels:**
  - CONFIRMED: traced end to end and still holds after an attempt to falsify it.
  - PLAUSIBLE: I believe it but could not trace it fully.

## Verdict

- **Earlier findings:** every finding I raised is fixed except the t16 part of M2, which the report says it left alone. H1 and H2 are properly fixed: binding by migration id is the right design.
- **One new defect (N1, Medium).** It is a gap left in the fix for Codex #1. The reader gate still trusts a `published` journal without binding it to the served store. Three boot branches leave a forged one in place on every boot. On a lost-record store, the held objects are then served with 200 across restarts.
- **Precondition and severity:** an attacker needs write access under `mirror-work/`, the same capability as the old L1. I rate this Medium because, unlike the documented journal-deletion residual, it survives reboots.
- **Everything else** I checked holds (see "Checked and found sound").

---

## Status of earlier findings

| # | Status | Where |
|---|---|---|
| **H1** crash after the second rename, then gated for good | **FIXED** | `migrate.rs:1291-1297`: `served_is_ours` runs before the attempt limit and before the `(true,true)` intruder branch, then goes to `publish`. `seal` writes `migration-id` before the marker (`:1607-1619`). f5 covers it over three crash states. |
| **H2** resume never wrote `Moved`, and the report was overwritten | **FIXED** | `Moved` is now written on both paths before the second rename (`:1566-1572`). `write_report` never replaces a report: same id stands, another id fails, and a new report goes through `create_exclusive` (`:2223-2233`, `:2283-2287`). Once moved, the old store is read only from recovery (`:1300`). |
| **M1** deviation 5 failed open, and probes short-circuited | **FIXED** | Each signal is evaluated on its own (`:964-1004`), and any positive is `Rebuild` (`:947-953`). An error with no positive goes to a `detection-error` gate journal (`:1249-1251`), re-examined under the lock (`:1262-1290`). The `Failed` resume constraint I raised is handled: only that reason is re-detected, and `rebuild` clears `reason` (`:1477`), so once a rebuild starts the journal is no longer removable. |
| **M2** test strength | **PARTLY FIXED** | Now covered: crash after the second rename (f5), the bad or symlinked journal (f3_a), lock failure (f7), detection error (f2), and 503 after give-up (f6). Still open: **t16 still does not prove the lock** (`migrate_tests.rs:983`; the report says so). The positive `Detected::Finish` path (no journal; marked, flagged, bound report, then finish) has no test: M5b's test only covers the negative. The forged-`published` test covers one branch only (see N1). |
| **L1** dangling-symlink journal opened the gate | **FIXED** | `read_journal` uses `symlink_metadata`, and any non-regular file is `Bad`, which gates (`:628-658`). |
| **L2** lock-failure journal written without the lock | **FIXED** (fail-closed residual accepted) | `gate` uses `create_exclusive` (`:1140-1155`, `:1050-1065`) and never replaces a journal. The leftover case (a peer finished, then this instance gates the clean store) is in the report's residuals. |
| **L3** unbounded lock wait | **FIXED** per store | `take_lock` polls `try_lock` up to `lock_wait` (`:1069-1098`); on timeout it writes a gate journal (`:1229-1237`). The total boot time is still unbounded, and the CHANGELOG now says so (`CHANGELOG.md:95-96`). |
| **L4** budget timeout orphaned helpers | **FIXED** | The `GROUPS` task-local registry (`:1422-1448`) is filled by `run_output` (`:264-265`), `for_each_line` and `run_with_input`. All remaining groups are killed before staging is removed (`:1513-1522`). |
| **L5** non-UTF-8 ref names | **FIXED** | `parse_ref_line` (`:1724-1730`) drops them as `unsafe-name` (`:1793`). The BTreeMap collision is gone, because names are only inserted when valid UTF-8. |
| **L6** CGI suffix and depth mismatch | **FIXED** | `smart_http_path` (`lib.rs:156-166`) and the CGI path rewrite to `/<o>/<n>.git<suffix>` (`lib.rs:772-777`). |
| **L7** CHANGELOG | **FIXED** | The part-1 "not migrated yet" sentence is gone. Post-move failure (404, old store in recovery) is described (`CHANGELOG.md:75-80`), as are detection errors, the incomplete-pass gate and the 512 MiB cap. One nuance under N1: `CHANGELOG.md:55-58` correctly says a `published` regular file does not gate, and that is exactly the gap. |

---

## New findings

### N1 (Medium): a forged `published` journal keeps a lost-record store open across reboots. CONFIRMED (needs write access under `mirror-work/`)

- **Anchors:**
  - `migrate.rs:656`: `Present(j) => j.state != Stage::Published`. The reader gate trusts a regular-file `published` journal without binding it to the served store.
  - The three boot branches that return without rewriting the journal:
    - `migrate.rs:1255-1261`: `contained` fails;
    - `:1301-1304`: `attempts >= MAX_ATTEMPTS`;
    - `:1305-1317`: served and recovery are both present.
  - `lib.rs:653-665`: after the journal check, an unmarked store with no record is not gated.
- **What the fix did, and what it left:**
  - It made `migrate_one` refuse to *finish* a store under a `published` journal that the store does not match (`:1291-1297`).
  - Only `fail()` rewrites such a journal to `failed`, and only on some paths.
  - f3_a's fixture takes one of the paths that rewrites: recovery `sample.git.1` is absent, so the code reaches `(true,false) if was_moved` and then `fail()` (`:1318-1327`).
- **What is new compared with the documented residual: persistence.** The CHANGELOG accepts that deleting a lost-record store's journal serves it until the next boot (`CHANGELOG.md:90-94`). A restart bounds that residual. This one survives every boot.
- **Failure (lost-record legacy store, no record, unmarked):**
  1. Plant `mirror-work/migrations/<o>/<n>.json` as a regular file reading `{"id":"x","ts":1,"class":"mirror","state":"published","recovery":"mirror-recovery/<o>/<n>.git.1","attempts":3}`. Any one of these variants also works:
     - a `recovery` path that fails `contained` (for example `"../x"`);
     - a `recovery` that names an existing directory (for example a real earlier recovery entry) with `attempts` below 3.
  2. Boot. `migrate_one` returns `Outcome::Failed` from one of the three branches and leaves the journal at `published`. `Failed` does not set `migration_incomplete` (`:725-730`).
  3. A reader calls `unpublished_mirror`:
     - `journal_gates` returns false;
     - the store is unmarked;
     - the incomplete flag is false;
     - `has_mirror_record` finds no record, so the gate is not applied.
  4. Raw, HTML, JSON, archive and smart-HTTP readers serve the legacy store, held objects included, with 200. The same happens on every later boot.
- **Related:** `gate`, when an entry already exists (`:1146`), and the `rebuild` journal-write failure (`:1483`) also defer to whatever journal is there. A forged `published` journal survives those paths too.
- **Why Medium:** the precondition is the same as the old L1, which I rated Low: write access under `mirror-work/` as the server user or an operator, not remotely exploitable. I rate this higher for two reasons:
  - this round put forged journals in scope (Codex #1, must-fix 3), and this is the remaining half;
  - unlike L1 and the documented deletion residual, it survives restarts.
- **Fix: either closes the persistence hole.**
  - **Reader side (preferred, one place):** in `journal_gates`, gate a `published` journal unless `served_is_ours(served, &j.id)`.
    - `publish` writes `published` only for a store that is in place (or immediately before the `served_is_ours` re-check at `:1291`), so a genuine `published` journal always matches its store.
    - The cost is one more `lstat` and one small read per request, on that rare state only.
    - It also ends a `published` journal's ability to open anything, so it is simpler to reason about.
  - **Boot side:** in `migrate_one`, rewrite any `published` journal that is not bound to the store to `failed` before every early return.
  - **Neither fix closes the runtime deletion path.** That path is already accepted and documented.
- **Test impact:**
  - Add f3_a variants on the lost-record fixture with `attempts: 3`, and with an existing recovery directory, asserting `assert_gated`. Both should be red today.
  - **The reader-side fix turns t18's `published` row red** (`migrate_tests.rs:1074`). t18 marks its store by hand (`:1047`) and has no `migration-id`, so a journal with id `t18` is unbound. Its fixture needs a `migration-id` file reading `t18`, as t13's gained (`mirror.rs:9091`). The boot-side fix leaves t18 unchanged, because t18 plants the journal after `new`.

### N2 (Low): the doc comment for `unpublished_mirror_error` now heads `smart_http_path`. CONFIRMED

- **Anchor:** `lib.rs:151-152`.
- **What happened:** the new function was inserted between `/// The 503 every reader returns for an unmigrated mirror.` and `fn unpublished_mirror_error`.
- **Effect:** the doc for `smart_http_path` now opens with the wrong sentence, and `unpublished_mirror_error` has no doc. Gate 9 does not catch it because both items are private.

### N3 (Low): copied `refs/replace/<held-sha>` names disclose held object ids. PLAUSIBLE

- **Anchors:** `migrate.rs:1802-1804`, `:1818-1827`.
- **How:** a non-root ref is copied when its *target* reaches nothing held. A legacy store with an owner-planted `refs/replace/<X>`, where `X` is a held commit and the replacement `Y` is approved, has the ref copied.
- **Effect:** the ref advertisement then names `X`. In the served store (readers do not set `GIT_NO_REPLACE_OBJECTS`), `/raw/X/...` resolves to `Y`'s content. No held *content* is served. Only the held id is disclosed, and that id is not reachable from any approved ref.
- **Fix:** drop `refs/replace/*` as `outside-roots` (or a new `replace` reason) unconditionally. Like notes, replacements are meaningless without their target.

---

## Checked and found sound

- **Held definition (R and H):**
  - R is exactly the copied branches and tags with valid SHAs (`:1797-1800`, `:1815-1816`).
  - H is closure(`--all`) minus closure(R), computed in two passes (`:1855-1902`).
  - A non-root ref is copied only if `rev-list <sha> ^R` lists nothing in H (`:1905-1927`). `^R` only ever over-lists, which cannot hide a hit.
  - So the copied set's closure is a subset of closure(R).
- **Verify with H empty or `refs/mirror` gone:**
  - `present_count` runs whatever H holds (`:2031`).
  - `outside_roots` runs unconditionally (`:2042`): every object physically in staging (`cat-file --batch-all-objects`, independent of replace refs) must be in closure(R) in the source. With R empty, any staged object fails.
  - This check also catches a ref added or removed between listing and `held_set`. The push is by explicit SHA (`:1939`), and the tombstone and lock block the forge's own writers meanwhile.
- **Replace refs:**
  - Every closure computation goes through `migration_command` with `GIT_NO_REPLACE_OBJECTS=1` (`:2308-2312`, used by `for_each_line` and `run_with_input`).
  - The `run_in` calls (ref listing and push) are not walks.
  - Whatever the push sends, `outside_roots` checks the bytes actually present.
- **Alternates and shallow sources:**
  - Alternates in the source are read through and copied. Staging is checked for alternates and shared inodes (`:2048-2062`).
  - From a shallow source, `receive-pack` refuses shallow updates by default, so the push fails closed.
- **Detection-error journal:**
  - It is removed only under the lock, by a `detect` that returns `None` (`:1269-1281`), and only for that reason prefix.
  - `rebuild` clears the reason as soon as an attempt starts (`:1477`).
  - `migration_incomplete` is fixed for the life of the service (`lib.rs:286`) and never cleared early.
  - The 64-hex condition matches `detect`'s own scope (`:909`), so it misses no store that migration would have handled.
  - **Blast-radius note, not a defect:** if owners are commonly 64-hex pubkeys, an incomplete pass sends 503 to every unmarked owner repository too, until restart.
- **`served_is_ours` and `migration-id`:**
  - The id is a random v4 uuid held only in the journal and the report, both outside `repos/`.
  - The router hands the CGI only the three smart suffixes (`router.rs:371`), so `migration-id` cannot be fetched through a dumb path.
  - The raw, HTML, JSON and archive readers resolve tree paths, not files in the git directory.
- **`read_regular`:**
  - `lstat`, then `open`, then `fstat` with the same `dev` and `ino` (`:595-625`).
  - A symlink swapped in after the `lstat` resolves to another inode and is refused.
  - The only remaining race is swapping in a FIFO, which blocks `open`. That needs local write access too. It is noted, not reported.
- **`contained`:** recovery and staging must be plain components of the exact shape, and an existing parent must canonicalise to the expected place (`:1162-1217`). A failure gates, except for the `published` case in N1.
- **`smart_http_path`:**
  - Exactly two segments, both passing `valid_name_segment`; nested, dot-led and `%`-encoded paths are refused.
  - The CGI path is rebuilt from the validated parts, and the gate and the write guard use the same `owner` and `name`.
  - `is_write` keys on the suffix and query, which are preserved.
  - Suffix-less pushes now auto-init `<n>.git`, the forge's own layout. Any `repos/<o>/<n>` created by the old behaviour becomes unreachable over smart HTTP. That is a deliberate and documented behaviour change (`CHANGELOG.md:102-103`).
- **Process-group killing:**
  - Groups are created with `process_group(0)`.
  - `note_group` is a no-op outside `GROUPS.scope`, so engine runs are never registered.
  - Normal completion forgets the group.
  - The only way to hit a foreign process is pid reuse: the leader is reaped after `kill_on_drop` and an unrelated process becomes a group leader with the same pid before `kill -- -pid` runs. That window is milliseconds, so I do not report it.
- **Notes dropped (deviation):** I accept it. A notes commit is never in closure(R), so copying `refs/notes/*` would re-open the F2 class (a note can carry any blob). The note is kept in recovery, the report lists it as `outside-roots`, and the CHANGELOG states it.
- **Residual "lock-timeout gate journals are never re-detected":** it is fail-closed. The journal is `failed` with 0 attempts, so readers get 503 (`:656`) and the next boot rebuilds under the lock. The costs are integrity and availability only: a second recovery entry, or non-root refs of a plain 64-hex owner store moved to recovery. Nothing held is exposed. The one exception is a pre-existing forged `published` journal (N1), because `gate` defers to it.
- **Crash-state table, re-traced:**
  - **Crash before the first rename** (`verified`, served present, no recovery): the store is rebuilt from served under the same id, and the report stands.
  - **Crash between the renames** (served absent, recovery present): the store is rebuilt from recovery.
  - **Crash after the second rename**, or after `published`: `served_is_ours`, then publish and finish.
  - **Placement failure:** `fail` keeps `moved: true` (`:1569`, `:1579-1589`), and the next boot rebuilds from recovery.
  - **Interaction with `MAX_ATTEMPTS`:** an attempt counts at start (`:1474-1479`), and in every case the attempt limit does not block finishing a store that is already in place.
- **Fork onto a journalled name:** returns 503 (`forks.rs:286-292`), after `refuse_mirror_write` and before the `dest` existence check.
