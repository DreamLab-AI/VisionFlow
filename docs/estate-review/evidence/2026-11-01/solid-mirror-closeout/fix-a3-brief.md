# Commit A fix round 3: brief

Same rules as round 2:
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- Run all seven gates on the final tree.
- Put the evidence in `fix-a3/`, with a round-3-only interdiff.
- Do not commit, push, or create or remove any worktree.
- Do not change existing assertions.

Your guess 1 (N3) is correct. Guesses 2–4 are NOT in scope: no ADR wording changes, no change to how states are dropped during a retry wait, no further L2 work, and no flip of the listing behaviour.

## Items

1. **Codex High 1: compare/PR injection.**
   - **Where:** `compare.rs` `resolve_head`, roughly lines 520–560.
   - **The bug:** when the base repository is a *marked* mirror and the head names another repository the caller owns, the head commit is fetched into the base's served object store. Any authenticated user can then plant their own objects in a mirror's served store, readable by SHA.
   - **Fix:** if the base repository has a mirror record (marked or not), refuse to fetch a foreign head into it. Return a 4xx or 409 with a clear reason, and fetch nothing.
   - **Check also:** any other path that writes objects or refs into a mirror's served store (`merge.rs` `merge_pull`, PR head refs `refs/forge/heads/*`, release paths). Refuse them all when the repository has a mirror record.
   - **Test:**
     - Set up a marked mirror base and a caller-owned head repo with a unique blob.
     - Call compare, and the PR create/open path.
     - Assert the request is refused, the blob is absent from the served store (`cat-file -e` fails), and no `refs/forge/heads/*` was written.

2. **Codex High 2: `mark_clean_store` stamps stores holding unreachable objects.**
   - **The bug:** it checks only refs and config.
   - **Fix:** before writing the marker, require that every object in the store is reachable from a non-`refs/mirror/*` ref. For example: `git rev-list --objects --all --exclude=refs/mirror/*` compared with `cat-file --batch-all-objects --batch-check`, or the equivalent `fsck --unreachable` restricted to those refs.
   - **Any extra objects:** return the "awaits migration" refusal, the same path as the legacy layout. No marker is written and the store stays gated.
   - **Test:** a store with a mirror record, no `refs/mirror`, no hideRefs, but an unreferenced held blob. Assert no marker is written and readers return 503. A clean store with the same layout and no extra object still gets marked.

3. **Security N3: sweep age.**
   - The pre-push sweep skips quarantines younger than `PUBLISH_TIMEOUT`, by mtime.
   - The sweep after a failed push removes only quarantines created during our push.
   - Backdate planted test quarantines with `File::set_modified`.
   - **Test:** a fresh concurrent quarantine survives both sweeps; an old one is removed.

4. **Codex: two tests to tighten.**
   - `a_receiver_ref_lock_failure_is_recorded_and_lands_only_approved_history`: also assert that the approved object *did* land in served (quarantine migrated), and that the recorded reason names the ref-lock failure. That rules out an earlier failure satisfying the negative assertions.
   - `held_side_commit_is_not_fetchable_over_smart_http`: when `git-http-backend` is missing, fail (panic) instead of printing SKIP and returning.

## Report

Write `fix-a3/fix-a3-report.md`. For each item, give the red output and the fix location, plus the gate results. Reply with only the report path.
