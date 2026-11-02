# Commit A fix round 6: brief (narrow, the final round for Commit A)

**Rules**, the same as round 5:
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`; no other cargo work is running.
- Run all nine gates: fmt; forge tests with `mirror`, with default features, and with `mirror,anchoring,announce`; server tests with `forge-full forge`; clippy `-D warnings` on forge (`mirror`), forge (`mirror,anchoring,announce`) and server (`forge-full`); and `cargo doc --no-deps` with `RUSTDOCFLAGS=-D warnings`.
- Save evidence in `.../solid-mirror-closeout/fix-a6/`: a report, the red logs, the gate logs, `fix-a6.patch` (the full diff against 31fe898) and `fix-a6-interdiff.patch` (the diff against `final-a5.patch`).
- Do not commit, push, or create or remove worktrees.
- Reply with only the path to the report.

**Sources:** `review-a5-security.md` (S1–S7, I1–I5) and `audit-a5-codex.txt` (the last `codex` block).

Do these items and nothing more:

1. **S2.1 and Codex: a corrupt record must fail closed.**
   - Make `MirrorStore::read` distinguish three cases: Missing, Unreadable or invalid, and Valid.
   - **Missing:** keep current behaviour (first follow, no scan).
   - **Unreadable or invalid:** `prepare` returns AwaitsMigration. Never overwrite the file with a fresh `isolated: true` record; leave it in place for the operator and Commit B.
   - Readers and the gate must treat a record that exists but can't be read as a mirror (503), not as absent.
2. **S2.2 and Codex: `legacy_traces` must fail closed on `git config` errors.**
   - Only exit code 1 means "unset".
   - Any other exit code, a timeout or a spawn error counts as a trace, so the store awaits migration.
3. **S4: fetch timeouts must stop after a limit.**
   - Generalise the timeout counter to cover consecutive fetch timeouts of the newest state as well as publish timeouts.
   - Apply the same `MAX_PUBLISH_TIMEOUTS` limit, and rename the constant if that reads better.
4. **S5: "clone URL refused" from `fetch_inner` must be terminal,** on both paths. Use a shared constant compared the same way as `GREW_PAST_CAP`, or a typed error.
5. **Codex C3/R4: a transient DNS failure inside the SSRF resolution is retryable.**
   - It still goes through the timeout and fetch limit from item 3.
   - A resolved address that the policy refuses stays terminal.
6. **S6: persist only AwaitsMigration verdicts backed by positive evidence:**
   - a trace was found;
   - an alternates file exists;
   - the scan reported an unreachable object;
   - an unreadable record (item 1).

   For an I/O error, a scan error or a timeout, skip the state without persisting the verdict and schedule a retry through the normal backoff.
7. **S7: a non-scan `prepare` error is retryable** through `schedule_retry`.
8. **I4 and I5, only if each is a small change:**
   - **I4:** a successful push must not be recorded as a terminal failure.
   - **I5:** the timeout counter must not reset for ever.

   If either is not small, record it in the report instead.
9. **S1 and I3, docs only:**
   - Update the `PUBLICATION_MARKER` rustdoc (`lib.rs` ~98) to the new meaning: "publication allowed; legacy layout not detected". Say that the marker does **not** certify that every object is reachable for a store first followed after this release.
   - Fix the stale comments that I3 lists.

Out of scope, recorded by the lead:
- forks of legacy mirrors (Commit B sweeps them through `forge.parent`, whatever their marker);
- verification CPU budget and scan memory limits (measure first);
- the publication marker being trusted once written;
- the `isolated: true` record reused after a store is replaced.
