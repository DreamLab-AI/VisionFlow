# Commit B2 fix round 1 (tests first)

**Same rules as `b2-brief.md`.** Evidence goes in `b2/fix1/`. Include an interdiff against `b2/after-b2.patch`, write the report to `b2/fix1/report.md`, and save `b2/fix1/after-fix1.patch` (run `git add -N` on every new file before you diff). Run all nine gates with `OUT=b2/fix1/gates.txt bash b1/gates.sh`. Capture each test red, then green, and run a mutation per fix.

The sources are `review-b2-security.md` (F1–F3, Opus) and `audit-b2-codex.txt` (Codex).

## Must fix

1. **F1: a remote-reachable leak.** A fork whose `forge.parent` fails `valid_owner` or `valid_repo` (for example a trailing `.git`) is never swept or gated, and its ancestor gets stamped.
   - For migration and for the incomplete-pass rule in `lib.rs`, parse lineage with the validator the readers use (`valid_name_segment` for both parts).
   - Treat a `forge.parent` key that is **present but unparseable** as a fork of unknown lineage: give it the orphan treatment (check it against every mirror-class report), gate it while the pass is incomplete, and never let it allow stamping.
   - Make `h_api_fork` refuse to write a lineage the parser would reject.
   - **Fixture:** a legacy mirror named `foo.git` (served as `repos/<hex>/foo.git.git`), forked to `carol/foo.git` with the production `build_fork_at`. Held must be denied on carol's raw, HTML and archive readers. Capture it red first.
2. **F2: objects pruned from the parent but kept in a fork's hardlinked packs.** For a fork whose lineage reaches a mirror-class ancestor with a report, also trigger when the fork holds **any object its own refs don't reach**.
   - Compare `cat-file --batch-all-objects --batch-check` with `rev-list --objects --all` (replace objects and grafts disabled), streamed and capped like `outside_roots`.
   - If either walk fails, the result is a gate or an incomplete pass, never "clean".
   - **Fixture:** a fork taken earlier keeps an object that gc pruned from the parent, and the object isn't in the parent's `.dropped`.
3. **F3:** on any `plan.gate`, block all stamping (`blocked.insert(String::new())`), not only the ids collected so far.
4. **Codex Medium 2:** a journalled ancestor fork under **any** owner, not only a 64-hex owner, gates its descendants (`migrate_fork.rs` `plan_for`, around line 224).
5. **Codex fail-open item 3:** run the gated-plan check **before** the `served_is_ours` resume path (`migrate.rs:~1573` versus `~1631`). A crash-resumed fork with a gated ancestor must not have its journal removed.
6. **Claim and CHANGELOG:**
   - The claim covers objects present at the time of migration. A fork owner who pushes held bytes they kept back into their own fork after migration is re-publishing content they hold. That's outside the claim, the same as risk 4. Say so.
   - Correct the "any ref reaching held is dropped" wording. Overlapping ancestor roots can instead fail the whole rebuild and leave it gated (Codex Medium 3). That's an availability residual, recorded rather than fixed.

## Record, don't fix

- **Codex Medium 3:** stale-root availability.
- **Codex's filesystem-writer items:** report deletion or truncation, counts not checked on read, symlinked owner or recovery directories in the tmp sweep, and timestamp collisions. All are outside the threat model under decision 32.
