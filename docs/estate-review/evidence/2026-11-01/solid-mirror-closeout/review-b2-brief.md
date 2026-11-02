# Review brief: Commit B2 (fork migration)

**Rules:** read-only. Do not run cargo, edit anything or create worktrees. The lead runs the gates.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-fork-sweep`, uncommitted on top of local main `8ba3372` (Commits A and B1). Diff: `b2/after-b2.patch` (`git diff 8ba3372`; the new files are `migrate_fork.rs` and `migrate_fork_tests.rs`). Worker report: `b2/report.md`. Brief: `b2-brief.md`. Design: `b-design.md`, which covers forks in §2.4, §3, lines 179–184, D1, D4 and risk 4. Decisions 1–33 are in `publication-design.md`.

**Claim to test:** for a remote attacker, once a boot completes, no fork (direct, transitive or followed) of a legacy mirror serves an object from any migrated ancestor's held set. A fork whose ancestor is unmigrated or failed is refused. A fork with no dropped object present is left byte-identical.

**Threat model:** the same as B1 (decision 32). Anyone who can write the plugin directory, recycled process-group IDs, and several instances sharing a plugin directory are out of scope. List such findings separately.

**Check especially:**
1. **Trigger completeness.**
   - Can a fork hold an ancestor's held or dropped object and still be classed clean or skipped? Consider an ancestor whose report is missing, unreadable or stamped `forks_clean_at`, a fork created after stamping, a `forge.parent` naming something other than a store, a parent that was renamed, or a depth limit.
   - Is `.dropped` read completely, or can a truncated file under-trigger?
2. **The inherited-held filter.** The filter runs `rev-list --objects R --not <present ancestor roots>`.
   - Can a fork ref reach held through objects that are reachable from ancestor roots but still held?
   - Can it reach held through grafts, replace objects, or `refs/forge/heads/*`?
   - Does verify check the fork's staging against the inherited H?
3. **Fail-open paths.** Look at every early return in the fork pass and `migrate_store` for forks: is a journal, or an incomplete pass, always in force? Pay particular attention to the review-round guard (a fork whose plan is gated is refused).
4. **The `lib.rs` incomplete-pass fork rule.** Which owners does it gate now, and can a fork under a non-64-hex owner (a normal user) be served while the pass is incomplete?
5. **Cycles and transitivity.** Is detection order-independent, so that a grandchild is never examined before its parent's report exists?
6. **The `.fork-*.tmp` sweep:** does it only ever move things, never delete them, and does it stay within `repos/<owner>/`?
7. **CHANGELOG accuracy.**

**Output:** a severity-ranked list with `file:line` anchors and a concrete failure scenario for each. Mark each finding CONFIRMED or PLAUSIBLE, then give an overall verdict on the claim.
