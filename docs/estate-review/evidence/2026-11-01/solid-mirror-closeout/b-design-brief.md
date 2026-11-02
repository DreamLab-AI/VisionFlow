# Commit B design brief (system-architect, read-only)

You are designing, not implementing, Commit B of SP-GIT-02 in solid-pod-rs: the migration of served stores created before this release. Your report goes to the delegating lead, who puts it to an advisor before locking it.

**Rules:**
- Do not edit the repo or create worktrees.
- You may run scratch git controls under `/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad/b-ctl/` (git 2.56 is installed). Do not run cargo.
- If you decline, begin the report with `DECLINED:`.

**Code:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication` on branch `fix/mirror-migration`, at `d6133e7` (Solid main with Commit A merged). The relevant files are:
- `crates/solid-pod-rs-forge/src/mirror.rs`: `GitCli`, `prepare`, `legacy_traces`, `holds_unreachable_objects`, `MirrorStore`, `StoredRecord`, `awaits_migration`
- `crates/solid-pod-rs-forge/src/lib.rs`: `ForgeService::new`, `resolve_repo`, `gated_git_dir`, `has_mirror_record`, `h_git_smart`
- `crates/solid-pod-rs-forge/src/forks.rs`: `build_fork`, `forge.parent`
- the server's `forge_service` startup in `crates/solid-pod-rs-server/src/main.rs`

**Read first** (all in `VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/`):
- `commit-b-brief.md`: the obligations and acceptance criteria. **This is your scope.**
- `publication-design.md`: the Commit A design and locked decisions 1–20, especially 2, 13, 14, 18 and 19.
- `review-a5-security.md`: the reasoning behind R1, I2 and the marker semantics.

## Deliverable

Write the design to `b-design.md` in the same evidence folder. Answer each of these concretely, with file:line anchors:

1. **Legacy detection.** Which stores are migrated?
   - Base the answer on the signals A uses (record, `refs/mirror`, `hideRefs`, `awaits_migration`), plus a decision on the lost-record case (obligation 5).
   - State what is decidable and what needs an operator runbook.
2. **Rebuild.** How exactly does an old served store become a clean one?
   - Which refs are copied: every ref except `refs/mirror/*` (decision 2), and decide `refs/forge/heads/*` and notes (decision 14).
   - How do objects get across without carrying held ones? Use a fresh store plus a push or fetch of the chosen refs only; verify it with a scratch control, including that a hardlinked or `--local` clone would *not* be acceptable.
   - Where do staging, the rename and the recovery copy live? They must sit outside `repos/`.
   - How does the marker get written, and how is `awaits_migration` cleared?
3. **Ordering and startup.**
   - Migration must finish before any request reaches a legacy store. Is it synchronous in `ForgeService::new`, or is it a gate that serves 503 until the store is done?
   - If synchronous, how does it avoid blocking the runtime (`spawn_blocking` or `block_in_place`)?
   - How do interruption, resumption and concurrent instances work under the existing `lock_repo` file lock?
4. **Forks** (obligation 3). Find forks of legacy mirrors through `forge.parent`, whatever their marker.
   - Decide whether to rebuild them the same way or sweep their unreachable objects.
   - Prove with a scratch control that the method removes hardlinked held objects.
5. **PR heads already fetched into legacy stores**, and unreadable records (obligation 6).
6. **Failure semantics.** Partial failure must never leave a served store that holds held objects, and must never lose approved data. Define the recovery path.
7. **Test plan.** A numbered list of tests the test-engineer writes first, each naming what it fails on before the fix. Restart-and-migrate tests must check that held SHAs return 404 on raw, HTML, JSON, archive and smart-HTTP, and that approved content and owner tags survive.
8. **Risks and open decisions** for the lead.

Keep the design inside the obligations in `commit-b-brief.md`, and add nothing outside them. When you finish, reply with only the path of `b-design.md`.
