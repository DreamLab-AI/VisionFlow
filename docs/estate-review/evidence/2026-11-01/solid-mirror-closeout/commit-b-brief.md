# Commit B brief: migration of pre-release served stores (SP-GIT-02)

**Base:** branch `fix/mirror-publication-isolation`, at the Commit A SHA in the handoff, in worktree `~/workspace/wt/solid-pod-rs-mirror-publication`.

**Process:** as for Commit A.
- A system-architect interrogates the design and the advisor reviews it before anything is locked.
- Tests are written first, captured red, then turned green.
- One cargo-running worker at a time, with `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- All nine gates (see `lead-gates-a8.sh`).
- Security and code reviews (Opus), plus cross-vendor Codex, using the stdin-bundle script pattern: `codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only -`.
- Every brief goes in a file, because inline messages are cut off at roughly 4 KB.

## What Commit A left for B

1. **Legacy served stores.** A store from before this release carries `refs/mirror` and `hideRefs`, and may hold held objects. Today `prepare` refuses it, so it stays at 503 and `awaits_migration` is set.
   - Migration rebuilds it from its approved public roots into staging under `mirror-work/`, then renames it into `repos/`.
   - It copies every ref except `refs/mirror/*` (decision 2).
   - Recovery copies are kept in `plugin_dir/mirror-recovery/`, outside `repos/`.
   - Migration must **clear `awaits_migration`** and write the marker.
2. **Migration runs before the engine serves.**
   - It happens in `ForgeService::new` or server `forge_service` startup, before any request reaches a legacy store.
   - Use `spawn_blocking` or `block_in_place` if git runs synchronously on a runtime thread (advisor item 6).
   - Review `resolve_repo` and `h_git_smart`.
3. **Forks of legacy mirrors.**
   - `build_fork` uses `git clone --local --bare`, which hardlinks the whole object store, held objects included.
   - Such a fork has no record and no traces, and may already carry a marker (decision 19).
   - Find these forks through `forge.parent` in their config, **whatever their marker**, and rebuild or sweep them.
4. **Copying PR heads and notes (decision 14).** Decide whether migration copies `refs/forge/heads/*` (open PRs) and notes.
   - Objects reachable only through them are gated today (decisions 13 and 18).
   - PR heads already fetched into legacy stores may carry foreign or held objects.
5. **Records cannot detect lost provenance.**
   - Consider a legacy store whose record was deleted and whose traces were all removed.
   - A3 marks it as a first follow (decision 18), so B cannot find it through records.
   - Decide how to detect it: for example, scan any unmarked store that has a candidate under `mirror-work/` or a recovery entry. Otherwise, document an operator runbook.
6. **Unreadable records (A6/A7)** are left in place and the store awaits migration. B defines what the operator does with them and how migration treats them.

## Acceptance (from the handoff and the design)

- Restart and migration of a pre-release store: held SHAs return 404 after the restart on raw, HTML, JSON, archive and smart-HTTP; approved content and owner tags survive.
- No request reaches a legacy store before migration completes.
- Forks of legacy mirrors are swept.
- Recovery data stays outside the served namespace.
- Migration can be interrupted and resumed.
- Hidden-ref deletion plus GC is **not** acceptance evidence.

## B2 additions (from the B1 reviews and advisor, 2026-11-02)

- Forks whose creation overlapped journal creation when several instances share a plugin directory.
- Operator recourse for stores stuck at the attempt limit, over 512 MiB, or behind a permanently unreadable owner directory (which today makes every boot incomplete, so every 64-hex owner serves 503 after every restart).
- The engine's race between process exit and the size check, noted by the B1 worker.
- Codex items only partly done: ancestor-directory fsync and descendant-process cleanup on normal completion.
- The B1 base for B2 is solid-pod-rs main 8ba3372 (B1 = bcaf9cf6).
