---
title: Solid mirror publication isolation and migration
date: 2026-11-02
status: repaired-local-main-open-deployment
type: explanation
supersedes: evidence/2026-11-01/solid-mirror-closeout/report-draft.md
---

# Solid mirror publication isolation and migration

**Outcome:** the mirror publication boundary that SP-GIT-02 left open is now
repaired on local solid-pod-rs `main` at `9b6143c`. A held mirror commit is no
longer readable through Forge, and stores and forks written by earlier
releases are migrated at boot. Nothing is pushed, released or deployed. This
continues the existing finding; it is not another distinct issue in the
estate total.

The claim, stated for remote attackers: no object that was present when
migration ran, and is held (not reachable from any approved branch or tag), is
served by a mirror or by a fork of one. Deployment evidence, an MSRV 1.89
build and consumer adoption remain unverified.

## What was wrong

At `31fe89856d0c`, `GitCli::fetch` put every upstream object under
`refs/mirror/*` in the same bare store that Forge serves. `branch_verdict`
only decided whether a public branch moved. Forge's raw, HTML, JSON and
archive readers call local Git with the caller's revision, so they never pass
through the Git HTTP backend's unadvertised-object controls. A native
regression, `held_mirror_commit_is_not_published_through_raw_route`, returned
200 with the held content where 404 was expected (exit 101, reproduced at the
start of this work).

## What changed

| Commit | Revision | Change |
| --- | --- | --- |
| A | `30135fec` (merged `d6133e7`) | Fetch and validate in a private candidate store under `mirror-work/`, outside `repos/`. Publish only approved commits into the served store by explicit SHA, one `--force-with-lease` per ref (no `+`), `--atomic`, and a fixed hardened receive-pack. A publication marker and a reader gate cover every serving route. Writes into a mirror are refused with 409. |
| B1 | `bcaf9cf6` (merged `8ba3372`) | Boot-time migration of pre-release mirror stores: rebuild from approved branches and tags only, verify object by object, swap behind a journal. Every journal gates readers while it exists, and an incomplete pass gates every mirror owner. Fixes the smart-HTTP pass-through for invalid path segments (such as `.fork-x.tmp`) that Commit A introduced. |
| B2 | `ff8048b` (merged `9b6143c`) | Boot-time migration of forks of legacy mirrors: transitive discovery through `forge.parent`, a rebuild triggered by an ancestor's dropped objects or by objects the fork's own refs do not reach, and ref selection that drops and reports anything reaching held. |

Two exposures that were live on local main before B1 are closed. A legacy
store whose mirror record was lost had no gate and served held objects with
200; that predates Commit A. The smart-HTTP pass-through came from Commit A
itself and was fixed in B1 before any push or release; it is recorded under
SP-GIT-02 and not counted separately.

## Evidence

The classes are kept separate and none is promoted to another.

- **Native route regressions.** Held-object denial on raw, HTML, JSON,
  archive and smart-HTTP; mixed approved and held branches; repeated updates;
  restart; migration of a pre-repair store; fork migration. Migration and fork
  fixtures are built so an in-place `gc` cleanup would fail them. Each fix
  round's tests were captured red first, and mutations that remove each fix
  turn the expected tests red. All nine gates (fmt; forge tests under default,
  `mirror` and `mirror,anchoring,announce`; server `forge-full`; Clippy with
  warnings denied on forge and server; rustdoc) passed on each exact committed
  tree, with sources touched so Cargo rebuilt them.
- **Subprocess controls.** Scratch Git 2.56 controls established that a `+`
  refspec overrides `--force-with-lease`, that `-c` settings do not reach the
  receiving side of a local push, that every clone variant carries held
  objects across, and that detached `HEAD` and broken repositories return
  distinct `symbolic-ref` exit codes.
- **Source tracing and review.** Opus security and correctness reviews and
  Codex `gpt-6-astra` cross-vendor audits, each round. The last verdicts were
  PARTLY SUPPORTED, with every residual named below. Some later fix rounds
  were reviewed by Codex only.
- **Deployment evidence.** None.

The smart-HTTP denial test is defence in depth only: the earlier `1a66c2d`
repair already refuses that request, so it would pass without these fixes.

Evidence, decisions 1 to 39 and checksums are in
[`evidence/2026-11-01/solid-mirror-closeout/`](evidence/2026-11-01/solid-mirror-closeout/)
(`SHA256SUMS`, `SHA256SUMS-commit-a`, `-commit-b1`, `-commit-b2`).

## Acceptance

Met on local main:

- Candidate objects stay outside every served repository during fetch and
  validation, including failed, cancelled and oversized fetches.
- Only approved commit closures are published; approved branches and normal
  Git, raw, HTML, JSON and archive reads still work.
- Mixed branches, repeated updates, restart and concurrent reads are
  exercised.
- Existing stores are migrated with recoverable handling of old storage, and
  denied reads are verified after migration, for mirrors and their forks.

Not met: the configured consumer and running artefact have not been checked,
and the MSRV 1.89 build has not been run.

## Residuals

Out of the threat model: anyone who can write the plugin directory (record,
report or journal tampering, symlinked owner or recovery directories),
several server instances sharing one plugin directory, recycled
process-group IDs, and a fork owner re-pushing held bytes they already hold.

Open, availability and durability only (queued as B3 under SP-GIT-02):

- No operator recourse for a store stuck at the attempt limit, over 512 MiB,
  or behind an unreadable owner directory. A permanently unreadable owner
  directory makes every boot incomplete and gates every 64-hex owner.
- Ancestor-directory fsync and descendant-process cleanup are only partly
  done; the engine has a race between process exit and its size check.
- Overlapping ancestor roots can fail a whole fork rebuild and leave the fork
  gated.

## ADR implications and scope

ADR-2009's reliance on ref hiding as the withholding mechanism is superseded
in practice by store isolation: the served store no longer contains held
objects, so hidden refs are no longer the boundary. ADR-2010's fail-closed
wording now matches the reader gate. Both ADRs still need their text updated
and are compound, so this assessment earns no whole-ADR credit. The branch
verdict still does not consult the chain; a matching trail is not
confirmation of an on-chain mark, and this repair does not change that.

## Utility and comparison

Forgejo documents periodic pull and push mirroring of branches, tags and
commits ([Forgejo repository mirrors](https://forgejo.org/docs/latest/user/repo-mirror/)).
That is the reference for ordinary synchronisation. What VisionFlow adds is a
publication decision tied to the followed identity and the declared trail.
After this repair, readers keep getting the approved version while a
replacement is held, and an old link or a known hash honours the same
boundary. The remaining product gap is the reason for a hold: readers see a
503 or 404, not why the content is being withheld.
