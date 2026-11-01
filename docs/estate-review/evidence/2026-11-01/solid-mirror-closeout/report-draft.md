---
title: Solid mirror publication and repository read paths
date: 2026-11-01
status: reviewed-open-findings
type: explanation
---

# Solid mirror publication and repository read paths

This assessment follows the open object-publication boundary in SP-GIT-02 at
Solid `31fe89856d0cddcf855e3656eb498ddb1df66c9c`. The earlier repair prevents
known hidden-tip requests through the shared Git CGI. Forge also reads the
repository through Git porcelain, outside that CGI policy. This is a continuation
of the existing finding, not another distinct issue added to the estate total.

## Source trace

`Mirror::on_event` verifies the event before applying it. The production
`GitCli` fetches remote branches into `refs/mirror/*` in the served bare
repository. `branch_verdict` then decides whether each public branch advances.
A marked candidate without the committed trail is held, but a successful fetch
has already populated the shared object store. Ref hiding changes discovery;
it does not change what local `cat-file` can read.

| Read surface | Source trace at the reviewed revision | Evidence boundary |
| --- | --- | --- |
| Raw file | `lib.rs::h_raw` resolves the repository and calls `browse::read_blob` with the caller's revision. `valid_rev` accepts a commit hash; the blob reader uses that revision in its Git object specification. | Candidate for the native lifecycle regression. |
| HTML tree, blob and commit history | `h_tree`, `h_blob`, `h_commits` and `h_commit` address the same repository, with no publication-membership check between resolving the repository and reading the revision. | Source-traced; each response needs its own runtime assertion. |
| JSON tree, blob and commits | `api.rs::h_api_repo_action` dispatches to handlers that resolve the caller's revision against the same object store. | Source-traced; not inferred from the raw-file result. |
| Archives | `releases.rs::h_archive` resolves a caller-supplied revision and invokes `git archive` on the same repository. | Source-traced; no decoded archive fixture in this checkpoint. |
| Git smart HTTP | `h_git_smart` calls the shared service with its repaired unadvertised-object settings. | Earlier regressions cover direct hidden-tip requests; they do not cover Forge's porcelain reads. |

The default Forge configuration has `private_repos=false`. Enabling private
repositories changes who may read a repository; it does not implement approval
of each fetched commit. The server adapter calls `ForgeService::handle` after
identity resolution. This source trace does not establish that a deployed
server has a configured follow entry or has fetched a held commit.

## Publication design control and existing data

A local Git subprocess control fetches two diverging branches into a private
staging repository, then pushes only the approved tip into a fresh served
repository. The approved file is readable and the unrelated held object is
absent. A second control starts from an old mirror containing both branches,
deletes its hidden refs, then adopts the same approved-only push. The held
file remains readable by hash. Both controls pass under Git 2.56.0; they use
synthetic content and temporary repositories.

This demonstrates why a change to future fetches alone is insufficient. It
is a design control, not an implemented Forge repair, and it does not prove
concurrency, cancellation, recovery, size-limit or migration behaviour. The
existing `push_guard.rs` uses a throw-away object directory for incoming pack
inspection, and all these adapters already share `hardened_git_async`. Reuse
those established process and isolation conventions where their contracts fit;
the push inspector is private and does not itself implement mirror publication.

Git explicitly distinguishes fetch validation from receive-pack quarantine.
For internal mirrors it describes a two-stage arrangement in which consumers
read the repository populated by a second push. Receive-pack quarantines new
objects until its pre-receive phase succeeds. These are useful primitives, but
Git's acceptance of a pack does not replace the application's mark verdict.
[Git transfer validation](https://git-scm.com/docs/git-config#Documentation/git-config.txt-transferfsckObjects),
[receive-pack quarantine](https://git-scm.com/docs/git-receive-pack#_quarantine_environment).

The remaining acceptance conditions are:

- Keep candidate objects outside every served repository during fetch and
  validation, including failed, cancelled and oversized fetches.
- Publish only approved commit closures; preserve legitimate history, approved
  branches and normal Git, raw, HTML, JSON and archive reads.
- Exercise mixed approved and held branches, multiple updates, restart and
  concurrent reads; a check only after a held verdict misses the fetch window.
- Define recovery and migration for existing repositories containing held
  objects. Preserve approved refs and metadata, provide recoverable handling
  of old storage, and verify denied reads after migration.
- Check the configured consumer and running artefact separately from the
  producer change. Do not infer deployment from `forge-full` compilation.

## ADR implications and scope

ADR-2009's ref-hiding claim and ADR-2010's fail-closed wording need the existing
qualification extended to the porcelain read paths. Their historical release
and activation observations remain dated. The branch verdict also explicitly
does not consult the chain: a matching committed trail is not confirmation of
an on-chain mark. A successful publication repair would not change that limit.

This checkpoint assesses the mirror publication clauses and their read-side
composition. It does not assess every comparator, anchoring, UI, egress or
release obligation in those compound ADRs, so it earns no whole-ADR credit.

## Utility and comparison

Forgejo documents periodic pull and push mirroring of branches, tags and
commits. That is a useful reference for ordinary repository synchronisation.
[Forgejo repository mirrors](https://forgejo.org/docs/latest/user/repo-mirror/).

VisionFlow's additional opportunity is a verifiable publication decision tied
to the followed identity and the declared trail. For an industrial maintenance
library, users should keep reading an approved procedure while a replacement
is held, and receive a clear reason for that hold. Downloads by an old link or
known hash must honour the same publication boundary. This is a product
recommendation inferred from the current contract and the comparison, not a
claim that the current mirror delivers industrial approval or chain finality.
