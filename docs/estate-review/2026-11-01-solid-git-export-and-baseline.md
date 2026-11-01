---
title: Solid Pod Git export and baseline evidence boundaries
date: 2026-11-01
status: implementation-validation
type: explanation
---

# Solid Pod Git export and baseline evidence boundaries

This assessment follows the Git disclosure hypothesis from the ACL repair
review and checks ADR-2001's documentation contract. Production source is
`b644e55c76d6e89b0e41ec1c7aa44510b20ccf68`. The concurrent parity worktree is
changing provisioning and PATCH parsing; those changes are outside this review.

## Git transport: source finding and acceptance boundary

`handle_git` builds its service root from `data_root/{pod}`, but passes the
original request path to `GitHttpService`. That service extracts the pod slug
and joins it to its root again. For `/alice/info/refs`, the candidate repository
is therefore `data_root/alice/alice`, not the normal `data_root/alice` location
that `git_mark_write` commits into. A missing nested repository stops delivery
before the CGI runs. This narrows the earlier reviewer's exposure claim.

The same route authorises only Read on `/alice/`. If the selected repository
exists, the service exports its objects without evaluating the ACL on each
child. Fixing the root mapping alone would make the normal history reachable
under that same insufficient permission check. Routing and export authority
must be repaired together. Root Read, root Control and ownership of all past
content are different contracts; no new authority policy is inferred here.

A disposable native probe is compiling; no runtime result is recorded yet. It uses only synthetic bytes in a fresh
filesystem pod, NIP-98 owner writes, local Git and an empty receiving repository.
It will distinguish ordinary routing, a correctly rooted transport control and
a nested repository. No runtime disclosure claim is made until its terminal
result and decoded pack assertions have been checked.

## ADR-2001 section assessment

| Section | Disposition |
| --- | --- |
| Context | Historical documentation drift is the rationale for consolidation. Current baseline summaries still mix old version and consumer facts with later scoped amendments; their presence does not prove current implementation or deployment. |
| Decision | Baseline, three-axis ledger, template, preamble and frozen archive exist. The local generator validates required fields, enumerations and reciprocal supersession. It does not resolve `verified_commit` or verify cited source behaviour. |
| Consequences | Separating historical records from governing invariants remains useful. A visible revision stamp and honest status labels require source review; structural validation alone cannot prevent overclaiming. Current consumer and deployment obligations remain open. |
| Verification | The current index check passes for eleven ADRs. All five archived records match their bytes before and at the actual archive cut, and remain unchanged. Historic source checks at `0ccad60` are not rerun or promoted into a current whole-system acceptance claim. |
| Review qualification, 2026-11-01 (isolated source branch) | Documentation commit `d40e97a` records the corrected archive boundary and the limits of structural validation. No runtime implementation or deployment status is advanced. |
| Closeout extension, 2026-09-04 | Source, consumed package, feature configuration and running artefact must be bound independently. The current Git path check demonstrates why component presence and route admission cannot substitute for that composed evidence. Cross-service acceptance remains open. |

Archive identity was checked at the actual move commit
`e093e88a09479b913b7be77d49ef6931c82627e0`, including its parent. The first
attempt used `0ccad60`, which is the code baseline named by the ADR, not the
archive-moving commit. Its missing-path results are retained as an invalid
comparison boundary; they do not show that archived files changed. All five
corrected comparisons are byte-identical before/after the cut and at current HEAD.

A temporary copy of ADR-2001 with `verified_commit: not-a-git-revision` still
passes the index check. This records the check's structural scope, not a claim
that its documented schema validation fails. Freshness and consumer acceptance
need the estate's existing evidence checks; they should not be implemented as
another independent protocol checker in this repository.

## Diagram review

SP-06.12 now shows the repeated path component and the missing-repository
branch at its existing source stamp. SP-14 separates current repository
selection from export authority. Thirteen diagrams render across both topics;
only SP-06.12 and the two SP-14 panels have been visually inspected and assessed
in this checkpoint. The gates resolve 123 historical and thirteen current
citations without warnings or fallback. One JEV advisory on unchanged SP-06.3
remains outside this selected-panel review. The new source panels are unchanged
by the documentation-only branch commit. Ledger credit is not yet published.

## External contract comparison

Git's HTTP backend combines its configured repository root with the request
path. The pod adapter and its framework-independent service must agree about
which component has already been removed. [Git HTTP backend](https://git-scm.com/docs/git-http-backend)

Git also warns that namespaces do not provide read isolation within a shared
repository. That supports treating history export as authority over the whole
repository object store; filtering a displayed file list or a few refs is not
an adequate substitute. [Git namespace security](https://git-scm.com/docs/gitnamespaces)

The practical product requirement is a usable private history export with an
explicit authority model, while ordinary public resource discovery remains
available. Acceptance must include a normal pod, a nested repository, private
children, historical versions and both permitted and denied callers. The forge
has its own prefix and repository-authorisation adapter; this observation does
not establish a forge disclosure or certify that separate path.
