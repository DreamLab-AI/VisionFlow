---
title: Solid Pod Git export and baseline evidence boundaries
date: 2026-11-01
status: reviewed-open-findings
type: explanation
---

# Solid Pod Git export and baseline evidence boundaries

This assessment follows the Git disclosure hypothesis from the ACL repair
review and checks ADR-2001's documentation contract. The reproduced baseline is
`b644e55c76d6e89b0e41ec1c7aa44510b20ccf68`; the direct hidden-object fetch repair
is `1a66c2d9222c52782891938da1b0373a0b41ab28`, merged through documentation
correction `31fe89856d0cddcf855e3656eb498ddb1df66c9c`. The concurrent parity worktree is
changing provisioning and PATCH parsing; those changes are outside this review.

## Git transport: reproduced routing and authority defects

`handle_git` builds its service root from `data_root/{pod}`, but passes the
original request path to `GitHttpService`. That service extracts the pod slug
and joins it to its root again. For `/alice/info/refs`, the candidate repository
is therefore `data_root/alice/alice`, not the normal `data_root/alice` location
that `git_mark_write` commits into. A missing nested repository stops delivery
before the CGI runs. This narrows the earlier reviewer's exposure claim.

This route authorises only Read on `/alice/`. If the selected repository
exists, the service exports its objects without evaluating the ACL on each
child. Fixing the root mapping alone would make the normal history reachable
under that same insufficient permission check. Routing and export authority
must be repaired together. Root Read, root Control and ownership of all past
content are different contracts; no new authority policy is inferred here.

A completed native probe uses synthetic bytes in a fresh filesystem pod,
NIP-98 owner writes, the production server routes, local Git and an empty
receiving repository. All assertions pass, reproducing the following defects:

| Case | Observed result |
| --- | --- |
| Ordinary pod layout | Private LDP read returns 401; Git discovery returns 404 because the nested repository is absent. |
| Correctly rooted direct-service control | Returned pack decodes to both the current and historical synthetic private file contents. This is a transport control, not the current server route. |
| Existing nested repository | Private LDP read returns 401; anonymous server Git discovery and upload return 200; the returned pack decodes to the protected synthetic file. |

The [native receipt](evidence/2026-11-01/solid-git-disclosure/native-check.json)
binds the executed fixture and raw log. The build took 25m49s under heavy I/O
pressure; the test took 2.18s. This is a local production-route reproduction,
not a live deployment test. The routing and whole-history authority repair
remains open.

## Mirror refs: configuration contradicts the withholding claim

At the reproduced baseline, the shared Git service forces `uploadpack.allowTipSHA1InWant=true` in its CGI
environment (`solid-pod-rs-git/src/service.rs:421`). The forge passes smart-HTTP
requests to that service (`solid-pod-rs-forge/src/lib.rs:626`). Meanwhile,
`mirror.rs:1211` hides `refs/mirror` from upload and transfer, and `fetch` places
remote branches in that namespace before the engine decides which branches may
move (`mirror.rs:1418`, `1442`, `2043`). A held verdict leaves the fetched
objects in the same repository. These settings compose differently from the
withholding claim in ADR-2009 and ADR-2010.

A completed local Git subprocess probe uses two synthetic commits: one public
tip and one held tip reachable only through `refs/mirror/held`. Discovery omits
the held tip. With the service's `allowTipSHA1InWant=true` setting, a request
naming that tip returns a pack which an empty receiver decodes to the held
file's exact contents. The same request with the setting false fails with
`not our ref`. Both outcomes are retained in
[the raw probe output](evidence/2026-11-01/solid-git-disclosure/hidden-ref-probe.txt)
alongside [the executed fixture](evidence/2026-11-01/solid-git-disclosure/hidden-ref-probe.sh).
Two integration regressions also reproduce pack delivery through the actual
Rust HTTP service, after successfully decoding an advertised public-tip pack.
One uses ordinary repository settings; the other enables all three permissive
unadvertised-object request options locally. Both fail at the hidden-pack
assertion before the repair and pass afterwards. No live forge or relay was
exercised.

Commit `1a66c2d` forces `allowTipSHA1InWant`, `allowReachableSHA1InWant` and
`allowAnySHA1InWant` false in the CGI environment. The Git crate suite passes
105 tests, including those two regressions, with one doctest ignored. Four
server Git WAC tests also pass; formatting, warning-denied Clippy and rustdoc
pass. The [repair receipt](evidence/2026-11-01/solid-git-disclosure/repair-checks.json)
records commands and source hashes. No package release, consumer upgrade or
deployment is claimed.

Git documents that this option explicitly permits requests for hidden tips.
It also warns that disabling it does not provide general read isolation within
a repository. A targeted repair must retain ordinary advertised-tip fetches
and resist permissive repository-local overrides; the regressions cover these
conditions. A complete withholding
boundary needs rejected objects separated from the repository being served,
including during fetch and validation; hiding their refs alone is insufficient.
[Git upload-pack configuration](https://git-scm.com/docs/git-config#Documentation/git-config.txt-uploadpackallowTipSHA1InWant)

Mirror branch-integrity and object-export decisions need separate
acceptance tests. This checkpoint does not reclassify all of ADR-2009 or
ADR-2010 as assessed, and does not infer deployed exposure from their historical
consumer observations.

## Security review disposition

Deepsec completed one source-file analysis and passed its HIGH threshold with
one MEDIUM and three BUG findings. All four predate the repair. Raw findings
and the configured verdict are retained; PASS does not mean no findings.

| Reviewer finding | Current disposition |
| --- | --- |
| Backend diagnostics and absolute paths can reach HTTP errors | Source-confirmed error propagation; a fresh sentinel-diagnostic HTTP fixture remains open. |
| `GitResponse::error` interpolates unescaped JSON | Helper defect confirmed; workspace search finds only its unit test and documentation, so the reviewer's deployed-caller claim is not established. |
| Service-name substring matching can classify a read as a write | Source-confirmed classification risk; authenticated GET auto-init and each adapter's exposure still need runtime tests. |
| `with_auth` docs describe anonymous push despite fail-closed code | Corrected in `31fe898`; warning-denied rustdoc passes. |

An initial working-diff invocation selected zero files after staging; its PASS
earns no review coverage. The corrected staged run selected `service.rs`.
Tests and documentation were not selected by that run; executed regression
tests, manual review and documentation checks cover those separately. Only
three rustdoc lines changed after review. The
[scope receipt](evidence/2026-11-01/solid-git-disclosure/security-review-scope.json)
binds reviewed source bytes, and the
[remediation queue](remediation/solid-git-export-2026-11-01.json)
preserves qualifications and acceptance conditions.

## ADR-2001 section assessment

| Section | Disposition |
| --- | --- |
| Context | Historical documentation drift is the rationale for consolidation. Current baseline summaries still mix old version and consumer facts with later scoped amendments; their presence does not prove current implementation or deployment. |
| Decision | Baseline, three-axis ledger, template, preamble and frozen archive exist. The local generator validates required fields, enumerations and reciprocal supersession. It does not resolve `verified_commit` or verify cited source behaviour. |
| Consequences | Separating historical records from governing invariants remains useful. A visible revision stamp and accurate status labels require source review; structural validation alone cannot prevent overclaiming. Current consumer and deployment obligations remain open. |
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
branch at its existing source stamp. SP-14 separates repository selection,
export authority and the direct hidden-object request repair at `1a66c2d`. Fourteen diagrams render
across both topics; only SP-06.12 and the three SP-14 panels have been visually
inspected and assessed in this checkpoint. The gates resolve 123 historical and eighteen current
citations without warnings or fallback. One JEV advisory on unchanged SP-06.3
remains outside this selected-panel review. The final documentation correction
does not change the code or line numbers cited by SP-14. This checkpoint records
one whole-ADR assessment and four panel assessments. Rebinding the unchanged
SP-06.7 body earns no additional credit.

Agentbox commit `cf04e56cd` adds the demonstrated routing/authority coupling and
discovery/direct-access distinction to the existing review-evidence skill
reference. Skill lint passes for 131 skills with existing warnings, and the
reference's Unicode and high-severity prose checks pass. JEV-facing descriptions
are unchanged. The baked skill still requires an Agentbox rebuild to adopt it.

## External contract comparison

Git's HTTP backend combines its configured repository root with the request
path. The pod adapter and its framework-independent service must agree about
which component has already been removed. [Git HTTP backend](https://git-scm.com/docs/git-http-backend)

Git also warns that namespaces do not provide read isolation within a shared
repository. That supports treating history export as authority over the whole
repository object store; filtering a displayed file list or a few refs is not
an adequate substitute. [Git namespace security](https://git-scm.com/docs/gitnamespaces)

Users need a usable private history export with an
explicit authority model, while ordinary public resource discovery remains
available. Acceptance must include a normal pod, a nested repository, private
children, historical versions and both permitted and denied callers. The forge
has its own prefix and repository-authorisation adapter; the pod reproduction
does not certify that separate path. The shared export configuration is
assessed separately above.

For product positioning, two existing user journeys are useful comparisons:

| Comparator | Documented user journey | Implication for VisionFlow |
| --- | --- | --- |
| [Forgejo repository permissions](https://forgejo.org/docs/v17.0/user/collaboration/repo-permissions/) | Repository Read permits viewing, cloning and pulling the repository. | Offer repository export only under an explicit repository-wide authority contract; a resource-container grant has a different scope. |
| [Nextcloud file versions](https://docs.nextcloud.com/server/stable/user_manual/en/files/version_control.html) | Users inspect a file's Versions tab, download a previous version or restore it. Named versions are exempt from automatic expiry. | Make authorised document recovery a direct user journey; users should not need a whole-pod Git export to recover one document. |

These implications are design recommendations, not claims of implemented
parity or audited security in either comparator. A useful industrial acceptance
scenario is to restore an approved maintenance document while a neighbouring
restricted document and its old versions remain inaccessible. Measure successful
authorised recovery and denied access to those unrelated bytes separately.
