---
title: Solid Pod empty-body commitment repair
date: 2026-11-01
status: source-repair-merged
type: explanation
---

# Solid Pod empty-body commitment repair

Solid Pod commit `8f2df863e230ae9b59ecbbe4cce36e76263f4380` repairs SP-ID-01 from the [identity assessment](2026-11-01-solid-identity-boundaries.md) in the shared verifier. A supplied body now requires a matching payload hash even when its length is zero. The token-minting helper now includes the hash for a deliberately empty body. This is a producer-source repair; release, consumer and deployment acceptance remain open.

## Behaviour and compatibility

A token committing to nonempty content cannot authorise a PUT that removes that content. Failed body binding occurs before signature-backed replay admission, so a rejected altered or empty body leaves the intended token available for its legitimate request. Successful authentication still consumes the token, and a second accepted write requires a fresh signed request.

The API distinguishes `Some(bytes)` from `None`. Supplying empty bytes requires a payload tag; callers that previously omitted that tag for an empty write must update. `mint_with_payload(..., Some(b""), ...)` now produces a compatible token. Passing `None` explicitly skips body verification, preserving the separate contract for callers that do not bind a body. This does not close policy gaps in callers that should supply a body but do not.

Three core regressions cover removal of a committed body, a missing payload tag on a supplied empty body and mint/verify round-tripping for empty bytes. The HTTP regression first proves a correctly signed private PUT succeeds, then checks anonymous and unbound requests, altered and removed content, unchanged stored bytes after rejection, acceptance of the original intended token, replay refusal and a deliberate empty write.

Before the repair, the selected core run had one existing pass and three regression failures; the route regression failed on the unbound-empty request. The preceding audit separately reproduced nonempty-to-empty substitution through that actual route. These failure causes are retained separately rather than describing every failed assertion as the same reproduction.

## Executed validation

Against the repaired source, `cargo test --locked --offline --workspace --all-targets` completed successfully: 2,170 passed and two ignored across 138 test summaries, plus successful benchmark smoke executions. These are test executions, not 2,170 unique requirements. The actual route regression is included in that count. Core crate doctests separately passed 30 with three ignored. Default workspace features were tested; this is not an all-feature-matrix claim.

`cargo fmt --all -- --check`, workspace Clippy with `--all-targets -- -D warnings`, and core crate documentation with `RUSTDOCFLAGS='-D warnings'` all passed. The tests used the real implementation and a disposable in-memory pod with an owner ACL; no live resources were changed. Commands, raw logs, source hashes and the reviewer disposition are retained in the [evidence directory](evidence/2026-11-01/solid-empty-body/manifest.json).

The new [SP-11.1 panel](../diagrams/solid-pod-rs/11-empty-body-binding.md) records the repaired flow. Its eleven citations resolve at the merged revision with no warnings or fallback. The panel was rendered and visually inspected; long citation labels were shortened after the first visual check exposed overflow. The eleven-citation JEV advisory check flagged none. The historical identity assessment and SP-05 remain evidence at their original source revision. This adds one panel assessment and no new whole-ADR credit. The [remediation record](remediation/solid-identity-2026-11-01.json) separates the producer repair from the remaining identity findings.

## Review disposition and remaining work

Deepsec reviewed the changed verifier and returned PASS at its HIGH threshold, with one MEDIUM finding about GitLenient string-prefix matching. That is the previously reproduced SP-ID-03, unchanged by this patch, despite the tool labelling it new. It remains open. The review did not cover the new untracked route-test file; that test received manual review and execution.

Short replay retention, default-clock boundary tests, Git repository scoping and misleading IdP delegation documentation remain open at their original evidence boundaries. The shipped `nip98_client` example also still constructs a dummy signature and describes signature checking as future work; it is not evidence of a working authenticated client journey and needs a separate adoption repair.

Publication, locked consumer upgrades and deployment are separate from merging this producer repair. Previously published versions and running containers do not acquire these source changes automatically. VisionClaw's payment repair remains separate, using its host verifier. No springs work, external account, live resource or funds are involved.

The general lessons are also merged into Agentbox at `9dfbf7cece25e49f41140b63ce0a0507faa9bba2`: contract-based consolidation, complete read/write journeys, boundary tests and explicit review coverage. Three JEV discovery probes chose the expected skills; that demonstrates routing only. The updated baked skills await an Agentbox rebuild.
