---
title: Review lessons incorporated into Agentbox skills
date: 2026-11-01
status: implemented-in-source
type: explanation
---

# Review lessons incorporated into Agentbox skills

Agentbox commit `686db14b1be0825ed9ff683790fb0ac550e30188` incorporates the estate review's reusable lessons into existing review skills. It is merged locally. Baked runtime copies will acquire the changes at the next Agentbox image rebuild; this checkpoint does not claim activation or a measured improvement in reviewer accuracy.

## Patterns and evidence

| Lesson | Evidence from this run | Instruction added |
| --- | --- | --- |
| Trace the implementation the consumer calls | VisionClaw had a replay-safe shared verifier, but payment routes called the stateless Solid verifier. The consumed crate was 0.5.0-alpha.17, distinct from the sibling checkout. | Follow the entry point, called helper and pinned dependency. Check cached or vendored source before declaring upstream unavailable. |
| Prove the cause of a failing test | The initial resource replay test failed on response-header construction after a debit, before reaching its replay assertion. | Read the first failure and stack; retain positive controls; separate setup failure, reproduced defect and inconclusive evidence. |
| Follow state beyond HTTP status | A completed debit preceded a response panic. Separate store writes and fresh-token retries remain open. | Inspect durable effects, secondary writes and delivery; distinguish token replay from logical-operation idempotency. |
| Read the actual gate policy | Deepsec returned PASS while reporting MEDIUM, HIGH_BUG and BUG findings; workspace lint failed outside changed code. | Record thresholds, severity categories, skipped scope, tested revision and baseline diagnostics. A gate result is bounded evidence. |
| Preserve uncertainty and counterevidence | Some upstream settlement suspicions were contradicted by pinned source; other deployment hypotheses remain untested. | Preserve raw findings and keyed dispositions. An unavailable reproduction is inconclusive, not a false positive. |
| Account for scope changes | Excluded repositories reduced scope; rendered diagrams and imported findings did not complete semantic assessments. | Separate source review, tests, merge and deployment. Count only explicitly assessed ADRs and panels. |

The [payment assessment](2026-11-01-visionclaw-payment-auth.md) supplies the concrete examples and their limits. The [completion ledger](completion/README.md) supplies the coverage distinctions. These rules extend the existing intake and review workflows; they do not introduce another tracking system.

## Skills and JEV discovery

[build-with-quality/references/review-evidence.md](../../../project/agentbox/skills/build-with-quality/references/review-evidence.md) contains the shared guidance. Its entry point and intake workflow now link it, and the Deepsec reference explains how to interpret gate receipts. [security-testing](../../../project/agentbox/skills/security-testing/SKILL.md) adds signed-request and stateful-effect checks. [diagrams-as-code verification](../../../project/agentbox/skills/diagrams-as-code/references/verification.md) adds source-version, semantic-coverage and denominator rules.

Two JEV routing checks against the updated source catalogue selected `security-testing` for the authentication review example, with `build-with-quality` also ranked, and `diagrams-as-code` for the corpus review example. Routing descriptions and front matter were unchanged. These are discovery checks, not an evaluation of downstream review quality.

Agentbox's authoritative skill lint passed across 131 skills, with its existing 33 suppressions and 29 warnings. Skill Creator's quick validator accepted `build-with-quality` and `security-testing`; it rejected the unchanged `compatibility` and `related_skills` metadata in `diagrams-as-code`. That validator/schema mismatch is retained, not reported as a pass. Logs and routing receipts are in the [evidence manifest](evidence/2026-11-01/pay-nip98/manifest.json).

One correction removes the old instruction to reject a finding that cannot be demonstrated. Reviewers must now leave such a finding inconclusive unless counterevidence contradicts it. This prevents missing access or an inadequate fixture from becoming a false assurance.
