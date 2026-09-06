---
title: Canon chapter claim reconciliation
status: evidence-qualified-review
date: 2026-09-05
type: explanation
---

# Canon chapter claim reconciliation

The book's [canon chapter](../../presentation/report/chapters/13-visionflow-canon.tex) presents the architectural intent and several dated reconciliations. Readers need a current implementation qualification alongside that narrative. This review preserves the chapter as an authored account; it does not silently change its historical dates or certify every statement in the book. [Source hashes](evidence/canon-claim-reconciliation.json) identify the chapter and the assessed evidence used here.

## Claims and current dispositions

| Chapter claim or framing | Current evidence-qualified account | Closeout route |
|---|---|---|
| Six components define the architecture | This is the chapter's topology, not the current assessment roster. Fourteen repository identities are in scope, including consumed dependencies and historical material; those categories must not be collapsed into fourteen deployed products. | CP-01: [scope](scope-and-evidence.md), [upstream identity](upstream-decision-scope.md) |
| VisionFlow runs no code | It hosts no substrate application server, but executes publishing, drift, fixture and evaluation scripts. Preserve the canon ownership distinction while avoiding literal zero-execution wording. | CP-01/08: [canon implementation](canon-and-verification.md) |
| One contract owner means one place to change/check security-sensitive behaviour | Contract ownership is a governance choice. The reviews find separate native/edge consumers, codecs and admission paths; a named owner does not eliminate multiple implementations. | CP-01/04/06: [storage](storage-and-authority.md), [identifiers](federation-identifiers.md), [sensing](sensing-extension.md) |
| One key replaces sessions and tokens | The identity design is useful, but current VisionClaw retains session and request-auth realms. Signature identity does not determine resource authority, delegated scope or expiry. | CP-04: [role and authentication review](role-authority.md) |
| Every HTTP request binds URL, method and body hash | NIP-98 support is path-specific; the reviewed body-binding and development/public-admission conditions qualify a universal statement. Do not extend a helper's checks to every route. | CP-04/08: [role authority](role-authority.md), [runtime ingress](runtime-ingress.md) |
| A precedent-replay system does not exist | PrecedentService storage/matching/application primitives now exist. Complete authorised, durable production replay remains unverified. The accurate distinction is implemented primitives versus a verified complete loop. | CP-05/07: [engineering governance](engineering-governance.md) |
| No decision-supersession authority or appeal mechanism is defined | The reviewed forum domain and relay implement supersession/appeal rules and additional authority checks. Their existence does not complete UI, projection and downstream application journeys. | CP-04/05: [forum decisions](forum-decisions.md) |
| Shared fixture drift is caught mechanically during release qualification | The inspected fixture workflow can succeed without a comparison. Existing scripts and checksum conventions do not prove every release ran the intended cross-repository gate. | CP-01/08/09: [fixture CI assessment](canon-and-verification.md#fixture-ci-cannot-establish-cross-repository-parity-as-configured) |
| A coordinated release manifest captures the estate | The reviewed generator covers six repositories and omits newer consumed corpus/model/configuration identities. A local draft manifest is not a complete released-system receipt. | CP-01/08: [release inventory](canon-and-verification.md#release-inventory-trails-the-public-estate) |
| Scoped voice targets establish least-authority command ingress | Current producer checks and target normalisation exist, but issuer/resource/revocation binding and applied outcome remain separate obligations. A named actor and signed dispatch do not prove authorised execution. | CP-04/05/06: [voice authority](agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope) |
| Roughly 90% automatic and 10% escalated decisions explain the operating gain | Treat this as a proposed operating model until measured on this estate. Current KPI denominators and observation health do not establish that split, completed work or causal organisational benefit. | CP-01/05/07/09: [KPI outcomes](kpi-outcomes.md) |

The chapter's explicit cautions remain valuable, but cautionary claims can also become stale. Correcting an old absence claim should credit implemented primitives without promoting them to complete operational acceptance. Conversely, a clear architectural principle should not be worded as a universal enforcement guarantee when consumers implement different boundaries.

## What remains of the architectural argument

The strongest supported proposition is that the estate has mechanisms for shared identity, governed knowledge, human decision records and observable execution. Its unresolved promise is that those mechanisms preserve meaning and authority across a complete user journey. The later sensing work makes that distinction concrete: a present helper, a successful optimisation return, an imputed metric and a model installed into a real consumer are different evidence stages.

The [open-questions chapter](../../presentation/report/chapters/15-open-questions.tex) already distinguishes information routing from legitimation, accountability and political negotiation. The implementation review supports keeping that qualification prominent. Even perfect routing cannot assign organisational authority or establish whether a human reviewer had enough context. Measure review workload, error/recovery rates, false approvals/rejections and completed outcomes before asserting that governance accelerates work.

This pass does not verify the external studies cited by the book or their transferability to this estate. It also does not complete the book's other component, roadmap or gap-register chapters. CP-01/09 must reconcile those remaining claims, source dates and evidence tiers before the whole documentation assessment can close.
