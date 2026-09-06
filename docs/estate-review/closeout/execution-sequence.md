---
title: Closeout execution sequence
status: proposed
date: 2026-09-04
type: how-to
---

# Closeout execution sequence

Use this sequence with the nine [work packages](README.md) and their six acceptance journeys. It consolidates the evidence gathered so far into implementation order. It is a proposed delivery plan, not authorisation to deploy changes or a declaration that the assessment is finished. Maintainer roles identify the expertise needed; named ownership and capacity have not been confirmed. No delivery dates are invented.

## 1. Establish what each result claims

CP-01/09, estate and domain maintainers. Agree on a release revision set, the authority of each record and the meaning of accepted, applied, observed, healthy and evaluated. Prioritise [liveness](../liveness-observation.md), [KPI definitions](../kpi-outcomes.md), [consultation independence](../agent-grounding-and-governance.md#cross-model-consultation-and-acceptance) and the [remaining historical obligations](../historical-decision-reconciliation.md). Keep a decision's implementation, activation and historical status separate.

Exit evidence: each claimed outcome has a source-to-consumer path, accountable role, failure meaning and exact acceptance predicate. Classify imported and archived records before using their status as local authority. This stage prevents later repairs from making the wrong measurement green.

## 2. Make durable state and authority trustworthy

CP-04/05/07/08, identity, governance and storage maintainers. Resolve [journal append acknowledgements](../shared-memory.md#execution-journal-durability-and-reconstruction), [ACSP case correlation/recovery](../forum-decisions.md#visionclaw-acsp-consumption-and-recovery), [voice grant scope](../agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope), [role revocation](../role-authority.md) and [pod authority/cache/recovery](../storage-and-authority.md). Establish idempotency and compensating recovery at the actual persistence boundaries.

Exit evidence: a signed, scoped request produces exactly one durable applied/rejected receipt; retries after partial failure do not disappear or duplicate effects; revocation and restart preserve the declared policy. A dispatch response, in-memory duplicate flag or signed event alone is insufficient.

## 3. Stabilise corpus and executable identity

CP-02/03/06/08, knowledge, runtime and release maintainers. Resolve [vault collision/inclusion](../authored-vault-transition.md), [publication privacy](../knowledge-production.md), [generation activation](../grounding-delivery.md), [vector artefact compatibility](../consumed-vector-storage.md), [development input invalidation](../configuration-projection.md#development-restart-and-build-input-coverage) and [loaded PTX identity](../rendered-state.md#ptx-build-acceptance-and-loaded-artefact-identity).

Exit evidence: the served corpus, model geometry, host binary and loaded device module match a versioned manifest; an incompatible or partial artefact is rejected or explicitly degraded. Recovery restores the same declared ownership and generation. This can proceed alongside stage 2 once its boundary contracts are agreed.

## 4. Verify consumers through failure and recovery

CP-03/05/06/08, application, actor and interaction maintainers. Exercise actual retrieval/answer paths, [GPU context recovery](../rendered-state.md#gpu-supervision-and-context-delivery), [all HUD controls and hierarchy semantics](../rendered-state.md#xr-control-coverage-and-hierarchy-semantics), desktop/XR output, session egress and stale/error presentation. Distinguish roster presence from rendered action and hardware sensing from simulation.

Exit evidence: the intended user sees the correct current result after normal operation, denial, stale data, disconnect and recovery. Component tests remain supporting evidence; they do not replace the target-client journey.

## 5. Admit and evaluate improvement work against a fixed candidate

CP-07/08, evaluation and release maintainers. Enforce [evaluator readiness](../self-improvement.md#evaluator-readiness-before-scheduling), candidate-bound checks, strict verdict interpretation and durable outcome reporting. Resolve capture completeness and model provenance before claiming learning benefit. Independently review the exact candidate and keep human approval distinct from automatic evaluation.

Exit evidence: missing or unusable evaluators prevent admission; a broken candidate cannot receive an accepted result; repeated/interrupted runs preserve the intended experiment and receipt. Neither a baseline test nor a successful consultant invocation closes this stage.

## 6. Ratify complete-system acceptance

CP-09 depends on CP-01–08. Execute all six roadmap journeys against one declared revision/profile set, with negative controls and recovery evidence. Reconcile public claims, operative ADRs, historical dispositions, generated indices and release receipts. Preserve unresolved limitations explicitly rather than averaging them into a pass rate.

Exit evidence: the acceptance owner can follow every claim to its actual producer, authority, consumer and receipt. This is system implementation closeout. The documentation assessment can finish earlier when every in-scope requirement and record has an evidence-qualified disposition; it must not confuse its own coverage with this delivery milestone.

## Immediate assessment backlog

Complete semantic classification of the RuVector/RuView decision packs and their copied records; reconcile remaining first-party candidate/support rows; finish section-level historical dispositions; review remaining book/report claims against the thematic findings. Do not rerun already sufficient helper tests merely to increase test counts. Gather new evidence where a missing consumer or boundary could change the conclusion.
