# DDD: Augmentation Conditions Bounded Context

**Status:** Living document
**Date:** 2026-09-14
**Scope:** Grading and instrumenting every surface where a human decides on an agent's behalf, against the six conditions of arXiv 2609.12482
**Governed by:** [ADR-2010](adr/ADR-2010-augmentation-conditions-are-the-canon-audit-lens.md), [ADR-2011](adr/ADR-2011-task-properties-set-the-boundary-not-agent-self-tiering.md), [PRD](PRD-augmentation-conditions.md)

---

## 1. Bounded Context

The Augmentation Conditions context owns the *grading* of human–agent decision surfaces and the *instrumentation* that makes the grade measurable. It does not own decisions (Judgment Broker), standing regulation (Harness Engineering) or identity (Identity Spine). It consumes their events and produces three things: an effective escalation posture per case, a set of receipts that close the recovery loop, and reviewer-level telemetry.

The Judgment Broker context asks *"was this decision made and routed?"*. This context asks *"could the human who made it have caught a failure, did they learn whether it applied, and are they still able to do this next year?"*.

---

## 2. Context Map

| Context | Relationship | Notes |
|---|---|---|
| **Judgment Broker** | Upstream supplier | Supplies `BrokerCase`, `ActionRequest`, `ActionResponse`, `DecisionOutcome`. This context adds `TaskProperties`, effective tier, application receipts, delegation admission. |
| **Harness Engineering** | Upstream supplier / Shared kernel | Harness template `escalation_rules[]` carry the panel's task-property defaults and `calibration_sample_rate`; the `Precedent` aggregate, when rebuilt, must satisfy C4. |
| **Identity Spine** | Conformist | Reviewer, delegatee, probe agent and manual executor are all `did:nostr`; `system:whelk-gate` is a reserved non-DID actor for reasoner outcomes. |
| **Intent Legibility (D7)** | Shared kernel | The `intent` envelope field is persisted here and compared with the act. The D7 honesty rule (never fabricate) generalises to rationale and confidence. |
| **Forum Social Dynamics (F10)** | Sibling | F10's escalation-on-amplification and this context's escalation-on-age share the same receipt vocabulary. |
| **Ecosystem Alignment** | Downstream | The compatibility matrix's augmentation-conditions table reads this context's status vocabulary. |
| **Dream cycle (CP-07)** | Downstream | Ledger gains reviewer identity and review minutes; the human's learning is a first-class column. |

---

## 3. Ubiquitous Language

| Term | Definition |
|---|---|
| **Augmentation condition (C1–C6)** | One of the six grading questions. C1–C3 are snapshot; C4–C6 are longitudinal and need two measurements to mean anything. |
| **Task-property triple** | `{verifiability, reversibility, stakes}` declared by an operator on a panel; a request may tighten it. Owned by `nostr-bbs-core`. |
| **Declared tier** | The requesting agent's `risk_tier`. Telemetry only. |
| **Effective tier** | Output of `effective_tier(panel, request, declared)`. The only tier stored, rendered or used for suppression. |
| **Vacuous verification** | A signed decision made without the proposal, provenance or a human-authored rationale in view. The failure mode this context exists to remove. |
| **Application receipt** | A receipt stage past `projection-committed`: `consumer-received`, `applied`, `not-applied`, `applied-manually`. Owned by the mutation owner, published to the forum. |
| **Escalated-on-age** | A side receipt the relay cron emits when a pending case exceeds the panel's `max_pending_hours`. |
| **Calibration sample** | A `Low`/`Reversible` request selected by a keyed HMAC of its id under a relay secret (compared against `calibration_sample_rate`) to be shown rather than suppressed, so reviewers keep exposure to routine agent output. Keyed, not merely deterministic: the agent chooses the id, so an unkeyed hash lets it grind for an id that is never sampled (found and fixed in forum commit 67d0d71). |
| **Probe** | A known-bad request tagged `probe=<sha256>` by a registered probe agent; the tag is hidden until decided. The share rejected is the **catch rate**. |
| **Override rate** | Share of a reviewer's decisions whose outcome differs from the agent's requested action. |
| **Warranted escalation** | A decided case where the human outcome differed from the request, the human amended, or `intent_match == false`. Numerator of HITL Precision. |
| **Intent match** | Whether the persisted declared intent names the operation and target actually recorded for the act. `null` when no intent was declared. |
| **Delegation admission** | The relay rule that admits a 31403 from a `reviewer`-role DID only for a case an admin has `Delegate`d to them. |
| **Manual continuation** | An operator executing an already-approved operation by hand during mesh unavailability, evidenced by an `applied-manually` receipt bound to the approved operation digest. |

---

## 4. Aggregates

| Aggregate | Owner | Description |
|---|---|---|
| `TaskProperties` | nostr-rust-forum (`nostr-bbs-core`) | Value object on panel and request; tightening-only merge; serialised as tags `tp-verifiability`, `tp-reversibility`, `tp-stakes`. |
| `EffectiveTier` | nostr-rust-forum (`nostr-bbs-core`) | Pure function result, stored on `broker_cases.effective_tier`. |
| `ApplicationReceipt` | agentbox (`ApplicationReceiptStore`) as mutation owner; VisionClaw for elevation cases | Local write-once record; mirrored to the forum via the receipts endpoint. |
| `ReviewerTelemetry` | nostr-rust-forum (auth worker) | Read model over `broker_decisions` × `broker_cases` × `governance_receipts`. |
| `AgentTrajectory` (+ `intent`, `intent_match`) | VisionClaw | Per-action row; the workflow record's "agent path". |
| `HitlPrecision` | VisionClaw (`kpi_compute.rs`) | Warranted ÷ decided, with denominator reported. |
| `Delegation` | nostr-rust-forum | `Delegate{to}` outcome projected to `case_delegations`; consulted by the 31403 gate. |
| `CalibrationPolicy` | nostr-rust-forum (relay) + VisionFlow harness template | `calibration_sample_rate`, `max_pending_hours`, probe agent DID. |
| `AugmentationMatrixRow` | VisionFlow | `{condition, substrate, status, citation}`; the ledger row in the compatibility matrix. |

---

## 5. Domain Events

| Event | Producer | Consumers |
|---|---|---|
| `EffectiveTierAssigned {case_id, declared, effective}` | relay | client, reviewer telemetry |
| `ApplicationReceiptPublished {response_event_id, stage}` | agentbox / VisionClaw | forum receipts, client decision chain |
| `CaseEscalatedOnAge {case_id, age_hours}` | relay cron | client badge, reviewer telemetry |
| `CaseDelegated {case_id, from, to}` | forum (human) | relay gate, telemetry |
| `CalibrationSampleShown {request_id}` | relay | telemetry |
| `ProbeDecided {request_id, caught}` | relay | telemetry catch rate |
| `IntentRecorded {agent_did, intent, intent_match}` | VisionClaw hub tap | `/api/trace`, HITL precision |
| `ManualContinuationRecorded {case_id, executed_by}` | agentbox | forum receipts, PROV-O |
| `AuthorityDenied {agent_did, stage, reason}` | agentbox authority gate | execution journal, `/v1/agent-events` |

---

## 6. Invariants

1. **No fabricated judgement.** A rationale, intent or confidence is either authored by the human or agent it is attributed to, or absent. A surface that fills the slot is non-conformant.
2. **Tightening only.** `merge(panel, request)` never yields a looser property than `panel`. Property test in `nostr-bbs-core`.
3. **Effective tier is authoritative.** No consumer reads `risk_tier` for suppression or routing after this context lands.
4. **Human resolves high and critical.** A case with effective tier ≥ `high` reaches `Decided` only through a human 31403; `applied-manually` presupposes a prior `Approve`.
5. **Receipts are monotonic.** `consumer-received` precedes `applied | not-applied | applied-manually`; a stage never regresses; `escalated-on-age` and `expired` are side receipts that do not advance the ladder.
6. **Delegation is scoped.** A delegatee may decide only the delegated case; the delegation itself remains in the chain and is attributable to the admin.
7. **Probes are blind until decided.** The `probe` tag is never rendered before the 31403 exists.
8. **System actors are not humans.** `system:whelk-gate` outcomes are excluded from every human-reviewer series.
9. **Oversight selection is unpredictable to the overseen.** Any sampling or probe selection that decides whether an agent's action is shown to a human takes a secret the agent does not hold as an input. A selection an agent can compute is a selection an agent can evade.
10. **Longitudinal rows start as baselines.** The first `measured` status on C4–C6 is a baseline; a pass requires a second measurement no earlier than 90 days later.

---

## 7. Services

| Service | Owner | Responsibility |
|---|---|---|
| `effective_tier()` | nostr-bbs-core | Pure boundary function (ADR-2011) |
| Relay ageing cron | relay worker | Emits `escalated-on-age` |
| Receipts endpoint | auth worker | Accepts application stages under NIP-98 |
| Reviewer telemetry endpoint | auth worker | Serves per-reviewer read model |
| Receipt publisher | agentbox broker-bridge | Mirrors `ApplicationReceiptStore` to the forum |
| `governance_manual_continue` | agentbox MCP | Records manual continuation |
| Intent persister / matcher | VisionClaw hub tap | Persists `intent`, computes `intent_match` |
| HITL precision | VisionClaw kpi_compute | Warranted ÷ decided |
| Matrix grader | VisionFlow scripts | Validates citations in the augmentation table |

---

## 8. Ownership Summary

| Substrate | Owns | Does not own |
|---|---|---|
| **nostr-rust-forum** | `TaskProperties`, effective tier, delegation admission, receipts and telemetry endpoints, ageing cron, decision UI rationale capture | Application outcome truth (mutation owner), intent, KPIs |
| **agentbox** | Application receipts as mutation owner, manual continuation, authority-deny journal, triple derivation from `authority_class` | Panel defaults, relay gates |
| **VisionClaw** | Intent persistence and matching, HITL precision, elevation TTL/reconciliation, case-queue rationale capture | Protocol schema, receipts endpoint |
| **VisionFlow** | The lens: matrix table, CP acceptance frame, terminology, this context | Any runtime |

---

## 9. Open Issues

1. Reviewer telemetry is only as good as the client's `created_at`; requests published by agents with skewed clocks distort time-to-decision. Relay `accepted_at` is preferred where present.
2. Probe agents must be excluded from Augmentation Ratio's denominator or they inflate it.
3. `Amend` has no UI even after this PRD; it is the natural carrier for "approve with change" and should be the next forum client increment.
4. The precedent system, when rebuilt, must record which human decisions it generalised and expire when the reviewer population changes (C4).
5. **Probe blindness is one-sided today (forum KF-10).** The `probe` tag is hidden from every relay projection and from the client, which protects the *reviewer's* blindness, but it still rides the raw signed 31402, so the *agent being measured* can recognise its own probes and the catch rate is not yet trustworthy against a motivated agent. The durable design is for probes to carry no tag at all: the registered probe agent publishes requests indistinguishable from real ones and submits the set of probe event ids to the relay out of band (a signed list, or an HMAC-keyed selection like calibration, per invariant 9), and the relay marks them only after the 31403 exists. This is ADR-2011's (forum) review trigger.
