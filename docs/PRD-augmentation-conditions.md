# PRD: Augmentation Conditions

**Owner:** DreamLab AI
**Status:** Active
**Date:** 2026-09-14
**Version:** 1.0
**Governed by:** [ADR-2010](adr/ADR-2010-augmentation-conditions-are-the-canon-audit-lens.md), [ADR-2011](adr/ADR-2011-task-properties-set-the-boundary-not-agent-self-tiering.md)
**Bounded context:** [DDD: Augmentation Conditions](DDD-augmentation-conditions-context.md)
**Source analysis:** [Judgment Broker Augmentation Audit](https://claude.ai/code/artifact/bd14aaf0-a387-4c57-b590-c49814ca6332) against arXiv 2609.12482 (CIVIC-AI 2026, *When Does AI Augment Work?*)
**Method:** build-with-quality (EDD → TDD, executed evidence, cross-family audit)

## TL;DR

The Judgment Broker loop is cryptographically complete and semantically hollow. Every human decision is a signed, supersession-chained Nostr event, but the decision card never shows the proposed change, the recorded rationale is fabricated by the UI, the boundary between agent and human is set by the requesting agent's own risk declaration, the human never learns whether an approval took effect, and nothing in the estate measures whether the humans doing the reviewing stay capable of doing it.

This PRD adopts the six augmentation conditions from arXiv 2609.12482 as the canon's audit lens for human judgement, and lands seven changes across four repositories so that the estate can be graded against them from code rather than declared compliant.

## Goals

| Goal | Outcome |
|---|---|
| G1 Audit lens | The six conditions are rows in the compatibility matrix and the acceptance frame for closeout packages CP-05 and CP-07 |
| G2 Non-vacuous verification | A 31403 records a judgement a human formed with the proposal in view, in their own words |
| G3 Task-set boundary | Escalation posture derives from operator-declared task properties, never solely from the requesting agent's self-tier |
| G4 Closed recovery loop | A human sees whether their decision applied; a stalled case ages visibly; a restarted actor recovers its pending cases |
| G5 Intent kept | Declared intent is persisted and compared against the act, and that comparison feeds HITL Precision |
| G6 Humans instrumented | Reviewer time, volume and override rate are measured; juniors get substantive review work; reviewers keep exposure to agent failure |
| G7 Continue during outage | An operator can execute an approved action manually with a signed receipt when the mesh is down |

## The six conditions (normative reference)

| # | Condition | Layer | Grading question |
|---|---|---|---|
| C1 | Durable net value | snapshot | Is verification, exception and repair effort counted against the productivity claim? |
| C2 | Meaningful human control | snapshot | Does the reviewer have the competence, time, information and authority to detect and override, and can they continue during unavailability? |
| C3 | Accountability and recovery | snapshot | Are authority, provenance, escalation and fallback explicitly assigned and observable? |
| C4 | Deepening learning | longitudinal | Is human learning measured, not only agent learning? |
| C5 | Career pathways | longitudinal | Do junior reviewers get substantive review work? |
| C6 | Job purpose | longitudinal | Is the human's judgement captured, not only their click? |

Boundary properties: **verifiability**, **reversibility**, **stakes**. The paper's design rule: assign agents work where gains scale, actions reverse and errors stay inspectable; retain human authority where decisions set goals, validity, interpretation or consequential use.

## Functional Requirements

Each FR names its owning repository per [ADR-2006](adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md) (canon owns the cross-repo view, substrates own implementation) and the expectation ids that prove it.

### FR1 Canon audit lens (VisionFlow) — EXP-AC-001

1. `docs/architecture/compatibility-matrix.md` gains a **Augmentation conditions** table with one row per condition per substrate, each cell a status (`absent | partial | measured`) with a `file:line` citation.
2. `docs/estate-review/closeout/README.md` CP-05 and CP-07 acceptance columns reference C1–C6 by id.
3. `docs/terminology.md` defines *augmentation condition*, *task-property triple*, *vacuous verification*, *calibration sample*.
4. The judgment-broker PRD and DDD carry an amendment pointing here.

### FR2 Non-vacuous decision surface (nostr-rust-forum, VisionClaw) — EXP-AC-002

1. The forum `ActionRow` renders `ActionRequest.fields` (the proposed change) and `context_url` in full, with the agent's `reasoning`, **below** the reviewer's controls. The agent's `risk_tier` and `confidence` render only after the reviewer has opened the rationale input, or below the controls, never above Approve.
2. The forum client collects a typed human rationale. For effective tier `high` or `critical` the rationale is mandatory (minimum 20 characters); the publish button is disabled until present. The published 31403 `reasoning` is the human's text verbatim. The string `"Human {action} via governance UI"` is removed from the codebase.
3. VisionClaw `AcspCaseQueue` renders the full proposal payload, proposal URN, reasoning hash, and generation provenance where present (`model`, `source_excerpt`, `confidence` only when not the hardcoded default), collects a typed rationale under the same rule, and never fabricates a rationale.
4. VisionClaw replaces the hardcoded `confidence: 0.5` with `None` so that absence renders as absence (mirrors the D7 honesty rule).

### FR3 Task-property triple (nostr-rust-forum, agentbox) — EXP-AC-003

1. `nostr-bbs-core` gains `TaskProperties { verifiability: Inspectable|Partial|Opaque, reversibility: Reversible|Compensable|Irreversible, stakes: Bounded|Significant|Critical }` serialised as tags on `PanelDefinition` (panel default, operator-set) and optionally on `ActionRequest` (may only tighten, never loosen, the panel default).
2. `effective_tier(panel, request)` is a pure function: `Irreversible` or `Critical` stakes ⇒ at least `high`; `Opaque` verifiability ⇒ at least `medium` and never member-suppressed; otherwise the agent's declared tier bounded below by the panel default. The relay stores the effective tier on `broker_cases` and the client reads only the effective tier.
3. The relay enforces its advertised `ESCALATION_DEFAULT_TIER` / `ESCALATION_DEFAULT_POSTURE`: an unlabelled request receives the default tier, and a request whose effective tier is `high` or `critical` cannot be resolved by anything other than a human 31403.
4. agentbox `governance_request_action` accepts `task_properties` and derives a default from the manifest `authority_class` (`zero-tolerance` ⇒ `Irreversible`, `recoverable` ⇒ `Compensable`), and the authority gate stamps the triple on its own 31402.

### FR4 Receipt ladder, ageing, reconciliation (nostr-rust-forum, agentbox, VisionClaw) — EXP-AC-004

1. The forum auth worker exposes `POST /api/governance/receipts/{response_event_id}/application` (NIP-98, registered agent or admin) accepting `{stage: consumer-received|applied|not-applied|applied-manually, acknowledgement}`; the relay-side `governance_receipts` row advances to that stage atomically and the decision chain in the UI shows it.
2. agentbox `broker-bridge` posts the receipt after `ApplicationReceiptStore.begin` and `finish`; failure to post is journalled, never silent.
3. The relay cron computes pending-case age; a case older than the panel's `max_pending_hours` (default 72) receives an `escalated-on-age` receipt and the UI badges it. Age is visible on every pending card (`created_at` differenced client-side).
4. agentbox authority-gate denials are appended to the execution journal as `authority.deny` records with stage and reason, readable at `/v1/agent-events`.
5. VisionClaw `ElevationActor` gains `OPEN_CASE_TTL` (14 days, same as `decision_elevation_actor.rs`), boot reconciliation of `pending` rows, and an `expired` receipt.

### FR5 Intent persistence and HITL Precision (VisionClaw) — EXP-AC-005

1. `kpi_agent_events` and `NewAgentTrajectory` gain `intent TEXT NULL`; the hub tap persists the envelope's `intent` verbatim and never synthesises one.
2. `GET /api/trace` returns `intent` and a per-row `intent_match: true|false|null` computed by comparing the declared operation and target with the recorded `action_type_name` and `target_urn`.
3. `kpi_compute.rs` HITL Precision is computed from decided cases: an escalation is *warranted* when the human outcome differs from the agent's requested action (reject, amend, delegate, or approve with a rationale that names a change) or when `intent_match == false`. The `awaiting_data_source` stub is removed; the KPI reports its denominator.
4. Whelk gate rejections carry `decided_by: system:whelk-gate` and are excluded from the Trust Variance human-outcome series.

### FR6 Human instrumentation (nostr-rust-forum, VisionClaw, agentbox, VisionFlow) — EXP-AC-006

1. Forum auth worker `GET /api/governance/reviewers` (admin NIP-98) returns per-reviewer: decisions, median and p90 time-to-decision, override rate (decisions ≠ agent's request), superseded count, calibration-sample catch rate.
2. **Reviewer tier via Delegate.** A governance role `reviewer` exists. An admin publishes `Delegate { to: <did> }` on a case; the relay then admits a 31403 from that delegatee for that case only, with the delegatee's outcome recorded and the admin's delegation preserved in the chain. Reviewers otherwise remain read-only.
3. **Calibration sampling.** `PanelDefinition.calibration_sample_rate` (default 0.1). A keyed selection — HMAC-SHA256 of the request id under a relay-held secret, compared against the rate — marks a `Low`/`Reversible` request `calibration_sample: true`; the key is never published, so the requesting agent (which chooses the request id) cannot search for an id the sampler never selects; such requests are shown to reviewers rather than suppressed. The reviewer telemetry reports the share of sampled requests that received a decision.
4. **Seeded probes.** An `ActionRequest` may carry tag `probe=<sha256>`; the client hides the tag until decided; telemetry counts probes rejected as the *catch rate*. Probes are published only by a registered probe agent named in panel metadata. *Known limit (2026-09-15):* the tag is on the signed event, so the measured agent can see it; the tagless design in the DDD §9.5 supersedes this once implemented.
5. VisionClaw `CaseView` gains `createdAt`; the case queue shows age and sorts oldest first.
6. The dream-cycle ledger schema (VisionFlow `docs/dream-cycle/LEDGER.md`, agentbox `dream-ledger.js`) gains `Reviewer` and `Review-minutes` columns, populated from the PR merge event.

### FR7 Manual continuation (agentbox, nostr-rust-forum) — EXP-AC-007

1. agentbox MCP tool `governance_manual_continue { case_id, executed_by, evidence }` records an operator-executed action: it writes an `applied-manually` application receipt bound to the approved operation digest, mints a PROV-O activity with `executed_by` a human DID, and posts the receipt to the forum.
2. The relay accepts `applied-manually` only from an admin pubkey and only for a case already `Decided: Approve`.
3. The authority gate's `no-decision-surface` deny path returns a structured hint naming the manual-continuation tool so an operator learns the option at the moment of denial.

## Non-functional requirements

- **No fabricated human text.** No surface may write a rationale, intent or confidence the human or agent did not author. Absence renders as absence.
- **Backward compatibility.** New tags and columns are additive; legacy events without task properties parse to the panel default; legacy receipts without application stages remain valid.
- **Evidence.** Every FR ships with an `EXP-AC-NNN` expectation, executed evidence with command, raw output, timestamp and git SHA, an auditor on a different model family, and a stabilising test named in `stabilized_by`.
- **Security gate.** `deepsec-gate.sh --diff` runs per repository on the change set; exit 78 is recorded as SKIPPED, never passed.

## Milestones

| Milestone | Delivers | Exit |
|---|---|---|
| M1 Canon | FR1, this PRD, ADR-2010, ADR-2011, DDD, EXP-AC-001..007 | Matrix table present; ledger index regenerated |
| M2 Substrates | FR2–FR7 on branches `feat/augmentation-conditions` in each repo | Tests green; evidence audited; deepsec receipts |
| M3 Publish | `nostr-bbs-core` and `nostr-bbs-mesh` republished; docs updated in all four repos | crates.io versions resolvable; diagram docs re-cited |
| M4 Live | Probe suite run against the live edge forum (relay, auth API) | Receipt of each probe; deploy status recorded honestly |

## Out of scope

- Multi-relay federation of receipts.
- Payment enforcement.
- A precedent system (deleted per agentbox; when it returns it must satisfy C4 by design).
- Replacing `RiskTier` — it remains as the agent's declaration; only its *authority* changes.

## Protocol reference

- Kinds 31400–31405 unchanged; new data rides tags and `outcome_detail`.
- Receipt stages: `signed → relay-accepted → projection-committed → consumer-received → applied | not-applied | applied-manually`, plus `escalated-on-age` and `expired` as side receipts.
- Identity: `did:nostr:<hex>` for humans and agents; `system:whelk-gate` for reasoner outcomes.
