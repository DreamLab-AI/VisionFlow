---
id: VC-24
title: ACSP — governed decision/elevation pipeline
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2006, ADR-2093, ADR-2101, ADR-2110, ADR-2148]
sources:
  - ../project/src/actors/elevation_actor.rs
  - ../project/src/services/kpi_compute.rs
  - ../project/src/actors/decision_elevation_actor.rs
  - ../project/src/adapters/decision_elevation_store.rs
  - ../project/src/services/decision_elevation.rs
  - ../project/src/services/decision_service.rs
  - ../project/src/services/broker_events.rs
  - ../project/src/services/proposal_spine.rs
  - ../project/src/services/nostr_bead_publisher.rs
  - ../project/src/services/nostr_bridge.rs
  - ../project/src/services/acsp/client.rs
  - ../project/src/services/acsp/events.rs
  - ../project/src/services/acsp/mod.rs
  - ../project/src/services/acsp/key_file.rs
  - ../project/src/services/ontology_enrichment_service.rs
  - ../project/src/adapters/sqlite_enrichment_repository.rs
  - ../project/src/handlers/enrichment_proposals_handler.rs
  - ../project/src/utils/agent_key.rs
  - ../project/src/handlers/broker_inbox_handler.rs
  - ../project/docs/reference/configuration.md
  - ../project/docker-compose.unified.yml
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/adr/ADR-2148-enable-local-typed-decisions.md
  - ../project/crates/visionclaw-jev/src/proposal_triage.rs
  - ../project/client/src/features/control-center/governance/brokerCaseQueue.ts
  - ../project/src/web_contract/ledger.rs
  - ../project/src/web_contract/reducer.rs
  - ../project/src/web_contract/ritual.rs
  - ../project/src/web_contract/state.rs
  - ../project/src/web_contract/trail.rs
  - ../project/src/app_state.rs
  - ../project/src/services/ontology_mutation_service.rs
  - ../project/src/services/voice_intent_client.rs
verified_commit: {visionclaw: e74bcddc772d47a45b1971156a2bab540eaa027c, agentbox: 0b0f7918961e629280faf86f88227535abf514b5}
---

## VC-24.1 Enrichment-proposal lifecycle (real status values)

```mermaid
stateDiagram-v2
    [*] --> pending: SqliteEnrichmentRepository.create_or_update<br/>sqlite_enrichment_repository.rs:401 status DEFAULT pending :61
    pending --> approved: status_for_outcome outcome approve accept accepted promote<br/>sqlite_enrichment_repository.rs:231
    pending --> rejected: status_for_outcome outcome starts_with reject<br/>sqlite_enrichment_repository.rs:232
    pending --> reviewed: status_for_outcome fallback amend delegate etc<br/>sqlite_enrichment_repository.rs:233
    approved --> elevated: ElevationActor GOV-2 terminal_for_pr_state Merged<br/>elevation_actor.rs:914, 1644 repo.set_status
    approved --> abandoned: ElevationActor GOV-2 terminal_for_pr_state ClosedUnmerged<br/>elevation_actor.rs:914, 1644 repo.set_status
    elevated --> [*]
    abandoned --> [*]
    rejected --> [*]
    reviewed --> [*]
    note right of approved
      RESOLVED (was forum-only pre-ADR-2101): DecisionElevationActor
      now carries its own store field (Option Arc DecisionElevationStore),
      decision_elevation_actor.rs:144, a typed facade OVER THIS SAME
      table (category=decision-elevation), decision_elevation_store.rs:60,
      202. Its terminal statuses reuse pending/approved/rejected/reviewed
      here but diverge past approved - published (not elevated) and
      applying/publish_failed/expired have no equivalent on this branch.
      decision_elevation_store.rs:68-90. See VC-24.6
    end note
```

## VC-24.2 Enrichment-proposal creation — the real call chain

```mermaid
sequenceDiagram
    autonumber
    participant EA as ElevationActor RunCycle handler<br/>elevation_actor.rs:1209
    participant ACSP as AcspClient.publish<br/>services/acsp/client.rs:119
    participant REPO as SqliteEnrichmentRepository<br/>adapters/sqlite_enrichment_repository.rs:401
    Note over EA: RunCycle scans owl_class frontier stubs<br/>ranked by voice demand then graph degree
    EA->>EA: case_for + pending_proposal build<br/>elevation_actor.rs:311, :373
    EA->>ACSP: publish build_action_request kind 31402<br/>elevation_actor.rs:1293 acsp/events.rs:307
    alt publish Ok
        EA->>REPO: create_or_update StoredProposal status pending<br/>elevation_actor.rs:1300
        REPO-->>EA: Ok case row upserted
    else publish Err
        EA->>EA: warn pending-case persist failed<br/>elevation_actor.rs:1301 (no row written)
    end
    Note over EA,REPO: DOC-DRIFT (re-verified at HEAD, corrected) — src/bin/sync_local.rs<br/>this note cited is DELETED. ontology_enrichment_service.rs still only<br/>classifies GraphData edges via EdgeClassifier (ontology_enrichment_service.rs:116),<br/>never touches the enrichment-proposal store (it also now has an optional Jev gap-fill for edges the classifier leaves unclassified, ontology_enrichment_service.rs:146-163, but with_jev :36 has test callers only and the production boot never builds this service). proposal_spine::governed_commit has<br/>exactly one non-test caller in the whole tree - decision_service.rs:677 - so the<br/>earlier claim that ontology_mutation_service.rs also calls it does not hold — that<br/>service uses the lower-level build_receipt/verify_envelope/WriteAheadIntent primitives<br/>directly (ontology_mutation_service.rs:26-28), never governed_commit. The real<br/>creator remains ElevationActor (above) plus the decide-route stub fallback (VC-24.3)
```

## VC-24.3 POST /api/enrichment-proposals/{id}/decide — decide route

```mermaid
sequenceDiagram
    autonumber
    participant BR as agentbox broker-bridge
    participant H as decide<br/>handlers/enrichment_proposals_handler.rs:424
    participant AUTH as require_agent_key<br/>handlers/enrichment_proposals_handler.rs:141
    participant AD as apply_decision<br/>handlers/enrichment_proposals_handler.rs:473
    participant REPO as SqliteEnrichmentRepository
    participant OXI as append_derived_summary call site<br/>handlers/enrichment_proposals_handler.rs:576
    participant ACSPC as AppState.acsp_client<br/>app_state.rs:301
    BR->>H: POST /api/enrichment-proposals/:id/decide X-Agent-Key
    H->>AUTH: require_agent_key compares X-Agent-Key to VISIONCLAW_AGENT_KEY<br/>via constant-time check_agent_key (subtle ct_eq, ADR-2093)<br/>utils/agent_key.rs:13, :15, handlers/enrichment_proposals_handler.rs:141, :146, :432
    alt key invalid or missing
        AUTH-->>H: 401 Unauthorized
        H-->>BR: 401 invalid or missing X-Agent-Key header
    else key valid
        H->>AD: apply_decision case_id body
        AD->>AD: record_decision pure core mints activity_urn<br/>handlers/enrichment_proposals_handler.rs:479 calling :168, :195
        AD->>REPO: get(case_id) read ONCE - presence drives broker:new_case,<br/>recorded tier drives the rationale gate<br/>handlers/enrichment_proposals_handler.rs:494
        AD->>AD: check_rationale(declared_tier, outcome, reasoning)<br/>handlers/enrichment_proposals_handler.rs:396, called at :510
        alt tier high or critical and rationale shorter than MIN_RATIONALE_CHARS 20
            AD-->>BR: 422 RationaleRejection code rationale_required<br/>handlers/enrichment_proposals_handler.rs:324, :516
            Note right of AD: ADR-2110 FR2.2 INVARIANT: refused BEFORE anything is minted or<br/>persisted, so a gated request leaves no trace and no partial state,<br/>and the server NEVER fills the rationale in - a server-authored<br/>rationale is the fabricated judgement DDD invariant 1 forbids.<br/>handlers/enrichment_proposals_handler.rs:304-306
            Note right of AD: INVARIANT one rule in two places, not two rules: the server<br/>constant must match the case queue's MIN_RATIONALE_CHARS.<br/>handlers/enrichment_proposals_handler.rs:320, mirrored at<br/>governance/brokerCaseQueue.ts:186 and asserted at :221
            Note right of AD: TENSION: the gate tier is now gate_tier(declared, effective) - the stricter of the<br/>risk_tier recorded on the case and the effective_tier the bridge MAY forward in the body (ER-7), so a<br/>forwarded tier can only tighten it. handlers/enrichment_proposals_handler.rs:70, :382, :508 - yet the in-code<br/>scope note still says effective_tier has not landed. handlers/enrichment_proposals_handler.rs:308-314
            Note right of AD: OPEN: a case with no row yet carries no recorded tier, so a first-contact bridge decision<br/>that forwards no effective_tier is still ungated. handlers/enrichment_proposals_handler.rs:502-508
        end
        opt case unknown to store
            AD->>REPO: create_or_update stub status pending is_new_case<br/>handlers/enrichment_proposals_handler.rs:537
        end
        AD->>REPO: record_decision atomic INSERT decision + UPDATE proposal.status<br/>handlers/enrichment_proposals_handler.rs:561 sqlite_enrichment_repository.rs:608
        alt outcome approve and attributed pubkey hex
            AD->>OXI: append_derived_summary owner_did activity_urn summary_triples
            OXI-->>AD: Ok fenced :summary write landed
            AD->>REPO: mark_writeback_committed case_id activity_urn now_ms<br/>handlers/enrichment_proposals_handler.rs:589
            Note right of AD: writeback_committed=true only on real Oxigraph write -<br/>writeback_triggered=true just means outcome qualified (approve+attributed)
        else unattributed or non-approve
            Note right of AD: writeback_committed stays false - no owner DID<br/>to scope an owner-less KG node, by design
        end
        AD->>AD: derive_kernel_decision via DecisionOrchestrator<br/>handlers/enrichment_proposals_handler.rs:616 calling :244 (domain broker kernel)
        Note over AD: DIVERGENCE: BrokerActor never merged (BASELINE-architecture.md<br/>Known divergences) - main uses this stateless ACSP producer plus the<br/>cherry-picked storage-agnostic domain kernel (BrokerCase/DecisionOrchestrator)<br/>invoked here, never a resurrected actor+transport
        alt case newly entered queue
            AD->>AD: broker_events.broadcast_new_case broker:new_case<br/>handlers/enrichment_proposals_handler.rs:618
        end
        AD->>AD: broker_events.broadcast_case_decided broker:case_decided<br/>handlers/enrichment_proposals_handler.rs:625
        alt ACSP client configured (handlers/enrichment_proposals_handler.rs:638)
            AD->>ACSPC: publish build_action_response kind 31403<br/>handlers/enrichment_proposals_handler.rs:643
            alt publish Ok
                ACSPC-->>AD: event id
                Note right of AD: forum_projection=published<br/>handlers/enrichment_proposals_handler.rs:646, :648
            else publish Err
                Note right of AD: forum_projection=failed - decision recorded<br/>locally but NOT visible in forum broker_decisions<br/>handlers/enrichment_proposals_handler.rs:654
            end
        else no AcspClient (FORUM_RELAY_URL unset)
            Note right of AD: forum_projection=skipped - degraded, not silent<br/>handlers/enrichment_proposals_handler.rs:638, :662
        end
        AD-->>H: DecideResponse success writeback_triggered writeback_committed forum_projection
        H-->>BR: 200 OK
    end
    Note over H,AD: EXTERNAL VC-24 addendum: a second entry point, decide_as_operator<br/>(handlers/enrichment_proposals_handler.rs:454, control-centre power-user<br/>session auth), funnels through the SAME apply_decision core - the auth differs,<br/>the decision path does not, and ADR-2110's rationale gate therefore binds both.
```

## VC-24.4 ElevationActor — draft → PR → GOV-2 merge poll → concept_elevated

```mermaid
sequenceDiagram
    autonumber
    participant CYCLE as run_interval CYCLE_INTERVAL 600s<br/>elevation_actor.rs:60, :1096
    participant EA as ElevationActor<br/>elevation_actor.rs:116
    participant ACSP as AcspClient
    participant GH as GitHubPRService.create_ontology_pr
    participant POLL as run_interval PR_POLL_INTERVAL 120s<br/>elevation_actor.rs:63, :1112
    CYCLE->>EA: RunCycle (frontier scan)<br/>elevation_actor.rs:1209
    EA->>ACSP: publish build_action_request kind 31402<br/>acsp/events.rs:307
    Note over EA: pending case held in-memory HashMap self.pending<br/>(elevation_actor.rs:140), capped at MAX_OPEN_CASES=5 concurrent<br/>cases (elevation_actor.rs:58, :1215, :1226)
    Note over EA: ADR-2110: pending is no longer only in memory. A Reconcile at boot<br/>reloads durable pending rows through the pure plan_elevation_reconciliation<br/>BEFORE the first cycle, so a kind-31403 arriving after a restart no longer<br/>hits an empty map and drops the human's signed decision.<br/>elevation_actor.rs:1009, :1330, :1340
    Note over EA: INVARIANT: a case past OPEN_CASE_TTL of 14 days is closed with an<br/>elevation_expired kind-31404 receipt AND a terminal durable status, and<br/>the store write does NOT depend on the receipt publishing.<br/>elevation_actor.rs:68, :70, :72, :1347, :1368
    ACSP-->>EA: CaseDecision via run_decision_subscription<br/>acsp/client.rs:164 (kind 31403, since Timestamp::now)
    alt action approve
        EA->>EA: approve_with_gate runs GOV-7 EL++ consistency gate<br/>elevation_actor.rs:1476, :1489 run_consistency_gate -> WhelkInferenceEngine.check_axiom_set :881
        alt gate inconsistent
            EA->>EA: record synthetic reject decision, case BLOCKED<br/>elevation_actor.rs:1508, :1516
            Note right of EA: ADR-2110 INVARIANT: the synthetic rejection is attributed to the<br/>reserved non-DID actor system:whelk-gate, not the human responder.<br/>Before this a reasoner outcome counted as a human override and<br/>polluted both Trust Variance and HITL Precision.<br/>elevation_actor.rs:1505-1512, kpi_compute.rs:140
        else gate consistent
            EA->>GH: create_ontology_pr draft class page<br/>elevation_actor.rs:1542
            GH-->>EA: pr_url
            EA->>EA: elevating.insert case_id TrackedPr pr_url<br/>elevation_actor.rs:1572
        end
    else action reject/amend/delegate
        EA->>EA: record_decision, rejected_count+=1<br/>elevation_actor.rs:1443, :1451
    end
    loop every PR_POLL_INTERVAL=120s
        POLL->>EA: PollPrs<br/>elevation_actor.rs:88, :1598
        alt no GitHub token configured
            Note right of EA: DEGRADED - concept_elevated can never fire<br/>elevation_actor.rs:1605
        else token present
            EA->>GH: pr_state pr_url
            alt PrState Merged
                GH-->>EA: Merged
                EA->>ACSP: publish build_case_status_update concept_elevated kind 31404<br/>elevation_actor.rs:914, :1633
                EA->>EA: repo.set_status case_id elevated<br/>elevation_actor.rs:1644
            else PrState ClosedUnmerged
                GH-->>EA: ClosedUnmerged
                EA->>ACSP: publish elevation_abandoned kind 31404
                EA->>EA: repo.set_status case_id abandoned
            else PrState Open
                GH-->>EA: Open (continue tracking, no event)
            end
        end
    end
```

## VC-24.5 GOV-2 terminal PR state mapping + publish-failure warn path

```mermaid
sequenceDiagram
    autonumber
    participant POLL as PollPrs handler<br/>elevation_actor.rs:1598
    participant MAP as terminal_for_pr_state<br/>elevation_actor.rs:914
    participant ACSP as AcspClient.publish
    participant REPO as SqliteEnrichmentRepository.set_status
    POLL->>MAP: terminal_for_pr_state(PrState)
    alt PrState::Merged
        MAP-->>POLL: Some concept_elevated, elevated
    else PrState::ClosedUnmerged
        MAP-->>POLL: Some elevation_abandoned, abandoned
    else PrState::Open
        MAP-->>POLL: None (keep polling, continue)
    end
    opt terminal status resolved
        POLL->>ACSP: publish build_case_status_update PANEL_ID case_id event_status pr_url<br/>elevation_actor.rs:1633
        alt publish Err
            ACSP-->>POLL: Err(e)
            POLL->>POLL: warn GOV-2 31404 publish failed for case_id<br/>elevation_actor.rs:1641
            Note right of POLL: publish failure does NOT block the store write below -<br/>the two facts (forum visibility vs durable terminal status)<br/>are independent, mirroring the writeback_triggered/committed split
        else publish Ok
            ACSP-->>POLL: event id
        end
        POLL->>REPO: set_status case_id store_status<br/>elevation_actor.rs:1644
        alt set_status Err
            REPO-->>POLL: Err(e)
            POLL->>POLL: warn GOV-2 terminal store status persist failed for case_id<br/>elevation_actor.rs:1646
        else set_status Ok
            POLL->>POLL: resolved.push(case_id) - remove from elevating map<br/>elevation_actor.rs:1653
        end
    end
    Note over POLL,REPO: RESOLVED ADR-2101 (2026-09-05) in part: the durable case-state<br/>authority ADR-2006 left unnamed is THIS store - enrichment.sqlite3 - reached for<br/>decision-elevation cases through DecisionElevationStore, and decision rows now<br/>carry the ADR-2006 signed-event correlation on both elevation paths.<br/>STILL OPEN on this path: the ElevationActor poll shown here persists set_status<br/>but has no boot reconciliation, so a case lost between publish and persist is not<br/>reloaded on restart. ADR-2101 adds that to DecisionElevationActor only - see VC-24.6
```

## VC-24.6 DecisionElevationActor — parallel path vs ElevationActor

```mermaid
sequenceDiagram
    autonumber
    participant DS as DecisionService.record_decision<br/>services/decision_service.rs:546 maybe_elevate
    participant SIG as is_significant<br/>services/decision_elevation.rs:75
    participant SINK as ActorElevationSink.elevate<br/>actors/decision_elevation_actor.rs:1014
    participant DEA as DecisionElevationActor<br/>actors/decision_elevation_actor.rs:128
    participant ACSP as AcspClient
    participant GH as create_ontology_pr call site<br/>decision_elevation_actor.rs:356
    Note over DS: governed write door already committed the DecisionRecord<br/>quads via proposal_spine::governed_commit (decision_service.rs:677)<br/>BEFORE maybe_elevate runs - elevation is fire-and-forget, fail-open
    DS->>SIG: is_significant(input, acsp_approved=false)<br/>decision_service.rs:525
    alt not significant (routine/edgeless)
        SIG-->>DS: false
        Note right of DS: runtime-only, no case opened - ADR-050 policy
    else significant (mutation/causal/precedent/influenced edges)
        SIG-->>DS: true
        DS->>SINK: elevate(ElevatedDecision)
        SINK->>DEA: try_send ElevateDecision (actor mailbox, non-blocking)<br/>decision_elevation_actor.rs:1014
        DEA->>DEA: draft_decision_page (services/decision_elevation.rs:257), open case CASE_PREFIX vc-decelev-<br/>decision_elevation_actor.rs:59, :607
        DEA->>ACSP: publish build_action_request kind 31402 PANEL_ID vc-decision-elevation<br/>decision_elevation_actor.rs:692
        ACSP-->>DEA: CaseDecision kind 31403<br/>decision_elevation_actor.rs:721
        alt action approve
            DEA->>GH: create_ontology_pr decision page (NO consistency gate), inside the<br/>spawn_decision_outcome future<br/>decision_elevation_actor.rs:284, :356
            Note right of DEA: Deliberately leaner than ElevationActor (module doc :12-14):<br/>decisions are ABox prov:Activity individuals adding no TBox<br/>axioms, so there is NO EL++ Whelk gate here (contrast VC-24.4<br/>approve_with_gate GOV-7)
            GH-->>DEA: pr_url
            DEA->>DEA: mark_elevating persists the PR url BEFORE elevating.insert case_id TrackedPr<br/>decision_elevation_actor.rs:363, decision_elevation_store.rs:259
        else reject/amend/delegate
            DEA->>DEA: rejected_count+=1, publish_state
        end
    end
    loop every PR_POLL_INTERVAL=120s
        DEA->>DEA: PollPrs calls terminal_for_pr_state<br/>decision_elevation_actor.rs:901, :921, :949
        alt Merged
            DEA->>ACSP: publish build_case_status_update decision_elevated kind 31404<br/>decision_elevation_actor.rs:948
            Note right of DEA: terminal_case_status maps decision_elevated to case_status::PUBLISHED<br/>(not elevated - the enrichment-proposal vocabulary), decision_elevation_actor.rs:905<br/>decision_elevation_store.rs:79. See VC-24.1 for the two vocabularies sharing one column
        else ClosedUnmerged
            DEA->>ACSP: publish decision_abandoned kind 31404
        end
    end
    Note over DEA: RESOLVED ADR-2101 (2026-09-05): the actor now holds a<br/>DecisionElevationStore, a typed facade over the SAME data/enrichment.sqlite3<br/>SqliteEnrichmentRepository the ElevationActor uses (decision_elevation_store.rs:60,<br/>202) - case rows tagged category=decision-elevation, decision rows carrying the<br/>ADR-2006 signed-event correlation. Every transition is durable and ordered - the<br/>case row is written BEFORE the 31402 publish, the PR url BEFORE the in-memory<br/>tracking insert (decision_elevation_actor.rs:363, :381), and the GOV-2 terminal<br/>status BEFORE the case leaves the map. At boot plan_reconciliation reloads every<br/>non-terminal case and re-arms the merge poll, re-opens a PR whose approval crashed<br/>before it landed, re-arms a case still awaiting a human, or expires it past a<br/>14-day TTL with a 31404 receipt (decision_elevation_actor.rs:819). A case tracking<br/>an open PR is never expired. Remaining gap - a resumed approval can open a<br/>duplicate PR, recorded as follow-on 1 in ADR-2101
```

## VC-24.8 nostr_bead_publisher / nostr_bridge — governance event publish

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as bead lifecycle caller
    participant PUB as NostrBeadPublisher<br/>services/nostr_bead_publisher.rs:31
    participant SRC as source relay NOSTR_RELAY_URL<br/>nostr_bead_publisher.rs:19
    participant BRIDGE as NostrBridge<br/>services/nostr_bridge.rs:28
    participant FORUM as forum relay FORUM_RELAY_URL<br/>nostr_bridge.rs:14
    rect rgb(226,228,246)
    Note over CALLER,SRC: trust boundary: bridge bot keypair VISIONCLAW_NOSTR_PRIVKEY<br/>shared by both publisher and bridge (nostr_bead_publisher.rs:18,<br/>nostr_bridge.rs:12)
    CALLER->>PUB: brief -> debrief cycle complete
    PUB->>PUB: from_env() validates ws:// or wss:// scheme<br/>nostr_bead_publisher.rs:39, :49
    PUB->>SRC: publish kind 30001 parameterized-replaceable (NIP-33) d=bead_id
    end
    BRIDGE->>SRC: subscribe kind 30001 bead provenance events
    SRC-->>BRIDGE: event(s)
    BRIDGE->>BRIDGE: re-sign under bridge keypair, preserve source_event tag<br/>nostr_bridge.rs:6-8
    BRIDGE->>FORUM: republish as NIP-29 group message kind 9
    Note over BRIDGE,FORUM: DIVERGENCE: NIP-26 delegation not wired, service-signing only<br/>(nostr_bridge.rs:29,65,169) - see VC-03 for the signing-authority detail<br/>and VC-09 for the env-var chain
```

## VC-24.9 Governance kind map (31400-31405)

```mermaid
flowchart LR
    subgraph kinds["ACSP Nostr kinds - services/acsp/events.rs:18-23"]
        K31400["31400 PanelDefinition"]
        K31401["31401 PanelState"]
        K31402["31402 ActionRequest"]
        K31403["31403 ActionResponse / CaseDecision"]
        K31404["31404 PanelUpdate / CaseStatusUpdate"]
        K31405["31405 PanelRetired"]
    end
    B31400["build_panel_definition<br/>acsp/events.rs:210"] -->|producer| K31400
    B31401["build_panel_state<br/>acsp/events.rs:219"] -->|producer| K31401
    B31402A["ElevationActor RunCycle publish<br/>elevation_actor.rs:1293"] -->|producer| K31402
    B31402B["DecisionElevationActor handle ElevateDecision then acsp.publish<br/>decision_elevation_actor.rs:692"] -->|producer| K31402
    B31402C["voice_intent_client.build_action_request<br/>voice_intent_client.rs:304"] -->|producer| K31402
    B31403A["build_action_response<br/>acsp/events.rs:288 enrichment_proposals_handler.rs:642"] -->|producer| K31403
    B31404A["build_case_status_update<br/>acsp/events.rs:237 elevation_actor.rs:1633"] -->|producer| K31404
    B31404B["build_panel_update<br/>acsp/events.rs:228"] -->|producer| K31404
    B31405["build_panel_retired<br/>acsp/events.rs:265"] -->|producer| K31405
    K31403 -->|consumer| C31403A["AcspClient.run_decision_subscription<br/>acsp/client.rs:164,169-171 filters since Timestamp::now"]
    C31403A -->|consumer| C31403B["ElevationActor.Decision handler<br/>elevation_actor.rs:1412"]
    C31403A -->|consumer| C31403C["DecisionElevationActor Decision handler<br/>decision_elevation_actor.rs:721"]
    K31402 -->|consumer| C31402["forum relay agent_registry gate<br/>acsp/client.rs:9-13 (relay-side, not this repo)"]
    Note1["Note: relay only accepts kinds 31400-31402 from<br/>registered pubkeys (acsp/client.rs:9) - 31403/31404/31405<br/>are consumer/admin-only, enforced relay-side"]
```

## VC-24.10 web_contract layer structure (reducer/state/ledger/trail/ritual)

```mermaid
flowchart TB
    subgraph L1["Layer 1 - Reducer - web_contract/reducer.rs"]
        TR["trait ContractReducer<br/>reducer.rs:69 validate + transition"]
        RE["ReducerError<br/>reducer.rs:32"]
    end
    subgraph L2["Layer 2 - State - web_contract/state.rs"]
        CS["CanonicalState<br/>state.rs:30 canonical_json + state_hash SHA-256"]
    end
    subgraph L3["Layer 3 - Ledger - web_contract/ledger.rs"]
        LE["Ledger / LedgerEntry<br/>ledger.rs:28,41 balance_sats integer-only"]
    end
    subgraph L4["Layer 4 - Trail - web_contract/trail.rs"]
        GM["GitMark 5-key envelope VERBATIM<br/>trail.rs:102 at-id genesis nick package repository"]
        BT["Blocktrails over solid-pod-rs Blocktrail<br/>trail.rs:184 walked by walk_blocktrail<br/>recomputed keys x(P_i) from pubkeyBase<br/>ritual.rs:215 verify_trail verdicts<br/>verified confirmed partial"]
    end
    subgraph RIT["Ritual + verify - web_contract/ritual.rs"]
        TL["TrustLevel L0/L1Anchored/L2/L3<br/>ritual.rs:81 gate() HARD-REFUSES L2/L3<br/>accepts() ritual.rs:119 L0 confirmed+<br/>L1 only verified"]
        CH["Checks registry run_all<br/>ritual.rs:147,157"]
        VF["verify() ritual.rs:362<br/>trail_ok = well_formed AND accepts(verdict)<br/>ritual.rs:386"]
    end
    TR -->|transition output| CS
    CS -->|hashed into| GM
    TR -->|balances recomputed| LE
    GM -->|commit SHA chained by| BT
    VF -->|1 recompute reducer replay genesis..event log| TR
    VF -->|2 replay ledger == stored ledger.json| LE
    VF -->|3 assert git-clean vs last gitmark commit SHA| GM
    VF -->|4 walk EVERY mark not tip only - each output key x of P_i recomputed,<br/>each mark spends its predecessor, each confirmed - reordered states derive<br/>different keys so the walk refuses| BT
    CH -->|3-gate registry| VF
    TL -->|capability gate| VF
    Note1["Note: this is the ADR-124/128 web-ledger contract<br/>layer (legacy numbering, not in the 20xx governing<br/>pack) - the deploy/verify ritual for gitmark.json /<br/>blocktrails.json payment contracts. It is NOT part of<br/>the ACSP proposal/case state machine (VC-24.1-24.7);<br/>included here only because the brief names it as a<br/>governance-adjacent entry point under src/"]
```

## VC-24.11 ACSP decision-projection client — app_state.rs boot + relay connect

```mermaid
sequenceDiagram
    autonumber
    participant BOOT as AppState::new<br/>app_state.rs:1307-1334
    participant ENV as env FORUM_RELAY_URL + panel key:<br/>KEY_FILE vars or privkey vars (acsp/key_file.rs:13-15)
    participant ACSP as AcspClient::connect<br/>acsp/client.rs:78
    participant RELAY as forum relay (nostr_sdk Client, auto-reconnect)
    BOOT->>ENV: read FORUM_RELAY_URL :1308, then panel secret via<br/>key_file::load_panel_secret (file first, env fallback)<br/>app_state.rs:1308-1315 acsp/key_file.rs:310
    alt relay and secret both Some
        BOOT->>ACSP: connect(secret, relay)
        ACSP->>RELAY: add_relay + connect (nostr_sdk relay pool)<br/>acsp/client.rs:83-89
        alt connect Ok
            RELAY-->>ACSP: relay added, nostr-sdk connects in the background and keeps retrying (acsp/client.rs:87-89)
            ACSP-->>BOOT: Arc AcspClient
            BOOT->>BOOT: acsp_client = Some(client)<br/>app_state.rs:1320 log relay set - REST/bridge decisions project as kind-31403
        else connect Err
            ACSP-->>BOOT: Err(e)
            BOOT->>BOOT: acsp_client = None, warn FAILED to connect - degraded, visible<br/>app_state.rs:1324
        end
    else unconfigured or key unusable
        BOOT->>BOOT: acsp_client = None, info OFF (key_file load error logs error!) -<br/>decisions record forum_projection=skipped<br/>app_state.rs:1330
    end
    Note over BOOT,RELAY: DEBT: connect Ok does not prove the relay is reachable - an unreachable relay surfaces later as a publish error,<br/>i.e. forum_projection=failed at decide time, not as the FAILED-to-connect branch above. acsp/client.rs:87-89, handlers/enrichment_proposals_handler.rs:654
    Note over BOOT,RELAY: this is the SAME AcspClient type ElevationActor and<br/>DecisionElevationActor each construct independently at their own<br/>startup (elevation_actor.rs:1056, decision_elevation_actor.rs:521) -<br/>three separate relay connections under three different panel<br/>identities can exist concurrently, all using the same bridge secret
```

## VC-24.12 Decision-page OKF/vault frontmatter — draft, wikilinks and the read-back

```mermaid
flowchart TB
    DEC["ElevatedDecision (from VC-24.6)"] --> DRAFT["draft_decision_page<br/>decision_elevation.rs:257"]
    SLUG["decision_slug summary-slug + 12hex urn tail<br/>decision_elevation.rs:152"] --> DRAFT
    DRAFT --> KEYS["DECISION_RELATION_KEYS causes precedes<br/>influences informed-by regulated-by<br/>decision_elevation.rs:186"]
    KEYS --> WL["wikilinks - each dl edge target wrapped<br/>as double-bracket link, empty set emits<br/>no key at all<br/>decision_elevation.rs:207"]
    WL --> META["vault.PageMeta type=Individual resource=decision_urn<br/>status=draft generated.by=agent_did<br/>extra_lists=BTreeMap key to wikilinks<br/>decision_elevation.rs:282-297, 323"]
    META --> RENDER["vault.render_page meta + body<br/>decision_elevation.rs:318"]
    RENDER --> PAGE["knowledge/decisions/&lt;slug&gt;.md (DECISIONS_DIR, flat vault root)<br/>frontmatter Properties, NOT a fenced JSON-LD block"]
    PAGE --> PARSE["parse_decision_page - type==Individual AND<br/>resource starts urn:agentbox:decision: required,<br/>else None (rejects class pages)<br/>decision_elevation.rs:360"]
    PARSE --> UNWL["unwikilink - strips brackets back to bare URNs<br/>per DECISION_RELATION_KEYS entry<br/>decision_elevation.rs:217"]
    UNWL --> IN["DecisionInput caused precedent_for influenced<br/>considered_inputs governed_by, rehydrated"]
    Note1["INVARIANT: the mapping is authoring-only - build_decision_quads still<br/>emits the same dl: predicates from DecisionInput regardless of how the<br/>page spells the edge, so this table decides HOW an edge is written,<br/>never WHAT it means in the graph. decision_elevation.rs:165-173"]
    Note2["DOC-DRIFT resolved: earlier VC-24 draft cited a JSON-LD fenced-block<br/>format for both class pages and decision pages. Both now emit vault<br/>YAML frontmatter (OKF Properties) - class pages via<br/>elevation_actor.rs:510 draft_class_page (see VC-23.9), decision<br/>pages via this module. Neither path writes a fenced JSON-LD block."]
```

## VC-24.13 GET /api/broker/inbox — opt-in Jev proposal triage (VC-24-S1-01)

```mermaid
sequenceDiagram
    autonumber
    participant BR as agentbox broker-bridge
    participant H as inbox<br/>handlers/broker_inbox_handler.rs:220
    participant ST as store::all projection over SqliteEnrichmentRepository<br/>handlers/enrichment_proposals_handler.rs:742
    participant TR as assemble_inbox<br/>handlers/broker_inbox_handler.rs:193
    participant JV as visionclaw_jev proposal_triage<br/>broker_inbox_handler.rs:42

    BR->>H: GET /api/broker/inbox - scope wrapped in RequireAuth::power_user, service credential accepted<br/>broker_inbox_handler.rs:248-252
    H->>ST: store::all - up to 500 proposals via repo.list, the bridge filters category and status itself<br/>broker_inbox_handler.rs:221 handlers/enrichment_proposals_handler.rs:742-753
    H->>TR: assemble_inbox(proposals, jev_triage())<br/>broker_inbox_handler.rs:222
    alt Jev unconfigured, item off or no endpoint - library default off, Compose sets VISIONCLAW_JEV=1 at docker-compose.unified.yml:8 (jev_triage resolves to None once via OnceLock)
        TR-->>H: plain projection, same cases, same order, no jev key<br/>broker_inbox_handler.rs:156-168, :195-198
    else triage state present
        TR->>JV: triage_queue over facts for pending or claimed cases only<br/>broker_inbox_handler.rs:172-173, :199-200
        JV-->>TR: ordered proposals plus per-case annotation, a fallback leaves the case unannotated
        TR-->>H: cases ordered by triage priority within the slots annotated cases already occupy<br/>broker_inbox_handler.rs:189-192, :201-210
    end
    H-->>BR: 200 {cases, total} - total is the unfiltered proposal count<br/>broker_inbox_handler.rs:210, :224
    Note over JV,H: INVARIANT: Jev annotates and orders only - it never approves, rejects or promotes, and the decide route (VC-24.3), signed decisions and receipts are untouched.<br/>crates/visionclaw-jev/src/proposal_triage.rs:8-10
    Note over TR,JV: OPEN: the proposal text sent to Jev is private, so PROPOSAL_TRIAGE is local-only and refuses the proxy and public endpoints unless the endpoint is declared local<br/>docs/reference/configuration.md:461 - the supplied Compose environment declares it local and<br/>turns the master flag on, so triage runs there once recreated (activation staged)<br/>docker-compose.unified.yml:8, :9, :11, docs/BASELINE-architecture.md:49-50
    Note over H,ST: DEBT: total is proposals.len() after store::all, which is capped at ALL_LIMIT 500 rows newest-updated first, so total is not the table size once more than 500 proposals exist<br/>handlers/enrichment_proposals_handler.rs:738-745, handlers/broker_inbox_handler.rs:194
```
