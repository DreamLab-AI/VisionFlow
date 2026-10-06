---
id: NF-06
title: Agent Control Surface Protocol — kinds 31400-31405, the broker aggregate and its consumers
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2010, ADR-2011, ADR-2014]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/ontology_governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/cron.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0002_governance.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0007_ontology_governance.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/board.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/panel_registry.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/governance_view.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/case_projection.rs
  - ../nostr-rust-forum/crates/nostr-bbs-governance-probe/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-governance-probe/src/main.rs
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
  - ../nostr-rust-forum/docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md
  - ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md
verified_commit: {nostr-rust-forum: 72463fbde35ac4c68539b1f65a08ff03b9941201, dreamlab-ai-website: ebaf16c0462407ba4eb09dcc3220a1846b0d5c80}
---

## NF-06.1 The six kinds and their publishers

```mermaid
classDiagram
    class ACSKinds {
        KIND_PANEL_DEFINITION 31400 : nostr-bbs-core/src/governance.rs:27
        KIND_PANEL_STATE 31401 : nostr-bbs-core/src/governance.rs:28
        KIND_ACTION_REQUEST 31402 : nostr-bbs-core/src/governance.rs:29
        KIND_ACTION_RESPONSE 31403 : nostr-bbs-core/src/governance.rs:30
        KIND_PANEL_UPDATE 31404 : nostr-bbs-core/src/governance.rs:31
        KIND_PANEL_RETIRED 31405 : nostr-bbs-core/src/governance.rs:32
        GOVERNANCE_KIND_RANGE 31400..=31405 : nostr-bbs-core/src/governance.rs:34
    }
    class TypedPayloads {
        PanelDefinition : nostr-bbs-core/src/governance.rs:102
        ActionRequest : nostr-bbs-core/src/governance.rs:1085
        ActionResponse : nostr-bbs-core/src/governance.rs:1111
    }
    class RegisteredAgent {
        agent registry record : nostr-bbs-core/src/governance.rs:1263
    }
    ACSKinds --> TypedPayloads
    ACSKinds --> RegisteredAgent

    note for ACSKinds "This repo OWNS the 31400-31405 schema for the estate. Every kind constant is re-exported at crate level so no consumer hardcodes a number nostr-bbs-core/src/lib.rs:150"
    note for TypedPayloads "DOC-DRIFT: the README table (README.md:250-255) and this module's own doc table (nostr-bbs-core/src/governance.rs:9-16) name six types, but only THREE have a Rust struct - PanelDefinition, ActionRequest and ActionResponse. PanelState, PanelUpdate and PanelRetired exist as kind constants and doc rows only."
    note for RegisteredAgent "INVARIANT: only kinds 31400, 31401, 31402, 31404 and 31405 are agent-published. 31403 is the HUMAN half - see NF-06.3"
```

## NF-06.2 Addressability and validation

```mermaid
flowchart TB
    V["validate_governance_event<br/>nostr-bbs-core/src/governance.rs:1235"]
    K["(a) kind must be in the governance range<br/>nostr-bbs-core/src/governance.rs:1216 via is_governance_kind nostr-bbs-core/src/governance.rs:1142"]
    D["non-empty d tag required - all six kinds are<br/>NIP-33 parameterised-replaceable nostr-bbs-core/src/governance.rs:1245"]
    AUD["31405 audit entries are APPEND-ONLY: a repeated d tag<br/>is rejected as a duplicate nostr-bbs-core/src/governance.rs:1254"]
    HELP["Tag helpers<br/>extract_d_tag nostr-bbs-core/src/governance.rs:1146<br/>extract_tag nostr-bbs-core/src/governance.rs:1153<br/>extract_e_tag_with_marker nostr-bbs-core/src/governance.rs:1170<br/>extract_supersedes_target nostr-bbs-core/src/governance.rs:1182<br/>extract_appeal_target nostr-bbs-core/src/governance.rs:1188"]

    V --> K --> D --> AUD
    V --> HELP

    N1["INVARIANT: d-tag addressability is what makes a panel replaceable - publish the same d again and<br/>the panel updates in place rather than duplicating nostr-bbs-core/src/governance.rs:1245-1249"]
    N2["KIND_GOVERNANCE_AUDIT_LOG is numerically the SAME kind as KIND_PANEL_RETIRED - both 31405<br/>nostr-bbs-core/src/governance.rs:1226 and nostr-bbs-core/src/governance.rs:32. The two roles are distinguished only by whether the caller<br/>supplies a seen_audit_ids set, so a PanelRetired and an audit entry are indistinguishable on the wire."]
    N3["The append-only rule exists because a same-d replay would otherwise OVERWRITE the prior audit<br/>entry, which is exactly what an audit log must not permit nostr-bbs-core/src/governance.rs:1213-1215"]
```

## NF-06.3 Publish path — agent asks, human signs

```mermaid
sequenceDiagram
    autonumber
    participant AG as Agent (registered pubkey)
    participant R as relay-worker DO
    participant REG as agent_registry (relay D1)
    participant FC as forum-client PanelRegistry
    participant HU as Human admin

    AG->>R: kind 31400 PanelDefinition
    R->>REG: is_registered_agent gate nip_handlers.rs:1037
    R-->>FC: subscription on 31400-31405 nostr-bbs-forum-client/src/app.rs:852
    AG->>R: kind 31402 ActionRequest
    R->>R: project_action_request into broker_cases nip_handlers.rs:2398
    FC->>FC: ingest_event into the panel registry nostr-bbs-forum-client/src/app.rs:858
    FC-->>HU: render the decision card
    HU->>FC: approve / reject on the decision card
    FC->>R: kind 31403 ActionResponse nostr-bbs-forum-client/src/pages/governance.rs:919
    R->>R: admin-or-delegated-reviewer gate nip_handlers.rs:1064 via response_admission nip_handlers.rs:234
    R->>R: correlate reads the request id, falling back to the first unmarked e tag nostr-bbs-relay-worker/src/relay_do/receipts.rs:130
    R->>R: project_action_response nip_handlers.rs:2606
    R-->>AG: subscription on 31403

    Note over R: INVARIANT P1-6+FR6.2: a Decision is a PRIVILEGED act, not a generic member action - kind 31403 from a non-admin, non-delegated-reviewer is blocked outright nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1054-1067
    Note over R: 31403 is EXEMPT from the agent-registry gate - it is the human half of the protocol nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1035
    Note over FC: The published response carries an UNMARKED e-tag naming the request it answers nostr-bbs-forum-client/src/pages/governance.rs:923 - the relay read only a request-marked tag until a5b809e, so every UI decision was stored and never projected - see NF-06.17
```

## NF-06.4 The two gates a governance event must pass at ingress

```mermaid
flowchart TB
    EV["governance-kind event at the relay"]
    G1{"is_governance_kind AND kind != 31403<br/>AND not a kanban approval request<br/>AND not a registered agent<br/>nip_handlers.rs:1001"}
    G2{"response_admission: admin admits any case; a reviewer<br/>admits only a case delegated to them; else blocked (FR6.2)<br/>nip_handlers.rs:1064 via response_admission nip_handlers.rs:234"}
    G3{"a 31403 carrying a supersedes e-tag<br/>supersession_authorised<br/>nip_handlers.rs:1102"}
    OK["saved and projected"]

    EV --> G1
    G1 -->|"unregistered"| B1["blocked: pubkey not in agent registry nip_handlers.rs:1043"]
    G1 -->|"pass"| G2
    G2 -->|"not admitted"| B2["blocked: admin-only governance action response, or<br/>case not delegated to this reviewer nip_handlers.rs:1066"]
    G2 -->|"admitted"| G3
    G3 -->|"unauthorised"| B3["blocked: unauthorised supersession nip_handlers.rs:1105"]
    G3 -->|"pass"| OK

    N1["Kanban exception: a 31402 tagged k=30302 is a MEMBER-initiated ask - may this card enter the<br/>approval-gated column - so it is admitted from any whitelisted author. Decisions stay admin/reviewer-only;<br/>every other 31402 remains registry-gated nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1029-1032"]
    N2["F6 supersession authority: only the ORIGINAL decision's signer, or a human of a strictly higher<br/>governance role, may supersede a published decision - rejected BEFORE the event is saved or<br/>projected nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1099-1105"]
    N3["The whitelist gate ran earlier in the pipeline, so registry membership is an ADDITIONAL<br/>requirement on top of forum membership - see NF-03.4 step 6b"]
```

## NF-06.6 The broker case aggregate — decision outcomes including ADR-2013 Promote/Demote

```mermaid
stateDiagram-v2
    [*] --> Open: 31402 ActionRequest projected<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2398
    Open --> Reopened: 31402 carrying an appeal e-tag<br/>project_appeal nip_handlers.rs:3028
    Open --> Decided: 31403 routed through the DecisionOrchestrator<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:523-524
    Decided --> Superseded: 31403 carrying a supersedes e-tag<br/>project_supersession nip_handlers.rs:2893
    Decided --> [*]

    note right of Decided
        DecisionOutcome::from_response_content parses the 31403 content JSON into a
        typed outcome nostr-bbs-core/src/governance.rs:1536, with an optional detail
        payload nostr-bbs-core/src/governance.rs:1543 (delegate_to, iri or scope)
        A delegate / promote / demote / precedent outcome now reaches its matching
        CaseState instead of the former fixed under_review fallback
        nostr-bbs-core/src/governance.rs:1776-1782
    end note
    note right of Open
        INVARIANT self-review forbidden: a broker may not decide their own case -
        CaseError::SelfReview nostr-bbs-core/src/governance.rs:1286, enforced in
        record_decision nostr-bbs-core/src/governance.rs:1738 and again on the
        supersession path nostr-bbs-core/src/governance.rs:1835
    end note
    note right of Superseded
        ADR-2013: Promote raises a corpus subject to status:stable and Demote
        lowers it to status:deprecated - its exact inverse, both carrying the
        subject IRI redundantly with the 31402's own context_url tag so the
        signed decision can never be repointed by a later edit to the request
        nostr-bbs-core/src/governance.rs:1494-1509
    end note
```

## NF-06.8 D1 projection tables

```mermaid
erDiagram
    agent_registry ||--o{ broker_cases : "opens"
    broker_cases ||--o{ broker_decisions : "decided by"
    broker_cases ||--o{ governance_receipts : "receipts for"
    broker_roles ||--o{ broker_decisions : "authorises"
    broker_cases ||--o{ case_delegations : "scoped to one case"
    broker_cases ||--o{ case_side_receipts : "ageing and expiry"

    agent_registry {
        migration_0002 line5
        inline_bootstrap relay_lib_813
        admin_gated_upsert governance_api_380
    }
    broker_cases {
        migration_0002 line17
        inline_bootstrap relay_lib_823
        category_state_created_by nip_handlers_137
        stale_after migration_0007_line16_nullable_ontology_expiry
    }
    broker_decisions {
        migration_0002 line39
        inline_bootstrap relay_lib_845
        outcome_and_detail nip_handlers_151
    }
    broker_roles {
        migration_0002 line52
        inline_bootstrap relay_lib_880
    }
    governance_receipts {
        migration_0005 line12
        inline_bootstrap relay_lib_862
        application_stage_columns applied_at_applied_by
    }
    case_delegations {
        migration_0006 mirrored_into_ensure_schema
        inline_bootstrap relay_lib_918
    }
    case_side_receipts {
        migration_0006 mirrored_into_ensure_schema
        inline_bootstrap relay_lib_906
        expiry_rows keyed_case_id_stage
    }
```

The auth-worker writes `agent_registry` and `whitelist` through the shared `RELAY_DB`
binding so the relay DO can read the registry at admission with no cross-worker call
(`nostr-bbs-auth-worker/src/governance_api.rs:31`). Migration 0007 is additive and
nullable: a case that is not an ontology proposal has no `stale_after` and is
invisible to the expiry sweep — see NF-06.16.

## NF-06.9 Agent lifecycle through the auth-worker REST surface

```mermaid
sequenceDiagram
    autonumber
    participant AD as Admin (NIP-98)
    participant API as auth-worker governance_api
    participant RD as relay D1 (RELAY_DB)
    participant R as relay-worker admission

    AD->>API: POST /api/governance/agents/register governance_api.rs:355
    API->>RD: INSERT OR REPLACE agent_registry governance_api.rs:381
    AD->>API: POST /api/governance/agents/provision governance_api.rs:434
    API->>RD: whitelist cohort MERGE, never a replace governance_api.rs:466
    API->>RD: agent_registry upsert reusing the same identity governance_api.rs:476
    AD->>API: POST /api/governance/agents/revoke governance_api.rs:503
    API->>RD: UPDATE agent_registry SET active = 0 governance_api.rs:532
    R->>RD: is_registered_agent at every governance-kind admission

    Note over API: Provisioning is a TWO-table act - allowlist write plus registry upsert - so an agent that can publish is also a forum member, issued as one D1 batch so it is all-or-nothing governance_api.rs:420-422 governance_api.rs:490
    Note over API: INVARIANT since 1a26e51 provisioning MERGES the agent cohorts into any row the pubkey already holds, so a re-provision or a human grant is never revoked by it governance_api.rs:429-431 governance_api.rs:464-466
    Note over API: Register, provision, revoke and role grant/revoke are require_admin - list agents, cases, decisions and roles are require_authed - see NF-02.6
    Note over R: EXTERNAL: agents mint their own did:nostr key at spawn in agentbox and are registered here - see AB-11 and ES-04
```

## NF-06.10 Consumers of the surface

```mermaid
flowchart TB
    RELAY["relay-worker - schema owner and gate"]
    FCADMIN["forum-client /governance/admin<br/>decision card publishes 31403 nostr-bbs-forum-client/src/pages/governance.rs:848<br/>panel actions publish 31403 nostr-bbs-forum-client/src/pages/governance.rs:501"]
    FCMEM["forum-client /governance member view<br/>read-only, no 31403 publish path compiles<br/>nostr-bbs-forum-client/src/pages/governance.rs:39"]
    BOARD["forum-client kanban board<br/>subscribes to 31402 and 31403 only<br/>nostr-bbs-forum-client/src/pages/board.rs:46"]
    REG["PanelRegistry store<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:212"]
    BBS["bbs-client governance bucket<br/>nostr-bbs-bbs-client/src/relay.rs:27"]
    KAN["kanban approval decisions are parsed FROM 31403<br/>nostr-bbs-core/src/kanban.rs:739 nostr-bbs-core/src/kanban.rs:745"]

    RELAY --> FCADMIN & FCMEM & BOARD & BBS
    FCADMIN --> REG
    BOARD --> KAN

    N1["The member view is enforced by COMPOSITION, not by a runtime flag - it mounts components that do<br/>not compile a 31403 publish path nostr-bbs-forum-client/src/pages/governance.rs:35"]
    N2["The registry resolves a decision chain to the most recent AUTHORISED 31403; superseded events<br/>remain in the store rather than being deleted nostr-bbs-forum-client/src/stores/panel_registry.rs:486"]
    N3["EXTERNAL: exactly ONE live consumer today - ontology-concept elevation in VisionClaw, a case queue<br/>capped at five concurrent README.md:310-309. Treat universal human-in-the-loop surface as the design<br/>target, not a claim of many production consumers. See VC-24."]
```

## NF-06.11 Guarded relay projection and separate consumer receipts

```mermaid
flowchart TB
    EVENT["Stored signed response; relay OK still means storage<br/>project_action_response nip_handlers.rs:2606"] --> CORR["Require matching event, case, request, signer and outcome at apply<br/>receipts::correlate receipts.rs:115"]
    CORR --> REPLAY["Completed receipt checked before planning against terminal case"]
    REPLAY --> PLAN["Existing case only; reject unknown persisted state"]
    PLAN --> ACCEPT["Record relay-accepted receipt by full event ID"]
    ACCEPT --> SQL["INSERT decision only if request, prior state, latest decision and receipt match"]
    SQL --> CASE["Case UPDATE only if decision changed one row"]
    CASE --> RECEIPT["Receipt UPDATE only if case changed one row"]
    RECEIPT --> CHECK["All three affected-row counts must equal one"]
    CHECK --> LIMIT["Relay projection receipt is not external application proof<br/>Agentbox and VisionClaw now retain separate bound consumer records"]
```

## NF-06.12 Task properties set the escalation boundary — ADR-2011, activation live

```mermaid
flowchart TB
    PANEL["PanelDefinition 31400 declares the task-property triple<br/>tp-verifiability nostr-bbs-core/src/governance.rs:331<br/>tp-reversibility nostr-bbs-core/src/governance.rs:333<br/>tp-stakes nostr-bbs-core/src/governance.rs:335"]
    REQUEST["ActionRequest 31402 may restate the triple<br/>TaskProperties nostr-bbs-core/src/governance.rs:390<br/>from_tags nostr-bbs-core/src/governance.rs:415"]
    MERGE["merge is TIGHTENING ONLY - the result is never looser than the panel<br/>nostr-bbs-core/src/governance.rs:451, over the optional pair<br/>nostr-bbs-core/src/governance.rs:463"]
    TIER["effective_tier - total and pure over four inputs<br/>nostr-bbs-core/src/governance.rs:515"]
    DECL["the agent's own risk_tier is TELEMETRY from here on - it can raise<br/>the tier and never lower it below the properties' floor<br/>nostr-bbs-core/src/governance.rs:508-509"]
    DEF["an entirely unlabelled request folds to the relay's advertised<br/>ESCALATION_DEFAULT_TIER nostr-bbs-core/src/governance.rs:525,<br/>nostr-bbs-relay-worker/wrangler.toml:48"]
    STORE["stored on broker_cases.effective_tier at projection<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2427,<br/>read back nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2458"]

    PANEL --> MERGE
    REQUEST --> MERGE
    MERGE --> TIER
    DECL --> TIER
    DEF --> TIER
    TIER --> STORE

    N1["INVARIANT ADR-2011: the OPERATOR's task properties set the escalation boundary, not the agent's own<br/>self-tiering. Both are read, and the tighter wins nostr-bbs-core/src/governance.rs:515-534"]
    N2["INVARIANT tightening only: a request may move a property up the tightness ordering and never down<br/>nostr-bbs-core/src/governance.rs:228, asserted over all 729 pairs<br/>nostr-bbs-core/src/governance.rs:3113"]
    N3["A case projected before migration 0006 carries no effective_tier and imposes nothing<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:463"]
    N4["RiskTier itself fails open: parse of an unlabelled or unrecognised tier falls back to the<br/>Medium default, so DECL is never silently hidden nostr-bbs-core/src/governance.rs:186,204.<br/>Only Low is ever member-suppressed nostr-bbs-core/src/governance.rs:217"]
```

## NF-06.13 Who may decide a 31403, and what a consequential decision must carry

```mermaid
sequenceDiagram
    autonumber
    participant H as Human or script
    participant R as relay admission<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1054
    participant D1 as broker_roles and case_delegations
    participant AD as response_admission<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:234
    participant RA as check_rationale<br/>nostr-bbs-core/src/governance.rs:764

    H->>R: signed kind 31403
    R->>R: extract the case id from the d tag nip_handlers.rs:1055
    R->>D1: is_reviewer nip_handlers.rs:2501
    R->>D1: is_delegated_for_case nip_handlers.rs:2525
    R->>AD: is_admin, is_reviewer, delegated for THIS case
    AD-->>R: Admit, BlockedNotAuthorised or BlockedNotDelegated nip_handlers.rs:240 nip_handlers.rs:243 nip_handlers.rs:248
    R->>D1: case_effective_tier nip_handlers.rs:1078
    R->>RA: effective tier, action, reasoning nip_handlers.rs:1082
    RA-->>R: rationale_required when High or Critical and the words are missing nostr-bbs-core/src/governance.rs:764 nostr-bbs-core/src/governance.rs:730-733
    R-->>H: OK false with the refusal reason nip_handlers.rs:1066

    Note over R: INVARIANT FR2.2 is enforced at the RELAY, before save_event, because a 31403 is a signed event any client or script can publish directly - a rule that lives only in the forum UI is a suggestion nostr-bbs-core/src/governance.rs:756-760
    Note over RA: Nothing ever FILLS IN a rationale - absence is refused, never papered over nostr-bbs-core/src/governance.rs:761
    Note over RA: The minimum is 20 Unicode SCALARS, not bytes, so a rationale in a multi-byte script is not asked for more words nostr-bbs-core/src/governance.rs:710
    Note over RA: Approve, reject, amend, delegate and - since d5f6308 - promote and demote are gated - only precedent, a scope note on an already-decided case, is not nostr-bbs-core/src/governance.rs:737-744
    Note over AD: One token for both refusals deliberately - telling an unauthorised caller HOW CLOSE they were tells them nothing useful nostr-bbs-core/src/governance.rs:726-729
```

## NF-06.14 Calibration sampling and seeded probes — what the relay actually enforces

```mermaid
flowchart TB
    SUP["is_member_suppressed_effective - a view filter over the EFFECTIVE tier<br/>nostr-bbs-core/src/governance.rs:541"]
    ONLY["only a case the member surface would otherwise HIDE is worth sampling<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:345"]
    CAL["is_calibration_sample - HMAC-SHA256 over the request id<br/>nostr-bbs-core/src/governance.rs:595"]
    KEY["ENV_CALIBRATION_SELECTION_KEY is a SECRET, never a plaintext var<br/>nostr-bbs-core/src/governance.rs:560, nostr-bbs-relay-worker/wrangler.toml:51-54"]
    RATE["rate declared on the panel, default 0.1<br/>nostr-bbs-core/src/governance.rs:122 nostr-bbs-core/src/governance.rs:368"]
    PROBE["a probe tag is honoured ONLY from the panel's registered probe agent<br/>nostr-bbs-core/src/governance.rs:695, dropped from anyone else<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:360-361"]
    BLIND["probe blinding: the tag is filtered at the tag-WRITING trigger<br/>nostr-bbs-relay-worker/src/lib.rs:665, mirrored into the live schema path<br/>nostr-bbs-relay-worker/src/lib.rs:942"]

    SUP --> ONLY --> CAL
    KEY --> CAL
    RATE --> CAL
    PROBE --> BLIND

    N1["INVARIANT the selection is KEYED. The request id is the 31402 d tag the agent chooses freely, so an<br/>unkeyed digest would let it grind ids until one is never sampled - opting itself out of oversight<br/>nostr-bbs-relay-worker/wrangler.toml:56-59"]
    N2["INVARIANT unset key still samples - silently disabling oversight is treated as the worse failure,<br/>and the relay logs a warning on every projection nostr-bbs-relay-worker/wrangler.toml:63-65"]
    N3["The blinding lives in the tag-writing trigger, not a watcher, because SQLite does not fire a trigger<br/>from inside another trigger without recursive_triggers, which is set nowhere here - the watcher was<br/>read as enforcement and was inert nostr-bbs-relay-worker/src/lib.rs:651-657"]
    N4["DIVERGENCE: what the relay enforces is NARROWER than DDD invariant 7. The tag remains on the raw<br/>signed 31402 served over REQ, because removing it would break the signature both clients verify<br/>strictly - blindness on the rendered surface is the CLIENT's responsibility<br/>nostr-bbs-core/src/governance.rs:357-364"]
    N5["The last purge statement is the one the original migration failed to mirror into the live path, so a<br/>deployed relay kept every probe row it had already indexed nostr-bbs-relay-worker/src/lib.rs:678"]
```

## NF-06.15 Two stale-identity classes the registry closed

```mermaid
flowchart TB
    REG["agent registration wrote the pubkey VERBATIM before this fix<br/>nostr-bbs-auth-worker/src/governance_api.rs:86"]
    LOOK["every other path canonicalises to lowercase, and is_registered_agent<br/>looks up with lower(pubkey)<br/>nostr-bbs-auth-worker/src/governance_api.rs:101-105"]
    BUG["so an agent registered with uppercase hex was written and then NEVER matched<br/>by the lookup that decides whether its governance events are admitted"]
    FIX["canonical_agent_pubkey - validate, then lowercase - now called by register too<br/>nostr-bbs-auth-worker/src/governance_api.rs:108"]
    ECHO["the response echoes what was STORED, not what was sent<br/>nostr-bbs-auth-worker/src/governance_api.rs:396-398"]
    NIP33["client side: NIP-33 replaceability per pubkey and d tag - a replayed OLDER<br/>31400 must not roll back the operator's declaration<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:242"]
    HELD["supersedes_held decides it, and accept_panel_state records the newest<br/>31401 or 31404 per address<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:461<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:199"]

    REG --> LOOK --> BUG --> FIX --> ECHO
    NIP33 --> HELD

    N1["INVARIANT: NIP-98 accepts case-insensitive hex and returns event.pubkey verbatim, and the event id is<br/>recomputed over that string, so the SAME key can legitimately present in two casings and both verify<br/>nostr-bbs-auth-worker/src/governance_api.rs:98-100"]
    N2["Canonicalisation is idempotent and still rejects everything it always rejected - length, non-hex and<br/>a blank name nostr-bbs-auth-worker/src/governance_api.rs:87-91"]
    N3["The registry keeps superseded events rather than deleting them - DecisionView tracks whether each<br/>entry is superseded and whether it is the current effective decision<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:174"]
```

## NF-06.16 Ontology proposal expiry — ADR-2013 stale_after and the cron sweep

```mermaid
sequenceDiagram
    autonumber
    participant REQ as 31402 PatchProposal
    participant PB as plan_request_boundary<br/>nip_handlers.rs:311
    participant OG as ontology_governance<br/>ontology_governance.rs
    participant DB as broker_cases.stale_after
    participant CRON as expire_stale_proposals<br/>cron.rs:748

    REQ->>PB: request tags and content
    PB->>OG: stale_after_from_content ontology_governance.rs:251
    OG-->>PB: Option i64, only for a PatchProposal level tag
    PB->>DB: stale_after copied onto the case at projection<br/>migrations/0007_ontology_governance.sql:16

    loop scheduled Worker cron nostr-bbs-relay-worker/src/lib.rs:1094
        CRON->>DB: SELECT id, stale_after WHERE state='pending' AND stale_after IS NOT NULL<br/>cron.rs:762-765
        CRON->>OG: is_expired(stale_after, now) ontology_governance.rs:265
        CRON->>DB: INSERT case_side_receipts (closed without a decision) cron.rs:794-802
        CRON->>DB: UPDATE broker_cases SET state=Closed cron.rs:834
    end

    Note over DB: INVARIANT ADR-2013: additive and nullable - a case that is not an ontology proposal has no<br/>stale_after and is invisible to the sweep migrations/0007_ontology_governance.sql:1-6
    Note over CRON: The sweep closes a case to CaseState::Closed nostr-bbs-core/src/governance.rs:1375 -<br/>closed WITHOUT a decision, never a silent Approve or Reject cron.rs:731
    Note over DB: The index is partial on stale_after IS NOT NULL so the sweep's scan carries only ontology<br/>cases, never every case the forum has opened migrations/0007_ontology_governance.sql:19-22
```

## NF-06.17 The decision card and the relay now agree — tier read and request correlation

```mermaid
flowchart TB
    CARD["decision card boundary computed from the signed events<br/>nostr-bbs-forum-client/src/pages/governance.rs:145"]
    TIERIN["effective_tier_in reads broker_cases.effective_tier from the case projection,<br/>answering only for the card's own author nostr-bbs-forum-client/src/stores/case_projection.rs:126-136"]
    WITH["with_relay_tier adopts the relay's stored tier wherever it has spoken<br/>nostr-bbs-forum-client/src/utils/governance_view.rs:280"]
    GATE["the rationale gate on every control reads that effective tier<br/>nostr-bbs-forum-client/src/pages/governance.rs:969-973"]
    PUB["the card signs d plus an UNMARKED e naming the request<br/>nostr-bbs-forum-client/src/pages/governance.rs:919-923"]
    CORR["correlate: marked request, else appeal target, else first unmarked e<br/>nostr-bbs-relay-worker/src/relay_do/receipts.rs:128-130"]
    UNM["unmarked_tag - fewer than four elements or an empty marker<br/>nostr-bbs-relay-worker/src/relay_do/receipts.rs:159"]
    BIND["the projection still requires that id to equal the case's own nostr_event_id<br/>nostr-bbs-relay-worker/src/relay_do/receipts.rs:126-127"]

    CARD --> WITH
    TIERIN --> WITH --> GATE --> PUB --> CORR
    UNM --> CORR --> BIND

    N1["INVARIANT: an unrecognised stored tier is IGNORED, not parsed - RiskTier::parse maps anything unknown<br/>to Medium, and a garbled column must not loosen a high case<br/>nostr-bbs-forum-client/src/utils/governance_view.rs:276-278"]
    N2["INVARIANT: a tag carrying any other marker - supersedes, appeal - is never read as the request, so<br/>the fallback cannot bind a decision to the wrong case nostr-bbs-relay-worker/src/relay_do/receipts.rs:124-127"]
    N3["DRIFT: the CaseProjection field doc still says effective_tier is carried for cross-checking and is<br/>not yet what the surfaces gate on nostr-bbs-forum-client/src/stores/case_projection.rs:47-48,<br/>while with_relay_tier now makes it the tier the card gates on"]
    N4["Panel actions such as Acknowledge all alerts now also carry an a tag binding them to THIS<br/>author's panel across republication nostr-bbs-forum-client/src/pages/governance.rs:460<br/>nostr-bbs-forum-client/src/pages/governance.rs:501"]
    N5["INVARIANT since ff5780a: a projection row answers only for the card its author published. The map is keyed<br/>by the d tag the requesting agent chooses, which collides across authors, so a second agent reusing a d would<br/>otherwise inherit the first one's tier and calibration flag case_projection.rs:22-28 - is_by matches created_by<br/>case-insensitively case_projection.rs:57-60, and the card passes its own agent pubkey<br/>nostr-bbs-forum-client/src/pages/governance.rs:145-148"]
```

## NF-06.18 The M4 probe suite and what live activation does not establish

```mermaid
flowchart TB
    SUITE["nostr-bbs-governance-probe - the ADR-2011 M4 suite: pure verdicts in the library,<br/>sockets in the binary nostr-bbs-governance-probe/src/lib.rs:1-8"]
    OWN["probes live on their own panel and never need the owner's key<br/>nostr-bbs-governance-probe/src/lib.rs:12-19, PANEL_D nostr-bbs-governance-probe/src/lib.rs:58"]
    TABLE["P01 to P11 - NIP-11 posture, routes, effective_tier stamped high, probe tag blinding,<br/>rationale refusal, system decider refusal, withdrawal nostr-bbs-governance-probe/src/lib.rs:23-35"]
    NOTRUN["NotRun never counts as a pass nostr-bbs-governance-probe/src/lib.rs:43-45"]
    BIN["the binary reads the signing key from a named variable and never prints it<br/>nostr-bbs-governance-probe/src/main.rs:10-11"]
    LIVE["ADR-2011 activation live: run 20261002t132144z passed 11 of 11 on the edge<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:187"]
    REC["ADR-2010 activation staged on the same run - the refusal path is live, the commit path is not<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:7, docs/adr/ADR-2010-durable-governance-outcome-receipts.md:153"]
    PIN["the edge kit pin is now 72463fb, a descendant of the correlation fix a5b809e<br/>../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:30"]

    SUITE --> OWN --> TABLE --> NOTRUN
    SUITE --> BIN
    TABLE --> LIVE --> REC
    PIN --> REC

    N1["OPEN: the requesting agent holds the admin flag, and the relay tells a human from a system by the<br/>31403's own decided_by - so the boundary is as strong as the admin list; not exercised live, on purpose<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:195"]
    N2["OPEN: no human has yet decided a high-tier case; the owner's first such 31403 at projection-committed<br/>is what closes ADR-2010 docs/adr/ADR-2010-durable-governance-outcome-receipts.md:157"]
    N3["RESOLVED 13cbe6c: ADR-2010 now records the correlation fix as on the edge, pinned through 341c5d2<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:155"]
    N4["Probe cases persist as open broker_cases rows after withdrawal, identifiable by subject_kind m4-probe<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:198"]
    N5["RESOLVED at dreamlab ebaf16c: the kit record pins 72463fb in both its SHA column and<br/>CANONICAL_KIT_SHA, its branch column reads main at 72463fb, and its narrative rows now carry the<br/>ontology Promote/Demote gating, the re-posted 31402 and the zone-key grant changes<br/>../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:26"]
```
