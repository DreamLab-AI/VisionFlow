---
id: NF-06
title: Agent Control Surface Protocol — kinds 31400-31405, the broker aggregate and its consumers
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2010]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0002_governance.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0005_governance_receipts.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/board.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/panel_registry.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
verified_commit: 380a595f150dd96bfe27ff278fff9ded1be7fbd0
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
        ActionRequest : nostr-bbs-core/src/governance.rs:188
        ActionResponse : nostr-bbs-core/src/governance.rs:204
    }
    class RegisteredAgent {
        agent registry record : nostr-bbs-core/src/governance.rs:332
    }
    ACSKinds --> TypedPayloads
    ACSKinds --> RegisteredAgent

    note for ACSKinds "This repo OWNS the 31400-31405 schema for the estate. Every kind constant is re-exported at crate level so no consumer hardcodes a number nostr-bbs-core/src/lib.rs:133"
    note for TypedPayloads "DOC-DRIFT: the README table (README.md:222-229) and this module's own doc table (nostr-bbs-core/src/governance.rs:9-14) name six types, but only THREE have a Rust struct - PanelDefinition, ActionRequest and ActionResponse. PanelState, PanelUpdate and PanelRetired exist as kind constants and doc rows only."
    note for RegisteredAgent "INVARIANT: only kinds 31400, 31401, 31402, 31404 and 31405 are agent-published. 31403 is the HUMAN half - see NF-06.3"
```

## NF-06.2 Addressability and validation

```mermaid
flowchart TB
    V["validate_governance_event<br/>nostr-bbs-core/src/governance.rs:304"]
    K["(a) kind must be in the governance range<br/>nostr-bbs-core/src/governance.rs:315 via is_governance_kind nostr-bbs-core/src/governance.rs:211"]
    D["non-empty d tag required - all six kinds are<br/>NIP-33 parameterised-replaceable nostr-bbs-core/src/governance.rs:322"]
    AUD["31405 audit entries are APPEND-ONLY: a repeated d tag<br/>is rejected as a duplicate nostr-bbs-core/src/governance.rs:329"]
    HELP["Tag helpers<br/>extract_d_tag nostr-bbs-core/src/governance.rs:215<br/>extract_tag nostr-bbs-core/src/governance.rs:222<br/>extract_e_tag_with_marker nostr-bbs-core/src/governance.rs:239<br/>extract_supersedes_target nostr-bbs-core/src/governance.rs:251<br/>extract_appeal_target nostr-bbs-core/src/governance.rs:257"]

    V --> K --> D --> AUD
    V --> HELP

    N1["INVARIANT: d-tag addressability is what makes a panel replaceable - publish the same d again and<br/>the panel updates in place rather than duplicating nostr-bbs-core/src/governance.rs:320-322"]
    N2["KIND_GOVERNANCE_AUDIT_LOG is numerically the SAME kind as KIND_PANEL_RETIRED - both 31405<br/>nostr-bbs-core/src/governance.rs:295 and nostr-bbs-core/src/governance.rs:32. The two roles are distinguished only by whether the caller<br/>supplies a seen_audit_ids set, so a PanelRetired and an audit entry are indistinguishable on the wire."]
    N3["The append-only rule exists because a same-d replay would otherwise OVERWRITE the prior audit<br/>entry, which is exactly what an audit log must not permit nostr-bbs-core/src/governance.rs:293-295"]
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
    R->>REG: is_registered_agent gate nip_handlers.rs:724
    R-->>FC: subscription on 31400-31405 nostr-bbs-forum-client/src/app.rs:767
    AG->>R: kind 31402 ActionRequest
    R->>R: project_action_request into broker_cases nip_handlers.rs:925
    FC->>FC: ingest_event into the panel registry nostr-bbs-forum-client/src/app.rs:773
    FC-->>HU: render the decision card
    HU->>FC: approve / reject
    FC->>R: kind 31403 ActionResponse nostr-bbs-forum-client/src/pages/governance.rs:473
    R->>R: admin-only gate nip_handlers.rs:740 via governance_response_blocked nip_handlers.rs:130
    R->>R: project_action_response nip_handlers.rs:938
    R-->>AG: subscription on 31403

    Note over R: INVARIANT P1-6: a Decision is a PRIVILEGED act, not a generic member action - kind 31403 from a non-admin is blocked outright nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:715-717
    Note over R: 31403 is EXEMPT from the agent-registry gate - it is the human half of the protocol nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:725
    Note over FC: The published response carries an e-tag naming the request it answers nostr-bbs-forum-client/src/pages/governance.rs:476
```

## NF-06.4 The two gates a governance event must pass at ingress

```mermaid
flowchart TB
    EV["governance-kind event at the relay"]
    G1{"is_governance_kind AND kind != 31403<br/>AND not a kanban approval request<br/>AND not a registered agent<br/>nip_handlers.rs:724"}
    G2{"governance_response_blocked - 31403 from a non-admin<br/>nip_handlers.rs:740"}
    G3{"a 31403 carrying a supersedes e-tag<br/>supersession_authorised<br/>nip_handlers.rs:757"}
    OK["saved and projected"]

    EV --> G1
    G1 -->|"unregistered"| B1["blocked: pubkey not in agent registry nip_handlers.rs:733"]
    G1 -->|"pass"| G2
    G2 -->|"non-admin"| B2["blocked: admin-only governance action response nip_handlers.rs:745"]
    G2 -->|"pass"| G3
    G3 -->|"unauthorised"| B3["blocked: unauthorised supersession nip_handlers.rs:763"]
    G3 -->|"pass"| OK

    N1["Kanban exception: a 31402 tagged k=30302 is a MEMBER-initiated ask - may this card enter the<br/>approval-gated column - so it is admitted from any whitelisted author. Decisions stay admin-only;<br/>every other 31402 remains registry-gated nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:719-723"]
    N2["F6 supersession authority: only the ORIGINAL decision's signer, or a human of a strictly higher<br/>governance role, may supersede a published decision - rejected BEFORE the event is saved or<br/>projected nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:750-756"]
    N3["The whitelist gate ran earlier in the pipeline, so registry membership is an ADDITIONAL<br/>requirement on top of forum membership - see NF-03.4 step 6b"]
```

## NF-06.5 Risk tiers — the agent's own declaration

```mermaid
stateDiagram-v2
    [*] --> Medium: default and fallback<br/>nostr-bbs-core/src/governance.rs:152
    [*] --> Low: "low"<br/>nostr-bbs-core/src/governance.rs:151
    [*] --> High: "high"
    [*] --> Critical: "critical"
    Low --> Suppressed: is_member_suppressed<br/>nostr-bbs-core/src/governance.rs:183
    Medium --> Shown
    High --> Shown
    Critical --> Shown

    note right of Medium
        FAIL-OPEN ON VISIBILITY: an unlabelled or unrecognised tier
        parses to Medium and is therefore SHOWN, not hidden
        parse nostr-bbs-core/src/governance.rs:169
        The field is optional on ActionRequest, absent on legacy requests
        nostr-bbs-core/src/governance.rs:194
    end note
    note right of Suppressed
        Only Low is suppressed from the member surface; medium and above
        always warrant member attention nostr-bbs-core/src/governance.rs:180-182
    end note
    note right of Critical
        The relay projects an escalation DEFAULT in its NIP-11 document so a mesh
        agent can discover the posture: ESCALATION_DEFAULT_TIER = medium,
        ESCALATION_DEFAULT_POSTURE = escalate_to_human
        nostr-bbs-relay-worker/wrangler.toml:48 nostr-bbs-relay-worker/wrangler.toml:49
        Declared a SCAFFOLD - the authoritative schema is owned by agentbox,
        EXTERNAL: see AB-15. An unrecognised tier folds to medium.
    end note
```

## NF-06.6 The broker case aggregate

```mermaid
stateDiagram-v2
    [*] --> Open: 31402 ActionRequest projected<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:929
    Open --> Reopened: 31402 carrying an appeal e-tag<br/>project_appeal nip_handlers.rs:927
    Open --> Decided: 31403 routed through the DecisionOrchestrator<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:166-169
    Decided --> Superseded: 31403 carrying a supersedes e-tag<br/>project_supersession nip_handlers.rs:940
    Decided --> [*]

    note right of Decided
        DecisionOutcome::from_response_content parses the 31403 content JSON into a
        typed outcome nostr-bbs-core/src/governance.rs:571, with an optional detail
        payload nostr-bbs-core/src/governance.rs:576
        A delegate / promote / precedent outcome now reaches its matching CaseState
        instead of the former fixed under_review fallback
        nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:163-169
    end note
    note right of Open
        INVARIANT self-review forbidden: a broker may not decide their own case -
        CaseError::SelfReview nostr-bbs-core/src/governance.rs:355, enforced in
        record_decision nostr-bbs-core/src/governance.rs:770 and again on the
        supersession path nostr-bbs-core/src/governance.rs:862
    end note
    note right of Superseded
        A response arriving before its request is unresolved and cannot invent a case.
        Redelivery may reconcile after the request has projected.
    end note
```

## NF-06.7 ADR-2010 receipt stages — what the relay's OK actually certifies

```mermaid
flowchart LR
    SIGNED["signed"] --> ACCEPTED["relay-accepted<br/>= the OK on the wire"]
    ACCEPTED --> PROJ["projection-committed"]
    PROJ -. "cross-repo contract still proposed" .-> RECV["consumer-received"]
    RECV -. "requires mutation-owner receipt" .-> APPLIED["applied / rejected"]

    OKNOW["Today the relay OK certifies STORAGE ONLY<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:877"]
    WARN["If the receipt is not applied the relay logs<br/>accepted but not applied rather than letting its OK stand<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:947"]
    TABLE["governance_receipts table<br/>migration 0005_governance_receipts.sql:12<br/>bootstrapped inline nostr-bbs-relay-worker/src/lib.rs:744"]

    ACCEPTED -.-> OKNOW
    PROJ -.-> WARN
    PROJ --> TABLE

    N1["DIVERGENCE ADR-2010 is PROPOSED, implementation partial, activation INACTIVE - see the ledger row in<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:1. The complete receipt contract requires<br/>agreement with the authority consumer and the mutation owner plus failure/restart evidence."]
    N2["INVARIANT already live: the relay does NOT let a bare OK imply the decision was applied - the gap<br/>between accepted and applied is logged, not hidden nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:942-945"]
    N3["EXTERNAL: consumer-received and applied are the OTHER repos' halves - VisionClaw's elevation queue<br/>see VC-24, agentbox's approvals pipeline see AB-14, estate view ES-05"]
```

## NF-06.8 D1 projection tables

```mermaid
erDiagram
    agent_registry ||--o{ broker_cases : "opens"
    broker_cases ||--o{ broker_decisions : "decided by"
    broker_cases ||--o{ governance_receipts : "receipts for"
    broker_roles ||--o{ broker_decisions : "authorises"

    agent_registry {
        migration_0002 line5
        inline_bootstrap relay_lib_695
        admin_gated_upsert governance_api_284
    }
    broker_cases {
        migration_0002 line17
        inline_bootstrap relay_lib_705
        category_state_created_by nip_handlers_137
    }
    broker_decisions {
        migration_0002 line39
        inline_bootstrap relay_lib_727
        outcome_and_detail nip_handlers_151
    }
    broker_roles {
        migration_0002 line52
        inline_bootstrap relay_lib_762
    }
    governance_receipts {
        migration_0005 line12
        inline_bootstrap relay_lib_744
    }
```

The auth-worker writes `agent_registry` and `whitelist` through the shared `RELAY_DB`
binding so the relay DO can read the registry at admission with no cross-worker call
(`nostr-bbs-auth-worker/src/governance_api.rs:30`).

## NF-06.9 Agent lifecycle through the auth-worker REST surface

```mermaid
sequenceDiagram
    autonumber
    participant AD as Admin (NIP-98)
    participant API as auth-worker governance_api
    participant RD as relay D1 (RELAY_DB)
    participant R as relay-worker admission

    AD->>API: POST /api/governance/agents/register governance_api.rs:259
    API->>RD: INSERT OR REPLACE agent_registry governance_api.rs:284
    AD->>API: POST /api/governance/agents/provision governance_api.rs:333
    API->>RD: whitelist INSERT governance_api.rs:367
    API->>RD: agent_registry upsert reusing the same identity governance_api.rs:381
    AD->>API: POST /api/governance/agents/revoke governance_api.rs:408
    API->>RD: UPDATE agent_registry SET active = 0 governance_api.rs:427
    R->>RD: is_registered_agent at every governance-kind admission

    Note over API: Provisioning is a TWO-table act - allowlist write plus registry upsert - so an agent that can publish is also a forum member governance_api.rs:310-317
    Note over API: Register, provision, revoke and role grant/revoke are require_admin - list agents, cases, decisions and roles are require_authed - see NF-02.6
    Note over R: EXTERNAL: agents mint their own did:nostr key at spawn in agentbox and are registered here - see AB-11 and ES-04
```

## NF-06.10 Consumers of the surface

```mermaid
flowchart TB
    RELAY["relay-worker - schema owner and gate"]
    FCADMIN["forum-client /governance/admin<br/>publishes 31403 nostr-bbs-forum-client/src/pages/governance.rs:473"]
    FCMEM["forum-client /governance member view<br/>read-only, no 31403 publish path compiles<br/>nostr-bbs-forum-client/src/pages/governance.rs:41"]
    BOARD["forum-client kanban board<br/>subscribes to 31402 and 31403 only<br/>nostr-bbs-forum-client/src/pages/board.rs:392"]
    REG["PanelRegistry store<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:242"]
    BBS["bbs-client governance bucket<br/>nostr-bbs-bbs-client/src/relay.rs:583"]
    KAN["kanban approval decisions are parsed FROM 31403<br/>nostr-bbs-core/src/kanban.rs:739 nostr-bbs-core/src/kanban.rs:741"]

    RELAY --> FCADMIN & FCMEM & BOARD & BBS
    FCADMIN --> REG
    BOARD --> KAN

    N1["The member view is enforced by COMPOSITION, not by a runtime flag - it mounts components that do<br/>not compile a 31403 publish path nostr-bbs-forum-client/src/pages/governance.rs:30"]
    N2["The registry resolves a decision chain to the most recent AUTHORISED 31403; superseded events<br/>remain in the store rather than being deleted nostr-bbs-forum-client/src/stores/panel_registry.rs:242"]
    N3["EXTERNAL: exactly ONE live consumer today - ontology-concept elevation in VisionClaw, a case queue<br/>capped at five concurrent README.md:255. Treat universal human-in-the-loop surface as the design<br/>target, not a claim of many production consumers. See VC-24."]
```

## NF-06.11 Guarded relay projection and separate consumer receipts

```mermaid
flowchart TB
    EVENT["Stored signed response; relay OK still means storage"] --> CORR["Require matching event, case, request, signer and outcome at apply"]
    CORR --> REPLAY["Completed receipt checked before planning against terminal case"]
    REPLAY --> PLAN["Existing case only; reject unknown persisted state"]
    PLAN --> ACCEPT["Record relay-accepted receipt by full event ID"]
    ACCEPT --> SQL["INSERT decision only if request, prior state, latest decision and receipt match"]
    SQL --> CASE["Case UPDATE only if decision changed one row"]
    CASE --> RECEIPT["Receipt UPDATE only if case changed one row"]
    RECEIPT --> CHECK["All three affected-row counts must equal one"]
    CHECK --> LIMIT["Relay projection receipt is not external application proof<br/>Agentbox and VisionClaw now retain separate bound consumer records"]
```

Execution update, 2026-09-07: request redelivery uses INSERT OR IGNORE and cannot reset an already-decided case. D1 statements execute serially in one transaction, and each dependent write requires its predecessor to change one row. Three exact-SQL projection tests cover wrong/missing request/case/receipt, prior decision/state races, duplicate commits and rollback on receipt failure; native relay tests cover correlation mismatch and unknown-state rejection. The earlier default-case and unconditional-update findings remain in the [audit](../../estate-review/2026-09-07-federation-audit.md). Agentbox now signs operation/digest and records received/outcome receipts; VisionClaw journals exact requests and claims PR application. Live D1 and external application are not certified by local tests.
