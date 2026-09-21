---
id: NF-06
title: Agent Control Surface Protocol — kinds 31400-31405, the broker aggregate and its consumers
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2010, ADR-2011]
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
verified_commit: 2f90c1916
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
        ActionRequest : nostr-bbs-core/src/governance.rs:1078
        ActionResponse : nostr-bbs-core/src/governance.rs:1104
    }
    class RegisteredAgent {
        agent registry record : nostr-bbs-core/src/governance.rs:1232
    }
    ACSKinds --> TypedPayloads
    ACSKinds --> RegisteredAgent

    note for ACSKinds "This repo OWNS the 31400-31405 schema for the estate. Every kind constant is re-exported at crate level so no consumer hardcodes a number nostr-bbs-core/src/lib.rs:134"
    note for TypedPayloads "DOC-DRIFT: the README table (README.md:237-243) and this module's own doc table (nostr-bbs-core/src/governance.rs:9-16) name six types, but only THREE have a Rust struct - PanelDefinition, ActionRequest and ActionResponse. PanelState, PanelUpdate and PanelRetired exist as kind constants and doc rows only."
    note for RegisteredAgent "INVARIANT: only kinds 31400, 31401, 31402, 31404 and 31405 are agent-published. 31403 is the HUMAN half - see NF-06.3"
```

## NF-06.2 Addressability and validation

```mermaid
flowchart TB
    V["validate_governance_event<br/>nostr-bbs-core/src/governance.rs:1204"]
    K["(a) kind must be in the governance range<br/>nostr-bbs-core/src/governance.rs:1209 via is_governance_kind nostr-bbs-core/src/governance.rs:1111"]
    D["non-empty d tag required - all six kinds are<br/>NIP-33 parameterised-replaceable nostr-bbs-core/src/governance.rs:1215"]
    AUD["31405 audit entries are APPEND-ONLY: a repeated d tag<br/>is rejected as a duplicate nostr-bbs-core/src/governance.rs:1222"]
    HELP["Tag helpers<br/>extract_d_tag nostr-bbs-core/src/governance.rs:1115<br/>extract_tag nostr-bbs-core/src/governance.rs:1122<br/>extract_e_tag_with_marker nostr-bbs-core/src/governance.rs:1139<br/>extract_supersedes_target nostr-bbs-core/src/governance.rs:1151<br/>extract_appeal_target nostr-bbs-core/src/governance.rs:1157"]

    V --> K --> D --> AUD
    V --> HELP

    N1["INVARIANT: d-tag addressability is what makes a panel replaceable - publish the same d again and<br/>the panel updates in place rather than duplicating nostr-bbs-core/src/governance.rs:1213-1215"]
    N2["KIND_GOVERNANCE_AUDIT_LOG is numerically the SAME kind as KIND_PANEL_RETIRED - both 31405<br/>nostr-bbs-core/src/governance.rs:1195 and nostr-bbs-core/src/governance.rs:32. The two roles are distinguished only by whether the caller<br/>supplies a seen_audit_ids set, so a PanelRetired and an audit entry are indistinguishable on the wire."]
    N3["The append-only rule exists because a same-d replay would otherwise OVERWRITE the prior audit<br/>entry, which is exactly what an audit log must not permit nostr-bbs-core/src/governance.rs:1192-1195"]
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
    R->>REG: is_registered_agent gate nip_handlers.rs:981
    R-->>FC: subscription on 31400-31405 nostr-bbs-forum-client/src/app.rs:767
    AG->>R: kind 31402 ActionRequest
    R->>R: project_action_request into broker_cases nip_handlers.rs:1220
    FC->>FC: ingest_event into the panel registry nostr-bbs-forum-client/src/app.rs:773
    FC-->>HU: render the decision card
    HU->>FC: approve / reject
    FC->>R: kind 31403 ActionResponse nostr-bbs-forum-client/src/pages/governance.rs:412
    R->>R: admin-only gate nip_handlers.rs:1001 via governance_response_blocked nip_handlers.rs:234
    R->>R: project_action_response nip_handlers.rs:1237
    R-->>AG: subscription on 31403

    Note over R: INVARIANT P1-6: a Decision is a PRIVILEGED act, not a generic member action - kind 31403 from a non-admin is blocked outright nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:971-974
    Note over R: 31403 is EXEMPT from the agent-registry gate - it is the human half of the protocol nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:982
    Note over FC: The published response carries an e-tag naming the request it answers nostr-bbs-forum-client/src/pages/governance.rs:415
```

## NF-06.4 The two gates a governance event must pass at ingress

```mermaid
flowchart TB
    EV["governance-kind event at the relay"]
    G1{"is_governance_kind AND kind != 31403<br/>AND not a kanban approval request<br/>AND not a registered agent<br/>nip_handlers.rs:981"}
    G2{"governance_response_blocked - 31403 from a non-admin<br/>nip_handlers.rs:1001"}
    G3{"a 31403 carrying a supersedes e-tag<br/>supersession_authorised<br/>nip_handlers.rs:1048"}
    OK["saved and projected"]

    EV --> G1
    G1 -->|"unregistered"| B1["blocked: pubkey not in agent registry nip_handlers.rs:990"]
    G1 -->|"pass"| G2
    G2 -->|"non-admin"| B2["blocked: admin-only governance action response nip_handlers.rs:1013"]
    G2 -->|"pass"| G3
    G3 -->|"unauthorised"| B3["blocked: unauthorised supersession nip_handlers.rs:1054"]
    G3 -->|"pass"| OK

    N1["Kanban exception: a 31402 tagged k=30302 is a MEMBER-initiated ask - may this card enter the<br/>approval-gated column - so it is admitted from any whitelisted author. Decisions stay admin-only;<br/>every other 31402 remains registry-gated nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:976-980"]
    N2["F6 supersession authority: only the ORIGINAL decision's signer, or a human of a strictly higher<br/>governance role, may supersede a published decision - rejected BEFORE the event is saved or<br/>projected nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1041-1047"]
    N3["The whitelist gate ran earlier in the pipeline, so registry membership is an ADDITIONAL<br/>requirement on top of forum membership - see NF-03.4 step 6b"]
```

## NF-06.5 Risk tiers — the agent's own declaration

```mermaid
stateDiagram-v2
    [*] --> Medium: default and fallback<br/>nostr-bbs-core/src/governance.rs:186
    [*] --> Low: "low"<br/>nostr-bbs-core/src/governance.rs:183
    [*] --> High: "high"
    [*] --> Critical: "critical"
    Low --> Suppressed: is_member_suppressed<br/>nostr-bbs-core/src/governance.rs:217
    Medium --> Shown
    High --> Shown
    Critical --> Shown

    note right of Medium
        FAIL-OPEN ON VISIBILITY: an unlabelled or unrecognised tier
        parses to Medium and is therefore SHOWN, not hidden
        parse nostr-bbs-core/src/governance.rs:204
        The field is optional on ActionRequest, absent on legacy requests
        nostr-bbs-core/src/governance.rs:1085
    end note
    note right of Suppressed
        Only Low is suppressed from the member surface; medium and above
        always warrant member attention nostr-bbs-core/src/governance.rs:215-216
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
    [*] --> Open: 31402 ActionRequest projected<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1220
    Open --> Reopened: 31402 carrying an appeal e-tag<br/>project_appeal nip_handlers.rs:1218
    Open --> Decided: 31403 routed through the DecisionOrchestrator<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:503-505
    Decided --> Superseded: 31403 carrying a supersedes e-tag<br/>project_supersession nip_handlers.rs:1231
    Decided --> [*]

    note right of Decided
        DecisionOutcome::from_response_content parses the 31403 content JSON into a
        typed outcome nostr-bbs-core/src/governance.rs:1471, with an optional detail
        payload nostr-bbs-core/src/governance.rs:1478
        A delegate / promote / precedent outcome now reaches its matching CaseState
        instead of the former fixed under_review fallback
        nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:399-402
    end note
    note right of Open
        INVARIANT self-review forbidden: a broker may not decide their own case -
        CaseError::SelfReview nostr-bbs-core/src/governance.rs:1255, enforced in
        record_decision nostr-bbs-core/src/governance.rs:1660 and again on the
        supersession path nostr-bbs-core/src/governance.rs:1762
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
    PROJ --> RECV["consumer-received"]
    RECV --> APPLIED["applied or not-applied or applied-manually<br/>a SET, never an order<br/>nostr-bbs-core/src/governance.rs:891"]

    OKNOW["Today the relay OK certifies STORAGE ONLY<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1168"]
    WARN["If the receipt is not applied the relay logs<br/>accepted but not applied rather than letting its OK stand<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1237"]
    TABLE["governance_receipts table<br/>migration 0005_governance_receipts.sql:12<br/>bootstrapped inline nostr-bbs-relay-worker/src/lib.rs:744"]

    ACCEPTED -.-> OKNOW
    PROJ -.-> WARN
    PROJ --> TABLE

    N1["DIVERGENCE ADR-2010 is still PROPOSED, implementation partial, activation INACTIVE in its ledger row<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:5-7, while the ladder now runs to ten stages<br/>owned by nostr-bbs-core/src/governance.rs:913 and the relay serves the application stages - see NF-11.10"]
    N4["FR4.1 moved the ladder out of the relay into the core crate because the auth worker's receipts<br/>endpoint and the relay projection must agree on it and share no other code<br/>nostr-bbs-core/src/governance.rs:882-885"]
    N2["INVARIANT already live: the relay does NOT let a bare OK imply the decision was applied - the gap<br/>between accepted and applied is logged, not hidden nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1233-1236"]
    N3["EXTERNAL: consumer-received and applied are the OTHER repos' halves - VisionClaw's elevation queue<br/>see VC-24, agentbox's approvals pipeline see AB-14, estate view ES-05"]
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
        inline_bootstrap relay_lib_754
        admin_gated_upsert governance_api_380
    }
    broker_cases {
        migration_0002 line17
        inline_bootstrap relay_lib_764
        category_state_created_by nip_handlers_137
    }
    broker_decisions {
        migration_0002 line39
        inline_bootstrap relay_lib_786
        outcome_and_detail nip_handlers_151
    }
    broker_roles {
        migration_0002 line52
        inline_bootstrap relay_lib_821
    }
    governance_receipts {
        migration_0005 line12
        inline_bootstrap relay_lib_803
        application_stage_columns applied_at_applied_by
    }
    case_delegations {
        migration_0006 mirrored_into_ensure_schema
        inline_bootstrap relay_lib_859
    }
    case_side_receipts {
        migration_0006 mirrored_into_ensure_schema
        inline_bootstrap relay_lib_847
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

    AD->>API: POST /api/governance/agents/register governance_api.rs:354
    API->>RD: INSERT OR REPLACE agent_registry governance_api.rs:380
    AD->>API: POST /api/governance/agents/provision governance_api.rs:431
    API->>RD: whitelist INSERT governance_api.rs:465
    API->>RD: agent_registry upsert reusing the same identity governance_api.rs:479
    AD->>API: POST /api/governance/agents/revoke governance_api.rs:506
    API->>RD: UPDATE agent_registry SET active = 0 governance_api.rs:535
    R->>RD: is_registered_agent at every governance-kind admission

    Note over API: Provisioning is a TWO-table act - allowlist write plus registry upsert - so an agent that can publish is also a forum member governance_api.rs:420-427
    Note over API: Register, provision, revoke and role grant/revoke are require_admin - list agents, cases, decisions and roles are require_authed - see NF-02.6
    Note over R: EXTERNAL: agents mint their own did:nostr key at spawn in agentbox and are registered here - see AB-11 and ES-04
```

## NF-06.10 Consumers of the surface

```mermaid
flowchart TB
    RELAY["relay-worker - schema owner and gate"]
    FCADMIN["forum-client /governance/admin<br/>publishes 31403 nostr-bbs-forum-client/src/pages/governance.rs:412"]
    FCMEM["forum-client /governance member view<br/>read-only, no 31403 publish path compiles<br/>nostr-bbs-forum-client/src/pages/governance.rs:35"]
    BOARD["forum-client kanban board<br/>subscribes to 31402 and 31403 only<br/>nostr-bbs-forum-client/src/pages/board.rs:392"]
    REG["PanelRegistry store<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:241"]
    BBS["bbs-client governance bucket<br/>nostr-bbs-bbs-client/src/relay.rs:583"]
    KAN["kanban approval decisions are parsed FROM 31403<br/>nostr-bbs-core/src/kanban.rs:739 nostr-bbs-core/src/kanban.rs:741"]

    RELAY --> FCADMIN & FCMEM & BOARD & BBS
    FCADMIN --> REG
    BOARD --> KAN

    N1["The member view is enforced by COMPOSITION, not by a runtime flag - it mounts components that do<br/>not compile a 31403 publish path nostr-bbs-forum-client/src/pages/governance.rs:33"]
    N2["The registry resolves a decision chain to the most recent AUTHORISED 31403; superseded events<br/>remain in the store rather than being deleted nostr-bbs-forum-client/src/stores/panel_registry.rs:241"]
    N3["EXTERNAL: exactly ONE live consumer today - ontology-concept elevation in VisionClaw, a case queue<br/>capped at five concurrent README.md:298. Treat universal human-in-the-loop surface as the design<br/>target, not a claim of many production consumers. See VC-24."]
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

## NF-06.12 Task properties set the escalation boundary — ADR-2011, live on main

```mermaid
flowchart TB
    PANEL["PanelDefinition 31400 declares the task-property triple<br/>tp-verifiability nostr-bbs-core/src/governance.rs:332<br/>tp-reversibility nostr-bbs-core/src/governance.rs:334<br/>tp-stakes nostr-bbs-core/src/governance.rs:336"]
    REQUEST["ActionRequest 31402 may restate the triple<br/>TaskProperties nostr-bbs-core/src/governance.rs:391<br/>from_tags nostr-bbs-core/src/governance.rs:416"]
    MERGE["merge is TIGHTENING ONLY - the result is never looser than the panel<br/>nostr-bbs-core/src/governance.rs:452, over the optional pair<br/>nostr-bbs-core/src/governance.rs:464"]
    TIER["effective_tier - total and pure over four inputs<br/>nostr-bbs-core/src/governance.rs:516"]
    DECL["the agent's own risk_tier is TELEMETRY from here on - it can raise<br/>the tier and never lower it below the properties' floor<br/>nostr-bbs-core/src/governance.rs:509-510"]
    DEF["an entirely unlabelled request folds to the relay's advertised<br/>ESCALATION_DEFAULT_TIER nostr-bbs-core/src/governance.rs:526,<br/>nostr-bbs-relay-worker/wrangler.toml:48"]
    STORE["stored on broker_cases.effective_tier at projection<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:320,<br/>read back nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2258"]

    PANEL --> MERGE
    REQUEST --> MERGE
    MERGE --> TIER
    DECL --> TIER
    DEF --> TIER
    TIER --> STORE

    N1["INVARIANT ADR-2011: the OPERATOR's task properties set the escalation boundary, not the agent's own<br/>self-tiering. Both are read, and the tighter wins nostr-bbs-core/src/governance.rs:523-534"]
    N2["INVARIANT tightening only: a request may move a property up the tightness ordering and never down<br/>nostr-bbs-core/src/governance.rs:229, asserted over all 729 pairs<br/>nostr-bbs-core/src/governance.rs:3062"]
    N3["A case projected before migration 0006 carries no effective_tier and imposes nothing<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:443-445"]
```

## NF-06.13 Who may decide a 31403, and what a consequential decision must carry

```mermaid
sequenceDiagram
    autonumber
    participant H as Human or script
    participant R as relay admission<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1001
    participant D1 as broker_roles and case_delegations
    participant AD as response_admission<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:234
    participant RA as check_rationale<br/>nostr-bbs-core/src/governance.rs:761

    H->>R: signed kind 31403
    R->>R: extract the case id from the d tag nip_handlers.rs:1002
    R->>D1: is_reviewer nip_handlers.rs:2276
    R->>D1: is_delegated_for_case nip_handlers.rs:2302
    R->>AD: is_admin, is_reviewer, delegated for THIS case
    AD-->>R: Admit, BlockedNotAuthorised or BlockedNotDelegated nip_handlers.rs:239 nip_handlers.rs:243 nip_handlers.rs:248
    R->>D1: case_effective_tier nip_handlers.rs:1025
    R->>RA: effective tier, action, reasoning nip_handlers.rs:1028
    RA-->>R: rationale_required when High or Critical and the words are missing nostr-bbs-core/src/governance.rs:766 nostr-bbs-core/src/governance.rs:773
    R-->>H: OK false with the refusal reason nip_handlers.rs:1033

    Note over R: INVARIANT FR2.2 is enforced at the RELAY, before save_event, because a 31403 is a signed event any client or script can publish directly - a rule that lives only in the forum UI is a suggestion nostr-bbs-core/src/governance.rs:754-759
    Note over RA: Nothing ever FILLS IN a rationale - absence is refused, never papered over nostr-bbs-core/src/governance.rs:759-760
    Note over RA: The minimum is 20 Unicode SCALARS, not bytes, so a rationale in a multi-byte script is not asked for more words nostr-bbs-core/src/governance.rs:712
    Note over RA: Only approve, reject, amend and delegate are gated - promote and precedent are bookkeeping on an already-decided case nostr-bbs-core/src/governance.rs:743
    Note over AD: One token for both refusals deliberately - telling an unauthorised caller HOW CLOSE they were tells them nothing useful nostr-bbs-core/src/governance.rs:728-731
```

## NF-06.14 Calibration sampling and seeded probes — what the relay actually enforces

```mermaid
flowchart TB
    SUP["is_member_suppressed_effective - a view filter over the EFFECTIVE tier<br/>nostr-bbs-core/src/governance.rs:542"]
    ONLY["only a case the member surface would otherwise HIDE is worth sampling<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:334"]
    CAL["is_calibration_sample - HMAC-SHA256 over the request id<br/>nostr-bbs-core/src/governance.rs:596"]
    KEY["CALIBRATION_SELECTION_KEY is a SECRET, never a plaintext var<br/>nostr-bbs-core/src/governance.rs:561, nostr-bbs-relay-worker/wrangler.toml:51-54"]
    RATE["rate declared on the panel, default 0.1<br/>nostr-bbs-core/src/governance.rs:338 nostr-bbs-core/src/governance.rs:369"]
    PROBE["a probe tag is honoured ONLY from the panel's registered probe agent<br/>nostr-bbs-core/src/governance.rs:696, dropped from anyone else<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:343-345"]
    BLIND["probe blinding: the tag is filtered at the tag-WRITING trigger<br/>nostr-bbs-relay-worker/src/lib.rs:607, mirrored into the live schema path<br/>nostr-bbs-relay-worker/src/lib.rs:872"]

    SUP --> ONLY --> CAL
    KEY --> CAL
    RATE --> CAL
    PROBE --> BLIND

    N1["INVARIANT the selection is KEYED. The request id is the 31402 d tag the agent chooses freely, so an<br/>unkeyed digest would let it grind ids until one is never sampled - opting itself out of oversight<br/>nostr-bbs-relay-worker/wrangler.toml:56-59"]
    N2["INVARIANT unset key still samples - silently disabling oversight is treated as the worse failure,<br/>and the relay logs a warning on every projection nostr-bbs-relay-worker/wrangler.toml:63-65"]
    N3["The blinding lives in the tag-writing trigger, not a watcher, because SQLite does not fire a trigger<br/>from inside another trigger without recursive_triggers, which is set nowhere here - the watcher was<br/>read as enforcement and was inert nostr-bbs-relay-worker/src/lib.rs:595-601"]
    N4["DIVERGENCE: what the relay enforces is NARROWER than DDD invariant 7. The tag remains on the raw<br/>signed 31402 served over REQ, because removing it would break the signature both clients verify<br/>strictly - blindness on the rendered surface is the CLIENT's responsibility<br/>nostr-bbs-core/src/governance.rs:357-364"]
    N5["The last purge statement is the one the original migration failed to mirror into the live path, so a<br/>deployed relay kept every probe row it had already indexed nostr-bbs-relay-worker/src/lib.rs:603-606"]
```

## NF-06.15 Two stale-identity classes the registry closed

```mermaid
flowchart TB
    REG["agent registration wrote the pubkey VERBATIM<br/>nostr-bbs-auth-worker/src/governance_api.rs:85"]
    LOOK["every other path canonicalises to lowercase, and is_registered_agent<br/>looks up with lower(pubkey)<br/>nostr-bbs-auth-worker/src/governance_api.rs:100-104"]
    BUG["so an agent registered with uppercase hex was written and then NEVER matched<br/>by the lookup that decides whether its governance events are admitted"]
    FIX["canonical_agent_pubkey - validate, then lowercase<br/>nostr-bbs-auth-worker/src/governance_api.rs:107"]
    ECHO["the response echoes what was STORED, not what was sent<br/>nostr-bbs-auth-worker/src/governance_api.rs:390-392"]
    NIP33["client side: NIP-33 replaceability per pubkey and d tag - a replayed OLDER<br/>31400 must not roll back the operator's declaration<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:241"]
    HELD["supersedes_held decides it, and accept_panel_state records the newest<br/>31401 or 31404 per address<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:454<br/>nostr-bbs-forum-client/src/stores/panel_registry.rs:198"]

    REG --> LOOK --> BUG --> FIX --> ECHO
    NIP33 --> HELD

    N1["INVARIANT: NIP-98 accepts case-insensitive hex and returns event.pubkey verbatim, and the event id is<br/>recomputed over that string, so the SAME key can legitimately present in two casings and both verify<br/>nostr-bbs-auth-worker/src/governance_api.rs:97-99"]
    N2["Canonicalisation is idempotent and still rejects everything it always rejected - length, non-hex and<br/>a blank name nostr-bbs-auth-worker/src/governance_api.rs:86-90"]
    N3["The registry keeps superseded events rather than deleting them - it resolves a decision chain to the<br/>most recent AUTHORISED 31403 nostr-bbs-forum-client/src/stores/panel_registry.rs:186"]
```
