---
id: ES-05
title: Human-approval governance loop across the estate
area: estate
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/docs/BASELINE-architecture.md
adrs: [visionclaw:ADR-2006, agentbox:ADR-2041, agentbox:ADR-2071]
sources:
  - ../project/src/services/acsp/mod.rs
  - ../project/src/services/acsp/events.rs
  - ../project/src/services/acsp/client.rs
  - ../project/src/services/decision_elevation.rs
  - ../project/src/handlers/broker_inbox_handler.rs
  - ../project/src/domain/broker/broker_case.rs
  - ../project/src/domain/broker/precedent_registry.rs
  - ../project/docs/adr/ADR-2006-acsp-human-approval.md
  - ../project/agentbox/management-api/lib/governance-correlation.js
  - ../project/agentbox/management-api/lib/governance-application-receipts.js
  - ../project/src/actors/decision_elevation_actor.rs
  - ../project/src/adapters/decision_elevation_store.rs
  - ../project/agentbox/management-api/lib/authority.js
  - ../project/agentbox/management-api/lib/governance-decision-waiter.js
  - ../project/agentbox/management-api/lib/elevation-publisher.js
  - ../project/agentbox/management-api/lib/kg-proposal-extractor.js
  - ../project/agentbox/management-api/lib/mandate.js
  - ../project/agentbox/management-api/lib/receipt-minter.js
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/agentbox/management-api/routes/kg-elevation.js
  - ../project/agentbox/mcp/nostr-bridge/relay-consumer.js
verified_commit: worktree-2026-09-07
---
## ES-05.1 Approval crosses independently durable systems
```mermaid
flowchart LR
    Request["Concrete operation"] --> Gate["Agentbox authority.js:137 buildAuthorityGate"]
    Gate --> Signed["Signed 31402 request commits to operation and digest"]
    Signed --> Forum["Forum relay stores event and guarded D1 projection"]
    Forum --> Human["Human signed 31403 references exact request"]
    Human --> AB["Agentbox verified allowlisted consumer<br/>request match, then local received/outcome ledger"]
    Human --> VC["VisionClaw client.rs:265 response_matches_request<br/>own signed request journal and one dispatch claim"]
    AB --> Mutation["Upstream committed acknowledgement is separate evidence"]
    VC --> PR["Decision-elevation approved-to-applying claim<br/>PR URL, merge and activation are separate stages"]
    Limits["No distributed transaction<br/>uncertain outcomes require reconciliation<br/>source tests are not live deployment receipts"] -.-> Mutation
    Limits -.-> PR
```

## ES-05.2 Wire fields and exact response binding
```mermaid
classDiagram
    class ActionRequest {
        +JSON fields
        +OptionString reasoning
        +OptionString context_url
    }
    class ActionResponse {
        +String action
        +String reasoning
    }
    class CaseDecision {
        +String case_id
        +String action
        +String reasoning
        +String responder_pubkey
        +String event_id
        +u64 created_at
    }
    ActionRequest --> ActionResponse : signed request e reference
    ActionResponse --> CaseDecision : validated consumer dispatch
    note for ActionRequest "events.rs:130. Agentbox embeds canonical operation and digest; decision-elevation embeds exact corpus draft/path."
    note for ActionResponse "events.rs:140. SDK signature verification plus own journalled request and d agreement at client.rs:265. Relay still owns admin admission."
    note for CaseDecision "A durable dispatch claim precedes delivery. It does not certify actor execution or external application."
```

## ES-05.3 Authority gate binds the signed operation
```mermaid
sequenceDiagram
    participant Caller
    participant Gate as authority.js:136 buildAuthorityGate
    participant Forum as Verified allowlisted consumer
    participant Owner as broker-bridge.js mutation owner
    participant Journal as governance-application-receipts.js:11 ApplicationReceiptStore
    Caller->>Gate: action class and concrete operation
    alt recoverable action
        Gate-->>Caller: allow under existing recoverable policy
    else escalation required
        Gate->>Gate: canonicalise operation and SHA256 digest
        Gate->>Forum: sign 31402 fields containing operation and digest
        Gate->>Gate: reject producer if signed content changed
        Forum-->>Gate: verified 31403 with exact request e reference
        Gate->>Gate: require optional case and panel agreement
        alt approval bound to this request
            Gate-->>Owner: released plus request ID, response ID, operation digest
            Owner->>Journal: durable immutable consumer-received claim
            alt fresh claim
                Owner->>Owner: send exact approved payload to upstream
                Owner->>Journal: applied only on writeback_committed acknowledgement
            else prior claim or unavailable storage
                Owner-->>Caller: refuse replay or require reconciliation
            end
        else mismatch, refusal or timeout
            Gate-->>Caller: deny
        end
    end
    Note over Owner,Journal: Timeout or crash is unknown, never proof of application. Local receipt is unsigned and does not prove deployment.
```

## ES-05.4 Decision waiter requires the exact request
```mermaid
sequenceDiagram
    participant Gate as authority.js
    participant Waiter as governance-decision-waiter.js:87 awaitDecision
    participant Relay as Existing relay consumer
    Gate->>Waiter: awaitDecision signed request, timeout
    Waiter->>Waiter: register by request event ID only
    par exact response arrives
        Relay->>Waiter: notify 31403
        Waiter->>Waiter: governance-correlation.js checks unambiguous e reference
        Waiter->>Waiter: optional case and panel must agree with request
        Waiter-->>Gate: resolve matching waiter and cancel timer
        Gate->>Gate: verify signature and outcome before release
    and timeout fires
        Waiter->>Waiter: remove pending entry
        Waiter-->>Gate: null, gate denies
    end
    Note over Gate,Waiter: Case-only or panel-only responses never release a wait. No extra relay subscription.
```

## ES-05.5 Broker inbox reads the current case state
```mermaid
sequenceDiagram
    participant Reviewer
    participant Bridge as broker-bridge.js:254 inbox route
    participant VC as broker_inbox_handler.rs:126
    Reviewer->>Bridge: authenticated inbox request with status filter
    Bridge->>VC: fetch current broker inbox
    alt upstream available
        VC-->>Bridge: cases and total
        Bridge-->>Reviewer: filtered/enriched inbox
    else upstream unavailable
        Bridge-->>Reviewer: explicit error
    end
    Note over Reviewer,VC: An inbox read is not an approval or application receipt.
```

## ES-05.6 Mutation-owner acknowledgement and replay refusal
```mermaid
sequenceDiagram
    participant Bridge as broker-bridge.js:373 decide route
    participant Gate as authority.js:137
    participant Ledger as governance-application-receipts.js:11
    participant VC as VisionClaw mutation owner
    Bridge->>Gate: concrete case, outcome, actor and reasoning operation
    Gate-->>Bridge: verified approval and operation digest, or refusal
    alt approval released
        Bridge->>Ledger: immutable consumer-received claim before external call
        alt fresh claim
            Bridge->>VC: exact approved operation payload
            VC-->>Bridge: committed acknowledgement or failure
            Bridge->>Ledger: applied only when writeback_committed is true
        else existing claim or storage unavailable
            Bridge-->>Bridge: refuse repeat execution, reconcile uncertain state
        end
    else refusal or timeout
        Bridge-->>Bridge: no governed writeback
    end
    Note over Gate,VC: Recoverable actions and an explicitly disabled authority table retain their existing policy. Live responder policy and remote state need their own evidence.
```

## ES-05.7 Governed ontology elevation — personal to shared, federated over Nostr
```mermaid
sequenceDiagram
    autonumber
    participant KE as kg-elevation<br/>routes/kg-elevation.js
    participant EX as kg-proposal-extractor<br/>lib/kg-proposal-extractor.js:330
    participant SC as scoreCandidate<br/>lib/kg-proposal-extractor.js:186
    participant BD as buildProposalDescriptor<br/>lib/kg-proposal-extractor.js:224
    participant EP as elevation-publisher
    participant ACS as agent-control-surface<br/>buildActionRequest / publishPanelEvent
    participant NB as NostrBridge (already connected)

    KE->>EX: extractProposals(entries, opts)
    loop each candidate entry
        EX->>EX: normaliseEntry — lib/kg-proposal-extractor.js:111
        EX->>SC: scoreCandidate(norm)
        SC-->>EX: score
        EX->>BD: buildProposalDescriptor(norm, score, opts)
        BD->>BD: buildProposeRequest — lib/kg-proposal-extractor.js:257
        Note over BD: propose_request is a GOVERNED {path, method, body} for<br/>/api/ontology-agent/propose — kg-proposal-extractor.js:214.<br/>governed_path is recorded at :286.
        BD-->>EX: descriptor plus an agent_action LINK beam
    end
    EX-->>KE: proposals
    KE->>EP: publish each proposal
    EP->>ACS: buildActionRequest — SIGNED ACSP kind 31402
    Note over EP,ACS: URN DISCIPLINE — the panel d-tag REUSES the proposal's own<br/>canonical urn-agentbox-thing-PUBKEY-proposal-SHA256_12,<br/>already minted through lib/uris.js. NIP-33 replaceability<br/>keys re-scans of the same concept to the SAME panel. No<br/>ad-hoc identifiers are invented (elevation-publisher.js:33-36).
    ACS->>NB: publishPanelEvent
    alt federation surface available
        NB-->>EP: published
        Note over NB: The relay's agent_registry gate plus broker_cases<br/>projection surface the elevation in the governance inbox.
    else nostr_bridge gate off, NOSTR_RELAYS empty, no signing<br/>stack, nostr-tools absent, or the key will not decrypt
        NB-->>EP: {published: false, reason}
        Note over EP,NB: STANDALONE-OR-FEDERATED CONTRACT ADR-005 —<br/>a no-op logged at debug. It NEVER throws into the request<br/>path: the existing beam plus propose response is returned<br/>unchanged. Federation is ADDITIVE, never load-bearing.
    end
    Note over KE,NB: INVARIANT — this is the SANCTIONED governed path.<br/>The ungoverned /api/ontology/load backdoor is never used<br/>(elevation-publisher.js:16-17).
```

## ES-05.8 Approval, dispatch and application remain distinct
```mermaid
stateDiagram-v2
    [*] --> Requested
    Requested --> Approved: exact signed request response
    Requested --> Refused: mismatch, denial or timeout
    Approved --> Received: durable local claim
    Received --> Applied: committed mutation acknowledgement
    Received --> NotApplied: explicit upstream refusal
    Received --> Unknown: timeout, crash or incomplete outcome
    Unknown --> Reconciliation: never blindly replay
    Applied --> [*]
    NotApplied --> [*]
    Refused --> [*]
    note right of Received
        Agentbox local ledger is unsigned.
        VisionClaw dispatch journal is not an applied receipt.
        PR creation and merge/activation are separate observations.
    end note
```

## ES-05.9 Mandate and receipt — the durable authority artefacts
```mermaid
sequenceDiagram
    autonumber
    participant I as issuer
    participant M as createMandate<br/>lib/mandate.js:99
    participant S as signMandate<br/>lib/mandate.js:163
    participant T as mandateToAclTurtle<br/>lib/mandate.js:137
    participant CK as isMandateActive<br/>lib/mandate.js:191
    participant RM as receipt-minter<br/>lib/receipt-minter.js

    I->>M: createMandate{issuer, agent, container, modes, issuedAt, expiresAt}
    M->>M: normalisePubkey :53, normaliseModes :60, normaliseContainer :80
    M-->>I: record
    I->>S: signMandate(record, signer)
    S-->>I: signedEvent
    opt reconstruct from the wire
        I->>M: recordFromSignedMandate(signedEvent) — lib/mandate.js:209
    end
    I->>T: mandateToAclTurtle(record)
    T-->>I: WAC turtle for the pod ACL
    loop on each authority check
        I->>CK: isMandateActive(record, nowSec)
        alt within window
            CK-->>I: true
        else expired or not yet valid
            CK-->>I: false — authority refused
        end
    end
    I->>RM: mintSpendReceipt{pubkey, origin, scheme, amountSats, outcome, idempotencyKey}
    RM-->>I: receipt — lib/receipt-minter.js:45
    I->>RM: mintSpendActivity(same shape) — lib/receipt-minter.js:78
    RM->>RM: crossActivityOutbound(activityUrn) — lib/receipt-minter.js:105
    Note over RM: crossActivityOutbound is the federation hop for the<br/>activity URN — see ES-03 for the closed kind map that<br/>governs what may cross.
    Note over I,RM: idempotencyKey makes receipt minting replay-safe, so a<br/>retried decision cannot double-spend.
```

## ES-05.10 Remaining governance boundaries after execution closeout
```mermaid
flowchart TB
    Local["Implemented local guards and receipts"] --> AB["Agentbox operation/request binding<br/>durable broker received/outcome records"]
    Local --> VC["VisionClaw signed-request journal<br/>conditional PR claim and uncertain-outcome reconciliation"]
    Local --> Forum["Forum ordinary-response guarded D1 projection"]
    Local --> Tasks["Task-spawn action plane journal<br/>see ES-05.11"]
    AB --> Runtime["Still requires live responder provisioning and deployed consumer evidence"]
    VC --> Runtime
    Forum --> Runtime
    Tasks --> Scope["Route-specific coverage does not prove mediation of every shell or nightly operation"]
    Scope --> UID["Same-UID agents are not isolated by application-level policy"]
    Runtime --> External["External applied, merged and activated states need their own witnessed receipts"]
    Note["ADR decision status is separate from source availability. No blanket governance closure follows from these changes."] -.-> Local
```

## ES-05.11 Journal coverage is route-specific

```mermaid
flowchart LR
    TASK["POST /v1/tasks"] --> PLANE["action-plane.dispatchTaskSpawn"]
    PLANE --> READY{"Events adapter and pipeline ready?"}
    READY -->|no| REFUSE["503: no unjournalled spawn"]
    READY -->|yes| LOCAL["local side-effect classification"]
    LOCAL --> SPAWN["Journalled task spawn; no extra approval receipt"]
    NIGHT["Nightly SSH, model calls, push and publication"] -.-> GAP["ADR-2071 proposed: no universal journal/approval claim"]
    PEER["Same-UID process"] -.-> LIMIT["Application gate does not establish OS isolation"]
```

Grounded in Agentbox `management-api/lib/action-plane.js` and `routes/tasks.js`; broader governance routes retain their individual contracts. A working task-spawn journal does not establish complete mediation of shell commands or nightly egress. See [audit](../../estate-review/2026-09-07-agentbox-audit.md).
