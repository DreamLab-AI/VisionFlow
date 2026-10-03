---
id: AB-14
title: Governance — journal, action pipeline, approvals
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2022, ADR-2027, ADR-2041, ADR-2071, ADR-2085, ADR-2087, ADR-2108, ADR-2122]
sources:
  - ../project/agentbox/management-api/lib/action-plane.js
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/management-api/routes/system.js
  - ../project/agentbox/docs/adr/ADR-2041-wire-execution-journal-and-action-pipeline.md
  - ../project/agentbox/management-api/lib/execution-journal.js
  - ../project/agentbox/management-api/lib/execution-projections.js
  - ../project/agentbox/management-api/lib/execution-coverage.js
  - ../project/agentbox/management-api/lib/agent-action-pipeline.js
  - ../project/agentbox/management-api/routes/approvals.js
  - ../project/agentbox/management-api/lib/governance-correlation.js
  - ../project/agentbox/management-api/lib/governance-decision-waiter.js
  - ../project/agentbox/management-api/lib/receipt-minter.js
  - ../project/agentbox/management-api/lib/audit-chain.js
  - ../project/agentbox/management-api/lib/elevation-publisher.js
  - ../project/agentbox/management-api/lib/project-tracker.js
  - ../project/agentbox/management-api/lib/project-primer.js
  - ../project/agentbox/management-api/routes/projects.js
  - ../project/agentbox/management-api/routes/tasks.js
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/docs/adr/ADR-2022-governed-ontology-writes.md
  - ../project/agentbox/docs/adr/ADR-2027-secret-custody-rotation-break-glass.md
  - ../project/agentbox/docs/archive/adr/ADR-057-replayable-agent-execution-journal.md
  - ../project/agentbox/docs/archive/adr/ADR-059-monotonic-agent-action-policy-pipeline.md
  - ../project/agentbox/mcp/mcp.json
  - ../project/agentbox/mcp/nostr-bridge/relay-consumer.js
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/tests/contract/agent-action-pipeline.contract.spec.js
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/hooks/project-tracking-publish.cjs
  - ../project/agentbox/management-api/lib/authority.js
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/skills/lint-skills.sh
  - ../project/agentbox/management-api/lib/task-properties.js
  - ../project/agentbox/management-api/lib/governance-application-receipts.js
  - ../project/agentbox/management-api/lib/authority-journal.js
  - ../project/agentbox/management-api/lib/governance-receipt-publisher.js
  - ../project/agentbox/management-api/lib/governance-manual-continue.js
  - ../project/agentbox/mcp/servers/governance-bridge.js
  - ../project/agentbox/docs/adr/ADR-2087-task-properties-receipts-and-manual-continuation.md
  - ../project/agentbox/docs/adr/ADR-2108-ontology-bridge-retired-vault-baked-by-nix.md
  - ../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/agentbox/management-api/routes/llm-marketplace.js
  - ../project/agentbox/management-api/routes/agent-events.js
  - ../project/agentbox/management-api/routes/exec-record.js
  - ../project/agentbox/scripts/activation/adr-2087-check.sh
  - ../project/agentbox/scripts/activation/adr-2071-api-down-night.sh
  - ../project/agentbox/scripts/experiments/exp-b8-label-log.cjs
  - ../project/agentbox/services/nostr-pod-bridge/src/main.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/lib.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/server.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/receipts.rs
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/flake.nix
  - ../project/agentbox/management-api/lib/pod-signer.js
  - ../project/agentbox/management-api/lib/junkiejarvis-agent.js
  - ../project/agentbox/scripts/dream-forum-suggestions.mjs
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
verified_commit: d03defbeaca6c52d6bf3f7338d3f465a109fcdbf
---

## AB-14.1 Governance plane — surfaces that reach the decision point vs surfaces that miss it

```mermaid
flowchart TB
    subgraph SURF["Agent-initiated side-effect surfaces GOVERNANCE-capabilities.md:46-70"]
        DTC["Direct tool call<br/>MCP fleet mcp/mcp.json"]
        CMS["Code-mode sub-call<br/>codeact / code-interpreter agentbox.toml:588,604"]
        ACISHELL["ACI shell<br/>test allowlist agentbox.toml:788<br/>raw Bash still reachable outside it"]
        CONSULT["Consultant / subagent action<br/>tree-search-coder agentbox.toml:824"]
        DREAM["Background job dream-engine<br/>01:00-05:00 UTC unattended"]
        BEADS["Background job beads work-DAG<br/>spawn_child mcp/mcp.json:219"]
        ALTHARNESS["Alternate harness path<br/>non-Claude harness"]
        TASKAPI["POST /v1/tasks<br/>routes/tasks.js:16 spawnTask"]
    end

    PIPE{{"AgentActionPipeline.dispatch<br/>agent-action-pipeline.js:110<br/>9-stage legacy-ADR-059 pipeline"}}
    JOURNAL[["ExecutionJournal.append<br/>execution-journal.js:133"]]
    ACTPLANE(["action-plane.js getActionPlane<br/>lazy singleton, ADR-2041, see AB-14.14"])
    COSTGATE["costGate middleware<br/>middleware/cost-gate.js, see AB-15.x"]
    ACSPGATE["ACSP authority gate<br/>authority.js:226 buildAuthorityGate.guard, see AB-11.10"]
    AXIOMGUARD["direct_axiom_load=false guard agentbox.toml:841<br/>ADR-2108 supersedes ADR-2022, see AB-25"]
    EXEC(["side effect executes"])

    DTC -.->|"designed target, never instantiated for this surface"| PIPE
    CMS -->|"own guard only, no interceptor"| EXEC
    ACISHELL -->|"allowlist only, no interceptor"| EXEC
    CONSULT -->|"spend cap only, see AB-15.x"| EXEC
    DREAM -->|"own evidence/merge gate, see AB-23"| EXEC
    DREAM -.->|"records only, never polices: POST /v1/exec/record exec-record.js:63, ADR-2071 Phase 1"| JOURNAL
    BEADS -->|"own work-DAG guard, see AB-26"| EXEC
    ALTHARNESS -->|"no interceptor at all"| EXEC
    TASKAPI --> COSTGATE --> ACTPLANE
    ACTPLANE -->|"ADR-2041: dispatchTaskSpawn, ready:true"| PIPE
    ACTPLANE -.->|"events adapter off/unresolvable: ready:false, 503"| EXEC
    PIPE --> JOURNAL
    PIPE -->|"executor seam, capability token minted"| EXEC
    ACSPGATE --> EXEC
    AXIOMGUARD --> EXEC

    DRIFT["DOC-DRIFT GOVERNANCE-capabilities.md:81-82 claims a repo-wide search of src/, services/, mcp/ for SessionEvent / execution-journal code returns nothing<br/>execution-journal.js:85 class ExecutionJournal and agent-action-pipeline.js:58 class AgentActionPipeline fully implement legacy-ADR-057/059 D1-D5<br/>under management-api/lib/, a path outside the doc's stated search scope"]
    DIVERGE1["DIVERGENCE (NARROWED by ADR-2041) TOP OPEN RISK GOVERNANCE-capabilities.md:276-281 named 'no single policy decision point' with zero production instantiations<br/>action-plane.js now builds a real ExecutionJournal + AgentActionPipeline singleton and POST /v1/tasks calls dispatchTaskSpawn — one surface is now wired, journalled and capability-tokened<br/>the dream engine now RECORDS each side effect into the same journal through POST /v1/exec/record (ADR-2071, activation live, decision still proposed) but nothing there approves or denies<br/>every OTHER surface above (direct tool call, code-mode, ACI shell, consultant, dream, beads, alt harness) still reaches EXEC with no interceptor — the gap is narrower, not closed, see AB-14.14"]
    DRIFT2["DOC-DRIFT the capability-surface list cites agentbox.toml line numbers that have moved and keep moving —<br/>GOVERNANCE-capabilities.md:53 says aci_shell is at agentbox.toml line 582 (it is agentbox.toml:782),<br/>GOVERNANCE-capabilities.md:56-57 says code_interpreter line 539 and codeact line 554 (they are agentbox.toml:588 and agentbox.toml:604),<br/>GOVERNANCE-capabilities.md:60 says the test allowlist is line 585 (it is agentbox.toml:788), GOVERNANCE-capabilities.md:64 says tree_search_coder line 624 (it is agentbox.toml:824),<br/>GOVERNANCE-capabilities.md:55 says spawn_child is at mcp.json line 197 (it is mcp/mcp.json:219)"]
    DIVERGE7["DIVERGENCE GOVERNANCE-capabilities.md:307-308 skill lint is advisory, not a runtime capability gate<br/>lint-skills.sh gates estate hygiene only; an enabled skill with clean frontmatter is trusted at runtime with no further check"]
```

## AB-14.2 Action lifecycle — intent through policy, approval, execution, receipt, journal

```mermaid
stateDiagram-v2
    [*] --> Normalise
    Normalise --> Enrich : stage 2, agent-action-pipeline.js:238, must not change operation/target
    Enrich --> Classify : stage 3, assigns side_effect_class privacy_class estimated_cost
    Classify --> FastPath : side_effect_class in read,local FAST_PATH :38
    Classify --> ApprovalGate : side_effect_class in mutate,egress,secret,spend APPROVAL_REQUIRED :34
    FastPath --> Guard : approval-free, still journalled
    ApprovalGate --> Approve : _obtainApproval :244
    Approve --> Guard : receipt valid, identity+cost ceiling frozen :143-144
    Approve --> Denied : approval_missing, approval_expired, approval_replayed, approval_mismatch or cost_exceeds_ceiling :250-273
    Guard --> Execute : every guard abstains :148-159
    Guard --> Denied : any guard denies, errors or returns a non-monotonic verdict :153-158
    Guard --> Denied : mutation_after_approval, identity changed post-approval :162-164
    Execute --> PostProcess : _protectedExecute returns within deadline :281-296
    Execute --> Denied : no_capability_token or timeout :283,289
    PostProcess --> Finalise : redaction ok, or degraded output for public/internal only :173-183
    PostProcess --> Denied : redaction_failed on mutate,egress,secret,spend or sensitive privacy_class :178-180
    Finalise --> Record : stage 8, definition-owned invariants :186
    Record --> Allowed : stage 9, journal.append tool.completed decision=allow :188-190
    Denied --> RecordDeny : _record still called with decision=deny :193
    RecordDeny --> [*]
    Allowed --> [*]

    note right of Guard
        INVARIANT legacy-ADR-059 D3 mutate, egress, secret and spend
        NEVER fail open. NEVER_FAIL_OPEN set agent-action-pipeline.js:36
        matches APPROVAL_REQUIRED exactly.
    end note
    note right of Record
        INVARIANT legacy-ADR-059 D2 after Approve begins, the tuple
        capability+operation+target+canonical_args_hash+cost_ceiling
        is frozen; re-checked at :162 before Execute. No later stage
        can rewrite an earlier approval.
    end note
    note right of RecordDeny
        DOC-DRIFT this state machine is fully implemented and
        contract-tested (tests/contract/agent-action-pipeline.contract.spec.js)
        but is never reached by real traffic — see AB-14.1 DIVERGENCE.
    end note
```

## AB-14.3 AgentActionPipeline.dispatch — intentSpec construction and policy evaluation

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as caller<br/>agent-action-pipeline.js:110
    participant AAP as AgentActionPipeline<br/>agent-action-pipeline.js:58
    participant CLS as classifier callback<br/>opts.classifier :78
    participant APP as approver callback<br/>opts.approver :244
    participant GRD as guards array<br/>opts.guards :76

    CALLER->>AAP: dispatch(raw, opts) agent-action-pipeline.js:110
    AAP->>AAP: _normalise(raw, opts.parentToken) :201
    Note over AAP: session_urn and capability+operation required, else ActionDenied bad_action :203-204
    opt parentToken supplied
        AAP->>AAP: _verifyToken(parentToken) :227, defined :347-358, HMAC-SHA256 _sign :375-377
        Note over AAP: D4 child inherits parent.authority into _delegatedAuthority :230
    end
    AAP->>AAP: action_id = hash(coreIdentity + dispatchSeq) :233
    AAP->>AAP: _enrich(action) :238, freezes operation/target
    alt enrich mutated operation or target
        AAP-->>CALLER: ActionDenied enrich_mutated_identity :120
    end
    AAP->>CLS: classifier(action) :124
    CLS-->>AAP: side_effect_class, privacy_class, estimated_cost
    Note over AAP: unknown side_effect_class defaults to mutate, the strictest class :125
    alt _delegatedAuthority set and class exceeds it
        AAP-->>CALLER: ActionDenied authority_exceeded :131-134
    end
    AAP->>AAP: identityHash = _identityHash(action) :138, capability+operation+target+canonical_args_hash
    alt side_effect_class in APPROVAL_REQUIRED and not FAST_PATH
        AAP->>APP: _obtainApproval(action, opts.approval) :244
        APP-->>AAP: approval receipt or opts.approval supplied
        AAP->>AAP: _validateReceipt(receipt, action, identityHash) :250
    end
    loop for each guard in this._guards agent-action-pipeline.js:148-159
        AAP->>GRD: guard(action)
        GRD-->>AAP: deny or abstain
        alt verdict is deny
            AAP-->>CALLER: ActionDenied guard_denied :155
        else verdict not abstain
            AAP-->>CALLER: ActionDenied bad_verdict, non-monotonic :157
        end
    end
    Note over AAP: D2 re-check — _approvedIdentityHash must still equal _identityHash(action) :162-164
```

## AB-14.4 F2 operator-gated approval — pending, decide, 409 already-decided

```mermaid
sequenceDiagram
    autonumber
    participant AG as authority gate<br/>see AB-11.10
    participant AC as authority consumer<br/>fastify.authorityConsumer, approvals.js:47
    participant OP as operator<br/>NIP-98 authed caller
    participant RT as POST /v1/approvals/:id/decide<br/>approvals.js:92
    participant AUTHZ as authz.isApprover<br/>approvals.js:143

    AG->>AC: publish kind-31402 ActionRequest, awaiting decision
    Note over AC: request now pending — c.getPending(id) can find it
    OP->>RT: POST /v1/approvals/:id/decide {outcome or decision, reasoning}
    alt request.auth.mode is not nip98
        RT-->>OP: 401 nip98_required approvals.js:132-137
    end
    RT->>AUTHZ: isApprover(request.auth.pubkey, manifest) :143
    alt pubkey not on approval allowlist
        RT-->>OP: 403 forbidden_not_approver approvals.js:148-152
    end
    alt c.isDecided(id) true
        RT->>AC: getDecision(id) :168
        RT-->>OP: 409 already_decided {request_event_id, outcome, response_event_id} :169-175
    end
    alt c.getPending(id) is falsy
        RT-->>OP: 404 unknown_request approvals.js:177-182
    end
    RT->>RT: normalise decision=deny to outcome=reject :186-187
    alt outcome not in approve,reject,defer
        RT-->>OP: 400 invalid_outcome approvals.js:188-193
    end
    RT->>AC: signAndPublishDecision({requestId, outcome, reasoning}) :198
    AC-->>RT: signed kind-31403 event, race-safe errors ALREADY_DECIDED/NOT_PENDING/DECISION_IN_FLIGHT
    alt err.code is ALREADY_DECIDED
        RT-->>OP: 409 already_decided approvals.js:203-204
    else err.code is NOT_PENDING
        RT-->>OP: 404 unknown_request approvals.js:206-207
    else err.code is DECISION_IN_FLIGHT
        RT-->>OP: 409 decision_in_flight approvals.js:209-212, concurrent decider already claimed it
    else sign/publish throws for another reason
        RT-->>OP: 502 sign_publish_failed approvals.js:214-215
    end
    RT-->>OP: 200 {success, request_event_id, response_event_id, outcome, decided_by} :223-229
    Note over AG,AC: governance-decision-waiter.js notify() then resolves any gate awaiting this request, see AB-14.6
```

## AB-14.5 routes/approvals.js — every route and status code

```mermaid
sequenceDiagram
    autonumber
    participant C as client
    participant FA as fastify<br/>approvals.js:33 approvalsRoutes
    participant CON as consumer()<br/>approvals.js:46-48

    C->>FA: GET /v1/approvals approvals.js:51
    FA->>CON: options.authorityConsumer or fastify.authorityConsumer
    alt no consumer or listPending not a function
        FA-->>C: 200 {approvals:[], count:0, wired:false, note} :85
    else consumer wired
        FA-->>C: 200 {approvals, count, wired:true} :87-88
    end

    C->>FA: POST /v1/approvals/:id/decide approvals.js:92
    Note over FA: body accepts outcome approve/reject/defer OR decision approve/deny/reject/defer :101-107
    alt not NIP-98 authed
        FA-->>C: 401 nip98_required :132-137
    else pubkey not on approval allowlist
        FA-->>C: 403 forbidden_not_approver :148-152
    else no consumer or signAndPublishDecision not a function
        FA-->>C: 503 authority_consumer_unwired :155-160
    else c.isDecided(id)
        FA-->>C: 409 already_decided :167-176
    else not c.getPending(id)
        FA-->>C: 404 unknown_request :177-182
    else outcome missing or invalid
        FA-->>C: 400 invalid_outcome :188-193
    else signAndPublishDecision throws ALREADY_DECIDED
        FA-->>C: 409 already_decided :203-204
    else signAndPublishDecision throws NOT_PENDING
        FA-->>C: 404 unknown_request :206-207
    else signAndPublishDecision throws DECISION_IN_FLIGHT
        FA-->>C: 409 decision_in_flight :209-212
    else signAndPublishDecision throws other
        FA-->>C: 502 sign_publish_failed :214-215
    else signed ok
        FA-->>C: 200 {success:true, request_event_id, response_event_id, outcome, decided_by} :223-229
    end
    Note over FA: legacy-ADR-043 D4.7 hard rule — the route never writes an unsigned approval, approvals.js:16-22
```

## AB-14.6 Decision waiter — exact request and bounded timeout

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

## AB-14.7 ExecutionJournal.append — what is written, where, and the ordering guarantee

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as caller<br/>e.g. AgentActionPipeline._record agent-action-pipeline.js:299
    participant EJ as ExecutionJournal<br/>execution-journal.js:85
    participant EVT as events adapter<br/>opts.eventsAdapter.dispatch, ADR-005 events slot

    CALLER->>EJ: append(event) execution-journal.js:133
    alt session_urn missing
        EJ-->>CALLER: JournalError bad_event :135
    else event.type not in VOCABULARY
        EJ-->>CALLER: JournalError bad_type :136-138, VOCABULARY has 14 entries :40-56
    else privacy_class invalid
        EJ-->>CALLER: JournalError bad_privacy_class :140-142, must be public/internal/sensitive/secret
    end
    opt event.event_id supplied and already seen for this session_urn
        EJ-->>CALLER: {envelope: prior, duplicate:true} :146-150, no re-append
    end
    EJ->>EJ: seq = this._nextSeq.get(session_urn) or 0 :152, _peekSeq
    EJ->>EJ: mint event_id via uris.js if absent :153, _mintEventId :331
    EJ->>EJ: build envelope {schema, event_id, session_urn, seq, occurred_at, harness, agent_did, turn, type, payload, privacy_class} :155-167
    EJ->>EVT: dispatch({kind: exec.<type>, session_id, execution_id, payload: envelope}) :174-179
    Note over EJ: session_urn + seq is the journal key, contiguous from 0, assigned HERE, never by the caller :16-18
    EJ->>EJ: commit _nextSeq.set(session_urn, seq+1) and _seenIds AFTER successful dispatch only :182-185
    EJ-->>CALLER: {envelope, duplicate:false} :187
    Note over EJ,EVT: INVARIANT the journal rides the ADR-005 events adapter — no sixth adapter slot, no new database :9,20-24
    Note over EJ: hydrate(envelopes) rebuilds _nextSeq and _seenIds from persisted envelopes after a restart, idempotent replay :289-300
    Note over EJ: RESOLVED (ADR-2041, partial) — action-plane.js now builds a live ExecutionJournal and this append path IS reached by POST /v1/tasks (task-spawn, side_effect_class 'local') when the events adapter is live. Every other surface in AB-14.1 still bypasses it — see AB-14.14
```

## AB-14.8 execution-projections and execution-coverage over the journal

```mermaid
flowchart LR
    ENV["AgentExecutionEvent envelope<br/>{session_urn, seq, type, payload}"]
    APPLY["Projection.apply(envelope)<br/>execution-projections.js:45"]
    WM{"envelope.seq <= watermark for session_urn ?<br/>:47-48"}
    NOOP["idempotent no-op<br/>return false"]
    REDUCE["_reduce(envelope)<br/>subclass-defined"]
    TRANS["TranscriptProjection<br/>execution-projections.js:77<br/>input.claimed, assistant.completed, tool.completed only"]
    COST["CostLedgerProjection<br/>execution-projections.js:115<br/>sums usage from assistant.completed only"]
    REBUILD["Projection.rebuild(envelopes)<br/>:55, sorts by session then seq, replays apply()"]
    COV["buildExecutionCoverage(live)<br/>execution-coverage.js:42"]
    SYS["/v1/system execution block"]

    ENV --> APPLY --> WM
    WM -->|"yes, already folded in"| NOOP
    WM -->|"no"| REDUCE
    REDUCE --> TRANS
    REDUCE --> COST
    REBUILD --> APPLY

    COV --> SYS
    COV -.->|"journal.status = live if live.journal passed, else declared"| SYS
    COV -.->|"action_pipeline.status = live if live.pipeline passed, else declared execution-coverage.js:65"| SYS

    DRIFTNOTE["DOC-DRIFT (PARTIALLY RESOLVED, ADR-2041) execution-coverage.js gained buildLiveExecutionCoverage(), which pulls a REAL singleton from action-plane.js getCoverageSnapshot() when POST /v1/tasks has built one<br/>but routes/system.js:26,61 still calls the OLD buildExecutionCoverage(live) with live={} — routes/system.js:48-49 reads options.execution.snapshot and server.js never supplies one — so GET /v1/system still reports status: declared always, even though a live instance now genuinely exists"]
    COV --- DRIFTNOTE
```

## AB-14.9 receipt-minter then audit-chain — the hash chain link and break detection

```mermaid
sequenceDiagram
    autonumber
    participant SPEND as spend attempt<br/>paid, denied, failed or pending-approval
    participant RM as receipt-minter<br/>management-api/lib/receipt-minter.js:45
    participant URIS as uris.mint<br/>management-api/lib/uris.js:181
    participant BC20 as bc20-provenance-bridge<br/>management-api/lib/receipt-minter.js:105 crossActivityOutbound
    participant WRITER as events writer<br/>daily-rotated JSONL, ADR-039
    participant AC as audit-chain<br/>management-api/lib/audit-chain.js:89

    SPEND->>RM: mintSpendReceipt({pubkey, origin, scheme, amountSats, outcome, idempotencyKey})<br/>management-api/lib/receipt-minter.js:45
    RM->>URIS: mint({kind:'receipt', pubkey, payload}) management-api/lib/receipt-minter.js:47-57
    URIS-->>RM: urn:agentbox:receipt:pubkey:sha256-12-hash, content-addressed
    alt mint throws
        RM-->>SPEND: urn:agentbox:receipt:error:mint-failed management-api/lib/receipt-minter.js:58-60, never throws
    end
    SPEND->>RM: mintSpendActivity({...}) management-api/lib/receipt-minter.js:78
    RM->>URIS: mint({kind:'activity', payload:{type:'pay-'+scheme, ...}})<br/>management-api/lib/receipt-minter.js:81-91
    RM->>BC20: crossActivityOutbound(urn) management-api/lib/receipt-minter.js:95,105
    Note over RM,BC20: fail-open — a crossOutbound failure is logged to stderr and never blocks the caller<br/>management-api/lib/receipt-minter.js:103-112

    Note over WRITER,AC: separately, each JSONL record written gets hash = SHA256(prev_hash || canonical_json(record minus<br/>prev_hash,hash)) management-api/lib/audit-chain.js:7,68-72
    WRITER->>AC: verifyLines(lines, {expectedPrev}) audit-chain.js:89
    loop each JSONL line management-api/lib/audit-chain.js:102-132
        alt record has neither hash nor prev_hash and chain not yet started
            AC->>AC: legacyPrefix += 1, tolerate as pre-ADR-039 prefix management-api/lib/audit-chain.js:113-118
        else record.prev_hash != prevHash
            AC-->>WRITER: fail(i, 'prev_hash mismatch') management-api/lib/audit-chain.js:121, splice detected
        else record.hash != hashRecord(prevHash, record)
            AC-->>WRITER: fail(i, 'hash mismatch (record content altered)') management-api/lib/audit-chain.js:124-125, edit<br/>detected
        else ok
            AC->>AC: prevHash = record.hash, chainStarted = true management-api/lib/audit-chain.js:128-129
        end
    end
    AC-->>WRITER: {ok, checked, legacy_prefix, broken_at, tail_hash, tail_seq}<br/>management-api/lib/audit-chain.js:134-137
    Note over AC: reorder is equivalent to a splice at the first moved record — deletion at the tail is the ONE mode a<br/>bare chain cannot see management-api/lib/audit-chain.js:13-16
    AC->>AC: readTail(dir) management-api/lib/audit-chain.js:195, walks newest file backward to resume prevHash<br/>and seq across restarts and daily rotation
```



## AB-14.11 ADR-2087 — the task-property triple, the deny journal, the receipt ladder and manual continuation

```mermaid
sequenceDiagram
    autonumber
    participant AG as agent
    participant GB as governance-bridge<br/>agentbox/mcp/servers/governance-bridge.js:136
    participant TP as task-properties<br/>agentbox/management-api/lib/task-properties.js:112
    participant GATE as buildAuthorityGate.guard<br/>agentbox/management-api/lib/authority.js:226
    participant AJ as authority-journal append<br/>agentbox/management-api/lib/authority-journal.js:92
    participant RC as ApplicationReceiptStore<br/>agentbox/management-api/lib/governance-application-receipts.js:52
    participant RP as receipt publisher<br/>agentbox/management-api/lib/governance-receipt-publisher.js:154
    participant OP as human operator

    AG->>GB: governance_request_action
    GB->>TP: derive verifiability, reversibility, stakes from authority_class
    TP-->>GB: zero-tolerance maps to irreversible, recoverable to compensable (task-properties.js:112)
    Note over TP: INVARIANT — every surface below the operator's manifest is TIGHTENING ONLY,<br/>so a skill's frontmatter may raise recoverable to zero-tolerance and never the reverse<br/>(management-api/lib/task-properties.js:40-55). The triple is stamped on every 31402 as tp- tags
    GB->>GATE: guard({operation, actionClass})
    alt denied at any stage
        GATE->>AJ: append({stage, reason, agent_did, action_class, operation_sha256})
        Note over AJ: an unlabelled denial is not a record — append THROWS without a stage<br/>and without a reason (authority-journal.js:96-101)
        Note over AJ: INVARIANT — fail-CLOSED on the decision, fail-OPEN and loud on the record of it<br/>(management-api/lib/authority-journal.js:31)
        AJ-->>AG: deny carries {code, hint} naming governance_manual_continue (management-api/lib/authority.js:401-403)
        Note over AJ,AG: since b18a52f03 the HTTP 403s keep it — management-api/routes/broker-bridge.js:525-526<br/>and management-api/routes/llm-marketplace.js:512-513 spread gate.code and gate.hint, see AB-14.15
    else admitted
        GATE->>RC: begin(gate, operation) — stage consumer-received (governance-application-receipts.js:64)
        RC->>RP: mirror the stage to the forum receipts endpoint under NIP-98
        alt transport failure
            RP->>AJ: authority.receipt-post-failed then queue in a DEDICATED outbox
            Note over RP: replayed at boot and once a minute by bootReceiptPublisher<br/>(management-api/lib/governance-receipt-publisher.js:471), see AB-14.15
            Note over RP,AJ: its own directory, governance-receipt-outbox under the agentbox state dir<br/>(management-api/lib/governance-receipt-publisher.js:69-72), shared with the governance-bridge MCP process
        end
        RC->>RC: finish(claim, stage, acknowledgement)
        Note over RC: INVARIANT — applied requires a committed mutation acknowledgement,<br/>and the terminal stages are a closed set applied, not-applied, unknown,<br/>applied-manually (governance-application-receipts.js:83-84)
    end
    OP->>GB: governance_manual_continue {case_id, executed_by, evidence} (mcp/servers/governance-bridge.js:218)
    GB->>GB: validate executed_by is a non-empty string up to 128 chars (mcp/servers/governance-bridge.js:377)
    GB->>GB: manualContinue requires a well-formed did:nostr (management-api/lib/governance-manual-continue.js:110-111)
    alt executed_by names an agent identity
        GB-->>OP: refused — a manual continuation must be attributed to a HUMAN operator<br/>(management-api/lib/governance-manual-continue.js:120)
    else a human did:nostr
        GB->>RC: stage applied-manually, bound to the approved operation digest
        Note over GB,RC: the PROV-O activity carries prov:wasAssociatedWith = the human did:nostr<br/>(management-api/lib/governance-manual-continue.js:178)
    end
    Note over GB,RP: this diagram REPLACES the precedent match, promote and retire sequence.<br/>precedent-service.js and precedent-bridge.js were deleted at commit 70d017a3b<br/>and colloquy took the ground — see AB-22.7 and AB-09.5
    Note over TP,GATE: DEBT — the classification table names payment_settlement zero-tolerance<br/>(agentbox.toml:1074) but no production path passes that actionClass. The only<br/>call sites are tests, and management-api/lib/authority.js:226 never sees it. see AB-15
```

## AB-14.12 project-tracker.js publish path and /v1/projects

```mermaid
sequenceDiagram
    autonumber
    participant SCHED as scan trigger<br/>project-tracker.js:604 startScheduler or POST /v1/projects/scan
    participant PT as ProjectTracker<br/>lib/project-tracker.js:230
    participant GIT as git via execFileSync<br/>SAFE_ENV, no shell :121-134
    participant URIS as uris.mint<br/>lib/uris.js
    participant PRIMER as PrimerGenerator<br/>lib/project-primer.js:71
    participant HOOK as project-tracking-publish.cjs<br/>config/hooks/, spawned

    SCHED->>PT: scan({dirs, githubEnrichment}) :379
    loop each repo under scanDirs project-tracker.js:391-405
        PT->>GIT: gitMetadata(repoPath) :193, branch/lastCommit/commits30d/commitDays/remote/language
        PT->>URIS: mint({kind:'thing', localId:'project-'+sha256_12}) :354, content-addressed on remote or path
        PT->>URIS: mint({kind:'dataset', localId:'commits-'+sha+'-30d'}) :363, requires pubkey scope
        opt githubEnrichment and GITHUB_TOKEN set and remote is github project-tracker.js:456
            PT->>PT: _enrichFromGithub(remote) via gh api :329, openIssues/stars, fail-open on error :344-346
        end
        Note over PT: fail-open per repo — one bad repo is logged, scan continues :397-404
    end
    PT->>URIS: mint({kind:'activity', payload:{type:'projscan', dirs, count, outcome}}) :422, one PROV-O scan receipt
    PT-->>SCHED: {projects, scanUrn, durationMs} :438

    Note over PT: GET /v1/projects routes/projects.js:106, GET /v1/projects/:id :146, GET /v1/projects/:id/activity :185
    Note over PT: POST /v1/projects/scan :233, POST /v1/projects/:id/primer :274 delegates to PRIMER.primer() :552-553

    Note right of PT: POST /v1/projects/:id/publish is an INLINE route handler in routes/projects.js:327, NOT a ProjectTracker method — projectDigest() and PUBLISH_HOOK are also defined there (routes/projects.js:28,35), not in project-tracker.js
    alt manifest.project_tracking.nostr_publish is not true
        PT-->>SCHED: {published:false, note:'nostr_publish disabled', urn} routes/projects.js:360-367, fail-open, no spawn
    else nostr_publish true
        PT->>HOOK: spawn('node', [PUBLISH_HOOK]) stdin = projectDigest(project) routes/projects.js:369-403
        HOOK-->>PT: exit code, 0 means published
        PT-->>SCHED: {published, urn} routes/projects.js:405
    end
    Note over PT: kind-30841 digest publish happens INSIDE the hook (nostr-pod-bridge track subcommand) — the calling route never signs or connects to a relay itself
```

## AB-14.13 elevation-publisher.js — outbound elevation boundary to VisionClaw

ADR-2108 deleted the agentbox-local ontology write path (`ontology-local.cjs`, `ontology-authoring-authority.js`); the
governed door is now `vault propose`/`vault edit --expect`, argv built by `lib/ontology-propose.js` and carried in the
31402 as `propose_command`/`propose_iri` instead of the retired HTTP `propose_request` descriptor.

```mermaid
sequenceDiagram
    autonumber
    participant KGE as kg-elevation candidate builder<br/>lib/kg-proposal-extractor.js, see AB-17
    participant EP as buildElevationPublisher<br/>lib/elevation-publisher.js:79
    participant BC20 as bc20-provenance-bridge<br/>durableStore :166
    participant BRIDGE as NostrBridge<br/>mcp/servers/nostr-bridge.js
    participant ACS as agent-control-surface<br/>buildActionRequest, publishPanelEvent

    KGE->>EP: publish(proposal) :151
    alt nostrBridgeEnabled(manifest) is false and no deps.bridge
        EP-->>KGE: {published:false, reason:'nostr-bridge-gate-off'} :86-87,102-103
    else resolveRelays(env) empty and no deps.bridge
        EP-->>KGE: {published:false, reason:'no-relays'} :88-89
    else no signing stack (AGENTBOX_STACK/AGENTBOX_PROFILE unset) and no deps.signer
        EP-->>KGE: {published:false, reason:'no-signing-stack'} :91-93
    end
    Note over EP: static eligibility decided ONCE at boot — reason is cached, logged at debug, never thrown into the request path :84-103
    EP->>EP: ensureConnected() :122, lazy cache bridge+signer, mirrors lib/pod-signer
    alt bridge/signer load throws
        EP-->>KGE: {published:false, reason: loadFailReason} :152-153
    end
    opt proposal.proposal_foreign_urn present
        EP->>BC20: durableStore().put({agentbox_urn, visionclaw_urn, owner_did}) :162-171
        Note over EP,BC20: additive — a storage failure never blocks the publish :159,172
    end
    EP->>ACS: buildActionRequest({panelId: proposal_urn, category:'ontology-elevation', subjectKind:'concept', fields:{propose_command, propose_iri, patch_proposal, ...}}) :179-206
    Note over EP: propose_command is proposal.propose_command.argv, the governed vault propose invocation ADR-2108 :199 — propose_iri is its target :200, patch_proposal carries the machine-gated PatchProposal when the route dry-ran it :204 — none of this is an HTTP request descriptor any more
    EP->>ACS: publishPanelEvent(bridge, signer, unsigned) :208
    ACS->>BRIDGE: publish signed kind-31402 ActionRequest
    alt publish throws
        EP-->>KGE: {published:false, reason: err.message} :214-220, federation is additive
    else published
        EP-->>KGE: {published:true, event_id, kind} :213
    end
    Note over EP,ACS: the relay agent_registry gate plus broker_cases projection then surface the elevation in the governance inbox a human approves from — never an ungoverned local write, see ADR-2108 (supersedes ADR-2022) and AB-25
```

## AB-14.14 action-plane.js — the one wired production instantiation (ADR-2041)

```mermaid
sequenceDiagram
    autonumber
    participant RT as POST /v1/tasks<br/>routes/tasks.js:16, preHandler costGate
    participant AP as getActionPlane<br/>lib/action-plane.js:114, lazy module singleton
    participant MAN as loadManifest + resolveAdapters<br/>action-plane.js:123,140
    participant EJ as new ExecutionJournal<br/>action-plane.js:168
    participant AAP as new AgentActionPipeline<br/>action-plane.js:184, classifier='local' :102-104
    participant DTS as dispatchTaskSpawn<br/>action-plane.js:252

    RT->>DTS: dispatchTaskSpawn({agent, task, provider, ...}, {processManager, logger}) :252
    DTS->>AP: getActionPlane({logger, processManager}) :253
    alt singleton already built
        AP-->>DTS: cached {ready, journal, pipeline}
    else first call
        AP->>MAN: loadManifest() then resolveAdapters(manifest) :123,140
        alt adapters.events missing, no dispatch(), or _implName==='off' action-plane.js:155-164
            AP-->>DTS: {ready:false, reason:'events adapter is not live...'} :156-163
        else events adapter live
            AP->>EJ: new ExecutionJournal({eventsAdapter}) :168
            AP->>AAP: new AgentActionPipeline({secret, journal, classifier:_classifyTaskSpawn, executor}) :184-195
            Note over AAP: executor closes over _boundProcessManager, calls processManager.spawnTask(...) only inside the protected seam :188-193
            AP-->>DTS: {ready:true, journal, pipeline} :204
        end
    end
    alt plane.ready is false
        DTS-->>RT: {ready:false, reason} :254-256, route replies 503, never spawns unjournalled
    else ready
        DTS->>DTS: sessionUrn = uris.mint({kind:'meta', localId:'session-task-...'}) :258
        DTS->>DTS: agentDid = resolveAgentDid(request) :259, auth.pubkey then X-Agentbox-Pubkey then AGENTBOX_AGENT_DID then null :221-231
        DTS->>AAP: pipeline.dispatch({session_urn, agent_did, capability:'agentbox.tasks', operation:'spawn', args, target}) :261-274
        AAP-->>DTS: {decision, output or reason, journal_event_id}
        alt decision is deny
            DTS-->>RT: {ready:true, decision:'denied', denyReason, journalEventId} :281-282
        else decision is allow
            DTS-->>RT: {ready:true, decision:'allow', output, journalEventId} :284
        end
    end
    Note over AP: fail-closed by design :146-154 — ADR-005's off events impl would satisfy ExecutionJournal's own<br/>dispatch-is-a-function guard and silently drop events — action-plane.js checks _implName equal to off<br/>explicitly, one level stricter
    Note over AAP: task-spawn is classified side_effect_class:'local' (FAST_PATH, no approval receipt) — a<br/>deliberate ADR-2041 rollout choice, not a permanent verdict — a future ADR may reclassify it 'mutate'<br/>once a real approver exists :85-101
    Note over RT,DTS: NARROWS but does not close GOVERNANCE-capabilities.md's TOP OPEN RISK (AB-14.1<br/>DIVERGENCE1) — exactly one surface (POST /v1/tasks) is now wired — every other agent-initiated<br/>side-effect surface still bypasses AgentActionPipeline entirely
```

## AB-14.15 ADR-2087 activation wiring — boot replay, per-request resolution, and the staged check

**What it shows.** The four wiring defects the 2 October activation found, as they are now wired at
`c4ed3ec65`: one receipt publisher booted by `server.js` and replayed without overlap, broker-bridge
resolving the boot-built gate and publisher per request, both HTTP denial routes carrying `{code, hint}`,
the governance-bridge MCP server recognising itself through the Nix-store symlink, and the activation
check that turns all of it into a `staged` or `live` verdict.

**Why it is this way.** Every unit suite passed while the shipped image stayed inert: Fastify loads a
top-level-registered plugin at the first awaited `register` inside `start()`, before the boot block
decorates the gate, journal and publisher, so anything read at registration was `undefined`
(ADR-2087 "Activation finding", fixed at `b18a52f03`). `scripts/activation/adr-2087-check.sh`
(`ddfb6d056`) checks the running process rather than the tests; ADR-2087 moved to `staged` on its exit 2
at `c4ed3ec65`.

```mermaid
sequenceDiagram
    autonumber
    participant SRV as server.js boot block<br/>management-api/server.js:1201
    participant BRP as bootReceiptPublisher<br/>management-api/lib/governance-receipt-publisher.js:471
    participant BB as broker-bridge decide<br/>management-api/routes/broker-bridge.js:501
    participant LLM as POST /v1/llm/revoke<br/>management-api/routes/llm-marketplace.js:438
    participant GB as governance-bridge entrypoint<br/>agentbox/mcp/servers/governance-bridge.js:553
    participant CHK as adr-2087-check.sh<br/>agentbox/scripts/activation/adr-2087-check.sh:12-18

    SRV->>BRP: bootReceiptPublisher({manifest, logger, journal: authorityDenyJournal}) management-api/server.js:1201-1203
    BRP->>BRP: log governance.receipt-replay.boot with configured true or false (management-api/lib/governance-receipt-publisher.js:494)
    BRP->>BRP: initialFlush, then publisher.start with the same replay tick every 60 s (management-api/lib/governance-receipt-publisher.js:500-501)
    Note over BRP: INVARIANT one replay at a time — a second tick returns the in-flight promise,<br/>so a hanging forum never posts an entry twice (management-api/lib/governance-receipt-publisher.js:480-491)
    alt forum_auth_api unset
        BRP->>BRP: count entries, skipped not-configured, spend no attempt (management-api/lib/governance-receipt-publisher.js:368-377)
    else forum_auth_api set (agentbox.toml:200, the auth worker since 60b60b48d)
        BRP->>BRP: post each queued entry, log governance.receipt-posted at info (management-api/lib/governance-receipt-publisher.js:407)
    end
    SRV->>SRV: decorate governanceReceiptPublisher, stop it onClose (management-api/server.js:1204-1205)

    BB->>BB: authorityGate() resolves options, then fastify.authorityGate, per request (management-api/routes/broker-bridge.js:286)
    alt nothing decorated yet
        BB->>BB: build a journalled fallback gate once and warn broker-bridge.authority-fallback (management-api/routes/broker-bridge.js:298)
    end
    BB->>BB: receiptPublisher() resolves the boot publisher per request (management-api/routes/broker-bridge.js:258)
    BB-->>BB: 403 carries gate.code and gate.hint (management-api/routes/broker-bridge.js:525-526)

    LLM->>LLM: gate journal is a per-denial shim onto fastify.authorityDenyJournal (management-api/routes/llm-marketplace.js:78-87)
    Note over LLM: an unwired journal throws, which the gate reports as authority.deny-unjournalled<br/>(management-api/lib/authority.js:313) — loud, never a silent success
    LLM-->>LLM: 403 carries {code: no-decision-surface, hint: governance_manual_continue} (management-api/routes/llm-marketplace.js:512-513)
    Note over LLM: an authority.deny row carries a did:nostr actor and a string target, and /v1/agent-events now serialises both<br/>(management-api/routes/agent-events.js:160-161, fc56e7a97)

    GB->>GB: compare realpath of argv[1] with realpath of the module URL (agentbox/mcp/servers/governance-bridge.js:550-554)
    Note over GB: path.resolve compared a /opt symlink with the /nix/store path Node hands an ES module,<br/>so the server exited 0 without connecting until b18a52f03

    CHK->>SRV: B1 replay boot line present and no fallback-gate line for this pid (agentbox/scripts/activation/adr-2087-check.sh:188-202)
    CHK->>LLM: B3 one live revoke of a non-existent grant, expect 403 with code and hint
    CHK->>GB: B6 launch exactly as mcp.json does, expect governance_manual_continue (agentbox/scripts/activation/adr-2087-check.sh:286)
    CHK->>CHK: B7 needs forum configured AND at least one governance.receipt-posted (agentbox/scripts/activation/adr-2087-check.sh:332-344)
    CHK->>CHK: C3 only with --api-down-night DATE, and only when the one-shot state shows phase done,<br/>a stop time and reason night-record or deadline (agentbox/scripts/activation/adr-2087-check.sh:411-413)
    Note over CHK: exit 0 LIVE, 2 STAGED, 1 FAIL. 2026-10-02 run on the rebuilt image exited 2 —<br/>posted 0, queued 0 (agentbox/docs/adr/ADR-2087-task-properties-receipts-and-manual-continuation.md:164-166)
```

**Invariant:** a receipt the forum has not yet received spends no retry budget while `forum_auth_api` is unset, so the degraded state queues rather than abandons (`management-api/lib/governance-receipt-publisher.js:368-377`).

**Drift (resolved 2026-10-02, `e8460a2e3`):** ADR-2087's Consequences used to say the forum receipts endpoint did not exist yet on the deployed edge; they now record it deployed, with `forum_auth_api` pointing at it, so the queue drains once a real governance response exists (`docs/adr/ADR-2087-task-properties-receipts-and-manual-continuation.md:94-98`), matching the staged section (`docs/adr/ADR-2087-task-properties-receipts-and-manual-continuation.md:164-166`) and the manifest (`agentbox.toml:200`).

**Invariant:** ADR-2071 clause (c) cannot pass on a night of failed journal posts alone: C3 also requires the one-shot's state file to record a clean stop and a `night-record` or `deadline` restart, and an `interrupted` night fails (`scripts/activation/adr-2087-check.sh:403-413`, `scripts/activation/adr-2071-api-down-night.sh:136-145`). AB-23.19 draws the one-shot.

**Open:** the EXP-B8 tick takes two outward side effects when its stopping rule fires, a signed forum post (`scripts/experiments/exp-b8-label-log.cjs:550-563`) and a pull request opened with `gh` (`scripts/experiments/exp-b8-label-log.cjs:573-606`), and records neither through `POST /v1/exec/record` or the authority journal; no record says whether an unattended experiment tick is meant to sit outside the governance journal that ADR-2071 put round the dream night.

**Open:** ADR-2087 reaches `live` only on the first real governance response after the rebuild (B7, `scripts/activation/adr-2087-check.sh:332-344`); the record does not say who is expected to produce that response, or when.

## AB-14.16 The identity port as the signer — named operations by caller uid, flag off versus flag on

```mermaid
flowchart TB
    subgraph OFF["role_isolation = false, the default the image ships: every signer holds its own key"]
        direction TB
        OEXIT["serve-identity prints one line and exits 0, reading no key<br/>services/nostr-pod-bridge/src/identity_port/mod.rs:92-96"]
        OJJ["JunkieJarvis, forum suggestions and EXP-B8 read the JJ key through role-secret,<br/>where the bare variable still counts while the flag is off<br/>management-api/lib/junkiejarvis-agent.js:92-99, scripts/dream-forum-suggestions.mjs:194-200"]
        OCOL["colloquy publish and the digests load the bridge key from AGENTBOX_BRIDGE_SK_FILE<br/>or the legacy variable, services/nostr-pod-bridge/src/lib.rs:179-192"]
        OPOD["pods NIP-98 keeps its key-file or stack-key source<br/>management-api/lib/pod-signer.js:76-79"]
        OEXIT --> OJJ --> OCOL --> OPOD
    end
    subgraph ON["role_isolation = true: one process holds the keys and signs named operations"]
        direction TB
        PROG["serve-identity runs under ab-identity in the isolated config,<br/>env -i so PID 1's inherited environment never reaches it, flake.nix:2515-2531"]
        SOCK["socket /run/secrets/ab-identity-port/identity.sock, mode 0660 in group ab-identity-port<br/>services/nostr-pod-bridge/src/identity_port/server.rs:17, config/role-accounts.json:94-102"]
        PEER["the peer uid comes from SO_PEERCRED on every connection, never from the socket mode<br/>server.rs:104-111"]
        GATE{"uid in the ACL, op in the closed list, op granted to that uid<br/>services/nostr-pod-bridge/src/identity_port/port.rs:361-379"}
        OPS["closed list: pubkey, nip98, sign_event, forum_event, nip42_auth, mirror_key<br/>services/nostr-pod-bridge/src/identity_port/acl.rs:43-50"]
        DEV["uid 1000 devuser: nip98 and nip42_auth on core, sign_event core kinds 38410-38413, 38415, 30840, 30841,<br/>forum_event junkiejarvis kinds 1, 42, 31923, mirror_key with the child secret<br/>config/custody/identity-port-acl.json:34-44"]
        GWY["uid 961 ab-gateway: pubkey, nip42_auth on core, mirror_key public half only<br/>identity-port-acl.json:45-52"]
        REC["one JSONL receipt per decision, admit or refuse, 0640, under /var/lib/agentbox/events/sign<br/>services/nostr-pod-bridge/src/identity_port/receipts.rs:124-138"]
        PROG --> SOCK --> PEER --> GATE --> OPS
        OPS --> DEV
        OPS --> GWY
        DEV --> REC
        GWY --> REC
    end
    OFF --> ON
    GATE -.-> INV1["INVARIANT: kinds 27235, 22242, 31400-31405 and 38414 can never be granted to sign_event or forum_event.<br/>The ACL loader refuses a file that tries, so a container key never signs a governance decision<br/>or a graduation, services/nostr-pod-bridge/src/identity_port/acl.rs:71-73, acl.rs:267-268"]
    REC -.-> INV2["INVARIANT: nothing is signed off the record. An admitted request whose receipt line<br/>cannot be written is answered as a refusal, port.rs:320-326"]
    DEV -.-> WIRE["pods NIP-98 is the one wired client: the flag forces the port, then<br/>nostr-pod-bridge sign-request nip98 over the socket, management-api/lib/pod-signer.js:132-136,<br/>pod-signer.js:152-170, services/nostr-pod-bridge/src/main.rs:81"]
    WIRE -.-> OPEN1["OPEN: sign_event, forum_event and nip42_auth have no client at this revision. JunkieJarvis,<br/>forum suggestions, colloquy publish and the gateway still read ab-identity key files<br/>that devuser cannot open under the flag, so they fail closed until the W3b cutover,<br/>docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:218-219"]
```

**What it shows.** Flag off, nothing changes: `serve-identity` exits at once and every governance-adjacent signer reads its own key as before. Flag on, `nostr-pod-bridge serve-identity` is the only holder of the core and JunkieJarvis keys. It authorises each connection by the kernel's peer uid, admits only a closed set of operations per uid, writes a receipt for every admit and refusal, and refuses outright any ACL that would let a container key sign a governance kind or a graduation.

**Why it is this way.** ADR-2122 (custody X-1 step 1) keeps the boundary in-container: a second supervisor config runs role programs under their own uids, and ADR-2101's rule that a generic "sign this payload" port is a bypass becomes the closed operation list. Only the pods signer was moved onto the port in W3. The remaining consumers are owed as W3b, which is why the flag must stay off until the owner's boot rehearsal says otherwise (see AB-36).

**Invariant (under role_isolation):** the governance kinds 31400-31405 and the graduation kind 38414 cannot be signed by any container process through the port, because no ACL that grants them loads (`../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs:71-73`, `../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs:267-268`). With the flag off the port does not run (`../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs:92-96`) and this says nothing about the env-key signers.

**Open:** under the flag the JunkieJarvis agent reads its key through role-secret and gets nothing it can open (`../project/agentbox/management-api/lib/junkiejarvis-agent.js:92-99`), so forum replies and governance-panel posts signed as JunkieJarvis stop until the W3b consumer cutover lands (`../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:218-219`); the port's `forum_event` grant (`../project/agentbox/config/custody/identity-port-acl.json:40`) has no caller yet.
