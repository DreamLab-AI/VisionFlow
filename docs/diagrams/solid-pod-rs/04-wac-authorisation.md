---
id: SP-04
title: Web Access Control — policy resolution, evaluation, conditions and the sidecar rule
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2002, ADR-2005]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/evaluator.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/parser.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/document.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/conditions.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/payment.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/client.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/issuer.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/anchor.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/origin.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/serializer.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
verified_commit: 1d9da5270
---

## SP-04.1 The write gate, end to end

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant EW as enforce_write_ctx<br/>solid-pod-rs-server/src/lib.rs:819
    participant ET as wac::effective_acl_target<br/>solid-pod-rs/src/wac/mod.rs:252
    participant RP as resolve_policy_dyn<br/>solid-pod-rs-server/src/lib.rs:1941
    participant BAL as resolve_balance_sats<br/>solid-pod-rs-server/src/lib.rs:670
    participant EV as wac::evaluate_access_ctx_with_registry<br/>solid-pod-rs/src/wac/evaluator.rs:234
    participant CH as charge_granted_payment<br/>solid-pod-rs-server/src/lib.rs:913

    H->>EW: (path, mode, agent_uri, Origin)
    EW->>EW: wac::Origin::parse of the raw Origin header<br/>solid-pod-rs-server/src/lib.rs:826
    EW->>ET: map (path, mode) to the resource actually governed
    ET-->>EW: (resource, eff_mode) — sidecars elevate to Control
    EW->>RP: typed policy resolution over the effective resource
    RP-->>EW: PolicyOutcome
    alt outcome.is_failure()
        EW-->>H: policy_failure_to_actix — 403 or 503<br/>solid-pod-rs-server/src/lib.rs:1957
    end
    EW->>BAL: Web-Ledger balance for the principal (None if anonymous)
    EW->>EV: (acl_doc, ctx, resource, eff_mode, origin, groups, registry)
    EV-->>EW: granted true or false
    alt not granted
        EW-->>H: acl_denial — 401 with WWW-Authenticate, or 403<br/>solid-pod-rs-server/src/lib.rs:939
    end
    opt resource == path, i.e. no sidecar elevation
        EW->>CH: debit the granting rule's PaymentCondition cost
    end
    EW-->>H: Ok
```

## SP-04.2 The read gate and the ADR-2002 audience classification

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_get / handle_glob_get
    participant ER as enforce_read_ctx<br/>solid-pod-rs-server/src/lib.rs:994
    participant ET as effective_acl_target with base Read<br/>solid-pod-rs-server/src/lib.rs:1012
    participant RP as resolve_policy_dyn<br/>solid-pod-rs-server/src/lib.rs:1941
    participant EV as evaluate_access_ctx_with_registry<br/>solid-pod-rs/src/wac/evaluator.rs:234

    H->>ER: (path, agent_uri, Origin)
    ER->>ET: an acl/meta sidecar becomes (protected, Control)
    ER->>RP: resolve typed policy
    RP-->>ER: PolicyOutcome, failures deny here
    ER->>EV: evaluate for the PRINCIPAL
    EV-->>ER: granted or denial
    ER->>EV: RE-evaluate the SAME document with web_id = None<br/>solid-pod-rs-server/src/lib.rs:1074
    EV-->>ER: anonymous_would_be_granted
    ER-->>H: Public only when anonymous is granted AND no elevation happened<br/>solid-pod-rs-server/src/lib.rs:1083
    Note over ER: The second evaluation is pure — no extra storage I/O. A sidecar read<br/>and a payment-gated read both classify Private by construction.
    Note over H: HEAD routes to handle_get, so HEAD inherits this same gate — see SP-02.11.
```

## SP-04.3 Typed policy outcomes — only absence inherits

```mermaid
stateDiagram-v2
    [*] --> Probe: acl_sidecar_key(path)<br/>solid-pod-rs/src/wac/resolver.rs:333
    Probe --> Absent: backend NotFound
    Probe --> Present: body read
    Probe --> Failed: any other backend error

    Absent --> Ascend: PolicyStep.Ascend<br/>solid-pod-rs/src/wac/resolver.rs:249
    Ascend --> Probe: parent_container, inherited = true<br/>solid-pod-rs/src/wac/resolver.rs:344
    Ascend --> Missing: no parent left — root probed<br/>solid-pod-rs/src/wac/resolver.rs:401

    Present --> Found: parsed JSON-LD or Turtle
    Present --> Invalid: PayloadTooLarge, depth bomb,<br/>not UTF-8, or malformed
    Failed --> Unavailable

    Missing --> [*]: deny by default (no ACL means no access)
    Found --> [*]: evaluate
    Invalid --> [*]: DENY, never inherit
    Unavailable --> [*]: DENY, never inherit

    note right of Invalid
      InvalidPolicyReason: TooLarge, TooDeep, NotUtf8, Malformed
      solid-pod-rs/src/wac/resolver.rs:65
      INVARIANT (ADR-2005): collapsing Invalid or Unavailable back into
      Ok(None) re-opens the escalation where a malformed RESTRICTIVE
      policy is silently replaced by a permissive inherited one.
    end note
    note right of Missing
      may_inherit() is true only for Missing
      solid-pod-rs/src/wac/resolver.rs:153
    end note
```

## SP-04.4 classify_policy_read — one level, runtime-free

```mermaid
flowchart TD
    IN["classify_policy_read(policy_path, read, inherited)<br/>solid-pod-rs/src/wac/resolver.rs:265"]
    ABS["PolicyRead::Absent -> Ascend<br/>solid-pod-rs/src/wac/resolver.rs:272"]
    FAIL["PolicyRead::Failed -> Unavailable<br/>solid-pod-rs/src/wac/resolver.rs:275"]
    JL["parse_jsonld_acl<br/>solid-pod-rs/src/wac/mod.rs:78"]
    TOOBIG["PayloadTooLarge -> Invalid TooLarge<br/>solid-pod-rs/src/wac/resolver.rs:298"]
    TOODEEP["BadRequest -> Invalid TooDeep<br/>solid-pod-rs/src/wac/resolver.rs:299"]
    UTF["not UTF-8 -> Invalid NotUtf8<br/>solid-pod-rs/src/wac/resolver.rs:305"]
    SNIFF["looks_turtle by content type, or contains @prefix / acl:Authorization<br/>solid-pod-rs/src/wac/resolver.rs:309"]
    MAL["neither JSON-LD nor Turtle -> Invalid Malformed<br/>solid-pod-rs/src/wac/resolver.rs:313"]
    TT["parse_turtle_acl<br/>solid-pod-rs/src/wac/parser.rs:26"]
    FOUND["Found(doc) with doc.inherited stamped<br/>solid-pod-rs/src/wac/resolver.rs:295"]

    IN --> ABS
    IN --> FAIL
    IN --> JL
    JL -- ok --> FOUND
    JL --> TOOBIG
    JL --> TOODEEP
    JL -- "not JSON at all" --> UTF
    UTF --> SNIFF
    SNIFF -- no --> MAL
    SNIFF -- yes --> TT
    TT -- ok --> FOUND
    TT --> TOOBIG

    N["INVARIANT: a bound violation is TERMINAL. An oversized or depth-bombed ACL<br/>must never fall through to the ancestor."]
    TOOBIG -.-> N
    N2["Pure and runtime-free — this is the function the wasm/edge tier can adopt<br/>with a version pin rather than a port. See SP-01.4."]
    IN -.-> N2
```

## SP-04.5 The sidecar rule — one function for read and write

```mermaid
flowchart LR
    IN["effective_acl_target(path, base_mode)<br/>solid-pod-rs/src/wac/mod.rs:252"]
    PR["protected_resource_for_acl(path)<br/>solid-pod-rs/src/wac/mod.rs:217"]
    ORD["ordinary resource -> (path, base_mode) unchanged<br/>solid-pod-rs/src/wac/mod.rs:255"]
    ELV["sidecar -> (protected_resource, Control)<br/>solid-pod-rs/src/wac/mod.rs:254"]
    RD["enforce_read_ctx passes base Read<br/>solid-pod-rs-server/src/lib.rs:1012"]
    WR["enforce_write_ctx passes the request's write mode<br/>solid-pod-rs-server/src/lib.rs:838"]

    IN --> PR
    PR -- None --> ORD
    PR -- "Some(p)" --> ELV
    RD --> IN
    WR --> IN

    N1["Suffix rules: /.acl and /dir/.acl strip to / and /dir/ respectively;<br/>/a/b.acl strips to the resource /a/b.<br/>solid-pod-rs/src/wac/mod.rs:221"]
    PR -.-> N1
    N2["INVARIANT (WAC 4.3.5): reading OR writing a sidecar discloses or rewrites<br/>the whole authorization graph of the governed resource, so both gate on<br/>acl:Control of that resource — never the base mode on the sidecar path.<br/>P0-2 and P0-3 were the write and read halves of the same bug."]
    ELV -.-> N2
    N3["Because read and write share this ONE function they cannot drift apart."]
    IN -.-> N3
```

## SP-04.6 The lockout guard — you may not ACL yourself out

```mermaid
sequenceDiagram
    autonumber
    participant C as Client with Control
    participant H as PUT / POST / PATCH handler
    participant G as proposed_acl_keeps_caller_control<br/>solid-pod-rs-server/src/lib.rs:721
    participant IDS as ids_of_acl_field<br/>solid-pod-rs-server/src/lib.rs:788

    C->>H: write a new .acl body
    H->>H: enforce Control on the protected resource (SP-04.5)
    H->>G: does the PROPOSED document still grant the caller Control?
    G->>IDS: inspect agent / agentClass ids on each authorization
    alt keeps Control via absolute WebID, foaf:Agent or acl:AuthenticatedAgent
        G-->>H: true
        H->>H: proceed with the write
    else
        G-->>H: false
        H-->>C: 409 Conflict — refused<br/>solid-pod-rs-server/src/lib.rs:1477
    end
    Note over H: The guard runs on all three write paths: PUT (lib.rs:1474),<br/>POST after Slug resolution (lib.rs:1582), and PATCH on the SERIALISED<br/>post-patch document (lib.rs:1718) — F7 closed the PATCH hole.
```

## SP-04.7 evaluate_access_ctx_inner — the decision loop

```mermaid
flowchart TD
    START["evaluate_access_ctx_inner<br/>solid-pod-rs/src/wac/evaluator.rs:255"]
    NODOC["acl_doc None or graph None -> DENY<br/>solid-pod-rs/src/wac/evaluator.rs:264"]
    INH["honour_access_to = NOT doc.inherited<br/>solid-pod-rs/src/wac/evaluator.rs:275"]
    LOOP["for each acl:Authorization in the graph"]
    MODE["get_modes contains the required mode?<br/>solid-pod-rs/src/wac/evaluator.rs:108"]
    AGENT["agent_matches_with_groups?<br/>solid-pod-rs/src/wac/evaluator.rs:116"]
    PATHA["acl:accessTo matches — only when NOT inherited<br/>solid-pod-rs/src/wac/evaluator.rs:287"]
    PATHD["else acl:default matches<br/>solid-pod-rs/src/wac/evaluator.rs:295"]
    COND["conjunctive condition gate — ALL must be Satisfied<br/>solid-pod-rs/src/wac/evaluator.rs:309"]
    GRANT["base_grant = true, break<br/>solid-pod-rs/src/wac/evaluator.rs:325"]
    CTRL["required_mode == Control -> return true, skip the origin gate<br/>solid-pod-rs/src/wac/evaluator.rs:335"]
    ORG["origin::check_origin (feature acl-origin)<br/>solid-pod-rs/src/wac/evaluator.rs:343"]
    DENY["DENY"]

    START --> NODOC
    START --> INH --> LOOP --> MODE
    MODE -- no --> LOOP
    MODE -- yes --> AGENT
    AGENT -- no --> LOOP
    AGENT -- yes --> PATHA
    PATHA -- no --> PATHD
    PATHD -- no --> LOOP
    PATHA -- yes --> COND
    PATHD -- yes --> COND
    COND -- "NotApplicable or Denied" --> LOOP
    COND -- all Satisfied --> GRANT
    LOOP -- exhausted --> DENY
    GRANT --> CTRL
    CTRL -- not Control --> ORG
    ORG --> DENY

    N["INVARIANT (WAC 4.2): an ACL resolved from an ANCESTOR honours ONLY acl:default.<br/>acl:accessTo names an exact resource and must not inherit to descendants —<br/>without this gate an ancestor accessTo-only grant leaks to every child."]
    INH -.-> N
    N2["INVARIANT (WAC 4.3): Control bypasses the origin gate so an owner can always<br/>repair a mis-configured ACL from any origin."]
    CTRL -.-> N2
```

## SP-04.8 The origin gate — off by default

```mermaid
stateDiagram-v2
    [*] --> FeatureCheck
    FeatureCheck --> Ignored: cfg(not(feature acl-origin))<br/>solid-pod-rs/src/wac/evaluator.rs:352
    FeatureCheck --> Active: cfg(feature acl-origin)<br/>solid-pod-rs/src/wac/evaluator.rs:341

    Active --> Decide: origin.check_origin(doc, request_origin)<br/>solid-pod-rs/src/wac/origin.rs:268
    Decide --> Allow: NoPolicySet or Permitted<br/>solid-pod-rs/src/wac/origin.rs:239
    Decide --> Reject: RejectedMismatch or RejectedNoOrigin
    Reject --> Counted: ACL_ORIGIN_REJECTED_TOTAL incremented<br/>solid-pod-rs/src/wac/evaluator.rs:345
    Counted --> [*]: access denied
    Allow --> [*]: access granted
    Ignored --> [*]: granted — pre-F4 behaviour preserved

    note right of Active
      Origin.parse normalises the raw header (solid-pod-rs/src/wac/origin.rs:50) —
      OriginPattern.parse and matches implement the wildcard forms
      (solid-pod-rs/src/wac/origin.rs:124 and :176) — extract_origin_patterns
      pulls acl:origin triples off an authorization
      (solid-pod-rs/src/wac/origin.rs:218).
    end note
    note right of Ignored
      DIVERGENCE: acl-origin is NOT in any default feature set (SP-01.6), so a
      stock build ignores the Origin header entirely and a cross-origin write
      is gated only by the agent's credential.
    end note
```

## SP-04.9 Parser bounds — fail closed at the boundary

```mermaid
flowchart LR
    JSONLD["parse_jsonld_acl<br/>solid-pod-rs/src/wac/mod.rs:78"]
    ENVB["JSS_MAX_ACL_BYTES, default MAX_ACL_BYTES 1 MiB<br/>solid-pod-rs/src/wac/mod.rs:28"]
    ENVD["JSS_MAX_ACL_JSON_DEPTH, default 32<br/>solid-pod-rs/src/wac/mod.rs:33"]
    WL["parse_jsonld_acl_with_limits<br/>solid-pod-rs/src/wac/mod.rs:96"]
    DEPTH["check_json_depth — counts braces without parsing,<br/>ignores braces inside string literals<br/>solid-pod-rs/src/wac/mod.rs:39"]
    TURTLE["parse_turtle_acl<br/>solid-pod-rs/src/wac/parser.rs:26"]
    TWL["parse_turtle_acl_with_limit — 413 above the cap<br/>solid-pod-rs/src/wac/parser.rs:40"]

    JSONLD --> ENVB
    JSONLD --> ENVD
    JSONLD --> WL --> DEPTH
    TURTLE --> TWL

    N["INVARIANT: the cap sits AT the parse boundary. check_json_depth fails fast<br/>before serde_json is reached, because serde allocates stack proportional to<br/>nesting depth — removing the cap reopens a parse-bomb DoS (CWE-400)."]
    DEPTH -.-> N
    N2["Both bound failures surface as PolicyOutcome::Invalid, so a bomb denies<br/>rather than inheriting. See SP-04.4."]
    TWL -.-> N2
```

## SP-04.10 The Turtle ACL parser

```mermaid
flowchart TD
    IN["parse_turtle_acl_with_limit<br/>solid-pod-rs/src/wac/parser.rs:40"]
    STRIP["strip_turtle_comments<br/>solid-pod-rs/src/wac/parser.rs:93"]
    SPLIT["split_turtle_statements<br/>solid-pod-rs/src/wac/parser.rs:118"]
    AUTH["parse_turtle_authorization<br/>solid-pod-rs/src/wac/parser.rs:150"]
    PRED["split_predicate_list<br/>solid-pod-rs/src/wac/parser.rs:242"]
    COBJ["parse_turtle_condition_objects<br/>solid-pod-rs/src/wac/parser.rs:287"]
    CBOD["parse_turtle_condition_body<br/>solid-pod-rs/src/wac/parser.rs:349"]
    NORM["normalise_condition_type<br/>solid-pod-rs/src/wac/parser.rs:460"]
    CURIE["expand_curie_or_iri against the prefix map<br/>solid-pod-rs/src/wac/parser.rs:545"]
    DOC["AclDocument { graph, inherited }<br/>solid-pod-rs/src/wac/document.rs:13"]
    SER["serialize_turtle_acl — the inverse<br/>solid-pod-rs/src/wac/serializer.rs:7"]

    IN --> STRIP --> SPLIT --> AUTH
    AUTH --> PRED --> CURIE
    AUTH --> COBJ --> CBOD --> NORM
    AUTH --> DOC
    DOC --> SER

    N["acl, foaf and vcard prefixes are pre-seeded so a document that omits its<br/>@prefix lines still parses.<br/>solid-pod-rs/src/wac/parser.rs:52"]
    IN -.-> N
```

## SP-04.11 Access modes and the ACL document model

```mermaid
classDiagram
    class AccessMode {
        <<enum>>
        Read  solid-pod-rs/src/wac/mod.rs:155
        Write  solid-pod-rs/src/wac/mod.rs:156
        Append  solid-pod-rs/src/wac/mod.rs:157
        Control  solid-pod-rs/src/wac/mod.rs:158
    }
    class MethodMap {
        GET and HEAD to Read  solid-pod-rs/src/wac/mod.rs:182
        PUT DELETE PATCH to Write  solid-pod-rs/src/wac/mod.rs:183
        POST to Append  solid-pod-rs/src/wac/mod.rs:184
        anything else to Read  solid-pod-rs/src/wac/mod.rs:185
    }
    class AclDocument {
        +graph Option~Vec~AclAuthorization~~  solid-pod-rs/src/wac/document.rs:13
        +inherited bool
    }
    class AclAuthorization {
        +access_to  solid-pod-rs/src/wac/document.rs:35
        +default
        +agent / agentClass / agentGroup
        +mode
        +condition
    }
    class IdOrIds {
        <<enum>>
        solid-pod-rs/src/wac/document.rs:88
    }
    AccessMode <.. MethodMap
    AclDocument *-- AclAuthorization
    AclAuthorization ..> IdOrIds
    note for MethodMap "method_to_mode (solid-pod-rs/src/wac/mod.rs:180) maps an UNKNOWN verb to\nRead, the least-privileged mode — fail closed, never to Write."
```

## SP-04.12 The condition registry

```mermaid
classDiagram
    class Condition {
        <<enum>>
        Client(ClientConditionBody)  solid-pod-rs/src/wac/conditions.rs:50
        Issuer(IssuerConditionBody)  solid-pod-rs/src/wac/conditions.rs:53
        Payment(PaymentConditionBody)  solid-pod-rs/src/wac/conditions.rs:56
        ProvenanceAnchor(body)  solid-pod-rs/src/wac/conditions.rs:61
        Unknown with type_iri preserved  solid-pod-rs/src/wac/conditions.rs:66
    }
    class ConditionOutcome {
        <<enum>>
        solid-pod-rs/src/wac/conditions.rs:30
    }
    class ConditionRegistry {
        +with_client  solid-pod-rs/src/wac/conditions.rs:265
        +with_issuer  solid-pod-rs/src/wac/conditions.rs:271
        +with_payment  solid-pod-rs/src/wac/conditions.rs:277
        +with_provenance_anchor  solid-pod-rs/src/wac/conditions.rs:286
        +default_with_client_and_issuer  solid-pod-rs/src/wac/conditions.rs:293
        +supported_iris  solid-pod-rs/src/wac/conditions.rs:304
    }
    class RequestContext {
        +web_id / client_id / issuer / payment_balance_sats  solid-pod-rs/src/wac/conditions.rs:215
    }
    class EmptyDispatcher {
        solid-pod-rs/src/wac/conditions.rs:355
    }
    ConditionRegistry ..> Condition
    ConditionRegistry ..> ConditionOutcome
    ConditionRegistry ..> RequestContext
    ConditionOutcome <.. EmptyDispatcher
    note for Condition "validate_for_write (solid-pod-rs/src/wac/conditions.rs:382) and\nvalidate_acl_document (:411) reject an ACL carrying an Unknown condition at\nWRITE time, so an unrecognised gate can never be stored and then ignored."
```

## SP-04.13 Payment, client, issuer and provenance-anchor conditions

```mermaid
flowchart TD
    PAY["PaymentConditionEvaluator::evaluate<br/>solid-pod-rs/src/wac/payment.rs:70"]
    BODY["PaymentConditionBody with cost_sats<br/>solid-pod-rs/src/wac/payment.rs:27"]
    TOT["total_payment_cost over a condition list<br/>solid-pod-rs/src/wac/payment.rs:85"]
    GPC["wac::granted_payment_cost — cost of the ONE granting rule<br/>solid-pod-rs/src/wac/evaluator.rs:375"]
    CG["charge_granted_payment then debit_ledger<br/>solid-pod-rs-server/src/lib.rs:913"]
    CLI["ClientConditionEvaluator::evaluate<br/>solid-pod-rs/src/wac/client.rs:54"]
    ISS["IssuerConditionEvaluator::evaluate<br/>solid-pod-rs/src/wac/issuer.rs:44"]
    ANC["ProvenanceAnchorEvaluator::evaluate — a MARKER, always satisfied<br/>solid-pod-rs/src/wac/anchor.rs:102"]
    MODE["anchor_mode_of — HighValue or Epoch<br/>solid-pod-rs/src/wac/anchor.rs:119"]

    BODY --> PAY --> GPC --> CG
    PAY --> TOT
    CLI --> GATE["conjunctive gate in SP-04.7"]
    ISS --> GATE
    PAY --> GATE
    ANC --> MODE --> PROV["feeds AnchorPolicy — see SP-07.3"]

    N1["A granted request is charged the cost of the rule it actually used, not the<br/>sum of every PaymentCondition in the document."]
    GPC -.-> N1
    N2["INVARIANT: a debit failure denies the request with the same shape as a WAC<br/>denial — the pod never serves unpaid, even if a concurrent spend raced the<br/>balance below cost after the gate passed.<br/>solid-pod-rs-server/src/lib.rs:928"]
    CG -.-> N2
    N3["ProvenanceAnchor is not an access gate — it never denies. Treating it as one<br/>would make an anchor-worthy resource unreadable."]
    ANC -.-> N3
```

## SP-04.14 Denial shape

```mermaid
flowchart LR
    IN["acl_denial(acl_doc, agent_uri, path)<br/>solid-pod-rs-server/src/lib.rs:939"]
    ANON{"agent_uri is None?"}
    U401["401 Unauthorized, body 'authentication required'<br/>solid-pod-rs-server/src/lib.rs:946"]
    F403["403 Forbidden, body 'access forbidden'<br/>solid-pod-rs-server/src/lib.rs:948"]
    WA["WAC-Allow advisory header on BOTH<br/>solid-pod-rs-server/src/lib.rs:952"]
    CH["WWW-Authenticate: Nostr, DPoP, Bearer — all three realms<br/>solid-pod-rs-server/src/lib.rs:966"]

    IN --> ANON
    ANON -- yes --> U401 --> WA
    ANON -- no --> F403 --> WA
    U401 --> CH

    N["Advertising the Nostr scheme matters: without it a did:nostr agent has no<br/>protocol signal that NIP-98 is accepted and cannot know how to retry."]
    CH -.-> N
    N2["A policy FAILURE is a different shape: Invalid gives 403 'governing ACL is<br/>invalid', Unavailable gives 503 'access control unavailable'.<br/>solid-pod-rs-server/src/lib.rs:1957"]
    IN -.-> N2
```
