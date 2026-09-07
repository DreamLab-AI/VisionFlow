---
id: ES-03
title: Cross-repo federation contract (agentbox <-> VisionClaw)
area: estate
governing:
  - ../project/docs/IDENTIFIER-taxonomy.md
  - ../project/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/docs/DATA-authority-erasure.md
  - ../project/docs/BASELINE-architecture.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [visionclaw:ADR-2023, visionclaw:ADR-2025, agentbox:ADR-2061]
sources:
  - ../project/src/uri/mod.rs
  - ../project/src/services/provenance_writer.rs
  - ../project/src/agent_events/schema.rs
  - ../project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/management-api/lib/bc20-provenance-bridge.js
  - ../project/agentbox/management-api/adapters/index.js
  - ../project/agentbox/management-api/observability/metrics.js
  - ../project/agentbox/management-api/middleware/privacy-filter.js
  - ../project/agentbox/management-api/middleware/linked-data/encoder.js
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/adapters/lifecycle.js
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/management-api/routes/uri-resolver.js
  - ../project/agentbox/management-api/utils/agent-event-publisher.js
  - ../project/agentbox/schema/federation-kinds.json
verified_commit: {visionclaw: 36bb64e1e, agentbox: 2c521c5bb}
---
## ES-03.1 Shared wire envelope: AgentActionNotification (agentbox emit -> VisionClaw ingest)
```mermaid
classDiagram
    class AgentActionNotification {
      +String jsonrpc
      +String method
      +AgentActionParams params
      is_canonical() bool
    }
    class AgentActionParams {
      +String kind
      +AgentActionEnvelope event
      +u8 message_type
      +u8 protocol_version
      +String timestamp
    }
    class AgentActionEnvelope {
      +u8 version
      +u64 id
      +u32 source_agent_id
      +u32 target_node_id
      +u8 action_type
      +String action_type_name
      +u64 timestamp
      +u32 duration_ms
      +Option~String~ source_urn
      +Option~String~ target_urn
      +Option~String~ pubkey
      +Option~u64~ token_count
      +Option~String~ handoff_id
      +Option~String~ verification
      +Option~String~ intent
      +Value metadata
      action_type() AgentActionType
      has_ctc() bool
      declared_intent() Option~str~
      to_binary_event() AgentActionEvent
    }
    class AgentActionEvent {
      +u32 source_agent_id
      +u32 target_node_id
      +u8 action_type
      +u32 timestamp
      +u16 duration_ms
      +List~u8~ payload
    }
    AgentActionNotification --> AgentActionParams : params
    AgentActionParams --> AgentActionEnvelope : event
    AgentActionEnvelope --> AgentActionEvent : to_binary_event
    note for AgentActionNotification "src/agent_events/schema.rs:26-30 struct; METHOD const :33;<br/>is_canonical requires jsonrpc 2.0, kind agent_action, version >= 3 at :36-41"
    note for AgentActionEnvelope "src/agent_events/schema.rs:60-140 mirrors agentbox<br/>management-api/utils/agent-event-publisher.js byte-for-field (ADR-059 par2)<br/>INVARIANT: canonical wire names (token_count/handoff_id/verification) come from agentbox;<br/>serde alias accepts the older draft spelling"
    note for AgentActionEvent "to_binary_event src/agent_events/schema.rs:173 is the identity-blind 0x23 projection:<br/>source_urn/target_urn/pubkey are dropped on purpose, already resolved server-side (ADR-059) :169-172"
```
## ES-03.2 VisionClaw urn:visionclaw grammar - 7 kinds, no agent kind
```mermaid
flowchart TD
    NS["NS = urn:visionclaw<br/>src/uri/mod.rs:41"]
    DID["did_nostr pubkey<br/>src/uri/mod.rs:220<br/>did:nostr plus hex-pubkey"]
    CONCEPT["concept domain, slug<br/>src/uri/mod.rs:229<br/>concept plus domain plus slug"]
    KG["kg owner_pubkey, content<br/>src/uri/mod.rs:242<br/>kg plus hex-pubkey plus sha256-12"]
    KGADDR["kg_with_address owner_pubkey, content_addr<br/>src/uri/mod.rs:255"]
    BEAD["bead owner_pubkey, content<br/>src/uri/mod.rs:265<br/>bead plus hex-pubkey plus sha256-12"]
    EXEC["execution content<br/>src/uri/mod.rs:277<br/>execution plus sha256-12, unscoped"]
    GROUP["group_members team<br/>src/uri/mod.rs:282<br/>group plus team plus members"]
    ROOM["room content<br/>src/uri/mod.rs:292<br/>room plus sha256-12, unscoped"]
    AVATAR["avatar pubkey<br/>src/uri/mod.rs:297<br/>avatar plus hex-pubkey, identity-bound 1:1 with DID"]
    NS --> DID
    NS --> CONCEPT
    NS --> KG
    KG -.-> KGADDR
    NS --> BEAD
    NS --> EXEC
    NS --> GROUP
    NS --> ROOM
    NS --> AVATAR
    ERR["UriError::MalformedUri"]
    KGADDR -- "require_content_address fails<br/>:169-174" --> ERR
    subgraph LEGACY["Legacy coexistence"]
        NGM["urn:ngm:* LEGACY_NGM_NS<br/>src/uri/mod.rs:46"]
        DUAL["parse_dual()<br/>src/uri/mod.rs:619<br/>resolve paths only"]
        NGM --> DUAL
    end
    NOTE1["INVARIANT: no urn:visionclaw:agent kind - identity IS the DID<br/>src/uri/mod.rs:26-27,51-52"]
    NOTE2["INVARIANT: every durable id minted only via these typed constructors<br/>ad-hoc format! is prohibited - src/uri/mod.rs:33-35"]
    NOTE3["DELIBERATE (2026-09-05, ADR-2061 review): parse rejects urn:ngm:* and parse_dual accepts it<br/>Not drift - a mint vs resolve split. parse refuses the retired namespace so no NEW durable id is minted under it<br/>parse_dual accepts it so ids persisted before the ADR-105 cutover keep resolving<br/>Collapsing them either strands legacy ids or re-opens minting - docs/IDENTIFIER-taxonomy.md invariant 7"]
    NOTE1 -.-> DID
    NOTE2 -.-> NS
    DUAL -.-> NOTE3
```
## ES-03.3 agentbox urn:agentbox grammar - 19 kinds (uris.js KINDS)
```mermaid
flowchart TD
    RE["URN_RE = urn:agentbox colon kind colon scope-or-local<br/>agentbox/management-api/lib/uris.js:114"]
    MINT["mint kind, pubkey, npub, payload, localId<br/>uris.js:157"]
    RE --> MINT
    subgraph CA["contentAddressed true - sha256-12 payload hash"]
        POD["pod - pods surface"]
        ENVELOPE["envelope - pods surface"]
        CRED["credential - pods surface"]
        MANDATE["mandate - pods surface"]
        RECEIPT["receipt - pods surface"]
        ACTIVITY["activity - agent-events surface"]
        EVENT["event - agent-events surface"]
        DECISION["decision - agent-events surface, ADR-048, ownerScope required"]
        BEAD["bead - beads surface, content-addressed to match urn:visionclaw:bead"]
    end
    subgraph SLUG["contentAddressed false - localId slug"]
        MCP["mcp - things, no owner scope"]
        MEMORY["memory - memory, ownerScope optional"]
        SKILL["skill - skills"]
        ADR["adr - docs"]
        PRD["prd - docs"]
        DDD["ddd - docs"]
        THING["thing - things, ownerScope optional"]
        DATASET["dataset - memory, ownerScope required"]
        AGENT["agent - agents, ownerScope optional"]
        META["meta - meta, no owner scope"]
    end
    MINT --> CA
    MINT --> SLUG
    NORM["_normalisePubkey supplied<br/>uris.js:203-227<br/>accepts hex, did:nostr hex, or npub1 bech32"]
    MINT -- "spec.ownerScope true" --> NORM
    NOTE_KIND["INVARIANT: 19 kinds total, decision added by ADR-048<br/>agentbox/CLAUDE.md Parallel namespace"]
    NOTE_R1["INVARIANT: R1 content-addressed local is sha256-12 plus first 12 hex chars, same input gives same URI<br/>uris.js:37-42"]
    RE -.-> NOTE_KIND
    CA -.-> NOTE_R1
```
## ES-03.4 cross_from_agentbox: shared kind policy and typed inbound targets
```mermaid
flowchart TD
    IN["cross_from_agentbox<br/>src/uri/mod.rs:809"] --> DID{"Valid did:nostr hex identity?<br/>src/uri/mod.rs:811-819"}
    DID -->|yes| SAME["Return identity unchanged with owner DID<br/>src/uri/mod.rs:813-816"]
    DID -->|malformed DID| NO["Refuse without inventing an identifier"]
    DID -->|URN| PARSE["Parse agentbox kind and optional hex scope<br/>src/uri/mod.rs:822-826"]
    PARSE --> ROW["Look up shared federation-kinds.json policy<br/>src/uri/mod.rs:832-834"]
    ROW -->|unknown or crossing disabled| NO
    ROW -->|crossing enabled| TARGET{"Typed target constructor<br/>src/uri/mod.rs:842"}
    TARGET --> ID["did:nostr: scoped identity<br/>src/uri/mod.rs:843"]
    TARGET --> EXEC["execution: hash original URN, unscoped<br/>src/uri/mod.rs:844"]
    TARGET --> KG["kg: hash original URN with owner scope<br/>src/uri/mod.rs:845"]
    TARGET --> BEAD["bead: preserve valid scoped content address<br/>src/uri/mod.rs:853-855"]
    TARGET -->|unsupported target| NO
    ID --> OUT["Return crossing with original identifier<br/>src/uri/mod.rs:860-863"]
    EXEC --> OUT
    KG --> OUT
    BEAD --> OUT
    NOTE["INVARIANT: policy comes from the shared artefact; constructors enforce target grammar.<br/>Memory elevation requires explicit domain and slug outside this hot path.<br/>Refusal and translation are narrower than resolved cross-service data."] --> ROW
```
## ES-03.5 bc20-provenance-bridge.js: forward policy and reversible identity recovery
```mermaid
flowchart TD
    FWD["toVisionclaw agentboxUrn, opts<br/>bc20-provenance-bridge.js:158"]
    PARSED["const parsed = uris.parse agentboxUrn, B02<br/>:160"]
    FWD --> PARSED
    PARSED -- "not urn scheme" --> DROP0["_countDrop unknown non-canonical, onDrop<br/>return null :161-164"]
    PARSED -- "kind agent" --> AGENTCHK{"pubkey present and PUBKEY_HEX_RE<br/>:169-170"}
    AGENTCHK -- "no" --> DROP1["_countDrop agent missing-scope<br/>return null :171-173"]
    AGENTCHK -- "yes" --> VCDID["vc = did:nostr plus pubkey<br/>:175"]
    PARSED -- "kind not agent" --> MAPLOOK["vcKind = AGENTBOX_TO_VISIONCLAW kind<br/>:180 - the map is DERIVED at :112-127 from the shared artefact, not written out<br/>schema/federation-kinds.json rows give activity to execution, thing to kg, memory to concept, bead to bead"]
    MAPLOOK -- "no mapping" --> DROP2["_countDrop kind unmapped-kind<br/>return null :181-184"]
    MAPLOOK -- "execution" --> VCEXEC["vc = urn:visionclaw:execution plus sha12 agentboxUrn<br/>:188-190"]
    MAPLOOK -- "bead" --> BEADCHK{"pubkey hex and local matches sha256-12 pattern"}
    BEADCHK -- "no" --> DROP3["_countDrop bead missing-scope or malformed-local<br/>return null :191-206"]
    BEADCHK -- "yes" --> VCBEAD["vc = urn:visionclaw:bead plus pubkey plus local, STRUCTURAL PASS-THROUGH<br/>:206"]
    MAPLOOK -- "kg" --> KGCHK{"pubkey present and hex"}
    KGCHK -- "no" --> DROP4["_countDrop kg missing-scope<br/>return null :207-213"]
    KGCHK -- "yes" --> VCKG["vc = urn:visionclaw:kg plus pubkey plus sha12 agentboxUrn<br/>:213"]
    MAPLOOK -- "concept" --> CONCEPTCHK{"opts.domain and opts.slug supplied"}
    CONCEPTCHK -- "no" --> DROP5["_countDrop concept missing-args<br/>return null :214-219"]
    CONCEPTCHK -- "yes" --> VCCONCEPT["vc = urn:visionclaw:concept plus slugify domain plus slugify slug<br/>:220"]
    VCDID --> RESULT["{visionclaw_id, mapping} plus _countCrossing<br/>:175-177,223-224"]
    VCEXEC --> RESULT
    VCBEAD --> RESULT
    VCKG --> RESULT
    VCCONCEPT --> RESULT
    NOTE_DIFF["RESOLVED ADR-2061 (2026-09-05): both translators now derive this map from one versioned artefact<br/>schema/federation-kinds.json v1.0.0 is read here at load and embedded in Rust with include_str!<br/>A paired fixture asserts per-kind agreement on crossed versus refused and on the target grammar<br/>Flipping one artefact row failed 3 of 50 jest cases and 4 of 7 cargo test cases - asymmetry is a test failure"]
    VCBEAD -.-> NOTE_DIFF
    subgraph REV["toAgentbox: source recovery, bc20-provenance-bridge.js:239"]
        REVDID{"Parsed did:nostr?<br/>bc20-provenance-bridge.js:243-244"}
        REVDID -->|yes| DIDSTORE["Return stored source if available;<br/>otherwise stable agent URI from pubkey<br/>bc20-provenance-bridge.js:249-253"]
        REVDID -->|no| VALID{"Known VisionClaw URN kind?<br/>bc20-provenance-bridge.js:256-266"}
        VALID -->|no| DROP["Count drop and return null"]
        VALID -->|yes| STORE{"Mapping store hit?<br/>bc20-provenance-bridge.js:268-272"}
        STORE -->|yes| HIT["Return original agentbox URN"]
        STORE -->|no| BEAD{"Valid scoped bead?<br/>bc20-provenance-bridge.js:275-285"}
        BEAD -->|yes| RECOVER["Preserve pubkey and existing content address<br/>bc20-provenance-bridge.js:281"]
        BEAD -->|no| DROP
        STORE -->|other kind without hit| MISS["Count store miss and return null<br/>bc20-provenance-bridge.js:287-289"]
    end
```
## ES-03.6 Content address byte-identity: sha256-12, first 6 bytes, lowercase hex (ADR-2023)
```mermaid
sequenceDiagram
    autonumber
    participant JSC as uris.js._contentAddress<br/>agentbox/management-api/lib/uris.js:281
    participant JSS as uris.js._stableStringify<br/>uris.js:292
    participant RSW as provenance_writer.rs::mint_assertion_version_urn<br/>src/services/provenance_writer.rs:301
    participant RSS as provenance_writer.rs::stable_stringify<br/>src/services/provenance_writer.rs:233
    participant RSC as provenance_writer.rs::content_address<br/>src/services/provenance_writer.rs:264
    Note over JSC,RSC: INVARIANT ADR-2023 - sha256-12 is SHA-256 truncated to first 6 bytes, 12 lowercase hex<br/>chars, byte-identical both sides
    JSC->>JSS: canon = _stableStringify(payload)
    Note right of JSS: sorted object keys, JSON.stringify primitives, no whitespace<br/>uris.js:292-297
    JSS-->>JSC: canon string
    JSC->>JSC: hex = sha256(canon).digest(hex).slice(0,12)
    JSC-->>JSC: sha256-12- + hex
    RSW->>RSS: canon = stable_stringify(&payload)
    Note right of RSS: same rule - keys sorted, ASCII keys so scalar sort matches JS UTF-16 order<br/>src/services/provenance_writer.rs:231-233
    RSS-->>RSW: canon string
    RSW->>RSC: content_address(&canon)
    RSC->>RSC: digest = Sha256::digest(canon.as_bytes)
    RSC-->>RSW: sha256-12- + hex::encode(digest[..6])
    Note over RSW: GOLDEN FIXTURE test entity_urn_matches_uris_js_golden<br/>src/services/provenance_writer.rs:623 (declared at :65) - identical payload gives<br/>urn:agentbox:event:aaa...:sha256-12-8c3913fd05a9 on both sides
    alt payload differs only by agent pubkey in scope
        RSW->>RSW: content_address_is_deterministic_and_agent_independent_in_hash<br/>:637-654 - same local hash, different scope segment
    end
    Note over JSC,RSC: DIVERGENCE - JS sha12 hashes the raw string via crypto.createHash on the URN or payload<br/>string, Rust src/uri/mod.rs content_address hashes caller-supplied bytes directly -<br/>equivalence depends on callers feeding byte-identical canonical input on both sides, not<br/>enforced by the type system
```
## ES-03.7 uris.js mint(): alt for every MalformedUri throw
```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant M as mint kind,pubkey,npub,payload,localId<br/>agentbox/management-api/lib/uris.js:157
    participant CA as _contentAddress<br/>uris.js:281
    participant SL as _slug<br/>uris.js:300
    participant NP as _normalisePubkey<br/>uris.js:203
    C->>M: mint({kind, ...})
    alt kind not in KINDS
        M-->>C: throw UnknownUriKind kind<br/>uris.js:158
    end
    M->>M: spec = KINDS[kind]
    alt spec.contentAddressed true
        alt payload is undefined
            M-->>C: throw MalformedUri content-addressed kind requires payload<br/>uris.js:164
        else payload present
            M->>CA: _contentAddress(payload)
            CA->>CA: canon = _stableStringify(payload)
            CA-->>M: local = sha256-12- + hex.slice(0,12)
        end
    else contentAddressed false
        alt localId supplied
            M->>SL: _slug(localId)
            SL-->>M: local, alnum plus dot underscore dash, max 96 chars
        else localId missing
            M-->>C: throw MalformedUri kind requires localId<br/>uris.js:170
        end
    end
    alt spec.ownerScope true
        alt no pubkey and no npub supplied
            alt spec.scopeRequired false
                M-->>C: return urn:agentbox: + kind + : + local, unscoped form<br/>uris.js:178
            else scopeRequired true (default)
                M-->>C: throw MalformedUri kind requires pubkey scope<br/>uris.js:180
            end
        else pubkey or npub supplied
            M->>NP: _normalisePubkey(supplied)
            alt already 64-hex
                NP-->>M: value unchanged
            else did:nostr prefix
                NP-->>M: strip did:nostr: prefix, validate hex
            else npub1 bech32
                NP->>NP: require nostr-tools, nip19.decode
                alt decoder unavailable or wrong type
                    NP-->>M: null
                else decoded ok
                    NP-->>M: hex pubkey
                end
            else unrecognised
                NP-->>M: null
            end
            alt normalised is null
                M-->>C: throw MalformedUri bad pubkey supplied<br/>:184
            else normalised ok
                M-->>C: return urn:agentbox: + kind + : + normalised + : + local<br/>:186
            end
        end
    else ownerScope false
        M-->>C: return urn:agentbox: + kind + : + local<br/>:189
    end
    Note over M: INVARIANT - fail-closed minting, a malformed input yields an error rather than a<br/>structurally-invalid identifier
```
## ES-03.8 URI resolver: redirect routing does not attest target availability
```mermaid
sequenceDiagram
    participant C as Client
    participant R as GET URI resolver<br/>uri-resolver.js:49
    participant U as URI grammar<br/>uris.js:261
    C->>R: request a canonical identifier
    R->>U: isCanonical then parse
    alt malformed
        R-->>C: 400 malformed-uri<br/>uri-resolver.js:53-58
    else did:nostr
        alt DID documents disabled
            R-->>C: 404 not-resolvable<br/>uri-resolver.js:66-74
        else enabled
            R-->>C: 307 pod well-known DID document<br/>uri-resolver.js:75
        end
    else operational URN
        R->>R: select known kind or return 404<br/>uri-resolver.js:80-84
        alt pod or envelope or credential or mandate or receipt
            R-->>C: scoped pod redirect, else 404<br/>uri-resolver.js:92-102
        else activity or event
            R-->>C: 307 agent-events query<br/>uri-resolver.js:108
        else mcp or thing
            R-->>C: 307 things path<br/>uri-resolver.js:113
        else memory or dataset
            R-->>C: 307 memory path and optional namespace<br/>uri-resolver.js:116-126
        else skill or document or meta or bead
            R-->>C: 307 kind-specific route<br/>uri-resolver.js:131-146
        else no mapped resolver
            R-->>C: 404 not-resolvable<br/>uri-resolver.js:149-155
        end
    end
    Note over R,C: DIVERGENCE: historical 410 language has no implemented 410 response.<br/>A 307 proves a route was selected, and the target may still be missing or refuse access.<br/>Deterministic naming does not guarantee collision-free or available data.
```
## ES-03.9 Adapter dispatch: observability -> privacy filter -> JSON-LD encoder, in that order
```mermaid
sequenceDiagram
    autonumber
    participant Rt as Route handler
    participant WD as wrapDispatch slot,impl,methodName,fn<br/>agentbox/management-api/observability/metrics.js:125
    participant PF as wrapWithPrivacyFilter<br/>agentbox/management-api/middleware/privacy-filter.js:649
    participant OPF as opf-router sidecar<br/>http 127.0.0.1:9092
    participant LD as LinkedDataEncoder.dispatch<br/>agentbox/management-api/middleware/linked-data/encoder.js:117
    participant AA as assertPrivacyFilterApplied<br/>privacy-filter.js:595
    participant Ad as Adapter impl call
    rect rgb(240,240,255)
    Note over Rt,Ad: DOC-DRIFT RESOLVED BY CORRECTION (ADR-2036) - the governing doc's own<br/>THREE-LAYER dispatch claim was the error, retracted at BASELINE-container.md:223.<br/>Observability and privacy wrap the dispatch (metrics.js:125, privacy-filter.js:649),<br/>JSON-LD encoding is a SEPARATE caller action, deliberately, not a third wrapper layer.
    Rt->>WD: instrumentedDispatch(...args)
    WD->>WD: executionId = uris.mint kind event,pubkey,payload
    WD->>PF: privacyWrapped(...args)
    alt OPF_MODE strict and sidecar unreachable
        PF->>OPF: redact payload
        OPF--xPF: connection error
        PF-->>WD: throw AdapterWriteRejected, fail-closed 503<br/>privacy-filter.js policy strict
    else OPF_MODE soft and sidecar unreachable
        PF->>OPF: redact payload
        OPF--xPF: connection error
        PF-->>WD: continue unredacted, fail-open, warn plus counter
    else OPF_MODE off
        PF->>PF: skip OPF entirely, pass-through
    else OPF reachable
        PF->>OPF: redact payload
        OPF-->>PF: redacted text, replaced count
    end
    PF->>PF: stamp WeakSet plus non-enumerable Symbol marker on payload<br/>privacy-filter.js:171 _hasPrivacyMark counterpart
    PF-->>WD: result
    WD->>WD: adapterDispatchTotal / adapterDurationSeconds metrics, structured log
    end
    opt route also passes payload to the JSON-LD surface
        Rt->>LD: dispatch({slot, operation, payload, adapterCall})
        LD->>AA: assertPrivacyFilterApplied(payload, slot, logger)
        alt payload carries the privacy mark
            AA-->>LD: pass, no-op
            LD->>LD: validatePayload input-validation limits
            LD->>LD: surface.encode(payload) -> JSON-LD document
            LD->>Ad: adapterCall(payload)
        else payload unmarked (route bypassed wrapWithPrivacyFilter)
            AA-->>LD: throw MiddlewareOrderViolation slot,payloadType<br/>privacy-filter.js:562-571
            LD->>LD: increment opf_middleware_order_violations_total
            alt slot is pods or memory (fail-closed slots)
                LD-->>Rt: throw, request fails
            else other slot
                LD-->>Rt: log and continue, fail-open
            end
        end
    end
    Note over LD: INVARIANT DDD-004 par L08 - privacy redaction completes before the encoder runs, verified<br/>per-dispatch not per-module-load
    Note over Ad: DIVERGENCE agentbox/docs/BASELINE-container.md - adapter contract versions are STALE<br/>PLACEHOLDERS. pods, memory, events and orchestrator all still declare 1.0.0 despite live<br/>churn, so a breaking change would need a MAJOR bump that has NOT happened. A consumer<br/>cannot tell from the version whether the contract it compiled against still holds.
    Note over Ad: CORRECTED (ADR-2035) - there is NO orchestrator-specific fatal probe. connectAdapters<br/>races EVERY slot against its OWN deadline (lifecycle.js:31), failure and timeout are<br/>equally fatal and both quarantine the adapter (lifecycle.js:256,274). If the off-replacement<br/>cannot be built the slot is left unavailable and dispatch throws AdapterQuarantined rather<br/>than reach a degraded adapter (lifecycle.js:292-299). toLegacyHealth maps all five slots<br/>uniformly into adapterHealth (server.js:1272), and /ready blocks on ANY manifest slot that<br/>is not healthy (server.js:485-486) - no slot is privileged.
```
## ES-03.10 Partial convergence: typed operational crossings and remaining RDF identity seams
```mermaid
flowchart TD
    subgraph VC["VisionClaw grammars in live code"]
        G1["urn:visionclaw:* operational URN<br/>src/uri/mod.rs:41 legacy ADR-105"]
        G2["vc:{domain}/{slug} semantic IRI, RDF display / JSON-LD predicate CURIE only<br/>legacy ADR-100"]
        G3["did:nostr:hex plus npub display<br/>src/uri/mod.rs:220"]
        G4["urn:ngm:class / property / axiom IRIs, still minted<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:20-25"]
        G5["visionclaw:owner:{npub}/kg/... NOT emitted anywhere in src/ or crates/<br/>legacy ADR-050, superseded by urn:visionclaw:kg pubkey scope"]
    end
    subgraph AB["agentbox grammars in live code"]
        G6["urn:agentbox:kind:scope:local hex-canonical<br/>agentbox/management-api/lib/uris.js:114 legacy ADR-053"]
        G7["did:nostr:hex identity, npub accepted at boundary only<br/>uris.js:203-227"]
    end
    G1 -. "cross_from_agentbox, ADR-2025 closed map" .-> G6
    G3 -. "structural round-trip, already converged" .-> G7
    NOTE1["PARTIAL CONVERGENCE: the shared federation-kinds.json governs operational URN crossings<br/>and both translators have typed refusal paths. RDF urn:ngm and semantic display<br/>identities remain separate seams; the historical visionclaw:owner form is not emitted.<br/>Distinct grammars alone are not evidence of a broken join; test each boundary."]
    NOTE2["DIVERGENCE docs/BASELINE-architecture.md par235 - identifier grammars unreconciled, DID doc ADR-074-D2prime vs ADR-125 conflict"]
    NOTE3["DELIBERATE REFUSAL: memory needs an explicit ontology elevation target;<br/>unknown kinds are refused. Preserve the raw identifier and refusal reason.<br/>This is a bounded translation contract, not proof that every identifier resolves."]
    G4 -.-> NOTE1
    G2 -.-> NOTE1
    G5 -.-> NOTE2
    G1 -.-> NOTE3
```


