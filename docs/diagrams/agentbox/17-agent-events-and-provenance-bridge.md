---
id: AB-17
title: Agent events and the BC20 provenance bridge
area: agentbox
governing:
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/INGRESS-identity.md
adrs: [ADR-2011, ADR-2022, ADR-2025, ADR-2061, ADR-2085]
sources:
  - ../project/agentbox/management-api/utils/agent-event-publisher.js
  - ../project/agentbox/management-api/routes/agent-events.js
  - ../project/agentbox/management-api/lib/agent-event-auth.js
  - ../project/agentbox/management-api/lib/bc20-provenance-bridge.js
  - ../project/agentbox/schema/federation-kinds.json
  - ../project/agentbox/management-api/lib/kg-proposal-extractor.js
  - ../project/agentbox/management-api/lib/memory-flash-notifier.js
  - ../project/agentbox/management-api/lib/failure-taxonomy.js
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/management-api/routes/kg-elevation.js
  - ../project/agentbox/management-api/lib/ontology-propose.js
  - ../project/agentbox/management-api/lib/elevation-stage.js
verified_commit: 6466e39313c3eb4ba0cadfc2efd4e7ffa3ccc296
---

## AB-17.1 The agent-event wire envelope — single canonical builder

```mermaid
classDiagram
    class AgentEventPublisher {
        <<agentbox/management-api/utils/agent-event-publisher.js:26 extends EventEmitter>>
        +Set subscribers
        +List~object~ eventBuffer
        +int maxBufferSize
        +int nextEventId
        +subscribe(callback) unsubscribe
        +emitAgentAction(event)
        +createMcpNotification(event)
    }
    class AgentActionType {
        <<frozen enum agent-event-publisher.js:13>>
        QUERY 0 blue
        UPDATE 1 yellow
        CREATE 2 green
        DELETE 3 red
        LINK 4 purple
        TRANSFORM 5 cyan
    }
    class WireEvent {
        <<params.event in createMcpNotification:305>>
        +int version
        +int id
        +string source_agent_id
        +string target_node_id
        +int action_type
        +string action_type_name
        +int timestamp
        +int duration_ms
        +string source_urn
        +string target_urn
        +string pubkey
        +string failure_mode
        +int token_count
        +string handoff_id
        +object verification
        +string authority_class
        +object metadata
    }
    class Notification {
        <<JSON-RPC 2.0 frame>>
        +string jsonrpc
        +string method
        +object params
        +int message_type
        +int protocol_version
        +string timestamp
    }
    AgentEventPublisher --> AgentActionType : numeric action_type, string name derived by reverse lookup
    AgentEventPublisher --> WireEvent : builds
    WireEvent --* Notification : params.event
    note for AgentEventPublisher "INVARIANT — createMcpNotification is the SINGLE canonical wire-envelope builder.<br/>Every transport (the /v1/agent-events/stream WebSocket and the deprecated MCP-TCP bridge) emits through here,<br/>so ADR-013 identity attribution (source_urn / target_urn / pubkey) is never dropped at the federation boundary.<br/>agentbox is the canonical schema source. VisionClaw mirrors this shape in src/agent_events/schema.rs. see ES-02"
    note for Notification "method notifications/agent_action. message_type 0x23 AGENT_ACTION for binary-frame parity.<br/>protocol_version 2. version 3 inside the event. eventBuffer is a ring capped at maxBufferSize 1000 — the oldest event is shift()ed out, so the buffer is NOT a durable record. see AB-14"
    note for WireEvent "BYTE-COMPATIBILITY DISCIPLINE — every additive field (failure_mode REC-5, token_count/handoff_id/verification REC-3 CTC,<br/>authority_class REC-6 AC4) is emitted as null when absent, so an existing success-only consumer sees an unchanged frame.<br/>DIVERGENCE — source_urn/target_urn/pubkey are OPTIONAL in Phase 1 and only become required under fail-closed attribution in Phase 5. Phase 5 is not here."
```

## AB-17.2 POST /v1/agent-events/emit — provable attribution

```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant RT as emit route<br/>agentbox/management-api/routes/agent-events.js:313
    participant AEA as verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:108
    participant RSU as reconcileSourceUrn<br/>agentbox/management-api/lib/agent-event-auth.js:169
    participant PUB as emitAgentAction<br/>agentbox/management-api/utils/agent-event-publisher.js:50
    participant TAX as failure-taxonomy classify<br/>agentbox/management-api/lib/failure-taxonomy.js
    participant BUF as eventBuffer ring (cap 1000)
    participant WS as stream subscribers

    C->>RT: POST /v1/agent-events/emit {source_agent_id, action_type, source_urn?, target_urn?, ...}
    RT->>AEA: verifyAgentEventRequest(request) (agent-events.js:365)
    alt auth.ok false
        AEA-->>RT: {ok false, status, error}
        RT-->>C: reply.code(auth.status) {success false, error, ...tag} (agent-events.js:371)
    end
    RT->>RSU: reconcileSourceUrn(claimed, auth.did) (agent-events.js:374)
    alt claimed source_urn does not match the verified identity
        RSU-->>RT: {ok false, status 403, error}
        RT-->>C: reply.code(403) {success false, error, ...tag} (agent-events.js:380)
        Note over RSU,RT: INVARIANT — source_urn becomes provable, derived from the SIGNATURE, not caller-asserted. see AB-11.14
    end
    RT->>PUB: emitAgentAction(event)
    PUB->>PUB: version 3, id = nextEventId++, timestamp = Date.now(), direction default 'outbound'
    alt action_type is a string
        PUB->>PUB: AgentActionType[action_type.toUpperCase()] or 0
    else numeric or absent
        PUB->>PUB: event.action_type or 0
    end
    opt caller supplied identity fields
        PUB->>PUB: forward source_urn / target_urn / pubkey verbatim (agent-event-publisher.js:70-72)
    end
    opt failure context present
        PUB->>TAX: taxonomy.classify(ctx)
        TAX-->>PUB: failure_mode
        PUB->>PUB: mirror the SAME tag into metadata.failure_mode
        Note over PUB,TAX: agent-event-publisher.js comment — "keep the mirror consistent with the canonical field so a consumer that reads either location sees the SAME tag (the disagreement is the bug)"
    end
    rect rgb(255,248,235)
    Note over PUB: ENVIRONMENT FALLBACK — attribution by default
    alt source_urn still empty
        PUB->>PUB: source_urn = AGENTBOX_URN or AGENTBOX_DID or null (agent-event-publisher.js:147-149)
    end
    alt pubkey still empty
        PUB->>PUB: pubkey = AGENTBOX_DID or null
    end
    end
    Note over PUB: DOC-DRIFT — ADR-2044 flipped the module default to 'nip98' (agent-event-auth.js:66) — 'off' is now an explicit dev-only opt-in, not the default.<br/>DIVERGENCE — when AGENTBOX_AGENT_EVENT_AUTH='off' is explicitly selected, this env fallback still stamps the CONTAINER identity<br/>on an event any caller could have raised. Attribution is then plausible, not provable. see AB-11.14
    PUB->>BUF: eventBuffer.push then shift() past 1000
    PUB->>WS: notify every subscriber via createMcpNotification. see AB-17.1
    RT-->>C: 2xx
    Note over RT,BUF: ADR-2026 — GET /v1/agent-events?id=<ref> for a BARE NUMERIC id stays process-local (resolved only against the in-memory ring, never the durable archive, because ids restart at 1 per process)<br/>a non-numeric urn reference additionally falls back to the durable agent-event-archive before returning 404 (agent-events.js:216-251). Route table for the whole surface: AB-03.9.
    Note over RT,AEA: POST /v1/agent-events/batch verifies auth ONCE for the whole batch (agent-events.js:469) but reruns reconcileSourceUrn PER ITEM inside the loop (agent-events.js:481), so a mixed batch cannot smuggle one mis-attributed event past a single header check.
```

## AB-17.4 BC20 kind map — the closed cross-namespace contract

```mermaid
classDiagram
    class FEDERATION_KINDS_json {
        <<versioned artefact 1.0.0, ADR-2061>>
        20 kind rows, 4 crossing
        agent to did:nostr identity
        activity to execution content-address
        thing to kg content-address
        bead to bead structural-passthrough
        memory to concept crosses false deliberate
        15 kinds crosses false not-federated
        knowledge added ADR-2085 federation-kinds.json:279
    }
    class AGENTBOX_TO_VISIONCLAW {
        <<frozen, DERIVED bc20-provenance-bridge.js:117-124>>
        fromEntries over kinds with target_kind
        excluding did:nostr - not a URN kind
        activity to execution
        thing to kg
        memory to concept
        bead to bead
    }
    class VISIONCLAW_TO_AGENTBOX {
        <<frozen, INVERTED bc20-provenance-bridge.js:125-127>>
        execution to activity
        kg to thing
        concept to memory
        bead to bead
    }
    class UrnMapping {
        <<returned by toVisionclaw>>
        +string agentbox_urn
        +string visionclaw_urn
        +string owner_did
    }
    class JsonlUrnMappingStore {
        <<durableStore bc20-provenance-bridge.js:346>>
        +string path
        +put(mapping)
        +get(id)
    }
    class InMemoryUrnMappingStore {
        <<reference and test store>>
        +Map _byAb
        +Map _byVc
    }
    FEDERATION_KINDS_json --> AGENTBOX_TO_VISIONCLAW : read at require time, derived not transcribed
    AGENTBOX_TO_VISIONCLAW <--> VISIONCLAW_TO_AGENTBOX : injective per owner_did
    UrnMapping --> JsonlUrnMappingStore : crossOutbound persists
    UrnMapping --> InMemoryUrnMappingStore : roundTrips proof helper
    note for AGENTBOX_TO_VISIONCLAW "There is deliberately NO agent kind in the map. An agent's identity IS its did:nostr,<br/>so urn:agentbox:agent:pubkey:name crosses as the BARE DID did:nostr:pubkey rather than a relabelled URN.<br/>bc20-provenance-bridge.js:14"
    note for FEDERATION_KINDS_json "INVARIANT: ADR-2061 — neither side transcribes the kind list. This bridge reads the artefact<br/>at require time (bc20-provenance-bridge.js:112-115); VisionClaw embeds the same bytes with<br/>include_str! in src/uri/mod.rs::cross_from_agentbox. A paired fixture<br/>(tests/contract/federation-kind-parity.contract.spec.js and federation_kind_artefact_matches_translator)<br/>makes a one-sided kind addition a test failure, not a runtime surprise. bc20-provenance-bridge.js:88-98<br/>INVARIANT ADR-2085: the colloquy knowledge kind is minted scoped and content-addressed<br/>(uris.js:114) and refused at the crossing (federation-kinds.json:279), so urn:agentbox:knowledge<br/>and the cq id ku_hex share the same twelve digest characters while the URN never crosses."
    note for VISIONCLAW_TO_AGENTBOX "B04 — the kind map is CLOSED. An unmapped kind is DROPPED and LOGGED, never silently mis-mapped.<br/>defaultLog writes '[bc20] drop: reason (urn)' to stderr (bc20-provenance-bridge.js:141,143)."
    note for UrnMapping "B01 — provenance is continuous, bidirectional and injective per owner_did.<br/>Where the VisionClaw kind is content-addressed (execution, kg) the local is a FRESH sha256-12 and the original<br/>urn:agentbox identity is recovered only from the durable UrnMapping store — lose the store, lose the crossing.<br/>Where it is identity-bearing (agent to did:nostr) the pubkey round-trips structurally with no store."
    note for JsonlUrnMappingStore "path = BC20_URN_MAPPING_PATH or /var/lib/agentbox/code-harness/bc20-urn-mappings.jsonl.<br/>DIVERGENCE PROTOCOL-registry 'Durable translation' row — the registry requires persistence, replay, round-trip and recovery receipts.<br/>roundTrips() proves the algebra against a FRESH IN-MEMORY store (bc20-provenance-bridge.js:307), which is not a durability or recovery proof."
```

## AB-17.5 toVisionclaw — outbound crossing and its drops

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as elevation or extractor
    participant TV as toVisionclaw<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:158
    participant U as uris.parse<br/>agentbox/management-api/lib/uris.js:266
    participant S12 as sha12<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:132
    participant SL as slugify<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:137
    participant DROP as onDrop or defaultLog
    participant ST as UrnMapping store

    CALLER->>TV: toVisionclaw(agentboxUrn, {domain, slug, onDrop})
    TV->>U: parse and validate through lib/uris.js
    Note over TV,U: B02 — agentbox URNs are parsed and validated through lib/uris.js. This module NEVER fabricates an ad-hoc urn:agentbox identifier. see AB-11.5
    alt non-canonical input
        TV->>DROP: drop(reason, urn) then return null
    end
    alt kind not in AGENTBOX_TO_VISIONCLAW
        TV->>DROP: B04 unmapped kind — drop and log, never mis-map
        TV-->>CALLER: null
    else activity
        TV->>S12: sha12(input) — sha256 over a UTF-8 string, first 12 hex
        S12-->>TV: sha256-12-<12hex>
        TV-->>CALLER: urn:visionclaw:execution:<sha256-12> plus mapping
    else thing
        alt no owner pubkey on the thing URN
            TV->>DROP: drop — bc20 drops an UNSCOPED thing
            Note over TV,DROP: consequence — the WS6 elevation path MUST mint its thing proposal WITH the owner pubkey or the crossing is lost. see AB-11.4
        else scoped
            TV->>S12: sha12
            TV-->>CALLER: urn:visionclaw:kg:<pubkey>:<sha256-12>
        end
    else memory
        alt opts.domain or opts.slug missing
            TV->>DROP: drop — memory to concept REQUIRES both
        else
            TV->>SL: slugify(slug) — lowercase, non [a-z0-9._-] to dash, trimmed
            TV-->>CALLER: urn:visionclaw:concept:<domain>:<slug>
        end
    else bead
        TV-->>CALLER: urn:visionclaw:bead:<pubkey>:<sha256-12> — PASS-THROUGH, local unchanged
        Note over TV,CALLER: both grammars are pubkey:sha256-12 now that agentbox beads are content-addressed,<br/>so content identity is preserved and the crossing round-trips with NO UrnMapping store (audit 2026-06-09 A3)
    end
    opt store supplied via crossOutbound (bc20-provenance-bridge.js:296)
        TV->>ST: store.put(mapping)
    end
    Note over TV: B03 — pure and synchronous. The fail-open posture lives at the NETWORK boundary (VisionClaw ingest), not here. This reference never calls a peer.
    Note over TV,ST: DIVERGENCE PROTOCOL-registry 'URN crossing' row — JS supports activity/thing/memory/bead plus the bare-DID agent case,<br/>the Rust side carries a NARROWER closed map. No versioned supported-kind agreement and no explicit unmapped-outcome contract exists. see ES-03
```

## AB-17.6 toAgentbox — inbound recovery

```mermaid
sequenceDiagram
    autonumber
    participant VC as VisionClaw identifier
    participant TA as toAgentbox<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:239
    participant RE as VC_URN_RE<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:130
    participant ST as UrnMapping store
    participant DROP as onDrop

    VC->>TA: toAgentbox(visionclawId, {store})
    alt input is a bare did:nostr:<64hex>
        TA-->>VC: urn:agentbox:agent:<pubkey>:… recovered STRUCTURALLY, no store needed
        Note over TA: the identity-bearing case — the pubkey round-trips on its own
    else matches VC_URN_RE ^urn:visionclaw:([a-z]+):(.+)$
        alt kind not in VISIONCLAW_TO_AGENTBOX
            TA->>DROP: B04 drop and log
            TA-->>VC: null
        else bead
            TA-->>VC: urn:agentbox:bead:<pubkey>:<sha256-12> — structural pass-through
        else execution or kg (content-addressed)
            TA->>ST: look up by visionclaw_urn
            alt mapping present
                ST-->>TA: agentbox_urn
                TA-->>VC: the ORIGINAL urn:agentbox identity
            else store missing the row
                TA-->>VC: null — the crossing is unrecoverable
                Note over TA,ST: DIVERGENCE — for execution and kg the local is a fresh sha256-12 with no algebraic inverse.<br/>Recovery is store-dependent, so store loss is identity loss. No recovery receipt exists for the durable store. see AB-17.4
            end
        end
    else neither
        TA-->>VC: null
    end
    Note over TA,ST: roundTrips(agentboxUrn) (bc20-provenance-bridge.js:307) is the B01 proof helper — cross out and back through a FRESH InMemoryUrnMappingStore,<br/>true iff the recovered URN equals the original. It proves the algebra, not the deployment.
    Note over ST: last-writer-wins on a VisionClaw id, which for did:nostr correctly collapses name variants of ONE identity — "injective per owner_did"
```

## AB-17.7 kg-proposal-extractor — lessons to a governed vault-propose command

```mermaid
sequenceDiagram
    autonumber
    participant SRC as memory entries / lessons
    participant EX as extractProposals<br/>agentbox/management-api/lib/kg-proposal-extractor.js:344
    participant NE as normaliseEntry / normaliseLesson<br/>agentbox/management-api/lib/kg-proposal-extractor.js:109,81
    participant SC as scoreCandidate<br/>agentbox/management-api/lib/kg-proposal-extractor.js:195
    participant BD as buildProposalDescriptor<br/>agentbox/management-api/lib/kg-proposal-extractor.js:233
    participant OWL as _owlClass<br/>agentbox/management-api/lib/kg-proposal-extractor.js:323
    participant U as uris.mint
    participant BC as bc20.toVisionclaw
    participant VP as buildVaultProposeCommand<br/>agentbox/management-api/lib/ontology-propose.js:214

    SRC->>EX: extractProposals(entries, opts)
    loop each entry
        EX->>NE: normaliseEntry then normaliseLesson
        NE->>NE: strip STOP words (kg-proposal-extractor.js:64)<br/>extract is_subclass_of and relationships from the raw value so they travel with the candidate (kg-proposal-extractor.js:151-159)
        NE-->>EX: normalised candidate
        EX->>SC: scoreCandidate(norm)
        SC-->>EX: score
        alt score < DEFAULT_MIN_SCORE 0.6 (kg-proposal-extractor.js:60)
            EX->>EX: reject — below the extraction floor
        else accepted
            EX->>BD: buildProposalDescriptor(norm, score, opts)
            BD->>OWL: _owlClass(term)
            BD->>U: uris.mint({kind thing, pubkey ownerPubkey, ...})
            Note over BD,U: the thing URN MUST carry the owner pubkey — bc20 drops an unscoped thing. see AB-17.5
            BD->>BC: toVisionclaw -> urn:visionclaw:kg:<pubkey>:<sha256-12>
            BD->>VP: buildVaultProposeCommand({action create, preferred_term, owl_class,<br/>is_subclass_of, relationships, ...}, env) (kg-proposal-extractor.js:268-277)
            VP-->>BD: {bin, argv, iri, level, hypothesis, dryRun, proposal, agentContext} (ontology-propose.js:249-257)
            BD-->>EX: descriptor with propose_command {argv, iri} — never propose_request
        end
    end
    EX-->>SRC: proposals
    Note over EX,VP: DOC-DRIFT resolved — ADR-2116 retired the /api/ontology-agent/propose HTTP descriptor.<br/>buildVaultProposeCommand is pure and synchronous: it shells to nothing, only asserts the `vault propose` argv (contract C2), which the route runs. see AB-17.10
    Note over EX,BC: INVARIANT ADR-2022 governed ontology writes — this path produces a PROPOSAL, not a write.<br/>direct_axiom_load = false keeps the remote direct-load descriptor disabled outside bootstrap.
    Note over EX: DIVERGENCE ADR-2022 — implementation is marked PARTIAL for the broad invariant. Forced-local dispatch can still edit authored Markdown BEFORE this guard,<br/>so local authoring and promotion need separately enforced authority and end-to-end receipts.
```

## AB-17.8 memory-flash-notifier — fail-open cross-repo notification

```mermaid
sequenceDiagram
    autonumber
    participant MEM as memory write path
    participant MF as notifyMemoryFlash<br/>agentbox/management-api/lib/memory-flash-notifier.js:68
    participant CFG as module-load config<br/>agentbox/management-api/lib/memory-flash-notifier.js:30-37
    participant PF as postFlash<br/>agentbox/management-api/lib/memory-flash-notifier.js:44
    participant VC as VisionClaw memory-flash endpoint

    Note over CFG: evaluated ONCE at module load — RAW_BASE from VISIONCLAW_MEMORY_FLASH_URL,<br/>FLASH_BASE = RAW_BASE with trailing slashes stripped,<br/>DISABLED when VISIONCLAW_MEMORY_FLASH is 'off',<br/>TIMEOUT_MS = VISIONCLAW_MEMORY_FLASH_TIMEOUT_MS or 1500,<br/>ENABLED = not DISABLED and FLASH_BASE non-empty and fetch is a function
    MEM->>MF: notifyMemoryFlash(flash)
    alt ENABLED false
        MF-->>MEM: no-op return — silently disabled
        Note over MF,MEM: three independent ways to be off, all silent — env 'off', an unset URL, or a runtime with no global fetch
    else enabled
        MF->>MF: logicalNamespace(namespace) (memory-flash-notifier.js:40)
        MF->>PF: postFlash(path, payload)
        PF->>VC: fetch with a TIMEOUT_MS budget
        alt response ok
            VC-->>PF: 2xx
        else error or timeout
            VC-->>PF: reject
            PF-->>MF: swallowed — FAIL-OPEN
            Note over PF,MF: the memory write already happened. This notification is advisory, so its loss is invisible to the writer and leaves the two repos' views divergent with no reconciliation path.
        end
    end
    MEM->>MF: notifyMemoryFlashBatch(flashes) (memory-flash-notifier.js:84)
    Note over MF,VC: DIVERGENCE — config is captured at MODULE LOAD, so changing VISIONCLAW_MEMORY_FLASH at runtime does nothing until the process restarts.<br/>Compare the proxy break-glass, captured the same way. see AB-16.9
```

## AB-17.10 End-to-end — a lesson becoming a VisionClaw KG proposal

```mermaid
sequenceDiagram
    autonumber
    participant AG as agent turn
    participant MEM as memory / lesson store
    participant EX as kg-proposal-extractor
    participant U as uris.mint
    participant BC as bc20.toVisionclaw
    participant ST as JsonlUrnMappingStore
    participant GATE as gateElevation<br/>agentbox/management-api/lib/elevation-stage.js:226
    participant EP as elevation-publisher
    participant PUB as agentEventPublisher
    participant VC as VisionClaw

    AG->>MEM: write a lesson
    MEM->>EX: extractProposals
    EX->>EX: normalise, score against the 0.6 floor,<br/>build propose_command {argv, iri}. see AB-17.7
    EX->>U: mint urn:agentbox:thing:<ownerPubkey>:proposal-<id>
    U-->>EX: scoped thing URN
    EX->>BC: toVisionclaw(thingUrn)
    BC->>BC: sha12 over the canonical payload
    BC-->>EX: urn:visionclaw:kg:<pubkey>:<sha256-12> plus UrnMapping
    BC->>ST: crossOutbound persists the mapping — the ONLY route back. see AB-17.6
    Note over EX: DIVERGENCE — routes/kg-elevation.js resolves ownerPubkey as auth.pubkey or<br/>AGENTBOX_X_ONLY_PUBKEY_HEX or AGENTBOX_PUBKEY (kg-elevation.js:162), so an elevation<br/>can be attributed to the operator identity when the request carries no verified<br/>pubkey — same shape as AB-17.2's env fallback.
    EX->>GATE: gateElevation(proposal, {env}) — kg-elevation.js:182,<br/>BEFORE anything is emitted or federated
    GATE->>GATE: stage proposal as a page, validate standalone,<br/>run vault propose --diff --dry-run. see AB-03.18
    alt blocked — stage failure or vault propose blockers
        GATE-->>EX: {proposable false, blocked true, blockers, stage_issues}
        Note over EX,EP: ADR-2116 — a blocked candidate keeps its observability beam<br/>(the scan found it, that is true)<br/>but gets NO signed 31402: elevationPublisher.publish is skipped for this candidate
    else proposable
        GATE-->>EX: {proposal PatchProposal, blocked false}
        EX->>EP: elevate — publish(proposal) carries propose_command.argv/iri<br/>and patch_proposal. see AB-14.13
        EP->>VC: signed elevation event
        Note over EP: elevation-publisher and the agent-control-surface authority gate share a<br/>producer-plus-injected-consumer dependency shape deliberately, testable without a<br/>live relay (see AB-11.10)<br/>the full eligibility branches (gate-off / no-relays / no-signing-stack) and the<br/>propose_command/propose_iri/patch_proposal fields are in AB-14.13.
    end
    par observability
        EX->>PUB: emitAgentAction {action_type CREATE, source_urn thingUrn,<br/>target_urn visionclaw kg urn}
        PUB->>PUB: createMcpNotification — both URNs travel on the wire,<br/>whether or not the candidate was blocked. see AB-17.1
    end
    VC-->>EP: governance decision (asynchronous, proposable candidates only).<br/>see ES-05 and AB-14
    Note over AG,VC: INVARIANT ADR-2025 — the crossing is the ONLY sanctioned way an agentbox<br/>identifier becomes a VisionClaw one.<br/>bc20-provenance-bridge is the ONLY module importing the urn:visionclaw grammar (B05) —<br/>every other aggregate speaks typed urn:agentbox value objects.
    Note over BC,ST: DIVERGENCE PROTOCOL-registry — all four contract rows are open.<br/>Content address: JS sha12 hashes a UTF-8 string, VisionClaw content_address hashes<br/>bytes, byte-parity asserted nowhere.<br/>URN crossing: no versioned supported-kind agreement. Precomputed KG address: the Rust<br/>constructor checks only the PREFIX, not the full grammar.<br/>Durable translation: helpers return a mapping, but persistence, replay, round-trip<br/>and recovery receipts do not exist.
```
