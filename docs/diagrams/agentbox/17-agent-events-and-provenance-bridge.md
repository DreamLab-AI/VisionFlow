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
  - ../project/agentbox/management-api/utils/agent-event-ws-subscriber.js
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
  - ../project/agentbox/management-api/lib/elevation-publisher.js
  - ../project/agentbox/docs/PROTOCOL-registry.md
verified_commit: c57d9c0508f4
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
        <<plain object, not frozen, agent-event-publisher.js:17>>
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
    note for AgentEventPublisher "INVARIANT — createMcpNotification is the SINGLE canonical wire-envelope builder (agent-event-publisher.js:296-303).<br/>The /v1/agent-events/stream WebSocket and the deprecated MCP-TCP bridge emit through here,<br/>so ADR-013 identity attribution (source_urn / target_urn / pubkey) is kept on those paths.<br/>DRIFT - the live outbound forwarder does NOT: it sends the bare event (agent-event-ws-subscriber.js:147),<br/>which VisionClaw ingest rejects as Malformed. see ES-02, finding W2-01<br/>agentbox is the canonical schema source. VisionClaw mirrors this shape in src/agent_events/schema.rs, PARTIALLY:<br/>that struct has no failure_mode or authority_class field. see ES-02"
    note for Notification "method notifications/agent_action. message_type 0x23 AGENT_ACTION for binary-frame parity (agent-event-publisher.js:338).<br/>protocol_version 2. version 3 inside the event. eventBuffer is a ring capped at maxBufferSize 1000 — the oldest event is shift()ed out,<br/>but every event is first appended to the durable agentEventArchive, fail-open (agent-event-publisher.js:154-160). see AB-14"
    note for WireEvent "BYTE-COMPATIBILITY DISCIPLINE — every additive field (failure_mode REC-5, token_count/handoff_id/verification REC-3 CTC,<br/>authority_class REC-6 AC4) is emitted as null when absent, so an existing success-only consumer sees an unchanged frame.<br/>DIVERGENCE — source_urn/target_urn/pubkey are OPTIONAL in Phase 1 and only become required under fail-closed attribution in Phase 5. Phase 5 is not here."
```

**Drift:** the publisher comment says VisionClaw mirrors this envelope in `src/agent_events/schema.rs` (`agentbox/management-api/utils/agent-event-publisher.js:296-298`), but at VisionClaw `86cdd7635582` `AgentActionEnvelope` carries `source_urn`, `target_urn`, `pubkey`, `token_count`, `handoff_id` and `verification` and no `failure_mode` or `authority_class` field, so serde drops those two wire fields on ingest. The mirror covers the identity and CTC fields only (`agentbox/management-api/utils/agent-event-publisher.js:326`, `agentbox/management-api/utils/agent-event-publisher.js:335`).

## AB-17.2 POST /v1/agent-events/emit — provable attribution

```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant RT as emit route<br/>agentbox/management-api/routes/agent-events.js:320
    participant AEA as verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:135
    participant RTS as reconcileTaskScope<br/>agentbox/management-api/lib/agent-event-auth.js:224
    participant RSU as reconcileSourceUrn<br/>agentbox/management-api/lib/agent-event-auth.js:204
    participant PUB as emitAgentAction<br/>agentbox/management-api/utils/agent-event-publisher.js:50
    participant TAX as failure-taxonomy classify<br/>agentbox/management-api/lib/failure-taxonomy.js
    participant BUF as eventBuffer ring (cap 1000) and archive
    participant WS as stream subscribers

    C->>RT: POST /v1/agent-events/emit {source_agent_id, target_node_id, action_type, ...}
    RT->>AEA: verifyAgentEventRequest(request) (agent-events.js:377)
    Note over AEA: a task-token request returns ok with did null and its taskId, under every policy (agent-event-auth.js:143-145).<br/>policy 'off' returns ok with did null (agent-event-auth.js:147).
    Note over AEA: INVARIANT: the proxy-stamped X-Agentbox-Pubkey branch counts only from a loopback TCP peer, peerIsProxy at<br/>agent-event-auth.js:98-105, checked first in proxyVerifiedPubkey at agent-event-auth.js:117, called at<br/>agent-event-auth.js:192, see AB-11.14
    alt auth.ok false
        AEA-->>RT: {ok false, status, error}
        RT-->>C: reply.code(auth.status) {success false, error, ...tag} (agent-events.js:383)
    end
    RT->>RTS: reconcileTaskScope(auth.taskId, body) (agent-events.js:385)
    alt task token names another task or asserts source_urn or pubkey
        RTS-->>RT: {ok false, status 403}
        RT-->>C: 403 {success false, error, ...tag}
    end
    RT->>RSU: reconcileSourceUrn(claimed, auth.did) (agent-events.js:391)
    alt claimed source_urn does not match the verified identity
        RSU-->>RT: {ok false, status 403, error}
        RT-->>C: reply.code(403) {success false, error, ...tag} (agent-events.js:397)
        Note over RSU,RT: INVARIANT — source_urn becomes provable, derived from the SIGNATURE, not caller-asserted. see AB-11.14
    end
    RT->>RT: when auth.did is set, stamp source_urn = auth.did and pubkey = auth.pubkey (agent-events.js:427-430).<br/>The route never copies the body's source_urn or target_urn into the payload.
    RT->>PUB: emitAgentAction(emitPayload) (agent-events.js:441)
    PUB->>PUB: version 3, id = nextEventId++, timestamp = Date.now(), direction default 'outbound'
    alt action_type is a string
        PUB->>PUB: AgentActionType[action_type.toUpperCase()] or 0
    else numeric or absent
        PUB->>PUB: event.action_type or 0
    end
    opt caller supplied identity fields
        PUB->>PUB: forward source_urn / target_urn / pubkey verbatim (agent-event-publisher.js:70-72)
    end
    opt failure signalled (outcome, failure context, or a failure_mode tag)
        PUB->>PUB: a valid top-level tag wins, else promote metadata.failure_mode (agent-event-publisher.js:120-124)
        PUB->>TAX: otherwise taxonomy.classify(ctx) (agent-event-publisher.js:129)
        TAX-->>PUB: failure_mode
        PUB->>PUB: mirror the SAME tag into metadata.failure_mode (agent-event-publisher.js:131-135)
    end
    rect rgb(255,248,235)
    Note over PUB: ENVIRONMENT FALLBACK — attribution by default, ADR-2042
    alt source_urn AND pubkey both empty
        PUB->>PUB: source_urn = AGENTBOX_URN or AGENTBOX_DID or null,<br/>pubkey = AGENTBOX_DID or null, set together (agent-event-publisher.js:146-151)
    end
    end
    Note over PUB: DOC-DRIFT — ADR-2044 flipped the module default to 'nip98' (agent-event-auth.js:66) — 'off' is now an explicit<br/>dev-only opt-in, not the default.<br/>DIVERGENCE — when 'off' is explicitly selected, or the caller holds a task token (did null), this env fallback stamps<br/>the CONTAINER identity on the event. Attribution is then plausible, not provable. see AB-11.14
    PUB->>BUF: eventBuffer.push, agentEventArchive.append (fail-open), then shift() past 1000 (agent-event-publisher.js:154-160)
    PUB->>WS: notify every subscriber with the event. see AB-17.1
    RT-->>C: 200 {success true, event_id, broadcast_count}
    Note over RT,BUF: ADR-2026 — GET /v1/agent-events?id=<ref> for a BARE NUMERIC id stays process-local (agent-events.js:217-224):<br/>resolved only against the in-memory ring, never the durable archive, because ids restart at 1 per process.<br/>A non-numeric urn reference also searches the durable agent-event archive before returning 404<br/>(agent-events.js:231-246). Route table for the whole surface: AB-03.9.
    Note over RT,AEA: POST /v1/agent-events/batch verifies auth ONCE for the whole batch (agent-events.js:487), checks task scope<br/>for EVERY item before emitting any (agent-events.js:498-504), then reruns reconcileSourceUrn PER ITEM inside the<br/>emit loop (agent-events.js:509), so one mis-attributed item is never emitted.
```

**Invariant:** the proxy-stamped `X-Agentbox-Pubkey` is read only when the TCP peer is the proxy: `proxyVerifiedPubkey` returns null unless `peerIsProxy` holds, which accepts `127.0.0.1`, `::1` and `::ffff:127.0.0.1` or the exact addresses in `AGENTBOX_PROXY_PEERS`, and refuses a request with no socket information (`agentbox/management-api/lib/agent-event-auth.js:98-105`, `agentbox/management-api/lib/agent-event-auth.js:117`). A `MANAGEMENT_API_KEY` holder on the docker network reaches the port because `server.js` binds `0.0.0.0`, but its forged header is ignored, so `reconcileSourceUrn` reconciles against an identity a cross-container caller cannot assert (`agentbox/management-api/lib/agent-event-auth.js:34-39`, `agentbox/management-api/lib/agent-event-auth.js:192-195`).

**Debt:** a process inside the container can still forge it over loopback (the same-UID boundary, TODO-unified X-1), so under path 2 provenance is bounded by that boundary (`agentbox/management-api/lib/agent-event-auth.js:37-39`).

**Debt:** the batch route rejects a mismatched item from inside the emit loop, after the items before it have already been emitted (`agentbox/management-api/routes/agent-events.js:506-519`, `agentbox/management-api/routes/agent-events.js:547`). The caller gets a 403 for a batch that was partly published, and the reply carries no ids for the events that went out.

**Tension:** a task-token request passes verification with `did` null (`agentbox/management-api/lib/agent-event-auth.js:143-145`), `reconcileTaskScope` forbids it from asserting an identity (`agentbox/management-api/lib/agent-event-auth.js:232-234`), and the route stamps nothing (`agentbox/management-api/routes/agent-events.js:427-430`), so the publisher's environment fallback attributes every task-token action to the container identity (`agentbox/management-api/utils/agent-event-publisher.js:146-151`). The task travels only in `metadata.taskId`.

## AB-17.4 BC20 kind map — the closed cross-namespace contract

```mermaid
classDiagram
    class FEDERATION_KINDS_json {
        <<versioned artefact 1.1.0, ADR-2061, federation-kinds.json:3>>
        21 kind rows, 4 crossing
        agent to did:nostr identity
        activity to execution content-address
        thing to kg content-address
        bead to bead structural-passthrough
        memory to concept crosses false deliberate
        16 kinds crosses false not-federated
        knowledge added ADR-2085 federation-kinds.json:279
        chain added ADR-2098 federation-kinds.json:291
        event_kinds band added ADR-2105
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
        +getByAgentbox(urn)
        +getByVisionclaw(id)
    }
    class InMemoryUrnMappingStore {
        <<reference and test store>>
        +Map _byAb
        +Map _byVc
    }
    FEDERATION_KINDS_json --> AGENTBOX_TO_VISIONCLAW : read at require time, derived not transcribed
    AGENTBOX_TO_VISIONCLAW <--> VISIONCLAW_TO_AGENTBOX : injective per owner_did
    UrnMapping --> JsonlUrnMappingStore : crossOutbound or elevation-publisher persists
    UrnMapping --> InMemoryUrnMappingStore : roundTrips proof helper
    note for AGENTBOX_TO_VISIONCLAW "There is deliberately NO agent kind in the map. An agent's identity IS its did:nostr,<br/>so urn:agentbox:agent:pubkey:name crosses as the BARE DID did:nostr:pubkey rather than a relabelled URN.<br/>bc20-provenance-bridge.js:14"
    note for FEDERATION_KINDS_json "INVARIANT: ADR-2061 — neither side transcribes the kind list. This bridge reads the artefact<br/>at require time (bc20-provenance-bridge.js:112-115); VisionClaw embeds the same bytes with<br/>include_str! in src/uri/mod.rs and derives cross_from_agentbox from it. A paired fixture<br/>(tests/contract/federation-kind-parity.contract.spec.js and federation_kind_artefact_matches_translator)<br/>makes a one-sided kind addition a test failure, not a runtime surprise. bc20-provenance-bridge.js:88-98<br/>INVARIANT ADR-2085: the colloquy knowledge kind is minted scoped and content-addressed<br/>(uris.js:111-115) and refused at the crossing (federation-kinds.json:279), so urn:agentbox:knowledge<br/>and the cq id ku_hex share the same twelve digest characters while the URN never crosses."
    note for VISIONCLAW_TO_AGENTBOX "B04 — the kind map is CLOSED. An unmapped kind is DROPPED and LOGGED, never silently mis-mapped.<br/>defaultLog writes '[bc20] drop: reason (urn)' to stderr (bc20-provenance-bridge.js:141,143)."
    note for UrnMapping "B01 — provenance is continuous, bidirectional and injective per owner_did.<br/>Where the VisionClaw kind is content-addressed (execution, kg) the local is a FRESH sha256-12 and the original<br/>urn:agentbox identity is recovered only from the durable UrnMapping store — lose the store, lose the crossing.<br/>Where it is identity-bearing (agent to did:nostr) the pubkey round-trips structurally with no store."
    note for JsonlUrnMappingStore "path = BC20_URN_MAPPING_PATH or /var/lib/agentbox/code-harness/bc20-urn-mappings.jsonl (bc20-provenance-bridge.js:380-387).<br/>A failed append degrades to in-memory only with a stderr line (bc20-provenance-bridge.js:359-368).<br/>DIVERGENCE PROTOCOL-registry 'Durable translation' row — the registry requires persistence, replay, round-trip and recovery receipts.<br/>roundTrips() proves the algebra against a FRESH IN-MEMORY store (bc20-provenance-bridge.js:307), which is not a durability or recovery proof."
```

## AB-17.5 toVisionclaw — outbound crossing and its drops

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as elevation, extractor or receipt-minter
    participant TV as toVisionclaw<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:158
    participant U as uris.parse<br/>agentbox/management-api/lib/uris.js:296
    participant S12 as sha12<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:132
    participant SL as slugify<br/>agentbox/management-api/lib/bc20-provenance-bridge.js:137
    participant DROP as onDrop or defaultLog
    participant ST as UrnMapping store

    CALLER->>TV: toVisionclaw(agentboxUrn, {domain, slug, onDrop})
    TV->>U: parse and validate through lib/uris.js (bc20-provenance-bridge.js:160)
    Note over TV,U: B02 — agentbox URNs are parsed and validated through lib/uris.js. This module NEVER fabricates an ad-hoc urn:agentbox identifier. see AB-11.5
    alt non-canonical input
        TV->>DROP: drop(reason, urn) then return null
    end
    alt agent kind
        alt no 64-hex owner pubkey
            TV->>DROP: drop — agent crossing needs a pubkey scope
        else
            TV-->>CALLER: did:nostr:<pubkey> plus mapping (bc20-provenance-bridge.js:169-178)
        end
    else kind not in AGENTBOX_TO_VISIONCLAW
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
            TV->>SL: slugify(domain) and slugify(slug) — lowercase, non [a-z0-9._-] to dash, trimmed
            TV-->>CALLER: urn:visionclaw:concept:<domain>:<slug>
        end
    else bead
        alt no 64-hex pubkey, or the local is not sha256-12
            TV->>DROP: drop — bead needs a scope and a content-addressed local (bc20-provenance-bridge.js:196-205)
        else
            TV-->>CALLER: urn:visionclaw:bead:<pubkey>:<sha256-12> — PASS-THROUGH, local unchanged
            Note over TV,CALLER: both grammars are pubkey:sha256-12 now that agentbox beads are content-addressed,<br/>so content identity is preserved and the crossing round-trips with NO UrnMapping store (audit 2026-06-09 A3)
        end
    end
    opt store supplied via crossOutbound (bc20-provenance-bridge.js:296)
        TV->>ST: store.put(mapping)
    end
    Note over TV: B03 — pure and synchronous. The fail-open posture lives at the NETWORK boundary (VisionClaw ingest), not here. This reference never calls a peer.
    Note over TV,ST: CONVERGED at the pins — VisionClaw cross_from_agentbox derives its map from the same artefact and has a bead arm,<br/>so both sides cross agent, activity, thing and bead. Only the JS side elevates memory, and only with explicit {domain, slug}.
```

**Drift:** the PROTOCOL-registry "URN crossing" row still says the Rust side has a narrower closed map with no `bead` arm and calls ADR-2061 planned (`agentbox/docs/PROTOCOL-registry.md:8`). The artefact is versioned and shared (`agentbox/schema/federation-kinds.json:3`), and at VisionClaw `86cdd7635582` `cross_from_agentbox` embeds it with `include_str!` and carries a structural `bead` arm, so the row's current-state column is stale.

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
        opt store holds the exact source URN
            ST-->>TA: agentbox_urn with its original name
        end
        TA-->>VC: otherwise urn:agentbox:agent:<pubkey>:_ (bc20-provenance-bridge.js:243-253)
        Note over TA: the identity-bearing case — the pubkey round-trips on its own. The agent NAME does not: without the store it becomes '_'.
    else matches VC_URN_RE ^urn:visionclaw:([a-z]+):(.+)$
        alt kind not in VISIONCLAW_TO_AGENTBOX
            TA->>DROP: B04 drop and log
            TA-->>VC: null
        else store supplied and holds the visionclaw_urn
            ST-->>TA: agentbox_urn (bc20-provenance-bridge.js:268-274)
            TA-->>VC: the ORIGINAL urn:agentbox identity
        else bead
            TA-->>VC: urn:agentbox:bead:<pubkey>:<sha256-12> — structural pass-through (bc20-provenance-bridge.js:275-286)
        else execution, kg or concept with no store row
            TA->>DROP: store-miss drop
            TA-->>VC: null — the crossing is unrecoverable
            Note over TA,ST: DIVERGENCE — for execution and kg the local is a fresh sha256-12 with no algebraic inverse, and a concept slug has none either.<br/>Recovery is store-dependent, so store loss is identity loss. No recovery receipt exists for the durable store. see AB-17.4
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
        NE->>NE: extract is_subclass_of and relationships from the raw value<br/>so they travel with the candidate (kg-proposal-extractor.js:151-159)
        NE-->>EX: normalised candidate
        EX->>SC: scoreCandidate(norm) — STOP words ignored when counting<br/>distinct content words (kg-proposal-extractor.js:64)
        SC-->>EX: score
        alt score < minScore, default DEFAULT_MIN_SCORE 0.6 (kg-proposal-extractor.js:60)
            EX->>EX: reject — below the extraction floor
        else accepted
            EX->>BD: buildProposalDescriptor(norm, score, opts)
            BD->>BD: ownerPubkey = opts.ownerPubkey or AGENTBOX_X_ONLY_PUBKEY_HEX<br/>or AGENTBOX_PUBKEY, else ExtractError (kg-proposal-extractor.js:235-241)
            BD->>U: uris.mint({kind thing, pubkey ownerPubkey, localId proposal-sha12})
            Note over BD,U: the thing URN MUST carry the owner pubkey — bc20 drops an unscoped thing. see AB-17.5
            BD->>BC: toVisionclaw(proposal_urn) -> proposal_foreign_urn<br/>urn:visionclaw:kg:<pubkey>:<sha256-12> (kg-proposal-extractor.js:253)
            BD->>BC: toVisionclaw(memory concept URN, {domain, slug}) -> target_urn<br/>urn:visionclaw:concept:<domain>:<slug> (kg-proposal-extractor.js:259-262)
            BD->>OWL: _owlClass(term)
            BD->>VP: buildVaultProposeCommand({action create, preferred_term, owl_class,<br/>is_subclass_of, relationships, ...}, env) (kg-proposal-extractor.js:268-279)
            VP-->>BD: {bin, argv, iri, level, hypothesis, dryRun, proposal, agentContext} (ontology-propose.js:247-256)
            BD-->>EX: descriptor with propose_command and a LINK emit payload<br/>(kg-proposal-extractor.js:284-288) — never propose_request
        end
    end
    EX-->>SRC: proposals
    Note over EX,VP: DOC-DRIFT resolved — ADR-2116 retired the /api/ontology-agent/propose HTTP descriptor.<br/>buildVaultProposeCommand is pure and synchronous: it shells to nothing,<br/>only asserts the `vault propose` argv (contract C2), which the route runs. see AB-17.10
    Note over EX,BC: INVARIANT ADR-2022 governed ontology writes — this path produces a PROPOSAL, not a write.<br/>The argv carries --dry-run unless post is true (ontology-propose.js:237,244).
    Note over EX: DIVERGENCE ADR-2022 — implementation is marked PARTIAL for the broad invariant.<br/>Forced-local dispatch can still edit authored Markdown BEFORE this guard,<br/>so local authoring and promotion need separately enforced authority and end-to-end receipts.
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

    Note over CFG: evaluated ONCE at module load — RAW_BASE from VISIONCLAW_MEMORY_FLASH_URL, else VISIONCLAW_API_URL,<br/>FLASH_BASE = RAW_BASE with trailing slashes stripped,<br/>DISABLED when VISIONCLAW_MEMORY_FLASH is 'off',<br/>TIMEOUT_MS = VISIONCLAW_MEMORY_FLASH_TIMEOUT_MS or 1500,<br/>ENABLED = not DISABLED and FLASH_BASE non-empty and fetch is a function
    MEM->>MF: notifyMemoryFlash(flash)
    alt ENABLED false, or the flash has no key
        MF-->>MEM: no-op return — silently disabled
        Note over MF,MEM: three independent ways to be off, all silent — env 'off', both URLs unset, or a runtime with no global fetch
    else enabled
        MF->>MF: logicalNamespace(namespace) (memory-flash-notifier.js:40)
        MF->>PF: postFlash('/api/memory-flash', payload)
        PF->>VC: fetch with a TIMEOUT_MS abort budget, not awaited by the caller
        alt network error or abort
            VC-->>PF: reject
            PF-->>MF: swallowed — FAIL-OPEN (memory-flash-notifier.js:60)
            Note over PF,MF: the memory write already happened. This notification is advisory, so its loss is invisible to the writer and leaves the two repos' views divergent with no reconciliation path.<br/>A non-2xx response is not inspected either.
        end
    end
    MEM->>MF: notifyMemoryFlashBatch(flashes) (memory-flash-notifier.js:84) — one event goes to /api/memory-flash, several to /api/memory-flash/batch
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
    participant GATE as gateElevation<br/>agentbox/management-api/lib/elevation-stage.js:226
    participant EP as elevation-publisher<br/>agentbox/management-api/lib/elevation-publisher.js:152
    participant ST as JsonlUrnMappingStore
    participant PUB as agentEventPublisher
    participant VC as VisionClaw

    AG->>MEM: write a lesson
    MEM->>EX: extractProposals
    EX->>EX: normalise, score against the 0.6 floor,<br/>build propose_command {argv, iri}. see AB-17.7
    EX->>U: mint urn:agentbox:thing:<ownerPubkey>:proposal-<sha12>
    U-->>EX: scoped thing URN
    EX->>BC: toVisionclaw(thingUrn) — plain toVisionclaw, NOT crossOutbound, so nothing is stored here
    BC->>BC: sha12 over the agentbox URN string
    BC-->>EX: urn:visionclaw:kg:<pubkey>:<sha256-12> as proposal_foreign_urn
    Note over EX: DIVERGENCE — routes/kg-elevation.js resolves ownerPubkey as auth.pubkey or<br/>AGENTBOX_X_ONLY_PUBKEY_HEX or AGENTBOX_PUBKEY (kg-elevation.js:162), so an elevation<br/>can be attributed to the operator identity when the request carries no verified<br/>pubkey — same shape as AB-17.2's env fallback.
    EX->>GATE: gateElevation(proposal, {env}) — kg-elevation.js:182,<br/>BEFORE anything is emitted or federated
    GATE->>GATE: stage proposal as a page, validate standalone,<br/>run vault propose --diff --dry-run. see AB-03.18
    par observability, when the request's emit flag is true (kg-elevation.js:202)
        EX->>PUB: emitAgentAction {action_type LINK, source/target ids hashed from the URNs,<br/>target_urn = the concept URN, source_urn = auth.did when verified} (kg-elevation.js:204-217)
        PUB->>PUB: proposal_urn and proposal_foreign_urn travel in metadata,<br/>whether or not the candidate was blocked. see AB-17.1
    end
    alt blocked — stage failure or vault propose blockers
        GATE-->>EX: {proposable false, blocked true, blockers, stage_issues}
        Note over EX,EP: ADR-2116 — a blocked candidate keeps its observability beam<br/>(the scan found it, that is true)<br/>but gets NO signed 31402: elevationPublisher.publish is skipped (kg-elevation.js:229-233)
    else proposable
        GATE-->>EX: {proposal PatchProposal, blocked false}
        EX->>EP: publish(proposal) carries propose_command.argv/iri<br/>and patch_proposal. see AB-14.13
        EP->>ST: durableStore().put(proposal_urn to proposal_foreign_urn), additive (elevation-publisher.js:163-173)
        EP->>VC: signed elevation event
        Note over EP: elevation-publisher and the agent-control-surface authority gate share a<br/>producer-plus-injected-consumer dependency shape deliberately, testable without a<br/>live relay (see AB-11.10)<br/>the full eligibility branches (gate-off / no-relays / no-signing-stack) and the<br/>propose_command/propose_iri/patch_proposal fields are in AB-14.13.
    end
    VC-->>EP: governance decision (asynchronous, proposable candidates only).<br/>see ES-05 and AB-14
    Note over AG,VC: INVARIANT ADR-2025 — the crossing is the ONLY sanctioned way an agentbox<br/>identifier becomes a VisionClaw one.<br/>bc20-provenance-bridge is the ONLY module importing the urn:visionclaw grammar (B05) —<br/>every other aggregate speaks typed urn:agentbox value objects.
    Note over BC,ST: DIVERGENCE PROTOCOL-registry — three contract rows stay open.<br/>Content address: JS sha12 hashes a UTF-8 string, VisionClaw content_address hashes<br/>bytes, byte-parity asserted nowhere.<br/>Precomputed KG address: the Rust constructor checks only the PREFIX, not the full grammar.<br/>Durable translation: helpers return a mapping, but replay, round-trip and recovery receipts do not exist.<br/>The URN crossing row is stale: see AB-17.5.
```

**Debt:** the kg URN a beam carries in `metadata.proposal_foreign_urn` is persisted to the durable mapping store only inside `elevationPublisher.publish` (`agentbox/management-api/lib/elevation-publisher.js:163-173`), which runs only for an unblocked candidate after the signing stack loads (`agentbox/management-api/lib/elevation-publisher.js:153-155`, `agentbox/management-api/routes/kg-elevation.js:229-233`). A blocked candidate, or one emitted while federation is unavailable, puts a `urn:visionclaw:kg` address on the wire that `toAgentbox` can never resolve back (`agentbox/management-api/lib/bc20-provenance-bridge.js:287-289`).
