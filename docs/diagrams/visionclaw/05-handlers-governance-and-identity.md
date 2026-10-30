---
id: VC-05
title: Governance and identity handler families
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/IDENTITY-authority-chain.md
adrs: [ADR-2006, ADR-2010, ADR-2011, ADR-2013, ADR-2016, ADR-2020, ADR-2058, ADR-2067, ADR-2075, ADR-2090, ADR-2091, ADR-2093, ADR-2094, ADR-2116]
sources:
  - ../project/src/handlers/admin_rbac_handler.rs
  - ../project/src/services/role_store.rs
  - ../project/src/handlers/nostr_handler.rs
  - ../project/src/services/nostr_service.rs
  - ../project/src/handlers/presence_handler.rs
  - ../project/src/actors/presence_actor.rs
  - ../project/src/handlers/broker_inbox_handler.rs
  - ../project/src/handlers/decision_handler.rs
  - ../project/src/handlers/enrichment_proposals_handler.rs
  - ../project/src/handlers/briefing_handler.rs
  - ../project/src/services/briefing_service.rs
  - ../project/src/handlers/insight_loop_handler.rs
  - ../project/src/services/insight_loop.rs
  - ../project/src/handlers/mcp_relay_handler.rs
  - ../project/src/services/mcp_relay_manager.rs
  - ../project/src/handlers/multi_mcp_websocket_handler.rs
  - ../project/src/actors/multi_mcp_visualization_actor.rs
  - ../project/src/services/multi_mcp_agent_discovery.rs
  - ../project/src/app_state.rs
  - ../project/src/services/bots_client.rs
  - ../project/src/services/proposal_spine.rs
  - ../project/src/handlers/solid_proxy_handler.rs
  - ../project/src/handlers/speech_socket_handler.rs
  - ../project/src/actors/elevation_actor.rs
  - ../project/src/actors/decision_elevation_actor.rs
  - ../project/src/domain/broker/mod.rs
  - ../project/src/domain/broker/broker_case.rs
  - ../project/src/domain/broker/broker_decision.rs
  - ../project/src/domain/broker/precedent_registry.rs
  - ../project/src/services/acsp/mod.rs
  - ../project/src/services/acsp/client.rs
  - ../project/src/services/nostr_bridge.rs
  - ../project/src/services/decision_service.rs
  - ../project/src/handlers/ingest_writeback_handler.rs
  - ../project/src/services/ontology_mutation_service.rs
  - ../project/src/uri/mod.rs
  - ../project/src/utils/auth.rs
  - ../project/src/utils/nip98.rs
  - ../project/src/main.rs
  - ../project/docs/adr/ADR-2006-acsp-human-approval.md
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/src/config/feature_access.rs
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/services/management_api_client.rs
  - ../project/src/services/nostr_bead_publisher.rs
  - ../project/src/handlers/ontology_handler.rs
  - ../project/docs/BASELINE-architecture.md
  - ../project/docker-compose.unified.yml
  - ../project/crates/visionclaw-jev/src/config.rs
verified_commit: {visionclaw: 8692b12ba987ecb87165566beb7c320a2446e4b4, agentbox: 6500582dfae23bf8a38a6942916d208154e43625}
---

## VC-05.1 `admin_rbac_handler` — whoami / list / assign / revoke (ADR-2010)
```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant H as admin_rbac_handler<br/>src/handlers/admin_rbac_handler.rs:285 scope /admin/rbac (registration see VC-01.13)
    participant VA as verify_access/verify_admin<br/>src/utils/auth.rs
    participant RS as RoleStore<br/>src/services/role_store.rs:408 assign_checked, :500 remove_checked

    Note over H: routes — GET /whoami :286, GET /users :287,<br/>PUT /users/{pubkey}/role :288, DELETE /users/{pubkey}/role :289
    Note over H: RbacGate now classes GET /api/admin/rbac/whoami as Authenticated (src/middleware/rbac_gate.rs:164-166, ER-26),<br/>so any signed-in user reaches this handler, while every other /api/admin path stays Admin (rbac_gate.rs:168-169)<br/>INVARIANT: only the exact GET whoami path is opened, POST to it and sub-paths of it stay Admin, rbac_gate.rs:164
    C->>H: GET /whoami
    H->>VA: verify_access(Authenticated) :82
    alt Err
        VA-->>C: deny response
    else Ok(pubkey)
        H->>RS: effective_role(pubkey, is_power_user) :88
        RS-->>C: 200 {pubkey, role, is_power_user}
    end
    C->>H: GET /users
    H->>VA: verify_admin(req, nostr) :102
    alt not Admin/Owner
        VA-->>C: 403
    else Ok
        H->>RS: list() :109
        RS-->>C: 200 {users: [UserRoleView]}
    end
    C->>H: PUT /users/{pubkey}/role {role} :124
    H->>VA: verify_admin :130
    alt not Admin/Owner
        VA-->>C: 403
    else Ok(caller)
        H->>H: UserRole::parse(body.role) :135
        alt unknown role string
            H-->>C: 400 "unknown role '{r}'"
        else parsed
            critical RS::assign_checked single tx — role_store.rs:423-478 (full sequence VC-03.8)
                H->>RS: assign_checked(target, new_role, CallerAuthority) :158
                RS-->>H: Ok(role) or RoleStoreError
            end
            alt Ok
                H-->>C: 200 {pubkey, role, assigned_by:caller}
            else Forbidden/LastOwner/CallerAuthorityChanged/InvalidPubkey
                H-->>C: role_error_response :241 — 403/409/400/500
            end
        end
    end
    C->>H: DELETE /users/{pubkey}/role
    H->>VA: verify_admin :182
    H->>RS: remove_checked(target, target_is_power, authority) :198
    RS-->>H: RemovalOutcome{had_explicit_role, previous_role, effective_after, authority_reduced}
    Note over H: ADR-2010 — removal is NOT revocation (:203) — response carries<br/>"note" warning the target may still hold effective_after by ambient default
    H-->>C: 200 {reverted_to, access_revoked:authority_reduced, note}
```

## VC-05.3 `nostr_handler` — session lifecycle routes (scope `/auth/nostr`)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as nostr_handler::config<br/>src/handlers/nostr_handler.rs:52 scope /auth/nostr :54
    participant RA as RequireAuth::authenticated<br/>wraps the /api-keys resource :61
    participant NS as NostrService<br/>src/services/nostr_service.rs
    participant FA as FeatureAccess<br/>src/config/feature_access.rs

    Note over H: this scope is on the public allowlist (api/auth prefix, RbacGate<br/>src/middleware/rbac_gate.rs:55-61 PUBLIC_SEGMENT_PREFIXES) — no RbacGate check runs<br/>before these handlers, so login, logout, verify and refresh each do their own session check<br/>and only the /api-keys resource carries its own RequireAuth wrap
    C->>H: POST empty-path login :55 -> login :130
    H->>NS: verify_auth_event(AuthEvent) :135
    alt Err(InvalidSignature)
        NS-->>C: 401 "Invalid signature" (nostr_handler.rs:159-161)
    else Err(other)
        NS-->>C: 500 "Authentication error: {e}" (nostr_handler.rs:162)
    else Ok(user)
        H->>H: expires_at = user.last_seen + AUTH_TOKEN_EXPIRY (env, :138-142, default 3600)
        H->>FA: get_available_features(pubkey) :144
        H-->>C: 200 AuthResponse{user,token:session_token,expires_at,features}
    end
    C->>H: DELETE empty-path logout :56 -> logout :166
    H->>NS: validate_session(pubkey,token) :170
    alt invalid
        NS-->>C: 401 "Invalid session" (nostr_handler.rs:174-176)
    else valid
        H->>NS: logout(pubkey) :179
        NS-->>C: 200 "Logged out successfully"
    end
    C->>H: POST /verify :57 -> verify :187
    H->>NS: validate_session(pubkey,token) :192
    H-->>C: 200 VerifyResponse{valid, user?, features} (nostr_handler.rs:214)
    C->>H: POST /refresh :58 -> refresh :221
    H->>NS: validate_session :226 then refresh_session(pubkey) :235
    alt invalid session
        NS-->>C: 401 "Invalid session" (nostr_handler.rs:230-232)
    else Ok(new_token)
        H-->>C: 200 AuthResponse{token:new_token, expires_at (AUTH_TOKEN_EXPIRY, :238-242), features}
    else Err
        H-->>C: 500 "Session refresh error: {e}" (nostr_handler.rs:260)
    end
    C->>H: GET /power-user-status :65 -> check_power_user_status :71
    H->>H: pubkey = X-Nostr-Pubkey header
    alt header missing
        H-->>C: 400 "Missing Nostr pubkey" (nostr_handler.rs:81-83)
    end
    H-->>C: 200 (feature_access power-user check)
    C->>H: GET /features :66 / GET /features/{feature} :67
    H->>FA: get_available_features(pubkey) / has_feature_access(pubkey,feature)
    H-->>C: 200 {features} / {has_access}
    C->>RA: POST /api-keys (nostr_handler.rs:59-63) -> update_api_keys nostr_handler.rs:277 / GET /api-keys -> get_api_keys nostr_handler.rs:310
    RA->>RA: NIP-98 identity, or a legacy session when VISIONCLAW_LEGACY_SESSIONS is on
    RA->>H: handler runs with the verified identity in request extensions
    H->>H: caller_pubkey(req) :267 reads get_authenticated_user — never the URL or body, so a caller acts only on their own record
    H->>NS: update_user_api_keys(caller, keys) :293 / state.get_api_keys(caller) :319
    Note over H: RESOLVED ER-24 (VisionClaw cefd9d1d7) — the route used to be "/api-keys" with no {pubkey} segment<br/>while both handlers extracted web::Path~String~, so the extractor failed and the routes could not succeed<br/>they now take the pubkey from RequireAuth, and get_api_keys no longer does its own Bearer check
    Note over H: INVARIANT: the pubkey an api-keys handler acts on comes from the authenticated request, nostr_handler.rs:267-275
```

## VC-05.4 `presence_handler` — PRD-008 `/ws/presence` handshake (T-WS-1)
```mermaid
sequenceDiagram
    autonumber
    participant C as XR client<br/>Godot
    participant WS as PresenceSession<br/>src/handlers/presence_handler.rs:135 ws_presence :501 (route src/main.rs:1119)
    participant NC as SeenNonces LRU<br/>presence_handler.rs:51 cap 4096
    participant IV as IdentityVerifier<br/>visionclaw_xr_presence::ports
    participant REG as PresenceRoomRegistry<br/>presence_handler.rs:41 Arc~DashMap~String,Addr~PresenceActor~~~
    participant PA as PresenceActor<br/>src/actors/presence_actor.rs:247

    WS->>WS: Actor::started :436 — send_challenge :163 {type:challenge,nonce,ts}<br/>run_interval HEARTBEAT_INTERVAL=15s (:35) :438-441
    C->>WS: text {type:auth, did, signature, room_id, metadata, ts?}
    WS->>WS: handle_auth :191 — must be in SessionPhase::Challenged
    alt wrong phase
        WS-->>C: close 4400 "auth in wrong phase"
    end
    opt client ts present
        WS->>WS: skew = |now_us - client_ts|
        alt skew > MAX_HANDSHAKE_SKEW_US=30_000_000us (presence_handler.rs:61, checked :224)
            WS-->>C: close 4401 "stale handshake timestamp"
        end
    end
    WS->>IV: verify_signed_challenge(SignedChallenge{nonce,timestamp_us,claimed_pubkey_hex,signature_hex})
    alt Err or verified.as_str() != did
        IV-->>WS: error / mismatch
        WS-->>C: close 4401 "auth: {e}" / "did/signature mismatch"
    else Ok(verified)
        WS->>NC: seen.put(nonce,()) :266
        alt nonce already present
            NC-->>WS: Some (replay)
            WS-->>C: close 4401 "replayed challenge"
        end
        WS->>WS: RoomId::parse(room_id)
        alt invalid room_id
            WS-->>C: close 4400 "room_id: {e}"
        end
        WS->>REG: entry(room).or_insert_with(PresenceActor::new(room).start())
        Note over REG: dead-Addr guard :286-291 — if entry.connected()==false<br/>(room previously emptied), replace with a fresh actor before join
        WS->>PA: JoinRoom{did,metadata,frame_recipient,event_recipient}
        alt Ok(Err(JoinRejection::DuplicateMember)) or other rejection
            PA-->>WS: JoinRejection
            WS-->>C: close 4400 duplicate member / rejection.to_string()
        else Ok(Ok(JoinAck{avatar_id,members}))
            PA-->>WS: JoinAck
            WS-->>C: text {type:joined, room_id, avatar_id, members:[MemberDescriptor]}
            WS->>WS: phase = SessionPhase::Joined{avatar_id, room_addr}
        end
    end
    loop every 15s heartbeat
        WS->>C: ws ping
        alt no pong/ping for > 30s (2x interval)
            WS->>WS: ctx.stop()
        end
        WS->>WS: enforce_handshake_deadline :395 — Challenged for >10s HANDSHAKE_TIMEOUT (:34)
        alt still Challenged after 10s
            WS-->>C: close 4401 "handshake timeout"
        end
    end
    Note over WS: Actor::stopping :444 — if Joined, room_addr.do_send(LeaveRoom{avatar_id})
```

## VC-05.5 `presence_handler`/`PresenceActor` — binary pose frame ingest and broadcast
```mermaid
sequenceDiagram
    autonumber
    participant C as XR client (Joined)
    participant WS as PresenceSession::handle_pose_frame<br/>presence_handler.rs:355
    participant PA as PresenceActor::handle_ingest<br/>src/actors/presence_actor.rs:751
    participant HR as configured_hand_reach_m<br/>presence_actor.rs:45 env PRESENCE_HAND_REACH_M :46

    Note over WS: opcode 0x43 OPCODE_AVATAR_POSE — presence_handler.rs:5 doc,<br/>const PREAMBLE_OPCODE = wire::OPCODE_AVATAR_POSE at presence_actor.rs:54<br/>envelope [opcode u8][len u16][room_hash 16B][avatar_id_len u8][avatar_id][payload] :800-802
    C->>WS: binary frame (0x43 sibling envelope)
    alt phase not Joined
        WS-->>C: close 4400 "binary before auth"
    end
    WS->>WS: check_rate_limit :175-188 — sliding 1s window, RATE_LIMIT_FRAMES_PER_SEC=120 (:32)
    alt over limit
        WS-->>C: close 4429 "rate limit exceeded"
    else within limit
        WS->>PA: IngestPose{avatar_id, frame_bytes}
        PA->>PA: not a room member? -> ValidationFailed
        PA->>PA: wire::decode(frame_bytes)
        alt decode error
            PA->>PA: record_violation — VIOLATION_KICK_THRESHOLD=10 within VIOLATION_WINDOW=1s
            alt threshold exceeded
                PA-->>WS: IngestOutcome::Kick("decode-violations exceeded: {e}")
                WS-->>C: close 4400
            else
                PA-->>WS: IngestOutcome::Decode(e)
            end
        else decoded
            alt decoded.avatar_id != msg.avatar_id
                PA-->>WS: Kick("avatar-id spoofing") after violation threshold
            else decoded.room_hash != room_id.wire_hash()
                PA-->>WS: ValidationFailed("room_hash mismatch")
            else run_validators fails (HR: hand_reach_m gate, joint_anatomy, velocity_gate)
                PA-->>WS: ValidationFailed(e) or Kick after threshold
            else Ok
                PA->>PA: room.update_pose(avatar_id, frame)
                PA->>PA: strip envelope, pending_poses.insert(avatar_id, payload)
                PA->>PA: dispatch_broadcast(avatar_id)
                PA-->>WS: IngestOutcome::Accepted
                PA->>C: BroadcastFrame{bytes} to every Subscriber.frame_recipient (fan-out)
            end
        end
    end
    Note over PA: RoomEventEnvelope :216 (JSON channel, not 0x43) — AvatarJoined :217,<br/>AvatarLeft :600, AgentPresenceExpired :234 (ADR-2020 co-presence retirement,<br/>SweepStaleAgentPresence handler :822-828) — distinct from IngestAgentPresence :814 0x44 deltas
```

## VC-05.6 `decision_handler` — record (authed, append-only) / trace (anon, bounded)
```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant SC as scope /decisions<br/>src/handlers/decision_handler.rs:321-331
    participant RA as RequireAuth::authenticated<br/>wraps nested scope /record only :326
    participant H as decision_handler<br/>src/handlers/decision_handler.rs:145 record_decision, :231 trace_decision
    participant DS as DecisionService::record_decision<br/>src/services/decision_service.rs:546
    participant SP as proposal_spine::governed_commit<br/>src/services/proposal_spine.rs:574 (Stage3 commit_quads :519)

    C->>SC: POST /decisions/record {summary,rationale,caused,precedent_for,...}
    SC->>RA: nested scope wrap (PRD-022 W-B / ADR-048)
    alt not authenticated
        RA-->>C: 401 (RequireAuth deny)
    else Ok(AuthenticatedUser)
        Note over H: INVARIANT ADR-048 Attribution — deciding principal is auth.pubkey,<br/>NEVER a body field, a caller cannot self-assert another agent's decision
        H->>DS: record_decision(&auth.pubkey, DecisionInput, idempotency_key, signature)
        DS->>SP: governed_commit — conflict gate, Whelk consistency, provenance append, single tx
        critical commit_quads — ONE oxigraph store.transaction, proposal_spine.rs:525-535
            SP->>SP: for q in provenance_quads: tx.insert(q) — INSERT-ONLY, no DELETE/DROP/CLEAR
            SP->>SP: for q in asserted_quads: tx.insert(q)
        end
        Note over SP: ADR-2016 — GRAPH_PROVENANCE append-only holds on this path too,<br/>governed_commit funnels through commit_quads which issues only tx.insert
        alt Ok(success)
            DS-->>H: DecisionSuccess{decision_urn, replayed, quads_written, receipt, gates}
            H-->>C: 200 {success:true, decision_urn, quads_written, replayed, receipt, gates}
        else Err CONFLICT_BLOCKED_PREFIX
            H-->>C: 409 {error:conflict_blocked, blockingConflicts, preExisting, conflictReport}
        else Err IDEMPOTENCY_CONFLICT_PREFIX
            H-->>C: 409 {error:idempotency_conflict, message}
        else Err ENVELOPE_REJECTED_PREFIX
            H-->>C: 403 {error:envelope_rejected, message}
        else Err other
            H-->>C: 500 "Decision write failed"
        end
    end
    C->>H: GET /decisions/{urn}/trace?direction&max_depth (anon, no RequireAuth)
    H->>H: max_depth = min(query.max_depth, MAX_DEPTH_CAP=64) decision_handler.rs:120 cap, :238 clamp
    loop up to max_depth SPARQL frontier expansions
        H->>H: direct_links_query(frontier, direction) — DIRECT dl:caused/dl:precedentFor only
        H->>H: store.query(sparql) — never a transitive property path
    end
    H->>H: bounded_bfs(root, max_depth, adjacency)
    H-->>C: 200 TraceResponse{derived:true, root, direction, max_depth, hops}
    Note over H: response is stamped derived:true — reachability, never a materialised<br/>or Whelk-classified transitive edge (ADR-048 Graph placement)
    Note over DS: ADR-2116 — ontology_agent_handler.rs's POST /ontology-agent/propose is<br/>retired to 410 Gone (handler :221, route :378), but OntologyMutationService and its three<br/>error-prefix consts are NOT removed: decision_service.rs:408-411 imports CONFLICT_BLOCKED_PREFIX,<br/>IDEMPOTENCY_CONFLICT_PREFIX, ENVELOPE_REJECTED_PREFIX from it directly (used above). The 410's<br/>body tells the callers still holding that URL — agentbox's ontology-propose.js, the<br/>ontology-augment/podcast-knowledge-ingest skills, the ontology-curator agent — to run<br/>`vault propose` instead (ontology_agent_handler.rs doc comment :193-217)
```

## VC-05.7 shared enrichment-decide core (ADR-2006 / ADR-130 Decision 2) — three callers, one path
```mermaid
sequenceDiagram
    autonumber
    participant AB as agentbox broker-bridge<br/>X-Agent-Key service caller
    participant OP as power-user operator<br/>via broker_inbox_handler scope
    participant GB as git-bridge write-back<br/>agentbox management-api
    participant D as decide / decide_as_operator<br/>src/handlers/enrichment_proposals_handler.rs:424<br/>decide_as_operator :454
    participant WB as ingest_writeback_handler::writeback<br/>src/handlers/ingest_writeback_handler.rs:75
    participant AD as apply_decision (shared core)<br/>enrichment_proposals_handler.rs:473
    participant OK as DecisionOrchestrator<br/>src/domain/broker/broker_decision.rs (ADR-130 Decision 2 kernel)
    participant OX as OntologyRepository::append_derived_summary
    participant FR as AcspClient::publish<br/>src/services/acsp/mod.rs kind 31403

    AB->>D: POST /api/enrichment-proposals/{id}/decide (X-Agent-Key)
    D->>D: require_agent_key :141 — header must equal VISIONCLAW_AGENT_KEY (env read :147, ADR-2093)
    alt key invalid/missing
        D-->>AB: 401 {success:false, error:"Invalid or missing X-Agent-Key header"}
    end
    OP->>D: POST /api/broker/cases/{id}/decide (decide_as_operator, power_user() scope)
    D->>D: if body has no broker_pubkey and user.pubkey is canonical hex -> attribute to operator (HITL)
    GB->>WB: POST /api/ingest/writeback {decision:{caseId,decision,...}} (no X-Agent-Key required, GOV-4)
    WB->>AD: apply_decision(case_id, BrokerDecisionRequest, state, client_coordinator)
    D->>AD: apply_decision(case_id, body, state, client_coordinator) (both agent + operator routes)
    rect rgb(225,245,225)
    Note over AD: single decision core — validate, gate, mint provenance, persist, write-back, broadcast, project
    AD->>AD: record_decision(case_id, req) :479 — trim outcome, classify(outcome)
    alt empty case_id or empty outcome
        AD-->>D: Err "empty case id"/"empty decision outcome" -> 400
    end
    AD->>AD: existing = repo.get(case_id) read ONCE :506 — supplies both the new-case round trip and the tier
    AD->>AD: declared_tier_of(proposal_json) :369, then gate_tier(declared, body.effective_tier) :382 (call :508), then check_rationale(tier, outcome, reasoning) :396
    alt tier is high or critical, outcome is a human verdict, rationale under 20 trimmed chars
        AD-->>D: 422 RationaleRejection{code:"rationale_required", tier, min_chars, received_chars} :516
    end
    Note over AD: ADR-2110 FR2.2 (EXP-AC-002) — the case queue's client-side gate is a courtesy,<br/>so the SAME rule runs here on the one shared core both routes funnel through, BEFORE<br/>anything is minted or persisted. check_rationale is a predicate, never a transformer:<br/>it refuses and names what is missing, it never authors the rationale, because a<br/>server-written rationale is the fabricated judgement DDD invariant 1 forbids :392-396
    Note over AD: INVARIANT — MIN_RATIONALE_CHARS :320 and the client's MIN_RATIONALE_CHARS in<br/>brokerCaseQueue.ts are one rule enforced in two places, not two rules :310-312
    Note over AD: ER-7 landed - gate_tier takes the STRICTER of the tier recorded on the case and the forum effective_tier the bridge forwards<br/>(request field :70, gate_tier :382-392, broker-bridge.js:488-497), so a forwarded tier can only tighten the gate<br/>A case with no row and no forwarded tier still carries no tier and the gate does not fire
    Note over AD: DRIFT: the scope comment at enrichment_proposals_handler.rs:308-314 still says effective_tier has not landed<br/>while the request field and gate_tier at enrichment_proposals_handler.rs:70 and :382 read it
    AD->>AD: attribution — 64-hex broker_pubkey -> did_nostr+kg URN, else unattributed
    AD->>AD: activity_urn = uri::execution(content-addressed over case_id,outcome,pubkey) — idempotent
    opt repo.get(case_id) is None
        AD->>AD: is_new_case=true, create pending StoredProposal stub
    end
    AD->>AD: repo.record_decision(StoredDecision) :561 — atomic INSERT decision + UPDATE proposal.status
    Note over AD: ADR-2006 — REST decisions carry decision_event_id:None (:558, in the<br/>StoredDecision built at :543-560) — no signed kind-31403 correlation is stored
    alt persist Err
        AD-->>D: 500 "failed to persist decision: {e}"
    end
    alt writeback_triggered AND attributed
        AD->>OX: append_derived_summary(owner_did, activity_urn, summary_triples_for(record))
        alt Ok
            AD->>AD: repo.mark_writeback_committed(case_id, activity_urn, now_ms)
        else Err
            AD->>AD: writeback_committed=false, warn DEGRADED
        end
    end
    AD->>D: BroadcastMessage{enrichment_decision} to client_coordinator (all WS clients)
    AD->>OK: derive_kernel_decision — DecisionOrchestrator::decide(case, activity_urn, outcome, broker, reasoning)
    alt kernel invariant note (e.g. no-self-review)
        OK-->>AD: Err(e) -> degrade to raw outcome string, warn (record-don't-reject)
    else Ok(report)
        OK-->>AD: kernel_action, share_plan
    end
    opt is_new_case
        AD->>AD: broker_events::broadcast_new_case(client_coordinator, case_id, case_id, knowledge_enrichment)
    end
    AD->>AD: broker_events::broadcast_case_decided(client_coordinator, case_id, activity_urn, kernel_action, share_plan)
    alt state.acsp_client Some
        AD->>FR: publish(build_action_response(case_id, kernel_action, reasoning)) kind 31403
        alt Ok(event_id)
            FR-->>AD: forum_projection="published"
        else Err
            FR-->>AD: forum_projection="failed" (DEGRADED, warn)
        end
    else None configured
        AD->>AD: forum_projection="skipped" (DEGRADED — needs FORUM_RELAY_URL and a panel key via acsp key_file)
    end
    AD->>AD: state.liveness_harness.observe(CANARY_REC2_CASE, evidence) — REC-2 canary, real traffic only
    end
    AD-->>D: DecideResponse{success:true, decision, attributed, writeback_triggered, writeback_committed, activity_urn, proposal_urn}
    D-->>AB: 200 DecideResponse
    D-->>OP: 200 DecideResponse
    AD-->>WB: 200 DecideResponse
```

## VC-05.8 `broker_inbox_handler` — dedicated privileged read scope (WS-12)
```mermaid
sequenceDiagram
    autonumber
    participant C as agentbox broker-bridge<br/>management-api/routes/broker-bridge.js
    participant SC as scope /broker<br/>src/handlers/broker_inbox_handler.rs:250 wrap RequireAuth::power_user() :251
    participant H as broker_inbox_handler<br/>inbox :220, case_by_id :230
    participant ST as enrichment_proposals_handler::store<br/>same durable store as WS-9, no second store

    Note over SC: doc comment :243-247 — "Mounted as a dedicated web::scope(broker) so the<br/>privileged RequireAuth::power_user() middleware wraps exactly these read routes<br/>and nothing else, mirroring the isolated privileged scope at ontology_handler.rs lines 913-918"
    Note over SC: DOC-DRIFT — that cross-reference is itself stale: WS-0 removed ontology_handler.rs<br/>own web::scope entirely (ontology_handler.rs:1095-1105), re-using its handlers from the single<br/>canonical scope in api_handler::ontology::config instead, see VC-20. No isolated privileged<br/>scope exists there any more — /broker is the odd one out now, not a mirror of it.
    C->>SC: GET /api/broker/inbox (X-Agent-Key or power-user session)
    SC->>SC: RequireAuth::power_user()
    alt not power user
        SC-->>C: 401/403 deny
    else Ok
        SC->>H: inbox() :252
        H->>ST: store::all()
        ST-->>H: Vec~EnrichmentProposal~
        H->>H: assemble_inbox (broker_inbox_handler.rs:193) projects each into BrokerCase{id,category:"knowledge_enrichment",status,metadata}
        opt VISIONCLAW_JEV_PROPOSAL_TRIAGE=1 (Jev option: library default off, but the Compose master flag docker-compose.unified.yml:21 turns it on and ALLOW_CLOUD=1 at :22 lets this local-only item take the shared agentbox proxy, VisionClaw ADR-2148, accepted / complete / live - see VC-24.13)
            H->>H: pending and claimed cases may gain an advisory jev annotation and be reordered by triage priority within annotated slots (broker_inbox_handler.rs:193-215, jev field broker_inbox_handler.rs:78)
        end
        Note over H: INVARIANT: the Jev triage is advisory, decide routes and signed decisions never read it, broker_inbox_handler.rs:78
        H-->>C: 200 {cases:[BrokerCase], total} — the bridge re-shapes it as total plus filtered_total (broker-bridge.js:385-386)
    end
    C->>SC: GET /api/broker/cases/{id} (bridge route broker-bridge.js:394)
    SC->>H: case_by_id(id) :230
    H->>ST: store::get(id)
    alt Some(p)
        ST-->>C: 200 BrokerCase::from(&p)
    else None
        ST-->>C: 404 {error:"not-found", message:"no broker case with id {id}"}
    end
    C->>SC: POST /api/broker/cases/{id}/decide (the bridge route broker-bridge.js:435 instead proxies its decision to /api/enrichment-proposals/{id}/decide at broker-bridge.js:576)
    SC->>H: delegates to enrichment_proposals_handler::decide_as_operator (see VC-05.7)
    Note over H: REC-2 / D3 (PRD-023 WP-4) — power-user-gated by the surrounding scope,<br/>funnels through the SAME decision core as the agentbox X-Agent-Key route
```

## VC-05.9 `briefing_handler` — brief submit / debrief, Management API bridge
```mermaid
sequenceDiagram
    autonumber
    participant C as Client (voice/UI)
    participant H as briefing_handler<br/>src/handlers/briefing_handler.rs:118 scope /briefs
    participant BS as BriefingService<br/>src/services/briefing_service.rs:11 submit_brief :24, request_debrief :80
    participant MA as ManagementApiClient<br/>src/services/management_api_client.rs
    participant NB as NostrBeadPublisher<br/>src/services/nostr_bead_publisher.rs (optional)

    rect rgb(225,225,245)
    Note over H,MA: process boundary — Management API runs in the agentbox agent container
    C->>H: POST "" {briefing:BriefingRequest, user_context} :119 -> submit_brief :22
    H->>BS: submit_brief(request, user_context) :34
    BS->>MA: create_brief(content, roles, user_context, version?, brief_type?, slug?) :37
    alt Err
        MA-->>C: 502 BadGateway {error:"Brief submission failed", message}
    else Ok(brief_result)
        MA-->>BS: {brief_id, brief_path, bead_id}
        BS->>MA: execute_brief(brief_id, brief_path, roles, user_context, bead_id) :55
        alt Err
            MA-->>C: 502 BadGateway "Failed to execute brief"
        else Ok(role_tasks)
            MA-->>BS: Vec~RoleTask~ (spawned role agents)
            BS-->>H: BriefingResponse{brief_id,brief_path,bead_id,role_tasks}
            H-->>C: 201 Created BriefingResponse
        end
    end
    end
    C->>H: POST /{brief_id}/debrief {user_context,role_tasks} :120 -> request_debrief :54
    H->>H: bead_id = first role_task.bead_id or brief_id :69-74
    H->>BS: request_debrief(brief_id, role_tasks, user_context) :77
    BS->>MA: create_debrief(brief_id, role_tasks, user_context) :93
    alt Err
        MA-->>C: 502 BadGateway "Debrief creation failed"
    else Ok(debrief_path)
        opt nostr_publisher configured
            H->>NB: tokio::spawn publish_bead_complete(bead_id, brief_id, user_pubkey, debrief_path) :88-97
            Note over NB: fire-and-forget — does not affect the HTTP response
        end
        H-->>C: 201 Created {brief_id, debrief_path}
    end
```

## VC-05.10 `insight_loop_handler` — REC-10 Mesh Velocity trace reads
```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant H as insight_loop_handler<br/>src/handlers/insight_loop_handler.rs:91 scope /insight-loop
    participant IL as insight_loop::summarise / build_trace<br/>src/services/insight_loop.rs:209 / :103
    participant DB as sqlite_enrichment_repository<br/>loop_traces(limit) / loop_trace_for(case_id)
    participant LH as liveness_harness.observe<br/>CANARY_REC10_LOOP

    Note over H: assembled from the governed write-back queue's persisted stage timestamps —<br/>propose/queued (proposal row), broker_decision/merged_enrichment (decision row).<br/>amplification stage is labelled "planned" (v1 scope, insight_loop_handler.rs:9-13)
    C->>H: GET /trace?limit= :92 -> traces :53
    H->>H: limit = query.limit.unwrap_or(DEFAULT_LIMIT=100).clamp(1,MAX_LIMIT=1000)
    H->>DB: loop_traces(limit)
    alt Err
        DB-->>C: 500 {error:"loop-trace read failed: {e}"}
    else Ok(rows)
        H->>IL: summarise(rows)
        IL-->>H: InsightLoopSummary{traces, mesh_velocity aggregate}
        opt any trace loop_closed AND monotonic
            H->>LH: observe(CANARY_REC10_LOOP, evidence) — REC-10, observed live traffic only
        end
        H-->>C: 200 InsightLoopSummary
    end
    C->>H: GET /trace/{case_id} :93 -> trace_by_case :68
    H->>DB: loop_trace_for(case_id)
    alt Ok(Some(row))
        H->>IL: build_trace(row)
        IL-->>H: InsightLoopTrace
        opt loop_closed AND monotonic
            H->>LH: observe(CANARY_REC10_LOOP, evidence)
        end
        H-->>C: 200 InsightLoopTrace
    else Ok(None)
        H-->>C: 404 {error:"not-found", message:"no insight-loop trace for case {id}"}
    else Err
        H-->>C: 500 {error:"loop-trace read failed: {e}"}
    end
```

## VC-05.12 `mcp_relay_handler` — `/ws/mcp-relay` bidirectional bridge to the orchestrator
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as mcp_relay_handler<br/>src/handlers/mcp_relay_handler.rs:438 (route src/main.rs:1115)
    participant A as MCPRelayActor<br/>src/handlers/mcp_relay_handler.rs:39
    participant O as orchestrator WS<br/>ORCHESTRATOR_WS_URL env mcp_relay_handler.rs:78 in connect_to_orchestrator :77

    C->>H: GET /ws/mcp-relay (Authorization: Bearer)
    H->>H: header_token = Authorization Bearer :475-480 — ADR-2058/ADR-2090 comment :469-474
    Note over H: ADR-2058 — the ?token= query carrier is compiled out of release: it survives<br/>only under cfg(debug_assertions, feature dev-auth) :482-494, and a release build that<br/>sees token= logs a SECURITY warning and ignores it :496-505
    alt no token at all
        H-->>C: 401 {error:"Authentication required"} :507-509 (unauthorised closure :459-467)
    else app_state.nostr_service is None
        H-->>C: 401 fail-closed — no session store to validate against :513-516
    else token does not name a live session
        H->>H: nostr_service.get_session(token).await.is_none() :518
        H-->>C: 401 "token does not name a live, unexpired session" :519
    else Ok — live, unexpired session
        Note over H: ADR-2090 (:445-452) — this upgrade previously accepted ANY non-empty string<br/>(presence-only). It now resolves the token through NostrService::get_session, which<br/>per ADR-2044 also enforces the AUTH_TOKEN_EXPIRY window. The old DOC-DRIFT is CLOSED.
        H->>A: ws::start(MCPRelayActor::new(), req, stream) :523
        A->>A: connect_to_orchestrator — CircuitBreaker::execute, connect_async(url), timeout=connect_timeout
        alt Ok(ws_stream)
            A->>O: split into tx/rx, do_send(SetOrchestratorTx(tx))
            A->>A: health_manager.check_service_now("orchestrator")
            A->>C: OrchestratorText("connected")
            par client -> orchestrator
                C->>A: ws::Message::Text(json)
                alt json.type == "ping"
                    A-->>C: {type:pong, timestamp}
                else orchestrator_tx set AND healthy
                    A->>O: tokio::time::timeout(5s, tx.send(Text))
                    alt timeout or Err
                        A->>A: health_manager.check_service_now (mark degraded)
                    end
                else unhealthy
                    A-->>C: {type:error, message:"Orchestrator unhealthy"}
                else not connected
                    A-->>C: {type:error, message:"Orchestrator not connected"}
                end
                C->>A: ws::Message::Binary(bin)
                A->>O: tokio::time::timeout(5s, tx.send(Binary)) (same health/degrade branches)
            and orchestrator -> client
                O->>A: TungsteniteMessage::Text/Binary via rx.next()
                A->>A: do_send(OrchestratorText/OrchestratorBinary)
                A->>C: ctx.text(text) / ctx.binary(bin) (Handler~OrchestratorText~ :261, Handler~OrchestratorBinary~ :285)
            end
        else Err/timeout
            A->>A: circuit_breaker records failure, retry on next connect_to_orchestrator
        end
    end
    C->>A: ws::Message::Close
    A->>A: ctx.stop()
```

## VC-05.13 `multi_mcp_websocket_handler` — `/multi-mcp/ws` only, `/status` + `/refresh` deleted (ADR-2091), discovery wired (ADR-2091 addendum 2026-10-12)
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as configure_multi_mcp_routes<br/>src/handlers/multi_mcp_websocket_handler.rs:867 scope /multi-mcp :869
    participant U as multi_mcp_visualization_ws<br/>multi_mcp_websocket_handler.rs:776 upgrade guard
    participant WS as MultiMcpVisualizationWs<br/>multi_mcp_websocket_handler.rs:50 new :119, Actor::started :410
    participant DS as MultiMcpAgentDiscovery<br/>src/services/multi_mcp_agent_discovery.rs:101

    Note over H: RESOLVED - the /status and /refresh dead stubs this diagram used to<br/>document<br/>were DELETED in ac3e12dd1 under ADR-2091 (removal comment :870-875).<br/>The scope now<br/>registers exactly one route, /ws :876 - get_mcp_server_status and<br/>refresh_mcp_discovery<br/>no longer exist anywhere in src/
    C->>U: GET /multi-mcp/ws (Authorization: Bearer)
    U->>U: header_token = Authorization Bearer :812-817 (ADR-2058/ADR-2090 comment :806-811)
    Note over U: ADR-2058 - the ?token= carrier survives only under<br/>cfg(debug_assertions, feature dev-auth) :819-831 - a release build<br/>that sees<br/>token= logs a SECURITY warning and drops it :833-842
    alt no token
        U-->>C: 401 {error:"Authentication required"} :844-846 (unauthorised closure :797-804)
    else no NostrService configured
        U-->>C: 401 fail-closed :851-853
    else get_session finds no live session
        U->>U: nostr_service.get_session(token).await.is_none() :855
        U-->>C: 401 "token does not name a live, unexpired session" :856
    else Ok
        Note over U: ADR-2090 (:783-790) - this upgrade previously accepted ANY non-empty<br/>string. It now resolves the token through NostrService::get_session,<br/>which per<br/>ADR-2044 enforces AUTH_TOKEN_EXPIRY. Same fix as /ws/mcp-relay, see<br/>VC-05.12
        U->>WS: ws::start(MultiMcpVisualizationWs::new(app_state.multi_mcp_discovery.clone())) :861
        Note over U,DS: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) - new()<br/>used to take (app_state, None) and never touch<br/>the discovery service. It now takes the shared<br/>Arc<MultiMcpAgentDiscovery> built at boot<br/>(src/handlers/multi_mcp_websocket_handler.rs:119, AppState field<br/>src/app_state.rs:364-365, construction src/app_state.rs:1166-1168)
        WS->>WS: started :410 - start_heartbeat :413 (5s ping :159-169)
        WS->>DS: discovery.attach() :417 - subscriber count +1 SYNCHRONOUSLY (multi_mcp_agent_discovery.rs:170)
        WS->>DS: actix::spawn sync_running() :419 - idempotent, serialised by a mutex (multi_mcp_agent_discovery.rs:184-193)
        DS->>DS: sync_running calls start_discovery when the count is above zero and<br/>the loop is not running :186-189, which re-checks :298-305, bumps<br/>epoch :307 and wakes the loop :298
        Note over DS: INVARIANT: a poll loop lives only while its own epoch is current<br/>(multi_mcp_agent_discovery.rs:327) and a stop or restart<br/>bumps the epoch and notifies the sleeping loop (:441-442, :425-428),<br/>so a quick reconnect leaves exactly one loop.<br/>Mutation-checked by a_quick_reconnect_leaves_exactly_one_loop (:991)
        WS->>WS: start_health_monitor :422 - ADR-2094 one health task per connection (fn :180, comment :173-179)
        WS->>WS: start_position_updates :433, then run_interval 60s recovery probe :441-463
        WS->>WS: run_interval 1s tick :431-437 - cycle_count() above last_frame_cycle triggers send_discovery_data :429
        Note over WS: Nothing is sent at connect (comment :428-430). The first frame arrives<br/>only after a discovery cycle has completed
        loop every discovery cycle (min enabled interval, floor 250 ms)
            DS->>DS: discover enabled servers concurrently :326-403, cycles += 1 :413, epoch recheck :417
            Note over DS: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) -<br/>initialize_default_servers :212 enables claude-flow only (:225).<br/>ruv-swarm is enabled only when RUV_SWARM_HOST or RUV_SWARM_PORT is<br/>non-empty (:242) and daa only for DAA_HOST or DAA_PORT (:259),<br/>via env_configured :134. CLAUDE_FLOW_HOST/MCP_TCP_PORT :220-224,<br/>RUV_SWARM_HOST/PORT :237-241, DAA_HOST/PORT :254-258 are now read by a<br/>CONSTRUCTED service (the earlier DIVERGENCE said the server list was<br/>dead configuration on the live path)
            DS->>DS: a failed first cycle now inserts a disconnected McpServerInfo<br/>(or_insert_with :382-393, is_connected=false :403)
        end
    end
    C->>WS: text "ping" (plain, pre-JSON) -> "pong" :486-489
    C->>WS: text {action:"configure", data:ClientConfig} :495 -> handle_client_config :311
    C->>WS: text {action:"request_discovery"} :504 -> handle_discovery_request :335
    WS->>WS: rate-limited to 1 per second :338-344
    WS->>WS: send_discovery_data :249 - has_healthy_services gate :252
    alt cached verdict says no connected server
        WS-->>C: {type:error, message:"No connected MCP servers"} :257-264
    else cached verdict usable
        WS->>DS: actix::spawn :272 get_server_statuses() :273 - re-judged on the state<br/>just read (not the verdict cached up to 30 s ago) :275-281
        alt fold_service_health false on fresh state
            WS-->>C: {type:error, message:"No connected MCP servers"} via DiscoveryFrame :282-290, handler :686-688
        else at least one server connected
            WS->>DS: get_all_agents :293, get_topology_data :294, get_discovery_stats :295
            WS-->>C: build_discovery_frame :297 (fn :357) - type discovery_data with<br/>servers, agents, topology, stats, filtered by SubscriptionFilters<br/>:366-383
            WS->>WS: DiscoverySuccess :307 -> record_success :713-722
        end
    end
    Note over WS: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) - the<br/>earlier DIVERGENCE said this path never reached the discovery service:<br/>a fastrand 20 per-cent synthetic ConnectionRefused and a 100 ms sleep<br/>inside retry_with_backoff over CircuitBreaker::execute, then Ok,<br/>and a Handler for RequestDiscoveryData that only logged. All of that<br/>is gone. Handler for RequestDiscoveryData now calls<br/>send_discovery_data (:674-680)
    Note over WS: DOC-DRIFT (RESOLVED 2026-10-12, project 6a1a74b8b/4f4345503) -<br/>MONITORED_SERVICES (claude-flow, ruv-swarm, flow-nexus) and the<br/>localhost port 8080+i<br/>HealthCheckManager registration are gone. The ADR-2094 monitor (30 s<br/>interval :26, sleep :187) now reads discovery.get_server_statuses()<br/>:202,<br/>maps each server_id and is_connected :203-206, folds with<br/>fold_service_health :208 (pure, an empty result set is usable :38-40)<br/>and stores :215
    Note over WS: The cached verdict starts true (:138) and the monitor sleeps 30 s<br/>before its first verdict (:187), so on a new connection it is the<br/>fresh re-judge below (:273-281) that produces the error frame, not the<br/>cached gate
    C->>WS: text {action:"request_agents"} :507 - wrapped in catch_unwind :508
    WS->>WS: reads CircuitBreaker stats then do_send(RequestAgentUpdate) in both the Open and the other arm :515-528
    Note over WS: DEBT: request_agents never reaches the discovery service. Handler for<br/>RequestAgentUpdate only logs at debug (:669-671)<br/>and so do the RequestPerformanceUpdate (:694-699) and<br/>RequestTopologyUpdate (:705-710) handlers, though the 100 ms position<br/>tick<br/>posts RequestAgentUpdate (:153-155). ADR-2091 addendum calls this<br/>circuit-breaker path a simulation left unchanged
    C->>WS: close frame or heartbeat timeout (30 s :160-166)
    WS->>DS: stopped :466 - detach() :468, then sync_running() :470 - the decrement<br/>saturates at zero (multi_mcp_agent_discovery.rs:175-178)
    DS->>DS: stop_discovery when count is zero :437-450 - running=false, epoch bumped :441, caches cleared :446-448
    rect rgb(255,235,235)
    Note over WS,DS: DEBT: cycles is never reset by stop_discovery<br/>(multi_mcp_agent_discovery.rs:437-450, counter :413) while a new<br/>socket starts last_frame_cycle at 0<br/>(multi_mcp_websocket_handler.rs:137),<br/>so after an idle gap the first 1 s tick (:431-433) can push a frame<br/>while the caches are still cleared<br/>(multi_mcp_agent_discovery.rs:446-448). An empty status set folds to<br/>usable<br/>(multi_mcp_websocket_handler.rs:38-40), so that frame carries empty<br/>servers and agents, which the comment at :428-430 says it avoids
    Note over WS,DS: OPEN: no client in client/src or xr-client consumes the discovery_data<br/>frame (ADR-2091 addendum) - its shape is defined only by<br/>build_discovery_frame :357-403 and is provisional
    Note over WS,DS: OPEN: not observed against a live claude-flow hub. The ruv-swarm and<br/>daa servers have no confirmed endpoint in this estate<br/>(agentbox lists ruflo-daa as purpose unknown), so they stay disabled<br/>unless configured
    Note over WS,DS: RESOLVED 2026-10-12 (project 9f98e3787): discovery now resolves the<br/>claude-flow host as BotsClient does (CLAUDE_FLOW_HOST, then MCP_HOST,<br/>then multi-agent-container), so both pollers target the same host on<br/>port 9500. The helper is claude_flow_host<br/>(multi_mcp_agent_discovery.rs:143, call :220), mirroring<br/>bots_client.rs:120-122.
    Note over WS,DS: DEBT: DiscoveryStats fields total_discoveries,<br/>total_agents_discovered, servers_online and servers_offline are<br/>declared (multi_mcp_agent_discovery.rs:91, :94, :97, :98)<br/>but never written by the poll loop (only successful_discoveries :367,<br/>failed_discoveries :383 and last_discovery_time :368 are), so<br/>build_discovery_frame now derives the four values from the frame<br/>itself, total_discoveries as successful plus failed and the rest from<br/>the filtered servers and agents<br/>(multi_mcp_websocket_handler.rs:393-402), and the stored fields stay<br/>dead
    end
```

## VC-05.15 `solid_proxy_handler` — pod lifecycle, DID resolution, double-auth DIVERGENCE
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant SC as scope /solid<br/>src/handlers/solid_proxy_handler.rs:1773 configure_routes (feature solid-pod-embed), scope :1781
    participant H as solid_proxy_handler<br/>solid_proxy_handler.rs:311 handle_solid_proxy, :1331 init_pod_nip98
    participant RG as RbacGate<br/>/api scope wrap src/main.rs:1137 (scope :1126, solid mounted :1206)
    participant SD as SolidPodState::extract_user_identity<br/>solid_proxy_handler.rs:188 own NIP-98 verification

    Note over SC: env SOLID_DATA_ROOT :130, SOLID_PROXY_SECRET_KEY :133, SOLID_ALLOW_ANONYMOUS :137
    Note over SC: ADR-2067 — WITHOUT solid-pod-embed the twin configure_routes :1821 registers<br/>NOTHING (doc :1810-1819): /solid/*, /.well-known/did.json and /did/* all 404. The<br/>503-stub route tree this diagram used to show is gone, and so are the individual<br/>stub twins — each leaves a removal note (handle_solid_proxy :382, solid_health_check :1630,<br/>init_pod_nip98 :1326-1327, SolidPodState::new :180-183)
    C->>SC: GET /health :1783 -> solid_health_check :1614
    SC-->>C: 200 {status:healthy, backend:solid-pod-rs, data_root}
    C->>SC: GET /.notifications :1785-1788 -> handle_solid_notifications_ws :1588 (solid-0.1 WS protocol)
    C->>SC: POST /pods :1790 -> create_pod :1186
    C->>SC: GET /pods/check :1791 -> check_pod_exists :1244
    C->>SC: POST /pods/init :1792 -> init_pod :1285
    rect rgb(255,235,235)
    Note over C,SD: DIVERGENCE — /api/solid/pods/init-nip98 sits inside the /api scope<br/>(main.rs:1206 configure_solid_routes, under scope :1126), so it is DOUBLE-authenticated (see VC-03.15)
    C->>RG: POST /api/solid/pods/init-nip98 solid_proxy_handler.rs:1793 (Authorization: Nostr event)
    RG->>RG: RbacGate verify_access(WriteGraph) — mutating method under /api
    alt RbacGate denies
        RG-->>C: 401/403
    else Ok
        RG->>H: init_pod_nip98(req, state) solid_proxy_handler.rs:1331
        H->>SD: extract_user_identity(req) :1332 — SECOND, fully independent NIP-98 re-verification
        alt SD returns None (bad/missing token)
            SD-->>C: 401 {error:"NIP-98 authentication required"} :1335-1338
        else Some(identity)
            H->>H: PublicKey::from_hex(identity.pubkey).to_bech32() -> npub :1343-1361
            H->>H: ensure_pod_exists(state, npub, pubkey, pod_base_url) :1148 (call :1366)
            H-->>C: 200 {pod_url, webid:structure.profile, created, structure, npub} :1369-1375
        end
    end
    end
    C->>SC: any method /{tail:.*} :1795-1799 plus PATCH :1800-1803 -> handle_solid_proxy :311 (LDP CRUD)
    Note over H: handle_solid_proxy auth flow doc :306-309 — NIP-98 header -> WAC, else<br/>SOLID_ALLOW_ANONYMOUS -> public ACL check, else 401
    C->>SC: GET /.well-known/did.json :1806 -> handle_did_wellknown :1639
    C->>SC: GET /did/{tail:.*} :1807 -> handle_did_proxy :1677 -> solid_pod_rs::interop::did_nostr::did_nostr_document
    Note over H: full pod/LDP internals (storage backend, ACL, containers) — see VC-26
```

## VC-05.16 `speech_socket_handler` — `/ws/speech`, ADR-2075 post-upgrade NIP-98
```mermaid
sequenceDiagram
    autonumber
    participant C as Client (browser voice / XR)
    participant H as speech_socket_handler<br/>src/handlers/speech_socket_handler.rs:989 (route src/main.rs:1114)
    participant SS as SpeechSocket actor<br/>speech_socket_handler.rs:82 struct, new() :103, Actor::started :479
    participant NS as NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs

    Note over H: RESOLVED — this upgrade no longer authenticates. ADR-2075 (doc :994-1001)<br/>moved the check AFTER the upgrade because browsers cannot set WebSocket headers,<br/>so the old presence-only Bearer/?token= gate rejected the browser voice client<br/>outright. There is no 401 on this path any more
    C->>H: GET /ws/speech (no credential required at upgrade)
    H->>H: connection_url = utils::auth::nip98_request_url(req) :1002 — the NIP-98 `u` tag
    H->>H: dev_bypass_ok = dev_bypass_permitted(req) under cfg(debug_assertions, dev-auth) :1004-1005, with no cfg(not) fallback binding any more
    H->>SS: SpeechSocket::new(socket_id "speech_{uuid}", app_state, connection_url, dev_bypass_ok) :1008-1014 - the unused hybrid_manager None argument is gone and dev_bypass_ok is passed only under the same cfg :1012-1013
    H->>SS: ws::start(socket, req, stream) :1016
    alt Err
        SS-->>C: start failure logged, propagated as actix_web::Error :1021-1024
    else Ok
        SS-->>C: 101 Switching Protocols :1017-1020
    end
    SS->>SS: Actor::started :479 — start_heartbeat, then run_later(AUTH_DEADLINE) :486
    Note over SS: ADR-2075 — AUTH_DEADLINE = 30s (:22, doc :19-21). A socket still carrying<br/>pubkey:None at the deadline is told "authentication deadline exceeded" and stopped<br/>(:487-500), so an unauthenticated peer cannot hold a broadcast subscription open
    C->>SS: text {"type":"authenticate","event":"<base64 NIP-98>"} -> handle_authenticate :155
    alt VISIONCLAW_DEV_MODE full bypass (dev builds only, :165-174)
        SS-->>C: {type:authenticate_success, pubkey} :172
    else dev-session-token without DEV_AUTH_LOOPBACK
        SS-->>C: {type:authenticate_error, "dev-session-token not permitted"} :182-190
    else no `event` field
        SS-->>C: {type:authenticate_error, "authenticate requires an `event` field"} :203-210
    else event present
        SS->>NS: verify_nip98_auth("Nostr {event_b64}", connection_url, "GET", None) :221-223
        alt Ok(user)
            NS-->>SS: pubkey
            SS->>SS: self.pubkey = Some(pubkey) :236
            SS-->>C: {type:authenticate_success, pubkey} :237
        else Err or no NostrService configured
            NS-->>SS: None :226-231
            SS-->>C: {type:authenticate_error, ...} :239-246
        end
    end
    C->>SS: any other typed frame before authenticate
    SS->>SS: reject_unauthenticated :133 (gate call :631) — pubkey.is_some() short-circuits :134
    SS-->>C: {type:error, "authentication required: send {type:authenticate,...} first"} :142-149
    Note over SS: voice internals (governed-voice command dispatch, clarification turns,<br/>ElevationActor VoiceTranscript hookup) — see VC-35
```

## VC-05.17 ADR-2006 human-approval journey — handler side, two disconnected producers, one store
```mermaid
sequenceDiagram
    autonumber
    participant V as voice/RunCycle trigger<br/>src/actors/elevation_actor.rs:1206 Handler~RunCycle~ (VC-02 internals)
    participant EA as ElevationActor/DecisionElevationActor<br/>src/actors/elevation_actor.rs:116, decision_elevation_actor.rs:128 (VC-02)
    participant FR as forum kind-31403<br/>src/services/acsp/client.rs:25 CaseDecision{event_id,created_at}
    participant DB as SqliteEnrichmentRepository::record_decision<br/>StoredDecision table — SHARED sink
    participant D as decide/decide_as_operator -> apply_decision<br/>src/handlers/enrichment_proposals_handler.rs:424 decide, :454 decide_as_operator, :473 apply_decision
    participant BI as broker_inbox_handler::inbox<br/>src/handlers/broker_inbox_handler.rs:220

    Note over V,FR: no HTTP handler in this file (decision_handler, enrichment_proposals_handler,<br/>broker_inbox_handler) ever sends a message to ElevationActor or DecisionElevationActor —<br/>grep across src/handlers finds zero references to either actor type
    V->>EA: RunCycle / VoiceTranscript (actor-internal, see VC-02)
    EA->>FR: publish ActionRequest kind-31402, poll PollPrs for the signed kind-31403 reply
    Note over FR: ADR-2013 — AcspClient signs with its OWN Keys (field acsp/client.rs:70,<br/>let keys = Keys::new(secret_key) :77, publish :101 — mirrors nostr_bridge.rs:65<br/>sign_with_keys) — the panel event carries the PANEL's authority, never the admin's key
    FR-->>EA: CaseDecision{case_id,action,responder_pubkey,event_id,created_at}
    EA->>EA: decision_record(&CaseDecision) :1694 — correlation on event_id when present
    EA->>DB: repo.record_decision(StoredDecision{decision_event_id:Some(event_id), decision_created_at_s:Some(...)}) :1443, :1516, :1523
    Note over EA,DB: this producer DOES retain signed-event correlation — decision_event_id is<br/>set from the signed event id (elevation_actor.rs:1749), rationale :1710-1719
    D->>DB: repo.record_decision(StoredDecision{decision_event_id:None,...}) :543-560 built, :561 written — REST path, no forum event to correlate
    Note over D,DB: DIVERGENCE — the agentbox/operator/git-bridge REST path (VC-05.7) writes<br/>decision_event_id:None every time, since it never carries a signed 31403 event
    BI->>DB: store::all() / store::get(id) — reads the SAME table both producers wrote to
    BI-->>BI: projects EITHER kind of row into the SAME BrokerCase shape, indistinguishable to the bridge
    Note over V,BI: DIVERGENCE (ADR-2006 closeout, docs/BASELINE-architecture.md:359) —<br/>"the retained domain kernel's presence does not prove integration into the elevation<br/>actor or inbox DTO. Current source review does not certify a complete human-approval journey."<br/>Verified precisely: the two producers only converge at the SQLite table, no handler route<br/>calls into either actor, and case authority/failure/restart receipts are not modelled here
```

## VC-05.18 `src/domain/broker/` verification — BrokerActor never merged (BASELINE-architecture.md:303)
```mermaid
flowchart TB
    KERNEL["src/domain/broker/ — storage-agnostic kernel (ADR-130 Decision 2)<br/>broker/mod.rs:1-43, broker_case.rs 490L, broker_decision.rs 437L, precedent_registry.rs 101L"]
    BC["BrokerCase aggregate<br/>CaseCategory, SubjectKind, ShareState (Private to Team to Mesh)"]
    BD["DecisionOrchestrator + DecisionOutcome<br/>six canonical outcomes, ShareTransitionPlan"]
    PR["PrecedentRegistry"]
    KERNEL --> BC
    KERNEL --> BD
    KERNEL --> PR
    ACSP["stateless ACSP producer<br/>src/services/acsp/mod.rs — kinds 31400-31405, AcspClient::publish"]
    ABSENT["src/actors/broker_actor.rs — VERIFIED ABSENT<br/>grep -r finds no file, no Neo4j adapter under src/adapters/ (listed: sqlite_*, oxigraph_*, actix_*)"]
    KERNEL -.->|"used by derive_kernel_decision (VC-05.7)"| CALLER["enrichment_proposals_handler::apply_decision"]
    KERNEL -.->|"used by decision_record (VC-05.17)"| CALLER2["elevation_actor.rs"]
    N1["DIVERGENCE (docs/BASELINE-architecture.md:304) — BrokerActor never merged, main uses<br/>a stateless ACSP producer + this cherry-picked storage-agnostic domain broker kernel.<br/>Confirmed against source: broker/mod.rs:11-18 states the crashbug BrokerActor + its Neo4j<br/>transport were deliberately left behind"]
    ABSENT --- N1
    ACSP --- N1
    N2["DOC-DRIFT — BASELINE-architecture.md:305 sizes the kernel at ~936 LOC, but the four files<br/>measure 1,071 lines today (490 + 437 + 101 + 43). ADR-2006 Verification (accepted) otherwise<br/>matches: src/services/acsp/mod.rs documents the producer, src/domain/broker/ holds the four<br/>listed files, broker_actor.rs and the neo4j adapters are absent"]
    KERNEL --- N2
```
