---
id: VC-03
title: Request lifecycle, identity and the RBAC lattice
area: visionclaw
governing:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/docs/SECURITY-profiles.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2003, ADR-2009, ADR-2010, ADR-2011, ADR-2012, ADR-2013, ADR-2026, ADR-2039, ADR-2043, ADR-2044, ADR-2141]
sources:
  - ../project/client/src/services/signedWebSocketProtocols.ts
  - ../project/client/src/services/nostrAuthService.ts
  - ../project/src/main.rs
  - ../project/src/middleware/mod.rs
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/middleware/public_demo.rs
  - ../project/src/middleware/rate_limit.rs
  - ../project/src/middleware/timeout.rs
  - ../project/src/utils/validation/middleware.rs
  - ../project/src/utils/validation/mod.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/utils/auth.rs
  - ../project/src/utils/nip98.rs
  - ../project/src/services/nostr_service.rs
  - ../project/src/services/nostr_identity_verifier.rs
  - ../project/src/services/nostr_bridge.rs
  - ../project/src/services/role_store.rs
  - ../project/src/models/rbac.rs
  - ../project/src/handlers/nostr_handler.rs
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/crates/visionclaw-domain/src/utils/visibility_filter.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/handlers/fastwebsockets_handler.rs
  - ../project/src/handlers/client_messages_handler.rs
  - ../project/src/handlers/mcp_relay_handler.rs
  - ../project/src/handlers/multi_mcp_websocket_handler.rs
  - ../project/src/handlers/speech_socket_handler.rs
  - ../project/src/handlers/solid_proxy_handler.rs
  - ../project/src/actors/client_filter.rs
  - ../project/src/uri/mod.rs
  - ../project/src/config/security_profile.rs
verified_commit: 8692b12ba987ecb87165566beb7c320a2446e4b4
---

## VC-03.1 REST request end-to-end — nginx to handler, real middleware order
```mermaid
sequenceDiagram
    autonumber
    participant NG as nginx (port 3001)
    participant AC as actix HttpServer<br/>src/main.rs:959-1243
    participant LG as Logger<br/>wrap #1 src/main.rs:1048
    participant CO as cors<br/>wrap #2 src/main.rs:1049
    participant CP as Compress<br/>wrap #3 src/main.rs:1050
    participant TO as TimeoutMiddleware<br/>wrap #4 src/main.rs:1051
    participant SC as scope /api<br/>src/main.rs:1126
    participant PD as PublicDemoGuard::from_env<br/>src/main.rs:1130
    participant RG as RbacGate::from_env<br/>src/main.rs:1137
    participant RL as RateLimit::per_minute 60<br/>src/main.rs:1144 (scope /api/settings only)
    participant H as route handler

    Note over LG,RG: actix applies .wrap() in REVERSE registration order at request time<br/>last .wrap() call = outermost layer that sees the request first
    Note over LG,TO: registration order in main.rs is Logger,cors,Compress,TimeoutMiddleware (src/main.rs:1048-1051)<br/>so the REAL request-time order is TimeoutMiddleware,Compress,cors,Logger,then routing
    NG->>AC: HTTP request
    AC->>TO: enter (outermost of the four)
    TO->>TO: get_timeout(path) — default 30s, override 600s for "/api/admin/sync" (src/main.rs:1051-1053)
    Note over TO: TimeoutConfig::new(30s).with_override(...) constructed src/main.rs:1052-1053 —<br/>get_timeout (timeout.rs:37-41) looks up an exact-path HashMap (endpoint_overrides) —<br/>no prefix/glob matching, so only "/api/admin/sync" itself gets 600s
    Note over LG,TO: DEBT: src/middleware/validation.rs (the old ValidationMiddleware) is deleted, so no request-size or content-type<br/>layer exists in this .wrap() chain. The nearest remaining guard, RequestSizeLimit<br/>(src/utils/validation/middleware.rs:17, default cap MAX_REQUEST_SIZE 16MiB at src/utils/validation/mod.rs:14),<br/>is not registered by any .wrap() in src/main.rs:1048-1051
    TO->>CP: enter
    CP->>CO: enter
    CO->>LG: enter
    LG->>SC: enter /api scope
    rect rgb(225,225,245)
    SC->>RG: RbacGate, wrapped LAST at src/main.rs:1137 so it is the outermost of the two /api wraps
    alt required_level present (not public route)
        RG->>RG: verify_access then record the identity in request extensions (see VC-03.6)
    else public route
        RG->>RG: pass straight through
    end
    RG->>PD: PublicDemoGuard, wrapped first at src/main.rs:1130 so it sees the request after the gate
    Note over PD: PublicDemoGuard::from_env() reads PUBLIC_DEMO ONCE at app start<br/>(public_demo.rs:49-53) — not re-read per request, so toggling the env var at<br/>runtime has no effect until restart. Truthy tokens (trim+lowercase):<br/>"read-only","readonly","1","true","on" (:26-31). DEFAULT OFF — unset means inert<br/>passthrough (:11-12)
    alt PUBLIC_DEMO read-only AND method not GET/HEAD/OPTIONS
        PD-->>NG: 403 read_only_demo
    end
    PD->>H: call handler
    end
    alt handler exceeds timeout_duration
        TO-->>NG: 504 Gateway Timeout "Request to {path} timed out after {ms}ms" (src/middleware/timeout.rs:120-124)
    end
    H-->>NG: HttpResponse
```

## VC-03.2 WebSocket upgrade lifecycle — every upgrade route, how each carrier is attributed
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant WS as socket_flow_handler<br/>src/handlers/socket_flow_handler/http_handler.rs:49
    participant NS as NostrService::get_session<br/>src/services/nostr_service.rs:611
    participant NV as NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs:648

    Note over WS: routes registered src/main.rs:1111-1119 — /wss, /wss/agent-events,<br/>/ws/speech, /ws/mcp-relay, /ws/client-messages, /ws/presence
    C->>WS: GET /wss (Upgrade: websocket)
    WS->>WS: WEBSOCKET_RATE_LIMITER.is_allowed(client_ip), else 429 (http_handler.rs:56-58)
    alt Origin header present
        WS->>WS: check against CORS_ALLOWED_ORIGINS / is_same_host (http_handler.rs:69-88)
        alt origin not allowed
            WS-->>C: 403 Forbidden "Origin not allowed" (http_handler.rs:113-118)
        end
    else Origin missing
        alt debug/dev-auth build AND ALLOW_INSECURE_DEFAULTS
            WS->>WS: allow_missing_origin = insecure_allowed (http_handler.rs:126-127)
        else release build
            WS-->>C: 400 BadRequest "Origin header required" (http_handler.rs:139)
        end
    end
    WS->>WS: read Sec-WebSocket-Protocol, websocket_auth_header (http_handler.rs:146-154)
    alt signed NIP-98 subprotocol present
        WS->>NV: verify_nip98_auth(header, signed_upgrade_url, GET) single-use (http_handler.rs:161)
        alt verified
            NV-->>WS: NostrUser { pubkey, is_power_user }
            WS->>WS: signed_user = Some(user) (http_handler.rs:162)
        else invalid or replayed
            WS-->>C: 401 "Invalid or replayed WebSocket signature" (http_handler.rs:163-166)
        end
    end
    opt signed_user is still None
        WS->>WS: token = Authorization Bearer header (http_handler.rs:179-184)
        Note over WS: DOC-DRIFT (was "DEPRECATED ADR-2044, kept for XR/native clients") — that premise is gone.<br/>ADR-2058 confined ?token= to dev/dev-auth builds only (http_handler.rs:187-202,<br/>fastwebsockets_handler.rs:243, client_messages_handler.rs:145, mcp_relay_handler.rs:482,<br/>multi_mcp_websocket_handler.rs:819) — a RELEASE build actively REJECTS it and logs a<br/>SECURITY warning, then reads the header alone (http_handler.rs:204-214, fastwebsockets_handler.rs:261-263,<br/>mcp_relay_handler.rs:498-500, multi_mcp_websocket_handler.rs:835-837). The code's own<br/>rationale for the old "XR/native clients cannot set headers" claim is that it was WRONG —<br/>neither xr-client/rust/src/ nor client/src/services/ ever used a token= query<br/>(client_messages_handler.rs:120-126) — the Godot client authenticates post-connect via the<br/>NIP-98 authenticate envelope instead. filter_auth.rs:138 is unrelated to this — that is a<br/>WS MESSAGE BODY legacy fallback, not an upgrade query param. speech_socket_handler.rs has<br/>NO query path and does full NIP-98 verification (verify_nip98_auth, speech_socket_handler.rs:222).
        alt token present and non-empty
            WS->>NS: get_session(token) (http_handler.rs:222)
            alt session found
                NS-->>WS: NostrUser { pubkey, is_power_user }
                WS->>WS: signed_user = session (http_handler.rs:262)
            else session not found
                alt debug/dev-auth build AND insecure_allowed
                    WS->>WS: warn, allow anyway with no identity (http_handler.rs:226-231)
                else release build
                    WS-->>C: 401 Unauthorized "Invalid or expired authentication token" (http_handler.rs:250-251)
                end
            end
        else token absent
            alt release build
                WS-->>C: 401 "Authentication required for WebSocket connections" (http_handler.rs:321-322)
            else dev build AND ALLOW_INSECURE_DEFAULTS
                WS->>WS: continue unauthenticated (anonymous session — visibility filter drops private nodes, see VC-03.9) (http_handler.rs:308-312)
            end
        end
    end
    WS->>WS: ws_server = SocketFlowServer::new(...) (http_handler.rs:384)
    WS->>WS: if signed_user, set ws_server.pubkey and ws_server.is_power_user (http_handler.rs:412-415)
    Note over WS: RESOLVED 0d636fd0e (S-7f) — the validated Authorization-header session used to be<br/>checked and then dropped, so a RELEASE build set ws_server.pubkey and .is_power_user only on the<br/>signed-subprotocol path and a header-session client stayed anonymous after authenticating.<br/>signed_user = session (http_handler.rs:262) now feeds the same assignment as the signed path<br/>(http_handler.rs:412-415), so both carriers are attributed at upgrade and the pubkey visibility<br/>filter in VC-03.9 sees the header-session client's identity.
    opt debug/dev-auth build only
        WS->>NS: second ?token= extraction, get_session again (http_handler.rs:371-379, http_handler.rs:430-440)
        Note over WS: DEBT — redundant since 0d636fd0e: the first extraction (http_handler.rs:187-202) already<br/>routes a query token through get_session into signed_user, so this pass re-sets pubkey for<br/>the same token (http_handler.rs:433-434). Absent from release (http_handler.rs:381-382).
    end
    Note over NS: RESOLVED ADR-2044 — get_session (nostr_service.rs:611) now enforces the SAME<br/>AUTH_TOKEN_EXPIRY window as validate_session (nostr_service.rs:510) through one shared rule<br/>session_is_fresh(last_seen, now, token_expiry) (nostr_service.rs:634). It previously had NO expiry check,<br/>so a WS token outlived its REST equivalent indefinitely. Empty tokens are rejected before<br/>lookup and a future last_seen (clock stepped back) is stale, not an unbounded lease.
    WS-->>C: 101 Switching Protocols (http_handler.rs:447-450)
    C->>WS: WS frames (binary positions / control JSON)
```

## VC-03.3 `AuthenticatedUser` extractor — settings API dual-auth
```mermaid
sequenceDiagram
    autonumber
    participant R as request
    participant FR as AuthenticatedUser::from_request<br/>src/settings/auth_extractor.rs:93
    participant DB as try_dev_bypass<br/>src/settings/auth_extractor.rs:21 dev / :64 release stub
    participant NS as NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs:648

    R->>FR: from_request(req)
    FR->>DB: try_dev_bypass(req)
    alt debug_assertions or feature dev-auth
        alt dev_full_bypass_active() (VISIONCLAW_DEV_MODE)
            DB-->>FR: Some(AuthenticatedUser{ DEV_MODE_PUBKEY, is_power_user:true })
        else Authorization == "Bearer dev-session-token"
            alt dev_bypass_permitted(req) (loopback + DEV_AUTH_LOOPBACK=1)
                DB-->>FR: Some(AuthenticatedUser{ pubkey: X-Nostr-Pubkey or "dev-user", true })
            else
                DB-->>FR: None (warn, rejected)
            end
        else
            DB-->>FR: None
        end
    else release build
        DB-->>FR: None (stub, auth_extractor.rs:62-66)
    end
    alt dev bypass returned Some
        FR-->>R: 200 Ok(user)
    end
    FR->>FR: app_data::<NostrService> lookup
    alt NostrService missing
        FR-->>R: 401 "Authentication service unavailable" (:109)
    end
    FR->>FR: reuse the identity RbacGate or RequireAuth left in request extensions, get_user(pubkey), else 401 "Authenticated user unavailable" (auth_extractor.rs:114-128)
    FR->>FR: Authorization header present and str-decodable?
    alt header absent
        FR-->>R: 401 "Missing authorization token" (:143)
    else header not valid UTF-8
        FR-->>R: 401 "Invalid authorization header" (:137)
    end
    alt header starts with "Nostr "
        FR->>NS: verify_nip98_auth(header, url, method, None)
        alt Ok(user)
            FR-->>R: 200 Ok(AuthenticatedUser{ pubkey, is_power_user })
        else Err(e)
            FR-->>R: 401 "NIP-98 auth failed: {e}" (:166)
        end
    else header starts with "Bearer "
        FR->>FR: pubkey = X-Nostr-Pubkey header
        alt X-Nostr-Pubkey missing
            FR-->>R: 401 "Missing pubkey" (:192)
        else header not UTF-8
            FR-->>R: 401 "Invalid pubkey header" (:187)
        end
        Note over FR: legacy Bearer requires validate_session(pubkey, token)<br/>auth_extractor.rs:227-228 - default-off migration flag applies
    else unrecognised prefix
        FR-->>R: 401 "Invalid authorization format" (:177)
    end
```

## VC-03.4 NIP-98 verification (kind 27235) — fixed order, single-use replay claim
```mermaid
sequenceDiagram
    autonumber
    participant H as caller<br/>auth.rs:191 / auth_extractor.rs:93
    participant NS as NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs:648
    participant V as validate_nip98_token<br/>src/utils/nip98.rs:428
    participant RC as REPLAY_CACHE<br/>src/utils/nip98.rs:221 Mutex~HashMap~

    Note over V,RC: INVARIANT (IDENTITY-authority-chain.md #1) — order is fixed:<br/>freshness -> tag match (host-checked) -> signature -> replay claim LAST.<br/>The claim must never precede signature verification.
    H->>NS: verify_nip98_auth(auth_header, url, method, body)
    NS->>NS: the caller reconstructs the url from X-Forwarded-Proto/-Host behind a TLS proxy (nip98_request_url, auth.rs:170-188)
    NS->>V: validate_nip98_token(token, expected_url, expected_method, body)
    V->>V: base64/UTF-8/JSON decode (nip98.rs:451-459)
    alt decode fails
        V-->>NS: InvalidBase64 / InvalidUtf8 / InvalidJson
    end
    V->>V: kind == 27235 HTTP_AUTH_KIND (const :20, check :462-464)
    alt kind mismatch
        V-->>NS: InvalidKind(got)
    end
    V->>V: age = now - created_at, symmetric window TOKEN_MAX_AGE_SECONDS=60 (:169, checks :476-483)
    alt age > 60
        V-->>NS: TokenExpired(age)
    else age < -60
        V-->>NS: TokenFromFuture(-age)
    end
    V->>V: extract "u" and "method" tags (:491-503), urls_match (nip98.rs:510, fn :683) host-checked
    alt "u" tag missing
        V-->>NS: MissingTag("u")
    else "method" tag missing
        V-->>NS: MissingTag("method")
    else url mismatch
        V-->>NS: UrlMismatch{expected,actual}
    else method mismatch (case-insensitive)
        V-->>NS: MethodMismatch{expected,actual}
    end
    V->>V: body binding by the route's declared BodyBinding policy (nip98.rs:525-557, ADR-2002)
    alt NoBody and a payload tag is present
        V-->>NS: UnexpectedPayloadHash (nip98.rs:529-530)
    else Required, body supplied
        V->>V: compute_payload_hash == SHA-256(body) (nip98.rs:535-539)
        alt hash mismatch
            V-->>NS: PayloadHashMismatch
        else tag absent
            V-->>NS: PayloadHashMissing (nip98.rs:540-541)
        end
    else Required, no body but a tag is present
        V-->>NS: PayloadHashMismatch (nip98.rs:545-546)
    else RequiredWhenTagged (validate_nip98_token default, legacy tolerance)
        V->>V: compare only when both body and tag are present (nip98.rs:551-556)
    end
    V->>V: Schnorr signature .verify() (nip98.rs:563-565)
    alt signature invalid
        V-->>NS: InvalidSignature
    end
    critical claim_signed_id(event.id, pubkey, now) under one Mutex lock
        V->>RC: admit_signed_in checks per-pubkey quota then claim_in atomically
        alt id already live within REPLAY_CACHE_TTL (2x60s=120s, :178)
            RC-->>V: TokenReplayed
        else signer already spent 1000 events in the120s window
            RC-->>V: PubkeyAdmissionExceeded, no live entry evicted
        else cache.len() >= REPLAY_CACHE_MAX_ENTRIES=100_000 after prune
            RC-->>V: ReplayCacheFull (generic auth failure upstream)
        else
            RC-->>V: Ok — id recorded with monotonic Instant
        end
    end
    Note over RC: INVARIANT #2 — replay-cache TTL (120s) >= 2x freshness window (60s),<br/>so no token still inside its freshness window can outlive its cache entry.<br/>Process-local scope — no cross-replica replay protection (SECURITY-profiles.md).
    V-->>NS: Ok(Nip98ValidationResult{pubkey,url,method,created_at,payload_hash})
    NS->>NS: get_or_create_user_from_pubkey, set is_power_user (nostr_service.rs:714)
    NS-->>H: NostrUser
```

## VC-03.5 Session-bearer realm (ADR-2009) — NIP-98 signing vs opaque session token
```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as /api scope (RbacGate)<br/>src/middleware/rbac_gate.rs
    participant SET as /api/settings (AuthenticatedUser)<br/>src/settings/auth_extractor.rs
    participant WSy as WS upgrade routes<br/>src/handlers/socket_flow_handler/http_handler.rs
    participant NS as NostrService<br/>src/services/nostr_service.rs

    Note over API,WSy: NIP-98 is default, legacy sessions exist only with explicit<br/>VISIONCLAW_LEGACY_SESSIONS=1 or true (nostr_service.rs:115)
    rect rgb(225,245,225)
    Note over C,API: REALM 1 — request-signing, verified ONCE per request (NIP-98 ids are single-use)
    C->>API: Authorization: Nostr <base64 kind-27235 event>
    API->>NS: verify_nip98_auth (verify_access, auth.rs:191), skipped when an outer layer already left the signer in request extensions (auth.rs:265-289)
    C->>SET: Authorization: Nostr <event> (primary path, auth_extractor.rs:148)
    end
    rect rgb(225,225,245)
    Note over C,WSy: REALM 2 — session-bearer, minted once, presented as an opaque token
    C->>NS: POST /api/auth/nostr (login) — verify_auth_event, issue Uuid::new_v4 (nostr_service.rs:428)
    NS-->>C: session token, TTL = AUTH_TOKEN_EXPIRY (default 3600s, nostr_service.rs:140)
    C->>WSy: Legacy Authorization Bearer token, release rejects query carrier
    WSy->>NS: get_session(token) — enforces AUTH_TOKEN_EXPIRY since ADR-2044 (see VC-03.2)
    C->>API: legacy X-Nostr-Pubkey + X-Nostr-Token headers, a missing one answers 401 (auth.rs:330-351)
    API->>NS: validate_session(pubkey, token) — DOES check now - last_seen <= token_expiry (nostr_service.rs:510-519)
    end
    Note over SET: Settings Bearer fallback validates the session before user lookup<br/>auth_extractor.rs:227-228 - caller pubkey alone is insufficient
    Note over API,SET: Graph browser signs the HTTP GET upgrade via NIP-98 subprotocol.<br/>Server verifies before upgrade and never echoes the credential.<br/>Legacy compatibility remains explicit for other consumers.
```

## VC-03.6 `RbacGate` decision sequence — env flags, allowlist, method classification
```mermaid
sequenceDiagram
    autonumber
    participant R as request
    participant RG as RbacGateMiddleware::call<br/>src/middleware/rbac_gate.rs:258
    participant RL as required_level<br/>src/middleware/rbac_gate.rs:147
    participant VA as verify_access<br/>src/utils/auth.rs:191

    RG->>RL: required_level(method, routed_path(req), public_reads) (rbac_gate.rs:262)
    Note over RL: routed_path is req.match_info().as_str() (rbac_gate.rs:69-71), the path the router matches with percent-escapes decoded,<br/>so an escaped /api/settings or /api/admin classifies as the real route and the raw URI path is no longer used
    Note over RG: INVARIANT - the gate classifies the routed path, never the raw URI, rbac_gate.rs:262
    RL->>RL: segments(path) split on "/" (:73)
    alt has_segment_prefix ["api","auth"] or ["api","client-logs"] or ["api","health"|"healthz"|"readyz"]
        RL-->>RG: None (public — allowlisted, rbac_gate.rs:56-60)
    else GET and segments == ["api","admin","rbac","whoami"]
        RL-->>RG: Some(AccessLevel::Authenticated) — any signed-in user may read their own role (:164-166, ER-26)
    else has_segment_prefix ["api","admin"]
        RL-->>RG: Some(AccessLevel::Admin) for EVERY other method and path (:168-169)
    else method.is_safe() (GET/HEAD/OPTIONS)
        alt RBAC_PUBLIC_READS=1 or true (public_reads_enabled, :135-141, fails closed on absence)
            RL-->>RG: None (public read)
        else
            RL-->>RG: Some(AccessLevel::ReadOnly)
        end
    else mutating method
        alt has_segment_prefix ["api","settings"]
            RL-->>RG: Some(AccessLevel::WriteSettings)
        else
            RL-->>RG: Some(AccessLevel::WriteGraph) (denies Viewer — Viewer maps to ReadOnly only)
        end
    end
    alt required is None
        RG->>R: pass through, no auth
    else required is Some(level)
        RG->>RG: app_data::<NostrService> lookup
        alt NostrService missing
            alt mode == Enforce
                RG-->>R: 401 Unauthorized
            else mode == Report
                RG->>R: pass through anyway
            end
        end
        RG->>VA: verify_access(req, nostr_service, level), verifies the NIP-98 event once and leaves the signer in request extensions for inner layers
        alt Ok(pubkey)
            RG->>RG: insert AuthenticatedUser{pubkey} into request extensions (:289-290)
            RG->>R: call handler
        else Err(deny_response)
            alt mode == Enforce (RBAC_GATE_MODE default)
                RG-->>R: deny_response as-is (401/403)
            else mode == Report
                RG->>RG: error!() log "would DENY ... allowed because auth is NOT enforced"
                RG->>R: call handler anyway
            end
        end
    end
    Note over RG: ADR-2011 — GateMode::from_env (:92-112): report requires debug_assertions<br/>OR RBAC_REPORT_MODE_ACK==today's UTC date, else falls back to Enforce with an error! log
```

## VC-03.7 Role lattice — Owner > Admin > Editor > Viewer, default-role resolution
```mermaid
flowchart TB
    OWN["Owner (4)<br/>src/models/rbac.rs:55 — full control"]
    ADM["Admin (3)<br/>manage users/settings, cannot touch Owner grants"]
    EDI["Editor (2)<br/>default_authenticated() — read + WriteGraph"]
    VIE["Viewer (1)<br/>ReadOnly only"]
    OWN --> ADM --> EDI --> VIE
    ACC1["AccessLevel::Admin<br/>to_access_level() rbac.rs:84"]
    ACC2["AccessLevel::Authenticated"]
    ACC3["AccessLevel::ReadOnly"]
    OWN -.->|"to_access_level"| ACC1
    ADM -.->|"to_access_level"| ACC1
    EDI -.->|"to_access_level"| ACC2
    VIE -.->|"to_access_level"| ACC3
    PREC["effective_role precedence — role_store.rs:359<br/>1. explicit row (RoleStore::get)<br/>2. Admin if POWER_USER_PUBKEYS match, no row<br/>3. else configured_default (RBAC_DEFAULT_ROLE_ENV)"]
    DEF["parse_default_role — role_store.rs:195<br/>unset or #quot;#quot; -> Editor (ADR-2010 compat default)<br/>#quot;editor#quot; -> Editor, #quot;viewer#quot; -> Viewer (case-insensitive, trimmed)<br/>anything else incl. #quot;admin#quot;/#quot;owner#quot; -> FAILS CLOSED to Viewer, error! logged"]
    ERR["any lookup/parse error (RoleStoreError)<br/>-> FAILS CLOSED to Viewer, role_store.rs:369-371"]
    PREC --> DEF
    PREC --> ERR
    N1["INVARIANT — RBAC_DEFAULT_ROLE can only NARROW (editor/viewer), never widen<br/>to admin/owner; an env typo cannot mass-grant elevated access"]
    DEF --- N1
    N2["CAN_ASSIGN lattice, rbac.rs:120 — Owner may assign anything;<br/>Admin may assign only target.level() lt Admin.level() (Editor/Viewer);<br/>Editor/Viewer may assign nothing"]
    OWN --- N2
```

## VC-03.8 `role_store` atomic mutations (ADR-2010) — grant/revoke/transfer
```mermaid
sequenceDiagram
    autonumber
    participant C as caller (Admin/Owner)
    participant RS as RoleStore::assign_checked<br/>src/services/role_store.rs:408
    participant TX as SQLite transaction<br/>tokio_rusqlite::Connection

    C->>RS: assign_checked(target, new_role, caller: CallerAuthority)
    critical single tx.transaction() scope — role_store.rs:423-478
        RS->>TX: tx = c.transaction()
        TX->>TX: resolve_caller_role_in_tx(caller, default_role) — re-reads CALLER's own row (role_store.rs:428)
        Note over TX: ADR-2010 — re-resolving the caller INSIDE the same tx closes the race where<br/>a concurrent demotion of the caller is not seen before the lattice check
        TX->>TX: effective_mutation_authority(admission_role, current)
        alt caller authority changed since admission
            TX-->>RS: TxOutcome (Forbidden/other) — no write
        end
        TX->>TX: read_role(tx, target) — existing role or ExistingRole::Invalid
        alt existing role fails to parse
            TX-->>RS: TxOutcome::InvalidExisting
        end
        alt !caller.can_assign(new_role)
            TX-->>RS: TxOutcome::Forbidden("{caller} may not assign role {new_role}")
        else target has existing role AND !caller.can_assign(existing)
            TX-->>RS: TxOutcome::Forbidden("may not modify a user currently holding {existing}")
        end
        alt existing == Owner AND new_role != Owner
            TX->>TX: COUNT(*) WHERE role='owner'
            alt owners <= 1
                TX-->>RS: TxOutcome::LastOwner
            end
        end
        TX->>TX: INSERT ... ON CONFLICT UPDATE role,assigned_by,updated_at
        TX->>TX: tx.commit()
        TX-->>RS: TxOutcome::Ok
    end
    RS-->>C: Ok(new_role) or RoleStoreError
    Note over RS: remove_checked (:500) runs the SAME atomic pattern — ADR-2010 —<br/>removal is NOT revocation: deleting the row drops the target to their AMBIENT<br/>authority (Admin if power-user, else configured_default), never denies outright.<br/>Genuine revocation requires an explicit Viewer assignment.
    Note over RS: ownerless resolution — bootstrap_owner_from_env (:643) grants Owner to<br/>RBAC_OWNER_PUBKEY idempotently at boot if unassigned — see VC-03.11 for the boot-fail path
```

## VC-03.9 `PUBKEY_VISIBILITY_FILTER` — per-client entitlement on the position stream and the initial load
```mermaid
sequenceDiagram
    autonumber
    participant EF as filter_enabled<br/>crates/visionclaw-domain/src/utils/visibility_filter.rs:38
    participant CC as ClientCoordinator pump_all / pump_one<br/>src/actors/client_coordinator_actor.rs:400 / :442
    participant VF as compute_private_opaque_ids<br/>crates/visionclaw-domain/src/utils/visibility_filter.rs:110
    participant SY as send_full_state_sync<br/>src/handlers/socket_flow_handler/types.rs:158
    participant PU as pubkey_visibility_filter_enabled<br/>src/handlers/socket_flow_handler/position_updates.rs:13

    Note over EF: parse_visibility_flag (:26-35) — default ON — only explicit<br/>"0"/"false"/"off"/"no" (case-insensitive, trimmed) disables.<br/>The env is read ONCE via OnceLock and cached for the process lifetime (:38-44),<br/>so toggling PUBKEY_VISIBILITY_FILTER at runtime has NO effect.
    PU->>EF: delegates, one definition for every path (position_updates.rs:13-15)
    CC->>EF: policy_on = filter_enabled() (client_coordinator_actor.rs:407)
    alt policy_on and node classification not yet loaded (!cx.nta.loaded)
        CC-->>CC: warn and send nothing — fail closed, "no classification" must not read as "nothing private" (:408-413)
    else classification in hand
        loop each client due a chunk (ADR-2141, until it acknowledges)
            CC->>VF: compute_private_opaque_ids(private_nodes, client.pubkey) — once per distinct pubkey per pump (:458-470)
            Note over VF: fail-closed — pubkey None (unauthenticated session) drops ALL private nodes<br/>is_dropped_for (visibility_filter.rs:93)
            VF-->>CC: hidden id set
            CC->>CC: visible(id) = client filter allows AND not hidden (:474-476), retain_nodes forgets ids no longer visible (:496)
            CC->>CC: plan the chunk from visible ids only, serialise V5 frame (:498-510)
        end
    end
    SY->>PU: visibility_filter_on (types.rs:166)
    SY->>SY: initial graph load keeps public nodes plus nodes whose owner_pubkey equals the caller (types.rs:257-276)
    Note over CC: INVARIANT: a frame is planned only from the visible set, so a private node the caller does not own cannot be in a position chunk, client_coordinator_actor.rs:474-476
    Note over SY: DRIFT: the comment at types.rs:255 says the filter is "Default OFF"<br/>while filter_enabled defaults ON (visibility_filter.rs:26-35)
    Note over CC: src/actors/client_filter.rs::recompute_filtered_nodes is a DIFFERENT filter —<br/>quality/authority-threshold + linked_page inclusion, NOT pubkey ownership.<br/>It feeds client.filter.filtered_node_ids, which pump_one ANDs with the entitlement set.
```

## VC-03.11 Dev bypass triple gate (ADR-2012 / ADR-2039) — nested conditions
```mermaid
sequenceDiagram
    autonumber
    participant R as request
    participant A as dev_full_bypass_active<br/>src/utils/auth.rs:116 dev / :129 release stub
    participant L as dev_bypass_permitted_for_addr<br/>src/utils/auth.rs:139
    participant B as boot: enforce_release_env_hygiene<br/>src/main.rs:118 real / :169 stub

    Note over B: gate 0 (compile-time) — this whole codepath is #[cfg(any(debug_assertions,#quot —dev-auth#quot —))]<br/>compiled OUT of a release binary entirely — release stub always returns false/None
    alt release build (no debug_assertions, no dev-auth feature)
        B->>B: enforce_release_env_hygiene runs at boot (src/main.rs:209)
        alt VISIONCLAW_DEV_MODE present (any value, presence not truthiness)
            B-->>R: FATAL eprintln, std::process::exit(2) (SUSPECT_ENVS, :129-133,:161)
        end
        Note over A,L: dev_full_bypass_active and dev_bypass_permitted are unreachable stubs in this binary
    else debug or dev-auth build
        R->>A: dev_full_bypass_active() (VISIONCLAW_DEV_MODE)
        alt env == "1" or "true" (trim, case-insensitive)
            A-->>R: true — LAN-local FULL bypass, EVERY request granted DEV_MODE_PUBKEY, no signature/token/peer check (auth.rs:116-123)
        else
            A-->>R: false
            R->>L: Authorization == "Bearer dev-session-token"?
            alt header matches
                R->>L: dev_bypass_permitted_for_addr(peer)
                alt DEV_AUTH_LOOPBACK=1 or true AND peer.ip().is_loopback()
                    L-->>R: true — dev-admin as X-Nostr-Pubkey or "dev-user"
                else opt-in missing OR non-loopback peer OR no peer addr
                    L-->>R: false — falls through to real NIP-98/session auth
                end
            end
        end
    end
    Note over B: ADR-2039 — VISIONCLAW_DEV_MODE full bypass is PEER-AGNOSTIC by design (Docker SNAT<br/>hides the real LAN headset address) — the loopback+opt-in gate above applies only to the<br/>SEPARATE Bearer dev-session-token path, not to VISIONCLAW_DEV_MODE itself — three gates<br/>(compile-time, runtime opt-in, boot-refusal in release) substitute for a peer check there.
```

## VC-03.12 `RateLimit` middleware — sliding window per identifier
```mermaid
sequenceDiagram
    autonumber
    participant R as request
    participant RL as RateLimitService::call<br/>src/middleware/rate_limit.rs:235
    participant ST as RateLimitState<br/>Arc~RwLock~ HashMap~String,VecDeque~Instant~~~ (:92-98)

    RL->>RL: identifier chosen inline — use_user_id? AuthenticatedUser.pubkey : realip_remote_addr (:242-263)
    RL->>ST: state.write().await
    ST->>ST: cleanup(config) — periodic prune (rate_limit.rs:136)
    ST->>ST: check_and_record(identifier, config) (rate_limit.rs:110)
    ST->>ST: window_start = now - config.window — pop_front expired entries
    alt history.len() < max_requests
        ST->>ST: push_back(now)
        ST-->>RL: true
    else at limit
        ST-->>RL: false
    end
    alt not allowed
        RL-->>R: 429 ErrorTooManyRequests(message) (:289)
    else allowed
        RL->>R: service.call(req)
    end
    Note over RL: applied at src/main.rs:1144 as RateLimit::per_minute(60) on the /api/settings<br/>scope only (rate_limit.rs:171) — 60 requests / 60s window, keyed by realip (use_user_id off by default). RateLimit is Clone and its state sits behind an Arc (rate_limit.rs:84-87), so clones share one request history
```

## VC-03.15 Identity authority chain end-to-end — Nostr key to role
```mermaid
sequenceDiagram
    autonumber
    participant K as secp256k1 keypair (client-held)
    participant U as uri::did_nostr<br/>src/uri/mod.rs:220
    participant NS as NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs:648
    participant SP as init_pod_nip98<br/>src/handlers/solid_proxy_handler.rs:1331
    participant RS as RoleStore::effective_role<br/>src/services/role_store.rs:359
    participant AL as AccessLevel<br/>src/utils/auth.rs:16

    K->>U: pubkey (64-char lowercase hex, Schnorr)
    U->>U: did_nostr(pubkey) -> "did:nostr:{pubkey}" (DID_NOSTR_PREFIX, uri/mod.rs:47)
    Note over U: parse() round-trips did:nostr:* to ParsedUri::DidNostr (uri/mod.rs:520-528)<br/>cross_from_agentbox treats an already-converged did:nostr:* as identity, structural passthrough (:811-818)
    K->>NS: sign kind-27235 NIP-98 event (see VC-03.4)
    NS-->>K: NostrUser{pubkey, is_power_user}
    opt Solid pod provisioning
        K->>SP: POST /api/solid/pods/init-nip98 (Authorization: Nostr ...)
        SP->>SP: extract_user_identity(req) — re-verifies NIP-98 (this route sits under /api,<br/>so RbacGate ALSO requires WriteGraph via verify_access before the handler runs)
        SP->>SP: PublicKey::from_hex(pubkey).to_bech32() -> npub
        SP->>SP: ensure_pod_exists(npub, pubkey, pod_base_url)
        SP-->>K: { pod_url, webid: structure.profile, npub } (:1369)
        Note over SP: GET /did/nostr:{pubkey} resolves a did+ld+json document via<br/>solid_pod_rs::interop::did_nostr::did_nostr_document (solid_proxy_handler.rs:1688,<br/>route registered :1807). Full pod/LDP detail: see VC-26.
    end
    K->>RS: canonicalise_pubkey(pubkey) (role_store.rs:154) then effective_role(pubkey, is_power_user)
    RS-->>K: UserRole (explicit row / power-user Admin / configured_default / fail-closed Viewer)
    RS->>AL: role.to_access_level() (rbac.rs:84-90)
    Note over AL: AccessLevel drives verify_access / RbacGate for every subsequent /api call.<br/>Client-side signing/identity storage: see VC-33.
```

## VC-03.16 ADR-2013 federation-delegation boundary — NOT implemented
```mermaid
flowchart TB
    U["user's own Nostr keypair<br/>signs NIP-98 directly (VC-03.4)"]
    SS["Service-signed — nostr_bridge.rs<br/>struct field keys: Keys (nostr_bridge.rs:29)<br/>Keys::new(secret_key) (nostr_bridge.rs:65)<br/>event.sign_with_keys(&self.keys) (nostr_bridge.rs:169)"]
    DA["Delegated-agent-signed (NIP-26)<br/>agent signs ON BEHALF OF a user with a<br/>verifiable delegation tag"]
    HOOK["would-hook point: validate_nip98_token<br/>src/utils/nip98.rs:428 — tag extraction loop (:491-500, u/method/payload only)<br/>a #quot;delegation#quot; tag is never read; no NIP-26 verifier exists in this crate"]
    U -->|"authority IS the user's own key"| REAL["real, implemented (auth.rs, nip98.rs)"]
    SS -->|"re-signs under the BRIDGE's key,<br/>not the user's — original authority NOT carried"| BRIDGE_REAL["real, implemented, but NOT delegation"]
    DA -.->|"DIVERGENCE — NOT WIRED"| HOOK
    N1["DIVERGENCE (IDENTITY-authority-chain.md #154-162) — NIP-26 delegation deferred to the<br/>unbuilt Phase 5. Until it lands, no request can be attributed to a user THROUGH an agent.<br/>Key custody/rotation (legacy ADR-081) and delegated admin (legacy ADR-094) are frozen 2026-07-03.<br/>Pod signing can fall back unsigned (legacy agentbox ADR-026)."]
    HOOK --- N1
    N2["ADR-2013 scope note — the enterprise/delegation deferral governs VisionClaw's own request<br/>realm only; it does not imply a single request-credential realm across other repositories'<br/>verifiers (Request-credential review, IDENTITY-authority-chain.md 2026-09-04)."]
    DA --- N2
```

2026-09-07 closeout: `settings/auth_extractor.rs` reuses the server-side
middleware identity extension. The NIP-98 token is consumed once within the
request; a new request presenting the same token still fails. The regression
test checks both outcomes and preserves the authenticated user's privilege bit.


## VC-03.17 Signed browser upgrade and legacy-session sunset

```mermaid
sequenceDiagram
    participant C as signedWebSocketProtocols<br/>signedWebSocketProtocols.ts:7
    participant Sign as signRequest<br/>nostrAuthService.ts:244
    participant H as socket_flow_handler<br/>http_handler.rs:49
    participant V as NostrService::verify_nip98_auth<br/>nostr_service.rs:648
    C->>Sign: Public HTTP(S) upgrade URL, GET, fresh nonce
    alt Signer absent or declines
        Sign-->>C: Fail before WebSocket construction
    else Signed event
        C->>H: visionclaw plus nostr.base64url event protocols
        H->>H: signed_upgrade_url delegates to utils::auth::nip98_request_url (http_handler.rs:44-46, auth.rs:170), the same URL REST NIP-98 checks (X-Forwarded-Proto and Host, else connection info)
        H->>V: verify_nip98_auth(header, signed_upgrade_url, GET) (http_handler.rs:161) then validate_nip98_token (nostr_service.rs:661): URL, method, signature, signer quota, replay
        alt Invalid or reused
            H-->>C: 401 before upgrade
        else Valid
            H-->>C: Upgrade and negotiate visionclaw only
        end
    end
    Note over H: Legacy sessions default off. Explicit compatibility<br/>does not reopen dev-token or query-token release bypasses.<br/>A legacy Authorization-header session that validates is carried into ws_server.pubkey and .is_power_user (http_handler.rs:262, http_handler.rs:412-415) - RESOLVED 0d636fd0e, see VC-03.2.
```
