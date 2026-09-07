---
id: AB-10
title: Ingress — nip98-proxy and the AoE door
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2002, ADR-2009, ADR-2010, ADR-2011, ADR-2013, ADR-2047, ADR-2080]
sources:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/config/nip98-proxy/README.md
  - ../project/agentbox/config/nip98-proxy/selftest.mjs
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/scripts/aoe-seed-sessions.mjs
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/middleware/auth.js
  - ../project/agentbox/scripts/ci/check-ports-loopback.sh
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/docs/adr/ADR-2002-aoe-token-auth-boundary.md
  - ../project/agentbox/docs/adr/ADR-2009-nip98-proxy-identity-boundary.md
  - ../project/agentbox/docs/adr/ADR-2010-bearer-gated-behind-nip98.md
  - ../project/agentbox/docs/adr/ADR-2011-hex-canonical-identity.md
  - ../project/agentbox/docs/adr/ADR-2013-loopback-publish-except-9096.md
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/voice/console/Caddyfile
  - ../project/agentbox/management-api/lib/agent-event-auth.js
  - ../project/agentbox/management-api/lib/action-plane.js
verified_commit: 2c521c5bb
---

## AB-10.1 Door inventory and port topology

```mermaid
flowchart TB
    LAN["LAN client"]
    P9096["nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:1154<br/>listen 0.0.0.0:9096"]
    AOE9095["aoe serve --auth token --behind-proxy<br/>agentbox/flake.nix:2353<br/>127.0.0.1:9095"]
    MGMT9090["management-api<br/>agentbox/management-api/server.js:1423,:47<br/>HOST 0.0.0.0 PORT 9090 in-container"]
    RELAY7777["nostr-rs-relay<br/>agentbox/flake.nix:1492,:1493<br/>binds relayCfg.bind, default 127.0.0.1:7777"]
    VOICE8444["voice cockpit Caddy origin<br/>docker-compose.voice.yml:40<br/>0.0.0.0:8444"]

    LAN -->|"9096:9096 SANCTIONED agentbox/scripts/ci/check-ports-loopback.mjs:94,:74-91 ADR-2013"| P9096
    P9096 -->|"default route, Authorization UNCONDITIONALLY replaced with daemon token agentbox/config/nip98-proxy/proxy.mjs:988-996"| AOE9095
    P9096 -->|"prefix /mgmt/ NIP98_PROXY_MGMT_UPSTREAM agentbox/config/nip98-proxy/proxy.mjs:407-413"| MGMT9090
    LAN -.->|"127.0.0.1:9090:9090 host publish agentbox/flake.nix:2692 not LAN-reachable"| MGMT9090
    LAN -->|"8443/8444 SANCTIONED docker-compose.voice.yml:40 agentbox/scripts/ci/check-ports-loopback.mjs:95-96"| VOICE8444
    VOICE8444 -->|"slash lo and slash docs ONLY - direct to management-api voice/console/Caddyfile:77-82"| MGMT9090
    VOICE8444 -->|"slash aoe, slash approvals, slash mgmt, slash dream, slash feed, slash bridge, slash nip07 - Authorization forwarded to the NIP-98 door voice/console/Caddyfile:54-75,:90-111"| P9096

N1["RESOLVED ADR-2047: not a breach - a decided, enumerated exposure. The LAN surface is TEN<br/>sanctioned publishes across five compose files, matched on the normalised host_ip/published/<br/>target/protocol tuple: 9096 sovereign ingress, voice 8443 and 8444, browsercontainer 5903<br/>8931 9222-to-9223, gui-tools 5905 9876 9877, xr-runtime 5904 - each cited at<br/>agentbox/scripts/ci/check-ports-loopback.mjs:93-104. 9096 is the sole IDENTITY ingress to the<br/>AoE plane; the others carry their own auth. The old framing understated the count as well."]
N2["RESOLVED ADR-2047: the stale --auth none bullet is gone. flake.nix:2687-2688 reads<br/>aoe serve, --auth token is NEVER published, matching the live supervisor command at<br/>flake.nix:2353, and INGRESS-identity now records the bullet as Resolved rather than<br/>open. The doc body's drifted verifyIdentity anchor (a stale 410-450 range) is<br/>corrected to proxy.mjs:626."]
N6["INVARIANT ADR-2013: sovereign_mesh.relay.expose does NOT open a LAN door. When true it adds<br/>ONE publish and that publish is loopback-pinned - 127.0.0.1:port:port at<br/>agentbox/flake.nix:2697-2698 - so the relay never reaches the SANCTIONED LAN list at<br/>check-ports-loopback.mjs:93-104. Default is expose=false, agentbox.toml:136."]
N3["INVARIANT ADR-2009: aoe serve binds 127.0.0.1 plus --behind-proxy - nip98-proxy is the sole<br/>IDENTITY ingress to :9095, nothing else may open that port<br/>agentbox/config/nip98-proxy/README.md:40-50"]
N4["RESOLVED ADR-2047: the compose-exposure qualification is superseded in the governing doc.<br/>The line-walker bypass is fixed - check-ports-loopback.sh is a wrapper that execs<br/>check-ports-loopback.mjs, a strict YAML reader for the compose subset, and anything outside that<br/>subset is REJECTED with file and line, never skipped. ADR-2013 stays partial for the DEPLOYMENT<br/>half only: overlay order, interpolation, external files and active-listener evidence.<br/>Receipt: agentbox/docs/estate-closeout/2026-09-05/adr-2013-ports-gate.json."]
N5["INVARIANT ADR-2013: the wrapper FAILS LOUDLY when the .mjs gate is missing - a copy of the<br/>wrapper without its gate exits 3 with an explicit message rather than looking like a pass<br/>agentbox/scripts/ci/check-ports-loopback.sh:38-42"]
```

## AB-10.2 proxy.mjs boot sequence — config, routes, verifier load

```mermaid
sequenceDiagram
    autonumber
    participant BOOT as proxy.mjs boot<br/>agentbox/config/nip98-proxy/proxy.mjs:96
    participant FS as Filesystem<br/>CONFIG_FILE / NOSTR_BRIDGE_PATH
    participant NB as loadNostrBridge<br/>agentbox/config/nip98-proxy/proxy.mjs:476
    participant LOG as log()<br/>agentbox/config/nip98-proxy/proxy.mjs:442

    BOOT->>BOOT: PORT, BIND, AOE_UPSTREAM parsed proxy.mjs:96-98
    BOOT->>BOOT: BREAK_GLASS = NIP98_PROXY_ALLOW_BEARER proxy.mjs:99
    BOOT->>BOOT: SESSION_TTL_S = NIP98_PROXY_SESSION_TTL default 43200 proxy.mjs:208
    break SESSION_TTL_S not a positive safe integer
        BOOT-->>BOOT: throw Error proxy.mjs:210
    end
    BOOT->>BOOT: SESSION_SECRET = NIP98_PROXY_SESSION_SECRET or crypto.randomBytes(32).toString(hex) proxy.mjs:212
    Note over BOOT: DIVERGENCE session secret is per-boot - NIP-07 sessions do not survive a proxy restart, by design agentbox/docs/INGRESS-identity.md known divergences bullet 5
    Note over BOOT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
    BOOT->>BOOT: NIP98_PROXY_ALLOWED_PUBKEYS split lowercased filtered proxy.mjs:218-221
    break any entry not 64-hex
        BOOT-->>BOOT: throw Error NIP98_PROXY_ALLOWED_PUBKEYS entries must be 64-character hex proxy.mjs:223
    end
    BOOT->>NB: loadNostrBridge() proxy.mjs:526
    alt NOSTR_BRIDGE_PATH explicit
        NB->>FS: existsSync(NOSTR_BRIDGE_PATH) proxy.mjs:484
        alt missing or exports no verifyNip98
            NB-->>LOG: error no fallback proxy.mjs:485,:494,:496
            NB-->>BOOT: null
        else
            NB-->>BOOT: NostrBridge proxy.mjs:492
        end
    else no explicit path
        loop source-tree then baked-image candidates proxy.mjs:501-510
            NB->>FS: existsSync(candidate)
            NB->>NB: require(candidate) proxy.mjs:514
        end
        NB-->>BOOT: NostrBridge or null proxy.mjs:512-523
    end
    alt NostrBridge is null
        BOOT-->>LOG: warn nostr-bridge unavailable NIP-98 verification DISABLED fail-closed proxy.mjs:528-534
        Note over BOOT: INVARIANT ADR-2009 - missing verifier rejects every NIP-98 token, only break-glass if configured survives
    else loaded
        BOOT-->>LOG: info nostr-bridge loaded proxy.mjs:491,:516
    end
    BOOT->>FS: readFileSync CONFIG_FILE default workspace/.agentbox/nip98-proxy-config.json proxy.mjs:352-358
    alt file absent
        FS-->>BOOT: FILE_CONFIG = routes [] allowedPubkeys [] proxy.mjs:360
    else present
        BOOT->>BOOT: JSON.parse, validate allowedPubkeys 64-hex proxy.mjs:363-371
        break malformed JSON or bad pubkey shape
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:375-376
        end
    end
    BOOT->>BOOT: parse NIP98_PROXY_ROUTES JSON array, normalizeRoute each entry proxy.mjs:382-387
    break not an array, or normalizeRoute throws e.g. bearer_env unset proxy.mjs:332-336
        BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:397-398
    end
    BOOT->>BOOT: FILE_CONFIG.routes appended, env route wins on prefix conflict proxy.mjs:389-392
    opt NIP98_PROXY_MGMT_UPSTREAM set and no existing slash mgmt slash route
        BOOT->>BOOT: push convenience route proxy.mjs:409-413
        break invalid URL
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:414-417
        end
    end
    BOOT->>BOOT: server.listen(PORT, BIND) proxy.mjs:1154
    BOOT-->>LOG: info nip98-proxy listening - bind, upstream, routes, nip98 enabled/DISABLED, breakGlass, nip07Sessions ttl and pinned-vs-per-boot, allowedPubkeys count proxy.mjs:1155-1177
```

## AB-10.3 verifyIdentity — full auth precedence

```mermaid
sequenceDiagram
    autonumber
    participant REQ as Request<br/>HTTP forward() or WS upgrade
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:626
    participant CTE as constantTimeEqual<br/>agentbox/config/nip98-proxy/proxy.mjs:540
    participant NB as NostrBridge.verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:459
    participant CAN as canonicalPubkey<br/>agentbox/config/nip98-proxy/proxy.mjs:241
    participant PA as pubkeyAllowed<br/>agentbox/config/nip98-proxy/proxy.mjs:227
    participant VST as verifySessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:561

    REQ->>VI: verifyIdentity(req, bearerToken, rawBody) proxy.mjs:626
    VI->>VI: authHeader = req.headers.authorization proxy.mjs:627
    alt BREAK_GLASS configured and a Bearer token or query bearerToken matches
        VI->>CTE: constantTimeEqual(token, BREAK_GLASS) proxy.mjs:636
Note over VI,CTE: DIVERGENCE (historical) break-glass bearer over the LAN - a single shared secret<br/>bypasses NIP-98 entirely while NIP98_PROXY_ALLOW_BEARER is set. See the RESOLVED note below -<br/>ADR-2027 (closeout 2026-09-05) bounds it with expiry + scope, still tracked as INGRESS-identity.md<br/>known divergences bullet 4 pending a doc update
        VI->>VI: RESOLVED ADR-2027 — breakGlassNotExpired() proxy.mjs:637
        alt expiry configured and elapsed, or malformed
            VI-->>REQ: ok false, reason break_glass_expired, audited by fingerprint proxy.mjs:638-646
        end
        VI->>VI: breakGlassScopeAllows(method, url) proxy.mjs:647
        alt NIP98_PROXY_BEARER_SCOPE configured and request outside it
            VI-->>REQ: ok false, reason break_glass_out_of_scope, audited by fingerprint proxy.mjs:648-656
        end
        Note over VI: SECURITY-profiles.md custody row Proxy break-glass bearer - captured at process start,<br/>ADR-2027 adds optional expiry + scope bounds and a per-use fingerprint audit trail (breakGlassUse<br/>counters), still an in-process receipt, not a durable audit store
        VI->>VI: breakGlassUse.accepted++, log warn break-glass USED with fingerprint proxy.mjs:657-669
        VI-->>REQ: ok true, pubkey NIP98_PROXY_BEARER_PUBKEY default break-glass, mode break-glass proxy.mjs:670
    else Authorization starts with Nostr
        VI->>NB: verifyNip98(authHeader, method, signedUrlFor(req), rawBody) proxy.mjs:680
        alt NostrBridge unavailable
            VI-->>REQ: ok false, reason nip98_verifier_unavailable proxy.mjs:676
        else verifyNip98 throws
            VI-->>REQ: ok false, reason nip98_verify_error proxy.mjs:681-682
        else result not valid
            VI-->>REQ: ok false, reason nip98_invalid proxy.mjs:698
        else result.valid true
            VI->>CAN: canonicalPubkey(result.pubkey) proxy.mjs:689
            alt not lowercase 64-hex
                VI-->>REQ: ok false, reason nip98_noncanonical_pubkey proxy.mjs:690-691
            else canonical
                VI->>PA: pubkeyAllowed(pubkey) proxy.mjs:693
                alt allowlist non-empty and pubkey missing
                    VI-->>REQ: ok false, reason pubkey_not_allowed proxy.mjs:693-694
                else
                    VI-->>REQ: ok true, pubkey, mode nip98 proxy.mjs:696
                end
            end
        end
    else Cookie carries agentbox_nip07_session
        VI->>VST: verifySessionToken(cookie) proxy.mjs:705
        VST->>CTE: constantTimeEqual(mac, expected) proxy.mjs:569
        VST->>PA: pubkeyAllowed(pubkey) proxy.mjs:570
        alt session valid, not expired, HMAC matches, and allowed
            VI-->>REQ: ok true, pubkey, mode nip07-session proxy.mjs:706-707
        else
            VI-->>REQ: ok false, reason session_invalid_or_expired proxy.mjs:709
        end
    else no credentials at all
        VI-->>REQ: ok false, reason no_credentials proxy.mjs:712
    end
```

## AB-10.4 verifyNip98 internals — header, tags, payload, signature, replay

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:680
    participant VN as verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:459
    participant TAG as getTag<br/>agentbox/mcp/servers/nostr-bridge.js:490
    participant SHA as crypto sha256<br/>agentbox/mcp/servers/nostr-bridge.js:525,:532
    participant SIG as verifyEvent nostr-tools<br/>agentbox/mcp/servers/nostr-bridge.js:542
    participant REPLAY as replay LRU<br/>agentbox/mcp/servers/nostr-bridge.js:97

    CALLER->>VN: verifyNip98(authHeader, method, url, rawBody) nostr-bridge.js:459
    break header missing or not Nostr-prefixed
        VN-->>CALLER: valid false, error missing or malformed Nostr header nostr-bridge.js:461
    end
    VN->>VN: event = JSON.parse(Buffer.from(encoded, base64).toString(utf8)) nostr-bridge.js:471
    break not valid base64 JSON
        VN-->>CALLER: valid false, error token is not valid base64 JSON nostr-bridge.js:473
    end
    break event.kind !== kinds.AUTH (27235) nostr-bridge.js:55,:476
        VN-->>CALLER: valid false, error expected kind 27235 got X nostr-bridge.js:477
    end
    break created_at not a number
        VN-->>CALLER: valid false, error missing created_at nostr-bridge.js:481
    end
    break abs(now - created_at) greater than VERIFY_NIP98_WINDOW_S 60s nostr-bridge.js:80,:485
        VN-->>CALLER: valid false, error event timestamp out of 60-second window nostr-bridge.js:486
    end
    VN->>TAG: getTag(method), getTag(u) nostr-bridge.js:497-498
    break method tag mismatch case-insensitive
        VN-->>CALLER: valid false, error method tag mismatch nostr-bridge.js:501
    end
    break url tag mismatch - urlTag not equal url and not endsWith(url)
        VN-->>CALLER: valid false, error url tag mismatch nostr-bridge.js:505
    end
    opt rawBody supplied - Finding 4 payload binding
        VN->>TAG: getTag(payload) nostr-bridge.js:520
        VN->>SHA: hex(sha256(rawBody)) nostr-bridge.js:525,:532
        break body non-empty and payload tag missing, or computed hash mismatch either branch
            VN-->>CALLER: valid false, error missing payload tag for request body OR payload hash mismatch nostr-bridge.js:523,:527,:534
        end
    end
    VN->>SIG: verifyEvent(event) BIP-340 schnorr constant-time nostr-bridge.js:543
    break signature invalid or verifyEvent throws
        VN-->>CALLER: valid false, error invalid Schnorr signature nostr-bridge.js:545,:548
    end
    VN->>REPLAY: _recordReplayId(event.id) nostr-bridge.js:555,:111
Note over REPLAY: REPLAY_TTL_MS = VERIFY_NIP98_WINDOW_S times 1000 = 60000ms,<br/>REPLAY_MAX_ENTRIES = 20000, keyed on event id, populated ONLY after signature verify so a<br/>forged token can never poison the cache nostr-bridge.js:95-97
    break id already seen inside the freshness window - live duplicate
        VN-->>CALLER: valid false, error replayed NIP-98 event id nostr-bridge.js:556
    end
    VN-->>CALLER: valid true, pubkey event.pubkey nostr-bridge.js:559
```

## AB-10.5 Request auth state machine

```mermaid
stateDiagram-v2
    [*] --> Unauthenticated
    Unauthenticated --> BearerAccepted: break-glass token constant-time match, unexpired, in scope proxy.mjs:632-636,:670
    Unauthenticated --> Nip98Verified: verifyNip98 valid, canonicalPubkey, pubkeyAllowed proxy.mjs:684-696
    Unauthenticated --> SessionCookieVerified: verifySessionToken valid and allowlisted proxy.mjs:704-707
    Unauthenticated --> Rejected302: browser GET, Accept text/html, auth failed proxy.mjs:919-926
    Unauthenticated --> Rejected401: auth failed, not an html GET proxy.mjs:928-933
    Unauthenticated --> FailClosed: NostrBridge null at boot, every Nostr header refused proxy.mjs:528-534,:676
    BearerAccepted --> Routed: routeFor(req.url) proxy.mjs:937,:1082
    Nip98Verified --> Routed: routeFor(req.url) proxy.mjs:937,:1082
    SessionCookieVerified --> Routed: routeFor(req.url) proxy.mjs:937,:1082
    Routed --> Upstream: default AoE route, daemon token replaces Authorization proxy.mjs:988-996
    Routed --> Upstream: named route, nip98 passthrough or bearer_env injection proxy.mjs:971-980
    Routed --> FailClosed: readAoeToken returned null, 503 ServiceUnavailable proxy.mjs:990-994,:1089-1094
    FailClosed --> [*]
    Rejected401 --> [*]
    Rejected302 --> [*]
    Upstream --> [*]
```

Every anchor above was re-derived in the 2026-09-07 pass; the previous set pointed
into the HANDSHAKE_PAGE string literal and into constantTimeEqual, not at the
transitions it named.

## AB-10.6 AoE board door — default upstream, daemon-token replacement

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser or client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:910
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:626
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:426
    participant TOK as readAoeToken<br/>agentbox/config/nip98-proxy/proxy.mjs:278
    participant FS as serve.url state file<br/>~/.config/agent-of-empires/serve.url
    participant AOE as aoe serve --auth token<br/>agentbox/flake.nix:2353 127.0.0.1:9095

    BR->>PX: GET /api/sessions (or dashboard asset) proxy.mjs:899
    PX->>PX: buffer request body, size-capped MAX_BODY_BYTES 25MiB proxy.mjs:906,:1027-1038
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:914
    alt auth not ok
        PX-->>BR: 302 to /nip07 (html GET) or 401 JSON proxy.mjs:919-933
    else auth ok
        PX->>RT: routeFor(req.url) - no prefix matches proxy.mjs:937,426-439
        RT-->>PX: upstream 127.0.0.1:9095, isAoe true proxy.mjs:439
        PX->>PX: drop hop-by-hop headers, drop inbound x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:941-954
        PX->>PX: inject x-forwarded-for, x-forwarded-proto, x-forwarded-host proxy.mjs:957-962
        PX->>PX: headers x-agentbox-pubkey = auth.pubkey, x-agentbox-auth-mode = auth.mode proxy.mjs:963-964
        Note over PX: INVARIANT N-05 boundary - route.isAoe forces Authorization replacement for EVERY auth mode including nip98, because aoe cannot verify a Nostr header itself proxy.mjs:982-987
        PX->>TOK: readAoeToken() proxy.mjs:989,278
        TOK->>FS: statSync, readFileSync, statSync - torn-read retry once proxy.mjs:281-289
        TOK-->>PX: token or null - last-good cache on transient error, NEVER caches null on error proxy.mjs:290-296
        alt token unavailable
            PX-->>BR: 503 ServiceUnavailable AoE token unavailable proxy.mjs:990-994
            Note over PX: SECURITY-profiles.md custody row AoE daemon token - deleting the state file alone is not a revocation mechanism, last-good cache persists
        else token present
            PX->>PX: headers.authorization = Bearer aoeTok proxy.mjs:996
            PX->>AOE: forward request, path = stripUrlCreds(route.path) proxy.mjs:1002-1011
            AOE-->>PX: proxyRes
            PX-->>BR: relay status and body, pipe proxyRes proxy.mjs:1009-1011
        end
    end
```

## AB-10.7 /mgmt/* router door — bearer_env gated behind NIP-98

```mermaid
sequenceDiagram
    autonumber
    participant BR as Client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:910
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:626
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:426
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1423
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant NB2 as verifyNip98Header<br/>agentbox/management-api/middleware/auth.js:67

    BR->>PX: request /mgmt/v1/system proxy.mjs:899
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:914
    alt not ok
        PX-->>BR: 401 or 302 proxy.mjs:919-933
    else ok
PX->>RT: routeFor(/mgmt/v1/system) matches prefix /mgmt/, strip true<br/>proxy.mjs:429-437
RT-->>PX: upstream 127.0.0.1:MANAGEMENT_API_PORT, path /v1/system, isAoe<br/>false proxy.mjs:437
        alt auth.mode is nip98
PX->>PX: headers.authorization = req.headers.authorization, unchanged<br/>proxy.mjs:971-978
Note over PX: ADR-2010 - a genuinely signed NIP-98 header always reaches<br/>a named upstream, so the governance surface re-verifies the operator<br/>signature itself
        else route declares bearer_env and auth.mode is not nip98
            PX->>PX: headers.authorization = Bearer route.bearer proxy.mjs:979-980
Note over PX: ADR-2010 - bearer_env is injected ONLY when auth.mode is<br/>not nip98, and normalizeRoute is fatal at boot if the named env var is<br/>unset proxy.mjs:332-336
        end
        PX->>MAPI: forward to 127.0.0.1:9090, path /v1/system proxy.mjs:1002-1011
MAPI->>AUTH: authMiddleware, authMode = MANAGEMENT_API_AUTH_MODE default<br/>hybrid agentbox/management-api/server.js:216-219
Note over AUTH: hybrid AUTO-ELEVATES to strict-nip98 when<br/>AGENTBOX_SOVEREIGN_MESH_ENABLED is true and no mode is set explicitly, and<br/>strict-nip98 rejects Bearer unconditionally<br/>agentbox/management-api/middleware/auth.js:152-162,:173-174
AUTH->>NB2: verifyNip98Header(authHeader, request) re-verifies the<br/>signature independently agentbox/management-api/middleware/auth.js:67-82
        alt hybrid and Bearer equals API_KEY
AUTH-->>MAPI: allow, bearer accepted<br/>agentbox/management-api/middleware/auth.js:105-110,178
        else nip98Result valid
AUTH-->>MAPI: allow, nip98 accepted<br/>agentbox/management-api/middleware/auth.js:169,:177
        else neither
            AUTH-->>MAPI: 401 agentbox/management-api/middleware/auth.js:192-195
        end
        MAPI-->>PX: response
        PX-->>BR: relay status and body proxy.mjs:1009-1011
    end
Note over MAPI,AUTH: the AUTH GATE never reads X-Agentbox-Pubkey - createAuthMiddleware<br/>re-verifies Authorization itself on every request<br/>agentbox/management-api/middleware/auth.js:164-199. Two consumers BEHIND the gate do<br/>trust the stamp under ADR-2042 - see AB-10.14
```

## AB-10.8 NIP-07 browser session handshake

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser
    participant PX as handleNip07<br/>agentbox/config/nip98-proxy/proxy.mjs:848
    participant SIGNER as window.nostr signer
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:626
    participant MINT as mintSessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:555
    participant UP as default or routed upstream

    BR->>PX: GET /nip07/ or /nip07/login proxy.mjs:852
    PX-->>BR: 200 HANDSHAKE_PAGE html proxy.mjs:748-842,853-855
    BR->>SIGNER: waitForSigner(3000ms), poll every 200ms proxy.mjs:789-798,802
    SIGNER-->>BR: window.nostr detected
    BR->>SIGNER: signEvent(kind 27235, tags u and method POST) proxy.mjs:811-817
    SIGNER-->>BR: signed event
    BR->>PX: POST /nip07/session, Authorization Nostr base64(signed) proxy.mjs:858,818-820
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:859
    alt not ok, or mode is not nip98
        PX-->>BR: 401 session minting requires a NIP-98 signature proxy.mjs:863-870
        Note over PX: an existing cookie cannot self-renew, and break-glass cannot launder its sentinel into a pubkey-bound session proxy.mjs:860-862
    else ok and mode nip98
        PX->>MINT: mintSessionToken(auth.pubkey) - v1.pubkey.expiry.hmac proxy.mjs:555-559,872
        Note over MINT: SESSION_SECRET is per-boot random unless NIP98_PROXY_SESSION_SECRET is pinned proxy.mjs:212
        Note over MINT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
        PX-->>BR: 200, Set-Cookie agentbox_nip07_session HttpOnly SameSite=Lax Secure-if-https Max-Age SESSION_TTL_S proxy.mjs:598-601,876-881
    end
    BR->>PX: subsequent request, Cookie agentbox_nip07_session=token proxy.mjs:213
    PX->>VI: verifyIdentity reads cookie via parseCookies proxy.mjs:574-582,703-704
    VI-->>PX: ok true, pubkey, mode nip07-session
    PX->>PX: stripSessionCookie removes agentbox_nip07_session before forwarding proxy.mjs:589-596,950
    PX->>UP: forward, x-agentbox-pubkey and x-agentbox-auth-mode nip07-session injected proxy.mjs:963-964
```

## AB-10.9 WebSocket upgrade auth path

```mermaid
sequenceDiagram
    autonumber
    participant BR as WS client
    participant PX as server.on(upgrade)<br/>agentbox/config/nip98-proxy/proxy.mjs:1049
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:626
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:426
    participant TOK as readAoeToken<br/>agentbox/config/nip98-proxy/proxy.mjs:278
    participant UP as upstream socket<br/>net.connect

    BR->>PX: Upgrade GET /sessions/id/live-ws with ?access_token= or ?auth= proxy.mjs:1049
    PX->>PX: parse query access_token or bearer proxy.mjs:1062-1063
    opt auth query param present and no Authorization header
        PX->>PX: req.headers.authorization = Nostr plus the auth param proxy.mjs:1064-1067
Note over PX: DIVERGENCE query-carried credentials, WS-handshake-only - regression fix<br/>restoring the console signer-only auth carrier lost when /feed moved behind this proxy under<br/>ADR-069 proxy.mjs:1050-1059
    end
    PX->>VI: verifyIdentity(req, queryToken) proxy.mjs:1070
    alt not ok
        PX-->>BR: HTTP/1.1 401 Unauthorized, then socket.destroy proxy.mjs:1071-1080
        Note over PX: DIVERGENCE break-glass bearer accepted via ?access_token= or ?bearer= on WS upgrades over the LAN agentbox/docs/INGRESS-identity.md known divergences bullet 4
    else ok
        PX->>RT: routeFor(req.url) proxy.mjs:1082
        alt route.isAoe
            PX->>TOK: readAoeToken() proxy.mjs:1088,278
            alt token unavailable
                PX-->>BR: HTTP/1.1 503 Service Unavailable, then socket.destroy proxy.mjs:1089-1094
            else token present
                PX->>UP: net.connect, rebuild request line, strip Authorization, inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:1096-1121
                PX->>UP: Authorization Bearer aoeWsToken - unconditional for AoE proxy.mjs:1132
            end
        else named route
            alt auth.mode is nip98
                PX->>UP: Authorization req.headers.authorization passthrough proxy.mjs:1124-1128
            else route.bearer set and auth.mode is not nip98
                PX->>UP: Authorization Bearer route.bearer proxy.mjs:1129-1130
            end
        end
        PX->>PX: strip session cookie before forwarding on WS too proxy.mjs:1110-1114
        PX->>UP: pipe socket bidirectionally proxy.mjs:1136-1137
    end
```

## AB-10.10 Direct :9090 path vs proxied :9090 path

```mermaid
sequenceDiagram
    autonumber
    participant DC as Direct caller (container-internal)
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1423, HOST 0.0.0.0 PORT 9090
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant BR2 as Browser via :9096
    participant PX2 as proxy /mgmt/ route<br/>agentbox/config/nip98-proxy/proxy.mjs:407

    Note over DC,MAPI: :9090 is published to the host only as 127.0.0.1:9090:9090 agentbox/flake.nix:2692 - DC must already be container-internal or on the loopback publish
    DC->>MAPI: request with its OWN Authorization Bearer API_KEY or Nostr header, no X-Agentbox-Pubkey
    MAPI->>AUTH: authMiddleware verifies bearer or nip98 directly agentbox/management-api/server.js:216-224
    AUTH-->>MAPI: allow or 401

    BR2->>PX2: request /mgmt/... over 9096, with NIP-98 or session cookie
    PX2->>PX2: verifyIdentity, then routeFor, then inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:914,:937,:963-964
    PX2->>MAPI: forward to 127.0.0.1:9090, Authorization is passthrough (nip98) or bearer_env (else) per ADR-2010
    MAPI->>AUTH: authMiddleware STILL independently verifies Authorization, same code path as the direct request agentbox/management-api/middleware/auth.js:164-194
Note over MAPI,AUTH: the only difference at the GATE is WHO ADDS the pubkey header - only the<br/>proxied path carries x-agentbox-pubkey and x-agentbox-auth-mode, and createAuthMiddleware<br/>requires neither agentbox/management-api/middleware/auth.js:164-199. Route handlers past the<br/>gate are a different matter - see AB-10.14
```

## AB-10.11 Direct :9095 token-bearing consumer — nostr-gateway aoeRequest

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator, /spawn command
    participant GW as aoeRequest<br/>agentbox/config/nostr-gateway/gateway.cjs:431
    participant TOKFN as readAoeToken<br/>agentbox/config/nostr-gateway/gateway.cjs:127
    participant FS2 as serve.url<br/>AGENTBOX_AOE_TOKEN_FILE override, default ~/.config/agent-of-empires/serve.url
    participant AOE2 as aoe serve :9095<br/>agentbox/flake.nix:2353

    OP->>GW: /spawn dir agent text - aoeCreateSession(repoPath, tool, title) agentbox/config/nostr-gateway/gateway.cjs:319-323,:452-453
    GW->>GW: POST /api/sessions?wait=ready via aoeRequest gateway.cjs:452-453
    GW->>TOKFN: readAoeToken() gateway.cjs:435
    TOKFN->>FS2: statSync then readFileSync then statSync, torn-read retry once gateway.cjs:130-138
    alt file missing or unreadable, and no last-good cache
        TOKFN-->>GW: null
        GW-->>OP: throw AoE token unavailable (N-05 fail-closed) - not sending unauthenticated request gateway.cjs:436
        Note over GW,TOKFN: ADR-2002 defence-in-depth - this is a SECOND independent enforcement point beyond the nip98-proxy, duplicated verbatim per the KEEP IN SYNC comment gateway.cjs:117-123
    else token cached or freshly read
        TOKFN-->>GW: token, last-good cache, NEVER caches null on a transient error
        GW->>AOE2: POST /api/sessions, Authorization Bearer tok gateway.cjs:437-444
        AOE2-->>GW: 200/201, session id and status
    end
    Note over GW: on any AoE failure the caller falls back to a plain tmux new-window, per the ADR-042/044 D5 comment gateway.cjs:105-109
```

## AB-10.12 selftest.mjs assertion families

```mermaid
flowchart TB
    HARNESS["selftest.mjs harness<br/>agentbox/config/nip98-proxy/selftest.mjs:275 main()"]
    PUT["proxy.mjs under test<br/>plus separate child processes<br/>for boot-config-specific cases"]
    FAKE["fake AoE, mgmt and governance upstreams"]
    A["A - no credentials, 401<br/>selftest.mjs:312"]
    B["B - break-glass bearer, 200, identity injected,<br/>Authorization stripped<br/>selftest.mjs:318"]
    C["C - valid NIP-98, 200, skips gracefully if<br/>nostr-tools unresolvable<br/>selftest.mjs:340"]
    D["D - WebSocket upgrade, forwarded with<br/>injected identity<br/>selftest.mjs:369"]
    E["E - routed prefix ADR-045, mgmt on second upstream,<br/>unrouted falls through<br/>selftest.mjs:421"]
    F["F - NIP-07 sessions, handshake page, redirect,<br/>cookie auth, strip, forged/expired reject<br/>selftest.mjs:455"]
    G["G - spoofed identity headers stripped and<br/>replaced, ADR-2009<br/>selftest.mjs:554"]
    H["H - verifier faults, absent/throwing/non-canonical,<br/>zero upstream contact, ADR-2009/2011<br/>selftest.mjs:602"]
    I["I - allowlist removal denies the next request,<br/>including a live session, ADR-2009<br/>selftest.mjs:693"]
    J["J - cookie expiry and restart under a rotated<br/>HMAC key, rejected<br/>selftest.mjs:734"]
    K["K - tokenless denial, AoE daemon token NOT<br/>injected, ADR-2002<br/>selftest.mjs:763"]
    L["L - bearer gated behind NIP-98 on a named route,<br/>ADR-2010<br/>selftest.mjs:785"]
    M["M - hex-canonical identity helper, 0600 key file,<br/>restart stability, ADR-2011, see AB-11<br/>selftest.mjs:874"]
    N["N - break-glass is BOUNDED authority, expiry,<br/>scope and fingerprint audit, ADR-2027<br/>selftest.mjs:948"]
    NOTE1["129 assert() call sites. NODE_PATH must resolve<br/>nostr-tools or the live-signature cases skip -<br/>README.md:189-206, harness at selftest.mjs:89"]

    HARNESS --> PUT
    HARNESS --> FAKE
    PUT --> FAKE
    HARNESS -.-> A
    A -.-> B
    B -.-> C
    C -.-> D
    D -.-> E
    E -.-> F
    F -.-> G
    G -.-> H
    H -.-> I
    I -.-> J
    J -.-> K
    K -.-> L
    L -.-> M
    M -.-> N
```

The A-to-N chain is execution order: `main()` runs the families in sequence, so a
failure in an early family short-circuits the ones after it.

## AB-10.13 aoe-seed-sessions.mjs — the 4th :9095 token consumer, and the ADR-2080 router seed
```mermaid
sequenceDiagram
    autonumber
    participant E as entrypoint-unified.sh<br/>Stage B interaction-plane seed block
    participant SEED as aoe-seed-sessions.mjs<br/>readAoeToken/fetchWithTimeout aoe-seed-sessions.mjs:428,:448
    participant FS as serve.url<br/>AGENTBOX_AOE_TOKEN_FILE, aoe-seed-sessions.mjs:425-426
    participant AOE as aoe serve --auth token<br/>agentbox/flake.nix:2353, 127.0.0.1:9095

    E->>SEED: nohup node aoe-seed-sessions.mjs (fire-and-forget)
    SEED->>FS: readAoeToken() — statSync/readFileSync/statSync, single retry on torn read (aoe-seed-sessions.mjs:428-441)
    alt token file absent or unreadable, no last-good cache
        SEED-->>SEED: fetchWithTimeout rejects "AoE token unavailable (N-05 fail-closed)" (aoe-seed-sessions.mjs:452-454)
        Note over SEED: same fail-closed contract as AB-10.11's readAoeToken — this script is<br/>DUPLICATED VERBATIM (modulo fs accessor) alongside proxy.mjs, gateway.cjs and<br/>tab0-bridge/server.mjs, KEEP IN SYNC comment at aoe-seed-sessions.mjs:418-421
    else token cached or freshly read
        SEED->>AOE: GET/POST /api/sessions with Authorization Bearer tok (aoe-seed-sessions.mjs:456-457)
        AOE-->>SEED: session records, incl. the router seed from agentbox.toml:1387-1391
    end
    Note over SEED,AOE: the `router` slug's customAgents program resolves via WRAPPER_SLUGS.router.file<br/>= config/harness-wrappers/router.sh (aoe-seed-sessions.mjs:110-114, ADR-2080) — see AB-02.20 for<br/>the full WRAPPER_SLUGS table and AB-02.19 for the realpath run-as-script guard fix (2026-09-06)<br/>that makes this reconciler actually execute in the baked image
```


## AB-10.14 ADR-2042 — who actually trusts X-Agentbox-Pubkey

```mermaid
sequenceDiagram
    autonumber
    participant CL as Caller
    participant PX as nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:910
    participant GATE as createAuthMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant AEA as verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:108
    participant APL as resolveAgentDid<br/>agentbox/management-api/lib/action-plane.js:221

    CL->>PX: request carrying its own x-agentbox-pubkey
    PX->>PX: inbound copy DROPPED, then re-injected from the AUTHENTICATED identity proxy.mjs:946-947,:963-964
    Note over PX: on WS the same strip-and-restamp applies proxy.mjs:1106,:1120-1121
    PX->>GATE: forward, Authorization passthrough (nip98) or bearer_env (else) proxy.mjs:971-980
    GATE->>GATE: verifyNip98Header and verifyBearerHeader on Authorization ONLY auth.js:168-178
    Note over GATE: INVARIANT the gate never reads x-agentbox-pubkey - it re-verifies the credential itself auth.js:164-199
    GATE->>AEA: POST /v1/agent-events/emit reaches the route past the gate
    alt Authorization starts with Nostr
        AEA->>AEA: verify the signature against this request, a bad signature is NEVER rescued by the header agent-event-auth.js:118-148
    else no Nostr header - NIP-07 session or break-glass hop
        AEA->>AEA: ADR-2042 path 2, trust x-agentbox-pubkey when it is 64 lowercase hex agent-event-auth.js:90-96,:157-160
        Note over AEA: DOC-DRIFT AB-10.7 and AB-10.10 previously asserted management-api has ZERO<br/>references to X-Agentbox-Pubkey. False - this is the ONLY identity carrier on that hop,<br/>because the proxy replaced Authorization with the route bearer
    end
    GATE->>APL: any action-plane dispatch past the gate
    APL->>APL: DID priority request.auth.pubkey, then x-agentbox-pubkey, then AGENTBOX_AGENT_DID, then null action-plane.js:221-231
    Note over AEA,APL: DIVERGENCE ADR-2042 Consequences, stated in-source - path 2 is sound only while<br/>nothing but the proxy reaches :9090, and that is NOT true today, server.js binds 0.0.0.0<br/>in-container so any holder of MANAGEMENT_API_KEY clears the gate and can then forge the<br/>stamp agent-event-auth.js:31-39
    Note over AEA: policy gate AGENTBOX_AGENT_EVENT_AUTH, default nip98 since ADR-2044, off is<br/>explicitly selectable and returns did null agent-event-auth.js:112-115
```

The governing doc states Invariant 2 as "X-Agentbox-Pubkey always proxy-injected,
never trusted inbound" — true at the ingress, and the two ADR-2042 consumers above
are the deliberate, documented exception on the far side of the gate.
