---
id: AB-10
title: Ingress — nip98-proxy named operator routes and the session socket
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2002, ADR-2009, ADR-2010, ADR-2011, ADR-2013, ADR-2047, ADR-2080, ADR-2153]
sources:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/config/nip98-proxy/README.md
  - ../project/agentbox/config/nip98-proxy/selftest.mjs
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/services/agentbox-manifest/src/proxy.rs
  - ../project/agentbox/services/agentbox-sessions/src/api.rs
  - ../project/agentbox/services/agentbox-sessions/src/plane.rs
  - ../project/agentbox/services/agentbox-sessions/README.md
  - ../project/agentbox/config/session-plane/client.cjs
  - ../project/agentbox/config/custody/session-plane-acl.json
  - ../project/agentbox/config/role-accounts.json
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
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/voice/console/Caddyfile
  - ../project/agentbox/management-api/lib/agent-event-auth.js
  - ../project/agentbox/management-api/lib/action-plane.js
verified_commit: 0b0f7918961e629280faf86f88227535abf514b5
---

## AB-10.1 Door inventory and port topology

```mermaid
flowchart TB
    LAN["LAN client"]
    P9096["nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:1045<br/>listen 0.0.0.0:9096"]
    MGMT9090["management-api<br/>agentbox/management-api/server.js:49,:1502<br/>HOST 0.0.0.0 PORT 9090 in-container"]
    BRIDGE8971["tab0-bridge<br/>agentbox/agentbox.toml:2106<br/>127.0.0.1:8971"]
    IDE["code-server 8080 and jupyter 8888<br/>agentbox/agentbox.toml:2128,:2133"]
    SOCK["agentbox-sessions socket<br/>agentbox/services/agentbox-sessions/src/api.rs:29<br/>unix 0660, no TCP port"]
    RELAY7777["nostr-rs-relay<br/>agentbox/flake.nix:1923-1924<br/>binds relayCfg.bind, default 127.0.0.1:7777"]
    VOICE8444["voice cockpit Caddy origin<br/>docker-compose.voice.yml:41<br/>0.0.0.0:8444"]

    LAN -->|"9096:9096 SANCTIONED agentbox/scripts/ci/check-ports-loopback.mjs:94,:74-91 ADR-2013"| P9096
    P9096 -->|"prefix mgmt, approvals, dream - bearer_env MANAGEMENT_API_KEY agentbox/agentbox.toml:2099-2118"| MGMT9090
    P9096 -->|"prefix bridge and feed - bearer_env BRIDGE_TOKEN agentbox/agentbox.toml:2104-2124"| BRIDGE8971
    P9096 -->|"prefix code and jupyter - preserve_host agentbox/agentbox.toml:2126-2135"| IDE
    P9096 -.->|"no prefix matches - routeFor returns null, 404 after auth, no upstream contact agentbox/config/nip98-proxy/proxy.mjs:846-851"| P9096
    LAN -.->|"127.0.0.1:9090:9090 host publish agentbox/flake.nix:3533 not LAN-reachable"| MGMT9090
    LAN -->|"8443/8444 SANCTIONED docker-compose.voice.yml:41 agentbox/scripts/ci/check-ports-loopback.mjs:95-96"| VOICE8444
    VOICE8444 -->|"slash lo and slash docs ONLY - direct to management-api voice/console/Caddyfile:72-77"| MGMT9090
    VOICE8444 -->|"slash approvals, mgmt, workspace, dream, feed, bridge, nip07 - Authorization forwarded to the NIP-98 door voice/console/Caddyfile:55-106"| P9096

N1["RESOLVED ADR-2047: not a breach - a decided, enumerated exposure. The LAN surface is TEN<br/>sanctioned publishes across five compose files, matched on the normalised host_ip/published/<br/>target/protocol tuple: 9096 sovereign ingress, voice 8443 and 8444, browsercontainer 5903<br/>8931 9222-to-9223, gui-tools 5905 9876 9877, xr-runtime 5904 - each cited at<br/>agentbox/scripts/ci/check-ports-loopback.mjs:93-104. 9096 is the sole IDENTITY ingress, now to<br/>named operator routes only; the others carry their own auth."]
N2["RETIRED ADR-2153: there is no session board door. aoe serve, its port 9095 listener and the proxy default<br/>upstream are gone; the flake comment at agentbox/flake.nix:3528-3529 records that session control<br/>is a unix socket with no TCP port, and the voice Caddyfile has no slash aoe handle (voice/console/Caddyfile:55-106)."]
N6["INVARIANT ADR-2013: sovereign_mesh.relay.expose does NOT open a LAN door. When true it adds<br/>ONE publish and that publish is loopback-pinned - 127.0.0.1:port:port at<br/>agentbox/flake.nix:3538-3539 - so the relay never reaches the SANCTIONED LAN list at<br/>check-ports-loopback.mjs:93-104. Default is expose=false, agentbox.toml:148."]
N3["INVARIANT ADR-2153: verification precedes routing. An unauthenticated request to an unknown path is<br/>refused 302 or 401 before routeFor runs, so the route table cannot be probed anonymously; only a verified<br/>caller sees the 404 (agentbox/config/nip98-proxy/proxy.mjs:823-851). The route table itself is projected from the<br/>manifest into nip98-proxy-config.json at every boot (agentbox/services/agentbox-manifest/src/proxy.rs:23,<br/>agentbox/config/entrypoint-unified.sh:2342)"]
N4["RESOLVED ADR-2047: the compose-exposure qualification is superseded in the governing doc.<br/>The line-walker bypass is fixed - check-ports-loopback.sh is a wrapper that execs<br/>check-ports-loopback.mjs, a strict YAML reader for the compose subset, and anything outside that<br/>subset is REJECTED with file and line, never skipped. ADR-2013 stays partial for the DEPLOYMENT<br/>half only: overlay order, interpolation, external files and active-listener evidence.<br/>Receipt: agentbox/docs/estate-closeout/2026-09-05/adr-2013-ports-gate.json."]
N5["INVARIANT ADR-2013: the wrapper FAILS LOUDLY when the .mjs gate is missing - a copy of the<br/>wrapper without its gate exits 3 with an explicit message rather than looking like a pass<br/>agentbox/scripts/ci/check-ports-loopback.sh:38-42"]
```

**Drift:** the governing doc still quotes a flake comment reading `# (aoe serve, --auth token) is NEVER published.` as live at flake.nix line 2647 (`agentbox/docs/INGRESS-identity.md:199-202`) and anchors `verifyIdentity` at proxy.mjs line 626 (`agentbox/docs/INGRESS-identity.md:68`), the break-glass bullet at proxy.mjs lines 753-760 (`agentbox/docs/INGRESS-identity.md:229`) and the session secret at proxy.mjs line 107 (`agentbox/docs/INGRESS-identity.md:232`), but that comment is gone, the function is at `proxy.mjs:523` (`agentbox/config/nip98-proxy/proxy.mjs:523`), and the doc's own 0.2.4 entry already records the AoE retirement (`agentbox/docs/INGRESS-identity.md:9`).

## AB-10.2 proxy.mjs boot sequence — config, routes, verifier load

```mermaid
sequenceDiagram
    autonumber
    participant BOOT as proxy.mjs boot<br/>agentbox/config/nip98-proxy/proxy.mjs:19
    participant FS as Filesystem<br/>CONFIG_FILE / NOSTR_BRIDGE_PATH
    participant NB as loadNostrBridge<br/>agentbox/config/nip98-proxy/proxy.mjs:373
    participant LOG as log()<br/>agentbox/config/nip98-proxy/proxy.mjs:339

    BOOT->>BOOT: PORT, BIND parsed, no default upstream URL any more (ADR-2153) proxy.mjs:19-20
    BOOT->>BOOT: BREAK_GLASS = roleSecret('NIP98_PROXY_ALLOW_BEARER') proxy.mjs:53<br/>role-secret loader: NAME_FILE, then $AGENTBOX_SECRETS_DIR/NAME,<br/>bare env only while role_isolation is off (proxy.mjs:28-52)
    BOOT->>BOOT: SESSION_TTL_S = NIP98_PROXY_SESSION_TTL default 43200 proxy.mjs:162
    break SESSION_TTL_S not a positive safe integer
        BOOT-->>BOOT: throw Error proxy.mjs:164
    end
    BOOT->>BOOT: SESSION_SECRET = PINNED_SESSION_SECRET or crypto.randomBytes(32).toString(hex) proxy.mjs:166-167<br/>the pinned secret takes the same role-secret path (proxy.mjs:166)
    Note over BOOT: DIVERGENCE session secret is per-boot - NIP-07 sessions do not survive a proxy restart, by design agentbox/docs/INGRESS-identity.md:231 (known divergences)
    Note over BOOT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
    BOOT->>BOOT: NIP98_PROXY_ALLOWED_PUBKEYS split lowercased filtered proxy.mjs:173-176
    break any entry not 64-hex
        BOOT-->>BOOT: throw Error NIP98_PROXY_ALLOWED_PUBKEYS entries must be 64-character hex proxy.mjs:178
    end
    BOOT->>NB: loadNostrBridge() proxy.mjs:423
    alt NOSTR_BRIDGE_PATH explicit
        NB->>FS: existsSync(NOSTR_BRIDGE_PATH) proxy.mjs:381
        alt missing or exports no verifyNip98
            NB-->>LOG: error no fallback proxy.mjs:382,:391,:393
            NB-->>BOOT: null
        else
            NB-->>BOOT: NostrBridge proxy.mjs:389
        end
    else no explicit path
        loop source-tree then baked-image candidates proxy.mjs:398-407
            NB->>FS: existsSync(candidate)
            NB->>NB: require(candidate) proxy.mjs:411
        end
        NB-->>BOOT: NostrBridge or null proxy.mjs:409-420
    end
    alt NostrBridge is null
        BOOT-->>LOG: warn nostr-bridge unavailable NIP-98 verification DISABLED fail-closed proxy.mjs:425-431
        Note over BOOT: INVARIANT ADR-2009 - missing verifier rejects every NIP-98 token, only break-glass if configured survives
    else loaded
        BOOT-->>LOG: info nostr-bridge loaded proxy.mjs:388,:413
    end
    BOOT->>FS: readFileSync CONFIG_FILE default workspace/.agentbox/nip98-proxy-config.json proxy.mjs:249-255
    alt file absent
        FS-->>BOOT: FILE_CONFIG = routes [] allowedPubkeys [] proxy.mjs:257
    else present
        BOOT->>BOOT: JSON.parse, validate allowedPubkeys 64-hex proxy.mjs:260-268
        break malformed JSON or bad pubkey shape
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:272-273
        end
    end
    BOOT->>BOOT: parse NIP98_PROXY_ROUTES JSON array, normalizeRoute each entry proxy.mjs:279-284
    break not an array, or normalizeRoute throws e.g. bearer_env unset proxy.mjs:226-230
        BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:294-295
    end
    BOOT->>BOOT: FILE_CONFIG.routes appended, env route wins on prefix conflict proxy.mjs:286-289
    opt NIP98_PROXY_MGMT_UPSTREAM set and no existing slash mgmt slash route
        BOOT->>BOOT: push convenience route proxy.mjs:306-310
        break invalid URL
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:311-314
        end
    end
    BOOT->>BOOT: server.listen(PORT, BIND) proxy.mjs:1045
    BOOT-->>LOG: info nip98-proxy listening - bind, upstream, routes, nip98 enabled/DISABLED, breakGlass, nip07Sessions ttl and pinned-vs-per-boot, allowedPubkeys count proxy.mjs:1046-1068
```

## AB-10.3 verifyIdentity — full auth precedence

```mermaid
sequenceDiagram
    autonumber
    participant REQ as Request<br/>HTTP forward() or WS upgrade
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:523
    participant CTE as constantTimeEqual<br/>agentbox/config/nip98-proxy/proxy.mjs:437
    participant NB as NostrBridge.verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:500
    participant CAN as canonicalPubkey<br/>agentbox/config/nip98-proxy/proxy.mjs:196
    participant PA as pubkeyAllowed<br/>agentbox/config/nip98-proxy/proxy.mjs:182
    participant VST as verifySessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:458

    REQ->>VI: verifyIdentity(req, bearerToken, rawBody) proxy.mjs:523
    VI->>VI: authHeader = req.headers.authorization proxy.mjs:524
    alt BREAK_GLASS configured and a Bearer token or query bearerToken matches
        VI->>CTE: constantTimeEqual(token, BREAK_GLASS) proxy.mjs:533
Note over VI,CTE: DIVERGENCE (historical) break-glass bearer over the LAN - a single shared secret<br/>bypasses NIP-98 entirely while NIP98_PROXY_ALLOW_BEARER is set. See the RESOLVED note below -<br/>ADR-2027 (closeout 2026-09-05) bounds it with expiry + scope, still tracked as INGRESS-identity.md:227<br/>(known divergences) pending a doc update
        VI->>VI: RESOLVED ADR-2027 — breakGlassNotExpired() proxy.mjs:534
        alt expiry configured and elapsed, or malformed
            VI-->>REQ: ok false, reason break_glass_expired, audited by fingerprint proxy.mjs:535-543
        end
        VI->>VI: breakGlassScopeAllows(method, url) proxy.mjs:544
        alt NIP98_PROXY_BEARER_SCOPE configured and request outside it
            VI-->>REQ: ok false, reason break_glass_out_of_scope, audited by fingerprint proxy.mjs:545-553
        end
        Note over VI: SECURITY-profiles.md custody row Proxy break-glass bearer - captured at process start,<br/>ADR-2027 adds optional expiry + scope bounds and a per-use fingerprint audit trail (breakGlassUse<br/>counters), still an in-process receipt, not a durable audit store
        VI->>VI: breakGlassUse.accepted++, log warn break-glass USED with fingerprint proxy.mjs:554-566
        VI-->>REQ: ok true, pubkey NIP98_PROXY_BEARER_PUBKEY default break-glass, mode break-glass proxy.mjs:567
    else Authorization starts with Nostr
        VI->>NB: verifyNip98(authHeader, method, signedUrlFor(req), rawBody) proxy.mjs:577
        alt NostrBridge unavailable
            VI-->>REQ: ok false, reason nip98_verifier_unavailable proxy.mjs:573
        else verifyNip98 throws
            VI-->>REQ: ok false, reason nip98_verify_error proxy.mjs:578-579
        else result not valid
            VI-->>REQ: ok false, reason nip98_invalid proxy.mjs:595
        else result.valid true
            VI->>CAN: canonicalPubkey(result.pubkey) proxy.mjs:586
            alt not lowercase 64-hex
                VI-->>REQ: ok false, reason nip98_noncanonical_pubkey proxy.mjs:587-588
            else canonical
                VI->>PA: pubkeyAllowed(pubkey) proxy.mjs:590
                alt allowlist non-empty and pubkey missing
                    VI-->>REQ: ok false, reason pubkey_not_allowed proxy.mjs:590-591
                else
                    VI-->>REQ: ok true, pubkey, mode nip98 proxy.mjs:593
                end
            end
        end
    else Cookie carries agentbox_nip07_session
        VI->>VST: verifySessionToken(cookie) proxy.mjs:602
        VST->>CTE: constantTimeEqual(mac, expected) proxy.mjs:466
        VST->>PA: pubkeyAllowed(pubkey) proxy.mjs:467
        alt session valid, not expired, HMAC matches, and allowed
            VI-->>REQ: ok true, pubkey, mode nip07-session proxy.mjs:603-604
        else
            VI-->>REQ: ok false, reason session_invalid_or_expired proxy.mjs:606
        end
    else no credentials at all
        VI-->>REQ: ok false, reason no_credentials proxy.mjs:609
    end
```

## AB-10.4 verifyNip98 internals — header, tags, payload, signature, replay

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:577
    participant VN as verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:500
    participant TAG as getTag<br/>agentbox/mcp/servers/nostr-bridge.js:531
    participant SHA as crypto sha256<br/>agentbox/mcp/servers/nostr-bridge.js:566,:573
    participant SIG as verifyEvent nostr-tools<br/>agentbox/mcp/servers/nostr-bridge.js:583
    participant REPLAY as replay LRU<br/>agentbox/mcp/servers/nostr-bridge.js:97

    CALLER->>VN: verifyNip98(authHeader, method, url, rawBody) nostr-bridge.js:500
    break header missing or not Nostr-prefixed
        VN-->>CALLER: valid false, error missing or malformed Nostr header nostr-bridge.js:502
    end
    VN->>VN: event = JSON.parse(Buffer.from(encoded, base64).toString(utf8)) nostr-bridge.js:512
    break not valid base64 JSON
        VN-->>CALLER: valid false, error token is not valid base64 JSON nostr-bridge.js:514
    end
    break event.kind !== kinds.AUTH (27235) nostr-bridge.js:55,:517
        VN-->>CALLER: valid false, error expected kind 27235 got X nostr-bridge.js:518
    end
    break created_at not a number
        VN-->>CALLER: valid false, error missing created_at nostr-bridge.js:522
    end
    break abs(now - created_at) greater than VERIFY_NIP98_WINDOW_S 60s nostr-bridge.js:80,:526
        VN-->>CALLER: valid false, error event timestamp out of 60-second window nostr-bridge.js:527
    end
    VN->>TAG: getTag(method), getTag(u) nostr-bridge.js:538-539
    break method tag mismatch case-insensitive
        VN-->>CALLER: valid false, error method tag mismatch nostr-bridge.js:542
    end
    break url tag mismatch - urlTag not equal url and not endsWith(url)
        VN-->>CALLER: valid false, error url tag mismatch nostr-bridge.js:546
    end
    opt rawBody supplied - Finding 4 payload binding
        VN->>TAG: getTag(payload) nostr-bridge.js:561
        VN->>SHA: hex(sha256(rawBody)) nostr-bridge.js:566,:573
        break body non-empty and payload tag missing, or computed hash mismatch either branch
            VN-->>CALLER: valid false, error missing payload tag for request body OR payload hash mismatch nostr-bridge.js:564,:568,:575
        end
    end
    VN->>SIG: verifyEvent(event) BIP-340 schnorr constant-time nostr-bridge.js:584
    break signature invalid or verifyEvent throws
        VN-->>CALLER: valid false, error invalid Schnorr signature nostr-bridge.js:586,:589
    end
    VN->>REPLAY: _recordReplayId(event.id) nostr-bridge.js:596,:111
Note over REPLAY: REPLAY_TTL_MS = VERIFY_NIP98_WINDOW_S times 1000 = 60000ms,<br/>REPLAY_MAX_ENTRIES = 20000, keyed on event id, populated ONLY after signature verify so a<br/>forged token can never poison the cache nostr-bridge.js:95-97
    break id already seen inside the freshness window - live duplicate
        VN-->>CALLER: valid false, error replayed NIP-98 event id nostr-bridge.js:597
    end
    VN-->>CALLER: valid true, pubkey event.pubkey nostr-bridge.js:600
```

## AB-10.5 Request auth state machine

```mermaid
stateDiagram-v2
    [*] --> Unauthenticated
    Unauthenticated --> BearerAccepted: break-glass token constant-time match, unexpired, in scope proxy.mjs:529-533,:567
    Unauthenticated --> Nip98Verified: verifyNip98 valid, canonicalPubkey, pubkeyAllowed proxy.mjs:586-593
    Unauthenticated --> SessionCookieVerified: verifySessionToken valid and allowlisted proxy.mjs:600-604
    Unauthenticated --> Rejected302: browser GET, Accept text/html, auth failed proxy.mjs:828-835
    Unauthenticated --> Rejected401: auth failed, not an html GET proxy.mjs:837-842
    Unauthenticated --> FailClosed: NostrBridge null at boot, every Nostr header refused proxy.mjs:425-431,:573
    BearerAccepted --> Routed: routeFor(req.url) proxy.mjs:846,:980
    Nip98Verified --> Routed: routeFor(req.url) proxy.mjs:846,:980
    SessionCookieVerified --> Routed: routeFor(req.url) proxy.mjs:846,:980
    Routed --> Upstream: named route, nip98 passthrough or bearer_env injection proxy.mjs:886-894
    Routed --> NotFound404: no named prefix matches, routeFor returned null, no upstream contact proxy.mjs:847-850,:981-983
    FailClosed --> [*]
    Rejected401 --> [*]
    Rejected302 --> [*]
    NotFound404 --> [*]
    Upstream --> [*]
```

Every anchor above was re-derived from its symbol at agentbox 0b0f791 (2026-10-29), after
ADR-2153 removed the AoE transitions, so the old set no longer applies.

## AB-10.6 Unknown-route door — no default upstream, 404 after verification

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser or client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:819
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:523
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:323
    participant SOCK as agentbox-sessions socket<br/>agentbox/services/agentbox-sessions/src/api.rs:29

    BR->>PX: GET /api/sessions (the retired board path) proxy.mjs:808
    PX->>PX: buffer request body, size-capped MAX_BODY_BYTES 25MiB proxy.mjs:157,:928-932
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:823
    alt auth not ok
        PX-->>BR: 302 to /nip07 (html GET) or 401 JSON proxy.mjs:828-842
    else auth ok
        PX->>RT: routeFor(req.url) - no prefix matches proxy.mjs:846,:325-336
        RT-->>PX: null, no default upstream proxy.mjs:336
        PX-->>BR: 404 NotFound, message No operator route, no upstream contact proxy.mjs:847-850
        Note over PX: WebSocket upgrades take the same decision after verifying: socket.end with 404 Not Found proxy.mjs:980-983
    end
    Note over BR,SOCK: session control is not a proxy route at all - it is a unix socket with no TCP port, so there is nothing for this door to forward to (agentbox/flake.nix:3528-3529, agentbox/services/agentbox-sessions/src/api.rs:3-7)
```

**Debt:** `routeFor` still returns an `isAoe: false` field and two branches still test `!route.isAoe`, so a dead flag outlives the retired daemon-token path and misleads the next reader (`agentbox/config/nip98-proxy/proxy.mjs:334`, `agentbox/config/nip98-proxy/proxy.mjs:886`, `agentbox/config/nip98-proxy/proxy.mjs:1016`).

## AB-10.7 /mgmt/* router door — bearer_env gated behind NIP-98

```mermaid
sequenceDiagram
    autonumber
    participant BR as Client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:819
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:523
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:323
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1502
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant NB2 as verifyNip98Header<br/>agentbox/management-api/middleware/auth.js:67

    BR->>PX: request /mgmt/v1/system proxy.mjs:808
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:823
    alt not ok
        PX-->>BR: 401 or 302 proxy.mjs:828-842
    else ok
PX->>RT: routeFor(/mgmt/v1/system) matches prefix /mgmt/, strip true<br/>proxy.mjs:326-331
RT-->>PX: upstream 127.0.0.1:MANAGEMENT_API_PORT, path /v1/system, isAoe<br/>false, preserveHost proxy.mjs:334
Note over PX,RT: a route may opt in to preserve_host, and only then is the client Host<br/>forwarded instead of the upstream host and port (proxy.mjs:239, :870-871).<br/>The connection TARGET stays fixed by the trusted config either way, so this<br/>lets a browser compare Host with Origin without loosening where a request goes
        alt auth.mode is nip98
PX->>PX: headers.authorization = req.headers.authorization, unchanged<br/>proxy.mjs:886-892
Note over PX: ADR-2010 - a genuinely signed NIP-98 header always reaches<br/>a named upstream, so the governance surface re-verifies the operator<br/>signature itself
        else route declares bearer_env and auth.mode is not nip98
            PX->>PX: headers.authorization = Bearer route.bearer proxy.mjs:893-894
Note over PX: ADR-2010 - bearer_env is injected ONLY when auth.mode is<br/>not nip98, and normalizeRoute is fatal at boot if the named env var is<br/>unset proxy.mjs:226-230
        end
        PX->>MAPI: forward to 127.0.0.1:9090, path /v1/system proxy.mjs:900-909
MAPI->>AUTH: authMiddleware, authMode = MANAGEMENT_API_AUTH_MODE default<br/>hybrid agentbox/management-api/server.js:223-226
Note over AUTH: hybrid AUTO-ELEVATES to strict-nip98 when<br/>AGENTBOX_SOVEREIGN_MESH_ENABLED is true and no mode is set explicitly, and<br/>strict-nip98 rejects Bearer unconditionally<br/>agentbox/management-api/middleware/auth.js:152-162,:173-174
AUTH->>NB2: verifyNip98Header(authHeader, request) re-verifies the<br/>signature independently agentbox/management-api/middleware/auth.js:67-82
Note over NB2: the URL it checks against the signed u tag is rebuilt from request.host, which<br/>keeps a non-default port, since the Fastify 5 upgrade, agentbox/management-api/middleware/auth.js:72
        alt hybrid and Bearer equals API_KEY
AUTH-->>MAPI: allow, bearer accepted<br/>agentbox/management-api/middleware/auth.js:105-110,178
        else nip98Result valid
AUTH-->>MAPI: allow, nip98 accepted<br/>agentbox/management-api/middleware/auth.js:169,:177
        else neither
            AUTH-->>MAPI: 401 agentbox/management-api/middleware/auth.js:192-195
        end
        MAPI-->>PX: response
        PX-->>BR: relay status and body proxy.mjs:907-909
    end
Note over MAPI,AUTH: the AUTH GATE never reads X-Agentbox-Pubkey - createAuthMiddleware<br/>re-verifies Authorization itself on every request<br/>agentbox/management-api/middleware/auth.js:164-199. Two consumers BEHIND the gate do<br/>trust the stamp under ADR-2042 (only from a loopback peer, peerIsProxy) - see AB-10.14
```

## AB-10.8 NIP-07 browser session handshake

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser
    participant PX as handleNip07<br/>agentbox/config/nip98-proxy/proxy.mjs:745
    participant SIGNER as window.nostr signer
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:523
    participant MINT as mintSessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:452
    participant UP as default or routed upstream

    BR->>PX: GET /nip07/ or /nip07/login proxy.mjs:749
    PX-->>BR: 200 HANDSHAKE_PAGE html proxy.mjs:645-739,749-752
    BR->>SIGNER: waitForSigner(3000ms), poll every 200ms proxy.mjs:686-695,692
    SIGNER-->>BR: window.nostr detected
    BR->>SIGNER: signEvent(kind 27235, tags u and method POST) proxy.mjs:708-714
    SIGNER-->>BR: signed event
    BR->>PX: POST /nip07/session, Authorization Nostr base64(signed) proxy.mjs:767,715-717
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:768
    alt not ok, or mode is not nip98
        PX-->>BR: 401 session minting requires a NIP-98 signature proxy.mjs:772-779
        Note over PX: an existing cookie cannot self-renew, and break-glass cannot launder its sentinel into a pubkey-bound session proxy.mjs:769-771
    else ok and mode nip98
        PX->>MINT: mintSessionToken(auth.pubkey) - v1.pubkey.expiry.hmac proxy.mjs:452-456,781
        Note over MINT: SESSION_SECRET is per-boot random unless NIP98_PROXY_SESSION_SECRET is pinned<br/>through the role-secret loader proxy.mjs:166-167
        Note over MINT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
        PX-->>BR: 200, Set-Cookie agentbox_nip07_session HttpOnly SameSite=Lax Secure-if-https Max-Age SESSION_TTL_S proxy.mjs:495-498,785-790
    end
    BR->>PX: subsequent request, Cookie agentbox_nip07_session=token proxy.mjs:168
    PX->>VI: verifyIdentity reads cookie via parseCookies proxy.mjs:471-479,600
    VI-->>PX: ok true, pubkey, mode nip07-session
    PX->>PX: stripSessionCookie removes agentbox_nip07_session before forwarding proxy.mjs:486-493,864
    PX->>UP: forward, x-agentbox-pubkey and x-agentbox-auth-mode nip07-session injected proxy.mjs:878-879
```

GET `/nip07/session` is a separate, unauthenticated-upstream probe: it reads the
cookie and calls `verifySessionToken` itself, nothing else, proxy.mjs:755-757.
INVARIANT: unrelated upstream availability or authorisation must not make a
signed-in browser start prompting its signer again — the probe never touches the
route or upstream, proxy.mjs:756-757. Response is `cache-control: no-store`,
proxy.mjs:761, so a cached 200 can never stand in for a live check.

## AB-10.9 WebSocket upgrade auth path

```mermaid
sequenceDiagram
    autonumber
    participant BR as WS client
    participant PX as server.on(upgrade)<br/>agentbox/config/nip98-proxy/proxy.mjs:947
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:523
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:323
    participant UP as upstream socket<br/>net.connect

    BR->>PX: Upgrade GET /feed with ?access_token= or ?auth= proxy.mjs:947
    PX->>PX: parse query access_token or bearer proxy.mjs:960-961
    opt auth query param present and no Authorization header
        PX->>PX: req.headers.authorization = Nostr plus the auth param proxy.mjs:962-965
Note over PX: DIVERGENCE query-carried credentials, WS-handshake-only - regression fix<br/>restoring the console signer-only auth carrier lost when /feed moved behind this proxy under<br/>ADR-069 proxy.mjs:948-957
    end
    PX->>VI: verifyIdentity(req, queryToken) proxy.mjs:968
    alt not ok
        PX-->>BR: HTTP/1.1 401 Unauthorized, then socket.destroy proxy.mjs:969-978
        Note over PX: DIVERGENCE break-glass bearer accepted via ?access_token= or ?bearer= on WS upgrades over the LAN agentbox/docs/INGRESS-identity.md:227 (known divergences)
    else ok
        PX->>RT: routeFor(req.url) proxy.mjs:980
        alt no named route matches
            PX-->>BR: HTTP/1.1 404 Not Found, Content-Length 0, then end, no upstream socket opened proxy.mjs:981-983
        else named route
            PX->>UP: net.connect, rebuild request line, strip Authorization, inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:985-1013
            alt auth.mode is nip98
                PX->>UP: Authorization req.headers.authorization passthrough proxy.mjs:1016-1020
            else route.bearer set and auth.mode is not nip98
                PX->>UP: Authorization Bearer route.bearer proxy.mjs:1021-1022
            end
            PX->>PX: strip session cookie before forwarding on WS too proxy.mjs:1002-1006
            PX->>UP: pipe socket bidirectionally proxy.mjs:1027-1028
        end
    end
```

## AB-10.10 Direct :9090 path vs proxied :9090 path

```mermaid
sequenceDiagram
    autonumber
    participant DC as Direct caller (container-internal)
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1502, HOST 0.0.0.0 PORT 9090
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant BR2 as Browser via port 9096
    participant PX2 as proxy /mgmt/ route<br/>agentbox/config/nip98-proxy/proxy.mjs:304

    Note over DC,MAPI: port 9090 is published to the host only as 127.0.0.1:9090:9090 agentbox/flake.nix:3533 - DC must already be container-internal or on the loopback publish
    DC->>MAPI: request with its OWN Authorization Bearer API_KEY or Nostr header, no X-Agentbox-Pubkey
    MAPI->>AUTH: authMiddleware verifies bearer or nip98 directly agentbox/management-api/server.js:223-231
    AUTH-->>MAPI: allow or 401

    BR2->>PX2: request /mgmt/... over 9096, with NIP-98 or session cookie
    PX2->>PX2: verifyIdentity, then routeFor, then inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:823,:846,:878-879
    PX2->>MAPI: forward to 127.0.0.1:9090, Authorization is passthrough (nip98) or bearer_env (else) per ADR-2010
    MAPI->>AUTH: authMiddleware STILL independently verifies Authorization, same code path as the direct request agentbox/management-api/middleware/auth.js:164-194
Note over MAPI,AUTH: the only difference at the GATE is WHO ADDS the pubkey header - only the<br/>proxied path carries x-agentbox-pubkey and x-agentbox-auth-mode, and createAuthMiddleware<br/>requires neither agentbox/management-api/middleware/auth.js:164-199. Route handlers past the<br/>gate are a different matter - see AB-10.14
```

## AB-10.11 Session-socket door — SO_PEERCRED ACL and the nostr-gateway client

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator, /spawn command
    participant GW as sessionCreate<br/>agentbox/config/nostr-gateway/gateway.cjs:415
    participant CL as client.call<br/>agentbox/config/session-plane/client.cjs:4
    participant SK as handle<br/>agentbox/services/agentbox-sessions/src/api.rs:67
    participant PL as Plane.create<br/>agentbox/services/agentbox-sessions/src/plane.rs:449

    OP->>GW: /spawn dir agent text - doSpawn prefers the session plane unless the launcher is zai gateway.cjs:462-464
    alt repoPath is not the project root
        GW-->>OP: throw using tmux for alternate project, doSpawn falls back to the confined tmux new-window gateway.cjs:418,:473-477
    else project root
        GW->>CL: call op create, slug from the title, tool gateway.cjs:419-421
        CL->>SK: connect to AGENTBOX_SESSIONS_SOCKET, write one JSON line, 12 s timeout client.cjs:4-11
        SK->>SK: socket_peercred reads the peer uid, unavailable credentials refuse api.rs:70-73
        alt uid not in the ACL
            SK-->>CL: ok false, uid N is not authorised api.rs:74-76
            CL-->>GW: throws, doSpawn logs and falls back to tmux client.cjs:18, gateway.cjs:473-475
        else uid 1000 devuser or 961 ab-gateway
            SK->>PL: dispatch op create api.rs:83-95
            PL-->>SK: bail on a bad slug, a manifest slug or an existing session plane.rs:451-461
            SK-->>CL: ok true client.cjs:19
        end
        GW->>CL: optional follow-up send, text then pause then Enter gateway.cjs:467-469, plane.rs:396-411
    end
    Note over SK: INVARIANT ADR-2153 - the ACL decides, never the file mode: the socket is 0660 in group ab-sessions (gid 973, members devuser and ab-gateway) but even root is refused unless listed<br/>api.rs:29,:128, config/role-accounts.json:138-143, config/custody/session-plane-acl.json:8-11
    Note over SK: the daemon is supervised as devuser with the ACL path and gid in its environment, autostart true<br/>flake.nix:2910-2915, gid taken from the one role table at eval time flake.nix:1779-1788
    Note over GW: before any command runs, handleWrap drops NIP-17 self-DM copies whose p tag<br/>names another identity and poker-coach table traffic, gateway.cjs:666,681-682,687 -<br/>and the operator key is read only through hooks/lib/operator-key.cjs, file-only<br/>under role_isolation (gateway.cjs:81-82,148)
```

## AB-10.12 selftest.mjs assertion families

```mermaid
flowchart TB
    HARNESS["selftest.mjs harness<br/>agentbox/config/nip98-proxy/selftest.mjs:275 main()"]
    PUT["proxy.mjs under test<br/>plus separate child processes<br/>for boot-config-specific cases"]
    FAKE["fake stub, mgmt and governance upstreams<br/>stub registered as named routes<br/>selftest.mjs:290-293"]
    A["A - no credentials, 401<br/>selftest.mjs:308"]
    B["B - break-glass bearer, 200, identity injected,<br/>Authorization stripped<br/>selftest.mjs:314"]
    C["C - valid NIP-98, 200, skips gracefully if<br/>nostr-tools unresolvable<br/>selftest.mjs:336"]
    D["D - WebSocket upgrade, forwarded with<br/>injected identity<br/>selftest.mjs:365"]
    E["E - routed prefix ADR-045, mgmt on second upstream,<br/>unrouted HTTP and WS get 404, ADR-2153<br/>selftest.mjs:417"]
    F["F - NIP-07 sessions, handshake page, redirect,<br/>cookie auth, strip, forged/expired reject<br/>selftest.mjs:450"]
    G["G - spoofed identity headers stripped and<br/>replaced, ADR-2009<br/>selftest.mjs:549"]
    H["H - verifier faults, absent/throwing/non-canonical,<br/>zero upstream contact, ADR-2009/2011<br/>selftest.mjs:597"]
    I["I - allowlist removal denies the next request,<br/>including a live session, ADR-2009<br/>selftest.mjs:688"]
    J["J - cookie expiry and restart under a rotated<br/>HMAC key, rejected<br/>selftest.mjs:729"]
    K["K - tokenless denial, nothing forwarded,<br/>ADR-2002<br/>selftest.mjs:758"]
    L["L - bearer gated behind NIP-98 on a named route,<br/>ADR-2010<br/>selftest.mjs:780"]
    M["M - hex-canonical identity helper, 0600 key file,<br/>restart stability, ADR-2011, see AB-11<br/>selftest.mjs:869"]
    N["N - break-glass is BOUNDED authority, expiry,<br/>scope and fingerprint audit, ADR-2027<br/>selftest.mjs:943"]
    NOTE1["129 assert() call sites. NODE_PATH must resolve<br/>nostr-tools or the live-signature cases skip -<br/>config/nip98-proxy/README.md:19-20, harness at selftest.mjs:89"]

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

**Debt:** family K still asserts that the retired AoE daemon token is neither injected nor leaked (`agentbox/config/nip98-proxy/selftest.mjs:762-767`), which can no longer fail because the proxy holds no such token; the live ADR-2153 coverage is E2's 404 pair for HTTP and WS (`agentbox/config/nip98-proxy/selftest.mjs:433-440`).

## AB-10.14 ADR-2042 — who actually trusts X-Agentbox-Pubkey

```mermaid
sequenceDiagram
    autonumber
    participant CL as Caller
    participant PX as nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:819
    participant GATE as createAuthMiddleware<br/>agentbox/management-api/middleware/auth.js:164
    participant AEA as verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:135
    participant APL as resolveAgentDid<br/>agentbox/management-api/lib/action-plane.js:222

    CL->>PX: request carrying its own x-agentbox-pubkey
    PX->>PX: inbound copy DROPPED, then re-injected from the AUTHENTICATED identity proxy.mjs:860-861,:878-879
    Note over PX: on WS the same strip-and-restamp applies proxy.mjs:995,:1012-1013
    PX->>GATE: forward, Authorization passthrough (nip98) or bearer_env (else) proxy.mjs:886-894
    GATE->>GATE: verifyNip98Header and verifyBearerHeader on Authorization ONLY auth.js:168-178
    Note over GATE: INVARIANT the gate never reads x-agentbox-pubkey - it re-verifies the credential itself auth.js:164-199
    GATE->>AEA: POST /v1/agent-events/emit reaches the route past the gate
    alt Authorization starts with Nostr
        AEA->>AEA: verify the signature against this request, a bad signature is NEVER rescued by the header agent-event-auth.js:145-175
    else no Nostr header - NIP-07 session or break-glass hop
        AEA->>AEA: ADR-2042 path 2, trust x-agentbox-pubkey when the TCP peer is the proxy and it is 64 lowercase hex agent-event-auth.js:116-122,:184-187
        Note over AEA: DOC-DRIFT AB-10.7 and AB-10.10 previously asserted management-api has ZERO<br/>references to X-Agentbox-Pubkey. False - this is the ONLY identity carrier on that hop,<br/>because the proxy replaced Authorization with the route bearer
    end
    GATE->>APL: any action-plane dispatch past the gate
    APL->>APL: DID priority request.auth.pubkey, then x-agentbox-pubkey, then AGENTBOX_AGENT_DID, then null action-plane.js:222-233
    Note over AEA,APL: INVARIANT both consumers go through proxyVerifiedPubkey, which returns null unless peerIsProxy holds,<br/>so the stamp counts only from a loopback peer or AGENTBOX_PROXY_PEERS, and a request with no socket fails closed<br/>agent-event-auth.js:98-105,117 and action-plane.js:229
    Note over AEA,APL: DEBT residual - a process INSIDE the container can still connect over loopback and forge the stamp, the same-UID boundary (TODO-unified X-1)<br/>a MANAGEMENT_API_KEY holder on the docker network reaches the 0.0.0.0 bind but its peer is not loopback, agent-event-auth.js:34-39
    Note over AEA: policy gate AGENTBOX_AGENT_EVENT_AUTH, default nip98 since ADR-2044, off is<br/>explicitly selectable and returns did null agent-event-auth.js:139-142
```

The governing doc states Invariant 2 as "X-Agentbox-Pubkey always proxy-injected,
never trusted inbound" — true at the ingress, and the two ADR-2042 consumers above
are the deliberate, documented exception on the far side of the gate.

**Invariant:** the stamp counts only when `request.raw.socket.remoteAddress` is loopback (`127.0.0.1`, `::1`, `::ffff:127.0.0.1`) or an exact address in `AGENTBOX_PROXY_PEERS`, and a request with no socket information fails closed; `server.js` binds `0.0.0.0`, so a `MANAGEMENT_API_KEY` holder on the docker network reaches the port but cannot forge any `did:nostr` (`agentbox/management-api/lib/agent-event-auth.js:98-105`, `agentbox/management-api/lib/agent-event-auth.js:117`, `agentbox/management-api/server.js:49`). `resolveAgentDid` calls the same exported helper, so the two ADR-2042 consumers share one rule (`agentbox/management-api/lib/action-plane.js:229`).

**Debt:** the loopback test authenticates the network position, not the process: anything inside the container can open its own loopback connection and forge the stamp (the same-UID boundary, TODO-unified X-1), (`agentbox/management-api/lib/agent-event-auth.js:37-39`).
