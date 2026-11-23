---
id: AB-10
title: Ingress — nip98-proxy named operator routes and the session socket
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2002, ADR-2009, ADR-2010, ADR-2011, ADR-2013, ADR-2047, ADR-2080, ADR-2153, ADR-2210]
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
  - ../project/agentbox/config/lib/nip98-config.sh
  - ../project/agentbox/tests/config/nip98-boot-failclosed.test.mjs
  - ../project/agentbox/docs/adr/ADR-2210-the-nip98-door-refuses-every-identity-until-its-boot-config-exists.md
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
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
---

## AB-10.1 Door inventory and port topology

```mermaid
flowchart TB
    LAN["LAN client"]
    P9096["nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:1065<br/>listen 0.0.0.0:9096"]
    MGMT9090["management-api<br/>agentbox/management-api/server.js:50,server.js:1543<br/>HOST 0.0.0.0 PORT 9090 in-container"]
    BRIDGE8971["tab0-bridge<br/>agentbox/agentbox.toml:2218<br/>127.0.0.1:8971"]
    IDE["code-server 8080 and jupyter 8888<br/>agentbox/agentbox.toml:2240,agentbox.toml:2245"]
    SOCK["agentbox-sessions socket<br/>agentbox/services/agentbox-sessions/src/api.rs:29<br/>unix 0660, no TCP port"]
    RELAY7777["nostr-rs-relay<br/>agentbox/flake.nix:2111-2112<br/>binds relayCfg.bind, default 127.0.0.1:7777"]
    VOICE8444["voice cockpit Caddy origin<br/>docker-compose.voice.yml:41<br/>0.0.0.0:8444"]

    LAN -->|"9096:9096 SANCTIONED agentbox/scripts/ci/check-ports-loopback.mjs:101,check-ports-loopback.mjs:74-86 ADR-2013"| P9096
    P9096 -->|"prefix mgmt, approvals, dream - bearer_env MANAGEMENT_API_KEY agentbox/agentbox.toml:2211-2230"| MGMT9090
    P9096 -->|"prefix bridge and feed - bearer_env BRIDGE_TOKEN agentbox/agentbox.toml:2216-2236"| BRIDGE8971
    P9096 -->|"prefix code and jupyter - preserve_host agentbox/agentbox.toml:2238-2247"| IDE
    P9096 -.->|"no prefix matches - routeFor returns null, 404 after auth, no upstream contact agentbox/config/nip98-proxy/proxy.mjs:866-871"| P9096
    LAN -.->|"127.0.0.1:9090:9090 host publish agentbox/flake.nix:3782 not LAN-reachable"| MGMT9090
    LAN -->|"8443/8444 SANCTIONED docker-compose.voice.yml:41 agentbox/scripts/ci/check-ports-loopback.mjs:102-103"| VOICE8444
    VOICE8444 -->|"slash lo and slash docs ONLY - direct to management-api voice/console/Caddyfile:71-76"| MGMT9090
    VOICE8444 -->|"slash approvals, mgmt, workspace, dream, feed, bridge, nip07 - Authorization forwarded to the NIP-98 door voice/console/Caddyfile:54-105"| P9096

N1["RESOLVED ADR-2047, amended 2026-11-17: not a breach - a decided, enumerated exposure. The LAN surface is THREE<br/>sanctioned publishes across two compose files, matched on the normalised host_ip/published/<br/>target/protocol tuple: 9096 sovereign ingress, voice 8443 and 8444, cited at<br/>agentbox/scripts/ci/check-ports-loopback.mjs:100-104. The browsercontainer, gui-tools and xr-runtime sidecar ports<br/>are no longer sanctioned: they publish as AGENTBOX_SIDECAR_BIND defaulting to 127.0.0.1, accepted only in those<br/>three files, check-ports-loopback.mjs:97-98,check-ports-loopback.mjs:657-660. DEBT: the gate proves the default, not the deployed<br/>address - setting the variable to 0.0.0.0 still passes. 9096 is the sole IDENTITY ingress, now to<br/>named operator routes only; the others carry their own auth."]
N2["RETIRED ADR-2153: there is no session board door. aoe serve, its port 9095 listener and the proxy default<br/>upstream are gone; the flake comment at agentbox/flake.nix:3777-3778 records that session control<br/>is a unix socket with no TCP port, and the voice Caddyfile has no slash aoe handle (voice/console/Caddyfile:54-105)."]
N6["INVARIANT ADR-2013: sovereign_mesh.relay.expose does NOT open a LAN door. When true it adds<br/>ONE publish and that publish is loopback-pinned - 127.0.0.1:port:port at<br/>agentbox/flake.nix:3787-3788 - so the relay never reaches the SANCTIONED LAN list at<br/>check-ports-loopback.mjs:100-104. Default is expose=false, agentbox.toml:148."]
N3["INVARIANT ADR-2153: verification precedes routing. An unauthenticated request to an unknown path is<br/>refused 302 or 401 before routeFor runs, so the route table cannot be probed anonymously; only a verified<br/>caller sees the 404 (agentbox/config/nip98-proxy/proxy.mjs:843-871). The route table itself is projected from the<br/>manifest into nip98-proxy-config.json at every boot (agentbox/services/agentbox-manifest/src/proxy.rs:23-24,<br/>agentbox/config/lib/nip98-config.sh:17, called at agentbox/config/entrypoint-unified.sh:2346). A mistyped routes or<br/>allowed_pubkeys value is an error and writes nothing, proxy.rs:55-62 and :23-24, so the door is not silently opened"]
N4["RESOLVED ADR-2047: the compose-exposure qualification is superseded in the governing doc.<br/>The line-walker bypass is fixed - check-ports-loopback.sh is a wrapper that execs<br/>check-ports-loopback.mjs, a strict YAML reader for the compose subset, and anything outside that<br/>subset is REJECTED with file and line, never skipped. ADR-2013 stays partial for the DEPLOYMENT<br/>half only: overlay order, interpolation, external files and active-listener evidence.<br/>Receipt: agentbox/docs/estate-closeout/2026-09-05/adr-2013-ports-gate.json."]
N5["INVARIANT ADR-2013: the wrapper FAILS LOUDLY when the .mjs gate is missing - a copy of the<br/>wrapper without its gate exits 3 with an explicit message rather than looking like a pass<br/>agentbox/scripts/ci/check-ports-loopback.sh:38-42"]
```

**Drift:** the governing doc still quotes a flake comment reading `# (aoe serve, --auth token) is NEVER published.` as live at flake.nix line 2647 (`agentbox/docs/INGRESS-identity.md:204-207`) and anchors `verifyIdentity` at proxy.mjs line 626 (`agentbox/docs/INGRESS-identity.md:73`), the break-glass bullet at proxy.mjs lines 753-760 (`agentbox/docs/INGRESS-identity.md:232-235`) and the session secret at proxy.mjs line 107 (`agentbox/docs/INGRESS-identity.md:236-237`), but that comment is gone, the function is at `proxy.mjs:543` (`agentbox/config/nip98-proxy/proxy.mjs:543`), and the doc's own 0.2.4 entry already records the AoE retirement (`agentbox/docs/INGRESS-identity.md:14`).

## AB-10.2 proxy.mjs boot sequence — config, routes, verifier load

```mermaid
sequenceDiagram
    autonumber
    participant BOOT as proxy.mjs boot<br/>agentbox/config/nip98-proxy/proxy.mjs:19
    participant FS as Filesystem<br/>CONFIG_FILE / NOSTR_BRIDGE_PATH
    participant NB as loadNostrBridge<br/>agentbox/config/nip98-proxy/proxy.mjs:393
    participant LOG as log()<br/>agentbox/config/nip98-proxy/proxy.mjs:359

    BOOT->>BOOT: PORT, BIND parsed, no default upstream URL any more (ADR-2153) proxy.mjs:19-20
    BOOT->>BOOT: BREAK_GLASS = roleSecret('NIP98_PROXY_ALLOW_BEARER') proxy.mjs:53<br/>role-secret loader: NAME_FILE, then $AGENTBOX_SECRETS_DIR/NAME,<br/>bare env only while role_isolation is off (proxy.mjs:28-52)
    BOOT->>BOOT: SESSION_TTL_S = NIP98_PROXY_SESSION_TTL default 43200 proxy.mjs:162
    break SESSION_TTL_S not a positive safe integer
        BOOT-->>BOOT: throw Error proxy.mjs:164
    end
    BOOT->>BOOT: SESSION_SECRET = PINNED_SESSION_SECRET or crypto.randomBytes(32).toString(hex) proxy.mjs:166-167<br/>the pinned secret takes the same role-secret path (proxy.mjs:166)
    Note over BOOT: DIVERGENCE session secret is per-boot - NIP-07 sessions do not survive a proxy restart, by design agentbox/docs/INGRESS-identity.md:236 (known divergences)
    Note over BOOT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
    BOOT->>BOOT: NIP98_PROXY_ALLOWED_PUBKEYS split lowercased filtered proxy.mjs:173-176
    break any entry not 64-hex
        BOOT-->>BOOT: throw Error NIP98_PROXY_ALLOWED_PUBKEYS entries must be 64-character hex proxy.mjs:178
    end
    BOOT->>NB: loadNostrBridge() proxy.mjs:443
    alt NOSTR_BRIDGE_PATH explicit
        NB->>FS: existsSync(NOSTR_BRIDGE_PATH) proxy.mjs:401
        alt missing or exports no verifyNip98
            NB-->>LOG: error no fallback proxy.mjs:402,proxy.mjs:411,proxy.mjs:413
            NB-->>BOOT: null
        else
            NB-->>BOOT: NostrBridge proxy.mjs:409
        end
    else no explicit path
        loop source-tree then baked-image candidates proxy.mjs:418-427
            NB->>FS: existsSync(candidate)
            NB->>NB: require(candidate) proxy.mjs:431
        end
        NB-->>BOOT: NostrBridge or null proxy.mjs:429-440
    end
    alt NostrBridge is null
        BOOT-->>LOG: warn nostr-bridge unavailable NIP-98 verification DISABLED fail-closed proxy.mjs:445-451
        Note over BOOT: INVARIANT ADR-2009 - missing verifier rejects every NIP-98 token, only break-glass if configured survives
    else loaded
        BOOT-->>LOG: info nostr-bridge loaded proxy.mjs:408,proxy.mjs:433
    end
    BOOT->>BOOT: REQUIRE_CONFIG = NIP98_PROXY_REQUIRE_CONFIG equals 1, set by the supervisor program proxy.mjs:260<br/>the [program:nip98-proxy] environment line carries it flake.nix:3170
    BOOT->>FS: readFileSync CONFIG_FILE default workspace/.agentbox/nip98-proxy-config.json, text kept as LOADED_CONFIG_RAW proxy.mjs:271-272
    alt file absent or unreadable, ENOENT or EACCES
        alt REQUIRE_CONFIG set
            FS-->>BOOT: CONFIG_PENDING true, errno kept, FILE_CONFIG routes [] allowedPubkeys [] proxy.mjs:276-277
            BOOT-->>LOG: error file absent, refusing every identity until it is readable, fail-closed proxy.mjs:1091-1093
            Note over BOOT: INVARIANT ADR-2210 - pending means pubkeyAllowed returns false for every key, so a validly signed key is refused, not served as any-signer proxy.mjs:183
        else standalone or selftest
            FS-->>BOOT: FILE_CONFIG = routes [] allowedPubkeys [] and the ADR-069 behaviour, any valid signer proxy.mjs:277
        end
    else present
        BOOT->>BOOT: JSON.parse, validate allowedPubkeys 64-hex proxy.mjs:280-288
        break malformed JSON or bad pubkey shape
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:292-293
        end
    end
    BOOT->>BOOT: parse NIP98_PROXY_ROUTES JSON array, normalizeRoute each entry proxy.mjs:299-304
    break not an array, or normalizeRoute throws e.g. bearer_env unset proxy.mjs:227-231
        BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:314-315
    end
    BOOT->>BOOT: FILE_CONFIG.routes appended, env route wins on prefix conflict proxy.mjs:306-309
    opt NIP98_PROXY_MGMT_UPSTREAM set and no existing slash mgmt slash route
        BOOT->>BOOT: push convenience route proxy.mjs:326-330
        break invalid URL
            BOOT-->>BOOT: log error, process.exit(1) proxy.mjs:331-334
        end
    end
    BOOT->>BOOT: server.listen(PORT, BIND) proxy.mjs:1065
    BOOT-->>LOG: info nip98-proxy listening - bind, upstream, routes, nip98 enabled/DISABLED, breakGlass, nip07Sessions ttl and pinned-vs-per-boot, allowedPubkeys count or PENDING proxy.mjs:1066-1088
    opt REQUIRE_CONFIG set, restart-on-config-change watcher
        loop every 2 s by default, NIP98_PROXY_REQUIRE_CONFIG_POLL_MS proxy.mjs:1095,proxy.mjs:1099
            BOOT->>FS: accessSync R_OK, readFileSync, JSON.parse - any failure keeps the state held proxy.mjs:1105-1109
            alt CONFIG_PENDING or file text differs from LOADED_CONFIG_RAW
                BOOT->>BOOT: return while process.uptime under 3.5 s, so the exit never falls inside startsecs proxy.mjs:1098,proxy.mjs:1113
                BOOT-->>LOG: info config appeared or changed, exiting for a supervised restart proxy.mjs:1114
                BOOT->>BOOT: process.exit(0), supervisord restarts it on the new file flake.nix:3172-3173 proxy.mjs:1115
            end
        end
    end
    Note over BOOT: DEBT - the restart reaches only a file that CHANGES. Stage B writes the projection from a program started alongside the proxy,<br/>flake.nix:2878-2884, entrypoint-unified.sh:2346 - until it lands the previous boot's allow-list and routes are served,<br/>and if projection fails the old file is served indefinitely, nip98-config.sh:26-27. ADR-2210 note of 2026-11-20
```

**Drift:** `agentbox/docs/INGRESS-identity.md:119-123` says an unset allowlist means any validly signed key and is silent on an absent or unreadable config file; under the supervisor that file now means refuse every identity until it is readable (`agentbox/config/nip98-proxy/proxy.mjs:183`, `agentbox/config/nip98-proxy/proxy.mjs:276`), and "any validly signed key" survives only as the explicit empty file the boot helper writes when the manifest has no section (`agentbox/config/lib/nip98-config.sh:30`). A validly signed key that is not on a non-empty allowlist is refused with `pubkey_not_allowed` (`agentbox/config/nip98-proxy/proxy.mjs:185`).

## AB-10.3 verifyIdentity — full auth precedence

```mermaid
sequenceDiagram
    autonumber
    participant REQ as Request<br/>HTTP forward() or WS upgrade
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:543
    participant CTE as constantTimeEqual<br/>agentbox/config/nip98-proxy/proxy.mjs:457
    participant NB as NostrBridge.verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:505
    participant CAN as canonicalPubkey<br/>agentbox/config/nip98-proxy/proxy.mjs:197
    participant PA as pubkeyAllowed<br/>agentbox/config/nip98-proxy/proxy.mjs:182
    participant VST as verifySessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:478

    REQ->>VI: verifyIdentity(req, bearerToken, rawBody) proxy.mjs:543
    VI->>VI: authHeader = req.headers.authorization proxy.mjs:544
    alt BREAK_GLASS configured and a Bearer token or query bearerToken matches
        VI->>CTE: constantTimeEqual(token, BREAK_GLASS) proxy.mjs:553
Note over VI,CTE: DIVERGENCE (historical) break-glass bearer over the LAN - a single shared secret<br/>bypasses NIP-98 entirely while NIP98_PROXY_ALLOW_BEARER is set. See the RESOLVED note below -<br/>ADR-2027 (closeout 2026-09-05) bounds it with expiry + scope, still tracked as INGRESS-identity.md:232<br/>(known divergences) pending a doc update
        VI->>VI: RESOLVED ADR-2027 — breakGlassNotExpired() proxy.mjs:554
        alt expiry configured and elapsed, or malformed
            VI-->>REQ: ok false, reason break_glass_expired, audited by fingerprint proxy.mjs:555-563
        end
        VI->>VI: breakGlassScopeAllows(method, url) proxy.mjs:564
        alt NIP98_PROXY_BEARER_SCOPE configured and request outside it
            VI-->>REQ: ok false, reason break_glass_out_of_scope, audited by fingerprint proxy.mjs:565-573
        end
        Note over VI: SECURITY-profiles.md custody row Proxy break-glass bearer - captured at process start,<br/>ADR-2027 adds optional expiry + scope bounds and a per-use fingerprint audit trail (breakGlassUse<br/>counters), still an in-process receipt, not a durable audit store
        VI->>VI: breakGlassUse.accepted++, log warn break-glass USED with fingerprint proxy.mjs:574-586
        VI-->>REQ: ok true, pubkey NIP98_PROXY_BEARER_PUBKEY default break-glass, mode break-glass proxy.mjs:587
    else Authorization starts with Nostr
        VI->>NB: verifyNip98(authHeader, method, signedUrlFor(req), rawBody) proxy.mjs:597
        alt NostrBridge unavailable
            VI-->>REQ: ok false, reason nip98_verifier_unavailable proxy.mjs:593
        else verifyNip98 throws
            VI-->>REQ: ok false, reason nip98_verify_error proxy.mjs:598-599
        else result not valid
            VI-->>REQ: ok false, reason nip98_invalid proxy.mjs:615
        else result.valid true
            VI->>CAN: canonicalPubkey(result.pubkey) proxy.mjs:606
            alt not lowercase 64-hex
                VI-->>REQ: ok false, reason nip98_noncanonical_pubkey proxy.mjs:607-608
            else canonical
                VI->>PA: pubkeyAllowed(pubkey) proxy.mjs:610
                alt config pending, or allowlist non-empty and pubkey missing
                    VI-->>REQ: ok false, reason pubkey_not_allowed proxy.mjs:610-611
                else
                    VI-->>REQ: ok true, pubkey, mode nip98 proxy.mjs:613
                end
            end
        end
    else Cookie carries agentbox_nip07_session
        VI->>VST: verifySessionToken(cookie) proxy.mjs:622
        VST->>CTE: constantTimeEqual(mac, expected) proxy.mjs:486
        VST->>PA: pubkeyAllowed(pubkey) proxy.mjs:487
        alt session valid, not expired, HMAC matches, and allowed
            VI-->>REQ: ok true, pubkey, mode nip07-session proxy.mjs:623-624
        else
            VI-->>REQ: ok false, reason session_invalid_or_expired proxy.mjs:626
        end
    else no credentials at all
        VI-->>REQ: ok false, reason no_credentials proxy.mjs:629
    end
```

## AB-10.4 verifyNip98 internals — header, tags, payload, signature, replay

```mermaid
sequenceDiagram
    autonumber
    participant CALLER as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:597
    participant VN as verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:505
    participant TAG as getTag<br/>agentbox/mcp/servers/nostr-bridge.js:536
    participant SHA as crypto sha256<br/>agentbox/mcp/servers/nostr-bridge.js:571,nostr-bridge.js:578
    participant SIG as verifyEvent nostr-tools<br/>agentbox/mcp/servers/nostr-bridge.js:588
    participant REPLAY as replay LRU<br/>agentbox/mcp/servers/nostr-bridge.js:102

    CALLER->>VN: verifyNip98(authHeader, method, url, rawBody) nostr-bridge.js:505
    break header missing or not Nostr-prefixed
        VN-->>CALLER: valid false, error missing or malformed Nostr header nostr-bridge.js:507
    end
    VN->>VN: event = JSON.parse(Buffer.from(encoded, base64).toString(utf8)) nostr-bridge.js:517
    break not valid base64 JSON
        VN-->>CALLER: valid false, error token is not valid base64 JSON nostr-bridge.js:519
    end
    break event.kind !== kinds.AUTH (27235) nostr-bridge.js:65,nostr-bridge.js:522
        VN-->>CALLER: valid false, error expected kind 27235 got X nostr-bridge.js:523
    end
    break created_at not a number
        VN-->>CALLER: valid false, error missing created_at nostr-bridge.js:527
    end
    break abs(now - created_at) greater than VERIFY_NIP98_WINDOW_S 60s nostr-bridge.js:85,nostr-bridge.js:531
        VN-->>CALLER: valid false, error event timestamp out of 60-second window nostr-bridge.js:532
    end
    VN->>TAG: getTag(method), getTag(u) nostr-bridge.js:543-544
    break method tag mismatch case-insensitive
        VN-->>CALLER: valid false, error method tag mismatch nostr-bridge.js:547
    end
    break url tag mismatch - urlTag not equal url and not endsWith(url)
        VN-->>CALLER: valid false, error url tag mismatch nostr-bridge.js:551
    end
    opt rawBody supplied - Finding 4 payload binding
        VN->>TAG: getTag(payload) nostr-bridge.js:566
        VN->>SHA: hex(sha256(rawBody)) nostr-bridge.js:571,nostr-bridge.js:578
        break body non-empty and payload tag missing, or computed hash mismatch either branch
            VN-->>CALLER: valid false, error missing payload tag for request body OR payload hash mismatch nostr-bridge.js:569,nostr-bridge.js:573,nostr-bridge.js:580
        end
    end
    VN->>SIG: verifyEvent(event) BIP-340 schnorr constant-time nostr-bridge.js:589
    break signature invalid or verifyEvent throws
        VN-->>CALLER: valid false, error invalid Schnorr signature nostr-bridge.js:591,nostr-bridge.js:594
    end
    VN->>REPLAY: _recordReplayId(event.id) nostr-bridge.js:601,nostr-bridge.js:116
Note over REPLAY: REPLAY_TTL_MS = VERIFY_NIP98_WINDOW_S times 1000 = 60000ms,<br/>REPLAY_MAX_ENTRIES = 20000, keyed on event id, populated ONLY after signature verify so a<br/>forged token can never poison the cache nostr-bridge.js:100-102
    break id already seen inside the freshness window - live duplicate
        VN-->>CALLER: valid false, error replayed NIP-98 event id nostr-bridge.js:602
    end
    VN-->>CALLER: valid true, pubkey event.pubkey nostr-bridge.js:605
```

## AB-10.5 Request auth state machine

```mermaid
stateDiagram-v2
    [*] --> Unauthenticated
    Unauthenticated --> BearerAccepted: break-glass token constant-time match, unexpired, in scope proxy.mjs:549-553,proxy.mjs:587
    Unauthenticated --> Nip98Verified: verifyNip98 valid, canonicalPubkey, pubkeyAllowed proxy.mjs:606-613
    Unauthenticated --> SessionCookieVerified: verifySessionToken valid and allowlisted proxy.mjs:620-624
    Unauthenticated --> Rejected302: browser GET, Accept text/html, auth failed proxy.mjs:848-855
    Unauthenticated --> Rejected401: auth failed, not an html GET proxy.mjs:857-862
    Unauthenticated --> FailClosed: NostrBridge null at boot, every Nostr header refused proxy.mjs:445-451,proxy.mjs:593
    Unauthenticated --> FailClosed: config pending under REQUIRE_CONFIG, every NIP-98 and session identity refused pubkey_not_allowed proxy.mjs:183,proxy.mjs:610-611
    BearerAccepted --> Routed: routeFor(req.url) proxy.mjs:866,proxy.mjs:1000
    Nip98Verified --> Routed: routeFor(req.url) proxy.mjs:866,proxy.mjs:1000
    SessionCookieVerified --> Routed: routeFor(req.url) proxy.mjs:866,proxy.mjs:1000
    Routed --> Upstream: named route, nip98 passthrough or bearer_env injection proxy.mjs:906-914
    Routed --> NotFound404: no named prefix matches, routeFor returned null, no upstream contact proxy.mjs:867-870,proxy.mjs:1001-1003
    FailClosed --> [*]
    Rejected401 --> [*]
    Rejected302 --> [*]
    NotFound404 --> [*]
    Upstream --> [*]
```

The current source check at `e458b308` retains the authentication order and named-route boundary. The redundant `isAoe` field and its conditionals have now been removed.

## AB-10.6 Unknown-route door — no default upstream, 404 after verification

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser or client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:839
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:543
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:343
    participant SOCK as agentbox-sessions socket<br/>agentbox/services/agentbox-sessions/src/api.rs:29

    BR->>PX: GET /api/sessions (the retired board path) proxy.mjs:828
    PX->>PX: buffer request body, size-capped MAX_BODY_BYTES 25MiB proxy.mjs:157,proxy.mjs:948-952
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:843
    alt auth not ok
        PX-->>BR: 302 to /nip07 (html GET) or 401 JSON proxy.mjs:848-862
    else auth ok
        PX->>RT: routeFor(req.url) - no prefix matches proxy.mjs:866,proxy.mjs:345-356
        RT-->>PX: null, no default upstream proxy.mjs:356
        PX-->>BR: 404 NotFound, message No operator route, no upstream contact proxy.mjs:867-870
        Note over PX: WebSocket upgrades take the same decision after verifying: socket.end with 404 Not Found proxy.mjs:1000-1003
    end
    Note over BR,SOCK: session control is not a proxy route at all - it is a unix socket with no TCP port, so there is nothing for this door to forward to (agentbox/flake.nix:3777-3778, agentbox/services/agentbox-sessions/src/api.rs:3-7)
```

**Resolved in source:** `routeFor` returns the upstream, path, bearer and host-preservation choice without an AoE flag. HTTP and WebSocket forwarding select signed-header passthrough by auth mode directly (`agentbox/config/nip98-proxy/proxy.mjs:354`, `agentbox/config/nip98-proxy/proxy.mjs:906-914`, `agentbox/config/nip98-proxy/proxy.mjs:1036-1042`).

## AB-10.7 /mgmt/* router door — bearer_env gated behind NIP-98

```mermaid
sequenceDiagram
    autonumber
    participant BR as Client
    participant PX as forward()<br/>agentbox/config/nip98-proxy/proxy.mjs:839
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:543
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:343
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1543
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:184
    participant NB2 as verifyNip98Header<br/>agentbox/management-api/middleware/auth.js:68

    BR->>PX: request /mgmt/v1/system proxy.mjs:828
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:843
    alt not ok
        PX-->>BR: 401 or 302 proxy.mjs:848-862
    else ok
PX->>RT: routeFor(/mgmt/v1/system) matches prefix /mgmt/, strip true<br/>proxy.mjs:346-351
RT-->>PX: upstream 127.0.0.1:MANAGEMENT_API_PORT, path /v1/system,<br/>bearer and preserveHost proxy.mjs:354
Note over PX,RT: a route may opt in to preserve_host, and only then is the client Host<br/>forwarded instead of the upstream host and port (proxy.mjs:240, proxy.mjs:890-891).<br/>The connection TARGET stays fixed by the trusted config either way, so this<br/>lets a browser compare Host with Origin without loosening where a request goes
        alt auth.mode is nip98
PX->>PX: headers.authorization = req.headers.authorization, unchanged<br/>proxy.mjs:906-912
Note over PX: ADR-2010 - a genuinely signed NIP-98 header always reaches<br/>a named upstream, so the governance surface re-verifies the operator<br/>signature itself
        else route declares bearer_env and auth.mode is not nip98
            PX->>PX: headers.authorization = Bearer route.bearer proxy.mjs:913-914
Note over PX: ADR-2010 - bearer_env is injected ONLY when auth.mode is<br/>not nip98, and normalizeRoute is fatal at boot if the named env var is<br/>unset proxy.mjs:227-231
        end
        PX->>MAPI: forward to 127.0.0.1:9090, path /v1/system proxy.mjs:920-929
MAPI->>AUTH: authMiddleware, authMode = MANAGEMENT_API_AUTH_MODE default<br/>hybrid agentbox/management-api/server.js:224-227
Note over AUTH: hybrid AUTO-ELEVATES to strict-nip98 when<br/>AGENTBOX_SOVEREIGN_MESH_ENABLED is true and no mode is set explicitly, and<br/>strict-nip98 rejects Bearer unconditionally<br/>agentbox/management-api/middleware/auth.js:172-182,auth.js:194-195
AUTH->>NB2: verifyNip98Header(authHeader, request) re-verifies the<br/>signature independently agentbox/management-api/middleware/auth.js:68-83
Note over NB2: the URL it checks against the signed u tag is rebuilt from request.host, which<br/>keeps a non-default port, since the Fastify 5 upgrade, agentbox/management-api/middleware/auth.js:73
        alt hybrid and Bearer equals API_KEY
AUTH-->>MAPI: allow, bearer accepted<br/>agentbox/management-api/middleware/auth.js:106-111, auth.js:199
        else nip98Result valid
AUTH-->>MAPI: allow, nip98 accepted<br/>agentbox/management-api/middleware/auth.js:190,auth.js:198
        else Bearer is a live per-task action token #40;ADR-2164#41;, checked only when nothing above authenticated
AUTH-->>MAPI: allow as mode task-token in EVERY auth mode, but only on POST /v1/agent-events/emit, batch<br/>or trajectory, any other route is 403 agentbox/management-api/middleware/auth.js:139-144, auth.js:207-219
        else neither
            AUTH-->>MAPI: 401 agentbox/management-api/middleware/auth.js:234-237
        end
        MAPI-->>PX: response
        PX-->>BR: relay status and body proxy.mjs:927-929
    end
Note over MAPI,AUTH: the AUTH GATE never reads X-Agentbox-Pubkey - createAuthMiddleware<br/>re-verifies Authorization itself on every request<br/>agentbox/management-api/middleware/auth.js:184-241. Two consumers BEHIND the gate do<br/>trust the stamp under ADR-2042 (only from a loopback peer, peerIsProxy) - see AB-10.14
```

## AB-10.8 NIP-07 browser session handshake

```mermaid
sequenceDiagram
    autonumber
    participant BR as Browser
    participant PX as handleNip07<br/>agentbox/config/nip98-proxy/proxy.mjs:765
    participant SIGNER as window.nostr signer
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:543
    participant MINT as mintSessionToken<br/>agentbox/config/nip98-proxy/proxy.mjs:472
    participant UP as named-route upstream, there is no default since ADR-2153

    BR->>PX: GET /nip07/ or /nip07/login proxy.mjs:769
    PX-->>BR: 200 HANDSHAKE_PAGE html proxy.mjs:665-759,769-772
    BR->>SIGNER: waitForSigner(3000ms), poll every 200ms proxy.mjs:706-715,712
    SIGNER-->>BR: window.nostr detected
    BR->>SIGNER: signEvent(kind 27235, tags u and method POST) proxy.mjs:728-734
    SIGNER-->>BR: signed event
    BR->>PX: POST /nip07/session, Authorization Nostr base64(signed) proxy.mjs:787,735-737
    PX->>VI: verifyIdentity(req, undefined, rawBody) proxy.mjs:788
    alt not ok, or mode is not nip98
        PX-->>BR: 401 session minting requires a NIP-98 signature proxy.mjs:792-799
        Note over PX: an existing cookie cannot self-renew, and break-glass cannot launder its sentinel into a pubkey-bound session proxy.mjs:789-791
    else ok and mode nip98
        PX->>MINT: mintSessionToken(auth.pubkey) - v1.pubkey.expiry.hmac proxy.mjs:472-476,801
        Note over MINT: SESSION_SECRET is per-boot random unless NIP98_PROXY_SESSION_SECRET is pinned<br/>through the role-secret loader proxy.mjs:166-167
        Note over MINT: SECURITY-profiles.md custody row Proxy browser-session signing secret - restart invalidation and multi-instance policy unconfirmed
        PX-->>BR: 200, Set-Cookie agentbox_nip07_session HttpOnly SameSite=Lax Secure-if-https Max-Age SESSION_TTL_S proxy.mjs:515-518,805-810
    end
    BR->>PX: subsequent request, Cookie agentbox_nip07_session=token proxy.mjs:168
    PX->>VI: verifyIdentity reads cookie via parseCookies proxy.mjs:491-499,620
    VI-->>PX: ok true, pubkey, mode nip07-session
    PX->>PX: stripSessionCookie removes agentbox_nip07_session before forwarding proxy.mjs:506-513,884
    PX->>UP: forward, x-agentbox-pubkey and x-agentbox-auth-mode nip07-session injected proxy.mjs:898-899
```

GET `/nip07/session` is a separate, unauthenticated-upstream probe: it reads the
cookie and calls `verifySessionToken` itself, nothing else, proxy.mjs:775-779.
INVARIANT: unrelated upstream availability or authorisation must not make a
signed-in browser start prompting its signer again — the probe never touches the
route or upstream, proxy.mjs:776-777. Response is `cache-control: no-store`,
proxy.mjs:781, so a cached 200 can never stand in for a live check.

## AB-10.9 WebSocket upgrade auth path

```mermaid
sequenceDiagram
    autonumber
    participant BR as WS client
    participant PX as server.on(upgrade)<br/>agentbox/config/nip98-proxy/proxy.mjs:967
    participant VI as verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:543
    participant RT as routeFor<br/>agentbox/config/nip98-proxy/proxy.mjs:343
    participant UP as upstream socket<br/>net.connect

    BR->>PX: Upgrade GET /feed with ?access_token= or ?auth= proxy.mjs:967
    PX->>PX: parse query access_token or bearer proxy.mjs:980-981
    opt auth query param present and no Authorization header
        PX->>PX: req.headers.authorization = Nostr plus the auth param proxy.mjs:982-985
Note over PX: DIVERGENCE query-carried credentials, WS-handshake-only - regression fix<br/>restoring the console signer-only auth carrier lost when /feed moved behind this proxy under<br/>ADR-069 proxy.mjs:968-977
    end
    PX->>VI: verifyIdentity(req, queryToken) proxy.mjs:988
    alt not ok
        PX-->>BR: HTTP/1.1 401 Unauthorized, then socket.destroy proxy.mjs:989-998
        Note over PX: DIVERGENCE break-glass bearer accepted via ?access_token= or ?bearer= on WS upgrades over the LAN agentbox/docs/INGRESS-identity.md:232 (known divergences)
    else ok
        PX->>RT: routeFor(req.url) proxy.mjs:1000
        alt no named route matches
            PX-->>BR: HTTP/1.1 404 Not Found, Content-Length 0, then end, no upstream socket opened proxy.mjs:1001-1003
        else named route
            PX->>UP: net.connect, rebuild request line, strip Authorization, inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:1005-1033
            alt auth.mode is nip98
                PX->>UP: Authorization req.headers.authorization passthrough proxy.mjs:1036-1040
            else route.bearer set and auth.mode is not nip98
                PX->>UP: Authorization Bearer route.bearer proxy.mjs:1041-1042
            end
            PX->>PX: strip session cookie before forwarding on WS too proxy.mjs:1022-1026
            PX->>UP: pipe socket bidirectionally proxy.mjs:1047-1048
        end
    end
```

## AB-10.10 Direct :9090 path vs proxied :9090 path

```mermaid
sequenceDiagram
    autonumber
    participant DC as Direct caller (container-internal)
    participant MAPI as management-api server<br/>agentbox/management-api/server.js:1543, HOST 0.0.0.0 PORT 9090
    participant AUTH as authMiddleware<br/>agentbox/management-api/middleware/auth.js:184
    participant BR2 as Browser via port 9096
    participant PX2 as proxy /mgmt/ route<br/>agentbox/config/nip98-proxy/proxy.mjs:324

    Note over DC,MAPI: port 9090 is published to the host only as 127.0.0.1:9090:9090 agentbox/flake.nix:3782 - DC must already be container-internal or on the loopback publish
    DC->>MAPI: request with its OWN Authorization Bearer API_KEY or Nostr header, no X-Agentbox-Pubkey
    MAPI->>AUTH: authMiddleware verifies bearer or nip98 directly agentbox/management-api/server.js:224-232
    AUTH-->>MAPI: allow, 401, or 403 for a task action token off its three agent-event routes agentbox/management-api/middleware/auth.js:207-219

    BR2->>PX2: request /mgmt/... over 9096, with NIP-98 or session cookie
    PX2->>PX2: verifyIdentity, then routeFor, then inject x-agentbox-pubkey and x-agentbox-auth-mode proxy.mjs:843,proxy.mjs:866,proxy.mjs:898-899
    PX2->>MAPI: forward to 127.0.0.1:9090, Authorization is passthrough (nip98) or bearer_env (else) per ADR-2010
    MAPI->>AUTH: authMiddleware STILL independently verifies Authorization, same code path as the direct request agentbox/management-api/middleware/auth.js:184-236
Note over MAPI,AUTH: the only difference at the GATE is WHO ADDS the pubkey header - only the<br/>proxied path carries x-agentbox-pubkey and x-agentbox-auth-mode, and createAuthMiddleware<br/>requires neither agentbox/management-api/middleware/auth.js:184-241. Route handlers past the<br/>gate are a different matter - see AB-10.14
```

## AB-10.11 Session-socket door — SO_PEERCRED ACL and the nostr-gateway client

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator, /spawn command
    participant GW as sessionCreate<br/>agentbox/config/nostr-gateway/gateway.cjs:467
    participant CL as client.call<br/>agentbox/config/session-plane/client.cjs:4
    participant SK as handle<br/>agentbox/services/agentbox-sessions/src/api.rs:67
    participant PL as Plane.create<br/>agentbox/services/agentbox-sessions/src/plane.rs:485

    OP->>GW: /spawn dir agent text - doSpawn prefers the session plane unless the launcher is zai gateway.cjs:514-516
    alt repoPath is not the project root
        GW-->>OP: throw using tmux for alternate project, doSpawn falls back to the confined tmux new-window gateway.cjs:470,gateway.cjs:525-529
    else project root
        GW->>CL: call op create, slug from the title, tool gateway.cjs:471-473
        CL->>SK: connect to AGENTBOX_SESSIONS_SOCKET, write one JSON line, 12 s timeout client.cjs:4-11
        SK->>SK: socket_peercred reads the peer uid, unavailable credentials refuse api.rs:70-73
        alt uid not in the ACL
            SK-->>CL: ok false, uid N is not authorised api.rs:74-76
            CL-->>GW: throws, doSpawn logs and falls back to tmux client.cjs:18, gateway.cjs:525-527
        else uid 1000 devuser or 961 ab-gateway
            SK->>PL: dispatch op create api.rs:83-95
            PL-->>SK: bail on a bad slug, a manifest slug or an existing session plane.rs:487-497
            SK-->>CL: ok true client.cjs:19
        end
        GW->>CL: optional follow-up send, text then pause then Enter gateway.cjs:519-521, plane.rs:432-447
    end
    Note over SK: INVARIANT ADR-2153 - the ACL decides, never the file mode: the socket is 0660 in group ab-sessions (gid 973, members devuser and ab-gateway) but even root is refused unless listed<br/>api.rs:29,api.rs:128, config/role-accounts.json:139-144, config/custody/session-plane-acl.json:8-11
    Note over SK: the daemon is supervised as devuser with the ACL path and gid in its environment, autostart true<br/>flake.nix:3153-3158, gid taken from the one role table at eval time flake.nix:1958-1967
    Note over GW: before any command runs, handleWrap drops NIP-17 self-DM copies whose p tag<br/>names another identity and poker-coach table traffic, gateway.cjs:720,735-736,741 -<br/>and with the flag off the operator key is read only through hooks/lib/operator-key.cjs, while under role_isolation<br/>the gateway asks the identity port for public keys and holds no key (gateway.cjs:81-82,153-170)
```

## AB-10.12 selftest.mjs assertion families

```mermaid
flowchart TB
    HARNESS["selftest.mjs harness<br/>agentbox/config/nip98-proxy/selftest.mjs:274 main()"]
    PUT["proxy.mjs under test<br/>plus separate child processes<br/>for boot-config-specific cases"]
    FAKE["fake stub, mgmt and governance upstreams<br/>stub registered as named routes<br/>selftest.mjs:288-291"]
    A["A - no credentials, 401<br/>selftest.mjs:306"]
    B["B - break-glass bearer, 200, identity injected,<br/>Authorization stripped<br/>selftest.mjs:312"]
    C["C - valid NIP-98, 200, skips gracefully if<br/>nostr-tools unresolvable<br/>selftest.mjs:333"]
    D["D - WebSocket upgrade, forwarded with<br/>injected identity<br/>selftest.mjs:362"]
    E["E - routed prefix ADR-045, mgmt on second upstream,<br/>unrouted HTTP and WS get 404, ADR-2153<br/>selftest.mjs:414"]
    F["F - NIP-07 sessions, handshake page, redirect,<br/>cookie auth, strip, forged/expired reject<br/>selftest.mjs:447"]
    G["G - spoofed identity headers stripped and<br/>replaced, ADR-2009<br/>selftest.mjs:546"]
    H["H - verifier faults, absent/throwing/non-canonical,<br/>zero upstream contact, ADR-2009/2011<br/>selftest.mjs:594"]
    I["I - allowlist removal denies the next request,<br/>including a live session, ADR-2009<br/>selftest.mjs:685"]
    J["J - cookie expiry and restart under a rotated<br/>HMAC key, rejected<br/>selftest.mjs:726"]
    K["K - tokenless denial, nothing forwarded,<br/>ADR-2002<br/>selftest.mjs:755"]
    L["L - bearer gated behind NIP-98 on a named route,<br/>ADR-2010<br/>selftest.mjs:775"]
    M["M - hex-canonical identity helper, 0600 key file,<br/>restart stability, ADR-2011, see AB-11<br/>selftest.mjs:864"]
    N["N - break-glass is BOUNDED authority, expiry,<br/>scope and fingerprint audit, ADR-2027<br/>selftest.mjs:937"]
    NOTE1["127 assert() call sites plus the helper itself. NODE_PATH must resolve<br/>nostr-tools or the live-signature cases skip -<br/>config/nip98-proxy/README.md:20-21, harness at selftest.mjs:88"]

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

**Resolved in source:** family K now asserts tokenless HTTP, WebSocket and named-route denial before any upstream contact (`agentbox/config/nip98-proxy/selftest.mjs:755-772`). The retired daemon-token assertion is gone; E2 still checks the authenticated unknown-route 404 pair (`agentbox/config/nip98-proxy/selftest.mjs:430-437`).

## AB-10.14 ADR-2042 — who actually trusts X-Agentbox-Pubkey

```mermaid
sequenceDiagram
    autonumber
    participant CL as Caller
    participant PX as nip98-proxy<br/>agentbox/config/nip98-proxy/proxy.mjs:839
    participant GATE as createAuthMiddleware<br/>agentbox/management-api/middleware/auth.js:184
    participant AEA as verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:135
    participant APL as resolveAgentDid<br/>agentbox/management-api/lib/action-plane.js:222

    CL->>PX: request carrying its own x-agentbox-pubkey
    PX->>PX: inbound copy DROPPED, then re-injected from the AUTHENTICATED identity proxy.mjs:880-881,proxy.mjs:898-899
    Note over PX: on WS the same strip-and-restamp applies proxy.mjs:1015,proxy.mjs:1032-1033
    PX->>GATE: forward, Authorization passthrough (nip98) or bearer_env (else) proxy.mjs:906-914
    GATE->>GATE: verifyNip98Header and verifyBearerHeader on Authorization ONLY auth.js:189-199
    Note over GATE: INVARIANT the gate never reads x-agentbox-pubkey - it re-verifies the credential itself auth.js:184-241
    GATE->>AEA: POST /v1/agent-events/emit reaches the route past the gate
    alt request.auth.mode is task-token, already verified at the gate
        AEA->>AEA: ok with did null and the token's taskId, under every policy agent-event-auth.js:139-145
        Note over AEA: reconcileTaskScope refuses a body that names another task or asserts source_urn or pubkey, 403 agent-event-auth.js:216-235
    else Authorization starts with Nostr
        AEA->>AEA: verify the signature against this request, a bad signature is NEVER rescued by the header agent-event-auth.js:153-183
    else no Nostr header - NIP-07 session or break-glass hop
        AEA->>AEA: ADR-2042 path 2, trust x-agentbox-pubkey when the TCP peer is the proxy and it is 64 lowercase hex agent-event-auth.js:116-122,agent-event-auth.js:192-195
        Note over AEA: DOC-DRIFT AB-10.7 and AB-10.10 previously asserted management-api has ZERO<br/>references to X-Agentbox-Pubkey. False - this is the ONLY identity carrier on that hop,<br/>because the proxy replaced Authorization with the route bearer
    end
    GATE->>APL: any action-plane dispatch past the gate
    APL->>APL: DID priority request.auth.pubkey, then x-agentbox-pubkey, then AGENTBOX_AGENT_DID, then null action-plane.js:222-233
    Note over AEA,APL: INVARIANT both consumers go through proxyVerifiedPubkey, which returns null unless peerIsProxy holds,<br/>so the stamp counts only from a loopback peer or AGENTBOX_PROXY_PEERS, and a request with no socket fails closed<br/>agent-event-auth.js:98-105,117 and action-plane.js:229
    Note over AEA,APL: DEBT residual - a process INSIDE the container can still connect over loopback and forge the stamp, the same-UID boundary (TODO-unified X-1)<br/>a MANAGEMENT_API_KEY holder on the docker network reaches the 0.0.0.0 bind but its peer is not loopback, agent-event-auth.js:34-39
    Note over AEA: policy gate AGENTBOX_AGENT_EVENT_AUTH, default nip98 since ADR-2044, off is<br/>explicitly selectable and returns did null agent-event-auth.js:147-150
```

The governing doc states Invariant 2 as "X-Agentbox-Pubkey always proxy-injected,
never trusted inbound" — true at the ingress, and the two ADR-2042 consumers above
are the deliberate, documented exception on the far side of the gate.

**Invariant:** the stamp counts only when `request.raw.socket.remoteAddress` is loopback (`127.0.0.1`, `::1`, `::ffff:127.0.0.1`) or an exact address in `AGENTBOX_PROXY_PEERS`, and a request with no socket information fails closed; `server.js` binds `0.0.0.0`, so a `MANAGEMENT_API_KEY` holder on the docker network reaches the port but cannot forge any `did:nostr` (`agentbox/management-api/lib/agent-event-auth.js:98-105`, `agentbox/management-api/lib/agent-event-auth.js:117`, `agentbox/management-api/server.js:50`). `resolveAgentDid` calls the same exported helper, so the two ADR-2042 consumers share one rule (`agentbox/management-api/lib/action-plane.js:229`).

**Debt:** the loopback test authenticates the network position, not the process: anything inside the container can open its own loopback connection and forge the stamp (the same-UID boundary, TODO-unified X-1), (`agentbox/management-api/lib/agent-event-auth.js:37-39`).
