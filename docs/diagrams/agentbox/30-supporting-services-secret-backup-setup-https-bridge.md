---
id: AB-30
title: Supporting services — secret-backup, setup server, https-bridge
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [legacy-ADR-024, ADR-2027]
sources:
  - ../project/agentbox/setup/server/src/main.rs
  - ../project/agentbox/setup/server/src/lib.rs
  - ../project/agentbox/setup/agentbox.default.toml
  - ../project/agentbox/scripts/start-agentbox.sh
  - ../project/agentbox/docs/user/quickstart.md
  - ../project/agentbox/https-bridge/https-proxy.js
  - ../project/agentbox/https-bridge/package.json
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/services/secret-backup/src/main.rs
  - ../project/agentbox/docs/archive/adr/ADR-024-setup-dashboard.md
  - ../project/agentbox/docs/BASELINE-container.md
verified_commit: 6500582dfae23bf8a38a6942916d208154e43625
---

## AB-30.2 setup/server — compiled binary or nothing (legacy-ADR-024 D1 superseded in the launcher)

```mermaid
flowchart TD
    START(["operator runs ./scripts/start-agentbox.sh"]) --> TUI{"first arg == --tui ?<br/>start-agentbox.sh:425"}
    TUI -->|yes| LEGACY["legacy shell wizard #40;whiptail or dialog#41;<br/>skips the browser UI entirely<br/>start-agentbox.sh:236"]
    TUI -->|no| BINCHECK{"agentbox-setup binary found at one of 3<br/>candidate paths?<br/>start-agentbox.sh:430-434"}
    BINCHECK -->|yes| EXECBIN["exec agentbox-setup CONFIG_FILE schema/agentbox.toml.schema.json<br/>start-agentbox.sh:436"]
    BINCHECK -->|no| REFUSE["print 'Browser wizard unavailable', name the build command<br/>and --tui, exit 1 — no python or static fallback<br/>start-agentbox.sh:439-447"]
    EXECBIN --> AXUM["axum::Router — /api/config, /api/config/edit, /api/shutdown,<br/>/api/proxy/#123;*path#125;, fallback serve_frontend<br/>setup/server/src/main.rs:260-266"]
    AXUM --> BIND["TcpListener::bind AGENTBOX_SETUP_BIND, default 127.0.0.1:0 — EPHEMERAL port<br/>setup/server/src/main.rs:270-274"]
    BIND --> WARN{"bound address loopback?<br/>setup/server/src/main.rs:275"}
    WARN -->|no| RED["red warning: unauthenticated, can rewrite config and use the mgmt key<br/>setup/server/src/main.rs:276"]
    WARN -->|yes| OPEN["open::that#40;url#41; unless AGENTBOX_SETUP_NO_OPEN<br/>setup/server/src/main.rs:291"]
    RED --> OPEN
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT: the Rust binary defaults to an ephemeral loopback port and prints the<br/>resolved address #40;setup/server/src/main.rs:270,279#41;. A fixed or non-loopback bind is an<br/>explicit AGENTBOX_SETUP_BIND opt-in #40;setup/server/src/main.rs:268-269#41;"]
        N2["CHANGE: the python http.server and pure-browser tiers are gone. A static page cannot<br/>apply comment-preserving edits, so the server is required #40;start-agentbox.sh:439-443#41;"]
        N1 ~~~ N2
    end
```

**Drift:** `docs/user/quickstart.md` still teaches the three-tier wizard, with a python `http.server` tier and a pure-browser drag-and-drop tier (`../project/agentbox/docs/user/quickstart.md:80-84`), and says no dependency is needed beyond Python 3 (`../project/agentbox/docs/user/quickstart.md:75`); the launcher now exits 1 without the compiled binary (`../project/agentbox/scripts/start-agentbox.sh:444-447`).

## AB-30.3 setup/server API surface — config round trip and the management-API proxy

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator's browser
    participant AX as axum Router<br/>setup/server/src/main.rs:260
    participant CFG as get_config / edit_config / save_config<br/>setup/server/src/main.rs:69,86,93
    participant DISK as agentbox.toml on disk
    participant PROXY as proxy_to_mgmt_api<br/>setup/server/src/main.rs:112
    participant MGMT as management API port 9090

    OP->>AX: GET /api/config
    AX->>CFG: get_config(state)
    CFG->>DISK: read config_path
    alt config file missing (fresh install)
        CFG-->>OP: seeded=true from the SHIPPED agentbox.default.toml<br/>#40;described as all learning/hygiene gates OFF#41;, NOT the live toml — setup/server/src/main.rs:70-76
    else config present
        CFG-->>OP: {toml_content, seeded, parsed, schema, ui, apply} — schema read from schema/agentbox.toml.schema.json<br/>setup/server/src/main.rs:77-82
    end
    OP->>AX: POST /api/config/edit {toml_content, edits}
    AX->>CFG: edit_config(req)
    CFG-->>OP: apply_edits rewrites only the named value tokens, comments kept<br/>text returned, disk untouched — setup/server/src/main.rs:86-91, lib.rs:63
    OP->>AX: POST /api/config {toml_content}
    AX->>CFG: save_config(state, req)
    CFG->>CFG: validate#40;#41; FIRST — toml_edit DocumentMut then toml::Value<br/>setup/server/src/main.rs:97, lib.rs:245-248 — malformed TOML never reaches disk
    alt parse fails
        CFG-->>OP: 400 Bad Request "Invalid TOML: <e>"
    else parse ok
        CFG->>DISK: atomic_write — temp file, fsync, rename over the target<br/>setup/server/src/main.rs:100, lib.rs:254
        CFG-->>OP: 200 OK
    end
    OP->>AX: any /api/proxy/{*path}
    AX->>PROXY: proxy_to_mgmt_api(method, path, query, body)
    PROXY->>PROXY: load_mgmt_key#40;#41; — tries /var/lib/agentbox/secrets/mgmt-key,<br/>~/.agentbox/mgmt-key, then MANAGEMENT_API_KEY env — setup/server/src/main.rs:193-211
    PROXY->>MGMT: forward with Authorization Bearer <key> if found
    alt management API unreachable
        MGMT--xPROXY: connection refused
        PROXY-->>OP: 503 container_unreachable, Is the agentbox container running<br/>setup/server/src/main.rs:162-167
    else reachable
        MGMT-->>PROXY: response
        PROXY-->>OP: status + body passed through
    end
    OP->>AX: POST /api/shutdown
    AX->>AX: shutdown.notify_one#40;#41; — setup/server/src/main.rs:107-108
    AX-->>AX: tokio::select! resolves the shutdown branch, process exits<br/>setup/server/src/main.rs:296-308
```

**Drift (default seed vs its own claim):** the seed comment says every learning and hygiene gate ships off (`../project/agentbox/setup/server/src/main.rs:70-72`, repeated in the file header `../project/agentbox/setup/agentbox.default.toml:5-6`), but the shipped default sets `[memory_learning] enabled = true` (`../project/agentbox/setup/agentbox.default.toml:451`) and `[seo_watch] enabled = true` (`../project/agentbox/setup/agentbox.default.toml:845`), so a fresh-install operator is offered both already on.

## AB-30.4 https-bridge — self-signed TLS termination in front of a plain-HTTP target

```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord [program:https-bridge]<br/>flake.nix:2697-2706
    participant BOOT as entrypoint root phase<br/>flake.nix:4245-4260
    participant OSSL as openssl req -x509
    participant NODE as https-proxy.js<br/>process.env-driven config
    participant BROWSER as Browser client
    participant TARGET as http://HOST_IP:TARGET_PORT

    BOOT->>BOOT: mkdir -p /var/lib/https-bridge/certs (tmpfs, uid 1000)<br/>flake.nix:4251, flake.nix:3728
    alt server.key already present
        BOOT-->>NODE: skip generation — cert persists for the tmpfs lifetime
    else missing
        BOOT->>OSSL: openssl req -x509 -newkey rsa:2048 -days 365 -nodes -subj "/CN=localhost"<br/>flake.nix:4253-4256
        OSSL-->>BOOT: server.key #40;0600#41;, server.crt #40;0644#41; — flake.nix:4258-4259
    end
    Note over SUP,BOOT: GATE — the program block is emitted only when [sovereign_mesh].enabled AND https_bridge are both true (flake.nix:2695, setup/agentbox.default.toml:44,47),<br/>while the certificate pre-generation above is not gated on either (flake.nix:4252), so a container without the program still carries the cert files
    SUP->>NODE: node https-proxy.js<br/>CERT_DIR=/var/lib/https-bridge/certs, MANAGEMENT_API_PORT env — flake.nix:2701
    Note over SUP,NODE: flake.nix:2173 copies ./https-bridge into the built image. SUP's full<br/>environment= is HOME, MANAGEMENT_API_PORT, CERT_DIR, SSL_KEY, SSL_CERT #40;flake.nix:2701#41;,<br/>stdout/stderr logged to /var/log/https-bridge.log + .error.log #40;flake.nix:2705-2706#41;.<br/>Its tmpfs #40;mode=755 size=8M uid/gid=1000#41; is declared in BOTH docker-compose.yml:115<br/>AND flake.nix:3728 — keep the two in sync
    NODE->>NODE: ensureCertificates#40;#41; — fs.existsSync check, hand-rolled node:crypto<br/>X.509 builder ONLY IF openssl's boot-time generation is somehow absent<br/>https-proxy.js:48-49,67 — buildSelfSignedX509 at :79
    Note over BOOT,NODE: DESIGN: the trusted path is openssl #40;flake.nix#41; — the hand-rolled builder in<br/>https-proxy.js is a fail-open FALLBACK only, per the boot-script's own comment<br/>#40;flake.nix:4245-4250#41; — not the primary certificate source
    NODE->>NODE: https.createServer#40;{key, cert}#41;.listen#40;HTTPS_PORT, HTTPS_HOST#41;<br/>https-proxy.js:190,262 — HTTPS_HOST defaults 0.0.0.0, published loopback-only<br/>via compose #40;R-003 comment, https-proxy.js:31-33#41;
    BROWSER->>NODE: HTTPS request to localhost:HTTPS_PORT
    NODE->>NODE: detectGatewayIP#40;#41; if HOST_IP unset — `ip route | grep default`<br/>https-proxy.js:21-28, falls back to 192.168.0.51 on any failure
    NODE->>TARGET: http.request — forwards method/headers, adds x-forwarded-proto https<br/>https-proxy.js:193-203
    TARGET-->>NODE: response
    NODE->>NODE: set Access-Control-Allow-Origin * AND Access-Control-Allow-Credentials true<br/>https-proxy.js:208-211
    NODE-->>BROWSER: response, CORS headers attached
    Note over NODE: DIVERGENCE: Access-Control-Allow-Origin "*" combined with<br/>Access-Control-Allow-Credentials "true" is a combination browsers reject for<br/>credentialed requests per the Fetch spec — the two headers as written cannot<br/>both take effect for any request that actually carries credentials
    Note over SUP: BASELINE-container.md:89 supervisord table names this program 'pod HTTPS<br/>bridge' — its own module doc-comment #40;https-proxy.js:3-4#41; describes it more<br/>generically as solving cross-origin issues for local dev against the<br/>management API. Both describe the same mechanism, framing differs by doc
```
