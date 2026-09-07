---
id: AB-30
title: Supporting services — secret-backup, setup server, https-bridge
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [legacy-ADR-024, ADR-2027]
sources:
  - ../project/agentbox/setup/server/src/main.rs
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
verified_commit: 2c521c5bb
---

## AB-30.1 Three name-dropped services, three different lifecycles

```mermaid
flowchart TB
    subgraph secretbackup["services/secret-backup — see AB-16.12 for the full diagram"]
        SB["agentbox-secret-backup Cargo bin<br/>age-encrypted archive backup/restore"]
    end
    subgraph setup["setup/ — the onboarding wizard"]
        SETUPBIN["setup/server #40;Rust, axum#41;<br/>src/main.rs"]
        FRONTEND["setup/frontend/dist #40;built SPA#41;"]
        LAUNCHER["scripts/start-agentbox.sh<br/>tier-selection logic"]
    end
    subgraph httpsbridge["https-bridge/ — TLS termination for local dev"]
        PROXY["https-proxy.js<br/>self-signed HTTPS to plain HTTP"]
    end
    LAUNCHER -->|"tier 1: binary present"| SETUPBIN
    LAUNCHER -->|"tier 2: binary absent, frontend present"| FRONTEND
    LAUNCHER -->|"tier 3: python3 absent too"| FRONTEND
    SETUPBIN --> FRONTEND
    subgraph notes["Why these three are grouped"]
        direction TB
        N1["None of the three is baked into the image's supervised process set as a<br/>continuously-running gate the way most agentbox.toml sections are — secret-backup<br/>is an operator-run CLI tool, setup/server runs ONCE per onboarding session then<br/>exits, and https-bridge is the one exception #40;it IS a supervised program, see<br/>AB-30.5#41;. All three were previously reduced to a manifest catalogue row or a<br/>single prose mention elsewhere in this topic tree"]
        N2["CI workflow coverage for agentbox is diagrammed in ES-09.11-09.18, not here or<br/>anywhere else in the AB-NN range — do not re-add it"]
        N1 ~~~ N2
    end
```

## AB-30.2 setup/server — three-tier fallback (legacy-ADR-024 D1)

```mermaid
flowchart TD
    START(["operator runs ./scripts/start-agentbox.sh"]) --> TUI{"first arg == --tui ?<br/>start-agentbox.sh:425"}
    TUI -->|yes| LEGACY["legacy shell wizard #40;gum/whiptail#41;<br/>skips the browser UI entirely"]
    TUI -->|no| BINCHECK{"agentbox-setup binary found at one of 3<br/>candidate paths?<br/>start-agentbox.sh:429-434"}
    BINCHECK -->|"yes — tier 1"| EXECBIN["exec agentbox-setup CONFIG_FILE schema/agentbox.toml.schema.json<br/>start-agentbox.sh:436"]
    BINCHECK -->|no| FRONTENDCHECK{"setup/frontend/dist/index.html exists?<br/>start-agentbox.sh:440"}
    FRONTENDCHECK -->|"yes, python3 present — tier 2"| COPYFILES["cp agentbox.toml + schema alongside the frontend HTML<br/>start-agentbox.sh:442-445"]
    COPYFILES --> PYSERVE["exec python3 -m http.server SETUP_PORT --directory DIST_DIR --bind 127.0.0.1<br/>ephemeral port via a throwaway socket bind, start-agentbox.sh:458,470"]
    FRONTENDCHECK -->|"no python3 — tier 3"| MANUAL["operator opens setup/frontend/dist/index.html directly<br/>drag-and-drop or file-picker load, save via browser download<br/>quickstart.md:70"]
    EXECBIN --> AXUM["axum::Router — /api/config, /api/shutdown,<br/>/api/proxy/#123;*path#125;, fallback serve_frontend<br/>setup/server/src/main.rs:222-228"]
    AXUM --> BIND["TcpListener::bind 127.0.0.1:0 — EPHEMERAL port, not fixed<br/>setup/server/src/main.rs:229-232"]
    BIND --> OPEN["open::that#40;url#41; — auto-launch the OS default browser<br/>setup/server/src/main.rs:246"]
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT: the Rust binary tier NEVER hard-codes a port — audit reports of<br/>fixed ports #40;2104-2106, 2126-2127#41; describe a DIFFERENT case list elsewhere in<br/>this topic tree #40;see AB-05#41;, not this service. This binary always binds ephemeral port 0 and<br/>prints the resolved ephemeral address #40;setup/server/src/main.rs:234-241#41;"]
        N2["DIVERGENCE: three fallback tiers exist so setup works with zero installed<br/>dependencies beyond python3 #40;quickstart.md:61#41; — but only tier 1 #40;the compiled<br/>binary#41; can write agentbox.toml server-side #40;save_config, setup/server/src/main.rs:60-77#41;;<br/>tier 3 saves via a browser file download instead of writing back in place"]
        N1 ~~~ N2
    end
```

## AB-30.3 setup/server API surface — config round trip and the management-API proxy

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator's browser
    participant AX as axum Router<br/>setup/server/src/main.rs:222
    participant CFG as get_config / save_config<br/>setup/server/src/main.rs:39,60
    participant DISK as agentbox.toml on disk
    participant PROXY as proxy_to_mgmt_api<br/>setup/server/src/main.rs:85
    participant MGMT as management API port 9090

    OP->>AX: GET /api/config
    AX->>CFG: get_config(state)
    CFG->>DISK: read config_path
    alt config file missing (fresh install)
        CFG-->>OP: seed from the SHIPPED agentbox.default.toml<br/>#40;all learning/hygiene gates OFF#41;, NOT the live toml — setup/server/src/main.rs:44
    else config present
        CFG-->>OP: {toml_content, schema} — schema read from schema/agentbox.toml.schema.json
    end
    OP->>AX: POST /api/config {toml_content}
    AX->>CFG: save_config(state, req)
    CFG->>CFG: parse as toml_edit::DocumentMut FIRST<br/>setup/server/src/main.rs:64-66 — malformed TOML never reaches disk
    alt parse fails
        CFG-->>OP: 400 Bad Request "Invalid TOML: <e>"
    else parse ok
        CFG->>DISK: write config_path (whole-file overwrite)
        CFG-->>OP: 200 OK
    end
    OP->>AX: any /api/proxy/{*path}
    AX->>PROXY: proxy_to_mgmt_api(method, path, query, body)
    PROXY->>PROXY: load_mgmt_key#40;#41; — tries /var/lib/agentbox/secrets/mgmt-key,<br/>~/.agentbox/mgmt-key, then MANAGEMENT_API_KEY env — setup/server/src/main.rs:166-184
    PROXY->>MGMT: forward with Authorization Bearer <key> if found
    alt management API unreachable
        MGMT--xPROXY: connection refused
        PROXY-->>OP: 503 container_unreachable, Is the agentbox container running<br/>setup/server/src/main.rs:135-139
    else reachable
        MGMT-->>PROXY: response
        PROXY-->>OP: status + body passed through
    end
    OP->>AX: POST /api/shutdown
    AX->>AX: shutdown.notify_one#40;#41; — setup/server/src/main.rs:81
    AX-->>AX: tokio::select! resolves the shutdown branch, process exits<br/>setup/server/src/main.rs:251-262
```

## AB-30.4 https-bridge — self-signed TLS termination in front of a plain-HTTP target

```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord [program:https-bridge]<br/>flake.nix:2180-2189
    participant BOOT as entrypoint root phase<br/>flake.nix:3360-3387
    participant OSSL as openssl req -x509
    participant NODE as https-proxy.js<br/>process.env-driven config
    participant BROWSER as Browser client
    participant TARGET as http://HOST_IP:TARGET_PORT

    BOOT->>BOOT: mkdir -p /var/lib/https-bridge/certs (tmpfs, uid 1000)<br/>flake.nix:3362,3379
    alt server.key already present
        BOOT-->>NODE: skip generation — cert persists for the tmpfs lifetime
    else missing
        BOOT->>OSSL: openssl req -x509 -newkey rsa:2048 -days 365 -nodes -subj "/CN=localhost"<br/>flake.nix:3378-3380
        OSSL-->>BOOT: server.key #40;0600#41;, server.crt #40;0644#41; — flake.nix:3381-3383
    end
    SUP->>NODE: node https-proxy.js<br/>CERT_DIR=/var/lib/https-bridge/certs, MANAGEMENT_API_PORT env — flake.nix:2184
    NODE->>NODE: ensureCertificates#40;#41; — fs.existsSync check, hand-rolled node:crypto<br/>X.509 builder ONLY IF openssl's boot-time generation is somehow absent<br/>https-proxy.js:48-49,67 — buildSelfSignedX509 at :79
    Note over BOOT,NODE: DESIGN: the trusted path is openssl #40;flake.nix#41; — the hand-rolled builder in<br/>https-proxy.js is a fail-open FALLBACK only, per the boot-script's own comment<br/>#40;flake.nix:3385-3386#41; — not the primary certificate source
    NODE->>NODE: https.createServer#40;{key, cert}#41;.listen#40;HTTPS_PORT, HTTPS_HOST#41;<br/>https-proxy.js:190,262 — HTTPS_HOST defaults 0.0.0.0, published loopback-only<br/>via compose #40;R-003 comment, https-proxy.js:31-33#41;
    BROWSER->>NODE: HTTPS request to localhost:HTTPS_PORT
    NODE->>NODE: detectGatewayIP#40;#41; if HOST_IP unset — `ip route | grep default`<br/>https-proxy.js:21-28, falls back to 192.168.0.51 on any failure
    NODE->>TARGET: http.request — forwards method/headers, adds x-forwarded-proto https<br/>https-proxy.js:193-203
    TARGET-->>NODE: response
    NODE->>NODE: set Access-Control-Allow-Origin * AND Access-Control-Allow-Credentials true<br/>https-proxy.js:208-211
    NODE-->>BROWSER: response, CORS headers attached
    Note over NODE: DIVERGENCE: Access-Control-Allow-Origin "*" combined with<br/>Access-Control-Allow-Credentials "true" is a combination browsers reject for<br/>credentialed requests per the Fetch spec — the two headers as written cannot<br/>both take effect for any request that actually carries credentials
```

## AB-30.5 https-bridge in the supervised process set — the one service of the three that IS

```mermaid
flowchart LR
    FLAKE["flake.nix baseline services list<br/>flake.nix:1682 copies ./https-bridge into the image"]
    SUP["[program:https-bridge]<br/>flake.nix:2180"]
    LOG["/var/log/https-bridge.log + .error.log<br/>flake.nix:2188-2189"]
    ENV["environment= HOME, MANAGEMENT_API_PORT,<br/>CERT_DIR, SSL_KEY, SSL_CERT<br/>flake.nix:2184"]
    TMPFS["/var/lib/https-bridge tmpfs<br/>mode=755 size=8M uid=1000 gid=1000<br/>docker-compose.yml:123, flake.nix:2877-2880"]
    FLAKE --> SUP --> LOG
    SUP --> ENV
    SUP --> TMPFS
    subgraph notes["Baseline doc note"]
        N1["BASELINE-container.md:61 supervisord table names this program 'pod HTTPS<br/>bridge' — its own module doc-comment (https-proxy.js:3-4) describes it more<br/>generically as solving cross-origin issues for local dev against the<br/>management API. Both describe the same mechanism; the framing differs by doc"]
    end
```
