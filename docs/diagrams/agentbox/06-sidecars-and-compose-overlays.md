---
id: AB-06
title: Compose overlays, sidecar topology and the loopback-publish invariant
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2013, ADR-2003, ADR-2040, ADR-2094, ADR-2118, ADR-2122, ADR-2140, ADR-2149, ADR-2152, ADR-2154, ADR-2207, ADR-2212]
sources:
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/xr-runtime/Dockerfile
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/docker-compose.override.yml
  - ../project/agentbox/docker-compose.browsercontainer.yml
  - ../project/agentbox/docker-compose.gui-tools.yml
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/docker-compose.xr-runtime.yml
  - ../project/agentbox/docker-compose.openmed.yml
  - ../project/agentbox/docker-compose.solid-pods.yml
  - ../project/agentbox/docker-compose.android.yml
  - ../project/agentbox/docker-compose.isaac-sim.yml
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md
  - ../project/agentbox/docs/adr/ADR-2152-render-mermaid-in-browsercontainer.md
  - ../project/agentbox/docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md
  - ../project/agentbox/docker-compose.speech-ming.yml
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/scripts/ci/check-ports-loopback.sh
  - ../project/agentbox/browsercontainer/server.js
  - ../project/agentbox/flake.nix
  - ../project/agentbox/.github/workflows/invariants.yml
  - ../project/agentbox/docker-compose.system-one.yml
  - ../project/agentbox/docker-compose.speech.yml
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/browsercontainer/podkey.pin
  - ../project/agentbox/browsercontainer/scripts/fetch-podkey.sh
  - ../project/agentbox/browsercontainer/Dockerfile
  - ../project/agentbox/browsercontainer/launch-chromium.sh
  - ../project/agentbox/browsercontainer/supervisord.conf
  - ../project/agentbox/browsercontainer/podkey-ctl.js
  - ../project/agentbox/browsercontainer/policies/podkey-only.json
  - ../project/agentbox/config/custody/g5-key-split.json
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/crates/system-one/system-one-proxy/src/server.rs
  - ../project/agentbox/services/agentbox-manifest/src/system_one.rs
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
---

## AB-06.1 Compose overlay topology on visionclaw_network
```mermaid
flowchart TB
    subgraph BASE["docker-compose.yml — AUTO-GENERATED from agentbox.toml via flake.nix, do not edit by hand (docker-compose.yml:1-2)"]
        PG["ruvector-postgres<br/>image pinned by digest docker-compose.yml:12<br/>db ruvector, healthcheck pg_isready docker-compose.yml:21-26"]
        AB["agentbox<br/>image AGENTBOX_IMAGE_REF docker-compose.yml:32<br/>depends_on ruvector-postgres service_healthy docker-compose.yml:36-38<br/>healthcheck curl localhost:9090/ready docker-compose.yml:41-45<br/>LOOM_RAW_BASE_URL, LOOM_MODEL dual-endpoint env vars docker-compose.yml:68-69,<br/>LOOM_MODEL default now empty so the advertised model is discovered, see AB-24"]
    end
    PG -->|"service_healthy gate"| AB
    AB ---|"9096:9096 LAN — the ONE identity-gated door"| LAN(("LAN"))
    AB ---|"127.0.0.1:9090 management-api"| LO(("host loopback"))
    AB ---|"127.0.0.1:9700"| LO
    AB ---|"127.0.0.1:9091 metrics"| LO
    AB ---|"127.0.0.1:8484 solid-pod"| LO
    AB ---|"127.0.0.1:8888 jupyter"| LO
    AB ---|"127.0.0.1:5901 vnc"| LO
    AB ---|"127.0.0.1:8080 code-server"| LO
    subgraph OVR["docker-compose.override.yml — operator layer, auto-loaded when present"]
        OV1["agentbox service overrides docker-compose.override.yml:9<br/>env_file docker-compose.override.yml:13, environment docker-compose.override.yml:21-51<br/>volumes docker-compose.override.yml:70-141, host docker socket bind docker-compose.override.yml:116<br/>group_add 965 docker-compose.override.yml:164-165"]
        OV2["ADR-2118: agentbox-claude-home is now a container-owned external<br/>volume in this overlay, not a host bind — see AB-06.3 docker-compose.override.yml:87-88, :188-190"]
    end
    OVR -.->|"-f base -f override (agentbox.sh:574-581)"| BASE
    OV1 --> OV2
    subgraph SIDE["sidecar overlays — own lifecycle, joined via visionclaw_network"]
        BC["browsercontainer<br/>5903 VNC, 8931 MCP SSE, 9222 to 9223 CDP<br/>host loopback unless AGENTBOX_SIDECAR_BIND is set"]
        GT["gui-tools-service<br/>5905 VNC, 9876 BlenderMCP, 9877 QGIS MCP<br/>host loopback unless AGENTBOX_SIDECAR_BIND is set"]
        VC["voice-console<br/>8443, 8444 Caddy origin"]
        XR["xr-runtime<br/>5904 VNC, host loopback unless AGENTBOX_SIDECAR_BIND is set"]
        OM["openmed<br/>127.0.0.1:9093"]
        AND["android redroid<br/>127.0.0.1:5555 adb — profile android"]
        CF["cloudflared-pod<br/>tunnel, no published port"]
    end
    AB --- NET(("visionclaw_network"))
    BC --- NET
    GT --- NET
    VC --- NET
    XR --- NET
    OM --- NET
    AND --- NET
    CF --- NET
    RET["RETIRED: the connected-node overlay (HP) no longer exists, ADR-2140 retired the HP agentbox;<br/>docs/BASELINE-container.md:285 records that the ADR-2118 migration therefore has no second target"]
    BASE -.-> RET
```

## AB-06.2 ADR-2013 — the loopback-publish invariant and its CI gate
```mermaid
flowchart TD
    CI[".github/workflows/invariants.yml"] --> W["scripts/ci/check-ports-loopback.sh<br/>stable entry point, resolves the gate<br/>relative to itself and FAILS LOUDLY if missing (check-ports-loopback.sh:29-30)"]
    W --> G["scripts/ci/check-ports-loopback.mjs<br/>real YAML reader"]
    G --> WALK["walk the WHOLE tree of every docker-compose*.yml<br/>at the repo root glob, not only services/*/ports (check-ports-loopback.mjs:1087-1092)"]
    WALK --> NORM["normalise every spelling to one tuple —<br/>long-form host_ip/published/target and<br/>short 0.0.0.0:8080:80 judge identically (check-ports-loopback.mjs:31-32)"]
    NORM --> J{"host_ip is 127.0.0.1?<br/>LOOPBACK const :106"}
    J -->|yes| PASS["pass"]
    J -->|no| S{"on the SANCTIONED list?<br/>check-ports-loopback.mjs:100-104 matched by<br/>isSanctioned() on file + host_ip + published + target + protocol (:632)"}
    S -->|yes| PASS
    S -->|no| FAIL["violation — publishes X, not loopback and not sanctioned (:664)"]
    S --> L1["docker-compose.yml 9096 to 9096 host_ip null — :101"]
    S --> L2["docker-compose.voice.yml 8443, 8444 on 0.0.0.0 — :102-103"]
    NORM --> SB["the browsercontainer, gui-tools and xr-runtime sidecars are NO LONGER sanctioned:<br/>they carry no authentication (check-ports-loopback.mjs:84-86). Their publishes take the host address from<br/>AGENTBOX_SIDECAR_BIND, default 127.0.0.1, the only interpolation the gate accepts,<br/>only in SIDECAR_BIND_FILES and only with that loopback default (check-ports-loopback.mjs:92-98).<br/>normalisePort judges such an entry as 127.0.0.1 (check-ports-loopback.mjs:538-541),<br/>and the same prefix in any other file is a violation (check-ports-loopback.mjs:657-660)"]
    SB --> J
    SB -.-> SBF["the three files publish every sidecar port that way:<br/>docker-compose.browsercontainer.yml:54, docker-compose.browsercontainer.yml:56, docker-compose.browsercontainer.yml:60,<br/>docker-compose.gui-tools.yml:46-50, docker-compose.xr-runtime.yml:64"]
    G --> LST["second rule, supervised listeners in flake.nix: a program binding a non-loopback address needs LISTENER_SANCTIONED (check-ports-loopback.mjs:735),<br/>system-one-proxy is sanctioned for 0.0.0.0:8094 when [system_one].network = true, docker networks only, never published (check-ports-loopback.mjs:754)"]
    G --> ALSO["also fails on env interpolation in a port value,<br/>and treats an unsanctioned IPv6 [::] bind as a public door (:36 and :39-40)"]
    G -.-> LIM["DIVERGENCE — the gate's own stated limit (:44): it does NOT resolve<br/>overlay order or --env-file interpolation, so a pass proves the compose files<br/>DECLARE no unsanctioned door, not that no unsanctioned door is OPEN"]
    NORM -.-> HIST["ADR-2013 replaced an awk line-walker after the estate review reproduced a<br/>structural bypass — a public port written as a nested service-flow or JSON-flow<br/>mapping passed a gate that only armed on a line beginning with ports: (:11 of the .mjs)"]
```

## AB-06.3 The gui-tools-exchange volume, and the ADR-2118 claude-home asymmetric mounts
```mermaid
flowchart LR
    V[("named volume gui-tools-exchange, pinned name<br/>agentbox_gui-tools-exchange unless GUI_TOOLS_EXCHANGE_VOLUME is set")] -->|"mounted at /home/devuser/exchange<br/>docker-compose.override.yml:130"| AB["agentbox container"]
    V -->|"also mounted at /home/devuser/gui-tools, the long-standing alias<br/>docker-compose.override.yml:131"| AB
    V -->|"mounted at /home/devuser/exchange<br/>docker-compose.browsercontainer.yml:68"| BC["browsercontainer"]
    V -->|"mounted at /home/devuser/exchange<br/>docker-compose.gui-tools.yml:59"| GT["gui-tools-service"]
    AB -->|"write file to /home/devuser/exchange/x.svg"| V
    V -->|"read as file:///home/devuser/exchange/x.svg"| BC
    BC -->|"screenshot or recording back into the volume"| V
    V -->|"agent reads /home/devuser/exchange/result"| AB
    AB -.-> NOTE["INVARIANT (ADR-2207 D2) — the canonical path /home/devuser/exchange is the SAME<br/>absolute path in agentbox and every sidecar, so a file path needs no translation<br/>(docker-compose.override.yml:126-129); the browser MCP child runs there<br/>(BROWSER_EXCHANGE_DIR, docker-compose.browsercontainer.yml:29)"]
    V -.-> DECL["declared in all three overlays with the same pinned name —<br/>docker-compose.override.yml:192-193, docker-compose.browsercontainer.yml:81-82, docker-compose.gui-tools.yml:72-73"]
    CH[("named volume<br/>agentbox-claude-home, external<br/>docker-compose.override.yml:188-190")] -->|"mounted at /home/devuser/.claude<br/>docker-compose.override.yml:87"| AB
    HB["host ~/.claude<br/>read AND written"] -->|"bound at /var/lib/agentbox/host-claude<br/>docker-compose.override.yml:88 — credentials only"| AB
    AB -.-> INV2["INVARIANT (ADR-2118) — the container no longer reads settings,<br/>hooks or agents from the host's ~/.claude; only claude-cred-sync<br/>reads AND writes across the host bind, converging .credentials.json"]
```

## AB-06.4 browsercontainer — HTTP surface and MCP transport
```mermaid
sequenceDiagram
    autonumber
    participant A as agent in agentbox
    participant S as createBridgeServer<br/>browsercontainer/server.js:308
    participant CH as Chrome on display 2<br/>launch-chromium.sh:123

    alt POST /display/fullscreen, /record/start or /record/stop (server.js:437-447)
        A->>S: request carrying an Origin header
        S-->>A: 403 before the body is read, never any CORS header (server.js:442-445)
    else any other path
        A->>S: OPTIONS any path
        S-->>A: 204 with Access-Control-Allow-Origin star (server.js:449-455)
    end
    alt GET /health (server.js:352)
        A->>S: GET port 8931 /health
        S->>CH: cdp.version() (server.js:330)
        S-->>A: 200 status ok, chrome true, cdp 127.0.0.1:9222, display size and build versions (server.js:331-332)
        S-->>A: or 503 status degraded, chrome false, when Chrome does not answer (server.js:334)
    end
    alt GET /display, POST /display/fullscreen (server.js:354, server.js:364)
        S->>CH: cdp.window(), cdp.fullscreen(on)
    end
    alt GET /record, POST /record/start, POST /record/stop (server.js:373-383)
        S->>S: record.sh status, start or stop, 409 when record.sh exits 3 or 4 (server.js:342)
    end
    alt GET /sse (server.js:385)
        A->>S: GET port 8931 /sse — MCP SSE stream opens
        S-->>A: event stream (registered as the browser-gpu MCP server)
        A->>S: POST /messages (server.js:414) — JSON-RPC tool calls
        S->>CH: drive via CDP
        CH-->>S: result
        S-->>A: 202 then the tool result on the stream, 404 for an unknown session (server.js:417-427)
    end
    Note over S,CH: raw CDP is also reachable — published 9222 on the host mapping to container 9223, where cdp-proxy.js<br/>admits only IP, localhost and CDP_PROXY_ALLOWED_HOSTS Host headers (supervisord.conf:88-92).<br/>The publish binds AGENTBOX_SIDECAR_BIND, default 127.0.0.1, and is no longer on the sanctioned list (docker-compose.browsercontainer.yml:60)
    Note over S: VNC port 5903 for eyes-on debugging, same loopback default (docker-compose.browsercontainer.yml:54)
    Note over A,S: GPU reservation and NVIDIA device request in the deploy block (docker-compose.browsercontainer.yml:39-51)
    Note over A,S: extra_hosts host.docker.internal maps to host-gateway (docker-compose.browsercontainer.yml:64-65)
```

**What it shows.** The sidecar's HTTP router since ADR-2207: Origin-refusing state-changing POSTs, CORS preflight for the rest, a `/health` that answers 503 when Chrome is down, the `/display` and `/record` endpoints, and the MCP SSE transport with its `/messages` POST (`../project/agentbox/browsercontainer/server.js:347-433`). The `/render-mermaid` endpoint and the `mmdc` install are gone from the sidecar; Mermaid renders in the Agentbox image.
**Why it is this way.** The exchange volume stays for agentbox and gui-tools hand-offs only (`../project/agentbox/docker-compose.browsercontainer.yml:67-68`), so the sidecar no longer carries a Mermaid renderer or its Chromium flags.

**Drift:** ADR-2152 (render in browsercontainer) reads `decision_status: superseded` yet still `implementation_status: partial`, `activation_status: staged` (`../project/agentbox/docs/adr/ADR-2152-render-mermaid-in-browsercontainer.md:1-10`), although the endpoint it describes is deleted; ADR-2154 (pinned Pele in the image) is `accepted`, `partial`, `staged` (`../project/agentbox/docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md:1-10`), see AB-01.1.

## AB-06.5 gui-tools-service — the FHS GPU presentation sidecar
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant SH as cmd_gui_tools<br/>agentbox.sh:2245
    participant DC as docker compose<br/>GUI_TOOLS_COMPOSE_ARGS
    participant GT as gui-tools-service
    participant HC as /opt/gui-tools/healthcheck.sh

    OP->>SH: ./agentbox.sh gui-tools up
    SH->>DC: docker compose GUI_TOOLS_COMPOSE_ARGS up -d --build (agentbox.sh:2251)
    Note over DC: Build context is the repository root,<br/>Dockerfile is gui-tools-sidecar/Dockerfile (docker-compose.gui-tools.yml:7-9)
    DC->>GT: start with DISPLAY set to display 2, NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics (docker-compose.gui-tools.yml:18-22)
    Note over GT: __GLX_VENDOR_LIBRARY_NAME=nvidia (docker-compose.gui-tools.yml:25) — the presentation path the Nix wrappers cannot provide, see AB-01.6
    GT->>GT: BlenderMCP binds 0.0.0.0:9876, QGIS MCP binds 0.0.0.0:9877 (docker-compose.gui-tools.yml:26-29)
    loop poll until deadline now plus 120 s, sleep 3 (agentbox.sh:2253-2257)
        SH->>HC: docker exec gui-tools-service bash /opt/gui-tools/healthcheck.sh
        alt healthy
            HC-->>SH: exit 0 — break
        else not yet
            HC-->>SH: non-zero
        end
    end
    alt deadline passed with ready 0
        SH-->>OP: Health check timed out then exit 1 (agentbox.sh:2258-2260)
    else
        SH-->>OP: BlenderMCP gui-tools-service:9876, QGIS gui-tools-service:9877, VNC vnc://localhost:5905 (agentbox.sh:2262-2265)
    end
    Note over SH,DC: sibling subcommands down agentbox.sh:2267-2270, logs agentbox.sh:2271, status agentbox.sh:2272, rebuild agentbox.sh:2283-2287 all reuse GUI_TOOLS_COMPOSE_ARGS
    Note over GT: everything runs under vglrun — interactive GL/Vulkan goes here, NOT through the wrapped Nix bins (BASELINE GPU wrappers limitation)
```

## AB-06.6 Per-sidecar compose argument sets and lifecycle entry points
```mermaid
flowchart LR
    SD["SCRIPT_DIR"] --> A1["COMPOSE_FILE docker-compose.yml — agentbox.sh:575"]
    SD --> A2["OVERRIDE_FILE docker-compose.override.yml — agentbox.sh:574"]
    A1 --> CA{"override file present?<br/>agentbox.sh:577"}
    A2 --> CA
    CA -->|yes| CA1["COMPOSE_ARGS = --project-name agentbox -f base -f override — :578"]
    CA -->|no| CA2["COMPOSE_ARGS = --project-name agentbox -f base — :580"]
    SD --> S1["SIDECAR_FILE browsercontainer — :576<br/>SIDECAR_COMPOSE_ARGS :582<br/>cmd_browsercontainer agentbox.sh:1462"]
    SD --> S2["XR_RUNTIME_FILE :583<br/>XR_RUNTIME_COMPOSE_ARGS :584<br/>cmd_xr_runtime agentbox.sh:1602"]
    SD --> S3["GUI_TOOLS_FILE :585<br/>GUI_TOOLS_COMPOSE_ARGS :586<br/>cmd_gui_tools agentbox.sh:2245"]
    SD --> S4["OPENMED_FILE :589<br/>OPENMED_COMPOSE_ARGS :590<br/>cmd_openmed agentbox.sh:2310"]
    SD --> S5["VOICE_FILE :600 plus voice/unmute-override.yml<br/>VOICE_COMPOSE_ARGS assembled inside _voice_compose_args agentbox.sh:2388-2394<br/>cmd_voice agentbox.sh:2708"]
    SD --> S6["ANDROID_FILE :622<br/>ANDROID_COMPOSE_ARGS adds --profile android :623<br/>cmd_android agentbox.sh:1237"]
    S5 --> VH["VOICE_HOST_ROOT default /mnt/mldata/githubs/AR-AI-Knowledge-Graph — :610<br/>compose bind SOURCES resolve on the HOST docker daemon,<br/>so they must be host paths"]
    S6 --> AG["EXPERIMENTAL and GATED OFF — additionally requires<br/>AGENTBOX_ENABLE_ANDROID=1, agentbox.sh:614"]
    CA1 --> MGMT["MGMT_PORT 9090 — agentbox.sh:628"]
    S5 -.-> VNOTE["voice-console uses its OWN project name agentbox-voice (agentbox.sh:2389),<br/>so it is a separate compose project from every other sidecar"]
    SD --> S7["ADR-2118: migrate-claude-home is NOT a sidecar lifecycle — it seeds<br/>the agentbox-claude-home volume the PRIMARY compose project depends on,<br/>cmd_migrate_claude_home agentbox.sh:1777, dispatched agentbox.sh:2906 — see AB-06.11"]
```

## AB-06.7 Sidecar surface census with published bindings
```mermaid
flowchart TB
    subgraph LANP["LAN-reachable — every one on the ADR-2013 SANCTIONED list (check-ports-loopback.mjs:100-104)"]
        P1["agentbox 9096:9096 — NIP-98 sovereign ingress<br/>docker-compose.yml:48"]
        P2["voice-console 0.0.0.0:8443 and 0.0.0.0:8444 Caddy origin<br/>docker-compose.voice.yml:40-41"]
    end
    subgraph SBIND["host loopback by default, LAN only when the operator sets AGENTBOX_SIDECAR_BIND"]
        P3["browsercontainer 5903 VNC, 8931 MCP SSE, 9222 to 9223 CDP<br/>docker-compose.browsercontainer.yml:54-60"]
        P4["gui-tools-service 5905 VNC, 9876 Blender, 9877 QGIS<br/>docker-compose.gui-tools.yml:46-50"]
        P5["xr-runtime 5904 VNC — docker-compose.xr-runtime.yml:64"]
    end
    subgraph LOOP["host-loopback only"]
        Q1["agentbox 9090 mgmt, 9700, 9091 metrics, 8484 pod,<br/>8888 jupyter, 5901 vnc, 8080 code-server<br/>docker-compose.yml:49-55"]
        Q2["openmed 127.0.0.1:9093 — docker-compose.openmed.yml:28"]
        Q3["android 127.0.0.1:5555 adb — docker-compose.android.yml:40"]
    end
    subgraph NONE["no published port"]
        R1["ruvector-postgres — network-internal only, docker-compose.yml:11-29"]
        R2["cloudflared-pod — outbound tunnel only, docker-compose.solid-pods.yml:25-33"]
    end
    Q3 -.-> AND["android comment: this is an authenticated Google session,<br/>never expose it on 0.0.0.0, prefer docker exec (docker-compose.android.yml:37-38)"]
    Q2 -.-> OM["openmed refuses to serve until the operator sets<br/>OPENMED_LICENSE_ACKNOWLEDGED, _ONNX_RUNTIME_PRESENT and<br/>_GOVERNANCE_ACKNOWLEDGED — all default false (docker-compose.openmed.yml:19-21)"]
    Q1 -.-> CSR["RESOLVED ADR-2040 (implementation_status: partial): code-server<br/>([program:code-server]) runs --auth password, credential minted at boot<br/>into /home/devuser/.local/share/code-server/config.yaml (0600),<br/>flake.nix:3093-3100. jupyter-lab's empty token was DELETED in favour of a<br/>minted JUPYTER_TOKEN. Listener-side CI gate is still open work."]
    R1 -.-> PGN["ADR-015 — mandatory memory sidecar, health-gated;<br/>the memory adapter fails closed with no fallback store"]
```

## AB-06.8 Container hardening posture declared in the base compose
```mermaid
flowchart TD
    AB["agentbox service<br/>docker-compose.yml:31"] --> CD["cap_drop docker-compose.yml:96"]
    AB --> CA2["cap_add docker-compose.yml:98"]
    AB --> TM["tmpfs docker-compose.yml:108, now also /etc/claude-code (root, noexec, 1M) docker-compose.yml:110 for the managed-policy file;<br/>/home/devuser/.codex left the tmpfs list and became the codex-home volume docker-compose.yml:147, docker-compose.yml:201-202"]
    TM --> SEC["NEW custody W1: /run/secrets is its OWN tmpfs, root 0711,<br/>noexec nosuid nodev, beside the uid-1000 /run<br/>docker-compose.yml:112"]
    AB --> SO["security_opt docker-compose.yml:127"]
    AB --> VOL["volumes docker-compose.yml:130"]
    AB --> NET["networks docker-compose.yml:154"]
    SO --> SCC["seccomp and no-new-privileges declarations,<br/>gated by CI invariants under scripts/ci/"]
    VOL --> CXP["codex-packages mounted at /home/devuser/.codex/packages docker-compose.yml:148,<br/>declared agentbox-codex-packages docker-compose.yml:203-204 - ADR-2120, a disk-backed<br/>exec-capable home for Codex daemon packages under the noexec tmpfs"]
    VOL --> NV["named volumes declared — ruvector-pg-data,<br/>agentbox-ruvector-data, agentbox-solid-data docker-compose.yml:177-182"]
    AB --> HC["healthcheck curl -f http://localhost:9090/ready<br/>interval 30s, timeout 10s, retries 5, start_period 60s — docker-compose.yml:40-45"]
    HC --> RDY["so compose readiness rides the SAME /ready contract<br/>the management API publishes — see AB-02"]
    AB --> DEP["depends_on ruvector-postgres condition service_healthy docker-compose.yml:36-38"]
    DEP --> ORD["memory sidecar must pass pg_isready before agentbox starts,<br/>which is what lets the memory adapter boot probe expect a live store — see AB-04.6"]
    AB -.-> GEN["INVARIANT — docker-compose.yml is AUTO-GENERATED from agentbox.toml<br/>via flake.nix (docker-compose.yml:1-2). Editing it by hand is overwritten by nix build .#compose,<br/>which agentbox.sh build and up --build now run first via refresh-compose.sh<br/>(agentbox.sh:762, agentbox.sh:936)"]
    TM -.-> TMS["tmpfs sizes widened: /tmp is now 8G, ~/.npm and ~/.cache are 4G each — docker-compose.yml:109,117-117"]
```

## AB-06.9 xr-runtime — operator CLI lifecycle (closes audit gap 5; sidecar internals in AB-27.13)

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant SH as cmd_xr_runtime<br/>agentbox.sh:1602
    participant DC as docker compose<br/>XR_RUNTIME_COMPOSE_ARGS
    participant XR as xr-runtime container<br/>Monado + Godot — see AB-27.13

    OP->>SH: ./agentbox.sh xr-runtime up
    SH->>DC: docker compose ... up -d --build (agentbox.sh:1610)
    DC->>XR: CMD supervisord -n -c /etc/supervisord.conf (xr-runtime/Dockerfile:117)
    loop poll .State.Health.Status until 720s deadline (agentbox.sh:1612-1620)
        SH->>XR: docker inspect --format .State.Health.Status
        alt healthy
            XR-->>SH: break
        else missing container
            XR-->>SH: exit 1 immediately (agentbox.sh:1618)
        end
    end
    alt not healthy within 12 min
        SH-->>OP: exit 1, check logs (agentbox.sh:1621-1624)
    else healthy
        SH-->>OP: VNC vnc://localhost:5904, Monado simulated stereo HMD, scene XRBoot→GraphScene (agentbox.sh:1626-1629)
    end
    Note over SH,DC: sibling subcommands down/logs/health/status/rebuild all reuse<br/>XR_RUNTIME_COMPOSE_ARGS (agentbox.sh:1631-1660)
```

## AB-06.10 System One and speech overlays: two sidecars added since 2026-09-06
```mermaid
flowchart TB
    CLI["agentbox.sh systemone<br/>cmd_systemone agentbox.sh:2456<br/>SYSTEMONE_FILE agentbox.sh:591"] --> SUB["subcommands up agentbox.sh:2484, down agentbox.sh:2515, logs agentbox.sh:2520,<br/>status agentbox.sh:2523, health agentbox.sh:2526, models agentbox.sh:2552,<br/>eval agentbox.sh:2556, rebuild agentbox.sh:2666"]
    CLI --> VEND["NEW ADR-2139 backend agentbox.sh:2595, proxy agentbox.sh:2659 - no compose: they act on the<br/>vendor-slot proxy inside the agentbox container (loopback for in-container consumers; 0.0.0.0:8094 on the<br/>private docker networks when network = true, agentbox.toml:915-916); from the host<br/>they re-exec through docker exec on the mounted checkout (agentbox.sh:2461-2479)"]
    SUB --> COMPOSE["docker-compose.system-one.yml"]

    COMPOSE --> FACADE["service systemone<br/>build laya-engine + system-one crate context<br/>image agentbox/system-one:latest<br/>docker-compose.system-one.yml:14-26"]
    FACADE --> PUB["published on host loopback only, port 8097<br/>docker-compose.system-one.yml:115"]
    FACADE --> NET["joined to visionclaw_network, so in-estate callers<br/>reach it by service name, not by a published port<br/>docker-compose.system-one.yml:120-121"]
    FACADE --> VOL["named volume systemone-models carries the weights<br/>docker-compose.system-one.yml:238"]

    COMPOSE --> ENGINE["service openjev, profile openjev<br/>docker-compose.system-one.yml:151-152"]
    ENGINE --> NM["network_mode service:systemone - no ports and no networks<br/>of its own, it shares the facade's namespace<br/>docker-compose.system-one.yml:167"]

    SPEECH["docker-compose.speech.yml"] --> ASR["nemotron-asr, host loopback port 8897<br/>docker-compose.speech.yml:10-11"]
    SPEECH --> TTS["pocket-tts, host loopback port 8898<br/>docker-compose.speech.yml:43-44"]

    PUB --> INV["INVARIANT - both new overlays publish on 127.0.0.1 only,<br/>so the ADR-2013 sanctioned-LAN list is unchanged,<br/>docker-compose.system-one.yml:115"]
    VEND --> PNET["the proxy port is never published. CONFIGURED, live once the new agentbox image is activated and the host services are recreated:<br/>containers on visionclaw_network dial agentbox:8094, in-container consumers still dial 127.0.0.1 (system_one.rs:95-103),<br/>and a browser-shaped POST, one carrying Origin or a non-JSON content type, gets 403 before any upstream call (server.rs:151-169)"]
```

**What it shows.** The two compose overlays that joined the estate since this topic was last stamped: the ADR-2094 System One façade with its optional engine profile, and the shared speech pair the voice console depends on.
**Why it is this way.** ADR-2094 keeps typed decisions on the LAN, so the façade is published to host loopback and reached in-estate by service name; the engine shares the façade's network namespace rather than opening a second surface (`../project/agentbox/docker-compose.system-one.yml:167`).

**Invariant:** enabling `features.sovereign_system_one` without this overlay running leaves both consumers failing open to their built-in paths rather than erroring (`../project/agentbox/management-api/lib/system-manifest.js:291`).

## AB-06.11 ADR-2118 — container-owned ~/.claude and the migrate-claude-home lifecycle
```mermaid
flowchart TB
    DEC["docker-compose.override.yml declares<br/>agentbox-claude-home EXTERNAL — compose refuses<br/>to start agentbox until the volume exists docker-compose.override.yml:188-190"] --> PRE["agentbox.sh preflight<br/>checks the override for that declaration and probes<br/>docker volume inspect agentbox-claude-home (agentbox.sh:2028-2033)"]
    PRE -->|"missing"| ERR["prints migrate-claude-home hint, errors+=1,<br/>preflight fails (agentbox.sh:2032-2033)"]
    PRE -->|"present"| OK["up proceeds"]
    OP["operator"] --> MIG["./agentbox.sh migrate-claude-home<br/>cmd_migrate_claude_home agentbox.sh:1777"]
    MIG --> STOP["stop agentbox if running, create the volume<br/>agentbox.sh:1831-1835"]
    STOP --> DEBRIS["tar excluded debris (settings/.claude.json backups,<br/>agentbox-superseded, archive) to ~/.claude-migrate-debris-DATE.tar.gz<br/>agentbox.sh:1837-1858"]
    DEBRIS --> RSYNC["rsync host ~/.claude into the volume, excluding<br/>CLAUDE.md and .credentials.json, then rewrite<br/>plugins/*.json host paths to /home/devuser/.claude<br/>agentbox.sh:1860-1869"]
    RSYNC --> DONE["volume seeded — run preflight then up or rebuild<br/>agentbox.sh:1871-1872"]
    DONE --> DEC
    AB2["agentbox container"] --> SYNC["[program:claude-cred-sync], only when toolchains claude_code<br/>agentbox-manifest cred-sync, interval 2s<br/>flake.nix:3528, :3363-3369"]
    SYNC --> MERGE["per-token merge between /home/devuser/.claude/.credentials.json<br/>and /var/lib/agentbox/host-claude/.credentials.json,<br/>later expiresAt wins — agentbox-manifest cred-sync, flake.nix:3530-3536"]
    MERGE -.-> ABSENT["exits 0 and stays stopped when the host bind is absent<br/>(deployments that don't share auth) — flake.nix:3533-3534, flake.nix:3540-3541"]
    MERGE -.-> INV["INVARIANT — the container reads settings, hooks and agents<br/>ONLY from the volume; the host bind exists solely for<br/>credential convergence, not configuration<br/>docker-compose.override.yml:83-85, system-manifest.js:274-276"]
```

**What it shows.** ADR-2118 replaced the whole-directory host `~/.claude` bind with a container-owned `agentbox-claude-home` volume: an external volume compose cannot start without, a `migrate-claude-home` command that seeds it once from the host, a preflight gate that fails loudly if it is missing, and a `claude-cred-sync` supervisor program that keeps only the rotating OAuth `.credentials.json` converged against the host's own `~/.claude`, which stays bound read-write at `/var/lib/agentbox/host-claude` for that purpose alone.
**Why it is this way.** Rotating OAuth refresh tokens meant a one-sided host bind logged one side out on every refresh; splitting configuration (volume-owned) from credentials (bind-synced, per-token merge on `expiresAt`) keeps both host and container sessions valid without the container trusting host-side settings, hooks or agents (`../project/agentbox/docker-compose.override.yml:78-86`).

**Open:** the credential bind stays writable both ways, so a compromised in-container tool could still modify the host's `.claude`; only the configuration-read path was closed (`../project/agentbox/docker-compose.override.yml:83-85`).

## AB-06.12 browsercontainer ships a pinned Podkey and keeps its profile on a volume (custody W9)
```mermaid
flowchart TB
    PIN["podkey.pin: upstream repo JavaScriptSolidServer/podkey,<br/>CI artefact podkey-extension, commit, run and artefact ids,<br/>sha256, version 0.0.11, artefact expiry, podkey.pin:2-11"]
    subgraph HOST["host, before the image build"]
        F1["agentbox.sh browsercontainer up or rebuild<br/>runs fetch-podkey.sh fetch, agentbox.sh:1454-1455, agentbox.sh:1468<br/>gh or GH_TOKEN only on first fetch"]
    end
    subgraph IMG["browsercontainer image build"]
        D1["COPY vendor/podkey-extension.zip, then fetch-podkey.sh install<br/>browsercontainer/Dockerfile:95-98, no token enters the build"]
        D2["install verifies the copy it will unpack against the pin sha256,<br/>refuses a fork repo or an escaping zip entry<br/>fetch-podkey.sh:95-101, :82, :153-157"]
        D3["managed policy (ADR-2212): Podkey's id allowed, Web Store installs blocked by update URL,<br/>no install sources, external extensions blocked, podkey-only.json:2-14.<br/>No star default block, which would make Chrome refuse the unpacked load.<br/>Installed for Chrome and Chromium browsercontainer/Dockerfile:102-103"]
    end
    subgraph RUN["sidecar runtime"]
        L1["launch-chromium.sh picks the load mode, flag for Chromium,<br/>cdp for branded Chrome, none without the extension<br/>launch-chromium.sh:80-95"]
        L2["program:podkey-loader runs podkey-ctl.js load --watch,<br/>Extensions.loadUnpacked once per Chrome process<br/>supervisord.conf:71-75, podkey-ctl.js:175"]
        L3["profile at /home/devuser/chrome-profile, stale Singleton files removed<br/>launch-chromium.sh:54, :73, :99"]
    end
    VOL[("named volume browsercontainer-profile, fixed name<br/>docker-compose.browsercontainer.yml:71, :85-86<br/>holds Podkey's encrypted vault - never down -v")]
    PIN --> F1 --> D1 --> D2 --> D3 --> L1 --> L2
    L1 --> L3 --> VOL
    VOL -.-> OPEN1["OPEN: the vault is meant to hold K_browser, not the house key,<br/>but k_browser's pubkey is still null, g5-key-split.json:16-19, and the<br/>nip98-proxy roster still admits the house key for it, agentbox.toml:2205"]
    PIN -.-> OPEN2["OPEN (RESOLVED 2026-10-12, agentbox 8fad68285): the earlier pin's CI artefact was due to expire on 2026-12-24.<br/>The pin now names run 38386898958 and expires 2027-01-11 (podkey.pin:11), and podkey CI has workflow_dispatch so the next re-pin needs no upstream commit"]
```

**What it shows.** The browser sidecar now carries an identity of its own. The Podkey extension is the upstream CI artefact named exactly by `podkey.pin`. It is fetched on the host and only verified and unpacked during the image build. A managed policy allows that one extension by id and blocks Web Store and external installs, without a default block that would stop the unpacked load (ADR-2212), and Chrome loads it in whichever mode the browser supports. The Chrome profile, which holds Podkey's encrypted vault, moved from a throwaway `/tmp` path to a named volume so it survives container recreation.
**Why it is this way.** The G-5 key split moves the browser-automation use off the house key onto a dedicated `K_browser` minted inside the sidecar (custody design §11.2, owner rule 2026-10-03). That key has to live somewhere the sidecar owns and that outlasts a rebuild. Pinning a published artefact by hash, rather than building from source or accepting a fork, keeps a credential-holding extension reproducible and auditable (`../project/agentbox/browsercontainer/Dockerfile:90-98`).

**Invariant:** the browsercontainer image is never built with an unverified Podkey: the zip copied into the build is checked against the pinned sha256 before a single entry is extracted, and a mismatch refuses the build (`../project/agentbox/browsercontainer/scripts/fetch-podkey.sh:150-151`, `../project/agentbox/browsercontainer/scripts/fetch-podkey.sh:95-101`).

**Open:** `K_browser` is not admitted anywhere yet. Its pubkey is null in the split register (`../project/agentbox/config/custody/g5-key-split.json:16-20`), and the LAN door's roster still names the house key as the browsercontainer vault's identity (`../project/agentbox/agentbox.toml:2205`). Until the owner mints it and the dual-admit step lands, the sidecar's flows still authenticate as the house key.

## AB-06.13 The host Docker socket and the /run/secrets mount: what compose can and cannot narrow (custody W0, W1)
```mermaid
flowchart TB
    subgraph COMPOSE["compose, both modes"]
        S1["host socket bound rw into agentbox<br/>docker-compose.override.yml:116"]
        S2["group_add 965 reaches PID 1 only, not programs that drop to devuser<br/>docker-compose.override.yml:159-165"]
        M1["/run tmpfs owned uid 1000, so devuser can rename its entries<br/>docker-compose.yml:111"]
        M2["/run/secrets its own tmpfs, uid 0 mode 711, a mount point<br/>devuser cannot rename, docker-compose.yml:112"]
    end
    subgraph BOOT["entrypoint, decided by role_isolation"]
        E0["flag off: chmod o+rw on the socket every boot<br/>entrypoint-unified.sh:609-611"]
        E1["flag on: socket left as found, devuser reaches the GET-only<br/>/run/docker-ro.sock, degraded docker-socket recorded if still reachable<br/>entrypoint-unified.sh:597-604, entrypoint-unified.sh:618-629"]
        E2["flag off: /run/secrets chowned back to devuser 0700<br/>entrypoint-unified.sh:359-362"]
    end
    S1 --> E0
    S1 --> E1
    M2 --> E2
    S1 -.-> R2["OPEN R2: the socket is the HOST inode, earlier boots left it o+rw,<br/>and only a host-side chmod 0660 narrows it,<br/>docker-compose.override.yml:114-116"]
```

**What it shows.** The two compose-level facts the custody change rests on. The host Docker socket is a bind of the host's own inode, so nothing inside the container can make it narrower than the host leaves it. `/run/secrets` became a separate root-owned tmpfs under the devuser-owned `/run`, because a mount point cannot be renamed by the owner of its parent. Both ship in every image. With `[security].role_isolation` off the entrypoint widens the socket and chowns `/run/secrets` to devuser exactly as before.
**Why it is this way.** The Docker daemon is root on the host, so while devuser can drive it every in-container boundary is void (custody design §0, bypass 1). The `/run` tmpfs is uid 1000 by construction (`../project/agentbox/docker-compose.yml:111`), so the secrets directory had to leave it.

**Invariant:** `/run/secrets` is its own mount, owned by root with mode 0711, in both modes, so devuser, as owner of `/run`, cannot rename it or replace it after boot (`../project/agentbox/docker-compose.yml:112`). The flag-off boot then hands the directory to devuser (`../project/agentbox/config/entrypoint-unified.sh:359-362`), so the root ownership only protects anything under `role_isolation`.

The host socket inode stays world-accessible until the host chmods it. The entrypoint under `role_isolation` stops widening it but never narrows it, and the compose comment names this as risk R2, owner question Q2 (`../project/agentbox/docker-compose.override.yml:114-116`); the open question is registered once, in AB-36.

## AB-06.14 Opt-in GPU sidecars added since the last stamp: Isaac Sim, Ming speech candidate, DCC gateway port
```mermaid
flowchart TB
    CLI["agentbox.sh isaac-sim<br/>ISAAC_SIM_FILE agentbox.sh:587, cmd_isaac_sim agentbox.sh:2173"] --> ISAAC["docker-compose.isaac-sim.yml<br/>service isaac-sim, image agentbox-isaac-sim:local docker-compose.isaac-sim.yml:19"]
    ISAAC --> GPU["GPU pinned by UUID, no all default:<br/>NVIDIA_VISIBLE_DEVICES uses ISAAC_SIM_GPU_UUID with :? so compose fails closed<br/>docker-compose.isaac-sim.yml:30"]
    ISAAC --> NOPORT["no ports: key, expose 9890 on visionclaw_network only<br/>docker-compose.isaac-sim.yml:54-56"]
    ISAAC --> EXV["gui-tools-exchange mounted at /exchange, runner confined to the isaac sub-directory;<br/>caches on the separate isaac-sim-cache volume<br/>docker-compose.isaac-sim.yml:61-63"]
    ISAAC --> TOK["ISAAC_RUNNER_TOKEN, no default and no :? guard: unset starts the runner closed,<br/>refusing every request but /healthz, docker-compose.isaac-sim.yml:42-43, docker-compose.isaac-sim.yml:7-10"]
    ISAAC --> SECOPT["security_opt no-new-privileges<br/>docker-compose.isaac-sim.yml:70"]
    GT["gui-tools-service"] --> DCC["expose 9881, DCC gateway MCP and REST, container network only, no new host port<br/>docker-compose.gui-tools.yml:53-55<br/>DCC_MCP_ENABLED defaults to 1 :30"]
    MING["docker-compose.speech-ming.yml<br/>own compose project agentbox-speech-ming docker-compose.speech-ming.yml:13, profiles tts-05b, tts-tiny, asr"] --> MP["host loopback only: ming-tts port 8911 docker-compose.speech-ming.yml:47,<br/>ming-asr port 8912 docker-compose.speech-ming.yml:81"]
    MING --> NOTE["experimental, additive: does not replace nemotron-asr or pocket-tts docker-compose.speech-ming.yml:1-3"]
    NOPORT --> INV["INVARIANT: the ADR-2013 gate sees no new LAN door from these overlays,<br/>isaac-sim publishes nothing and ming publishes on 127.0.0.1<br/>docker-compose.isaac-sim.yml:54, docker-compose.speech-ming.yml:47"]
    MP --> INV
    ISAAC -.-> OPENA["OPEN: ADR-2149 is proposed, implementation partial, activation staged,<br/>while agentbox.sh already dispatches the isaac-sim subcommand agentbox.sh:2896"]
```

**What it shows.** Two overlays and one port that joined the compose estate since the last stamp. Isaac Sim runs headless behind a typed job API with no published port and a mandatory GPU UUID. The Ming speech candidate is a separate compose project on host loopback. The Blender and QGIS sidecar gained an internal-only DCC gateway port.
**Why it is this way.** The Isaac sidecar shares the existing exchange volume so agentbox and Blender see results without recreating either, and fails closed on GPU choice so an agent cannot grab every device (`../project/agentbox/docker-compose.isaac-sim.yml:3`).

**Open:** ADR-2149 is `decision_status: proposed` and `activation_status: staged`, yet the CLI dispatches `isaac-sim` unconditionally (`../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:1-10`, `../project/agentbox/agentbox.sh:2896`).
