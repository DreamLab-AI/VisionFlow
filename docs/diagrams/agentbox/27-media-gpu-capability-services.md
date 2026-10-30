---
id: AB-27
title: Media and GPU capability services — manifest-gated dispatch
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2006, ADR-2020, ADR-2040, ADR-2057, ADR-2142, ADR-2149, ADR-2151, ADR-2153]
sources:
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/lib/gpu-wrap.nix
  - ../project/agentbox/lib/gpu-backend.nix
  - ../project/agentbox/lib/3dgs-stack.nix
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/management-api/routes/comfyui.js
  - ../project/agentbox/management-api/utils/comfyui-manager.js
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/services/agentbox-ops/src/bin/comfyui-generate.rs
  - ../project/agentbox/services/agentbox-mcp/src/main.rs
  - ../project/agentbox/services/agentbox-mcp/src/imagemagick/mod.rs
  - ../project/agentbox/services/agentbox-mcp/src/imagemagick/exec.rs
  - ../project/agentbox/mcp/mcp.json
  - ../project/agentbox/skills/blender/tools/blender-mcp-proxy.js
  - ../project/agentbox/skills/comfyui/mcp-server/server.js
  - ../project/agentbox/skills/lichtfeld-studio/SKILL.md
  - ../project/agentbox/gui-tools-sidecar/supervisord.conf
  - ../project/agentbox/gui-tools-sidecar/launch-blender.sh
  - ../project/agentbox/gui-tools-sidecar/launch-qgis.sh
  - ../project/agentbox/docker-compose.gui-tools.yml
  - ../project/agentbox/docker-compose.openmed.yml
  - ../project/agentbox/openmed-sidecar/prereq-check.sh
  - ../project/agentbox/openmed-sidecar/entrypoint.sh
  - ../project/agentbox/voice/README.md
  - ../project/agentbox/voice/console/Caddyfile
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/scripts/agentbox-config-validate.js
  - ../project/agentbox/gui-tools-sidecar/qgis-mcp-headless.py
  - ../project/agentbox/xr-runtime/launch-godot.sh
  - ../project/agentbox/xr-runtime/launch-monado.sh
  - ../project/agentbox/xr-runtime/supervisord.conf
  - ../project/agentbox/xr-runtime/build-gdext.sh
  - ../project/agentbox/xr-runtime/healthcheck.sh
  - ../project/agentbox/xr-runtime/README.md
  - ../project/agentbox/voice/unmute-override.yml
  - ../project/agentbox/docker-compose.isaac-sim.yml
  - ../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
  - ../project/agentbox/docs/adr/ADR-2142-blender-runs-sidecar-first-the-in-container-binary-is-optional.md
  - ../project/agentbox/docker-compose.voice.yml
  - ../project/agentbox/docker-compose.speech.yml
  - ../project/agentbox/scripts/qgis_mcp_standalone.py
verified_commit: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc
---

## AB-27.1 ComfyUI — builtin loopback gate vs external sidecar integration

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Claude / agent
    participant Skill as comfyui MCP skill<br/>agentbox/skills/comfyui/mcp-server/server.js:11
    participant Builtin as comfyui-builtin<br/>agentbox/flake.nix:2874
    participant MgmtAPI as ComfyUIManager<br/>agentbox/management-api/utils/comfyui-manager.js:25
    participant Gen as comfyui-generate<br/>agentbox/services/agentbox-ops/src/bin/comfyui-generate.rs:15
    participant Ext as comfyui:8188<br/>visionclaw_network sidecar

    Agent->>Skill: generate_image(prompt) via stdio MCP<br/>agentbox/mcp/mcp.json:137
    Note over Skill: class default localhost:8188 (skills/comfyui/mcp-server/server.js:11) but the entry point<br/>defaults COMFYUI_URL to http://comfyui:8188, the sidecar (skills/comfyui/mcp-server/server.js:176), and mcp.json says the image sets it (mcp/mcp.json:148)

    alt skills.media.comfyui_builtin = true (agentbox.toml:372)
        Skill->>Builtin: POST /prompt (loopback 127.0.0.1:8188)
        Note over Builtin: [program:comfyui-builtin]<br/>apply_class rebuild (system-manifest.js:60)
        Builtin-->>Skill: prompt_id, history poll
    else comfyui_builtin = false (default) — integrations.comfyui_external.enabled = true (agentbox.toml:513)
        Skill->>Ext: COMFYUI_URL=http://comfyui:8188 POST /prompt
        Note over Ext: agentbox.toml:515 url, not baked into the image
    else both gates off (documented default)
        Note over Skill: ADR-2020 byte-identical-when-off — no process on port 8188<br/>Skill--xAgent: connection refused
    end

    par management-api GPU-metered route
        Agent->>MgmtAPI: POST /v1/comfyui/workflow (X-API-Key, payment-gate)<br/>agentbox/management-api/routes/comfyui.js:23
        MgmtAPI->>MgmtAPI: _isBackendAvailable() GET {comfyuiUrl}/system_stats<br/>comfyui-manager.js:45
        alt backend reachable
            MgmtAPI->>Ext: submit workflow, track queue
        else unreachable (default state, comfyui_external off)
            MgmtAPI--xAgent: 503 backend unavailable<br/>comfyui-manager.js:326-329
        end
        Note over MgmtAPI: route is registered unconditionally<br/>management-api/server.js:355 — the 503 is the off-state, not a manifest gate
    and one-shot CLI generation
        Agent->>Gen: comfyui-generate <prompt> [out.png]
        Gen->>Ext: POST {base}/prompt FLUX2 workflow (DEFAULT_URL comfyui-generate.rs:15,84)
        loop poll GET /history/{prompt_id} every 2s<br/>comfyui-generate.rs:136
            Ext-->>Gen: status pending/success/error
        end
        Gen->>Ext: GET /view?filename=... on success<br/>comfyui-generate.rs:122-127
    end
```

## AB-27.2 Blender MCP dispatch — proxy to the gui-tools GPU sidecar

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Claude / agent
    participant UVX as uvx blender-mcp<br/>agentbox/mcp/mcp.json:152
    participant Proxy as blender-mcp-proxy.js<br/>agentbox/skills/blender/tools/blender-mcp-proxy.js:30
    participant Sidecar as gui-tools-service:9876<br/>agentbox/docker-compose.gui-tools.yml
    participant Launch as launch-blender.sh<br/>agentbox/gui-tools-sidecar/launch-blender.sh:18
    participant BlenderApp as Blender + BlenderMCP addon<br/>agentbox/gui-tools-sidecar/supervisord.conf:35

    Agent->>UVX: MCP tool call (stdio, BLENDER_HOST=localhost BLENDER_PORT=9876)
    alt skills.spatial_and_3d.blender = true (agentbox.toml:630)
        UVX->>Proxy: TCP connect 127.0.0.1:9876
        Note over Proxy: [program:blender-mcp] agentbox/flake.nix:2381<br/>apply_class rebuild (system-manifest.js:137)
        Proxy->>Sidecar: bridge -> GUI_CONTAINER_HOST:GUI_BLENDER_PORT (default gui-tools-service:9876)
        Sidecar->>Launch: supervisord [program:blender] startsecs=15
        alt vglrun present (launch-blender.sh:18)
            Launch->>BlenderApp: exec vglrun -d egl blender --factory-startup ...<br/>launch-blender.sh:19
            Note right of BlenderApp: GPU EGL context — Cycles CUDA/OptiX + real GL viewport
        else vglrun absent (fallback, launch-blender.sh:21-22)
            Launch->>BlenderApp: exec blender ... (software GL)
            Note right of BlenderApp: DEGRADED — CPU render, no GPU viewport
        end
        BlenderApp-->>UVX: BlenderMCP socket response
        UVX-->>Agent: tool result
    else skills.spatial_and_3d.blender = false
        Note over Proxy: [program:blender-mcp] omitted (flake.nix:2723 lib.optionalString)<br/>ADR-2020: package + supervisor block both absent, no runtime trace
    end
    Note over BlenderApp: DIVERGENCE — flake.nix:1470 wraps pkgs.qgis and flake.nix:1476-1480 wraps pkgs.blender in the main image,<br/>the latter only when blender and blender_local are both true (blender_local absent = true).<br/>This manifest sets blender_local = false (agentbox.toml:635), so no in-container Blender is baked and<br/>batch work goes to the sidecar through blender-job.js (flake.nix:2376-2377, ADR-2142).
```

## AB-27.3 QGIS MCP dispatch — headless offscreen server in the same sidecar

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Claude / agent
    participant PyClient as qgis MCP client<br/>agentbox/mcp/mcp.json:80
    participant Standalone as qgis_mcp_standalone.py<br/>agentbox/flake.nix:2354
    participant Sidecar as gui-tools-service:9877<br/>agentbox/docker-compose.gui-tools.yml
    participant QLaunch as launch-qgis.sh<br/>agentbox/gui-tools-sidecar/launch-qgis.sh:17
    participant QgisApp as QgsApplication + QgisMCPServer<br/>agentbox/gui-tools-sidecar/qgis-mcp-headless.py

    Agent->>PyClient: MCP tool call (stdio, QGIS_HOST=localhost QGIS_PORT=9877)
    alt skills.spatial_and_3d.qgis = true (agentbox.toml:629)
        PyClient->>Standalone: TCP connect 127.0.0.1:9877
        Note over Standalone: [program:qgis-mcp] flake.nix:2354<br/>thin TCP proxy (localhost:9877 -> gui-tools-service:9877)<br/>apply_class rebuild (system-manifest.js:134)
        Standalone->>Sidecar: proxy -> GUI_CONTAINER_HOST:9877
        Sidecar->>QLaunch: supervisord [program:qgis] startsecs=15
        QLaunch->>QgisApp: python3 qgis-mcp-headless.py<br/>QT_QPA_PLATFORM=offscreen (launch-qgis.sh:15)
        Note right of QgisApp: iface is None (no desktop, no vglrun)<br/>layer/feature/processing-algorithm/layout-export tools work<br/>canvas/render tools are unavailable
        QgisApp-->>PyClient: JSON-RPC response
        PyClient-->>Agent: tool result
    else skills.spatial_and_3d.qgis = false
        Note over Standalone: [program:qgis-mcp] omitted (flake.nix:2722 lib.optionalString)<br/>ADR-2020: package + supervisor block both absent, no runtime trace
    end
    Note over QgisApp: DIVERGENCE — QGIS does NOT share Blender's vglrun GPU path (see AB-27.12):<br/>the desktop QGIS full app never initialises headlessly in this sidecar (no window<br/>manager)
```

## AB-27.4 ImageMagick MCP dispatch — the 7 rmcp tools

```mermaid
sequenceDiagram
    autonumber
    participant Agent as Claude / agent
    participant Registry as .mcp.json stdio spawn<br/>agentbox/mcp/mcp.json:67-76
    participant Bin as agentbox-mcp imagemagick<br/>agentbox/services/agentbox-mcp/src/main.rs:28
    participant Router as ImageMagickServer tool_router<br/>agentbox/services/agentbox-mcp/src/imagemagick/mod.rs:47
    participant Exec as run_imagemagick / run_identify<br/>agentbox/services/agentbox-mcp/src/imagemagick/exec.rs:41,112
    participant CLI as magick/convert binary<br/>exec.rs:60

    Agent->>Registry: spawn agentbox-mcp imagemagick (stdio, IMAGEMAGICK_TIMEOUT=300)
    Registry->>Bin: rmcp::transport::stdio() serve() main.rs:75-76
    Note over Router: 7 tools (mod.rs:34-41): create_image, convert_image,<br/>resize_image, crop_image, composite_images, identify_image, batch_process
    Agent->>Router: call_tool(name, params)
    Router->>Exec: run_imagemagick(args, timeout) or run_identify(path, verbose)
    Exec->>CLI: Command::new(magick|convert).args(...) exec.rs:14-21,60
    CLI-->>Exec: stdout/stderr, exit status
    Exec-->>Router: json result (success/error shape preserved from Python original)
    Router-->>Agent: CallToolResult

    Note over Bin: [program:imagemagick-mcp] flake.nix:2711, gate skills.media.imagemagick<br/>(agentbox.toml:371, apply_class rebuild, system-manifest.js:143)<br/>runs the SAME binary as a redundant always-on supervisord instance<br/>whose stdio goes to /var/log/imagemagick-mcp.log, not a live MCP client
```

## AB-27.5 JupyterLab dispatch

```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator / agent browser
    participant Sup as supervisord
    participant Lab as jupyter-lab<br/>agentbox/flake.nix:2453
    participant Entry as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1058

    alt skills.data_science.jupyter = true (agentbox.toml:640)
        Entry->>Entry: mint JUPYTER_TOKEN into a 0600 devuser file, export before supervisord starts<br/>(entrypoint-unified.sh:1058-1083)
        Sup->>Lab: jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser (flake.nix:2469)
        Note right of Lab: DOC-DRIFT (fixed by ADR-2040): the old --IdentityProvider.token= flag is REMOVED —<br/>an explicit empty token short-circuited jupyter_server's own env default. Dropping it lets<br/>IdentityProvider fall through to JUPYTER_TOKEN, inherited from Entry, never interpolated<br/>into the generated supervisor text (flake.nix:2455-2468)
        Op->>Lab: GET /lab on host port 8888 (token required)
        Lab-->>Op: notebook UI, kernel execution
    else skills.data_science.jupyter = false
        Note over Sup: [program:jupyter-lab] omitted (flake.nix:2724 lib.optionalString)<br/>apply_class rebuild (system-manifest.js:53), no runtime trace
    end
```

## AB-27.6 code-server dispatch

```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator browser
    participant Sup as supervisord
    participant CS as code-server<br/>agentbox/flake.nix:2850
    participant Entry as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh:1013

    alt toolchains.code_server = true (agentbox.toml:2250)
        Entry->>Entry: mint/adopt CODE_SERVER_PASSWORD, write --config file 0600 devuser-owned<br/>(entrypoint-unified.sh:1013-1044)
        Sup->>CS: code-server --bind-addr 0.0.0.0:8080 --auth password<br/>--config .../config.yaml (flake.nix:2858)
        Note right of CS: DOC-DRIFT (fixed by ADR-2040): binds 0.0.0.0:8080 (still not 127.0.0.1 — the<br/>docker-compose loopback publish only constrains host->container, not bridge->container<br/>on visionclaw_network) but --auth none is GONE, replaced by a minted password
        Op->>CS: GET host port 8080
        CS-->>Op: VS Code web UI over /home/devuser/workspace, password prompt
    else toolchains.code_server = false
        Note over Sup: [program:code-server] omitted (flake.nix:2848 lib.optionalString)<br/>apply_class rebuild (system-manifest.js:51), no runtime trace
    end
```

## AB-27.7 Voice / Unmute operator console dispatch — own-lifecycle sidecar

```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator
    participant CLI as agentbox.sh voice<br/>agentbox/agentbox.sh:2702
    participant Certs as _voice_ensure_certs<br/>agentbox/agentbox.sh:2353
    participant Speech as docker compose (agentbox-speech project)<br/>docker-compose.speech.yml, agentbox.sh:594
    participant Compose as docker compose (agentbox-voice project)<br/>agentbox/agentbox.sh:2382
    participant Caddy as voice-console Caddy port 8444/port 8443<br/>agentbox/voice/console
    participant Bridge as tab0-bridge port 8971<br/>agentbox/agentbox.toml
    participant Unmute as Kyutai Unmute backend<br/>voice-stack/unmute (external, not vendored)

    Op->>CLI: ./agentbox.sh voice up
    Note over CLI: [voice] agentbox.toml:2209 enabled=true (flipped 2026-10-02 after the<br/>console ran 5 days while the gate said false) — comment still says "sidecar state,<br/>its own lifecycle, not agentbox up" — system-manifest.js:182 catalogues<br/>gate 'voice' apply_class 'live' (id voice-console)
    CLI->>Certs: gen self-signed TLS if absent (agentbox.sh:2355-2376)
    CLI->>Speech: up --wait (agentbox.sh:2723): shared Nemotron ASR + Pocket TTS<br/>(system-manifest.js:183 "follow the core AgentBox lifecycle")
    Note over Speech: NOT gated by BRIDGE_TOKEN — brought up before the check below
    CLI->>CLI: BRIDGE_TOKEN="$(_read_bridge_token)" (agentbox.sh:2734)
    alt BRIDGE_TOKEN unset/empty
        CLI--xOp: refuse to start voice stack (agentbox.sh:2735-2742)
    else BRIDGE_TOKEN set
        CLI->>Compose: up docker-compose.voice.yml<br/>(+ voice/compose.web.yml + unmute-override.yml when VOICE_UNMUTE_DIR clone present, agentbox.sh:2382-2388, voice up at 2448)
        Compose->>Caddy: start console (cockpit port 8444, debug port 8443)
        Compose->>Unmute: start frontend/backend (if VOICE_UNMUTE_DIR clone present)
        Op->>Caddy: /embed voice strip, /feed+/bridge -> Bridge, /approvals/* (voice/console/Caddyfile:55, voice/console/Caddyfile:92, voice/console/Caddyfile:97) - the /aoe/* route is GONE (ADR-2153, voice/README.md:1-4)
        Caddy->>Bridge: forward Authorization header
        Caddy->>Unmute: forward /api/* (STT/TTS, /v1/realtime)
        Note over Unmute: LLM = tab0-bridge (OpenAI-compatible /v1/chat/completions),<br/>KYUTAI_LLM_API_KEY=${BRIDGE_TOKEN} (agentbox.sh:2749)
    end
    Note over Caddy: if the Unmute clone is absent, console still starts —<br/>/embed and /api 502 until the speech stack is present (voice/README.md)
```

**Tension (ADR-2151 vs manifest):** ADR-2151 records edge voice as `decision_status: proposed`, `implementation_status: none`, `activation_status: inactive` (`../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`), while the live manifest sets `[voicelink] enabled = true` and `record_events = true` (`../project/agentbox/agentbox.toml:2226`, `../project/agentbox/agentbox.toml:2233`); it is a separate path from the cockpit drawn here and is mapped in AB-12.

## AB-27.8 openmed clinical-PHI sidecar — triple fail-closed prerequisite gate

```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator
    participant CLI as agentbox.sh openmed up<br/>agentbox/agentbox.sh:2304
    participant Compose as docker compose openmed<br/>agentbox/docker-compose.openmed.yml
    participant Entry as entrypoint.sh<br/>agentbox/openmed-sidecar/entrypoint.sh:7
    participant Prereq as prereq-check.sh<br/>agentbox/openmed-sidecar/prereq-check.sh:17
    participant Server as helix pipeline server<br/>OPENMED_SERVER_ENTRY (operator-provisioned)

    Op->>CLI: ./agentbox.sh openmed up
    Note over CLI: [privacy_filter.openmed] agentbox.toml:1817 enabled=false (default)<br/>no system-manifest.js catalogue id — only the generic 'privacy-filter'<br/>entry (:241-243) mentions openmed as "compose-managed, separately fail-closed gated"
    CLI->>Compose: docker compose up -d --build (agentbox.sh:2313)
    Compose->>Entry: container starts, OPENMED_* env from [privacy_filter.openmed]
    Entry->>Prereq: bash prereq-check.sh
    alt all three of license_acknowledged, governance_acknowledged, onnx_runtime_present = true<br/>AND model_artifact exists AND sha256 matches artifact_lock_sha256
        Prereq-->>Entry: "prerequisites satisfied" (prereq-check.sh:40)
        Entry->>Server: exec node ${OPENMED_SERVER_ENTRY:-/opt/openmed/server/index.js}
        Server-->>Op: clinical redaction routes (per [privacy_filter.openmed.policy] agentbox.toml:1830:<br/>pods=strict, memory=strict, inbound=soft, outbound=soft)
    else any prerequisite false (documented default — all three false)
        Prereq--xEntry: fail() exit 1, e.g. "license_acknowledged is false" (prereq-check.sh:19-20)
        Note over Entry: SERVER path is unreachable with default gates (entrypoint.sh comment, :12-13)
    else server file absent even after prereqs pass
        Entry--xServer: "no server at ${SERVER}" exit 1 (entrypoint.sh:18-19)
        Note over Server: helix-openmed / helix-wasm / openmedkit-web is not vendored (licence-gated)
    end
```

## AB-27.9 3D Gaussian Splatting / LichtFeld — two unrelated paths behind one skill name

```mermaid
sequenceDiagram
    autonumber
    participant Op as Operator
    participant Toml as agentbox.toml:636<br/>skills.spatial_and_3d.gaussian_splatting
    participant Flake as flake.nix gauss3dPackages<br/>agentbox/flake.nix:771
    participant Nix3dgs as lib/3dgs-stack.nix makeLichtfeld<br/>agentbox/lib/3dgs-stack.nix:79
    participant Bridge as lichtfeld_mcp_bridge.py<br/>agentbox/skills/lichtfeld-studio/SKILL.md:50
    participant LFS as LichtFeld-Studio binary port 45677<br/>workspace/gaussians (external, ungated)

    alt gaussian_splatting = true (requires gpu.backend = local-cuda, E006)
        Flake->>Nix3dgs: makeGaussianSplattingPackages (colmap, metis, lichtfeld) (flake.nix:772)<br/>wrapGpuAll gauss3dPackages (flake.nix:1486)
        Nix3dgs--xFlake: DIVERGENCE: throw at eval — lichtfeldRev is the placeholder<br/>"0000...0" (3dgs-stack.nix:85-97): "gaussian_splatting cannot be enabled"
        Note over Toml: E006 validator (scripts/agentbox-config-validate.js:222-227)<br/>additionally requires gpu.backend="local-cuda"
    else gaussian_splatting = false (documented default, agentbox.toml:636)
        Note over Flake: gauss3dPackages = [] (flake.nix:771 lib.optionals)<br/>apply_class rebuild (system-manifest.js:416, id gaussian-splatting)
    end
    Note over Bridge,LFS: DIVERGENCE — the actually-used lichtfeld-studio skill is<br/>completely independent of this gate: Claude spawns lichtfeld_mcp_bridge.py<br/>(stdio) which HTTP-POSTs JSON-RPC to a manually built<br/>/home/devuser/workspace/gaussians/LichtFeld-Studio/build/LichtFeld-Studio<br/>at localhost port 45677 — never baked by lib/3dgs-stack.nix, never gated by this toml key
    Op->>Bridge: tools/lfs-mcp.sh call training.get_state
    Bridge->>LFS: POST to localhost port 45677, path /mcp (JSON-RPC 2.0)
    LFS-->>Bridge: result
    Bridge-->>Op: tool response
```

## AB-27.10 Supervision tree — media/GPU supervised programs, ports on the edges

```mermaid
flowchart TB
    SUP["supervisord (PID 1, root)<br/>agentbox/flake.nix"]
    SUP -->|priority 210| IM["imagemagick-mcp<br/>flake.nix:2711"]
    SUP -->|priority 220| CB["comfyui-builtin port 8188 loopback<br/>flake.nix:2874"]
    SUP -->|priority 230| QM["qgis-mcp port 9877 -> proxy<br/>flake.nix:2354"]
    SUP -->|priority 231| BM["blender-mcp port 9876 -> proxy<br/>flake.nix:2381"]
    SUP -->|priority 232| JL["jupyter-lab port 8888<br/>flake.nix:2454"]
    SUP -->|priority 50| CS["code-server 0.0.0.0:8080<br/>flake.nix:2850"]
    SUP -->|priority 250, gated| PC["podcast-cron (supercronic)<br/>flake.nix:2984 — RESOLVED ADR-2057 gap 1: now wrapped in<br/>lib.optionalString podcastIngestEnabled (flake.nix:296, flake.nix:2977),<br/>gate skills.podcast_ingest.enabled default true — off removes the<br/>program but the binary/supercronic stay in the closure (shared<br/>with podcast-{knowledge,bulk}-ingest and forum-backup-cron)"]

    QM -->|TCP 9877| GTS["gui-tools-service<br/>docker-compose.gui-tools.yml"]
    BM -->|TCP 9876| GTS

    subgraph sidecar["gui-tools-service (own supervisord, compose sidecar)"]
        GSUP["supervisord"]
        GSUP -->|priority 30| GB["blender + BlenderMCP port 9876<br/>gui-tools-sidecar/supervisord.conf:35"]
        GSUP -->|priority 40| GQ["qgis + QgisMCPServer port 9877<br/>gui-tools-sidecar/supervisord.conf:47"]
    end
    GTS --- GSUP

    subgraph voicesc["voice-console sidecar (own compose project agentbox-voice)"]
        CADDY["Caddy port 8444 / port 8443<br/>voice/console"]
        UNMUTE["Unmute frontend/backend<br/>voice-stack/unmute (external)"]
    end

    subgraph openmedsc["openmed sidecar (own compose project)"]
        OM["openmed port 9093 loopback<br/>docker-compose.openmed.yml"]
    end
```

## AB-27.11 Manifest gate to apply_class — media/GPU capability catalogue

```mermaid
classDiagram
    direction LR
    class ApplyClass {
        <<enumeration>>
        live
        boot
        rebuild
    }
    class CatalogueEntry {
        +String id
        +String gate
        +String service
        +ApplyClass apply_class
    }
    CatalogueEntry --> ApplyClass

    class comfyui {
        id = comfyui
        gate = skills.media.comfyui_builtin
        service = comfyui-builtin
        apply_class = rebuild
    }
    class imagemagick_mcp {
        id = imagemagick-mcp
        gate = skills.media.imagemagick
        service = imagemagick-mcp
        apply_class = rebuild
    }
    class ffmpeg {
        id = ffmpeg
        gate = skills.media.ffmpeg
        service = null
        apply_class = rebuild
    }
    class qgis_mcp {
        id = qgis-mcp
        gate = skills.spatial_and_3d.qgis
        service = qgis-mcp
        apply_class = rebuild
    }
    class blender_mcp {
        id = blender-mcp
        gate = skills.spatial_and_3d.blender
        service = blender-mcp
        apply_class = rebuild
    }
    class gaussian_splatting {
        id = gaussian-splatting
        gate = skills.spatial_and_3d.gaussian_splatting
        service = null
        apply_class = rebuild
    }
    class jupyter {
        id = jupyter
        gate = skills.data_science.jupyter
        service = jupyter-lab
        apply_class = rebuild
    }
    class code_server {
        id = code-server
        gate = toolchains.code_server
        service = code-server
        apply_class = rebuild
    }
    class desktop {
        id = desktop
        gate = desktop.enabled
        service = xvnc
        apply_class = rebuild
    }
    class gui_tools_sidecar {
        id = gui-tools-sidecar
        gate = null
        service = gui-tools-service
        apply_class = live
    }
    class voice_console {
        id = voice-console
        gate = voice
        service = voice-console
        apply_class = live
    }
    class privacy_filter {
        id = privacy-filter
        gate = privacy_filter
        service = null
        apply_class = boot
        note = "openmed sidecar is separately fail-closed gated, no own catalogue id"
    }
    class podcast_ingest {
        id = podcast-ingest
        gate = skills.podcast_ingest
        service = podcast-cron
        apply_class = rebuild
        note = "RESOLVED ADR-2057 gap 1 - new entry; see AB-27.10"
    }

    CatalogueEntry <|-- comfyui
    CatalogueEntry <|-- imagemagick_mcp
    CatalogueEntry <|-- ffmpeg
    CatalogueEntry <|-- qgis_mcp
    CatalogueEntry <|-- blender_mcp
    CatalogueEntry <|-- gaussian_splatting
    CatalogueEntry <|-- jupyter
    CatalogueEntry <|-- code_server
    CatalogueEntry <|-- desktop
    CatalogueEntry <|-- gui_tools_sidecar
    CatalogueEntry <|-- voice_console
    CatalogueEntry <|-- privacy_filter
    CatalogueEntry <|-- podcast_ingest

    note for CatalogueEntry "system-manifest.js:27-30 ApplyClass semantics,<br/>50-61 code-server/jupyter/desktop/comfyui,<br/>133-145 qgis-mcp/blender-mcp/blender-local/imagemagick-mcp/ffmpeg,<br/>175-183 gui-tools-sidecar/isaac-sim-sidecar/voice-console, 241-243 privacy-filter,<br/>373-375 podcast-ingest (RESOLVED ADR-2057 gap 1, new entry —<br/>see AB-22 for harness-bridge :367 and colloquy :370, the ADR-2057<br/>gap 2/gap 4 entries, out of this topic's scope),<br/>415-417 gaussian-splatting"
```

## AB-27.12 gui-tools sidecar GPU path — vglrun/nixGL, shared and unshared

```mermaid
sequenceDiagram
    autonumber
    participant Nix as main image Nix package set<br/>agentbox/flake.nix:1461-1486
    participant Wrap as wrapGpuBin (flake.nix:444) -> gpuWrap.wrapGpuBins<br/>agentbox/lib/gpu-wrap.nix:76
    participant BlenderProxy as blender-mcp proxy (main image)
    participant QgisProxy as qgis-mcp proxy (main image)
    participant Sidecar as gui-tools-service (FHS container)<br/>agentbox/gui-tools-sidecar/Dockerfile
    participant BLaunch as launch-blender.sh:18-22
    participant QLaunch as launch-qgis.sh:15

    Note over Nix,Wrap: ADR-2006 — gpu.backend == "local-cuda" (agentbox.toml:1567)<br/>gpuActive gate flake.nix:443 — wrapGpuBin appends host driver dirs to<br/>LD_LIBRARY_PATH with --suffix (gpu-wrap.nix:56-63), CUDA-only, no GLX/Vulkan surface
    Nix->>Wrap: wrapGpuBin pkgs.qgis ["qgis"] (flake.nix:1470)
    Nix->>Wrap: wrapGpuBin pkgs.blender ["blender"] only if blender_local (flake.nix:1476-1480)<br/>agentbox.toml:635 sets blender_local = false, so not baked here
    Note over Wrap: DIVERGENCE — these two nixGL-wrapped derivations exist in the<br/>main image's package set (Blender only when blender_local, ADR-2142) but are NEVER what serves blender-mcp/qgis-mcp:<br/>both MCP servers proxy to the separate gui-tools-service sidecar instead<br/>(flake.nix:2348-2352 comment: 'nix-built QGIS in agentbox-main cannot reach<br/>the nvidia driver libs ... the same constraint as Blender')

    BlenderProxy->>Sidecar: TCP 9876 (gui-tools-service, own Xvfb display 2 + FHS rootfs)
    Sidecar->>BLaunch: [program:blender] gui-tools-sidecar/supervisord.conf:35
    alt vglrun found on PATH (launch-blender.sh:18)
        BLaunch->>Sidecar: vglrun -d egl blender ... — VirtualGL intercepts GLX,<br/>renders on the real GPU via DRM render node, composites back to Xvfb display 2
    else vglrun missing
        BLaunch->>Sidecar: plain blender ... (software GL, degraded)
    end

    QgisProxy->>Sidecar: TCP 9877
    Sidecar->>QLaunch: [program:qgis] gui-tools-sidecar/supervisord.conf:47
    QLaunch->>Sidecar: python3 qgis-mcp-headless.py, QT_QPA_PLATFORM=offscreen<br/>NO vglrun invocation (launch-qgis.sh has no vglrun reference)
    Note over QLaunch: INVARIANT violation risk if assumed shared — only Blender's<br/>GUI viewport actually runs under VirtualGL in this sidecar —<br/>QGIS's headless QgsApplication has no GL/window surface to accelerate
```


## AB-27.13 xr-runtime sidecar — Monado OpenXR plus Godot, behind Xvfb/VNC

```mermaid
flowchart TB
    subgraph boot["xr-runtime supervisord — priority order"]
        INIT["init-perms #40;priority 5#41;<br/>chown cargo registry + gdext target volumes to devuser<br/>xr-runtime/supervisord.conf:19-27, one-shot, exitcodes=0"]
        XVFB["xvfb #40;priority 10#41;<br/>Xvfb :3 1920x1080x24<br/>xr-runtime/supervisord.conf:30-38"]
        VNC["x11vnc #40;priority 20#41;<br/>mirrors :3 on VNC 5904, -nopw<br/>xr-runtime/supervisord.conf:41-49"]
        MONADO["monado #40;priority 25#41;<br/>launch-monado.sh<br/>xr-runtime/supervisord.conf:52-60"]
        GODOT["godot #40;priority 30, startsecs=10, startretries=10#41;<br/>launch-godot.sh<br/>xr-runtime/supervisord.conf:63-72"]
    end
    INIT --> XVFB --> VNC --> MONADO --> GODOT
    MONADO --> SOCK["monado_comp_ipc socket<br/>XDG_RUNTIME_DIR/monado_comp_ipc<br/>launch-godot.sh:32-38 — godot polls up to 60s"]
    SOCK --> GODOT
    GODOT --> GDEXT["build-gdext.sh #40;first boot only, ~5-10 min cold#41;<br/>launch-godot.sh:28 — cached in the target volume after"]
    GODOT --> IMPORT["godot --headless --import #40;one-shot resource import#41;<br/>launch-godot.sh:47"]
    GODOT --> SCENE["godot --verbose XRBoot.tscn against Monado<br/>launch-godot.sh:57-58"]
    subgraph driver["Monado input driver — XR_INPUT_DRIVER"]
        SIM["simulated #40;default#41;<br/>static stereo HMD, view count 2<br/>launch-monado.sh:17-24 — the reliable path"]
        QWERTY["qwerty #40;EXPERIMENTAL#41;<br/>WASD + click-drag 6DoF via keyboard/mouse<br/>launch-monado.sh:15-16,19-20 — produces NO head device<br/>and segfaults the compositor in this Monado build"]
    end
    MONADO -.-> SIM
    MONADO -.->|"opt-in only"| QWERTY
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT: stdin must be a pollable pipe that never EOFs — Monado's IPC mainloop epoll#40;#41;s<br/>stdin, and supervisord's closed/non-pollable stdin fd would fail XRT_ERROR_IPC_MAINLOOP_FAILED_TO_INIT<br/>#40;launch-monado.sh:50-52#41; — fixed by exec monado-service reading a pipe that never EOFs #40;launch-monado.sh:55#41;"]
        N2["DIVERGENCE: this sidecar has its OWN supervisord.conf and Dockerfile #40;xr-runtime/#41;, unlike<br/>gui-tools-sidecar/openmed-sidecar which share one FHS pattern with the main image's flake.nix<br/>#40;see AB-27.10#41; — it is a genuinely separate container, not a variant of the same compose overlay"]
        N3["Godot 4.3's mobile renderer #40;required to match Quest#41; logs a benign tonemapper shader-missing<br/>warning every frame #40;~2k/min unbounded#41; against Monado's multiview swapchain — launch-godot.sh:53-63<br/>filters ONLY that exact signature out of stderr with a targeted grep -vE, so a genuinely<br/>different shader failure #40;distinct at: line#41; still surfaces"]
        N1 ~~~ N2 ~~~ N3
    end
```

## AB-27.14 Isaac Sim sidecar — opt-in headless physics behind a typed job API

```mermaid
flowchart TB
    CLI["./agentbox.sh isaac-sim up|smoke|health<br/>host shell only, GPU UUID required<br/>agentbox.sh:2167"]
    SVC["isaac-sim container, uid 1000, no-new-privileges<br/>docker-compose.isaac-sim.yml:5, docker-compose.isaac-sim.yml:58"]
    GPU["GPU pinned by ISAAC_SIM_GPU_UUID, no all default<br/>docker-compose.isaac-sim.yml:20-21"]
    RUN["internal job runner, port 9890 on visionclaw_network only<br/>expose, no ports: docker-compose.isaac-sim.yml:42-43"]
    VOL["gui-tools exchange volume, runner confined to isaac/<br/>docker-compose.isaac-sim.yml:49"]
    BLD["Blender in gui-tools-service reads results<br/>no recreation of either container<br/>docker-compose.isaac-sim.yml:47-48"]
    DCC["gui-tools-service also exposes port 9881 DCC gateway<br/>container network only, DCC_MCP_ENABLED<br/>docker-compose.gui-tools.yml:30, docker-compose.gui-tools.yml:53-55"]
    CAT["system-manifest catalogue: isaac-sim-sidecar, live, gate null<br/>management-api/lib/system-manifest.js:178-180"]
    CLI --> SVC
    GPU --> SVC
    SVC --> RUN
    SVC --> VOL
    VOL --> BLD
    BLD --- DCC
    CAT -.-> SVC
```

**Open:** ADR-2149 is recorded as `decision_status: proposed`, `implementation_status: partial`, `activation_status: staged`, so this sidecar is drawn as an opt-in design, not a running service; health beyond the control level becomes true only from a succeeded smoke job (`../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:5-7`, `../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:47-48`).

**Tension (ADR-2149 vs catalogue):** the ADR says the sidecar is live-class and ungated with no manifest section (`../project/agentbox/docs/adr/ADR-2149-isaac-sim-sidecar.md:29`), and the catalogue lists it as `apply_class: 'live'` with `gate: null` (`../project/agentbox/management-api/lib/system-manifest.js:178-180`); the decision itself is still `proposed`, so the catalogue already advertises a capability its record has not accepted.
