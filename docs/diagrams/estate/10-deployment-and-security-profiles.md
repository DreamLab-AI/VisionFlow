---
id: ES-10
title: Deployment and security profiles across the estate
area: estate
governing:
  - ../project/docs/SECURITY-profiles.md
  - ../project/agentbox/docs/SECURITY-profiles.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/docs/IDENTITY-authority-chain.md
adrs: [visionclaw:ADR-2003, visionclaw:ADR-2010, visionclaw:ADR-2012, agentbox:ADR-2012, agentbox:ADR-2013, visionclaw:ADR-2026, visionclaw:ADR-2027, agentbox:ADR-2027, visionclaw:ADR-2037, visionclaw:ADR-2038, visionclaw:ADR-2039, agentbox:ADR-2062, visionclaw:ADR-2086, visionclaw:ADR-2087]
sources:
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/main.rs
  - ../project/src/utils/auth.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/src/services/role_store.rs
  - ../project/docker-compose.unified.yml
  - ../project/docker-compose.cloudflared.yml
  - ../project/docs/adr/ADR-2027-three-deployment-profiles.md
  - ../project/docs/adr/ADR-2037-production-build-excludes-dev-auth.md
  - ../project/docs/adr/ADR-2038-boot-time-profile-assertion.md
  - ../project/docs/adr/ADR-2039-visionclaw-dev-mode-lan-local-bypass.md
  - ../project/agentbox/docs/adr/ADR-2013-loopback-publish-except-9096.md
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/agentbox.toml
  - ../project/src/config/security_profile.rs
  - ../project/agentbox/.github/workflows/invariants.yml
  - ../project/scripts/backup-secrets.sh
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/services/secret-backup/src/main.rs
verified_commit: {visionclaw: dd82a07b0, agentbox: 2c521c5bb}
---
## ES-10.1 Three named profiles — exact flag set per profile vs the fail-closed code default
```mermaid
flowchart TB
    subgraph code["Code defaults — fail-closed (ADR-2026)"]
        CD1["RBAC_PUBLIC_READS<br/>unwrap_or(false) = OFF<br/>rbac_gate.rs:126-132"]
        CD2["RBAC_ALLOW_OWNERLESS<br/>absent = refuse boot<br/>project/src/main.rs:735-755"]
        CD3["PUBKEY_VISIBILITY_FILTER<br/>parse_visibility_flag = ON<br/>position_updates.rs:34-58"]
        CD4["RBAC_DEFAULT_ROLE<br/>RBAC_DEFAULT_ROLE_ENV role_store.rs:41<br/>parse_default_role :195 = Editor<br/>fails closed to viewer on an unknown value :204"]
        CD5["RBAC_GATE_MODE<br/>enforce<br/>rbac_gate.rs:82-102"]
    end
    subgraph demo["demo-open — public read-only kiosk"]
        D1["RBAC_PUBLIC_READS=1"]
        D2["RBAC_ALLOW_OWNERLESS=1"]
        D3["RBAC_OWNER_PUBKEY unset"]
        D4["RBAC_DEFAULT_ROLE=editor"]
        D5["PUBKEY_VISIBILITY_FILTER=1"]
    end
    subgraph single["single-tenant — one operator, private graph"]
        S1["RBAC_PUBLIC_READS=0"]
        S2["RBAC_ALLOW_OWNERLESS=1"]
        S3["RBAC_OWNER_PUBKEY set 64-hex"]
        S4["RBAC_DEFAULT_ROLE=editor"]
        S5["PUBKEY_VISIBILITY_FILTER=1"]
    end
    subgraph locked["multi-user-locked — hardened multi-tenant"]
        L1["RBAC_PUBLIC_READS=0"]
        L2["RBAC_ALLOW_OWNERLESS=0"]
        L3["RBAC_OWNER_PUBKEY set 64-hex"]
        L4["RBAC_DEFAULT_ROLE=viewer"]
        L5["PUBKEY_VISIBILITY_FILTER=1"]
    end
    ALL["All three profiles pin<br/>APP_ENV=production<br/>RBAC_GATE_MODE=enforce<br/>SETTINGS_AUTH_BYPASS unset"]
    INV["INVARIANT ADR-2027 — a flag left unlisted takes its<br/>fail-closed code default. Any combination outside<br/>these three rows is UNSUPPORTED, a defect not a variant."]
    DRIFT["ADR-2038 HAS LANDED — profiles are machine-selected by<br/>src/config/security_profile.rs via VISIONCLAW_SECURITY_PROFILE (security_profile.rs:54),<br/>missing intent now refuses non-debug startup. The file is now tracked<br/>and the call site is in HEAD (project/src/main.rs:883), so the record reads<br/>decision_status accepted / activation_status live. NOTE the table above<br/>is SIX flags, not four — PROFILE_FLAGS (security_profile.rs:73) adds RBAC_OWNER_PUBKEY<br/>and RBAC_GATE_MODE. ADR-2027 corrected. see ES-10.7"]

    code --> demo
    code --> single
    code --> locked
    demo --> ALL
    single --> ALL
    locked --> ALL
    ALL --> INV
    INV --> DRIFT
```

## ES-10.2 Boot gate order — every check before the listener binds
```mermaid
sequenceDiagram
    autonumber
    participant OS as docker entrypoint
    participant M as main<br/>project/src/main.rs:195
    participant EH as enforce_release_env_hygiene<br/>project/src/main.rs:118
    participant EV as env validation<br/>project/src/main.rs:59-85
    participant RS as RoleStore bootstrap<br/>project/src/main.rs:754
    participant L as HTTP listener

    OS->>M: exec visionclaw binary
    rect rgb(240,230,230)
    M->>EH: enforce_release_env_hygiene()
    Note over EH: Compiled ONLY when neither debug_assertions<br/>nor feature dev-auth holds. Dev builds get the<br/>no-op stub at project/src/main.rs:169 (ADR-2037)
    alt argv contains --allow-skip-auth
        EH-->>M: FATAL exit — project/src/main.rs:120-122
    else env has SETTINGS_AUTH_BYPASS or ALLOW_INSECURE_DEFAULTS or VISIONCLAW_DEV_MODE
        EH-->>M: FATAL exit 2 — project/src/main.rs:130-132
    else NODE_ENV=development AND DOCKER_ENV both set
        EH-->>M: FATAL exit — project/src/main.rs:140-146
    else clean
        EH-->>M: ok
    end
    end
    M->>EV: read APP_ENV — project/src/main.rs:85
    Note over EV: DOC-DRIFT resolved in code — the case-sensitive<br/>APP_ENV=production runtime guard was REMOVED as a T2<br/>anti-pattern (APP_ENV=Production defeated it). The<br/>fence moved to the binary level (project/src/main.rs:76-79)
    alt is_production
        EV-->>M: missing required vars are a hard failure
    else non-production
        EV-->>M: permissive
    end
    M->>RS: bootstrap_owner_from_env(RBAC_OWNER_PUBKEY)
    alt an Owner is assigned
        RS-->>M: ok
    else no Owner and RBAC_ALLOW_OWNERLESS=1
        RS-->>M: warn and continue — project/src/main.rs:742
    else no Owner and flag unset
        RS-->>M: FATAL PermissionDenied — project/src/main.rs:783
    end
    M->>L: bind
    Note over M,L: RESOLVED — ADR-2038 has LANDED. assert_effective_profile_or_exit is called<br/>at project/src/main.rs:912 BEFORE HttpServer::new and .bind,<br/>exiting 2 on any finding in a non-debug artefact, including release/dev-auth.<br/>Only a debug build<br/>logs and continues (project/src/main.rs:875-877). src/config/security_profile.rs<br/>is tracked and the call site is in HEAD, so the record now reads<br/>decision_status accepted / activation_status live. Boot receipt logged<br/>project/src/main.rs:889-891. see ES-10.7 and VC-09.4
```

## ES-10.3 Shipped compose inverts two fail-closed code defaults
```mermaid
flowchart LR
    subgraph codeside["src/ — fail-closed defaults"]
        A1["public_reads_enabled()<br/>.unwrap_or(false)<br/>rbac_gate.rs:132"]
        A2["RBAC_ALLOW_OWNERLESS_ENV absent<br/>refuse to start<br/>project/src/main.rs:783"]
        A3["parse_visibility_flag<br/>defaults ON<br/>position_updates.rs:34"]
    end
    subgraph composeside["docker-compose.unified.yml — shipped"]
        B1["RBAC_PUBLIC_READS: ${RBAC_PUBLIC_READS:-1}<br/>line 93"]
        B2["RBAC_ALLOW_OWNERLESS: ${RBAC_ALLOW_OWNERLESS:-1}<br/>line 94"]
        B3["PUBKEY_VISIBILITY_FILTER: ${PUBKEY_VISIBILITY_FILTER:-1}<br/>line 107"]
        B4["RBAC_DEFAULT_ROLE: ${RBAC_DEFAULT_ROLE:-editor}<br/>line 101"]
        B5["VISIONCLAW_DEV_MODE: ${VISIONCLAW_DEV_MODE:-0}<br/>line 85"]
    end
    NET["Net shipped posture = demo-open<br/>anonymous /api reads ON, owner-less boot permitted"]
    DRIFT1["Compose defaults describe demo-open flags, but release now<br/>requires explicit VISIONCLAW_SECURITY_PROFILE intent.<br/>Missing intent or unnamed flags refuse listener bind (ADR-2038)"]
    DRIFT2["RESOLVED ADR-2087: docs/SECURITY-profiles.md now cites the<br/>code default and the compose default as two SEPARATE facts —<br/>fail-closed at rbac_gate.rs:126-132 (unwrap_or(false)) vs the<br/>demo-open override at docker-compose.unified.yml:93-94.<br/>docs/DATA-authority-erasure.md still says default ON at<br/>rbac_gate.rs:123-126 — routed to vc-knowledge, open until applied."]

    A1 -- "inverted by" --> B1
    A2 -- "inverted by" --> B2
    A3 -- "matched by" --> B3
    B1 --> NET
    B2 --> NET
    B3 --> NET
    B4 --> NET
    B5 --> NET
    NET --> DRIFT1
    A1 --> DRIFT2
```

## ES-10.4 Illegal combinations and where each is actually enforced
```mermaid
flowchart TB
    subgraph hard["Machine-enforced — hard-fail at boot"]
        H1["SETTINGS_AUTH_BYPASS or VISIONCLAW_DEV_MODE or<br/>ALLOW_INSECURE_DEFAULTS set in a release build"]
        H1E["exit 2 — project/src/main.rs:130-132"]
        H2["--allow-skip-auth argv in release"]
        H2E["FATAL — project/src/main.rs:120-122"]
        H3["NODE_ENV=development plus DOCKER_ENV"]
        H3E["FATAL — project/src/main.rs:140-146"]
        H4["RBAC_ALLOW_OWNERLESS=0 with no RBAC_OWNER_PUBKEY<br/>and no prior Owner"]
        H4E["PermissionDenied, refuses to start<br/>project/src/main.rs:783"]
    end
    subgraph soft["Refuses to activate — falls back to safe value"]
        S1["RBAC_GATE_MODE=report in release without<br/>RBAC_REPORT_MODE_ACK = today UTC"]
        S1E["refuses to disable auth, stays enforce<br/>rbac_gate.rs:96-102 — see ES-10.5"]
    end
    subgraph none["Effective-profile enforcement before bind"]
        N1["RBAC_PUBLIC_READS=1 with PUBKEY_VISIBILITY_FILTER=0"]
        N1E["Illegal pair has an explicit finding<br/>evaluate_effective_profile security_profile.rs:485<br/>Non-debug builds refuse binding"]
        N2["VISIONCLAW_SECURITY_PROFILE unset"]
        N2E["MissingDeclaredProfile finding<br/>Non-debug builds refuse binding"]
    end

    H1 --> H1E
    H2 --> H2E
    H3 --> H3E
    H4 --> H4E
    S1 --> S1E
    N1 --> N1E
    N2 --> N2E
```

## ES-10.5 RBAC_GATE_MODE=report — dated acknowledgement or refuse
```mermaid
sequenceDiagram
    autonumber
    participant B as boot
    participant RG as RbacGate::from_env<br/>src/middleware/rbac_gate.rs:183
    participant RA as report_acknowledged<br/>src/middleware/rbac_gate.rs:105-111
    participant REQ as inbound /api request

    B->>RG: construct from env
    RG->>RG: read RBAC_GATE_MODE (default enforce)
    alt RBAC_GATE_MODE=report
        RG->>RA: report_mode_acknowledged(env, build, today)
        Note over RA: today = Utc::now().format("%Y-%m-%d")<br/>ack must equal TODAY's UTC date — a stale ack<br/>from yesterday silently stops working
        alt debug build or RBAC_REPORT_MODE_ACK = today UTC
            RA-->>RG: true
            RG-->>B: warn RBAC_GATE_MODE=report is ACTIVE — denials LOGGED not enforced<br/>rbac_gate.rs:90
        else release build without a dated ack
            RA-->>RG: false
            RG-->>B: refuse — rbac_gate.rs:96-98, falls back to enforce
        end
    else enforce
        RG-->>B: enforce
    end
    RG->>RG: public_reads_enabled() — rbac_gate.rs:126
    alt RBAC_PUBLIC_READS is "1" or "true"
        RG-->>B: log anonymous /api reads ENABLED — rbac_gate.rs:194
    else absent or any other value
        RG-->>B: fail closed, reads require auth — rbac_gate.rs:132
    end
    REQ->>RG: method + path
    RG->>RG: required_level(method, path, public_reads) — rbac_gate.rs:138
    Note over RG: INVARIANT — absence of a security flag must never<br/>widen access. RBAC_PUBLIC_READS may only widen reads<br/>when EXPLICITLY set (public_reads_enabled rbac_gate.rs:125-131,<br/>unwrap_or(false) at rbac_gate.rs:132)
```

## ES-10.6 VISIONCLAW_DEV_MODE — peer-agnostic LAN-local full bypass (ADR-2039)
```mermaid
sequenceDiagram
    autonumber
    participant HP as Godot client on HP-Desktop<br/>ws to 192.168.2.132 port 4000
    participant DK as Docker bridge SNAT
    participant AX as AuthenticatedUser extractor<br/>src/settings/auth_extractor.rs
    participant DB as dev_full_bypass_active<br/>src/utils/auth.rs:99
    participant VA as verify_access<br/>src/utils/auth.rs:142
    participant WS as WS handshake<br/>src/handlers/socket_flow_handler/filter_auth.rs

    Note over HP,DK: CONTEXT ADR-2039 — Docker port-publishing SNATs the<br/>source, so the backend sees the bridge gateway not the<br/>real HP. Neither a loopback check nor a LAN-CIDR<br/>allow-list can express trust my headset.
    HP->>DK: graph write (layout-DAG trigger, node drag, settings)
    DK->>AX: request with rewritten source address
    AX->>DB: dev_full_bypass_active()
    rect rgb(240,230,230)
    alt release build
        DB-->>AX: false — codepath cfg-stripped, and mere presence<br/>of VISIONCLAW_DEV_MODE hard-fails boot (see ES-10.2)
    else dev or dev-auth build with VISIONCLAW_DEV_MODE unset or 0
        DB-->>AX: false — src/utils/auth.rs:99
    else dev or dev-auth build with VISIONCLAW_DEV_MODE=1 or true
        DB-->>AX: true (whitespace and case insensitive)
        AX->>VA: grant dev-admin identity dev-mode-local-admin
        Note over VA: BYPASS IS TOTAL — no NIP-98, no token, no peer check,<br/>across REST verify_access, the settings extractor and<br/>the WS handshake. Peer-agnostic BY DESIGN.
        VA-->>AX: authorised
        AX->>WS: same bypass on WS upgrade
    end
    end
    Note over HP,WS: DIVERGENCE ADR-2039 is decision_status proposed but<br/>implementation_status COMPLETE and activation inactive.<br/>Root cause it works around — the client's NIP-98 signing<br/>has a standing u-tag URL-mismatch bug, so every<br/>server-write HUD action returns 403.
```

## ES-10.7 Boot-time profile assertion and remaining decision divergence

```mermaid
stateDiagram-v2
    [*] --> Evaluate
    Evaluate --> Named: declared supported flags
    Evaluate --> Findings: missing intent or unnamed flags
    Evaluate --> Findings: forbidden build or environment
    Named --> Bind: no findings
    Findings --> Abort: non-debug build
    Findings --> Report: debug build only
    Report --> Bind
    Abort --> [*]
    Bind --> [*]
    note right of Evaluate
        evaluate_effective_profile security_profile.rs:485
        validates explicit intent and effective flags.
        The illegal disclosure pair is explicitly checked.
    end note
    note right of Abort
        assert_effective_profile_or_exit security_profile.rs:624
        refuses before listener binding, including release/dev-auth.
        Actual release negative probes return exit 2.
    end note
    note right of Named
        ADR-2038 still proposes an implicit locked selector.
        Implementation requires explicit intent instead.
        Partial status retains that decision divergence;
        deployment activation is separately evidenced.
    end note
```

## ES-10.8 agentbox exposure policy — loopback-by-default with a sanctioned list (ADR-2013)
```mermaid
flowchart TB
    subgraph ci["CI gate — agentbox/.github/workflows/invariants.yml"]
        SC["check-ports-loopback.sh<br/>sweeps EVERY docker-compose*.yml"]
    end
    subgraph rule["Rule"]
        R1["A publish must bind 127.0.0.1:<br/>OR appear on the in-script SANCTIONED list"]
        R2["Long-syntax published: / host_ip: mappings<br/>are FORBIDDEN in every file"]
    end
    subgraph sanctioned["SANCTIONED LAN doors — each cites its rationale"]
        P1[" port 9096 nip98-proxy — sovereign ingress (ADR-045)"]
        P2[" port 8443 and  port 8444 voice cockpit TLS door"]
        P3[" port 5903 /  port 8931 /  port 9222 browsercontainer<br/>VNC / MCP SSE / raw CDP"]
        P4[" port 5905 /  port 9876 /  port 9877 gui-tools"]
        P5[" port 5904 xr-runtime"]
    end
    FAIL["Anything else FAILS CI"]
    INV["INVARIANT —  port 9095 AoE serve is NEVER published to the LAN. It runs<br/>aoe serve --auth token --behind-proxy --allowed-host 127.0.0.1<br/>--host 127.0.0.1 (agentbox/flake.nix:2353) — it binds loopback EXPLICITLY.<br/> port 9096 is the one identity-gated door, proxy_port at agentbox/flake.nix:206,<br/>published 9096 port 9096 at agentbox/docker-compose.yml:54 (noted flake.nix:2373)."]
    D1["RESOLVED ADR-2013 — the estate has TEN sanctioned LAN publishes,<br/>not one front door and not two. the main compose publishes only  port 9096<br/>(agentbox/docker-compose.yml:54) but the overlays add nine more, each with a<br/>cited rationale on the SANCTIONED list (check-ports-loopback.mjs:93-104)<br/>and CI-enforced. These are DECIDED exposures, not an admitted breach."]
    D2["RESOLVED ADR-2013 closeout 2026-09-05 — the scanner is now a<br/>strict YAML PARSER, not an awk line-walker<br/>(check-ports-loopback.mjs:8-24). It rejects the flow-mapping<br/>and JSON-flow bypasses that previously passed, plus IPv6<br/>binds and non-sequence ports values (:39-41)."]
    D3["RESOLVED ADR-2040 — code-server still binds 0.0.0.0:8080 inside the<br/>container while compose publishes 127.0.0.1:8080:8080<br/>(agentbox/docker-compose.yml:61), and a loopback PUBLISH constrains the<br/>HOST only: agentbox also joins visionclaw_network<br/>(agentbox/docker-compose.yml:162), so a PEER CONTAINER still reaches it.<br/>The hole was closed by AUTHENTICATION, not by rebinding — the flag is now<br/>--auth password, not --auth none (agentbox/flake.nix:2286), with the<br/>password minted 0600 at boot by entrypoint-unified.sh and never baked into<br/>the generated supervisor text (agentbox/flake.nix:2279-2285)."]
    D5["PROPOSED ADR-2062: the gate reasons about PUBLISHED ports and<br/>is structurally blind to a container-internal 0.0.0.0 bind on a<br/>shared bridge. The invariant is to be restated in terms of<br/>LISTENERS, with each supervised program declaring its bind address."]
    D4["DOC-DRIFT — the surviving stale --auth none claim is a COMMENT at<br/>agentbox/docker-compose.yml:52. The GENERATOR is already correct<br/>(agentbox/flake.nix:2286 emits --auth password); the committed ARTEFACT<br/>predates that fix and has not been regenerated. It self-heals on the next<br/>nix build .#compose. DECISION: left unpatched by hand — the file header<br/>says AUTO-GENERATED, do not edit by hand, and hand-patching a generated<br/>file teaches the wrong habit."]

    SC --> R1
    SC --> R2
    R1 --> sanctioned
    R1 --> FAIL
    sanctioned --> INV
    P2 --> D1
    SC --> D2
    R1 --> D3
    INV --> D4
    D3 --> D5
```

## ES-10.9 cloudflared — the tunnel exposure path
```mermaid
flowchart LR
    subgraph internet["Public internet"]
        U["visitor"]
        CF["Cloudflare edge<br/>Zero Trust Public Hostname"]
    end
    subgraph host["Docker host — visionclaw_network (external)"]
        CFD["cloudflared-visionclaw<br/>tunnel --no-autoupdate run<br/>docker-compose.cloudflared.yml"]
        NG["visionclaw-server network alias<br/>nginx"]
        BE["Rust backend"]
    end
    U -- "https junkiejarvis.com" --> CF
    CF -- "outbound-initiated tunnel<br/>no inbound port opened on the host" --> CFD
    CFD -- " port 3001 http" --> NG
    NG -- " port 4000" --> BE

    T1["TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN} — compose FAILS<br/>fast if unset (:? guard). Image pinned by sha256 digest."]
    T2["INVARIANT — no local config.yml. The ingress mapping is<br/>configured in the Cloudflare dashboard, so the exposed<br/>route is NOT reviewable from this repo."]
    T3["DIVERGENCE — the tunnel bypasses the loopback-publish<br/>posture entirely: it needs no published port, so a<br/>compose port audit cannot see this exposure.<br/>It reaches nginx  port 3001 as an ordinary network peer."]
    T4["DIVERGENCE — cloudflared fronts whatever RBAC profile the<br/>backend booted. With the shipped demo-open compose<br/>(see ES-10.3) that is anonymous /api reads on the<br/>public internet."]

    CFD --> T1
    CF --> T2
    CFD --> T3
    NG --> T4
```

## ES-10.10 agentbox credential custody register — roles and open acceptance evidence
```mermaid
flowchart TB
    subgraph reg["Provisional custody register — agentbox/docs/SECURITY-profiles.md 2026-09-04"]
        C1["Bridge identity / unwrap key<br/>AGENTBOX_BRIDGE_SK_FILE default /run/secrets/nostr.key<br/>legacy env fallback remains"]
        C2["Shared server publisher identity<br/>ADR-2012 per-consumer split PENDING<br/>relay key list is build-projected"]
        C3["Proxy break-glass bearer<br/>NIP98_PROXY_ALLOW_BEARER captured at process start"]
        C4["Proxy browser-session signing secret<br/>NIP98_PROXY_SESSION_SECRET or per-boot random"]
        C5["AoE daemon token<br/>state file read by proxy with last-good cache"]
        C6["Dream remote-execution identity<br/>ssh/scp uses AMBIENT ssh config, no explicit identity file"]
        C7["VisionClaw legacy backup<br/>scripts/backup-secrets.sh: ZIP plus manifest"]
        C8["Agentbox Rust backup source<br/>services/secret-backup: tar inside age, owner-only output"]
    end
    ST["STATUS — proposed governing surface. Every custodian,<br/>deployed location, rotation cadence and incident response<br/>time is UNCONFIRMED. No cadence is invented."]
    D1["PARTIAL — break-glass checks optional expiry and method/path scope.<br/>Unset bounds allow unbounded use. Acceptance/refusal logs include<br/>a fingerprint and counters, but durable per-use audit is unproven."]
    D2["TWO SOURCE PATHS — VisionClaw backup-secrets.sh still writes<br/>ordinary ZIP with an integrity check, without explicit encryption<br/>or permission hardening. Agentbox Rust backup refuses missing<br/>encryption authority and hardens output to 0600. Its existence<br/>does not replace the legacy script or prove deployed adoption."]
    D3["DIVERGENCE ADR-040 D3 — agentbox/agentbox.toml:148 marks the<br/>governance publisher key-split PENDING, so governance<br/>events and server identity still share a key."]
    D4["DIVERGENCE — deleting the AoE state file alone does NOT<br/>rotate the daemon token, because the proxy holds a<br/>last-good cache. Daemon and proxy must rotate coherently."]
    D5["DIVERGENCE SOPS never executed (legacy ADR-109, accepted<br/>2026-05-09) — VisionClaw .env is PLAINTEXT today, no SOPS<br/>artifacts in tree."]

    reg --> ST
    C3 --> D1
    C7 --> D2
    C8 --> D2
    C2 --> D3
    C5 --> D4
    ST --> D5
```

## Custody audit qualification — 2026-09-07

ES-10.10 follows `proxy.mjs::verifyIdentity`, `breakGlassNotExpired` and `breakGlassScopeAllows`, the existing VisionClaw ZIP script, and Agentbox `services/secret-backup/src/main.rs::backup`. The encrypted implementation and legacy script coexist. No actual credential, backup, deployed configuration or restoration was inspected or exercised. Lifecycle acceptance remains open; see the [Agentbox audit](../../estate-review/2026-09-07-agentbox-audit.md).


The source closeout adds findings for missing profile intent and unnamed effective
flags, and refuses every finding in non-debug builds. Explicit intent is required;
no implicit locked profile is selected. That differs from ADR-2038's original
production-selector proposal, so the record remains partial pending owner decision.
Actual default release artefact probes rejected forbidden variables even when set
to zero; the receipt identifies that artefact and does not certify every image.
