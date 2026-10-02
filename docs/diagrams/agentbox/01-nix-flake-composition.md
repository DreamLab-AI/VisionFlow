---
id: AB-01
title: Nix flake composition and apply-class gates
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2006, ADR-2029, ADR-2039, ADR-2080]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/lib/gpu-wrap.nix
  - ../project/agentbox/lib/npm-cli.nix
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/scripts/agentbox-config-validate.js
  - ../project/agentbox/management-api/lib/system-manifest.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/flake.lock
  - ../project/agentbox/lib/rune.nix
  - ../project/agentbox/lib/vault.nix
  - ../project/agentbox/scripts/agentbox-config-validate.sh
  - ../project/agentbox/scripts/post-deploy-cleanup.sh
  - ../project/agentbox/scripts/skill-count-check.js
  - ../project/agentbox/config/model-router/artefacts.json
  - ../project/agentbox/scripts/refresh-compose.sh
verified_commit: 5ab197a9d49e9721b85b791bf9efe30842c9e047
---

## AB-01.1 agentbox.toml gates to flake.nix conditionals to package set and supervisord text

```mermaid
flowchart TB
    TOML["agentbox.toml<br/>flake.nix:128 builtins.fromTOML"] --> CFG["agentboxConfig"]
    CFG --> DESK["desktopCfg = agentboxConfig.desktop or {}<br/>flake.nix:150"]
    CFG --> MEDIA["mediaCfg = skillsCfg.media or {}<br/>flake.nix:220"]
    CFG --> VAULT["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:725"]

    DESK -->|"desktopCfg.enabled or false"| DESKOPT["lib.optionals<br/>flake.nix:1680 desktopPackages"]
    MEDIA -->|"mediaCfg.comfyui_builtin or false"| COMFYOPT["lib.optionals<br/>flake.nix:1249 comfyuiPackages"]
    MEDIA -->|"mediaCfg.ffmpeg or false"| FFOPT["lib.optionals<br/>flake.nix:1257 wrapGpuBin ffmpeg"]
    VAULT -->|"vaultCfg.tui == rune"| RUNEOPT["runeActive<br/>flake.nix:727"]

    DESKOPT --> ALLPKG["allPackages closure"]
    COMFYOPT --> ALLPKG
    FFOPT --> ALLPKG
    RUNEOPT -->|"lib.optionals runeActive"| RUNEPKG["runePackages<br/>flake.nix:729"]
    RUNEPKG --> ALLPKG

    DESK -->|"lib.optionalString (desktopCfg.enabled or false)"| DESKSUP["desktopBlocks text<br/>flake.nix:2176<br/>spliced flake.nix:2372"]
    MEDIA -->|"lib.optionalString (mediaCfg.comfyui_builtin or false)"| COMFYSUP["program:comfyui-builtin block<br/>flake.nix:2466-2475"]

    CFG --> SIDE["sidechainCfg = agentboxConfig.sidechain or {}<br/>flake.nix:239, enabled default false (:240)<br/>mirror and faucet only with enabled (:241-242)"]
    SIDE -->|"lib.optionals sidechainFaucet"| SIDEPKG["sidechainPackages = sidestr-agent<br/>flake.nix:1576"]
    SIDEPKG --> ALLPKG
    SIDE -->|"lib.optionalString sidechainEnabled / Mirror / Faucet"| SIDESUP["program:sidestr-producer, -mirror, -faucet blocks<br/>flake.nix:2631, :2655, :2672"]
    SIDESUP --> SUPTEXT
    CFG --> JEVON["jevCompactionOn = features.jev_compaction.enabled<br/>flake.nix:1580, ADR-2121"]
    JEVON -->|"lib.optionals jevCompactionOn"| FRPKG["factrailPackages<br/>flake.nix:1582"]
    FRPKG --> ALLPKG
    JEVON -->|"lib.optionalString jevCompactionOn"| FRBAKE["/opt/agentbox/bin/factrail symlink (flake.nix:1802)<br/>plugin copied to config/claude-plugins/factrail (flake.nix:1858)"]

    DESKSUP --> SUPTEXT["supervisorText — AUTO-GENERATED supervisord.conf<br/>flake.nix:2268 supervisorText = header"]
    COMFYSUP --> SUPTEXT
    ALLPKG --> MKIMAGE["mkImage layers<br/>flake.nix:3939"]

    ALLPKG -.->|"see AB-01.6"| WRAPTGT["wrapped GPU targets"]

    note1["INVARIANT ADR-2003 - every gate touches package set AND supervisor text AND<br/>a system-manifest.js catalogue entry, management-api/lib/system-manifest.js line 39"]
    SUPTEXT --- note1
```

## AB-01.2 apply_class taxonomy - live, boot, rebuild

```mermaid
stateDiagram-v2
    [*] --> live
    [*] --> boot
    [*] --> rebuild

    state live {
        [*] --> LiveRead
        LiveRead: Read at operation time
        LiveRead --> LiveEffect
        LiveEffect: flipping the key affects the running box, no restart
    }
    state boot {
        [*] --> BootRead
        BootRead: Read once at container boot
        BootRead --> BootEffect
        BootEffect: takes effect on next restart, entrypoint reconciles every boot
    }
    state rebuild {
        [*] --> RebuildRead
        RebuildRead: Changes the Nix image composition
        RebuildRead --> RebuildEffect
        RebuildEffect: needs agentbox.sh rebuild, gates package set AND supervisor block
    }

    note right of live
        APPLY_CLASSES const system-manifest.js line 27
        entries browser-sidecar 160, gui-tools-sidecar 163, voice-console 166, memory-hygiene 181
    end note
    note right of boot
        entries management-api 41, terminal 44, setup-wizard 47, vault root/pages/format 267, memory-learning 178
        newer boot entries claude-code-permissions 212, instruction-tiers 215,<br/>sovereign-system-one 224, skill-router 232, skill-router-cascade 235, routing-teacher-labels 238
        stateOf counts off and none as off, plus a per-entry off_values set, system-manifest.js line 339
    end note
    note right of rebuild
        entries code-server 50, jupyter 53, desktop 56, comfyui 59, vault-tui 270
        newer rebuild entries claude-cred-sync 218, jev-compaction 221 (boot until ADR-2121), sidechain 257,<br/>vault-cli 273, mcp-hub 291, hook-shim 294, teammate-gc 297
        DOES NOT reconcile on restart, needs full Nix re-evaluation via agentbox.sh rebuild
    end note

    live --> [*]: does NOT reconcile at boot, only live reads
    boot --> [*]: does NOT re-evaluate Nix composition
    rebuild --> [*]: does NOT take effect on a plain restart
```

## AB-01.3 ./agentbox.sh rebuild - down, build --variant runtime, up --build, cleanup

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant REB as cmd_rebuild<br/>agentbox.sh:1068
    participant DOWN as cmd_down<br/>agentbox.sh:870
    participant BUILD as cmd_build<br/>agentbox.sh:907
    participant NIX as nix build<br/>agentbox.sh:924
    participant UP as cmd_up<br/>agentbox.sh:728
    participant DOCKER as docker compose
    participant CLEAN as post-deploy-cleanup.sh

    OP->>REB: ./agentbox.sh rebuild [--no-cleanup]
    REB->>DOWN: cmd_down (agentbox.sh:1081)
    DOWN->>DOCKER: docker compose down (agentbox.sh:895)
    DOCKER-->>DOWN: stack stopped
    REB->>BUILD: cmd_build --variant runtime (agentbox.sh:1084)
    BUILD->>BUILD: validate variant in runtime|desktop|full<br/>agentbox.sh:917-920
    alt unknown variant
        BUILD-->>OP: exit 1 Unknown variant
    end
    BUILD->>BUILD: refresh-compose.sh, return 1 on failure (agentbox.sh:922)
    Note over BUILD: refresh-compose.sh nix-builds .#compose, validates it with<br/>docker compose config --quiet, then atomically replaces docker-compose.yml<br/>(refresh-compose.sh:7-22), so a rebuild applies mount changes too
    BUILD->>NIX: nix build .#runtime (agentbox.sh:924)
    NIX-->>BUILD: result symlink resolved (agentbox.sh:926-927)
    REB->>UP: cmd_up --build (agentbox.sh:1087)
    UP->>UP: mutually exclusive check --build vs --registry<br/>agentbox.sh:742-746
    UP->>UP: refresh-compose.sh again before the build (agentbox.sh:750)
    UP->>NIX: nix build .#runtime (agentbox.sh:752)
    UP->>DOCKER: nix run .#runtime.copyToDockerDaemon<br/>agentbox.sh:756
    UP->>UP: unset AGENTBOX_IMAGE_REF (agentbox.sh:758)
    UP->>UP: resolve image hash + manifest checksum<br/>agentbox.sh:781-791
    opt visionclaw_network absent
        UP->>DOCKER: docker network create visionclaw_network<br/>agentbox.sh:724
    end
    opt orphaned ruvector-postgres container
        UP->>DOCKER: docker rm -f ruvector-postgres<br/>agentbox.sh:809
    end
    UP->>DOCKER: docker compose up -d (agentbox.sh:821)
    loop poll every 2s up to 120s
        UP->>DOCKER: curl READY_URL (agentbox.sh:849)
    end
    alt readiness times out
        UP-->>OP: exit 1 Readiness check timed out<br/>agentbox.sh:857-859
    end
    UP-->>REB: Stack is up and ready (agentbox.sh:863)
    alt skip_cleanup == 0
        REB->>CLEAN: bash scripts/post-deploy-cleanup.sh<br/>agentbox.sh:1091
        CLEAN->>CLEAN: 1/5 prune old agentbox images, keep CURRENT_ID
        CLEAN->>CLEAN: 2/5 docker system prune -f
        CLEAN->>CLEAN: 3/5 nix store gc
        CLEAN->>CLEAN: 4/5 clean tmp build files
        CLEAN->>CLEAN: 5/5 reap stale cargo target dirs, AGENTBOX_REAP_CARGO
    else --no-cleanup
        Note over REB,CLEAN: cleanup skipped, skip_cleanup=1, agentbox.sh:1077-1078,1089
    end
```

## AB-01.4 Static config validation - schema, semantic rules, then the skill-count gate

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator or CI
    participant WRAP as agentbox-config-validate.sh
    participant JS as agentbox-config-validate.js
    participant AJV as Ajv 2020 compiler<br/>agentbox-config-validate.js:126
    participant SCHEMA as agentbox.toml.schema.json
    participant SEM as semantic rule functions
    participant SKC as skill-count-check.js

    OP->>WRAP: ./scripts/agentbox-config-validate.sh agentbox.toml
    WRAP->>WRAP: probe node_modules/@iarna/toml, ajv<br/>agentbox-config-validate.sh:27-29
    opt node_modules missing
        WRAP->>WRAP: npm ci bootstrap or exit 2 if AGENTBOX_VALIDATOR_NO_BOOTSTRAP=1
    end
    WRAP->>JS: exec node agentbox-config-validate.js agentbox.toml
    JS->>JS: TOML.parse(raw)<br/>agentbox-config-validate.js:108
    alt TOML parse error
        JS-->>OP: emit E000, exit 1<br/>agentbox-config-validate.js:110-111
    end
    JS->>SCHEMA: JSON.parse schema file<br/>agentbox-config-validate.js:117
    JS->>AJV: ajv.compile(schema)<br/>agentbox-config-validate.js:126
    JS->>AJV: validate(manifest)<br/>agentbox-config-validate.js:127
    alt schemaValid == false
        AJV-->>JS: additionalProperty violations
        JS->>JS: push E016 UnknownManifestKey per error<br/>agentbox-config-validate.js:137-143
    end
    JS->>SEM: run E0xx/W0xx rule families<br/>adapters, providers, nostr relay, privacy filter, linked-data
    SEM-->>JS: errors[] and warnings[] arrays populated
    JS->>SKC: checkSkillCount repoRoot<br/>E-SKILL1, ADR-037 D8, agentbox-config-validate.js:1657-1659
    alt skills/*/SKILL.md count diverges from a README/SKILL-DIRECTORY claim
        SKC-->>JS: divergences[] -> push E016-sibling E-SKILL1 per drift<br/>agentbox-config-validate.js:1660-1665
    else checkSkillCount throws (e.g. skills/ absent)
        JS->>JS: push W067, RES-d gate skipped this pass<br/>agentbox-config-validate.js:1666-1671
    end
    JS->>JS: emit every warning to stderr<br/>agentbox-config-validate.js:1677-1679
    alt errors.length == 0
        JS-->>OP: stdout "agentbox manifest valid" + advisory count, exit 0<br/>agentbox-config-validate.js:1681-1684
    else errors present
        JS->>JS: emit every error to stderr<br/>agentbox-config-validate.js:1686-1688
        JS-->>OP: exit 1<br/>agentbox-config-validate.js:1689
    end

    Note over JS,SEM: DIVERGENCE - static-schema stage is advisory for W0xx dead-policy warnings,<br/>only E016/E-SKILL1 schema/RES-d violations and other E-code semantic rules hard-fail<br/>see agentbox-config-validate.js line 4 comment and lines 1677-1689 exit logic
    Note over SKC: see AB-22 for skill-count-check.js internals (skills/*/SKILL.md as the single count source)
```

## AB-01.5 npm-cli.nix pinned exact-semver closure - ruvector always in package set

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:315
    participant MK as makeNpmCli<br/>lib/npm-cli.nix:120
    participant FETCH as pkgs.fetchurl stage 1<br/>lib/npm-cli.nix:184
    participant FOD as packageWithDeps FOD stage 2<br/>lib/npm-cli.nix:204
    participant WRAP as wrapper derivation stage 3
    participant ALWAYS as npmCliAlwaysPackages<br/>flake.nix:564

    FLAKE->>MK: mkNpmCli pkgName=ruvector version=0.3.3<br/>flake.nix:320-327
    MK->>FETCH: registryUrl ruvector 0.3.3<br/>lib/npm-cli.nix:102-115
    FETCH->>FETCH: sha256 = SRI hash of the .tgz<br/>lib/npm-cli.nix:186-187
    alt sha256 is lib.fakeHash placeholder
        FETCH-->>MK: eval-time hint, realisation-time hash mismatch<br/>lib/npm-cli.nix:161-173
    end
    MK->>FOD: npm ci from checked-in packageLock when supplied<br/>lib/npm-cli.nix:266-273, scripts disabled<br/>legacy-peer flags remain package-specific
    FOD->>FOD: outputHash = nodeModulesHash, network allowed inside sandbox<br/>lib/npm-cli.nix header Stage 2 rationale lines 29-37
    FOD-->>MK: $out/lib/ruvector with populated node_modules
    MK->>WRAP: thin mkDerivation, no network, writes $out/bin/ruvector wrapper<br/>lib/npm-cli.nix Stage 3 rationale lines 39-41
    WRAP-->>FLAKE: ruvectorPkg derivation
    FLAKE->>ALWAYS: npmCliAlwaysPackages = [ ruvectorPkg wranglerPkg ]<br/>flake.nix:564
    Note over FLAKE,ALWAYS: the nix-prefetch-url comment at flake.nix:319 and the version field<br/>at flake.nix:322 now agree on ruvector-0.3.3 (bumped together 2026-10-01)
    Note over FLAKE,ALWAYS: DRIFT: BASELINE-container.md:52 still states ruvector pins 0.3.0<br/>and that the prefetch comment names the old tarball, but flake.nix:322 pins 0.3.3<br/>and the comment at flake.nix:319 matches it - only the doc is stale now
```

## AB-01.6 gpu-wrap.nix wrapGpuBins - LD_LIBRARY_PATH suffix and vendor ICDs

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:292-294
    participant WRAP as gpuWrap.wrapGpuBins<br/>lib/gpu-wrap.nix:76
    participant JOIN as pkgs.symlinkJoin<br/>lib/gpu-wrap.nix:77
    participant MAKEW as makeWrapper wrapProgram<br/>lib/gpu-wrap.nix:87
    participant BIN as wrapped binary at runtime

    FLAKE->>FLAKE: gpuActive = agentbox.toml gpu.backend == local-cuda<br/>flake.nix:292
    alt gpu.backend == none
        FLAKE->>FLAKE: wrapGpuBin pkg bins = pkg, unwrapped passthrough<br/>flake.nix:293-296
        Note over FLAKE: alt gpu.backend=none - wrapping is inert without injected driver libs, gpu-wrap.nix comment lines 21-22
    else gpu.backend == local-cuda
        FLAKE->>WRAP: wrapGpuBins pkg=pkgs.blender bins=[blender]<br/>flake.nix:1272
        WRAP->>JOIN: paths=[pkg], nativeBuildInputs=[makeWrapper]<br/>lib/gpu-wrap.nix:77-80
        JOIN->>MAKEW: for each bin, wrapProgram target gpuEnvArgs<br/>lib/gpu-wrap.nix:81-89
        MAKEW->>MAKEW: --suffix LD_LIBRARY_PATH : /usr/lib:/usr/lib/x86_64-linux-gnu:/run/opengl-driver/lib<br/>lib/gpu-wrap.nix:46-51,56
        MAKEW->>MAKEW: --set-default __GLX_VENDOR_LIBRARY_NAME nvidia<br/>lib/gpu-wrap.nix:57
        MAKEW->>MAKEW: --set-default __EGL_VENDOR_LIBRARY_FILENAMES /usr/share/glvnd/egl_vendor.d/10_nvidia.json<br/>lib/gpu-wrap.nix:58-59
        MAKEW->>MAKEW: --set-default VK_ICD_FILENAMES /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json<br/>lib/gpu-wrap.nix:60-65
        JOIN-->>WRAP: symlinkJoin derivation, meta description appended C-9<br/>lib/gpu-wrap.nix:93-96
        WRAP-->>FLAKE: gpu-wrapped drop-in replacement for pkg
        FLAKE->>BIN: dlopen libcuda.so.1 resolves via appended LD_LIBRARY_PATH
        BIN-->>FLAKE: CUDA devices enumerated, verified RTX A6000 + 2x RTX 6000 Ada 2026-08-31
    end

    Note over MAKEW: INVARIANT - --suffix never --prefix, so Nix's own libstdc++/libc stays authoritative,<br/>ADR-2006 and gpu-wrap.nix comment lines 23-26
    Note over BIN: DIVERGENCE - GPU wrapper is CUDA-only by design, no Nix-binary Vulkan/GLX presentation path,<br/>interactive 3D depends on the FHS gui-tools sidecar, ADR-2006 Context and Consequences
    Note over MAKEW: DIVERGENCE - BASELINE GPU scope and evidence qualification 2026-09-04 -<br/>current wrappers include GLX/EGL/Vulkan defaults alongside CUDA library-path config,<br/>so ADR-2006's graphics review trigger has been reached, agentbox/docs/BASELINE-container.md line 257-259

    Note over WRAP: wrapped-target gates - ffmpeg mediaCfg.ffmpeg flake.nix:1258,<br/>qgis spatialCfg.qgis flake.nix:1265, blender spatialCfg.blender flake.nix:1272,<br/>3DGS spatialCfg.gaussian_splatting map wrapGpuAll gauss3dPackages flake.nix:1278<br/>(see AB-27 for the fuller MCP-proxy treatment of qgis/blender/ffmpeg)
    Note over WRAP: INVARIANT - wrapGpuBin names exact bins, wrapGpuAll wraps every<br/>executable under out/bin for upstream-versioned bin sets - colmap and lichtfeld<br/>(both CUDA) need the wrapper, metis is CPU-only so wrapping is inert for it,<br/>flake.nix comment lines 1274-1276
```

## AB-01.8 agentbox.toml schema shape - top-level sections

```mermaid
classDiagram
    class AgentboxToml {
        +GpuSection gpu
        +VaultSection vault
        +SkillsSection skills
        +AdaptersSection adapters
        +CoreSection core
        +FederationSection federation
    }
    class GpuSection {
        +string backend
    }
    class VaultSection {
        +string root
        +string pages
        +string format
        +string tui
        +bool cli
        +string working
        +string transcripts
    }
    class SkillsSection {
        +BrowserSkills browser
        +MediaSkills media
        +SpatialSkills spatial_and_3d
        +DataScienceSkills data_science
    }
    class AdaptersSection {
        +string beads
        +string pods
        +string memory
        +string events
        +string orchestrator
    }
    class CoreSection {
        +string orchestration
        +string vector_db
    }
    class FederationSection {
        +string mode
        +string external_url
    }

    AgentboxToml "1" --> "1" GpuSection
    AgentboxToml "1" --> "1" VaultSection
    AgentboxToml "1" --> "1" SkillsSection
    AgentboxToml "1" --> "1" AdaptersSection
    AgentboxToml "1" --> "1" CoreSection
    AgentboxToml "1" --> "1" FederationSection

    note for GpuSection "backend enum: none, ollama-rocm, ollama-cuda, local-cuda - agentbox.toml.schema.json:106-115"
    note for VaultSection "required: root - agentbox.toml.schema.json:222-224. format enum obsidian ONLY (logseq-legacy withdrawn 2026-09-22, corpus migration complete) - agentbox.toml.schema.json:248-255. tui enum rune, none, default none - agentbox.toml.schema.json:270-278, ADR-2029. cli bool default true - agentbox.toml.schema.json:279-283, ADR-2107/ADR-2108"
    note for AdaptersSection "each value resolves to local-star, external, or off per slot - agentbox/CLAUDE.md adapter contract"
```

## AB-01.9 flake.lock inputs to flake outputs

```mermaid
flowchart LR
    subgraph INPUTS["flake.lock root inputs"]
        NIXPKGS["nixpkgs - nixpkgs_3<br/>github NixOS/nixpkgs pin b19cbd07b1a6<br/>flake.nix:5"]
        FU["flake-utils<br/>github numtide/flake-utils<br/>flake.nix:6"]
        N2C["nix2container<br/>github nlewo/nix2container<br/>flake.nix:7"]
        RO["rust-overlay<br/>github oxalica/rust-overlay<br/>flake.nix:8"]
        AOE["aoe<br/>github DreamLab-AI/agentbox-of-empires pin 33e806ad<br/>flake.nix:18"]
        SKILLS["skills - path:./skills, flake=false<br/>flake.nix:26-27"]
        VAULTSRC["vaultSrc - github DreamLab-AI/VisionClaw pin 0c195f7605f3<br/>flake=false, flake.nix:41-44"]
        CODEX["codexPlugin<br/>github openai/codex-plugin-cc pin db52e28f<br/>flake.nix:52-55"]
    end

    NIXPKGS --> EVAL["flake outputs eval, per-system<br/>flake.nix:58-64 flake-utils.lib.eachSystem"]
    FU --> EVAL
    N2C --> EVAL
    RO --> EVAL
    AOE --> EVAL
    SKILLS --> EVAL
    VAULTSRC --> EVAL
    CODEX --> EVAL

    EVAL --> PACKAGES["packages - flake.nix:3978<br/>lib.optionalAttrs pkgs.stdenv.isLinux"]
    EVAL --> DEVSHELL["devShells.default<br/>flake.nix:4056"]

    PACKAGES --> RUNTIME["runtime = mkImage tag runtime-system<br/>flake.nix:3980"]
    PACKAGES --> FULL["full = mkImage extraPackages allPackages<br/>flake.nix:3981-3985"]
    PACKAGES --> DESKTOP["desktop = mkImage extraPackages desktopPackages<br/>flake.nix:3986-3990"]
    PACKAGES --> CUDART["cuda-runtime adds explicitly dispatched local-cuda packages<br/>flake.nix:4008-4017"]
    PACKAGES --> GSPLAT["gaussian-splatting = 3DGS stack over cuda-runtime<br/>flake.nix:4031-4042"]
    PACKAGES --> COMPOSE["compose = docker-compose.yml text, cross-platform<br/>flake.nix:4050-4053"]

    RUNTIME --> MKIMG["mkImage - n2c.buildImage 4 layers<br/>flake.nix:3939-3948"]
    FULL --> MKIMG
    DESKTOP --> MKIMG

    MKIMG --> ENTRYPOINT["config = Entrypoint entrypoint/bin/entrypoint<br/>flake.nix:3959"]
    VAULTSRC -.->|"see AB-01.10"| VAULTPKG["vaultPkg = import lib/vault.nix"]

    NOTE1["INVARIANT - container-image outputs are Linux-only,<br/>darwin exposes only compose and devShells, flake.nix comment lines 3974-3977"]
    PACKAGES --- NOTE1
```

## AB-01.10 ADR-2029/ADR-2107 vault gates - three catalogue entries, one section

```mermaid
flowchart TB
    TOMLVAULT["[vault] in agentbox.toml<br/>agentbox.toml:896<br/>tui = rune (:904), cli = true (:911)"] --> VAULTCFG["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:725"]

    VAULTCFG --> TUIVAL["vaultTui = vaultCfg.tui or none<br/>flake.nix:726"]
    TUIVAL --> RUNEACTIVE["runeActive = vaultTui == rune<br/>flake.nix:727"]
    RUNEACTIVE -->|"true"| RUNEPKGIMPORT["runePkg = import lib/rune.nix<br/>flake.nix:728, lazy import"]
    RUNEACTIVE -->|"lib.optionals runeActive"| RUNEPACKAGES["runePackages<br/>flake.nix:729"]
    RUNEPKGIMPORT --> RUNEPACKAGES

    VAULTCFG --> CLIVAL["vaultCliActive = root != '' and (vaultCfg.cli or true)<br/>flake.nix:750 - defaults ON, unlike tui"]
    CLIVAL -->|"true"| VAULTPKGIMPORT["vaultPkg = import lib/vault.nix src=vaultSrc<br/>flake.nix:751-755"]
    CLIVAL -->|"lib.optionals vaultCliActive"| VAULTPACKAGES["vaultPackages<br/>flake.nix:756"]
    VAULTPKGIMPORT --> VAULTPACKAGES

    RUNEPACKAGES --> ALLPKG["allPackages closure - Nix image composition"]
    VAULTPACKAGES --> ALLPKG

    RUNEBUILD["rune.nix pkgs.rustPlatform.buildRustPackage<br/>fetchFromGitHub jjohare/rune (DreamLab fork) tag v1.5.0-dreamlab.1<br/>lib/rune.nix:26-40, cargoBuildFlags -p rune-cli, doCheck=false<br/>cargoLock is the vendored lib/rune-Cargo.lock, not the source's (rune.nix:38),<br/>so nix flake check --no-build needs no import-from-derivation<br/>adds callouts, highlights, tags, backlinks panel, daily notes"]
    RUNEPKGIMPORT --> RUNEBUILD

    subgraph MANIFEST["system-manifest.js CATALOGUE - management-api/lib/system-manifest.js"]
        VAULTENTRY["id vault<br/>gate vault.format, apply_class boot<br/>line 267-269"]
        VAULTTUIENTRY["id vault-tui<br/>gate vault.tui, apply_class rebuild<br/>line 270-272"]
        VAULTCLIENTRY["id vault-cli<br/>gate vault.cli, apply_class rebuild<br/>line 273-275<br/>ADR-2107/ADR-2108, the one door to the corpus"]
    end

    TOMLVAULT -.->|"vault.format read at boot"| VAULTENTRY
    TOMLVAULT -.->|"vault.tui read at rebuild"| VAULTTUIENTRY
    TOMLVAULT -.->|"vault.cli read at rebuild"| VAULTCLIENTRY

    STATEOF["stateOf function<br/>system-manifest.js:317-342<br/>a false parent gate now dominates child gates (:318-319)"]
    VAULTTUIENTRY --> STATEOF
    STATEOF --> MODESTRING["off and none count as off, plus a per-entry off_values set<br/>skill-router declares table as its off value<br/>system-manifest.js:338-339, entry at system-manifest.js:232"]

    NOTE1["DOC-DRIFT - the code comment at system-manifest.js:260-266 still says the<br/>split is into TWO catalogue entries (ADR-2020), but [vault] now has THREE:<br/>root/pages/format are boot-class, tui and cli are both rebuild-class but gate<br/>different binaries (Rune TUI vs the vault CLI door)"]
    VAULTTUIENTRY --- NOTE1

    NOTE2["DIVERGENCE - vault-tui and vault-cli are REBUILD-class, flipping either key and<br/>restarting the container is NOT sufficient - the Rune binary or the vault CLI is<br/>absent from the package set until agentbox.sh rebuild runs"]
    STATEOF --- NOTE2

    NOTE3["INVARIANT ADR-2107/ADR-2108 - cli defaults to true whenever root is set (opt-OUT),<br/>unlike tui which defaults to none (opt-IN); agents reach the corpus through<br/>/opt/agentbox/bin/vault and the Loom over HTTP since ontology-bridge was retired,<br/>agentbox.toml.schema.json:279-283"]
    VAULTCLIENTRY --- NOTE3
```

## AB-01.11 ADR-2080 model_routing.neural gate - pinned artefacts baked into the image (rebuild-class)
```mermaid
flowchart TB
    TOML2["[model_routing.neural] in agentbox.toml<br/>agentbox.toml:1387-1397<br/>enabled/provider/quality_bar/cost_ceiling_usd_per_mtok/privacy_tier/trajectory/assets_dir"] --> CFG2["agentboxConfig<br/>flake.nix:128"]
    CFG2 --> MRNCFG["modelRoutingNeuralCfg = (agentboxConfig.model_routing or {}).neural or {}<br/>flake.nix:136"]
    ARTJSON["config/model-router/artefacts.json<br/>hash-pinned files[] list"] --> MRNFETCH["modelRouterArtefacts = builtins.fromJSON ...<br/>flake.nix:137"]
    MRNFETCH --> MRNASSETS["modelRouterAssets = pkgs.runCommand ...<br/>per-file pkgs.fetchurl url+sha256, copied to $out/dest<br/>flake.nix:138-146"]

    MRNCFG -->|"modelRoutingNeuralCfg.enabled or false"| GATE{"lib.optionalString<br/>flake.nix:1893"}
    MRNASSETS --> GATE
    GATE -->|true| BAKE["mkdir $out/opt/agentbox/model-router<br/>cp -r modelRouterAssets/. into it<br/>flake.nix:1896-1898"]
    GATE -->|false| SKIP["byte-identical-when-off: nothing copied<br/>flake.nix comment line 1895"]
    BAKE --> ALLPKG2["mkImage layers<br/>flake.nix:3939"]

    subgraph MANIFEST2["system-manifest.js CATALOGUE"]
        MRNENTRY["id model-routing-neural<br/>gate model_routing.neural.enabled, apply_class rebuild<br/>system-manifest.js:109-111"]
    end
    TOML2 -.->|"enabled read at rebuild"| MRNENTRY

    subgraph SCHEMA2["schema/agentbox.toml.schema.json"]
        NEURALSCHEMA["model_routing.properties.neural<br/>additionalProperties false<br/>agentbox.toml.schema.json:1405-1407"]
    end
    TOML2 -.-> NEURALSCHEMA

    NOTE1["INVARIANT ADR-2080 - the npm @claude-flow/cli tarball ships no router<br/>artefacts (files list excludes assets/model-router), so ONE hash-verified<br/>manifest (config/model-router/artefacts.json) is the single source for both<br/>this Nix bake and scripts/model-router-fetch.sh's pre-rebuild fallback dir"]
    MRNASSETS --- NOTE1

    NOTE2["see AB-02.20 (aoe-seed-sessions.mjs WRAPPER_SLUGS.router),<br/>AB-05.11 (agentbox.sh model-router CLI), AB-29 (router console itself)"]
    BAKE --- NOTE2
```
