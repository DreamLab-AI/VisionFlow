---
id: AB-01
title: Nix flake composition and apply-class gates
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2006, ADR-2029, ADR-2039, ADR-2080, ADR-2151, ADR-2152]
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
  - ../project/agentbox/scripts/runtime-delivery.sh
  - ../project/agentbox/scripts/runtime-delivery.cjs
  - ../project/agentbox/scripts/nix-build-budget.cjs
  - ../project/agentbox/docs/adr/ADR-2151-voicelink-edge-voice.md
  - ../project/agentbox/lib/voice-paned.nix
verified_commit: 1da8d21719a27394f190866223b88533b52d87f9
---

## AB-01.1 agentbox.toml gates to flake.nix conditionals to package set and supervisord text

```mermaid
flowchart TB
    TOML["agentbox.toml<br/>flake.nix:186 builtins.fromTOML"] --> CFG["agentboxConfig"]
    CFG --> DESK["desktopCfg = agentboxConfig.desktop or {}<br/>flake.nix:208"]
    CFG --> MEDIA["mediaCfg = skillsCfg.media or {}<br/>flake.nix:293"]
    CFG --> VAULT["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:937"]

    DESK -->|"desktopCfg.enabled or false"| DESKOPT["lib.optionals<br/>flake.nix:1982 desktopPackages"]
    MEDIA -->|"mediaCfg.comfyui_builtin or false"| COMFYOPT["lib.optionals<br/>flake.nix:1471 comfyuiPackages"]
    CFG -->|"comfyui_builtin or integrations.comfyui_external.enabled"| COMFYMCP["comfyuiMcpPkg only<br/>flake.nix:920"]
    COMFYMCP --> ALLPKG
    CFG --> VOICELINK["voicelinkOn = voicelink.enabled or false<br/>flake.nix:1848, voicelinkPackages flake.nix:1850<br/>voice-paned linked at /opt/agentbox/bin flake.nix:2127"]
    VOICELINK --> ALLPKG
    MEDIA -->|"mediaCfg.ffmpeg or false"| FFOPT["lib.optionals<br/>flake.nix:1479 wrapGpuBin ffmpeg"]
    VAULT -->|"vaultCfg.tui == rune"| RUNEOPT["runeActive<br/>flake.nix:939"]

    DESKOPT --> ALLPKG["allPackages closure"]
    COMFYOPT --> ALLPKG
    FFOPT --> ALLPKG
    RUNEOPT -->|"lib.optionals runeActive"| RUNEPKG["runePackages<br/>flake.nix:941"]
    RUNEPKG --> ALLPKG

    DESK -->|"lib.optionalString (desktopCfg.enabled or false)"| DESKSUP["desktopBlocks text<br/>flake.nix:2528<br/>spliced flake.nix:2724"]
    MEDIA -->|"lib.optionalString (mediaCfg.comfyui_builtin or false)"| COMFYSUP["program:comfyui-builtin block<br/>flake.nix:2873-2882"]

    CFG --> SIDE["sidechainCfg = agentboxConfig.sidechain or {}<br/>flake.nix:330, enabled default false (:331)<br/>mirror and faucet only with enabled (:332-333)"]
    SIDE -->|"lib.optionals sidechainFaucet"| SIDEPKG["sidechainPackages = sidestr-agent<br/>flake.nix:1844"]
    SIDEPKG --> ALLPKG
    SIDE -->|"lib.optionalString sidechainEnabled / Mirror / Faucet"| SIDESUP["program:sidestr-producer, -mirror, -faucet blocks<br/>flake.nix:2104, :3127, :3145, :3162"]
    SIDESUP --> SUPTEXT
    CFG --> JEVON["jevCompactionOn = features.jev_compaction.enabled<br/>flake.nix:1866, ADR-2121"]
    JEVON -->|"lib.optionals jevCompactionOn"| FRPKG["factrailPackages<br/>flake.nix:1882"]
    FRPKG --> ALLPKG
    JEVON -->|"lib.optionalString jevCompactionOn"| FRBAKE["/opt/agentbox/bin/factrail symlink (flake.nix:2114)<br/>plugin copied to config/claude-plugins/factrail (flake.nix:2195)"]

    CFG --> SOON["systemOneProxyOn = system_one.enabled<br/>flake.nix:324, ADR-2139"]
    SOON -->|"lib.optionals systemOneProxyOn"| SOPKG["systemOneProxyPackages (flake.nix:1884)<br/>/opt/agentbox/bin/system-one-proxy link (flake.nix:2132)"]
    SOPKG --> ALLPKG
    SOON -->|"lib.optionalString systemOneProxyOn"| SOSUP["program:system-one-proxy<br/>flake.nix:3057"]
    SOSUP --> SUPTEXT
    CFG --> JKON["jevCheckpointsOn = features.jev_checkpoints.enabled<br/>flake.nix:1871, ADR-2135"]
    JKON -->|"config copy filter"| JKBAKE["plugin dir config/claude-plugins/jev-checkpoints dropped when the gate is off,<br/>its tests/ never shipped - flake.nix:2166-2170"]
    CFG --> SEOON["seoWatchEnabled = seo_watch.enabled (flake.nix:318)<br/>program:seo-watch-cron flake.nix:3084"]
    CFG --> TGCON["targetGcEnabled = resources.target_gc.enabled (flake.nix:253)<br/>program:target-gc flake.nix:3376"]
    SEOON --> SUPTEXT
    TGCON --> SUPTEXT

    DESKSUP --> SUPTEXT["supervisorText — AUTO-GENERATED supervisord.conf<br/>flake.nix:2620 supervisorText = header"]
    COMFYSUP --> SUPTEXT
    ALLPKG --> MKIMAGE["mkImage layers<br/>flake.nix:4611"]

    ALLPKG -.->|"see AB-01.6"| WRAPTGT["wrapped GPU targets"]

    note1["INVARIANT ADR-2003 - every gate touches package set AND supervisor text AND<br/>a system-manifest.js catalogue entry, management-api/lib/system-manifest.js line 39"]
    SUPTEXT --- note1
```

**Tension (ADR-2151 vs manifest):** ADR-2151 is `decision_status: proposed`, `implementation_status: none`, `activation_status: inactive` (`docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`), yet `[voicelink] enabled = true` (`agentbox.toml:2225`) makes `voicelinkOn` bake `voice-paned` into the package set and `/opt/agentbox/bin` (`flake.nix:1848-1850`, `flake.nix:2127`).
**Invariant:** the voicelink gate is the one rebuild-class gate that adds a package and a catalogue entry (`management-api/lib/system-manifest.js:184`) but no supervisor block, by design: `voice-paned` runs per ssh connection (`management-api/lib/system-manifest.js:186`).
**Open:** ADR-2152 (proposed, partial, staged) replaces the Nix Chromium-backed `mmdc` closure with a shell shim onto `/opt/agentbox/scripts/mmdc-sidecar.sh` when `docsCfg.mermaid` is on (`flake.nix:707-709`, `flake.nix:788`); which renderer finally owns Mermaid is not settled by the record.

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
        entries browser-sidecar 172, gui-tools-sidecar 175, voice-console 181, memory-hygiene 210
    end note
    note right of boot
        entries management-api 41, terminal 44, setup-wizard 47, vault root/pages/format 337, memory-learning 207
        newer boot entries claude-code-permissions 253, instruction-tiers 259,<br/>sovereign-system-one 277, skill-router 285, skill-router-cascade 288, routing-teacher-labels 291
        stateOf counts off and none as off, plus a per-entry off_values set, system-manifest.js lines 400-403
    end note
    note right of rebuild
        entries code-server 50, jupyter 53, desktop 56, comfyui 59, vault-tui 340
        newer rebuild entries claude-cred-sync 265, jev-compaction 268 (boot until ADR-2121), sidechain 322,<br/>vault-cli 343, mcp-hub 364, hook-shim 367, teammate-gc 370<br/>voicelink 184 is also rebuild-class (ADR-2151) but adds no supervisor block
        DOES NOT reconcile on restart, needs full Nix re-evaluation via agentbox.sh rebuild
    end note

    live --> [*]: does NOT reconcile at boot, only live reads
    boot --> [*]: does NOT re-evaluate Nix composition
    rebuild --> [*]: does NOT take effect on a plain restart
```

## AB-01.3 ./agentbox.sh rebuild - runtime-delivery prepare, then activate

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant REB as cmd_rebuild<br/>agentbox.sh:1084
    participant LOCK as runtime-delivery.sh<br/>flock lifecycle.lock :11-12
    participant RD as Delivery rebuild<br/>runtime-delivery.cjs:292-296
    participant PREP as Delivery.prepare<br/>runtime-delivery.cjs:152
    participant REG as Delivery.registry<br/>runtime-delivery.cjs:91
    participant ACT as Delivery.activate<br/>runtime-delivery.cjs:260
    participant DOCKER as docker

    OP->>REB: ./agentbox.sh rebuild [--prepare-only] [--delivery ...]
    REB->>LOCK: exec bash scripts/runtime-delivery.sh rebuild
    LOCK->>RD: flock -n .agentbox-build/lifecycle.lock then node runtime-delivery.cjs
    RD->>RD: parse --delivery registry|daemon|none, --prepare-only<br/>unknown option rejected, :279-287
    alt delivery none without --prepare-only
        RD-->>OP: error, :293
    end
    RD->>PREP: prepare(delivery) — the running box is untouched
    PREP->>PREP: mkdtemp generation dir, inspect running agentbox container<br/>record sourceCommit and dirty state, manifest checksum :156-162
    PREP->>PREP: bash scripts/refresh-compose.sh :163
    PREP->>PREP: nix build .#runtime .#runtime.copyTo --json,<br/>--max-jobs and --cores from nixBuildBudget() (60% of CPUs) :164-167<br/>nix-build-budget.cjs:5-20, NIX_CONFIG set at runtime-delivery.cjs:57-60
    PREP->>PREP: layer report vs previous candidate, duplicate store path = error :179-180
    alt delivery registry (default)
        PREP->>REG: loopback-only digest-pinned registry from config/build-registry.json :92-102
        REG->>DOCKER: start or create the pinned registry container, curl /v2 :112-124
        PREP->>DOCKER: copy-to docker://agentbox candidate tag, digestfile :190-191
        PREP->>DOCKER: docker pull image@sha256 digest :195-196
    else delivery daemon
        PREP->>DOCKER: copy-to docker-daemon:agentbox candidate tag :198-200
    end
    PREP->>DOCKER: inspect image, RootFS layers must equal the Nix image :203-207
    PREP->>DOCKER: offline no-network read-only smoke test of the candidate :127-136
    PREP->>PREP: quiescence - running container id, start time and manifest<br/>checksum must be unchanged or the candidate is NOT promoted :211-215
    PREP->>PREP: write generation receipt.json, candidate.json, previous.json<br/>release old generation store roots :216-236
    alt --prepare-only
        RD-->>OP: prepared, nothing deployed :294-295
    else activate
        RD->>ACT: activate()
        ACT->>ACT: validate() - candidate source, manifest checksum, configuration<br/>hash, image identity, running container identity and persistent<br/>mounts all unchanged since preparation :242-254
        ACT->>DOCKER: tag current image agentbox:recovery-<id12> :262
        ACT->>DOCKER: compose up -d --no-deps --force-recreate --pull never agentbox :263
        ACT->>DOCKER: verify activated image id and persistent mounts :264-266
        ACT->>ACT: poll GET 127.0.0.1:9090/ready, 60 retries x 2s :254-255
        ACT->>ACT: write .agentbox-build/active.json :269
    end
    Note over REB,ACT: RESHAPED (2026-10, HEAD 6466e393) - cmd_rebuild no longer runs cmd_down,<br/>cmd_build --variant runtime, cmd_up --build and post-deploy-cleanup.sh.<br/>The lifecycle is prepare (build + deliver + smoke, never touching the<br/>running box) then activate (validated single-service recreation).<br/>--no-cleanup is still accepted for compatibility but does nothing (:278)
    Note over PREP: persistent-mount changes are REFUSED at activate - validate() errors with<br/>use a separately reviewed migration, not fast activation (:255-257)
```

## AB-01.4 Static config validation - schema, semantic rules, then the skill-count gate

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator or CI
    participant WRAP as agentbox-config-validate.sh
    participant JS as agentbox-config-validate.js
    participant AJV as Ajv 2020 compiler<br/>agentbox-config-validate.js:127
    participant SCHEMA as agentbox.toml.schema.json
    participant SEM as semantic rule functions
    participant SKC as skill-count-check.js

    OP->>WRAP: ./scripts/agentbox-config-validate.sh agentbox.toml
    WRAP->>WRAP: probe node_modules/@iarna/toml, ajv<br/>agentbox-config-validate.sh:27-29
    opt node_modules missing
        WRAP->>WRAP: npm ci bootstrap or exit 2 if AGENTBOX_VALIDATOR_NO_BOOTSTRAP=1
    end
    WRAP->>JS: exec node agentbox-config-validate.js agentbox.toml
    JS->>JS: TOML.parse(raw)<br/>agentbox-config-validate.js:109
    alt TOML parse error
        JS-->>OP: emit E000, exit 1<br/>agentbox-config-validate.js:109-111
    end
    JS->>SCHEMA: JSON.parse schema file<br/>agentbox-config-validate.js:118
    JS->>AJV: ajv.compile(schema)<br/>agentbox-config-validate.js:127
    JS->>AJV: validate(manifest)<br/>agentbox-config-validate.js:128
    alt schemaValid == false
        AJV-->>JS: additionalProperty violations
        JS->>JS: push E016 UnknownManifestKey per error<br/>agentbox-config-validate.js:138-145
    end
    JS->>SEM: run E0xx/W0xx rule families<br/>adapters, providers, nostr relay, privacy filter, linked-data
    SEM-->>JS: errors[] and warnings[] arrays populated
    JS->>SKC: checkSkillCount repoRoot<br/>E-SKILL1, ADR-037 D8, agentbox-config-validate.js:1834-1835
    alt skills/*/SKILL.md count diverges from a README/SKILL-DIRECTORY claim
        SKC-->>JS: divergences[] -> push E016-sibling E-SKILL1 per drift<br/>agentbox-config-validate.js:1836-1841
    else checkSkillCount throws (e.g. skills/ absent)
        JS->>JS: push W067, RES-d gate skipped this pass<br/>agentbox-config-validate.js:1842-1847
    end
    JS->>JS: emit every warning to stderr<br/>agentbox-config-validate.js:1853-1855
    alt errors.length == 0
        JS-->>OP: stdout "agentbox manifest valid" + advisory count, exit 0<br/>agentbox-config-validate.js:1857-1860
    else errors present
        JS->>JS: emit every error to stderr<br/>agentbox-config-validate.js:1862-1864
        JS-->>OP: exit 1<br/>agentbox-config-validate.js:1865
    end

    Note over JS,SEM: DIVERGENCE - static-schema stage is advisory for W0xx dead-policy warnings,<br/>only E016/E-SKILL1 schema/RES-d violations and other E-code semantic rules hard-fail<br/>see agentbox-config-validate.js line 4 comment and lines 1853-1865 exit logic
    Note over SKC: see AB-22 for skill-count-check.js internals (skills/*/SKILL.md as the single count source)
```

**Tension:** two independent semantic blocks of the same validator emit the code `E080`: the ADR-2138 absolute-path check on `[resources.target_gc]` (`../project/agentbox/scripts/agentbox-config-validate.js:656`) and the six ADR-2139 `[system_one]` slot checks (`../project/agentbox/scripts/agentbox-config-validate.js:1582`), so the code alone does not identify the failing gate and only the message text does; the ADR-2139 block also documents `E081` and `W076`, while the ADR-2138 block owns `W077` (`../project/agentbox/scripts/agentbox-config-validate.js:666`).

## AB-01.5 npm-cli.nix pinned exact-semver closure - ruvector always in package set

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:485-486
    participant MK as makeNpmCli<br/>lib/npm-cli.nix:120
    participant FETCH as pkgs.fetchurl stage 1<br/>lib/npm-cli.nix:184
    participant FOD as packageWithDeps FOD stage 2<br/>lib/npm-cli.nix:204
    participant WRAP as wrapper derivation stage 3
    participant ALWAYS as npmCliAlwaysPackages<br/>flake.nix:774

    FLAKE->>MK: mkNpmCli pkgName=ruvector version=0.3.3<br/>flake.nix:490-497
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
    FLAKE->>ALWAYS: npmCliAlwaysPackages = [ ruvectorPkg wranglerPkg ]<br/>flake.nix:774
    Note over FLAKE,ALWAYS: the nix-prefetch-url comment at flake.nix:489 and the version field<br/>at flake.nix:492 agree on ruvector-0.3.3 (bumped together 2026-10-01)
    Note over FLAKE,ALWAYS: DRIFT: BASELINE-container.md:61 says the nix-prefetch-url comment above the pin<br/>still names the old tarball, but flake.nix:489 names ruvector-0.3.3 like the pin at flake.nix:492 -<br/>and BASELINE-container.md:59 gives flake.nix as 4,684 lines while it is 4,774 at this stamp
```

## AB-01.6 gpu-wrap.nix wrapGpuBins - LD_LIBRARY_PATH suffix and vendor ICDs

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:462-464
    participant WRAP as gpuWrap.wrapGpuBins<br/>lib/gpu-wrap.nix:76
    participant JOIN as pkgs.symlinkJoin<br/>lib/gpu-wrap.nix:77
    participant MAKEW as makeWrapper wrapProgram<br/>lib/gpu-wrap.nix:87
    participant BIN as wrapped binary at runtime

    FLAKE->>FLAKE: gpuActive = agentbox.toml gpu.backend == local-cuda<br/>flake.nix:462
    alt gpu.backend == none
        FLAKE->>FLAKE: wrapGpuBin pkg bins = pkg, unwrapped passthrough<br/>flake.nix:463-466
        Note over FLAKE: alt gpu.backend=none - wrapping is inert without injected driver libs, gpu-wrap.nix comment lines 21-22
    else gpu.backend == local-cuda
        FLAKE->>WRAP: wrapGpuBins pkg=pkgs.blender bins=[blender]<br/>flake.nix:1497
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
    Note over MAKEW: DIVERGENCE - BASELINE GPU scope and evidence qualification 2026-09-04 -<br/>current wrappers include GLX/EGL/Vulkan defaults alongside CUDA library-path config,<br/>so ADR-2006's graphics review trigger has been reached, agentbox/docs/BASELINE-container.md lines 307-309

    Note over WRAP: wrapped-target gates - ffmpeg mediaCfg.ffmpeg flake.nix:1479,<br/>qgis spatialCfg.qgis flake.nix:1485, blender spatialCfg.blender AND spatialCfg.blender_local (default true)<br/>flake.nix:1493 - the blender-mcp proxy block stays gated on blender alone, flake.nix:2722,<br/>3DGS spatialCfg.gaussian_splatting map wrapGpuAll gauss3dPackages flake.nix:1503<br/>(see AB-27 for the fuller MCP-proxy treatment of qgis/blender/ffmpeg)
    Note over WRAP: INVARIANT - wrapGpuBin names exact bins, wrapGpuAll wraps every<br/>executable under out/bin for upstream-versioned bin sets - colmap and lichtfeld<br/>(both CUDA) need the wrapper, metis is CPU-only so wrapping is inert for it,<br/>flake.nix comment lines 1499-1502
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

    note for GpuSection "backend enum: none, ollama-rocm, ollama-cuda, local-cuda - agentbox.toml.schema.json:108-117"
    note for VaultSection "required: root - agentbox.toml.schema.json:226-228. format enum obsidian ONLY (logseq-legacy withdrawn 2026-09-22, corpus migration complete) - agentbox.toml.schema.json:253-260. tui enum rune, none, default none - agentbox.toml.schema.json:275-283, ADR-2029. cli bool default true - agentbox.toml.schema.json:284-304, ADR-2107/ADR-2108"
    note for AdaptersSection "each value resolves to local-star, external, or off per slot - agentbox/CLAUDE.md adapter contract"
```

## AB-01.9 flake.lock inputs to flake outputs

```mermaid
flowchart LR
    subgraph INPUTS["flake.lock root inputs"]
        NIXPKGS["nixpkgs - nixpkgs_3<br/>github NixOS/nixpkgs pin 02c087f3fe67 (flake.lock)<br/>flake.nix:5"]
        FU["flake-utils<br/>github numtide/flake-utils<br/>flake.nix:6"]
        N2C["nix2container<br/>github nlewo/nix2container<br/>flake.nix:7"]
        RO["rust-overlay<br/>github oxalica/rust-overlay<br/>flake.nix:8"]
        AOE["aoe<br/>github DreamLab-AI/agentbox-of-empires pin 33e806ad<br/>flake.nix:18"]
        SKILLS["skills - path:./skills, flake=false<br/>flake.nix:25-28"]
        VAULTSRC["vaultSrc - github DreamLab-AI/VisionClaw pin e2d4e2cc1314<br/>flake=false, flake.nix:41-44"]
        CODEX["codexPlugin<br/>github openai/codex-plugin-cc pin db52e28f<br/>flake.nix:52-55"]
    end

    NIXPKGS --> EVAL["flake outputs eval, per-system<br/>flake.nix:72-78 flake-utils.lib.eachSystem"]
    FU --> EVAL
    N2C --> EVAL
    RO --> EVAL
    AOE --> EVAL
    SKILLS --> EVAL
    VAULTSRC --> EVAL
    CODEX --> EVAL
    RUFLO["rufloConsole<br/>github ruvnet/ruflo pin 631c1557<br/>flake=false, flake.nix:66-69"]
    RUFLO --> EVAL

    EVAL --> PACKAGES["packages - flake.nix:4670<br/>lib.optionalAttrs pkgs.stdenv.isLinux"]
    EVAL --> DEVSHELL["devShells.default<br/>flake.nix:4747"]

    PACKAGES --> RUNTIME["runtime = mkImage tag runtime-system<br/>flake.nix:4675"]
    PACKAGES --> FULL["full = mkImage extraPackages allPackages<br/>flake.nix:4676-4679"]
    PACKAGES --> DESKTOP["desktop = mkImage extraPackages desktopPackages<br/>flake.nix:4680-4683"]
    PACKAGES --> CUDART["cuda-runtime adds explicitly dispatched local-cuda packages<br/>flake.nix:4701-4709"]
    PACKAGES --> GSPLAT["gaussian-splatting = 3DGS stack over cuda-runtime<br/>flake.nix:4723-4733"]
    PACKAGES --> SECTOOLS["security-tools and agentbox-vex, host-only supply-chain tooling (ADR-2137)<br/>flake.nix:4672-4674, built from securityTools flake.nix:178"]
    MKIMG2["each mkImage result also carries securityEvidence (SBOM and VEX inputs)<br/>flake.nix:4643"] -.-> SECTOOLS
    PACKAGES --> COMPOSE["compose = docker-compose.yml text, cross-platform<br/>flake.nix:4741-4744"]

    RUNTIME --> MKIMG["mkImage - n2c.buildImage 4 layers, platform toolchains agent-clis services<br/>flake.nix:4591-4616"]
    FULL --> MKIMG
    DESKTOP --> MKIMG

    MKIMG --> ENTRYPOINT["config = Entrypoint entrypoint/bin/entrypoint<br/>flake.nix:4627"]
    VAULTSRC -.->|"see AB-01.10"| VAULTPKG["vaultPkg = import lib/vault.nix"]

    NOTE1["INVARIANT - container-image outputs are Linux-only,<br/>darwin exposes only compose and devShells, flake.nix comment lines 4666-4669"]
    PACKAGES --- NOTE1
```

## AB-01.10 ADR-2029/ADR-2107 vault gates - three catalogue entries, one section

```mermaid
flowchart TB
    TOMLVAULT["[vault] in agentbox.toml<br/>agentbox.toml:1125<br/>tui = rune (:1133), cli = true (:1140)"] --> VAULTCFG["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:937"]

    VAULTCFG --> TUIVAL["vaultTui = vaultCfg.tui or none<br/>flake.nix:938"]
    TUIVAL --> RUNEACTIVE["runeActive = vaultTui == rune<br/>flake.nix:939"]
    RUNEACTIVE -->|"true"| RUNEPKGIMPORT["runePkg = import lib/rune.nix<br/>flake.nix:940, lazy import"]
    RUNEACTIVE -->|"lib.optionals runeActive"| RUNEPACKAGES["runePackages<br/>flake.nix:941"]
    RUNEPKGIMPORT --> RUNEPACKAGES

    VAULTCFG --> CLIVAL["vaultCliActive = root != '' and (vaultCfg.cli or true)<br/>flake.nix:962 - defaults ON, unlike tui"]
    CLIVAL -->|"true"| VAULTPKGIMPORT["vaultPkg = import lib/vault.nix src=vaultSrc<br/>flake.nix:963-967"]
    CLIVAL -->|"lib.optionals vaultCliActive"| VAULTPACKAGES["vaultPackages = vaultPkg + ontomarkPkg<br/>flake.nix:975-976"]
    VAULTPKGIMPORT --> VAULTPACKAGES

    RUNEPACKAGES --> ALLPKG["allPackages closure - Nix image composition"]
    VAULTPACKAGES --> ALLPKG

    RUNEBUILD["rune.nix pkgs.rustPlatform.buildRustPackage<br/>fetchFromGitHub jjohare/rune (DreamLab fork) tag v1.5.0-dreamlab.1<br/>lib/rune.nix:26-40, cargoBuildFlags -p rune-cli, doCheck=false<br/>cargoLock is the vendored lib/rune-Cargo.lock, not the source's (rune.nix:38),<br/>so nix flake check --no-build needs no import-from-derivation<br/>adds callouts, highlights, tags, backlinks panel, daily notes"]
    RUNEPKGIMPORT --> RUNEBUILD

    subgraph MANIFEST["system-manifest.js CATALOGUE - management-api/lib/system-manifest.js"]
        VAULTENTRY["id vault<br/>gate vault.format, apply_class boot<br/>line 337-339"]
        VAULTTUIENTRY["id vault-tui<br/>gate vault.tui, apply_class rebuild<br/>line 340-342"]
        VAULTCLIENTRY["id vault-cli<br/>gate vault.cli, apply_class rebuild<br/>line 343-345<br/>ADR-2107/ADR-2108, the one door to the corpus"]
    end

    TOMLVAULT -.->|"vault.format read at boot"| VAULTENTRY
    TOMLVAULT -.->|"vault.tui read at rebuild"| VAULTTUIENTRY
    TOMLVAULT -.->|"vault.cli read at rebuild"| VAULTCLIENTRY

    STATEOF["stateOf function<br/>system-manifest.js:393-420<br/>a false parent gate now dominates child gates (:397-398)"]
    VAULTTUIENTRY --> STATEOF
    STATEOF --> MODESTRING["off and none count as off, plus a per-entry off_values set<br/>skill-router declares table as its off value<br/>system-manifest.js:415-418, entry at system-manifest.js:285"]

    NOTE1["DOC-DRIFT - the code comment at system-manifest.js:330-336 still says the<br/>split is into TWO catalogue entries (ADR-2020), but [vault] now has THREE:<br/>root/pages/format are boot-class, tui and cli are both rebuild-class but gate<br/>different binaries (Rune TUI vs the vault CLI door)"]
    VAULTTUIENTRY --- NOTE1

    NOTE2["DIVERGENCE - vault-tui and vault-cli are REBUILD-class, flipping either key and<br/>restarting the container is NOT sufficient - the Rune binary or the vault CLI is<br/>absent from the package set until agentbox.sh rebuild runs"]
    STATEOF --- NOTE2

    NOTE3["INVARIANT ADR-2107/ADR-2108 - cli defaults to true whenever root is set (opt-OUT),<br/>unlike tui which defaults to none (opt-IN); agents reach the corpus through<br/>/opt/agentbox/bin/vault and the Loom over HTTP since ontology-bridge was retired,<br/>agentbox.toml.schema.json:284-304"]
    VAULTCLIENTRY --- NOTE3
```

## AB-01.11 ADR-2080 model_routing.neural gate - pinned artefacts baked into the image (rebuild-class)
```mermaid
flowchart TB
    TOML2["[model_routing.neural] in agentbox.toml<br/>agentbox.toml:1709-1719<br/>enabled/provider/quality_bar/cost_ceiling_usd_per_mtok/privacy_tier/trajectory/assets_dir"] --> CFG2["agentboxConfig<br/>flake.nix:186"]
    CFG2 --> MRNCFG["modelRoutingNeuralCfg = (agentboxConfig.model_routing or {}).neural or {}<br/>flake.nix:194"]
    ARTJSON["config/model-router/artefacts.json<br/>hash-pinned files[] list"] --> MRNFETCH["modelRouterArtefacts = builtins.fromJSON ...<br/>flake.nix:195"]
    MRNFETCH --> MRNASSETS["modelRouterAssets = pkgs.runCommand ...<br/>per-file pkgs.fetchurl url+sha256, copied to $out/dest<br/>flake.nix:196-204"]

    MRNCFG -->|"modelRoutingNeuralCfg.enabled or false"| GATE{"lib.optionalString<br/>flake.nix:2241"}
    MRNASSETS --> GATE
    GATE -->|true| BAKE["mkdir $out/opt/agentbox/model-router<br/>cp -r modelRouterAssets/. into it<br/>flake.nix:2244-2246"]
    GATE -->|false| SKIP["byte-identical-when-off: nothing copied<br/>flake.nix comment line 2243"]
    BAKE --> ALLPKG2["mkImage layers<br/>flake.nix:4611"]

    subgraph MANIFEST2["system-manifest.js CATALOGUE"]
        MRNENTRY["id model-routing-neural<br/>gate model_routing.neural.enabled, apply_class rebuild<br/>system-manifest.js:118-120"]
    end
    TOML2 -.->|"enabled read at rebuild"| MRNENTRY

    subgraph SCHEMA2["schema/agentbox.toml.schema.json"]
        NEURALSCHEMA["model_routing.properties.neural<br/>additionalProperties false<br/>agentbox.toml.schema.json:1604-1606"]
    end
    TOML2 -.-> NEURALSCHEMA

    NOTE1["INVARIANT ADR-2080 - the npm @claude-flow/cli tarball ships no router<br/>artefacts (files list excludes assets/model-router), so ONE hash-verified<br/>manifest (config/model-router/artefacts.json) is the single source for both<br/>this Nix bake and scripts/model-router-fetch.sh's pre-rebuild fallback dir"]
    MRNASSETS --- NOTE1

    NOTE2["see AB-02.20 (aoe-seed-sessions.mjs WRAPPER_SLUGS.router),<br/>AB-05.11 (agentbox.sh model-router CLI), AB-29 (router console itself)"]
    BAKE --- NOTE2
```
