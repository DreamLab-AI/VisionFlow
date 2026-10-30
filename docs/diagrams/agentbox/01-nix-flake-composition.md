---
id: AB-01
title: Nix flake composition and apply-class gates
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2003, ADR-2006, ADR-2029, ADR-2039, ADR-2080, ADR-2151, ADR-2152, ADR-2154, ADR-2155]
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
  - ../project/agentbox/lib/pele.nix
  - ../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md
  - ../project/agentbox/docs/adr/ADR-2154-pele-renders-mermaid-in-the-image.md
verified_commit: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc
---

## AB-01.1 agentbox.toml gates to flake.nix conditionals to package set and supervisord text

```mermaid
flowchart TB
    TOML["agentbox.toml<br/>flake.nix:176 builtins.fromTOML"] --> CFG["agentboxConfig"]
    CFG --> DESK["desktopCfg = agentboxConfig.desktop or {}<br/>flake.nix:198"]
    CFG --> MEDIA["mediaCfg = skillsCfg.media or {}<br/>flake.nix:283"]
    CFG --> VAULT["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:916"]

    DESK -->|"desktopCfg.enabled or false"| DESKOPT["lib.optionals<br/>flake.nix:1981 desktopPackages"]
    MEDIA -->|"mediaCfg.comfyui_builtin or false"| COMFYOPT["lib.optionals<br/>flake.nix:1454 comfyuiPackages"]
    CFG -->|"comfyui_builtin or integrations.comfyui_external.enabled"| COMFYMCP["comfyuiMcpPkg only<br/>flake.nix:899"]
    COMFYMCP --> ALLPKG
    CFG --> VOICELINK["voicelinkOn = voicelink.enabled or false<br/>flake.nix:1847, voicelinkPackages flake.nix:1849<br/>voice-paned linked at /opt/agentbox/bin flake.nix:2128"]
    VOICELINK --> ALLPKG
    MEDIA -->|"mediaCfg.ffmpeg or false"| FFOPT["lib.optionals<br/>flake.nix:1462 wrapGpuBin ffmpeg"]
    VAULT -->|"vaultCfg.tui == rune"| RUNEOPT["runeActive<br/>flake.nix:918"]

    DESKOPT --> ALLPKG["allPackages closure"]
    COMFYOPT --> ALLPKG
    FFOPT --> ALLPKG
    RUNEOPT -->|"lib.optionals runeActive"| RUNEPKG["runePackages<br/>flake.nix:920"]
    RUNEPKG --> ALLPKG

    DESK -->|"lib.optionalString (desktopCfg.enabled or false)"| DESKSUP["desktopBlocks text<br/>flake.nix:2529<br/>spliced flake.nix:2725"]
    MEDIA -->|"lib.optionalString (mediaCfg.comfyui_builtin or false)"| COMFYSUP["program:comfyui-builtin block<br/>flake.nix:2874-2883"]

    CFG --> SIDE["sidechainCfg = agentboxConfig.sidechain or {}<br/>flake.nix:320, enabled default false (:321)<br/>mirror and faucet only with enabled (:322-323)"]
    SIDE -->|"lib.optionals sidechainFaucet"| SIDEPKG["sidechainPackages = sidestr-agent<br/>flake.nix:1843"]
    SIDEPKG --> ALLPKG
    SIDE -->|"lib.optionalString sidechainEnabled / Mirror / Faucet"| SIDESUP["program:sidestr-producer, -mirror, -faucet blocks<br/>flake.nix:2105, :3105, :3123, :3140"]
    SIDESUP --> SUPTEXT
    CFG --> JEVON["jevCompactionOn = features.jev_compaction.enabled<br/>flake.nix:1865, ADR-2121"]
    JEVON -->|"lib.optionals jevCompactionOn"| FRPKG["factrailPackages<br/>flake.nix:1881"]
    FRPKG --> ALLPKG
    JEVON -->|"lib.optionalString jevCompactionOn"| FRBAKE["/opt/agentbox/bin/factrail symlink (flake.nix:2115)<br/>plugin copied to config/claude-plugins/factrail (flake.nix:2196)"]

    CFG --> SOON["systemOneProxyOn = system_one.enabled<br/>flake.nix:314, ADR-2139"]
    SOON -->|"lib.optionals systemOneProxyOn"| SOPKG["systemOneProxyPackages (flake.nix:1883)<br/>/opt/agentbox/bin/system-one-proxy link (flake.nix:2133)<br/>rebuild-class consumers add jev: flake.nix:946, :957, :1537"]
    SOPKG --> ALLPKG
    SOON -->|"lib.optionalString systemOneProxyOn"| SOSUP["program:system-one-proxy<br/>flake.nix:3035"]
    SOSUP --> SUPTEXT
    CFG --> JKON["jevCheckpointsOn = features.jev_checkpoints.enabled<br/>flake.nix:1870, ADR-2135"]
    JKON -->|"config copy filter"| JKBAKE["plugin dir config/claude-plugins/jev-checkpoints dropped when the gate is off,<br/>its tests/ never shipped - flake.nix:2167-2171"]
    CFG --> SEOON["seoWatchEnabled = seo_watch.enabled (flake.nix:308)<br/>program:seo-watch-cron flake.nix:3062"]
    CFG --> TGCON["targetGcEnabled = resources.target_gc.enabled (flake.nix:243)<br/>program:target-gc flake.nix:3354"]
    CFG --> IPON["interactionPlaneEnabled flake.nix:418, ADR-2153<br/>agentboxSessionsPkg flake.nix:2057, link :2098"]
    IPON --> ALLPKG
    IPON -->|"program:agentbox-sessions flake.nix:2910"| SUPTEXT
    SEOON --> SUPTEXT
    TGCON --> SUPTEXT

    DESKSUP --> SUPTEXT["supervisorText — AUTO-GENERATED supervisord.conf<br/>flake.nix:2621 supervisorText = header"]
    COMFYSUP --> SUPTEXT
    ALLPKG --> MKIMAGE["mkImage layers<br/>flake.nix:4589"]

    ALLPKG -.->|"see AB-01.6"| WRAPTGT["wrapped GPU targets"]

    note1["INVARIANT ADR-2003 - every gate touches package set AND supervisor text AND<br/>a system-manifest.js catalogue entry, management-api/lib/system-manifest.js line 39"]
    SUPTEXT --- note1
```

**Tension (ADR-2151 vs manifest):** ADR-2151 is `decision_status: proposed`, `implementation_status: none`, `activation_status: inactive` (`docs/adr/ADR-2151-voicelink-edge-voice.md:5-7`), yet `[voicelink] enabled = true` (`agentbox.toml:2226`) makes `voicelinkOn` bake `voice-paned` into the package set and `/opt/agentbox/bin` (`flake.nix:1847-1849`, `flake.nix:2128`).
**Invariant:** the voicelink gate is the one rebuild-class gate that adds a package and a catalogue entry (`management-api/lib/system-manifest.js:184`) but no supervisor block, by design: `voice-paned` runs per ssh connection (`management-api/lib/system-manifest.js:186`).
**Open:** ADR-2154 (accepted, partial, staged; supersedes ADR-2152 and its browsercontainer `mmdc` shim) puts the pinned Pele renderer and an `mmdc` compatibility shim into the package set when `docsCfg.mermaid` is on (`flake.nix:688`, `flake.nix:767`, `lib/pele.nix:61`, `lib/pele.nix:67`); the record's activation status is still staged. ADR-2155 (`accepted`, `complete`, `staged`, `../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md:5-7`) likewise records the System One consumer activation as staged while `[system_one.consumers]` already sets all nineteen true, three of them rebuild-class (`../project/agentbox/agentbox.toml:902-923`).

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
        entries management-api 41, terminal 44, setup-wizard 47, vault root/pages/format 397, memory-learning 207
        newer boot entries claude-code-permissions 253, instruction-tiers 259,<br/>sovereign-system-one 280, system-one-network 277 (2026-10-30), skill-router 345, skill-router-cascade 348, routing-teacher-labels 351,<br/>sixteen boot system-one-consumer-* entries from failure-taxonomy 288 (requires system_one.enabled)
        stateOf counts off and none as off, plus a per-entry off_values set, system-manifest.js lines 475-478
    end note
    note right of rebuild
        entries code-server 50, jupyter 53, desktop 56, comfyui 59, interaction-plane 74 and agentbox-sessions-binary 124 (ADR-2153, interaction-plane was boot), vault-tui 400
        newer rebuild entries claude-cred-sync 265, jev-compaction 268 (boot until ADR-2121), sidechain 382,<br/>vault-cli 403, mcp-hub 424, hook-shim 427, teammate-gc 430,<br/>consumers solid-pod-advisories 336, ontomark-suggestions 339, vault-rank 342 (build Jev into the package)<br/>voicelink 184 is also rebuild-class (ADR-2151) but adds no supervisor block
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
    JS->>SKC: checkSkillCount repoRoot<br/>E-SKILL1, ADR-037 D8, agentbox-config-validate.js:1849-1850
    alt skills/*/SKILL.md count diverges from a README/SKILL-DIRECTORY claim
        SKC-->>JS: divergences[] -> push E016-sibling E-SKILL1 per drift<br/>agentbox-config-validate.js:1851-1856
    else checkSkillCount throws (e.g. skills/ absent)
        JS->>JS: push W067, RES-d gate skipped this pass<br/>agentbox-config-validate.js:1857-1862
    end
    JS->>JS: emit every warning to stderr<br/>agentbox-config-validate.js:1868-1870
    alt errors.length == 0
        JS-->>OP: stdout "agentbox manifest valid" + advisory count, exit 0<br/>agentbox-config-validate.js:1872-1875
    else errors present
        JS->>JS: emit every error to stderr<br/>agentbox-config-validate.js:1877-1879
        JS-->>OP: exit 1<br/>agentbox-config-validate.js:1880
    end

    Note over JS,SEM: DIVERGENCE - static-schema stage is advisory for W0xx dead-policy warnings,<br/>only E016/E-SKILL1 schema/RES-d violations and other E-code semantic rules hard-fail<br/>see agentbox-config-validate.js line 4 comment and lines 1862-1874 exit logic
    Note over SKC: see AB-22 for skill-count-check.js internals (skills/*/SKILL.md as the single count source)
```

**Tension:** two independent semantic blocks of the same validator emit the code `E080`: the ADR-2138 absolute-path check on `[resources.target_gc]` (`../project/agentbox/scripts/agentbox-config-validate.js:656`) and the eight ADR-2139 `[system_one]` slot checks, now including `local_backend`, the bind check that admits a private address only with `network = true` (2026-10-30) and boolean `consumers` (`../project/agentbox/scripts/agentbox-config-validate.js:1582`, `:1589`, `:1611`, `:1615`), so the code alone does not identify the failing gate and only the message text does; the ADR-2139 block also documents `E081` and `W076`, while the ADR-2138 block owns `W077` (`../project/agentbox/scripts/agentbox-config-validate.js:656`).

## AB-01.5 npm-cli.nix pinned exact-semver closure - ruvector always in package set

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:466-467
    participant MK as makeNpmCli<br/>lib/npm-cli.nix:120
    participant FETCH as pkgs.fetchurl stage 1<br/>lib/npm-cli.nix:184
    participant FOD as packageWithDeps FOD stage 2<br/>lib/npm-cli.nix:204
    participant WRAP as wrapper derivation stage 3
    participant ALWAYS as npmCliAlwaysPackages<br/>flake.nix:753

    FLAKE->>MK: mkNpmCli pkgName=ruvector version=0.3.3<br/>flake.nix:471-478
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
    FLAKE->>ALWAYS: npmCliAlwaysPackages = [ ruvectorPkg wranglerPkg ]<br/>flake.nix:753
    Note over FLAKE,ALWAYS: the nix-prefetch-url comment at flake.nix:470 and the version field<br/>at flake.nix:473 agree on ruvector-0.3.3 (bumped together 2026-10-01)
    Note over FLAKE,ALWAYS: DRIFT: BASELINE-container.md:63 says the nix-prefetch-url comment above the pin<br/>still names the old tarball, but flake.nix:470 names ruvector-0.3.3 like the pin at flake.nix:473 -<br/>and BASELINE-container.md:61 gives flake.nix as 4,684 lines while it is 4,774 at this stamp
```

## AB-01.6 gpu-wrap.nix wrapGpuBins - LD_LIBRARY_PATH suffix and vendor ICDs

```mermaid
sequenceDiagram
    autonumber
    participant FLAKE as flake.nix eval<br/>flake.nix:443-445
    participant WRAP as gpuWrap.wrapGpuBins<br/>lib/gpu-wrap.nix:76
    participant JOIN as pkgs.symlinkJoin<br/>lib/gpu-wrap.nix:77
    participant MAKEW as makeWrapper wrapProgram<br/>lib/gpu-wrap.nix:87
    participant BIN as wrapped binary at runtime

    FLAKE->>FLAKE: gpuActive = agentbox.toml gpu.backend == local-cuda<br/>flake.nix:443
    alt gpu.backend == none
        FLAKE->>FLAKE: wrapGpuBin pkg bins = pkg, unwrapped passthrough<br/>flake.nix:444-447
        Note over FLAKE: alt gpu.backend=none - wrapping is inert without injected driver libs, gpu-wrap.nix comment lines 21-22
    else gpu.backend == local-cuda
        FLAKE->>WRAP: wrapGpuBins pkg=pkgs.blender bins=[blender]<br/>flake.nix:1480
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

    Note over WRAP: wrapped-target gates - ffmpeg mediaCfg.ffmpeg flake.nix:1462,<br/>qgis spatialCfg.qgis flake.nix:1468, blender spatialCfg.blender AND spatialCfg.blender_local (default true)<br/>flake.nix:1476 - the blender-mcp proxy block stays gated on blender alone, flake.nix:2723,<br/>3DGS spatialCfg.gaussian_splatting map wrapGpuAll gauss3dPackages flake.nix:1486<br/>(see AB-27 for the fuller MCP-proxy treatment of qgis/blender/ffmpeg)
    Note over WRAP: INVARIANT - wrapGpuBin names exact bins, wrapGpuAll wraps every<br/>executable under out/bin for upstream-versioned bin sets - colmap and lichtfeld<br/>(both CUDA) need the wrapper, metis is CPU-only so wrapping is inert for it,<br/>flake.nix comment lines 1483-1485
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
    note for VaultSection "required: root - agentbox.toml.schema.json:226-228. format enum obsidian ONLY (logseq-legacy withdrawn 2026-09-22, corpus migration complete) - agentbox.toml.schema.json:253-260. tui enum rune, none, default none - agentbox.toml.schema.json:275-283, ADR-2029. cli bool default true - agentbox.toml.schema.json:284-308, ADR-2107/ADR-2108"
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
        SKILLS["skills - path:./skills, flake=false<br/>flake.nix:15-18"]
        VAULTSRC["vaultSrc - github DreamLab-AI/VisionClaw pin f3bbb430ea00<br/>flake=false, flake.nix:31-34"]
        CODEX["codexPlugin<br/>github openai/codex-plugin-cc pin db52e28f<br/>flake.nix:42-45"]
    end

    NIXPKGS --> EVAL["flake outputs eval, per-system<br/>flake.nix:62-69 flake-utils.lib.eachSystem"]
    FU --> EVAL
    N2C --> EVAL
    RO --> EVAL
    SKILLS --> EVAL
    VAULTSRC --> EVAL
    CODEX --> EVAL
    RUFLO["rufloConsole<br/>github ruvnet/ruflo pin 631c1557<br/>flake=false, flake.nix:56-59"]
    RUFLO --> EVAL

    EVAL --> PACKAGES["packages - flake.nix:4648<br/>lib.optionalAttrs pkgs.stdenv.isLinux"]
    EVAL --> DEVSHELL["devShells.default<br/>flake.nix:4726"]

    PACKAGES --> RUNTIME["runtime = mkImage tag runtime-system<br/>flake.nix:4654"]
    PACKAGES --> FULL["full = mkImage extraPackages allPackages<br/>flake.nix:4655-4658"]
    PACKAGES --> DESKTOP["desktop = mkImage extraPackages desktopPackages<br/>flake.nix:4659-4662"]
    PACKAGES --> CUDART["cuda-runtime adds explicitly dispatched local-cuda packages<br/>flake.nix:4680-4688"]
    PACKAGES --> GSPLAT["gaussian-splatting = 3DGS stack over cuda-runtime<br/>flake.nix:4702-4712"]
    PACKAGES --> SECTOOLS["security-tools and agentbox-vex, host-only supply-chain tooling (ADR-2137)<br/>flake.nix:4650-4653, built from securityTools flake.nix:168"]
    MKIMG2["each mkImage result also carries securityEvidence (SBOM and VEX inputs)<br/>flake.nix:4621"] -.-> SECTOOLS
    PACKAGES --> COMPOSE["compose = docker-compose.yml text, cross-platform<br/>flake.nix:4720-4723"]

    RUNTIME --> MKIMG["mkImage - n2c.buildImage 4 layers, platform toolchains agent-clis services<br/>flake.nix:4569-4594"]
    FULL --> MKIMG
    DESKTOP --> MKIMG

    MKIMG --> ENTRYPOINT["config = Entrypoint entrypoint/bin/entrypoint<br/>flake.nix:4605"]
    VAULTSRC -.->|"see AB-01.10"| VAULTPKG["vaultPkg = import lib/vault.nix"]

    NOTE1["INVARIANT - container-image outputs are Linux-only,<br/>darwin exposes only compose and devShells, flake.nix comment lines 4644-4647"]
    PACKAGES --- NOTE1
```

## AB-01.10 ADR-2029/ADR-2107 vault gates - three catalogue entries, one section

```mermaid
flowchart TB
    TOMLVAULT["[vault] in agentbox.toml<br/>agentbox.toml:1152<br/>tui = rune (:1160), cli = true (:1167)"] --> VAULTCFG["vaultCfg = agentboxConfig.vault or {}<br/>flake.nix:916"]

    VAULTCFG --> TUIVAL["vaultTui = vaultCfg.tui or none<br/>flake.nix:917"]
    TUIVAL --> RUNEACTIVE["runeActive = vaultTui == rune<br/>flake.nix:918"]
    RUNEACTIVE -->|"true"| RUNEPKGIMPORT["runePkg = import lib/rune.nix<br/>flake.nix:919, lazy import"]
    RUNEACTIVE -->|"lib.optionals runeActive"| RUNEPACKAGES["runePackages<br/>flake.nix:920"]
    RUNEPKGIMPORT --> RUNEPACKAGES

    VAULTCFG --> CLIVAL["vaultCliActive = root != '' and (vaultCfg.cli or true)<br/>flake.nix:941 - defaults ON, unlike tui"]
    CLIVAL -->|"true"| VAULTPKGIMPORT["vaultPkg = import lib/vault.nix src=vaultSrc<br/>flake.nix:942-947"]
    CLIVAL -->|"lib.optionals vaultCliActive"| VAULTPACKAGES["vaultPackages = vaultPkg + ontomarkPkg<br/>flake.nix:955-959"]
    VAULTPKGIMPORT --> VAULTPACKAGES

    RUNEPACKAGES --> ALLPKG["allPackages closure - Nix image composition"]
    VAULTPACKAGES --> ALLPKG

    RUNEBUILD["rune.nix pkgs.rustPlatform.buildRustPackage<br/>fetchFromGitHub jjohare/rune (DreamLab fork) tag v1.5.0-dreamlab.1<br/>lib/rune.nix:26-40, cargoBuildFlags -p rune-cli, doCheck=false<br/>cargoLock is the vendored lib/rune-Cargo.lock, not the source's (rune.nix:38),<br/>so nix flake check --no-build needs no import-from-derivation<br/>adds callouts, highlights, tags, backlinks panel, daily notes"]
    RUNEPKGIMPORT --> RUNEBUILD

    subgraph MANIFEST["system-manifest.js CATALOGUE - management-api/lib/system-manifest.js"]
        VAULTENTRY["id vault<br/>gate vault.format, apply_class boot<br/>line 394-396"]
        VAULTTUIENTRY["id vault-tui<br/>gate vault.tui, apply_class rebuild<br/>line 397-399"]
        VAULTCLIENTRY["id vault-cli<br/>gate vault.cli, apply_class rebuild<br/>line 403-405<br/>ADR-2107/ADR-2108, the one door to the corpus"]
    end

    TOMLVAULT -.->|"vault.format read at boot"| VAULTENTRY
    TOMLVAULT -.->|"vault.tui read at rebuild"| VAULTTUIENTRY
    TOMLVAULT -.->|"vault.cli read at rebuild"| VAULTCLIENTRY

    STATEOF["stateOf function<br/>system-manifest.js:453-466<br/>a false parent gate now dominates child gates (:457-458)"]
    VAULTTUIENTRY --> STATEOF
    STATEOF --> MODESTRING["off and none count as off, plus a per-entry off_values set<br/>skill-router declares table as its off value<br/>system-manifest.js:475-478, entry at system-manifest.js:345"]

    NOTE1["DOC-DRIFT - the code comment at system-manifest.js:390-396 still says the<br/>split is into TWO catalogue entries (ADR-2020), but [vault] now has THREE:<br/>root/pages/format are boot-class, tui and cli are both rebuild-class but gate<br/>different binaries (Rune TUI vs the vault CLI door)"]
    VAULTTUIENTRY --- NOTE1

    NOTE2["DIVERGENCE - vault-tui and vault-cli are REBUILD-class, flipping either key and<br/>restarting the container is NOT sufficient - the Rune binary or the vault CLI is<br/>absent from the package set until agentbox.sh rebuild runs"]
    STATEOF --- NOTE2

    NOTE3["INVARIANT ADR-2107/ADR-2108 - cli defaults to true whenever root is set (opt-OUT),<br/>unlike tui which defaults to none (opt-IN); agents reach the corpus through<br/>/opt/agentbox/bin/vault and the Loom over HTTP since ontology-bridge was retired,<br/>agentbox.toml.schema.json:284-308"]
    VAULTCLIENTRY --- NOTE3
```

## AB-01.11 ADR-2080 model_routing.neural gate - pinned artefacts baked into the image (rebuild-class)
```mermaid
flowchart TB
    TOML2["[model_routing.neural] in agentbox.toml<br/>agentbox.toml:1736-1746<br/>enabled/provider/quality_bar/cost_ceiling_usd_per_mtok/privacy_tier/trajectory/assets_dir"] --> CFG2["agentboxConfig<br/>flake.nix:176"]
    CFG2 --> MRNCFG["modelRoutingNeuralCfg = (agentboxConfig.model_routing or {}).neural or {}<br/>flake.nix:184"]
    ARTJSON["config/model-router/artefacts.json<br/>hash-pinned files[] list"] --> MRNFETCH["modelRouterArtefacts = builtins.fromJSON ...<br/>flake.nix:185"]
    MRNFETCH --> MRNASSETS["modelRouterAssets = pkgs.runCommand ...<br/>per-file pkgs.fetchurl url+sha256, copied to $out/dest<br/>flake.nix:186-194"]

    MRNCFG -->|"modelRoutingNeuralCfg.enabled or false"| GATE{"lib.optionalString<br/>flake.nix:2242"}
    MRNASSETS --> GATE
    GATE -->|true| BAKE["mkdir $out/opt/agentbox/model-router<br/>cp -r modelRouterAssets/. into it<br/>flake.nix:2245-2247"]
    GATE -->|false| SKIP["byte-identical-when-off: nothing copied<br/>flake.nix comment lines 2243-2244"]
    BAKE --> ALLPKG2["mkImage layers<br/>flake.nix:4589"]

    subgraph MANIFEST2["system-manifest.js CATALOGUE"]
        MRNENTRY["id model-routing-neural<br/>gate model_routing.neural.enabled, apply_class rebuild<br/>system-manifest.js:118-120"]
    end
    TOML2 -.->|"enabled read at rebuild"| MRNENTRY

    subgraph SCHEMA2["schema/agentbox.toml.schema.json"]
        NEURALSCHEMA["model_routing.properties.neural<br/>additionalProperties false<br/>agentbox.toml.schema.json:1612-1614"]
    end
    TOML2 -.-> NEURALSCHEMA

    NOTE1["INVARIANT ADR-2080 - the npm @claude-flow/cli tarball ships no router<br/>artefacts (files list excludes assets/model-router), so ONE hash-verified<br/>manifest (config/model-router/artefacts.json) is the single source for both<br/>this Nix bake and scripts/model-router-fetch.sh's pre-rebuild fallback dir"]
    MRNASSETS --- NOTE1

    NOTE2["see AB-02.20 (the router seed in the session plane),<br/>AB-05.11 (agentbox.sh model-router CLI), AB-29 (router console itself)"]
    BAKE --- NOTE2
```
