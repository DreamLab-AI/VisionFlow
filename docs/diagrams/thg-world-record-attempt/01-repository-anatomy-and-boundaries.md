---
id: TH-01
title: Repository anatomy, external boundaries and the absent build path
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/README.md]
adrs: []
sources:
  - ../THG-world-record-attempt/README.md
  - ../THG-world-record-attempt/configs/topshop.json
  - ../THG-world-record-attempt/docs/prd/campaign-generator-v3.md
  - ../THG-world-record-attempt/scripts/clean_scenes.py
  - ../THG-world-record-attempt/scripts/generate_composites.py
  - ../THG-world-record-attempt/scripts/generate_veo_branded.py
  - ../THG-world-record-attempt/scripts/poll-veo-operations.py
  - ../THG-world-record-attempt/scripts/render_3d_pass.py
  - ../THG-world-record-attempt/scripts/render_tracked_composite.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_riffs.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_expanded_riffs.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_flux2_riffs.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch1_repose.py
  - ../THG-world-record-attempt/task-two-repose/scripts/repose_batch2.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch3_repose.py
  - ../THG-world-record-attempt/src/vfx_pipeline/__init__.py
  - ../THG-world-record-attempt/src/vfx_pipeline/__main__.py
  - ../THG-world-record-attempt/src/vfx_pipeline/pipeline.py
  - ../THG-world-record-attempt/src/vfx_pipeline/comfyui_client.py
  - ../THG-world-record-attempt/src/vfx_pipeline/blender_client.py
  - ../THG-world-record-attempt/tests/test_vfx_pipeline.py
  - ../THG-world-record-attempt/reports/world-record-report.tex
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-01.1 What the repository holds, and which files run

```mermaid
flowchart TB
    subgraph CAMPAIGN["Campaign scripts, run one by one as python3 file.py, no shared module"]
        direction TB
        SC["scene clean and direct composites<br/>clean_scenes.py:165<br/>generate_composites.py:156"]
        RF["Gemini riffs and local Flux 2 riffs<br/>_generate_riffs.py:127<br/>_generate_expanded_riffs.py:104<br/>_generate_flux2_riffs.py:273"]
        RP["mannequin repose, three batch scripts<br/>batch1_repose.py:160<br/>repose_batch2.py:101<br/>batch3_repose.py:73"]
        VE["Veo branded films and a separate poller<br/>generate_veo_branded.py:395<br/>poll-veo-operations.py:20"]
    end
    subgraph VFX["VFX object-tracking package, the only importable code"]
        direction TB
        INIT["vfx_pipeline v2.0.0<br/>__init__.py:6"]
        MAIN["python -m vfx_pipeline runs main<br/>__main__.py:2,4<br/>pipeline.py:514"]
        BR["Blender-side scripts run under blender --background<br/>render_tracked_composite.py:13<br/>render_3d_pass.py:13"]
    end
    subgraph EVID["Records and evidence, read by no code"]
        direction TB
        CFG["brand and engine config<br/>topshop.json:2"]
        REP["formal report source, figures from reports/figures<br/>world-record-report.tex:1,88"]
    end
    CAMPAIGN --> VFX --> EVID
    DRIFT["DOC-DRIFT: the README structure block lists reports, configs, assets, task-two-repose and scripts, but no src, tests or exports, and never names the VFX pipeline<br/>README.md:171,204"]
    VFX --- DRIFT
    DEBT["DEBT: campaign scripts hard-code an absolute checkout under a campaign directory, not this repository root<br/>generate_composites.py:11<br/>clean_scenes.py:26<br/>generate_veo_branded.py:25<br/>batch1_repose.py:14<br/>_generate_riffs.py:10"]
    CAMPAIGN --- DEBT
    INV["INVARIANT: the Blender render scripts resolve every path from their own file location<br/>render_3d_pass.py:20,21<br/>render_tracked_composite.py:28,29"]
    BR --- INV
    DIV["DIVERGENCE: the README describes voice-captured prompts decomposed by a Claude Flow agent swarm, no orchestration or voice-capture code is committed, only the scripts it ran<br/>README.md:33,133"]
    CAMPAIGN --- DIV
```

**What it shows.** Two unrelated programmes share the tree: the February 2026 campaign scripts and the March 2026 VFX tracking package; everything else is records.

## TH-01.2 External services and how each is reached

```mermaid
flowchart LR
    subgraph LOCALSIDE["Local scripts"]
        direction TB
        GEM_CALLERS["Gemini image callers<br/>generate_composites.py:9<br/>clean_scenes.py:24<br/>batch1_repose.py:12"]
        VEO_CALLER["Veo submitter and poller<br/>generate_veo_branded.py:31,32"]
        FLUX_CALLER["Flux 2 riffs, ComfyUI host comfyui port 8188<br/>_generate_flux2_riffs.py:9"]
        VFX_CALLER["VFX pipeline clients, ComfyUI port 8188 and Blender MCP port 8765<br/>comfyui_client.py:16<br/>blender_client.py:11"]
        TOOLS["ffprobe, ffmpeg and curl as subprocesses<br/>generate_veo_branded.py:290,322,371"]
    end
    subgraph REMOTE["Remote endpoints"]
        direction TB
        GAPI["Google generativelanguage v1beta generateContent"]
        VAPI["Google generativelanguage v1beta predictLongRunning and operations"]
        COMFY["ComfyUI container"]
        BLENDER["Blender with a Unified Blender MCP addon"]
    end
    GEM_CALLERS --> GAPI
    VEO_CALLER --> VAPI
    FLUX_CALLER --> COMFY
    VFX_CALLER --> COMFY
    VFX_CALLER --> BLENDER
    KEY["credential from the environment, sent as a key query parameter<br/>generate_composites.py:8<br/>generate_veo_branded.py:30"]
    KEY --> GEM_CALLERS
    DEBT["DEBT: a literal Google API key is committed in the poller<br/>poll-veo-operations.py:5"]
    VEO_CALLER --- DEBT
    DRIFT["DOC-DRIFT: the PRD says all API keys come from the environment and are never hardcoded<br/>campaign-generator-v3.md:75"]
    DEBT --- DRIFT
    OPEN["OPEN: the config declares Imagen 4, Perplexity and a Salad API engine, no script in the tree calls any of them<br/>topshop.json:91,106,116"]
    REMOTE --- OPEN
```

**What it shows.** Every remote call is a raw HTTP request from a script; the only credential path is `GOOGLE_API_KEY`, and one script bypasses it with a committed key.

## TH-01.3 Build, test and release path

```mermaid
flowchart TB
    SRC["src/vfx_pipeline package, no pyproject, setup or requirements file at this commit<br/>__init__.py:1"]
    TEST["pytest suite reaches the package by editing sys.path<br/>test_vfx_pipeline.py:22,24"]
    SKIP["KeenTools example tests skip when the example folder is absent<br/>test_vfx_pipeline.py:104,109"]
    SRC --> TEST --> SKIP
    TEX["LaTeX report, figures from reports/figures<br/>world-record-report.tex:1,88"]
    PDF["compiled PDF committed beside it, 29 pages per the README<br/>README.md:175"]
    TEX --> PDF
    DEBT["DEBT: no CI workflow, no dependency manifest and no build script, so the tests, the report and the scripts run only by hand"]
    SKIP --- DEBT
    PDF --- DEBT
```

**What it shows.** Nothing builds, tests or publishes automatically; the committed PDF and media are the release.
