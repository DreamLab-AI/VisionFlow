---
id: VO-08
title: Test tiers, environment, evidence tools and the closeout report build
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/README.md
  - ../vitrine-object-sidecar/smoke/RESULTS.md
adrs: [vitrine-object-sidecar:PRD-001]
sources:
  - ../vitrine-object-sidecar/pytest.ini
  - ../vitrine-object-sidecar/tests/conftest.py
  - ../vitrine-object-sidecar/tests/test_identify.py
  - ../vitrine-object-sidecar/tests/test_integration_generate.py
  - ../vitrine-object-sidecar/tests/test_integration_pipeline.py
  - ../vitrine-object-sidecar/smoke/env.sh
  - ../vitrine-object-sidecar/smoke/sam31_smoke.py
  - ../vitrine-object-sidecar/smoke/RESULTS.md
  - ../vitrine-object-sidecar/requirements.lock
  - ../vitrine-object-sidecar/README.md
  - ../vitrine-object-sidecar/sidecar/__main__.py
  - ../vitrine-object-sidecar/sidecar/generate.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/texture.py
  - ../vitrine-object-sidecar/sidecar/compose.py
  - ../vitrine-object-sidecar/sidecar/charts.py
  - ../vitrine-object-sidecar/scripts/make_closeout_charts.py
  - ../vitrine-object-sidecar/scripts/render_turntable.py
  - ../vitrine-object-sidecar/scripts/validate_gallery.py
  - ../vitrine-object-sidecar/scripts/compare_passes.py
  - ../vitrine-object-sidecar/tools_blender_turntable.py
  - ../vitrine-object-sidecar/tools_score_bakeoff.py
  - ../vitrine-object-sidecar/tools_contact_sheets.py
  - ../vitrine-object-sidecar/tools_recarve_radio.py
  - ../vitrine-object-sidecar/tools_score_refined.py
  - ../vitrine-object-sidecar/tools_composed_proof.py
  - ../vitrine-object-sidecar/report/main.tex
  - ../vitrine-object-sidecar/report/sections/04-live-results.tex
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-08.1 Two test tiers and what makes the slow tier skip

```mermaid
flowchart TB
    INI["pytest.ini registers slow, addopts -ra, nothing deselected by default<br/>pytest.ini:2,3,4"]
    FAST["fast CPU tier: stub models, one slow identify test hits the GPU<br/>test_identify.py:1"]
    GAL["gallery_output golden set at a fixed dev-host path<br/>conftest.py:10,11"]
    SKIP["slow fixtures skip when the gallery cameras.txt is missing<br/>conftest.py:15,16,21,22"]
    GEN["real TRELLIS.2 smoke skips without the models-lane2 venv, weights or a pass-1 crop<br/>test_integration_generate.py:21,26,27,28,29"]
    DRY["slow dry run mocks the subprocess, everything downstream real<br/>test_integration_pipeline.py:4,5,6,7,100"]
    INI --> FAST
    INI --> GAL --> SKIP
    INI --> GEN
    SKIP --> DRY
    OPEN["OPEN: no CI workflow and no package metadata are tracked, the README names a venv rebuilt from requirements.lock as the environment<br/>README.md:36,44"]
    INI --- OPEN
```

**What it shows.** On a machine without the dev host's datasets and model venv, every real-data test skips and only the stubbed CPU tier runs.

## VO-08.2 The dev-host environment the code assumes

```mermaid
flowchart TB
    ENV["smoke/env.sh: Nix libstdc++ on LD_LIBRARY_PATH, PCI_BUS_ID device order<br/>env.sh:2,4"]
    DEV0["env.sh default device 0, the A6000<br/>env.sh:5"]
    DEV2["generate default device 2, never GPU1, and the CLI docstring uses 2<br/>generate.py:91,93<br/>__main__.py:8"]
    HF1["env.sh HF_HOME under .hfcache in project5/object-sidecar<br/>env.sh:7,10"]
    HF2["generate HF_HOME .hf-cache beside the package, offline<br/>generate.py:36,95,98"]
    LOCK["lock: torch 2.6.0 cu124, numpy 1.26.4, groundingdino-py 0.4.0, plyfile, trimesh<br/>requirements.lock:83,37,17,58,87"]
    OPTS["optional imports degrade silently: pillow-heif, xatlas, and pygltflib imported bare<br/>pipeline.py:98,100<br/>texture.py:85,92<br/>compose.py:251"]
    SMK["smoke tests load SAM 3.1 and GroundingDINO from fixed paths<br/>sam31_smoke.py:17,18,19"]
    ENV --> DEV0
    DEV0 -.- DEV2
    HF1 -.- HF2
    LOCK --> OPTS
    ENV --> SMK
    TENS["TENSION: two HF caches and two default GPUs, the generator overrides env.sh's cache and device for the model venv<br/>env.sh:5,10<br/>generate.py:158,162"]
    HF2 --- TENS
    DEBT["DEBT: pillow-heif, xatlas and pygltflib are imported by the code but none is pinned in requirements.lock<br/>pipeline.py:98<br/>texture.py:85<br/>compose.py:251"]
    OPTS --- DEBT
    REC["smoke results record GPU1 in use and avoided, SAM 3.1 failing on GPU0<br/>RESULTS.md:4,13"]
    SMK --- REC
```

**What it shows.** Paths, devices and caches are written for one container; the README's environment note (`README.md:44,45,46`) is the only portability guidance.

## VO-08.3 From run JSON to the closeout PDF

```mermaid
flowchart TB
    TOOLS["root tools pin project5 runs and datasets and push project5/object-sidecar onto sys.path<br/>tools_score_bakeoff.py:12,16,17,18,19"]
    VAL["validate_gallery: 3D F1, observed-vs-withheld reprojected IoU, mesh_sanity on the golden set<br/>validate_gallery.py:9,10,13,14,15"]
    CMP["compare_passes: pass1 vs pass2 summary.json deltas<br/>compare_passes.py:2,5,6"]
    SHEET["contact sheets via charts.contact_sheet<br/>tools_contact_sheets.py:11,66"]
    CH["closeout charts in the charts house style, PDF plus PNG<br/>make_closeout_charts.py:3,4,25<br/>charts.py:10,11"]
    TT1["render_turntable sets CYCLES<br/>render_turntable.py:49,50"]
    TT2["tools_blender_turntable: EEVEE crashes headless, CYCLES on CPU<br/>tools_blender_turntable.py:24,26,28"]
    TEX["main.tex: two-column, inputs sections 00 to 10<br/>main.tex:2,4,53,63"]
    TOOLS --> CH
    VAL --> CH
    CMP --> TEX
    SHEET --> TEX
    CH --> TEX
    TT1 --> TEX
    TT2 --> TEX
    TENS["TENSION: the chart script says every number is read from run records and nothing is hand-entered, the Lane-B VRAM and runtime bars are literals<br/>make_closeout_charts.py:4,5,164,165"]
    CH --- TENS
    DRIFT["DOC-DRIFT: render_turntable's header says the engine is EEVEE, the code sets CYCLES<br/>render_turntable.py:8,50"]
    TT1 --- DRIFT
```

**What it shows.** Every reported number passes through root tools pinned to `project5` paths outside this repository; the report sources then build with `latexmk` from the report Makefile.

## VO-08.4 Root tools behind the recarve, refined scores and composed-scene figure

```mermaid
flowchart TB
    RC["tools_recarve_radio: re-carve the nc04 radio from pass-2 masks, sweep min_views and fg_threshold<br/>tools_recarve_radio.py:1,2,42,43,46"]
    RCO["keeps the tightest p5-p95 extent, saves radio_00_recarved.npy under the project5 run<br/>tools_recarve_radio.py:52,56,57"]
    SR["tools_score_refined: scene_scale prior 9.2034 plus or minus 15 percent, 24 yaws, refine on<br/>tools_score_refined.py:1,2,3,15,57,58,59"]
    SRO["writes radio_bakeoff_refined.json beside the pass-1 generate run<br/>tools_score_refined.py:65,67"]
    CP["tools_composed_proof: Lane-A green plus Lane-B blue silhouettes over real rectified frames<br/>tools_composed_proof.py:1,2,3,57,58"]
    PL["placement is carve-init: mean-extent scale onto the carve centroid, no emplace solve<br/>tools_composed_proof.py:43,44,45"]
    FIG["report figure renders/composed_proof.png, captioned carve-init and yaw unrefined<br/>04-live-results.tex:129,133,134"]
    RC --> RCO
    SR --> SRO
    CP --> PL --> FIG
    EXT["EXTERNAL: all three read and write project5 runs and datasets outside this repository<br/>tools_recarve_radio.py:9,10<br/>tools_score_refined.py:11,12,13,14<br/>tools_composed_proof.py:9,10,11,12"]
    RC --- EXT
    OPEN["OPEN: no tracked file reads radio_bakeoff_refined.json or radio_00_recarved.npy, and the report says refined scoring is still running with no numeric winner claimed<br/>04-live-results.tex:172,173,174"]
    SRO --- OPEN
    DRIFT["DOC-DRIFT: the figure's SRC comment credits the sidecar compose stage, the PNGs come from this root tool calling compose_view directly<br/>04-live-results.tex:128<br/>tools_composed_proof.py:7,57"]
    FIG --- DRIFT
```

**What it shows.** The recarve, refined-scoring and composed-proof evidence are produced by root scripts outside the CLI; only the composed-proof PNG reaches the report, and the report's section text is otherwise covered only through `main.tex` (VO-08.3) and the interface-contract section (VO-01).

