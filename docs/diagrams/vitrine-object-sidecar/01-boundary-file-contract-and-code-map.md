---
id: VO-01
title: Boundary, file contract and what the CLI actually wires
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/STATE-OF-PLAY.md
  - ../vitrine-object-sidecar/docs/decisions/adr-001-sidecar-boundary.md
adrs: [vitrine-object-sidecar:ADR-001, vitrine-object-sidecar:ADR-002, vitrine-object-sidecar:DDD-001]
sources:
  - ../vitrine-object-sidecar/README.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/STATE-OF-PLAY.md
  - ../vitrine-object-sidecar/docs/decisions/adr-001-sidecar-boundary.md
  - ../vitrine-object-sidecar/docs/decisions/ddd-001-domain-model.md
  - ../vitrine-object-sidecar/report/sections/06-interface-contract.tex
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/manifest.py
  - ../vitrine-object-sidecar/sidecar/colmap_model.py
  - ../vitrine-object-sidecar/sidecar/generate.py
  - ../vitrine-object-sidecar/sidecar/identify.py
  - ../vitrine-object-sidecar/sidecar/workers/trellis_worker.py
  - ../vitrine-object-sidecar/sidecar/texture.py
  - ../vitrine-object-sidecar/sidecar/compose.py
  - ../vitrine-object-sidecar/sidecar/emplace.py
  - ../vitrine-object-sidecar/sidecar/evalkit.py
  - ../vitrine-object-sidecar/sidecar/charts.py
  - ../vitrine-object-sidecar/tests/test_integration_pipeline.py
  - ../vitrine-object-sidecar/tests/test_compose_scene.py
  - ../vitrine-object-sidecar/tools_score_bakeoff.py
  - ../vitrine-object-sidecar/requirements.lock
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-01.1 The two systems meet at the filesystem

```mermaid
flowchart TB
    subgraph UOS["UoS Vitrine preservation package, read-only input"]
        SFM["sfm/ COLMAP text model, cameras.txt probed<br/>pipeline.py:79,80"]
        ALT["fallback colmap/exported layout<br/>pipeline.py:82,83,84"]
        PLY["model/scene.ply, None when absent<br/>pipeline.py:85,89"]
        ORIG["originals/ frames<br/>pipeline.py:90"]
    end
    OPEN["Package.open builds the four paths, nothing else is read<br/>pipeline.py:77"]
    RUN["run over --package into --out<br/>pipeline.py:562,568"]
    subgraph OUT["--out run directory, written by the sidecar"]
        RECT["rectified/ PNGs and intrinsics.json<br/>pipeline.py:235,269"]
        IDENT["identify/ masks and identify.json<br/>pipeline.py:331,343"]
        OBJ["objects/label_NN.ply carved subsets<br/>pipeline.py:647"]
        SEEDD["seed/label_NN_crop.png<br/>pipeline.py:716"]
        MAN["objects.json at the --out root<br/>pipeline.py:831"]
        SUM["summary.json<br/>pipeline.py:669"]
    end
    SFM --> OPEN
    ALT --> OPEN
    PLY --> OPEN
    ORIG --> OPEN
    OPEN --> RUN --> RECT --> IDENT --> OBJ --> SEEDD --> MAN --> SUM
    DRIFT1["DOC-DRIFT: README and STATE-OF-PLAY say the sidecar writes objects/objects.json and reads the package manifest, the driver writes objects.json beside objects/ and never opens a manifest<br/>README.md:10<br/>STATE-OF-PLAY.md:44,45,46"]
    MAN --- DRIFT1
    DRIFT2["DOC-DRIFT: AssetRef.path is documented relative to objects/, the driver stores paths that already start objects/<br/>manifest.py:73<br/>pipeline.py:647"]
    OBJ --- DRIFT2
    EXT["EXTERNAL: uos PR adds an objects manifest key, a vitrine objects subcommand and a read-only panel<br/>DESIGN.md:103,105,106"]
    RUN --- EXT
    DRIFT3["DOC-DRIFT: STATE-OF-PLAY says both UoS PRs merged, the report says two pull requests are open<br/>STATE-OF-PLAY.md:53,54,55<br/>06-interface-contract.tex:27"]
    EXT --- DRIFT3
```

**What it shows.** The package is consumed read-only through four paths; everything the driver writes lands under the `--out` directory, never inside the package (`ddd-001-domain-model.md:15`).

## VO-01.2 Three environments and the licence fence

```mermaid
flowchart TB
    subgraph UOSENV["uos-vitrine tree, MIT, no model dependencies"]
        UPKG["must not import the custom-licensed model stack<br/>adr-001-sidecar-boundary.md:16,17,18"]
    end
    subgraph SIDE["sidecar venv, Python 3.12, torch cu124"]
        VEND["COLMAP reader logic-copied, never imported across the boundary<br/>colmap_model.py:3,4,5,6"]
        GD["GroundingDINO loaded in-process<br/>identify.py:459,465"]
        SAM["SAM 3.1 repo put on sys.path and built in-process<br/>identify.py:539,540,548"]
        LOCK["torch 2.6.0 cu124 pinned<br/>requirements.lock:83"]
    end
    subgraph MODEL["models-lane2 py3.10 venv, separate licences"]
        WRK["trellis_worker is the only file importing TRELLIS.2<br/>trellis_worker.py:3,4,5"]
        TR["Trellis2ImageTo3DPipeline from_pretrained<br/>trellis_worker.py:40,42"]
    end
    DRV["generate_lane_a shells into the venv with bash -c<br/>generate.py:155,171,210"]
    VEND -.->|file contract only| UPKG
    DRV -->|job.json then candidates.json| WRK --> TR
    GD --> SAM
    TENS["TENSION: the worker and driver call the sidecar env MIT-clean, yet SAM 3.1 runs in it with licence unknown<br/>trellis_worker.py:6<br/>generate.py:7,8<br/>identify.py:593"]
    SAM --- TENS
```

**What it shows.** Only TRELLIS.2 is fenced off by a process boundary; the identify models run inside the sidecar's own interpreter (`pipeline.py:597,598`).

## VO-01.3 Wired by the CLI, library-only, or outside the repository

```mermaid
flowchart TB
    subgraph WIRED["wired: python -m sidecar run"]
        STAGES["ALL_STAGES is identify, carve, seed, emplace<br/>pipeline.py:59"]
        EINIT["emplace is init-only: identity rotation, scale 1.0, refine deferred<br/>pipeline.py:739,742,745"]
        WMAN["objects.json and summary.json<br/>pipeline.py:664,669"]
    end
    subgraph LIB["library-only: reached from tests and root tools scripts"]
        GEN["generate_lane_a and score_candidates<br/>generate.py:175,243"]
        EMP["emplace 7-DoF silhouette fit<br/>emplace.py:187"]
        TEX["bake_texture<br/>texture.py:254"]
        CMP["compose_scene and export_composed_scene<br/>compose.py:93,228"]
        EVK["evalkit seed_variance and per_camera_scores<br/>evalkit.py:303,405"]
        CHT["charts house style<br/>charts.py:31,63"]
    end
    subgraph TESTS["only callers of the library chain"]
        DRY["slow dry run chains generate, score, bake, compose<br/>test_integration_pipeline.py:101,110,124,128"]
        CST["composed_scene.glb written only by a test<br/>test_compose_scene.py:40,41"]
    end
    subgraph EXTREPO["outside the repository"]
        LB["EXTERNAL: Lane-B ReconViaGen GLBs read from project5 runs<br/>tools_score_bakeoff.py:19,51"]
    end
    STAGES --> EINIT --> WMAN
    DRY --> GEN
    DRY --> TEX
    DRY --> CMP
    GEN --> EMP
    GEN --> EVK
    CST --> CMP
    LB --> EMP
    DRIFT["DOC-DRIFT: README lists ten stages including generate, select, texture, compose under one run command<br/>README.md:17,21,24,25,35"]
    STAGES --- DRIFT
    DRIFT2["DOC-DRIFT: ADR-001 and STATE-OF-PLAY say the sidecar emits per-object GLBs and composed_scene.glb, the CLI writes Gaussian-subset PLYs only<br/>adr-001-sidecar-boundary.md:10<br/>STATE-OF-PLAY.md:46,47<br/>pipeline.py:647"]
    WMAN --- DRIFT2
    DRIFT3["DOC-DRIFT: the report says the sidecar validates a handoff, can write scene_cleaned.ply and imports uos-vitrine cleanly, no code does the first two and the COLMAP reader exists to avoid the import<br/>06-interface-contract.tex:6,7,8,18<br/>colmap_model.py:3,4,5,6"]
    WMAN --- DRIFT3
```

**What it shows.** Four stages are wired; generation, refinement, texture and composition exist as tested functions with no tracked production caller, and Lane B has no code here.
