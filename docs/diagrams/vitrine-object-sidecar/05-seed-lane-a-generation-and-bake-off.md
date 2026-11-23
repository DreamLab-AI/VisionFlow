---
id: VO-05
title: Seed selection, the Lane-A generation subprocess and the bake-off evidence
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/decisions/adr-002-bakeoff-lanes-and-licences.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
adrs: [vitrine-object-sidecar:ADR-002]
sources:
  - ../vitrine-object-sidecar/sidecar/seed.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/generate.py
  - ../vitrine-object-sidecar/sidecar/workers/trellis_worker.py
  - ../vitrine-object-sidecar/sidecar/emplace.py
  - ../vitrine-object-sidecar/sidecar/evalkit.py
  - ../vitrine-object-sidecar/scripts/score_bakeoff.py
  - ../vitrine-object-sidecar/scripts/score_bakeoff_run.log
  - ../vitrine-object-sidecar/scripts/object_views.py
  - ../vitrine-object-sidecar/tools_recarve_and_score.py
  - ../vitrine-object-sidecar/tools_score_bakeoff.py
  - ../vitrine-object-sidecar/docs/STATE-OF-PLAY.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/decisions/adr-002-bakeoff-lanes-and-licences.md
  - ../vitrine-object-sidecar/report/sections/07-licence-deployment.tex
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-05.1 Seed: evidence-scored frames for both lanes

```mermaid
flowchart TB
    SUP["restrict to the instance's own support frames when carve supplied them<br/>pipeline.py:696,698,699"]
    EV["FrameEvidence: mask, pose, K and rectified pixels<br/>pipeline.py:705,706"]
    SCORE["score is area times normalised sharpness times centrality times edge clearance<br/>seed.py:123,124,130,132"]
    SH["sharpness is variance of Laplacian over the mask, 1.0 when there is no image or no mask<br/>seed.py:86,88,89,92"]
    LA["Lane A: best frame, square RGBA matte, pad 0.12, at least 1024 px<br/>seed.py:228,230,136,164,165"]
    LB["Lane B: farthest-point in optical-axis space, skip views within 10 degrees<br/>seed.py:173,194,203,204"]
    PROV["provenance: score components, crop box, frames, k requested<br/>seed.py:233,238,245,247"]
    REC["c.seed holds lane_a_frame, crop path, lane_b_frames, provenance<br/>pipeline.py:718,719,721,722"]
    SUP --> EV --> SCORE
    SH --> SCORE
    SCORE --> LA --> PROV
    SCORE --> LB --> PROV
    PROV --> REC
    INV["INVARIANT: conditioning is real frames and SAM mattes only, never splat renders<br/>seed.py:6,7"]
    LA --- INV
    EXT["EXTERNAL: no code here feeds lane_b_frames to a generator, object_views writes multi-view crops for an outside Lane-B run<br/>object_views.py:2,9"]
    LB --- EXT
```

**What it shows.** Seeding runs inside the CLI pass; its outputs are a crop file per object and a recorded Lane-B frame list.

## VO-05.2 generate_lane_a and the TRELLIS.2 worker contract

```mermaid
sequenceDiagram
    autonumber
    participant DRV as generate_lane_a<br/>generate.py:175
    participant SH as bash in models-lane2 venv<br/>generate.py:155
    participant WK as trellis_worker main<br/>trellis_worker.py:98
    participant TR as Trellis2ImageTo3DPipeline<br/>trellis_worker.py:40
    DRV->>DRV: write job.json with weights, crop, seeds, pipeline_type, texture size, memory_fraction (generate.py:193,202,205)
    DRV->>SH: subprocess.run of bash -c (generate.py:210)
    SH->>SH: source activate and env.sh, CUDA_VISIBLE_DEVICES 2, HF cache, offline, SPARSE_CONV_BACKEND, TRITON_LIBCUDA_PATH (generate.py:157,158,162,166,168,170)
    SH->>WK: exec python worker --job (generate.py:171)
    WK->>WK: optional per-process VRAM fraction to emulate 32 GB (trellis_worker.py:114,117)
    WK->>TR: from_pretrained, cuda (trellis_worker.py:42,43)
    loop each seed
        WK->>TR: run with seed, pipeline_type, return latent (trellis_worker.py:53,56,57)
        TR-->>WK: mesh, to_glb, export seed_NNNN.glb (trellis_worker.py:64,78,79)
        WK->>WK: runtime, allocated and reserved VRAM peaks (trellis_worker.py:87,90,91)
    end
    WK->>WK: candidates.json, exit 0 if any seed succeeded (trellis_worker.py:141,144)
    WK-->>DRV: candidates.json parsed into Candidate list (generate.py:211,217,218)
    Note over WK: INVARIANT: a failing seed records its error and traceback and does not abort the run (trellis_worker.py:124,126,127,128)
    Note over DRV: DEBT: only a missing candidates.json raises, the worker exit code is otherwise ignored (generate.py:210,212,213)
```

**What it shows.** The model environment is reached only through two JSON files and one subprocess; the sidecar never imports TRELLIS.2.

## VO-05.3 Candidate selection and the bake-off result

```mermaid
flowchart TB
    SC["score_candidates: emplace each GLB against real masks, score is mean silhouette IoU<br/>generate.py:243,271,276"]
    SV["seed variance: mean, std, min, max, range per metric<br/>evalkit.py:405,428,429,430,433"]
    PCS["per_camera_scores with PSNR and SSIM exists, unused by the selector<br/>evalkit.py:303,335,336"]
    BEST["best: highest score, else first ok candidate<br/>generate.py:148,149,151,152"]
    SC --> SV
    SC --> BEST
    PCS -.-> SC
    TENS1["TENSION: DESIGN select is IoU plus DINO similarity plus depth plus mesh sanity, the code selects on silhouette IoU alone<br/>DESIGN.md:62,63,64,65<br/>generate.py:258,259"]
    SC --- TENS1
    STUB["scripts/score_bakeoff.py is a docstring only, no code<br/>score_bakeoff.py:1,4,5"]
    LOG["its run log: 24 instance views, laneA_best 0.2115, laneB single 0.1272, multiview 0.1359<br/>score_bakeoff_run.log:1,2,3,4"]
    CLEAN["tools_recarve_and_score writes the same radio_bakeoff_clean.json, Lane-A best is fixed at seed_0000<br/>tools_recarve_and_score.py:60,84"]
    HEAD["STATE-OF-PLAY headline: Lane-A 0.212 vs Lane-B 0.136 and 0.127<br/>STATE-OF-PLAY.md:69,70,71"]
    STUB --> LOG --> HEAD
    CLEAN --> HEAD
    DEBT1["DEBT: the instance-aware scorer behind the headline ranking is not in the repository, only its docstring and log<br/>score_bakeoff.py:1,5<br/>score_bakeoff_run.log:5"]
    STUB --- DEBT1
    TENS2["TENSION: DESIGN fairness asks for fixed-count and fixed-GPU-time comparisons with seed variance, the clean ranking is one object, one Lane-A seed, two Lane-B GLBs<br/>DESIGN.md:27,28,29,30<br/>tools_recarve_and_score.py:60,61,62"]
    CLEAN --- TENS2
    LIN["lane_a_checkpoint_lineage: TRELLIS.2 commit unknown, custom-noncommercial, DINOv3 gated, downloaded at install<br/>generate.py:48,64,66,70,73"]
    TENS3["TENSION: ADR-002 and the report list TRELLIS.2-4B as MIT clean, the lineage helper classes it custom-noncommercial<br/>adr-002-bakeoff-lanes-and-licences.md:22<br/>07-licence-deployment.tex:51<br/>generate.py:66"]
    LIN --- TENS3
    DEBT2["DEBT: venv, weights and TRELLIS.2 paths are absolute dev-host constants<br/>generate.py:28,29,30,31"]
    LIN --- DEBT2
```

**What it shows.** Selection exists as a library call; the published ranking was produced by root tools and an uncommitted scorer, on one object.
