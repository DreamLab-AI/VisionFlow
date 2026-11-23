---
id: VO-02
title: CLI, stage driver and configuration surface
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/README.md
adrs: [vitrine-object-sidecar:DDD-001]
sources:
  - ../vitrine-object-sidecar/sidecar/__main__.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/colmap_model.py
  - ../vitrine-object-sidecar/docs/decisions/ddd-001-domain-model.md
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-02.1 One run from the command line to summary.json

```mermaid
sequenceDiagram
    autonumber
    participant CLI as main<br/>__main__.py:59
    participant RUN as run<br/>pipeline.py:562
    participant PKG as Package and COLMAP<br/>pipeline.py:568
    participant FS as out directory<br/>pipeline.py:575
    participant ID as identify models<br/>pipeline.py:597
    CLI->>CLI: unknown stage names print and return 2 (:66,69)
    CLI->>RUN: RunConfig from flags, then run(package, out, cfg) (__main__.py:75,86)
    RUN->>PKG: Package.open, read_model, scene_scale (pipeline.py:568,569,570)
    alt rectified cache present and identify not requested
        RUN->>FS: load_cached_rect_frames, skip the re-undistort (pipeline.py:577,580)
    else
        RUN->>PKG: select_frames, RuntimeError when nothing resolves (pipeline.py:585,587,588)
        RUN->>FS: rectify_frames persists rectified PNGs (pipeline.py:590)
    end
    alt identify in stages
        RUN->>ID: load_default_models unless injected (pipeline.py:597,598)
        ID-->>RUN: IdentifyResult, VRAM peak recorded (pipeline.py:610,615)
    else
        RUN->>FS: rehydrate identify.json (pipeline.py:617,618)
    end
    RUN->>PKG: carve needs model/scene.ply or raises (pipeline.py:624,625,626)
    RUN->>FS: per-object subset PLYs (pipeline.py:645,648)
    RUN->>RUN: seed only when candidates exist (pipeline.py:652)
    RUN->>RUN: emplace init only when carve ran in this call (pipeline.py:658)
    RUN->>FS: objects.json then summary.json (pipeline.py:664,669)
    RUN-->>CLI: summary printed, exit 0 (__main__.py:88,102)
    Note over RUN: DEBT: seed and emplace read candidates built by carve in the same call, so --stages seed or emplace alone writes no objects (:621,652,658)
```

**What it shows.** Stage re-runs reuse cached rectified frames and `identify.json`, but carve is the only producer of candidates, so the later stages cannot run without it.

## VO-02.2 Flags, RunConfig defaults and what reaches summary.json

```mermaid
flowchart TB
    subgraph FLAGS["CLI flags on the run subcommand"]
        F1["--package, --out required<br/>__main__.py:28,29"]
        F2["--stages default all four, --prompts default nested-cinema list<br/>__main__.py:30,32"]
        F3["--max-frames 0 means all, --detect-max-side 2048, --no-per-prompt<br/>__main__.py:34,36,38"]
        F4["--box-threshold 0.30, --text-threshold 0.25<br/>__main__.py:40,41"]
        F5["--depth-band-frac 0.01, --fg-threshold 0.5, --min-views 4, --carve-min-votes 3<br/>__main__.py:42,44,45,47"]
        F6["--min-component-points 200, --max-components-per-label 6, --lane-b-k 8<br/>__main__.py:49,50,51"]
        F7["--extra-originals, --device cuda, -v<br/>__main__.py:52,54,55"]
    end
    subgraph CFG["RunConfig, the baked pass-2 recipe"]
        C1["detect_max_side 2048 chosen by mask IoU<br/>pipeline.py:534"]
        C2["depth_band_frac scales by scene_scale<br/>pipeline.py:538"]
        C3["per_instance_carve True, cluster_radius_frac 0.04, max_views_per_label 80<br/>pipeline.py:544,545,546"]
        C4["thirteen nested-cinema prompts, no person word<br/>pipeline.py:53,54,56"]
    end
    SCALE["scene_scale is half the camera-centre bounding-box diagonal<br/>colmap_model.py:301,309,311"]
    SUMCFG["summary config block<br/>pipeline.py:847"]
    FLAGS --> CFG
    C2 --> SCALE
    CFG --> SUMCFG
    DEBT1["DEBT: per_instance_carve, cluster_radius_frac and max_views_per_label have no CLI flag<br/>pipeline.py:544,545,546<br/>__main__.py:75,84"]
    C3 --- DEBT1
    DEBT2["DEBT: the recorded config omits lane_b_k, max_views_per_label, extra_originals and device, so a run cannot be replayed from summary.json alone<br/>pipeline.py:848,856"]
    SUMCFG --- DEBT2
```

**What it shows.** Every carve threshold is relative to `scene_scale`, and three of the baked carve settings can be changed only in code.

## VO-02.3 Degrade, never fake: what is recorded instead of invented

```mermaid
flowchart TB
    FR["frame resolution by basename then extension fallback, extra dirs after originals<br/>pipeline.py:117,119,125"]
    FRR["per-camera registered, resolved_on_disk, selected counts<br/>pipeline.py:192,193,194"]
    STR["stratified quota per camera group when max_frames is set<br/>pipeline.py:179,181,182"]
    HEIC["pillow-heif missing: HEIC frames skipped with a warning<br/>pipeline.py:100,101"]
    REJ1["label below min_views: rejected insufficient_views<br/>pipeline.py:463,464"]
    REJ2["no instance above the point floor: rejected<br/>pipeline.py:482,483"]
    KEEP["rejected list appended to objects.json, not deleted<br/>pipeline.py:833,836"]
    NOASSET["a candidate without an asset gets no ObjectRecord<br/>pipeline.py:810"]
    FR --> FRR
    FR --> STR
    HEIC --> FR
    REJ1 --> KEEP
    REJ2 --> KEEP
    INV["INVARIANT: missing stage output degrades the record, never fakes it<br/>ddd-001-domain-model.md:35"]
    KEEP --- INV
    DEBT["DEBT: the rectify cache is reused whenever identify is skipped, whatever max_frames or extra_originals now say<br/>pipeline.py:577,580"]
    STR --- DEBT
```

**What it shows.** Missing frames, sparse labels and empty carves surface as counts and `rejected` entries in the outputs rather than as silent drops.
