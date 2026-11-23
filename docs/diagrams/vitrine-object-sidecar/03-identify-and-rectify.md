---
id: VO-03
title: Identify (GroundingDINO to SAM 3.1) and the rectified pixel domain
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/decisions/adr-003-carve-emplace-texture.md
adrs: [vitrine-object-sidecar:ADR-002, vitrine-object-sidecar:ADR-003]
sources:
  - ../vitrine-object-sidecar/sidecar/identify.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/rectify.py
  - ../vitrine-object-sidecar/sidecar/colmap_model.py
  - ../vitrine-object-sidecar/tests/test_rectify.py
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/decisions/adr-003-carve-emplace-texture.md
  - ../vitrine-object-sidecar/README.md
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-03.1 Detect, segment and record one frame

```mermaid
sequenceDiagram
    autonumber
    participant ST as run_identify_stage<br/>pipeline.py:316
    participant RI as run_identify<br/>identify.py:345
    participant FR as identify_frame<br/>identify.py:280
    participant GD as GroundingDinoDetector<br/>identify.py:450
    participant SM as Sam3Segmenter<br/>identify.py:528
    ST->>ST: downscale long side to detect_max_side, record detect_scale (pipeline.py:335,336)
    ST->>RI: frames as id, name, pixels (pipeline.py:339)
    RI->>RI: lineage from both models, failure only noted (identify.py:370,372,373)
    RI->>FR: per frame (identify.py:376)
    FR->>GD: detect with prompts and thresholds (identify.py:292)
    GD->>GD: per-prompt mode runs one caption per prompt and forces the label to the prompt (identify.py:501,505)
    GD-->>FR: boxes, then score floor re-applied (identify.py:296)
    FR->>SM: segment the surviving boxes (identify.py:297)
    SM->>SM: one text prompt per unique label, best box-IoU mask per detection (identify.py:563,564,580,583)
    SM-->>FR: a mask or None per detection
    FR->>FR: count mismatch raises, wrong mask shape raises (identify.py:298,299,318,319)
    FR->>FR: mask below the area floor stays unsegmented (identify.py:324,326)
    FR->>FR: write masks/frameNNNNN_ii_label.png (identify.py:336,339)
    RI-->>ST: IdentifyResult, identify.json written (identify.py:399)
    Note over SM: DEBT: seg_match_iou 0.30 is configured and recorded but the match accepts any IoU at or above 0.0 (identify.py:72,411,583)
    Note over FR: INVARIANT: an unsegmented box is kept with status unsegmented, never a box-shaped mask (identify.py:314,316)
```

**What it shows.** GroundingDINO decides what and where, SAM 3.1 supplies the silhouette, and boxes and masks are matched only by box IoU.

## VO-03.2 People exclusion, model paths and lineage

```mermaid
flowchart TB
    EXCL["exclusion set: person, people, visitor, man, woman, child, human, hand<br/>identify.py:56,57"]
    ISX["is_excluded matches whole words of the label<br/>identify.py:270,272"]
    OBJS["objects are non-excluded, segmented instances<br/>identify.py:166,168,169"]
    PROMPTS["default driver prompts, no person word<br/>pipeline.py:53,54,56"]
    FORCE["per-prompt labels are the prompt text<br/>identify.py:505"]
    EXCL --> ISX --> OBJS
    PROMPTS --> FORCE --> ISX
    TENS1["TENSION: with the default prompts and per-prompt labels no detection can carry an excluded word, so README people excluded holds only by never asking for people<br/>README.md:17<br/>identify.py:306"]
    ISX --- TENS1
    PATHS["GroundingDINO weights and config, SAM 3.1 repo and checkpoint are absolute paths on the dev host<br/>identify.py:438,442,446,447"]
    LIN1["GroundingDINO Apache-2.0, obtained redistributed<br/>identify.py:515,516"]
    LIN2["SAM 3.1 licence unknown, obtained redistributed, strict=False load flag<br/>identify.py:593,594,595"]
    PATHS --> LIN1
    PATHS --> LIN2
    DEBT["DEBT: model locations are code constants, not configuration<br/>identify.py:438,446"]
    PATHS --- DEBT
    OPEN["OPEN: lineage says redistributed for a licence-unknown checkpoint while the README says nothing gated is redistributed<br/>identify.py:594<br/>README.md:49"]
    LIN2 --- OPEN
    DRIFT["DOC-DRIFT: the module header says carve_components splits instances, the driver default is carve_instances<br/>identify.py:20,21<br/>pipeline.py:544,475"]
    OBJS --- DRIFT
```

**What it shows.** The lineage entries written here are copied into every `objects.json` record (`pipeline.py:767,776`).

## VO-03.3 Rectify contract: distorted frame to a K-only pinhole domain

```mermaid
flowchart TB
    LOAD["load frame, resize to the COLMAP camera size if it differs<br/>pipeline.py:244,247,248"]
    MODEL{"camera model in UNDISTORTABLE?<br/>rectify.py:124"}
    WARN["warn and return the frame unchanged, is_noop True<br/>rectify.py:125,129"]
    NODIST{"has_distortion?<br/>rectify.py:130"}
    WARP["backward warp: ideal pixel, distort, sample source<br/>rectify.py:141,143,144"]
    REMAP["cv2.remap bilinear, replicate border<br/>rectify.py:149,151,152"]
    INSET["largest all-valid rectangle, at most 12 percent per side<br/>rectify.py:43,84,155"]
    KNEW["cx, cy shifted by the crop<br/>rectify.py:159,160,161"]
    PERSIST["rectified PNG plus K, is_noop and inset in intrinsics.json<br/>pipeline.py:262,263,266,269"]
    LOAD --> MODEL
    MODEL -->|no| WARN --> PERSIST
    MODEL -->|yes| NODIST
    NODIST -->|no| PERSIST
    NODIST -->|yes| WARP --> REMAP --> INSET --> KNEW --> PERSIST
    DEBT1["DEBT: FULL_OPENCV counts as undistortable but only k1, k2, k3, p1, p2 are applied, its k4 to k6 terms are dropped<br/>colmap_model.py:40,41,53,55<br/>rectify.py:63,65,70"]
    WARP --- DEBT1
    TENS1["TENSION: fisheye frames pass through undistorted yet every identify record defaults to domain rectified<br/>rectify.py:124<br/>identify.py:144"]
    WARN --- TENS1
    TENS2["TENSION: DESIGN and ADR-003 promise golden tests against uos ViewSet output, the test reference reuses the sidecar's own distort_normalised, only the cv2 oracle test is independent<br/>DESIGN.md:45,46<br/>adr-003-carve-emplace-texture.md:24,25<br/>test_rectify.py:20,33,114"]
    INSET --- TENS2
```

**What it shows.** All masks, crops and carve views downstream live in this rectified domain; the cache in VO-02.1 stores exactly these outputs.
