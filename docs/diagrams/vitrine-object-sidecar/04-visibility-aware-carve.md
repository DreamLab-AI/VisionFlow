---
id: VO-04
title: Visibility-aware carve, from 2D masks to per-object Gaussian subsets
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/decisions/adr-003-carve-emplace-texture.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
adrs: [vitrine-object-sidecar:ADR-003]
sources:
  - ../vitrine-object-sidecar/sidecar/carve.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/splat_ply.py
  - ../vitrine-object-sidecar/tests/test_integration_carve.py
  - ../vitrine-object-sidecar/tools_recarve_and_score.py
  - ../vitrine-object-sidecar/README.md
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-04.1 One view's vote and the aggregate label

```mermaid
flowchart TB
    PROJ["project Gaussian centres with R, t and pinhole K, behind camera or off image abstains<br/>carve.py:89,91,97,98"]
    ZBUF["per-pixel nearest depth over projected centres<br/>carve.py:114,115,116"]
    FRONT["front if depth within band of the pixel's nearest<br/>carve.py:119"]
    FG["in mask and front: foreground vote<br/>carve.py:121,124"]
    BG["inside image, outside mask: background vote<br/>carve.py:122,125"]
    AB["in mask but deeper than the band: abstain<br/>carve.py:123"]
    AGG["foreground iff fg votes reach min_views and fg over fg plus bg beats fg_threshold<br/>carve.py:229,231,232"]
    CC["keep the largest 26-connected voxel component<br/>carve.py:128,236,242"]
    EXT["median centroid, p5 to p95 extents<br/>carve.py:254,255,256"]
    PROJ --> ZBUF --> FRONT
    FRONT --> FG
    FRONT --> AB
    PROJ --> BG
    FG --> AGG
    BG --> AGG
    AGG --> CC --> EXT
    INV["INVARIANT: an occluded Gaussian inside the mask never votes foreground, the fix for the naive mask_projector<br/>carve.py:3,4,5,121,123"]
    AB --- INV
```

**What it shows.** The depth band is what separates the visible surface from what sits behind it along the same ray; robust extents keep a few stray Gaussians from inflating the object's scale.

## VO-04.2 Three entry points, one used by the driver

```mermaid
flowchart TB
    C1["carve: single object, largest component<br/>carve.py:183,236"]
    C2["carve_components: one result per component above a floor<br/>carve.py:439,470,485"]
    C3["carve_instances: detection-anchored tracks<br/>carve.py:550,554,557"]
    PV["_carve_prevote: shared vote for carve_components<br/>carve.py:349,381,387"]
    AUTO["auto band 2 percent of the robust p5 to p95 diagonal, only when depth_band is None<br/>carve.py:211,218,219,223"]
    DRV["driver default per_instance True, legacy path when False<br/>pipeline.py:460,475,494,500"]
    BAND["driver band is depth_band_frac times scene_scale, always explicit<br/>pipeline.py:628"]
    C2 --> PV
    C1 --> AUTO
    PV --> AUTO
    DRV --> C3
    DRV --> C2
    BAND --> DRV
    DEBT1["DEBT: _carve_prevote says it is factored so the variants cannot drift, yet carve repeats the vote and band code inline, and the union-find exists twice<br/>carve.py:358,359,360<br/>carve.py:211,226,232<br/>carve.py:128,288"]
    PV --- DEBT1
    TENS["TENSION: the robust auto band that fixed the nc04 outlier inflation is reached only by direct callers such as the recarve tool, never by the driver<br/>tools_recarve_and_score.py:43<br/>pipeline.py:628"]
    AUTO --- TENS
    TENS2["TENSION: README carve IoU 0.928 comes from the single-object carve on projected cheat masks, the integration test gates only above 0.6, and the driver runs carve_instances<br/>README.md:54<br/>test_integration_carve.py:3,108,119"]
    C1 --- TENS2
```

**What it shows.** Only `carve_instances` runs in a default pass; the other two serve the legacy flag, the tests and the root tools.

## VO-04.3 carve_instances as the driver calls it

```mermaid
flowchart TB
    CAP["driver caps views per label at 80, evenly spaced, frame ids kept aligned<br/>pipeline.py:470,471,472,474"]
    CALL["min_views is min of carve_min_votes and view count, cluster_radius 0.04 times scene_scale<br/>pipeline.py:477,478,637"]
    ONCE["project each view once, cache visible set and front mask<br/>carve.py:592,593,598,599"]
    ANCH["anchor per detection: median of front, in-mask centres, needs 3 or more<br/>carve.py:600,602,603,604"]
    CLUS["union-find anchors closer than cluster_radius into tracks<br/>carve.py:624,625,627,629"]
    TOPK["keep the largest tracks, cap max_instances or 24<br/>carve.py:639,640"]
    BGV["shared background: pixels outside every kept-track mask<br/>carve.py:660,666,671,673"]
    BEST["each Gaussian goes to the track with most fg votes<br/>carve.py:676,685,686,687"]
    GATE["track gates: enough views, enough points, conf above threshold<br/>carve.py:695,697,698"]
    VOX["voxel size coarsened to at least diag over 55, then largest component<br/>carve.py:708,709,710"]
    OUT["ObjectCandidate per instance, support frames from view_indices<br/>pipeline.py:487,488,489"]
    PLY["subset PLY written per candidate<br/>pipeline.py:642,647,648"]
    CAP --> CALL --> ONCE --> ANCH --> CLUS --> TOPK --> BGV --> BEST --> GATE --> VOX --> OUT --> PLY
    NOTE1["detections whose track was dropped by the cap are skipped before the coverage union, so their pixels count as background<br/>carve.py:662,663,664,666"]
    BGV --- NOTE1
    RD["reader needs x, y, z, other splat properties default<br/>splat_ply.py:18,19,63"]
    PLY --- RD
```

**What it shows.** Instances are separated by clustering per-detection 3D anchors before voting, so two touching objects with one label each carve from their own masks.
