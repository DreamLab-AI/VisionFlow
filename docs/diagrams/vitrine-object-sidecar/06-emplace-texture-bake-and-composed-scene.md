---
id: VO-06
title: Emplacement, real-frame texture bake and the composed scene
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/decisions/adr-003-carve-emplace-texture.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
adrs: [vitrine-object-sidecar:ADR-003]
sources:
  - ../vitrine-object-sidecar/sidecar/emplace.py
  - ../vitrine-object-sidecar/sidecar/texture.py
  - ../vitrine-object-sidecar/sidecar/compose.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/manifest.py
  - ../vitrine-object-sidecar/sidecar/colmap_model.py
  - ../vitrine-object-sidecar/tests/test_compose_scene.py
  - ../vitrine-object-sidecar/docs/DESIGN.md
  - ../vitrine-object-sidecar/docs/decisions/adr-003-carve-emplace-texture.md
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-06.1 7-DoF silhouette emplacement and scene-up

```mermaid
flowchart TB
    SIL["silhouette is the OR of every in-front projected triangle, filled one at a time on the CPU<br/>emplace.py:80,90,105,106"]
    OBJ["objective: mean IoU of rendered silhouette vs real mask over all views<br/>emplace.py:117,119,120"]
    INIT["initial scale from carve extent over mesh extent, or a scale_center override<br/>emplace.py:216,219,220"]
    GRID["grid: 24 yaws about up, 5 scales, 7 translation jitters<br/>emplace.py:221,222,223,230"]
    POW["Powell refine of rotvec, translation, log-scale from the top seeds<br/>emplace.py:243,246,254"]
    AMB["ambiguity list: other optima within eps and more than 5 degrees away<br/>emplace.py:266,273,274"]
    OBS["per-axis observability by finite-difference sensitivity<br/>emplace.py:277,287,288"]
    TF["4x4 with scale, world equals s R (v minus pivot) plus t<br/>emplace.py:290,292,293"]
    UP["scene-up: RANSAC plane, normal turned toward the point centroid<br/>emplace.py:127,151,169,170"]
    INIT --> GRID --> POW --> AMB --> TF
    POW --> OBS
    SIL --> OBJ --> GRID
    UP --> GRID
    TENS1["TENSION: DESIGN and emplace say scene-up is never assumed +Y, yet both the estimator and the driver fall back to +Y<br/>DESIGN.md:72,73<br/>emplace.py:140,144,165<br/>pipeline.py:736,737"]
    UP --- TENS1
    DRIFT1["DOC-DRIFT: the estimator docstring orients the normal so most points lie below it, the code orients it toward the centroid<br/>emplace.py:137,138,167,170"]
    UP --- DRIFT1
    TENS2["TENSION: ADR-003 refines by silhouette plus photometric alignment, the objective is silhouette IoU only<br/>adr-003-carve-emplace-texture.md:13,14<br/>emplace.py:119"]
    OBJ --- TENS2
```

**What it shows.** The CLI never calls this fit; its placement stays the carve centroid with identity rotation (`pipeline.py:739,740,745`), so these transforms exist only where a tool or test runs `emplace`.

## VO-06.2 Projective texture bake and per-texel provenance

```mermaid
flowchart TB
    UV["unwrap: xatlas when importable, else a per-triangle grid atlas<br/>texture.py:84,85,86,93,94"]
    RAST["rasterise the atlas into per-texel position, normal, island<br/>texture.py:141,170,173,174"]
    DEPTH["per view: mesh z-buffer<br/>texture.py:197,298"]
    FACE["two-sided facing, abs cos above facing_min<br/>texture.py:315,316"]
    OCC["occlusion: texel depth within eps of the z-buffer<br/>texture.py:322,323"]
    W["weight is facing times frame sharpness, colours blended across views<br/>texture.py:328,331,332"]
    PROV["codes: EMPTY, OBSERVED one view, BLENDED two or more, UNOBSERVED in-island unseen<br/>texture.py:42,43,44,45,345,346,347"]
    FILL["unseen texels: neutral grey by default, cv2.inpaint only when asked<br/>texture.py:265,350,354,357"]
    OUTF["albedo and provenance PNGs, GLB with V flipped, coverage stats<br/>texture.py:391,407,422,430"]
    UV --> RAST --> DEPTH --> FACE --> OCC --> W --> PROV --> FILL --> OUTF
    INV["INVARIANT: inpainted texels stay tagged UNOBSERVED so a later generative pass can replace them in place<br/>texture.py:351,352"]
    FILL --- INV
    DRIFT["DOC-DRIFT: the module header names a FILLED code and a cv2.inpaint generative fill step, the codes have no FILLED and the default is neutral fill<br/>texture.py:17,18,19,20<br/>texture.py:42,45,357"]
    PROV --- DRIFT
```

**What it shows.** Observed colour comes only from real rectified frames; anything no camera saw is flagged, never painted as if observed.

## VO-06.3 Proof renders and composed_scene.glb

```mermaid
flowchart TB
    PROOF["compose_view: real frame or projected scene points, object silhouette blended and outlined<br/>compose.py:45,59,67,82,86"]
    CONV["colmap_to_gltf_matrix: rotate estimated up onto +Y, divide by scene_scale<br/>compose.py:123,140,141,143"]
    GATE1["provenance gate: geometry, placement, material class and sha256 required<br/>compose.py:153,179,182,185"]
    NODE["one node per object, transform is basis times placement<br/>compose.py:266,269,270"]
    WRITE["trimesh writes the GLB<br/>compose.py:273"]
    GATE2["basis gate: units, handedness, axes, scene_scale, matrices<br/>compose.py:155,275,276,277"]
    EXTRAS["pygltflib injects node, scene and asset extras, derivative_class composed-derivative<br/>compose.py:280,284,290,296"]
    PROOF -.-> CONV
    GATE1 --> NODE --> WRITE --> GATE2 --> EXTRAS
    CONV --> NODE
    TENS["TENSION: SceneObject needs provenance and sha256 in its record, ObjectRecord emits neither, only the test's hand-built record has them<br/>compose.py:174,179,185<br/>manifest.py:124,125,133<br/>test_compose_scene.py:13,19,22"]
    GATE1 --- TENS
    OPEN["OPEN: the basis declares units metre after dividing by scene_scale, half the camera-centre diagonal of an arbitrary-scale reconstruction<br/>compose.py:195,141<br/>colmap_model.py:302,304,311"]
    GATE2 --- OPEN
    DEBT["DEBT: the basis gate runs after the GLB is written, so a refusal leaves the file on disk, and pygltflib is imported unguarded<br/>compose.py:251,273,278"]
    WRITE --- DEBT
```

**What it shows.** The composed scene keeps COLMAP-world placement recoverable by inverting the declared basis; no tracked code outside the tests calls the exporter (see VO-01.3).
