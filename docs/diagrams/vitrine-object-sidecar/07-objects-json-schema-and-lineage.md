---
id: VO-07
title: objects.json (vitrine/object/1), its validator and checkpoint lineage
area: vitrine-object-sidecar
governing:
  - ../vitrine-object-sidecar/docs/decisions/ddd-001-domain-model.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
adrs: [vitrine-object-sidecar:DDD-001, vitrine-object-sidecar:ADR-002]
sources:
  - ../vitrine-object-sidecar/sidecar/manifest.py
  - ../vitrine-object-sidecar/sidecar/pipeline.py
  - ../vitrine-object-sidecar/sidecar/identify.py
  - ../vitrine-object-sidecar/docs/decisions/ddd-001-domain-model.md
  - ../vitrine-object-sidecar/docs/DESIGN.md
verified_commit: {vitrine-object-sidecar: 8011b32a57d183dd40af38fcf7bfcf952167603d}
---

## VO-07.1 The record model

```mermaid
classDiagram
    class ObjectRecord {
        label str manifest.py:115
        stage_scores dict manifest.py:116
        carve CarveStats manifest.py:117
        placement PlacementTransform manifest.py:118
        assets list of AssetRef manifest.py:119
        checkpoint_lineage list manifest.py:120
        vram_peaks dict manifest.py:121
        schema vitrine/object/1 manifest.py:122
    }
    class CarveStats {
        n_foreground, n_prevote, n_removed_by_cc, n_views manifest.py:85
        centroid, extent_size, depth_band manifest.py:89
    }
    class PlacementTransform {
        translation, uniform scale, 3x3 rotation manifest.py:101
        up_source exif or support-plane or config manifest.py:104
        orientation_observability dict manifest.py:105
    }
    class AssetRef {
        role, path, sha256 manifest.py:72
        domain rectified or distorted or world manifest.py:75
    }
    class CheckpointLineage {
        repo, commit, checkpoint_hash manifest.py:58
        licence from LICENCE_CLASSES manifest.py:61
        obtained from OBTAINED_MODES manifest.py:62
    }
    ObjectRecord --> CarveStats
    ObjectRecord --> PlacementTransform
    ObjectRecord --> "1..n" AssetRef
    ObjectRecord --> "0..n" CheckpointLineage
    note for ObjectRecord "DOC-DRIFT: DDD-001 puts Candidate, TextureBake,<br/>FinishStats and residuals in the aggregate and a<br/>colmap_world frame tag on world quantities, the<br/>record has none of them<br/>ddd-001-domain-model.md:27,28,29,36,37"
```

**What it shows.** The schema is plain JSON checked by a hand-written validator with no third-party dependency (`manifest.py:11,12,13`).

## VO-07.2 How the driver fills and writes objects.json

```mermaid
flowchart TB
    LIN["identify lineage copied into every record<br/>pipeline.py:767,776"]
    COERCE["licence not in LICENCE_CLASSES becomes unknown, obtained not in OBTAINED_MODES becomes generated<br/>pipeline.py:771,772,773,774"]
    ASSET["one asset: object_ply, sha256 of the file, domain world<br/>pipeline.py:786,787,788,789"]
    PLACE["placement: carve centroid or zeros, scale 1.0, identity, observability empty<br/>pipeline.py:792,793,794,796"]
    SCORES["stage_scores: carve_confidence, n_views, seed_lane_a<br/>pipeline.py:805,806,807,808"]
    CONF["carve_confidence is mean fg over fg plus bg on the object's Gaussians<br/>pipeline.py:749,758,759"]
    VR["vram_peaks is the run-wide identify peak, the same in every record<br/>pipeline.py:615,814"]
    ONLY["record only when an asset exists<br/>pipeline.py:810"]
    VAL["write_manifest validates, refuses to write an invalid document<br/>manifest.py:229,230,232"]
    REJ["re-read, add rejected, rewrite without re-validation<br/>pipeline.py:835,836,837"]
    LIN --> COERCE --> ONLY
    ASSET --> ONLY
    PLACE --> ONLY
    SCORES --> ONLY
    CONF --> SCORES
    VR --> ONLY
    ONLY --> VAL --> REJ
    TENS["TENSION: LICENCE_CLASSES says an unknown licence is a hard error that must be resolved, yet unknown is an allowed class and the driver coerces any unrecognised licence to it<br/>manifest.py:25,26,30<br/>pipeline.py:771,772"]
    COERCE --- TENS
    INV["INVARIANT: every asset carries a 64-hex sha256 and every lineage entry a known licence and obtained mode<br/>manifest.py:195,204,206"]
    VAL --- INV
    DRIFT["DOC-DRIFT: DESIGN manifest promises sha256 of every checkpoint, coverage, orientation observability and stage VRAM peaks, a driver record has no coverage and an empty observability map<br/>DESIGN.md:84,85,86,87<br/>pipeline.py:796"]
    PLACE --- DRIFT
```

**What it shows.** The validator gates shape and hashes, not truth: a record's placement in a CLI run is the carve initialisation, and its lineage is whatever the identify models reported.

## VO-07.3 The three JSON documents a run leaves

```mermaid
flowchart LR
    IDJ["identify.json, schema vitrine/identify/1: config, lineage, notes, frames, instances<br/>identify.py:179,180,181,183,187"]
    OBJJ["objects.json, schema vitrine/object/1: run, objects, rejected<br/>manifest.py:223,224,225<br/>pipeline.py:836"]
    SUMJ["summary.json, schema vitrine/object-pass/1: scene scale, config, frames, objects, rejected, timings, VRAM<br/>pipeline.py:844,846,858,867,869,870,871"]
    IDJ -->|labels and masks feed carve| OBJJ
    OBJJ -->|object table mirrors records| SUMJ
    RH["a later stage re-run rehydrates identify.json into an IdentifyResult<br/>pipeline.py:875,877,881"]
    IDJ --- RH
```

**What it shows.** Only `summary.json` records the scene scale every threshold depended on (`pipeline.py:846`); `objects.json` does not.
