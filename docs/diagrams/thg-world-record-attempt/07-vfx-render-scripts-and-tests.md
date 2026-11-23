---
id: TH-07
title: The VFX render scripts that made the committed composite, and the test suite
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/docs/PRD-vfx-object-tracking.md]
adrs: []
sources:
  - ../THG-world-record-attempt/docs/PRD-vfx-object-tracking.md
  - ../THG-world-record-attempt/scripts/render_3d_pass.py
  - ../THG-world-record-attempt/scripts/render_tracked_composite.py
  - ../THG-world-record-attempt/exports/vfx-composite/composite_metadata.json
  - ../THG-world-record-attempt/exports/vfx-composite/render_3d_pass_meta.json
  - ../THG-world-record-attempt/src/vfx_pipeline/blender_scripts.py
  - ../THG-world-record-attempt/src/vfx_pipeline/pipeline.py
  - ../THG-world-record-attempt/tests/test_vfx_pipeline.py
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-07.1 How tracked_car_composite.mp4 was made

```mermaid
flowchart TB
    BLEND["KeenTools example CameraTrack.blend, already tracked, opened as the scene<br/>composite_metadata.json:5<br/>render_tracked_composite.py:4,13"]
    subgraph PASS1["Pass 1, render_3d_pass.py under Blender"]
        direction TB
        CLIP["relink the clip, fix frames 1 to 314 at 25 fps<br/>render_3d_pass.py:36,37,41,42"]
        MAT["camera-projected texture from ref_texture_frame.png, procedural paint fallback<br/>render_3d_pass.py:25,90,100,105,133"]
        LIGHT["sun 5.0, sky fill area 100, ground bounce area 30<br/>render_3d_pass.py:179,182,189,192,198,201"]
        RGBA["compositor off, transparent film, RGBA PNG frames render_NNNN<br/>render_3d_pass.py:218,227,229,231"]
        CLIP --> MAT --> LIGHT --> RGBA
    end
    META1["render_3d_pass_meta.json, camera projected material, 314 frames<br/>render_3d_pass.py:268<br/>render_3d_pass_meta.json:3,4"]
    PASS2["Pass 2, ffmpeg overlay of the RGBA frames on the source video, libx264 crf 18<br/>composite_metadata.json:18,20,21,22"]
    MP4["exports/vfx-composite/tracked_car_composite.mp4<br/>composite_metadata.json:7"]
    BLEND --> PASS1 --> META1
    RGBA --> PASS2 --> MP4
    DRIFT["DOC-DRIFT: the final metadata describes pass 1 as metallic teal with a 4.0 sun and a 200 fill, the script now sets a projected texture and three lights<br/>composite_metadata.json:15,16"]
    PASS2 --- DRIFT
    DEBT["DEBT: no committed script runs pass 2 or extracts ref_texture_frame.png, both exist only as outputs and metadata<br/>composite_metadata.json:19"]
    PASS2 --- DEBT
    DIV["DIVERGENCE: the PRD success criteria want the example files tracked end to end by the pipeline, the composite instead rests on KeenTools tracking data and python -m vfx_pipeline produced no part of it<br/>PRD-vfx-object-tracking.md:117,118<br/>composite_metadata.json:3,5"]
    MP4 --- DIV
```

**What it shows.** The committed VFX output is a two-pass render of the vendor's pre-tracked example scene, made outside the v2 pipeline package.

## TH-07.2 Three compositor approaches that disagree

```mermaid
flowchart TB
    A["v2 package compositor: MovieClip node, AlphaOver inputs addressed by name Image and Image_001<br/>blender_scripts.py:539,541,566,567"]
    B["render_tracked_composite.py: MovieClip node, AlphaOver inputs addressed by index 1 and 2<br/>render_tracked_composite.py:216,236,237,238"]
    C["render_3d_pass.py: compositor disabled, RGBA frames, overlay done outside Blender<br/>render_3d_pass.py:217,218"]
    A --> B --> C
    TENS1["TENSION: pass 1 says the MovieClip compositor node does not load video headless, the v2 package and the second script both depend on it<br/>render_3d_pass.py:3,4"]
    C --- TENS1
    TENS2["TENSION: the second script says Blender 5.x AlphaOver inputs are indexed, the package links Image_001 by name<br/>render_tracked_composite.py:236"]
    B --- TENS2
    DEBT["DEBT: frame range fixed at 314 in pass 1, taken from the clip in the other script<br/>render_3d_pass.py:41<br/>render_tracked_composite.py:70"]
    C --- DEBT
```

**What it shows.** The approach that worked, a transparent 3D pass overlaid by ffmpeg, was never folded back into the package's step 4.

## TH-07.3 What the test suite covers

```mermaid
flowchart TB
    IMP["imports only vfx_pipeline modules, through a sys.path edit<br/>test_vfx_pipeline.py:22,24,30,35,48"]
    subgraph PURE["Pure functions"]
        direction TB
        TV["validators, GLB, scene, tracking, composite<br/>test_vfx_pipeline.py:117,156,176,199"]
        TW["Hunyuan3D builder, reference load, image patch<br/>test_vfx_pipeline.py:231,279,297"]
        TB2["Blender script strings and v1 regressions<br/>test_vfx_pipeline.py:321,656"]
    end
    subgraph MOCKED["Orchestration with mocks"]
        direction TB
        TP["VFXPipeline with a MagicMock ComfyUI client and a patched BlenderClient<br/>test_vfx_pipeline.py:401,404,441"]
    end
    subgraph EXAMPLE["KeenTools example files, skipped when absent"]
        direction TB
        TK["file presence, GLB header, workflow node types, sizes<br/>test_vfx_pipeline.py:569,578,583"]
    end
    IMP --> PURE --> MOCKED --> EXAMPLE
    TENS["TENSION: the multi-pin test formats the track script with three frames and asserts only button names, never that pins are used<br/>test_vfx_pipeline.py:362,364,367,368"]
    TB2 --- TENS
    DEBT["DEBT: step 1 test mocks get_output_path to return a local GLB, hiding the ComfyUI-relative path<br/>test_vfx_pipeline.py:415"]
    TP --- DEBT
    GAP["no test imports the render scripts, the campaign scripts or the Veo scripts"]
    EXAMPLE --- GAP
```

**What it shows.** Coverage stops at the package boundary; every script that produced a committed asset is untested.
