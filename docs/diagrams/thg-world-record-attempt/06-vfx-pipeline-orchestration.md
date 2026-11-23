---
id: TH-06
title: VFX tracking pipeline v2, steps, clients and the Hunyuan3D graph
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/docs/PRD-vfx-object-tracking.md]
adrs: []
sources:
  - ../THG-world-record-attempt/docs/PRD-vfx-object-tracking.md
  - ../THG-world-record-attempt/src/vfx_pipeline/pipeline.py
  - ../THG-world-record-attempt/src/vfx_pipeline/comfyui_client.py
  - ../THG-world-record-attempt/src/vfx_pipeline/blender_client.py
  - ../THG-world-record-attempt/src/vfx_pipeline/blender_scripts.py
  - ../THG-world-record-attempt/src/vfx_pipeline/validators.py
  - ../THG-world-record-attempt/src/vfx_pipeline/workflows.py
  - ../THG-world-record-attempt/tests/test_vfx_pipeline.py
  - ../THG-world-record-attempt/assets/keentools-example/ComfyUI+GeoTracker_example/3dmodelgeneration.json
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-06.1 What run does, step by step

```mermaid
flowchart TB
    CLI["CLI flags video, screenshot or glb, pins, transform, engine, skip-to, both service URLs<br/>pipeline.py:531,532,533,539,543,547,553,554,555"]
    DIR["create the output directory<br/>pipeline.py:432"]
    S1{"no glb given and not skipping past step 1<br/>pipeline.py:448"}
    GEN["step 1 Hunyuan3D generation, abort on failure<br/>pipeline.py:451,453,454"]
    VG["validate the GLB header, abort if invalid<br/>pipeline.py:461,462,463"]
    S2["step 2 scene setup, abort on failure<br/>pipeline.py:467,468,470,471"]
    S3["step 3 tracking, result recorded, never aborts<br/>pipeline.py:474,475,477"]
    S4["step 4 composite and render, result recorded<br/>pipeline.py:480,481"]
    META["pipeline_metadata.json with each step ok, message, duration<br/>pipeline.py:493,501,502,506"]
    CLI --> DIR --> S1
    S1 -->|yes| GEN --> VG
    S1 -->|no| VG
    VG --> S2 --> S3 --> S4 --> META
    FO1["DEBT: step 2 returns ok whatever the footage, GeoTracker, analysis and state scripts print<br/>pipeline.py:224,243,248,253,260"]
    S2 --- FO1
    FO2["DEBT: a script output with no RESULT line passes both scene and tracking validation<br/>validators.py:55,76"]
    S2 --- FO2
    FO3["DEBT: tracking validation failure only prints a warning<br/>pipeline.py:326,328,329"]
    S3 --- FO3
```

**What it shows.** Only step 1, the GLB check and step 2's reset and import can stop a run; tracking and render outcomes are recorded, not enforced.

## TH-06.2 The two clients and the Blender script sequence

```mermaid
sequenceDiagram
    autonumber
    participant P as VFXPipeline<br/>pipeline.py:87
    participant C as ComfyUIClient<br/>comfyui_client.py:13
    participant B as BlenderClient<br/>blender_client.py:8
    participant BS as Blender MCP addon
    P->>C: upload the screenshot unless it already sits under the ComfyUI root (:127,129)
    C->>C: multipart POST /upload/image (:55,67,71)
    P->>C: run_workflow, POST /prompt then poll /history every 3 s (comfyui_client.py:77,78,89,90,99)
    C-->>P: outputs, then node 70 textured or node 24 untextured GLB path (pipeline.py:145,146,149)
    Note over C: get_output_path returns subfolder slash filename, a ComfyUI-relative name<br/>comfyui_client.py:114,115,117
    Note over P: OPEN: validate_glb opens that name as a local file, the unit test mocks the path to a local GLB<br/>validators.py:19,20<br/>test_vfx_pipeline.py:415
    P->>C: POST /free to unload models before Blender work (:206)
    P->>B: open a WebSocket per step (:213,299,366)
    B->>BS: JSON id, tool execute_python, params code (:41,42,43,59)
    BS-->>B: status and data with stdout, error status raises (blender_client.py:52,53,54,60,61)
    Note over P: step 2 runs reset, footage, import, GeoTracker setup, analyse, state check<br/>pipeline.py:216,224,233,243,248,253
    Note over P: step 3 runs align, track and refine, save blend<br/>pipeline.py:303,316,335
    Note over P: step 4 runs compositor setup, then render with a one hour timeout<br/>pipeline.py:369,380
```

**What it shows.** Each Blender step is Python source sent as one `execute_python` call; results come back only as printed stdout that the validators parse.

## TH-06.3 The Hunyuan3D v2 graph built when no reference workflow is given

```mermaid
flowchart TB
    LOAD["LoadImage then ImageResize+ pad to 512<br/>workflows.py:70,74,77"]
    BG["TransparentBGSession+ and ImageRemoveBackground+<br/>workflows.py:87,91"]
    DIT["Hy3DModelLoader dit-v2-0, Hy3DGenerateMesh 50 steps, guidance 5.5<br/>workflows.py:99,101,107,112,113"]
    MESH["Hy3DVAEDecode octree 512, PostprocessMesh to 25000 faces<br/>workflows.py:119,123,130,136"]
    EXP1["node 24 Hy3DExportMesh untextured<br/>workflows.py:141,142"]
    DL["Hy3DDelightImage on the masked input<br/>workflows.py:152,175"]
    UV["MeshUVWrap and RenderMultiView<br/>workflows.py:189,193"]
    PAINT["paint-v2-0 SampleMultiView, Remacri 4x upscale<br/>workflows.py:203,207,223,226"]
    BAKE["BakeFromMultiview, vertex inpaint, CV2 inpaint, ApplyTexture<br/>workflows.py:234,241,249,258"]
    EXP2["node 70 Hy3DExportMesh textured<br/>workflows.py:265,266"]
    LOAD --> BG --> DIT --> MESH
    MESH --> EXP1
    MESH --> UV --> PAINT
    DL --> PAINT --> BAKE --> EXP2
    REF["reference path: load a workflow file and patch its first LoadImage<br/>workflows.py:11,21,30,37"]
    OPEN["OPEN: a UI-format file with last_node_id is patched and posted to /prompt unchanged<br/>workflows.py:27,32<br/>pipeline.py:134,135,142"]
    REF --- OPEN
    KT["the only reference file in the tree has no node 70 and its node 24 is a VAE decode, so it reaches the GLB only through the scan of every output<br/>3dmodelgeneration.json:1<br/>pipeline.py:150,152,154"]
    OPEN --- KT
```

**What it shows.** Step 1 prefers node 70 then node 24 and falls back to the first `.glb` or `.gltf` output from any node.

## TH-06.4 What the PRD asks for against what the code does

```mermaid
flowchart TB
    FR41["FR-4.1 accept vertex groups to mask from tracking<br/>PRD-vfx-object-tracking.md:55"]
    M41["CREATE_SURFACE_MASK exists but run never imports it, and no CLI flag takes a group<br/>blender_scripts.py:258<br/>pipeline.py:42,53"]
    FR51["FR-5.1 pin at user frames or auto-spaced frames<br/>PRD-vfx-object-tracking.md:60"]
    M51["pin frames are parsed into a list and never read again<br/>blender_scripts.py:369,373"]
    FR54["FR-5.4 track to end and to start<br/>PRD-vfx-object-tracking.md:63"]
    M54["only track_to_end_btn is called<br/>blender_scripts.py:409"]
    NF1["NFR-1 headless, no GUI interaction<br/>PRD-vfx-object-tracking.md:75"]
    MN1["align and track raise when no 3D viewport exists, analysis falls back to precalcless<br/>blender_scripts.py:334,335,395,396,226"]
    NF2["NFR-2 each step validates before the next<br/>PRD-vfx-object-tracking.md:76"]
    MN2["step 3 and step 4 never stop the run<br/>pipeline.py:477"]
    FR41 --> M41
    FR51 --> M51
    FR54 --> M54
    NF1 --> MN1
    NF2 --> MN2
    DRIFT["DOC-DRIFT: the module docstring claims multi-keyframe pin support added in v2<br/>blender_scripts.py:11"]
    M51 --- DRIFT
    DEBT["DEBT: paths and names are pasted into Blender source inside double quotes, a quote in a path breaks the script<br/>blender_scripts.py:53,80,81"]
    M41 --- DEBT
```

**What it shows.** Five PRD requirements are unmet or contradicted in code, each marked against the line that decides it.
