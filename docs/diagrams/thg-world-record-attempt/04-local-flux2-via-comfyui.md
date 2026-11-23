---
id: TH-04
title: Local Flux 2 generation through ComfyUI and the committed workflow files
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/configs/AGENT_NOTICE.md]
adrs: []
sources:
  - ../THG-world-record-attempt/README.md
  - ../THG-world-record-attempt/configs/AGENT_NOTICE.md
  - ../THG-world-record-attempt/configs/pipeline.json
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_flux2_riffs.py
  - ../THG-world-record-attempt/configs/workflows/flux2-multigpu-campaign.json
  - ../THG-world-record-attempt/configs/workflows/flux2-img2img-workflow.json
  - ../THG-world-record-attempt/configs/workflows/flux2-optimal-workflow.json
  - ../THG-world-record-attempt/configs/workflows/nano-banana-garment-reskin.json
  - ../THG-world-record-attempt/configs/workflows/nano-banana-repose.json
  - ../THG-world-record-attempt/configs/workflows/veo3-branded-video.json
  - ../THG-world-record-attempt/configs/workflows/sam3-video-segmentation.json
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-04.1 Submit, poll and download one Flux 2 image

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>_generate_flux2_riffs.py:273
    participant W as make_workflow<br/>_generate_flux2_riffs.py:15
    participant CU as ComfyUI host comfyui port 8188
    participant FS as assets/scene-riffs/variations
    M->>W: prompt, seed, width, height, 20 steps per entry (:291,292,293,294,295,296)
    W-->>M: API-format graph of nine nodes (_generate_flux2_riffs.py:17,85)
    M->>CU: POST /prompt with the graph, read prompt_id (_generate_flux2_riffs.py:97,100)
    loop every 3 s up to 600 s
        M->>CU: GET /history for prompt_id (_generate_flux2_riffs.py:104,106,128)
    end
    CU-->>M: status error, or the first images output (_generate_flux2_riffs.py:115,121,123)
    M->>CU: GET /view with filename, subfolder, type (_generate_flux2_riffs.py:136,141)
    M->>FS: write flux2_NAME.png (_generate_flux2_riffs.py:143,144)
    Note over M: exit status 1 unless all eight succeed (:326)
```

**What it shows.** The only script that drives ComfyUI posts an API-format graph it builds itself and polls history; it never loads a committed workflow file.

## TH-04.2 The graph it builds and where each model sits

```mermaid
flowchart TB
    UNET["UNETLoaderMultiGPU, flux2_dev_fp8mixed, device cuda 0<br/>_generate_flux2_riffs.py:19,21,23"]
    CLIP["CLIPLoaderMultiGPU, Mistral 3 Small text encoder, device cuda 1<br/>_generate_flux2_riffs.py:27,29,31"]
    POS["positive prompt text<br/>_generate_flux2_riffs.py:35,37"]
    NEG["empty negative prompt<br/>_generate_flux2_riffs.py:42,44"]
    LAT["EmptySD3LatentImage at the requested size<br/>_generate_flux2_riffs.py:49,51,52"]
    KS["KSampler, cfg 1.0, euler, simple, denoise 1.0<br/>_generate_flux2_riffs.py:57,65,66,67,68"]
    VAE["VAELoaderMultiGPU, flux2-vae, device cuda 1<br/>_generate_flux2_riffs.py:72,74,75"]
    DEC["VAEDecode then SaveImage prefix flux2_riff<br/>_generate_flux2_riffs.py:79,86,89"]
    CLIP --> POS --> KS
    CLIP --> NEG --> KS
    UNET --> KS
    LAT --> KS
    KS --> DEC
    VAE --> DEC
    INV["INVARIANT: UNet on GPU 0, text encoder and VAE on GPU 1, because the encoder alone filled GPU 0<br/>AGENT_NOTICE.md:4,34"]
    VAE --- INV
    TENS["TENSION: there is no LoadImage node, so garment fidelity rests on the prompt text alone<br/>_generate_flux2_riffs.py:159,160,161"]
    POS --- TENS
    DRIFT["DOC-DRIFT: the ledger records phase 1 at 25 steps and 768x1024, the committed phase 1 workflow says 20 steps at 1024x1024<br/>pipeline.json:18,22<br/>flux2-multigpu-campaign.json:83,104"]
    KS --- DRIFT
```

**What it shows.** The two-GPU split is fixed in the graph by device name, not negotiated by ComfyUI at run time.

## TH-04.3 Committed workflow files and what uses them

```mermaid
flowchart TB
    subgraph UI["UI-format workflow JSON in configs/workflows"]
        direction TB
        F1["Flux 2 multi-GPU campaign, devices cuda 0 and cuda 1<br/>flux2-multigpu-campaign.json:7,17,32,119"]
        F2["Flux 2 img2img with LoadImage and ReferenceLatent<br/>flux2-img2img-workflow.json:83,280"]
        F3["Flux 2 with a LoRA loader<br/>flux2-optimal-workflow.json:388"]
        G1["GeminiImageNode reskin, gemini-2.5-flash-image<br/>nano-banana-garment-reskin.json:57,73"]
        G2["GeminiImageNode repose<br/>nano-banana-repose.json:57,73"]
        V1["Veo3VideoGenerationNode then SaveVideo, model veo-3.1-generate<br/>veo3-branded-video.json:7,32,38"]
        S1["SAM 3D model loader and segmentation<br/>sam3-video-segmentation.json:47"]
    end
    USERS["named by no script in the tree"]
    F1 --> USERS
    F2 --> USERS
    F3 --> USERS
    G1 --> USERS
    G2 --> USERS
    V1 --> USERS
    S1 --> USERS
    DRIFT["DOC-DRIFT: the README table lists four workflows as agent-generated and loadable, the folder holds seven and no script names any of them<br/>README.md:186,219,223,226"]
    USERS --- DRIFT
```

**What it shows.** The workflow files are artefacts for a human to open in ComfyUI; the scripts that produced assets used REST calls or a self-built graph instead.
