---
id: TH-03
title: Gemini image scripts, scene clean, composites, riffs and repose
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/README.md]
adrs: []
sources:
  - ../THG-world-record-attempt/README.md
  - ../THG-world-record-attempt/scripts/clean_scenes.py
  - ../THG-world-record-attempt/scripts/generate_composites.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_riffs.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_expanded_riffs.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch1_repose.py
  - ../THG-world-record-attempt/task-two-repose/scripts/repose_batch2.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch3_repose.py
  - ../THG-world-record-attempt/configs/workflows/nano-banana-repose.json
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-03.1 One generateContent call, the shape every image script shares

```mermaid
sequenceDiagram
    autonumber
    participant GC as generate<br/>generate_composites.py:30
    participant PIL as Pillow resize<br/>generate_composites.py:18
    participant API as Gemini 2.5 Flash Image
    participant FS as assets/scene-riffs/composited
    GC->>PIL: garment at 1024 px long edge, scene at 1536 px (:31,32)
    PIL-->>GC: base64 and mime type, JPEG for .jpg else PNG (generate_composites.py:25,27)
    GC->>API: POST generateContent, inlineData garment, inlineData scene, text prompt, TEXT and IMAGE modalities, fixed seed (:34,35,36,37,38,40,44)
    API-->>GC: candidates, first candidate content parts (generate_composites.py:63)
    GC->>FS: first inlineData part decoded and written as the output PNG (:64,65,66,67)
    Note over GC: no image part, print any text part, retry with seed plus one (:73,74,75,80)
    Note over API: DEBT: the key rides in the URL query, never a header<br/>generate_composites.py:9
```

**What it shows.** Scene cleaning, composites, riffs and repose all build this one-or-two-image request and save the first image part.

## TH-03.2 Retry and fallback, script by script

```mermaid
flowchart TB
    subgraph CLEAN["clean_scenes.py, three scenes"]
        direction TB
        CL1["loop of 3 attempts, 429 waits 30 s times attempt<br/>clean_scenes.py:83,88,89,90,91,92"]
        CL2["error body, no candidates or no image all retry<br/>clean_scenes.py:104,113,140"]
    end
    subgraph COMP["generate_composites.py, nine composites"]
        direction TB
        CO1["recursion, 429 up to 4 attempts, other HTTP errors up to 3<br/>generate_composites.py:45,49,54,56"]
        CO2["no image retries with seed plus one, so the stored seed is not the seed that produced the file<br/>generate_composites.py:80,100"]
    end
    subgraph RIFF["_generate_riffs.py, ten riffs"]
        direction TB
        RI1["tries a preview model, then the stable model<br/>_generate_riffs.py:26,27,28"]
        RI2["one 15 s 429 retry per model, then a second pass over failures<br/>_generate_riffs.py:41,43,44,157,165"]
    end
    subgraph EXP["_generate_expanded_riffs.py, fifteen riffs"]
        direction TB
        EX1["no status check, one retry after 3 s<br/>_generate_expanded_riffs.py:34,35,120,123"]
    end
    subgraph REP["repose batches, 28 images"]
        direction TB
        RE1["batch 1 raises on HTTP error, no retry<br/>batch1_repose.py:75,76,81,82"]
        RE2["batches 2 and 3 no retry, batch 3 has no exception handling at all<br/>repose_batch2.py:41,67<br/>batch3_repose.py:39,40"]
    end
    CLEAN --> COMP --> RIFF --> EXP --> REP
    DEBT1["DEBT: ThreadPoolExecutor and generate_one are defined but the loop is sequential<br/>_generate_riffs.py:7,119,137,143"]
    RIFF --- DEBT1
    DEBT2["DEBT: the function name says pro then flash, the docstring and list say flash preview then flash stable<br/>_generate_riffs.py:24,25"]
    RIFF --- DEBT2
    DEBT3["DEBT: batch 1 takes a seed per image but never puts it in the payload<br/>batch1_repose.py:32,65,71,135"]
    REP --- DEBT3
```

**What it shows.** Five scripts, five different retry policies; none shares a helper, so behaviour under rate limiting differs per phase.

## TH-03.3 Mannequin repose, how thirty source images were split

```mermaid
flowchart TB
    SRC["source folder Posed Mannequins, 28 outfits plus two pose references<br/>batch1_repose.py:14"]
    REF["pose references 24 and 26, each with a fixed text description<br/>batch1_repose.py:127,128,130,131"]
    B1["batch 1, images 1 to 10, alternating 24 then 26<br/>batch1_repose.py:134,135,144"]
    B2["batch 2, images 11 to 23, skips reference 24<br/>repose_batch2.py:76,78,90"]
    B3["batch 3, images 25 and 27 to 30, skips reference 26<br/>batch3_repose.py:59,60,64"]
    OUT["output-reposed, named reposed_NN_poseNN.png<br/>batch1_repose.py:15"]
    SRC --> B1
    SRC --> B2
    SRC --> B3
    REF --> B1
    REF --> B2
    REF --> B3
    B1 --> OUT
    B2 --> OUT
    B3 --> OUT
    PROMPT["batch 1 sends a long requirements prompt, batches 2 and 3 a short one<br/>batch1_repose.py:47,55<br/>repose_batch2.py:24,30"]
    B1 --- PROMPT
    DRIFT["DOC-DRIFT: the README calls Nano Banana a ComfyUI custom node, the repose scripts call the REST endpoint directly and the repose workflow JSON is never loaded<br/>README.md:214<br/>batch1_repose.py:12<br/>nano-banana-repose.json:57"]
    OUT --- DRIFT
```

**What it shows.** The 28 outputs are three hand-partitioned batches with two different prompts; the ComfyUI repose workflow is documentation of an equivalent graph, not the path that ran.
