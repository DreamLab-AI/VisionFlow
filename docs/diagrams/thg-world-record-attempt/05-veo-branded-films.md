---
id: TH-05
title: Veo branded films, the generator and the separate poller
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/README.md]
adrs: []
sources:
  - ../THG-world-record-attempt/README.md
  - ../THG-world-record-attempt/scripts/generate_veo_branded.py
  - ../THG-world-record-attempt/scripts/veo-branded-prompts.json
  - ../THG-world-record-attempt/scripts/poll-veo-operations.py
  - ../THG-world-record-attempt/scripts/veo-operations.json
verified_commit: {thg-world-record-attempt: 27262fa0e91f2b0d78c94b3b3fe248c7c53dba1c}
---

## TH-05.1 One film from prompt to branded MP4

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>generate_veo_branded.py:395
    participant V as Veo predictLongRunning
    participant DL as curl subprocess
    participant FF as ffprobe and ffmpeg
    M->>M: write all five prompts and payloads to veo-branded-prompts.json first (:155,184,399)
    M->>M: no key, stop after the JSON, existing branded file over 50 KB, skip (:402,406,423,427)
    M->>V: POST instances prompt, aspectRatio, durationSeconds 8, up to 3 tries (:192,193,196,200,431)
    V-->>M: operation name (generate_veo_branded.py:203)
    loop every 15 s up to 600 s
        M->>V: GET the operation (generate_veo_branded.py:220,224,270)
    end
    V-->>M: done, video uri from one of three response shapes (generate_veo_branded.py:245,247,249,259)
    M->>DL: curl -L to the raw file, key appended for googleapis hosts (generate_veo_branded.py:284,286,290)
    M->>FF: probe width and height, two drawtext lines, libx264 crf 18 (generate_veo_branded.py:321,349,375,377)
    M->>M: branded file kept, raw removed, or raw renamed when branding fails (:461,465,472)
    Note over M: DOC-DRIFT: source_image is recorded for each film but the payload is text only<br/>generate_veo_branded.py:44,193
```

**What it shows.** Every film is text to video; the riff image named as its source never reaches the API, and branding is an ffmpeg overlay afterwards.

## TH-05.2 Outcomes a film can end in

```mermaid
flowchart TB
    START["for each of five films<br/>generate_veo_branded.py:415"]
    SKIP["already_exists<br/>generate_veo_branded.py:426"]
    SUBF["submit_failed after 3 attempts<br/>generate_veo_branded.py:442"]
    GENF["generation_failed, error, timeout or no uri<br/>generate_veo_branded.py:450"]
    DLF["download_failed<br/>generate_veo_branded.py:456"]
    OK["success, branded<br/>generate_veo_branded.py:462"]
    NOB["success_no_brand, raw renamed to the branded name<br/>generate_veo_branded.py:473"]
    BRF["brand_failed<br/>generate_veo_branded.py:475"]
    START --> SKIP
    START --> SUBF
    START --> GENF
    START --> DLF
    START --> OK
    START --> NOB
    START --> BRF
    INV["INVARIANT: the two brand lines show only between 1 s and 7.5 s, from a fixed Noto Sans Bold path<br/>generate_veo_branded.py:28,35,36,358,367"]
    OK --- INV
    DEBT1["DEBT: any submit failure waits 60 s times attempt, including after the third and last try<br/>generate_veo_branded.py:431,436,438"]
    SUBF --- DEBT1
    DEBT2["DEBT: the summary counts any file present as generated, so an unbranded rename counts the same<br/>generate_veo_branded.py:492,493,495"]
    NOB --- DEBT2
    DRIFT["DOC-DRIFT: the README says the films used a ComfyUI Veo node, the script calls the REST endpoint, as another README table also says<br/>README.md:62,132"]
    START --- DRIFT
    REC["DOC-DRIFT: the committed prompt record carries status and branding-method fields the writer never emits, its payloads add a sampleCount the live submit omits, and a re-run overwrites the record first<br/>veo-branded-prompts.json:8,9,10,29<br/>generate_veo_branded.py:157,165,180,196,399"]
    OK --- REC
```

**What it shows.** A brand failure still yields a file under the branded name, and the summary does not distinguish it.

## TH-05.3 The separate poller and its operation list

> **Superseded** by TH-05.4 at 27262fa (frozen at e795254).

```mermaid
flowchart TB
    OPS["five operation names, model veo-3.0-generate-001<br/>veo-operations.json:3,4,6"]
    POLL["poll each pending operation up to 30 times, 30 s apart<br/>poll-veo-operations.py:9,17,20,27,76"]
    URI["done with a uri, download with the key appended<br/>poll-veo-operations.py:35,38,43,44,45"]
    B64["done with inline base64, decode to file<br/>poll-veo-operations.py:51,55"]
    ERR["done with error or no samples, marked complete anyway<br/>poll-veo-operations.py:62,64,66,67"]
    OUT["assets/animated, output name veo_greenhouse.mp4 and so on<br/>poll-veo-operations.py:6,42"]
    OPS --> POLL
    POLL --> URI --> OUT
    POLL --> B64 --> OUT
    POLL --> ERR
    DEBT["DEBT: the API key is a string literal in this file, not read from the environment<br/>poll-veo-operations.py:5"]
    POLL --- DEBT
    DRIFT["DOC-DRIFT: the operations name Veo 3.0, the generator and prompt record name veo-3.1-generate-preview<br/>generate_veo_branded.py:31<br/>veo-branded-prompts.json:3"]
    OPS --- DRIFT
    TENS["TENSION: the poller writes an unbranded file to the same name the generator gives its branded output, and the generator then skips that film as already done<br/>veo-operations.json:6<br/>generate_veo_branded.py:43,423"]
    OUT --- TENS
    DEBT2["DEBT: a test operation is defined and never polled, and the operations file is opened relative to the working directory<br/>poll-veo-operations.py:13,15,9"]
    OPS --- DEBT2
```

**What it shows.** A second, older Veo path exists beside the generator; which of the two produced the committed films is not recorded in code.

## TH-05.4 The poller's key, read from the environment

```mermaid
flowchart TB
    K["API_KEY from GOOGLE_API_KEY, else GOOGLE_GEMINI_API_KEY<br/>poll-veo-operations.py:5"]
    X["neither set: exit with a message before any request<br/>poll-veo-operations.py:6-7"]
    OPS["operations file, still opened relative to the working directory<br/>poll-veo-operations.py:11"]
    POLL["poll each pending operation up to 30 times, 30 s apart<br/>poll-veo-operations.py:19,22,29,78"]
    URI["done with a uri, download with the key appended<br/>poll-veo-operations.py:45"]
    K --> OPS --> POLL --> URI
    K --> X
    SAME["same fallback order as the submitter<br/>generate_veo_branded.py:30"]
    K -.- SAME
```

- **What it shows:** the frozen TH-05.3 recorded the key as a string literal in the poller. Since 27262fa it is read from the environment, with the submitter's fallback, and a missing key stops the script. The rest of TH-05.3 (model name drift, the unbranded-output tension, the unpolled test operation) is unchanged at this revision.
- **Why it is this way:** PR #1 (27262fa); the replaced key was already invalid (tested 2026-11-23), so the fix closes the pattern rather than a live credential.
