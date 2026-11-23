---
id: TH-02
title: The campaign phase record against the committed code
area: thg-world-record-attempt
governing: [../THG-world-record-attempt/README.md, ../THG-world-record-attempt/docs/prd/campaign-generator-v3.md]
adrs: []
sources:
  - ../THG-world-record-attempt/README.md
  - ../THG-world-record-attempt/configs/pipeline.json
  - ../THG-world-record-attempt/configs/topshop.json
  - ../THG-world-record-attempt/configs/prompts/phase1_prompts.json
  - ../THG-world-record-attempt/configs/workflows/flux2-multigpu-campaign.json
  - ../THG-world-record-attempt/configs/workflows/nano-banana-garment-reskin.json
  - ../THG-world-record-attempt/docs/prd/campaign-generator-v3.md
  - ../THG-world-record-attempt/reports/world-record-report.tex
  - ../THG-world-record-attempt/scripts/veo-branded-prompts.json
  - ../THG-world-record-attempt/scripts/clean_scenes.py
  - ../THG-world-record-attempt/scripts/generate_composites.py
  - ../THG-world-record-attempt/scripts/generate_veo_branded.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_riffs.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_expanded_riffs.py
  - ../THG-world-record-attempt/assets/scene-riffs/variations/_generate_flux2_riffs.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch1_repose.py
  - ../THG-world-record-attempt/task-two-repose/scripts/repose_batch2.py
  - ../THG-world-record-attempt/task-two-repose/scripts/batch3_repose.py
verified_commit: {thg-world-record-attempt: e79525441acff60a4e3ca4a0c1111535a39cc1e9}
---

## TH-02.1 Each recorded phase and what in the tree produced it

```mermaid
flowchart TB
    P1["Phase 1 base generation, Flux 2 Dev on the local GPU<br/>pipeline.json:9,11"]
    P1B["backing: a UI-format ComfyUI workflow only, no script<br/>flux2-multigpu-campaign.json:7"]
    P2["Phase 2 style refinement, Gemini 2.5 Flash Image<br/>pipeline.json:34,36"]
    P3["Phase 3 static compositing, PIL text overlay<br/>pipeline.json:49,51"]
    P4["Phase 4 animation, Veo 3.1 text to video<br/>pipeline.json:57,59,64"]
    P4B["Phase 4b garment reskinning<br/>pipeline.json:68,73"]
    P4W["backing: a UI-format GeminiImageNode workflow only<br/>nano-banana-garment-reskin.json:57"]
    NONE["no script in the tree for phases 2, 3 and 4, outputs exist only as committed media"]
    P5["Phase 5 scene riffs<br/>pipeline.json:80,82"]
    P5S["backing scripts<br/>clean_scenes.py:165<br/>generate_composites.py:156<br/>_generate_riffs.py:127<br/>_generate_expanded_riffs.py:104<br/>_generate_flux2_riffs.py:273"]
    P6["Phase 6 mannequin repose<br/>pipeline.json:99,101"]
    P6S["backing scripts<br/>batch1_repose.py:160<br/>repose_batch2.py:101<br/>batch3_repose.py:73"]
    P7["Phase 7 branded Veo films<br/>pipeline.json:114,116"]
    P7S["backing script<br/>generate_veo_branded.py:395"]
    P1 --> P1B
    P2 --> NONE
    P3 --> NONE
    P4 --> NONE
    P4B --> P4W
    P5 --> P5S
    P6 --> P6S
    P7 --> P7S
    DIV["DIVERGENCE: the README credits a Pillow overlay for the campaign text, no file in the tree imports ImageDraw or ImageFont<br/>README.md:57,210"]
    NONE --- DIV
    DIV2["DIVERGENCE: the PRD routes every asset through an automated QA and brand-compliance loop into an outputs tree, no QA code or outputs directory is committed<br/>campaign-generator-v3.md:291,293,313"]
    P7S --- DIV2
```

**What it shows.** Only phases 5 to 7 have runnable code; phases 1 to 4b are a status record, workflow JSON and the media they produced.

## TH-02.2 Counts, timings and statuses that disagree

```mermaid
flowchart TB
    subgraph COUNT["Asset counts"]
        direction TB
        C1["138 assets in 96 minutes<br/>README.md:5"]
        C2["the dress is constant across all 133 generated assets<br/>README.md:49"]
        C3["138 assets at T+125m, branded films finished 12.25<br/>world-record-report.tex:1224"]
        C4["metrics: 138 total, 96 minutes wall clock<br/>pipeline.json:138,145"]
    end
    subgraph TIME["Active generation time"]
        direction TB
        T1["64 minutes of active generation<br/>world-record-report.tex:159"]
        T2["69 active generation minutes<br/>pipeline.json:144"]
        T3["about 69 cumulative<br/>world-record-report.tex:1247"]
    end
    subgraph STATUS["Phase 7 status"]
        direction TB
        S1["Phase 7 queued, via a ComfyUI Veo node<br/>README.md:62"]
        S2["Phase 7 completed, 13.14Z to 13.27Z, REST predictLongRunning<br/>pipeline.json:115,117,118,119"]
        S3["branded prompts written 13.23.51Z<br/>veo-branded-prompts.json:7"]
        S4["campaign status phase7_completed but completed_at null<br/>pipeline.json:3,5"]
        S5["Phase 5 still in_progress while 45 of 45 are counted complete<br/>pipeline.json:81,96,97"]
    end
    COUNT --> TIME --> STATUS
    DRIFT1["DOC-DRIFT: 138 in 96 minutes and 138 at T+125m cannot both hold, the 96-minute mark is where the report puts 133<br/>world-record-report.tex:1223"]
    C3 --- DRIFT1
    DRIFT2["DOC-DRIFT: the report states 64 and 69 active minutes in two places"]
    T3 --- DRIFT2
    DRIFT3["DOC-DRIFT: the README calls Phase 7 queued while the ledger records it completed with five films<br/>pipeline.json:133"]
    S2 --- DRIFT3
```

**What it shows.** The record carries two totals (133 and 138), two active-time figures and two completion times for the same run.

## TH-02.3 The configuration surface and who reads it

```mermaid
flowchart TB
    TS["brand config: garment, palette, typography, formats, targets, engines, container rights<br/>topshop.json:4,17,70,82,88,121"]
    PH1["phase 1 prompt set, six seeded shots<br/>phase1_prompts.json:4,6"]
    LEDGER["phase ledger and metrics<br/>pipeline.json:8,137"]
    CONSUMERS["readers in code: none, the only JSON any script opens is its own Veo prompt or operation file<br/>generate_veo_branded.py:27"]
    TS --> CONSUMERS
    PH1 --> CONSUMERS
    LEDGER --> CONSUMERS
    DRIFT["DOC-DRIFT: both config files still describe a diagonal black striped skirt, the scripts that ran describe thin vertical pinstripes<br/>topshop.json:8<br/>phase1_prompts.json:3<br/>generate_composites.py:101"]
    TS --- DRIFT
    WHY["the correction came from a voice prompt and was applied to prompts, not to the config<br/>README.md:230"]
    DRIFT --- WHY
    OPEN["OPEN: targets of a 0.7 pass rate and 3 retries per asset are declared, nothing enforces them<br/>topshop.json:85,86"]
    TS --- OPEN
    DEBT["DEBT: the config grants agents model download, custom node install and venv creation inside the ComfyUI container<br/>topshop.json:125,126,127,128"]
    TS --- DEBT
```

**What it shows.** The config files are descriptions written for agents, not inputs to the scripts, so they drifted from what actually ran.
