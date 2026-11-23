---
id: OO-03
title: Settings surface — persisted fields, session keys, two renderers and save side effects
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-003-ingest-action-and-offline-contract.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-005-excluded-folders.md
adrs: [ADR-OBS-003, ADR-OBS-005]
sources:
  - ../ontomark-obsidian/src/settings.ts
  - ../ontomark-obsidian/src/main.ts
  - ../ontomark-obsidian/src/excludes.ts
  - ../ontomark-obsidian/src/jev.ts
  - ../ontomark-obsidian/manifest.json
verified_commit: {ontomark-obsidian: 64bdc5bf6309fca94bf0437b52c1484618470ed0}
---

## OO-03.1 What is stored in data.json and what lives only in memory

```mermaid
flowchart TD
    subgraph DATA["Persisted through saveData, main.ts:167"]
        CORE["profile generic, autoClassify on, queryLimit 1000<br/>settings.ts:35,36,37"]
        EXC["excludedFolders, default templates<br/>settings.ts:38<br/>excludes.ts:8"]
        ING["ingestEndpoint and ingestModel, empty keeps ingest off<br/>settings.ts:16,17,39,40"]
        JV["jevEnabled false, jevEndpoint, jevModel empty<br/>settings.ts:41,42,43"]
        THR["jevWithholdAtOrBelow 0, jevMinConfidence 0.6<br/>settings.ts:44,45<br/>jev.ts:49"]
        SEC["allowInsecureHttp false<br/>settings.ts:46"]
    end
    subgraph MEM["Plugin fields, never saved, cleared on unload"]
        IK["ingestKey<br/>main.ts:97,98"]
        JK["jevKey<br/>main.ts:99,100"]
    end
    RK["renderKey: password input writes the plugin field, never settings<br/>settings.ts:283,284,288,291"]
    RK --> IK
    RK --> JK
    UNL["onunload empties both keys<br/>main.ts:148,149"]
    MEM --- UNL
    INV["INVARIANT: the key rows are render items, not controls, so no key can reach the persisted settings<br/>settings.ts:167,202"]
    RK --- INV
    PRIV["Local by design text names ingest and Jev as the only network exceptions<br/>settings.ts:67,68,102"]
    DATA --- PRIV
```

**What it shows.** Endpoints and thresholds persist; API keys are typed per session and die with the plugin. An empty endpoint is the off switch for ingest, and Jev additionally needs its own toggle.

## OO-03.2 Load-time repair and the two renderers of one definition list

```mermaid
flowchart TD
    LD["loadData then normaliseSettings<br/>main.ts:120,121<br/>settings.ts:50"]
    MRG["saved data merged over defaults<br/>settings.ts:51"]
    FOL["excludedFolders re-parsed, non-strings dropped<br/>settings.ts:52,53"]
    FC["jevEnabled and allowInsecureHttp are on only for a literal true<br/>settings.ts:55,57"]
    UNIT["both thresholds must be numbers in 0 to 1, else defaults<br/>settings.ts:58,59,60"]
    STR["jevEndpoint and jevModel forced to strings<br/>settings.ts:61"]
    LD --> MRG --> FOL --> FC --> UNIT --> STR
    DEF["getSettingDefinitions: one list, groups for ingest, Jev and endpoint security<br/>settings.ts:100,143,172,219"]
    NEW["Obsidian 1.13 and later render it declaratively and index it for search<br/>settings.ts:86,87"]
    OLD["display fallback renders the same list imperatively, down to the 1.7.2 floor<br/>settings.ts:298,301<br/>manifest.json:5"]
    TXT["text controls commit on the DOM change event and validate first<br/>settings.ts:344,347,350"]
    GC["getControlValue presents numbers and folders as strings<br/>settings.ts:239,242,243"]
    DEF --> NEW
    DEF --> OLD
    OLD --> TXT
    NEW --> GC
    OLD --> GC
    INV["INVARIANT: a hand-edited data.json fails closed for Jev and plain http<br/>settings.ts:54,55,56,57"]
    FC --- INV
```

**What it shows.** Malformed or hostile saved values are repaired before use rather than rejected. Both renderers read and write through the same `getControlValue` and `setControlValue` pair (`settings.ts:89,90`).

## OO-03.3 setControlValue: what a single change sets in motion

```mermaid
flowchart TD
    SCV["setControlValue for a known key<br/>settings.ts:252,253"]
    EXQ{"excludedFolders?<br/>settings.ts:255"}
    BAD["folder list problem: store nothing<br/>settings.ts:257<br/>excludes.ts:86"]
    PARSE["parse, store, saveSettings<br/>settings.ts:258,260,261<br/>excludes.ts:72"]
    RX["reindex unless saveSettings already did, or the list is unchanged<br/>settings.ts:262"]
    UNI["threshold outside 0 to 1: store nothing<br/>settings.ts:265,362,365"]
    NUM["queryLimit and thresholds stored as numbers, strings trimmed<br/>settings.ts:267,268,269,271"]
    SAVE["saveSettings: saveData, re-read app.json filters<br/>main.ts:166,167,168"]
    FR{"Excluded files changed?<br/>main.ts:344,346"}
    REIDX["reindex the vault<br/>main.ts:169,179"]
    PROF{"profile?<br/>settings.ts:274"}
    SETP["push the profile to the engine, errors reported<br/>settings.ts:276,278"]
    SCV --> EXQ
    EXQ -->|yes| BAD
    EXQ -->|yes| PARSE --> RX
    EXQ -->|no| UNI
    EXQ -->|no| NUM --> SAVE --> FR
    FR -->|yes| REIDX
    SAVE --> PROF
    PROF -->|yes| SETP
    PARSE --> SAVE
    EP["endpoint fields validated against the current plain-http setting<br/>settings.ts:154,189,234,236"]
    INV["INVARIANT: the stored endpoint is checked again at use, so turning plain http off later still refuses a LAN http URL<br/>main.ts:458<br/>jev.ts:93"]
    EP --- INV
```

**What it shows.** Any save can reindex, because a save is one of the two moments Obsidian's own Excluded files are re-read; a profile change instead mutates the live engine. Validation of an endpoint is advisory at entry and enforced at use.
