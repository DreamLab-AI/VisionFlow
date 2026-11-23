---
id: OO-05
title: Network boundary — endpoint policy, the single transport and the Jev decision paths
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-002-assisted-ingest-boundary-and-estate-notes.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-003-ingest-action-and-offline-contract.md
adrs: [ADR-OBS-002, ADR-OBS-003]
sources:
  - ../ontomark-obsidian/src/endpoint.ts
  - ../ontomark-obsidian/src/ingest-transport.ts
  - ../ontomark-obsidian/src/jev.ts
  - ../ontomark-obsidian/src/jev-ingest.ts
  - ../ontomark-obsidian/src/query-templates.ts
  - ../ontomark-obsidian/src/views/query-view.ts
  - ../ontomark-obsidian/src/settings.ts
  - ../ontomark-obsidian/src/main.ts
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-003-ingest-action-and-offline-contract.md
verified_commit: {ontomark-obsidian: 64bdc5bf6309fca94bf0437b52c1484618470ed0}
---

## OO-05.1 endpointProblem: one policy, checked at entry and at use

```mermaid
flowchart TD
    IN["endpointProblem<br/>endpoint.ts:38"]
    ABS{"absolute URL?<br/>endpoint.ts:40,41,43"}
    PROTO{"http or https?<br/>endpoint.ts:45"}
    CRED{"embedded user or password?<br/>endpoint.ts:46"}
    FRAG{"has a fragment?<br/>endpoint.ts:47"}
    HTTP{"plain http to a non-loopback host without the opt-in?<br/>endpoint.ts:48"}
    LOOP["loopback: localhost, any .localhost name, 127.0.0.0/8, the IPv6 ::1<br/>endpoint.ts:18,19,20,21"]
    OK["null, usable<br/>endpoint.ts:51"]
    REFUSE["a reason string"]
    IN --> ABS
    ABS -->|no| REFUSE
    ABS -->|yes| PROTO
    PROTO -->|no| REFUSE
    PROTO -->|yes| CRED
    CRED -->|yes| REFUSE
    CRED -->|no| FRAG
    FRAG -->|yes| REFUSE
    FRAG -->|no| HTTP
    HTTP -->|yes| REFUSE
    HTTP -->|no| OK
    HTTP --- LOOP
    U1["settings field validation<br/>settings.ts:234,236"]
    U2["extract facts refuses with a Notice<br/>main.ts:458,460"]
    U3["Jev: unusable URL is no-endpoint, http-only refusal is insecure-endpoint<br/>jev.ts:91,93"]
    U1 --> IN
    U2 --> IN
    U3 --> IN
    TENS["TENSION: requestUrl follows redirects without reporting them, so the one-destination promise rests on the endpoint answering directly<br/>endpoint.ts:27,28,29,30<br/>README.md:74"]
    OK --- TENS
```

**What it shows.** The same pure function guards both optional features. A 3xx that does reach the plugin is refused (OO-05.2), but a redirect the host layer already followed is invisible to it.

## OO-05.2 postJson: the one place a request leaves the plugin

```mermaid
flowchart TD
    PJ["postJson: POST JSON, bearer header only when a key is set, HTTP status returned not thrown<br/>ingest-transport.ts:90,97,98,101,103"]
    RACE["timeout races the host request with no abort handle<br/>ingest-transport.ts:92,93,96"]
    subgraph INGEST["Assisted ingest, ingest-transport.ts:139"]
        CAPN["more than 64 chunks: nothing sent<br/>ingest-transport.ts:17,147,150"]
        LOOPI["per chunk: abort check, then the 10-minute run deadline<br/>ingest-transport.ts:158,159,163,164"]
        BODY["system instructions plus one chunk, json_object, temperature 0<br/>ingest-transport.ts:172,173,174,176,177"]
        CALLT["per-call timeout min of 60 s and time left<br/>ingest-transport.ts:19,24,180"]
        R3["3xx or non-2xx becomes model_unavailable<br/>ingest-transport.ts:116,117,119"]
        DROP["malformed answer drops that chunk only<br/>ingest-transport.ts:181,182"]
    end
    subgraph JEVP["Jev, jev.ts:82"]
        JT["4000 ms deadline, no retries<br/>jev.ts:17,18,100"]
        JBODY["model defaults to jev-latest, body is state plus questions<br/>jev.ts:48,99"]
        JF["timeout, transport, http status, malformed answers each become a fallback<br/>jev.ts:104,106,108,112"]
    end
    CAPN --> LOOPI --> BODY --> CALLT --> PJ
    PJ --> R3
    PJ --> DROP
    JBODY --> JT --> PJ
    PJ --> JF
    PJ --- RACE
    INV["INVARIANT: requestUrl is imported once, here, and shared by Jev through an injected post<br/>ingest-transport.ts:1,2,12<br/>jev.ts:3,4,5,6"]
    PJ --- INV
    LOG["fallback logs one line naming the reason and question set, never note content<br/>jev.ts:72,73"]
    JF --- LOG
    NOTE["see OM-09.4 for the ingest run lifecycle, review and apply"]
    LOOPI --- NOTE
```

**What it shows.** Ingest and Jev share a transport but not a failure model: an ingest failure aborts the run with an error, a Jev failure returns a fallback and the caller keeps its pre-Jev behaviour. Timeouts stop waiting; they do not recall a request already sent.

## OO-05.3 Three Jev call sites, one ask, nothing written by Jev

```mermaid
flowchart TD
    ASK["jevAsk: flag, endpoint, POST, every question answered with its own type<br/>jev.ts:82,89,91,93,110,111,112"]
    QS["question sets fact-support, query-template, edit-intent<br/>jev.ts:21,22,23,24"]
    subgraph FACTS["Extract facts verifier"]
        F1["judge passed only when Jev is enabled<br/>main.ts:491,492"]
        F2["verifyFacts: one Noul per quoted fact<br/>jev.ts:177,185,189"]
    end
    subgraph TPL["Query panel ask row"]
        T1["candidates from labels and aliases on word boundaries, at most 20<br/>query-view.ts:104<br/>query-templates.ts:83"]
        T2["pickTemplate: template, entity, dataset choices<br/>query-view.ts:105<br/>jev.ts:267"]
        T3["compileTemplate in code, unknown or unsafe IRI gives null<br/>query-view.ts:110<br/>query-templates.ts:60,70,74"]
        T4["fills the textarea and dataset, user presses Run<br/>query-view.ts:115,117"]
    end
    subgraph EDIT["Propose frontmatter edit from a sentence"]
        E1["parents: mentioned entities, not properties, unique note name<br/>main.ts:527,529,530"]
        E2["pickEdit: make class, individual, property or add parent<br/>main.ts:532<br/>jev.ts:355"]
        E3["proposal by the same API as the menu, then the diff<br/>main.ts:538,539,540,541,542"]
    end
    F1 --> F2 --> ASK
    T1 --> T2 --> ASK
    E1 --> E2 --> ASK
    ASK --- QS
    GATE["none, unresolved target or confidence below jevMinConfidence each fall back<br/>jev.ts:281,287,291,368,372,376"]
    T2 --> GATE
    E2 --> GATE
    T2 --> T3 --> T4
    E2 --> E3
    DS["dataset widens past asserted only at the same confidence<br/>jev.ts:294"]
    T2 --- DS
    OPEN["OPEN: the offline release test runs with Jev off, the enabled path is covered by mocked-transport unit tests only<br/>README.md:100"]
    ASK --- OPEN
```

**What it shows.** Jev only selects among options the code generated, and its output is either an annotation, a filled-but-unrun query, or the input to an existing reviewed edit. The model never sees IRIs for template choices: entities are offered as `e0`, `e1` with labels (`jev.ts:226`).

## OO-05.4 judgeIngest: advisory by default, fail-closed above a threshold

```mermaid
flowchart TD
    J["judgeIngest after the engine grounded the answer<br/>jev-ingest.ts:117"]
    OFF{"Jev enabled?<br/>jev-ingest.ts:126"}
    Q["split accepted facts into quoted and unquoted<br/>jev-ingest.ts:128,130"]
    NONE["no facts: verified, nothing to judge<br/>jev-ingest.ts:151,152"]
    V["verifyFacts on the quoted facts<br/>jev-ingest.ts:158"]
    VF{"verifier answered?<br/>jev-ingest.ts:164"}
    ADV["threshold 0: keep the plain result, mark not_verified<br/>jev-ingest.ts:134,135"]
    FC["threshold above 0: withhold as not_verified or judged_unsupported<br/>jev-ingest.ts:127,137,138,139,140"]
    NOTES["doubt note below 0.5, reject at or below the threshold<br/>jev-ingest.ts:175,177<br/>jev.ts:51"]
    RERUN["withhold reruns ingest_accept without the dropped proposals<br/>jev-ingest.ts:78,87"]
    MONO{"rerun writes only facts the plain run wrote, none it was told to drop?<br/>jev-ingest.ts:89,91"}
    NW["nothingWritten: no preview, every fact listed as withheld<br/>jev-ingest.ts:102,109,202"]
    OUT["review dialog then guarded apply<br/>jev-ingest.ts:204,205"]
    J --> OFF
    OFF -->|no| OUT
    OFF -->|yes| Q
    Q --> NONE --> OUT
    Q --> V --> VF
    VF -->|no| ADV --> OUT
    VF -->|no| FC
    VF -->|yes| NOTES
    NOTES -->|nothing rejected| OUT
    NOTES -->|rejections| RERUN
    FC --> RERUN --> MONO
    MONO -->|yes| OUT
    MONO -->|no| NW --> OUT
    INV["INVARIANT: never fall back to the plain patch once Jev rejected a fact<br/>jev-ingest.ts:200,201"]
    NW --- INV
    DRIFT["DOC-DRIFT: ADR-OBS-003 still records implementation none in this plugin, yet the ingest command and this verifier ship<br/>ADR-OBS-003-ingest-action-and-offline-contract.md:6,19<br/>main.ts:558"]
    J --- DRIFT
```

**What it shows.** The patch is always the engine's own output, re-run on fewer proposals, so grounding, conflict checks and the digest guard still apply after a withhold. OM-09.4 states the same threshold rule as one invariant; this panel draws its branches.
