---
id: VF-06
title: Diagram tooling — the report render gate and this tree's own generator
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
adrs: [ADR-2004]
sources:
  - .github/workflows/diagram-render.yml
  - .github/workflows/diagram-index.yml
  - .github/workflows/deploy.yml
  - scripts/diagram-render/render.mjs
  - scripts/diagram-render/lib/cdp.mjs
  - scripts/diagram-render/lib/preprocess.mjs
  - scripts/diagram-render/package.json
  - scripts/diagram-render/vendor/mermaid.min.js
  - scripts/check-diagram-text.js
  - scripts/check-cdp-sidecar.mjs
  - scripts/diagram-index-gen.cjs
  - scripts/validate-mermaid-diagrams.sh
  - scripts/generate-diagram-art.sh
  - docs/diagrams/README.md
  - docs/adr/ADR-2004-diagram-baseline-vendored-render-gate.md
  - docs/BASELINE-visionflow.md
  - docs/site-verification.md
verified_commit: bec06dc3a
---

## VF-06.1 Two diagram pipelines, one repository — what each owns
```mermaid
flowchart TB
    classDef pipeA fill:#e4ecf8,stroke:#33559a,color:#222
    classDef pipeB fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef orphan fill:#f0f0f0,stroke:#888,color:#222

    subgraph A["Pipeline A — report render gate (RES-b)"]
        direction TB
        AS["presentation/report/diagrams/*.mmd<br/>ten dark-theme authored sources"]:::pipeA
        APRE["preprocess.mjs — strip theme directive<br/>and dark palette<br/>preprocess.mjs:19"]:::pipeA
        AREN["render.mjs — vendored mermaid.min.js<br/>driven over raw CDP in a real Chrome<br/>render.mjs:127"]:::pipeA
        ABASE["presentation/report/diagrams/rendered/*.svg<br/>COMMITTED baseline — the authority"]:::pipeA
        ACHK["check-diagram-text.js — browserless<br/>visibility and label probe<br/>check-diagram-text.js:160"]:::pipeA
        ADIFF["check-diagram-text.js --diff —<br/>visible-word set comparison<br/>check-diagram-text.js:123"]:::pipeA
        AS --> APRE --> AREN --> ABASE
        ABASE --> ACHK
        AREN -.->|"secondary drift check only"| ADIFF
        ABASE --> ADIFF
    end

    subgraph B["Pipeline B — this diagrams-as-code tree"]
        direction TB
        BS["docs/diagrams/AREA/NN-*.md<br/>frontmatter plus fenced mermaid blocks"]:::pipeB
        BGEN["diagram-index-gen.cjs — walk, parse,<br/>validate, optionally render and index<br/>diagram-index-gen.cjs:451"]:::pipeB
        BREND["docs/diagrams/rendered/ via mmdc,<br/>gitignored and regenerable<br/>outDir at diagram-index-gen.cjs:341"]:::pipeB
        BIDX["docs/diagrams/README.md index block<br/>plus COVERAGE.md inverted indexes<br/>writeIndexes at diagram-index-gen.cjs:368, COVERAGE.md written :445"]:::pipeB
        BS --> BGEN
        BGEN -->|"--render"| BREND
        BGEN -->|"no --check, no --only"| BIDX
    end

    WA["diagram-render.yml — PR-triggered on<br/>presentation/report/diagrams/**<br/>diagram-render.yml:18"] --> A
    WB["diagram-index.yml — push and PR on<br/>docs/diagrams/**<br/>diagram-index.yml:7"] --> B
    WD["deploy.yml — BLOCKING publication gate runs<br/>the browser-free baseline probe only<br/>deploy.yml:124"] --> ACHK

    subgraph O["Neither pipeline — unwired scripts"]
        OV["validate-mermaid-diagrams.sh<br/>hardcoded to a file in ANOTHER repo<br/>validate-mermaid-diagrams.sh:7"]:::orphan
        OG["generate-diagram-art.sh<br/>Gemini image generation, needs an API key<br/>generate-diagram-art.sh:5"]:::orphan
    end

    NOTE1["INVARIANT: the COMMITTED baseline is the authority, not a CI render.<br/>A green build never depends on a browser matching byte for byte<br/>ADR-2004-diagram-baseline-vendored-render-gate.md:30"]
    NOTE2["INVARIANT: this very file was validated by pipeline B —<br/>diagram-index-gen.cjs docs/diagrams --check --cite-check --only visionflow/<br/>diagrams/README.md:68"]
    NOTE3["DIVERGENCE: the two pipelines share no code. Pipeline A vendors mermaid 11.16.0<br/>and drives Chrome; pipeline B shells out to the Nix-installed mmdc"]
```

## VF-06.2 diagram-render.yml — three stages, baseline first
```mermaid
sequenceDiagram
    autonumber
    participant GH as "diagram-render.yml job render-gate"
    participant CHK as "check-diagram-text.js"
    participant NPM as "npm install -g pinned toolchain"
    participant RES as "step chrome — resolve a Chrome binary"
    participant REN as "render.mjs"

    Note over GH: triggers only on presentation/report/diagrams/**,<br/>scripts/diagram-render/**, check-diagram-text.js and this workflow<br/>diagram-render.yml:18

    rect rgb(228, 240, 232)
    Note over GH,CHK: STAGE 1 — the AUTHORITATIVE guard. No browser, fully deterministic
    GH->>CHK: node check-diagram-text.js presentation/report/diagrams/rendered<br/>diagram-render.yml:47
    CHK-->>GH: exit 1 unless every committed SVG has visible text and its key labels
    end

    rect rgb(232, 238, 250)
    Note over GH,RES: STAGE 2 — provision a browser, NOT a renderer
    GH->>NPM: install mermaid-cli 11.16.0 and puppeteer 23.11.1<br/>diagram-render.yml:34
    Note over NPM: the mermaid-cli version is pinned to match the bundled mermaid in<br/>scripts/diagram-render/vendor/mermaid.min.js — two edits kept in lockstep
    NPM-->>RES: puppeteer downloads a chromium into PUPPETEER_CACHE_DIR
    RES->>RES: find a chrome binary in that cache, else fall back to<br/>google-chrome or chromium on PATH<br/>diagram-render.yml:61
    RES-->>GH: no Chrome or Chromium binary found is an error, exit 1<br/>diagram-render.yml:66
    end

    rect rgb(250, 244, 228)
    Note over GH,REN: STAGE 3 — re-render from source, then diff. Secondary, not truth
    GH->>GH: copy the committed baseline aside into RUNNER_TEMP/baseline<br/>diagram-render.yml:80
    GH->>REN: DIAGRAM_CHROME_BIN set, node render.mjs<br/>diagram-render.yml:81
    REN-->>GH: rendered/*.svg overwritten in place
    GH->>CHK: re-run the visibility probe on the fresh output<br/>diagram-render.yml:82
    GH->>CHK: --diff RUNNER_TEMP/baseline against rendered<br/>diagram-render.yml:83
    CHK-->>GH: exit 1 on any visible-word drift — this is what catches an<br/>edited .mmd whose baseline was never regenerated
    GH->>GH: git diff --stat over rendered/ is INFORMATIONAL only —<br/>sub-pixel font metrics differ across environments<br/>diagram-render.yml:85
    end

    Note over GH: DIVERGENCE: this workflow has never run hosted. The three stages were<br/>executed by hand against the sidecar Chrome, so cross-environment byte<br/>parity remains untested<br/>ADR-2004-diagram-baseline-vendored-render-gate.md:107
```

## VF-06.3 render.mjs — how a browser is obtained
```mermaid
flowchart TB
    classDef ci fill:#e4ecf8,stroke:#33559a,color:#222
    classDef local fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef fail fill:#ffe0e0,stroke:#aa3333,color:#222

    START["resolveConnection<br/>render.mjs:97"] --> Q1{"DIAGRAM_CHROME_BIN set?"}

    Q1 -->|yes| LAUNCH["launchLocalChrome spawns headless=new, no-sandbox,<br/>disable-gpu, remote-debugging-port=0, temp user-data-dir<br/>cdp.mjs:54"]:::ci
    LAUNCH --> PORT["poll DevToolsActivePort in that directory,<br/>20s deadline; the first line is the port<br/>cdp.mjs:64"]:::ci
    PORT --> TO["Chrome did not expose a DevTools port within 20s<br/>cdp.mjs:83"]:::fail
    PORT --> EP

    Q1 -->|no| CANDS["candidate list: DIAGRAM_CDP_URL first, then<br/>BROWSER_CDP_HOST and BROWSER_CDP_PORT,<br/>defaulting to browsercontainer 9223<br/>render.mjs:105"]:::local
    CANDS --> EP["browserEndpointFrom does GET /json/version<br/>under an 8s AbortController timeout<br/>cdp.mjs:30"]
    EP --> REWRITE["rewrite the webSocketDebuggerUrl host back to the reachable one:<br/>Chrome's DevTools HTTP endpoint rejects a hostname Host header,<br/>so the sidecar name is DNS-resolved to an IP first<br/>cdp.mjs:22"]
    REWRITE --> OPEN["CdpConnection.open uses node's built-in global WebSocket,<br/>so the gate carries NO npm runtime dependency<br/>cdp.mjs:92"]
    OPEN --> PAGE["newPage — Target.createTarget, attachToTarget flatten,<br/>Page.enable, Runtime.enable<br/>cdp.mjs:122"]

    CANDS --> ALLFAIL["No reachable browser, listing everything tried<br/>render.mjs:116"]:::fail

    PAGE --> LOADM["inject the vendored bundle as a Runtime.evaluate source<br/>string, then assert typeof mermaid<br/>render.mjs:134"]
    LOADM --> MISSING["a missing vendored bundle throws before any work<br/>render.mjs:120"]:::fail

    NOTE["INVARIANT: no CDN and no npm renderer. vendor/mermaid.min.js is the<br/>engine; mermaid-cli exists in CI only to produce a Chrome binary<br/>diagram-render/package.json:4"]
    NOTE2["EXTERNAL: browsercontainer 9223 is the shared browser sidecar; the same<br/>host-header rewrite lives in check-cdp-sidecar.mjs:20 and the site<br/>verification contract in docs/site-verification.md"]
```

## VF-06.4 render.mjs — in-page render, baked background and baked text fill
```mermaid
sequenceDiagram
    autonumber
    participant N as "render.mjs in node"
    participant P as "preprocess.mjs"
    participant B as "page — window.__renderDiagram<br/>render.mjs:26"
    participant M as "vendored mermaid 11.16.0"
    participant FS as "presentation/report/diagrams/rendered/"

    loop each .mmd in sorted order
        N->>P: preprocessSource on the raw source<br/>render.mjs:143
        P-->>N: light-theme-safe definition
        N->>B: Runtime.evaluate window.__renderDiagram, awaitPromise true
        B->>M: mermaid.initialize theme default, deterministicIds,<br/>securityLevel loose<br/>render.mjs:29
        Note over B,M: TOP-LEVEL htmlLabels false is what actually forces SVG text nodes —<br/>the flowchart-scoped key alone is ignored by mermaid<br/>render.mjs:37
        M-->>B: svg string
        B->>B: append into an off-screen holder at left minus 99999px<br/>so getComputedStyle resolves real values<br/>render.mjs:46
        B->>B: insert a full-size white rect as the FIRST child —<br/>the transparent-background trap that let dark text vanish<br/>render.mjs:52
        loop each text or tspan with non-empty content
            B->>B: read the computed fill and write it back as an attribute<br/>render.mjs:65
            B->>B: carry fill-opacity through only when below 1<br/>render.mjs:67
            B->>B: count as visible unless r, g and b are all at least 240<br/>render.mjs:71
        end
        B-->>N: JSON carrying svg plus total and visible counts<br/>render.mjs:78
        N->>FS: write rendered NAME.svg with an XML prolog<br/>render.mjs:150
    end

    alt any diagram threw
        N-->>N: log FAILED per file, then exit 1<br/>render.mjs:162
    else all rendered
        N-->>N: report the number of diagrams rendered<br/>render.mjs:166
    end

    Note over N,FS: INVARIANT: the browser-computed fill is BAKED into the file so the<br/>standalone checker reads the visible colour with no browser at all<br/>render.mjs:57
```

## VF-06.5 preprocess.mjs — dark-to-light transform and label-word extraction
```mermaid
flowchart TB
    classDef strip fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef keep fill:#e0f2e4,stroke:#2f7a45,color:#222

    RAW["raw .mmd authored for a DARK mermaid theme: a leading<br/>init directive plus hardcoded style, linkStyle and classDef<br/>lines with near-white text<br/>preprocess.mjs:3"]

    RAW --> T1["drop every full-line init directive —<br/>they force theme dark and dark themeVariables<br/>preprocess.mjs:22"]:::strip
    T1 --> T2["drop every style, linkStyle and classDef line —<br/>the hardcoded dark palette overrides<br/>preprocess.mjs:24"]:::strip
    T2 --> T3["drop b and i emphasis tags: the non-HTML label<br/>renderer would print them verbatim<br/>preprocess.mjs:26"]:::strip
    T3 --> T4["keep the line-break tag — mermaid splits on it<br/>preprocess.mjs:16"]:::keep
    T4 --> T5["collapse runs of three or more newlines the strips left, then trim<br/>preprocess.mjs:28"]
    T5 --> OUT["definition rendered with mermaid's LIGHT default theme,<br/>so all text resolves to a dark print-safe fill"]:::keep

    RAW --> W1["extractLabelWords reads the ORIGINAL source, not the transform<br/>preprocess.mjs:38"]
    W1 --> W2["fragments: every quoted string and every bracketed label<br/>preprocess.mjs:40"]
    W2 --> W3["strip markup and entities, keep letters only<br/>preprocess.mjs:54"]
    W3 --> W4["keep words of four or more characters that are not in the stop set<br/>preprocess.mjs:57"]
    W4 --> W5["word level, not phrase level, so it survives mermaid wrapping<br/>one label across several tspan lines<br/>preprocess.mjs:35"]:::keep
    W5 --> CONS["consumed by check-diagram-text.js as the required-label set<br/>check-diagram-text.js:25"]

    NOTE["DIVERGENCE: the transform is lossy by design — a diagram whose meaning<br/>depended on a classDef colour loses it. The gate certifies text visibility<br/>and label presence, never that a diagram is semantically correct<br/>ADR-2004-diagram-baseline-vendored-render-gate.md:110"]
```

## VF-06.6 check-diagram-text.js — what counts as invisible
```mermaid
flowchart TB
    classDef bad fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222

    IN["runCheck over a render directory, defaulting to<br/>presentation/report/diagrams/rendered<br/>check-diagram-text.js:160"] --> E0{"directory exists<br/>and holds any .svg?"}
    E0 -->|no| F0["render directory not found, or no rendered SVGs — exit 1<br/>check-diagram-text.js:166"]:::bad
    E0 -->|yes| WALK["extractTextNodes walks every text element, resolving fill<br/>from tspan, then text, then the style default, then black<br/>check-diagram-text.js:100"]

    WALK --> J{"isInvisibleFill<br/>check-diagram-text.js:46"}
    J -->|"fill-opacity below 0.1"| INV["invisible"]:::bad
    J -->|"none, transparent, white or #fff"| INV
    J -->|"rgb with alpha below 0.1"| INV
    J -->|"r, g and b all at least 240 — hex or rgb"| INV
    J -->|"a named colour other than white"| VIS["visible — named colours are trusted<br/>check-diagram-text.js:69"]:::ok
    J -->|"anything else"| VIS

    VIS --> P1{"at least one visible node?"}
    INV --> P2["any invisible node is a problem, with up to four samples<br/>printed as text equals fill<br/>check-diagram-text.js:179"]:::bad
    P1 -->|no| P3["no visible text nodes<br/>check-diagram-text.js:178"]:::bad

    P1 -->|yes| L1{"matching .mmd source exists?"}
    L1 -->|no| L2["no matching .mmd source is a problem, so an orphan<br/>SVG in rendered/ fails the gate<br/>check-diagram-text.js:196"]:::bad
    L1 -->|yes| L3["extractLabelWords over the raw source, then a substring<br/>search in the lower-cased VISIBLE text only<br/>check-diagram-text.js:189"]
    L3 --> L4{"every wanted word found?"}
    L4 -->|no| L5["missing label words in visible text, first eight listed<br/>check-diagram-text.js:192"]:::bad
    L4 -->|yes| PASS["ok NAME with the visible-node count and label tally<br/>check-diagram-text.js:203"]:::ok

    P2 --> AGG
    P3 --> AGG
    L2 --> AGG
    L5 --> AGG["failures counted, one exit 1 after every file is judged<br/>check-diagram-text.js:208"]:::bad

    NOTE["INVARIANT: pure node built-ins — no browser, no npm dependency — so<br/>this probe runs anywhere, including as a blocking step at deploy.yml:124<br/>check-diagram-text.js:16"]
```

## VF-06.7 check-diagram-text.js --diff — word-set drift against the baseline
```mermaid
sequenceDiagram
    autonumber
    participant CI as "diagram-render.yml stage 3"
    participant D as "runDiff over baseline and candidate<br/>check-diagram-text.js:123"
    participant BL as "RUNNER_TEMP/baseline — the committed SVGs"
    participant CA as "rendered/ — freshly re-rendered SVGs"

    CI->>D: --diff with exactly three argv entries, else usage exit 2<br/>check-diagram-text.js:216
    D->>BL: list the SVGs
    D->>CA: list the SVGs
    loop union of both lists, sorted
        alt present in candidate, absent from baseline
            D-->>CI: DRIFT — an SVG appeared with no committed twin<br/>check-diagram-text.js:137
        else present in baseline, absent from candidate
            D-->>CI: DRIFT — a committed SVG no longer renders<br/>check-diagram-text.js:138
        else present in both
            D->>D: reduce each file to a SET of lower-cased visible words,<br/>splitting on every non-alphanumeric run<br/>check-diagram-text.js:127
            D->>D: removed is baseline minus candidate, added is the converse<br/>check-diagram-text.js:141
            alt either set is non-empty
                D-->>CI: DRIFT with up to eight removed and eight added words
            else identical
                D-->>CI: ok NAME, visible words match baseline<br/>check-diagram-text.js:149
            end
        end
    end
    D-->>CI: exit 1 if any file drifted<br/>check-diagram-text.js:154

    Note over D: INVARIANT: word level, not phrase level and not byte level. Invariant to<br/>the sub-pixel font metrics that make one Chrome wrap a label across a<br/>different number of tspan lines, and to SVG element ordering<br/>check-diagram-text.js:118
    Note over CI,CA: What it still catches: a label added, removed or changed because the<br/>committed baseline went stale against an edited .mmd. That forgotten-regen<br/>failure is the cost ADR-2004 accepts for the design
```

## VF-06.8 diagram-index-gen.cjs — walk, parse, validate
```mermaid
sequenceDiagram
    autonumber
    participant CLI as main IIFE<br/>diagram-index-gen.cjs:553
    participant W as walk<br/>diagram-index-gen.cjs:97
    participant P as parseTopic<br/>diagram-index-gen.cjs:150
    participant F as parseFrontmatter<br/>diagram-index-gen.cjs:126
    CLI->>W: traverse topics excluding generated and archived trees
    loop each topic
        CLI->>P: parse topic, collect errors
        P->>F: parse flat metadata and lists
        F-->>P: frontmatter and body
        P->>P: check metadata, source paths, headings and diagram structure
        P-->>CLI: topic and diagrams
    end
    CLI->>CLI: reject duplicate topic and diagram identifiers
    Note over CLI,P: Source paths may be skipped by hosted structural checks.<br/>Structure does not attest behaviour, clean source or activation.
```

## VF-06.9 Citation refusal and structural failures
```mermaid
flowchart TB
    I["Topic input"] --> P["Structural parse and duplicate checks<br/>diagram-index-gen.cjs:150"]
    P --> ERR["Accumulated errors"]
    I --> C["Citation and symbol diagnostics<br/>diagram-index-gen.cjs:280<br/>diagram-index-gen.cjs:371"]
    C --> STRICT{"strict-citations?"}
    STRICT -->|yes| ERR
    STRICT -->|no| WARN["Advisory warnings remain visible"]
    I --> R["Optional render and width checks<br/>diagram-index-gen.cjs:442"]
    R --> ERR
    ERR --> FAIL["Exit 1 when errors exist<br/>diagram-index-gen.cjs:581"]
    MODE["Strict citations reject no-source-paths<br/>diagram-index-gen.cjs:92"] --> STRICT
    LIMIT["A strict pass establishes citation hygiene;<br/>semantic and deployment evidence remain separate."]
    WARN --> LIMIT
```

## VF-06.10 Citation resolution and inference limits
```mermaid
flowchart TB
    A["Mermaid source"] --> SCAN["Dotted path plus line/range scan<br/>diagram-index-gen.cjs:218"]
    SCAN --> MATCH["Exact source entry wins; otherwise resolve unique suffix<br/>diagram-index-gen.cjs:296"]
    MATCH --> MODE["Choose source bytes<br/>diagram-index-gen.cjs:263"]
    MODE --> PIN["Default: declared revision via git show;<br/>unavailable revision falls back to working tree"]
    MODE --> WT["worktree-citations: current working tree only"]
    PIN --> READ["Check line bounds and anchors<br/>diagram-index-gen.cjs:301"]
    WT --> READ
    A --> BARE["Bare-line context from explicit citations and participant bindings<br/>diagram-index-gen.cjs:329"]
    BARE --> READ
    A --> SYMBOL["Function-labelled participant checked against a unique definition<br/>diagram-index-gen.cjs:371"]
    READ --> DIAG["Diagnostic: ambiguous, missing, unreadable, out of bounds or empty anchor"]
    SYMBOL --> DIAG
    DIAG --> EXIT["Strict mode adds diagnostics to errors<br/>diagram-index-gen.cjs:574"]
    LIMIT["Context inference and brace counting are heuristics.<br/>A real source line can still support the wrong claim."] --> EXIT
```

## VF-06.11 Render invocation and width boundary
```mermaid
sequenceDiagram
    participant G as renderAll<br/>diagram-index-gen.cjs:442
    participant R as renderOne<br/>diagram-index-gen.cjs:417
    participant M as mmdc
    participant O as rendered topic directory
    G->>G: queue blocks with bounded worker concurrency
    G->>R: render current block
    R->>O: write exact Mermaid input
    R->>M: spawn renderer with input and SVG paths
    alt successful process
        M-->>R: SVG output
        R->>R: reject viewBox width above 4500px
    else render failure
        M-->>R: captured failure details
    end
    R-->>G: success or diagnostic
    Note over G,O: Exact-source comparison and execution logs are distinct evidence.<br/>A cached SVG by itself is not proof of a fresh successful render.
```

## VF-06.12 Index identity and hosted checks
```mermaid
sequenceDiagram
    participant CI as diagram-index.yml
    participant T as diagram-index.test.cjs
    participant G as writeIndexes<br/>diagram-index-gen.cjs:470
    participant RM as README.md
    participant CV as COVERAGE.md
    CI->>T: exercise namespace collisions and strict citation refusal
    CI->>G: structural check with sibling paths skipped
    CI->>CV: retain pre-generation content
    CI->>G: regenerate coverage from topic metadata
    G->>RM: emit topic tables and regeneration command
    G->>CV: qualify ADR identities by repository<br/>diagram-index-gen.cjs:507
    G->>CV: label revisions as author declarations
    CI->>CI: compare regenerated coverage with committed content
    Note over G,RM: check and only suppress index writes<br/>diagram-index-gen.cjs:586
    Note over CI,CV: Hosted structural success does not check sibling source behaviour.<br/>Run strict citations in a source-accessible checkout and preserve claim evidence.
```

## VF-06.13 Unwired diagram scripts — what they point at and why they are orphaned
```mermaid
flowchart TB
    classDef orphan fill:#f0f0f0,stroke:#888,color:#222
    classDef ext fill:#f2eaf6,stroke:#7a4a9a,color:#222

    subgraph V["validate-mermaid-diagrams.sh"]
        V1["DOCS_DIR is a HARDCODED absolute path into<br/>ANOTHER repository's checkout<br/>validate-mermaid-diagrams.sh:7"]:::orphan
        V2["FILE is that repo's event-flow-diagrams.md;<br/>the script exits 1 if it is absent<br/>validate-mermaid-diagrams.sh:8"]:::orphan
        V3["counts mermaid fences and asserts exactly ten<br/>validate-mermaid-diagrams.sh:31"]
        V4["greps for named participants — EventBus, GraphRepository,<br/>PhysicsService, EventStore<br/>validate-mermaid-diagrams.sh:60"]:::ext
        V5["greps for event type names and for figures such as<br/>316 nodes, 450 edges and 60 FPS<br/>validate-mermaid-diagrams.sh:136"]:::ext
        V6["exit 0 only when the fence count is exactly ten;<br/>every component and detail check is advisory<br/>validate-mermaid-diagrams.sh:154"]
        V1 --> V2 --> V3 --> V4 --> V5 --> V6
    end

    subgraph A["generate-diagram-art.sh"]
        A1["needs a Gemini API key from the environment;<br/>without one every call silently returns no image<br/>generate-diagram-art.sh:5"]:::orphan
        A2["writes into assets/generated/ at an absolute path<br/>generate-diagram-art.sh:6"]:::orphan
        A3["curl to the Google generative-language endpoint with<br/>response modalities TEXT and IMAGE<br/>generate-diagram-art.sh:16"]
        A4["inline python3 pulls the first inlineData part;<br/>failures land in a per-name debug file under /tmp<br/>generate-diagram-art.sh:30"]
        A5["five fixed prompts: evolution-line, five-substrates,<br/>judgment-broker, coordination-topology, identity-spine<br/>generate-diagram-art.sh:62"]
        A6["no exit-code contract — the loop reports per image and<br/>the script always ends by listing the directory<br/>generate-diagram-art.sh:82"]:::orphan
        A1 --> A2 --> A3 --> A4 --> A5 --> A6
    end

    V6 ~~~ A1

    N1["DOC-DRIFT: validate-mermaid-diagrams.sh asserts a diagram count<br/>and component set for a file this repository does not contain,<br/>and the model it describes — EventBus, CQRS, cache invalidation —<br/>is VisionClaw's, not the canon's. No workflow, no package script<br/>and no doc invokes it"]

    N2["DOC-DRIFT: generate-diagram-art.sh's header names<br/>gemini-2.0-flash-exp while the URL it actually calls is<br/>gemini-2.5-flash-image. Its outputs under assets/generated/<br/>are referenced by no workflow, gate or generator"]

    N3["EXTERNAL: the subjects validate-mermaid-diagrams.sh probes<br/>belong to VisionClaw — see VC-NN for the real event-flow<br/>and CQRS topology"]

    A6 ~~~ N1
    N1 ~~~ N2
    N2 ~~~ N3
```
