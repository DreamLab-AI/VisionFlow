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
        BGEN["diagram-index-gen.cjs — walk, parse,<br/>validate, optionally render and index<br/>diagram-index-gen.cjs:431"]:::pipeB
        BREND["docs/diagrams/rendered/ via mmdc,<br/>gitignored and regenerable<br/>diagram-index-gen.cjs:337"]:::pipeB
        BIDX["docs/diagrams/README.md index block<br/>plus COVERAGE.md inverted indexes<br/>diagram-index-gen.cjs:361"]:::pipeB
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
    NOTE2["INVARIANT: this very file was validated by pipeline B —<br/>diagram-index-gen.cjs docs/diagrams --check --cite-check --only visionflow/<br/>diagrams/README.md:64"]
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
    participant CLI as "diagram-index-gen.cjs with a dir and flags"
    participant W as "walk<br/>diagram-index-gen.cjs:90"
    participant PT as "parseTopic"
    participant FM as "parseFrontmatter"
    participant FS as "repo filesystem"

    CLI->>CLI: parse check, render, cite-check, jobs, only and<br/>no-source-paths — anything else is a usage error, exit 2<br/>diagram-index-gen.cjs:76
    CLI->>CLI: repoRoot is the diagrams root two levels up —<br/>every source path resolves from there<br/>diagram-index-gen.cjs:87

    CLI->>W: recurse the diagrams root
    W->>W: skip hero, archive, rendered, src, upgraded, node_modules<br/>and any dot-directory<br/>diagram-index-gen.cjs:61
    W->>W: skip README.md and COVERAGE.md, and skip the root entirely —<br/>topic files live in area subdirectories only<br/>diagram-index-gen.cjs:97
    W-->>CLI: candidate .md files, then filtered by the only substring<br/>diagram-index-gen.cjs:434

    loop each topic file
        CLI->>PT: parse it, accumulating errors
        PT->>FM: frontmatter must open with a fence line and terminate<br/>diagram-index-gen.cjs:120
        FM->>FM: minimal YAML — key colon value, inline bracket lists,<br/>and dash continuation lines under the last key<br/>diagram-index-gen.cjs:132
        FM-->>PT: the frontmatter object plus the body after it
        PT->>PT: assert all seven required keys are present<br/>diagram-index-gen.cjs:149
        PT->>PT: area must be a known area AND equal the directory name<br/>diagram-index-gen.cjs:153
        PT->>PT: id must match the two-or-three-letter, two-or-three-digit shape<br/>diagram-index-gen.cjs:154
        PT->>FS: every sources path must exist under repoRoot,<br/>unless no-source-paths is set<br/>diagram-index-gen.cjs:157
        PT->>FS: every governing doc must exist, anchor stripped<br/>diagram-index-gen.cjs:161
        PT->>PT: scan the body line by line, tracking fences and H2 headings<br/>diagram-index-gen.cjs:173
        PT->>PT: count prose — any non-blank line outside a fence that is<br/>neither a heading nor an HTML comment<br/>diagram-index-gen.cjs:178
        PT-->>CLI: diagram records carrying id, title, source, md line and kind
    end

    CLI->>CLI: topic ids unique tree-wide, diagram ids unique tree-wide<br/>diagram-index-gen.cjs:438
    CLI-->>CLI: report the topic-file and diagram totals<br/>diagram-index-gen.cjs:446

    Note over CLI,FS: INVARIANT: this very file passed that pipeline before it was committed.<br/>diagrams/README.md:62 records the four invocations authors run
```

## VF-06.9 diagram-index-gen.cjs failure taxonomy — what errors, what warns
```mermaid
stateDiagram-v2
    [*] --> Parsing

    state "HARD ERRORS — collected, then exit 1" as Errors {
        [*] --> Frontmatter
        Frontmatter --> Frontmatter : "missing frontmatter — no opening fence"
        Frontmatter --> Frontmatter : "unterminated — no closing fence"
        Frontmatter --> Frontmatter : "unparseable line — neither key-value nor list item"
        Frontmatter --> Frontmatter : "missing field — any of the seven required keys"
        Frontmatter --> Frontmatter : "bad area — not one of the ten known areas"
        Frontmatter --> Frontmatter : "area does not equal the directory name"
        Frontmatter --> Frontmatter : "bad id — fails the AREA-NN shape"
        Frontmatter --> Frontmatter : "a sources path does not exist"
        Frontmatter --> Frontmatter : "a governing doc does not exist"

        Frontmatter --> Structure
        Structure --> Structure : "orphan block — a mermaid fence with no H2 above it"
        Structure --> Structure : "H2 id is not file-id dot n"
        Structure --> Structure : "dark rect — rgb fill luminance below 140"
        Structure --> Structure : "forbidden kind — mindmap, pie, quadrantChart, timeline, journey"
        Structure --> Structure : "a code fence never closes"
        Structure --> Structure : "a topic file with zero mermaid blocks"
        Structure --> Structure : "prose lines exceed three per diagram"

        Structure --> Identity
        Identity --> Identity : "two files claim one topic id"
        Identity --> Identity : "two blocks claim one diagram id"

        Identity --> Render
        Render --> Render : "render only — mmdc rejected the block"
        Render --> Render : "render only — viewBox wider than 4500px"
    }

    state "WARNINGS — printed, never fatal" as Warnings {
        [*] --> Cite
        Cite --> Cite : "cited line number exceeds the file length"
        Cite --> Cite : "the anchor line is empty"
        Cite --> Cite : "the anchor line is punctuation only"
        Cite --> Symbol
        Symbol --> Symbol : "a function-labelled participant cites outside its span"
    }

    Parsing --> Errors
    Parsing --> Warnings
    Errors --> [*] : "exit 1 after listing every error"
    Warnings --> [*] : "exit 0 — cite-check never fails the run"

    note right of Errors
        Ordering: parse and identity errors are collected first,
        render errors are appended, then one exit.
        diagram-index-gen.cjs:457
    end note

    note right of Warnings
        A blank or punctuation-only anchor is a REAL defect for an
        author even though the tool only warns — re-cite a meaningful
        line. diagram-index-gen.cjs:242
    end note

    note right of Render
        MAX_WIDTH is 4500px because wider renders are illegible at any
        zoom. The remedy is line-break wrapping, Notes capped near 90
        characters, or a split. diagram-index-gen.cjs:63
    end note
```

## VF-06.10 --cite-check and the symbol check — suffix resolution and its blind spots
```mermaid
flowchart TB
    classDef warn fill:#fff4d6,stroke:#aa8833,color:#222
    classDef blind fill:#f0f0f0,stroke:#888,color:#222

    SRC["every mermaid block's source text"] --> RE["the citation regex scans for a dotted path,<br/>a colon, a line number and an optional range end<br/>diagram-index-gen.cjs:211"]
    RE --> RES["resolve the cited path against THIS file's sources —<br/>exact equality, or a source ending with slash plus the citation<br/>diagram-index-gen.cjs:226"]
    RES --> HITS{"exactly one source matched?"}
    HITS -->|"zero or many"| SKIP["silently skipped — an ambiguous basename shared by two<br/>sources is unchecked, not an error<br/>diagram-index-gen.cjs:230"]:::blind
    HITS -->|one| READ["read the file, cached per run<br/>diagram-index-gen.cjs:217"]

    READ --> C1{"either endpoint past EOF?"}
    C1 -->|yes| W1["warn: past EOF, with the real line count<br/>diagram-index-gen.cjs:237"]:::warn
    READ --> C2["judge ONLY the anchor line's content —<br/>a range legitimately ends on a closing brace<br/>diagram-index-gen.cjs:235"]
    C2 --> C3{"anchor line blank?"}
    C3 -->|yes| W2["warn: the line is blank<br/>diagram-index-gen.cjs:242"]:::warn
    C2 --> C4{"anchor line only punctuation?"}
    C4 -->|yes| W3["warn: the line is punctuation only<br/>diagram-index-gen.cjs:243"]:::warn

    SRC --> PART["the participant regex finds every participant-as line<br/>diagram-index-gen.cjs:255"]
    PART --> FN["the function regex pulls lowercase identifiers of four or<br/>more characters from the label text BEFORE the citation<br/>diagram-index-gen.cjs:256"]
    FN --> ESC["escape hatches: a label carrying two names, or a<br/>path with a comma-separated line list, is left alone<br/>diagram-index-gen.cjs:285"]
    FN --> NEAR{"name appears within four lines above<br/>or three below the cited line?"}
    NEAR -->|yes| OKS["accepted"]
    NEAR -->|no| DEF["look for exactly one fn or function definition of that name<br/>diagram-index-gen.cjs:288"]
    DEF --> SPAN["walk braces from the definition to its close to get the body span<br/>diagram-index-gen.cjs:295"]
    SPAN --> JUDGE{"cited line inside the body, or in<br/>the three doc-comment lines above?"}
    JUDGE -->|no| W4["warn: labelled with a name whose function spans elsewhere<br/>diagram-index-gen.cjs:300"]:::warn

    WHY["WHY the symbol check exists: relocating a citation by diff preserves<br/>whatever the citation MEANT, including one already pointing at the wrong<br/>line. Re-derive from the SYMBOL, never from a computed offset<br/>diagrams/README.md:73"]

    BLIND["Documented blind spots: a bare line-number continuation, an extensionless<br/>path, a basename two sources share, a governing doc that is not also a<br/>source, and a citation resolving to a real, non-blank line whose behaviour<br/>has since been deleted<br/>diagrams/README.md:78"]:::blind
```

## VF-06.11 --render — mmdc invocation, concurrency and the width cap
```mermaid
sequenceDiagram
    autonumber
    participant G as "renderAll<br/>diagram-index-gen.cjs:334"
    participant Q as "job queue"
    participant R as "renderOne"
    participant MM as "mmdc — Nix-installed Mermaid CLI 11.16"
    participant OUT as "docs/diagrams/rendered/TOPIC/"

    G->>OUT: create rendered/ plus the topic path with the .md suffix dropped<br/>diagram-index-gen.cjs:337
    G->>Q: one job per diagram across every topic file
    G->>G: spawn the smaller of the jobs flag and the queue length,<br/>defaulting to six workers<br/>diagram-index-gen.cjs:350

    loop each queued diagram
        R->>OUT: write the block source to an .mmd file named by diagram id
        R->>MM: spawn mmdc with -i, -o and -q<br/>diagram-index-gen.cjs:314
        alt exit code 0
            MM-->>R: SVG written
            R->>R: read the viewBox width from the output<br/>diagram-index-gen.cjs:322
            alt width above MAX_WIDTH
                R-->>G: error naming the measured width and the remedies —<br/>wrap long Notes, cap them near 90 chars, or split<br/>diagram-index-gen.cjs:324
            else within budget
                R-->>G: no error
            end
        else non-zero
            MM-->>R: stderr and stdout are both accumulated
            R->>R: extract the mermaid parse-error line and its context,<br/>else the first three non-stack lines<br/>diagram-index-gen.cjs:328
            R-->>G: error reported as file, block id and the markdown line<br/>diagram-index-gen.cjs:330
        end
    end

    G-->>G: report how many of the queued diagrams rendered<br/>diagram-index-gen.cjs:454
    G-->>G: render errors join the same error list and exit 1<br/>diagram-index-gen.cjs:455

    Note over OUT: rendered/ is gitignored and regenerable, and is also one of the<br/>directories the walk skips, so output can never become input<br/>diagram-index-gen.cjs:61
    Note over G,MM: DIVERGENCE: mmdc must be on PATH. Nothing installs it and nothing<br/>checks for it — a missing binary surfaces as a spawn failure per diagram
```

## VF-06.12 Index emission and the diagram-index.yml sync gate
```mermaid
sequenceDiagram
    autonumber
    participant CI as "diagram-index.yml"
    participant GEN as "diagram-index-gen.cjs"
    participant RM as "docs/diagrams/README.md"
    participant CV as "docs/diagrams/COVERAGE.md"

    Note over CI: triggers on push to main and on pull_request touching<br/>docs/diagrams/**, the generator, or this workflow<br/>diagram-index.yml:7

    rect rgb(232, 238, 250)
    Note over CI,GEN: STEP 1 — structural check, sibling paths tolerated
    CI->>GEN: run with check and no-source-paths<br/>diagram-index.yml:34
    Note over CI: sources reach sibling checkouts a hosted runner does not have,<br/>so path existence AND cite-check are LOCAL gates only<br/>diagram-index.yml:28
    GEN-->>CI: frontmatter shape, unique ids, block structure, prose limits
    end

    rect rgb(228, 240, 232)
    Note over CI,CV: STEP 2 — the index must already be in sync
    CI->>CV: copy COVERAGE.md aside
    CI->>GEN: regenerate with no-source-paths and without check<br/>diagram-index.yml:39
    GEN->>RM: rewrite between the generated-index markers, appending a<br/>Diagram index section if they are absent<br/>diagram-index-gen.cjs:384
    GEN->>CV: emit three inverted indexes — by ADR, by governing document<br/>and by source path — plus the full diagram table<br/>diagram-index-gen.cjs:415
    Note over CV: verified_commit renders as a bare sha, or as repo-at-sha for an<br/>estate topic whose sources span repositories<br/>diagram-index-gen.cjs:407
    CI->>CI: diff the copy against the regenerated file<br/>diagram-index.yml:40
    CI-->>CI: a stale COVERAGE.md is an error, exit 1
    end

    Note over GEN,RM: INVARIANT: check and only both SUPPRESS index writing, so a scoped<br/>author run can never rewrite the tree README<br/>diagram-index-gen.cjs:462
    Note over GEN: DOC-DRIFT: the generated README line and the file's own usage banner<br/>both tell readers to run diagram-index-gen.js — the committed file<br/>has a .cjs extension<br/>diagram-index-gen.cjs:383
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
