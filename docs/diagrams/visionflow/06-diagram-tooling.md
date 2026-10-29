---
id: VF-06
title: Diagram tooling — the report render gate and this tree's own generator, both rendering with Pele
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
adrs: [ADR-2004, ADR-2020]
sources:
  - scripts/capture-diagram-source-snapshot.cjs
  - scripts/check-diagram-render-sources.py
  - tests/gates/diagram-index.test.cjs
  - tests/gates/run-all.sh
  - .github/workflows/diagram-render.yml
  - .github/workflows/diagram-index.yml
  - .github/workflows/deploy.yml
  - scripts/diagram-render/render.mjs
  - scripts/diagram-render/lib/preprocess.mjs
  - scripts/diagram-render/package.json
  - scripts/lib/pele-static.cjs
  - scripts/vendor/pele/VERSION
  - scripts/vendor/pele/pele.js
  - scripts/check-diagram-text.js
  - scripts/check-cdp-sidecar.mjs
  - scripts/diagram-index-gen.cjs
  - scripts/validate-mermaid-diagrams.sh
  - scripts/generate-diagram-art.sh
  - docs/diagrams/README.md
  - docs/diagrams/citation-allowlist.json
  - docs/adr/ADR-2004-diagram-baseline-vendored-render-gate.md
  - docs/adr/ADR-2020-pele-renders-every-estate-diagram.md
  - scripts/lib/jev.cjs
  - docs/jev-option.md
  - docs/BASELINE-visionflow.md
  - docs/site-verification.md
verified_commit: 1d7183a73c1eae4996a7006067d5047308fab7a5
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
        AREN["render.mjs — vendored Pele, sha256-checked, in-process,<br/>no browser; output flattened to static SVG with a baked<br/>white background and text fill<br/>render.mjs:31-61"]:::pipeA
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
        BGEN["diagram-index-gen.cjs main IIFE — walk, parse,<br/>validate, optionally render and index<br/>diagram-index-gen.cjs:839"]:::pipeB
        BALLOW["citation-allowlist.json — loadAllowlist waives the revision<br/>requirement for one named citation, never the line checks<br/>loadAllowlist diagram-index-gen.cjs:387"]:::pipeB
        BVER["a whole-tree --cite-check run writes VERIFICATION.md — verified,<br/>unverified, unresolvable and allowlisted counts, plus a<br/>declared-revisions anchor; an --only run never does<br/>writeVerificationReport diagram-index-gen.cjs:448<br/>guard diagram-index-gen.cjs:881"]:::pipeB
        BCHKV["--check-verification (hosted CI) — refuses a VERIFICATION.md<br/>written against revisions the tree no longer declares<br/>diagram-index-gen.cjs:887"]:::pipeB
        BREND["docs/diagrams/rendered/ via the same Pele build,<br/>widths measured on its layout, gitignored and regenerable<br/>renderOne diagram-index-gen.cjs:676, renderAll diagram-index-gen.cjs:688"]:::pipeB
        BIDX["docs/diagrams/README.md index block<br/>plus COVERAGE.md inverted indexes<br/>writeIndexes at diagram-index-gen.cjs:712, COVERAGE.md written diagram-index-gen.cjs:781"]:::pipeB
        BS --> BGEN
        BGEN -->|"--cite-check"| BALLOW --> BVER
        BVER -.->|"CI freshness gate"| BCHKV
        BGEN -->|"--render"| BREND
        BGEN -->|"no --check, no --only"| BIDX
    end

    WA["diagram-render.yml — PR-triggered on<br/>presentation/report/diagrams/** and the vendored Pele<br/>diagram-render.yml:17-24"] --> A
    WB["diagram-index.yml — push and PR on<br/>docs/diagrams/**, the generator and the Pele files<br/>diagram-index.yml:3-20"] --> B
    WD["deploy.yml — BLOCKING publication gate runs<br/>the browser-free baseline probe only<br/>deploy.yml:128"] --> ACHK

    subgraph O["Neither pipeline — unwired scripts"]
        OV["validate-mermaid-diagrams.sh<br/>hardcoded to a file in ANOTHER repo<br/>validate-mermaid-diagrams.sh:7"]:::orphan
        OG["generate-diagram-art.sh<br/>Gemini image generation, needs an API key<br/>generate-diagram-art.sh:5"]:::orphan
    end

    NOTE1["INVARIANT: the COMMITTED baseline is the authority, not a CI render.<br/>A green build never depends on a browser matching byte for byte<br/>ADR-2004-diagram-baseline-vendored-render-gate.md:30, kept by docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:39-40"]
    NOTE2["INVARIANT: this very file was validated by pipeline B —<br/>diagram-index-gen.cjs docs/diagrams --check --cite-check --only visionflow/<br/>diagrams/README.md:79"]
    PELE["scripts/lib/pele-static.cjs — the one shared renderer: loadPele checks the vendored<br/>pele.js against the sha256 in VERSION before import, renderStatic flattens every CSS variable<br/>pele-static.cjs:26-32 and pele-static.cjs:55"]:::pipeA
    AREN --> PELE
    BREND --> PELE
    NOTE3["INVARIANT: both pipelines now render with the same pinned Pele 1.12.0 (ADR-2020) and share<br/>pele-static.cjs, so a diagram that parses in one parses in the other; they share no other code<br/>scripts/vendor/pele/VERSION:1 and docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:28-33"]
    NOTE4["OPEN: ADR-2020 is accepted and live but implementation_status partial — agentbox, the<br/>lesson pages, the Quartz notes site and the sealmap and dream-engine generators are follow-on surfaces<br/>docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:7 and docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:52-54"]
    NOTE5["DRIFT: the generator renders with Pele but the tooling block of docs/diagrams/README.md still reads parse every block with mmdc<br/>docs/diagrams/README.md:79"]
```

## VF-06.2 diagram-render.yml — two stages, baseline first, no browser
```mermaid
sequenceDiagram
    autonumber
    participant GH as "diagram-render.yml job render-gate"
    participant CHK as "check-diagram-text.js"
    participant REN as "render.mjs"

    Note over GH: triggers only on presentation/report/diagrams/**, scripts/diagram-render/**,<br/>scripts/vendor/pele/**, pele-static.cjs, check-diagram-text.js and this workflow<br/>diagram-render.yml:16-25

    rect rgb(228, 240, 232)
    Note over GH,CHK: STAGE 1 — the AUTHORITATIVE guard. No browser, fully deterministic
    GH->>CHK: node check-diagram-text.js presentation/report/diagrams/rendered<br/>diagram-render.yml:42-43
    CHK-->>GH: exit 1 unless every committed SVG has visible text and its key labels
    end

    rect rgb(250, 244, 228)
    Note over GH,REN: STAGE 2 — re-render from source with the vendored Pele, then diff. Secondary, not truth
    GH->>GH: copy the committed baseline aside into RUNNER_TEMP/baseline<br/>diagram-render.yml:50-51
    GH->>REN: node render.mjs, in-process, no Chrome to provision<br/>diagram-render.yml:52
    REN-->>GH: rendered/*.svg overwritten in place
    GH->>CHK: re-run the visibility probe on the fresh output<br/>diagram-render.yml:53
    GH->>CHK: --diff RUNNER_TEMP/baseline against rendered<br/>diagram-render.yml:54
    CHK-->>GH: exit 1 on any visible-word drift — this is what catches an<br/>edited .mmd whose baseline was never regenerated
    GH->>GH: git diff --stat over rendered/ is INFORMATIONAL only<br/>diagram-render.yml:55-56
    end

    Note over GH: INVARIANT: the stage that installed pinned mermaid-cli and puppeteer and resolved a Chrome binary<br/>is gone, so the workflow can run on a plain hosted runner with Node 22<br/>diagram-render.yml:36-39
    Note over GH: INVARIANT: the committed baseline was re-rendered with Pele in the same change that replaced the engine,<br/>and ADR-2020 records that its text-diff against the previous Mermaid baseline reported identical visible words for all ten diagrams<br/>docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:39-40 and docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:57-60
```

## VF-06.3 render.mjs — how Pele is obtained and verified
```mermaid
flowchart TB
    classDef ci fill:#e4ecf8,stroke:#33559a,color:#222
    classDef fail fill:#ffe0e0,stroke:#aa3333,color:#222

    START["main — list the .mmd sources, create rendered/<br/>render.mjs:31-34"]:::ci --> LOAD["peleStatic.loadPele — import the vendored build<br/>render.mjs:35"]:::ci
    LOAD --> PIN["read the sha256 line from scripts/vendor/pele/VERSION<br/>pele-static.cjs:29"]:::ci
    PIN --> NOPIN["VERSION carries no sha256 line — throw<br/>pele-static.cjs:30"]:::fail
    PIN --> HASH["hash pele.js and compare<br/>pele-static.cjs:31"]:::ci
    HASH --> MISMATCH["vendored pele.js sha256 does not match VERSION — throw<br/>pele-static.cjs:32"]:::fail
    HASH --> IMPORT["import the file by URL, once per process<br/>pele-static.cjs:26"]:::ci
    IMPORT --> RENDER["renderStatic with responsive false and autoDirection false<br/>so a width belongs to the diagram, not a viewport<br/>pele-static.cjs:21 and pele-static.cjs:55"]:::ci

    NOTE["INVARIANT: no CDN, no npm renderer, no browser and no CDP client. The engine is the committed pele.js,<br/>pinned by npm integrity and sha256 in VERSION; npm pele 1.0.0 is an unrelated 2020 package and is never accepted<br/>scripts/vendor/pele/VERSION:3-5 and docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:30-33"]
    NOTE2["DRIFT: scripts/check-cdp-sidecar.mjs and docs/site-verification.md still describe a raw CDP connection to the shared browser sidecar<br/>for site checks, which is now unrelated to diagram rendering and is not touched by ADR-2020<br/>scripts/check-cdp-sidecar.mjs:20"]
```

## VF-06.4 render.mjs — static flatten, baked background and baked text fill
```mermaid
sequenceDiagram
    autonumber
    participant N as "render.mjs in node"
    participant P as "preprocess.mjs"
    participant S as "pele-static.cjs"
    participant FS as "presentation/report/diagrams/rendered/"

    loop each .mmd in sorted order
        N->>P: preprocessSource on the raw source<br/>render.mjs:40
        P-->>N: light-theme-safe definition
        N->>S: renderStatic with a white background and a per-diagram id prefix<br/>render.mjs:42-45
        S->>S: pele.render with the gate options, then flatten<br/>pele-static.cjs:56-57
        S->>S: replace each var of a Pele custom property by the default Pele declares for it,<br/>inside tag markup only so diagram text reading var is left alone<br/>pele-static.cjs:40-41
        S->>S: resolve the root-local aliases over four passes, throw if any variable survives<br/>pele-static.cjs:42-49
        S->>S: insert a full-size white rect as the FIRST child —<br/>the transparent-background trap that let dark text vanish<br/>pele-static.cjs:50
        S-->>N: svg, width, height and diagram type
        N->>N: give every text element without a fill the print fill 1f1f1f<br/>render.mjs:24-29
        N->>FS: write rendered NAME.svg with an XML prolog<br/>render.mjs:48
    end

    alt any diagram threw
        N-->>N: log FAILED per file with the Pele code, line and snippet, then exit 1<br/>render.mjs:51-59
    else all rendered
        N-->>N: report the number of diagrams rendered<br/>render.mjs:60
    end

    Note over N,FS: INVARIANT: the fill is BAKED into the file so the standalone checker reads the visible colour<br/>with no browser and no stylesheet, as the earlier gate did with browser-computed colours<br/>render.mjs:3-9
    Note over S: DIVERGENCE: the earlier text said TOP-LEVEL htmlLabels false forced SVG text nodes and measured<br/>visible counts in a page. Neither step exists now, so visible-text evidence rests on the baked fill and check-diagram-text.js<br/>render.mjs:26-29
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
    T5 --> OUT["definition rendered by Pele, whose default text colour is the<br/>dark print-safe 1f1f1f the flatten step bakes in"]:::keep

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

    NOTE["INVARIANT: pure node built-ins — no browser, no npm dependency — so<br/>this probe runs anywhere, including as a blocking step at deploy.yml:128<br/>check-diagram-text.js:16"]
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
    participant CLI as main IIFE<br/>diagram-index-gen.cjs:790
    participant W as walk<br/>diagram-index-gen.cjs:115
    participant P as parseTopic<br/>diagram-index-gen.cjs:193
    participant F as parseFrontmatter<br/>diagram-index-gen.cjs:144
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

## VF-06.9 Citation refusal, the allowlist waiver, and structural failures
```mermaid
flowchart TB
    I["Topic input"] --> P["Structural parse and duplicate checks<br/>diagram-index-gen.cjs:193"]
    P --> ERR["Accumulated errors"]
    I --> C["Citation and symbol diagnostics<br/>citeCheck diagram-index-gen.cjs:496<br/>symbolCheck diagram-index-gen.cjs:624"]
    C --> WAIVE{"allowMatch against<br/>citation-allowlist.json?<br/>diagram-index-gen.cjs:408"}
    WAIVE -->|yes| NOTE["counted allowlisted, printed with its reason;<br/>never folded into verified or into warnings<br/>diagram-index-gen.cjs:511"]
    WAIVE -->|no| STRICT{"strict-citations?"}
    STRICT -->|yes| ERR
    STRICT -->|no| WARN["Advisory warnings remain visible,<br/>per-citation reason tracked: unverified,<br/>unresolvable or allowlisted<br/>diagram-index-gen.cjs:423"]
    C --> VER["a --cite-check run without --only writes VERIFICATION.md,<br/>whatever the verdict, strict or not; an --only run<br/>leaves the whole-tree report untouched<br/>diagram-index-gen.cjs:489 and diagram-index-gen.cjs:881"]
    I --> R["Optional render and width checks<br/>diagram-index-gen.cjs:910-913"]
    R --> ERR
    ERR --> FAIL["Exit 1 when errors exist<br/>diagram-index-gen.cjs:915-918"]
    MODE["Strict citations reject no-source-paths<br/>diagram-index-gen.cjs:109"] --> STRICT
    LIMIT["A strict pass establishes citation hygiene;<br/>semantic and deployment evidence remain separate.<br/>An allowlist entry that matches nothing is stale<br/>and fails the run rather than broadening silently<br/>diagram-index-gen.cjs:606"]
    WARN --> LIMIT
    NOTE --> LIMIT
```

## VF-06.10 Citation resolution: dual paths, allowlist exemptions, inference limits
```mermaid
flowchart TB
    A["Mermaid source"] --> SCAN["Dotted path plus line/range scan<br/>CITE_RE diagram-index-gen.cjs:287"]
    SCAN --> MATCH["Exact sources: entry wins; otherwise the unique<br/>suffix match; zero or >1 hits is unresolvable/ambiguous<br/>diagram-index-gen.cjs:522-528"]
    MATCH --> EXIST["existsForTopic — worktree first, else<br/>git cat-file -e &lt;sha&gt;:&lt;path&gt; at the topic's<br/>declared revision for that repo<br/>diagram-index-gen.cjs:175-185"]
    MATCH --> MODE["revisionLines chooses source bytes<br/>diagram-index-gen.cjs:350"]
    MODE --> PIN["declared revision: git show &lt;sha&gt;:&lt;path&gt;<br/>— 'verified at revision'<br/>diagram-index-gen.cjs:364"]
    MODE --> WT["no repo mapping, no sha for repo, or a failed<br/>git show — falls back to the working tree,<br/>each reason tracked separately<br/>— 'unverified (working-tree fallback)'<br/>diagram-index-gen.cjs:344-349"]
    MODE --> WTFLAG["--worktree-citations: current working tree only,<br/>by explicit flag, not fallback"]
    PIN --> READ["Check line bounds and anchors<br/>diagram-index-gen.cjs:548"]
    WT --> WAIVE{"allowMatch in citation-allowlist.json?<br/>diagram-index-gen.cjs:408"}
    WAIVE -->|yes| ALLOWED["counted allowlisted, never verified;<br/>line checks still run against it<br/>diagram-index-gen.cjs:511-518"]
    WAIVE -->|no| UNVER["counted unverified, reason recorded<br/>diagram-index-gen.cjs:541"]
    ALLOWED --> READ
    UNVER --> READ
    WTFLAG --> READ
    A --> BARE["Bare-line context from explicit citations and participant bindings<br/>diagram-index-gen.cjs:596-598"]
    BARE --> READ
    A --> SYMBOL["Function-labelled participant checked against a unique definition<br/>symbolCheck diagram-index-gen.cjs:618"]
    READ --> DIAG["Diagnostic: ambiguous, missing, unreadable, out of bounds or empty anchor"]
    SYMBOL --> DIAG
    DIAG --> EXIT["Strict mode adds diagnostics to errors<br/>diagram-index-gen.cjs:884"]
    LIMIT["Context inference and brace counting are heuristics.<br/>A real source line can still support the wrong claim.<br/>An allowlist waiver is of the revision requirement<br/>alone, never of whether the cited line is right."] --> EXIT
```

## VF-06.11 Render invocation and width boundary
```mermaid
sequenceDiagram
    participant G as renderAll<br/>diagram-index-gen.cjs:688
    participant R as renderOne<br/>diagram-index-gen.cjs:676
    participant M as Pele via renderStatic
    participant O as rendered topic directory
    G->>G: load Pele once, then walk every block in order
    G->>R: render current block
    R->>O: write exact Mermaid input
    R->>M: renderStatic of the block source
    alt Pele throws
        M-->>R: PeleError with code, line and snippet
        R-->>G: diagnostic naming the topic, diagram and md line
    else successful render
        M-->>R: flattened SVG and its width
        R->>O: write the SVG
        R->>R: reject a width above 4500px, with advice to wrap Notes or split<br/>diagram-index-gen.cjs:685
    end
    R-->>G: success or diagnostic
    Note over G,O: Exact-source comparison and execution logs are distinct evidence.<br/>A cached SVG by itself is not proof of a fresh successful render.
    Note over G,M: TENSION: widths come from Pele's own Node text-width table and are not comparable with the earlier mmdc numbers,<br/>so a block that passed the old 4500px ceiling may fail now. The refresh found 17 corpus blocks over the ceiling and 4 with a bare semicolon<br/>diagram-index-gen.cjs:670-674 and docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:60-63
```

## VF-06.12 diagram-index.yml — four steps, and what a hosted runner cannot see
```mermaid
sequenceDiagram
    autonumber
    participant CI as "diagram-index.yml job index"
    participant G as writeIndexes<br/>diagram-index-gen.cjs:717
    participant T as "tests/gates/diagram-index.test.cjs"
    participant RM as "docs/diagrams/README.md"
    participant CV as "docs/diagrams/COVERAGE.md"

    Note over CI: "push to main and pull_request, path-filtered to docs/diagrams/**,<br/>the generator, the gate test and the workflow itself<br/>diagram-index.yml:3-20"

    rect rgb(232, 238, 250)
    Note over CI,G: "STEP 1 — structural check, sibling paths tolerated"
    CI->>G: "diagram-index-gen.cjs docs/diagrams --check --no-source-paths<br/>diagram-index.yml:55-56"
    Note over G: "frontmatter shape, unique ids, heading and block structure,<br/>prose limits. sources: path existence is WAIVED"
    end

    rect rgb(228, 240, 232)
    Note over CI,T: "STEP 2 — prove the gate can still fail"
    CI->>T: "node tests/gates/diagram-index.test.cjs<br/>diagram-index.yml:58-59"
    T->>T: "three fixture topics carrying the SAME ADR number in three areas<br/>diagram-index.test.cjs:10"
    T-->>CI: "coverage must key them visionflow:ADR-2002, agentbox:ADR-2002,<br/>estate-unresolved:ADR-2002 and must NOT collapse them<br/>diagram-index.test.cjs:19"
    T->>T: "a past-EOF citation: advisory under --cite-check,<br/>exit 1 under --strict-citations<br/>diagram-index.test.cjs:29"
    T-->>CI: "--strict-citations with --no-source-paths is refused with exit 2,<br/>never silently downgraded — diagram-index.test.cjs:30"
    T->>T: "declared-revision mode reads git show at the stamped sha,<br/>so a drifted working tree passes and --worktree-citations fails<br/>diagram-index.test.cjs:51 and diagram-index.test.cjs:52"
    T->>T: "the citation allowlist: an unverifiable citation not named in it<br/>still fails strict mode — naming it waives only that one citation,<br/>counted apart as allowlisted<br/>diagram-index.test.cjs:54"
    T-->>CI: "a waiver that matches nothing fails the run rather than<br/>broadening silently<br/>diagram-index.test.cjs:95"
    end

    rect rgb(232, 245, 245)
    Note over CI,G: "STEP 2b — every block must parse and stay under 4500px"
    CI->>G: "diagram-index-gen.cjs docs/diagrams --check --no-source-paths --render<br/>diagram-index.yml:61-65"
    Note over G: "Pele renders in-process, so the hosted runner needs neither a browser nor a sibling checkout.<br/>A diagram that does not parse cannot reach main — the corpus render gate had never run hosted before this step"
    end

    rect rgb(245, 232, 250)
    Note over CI,CV: "STEP 3 — the local strict-citations run must be fresh"
    CI->>CV: "diagram-index-gen.cjs docs/diagrams --check --no-source-paths<br/>--check-verification<br/>diagram-index.yml:70-71"
    Note over CV: "recomputes declaredRevisions from frontmatter alone (no source<br/>access) and refuses VERIFICATION.md if its declared-revisions<br/>anchor no longer matches — also checks citation-allowlist.json<br/>is well formed and names real topics<br/>diagram-index-gen.cjs:887"
    end

    rect rgb(250, 244, 228)
    Note over CI,CV: "STEP 4 — the index must already be in sync"
    CI->>CV: "copy COVERAGE.md aside, then regenerate<br/>diagram-index.yml:73-76"
    G->>RM: "emit topic tables and the regeneration command"
    G->>CV: "qualify ADR identities by repository<br/>diagram-index-gen.cjs:769"
    CI->>CI: "diff the copy against the regenerated file, and a mismatch<br/>raises a workflow error saying COVERAGE.md is stale, exit 1<br/>diagram-index.yml:77"
    end

    Note over CI,CV: "INVARIANT: sources: reach sixteen repositories at forty-one declared<br/>revisions, two of them unfetchable here (a private repo, a local-only<br/>clone), so --no-source-paths is the only runnable form on this runner.<br/>Verification stays LOCAL — this runner gates its FRESHNESS instead<br/>diagram-index.yml:30"
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

The companion `scripts/capture-diagram-source-snapshot.cjs` runs the strict current-worktree check between two source-hash captures and refuses to publish a snapshot if the input bytes changed. Its directory entries are markers rather than recursive attestations, and it excludes RuView source reads. Semantic audit and render execution remain separate evidence.

## VF-06.14 check-diagram-render-sources.py — render-input byte parity, and what it refuses to claim
```mermaid
flowchart TB
    classDef parse fill:#e4ecf8,stroke:#33559a,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef fail fill:#ffe0e0,stroke:#aa3333,color:#222

    CLI["python3 scripts/check-diagram-render-sources.py ROOT [--report FILE]<br/>check-diagram-render-sources.py:9"]:::parse
    CLI --> WALK["sorted glob of every AREA/NN-*.md under the root;<br/>a file not starting with a frontmatter fence is skipped<br/>check-diagram-render-sources.py:14"]:::parse
    WALK --> SCAN["track the last '## ID.n ' heading, then capture each<br/>fenced block's body verbatim<br/>check-diagram-render-sources.py:23"]:::parse
    SCAN -->|"a mermaid block with no heading above it"| HALT["SystemExit Missing diagram heading — the walk stops,<br/>it does not warn and continue<br/>check-diagram-render-sources.py:32"]:::fail

    SCAN --> CMP["for each mermaid block compare THREE things<br/>check-diagram-render-sources.py:40"]:::parse
    CMP --> T1["render_input_matches — the committed rendered/.../ID.mmd<br/>must equal the current block bytes exactly"]:::parse
    CMP --> T2["the sibling SVG must exist and its viewBox must parse<br/>check-diagram-render-sources.py:38"]:::parse
    CMP --> T3["width must be present and at most 4500 px"]:::parse

    T1 --> VERD{"ok = all three"}
    T2 --> VERD
    T3 --> VERD
    VERD -->|no| E1["exit 1, and the printed summary names<br/>topics_needing_render<br/>check-diagram-render-sources.py:49"]:::fail
    VERD -->|"yes, and at least one record"| E0["exit 0<br/>check-diagram-render-sources.py:50"]:::ok
    VERD -->|"zero records"| E2["exit 1 — an empty run is a failure,<br/>not a vacuous pass"]:::fail

    REP["--report writes source_sha256, svg_sha256, width and the<br/>per-diagram verdict, alongside a method string that states<br/>the limit in the artefact itself<br/>check-diagram-render-sources.py:45"]:::ok
    E0 --> REP

    SIB["sibling instrument: capture-diagram-source-snapshot.cjs hashes every<br/>topic and every declared source BEFORE and AFTER a passing strict<br/>worktree check, and refuses to publish if the bytes moved between<br/>capture-diagram-source-snapshot.cjs:48"]:::ok
    REP --- SIB

    LIM["INVARIANT: neither instrument makes a semantic claim. Byte parity<br/>between a topic block and its render input says the SVG came from<br/>this text, never that the text is true<br/>check-diagram-render-sources.py:45"]
    REP -.-> LIM

    DIV["DIVERGENCE: this checker is wired to no workflow and no npm script.<br/>It reads docs/diagrams/rendered/, which is gitignored and regenerable,<br/>so on a fresh checkout every diagram reports render_input_matches false"]:::fail
    E1 -.-> DIV
```

**Invariant (--only leaves VERIFICATION.md alone; Debt resolved at VisionFlow 54e1d02):** `--only` filters the topic set before anything else (`scripts/diagram-index-gen.cjs:843`) and suppresses the README and COVERAGE writes (`scripts/diagram-index-gen.cjs:920`); since 54e1d02 the cite-check block also skips `writeVerificationReport` under `--only` (`scripts/diagram-index-gen.cjs:832`), so a per-topic `--cite-check --only` run no longer overwrites the committed VERIFICATION.md with one topic's counts and a subset declared-revisions anchor (`scripts/diagram-index-gen.cjs:442`). Only a whole-tree run writes it, and `--check-verification` (`scripts/diagram-index-gen.cjs:837`) judges that.

## VF-06.15 The opt-in Jev advisory step — a typed-decision review that never gates
```mermaid
flowchart TB
    FLAG["--jev, --jev=diagram, --jev-report PATH, --jev-max N parsed beside the other flags;<br/>VISIONFLOW_JEV enables the same from the environment, the CLI value wins<br/>scripts/diagram-index-gen.cjs:104-106 and scripts/lib/jev.cjs:51-58"]
    OFF["flag off, or no endpoint, or a local-only consumer refused:<br/>the generator prints what it printed before and exits the same<br/>scripts/lib/jev.cjs:13-15"]
    RUN["jevDiagramReview runs AFTER every deterministic check, for at most 5 diagrams by default<br/>scripts/diagram-index-gen.cjs:810-812 and scripts/diagram-index-gen.cjs:886"]
    ASK["per diagram: the Mermaid source plus bounded excerpts of the cited source lines<br/>sent as a typed question set, data class local-only<br/>scripts/diagram-index-gen.cjs:819-823"]
    OUT["a contradicted verdict is printed to stderr as JEV-ADVISORY with its path:line anchors,<br/>optionally written to a JSON report; nothing reaches an index, VERIFICATION.md or verified_commit<br/>scripts/diagram-index-gen.cjs:829-834"]
    FLAG --> RUN --> ASK --> OUT
    FLAG -.-> OFF
    N1["INVARIANT: authority stays in code — Jev only asks and validates, writes nothing and approves nothing,<br/>and every fallback must behave exactly as before Jev existed<br/>scripts/lib/jev.cjs:7-18"]
    N2["OPEN: every item in the option is a proposed research candidate with no measured uplift,<br/>and the confidence floor of 0.5 is explicitly unfitted<br/>docs/jev-option.md:5 and scripts/lib/jev.cjs:41-42"]
    OUT --- N1
    ASK --- N2
```

**What it shows.** Since the earlier stamp the generator can ask a typed-decision model whether a diagram contradicts the source lines it cites, as an advisory line after all the deterministic gates have already passed or failed.
**Why it is this way.** The checks that decide the exit code, citation resolution, render and width, stay deterministic; the option adds a second opinion that can flag a claim for human review before a restamp, and cannot change a verdict (docs/jev-option.md:3-5, scripts/diagram-index-gen.cjs:785-790).
