---
id: VF-07
title: Pitch, presentation and report pipelines — release manifests, Wardley exports, LaTeX and the content that has no pipeline
area: visionflow
governing:
  - docs/BASELINE-visionflow.md
  - docs/architecture/compatibility-matrix.md
adrs: [ADR-2006]
sources:
  - scripts/generate-release-manifest.sh
  - scripts/render-wardley.mjs
  - scripts/check-fixture-drift.sh
  - docs/releases/ecosystem-release.schema.json
  - docs/releases/candidate-2026-05-22.json
  - docs/releases/README.md
  - tests/gates/release-manifest.test.sh
  - .github/workflows/harness-fitness-gates.yml
  - presentation/report/main.tex
  - presentation/report/ARXIV.md
  - presentation/report/scripts/build-arxiv-package.sh
  - presentation/report/chapters/14-implementation.tex
  - presentation/pitch-deck-2026-07/README.md
  - presentation/pitch-deck-2026-07/make-pdf.sh
  - presentation/pitch-deck-2026-07/render-batch.sh
  - pitch/visionflow-onepager.tex
  - pitch/visionflow-ecosystem-pitch.tex
  - texput.log
  - the-bubble-is-the-architecture.md
  - the-bubble-is-the-architecture-v3.md
  - docs/estate-review/evidence/adr-inventory.json
  - docs/BASELINE-visionflow.md
  - docs/architecture/compatibility-matrix.md
verified_commit: bec06dc3a
---

## VF-07.1 Artefact map — what produces each output and where it lands
```mermaid
flowchart LR
    classDef auto fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef manual fill:#fff4d6,stroke:#aa8833,color:#222
    classDef content fill:#f0f0f0,stroke:#888,color:#222

    subgraph SRC["Sources in-tree"]
        R1["presentation/report/main.tex<br/>XeLaTeX book, 28 chapters, 3 appendices<br/>ARXIV.md:5"]
        R2["presentation/report/wardley/*.html<br/>five self-contained maps with tool chrome"]
        R3["presentation/report/diagrams/*.mmd — see VF-06"]
        P1["pitch/visionflow-onepager.tex<br/>onepager.tex:1"]
        P2["pitch/visionflow-ecosystem-pitch.tex<br/>ecosystem-pitch.tex:1"]
        D1["presentation/pitch-deck-2026-07/prompt-files/*.prompt<br/>twenty self-contained image prompts<br/>pitch-deck-2026-07/README.md:5"]
        B1["the-bubble-is-the-architecture.md"]:::content
        B2["the-bubble-is-the-architecture-v3.md"]:::content
        RM["the estate itself — fourteen repository checkouts"]
    end

    R2 -->|"render-wardley.mjs<br/>render-wardley.mjs:88"| O2["presentation/report/wardley/rendered/*.svg + *.png<br/>AND presentation/report/images/wardley-0N-*.png<br/>render-wardley.mjs:78"]:::auto
    O2 --> R1
    R3 --> R1
    R1 -->|"xelatex, biber, xelatex, xelatex — BY HAND<br/>main.tex:6"| O1["presentation/report/main.pdf"]:::manual
    R1 -->|"build-arxiv-package.sh<br/>ARXIV.md:10"| O3["presentation/report/dist/arxiv-package/<br/>+ dist/arxiv-YYYY-MM-DD.tar.gz<br/>ARXIV.md:16"]:::auto

    P1 -->|"latex by hand — no script, no workflow"| O4["pitch/visionflow-onepager.pdf"]:::manual
    P2 -->|"latex by hand"| O5["pitch/visionflow-ecosystem-pitch.pdf"]:::manual
    O4 -->|"copied, not generated"| O6["pdf-reports/visionflow-onepager.pdf"]:::manual
    O5 -->|"copied, not generated"| O7["pdf-reports/visionflow-ecosystem-pitch.pdf"]:::manual

    D1 -->|"render-batch.sh, one retry per slide<br/>render-batch.sh:13"| O8["pitch-deck-2026-07/slides/slide-NN.png"]:::auto
    O8 -->|"make-pdf.sh — resize to jpg, then img2pdf<br/>make-pdf.sh:9"| O9["pitch-deck-2026-07/visionflow-pitch-deck-2026-07.pdf"]:::auto

    RM -->|"generate-release-manifest.sh<br/>generate-release-manifest.sh:240"| O10["stdout — docs/releases/*.json by redirection<br/>releases/README.md:8"]:::auto

    B1 --> NOP["NO pipeline. Two long-form root documents, 249 lines each,<br/>differing in content. Nothing builds, validates or publishes them;<br/>they are referenced once, as content the two real surfaces publish<br/>BASELINE-visionflow.md:51"]:::content
    B2 --> NOP

    ORPH["texput.log at the repo root — evidence of a real XeTeX run on<br/>2026-07-08 that ABORTED because main.tex is not at the root.<br/>Same date as dist/arxiv-2026-07-08.tar.gz<br/>texput.log:1"]:::content

    NOTE["DOC-DRIFT: pdf-reports/ holds byte-identical copies of the two pitch/ PDFs.<br/>No script produces them and no doc explains the duplication; README.md:15<br/>links the pdf-reports/ copies while README.md:232 points at pitch/"]
```

## VF-07.2 generate-release-manifest.sh — building the fourteen-repository roster
```mermaid
sequenceDiagram
    autonumber
    participant OP as "operator or tests/gates suite"
    participant G as "generate-release-manifest.sh"
    participant RO as "ROSTER — 14 pipe-delimited entries<br/>generate-release-manifest.sh:86"
    participant GIT as "each sibling checkout under the workspace"
    participant JQ as "jq -n emitter"

    OP->>G: optional --status, --fixtures-canonical,<br/>--require-fixtures, --workspace<br/>generate-release-manifest.sh:59
    G->>G: status must be local-draft, candidate or released, else exit 2<br/>generate-release-manifest.sh:70
    G->>G: jq is a hard dependency, else exit 2<br/>generate-release-manifest.sh:75

    loop each roster entry NAME, PATH, PROVENANCE, UPSTREAM, ROLE
        G->>GIT: git -C PATH rev-parse HEAD
        alt not a repository, or absent
            GIT-->>G: record head as forty zeroes, branch "missing",<br/>dirty true, present false — never omitted<br/>generate-release-manifest.sh:114
        else resolved
            GIT-->>G: head, branch --show-current, dirty from git status --short<br/>generate-release-manifest.sh:120
        end
    end

    Note over G,GIT: DETECTION FIX: presence is tested with git rev-parse, not by looking<br/>for a .git DIRECTORY — a submodule's .git is a gitdir FILE, which made<br/>the agentbox submodule read as missing at its correct path<br/>generate-release-manifest.sh:110

    Note over RO: DEFECT CLOSED: the generator covered six repositories while the ADR<br/>inventory records fourteen. loom, knowledgeGraph, WasmVOWL, visionGraph,<br/>dream-machine, logseq, ruvector and RuView could each move under a release<br/>with no record<br/>generate-release-manifest.sh:11

    G->>JQ: assemble manifest_version 2, generated_at, status,<br/>repositories, fixtures and compatibility<br/>generate-release-manifest.sh:246
    JQ-->>OP: JSON on stdout — the caller redirects it<br/>generate-release-manifest.sh:240

    Note over OP,RO: EXTERNAL: the roster spans VisionClaw (VC-NN), agentbox (AB-NN),<br/>solid-pod-rs (SP-NN), nostr-rust-forum (NF-NN), dreamlab-ai-website (DW-NN),<br/>vowl-wasm (VW-NN), knowledgeGraph (KG-NN) and visionGraph (VG-NN),<br/>plus loom, dream-machine, logseq, ruvector and RuView
    Note over RO: DIVERGENCE: the roster names the WASM ontology visualiser "WasmVOWL"<br/>at path WasmVOWL, while repository-map.md names it vowl-wasm at ../vowl-wasm —<br/>see VF-08 for the ownership consequence<br/>generate-release-manifest.sh:95
```

## VF-07.3 The fixtures block — a canonical revision set, or an explicit refusal
```mermaid
flowchart TB
    classDef refuse fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef honest fill:#fff4d6,stroke:#aa8833,color:#222

    START["--fixtures-canonical supplied?<br/>generate-release-manifest.sh:214"] -->|no| Q1{"status is local-draft<br/>AND --require-fixtures absent?"}

    Q1 -->|no| REF["ERROR: no canonical fixture revision set supplied.<br/>The message names the exact remedy and exits 3<br/>generate-release-manifest.sh:233"]:::refuse
    Q1 -->|yes| DRAFT["fixtures.status = not-compared with a reason string,<br/>so the gap is VISIBLE in the artefact rather than implied<br/>generate-release-manifest.sh:235"]:::honest

    START -->|yes| PARSE{"matches REPO@REV:DIR ?"}
    PARSE -->|no| E1["ERROR: --fixtures-canonical must be REPO@REV:DIR, exit 3<br/>generate-release-manifest.sh:157"]:::refuse
    PARSE -->|yes| ROSTER{"REPO resolves to a roster<br/>name or path?"}
    ROSTER -->|no| E2["ERROR: canonical repo is not in the release roster, exit 3<br/>generate-release-manifest.sh:164"]:::refuse
    ROSTER -->|yes| REV{"git rev-parse --verify REV^{commit}"}
    REV -->|"fails"| E3["ERROR: canonical revision does not resolve to a commit, exit 3.<br/>Plain rev-parse would echo any well-formed 40-hex string back<br/>unverified, recording a nonexistent commit as canonical<br/>generate-release-manifest.sh:173"]:::refuse
    REV -->|"resolves"| DIR{"corpus directory exists?"}
    DIR -->|no| E4["ERROR: canonical corpus directory not found, exit 3<br/>generate-release-manifest.sh:178"]:::refuse
    DIR -->|yes| DIG["digest: sha256 over the SORTED sha256-plus-filename lines<br/>of the corpus, so two manifests agree if and only if they<br/>qualified against byte-identical fixtures<br/>generate-release-manifest.sh:186"]
    DIG --> CNT{"any *.json fixtures?"}
    CNT -->|no| E5["ERROR: corpus contains no *.json fixtures, exit 3<br/>generate-release-manifest.sh:189"]:::refuse
    CNT -->|yes| CMP["run check-fixture-drift.sh against the corpus when it is<br/>executable here; verdict is match, drift, or not-run<br/>generate-release-manifest.sh:197"]
    CMP --> OUT["fixtures.status = compared, with canonical repo, resolved<br/>revision, dir, corpus_sha256 and fixture_count<br/>generate-release-manifest.sh:205"]:::ok

    NOTE["INVARIANT: the release qualification CLAIMS fixture parity, so the<br/>generator will not manufacture one. Version 1 asserted parity in prose —<br/>run npm run verify plus substrate-specific CI — with no evidence at all<br/>generate-release-manifest.sh:20"]

    NOTE2["DIVERGENCE: consumers is always an empty array. The comparison is run<br/>for a verdict but its per-repository detail is never recorded, so a<br/>compared manifest names WHICH corpus but not WHICH consumers matched<br/>generate-release-manifest.sh:196"]

    NOTE3["DOC-DRIFT: check-fixture-drift.sh is invoked with only --canonical and<br/>--quiet, so it falls back to its hardcoded consumer paths — see VF-05.6,<br/>where the same script's canonical default points at a deleted directory"]
```

## VF-07.4 ecosystem-release.schema.json — the shape a manifest must satisfy
```mermaid
classDiagram
    class EcosystemReleaseManifest {
        +integer manifest_version "const 2 — schema.json:9"
        +string generated_at "format date-time"
        +string status "local-draft | candidate | released — schema.json:11"
        +Repository[] repositories "minItems 14 — schema.json:15"
        +Fixtures fixtures
        +Compatibility compatibility
        additionalProperties = false "schema.json:102"
    }

    class Repository {
        +string name
        +string path
        +string head "pattern — exactly 40 lowercase hex chars — schema.json:22"
        +string branch
        +boolean dirty
        +boolean present "absent repos are recorded, never omitted — schema.json:27"
        +string provenance "first-party | imported | upstream — schema.json:31"
        +string|null upstream
        +string role
        additionalProperties = false
    }

    class Fixtures {
        +string status
        oneOf NotCompared or Compared "schema.json:47"
    }

    class NotCompared {
        +const status "not-compared"
        +null canonical
        +array consumers "maxItems 0"
        +string reason "REQUIRED — the gap must be stated"
    }

    class Compared {
        +const status "compared"
        +Canonical canonical "REQUIRED"
        +string verdict "match | drift | not-run — schema.json:79"
        +object[] consumers
    }

    class Canonical {
        +string repo
        +string revision "40 hex — HEAD resolved at generation time — schema.json:66"
        +string dir
        +string corpus_sha256 "64 hex over the sorted digest lines — schema.json:70"
        +integer fixture_count "minimum 1"
    }

    class Compatibility {
        +string identity
        +string mesh
        +string pod
        +string governance
        +string verification "requires fixtures.status compared before released"
    }

    EcosystemReleaseManifest o-- Repository
    EcosystemReleaseManifest o-- Fixtures
    EcosystemReleaseManifest o-- Compatibility
    Fixtures <|-- NotCompared
    Fixtures <|-- Compared
    Compared o-- Canonical

    note for EcosystemReleaseManifest "INVARIANT: minItems 14 equals the repo count in\nadr-inventory.json. A manifest listing fewer is a\ncoordinated view of PART of the estate, which is what\nversion 1 silently shipped. schema.json:14"
    note for Repository "provenance decides qualification: first-party and imported\nare qualified for release; upstream repositories are PINNED,\nnot qualified. schema.json:32"
```

## VF-07.5 candidate-2026-05-22.json against the committed schema — a v1 artefact under a v2 contract
```mermaid
flowchart TB
    classDef fail fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef doc fill:#fff4d6,stroke:#aa8833,color:#222

    CAND["docs/releases/candidate-2026-05-22.json<br/>the only committed manifest, status candidate<br/>candidate-2026-05-22.json:4"] --> V

    V["validate against docs/releases/ecosystem-release.schema.json"]

    V --> F1["manifest_version is 1; the schema pins const 2<br/>candidate-2026-05-22.json:2 vs schema.json:9"]:::fail
    V --> F2["repositories has SIX entries; the schema requires<br/>minItems 14<br/>candidate-2026-05-22.json:5 vs schema.json:15"]:::fail
    V --> F3["no repository declares provenance, which the schema<br/>lists as required for every item<br/>schema.json:18"]:::fail
    V --> F4["no fixtures block at all, and fixtures is a top-level<br/>required key — so this candidate asserts nothing about<br/>fixture parity while being labelled a candidate<br/>schema.json:7"]:::fail
    V --> F5["compatibility.verification claims parity through<br/>cargo test in named versions rather than a compared corpus<br/>candidate-2026-05-22.json:54"]:::fail

    F1 --> VERD["DOC-DRIFT: the committed release candidate cannot validate<br/>against the committed schema. The ADR-2006 closeout rewrote the<br/>generator and schema on 2026-09-05 and left the artefact at v1"]:::fail

    RM["docs/releases/README.md still describes the v1 world"] --> D1["the candidate row reads 'Pins all 6 repos at main branch HEADs'<br/>releases/README.md:17"]:::doc
    RM --> D2["the Required Fields table lists neither provenance nor<br/>fixtures, both of which the v2 schema requires<br/>releases/README.md:23"]:::doc
    RM --> D3["the generate command still writes to<br/>ecosystem-release.local.json, uncommitted by default<br/>releases/README.md:8"]

    VERSIONS["DOC-DRIFT in the pinned version claims, cross-checked against the<br/>sibling checkouts: the candidate pins solid-pod-rs v0.4.0-alpha.15 and<br/>nostr-rust-forum v3.0-rc3, which the compatibility matrix repeats"]:::doc
    VERSIONS --> X1["EXTERNAL: solid-pod-rs Cargo.toml now declares 0.5.0-alpha.9 — see SP-NN"]
    VERSIONS --> X2["EXTERNAL: nostr-rust-forum workspace crates now declare 1.0.0-beta.10,<br/>and status-reconciliation.md separately says 3.0.0-rc11 — see NF-NN"]

    GATE["The gate that would catch this does not run on the artefact:<br/>harness-fitness-gates.yml runs only the regression suite, never the<br/>generator or a validator against the committed candidate<br/>harness-fitness-gates.yml:166"]:::fail
```

## VF-07.6 release-manifest.test.sh — what the release gate actually proves
```mermaid
sequenceDiagram
    autonumber
    participant CI as "harness-fitness-gates.yml job release-manifest"
    participant T as "tests/gates/release-manifest.test.sh"
    participant G as "generate-release-manifest.sh"
    participant INV as "docs/estate-review/evidence/adr-inventory.json"
    participant SC as "docs/releases/ecosystem-release.schema.json"

    CI->>T: bash the suite — the ONLY step in this job<br/>harness-fitness-gates.yml:166

    T->>INV: read the repos inventory
    T->>G: generate a manifest
    T->>T: every inventoried repository path appears in the roster<br/>release-manifest.test.sh:27
    T->>T: every repository declares provenance<br/>release-manifest.test.sh:46

    rect rgb(250, 244, 228)
    Note over T,G: the core regression — a claim with no comparison behind it
    T->>G: local draft with no canonical revision set
    G-->>T: fixtures.status not-compared, and it SAYS so<br/>release-manifest.test.sh:59
    T->>G: --status candidate with no canonical revision set
    G-->>T: REFUSED, exit 3<br/>release-manifest.test.sh:70
    T->>G: --require-fixtures on a local draft
    G-->>T: refused too<br/>release-manifest.test.sh:78
    T->>G: --status released with no canonical revision set
    G-->>T: refused<br/>release-manifest.test.sh:83
    T->>G: a canonical revision set that does not resolve
    G-->>T: refused<br/>release-manifest.test.sh:89
    T->>G: a valid canonical revision set
    G-->>T: a comparison record with repo, revision, digest and count<br/>release-manifest.test.sh:101
    end

    T->>SC: assert the generated output against the committed schema<br/>release-manifest.test.sh:125
    alt no JSON-Schema validator installed
        T->>SC: fall back to the schema's own hard constraints —<br/>roster size against minItems, manifest_version const,<br/>and that fixtures is required<br/>release-manifest.test.sh:132
    end

    T-->>CI: RELEASE-MANIFEST-TESTS-OK or ...-FAIL

    Note over CI,SC: DIVERGENCE: the suite validates GENERATED output only. The committed<br/>candidate-2026-05-22.json is never validated by any gate — see VF-07.5
```

## VF-07.7 render-wardley.mjs — chrome-free print-resolution map export
```mermaid
sequenceDiagram
    autonumber
    participant N as "render-wardley.mjs"
    participant H as "presentation/report/wardley/0N-*.html"
    participant SVG as "wardley/rendered/0N-*.svg"
    participant MG as "ImageMagick with the librsvg delegate"
    participant IMG as "presentation/report/images/wardley-0N-*.png"

    Note over N,H: the maps ship as self-contained HTML from the map generator — each embeds<br/>a clean 1200x800 svg wrapped in the tool's page card, heading and export buttons.<br/>The book figures had been baked as full-page SCREENSHOTS of that chrome,<br/>below print resolution<br/>render-wardley.mjs:5

    N->>H: read every file matching a two-digit prefix
    N->>N: exactly five numbered maps expected, else exit 1<br/>render-wardley.mjs:66
    N->>N: extract the first svg element, dropping the card,<br/>heading and Export buttons<br/>render-wardley.mjs:43
    N->>N: collapse whitespace inside every text node — component names<br/>carry literal newlines that a browser collapses to a space<br/>and librsvg would otherwise drop, joining the words<br/>render-wardley.mjs:50
    N->>N: rewrite the opening svg tag with an Arial-metric-compatible<br/>sans stack so librsvg matches the browser figure<br/>render-wardley.mjs:37
    N->>N: insert a full-size white background rect<br/>render-wardley.mjs:57
    N->>SVG: write the clean SVG
    N->>MG: magick -density 240 -background white SVG to rendered PNG<br/>render-wardley.mjs:81
    N->>MG: magick again straight into the images/ filename the .tex expects<br/>render-wardley.mjs:78
    MG-->>IMG: 1200px SVG at density 240 gives 3000x2000 —<br/>at or above 300 DPI for the book's ~9.6in figure width<br/>render-wardley.mjs:35
    N->>MG: identify the written dimensions and log them per map<br/>render-wardley.mjs:84
    N-->>N: five maps re-exported clean at print resolution<br/>render-wardley.mjs:88

    IMG-->>IMG: consumed by the book through graphicspath images, diagrams, wardley<br/>main.tex:154
    Note over IMG: e.g. chapter 14 includes wardley-03-middle-manager-evolution.png<br/>14-implementation.tex:209
    Note over N,MG: DIVERGENCE: this is RES-e and it is a MANUAL script. No workflow runs it,<br/>nothing asserts the exported PNGs are current against the .html sources,<br/>and it hard-depends on magick and identify being on PATH
```

## VF-07.8 The LaTeX reality — what is automated, what is manual, what is orphaned
```mermaid
flowchart TB
    classDef auto fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef manual fill:#fff4d6,stroke:#aa8833,color:#222
    classDef orphan fill:#ffe0e0,stroke:#aa3333,color:#222

    subgraph BOOK["presentation/report — the book"]
        BT["main.tex declares its own toolchain in a magic comment:<br/>xelatex, with fontspec loading FreeSerif, FreeSans, FreeMono<br/>main.tex:1"]:::manual
        BC["the compile sequence is stated in a COMMENT, not a script:<br/>xelatex, biber, xelatex, xelatex<br/>main.tex:6"]:::manual
        BA["build-arxiv-package.sh IS automated — but it packages<br/>sources, it does not compile them<br/>ARXIV.md:10"]:::auto
        BOUT["main.pdf, main.aux, main.bbl, main.log, main.toc and the rest<br/>are COMMITTED build output next to the sources"]:::manual
        BT --> BC --> BOUT
        BT --> BA
    end

    subgraph PITCH["pitch/ — two standalone documents"]
        PT["onepager.tex and ecosystem-pitch.tex are article-class<br/>dark-palette documents with no shared preamble<br/>ecosystem-pitch.tex:1"]:::manual
        PO["each ships its .pdf, .aux, .log and .out beside the source"]:::manual
        PN["NO build script, NO Makefile, NO workflow —<br/>the PDFs are produced by hand and committed"]:::orphan
        PT --> PO --> PN
    end

    subgraph ORPH["Orphaned LaTeX evidence at the repo root"]
        T1["texput.log — a real XeTeX run, TeX Live 2025 on nixos,<br/>8 JUL 2026 18:23<br/>texput.log:1"]:::orphan
        T2["it was fed **main.tex at the REPO ROOT, where no main.tex exists<br/>texput.log:5"]:::orphan
        T3["Emergency stop, job aborted, file error in nonstop mode<br/>texput.log:10"]:::orphan
        T4["No pages of output<br/>texput.log:21"]:::orphan
        T1 --> T2 --> T3 --> T4
    end

    T4 --> DATE["Same date as presentation/report/dist/arxiv-2026-07-08.tar.gz —<br/>consistent with an arXiv-packaging session run from the wrong<br/>working directory. The log is an untracked-looking artefact<br/>that survived into the tree"]:::orphan

    VERD["DOC-DRIFT: there is NO LaTeX pipeline in this repository. No workflow,<br/>no package.json script and no shell script anywhere under .github/ or<br/>scripts/ invokes latex, xelatex, pdflatex or latexmk. Every PDF here was<br/>compiled by hand outside the repo's own automation"]:::orphan

    GUARD["The one automated protection over this area is copyright-guard.yml,<br/>which polices root-level *.txt — a .log is outside its pattern,<br/>so texput.log is invisible to it. See VF-05.9"]
```

## VF-07.9 build-arxiv-package.sh — assembling a self-contained arXiv tree
```mermaid
sequenceDiagram
    autonumber
    participant OP as "operator"
    participant S as "build-arxiv-package.sh"
    participant PY as "inline python3"
    participant PKG as "dist/arxiv-package/"
    participant TAR as "dist/arxiv-YYYY-MM-DD.tar.gz"

    OP->>S: run it from anywhere — paths derive from the script's own directory<br/>build-arxiv-package.sh:26
    S->>S: locate GNU FreeFont OTFs, first under /nix/store, then anywhere —<br/>absence is FATAL<br/>build-arxiv-package.sh:34
    S->>PKG: rm -rf and recreate chapters, appendices, images and fonts<br/>build-arxiv-package.sh:53

    S->>PY: rewrite main.tex's fontspec block to load the bundled fonts BY PATH<br/>build-arxiv-package.sh:57
    Note over PY: arXiv resolves setmainfont by NAME through its own fontconfig —<br/>the classic works-locally, fails-on-arXiv trap<br/>build-arxiv-package.sh:12
    S->>PKG: copy every input'd chapter and appendix<br/>build-arxiv-package.sh:101
    S->>PKG: copy the PRE-BUILT main.bbl, because arXiv does not run biber<br/>build-arxiv-package.sh:107
    S->>PKG: copy the enumerated FreeFont .otf files<br/>build-arxiv-package.sh:118

    S->>S: grep every includegraphics target and resolve each basename<br/>against the graphicspath roots<br/>build-arxiv-package.sh:124
    alt any target unresolved
        S-->>OP: FATAL: unresolved includegraphics targets, exit 1<br/>build-arxiv-package.sh:138
    else all resolved
        S->>PKG: copy ONLY the referenced images — the .mmd, .owm and orphan<br/>renders are deliberately left out<br/>build-arxiv-package.sh:139
    end
    S->>S: verify every referenced image is pdf, png or jpg<br/>build-arxiv-package.sh:141

    S->>PKG: write 00README.json declaring compiler xelatex and toplevel main.tex<br/>build-arxiv-package.sh:150
    S->>S: measure the tree against a 45 MB ceiling, leaving headroom<br/>under arXiv's ~50 MB limit<br/>build-arxiv-package.sh:43
    alt over the ceiling
        S->>S: downscale rasters to a max width with a fixed JPEG quality<br/>build-arxiv-package.sh:167
    else within budget
        S->>S: say so and change nothing<br/>build-arxiv-package.sh:181
    end

    S->>PY: build the tarball with python's tarfile rather than a system tar,<br/>members stored flat at the archive root and sorted for reproducibility<br/>build-arxiv-package.sh:185
    PY-->>TAR: gzipped source package
    S-->>OP: DONE, with package and tarball sizes<br/>build-arxiv-package.sh:204

    Note over S,PKG: INVARIANT: the repo originals are never mutated — everything is copied<br/>into dist/, and the script is idempotent<br/>build-arxiv-package.sh:20
    Note over TAR: DOC-DRIFT: ARXIV.md:18 says neither the tarball nor the assembled tree<br/>is committed, and ARXIV.md:13 calls dist/ git-ignored — yet dist/ holds two<br/>committed tarballs and an unpacked arxiv-package/ tree
```

## VF-07.10 pitch-deck-2026-07 — prompt-driven image generation to a single PDF
```mermaid
sequenceDiagram
    autonumber
    participant OP as "operator"
    participant RB as "render-batch.sh"
    participant NB as "agentbox skills/art/tools/nb-generate.cjs"
    participant SL as "slides/slide-NN.png"
    participant MP as "make-pdf.sh"
    participant PDF as "visionflow-pitch-deck-2026-07.pdf"

    OP->>RB: bash render-batch.sh
    RB->>RB: source the operator's ~/.claude/.env for credentials<br/>render-batch.sh:4
    Note over RB,NB: the generator is an ABSOLUTE path into the agentbox checkout —<br/>the deck cannot be rebuilt without that sibling repo present<br/>render-batch.sh:5

    loop each prompt-files/slide-NN.prompt
        RB->>RB: skip when a non-empty slides/slide-NN.png already exists<br/>render-batch.sh:10
        loop up to two attempts
            RB->>NB: nb-generate with the prompt text, 4K, aspect 16:9<br/>render-batch.sh:13
            alt success and the file is non-empty
                NB-->>SL: PNG written, break out of the retry loop
            else failure
                RB->>RB: delete the partial file, sleep 10, retry once<br/>render-batch.sh:16
            end
        end
    end
    RB-->>OP: BATCH DONE with a timestamp<br/>render-batch.sh:19

    OP->>MP: bash make-pdf.sh
    MP->>MP: magick each slide PNG to a 2200px-wide JPEG at quality 82<br/>make-pdf.sh:7
    MP->>PDF: img2pdf over the sorted JPEGs, falling back to magick<br/>make-pdf.sh:9
    MP-->>OP: list the resulting PDF

    Note over OP,PDF: twenty slides, each one self-contained infographic mapped to a<br/>book chapter in the deck's own narrative-arc table<br/>pitch-deck-2026-07/README.md:10
    Note over RB: DIVERGENCE: no exit-code contract. A slide that fails both attempts leaves<br/>no file and the batch still reports DONE — make-pdf.sh then silently builds a<br/>PDF with that slide missing. render.log, render2.log, render3.log and<br/>render4.log beside the script are the only record of past runs
    Note over NB: EXTERNAL: the image generator lives in agentbox — see AB-NN. Nothing in<br/>this repository pins its version or asserts its presence
```

## VF-07.11 Content with no pipeline — stated as such, not invented
```mermaid
flowchart TB
    classDef content fill:#f0f0f0,stroke:#888,color:#222
    classDef surface fill:#e0f2e4,stroke:#2f7a45,color:#222

    BASE["BASELINE-visionflow.md: the repo has exactly TWO concrete surfaces —<br/>a static marketing website and the governance canon under docs/.<br/>Everything else is content that those two surfaces publish<br/>BASELINE-visionflow.md:50"]

    BASE --> C1["the-bubble-is-the-architecture.md<br/>249 lines, long-form essay with numbered citations"]:::content
    BASE --> C2["the-bubble-is-the-architecture-v3.md<br/>249 lines, a revised edition with different content —<br/>its method note records how the revision was checked<br/>the-bubble-is-the-architecture-v3.md:117"]:::content
    BASE --> C3["presentation/the-coordination-collapse.md and<br/>presentation/google-analysis.md — the only two presentation<br/>files the drift-counter workflow watches, see VF-05"]:::content
    BASE --> C4["presentation/2026-04-20-best-companies-ai/ and<br/>presentation/enterprise-prd.md — research and narrative,<br/>no build step of any kind"]:::content

    C1 --> N1["NO automation: no linter, no link check, no word-count gate,<br/>no publish step, no reference from any workflow. The only<br/>mention anywhere in the tree is the BASELINE line above"]:::content
    C2 --> N1

    C3 --> N2["Watched but not built: drift-counter.yml lists these two files<br/>in its path filter so an edit re-runs the count gate; the book<br/>prose under presentation/ is deliberately EXCLUDED from the<br/>skills scan and reconciled by hand instead"]:::surface

    SURF["The two real surfaces"] --> S1["the static website — see VF-02"]:::surface
    SURF --> S2["the governance canon and its gates — see VF-05"]:::surface

    HONEST["INVARIANT stated plainly: three of the four items above have no pipeline<br/>at all. Their being in a repository whose CI is dense with gates does not<br/>mean anything checks them; the honest description is 'published content',<br/>which is exactly what the baseline calls them"]

    XREF["Cross-references rather than duplication: the report's mermaid figures<br/>and their render gate are VF-06; the release manifest is VF-07.2 to VF-07.6;<br/>the compatibility posture those documents cite is VF-08"]
```
