---
id: VG-01
title: Vault structure and the publication contract — frontmatter gate, inclusion divergence
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-001, ADR-VG-002]
sources:
  - ../visionGraph/README.md
  - ../visionGraph/CLAUDE.md
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/docs/adr/README.md
  - ../visionGraph/pipeline/jsonld_parser.py
  - ../visionGraph/licensing/NOTICE
verified_commit: 9e308164c
---

## VG-01.1 Repo composition — two vaults, one pipeline, one publishing tool

```mermaid
flowchart TB
    subgraph REPO["visionGraph — the authored corpus, NOT a distribution mirror"]
        KV["knowledge/ — the PUBLISHED vault<br/>pages/ journals/ assets#40;symlink#41; .obsidian/<br/>8,671 markdown pages under pages/"]
        WV["working/ — the RESEARCH vault<br/>574 pages — same shape, not published"]
        PIPE["pipeline/ — JSON-LD → Turtle/WebVOWL/NGG1<br/>17 modules — visionGraph/README.md:19-21"]
        PUBTOOLS["publishing-tools/WasmVOWL/<br/>vendored explorer checkout — its OWN copy,<br/>distinct from knowledgeGraph's explorer/"]
        STATIC["static/ns/v2.jsonld"]
        LIC["licensing/ — LICENSE-AGPL-3.0.txt<br/>LICENSE-ODbL-1.0.txt, NOTICE"]
        TRANS["transcripts/ — podcast transcript store<br/>outside both vaults on purpose #40;README.md#41;"]
        GH[".github/workflows/publish.yml<br/>THE actual publisher — see VG-03"]
    end
    KV -->|"symlink"| ASSETS["working/assets — 961 MB, stored once"]
    WV --> ASSETS
    KV --> PIPE
    PIPE --> PUBTOOLS
    GH --> PIPE
    GH --> PUBTOOLS
    note1["INVARIANT: nothing hard-codes a corpus path — VAULT_ROOT is the single<br/>path authority, every consumer derives sub-paths from it #40;README.md 'How<br/>consumers bind to it'#41;"]
```

## VG-01.2 Vault contract — the frontmatter publication gate

```mermaid
flowchart LR
    PAGE["knowledge/pages/&lt;Ns&gt;/&lt;Title&gt;.md"]
    FM["YAML frontmatter block<br/>public: true — the KG inclusion gate<br/>CLAUDE.md:10-11"]
    OWL["non-empty owl-class — ingests UNCONDITIONALLY<br/>regardless of public: value"]
    NONE["NO frontmatter at all"]
    PAGE --> FM
    FM -->|"public: true"| INCLUDED["included in knowledge graph"]
    FM -->|"owl-class set"| INCLUDED
    PAGE -.-> NONE
    NONE -->|"gate FAILS CLOSED"| PRIVATE["private — never ingested"]
    note1["INVARIANT: identity is the vault-relative path under pages/ WITHOUT .md,<br/>with / as the namespace separator #40;README.md 'vault contract'#41;. Journals<br/>are YYYY-MM-DD.md and excluded from graph ingest"]
    note2["EXTERNAL: normative spec is VisionClaw's docs/VAULT-corpus-format.md<br/>#40;ADR-2040/2041/2042#41; — see VC-21"]
```

## VG-01.3 Current behaviour vs proposed closeout — three distinct producers, three inclusion policies

```mermaid
flowchart TB
    SRC["visionGraph — authored corpus of record<br/>PUBLICATION-contract.md:7"]
    SITE["site publisher — reads JSON-LD Page vc:public<br/>excludes _misc and dot-directories, recurses namespaces"]
    VC["VisionClaw ingest — documented frontmatter interface<br/>public OR owl-class — EXTERNAL: VC-21"]
    AB["agentbox local ontology reader<br/>DIFFERENT top-level/private-page projection<br/>EXTERNAL: see AB-25"]
    KG["knowledgeGraph — separate distribution tree<br/>deploy target of the publisher, not source — EXTERNAL: KG-*"]
    SRC --> SITE
    SRC --> VC
    SRC --> AB
    SITE --> KG
    note1["DIVERGENCE: these three consumers have DISTINCT inclusion policies —<br/>changing one flag does not establish removal from every consumer<br/>#40;PUBLICATION-contract.md:9#41;"]
    note2["DIVERGENCE: source, distribution and deployed artefact identities<br/>must NOT be collapsed into one revision #40;PUBLICATION-contract.md:7#41;"]
```

## VG-01.4 Reproducible boundary cases the proposed ADRs name

```mermaid
flowchart LR
    B1["omitted malformed json-ld fences<br/>#40;source parser silently drops them#41;"]
    B2["truthy string publication flags<br/>#40;'false' as a Python truthy value#41;"]
    B3["private-ancestor details leaking<br/>into derived public exports"]
    B4["asserted vs inferred IRI representation<br/>divergence across outputs"]
    B1 & B2 & B3 & B4 --> EVID["PUBLICATION-contract.md:11 — reproducible boundary<br/>cases; NO real private-page disclosure established"]
    EVID --> ADR1["ADR-VG-001 — proposed<br/>make inclusion + inferred visibility explicit"]
    EVID --> ADR2["ADR-VG-002 — proposed<br/>bind corpus/bundle/consumer to one generation"]
    note1["Both ADR-VG-001 and ADR-VG-002 are decision_status: proposed,<br/>activation_status: inactive #40;docs/adr/README.md#41; — presence in the<br/>index is NOT activation"]
```
