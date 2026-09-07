---
id: KG-06
title: Consumers — VisionClaw ingest, agentbox Loom grounding, vowl-wasm, OntoCast staging
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/ecosystem.md
adrs: []
sources:
  - ../knowledgeGraph/docs/ecosystem.md
  - ../knowledgeGraph/docs/integrations/ontocast.md
  - ../knowledgeGraph/pipeline/ontocast_import.py
  - ../knowledgeGraph/README.md
  - docs/architecture/licensing.md
verified_commit: 2791111fc
---

## KG-06.1 Sibling repository map — what actually depends on what

```mermaid
flowchart TB
    KG["this repo — knowledgeGraph<br/>ODbL corpus + AGPL pipeline + MIT explorer"]
    VC["DreamLab-AI/VisionClaw — AGPL-3.0<br/>EXTERNAL: see VC-20"]
    WV["DreamLab-AI/WasmVOWL — MIT<br/>direct upstream of explorer/"]
    VW["vowl-wasm — the NGG1 reader crate<br/>EXTERNAL: see VW-*"]
    VF["DreamLab-AI/VisionFlow — no root licence<br/>documentation/website canon"]
    AB["DreamLab-AI/agentbox — AGPL-3.0<br/>EXTERNAL: see AB-24, AB-25 — tooling provenance, no code dep"]
    MO["DreamLab-AI/Metaverse-Ontology — closest independent precedent<br/>Logseq properties + Rust extractor, different method"]
    SPR["DreamLab-AI/solid-pod-rs — AGPL-3.0<br/>indirect: cause of VisionClaw's AGPL relicence"]
    KG -->|"fetches markdown + Turtle over GitHub"| VC
    WV -->|"explorer/ is a fork, MIT lineage"| KG
    WV -.->|"same MIT-derivative lineage"| VW
    VW -->|"@dreamlab-ai/vowl-wasm release tarball"| KG
    VC -.->|"AGPL relicence caused by linking"| SPR
    note1["Excluded deliberately: VisionFlow-Ontology-Engine is a fork of<br/>OpenPlanter #40;op-cli/op-core/...#41;, nothing to do with this corpus<br/>#40;ecosystem.md:213-219#41;"]
```

## KG-06.2 VisionClaw ingest — static artefact becomes a live, reasoned graph

```mermaid
sequenceDiagram
    autonumber
    participant KG as this repo — dist/data/ontology.ttl<br/>258,200 triples
    participant SYNC as GitHubSyncService<br/>EXTERNAL: VisionClaw — see VC-20/VC-21
    participant OXI as Oxigraph<br/>embedded RocksDB RDF quad store
    participant WHELK as Whelk-rs<br/>OWL 2 EL reasoner

    KG->>SYNC: fetch markdown + ontology #40;ecosystem.md:87-88#41;
    SYNC->>OXI: write asserted triples #40;SHACL-lite gated#41;
    OXI->>WHELK: classify
    WHELK-->>OXI: inferred axioms + PROV-O
    Note over KG,WHELK: transcribed from VisionClaw's own system-overview.md —<br/>message sequence lines 193-196, store description line 122,<br/>stack table line 244 #40;ecosystem.md:102-108#41;
    Note over KG: DOC-DRIFT #40;flagged upstream#41;: VisionFlow's docs/architecture/licensing.md:11<br/>still records VisionClaw as MPL 2.0 — VisionClaw's root LICENSE is<br/>AGPL-3.0 since linking solid-pod-rs crates #40;ecosystem.md:121-127#41;
```

## KG-06.3 agentbox — Loom grounding and the ontology-bridge MCP server

```mermaid
flowchart LR
    TTL["dist/data/ontology.ttl<br/>the TBox VisionClaw reasons over"]
    LOOM["agentbox Ontology Loom facade<br/>EXTERNAL: see AB-24 — model-swappable, static scaffold ~3.5x recall"]
    BRIDGE["agentbox ontology-bridge MCP tools<br/>EXTERNAL: see AB-25 — ontology_ask/ontology_search/kg_pathfind"]
    VC20["VisionClaw ontology pipeline<br/>Oxigraph + Whelk — EXTERNAL: VC-20"]
    TTL --> VC20 --> LOOM
    VC20 --> BRIDGE
    note1["No agentbox program serves knowledgeGraph directly — the Loom and<br/>ontology-bridge ground against VisionClaw's REASONED graph #40;VC-20#41;,<br/>not this repo's static Turtle file"]
```

## KG-06.4 OntoCast integration — RDF-to-Logseq staging boundary, never a direct write

```mermaid
sequenceDiagram
    autonumber
    participant OC as OntoCast e44005d1 #40;0.6.1#41;<br/>EXTERNAL — not vendored, not forked
    participant IMPORT as pipeline.ontocast_import<br/>ontocast_import.py:140 import_graph
    participant REVIEW as review/ directory<br/>ISOLATED, not ontology/pages/
    participant HUMAN as human reviewer
    participant CORPUS as ontology/pages/

    OC->>IMPORT: standards-compliant Turtle #40;facts mode against dist/data/ontology.ttl seed#41;
    IMPORT->>IMPORT: one candidate .md per explicit owl:Class/owl:NamedIndividual<br/>public:: false, schema v3, pending-review — ontocast.md:39-41
    IMPORT->>IMPORT: every predicate/object copied into vc:ImportEvidence<br/>#40;staging record — publisher ignores it by design#41;
    IMPORT->>REVIEW: --write #40;preview is the default#41; — ontocast.md:39
    REVIEW->>HUMAN: inspect JSON report, esp. skipped#91;#93;
    HUMAN->>CORPUS: promote accepted pages via normal git history
    Note over IMPORT: grounding starts at confidence 0.5, method inferred —<br/>importer cannot know a generated claim is correct #40;ontocast.md:52-54#41;
    Note over IMPORT: INVARIANT: slug collisions reported and skipped —<br/>existing files NEVER overwritten #40;ontocast.md:49-50#41;
```
