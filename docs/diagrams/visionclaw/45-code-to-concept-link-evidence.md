---
id: VC-45
title: Code annotations, runtime concept joins and missing provenance
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md
adrs: [ADR-2143, ADR-2159, ADR-2164]
sources:
  - ../project/docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md
  - ../project/codescape/crates/codescape-ingest/src/annotations.rs
  - ../project/codescape/crates/codescape-server/src/xr.rs
  - ../project/codescape/crates/codescape-server/src/engine.rs
  - ../project/xr-client/rust/src/link_index.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/scripts/graph_scene.gd
verified_commit: {codescape: e3b76f63e801c6a8460110b93e88a0b420e9b0b4, visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The [executable audit](../../estate-review/2026-10-31-xr-volumes-and-code-links.md) follows real Codescape annotation scanning and topology emission into the verbatim pure XR join module. It does not exercise the gdext wrapper or live streams.

## For the business

A code-to-concept edge is an authored annotation resolved against the loaded graph. It does not prove that the code implements the concept, and current records cannot identify the repository revision that supplied the claim.

## VC-45.1 The scanner recognises line prefixes without language lexical context

```mermaid
flowchart TD
    FILE["Read UTF-8 file up to 4 MiB<br/>annotations.rs:247"] --> LINE["Trim leading whitespace and recognise comment leader<br/>annotations.rs:125"]
    LINE --> SCAN["Scan each matching line for marker and IRI shapes<br/>annotations.rs:159"]
    EX["Comment-looking line inside a Rust raw string<br/>also enters this branch<br/>annotations.rs:125,159"] --> LINE
    SCAN --> BIND["Prefer directly documented symbol, then containing symbol, then file<br/>annotations.rs:191"]
    BIND --> ROLL["Rebind side table and roll symbols up to file<br/>annotations.rs:314"]
    ROLL --> CAP["Sort, deduplicate and keep at most 32 refs per node<br/>annotations.rs:236,239"]
    CAP --> WIRE["Attach ontology_refs to projected topology nodes<br/>codescape-server/src/xr.rs:26,52"]
```

**What it shows.** The probe compiles its raw-string fixture as valid Rust, then observes the example become a bound annotation and two target links. Real comments work and bare, inline and malformed tokens are refused. **Why it is this way.** Scanning is line-oriented; it does not ask the language parser whether a matching line lies inside a string.

**Tension:** The contract describes a comment annotation, while the recogniser accepts comment-shaped string content (`annotations.rs:125`).

## VC-45.2 The consumer now reads repository provenance that the pinned producer never sends

```mermaid
flowchart TD
    PROD["Pinned producer metadata: source_file, domain, size, type, community, tier, no repo or revision<br/>codescape-server/src/xr.rs:42-49"] --> CS
    CS["Parse ontology_refs into code-node refs<br/>link_index.rs:93,124"] --> READ["Read metadata.repo, else empty, and metadata.revision, else None<br/>link_index.rs:132,138"]
    READ --> PICK["Deduplicate IRI, path and line<br/>symbol wins over file roll-up<br/>link_index.rs:253,268"]
    VC["Build IRI to wire-id lists from topology metadata<br/>binary_protocol.rs:210,1409"] --> JOIN["Join targets and filter by knowledge or ontology class<br/>link_index.rs:276"]
    PICK --> JOIN
    JOIN --> REC["Emit one record per eligible target<br/>link_index.rs:281"]
    REC --> GEN["Retain both stream generations<br/>link_index.rs:301"]
    REC --> PROV["Provenance repo and revision copied from the code ref<br/>link_index.rs:297,300"]
    PROV -.-> EMPTY["TENSION: against the pinned producer both stay empty,<br/>although the field docs call that the legacy case<br/>link_index.rs:198,204"]
    JOIN -->|no eligible target| MISS["Increment unresolved once per chosen annotation<br/>link_index.rs:305,306"]
```

**What it shows.** VisionClaw's half of the provenance repair has landed: the join copies whatever repository and revision the code stream supplies. **Why it is this way.** The revision is a monotonic Codescape snapshot counter, not a Git commit (`link_index.rs:114`), so even when populated it does not name the source revision of a claim.
**Tension (cross-repo):** The producer half (`repo` and `revision` in `initial_graph_load` metadata) exists only on standalone Codescape after `e3b76f6`; VisionClaw `ec785afa9` pins `e3b76f6`, whose `codescape-server/src/xr.rs:42-49` emits neither, so every deployed record is the "legacy" empty case.

## VC-45.3 Producer truncation and connection invalidation have different counters

```mermaid
flowchart TD
    ALL["Bound annotation table retains all rows<br/>annotations.rs:369"] --> CUT["Only first 32 sorted refs per node reach wire metadata<br/>annotations.rs:239"]
    CUT --> JOIN["Join sees only received refs<br/>link_index.rs:253"]
    JOIN --> COUNT["Missing targets count as unresolved<br/>link_index.rs:306"]
    TOPO["Topology replaces cached IRI and annotation tables<br/>binary_protocol.rs:1409,1410"] --> REBUILD["Scene joins on generation and population-count changes<br/>graph_scene.gd:750,758"]
    DISC["Disconnect drops records for that connection<br/>graph_scene.gd:4650,4651"] --> REBUILD
    NEW["Topology callback drops old-generation records<br/>graph_scene.gd:4662,4682"] --> REBUILD
```

**What it shows.** With 33 bound refs and every target available, the probe receives 32 refs, emits 32 links and reports zero unresolved. Generation-change cleanup is a separate mechanism. **Why it is this way.** The cap is documented, but omitted producer refs cannot be counted by a consumer that never receives them.

**Drift:** ADR-2164 still records `implementation_status: none` (`ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md:6`) although producer, join and scene wiring exist, and its 2026-11-02 note says the join leaves repo empty at `link_index.rs:281-285` (same ADR, line 26), which the join no longer does (`link_index.rs:297`).
