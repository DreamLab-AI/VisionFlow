---
id: VC-45
title: Code annotations, runtime concept joins and missing provenance
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md
adrs: [ADR-2143, ADR-2159, ADR-2164]
sources:
  - ../codescape/crates/codescape-ingest/src/annotations.rs
  - ../codescape/crates/codescape-server/src/xr.rs
  - ../codescape/crates/codescape-server/src/engine.rs
  - ../project/xr-client/rust/src/link_index.rs
  - ../project/xr-client/rust/src/binary_protocol.rs
  - ../project/xr-client/scripts/graph_scene.gd
verified_commit: {codescape: 58108c0357bd0dbf6588f8b6adf6e044f4c04556, visionclaw: 75b72416bac176ca8fde715cac3bd7c77657b3e7}
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

## VC-45.2 The join preserves generation identity but leaves repository provenance empty

```mermaid
flowchart TD
    CS["Parse ontology_refs into code-node refs<br/>link_index.rs:91,118"] --> PICK["Deduplicate IRI, path and line<br/>symbol wins over file roll-up<br/>link_index.rs:237,252"]
    VC["Build IRI to wire-id lists from topology metadata<br/>binary_protocol.rs:206,1298"] --> JOIN["Join targets and filter by knowledge or ontology class<br/>link_index.rs:260"]
    PICK --> JOIN
    JOIN --> REC["Emit one record per eligible target<br/>link_index.rs:265"]
    REC --> GEN["Retain both stream generations<br/>link_index.rs:285"]
    REC --> EMPTY["Repository empty and revision None<br/>link_index.rs:281,284"]
    JOIN -->|no eligible target| MISS["Increment unresolved once per chosen annotation<br/>link_index.rs:289,290"]
```

**What it shows.** Multiple target ids and symbol/file de-duplication work, but a snapshot labelled with repository and revision produces records with neither value. **Why it is this way.** The stream omits these provenance values and the join deliberately supplies empty defaults. Stream generations protect transient ids; they are not repository revisions.

## VC-45.3 Producer truncation and connection invalidation have different counters

```mermaid
flowchart TD
    ALL["Bound annotation table retains all rows<br/>annotations.rs:369"] --> CUT["Only first 32 sorted refs per node reach wire metadata<br/>annotations.rs:239"]
    CUT --> JOIN["Join sees only received refs<br/>link_index.rs:237"]
    JOIN --> COUNT["Missing targets count as unresolved<br/>link_index.rs:290"]
    TOPO["Topology replaces cached IRI and annotation tables<br/>binary_protocol.rs:1298,1299"] --> REBUILD["Scene joins on generation and population-count changes<br/>graph_scene.gd:677,685"]
    DISC["Disconnect drops records for that connection<br/>graph_scene.gd:3922,3923"] --> REBUILD
    NEW["Topology callback drops old-generation records<br/>graph_scene.gd:3929,3940"] --> REBUILD
```

**What it shows.** With 33 bound refs and every target available, the probe receives 32 refs, emits 32 links and reports zero unresolved. Generation-change cleanup is a separate mechanism. **Why it is this way.** The cap is documented, but omitted producer refs cannot be counted by a consumer that never receives them.

**Drift:** ADR-2164's “nothing built” verification predates the implemented producer, join and scene wiring. Live seed resolution remains unverified in this pass.
