---
id: VG-07
title: Reviewed source admission, header staging and importer limits
area: visiongraph
governing:
  - ../visionGraph/docs/adr/ADR-VG-005-licensed-source-admission.md
  - ../visionGraph/docs/adr/ADR-VG-006-ontology-to-obsidian.md
sources:
  - ../visionGraph/docs/adr/ADR-VG-005-licensed-source-admission.md
  - ../visionGraph/docs/adr/ADR-VG-006-ontology-to-obsidian.md
  - ../visionGraph/planning/space-science-and-geo-sensing/sources.md
  - ../visionGraph/planning/space-science-and-geo-sensing/conversion-and-linking.md
  - ../visionGraph/scripts/space-corpus/README.md
  - ../visionGraph/scripts/space-corpus/Cargo.toml
  - ../visionGraph/scripts/space-corpus/src/main.rs
adrs: [ADR-VG-005, ADR-VG-006]
verified_commit: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6
---

## For developers

Follow the project-specific importer from reviewed RDF snapshots to catalogue-driven headers and delegated writes; the [source-admission audit](../../estate-review/2026-11-01-visiongraph-source-admission.md) distinguishes implemented checks from manual review and later publisher gates.

## For the business

Curators can preserve source references while building editable knowledge pages, but successful staging does not establish licence admission, scientific correctness or readiness to publish; those decisions need their own evidence.

## VG-07.1 Admission policy and RDF inventory are separate steps

```mermaid
flowchart TB
    POLICY["Reviewed source register<br/>version, digest, notice and adaptations"]
    FILE["Caller supplies a local RDF file"]
    PARSE["Oxigraph parser<br/>Turtle for .ttl, otherwise RDF/XML"]
    CLASSES["Explicit named owl:Class subjects<br/>labels, named parents, direct predicates"]
    OUT["JSON inventory<br/>file hash and total quad count"]
    LIMIT["Restriction references remain<br/>nested blank-node triples are not enumerated here"]
    POLICY -.-> FILE --> PARSE --> CLASSES --> OUT --> LIMIT
    %% Source: docs/adr/ADR-VG-005-licensed-source-admission.md:23
    %% Source: planning/space-science-and-geo-sensing/sources.md:25
    %% Source: scripts/space-corpus/src/main.rs:311
    %% Source: scripts/space-corpus/src/main.rs:321
    %% Source: scripts/space-corpus/src/main.rs:341
    %% Source: scripts/space-corpus/src/main.rs:370
```

**What it shows:** the parser reads local bytes without following imports; inventory computes a digest but accepts no expected digest or admission register (`scripts/space-corpus/src/main.rs:311`).
**Why it is this way:** the decision admits reviewed selections rather than recursive upstream imports, with raw sources and separate construct accounting retained (`docs/adr/ADR-VG-005-licensed-source-admission.md:28`, `scripts/space-corpus/README.md:25`).

## VG-07.2 Catalogue validation checks names and targets, not complete semantics

```mermaid
flowchart TB
    INPUT["Catalogue plus live knowledge vault<br/>and ontology vocabulary"]
    IDS["Unique titles, safe path spelling<br/>known domain, existing reuse identity"]
    IRI["New page ID and derived resource collision checks"]
    EDGES["Known relation keys<br/>existing or staged targets, no self-link"]
    MAP["Mapping disposition allowlist<br/>create, reuse, alias or excluded"]
    LIMIT["DEBT: no mapping coverage join<br/>no alias uniqueness or multi-page cycle check"]
    INPUT --> IDS --> IRI --> EDGES --> MAP --> LIMIT
    %% Source: scripts/space-corpus/src/main.rs:379
    %% Source: scripts/space-corpus/src/main.rs:404
    %% Source: scripts/space-corpus/src/main.rs:418
    %% Source: scripts/space-corpus/src/main.rs:430
    %% Source: scripts/space-corpus/src/main.rs:441
    %% Source: scripts/space-corpus/src/main.rs:454
    %% Source: planning/space-science-and-geo-sensing/conversion-and-linking.md:47
```

**What it shows:** these checks reject selected structural errors but do not implement the conversion contract's complete collision, axiom and source-accounting requirements (`scripts/space-corpus/src/main.rs:404`, `planning/space-science-and-geo-sensing/conversion-and-linking.md:47`).
**Why it is this way:** the CLI consumes a reviewed catalogue; scientific review, reasoning and final-public-state checks remain separate acceptance work (`docs/adr/ADR-VG-006-ontology-to-obsidian.md:43`).

## VG-07.3 Staging writes sequential headers using the catalogue visibility flag

```mermaid
flowchart TB
    VALID["Validate catalogue<br/>refuse an existing output directory"]
    CREATE["Create output directory"]
    EACH["For each new candidate<br/>require a nonempty sources list"]
    HEADER["Class, minted IRI, draft lifecycle<br/>generated actor/time and source values"]
    PUBLIC["public comes from catalogue<br/>default false, not forced private"]
    WRITE["Parse YAML and write this header"]
    FAIL["Later error leaves earlier files<br/>no directory rollback"]
    VALID --> CREATE --> EACH --> HEADER --> PUBLIC --> WRITE --> FAIL
    %% Source: scripts/space-corpus/src/main.rs:102
    %% Source: scripts/space-corpus/src/main.rs:388
    %% Source: scripts/space-corpus/src/main.rs:466
    %% Source: scripts/space-corpus/src/main.rs:470
    %% Source: scripts/space-corpus/src/main.rs:473
```

**What it shows:** staging skips reuse entries and writes body-free headers individually; source objects and generation time are copied without admission-register checks (`scripts/space-corpus/src/main.rs:388`, `scripts/space-corpus/src/main.rs:473`).
**Why it is this way:** the original header stage deferred article bodies and verification claims; the current function also supports the catalogue's later visibility state (`docs/adr/ADR-VG-006-ontology-to-obsidian.md:28`, `scripts/space-corpus/src/main.rs:477`).

## VG-07.4 Apply and promote delegate to a PATH-resolved vault executable

```mermaid
flowchart TB
    APPLY["Apply: count .md paths<br/>optional resume requires identical live bytes"]
    CREATE["vault create file --expect docs=1 --json<br/>require successful exit and created=true"]
    PROMOTE["Promote: select new entries and named pilots<br/>count guard, empty body and draft lifecycle"]
    EDIT["vault edit ID --set public=true<br/>--expect docs=1,blocks=1 --json"]
    RESULT["Successful exit is the promotion result<br/>no post-write read or JSON outcome check"]
    APPLY --> CREATE
    PROMOTE --> EDIT --> RESULT
    NOTE["DEBT: no staged digest or review receipt binding<br/>CLI dependency resolved through PATH"]
    CREATE --> NOTE
    RESULT --> NOTE
    %% Source: scripts/space-corpus/src/main.rs:153
    %% Source: scripts/space-corpus/src/main.rs:167
    %% Source: scripts/space-corpus/src/main.rs:178
    %% Source: scripts/space-corpus/src/main.rs:210
    %% Source: scripts/space-corpus/src/main.rs:223
    %% Source: scripts/space-corpus/src/main.rs:488
    %% Source: scripts/space-corpus/src/main.rs:500
    %% Source: scripts/space-corpus/src/main.rs:528
    %% Source: scripts/space-corpus/src/main.rs:543
```

**What it shows:** the importer checks per-operation counts and selected outcomes; a downstream CLI still owns actual creation/edit enforcement (`scripts/space-corpus/src/main.rs:528`, `scripts/space-corpus/src/main.rs:210`).
**Why it is this way:** guarded creation avoids direct overwrites, but sequential or parallel per-page calls provide no batch rollback guarantee (`scripts/space-corpus/README.md:15`, `scripts/space-corpus/src/main.rs:522`).

## VG-07.5 Audit and competency checks answer bounded questions

```mermaid
flowchart TB
    AUDIT["Audit: new headers equal generated frontmatter<br/>new bodies empty, reused pages exist"]
    TARGET["Declared relation keys and targets resolve<br/>reused-page edge presence is not compared"]
    CASES["Competency checks over actual header edges<br/>required and forbidden assertions"]
    WALK["Only is-a traverses ancestors<br/>visited set terminates cycles"]
    RESULT["Zero failed assertions passes<br/>empty case lists also pass"]
    AUDIT --> TARGET
    CASES --> WALK --> RESULT
    %% Source: scripts/space-corpus/src/main.rs:570
    %% Source: scripts/space-corpus/src/main.rs:581
    %% Source: scripts/space-corpus/src/main.rs:600
    %% Source: scripts/space-corpus/src/main.rs:268
    %% Source: scripts/space-corpus/src/main.rs:249
    %% Source: scripts/space-corpus/src/main.rs:282
    %% Source: scripts/space-corpus/src/main.rs:303
```

**What it shows:** an audit result counts catalogue-declared targets; competency results instead traverse actual headers, without running the emitted OWL reasoner (`scripts/space-corpus/src/main.rs:600`, `scripts/space-corpus/src/main.rs:273`).
**Why it is this way:** association edges must not gain invented transitivity; full reasoning, attribution round trips and public-projection checks remain separate contract requirements (`scripts/space-corpus/README.md:20`, `planning/space-science-and-geo-sensing/conversion-and-linking.md:62`).

**Drift:** the README describes private staging, but the implementation consumes the catalogue visibility flag; no private-only invariant is enforced by `stage` (`scripts/space-corpus/README.md:14`, `scripts/space-corpus/src/main.rs:477`).
**Debt:** the crate depends on a relative sibling `vault-core` checkout, not a released parser pin; reproducibility requires that dependency's revision as well as this source revision (`scripts/space-corpus/Cargo.toml:18`).
