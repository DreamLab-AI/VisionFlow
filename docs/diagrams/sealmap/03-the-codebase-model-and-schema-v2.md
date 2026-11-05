---
id: SM-03
title: The codebase model, its sources and schema v2
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-model/src/lib.rs
  - ../sealmap/crates/sealmap-model/src/codebase.rs
  - ../sealmap/crates/sealmap-model/src/symbol.rs
  - ../sealmap/crates/sealmap-model/src/flow.rs
  - ../sealmap/crates/sealmap-model/src/path.rs
  - ../sealmap/crates/sealmap-model/src/source.rs
  - ../sealmap/crates/sealmap-model/Cargo.toml
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

`sealmap-model` is the vocabulary both halves of the workspace meet in:
language adapters produce a `Codebase`, projections consume one, and neither
side knows the other (`crates/sealmap-model/src/lib.rs:6`-`10`). This topic
draws that value: files, symbols, relations and the ordered call `Flow` of
every callable; the `SourceSet` adapters read from instead of the disk; the
path normalisation that makes keys identical on every platform; and the
schema-v2 JSON reader that refuses anything else.

The crate's determinism contract has four clauses: ordered collections only,
normalised paths, stable hashing, no ambient data
(`crates/sealmap-model/src/lib.rs:18`-`34`). Every diagram here is one of
those clauses made concrete. Schema v2 came with commit `01f82b4`; v1 has no
reader because nothing outside the workspace consumed it
(`crates/sealmap-model/src/lib.rs:106`-`111`).

Not covered: the id grammar (SM-01), the fingerprints (SM-02), and how an
adapter fills the model (SM-04 onwards).

## For the business

Everything sealmap claims about a codebase is a claim about this one value.
An adopter can persist it (`_model.json`), diff it, or feed it to a tool of
their own, and the same checkout produces the same bytes on any machine.
That is what makes the output safe to cache and cheap to compare in CI: no
timestamps, no machine paths, no user names.

Two properties lower risk directly. The loader honours the repository's own
ignore files, so vendored and generated trees do not inflate the model (the
design measured 47,229 files walked instead of 934 before this,
`docs/DESIGN.md:233`). And a model written by another version is refused
outright rather than half-read, so a stale artefact cannot be mistaken for a
current one.

## SM-03.1 The model as types

```mermaid
classDiagram
    direction LR
    class Codebase {
        +schema_version u32  codebase.rs:19
        +files BTreeMap path to SourceFile  codebase.rs:23
        +symbols BTreeMap SymbolId to Symbol  codebase.rs:25
        +relations BTreeSet Relation  codebase.rs:27
    }
    class SourceFile {
        +path language module hash lines  source.rs:16
    }
    class Symbol {
        +id name kind visibility file span  symbol.rs:192
        +sig_hash body_hash  symbol.rs:209
        +parent signature doc generics tags  symbol.rs:214
        +members  symbol.rs:226
        +flow Option Flow  symbol.rs:228
    }
    class Relation {
        +from to kind confidence  symbol.rs:325
    }
    class Flow {
        +steps  flow.rs:40
    }
    class Confidence {
        <<enum>>
        Exact Inferred External  symbol.rs:299
    }
    Codebase *-- SourceFile
    Codebase *-- Symbol
    Codebase *-- Relation
    Symbol o-- Flow
    Relation --> Confidence
```

**What it shows.** Three ordered maps or sets and nothing else. A symbol may
carry a flow; a relation always carries a confidence.

**Why it is this way.** Ordered collections make serialisation deterministic
and every query iterate in key order (`crates/sealmap-model/src/codebase.rs:13`-`15`).
Confidence is on every relation because there is no type checker behind the
model, and nothing is meant to be guessed silently (`README.md:321`-`323`).

**Invariant:** a relation from a symbol to itself is never stored
(`crates/sealmap-model/src/codebase.rs:119`).

**Debt:** `relations_from` is a range scan because `Relation` orders by
`from` first (`crates/sealmap-model/src/codebase.rs:158`-`161`), but
`relations_to`, `children` and `symbols_in_file` scan everything
(`crates/sealmap-model/src/codebase.rs:164`,
`crates/sealmap-model/src/codebase.rs:150`,
`crates/sealmap-model/src/codebase.rs:140`); projections that call them per
document have had to build their own indexes (SM-12).

## SM-03.2 Loading a directory

```mermaid
flowchart TB
    ROOT["root directory"]
    W["ignore walker<br/>WalkBuilder, source.rs:154"]
    CFG["hidden skipped, parents off, no global<br/>or git exclude, no symlinks<br/>source.rs:155-162"]
    SKIP["skip target, node_modules, vendor,<br/>dist, build, out below the root<br/>source.rs:163"]
    EXT{"extension wanted<br/>and under 2 MiB?<br/>source.rs:175-177"}
    UTF{"valid UTF-8?<br/>source.rs:181"}
    REL["path made relative and normalised<br/>source.rs:182"]
    SET["SourceSet, newlines normalised<br/>source.rs:183"]
    DROP["skipped silently"]
    ROOT --> W --> CFG --> SKIP --> EXT
    EXT -->|no| DROP
    EXT -->|yes| UTF
    UTF -->|no| DROP
    UTF -->|yes| REL --> SET
```

**What it shows.** `SourceSet::load_dir` walks with the `ignore` crate,
honouring `.gitignore` and `.ignore` files under the root only, then keeps
files by extension and size and stores their text with `\n` line endings.

**Why it is this way.** Reading only what is under the root, and never the
user's global excludes, keeps the result a function of the checkout's bytes
(`crates/sealmap-model/src/source.rs:53`-`57`). The ignore support was the
hardening fix for the 48-second VisionClaw walk (`docs/DESIGN.md:233`). It
also makes `ignore` a dependency of the model crate
(`crates/sealmap-model/Cargo.toml:21`), which SM-19 follows to the MSRV.

**Debt:** a file that is not valid UTF-8 is skipped with no diagnostic
(`crates/sealmap-model/src/source.rs:181`), so it disappears from the model
and from every count, while one unreadable directory entry aborts the whole
load (`crates/sealmap-model/src/source.rs:170`).

## SM-03.3 Normalising a path

```mermaid
flowchart TB
    RAW["raw path text"]
    BS["backslashes become slashes<br/>path.rs:48"]
    ABS{"leading slash or drive letter?<br/>path.rs:49"}
    E1["PathError.Absolute"]
    LOOP["split on slash<br/>path.rs:53"]
    DOT["empty and dot parts dropped<br/>path.rs:55"]
    UP{"dot-dot with nothing to pop?<br/>path.rs:57"}
    E2["PathError.Escapes"]
    EMP{"nothing left?<br/>path.rs:64"}
    E3["PathError.Empty"]
    OK["parts joined with slash<br/>path.rs:67"]
    RAW --> BS --> ABS
    ABS -->|yes| E1
    ABS -->|no| LOOP --> DOT --> UP
    UP -->|yes| E2
    UP -->|no| EMP
    EMP -->|yes| E3
    EMP -->|no| OK
```

**What it shows.** Every path becomes relative, slash-separated and free of
`.` and `..` before it is used as a key, or it is refused.

**Why it is this way.** `src\lib.rs` on Windows and `src/lib.rs` on Linux
must be the same key for the corpus to be byte-identical across machines
(`crates/sealmap-model/src/lib.rs:27`-`29`). The same type names the 1:1
corpus documents, by appending a suffix (`crates/sealmap-model/src/path.rs:109`).

**Invariant:** no `SourcePath` can escape the codebase root, because a `..`
with nothing to pop is an error rather than a clamp
(`crates/sealmap-model/src/path.rs:57`-`58`).

## SM-03.4 Reading a persisted model

```mermaid
sequenceDiagram
    autonumber
    participant CL as tool or agent
    participant FJ as from_json<br/>codebase.rs:48
    participant SJ as serde_json
    CL->>FJ: text of a _model.json
    FJ->>SJ: probe only schema_version and schema (codebase.rs:50)
    SJ-->>FJ: both optional numbers
    FJ->>FJ: take schema_version, else v1 schema (:55)
    alt not version 2
        FJ-->>CL: ModelJsonError.Version with found and expected (codebase.rs:57)
    else version 2
        FJ->>SJ: full deserialise (codebase.rs:59)
        SJ-->>FJ: Codebase, ids parsed as canonical sym text
        FJ-->>CL: Codebase
    end
```

**What it shows.** The reader looks at the version first, through a two-field
probe, and only then deserialises; a v1 file is reported as "found 1" rather
than failing on a missing field.

**Why it is this way.** Schema v2 changed the meaning of every id, so a v1
model read as v2 would be wrong in every row. The README states the refusal
for users (`README.md:205`-`208`). Every `SymbolId` inside is parsed through
the canonical-only parser, so a hand-edited id fails the whole read.

**Invariant:** `Codebase::new` always stamps the current schema version
(`crates/sealmap-model/src/codebase.rs:33`), so nothing written by this build
can fail its own reader on version.

## SM-03.5 A flow is the skeleton of a sequence diagram

```mermaid
flowchart TB
    F["Flow, steps in source order<br/>flow.rs:38"]
    C["Call: target, label, kind,<br/>confidence, awaited, fallible, line<br/>flow.rs:165"]
    B["Branch: arms<br/>flow.rs:108"]
    L["Loop: label, body<br/>flow.rs:116"]
    O["Optional: label, body<br/>flow.rs:124"]
    P["Parallel: arms<br/>flow.rs:132"]
    R["Return: exit label and line<br/>flow.rs:138"]
    M1["message arrow"]
    M2["alt and else"]
    M3["loop"]
    M4["opt"]
    M5["par and and"]
    M6["note"]
    F --> C --> M1
    F --> B --> M2
    F --> L --> M3
    F --> O --> M4
    F --> P --> M5
    F --> R --> M6
```

**What it shows.** Calls are leaves; branches, loops, optional blocks and
parallel arms are interior nodes that map one to one onto Mermaid fragments.

**Why it is this way.** A flow holds only what contains calls, so it is
already the minimal "who talks to whom, in what order, under which
condition" (`crates/sealmap-model/src/flow.rs:4`-`8`). Every call target is a
canonical id, which is what lets an orchestrator inline one function's
sequence into another's without guessing (`crates/sealmap-model/src/flow.rs:10`-`13`).

**Debt:** `Flow::calls` collects every call into a fresh vector before
iterating, and `call_count` builds that vector just to count it
(`crates/sealmap-model/src/flow.rs:55`-`64`); `Codebase::stats` does this
once per symbol (`crates/sealmap-model/src/codebase.rs:197`).
