---
id: SM-04
title: The language-adapter pipeline
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-rust/src/lib.rs
  - ../sealmap/crates/sealmap-rust/src/layout.rs
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-rust/src/raw.rs
  - ../sealmap/crates/sealmap-rust/src/resolve.rs
  - ../sealmap/crates/sealmap-extract/src/lib.rs
  - ../sealmap/crates/sealmap-extract/src/isolate.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
  - ../sealmap/crates/sealmap-rust/README.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

A language adapter turns a `SourceSet` into a `Codebase` in three passes:
**layout** maps each file to a crate and module path by Cargo's conventions,
**collect** parses each file on its own into plain owned data, and
**resolve** builds the model from all of it in one sequential pass
(`crates/sealmap-rust/src/lib.rs:14`-`30`). `sealmap-rust` is the only adapter
today; the language-neutral half it shares with any future adapter lives in
`sealmap-extract`, split out in commit `a02b381` with byte-identical output
before and after (`docs/DESIGN.md:201`).

This topic is the skeleton those passes hang on: the `extract` entry point,
the Cargo layout rules, the panic-isolated big-stack collection that came
from the hardening work (`docs/DESIGN.md:302`), and the order in which
`resolve::build` assembles the model. The passes' insides are their own
topics: ids (SM-05), the flow walker (SM-06), lowering and confidence
(SM-07), resolution (SM-08) and fingerprints (SM-09).

## For the business

The adapter is the part that reads an adopter's code, so it decides what
running sealmap costs and whether it can be trusted on a real repository.
Three properties come from this skeleton. It never compiles anything, so it
runs on any checkout, including one that does not build, and needs no
toolchain at run time. It never fails a whole run because of one bad file:
a file that does not parse, or that trips a bug, becomes a warning and a
placeholder entry. And it is fast enough to sit in CI: the extractor's README says a
934-file workspace extracts and renders in about 1 s
(`crates/sealmap-rust/README.md:20`-`22`).

The residual risk is a file nested deeply enough to exhaust even the large
stack: that still ends the process, because a stack overflow cannot be
caught.

## SM-04.1 One call from sources to model

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant EX as extract<br/>sealmap-rust/src/lib.rs:162
    participant LY as plan<br/>layout.rs:152
    participant CA as collect_all<br/>sealmap-rust/src/lib.rs:182
    participant MI as map_isolated<br/>isolate.rs:38
    participant RB as build<br/>resolve.rs:213
    CL->>EX: SourceSet and RustOptions
    EX->>LY: map every .rs file to a crate and module (sealmap-rust/src/lib.rs:163)
    LY-->>EX: Plan, each path's FileRole and each crate's reach
    EX->>EX: drop test, example and bench targets unless asked (sealmap-rust/src/lib.rs:167)
    EX->>CA: jobs in path order (sealmap-rust/src/lib.rs:171)
    CA->>MI: collect_file per job, failed_file on panic (sealmap-rust/src/lib.rs:186)
    MI-->>CA: RawFile per job, input order
    CA-->>EX: raw files
    EX->>RB: name, raw files, the plan, options (sealmap-rust/src/lib.rs:173)
    RB-->>EX: Codebase and diagnostics
    EX-->>CL: Extraction (sealmap-rust/src/lib.rs:174)
```

**What it shows.** Extraction never fails: unparsable files surface in
`Extraction::diagnostics` (`crates/sealmap-rust/src/lib.rs:159`-`161`), and
everything between collection and resolution is owned data, so the parallel
and the sequential halves meet only in a `Vec<RawFile>`.

**Why it is this way.** Collecting to plain strings and vectors lets files be
parsed in parallel and resolved afterwards in one deterministic pass
(`crates/sealmap-rust/src/raw.rs:3`-`6`, drawn as the IR in EXT-3).

**Invariant:** parallel collection returns results in input order, so the
model is identical with or without the `parallel` feature
(`crates/sealmap-rust/src/lib.rs:78`-`81`).

## SM-04.2 Where a file sits in the workspace

```mermaid
flowchart TB
    F["a .rs file"]
    PK{"inside a directory with a<br/>Cargo.toml holding package?<br/>layout.rs:203"}
    FB["fallback crate named after the codebase<br/>layout.rs:213"]
    LIB{"the lib path from the manifest?<br/>layout.rs:267"}
    SB{"under src/bin?<br/>layout.rs:276"}
    MAIN{"src/main.rs?<br/>layout.rs:279"}
    SRC["other src files: the library crate<br/>layout.rs:285"]
    TEB{"under tests, examples or benches?<br/>layout.rs:291"}
    NONE["build.rs and stray files:<br/>no target, never extracted<br/>layout.rs:296"]
    OWN["own crate, prefixed test_, example_, bench_<br/>layout.rs:304"]
    BIN["binary crate; pkg_main when a lib exists<br/>layout.rs:280"]
    KEY["crate key: the name, or dir/name when<br/>another directory holds that name<br/>layout.rs:242"]
    F --> PK
    PK -->|no| FB
    PK -->|yes, longest dir wins, layout.rs:194| LIB
    LIB -->|yes| SRC
    LIB -->|no| SB
    SB -->|yes| OWN
    SB -->|no| MAIN
    MAIN -->|yes| BIN
    MAIN -->|no| SRC
    SRC --> TEB
    TEB -->|tests etc| OWN
    TEB -->|neither| NONE
    FB --> KEY
    OWN --> KEY
    BIN --> KEY
```

**What it shows.** Packages are discovered from every `Cargo.toml` that has a
`[package]` table; the longest matching directory wins, so nested packages
beat their parents; each binary, test, example and bench target becomes a
crate of its own.

**Why it is this way.** Separate crates per target mean a binary and the
library never share an id (`crates/sealmap-rust/src/lib.rs:36`-`38`); the
`_main` suffix applies only when a library exists to collide with
(`crates/sealmap-rust/src/layout.rs:280`). Crate names replace `-` with `_`,
as Rust paths do (`crates/sealmap-rust/src/layout.rs:186`). The plan also
records, per crate, the workspace crates its manifest lets it name, which
bounds the resolver's name-only guesses (`crates/sealmap-rust/src/layout.rs:247`, SM-08).
A crate name that more than one directory holds (two packages called `core`,
or `tests/it.rs` in two packages) is qualified with its package directory,
`./` at the root, so same-named crates never merge
(`crates/sealmap-rust/src/layout.rs:140`-`146`,
`crates/sealmap-rust/src/lib.rs:38`-`44`).

**Open:** packages come from any manifest with a `[package]` table
(`crates/sealmap-rust/src/layout.rs:171`-`192`); of `[workspace]` only the
`dependencies` table is read, for `path` dependencies
(`crates/sealmap-rust/src/layout.rs:161`-`170`), and membership and `exclude`
are never read, so is a package the workspace excludes meant to
appear in the model?

## SM-04.3 Big stacks and a panic guard

```mermaid
sequenceDiagram
    autonumber
    participant CA as collect_all<br/>sealmap-rust/src/lib.rs:182
    participant MI as map_isolated<br/>isolate.rs:38
    participant RP as rayon pool, 64 MiB stacks<br/>isolate.rs:52
    participant CF as collect_file<br/>collect.rs:33
    participant FF as failed_file<br/>collect.rs:60
    CA->>MI: jobs, stack size, job, on_panic
    MI->>RP: build pool with stack_size (isolate.rs:52)
    alt pool built
        RP->>CF: each job under catch_unwind (isolate.rs:46)
    else pool refused
        MI->>MI: one scoped thread with the same stack (isolate.rs:56)
    end
    CF-->>MI: RawFile, or a panic
    MI->>FF: on panic, message as the error (isolate.rs:47)
    FF-->>MI: module-only RawFile tagged parse_error (collect.rs:74)
    MI-->>CA: results in input order (isolate.rs:53)
    Note over RP: DEBT: a stack overflow is an abort,<br/>not a panic, and ends the run, isolate.rs:8
```

**What it shows.** Every file is collected on a thread with a 64 MiB stack;
a panic in one file becomes a placeholder `RawFile` whose error becomes a
diagnostic, and the run continues.

**Why it is this way.** The VisionClaw stack overflow came from deeply nested
generics on rayon's default 2 MiB stacks (`docs/DESIGN.md:302`); syn costs
about 40 KiB per nesting level in a debug build
(`crates/sealmap-extract/src/isolate.rs:3`-`7`). Address space is reserved,
not memory, so the large stack is cheap (`crates/sealmap-extract/src/isolate.rs:27`-`29`).
The flow walker's own depth cap is the second guard
(`crates/sealmap-rust/src/collect.rs:30`).

**Invariant:** behaviour does not depend on the `parallel` feature: without
it, or if the pool cannot be built, the same stack size is used on one scoped
thread (`crates/sealmap-extract/src/isolate.rs:34`-`37`).

## SM-04.4 What becomes of one file

```mermaid
stateDiagram-v2
    [*] --> Planned: layout assigns a role
    Planned --> Skipped: test, example or bench target and tests off
    Planned --> Parsing: collect_file
    Parsing --> Collected: syn parses
    Parsing --> Failed: syn error with line and column
    Parsing --> Failed: panic caught by the guard
    Collected --> Modelled: resolve builds symbols and relations
    Failed --> Placeholder: module symbol only, tag parse_error
    Placeholder --> Modelled: plus a diagnostic
    Skipped --> [*]
    Modelled --> [*]
```

**What it shows.** A file either leaves the run because its target is
excluded, or ends up in the model; a failed file is represented by its module
alone, tagged `parse_error`, with its error reported.

**Why it is this way.** Keeping a module symbol for a failed file keeps the
1:1 corpus contract (one document per source) true even for broken code
(`crates/sealmap-rust/src/collect.rs:57`-`59`). The placeholder's
fingerprints hash the raw text, so any edit to a broken file still shows
(`crates/sealmap-rust/src/collect.rs:62`).

**Invariant:** a parse error's message carries the `line:col` of the error
(`crates/sealmap-rust/src/collect.rs:47`), and `build` emits one diagnostic
per failed file in file order (`crates/sealmap-rust/src/resolve.rs:218`-`222`).

## SM-04.5 The order resolve builds the model in

```mermaid
flowchart TB
    T["Resolver tables from every raw file<br/>resolve.rs:214"]
    D["diagnostics for failed files<br/>resolve.rs:218"]
    subgraph PERFILE["for each file in path order, resolve.rs:225"]
        FI["SourceFile entry<br/>resolve.rs:226"]
        MO["module symbols and their imports<br/>resolve.rs:233"]
        IT["items, members, field and uses<br/>relations, trait methods<br/>resolve.rs:257"]
        IM["impl blocks: implements relation,<br/>method symbols<br/>resolve.rs:297"]
        FI --> MO --> IT --> IM
    end
    AG["one calls relation per caller and target<br/>aggregate_calls, resolve.rs:318"]
    T --> D --> PERFILE --> AG
```

**What it shows.** Lookup tables are built from all files first, then each
file contributes its file entry, modules, items and impls in that order, and
call edges are aggregated from the finished flows last.

**Why it is this way.** Modules and items go first so field types are known
before any flow is resolved (`crates/sealmap-rust/src/resolve.rs:224`); call
relations come from flows rather than being recorded during the walk, so the
edge confidence is the strongest of the calls behind it (EXT-4).
