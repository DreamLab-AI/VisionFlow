---
id: SM-14
title: Generate --check, write and the command line
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-corpus/src/contract.rs
  - ../sealmap/crates/sealmap-corpus/src/document.rs
  - ../sealmap/crates/sealmap-corpus/src/lib.rs
  - ../sealmap/crates/sealmap-corpus/tests/contract.rs
  - ../sealmap/crates/sealmap/src/main.rs
  - ../sealmap/crates/sealmap/src/lib.rs
  - ../sealmap/.github/workflows/ci.yml
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

`verify` and `write` in `contract.rs` are the 0.1 contract between a generated
corpus and a directory on disk. `verify_against` compares a freshly generated
`Corpus` with what a directory holds and classifies every difference as
missing, orphaned, stale or modified; `write` brings the directory into
compliance and only ever deletes files that carry the generator's header
(`crates/sealmap-corpus/src/contract.rs:1`-`2`,
`crates/sealmap-corpus/src/lib.rs:12`-`29`).

Since step 3 the command line reaches that contract only through
`sealmap generate`: plain `generate` writes, and `generate --check` compares
and writes nothing (`crates/sealmap/src/main.rs:311`-`322`). The name `verify`
now belongs to the seal gate (SM-15, SM-16), and nothing generated is
committed; CI checks instead that two fresh generations agree (SM-19). This
topic draws the generate path, the drift classes and the shared flags.

## For the business

The generated corpus is a view, rebuilt in seconds whenever someone wants it,
and never a file set a pipeline has to keep in step. A developer who keeps a
copy on disk can ask whether it is still current with one flag, and gets an
exact answer: which files are missing, which are left over, which are out of
date because the code changed, and which were edited by hand or by a
different generator version.

For an adopter migrating from 0.1, the one change to plan for is the name:
a pipeline that ran `sealmap verify` against a committed corpus now wants
`sealmap generate --check`, and `sealmap verify` means the seal gate.

## SM-14.1 sealmap generate, end to end

```mermaid
sequenceDiagram
    autonumber
    participant US as developer or CI
    participant GE as generate<br/>main.rs:300
    participant EX as extract<br/>main.rs:381
    participant AD as sealmap-rust extract
    participant PR as sealmap generate
    participant CT as contract verify or write
    US->>GE: generate PATH -o DIR, maybe --check
    GE->>GE: map flags onto the corpus options (main.rs:301)
    GE->>EX: sources, single dir or several repos (main.rs:309)
    EX->>EX: codebase name from --name, the dir or the repo names (main.rs:393)
    EX->>AD: extract once (main.rs:402)
    AD-->>EX: Extraction, diagnostics printed as warnings (main.rs:403)
    EX-->>GE: model and the name used (main.rs:382)
    GE->>PR: corpus in memory (main.rs:310)
    alt --check
        GE->>CT: compare DIR with the fresh corpus (main.rs:312)
        CT-->>GE: Report
        GE-->>US: one padded line per drift, exit 1 if any (main.rs:314, main.rs:321)
    else write
        GE->>CT: bring DIR into compliance (main.rs:323)
        GE-->>US: counts written, removed, unchanged (main.rs:326)
    end
```

**What it shows.** Both branches regenerate the whole corpus in memory first;
`--check` then compares and `write` repairs, and nothing is trusted from disk
except the bytes being compared.

**Why it is this way.** The extraction helper returns the codebase name it
used, so `stale --since` can read a second tree under the same name
(`crates/sealmap/src/main.rs:378`-`381`). Errors in arguments or IO exit 2,
drift under `--check` exits 1 (`crates/sealmap/src/main.rs:256`,
`crates/sealmap/src/main.rs:321`). A drift name is padded through
`to_string`, because `Drift`'s `Display` writes with `write_str` and ignores a
width (`crates/sealmap/src/main.rs:314`,
`crates/sealmap-corpus/src/contract.rs:32`).

## SM-14.2 Classifying a difference

```mermaid
flowchart TB
    E["each expected file<br/>src/contract.rs:81"]
    A{"present in the directory?<br/>src/contract.rs:82"}
    MIS["Missing<br/>src/contract.rs:83"]
    EQ{"byte-identical?<br/>src/contract.rs:84"}
    OK["clean"]
    RS{"reserved underscore file?<br/>src/contract.rs:86"}
    MOD1["Modified<br/>src/contract.rs:87"]
    FH{"same source_hash in<br/>the front matter?<br/>src/contract.rs:90"}
    MOD2["Modified: hand edit or new<br/>generator or options<br/>src/contract.rs:91"]
    STA["Stale: the source changed<br/>src/contract.rs:92"]
    ORP["each md on disk not expected and<br/>carrying the header, or a sealmap<br/>_model.json left out: Orphaned<br/>src/contract.rs:100-102"]
    E --> A
    A -->|no| MIS
    A -->|yes| EQ
    EQ -->|yes| OK
    EQ -->|no| RS
    RS -->|yes| MOD1
    RS -->|no| FH
    FH -->|yes| MOD2
    FH -->|no| STA
    E -.-> ORP
```

**What it shows.** The front-matter source hash is what separates "the code
changed" from "the document was touched"; anything in the directory that is
not a generated document is ignored. The one reserved file a generation can
leave out, `_model.json` under `emit_model: false`, is orphaned when it is a
serialised model, so `--no-model` cannot leave a stale model beside a clean
corpus (`crates/sealmap-corpus/src/contract.rs:112`-`115`,
`crates/sealmap-corpus/tests/contract.rs:333`).

**Why it is this way.** Recording the source hash in each document makes drift
detectable without re-parsing the old state (`crates/sealmap-corpus/src/lib.rs:16`-`17`);
`verify_classifies_every_kind_of_drift` pins all four classes
(`crates/sealmap-corpus/tests/contract.rs:106`).

**Debt:** a reserved file is always `Modified` when it differs
(`crates/sealmap-corpus/src/contract.rs:86`-`87`), even when the difference is
that the code changed, so `_index.json` and `_model.json` never report stale.

## SM-14.3 A corpus file's drift states

```mermaid
stateDiagram-v2
    [*] --> Clean: generate then write
    Clean --> Stale: source edited
    Clean --> Modified: document hand-edited
    Clean --> Modified: generator version or options change
    Clean --> Orphaned: source deleted
    [*] --> Missing: new source
    Stale --> Clean: write rewrites it
    Modified --> Clean: write rewrites it
    Missing --> Clean: write creates it
    Orphaned --> [*]: write deletes it and empty parents
```

**What it shows.** Every drift is repaired by `write` in one step: stale,
modified and missing files are written, orphaned generated documents are
deleted, and unchanged files are not touched.

**Why it is this way.** Untouched files keep their mtimes, so a re-run is cheap
for anything watching the directory (`crates/sealmap-corpus/src/contract.rs:148`-`152`);
`write_converges_and_is_idempotent` pins that a second `write` changes
nothing (`crates/sealmap-corpus/tests/contract.rs:138`).

**Invariant:** `write` deletes only Markdown files whose YAML front matter
opens on the first line (`---`) and whose first key is the `sealmap: `
marker, and a `_model.json` whose first key is `schema_version`; a file that
merely mentions the marker anywhere else is never touched
(`crates/sealmap-corpus/src/contract.rs:100`, `crates/sealmap-corpus/src/document.rs:153`-`155`).

**Invariant:** `write` never follows a symbolic link below the directory. It
checks every path it would write or delete, component by component, and
refuses the whole run before changing anything if one runs through a link
(`crates/sealmap-corpus/src/contract.rs:160`-`162`,
`crates/sealmap-corpus/src/contract.rs:182`-`199`); a linked file and a linked
directory are both pinned by `write_never_writes_through_a_symlink`
(`crates/sealmap-corpus/tests/contract.rs:295`).

## SM-14.4 Reading a directory

```mermaid
flowchart TB
    D["corpus directory"]
    EX{"exists?<br/>src/contract.rs:121"}
    EMP["empty map: every file Missing"]
    RR["recursive read, entries sorted by name<br/>src/contract.rs:129"]
    UT{"readable as UTF-8 and<br/>relative to the root?<br/>src/contract.rs:135"}
    IN["path to text<br/>src/contract.rs:136"]
    SK["skipped"]
    D --> EX
    EX -->|no| EMP
    EX -->|yes| RR --> UT
    UT -->|yes| IN
    UT -->|no| SK
```

**What it shows.** The directory is read into the same path-to-text map the
generator produces, in sorted order, so the comparison is a map diff.

**Why it is this way.** Sorting entries keeps the report order independent of
the file system (`crates/sealmap-corpus/src/contract.rs:129`); the whole corpus
can also be summarised as one hash for cheap cross-machine equality
(`crates/sealmap-corpus/src/contract.rs:213`-`222`).

## SM-14.5 Commands, flags and the facade

```mermaid
flowchart TB
    CMD["generate, model, dense, resolve, seal-check,<br/>verify, stale, seal sign, pack<br/>main.rs:45-71"]
    SRC["shared source flags: --repo, --name, --tests<br/>main.rs:76-86"]
    GEN["generate only: -o, --check, --external,<br/>--owner-lanes, --min-calls, --public-only,<br/>--no-model, --pretty<br/>main.rs:89-119"]
    MOD["model: path, source flags, --external<br/>main.rs:122-131"]
    DNS["dense: path, source flags, -o, --depth,<br/>--external, --stats<br/>main.rs:134-152"]
    SITE["seal commands and pack: -C ROOT,<br/>--diagrams DIR, source flags<br/>main.rs:156-165"]
    PCK["pack only: topics, --diff, --budget,<br/>--depth, --source-window, -o, --shard<br/>main.rs:206-232, SM-17"]
    OP["Options: rust and corpus halves<br/>sealmap/src/lib.rs:98"]
    ONE["one directory: load_dir<br/>main.rs:394"]
    MANY["several repos, each prefixed NAME/<br/>load_repos, sealmap/src/lib.rs:129"]
    CMD --> SRC
    CMD --> GEN
    CMD --> MOD
    CMD --> DNS
    CMD --> SITE
    SITE --> PCK
    SITE --> SRC
    GEN --> OP
    SRC --> ONE
    SRC --> MANY
```

**What it shows.** Every command shares the flags that shape ids; only
`generate` carries the projection flags, and `model` takes the external-call
policy because it changes the flows it prints. The seal commands and `pack`
share one `Site` group (root, topic directory, source flags); the report
flags `--json` and `--all` sit beside it only where a command reports. Several repositories are
loaded into one `SourceSet` under name prefixes, so calls between them
resolve and the overview groups crates by repository
(`crates/sealmap/src/lib.rs:112`-`113`).

**Why it is this way.** The source flags decide which symbols exist and what
they are called, so a seal must be checked with the flags it was signed with
(`crates/sealmap/src/main.rs:73`-`74`); the facade exists so the common path
is one dependency (`crates/sealmap/src/lib.rs:35`-`36`), and the CLI is behind
the default `cli` feature, so a library user does not pull in clap.
