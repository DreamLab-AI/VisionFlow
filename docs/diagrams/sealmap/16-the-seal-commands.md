---
id: SM-16
title: The seal commands, resolve, verify, stale and sign
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap/src/main.rs
  - ../sealmap/crates/sealmap/tests/cli.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/check.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

The seal commands are thin: each one loads the same context (a model of the
source root, the lock text and the topic files), calls one function of
`sealmap_corpus::seal` (SM-15), and prints text or JSON. The only processes
the CLI spawns are for `stale --since` and `pack --diff`, which export a git
revision into a temporary directory so the library can compare two models,
and `git rev-parse` and `git status` for a pack's revision line; git stays out
of the libraries (`docs/DESIGN.md:166`-`169`). `pack` itself is SM-17.

After this topic you should know what each command reads and writes, how a
revision becomes a second model, what the exit codes mean, and which options
a seal silently depends on.

## For the business

Five commands cover the whole seal workflow for a pipeline or a skill: ask
where a function is now, seal a reviewed topic, gate CI on every seal,
classify a chosen few, and list what changed since a release or a branch
point. The last one always succeeds, so it can feed a report without failing
a build; the gate fails on anything short of "still true".

One caution for adopters: a seal remembers functions by id, and the ids
depend on how the code was read (whether tests were included, what the
codebase is called when no manifest names it). Checking a seal with different
settings from the ones it was signed with makes sealed functions look absent.

## SM-16.1 What every seal command loads

```mermaid
sequenceDiagram
    autonumber
    participant CMD as resolve, seal-check, verify,<br/>stale or seal sign
    participant CX as Ctx load<br/>main.rs:438
    participant EX as extract<br/>main.rs:381
    participant FS as the file system
    participant RT as read_topics<br/>main.rs:472
    CMD->>CX: -C ROOT, --diagrams DIR, source flags
    CX->>EX: model of ROOT, the name used and the source texts (main.rs:439)
    CX->>CX: DIR relative to ROOT unless absolute (main.rs:440)
    CX->>FS: DIR/seals.lock, a missing file is no lock (main.rs:441-445)
    CX->>RT: every md under DIR with a front-matter id (main.rs:447)
    RT->>FS: sorted entries, hidden names skipped (main.rs:482-487)
    RT-->>CX: topics keyed by path under DIR (main.rs:494-496)
    CX-->>CMD: model, sources, name, lock text, topics (main.rs:448)
```

**What it shows.** One loader feeds every seal command, so they all see the
same model, the same lock bytes and the same topic set.

**Why it is this way.** The lock is kept as text, not parsed here, because
`verify` must judge its bytes for canonicality while `seal sign` only needs to
read it (`crates/sealmap/src/main.rs:433`,
`crates/sealmap/src/main.rs:662`-`663`). A topic is any Markdown file whose
front matter has an `id:`, so the corpus's README and indexes are ignored
(`crates/sealmap/src/main.rs:494`).

**Debt:** a seal does not record the extraction options that shape ids, so a
lock signed with `--tests` and checked without it reports its test symbols as
absent; only the flag's doc comment says so
(`crates/sealmap/src/main.rs:73`-`74`).

## SM-16.2 stale --since, two trees

```mermaid
sequenceDiagram
    autonumber
    participant US as developer or skill
    participant ST as stale<br/>main.rs:574
    participant GT as GitTree export<br/>main.rs:608
    participant GI as git and tar
    participant EX as extract<br/>main.rs:381
    participant LB as seal stale<br/>check.rs:481
    US->>ST: stale --since REV
    ST->>ST: Ctx model_at, shared with pack --diff (main.rs:579)
    ST->>ST: refuse with --repo (main.rs:461-462)
    ST->>GT: ROOT and REV (main.rs:464)
    GT->>GI: rev-parse --show-prefix (main.rs:609)
    GT->>GI: archive REV:prefix piped into tar -x, temp dir (main.rs:614-622)
    GI-->>GT: the revision's tree, or an error (main.rs:630-634)
    ST->>EX: the old tree under the current tree's name (main.rs:465-466)
    ST->>LB: lock, old model, current model (main.rs:581)
    LB-->>ST: changed sealed symbols
    ST-->>US: topic, class, symbol, candidates, always exit 0 (main.rs:586, main.rs:599)
    Note over GT: the temp dir is removed on drop, main.rs:644-646
```

**What it shows.** A revision becomes a second model by exporting only the
subdirectory the root points at, so paths line up, and reading it under the
current codebase name, so ids line up.

**Why it is this way.** Each sealed symbol's baseline is its hashes in the old
model when it is there, else its sealed hashes, so `--since` reports what
changed since that revision whatever the lock says
(`crates/sealmap-corpus/src/seal/check.rs:486`-`489`). The CLI test commits a
behaviour change, edits a signature after it, and sees only the contract
change with `--since HEAD` (`crates/sealmap/tests/cli.rs:125`, `crates/sealmap/tests/cli.rs:150`).

**Debt:** `--since` needs `tar` on the path as well as `git`
(`crates/sealmap/src/main.rs:622`), and a process killed mid-run leaves its
`sealmap-since-*` directory in the system temp directory, since only a normal
drop removes it (`crates/sealmap/src/main.rs:644`-`646`).

## SM-16.3 seal sign, from the command line

```mermaid
flowchart TB
    A["seal sign TOPIC --reviewer --model, maybe --date<br/>main.rs:660"]
    L["lock parsed leniently, a non-canonical<br/>lock is rewritten canonically<br/>main.rs:662-663"]
    T{"TOPIC an id? else a path under the<br/>cwd or the diagrams dir<br/>main.rs:775-789"}
    D["date: --date, else today in UTC<br/>main.rs:669"]
    S["seal sign in the library<br/>main.rs:671"]
    R["refused: message, exit 1, nothing written<br/>main.rs:673-675"]
    W1["topic written only if the pointer changed it<br/>main.rs:679-680"]
    W2["lock written canonically<br/>main.rs:682"]
    A --> L --> T --> D --> S
    S -->|error| R
    S -->|ok| W1 --> W2
```

**What it shows.** Signing reads, derives and refuses before it writes; a
successful sign writes at most two files, the topic first and the lock last.

**Why it is this way.** The library does the derivation and the CLI only the
IO, so a skill can call either (`docs/DESIGN.md:80`-`85`). The default date is
computed from the system clock without a date library
(`crates/sealmap/src/main.rs:793`-`800`).

**Invariant:** a failure between the two writes leaves a pointer with no
entry, which `verify` reports as a lock fault, so an interrupted sign can
never pass the gate (`crates/sealmap/src/main.rs:680`-`682`,
`crates/sealmap-corpus/src/seal/check.rs:383`).

## SM-16.4 Exit codes

```mermaid
flowchart TB
    E2["2: usage or IO error, any command<br/>main.rs:256"]
    V["verify, seal-check: 0 when every finding<br/>holds, else 1<br/>main.rs:569"]
    R["resolve: 0 when found, 1 when absent<br/>or unparsable<br/>main.rs:538"]
    S["stale: always 0<br/>main.rs:599"]
    G["seal sign: 0 sealed, 1 refused<br/>main.rs:675"]
    C["generate --check: 0 clean, 1 drift<br/>main.rs:321"]
```

**What it shows.** Exit 1 always means "the check ran and said no", and 2
always means the command could not run.

**Why it is this way.** `stale` is a report, not a gate, so a skill can call
it on every commit without failing the build; the design gives it exit 0
(`docs/DESIGN.md:157`). `resolve` fails when the id is not found so a script
that checks diagram edges can branch on it (`README.md:294`-`295`); the codes
are pinned end to end (`crates/sealmap/tests/cli.rs:64`,
`crates/sealmap/tests/cli.rs:105`).
