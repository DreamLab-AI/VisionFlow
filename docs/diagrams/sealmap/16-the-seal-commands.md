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
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

The seal commands are thin: each one loads the same context (a model of the
source root, the lock text and the topic files), calls one function of
`sealmap_corpus::seal` (SM-15), and prints text or JSON. The only processes
the CLI spawns are for `stale --since` and `pack --diff`, which export a git
revision into a temporary directory so the library can compare two models,
and `git rev-parse` and `git status` for a pack's revision line; git stays out
of the libraries (`docs/DESIGN.md:236`-`239`). `pack` itself is SM-17.

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
    participant CX as Ctx load<br/>main.rs:527
    participant EX as extract<br/>main.rs:468
    participant FS as the file system
    participant RT as read_topics<br/>main.rs:563
    CMD->>CX: -C ROOT, --diagrams DIR, source flags
    CX->>EX: model of ROOT, the name used and the source texts (main.rs:528)
    CX->>CX: DIR under the canonical ROOT unless absolute, path checked (main.rs:529-531)
    CX->>FS: DIR/seals.lock, a missing file is no lock (main.rs:532-537)
    CX->>CX: a lock that is present must parse or the load fails (main.rs:538-540)
    CX->>RT: every md under DIR with a front-matter id (main.rs:541)
    RT->>FS: sorted entries, hidden names skipped (main.rs:573-578)
    RT-->>CX: topics keyed by path under DIR (main.rs:585-587)
    CX-->>CMD: model, sources, name, lock text, topics (main.rs:542)
```

**What it shows.** One loader feeds every seal command, so they all see the
same model, the same lock bytes and the same topic set.

**Why it is this way.** The lock is kept as text because
`verify` must judge its bytes for canonicality while `seal sign` only needs to
read it, and load parses it once only to reject a malformed lock
(`crates/sealmap/src/main.rs:538`-`540`) (`crates/sealmap/src/main.rs:522`,
`crates/sealmap/src/main.rs:750`-`751`). A topic is any Markdown file whose
front matter has an `id:`, so the corpus's README and indexes are ignored
(`crates/sealmap/src/main.rs:585`).

**Debt:** a seal does not record the extraction options that shape ids, so a
lock signed with `--tests` and checked without it reports its test symbols as
absent; only the flag's doc comment says so
(`crates/sealmap/src/main.rs:78`-`79`).

## SM-16.2 stale --since, two trees

```mermaid
sequenceDiagram
    autonumber
    participant US as developer or skill
    participant ST as stale<br/>main.rs:665
    participant GT as GitTree export<br/>main.rs:699
    participant GI as git and tar
    participant EX as extract<br/>main.rs:468
    participant LB as seal stale<br/>check.rs:481
    US->>ST: stale --since REV
    ST->>ST: Ctx model_at, shared with pack --diff (main.rs:670)
    ST->>ST: refuse with --repo (main.rs:552-553)
    ST->>GT: ROOT and REV (main.rs:555)
    GT->>GI: rev-parse --show-prefix (main.rs:703)
    GT->>GI: archive REV:prefix piped into tar -x, temp dir (main.rs:712-720)
    GI-->>GT: the revision's tree, or an error (main.rs:724-728)
    ST->>EX: the old tree under the current tree's name (main.rs:556-557)
    ST->>LB: lock, old model, current model (main.rs:672)
    LB-->>ST: changed sealed symbols
    ST-->>US: topic, class, symbol, candidates, always exit 0 (main.rs:677, main.rs:690)
    Note over GT: the temp dir is a tempfile TempDir, removed when it drops, main.rs:696
```

**What it shows.** A revision becomes a second model by exporting only the
subdirectory the root points at, so paths line up, and reading it under the
current codebase name, so ids line up.

**Why it is this way.** Each sealed symbol's baseline is its hashes in the old
model when it is there, else its sealed hashes, so `--since` reports what
changed since that revision whatever the lock says
(`crates/sealmap-corpus/src/seal/check.rs:486`-`489`). The CLI test commits a
behaviour change, edits a signature after it, and sees only the contract
change with `--since HEAD` (`crates/sealmap/tests/cli.rs:135`, `crates/sealmap/tests/cli.rs:160`).

**Debt:** `--since` needs `tar` on the path as well as `git`
(`crates/sealmap/src/main.rs:720`), and a process killed mid-run leaves its
`sealmap-since-*` directory in the system temp directory, since only a normal
drop removes it (`crates/sealmap/src/main.rs:696`, `crates/sealmap/src/main.rs:704`-`711`).

## SM-16.3 seal sign, from the command line

```mermaid
flowchart TB
    A["seal sign TOPIC --reviewer --model, maybe --date<br/>main.rs:748"]
    L["lock parsed leniently, a non-canonical<br/>lock is rewritten canonically<br/>main.rs:750-751"]
    T{"TOPIC an id? else a path under the<br/>cwd or the diagrams dir<br/>main.rs:1025-1039"}
    D["date: --date, else today in UTC<br/>main.rs:757"]
    S["seal sign in the library<br/>main.rs:759"]
    R["refused: message, exit 1, nothing written<br/>main.rs:761-763"]
    W1["both writes staged first, topic only if the pointer changed it<br/>main.rs:768-771"]
    W2["lock replaced canonically, then the topic<br/>main.rs:774-777"]
    A --> L --> T --> D --> S
    S -->|error| R
    S -->|ok| W1 --> W2
```

**What it shows.** Signing reads, derives and refuses before it writes; a
successful sign stages both files, then replaces the lock first and the topic last.

**Why it is this way.** The library does the derivation and the CLI only the
IO, so a skill can call either (`docs/DESIGN.md:150`-`155`). The default date is
computed from the system clock without a date library
(`crates/sealmap/src/main.rs:1043`-`1050`).

**Invariant:** a refused path or staging error leaves the topic and the lock
untouched (`crates/sealmap/src/main.rs:768`-`769`). A stop between the two
replaces leaves a lock entry whose topic hash no longer matches the old topic
text, which `verify` reports as edited prose or a pointer fault, so an interrupted
sign can never pass the gate (`crates/sealmap/src/main.rs:774`-`777`,
`crates/sealmap-corpus/src/seal/check.rs:36`, `crates/sealmap-corpus/src/seal/check.rs:268`-`275`).

## SM-16.4 Exit codes

```mermaid
flowchart TB
    E2["2: usage or IO error, any command<br/>main.rs:263"]
    V["verify, seal-check: 0 when every finding<br/>holds, else 1<br/>main.rs:660"]
    R["resolve: 0 when found, 1 when absent<br/>or unparsable<br/>main.rs:629"]
    S["stale: always 0<br/>main.rs:690"]
    G["seal sign: 0 sealed, 1 refused<br/>main.rs:763"]
    C["generate --check: 0 clean, 1 drift<br/>main.rs:330"]
```

**What it shows.** Exit 1 always means "the check ran and said no", and 2
always means the command could not run.

**Why it is this way.** `stale` is a report, not a gate, so a skill can call
it on every commit without failing the build; the design gives it exit 0
(`docs/DESIGN.md:227`). `resolve` fails when the id is not found so a script
that checks diagram edges can branch on it (`README.md:419`-`423`); the codes
are pinned end to end (`crates/sealmap/tests/cli.rs:74`,
`crates/sealmap/tests/cli.rs:115`).
