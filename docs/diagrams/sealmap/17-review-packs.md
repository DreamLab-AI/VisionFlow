---
id: SM-17
title: Review packs
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-corpus/src/pack.rs
  - ../sealmap/crates/sealmap-corpus/tests/pack.rs
  - ../sealmap/crates/sealmap/src/main.rs
  - ../sealmap/crates/sealmap/tests/cli.rs
  - ../sealmap/crates/sealmap-dense/src/lib.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: eacc59ece21274273c6bf2c83d4642b4fa7acaf5
---
## For developers

Historical snapshot: this topic describes the earlier pack implementation.
For the 0.4.0 selection and provenance changes, see
[SM-18](18-review-evidence-and-selection.md). Its original revision and source
coordinates below are retained rather than presented as a new verification.

`sealmap pack` turns chosen topics into one text a reviewer can read without
the repository (`docs/DESIGN.md:84`). For each topic the pack holds the file
verbatim, the `sealmap-dense` slice around the symbols it cites, the
citations that do not resolve, and a window of each cited symbol's current
source (`crates/sealmap-corpus/src/pack.rs:477`-`495`). The library half,
`sealmap_corpus::pack`, is pure: it takes the model, the source texts, the
topic texts and an opaque revision string, and never reads the disk or runs
git (`crates/sealmap-corpus/src/pack.rs:6`-`9`). The CLI half reads the
checkout, picks the topics and writes the result
(`crates/sealmap/src/main.rs:696`).

Three rules carry the design:

- **Identical inputs give identical bytes.** Topics are a set and are
  written in id order (`crates/sealmap-corpus/src/pack.rs:417`-`433`), and
  citations are deduplicated and sorted
  (`crates/sealmap-corpus/src/pack.rs:451`). A test feeds the ids in another
  order, with a duplicate, and from a freshly extracted model, and gets the
  same pack (`crates/sealmap-corpus/tests/pack.rs:163`). CI checks the
  same on this repository (SM-15 and SM-21 in both orders, SM-19).
- **Nothing is truncated.** Over budget, `pack` refuses with the header's
  size and each topic's (`crates/sealmap-corpus/src/pack.rs:278`-`282`).
  `shard` instead fills numbered packs with whole topics and refuses only a
  topic too large to fit alone (`crates/sealmap-corpus/src/pack.rs:295`-`331`).
- **Every block states its length.** A block line ends its fixed fields with
  the payload's byte count, so a reader can skip a block exactly even when a
  topic contains a line that looks like one
  (`crates/sealmap-corpus/src/pack.rs:561`-`571`). A test walks the golden
  pack by those lengths alone (`crates/sealmap-corpus/tests/pack.rs:130`).

`--diff REV` adds every topic whose sealed or cited symbols changed since a
revision. It reuses `stale` for sealed topics and compares each cited
symbol's hashes in the two models, ignoring moves
(`crates/sealmap-corpus/src/pack.rs:357`-`370`).

## For the business

A reviewer outside the team, a seal review or a debugging session needs the
claims and the code they rest on, and nothing else. Handing over the
repository costs far more context than the question needs, and an excerpt
cut to fit can hide the line that matters. A pack is the bounded, repeatable
alternative. The same request always produces the same file, so two
reviewers, or two runs of one reviewer, see the same thing. If the file
would be too big, the tool says by how much and which topics are largest,
or splits the request into whole topics. It never quietly drops content.

`sealmap pack --diff main` selects the topics whose cited symbols changed,
and leaves out topics that merely share a file with the change. That is the
per-symbol rule the evidence found loses real changes, so `--diff` is a
convenience, not an answer to "what needs another look after this branch?"
(`README.md:41`-`46`, `README.md:291`). Naming the topics, for example every
topic a per-file flag selects, is the dependable way to build a pack.

## SM-17.1 From a request to a pack

```mermaid
flowchart TB
    REQ["sealmap pack ids, --diff REV,<br/>--budget, --depth, --source-window<br/>main.rs:696"]
    SEL["topic ids or files, plus changed topics<br/>main.rs:700, main.rs:708"]
    NONE{"no topic named and none changed?<br/>main.rs:712"}
    REV["revision: HEAD, +dirty for modified<br/>tracked files, else unknown<br/>main.rs:767-770"]
    BLD["Builder.new: unknown or duplicate ids refused<br/>src/pack.rs:407"]
    SEC["one section per topic, in id order<br/>src/pack.rs:437-443"]
    MODE{"--shard?<br/>main.rs:725"}
    PK{"pack: longer than the budget?<br/>src/pack.rs:279"}
    SH["shard: whole topics per file<br/>src/pack.rs:295"]
    OUT["stdout, -o FILE, or pack-01.txt ...<br/>main.rs:746-757"]
    REF["exit 1: total, header and per-topic sizes<br/>main.rs:736-743"]
    REQ --> SEL --> NONE
    NONE -->|yes, exit 0, nothing written| OUT
    NONE -->|no| REV --> BLD --> SEC --> MODE
    MODE -->|no| PK
    MODE -->|yes| SH
    PK -->|no| OUT
    PK -->|yes| REF
    SH -->|every topic fits alone| OUT
    SH -->|a topic too large alone| REF
```

**What it shows.** Selection and IO live in the CLI. Validation, rendering
and the budget decision live in the library, so another caller, such as the
`sealmap-review` skill, gets the same refusals from either.

**Why it is this way.** Git stays out of the libraries
(`docs/DESIGN.md:208`-`211`). The revision is therefore an opaque string the
CLI supplies. Untracked files do not make it dirty, so writing a pack into
the checkout does not change the next pack's header
(`crates/sealmap/src/main.rs:763`-`769`).

**Tension (revision vs reproducibility):** `+dirty` says that tracked files
differ from `HEAD`, not how (`crates/sealmap/src/main.rs:769`-`770`). Two
packs of different uncommitted edits carry the same header. Their bodies
differ, but the header alone cannot tell them apart.

**Debt:** when `--diff` finds nothing to pack, the command exits 0 and writes
nothing (`crates/sealmap/src/main.rs:712`-`714`). With `-o FILE` a pack from
an earlier run is left in place, where a caller may read it as current.

## SM-17.2 One topic's section

```mermaid
sequenceDiagram
    autonumber
    participant BS as Builder.sections<br/>src/pack.rs:437
    participant SE as Builder.section<br/>src/pack.rs:445
    participant RS as seal resolve
    participant DN as Dense.slice<br/>sealmap-dense/src/lib.rs:347
    participant WI as Builder.window<br/>src/pack.rs:501
    BS->>BS: one Dense for every topic (src/pack.rs:438)
    BS->>SE: topic id, file, text
    SE->>SE: citations, deduplicated, in id order (src/pack.rs:451-455)
    loop each citation
        SE->>RS: where is it now
        RS-->>SE: found, unparsable or absent, or an invalid id (src/pack.rs:456-471)
    end
    SE->>SE: topic block, the file verbatim (src/pack.rs:477)
    SE->>DN: found ids as seeds, depth, no byte limit (src/pack.rs:479-481)
    DN-->>SE: skeletons, calls, callers, index
    SE->>SE: unresolved block, when any (src/pack.rs:484-486)
    loop each found id, in id order
        SE->>WI: file, span start and end (src/pack.rs:490)
        WI-->>SE: at most N lines from the first, label says when clipped (src/pack.rs:508-517)
    end
    SE->>SE: end line (src/pack.rs:495)
```

**What it shows.** A section is built from the model alone. Every citation
lands in exactly one place: the slice and a source window when it resolves,
the unresolved list when it does not.

**Why it is this way.** A pack is evidence for a review, so a citation that
no longer resolves must stay visible rather than vanish
(`docs/DESIGN.md:229`-`230`). The slice carries its own index, so the pack
needs no `_index.txt` beside it (`crates/sealmap-dense/src/lib.rs:75`-`79`).

**Invariant:** the dense slice is taken without a byte limit, so it cannot
fail on size. The only budget is the pack's
(`crates/sealmap-corpus/src/pack.rs:479`-`481`).

**Open:** a topic that cites `path:line` rather than `sym:` ids packs as its
text alone (`docs/DESIGN.md:232`-`233`). Every topic of this repository's
own corpus is such a topic, and the step that would have migrated it to
`sym:` citations is closed with the seal work (`docs/DESIGN.md:26`-`30`), so a
pack of it carries no slices and no source windows. Should `pack` resolve
`path:line` citations itself, so that corpora which never migrate get them?

## SM-17.3 The budget: refuse, or shard by topic

```mermaid
flowchart TB
    T["sections in id order"]
    P{"pack: header plus every section<br/>over the budget?<br/>src/pack.rs:279"}
    OK["the pack, unchanged"]
    OB["OverBudget: bytes, budget,<br/>header size, each topic's size<br/>src/pack.rs:280"]
    S["shard: try the topic in the current file<br/>src/pack.rs:308-309"]
    F{"fits, with this file's header?<br/>src/pack.rs:309"}
    C["close the file, start the next<br/>src/pack.rs:314"]
    A{"fits a file alone?<br/>src/pack.rs:316-317"}
    TO["TopicOverBudget names the topic<br/>src/pack.rs:318"]
    T --> P
    P -->|no| OK
    P -->|yes| OB
    T --> S --> F
    F -->|yes| S
    F -->|no| C --> A
    A -->|yes| S
    A -->|no| TO
```

**What it shows.** `pack` either fits whole or refuses with what a caller
needs to plan shards. `shard` does that planning itself, one topic at a time,
and fails only on a topic no file could hold.

**Why it is this way.** The design forbids silent truncation
(`docs/DESIGN.md:203`). Sizes are planned from the header's length and the
section lengths, so each file is assembled once
(`crates/sealmap-corpus/src/pack.rs:301`-`332`). The first version assembled
every candidate file in full. The dense projection of that code showed the
quadratic copy, and commit `673d58c` replaced it.

**Invariant:** at exactly the budget the pack is returned, one byte under it
is refused, and the refusal's sizes add up to the pack's length
(`crates/sealmap-corpus/tests/pack.rs:181`). Every topic lands in exactly one
shard, with its section unchanged (`crates/sealmap-corpus/tests/pack.rs:208`).

**Debt (designed, not built):** the diagrams-only `--review` pack for an
outside reviewer is in the design (`docs/DESIGN.md:84`) but not in the
binary (`docs/DESIGN.md:223`).

## SM-17.4 Choosing topics by change

```mermaid
sequenceDiagram
    autonumber
    participant CL as sealmap pack<br/>main.rs:696
    participant CX as Ctx.model_at<br/>main.rs:460
    participant CT as changed_topics<br/>src/pack.rs:357
    participant ST as seal stale
    participant SS as SymbolState.of<br/>src/pack.rs:382
    CL->>CX: --diff REV (main.rs:705)
    CX-->>CL: the model of REV, same name and options (main.rs:464-466)
    CL->>CT: lock, topics, model at REV, current model (main.rs:706)
    CT->>ST: sealed topics against the model at REV (src/pack.rs:359)
    ST-->>CT: topics with changed sealed symbols
    loop every topic, every valid citation
        CT->>SS: the symbol in both models (src/pack.rs:365)
        SS-->>CT: hashes, unparsable or absent, never its place (src/pack.rs:383-387)
    end
    CT-->>CL: changed topic ids, in id order
    CL->>CL: joined with the named topics (main.rs:708)
```

**What it shows.** A topic is chosen when the lock says its sealed code
changed since REV, or when any symbol it cites was added, removed, made
unparsable, or given a new signature or body hash. A move is not a change.

**Why it is this way.** Sealed topics and topics that only cite symbols both
need review after a change, and this repository's corpus has no seals yet.
The CLI test commits a base, edits one function and sees only the topic
citing it chosen, with the current code in its source window. It then
commits the edit and sees nothing chosen against `HEAD`
(`crates/sealmap/tests/cli.rs:250`).
