---
id: SM-15
title: Seals, the lock and the checks
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-corpus/src/seal/mod.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/lock.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/topic.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/check.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/sign.rs
  - ../sealmap/crates/sealmap-corpus/tests/seal.rs
  - ../sealmap/crates/sealmap-corpus/tests/lock_props.rs
  - ../sealmap/crates/sealmap-model/src/symbol.rs
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-extract/src/lower.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: eacc59ece21274273c6bf2c83d4642b4fa7acaf5
---
## For developers

A seal records that a hand-written topic was reviewed against exact versions
of the code it cites. `sealmap_corpus::seal` holds everything that makes that
record checkable: the lock format and its one canonical text, the hash that
binds a topic's prose, the rule that finds citations in Markdown, the
classifier, and the step that writes a seal. It is pure: callers pass models
and texts, and nothing reads the file system or runs git
(`crates/sealmap-corpus/src/seal/mod.rs:1`-`6`). The CLI around it is SM-16.

The module implements DESIGN §3 (`docs/DESIGN.md:92`-`163`). Where the design
was loose the build chose, and DESIGN now records the choices: the exact lock
layout, `topic_hash`, the citation rule, failing closed from the model alone,
and legacy topics as coverage rather than failures. After this topic you
should know what each class means, where it is decided, and which limits the
tests pin.

## For the business

A seal records a judgement mechanically. A reviewer, human or model, decides
once that a topic matches the code; the lock then remembers exactly which
versions of which functions that judgement was about. From then on a CI job
can say "no cited function changed" or name the one that did and how (its
behaviour, its contract, or its name), with no model and no tokens involved.

That is narrower than "the topic is still true", and the evidence says the
gap matters. On blind-labelled commits, per-symbol body hashes caught 0.76 of
the changes that made a topic wrong, against 1.0 for per-file flags, while
flagging only 1.19× fewer topics (`README.md:81`-`85`). The seal surface is
therefore experimental and frozen, not the recommended way to keep a corpus
true (`docs/DESIGN.md:19`-`30`).

It is deliberately strict where strictness is cheap. An edit to the prose
needs a re-seal, a function that cannot be parsed counts as changed rather
than as fine, and a lock edited by hand is refused. What it does not do is
judge: whether a behaviour change makes a diagram wrong is still the
reviewer's call, and the lock records who made it without policing who that
may be.

## SM-15.1 The lock and the report as types

```mermaid
classDiagram
    class Lock {
        +version u32
        +algorithm String
        +generator String
        +topics BTreeMap of id to TopicSeal
        +parse(text) lock.rs:203
        +parse_canonical(text) lock.rs:231
        +to_toml() lock.rs:248
    }
    class TopicSeal {
        +id String
        +file SourcePath
        +topic_hash Fingerprint
        +reviewer String
        +model String
        +date String
        +symbols BTreeMap of SymbolId to SymbolSeal
    }
    class SymbolSeal {
        +sig Fingerprint
        +body Fingerprint
    }
    class Class {
        <<enum>>
        Holds
        Behaviour
        Contract
        Absent
        Unparsable
        UnsealedCitation
        Orphan
        ProseEdited
        LockFault
    }
    class Finding {
        +topic, file, symbol
        +class Class
        +detail String
        +candidates Vec of SymbolId
    }
    class SealReport {
        +findings Vec of Finding
        +unsealed_topics Vec of SourcePath
        +passes() check.rs:104
    }
    Lock *-- TopicSeal : topics
    TopicSeal *-- SymbolSeal : symbols
    SealReport *-- Finding : findings
    Finding --> Class
    note for Lock "lock.rs:57"
    note for Class "one variant per row of<br/>the design's table, check.rs:17"
```

**What it shows.** A lock is a map of topic seals keyed by topic id, each
holding the prose hash, two opaque strings and the two fingerprints of every
sealed symbol; a check produces findings, one class each.

**Why it is this way.** Ordered maps make the canonical text a function of the
data: topics in id order, symbols in id order, so two people sealing
different topics touch disjoint hunks (`crates/sealmap-corpus/src/seal/lock.rs:23`-`30`).
`reviewer` and `model` stay opaque so reviewer policy lives in the skill, not
the crate (`crates/sealmap-corpus/src/seal/lock.rs:79`-`82`).

**Invariant:** only `Holds` passes (`crates/sealmap-corpus/src/seal/check.rs:45`-`46`),
and a report passes only when every finding does
(`crates/sealmap-corpus/src/seal/check.rs:104`-`105`).

## SM-15.2 Reading a lock

```mermaid
flowchart TB
    T["lock text"]
    N["line endings normalised<br/>lock.rs:204"]
    TP["TOML into raw structs that refuse<br/>unknown keys, lock.rs:142"]
    V{"version 1 and<br/>algorithm sm1?<br/>lock.rs:207, lock.rs:210"}
    F["each topic validated: id, normalised file,<br/>hash forms, YYYY-MM-DD date,<br/>no repeated symbol<br/>lock.rs:282-304"]
    D{"a file or a topic id<br/>sealed twice?<br/>lock.rs:216, lock.rs:219"}
    L["Lock"]
    C{"to_toml equals the<br/>text byte for byte?<br/>lock.rs:233"}
    NC["NonCanonical<br/>lock.rs:234"]
    E["LockError"]
    T --> N --> TP --> V
    V -->|no| E
    V -->|yes| F --> D
    D -->|yes| E
    D -->|no| L
    L -->|parse_canonical| C
    C -->|no| NC
```

**What it shows.** `parse` validates every field and returns a lock; only
`parse_canonical` also demands the exact bytes the writer would produce.

**Why it is this way.** Canonicality is checked by re-serialising, so the
writer is the only definition of the format and there is no second grammar to
drift from it (`crates/sealmap-corpus/src/seal/lock.rs:227`-`235`). `parse`
alone normalises line endings, so a lock checked out with `\r\n` can be read
and re-signed, while `verify` still reports it as not canonical
(`crates/sealmap-corpus/src/seal/lock.rs:229`-`230`).

**Invariant:** the writer round-trips byte for byte and is independent of
signing order, pinned by a fixed-layout test and a property test over
arbitrary strings (`crates/sealmap-corpus/tests/seal.rs:554`,
`crates/sealmap-corpus/tests/lock_props.rs:46`).

## SM-15.3 What a topic contributes

```mermaid
flowchart TB
    TX["topic text, line endings normalised<br/>topic.rs:19"]
    FM["front matter: first line ---,<br/>closed by the next ---<br/>topic.rs:26"]
    PH["topic_hash: drop every sealed: line in<br/>the front matter, BLAKE3, first 16 bytes<br/>topic.rs:110-113"]
    PT["pointers: every sealed: value<br/>topic.rs:72"]
    ID["topic_id: the id: value<br/>topic.rs:64"]
    BODY["body after the front matter"]
    FE["fenced blocks skipped, so Mermaid<br/>is never scanned<br/>topic.rs:246, topic.rs:252"]
    PA["paragraphs, ended by a blank line<br/>topic.rs:260"]
    SP["CommonMark code spans<br/>topic.rs:284"]
    GL{"starts like a global id:<br/>sym:, a manager, a space?<br/>topic.rs:331"}
    CI["citation, with its parse result<br/>topic.rs:182"]
    MN["a grammar mention, not a citation"]
    TX --> FM
    FM --> PH
    FM --> PT
    FM --> ID
    FM --> BODY --> FE --> PA --> SP --> GL
    GL -->|yes| CI
    GL -->|no| MN
```

**What it shows.** The front matter yields the id, the pointer and (with the
pointer removed) the bytes the prose hash covers; the body yields citations,
which are code spans outside fences that start like a global `sym:` id.

**Why it is this way.** Ids contain spaces, so nothing but a code span can
delimit one, and a span can wrap across lines inside a paragraph
(`crates/sealmap-corpus/src/seal/topic.rs:198`-`214`). Removing the pointer
before hashing is what lets a re-seal leave the prose untouched
(`crates/sealmap-corpus/src/seal/topic.rs:92`-`94`). A span that starts like a
global id but does not parse is kept with its error, so a typo fails a check
instead of vanishing (`crates/sealmap-corpus/src/seal/topic.rs:198`-`200`).

**Open:** an example id written in prose is indistinguishable from a citation
(`crates/sealmap-corpus/src/seal/topic.rs:343`-`350`); the only escape is a
fenced block, and nothing records whether topics that explain the grammar are
meant to need one.

## SM-15.4 One verify run

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant VF as verify<br/>check.rs:359
    participant LK as Lock
    participant SC as seal_check<br/>check.rs:222
    participant CE as check_entry<br/>check.rs:243
    participant TP as topic functions
    participant CF as classify<br/>check.rs:138
    CL->>VF: lock text or none, lock name, topics, model
    VF->>LK: parse (check.rs:360)
    alt the lock does not parse
        VF-->>CL: one LockFault and nothing else (check.rs:363-367)
    end
    VF->>SC: every entry (check.rs:370)
    loop each lock entry (check.rs:236)
        SC->>CE: entry
        CE->>CE: topic file missing, Orphan, stop (check.rs:259-260)
        CE->>TP: topic_id, pointers (check.rs:263, check.rs:268)
        CE->>CE: id or pointer disagrees, LockFault (check.rs:265, check.rs:269)
        CE->>TP: topic_hash (check.rs:271)
        CE->>CE: Holds or ProseEdited for the prose (check.rs:272-275)
        loop each sealed symbol (check.rs:280)
            CE->>CF: id and sealed hashes (check.rs:281)
            CE->>CE: sealed but no longer cited, Orphan (check.rs:283-284)
        end
        CE->>CE: cited but not sealed, UnsealedCitation (check.rs:287)
    end
    VF->>VF: text not canonical, LockFault (check.rs:372-373)
    loop each topic without an entry (check.rs:376)
        VF->>VF: pointer without entry, LockFault (check.rs:383)
        VF->>VF: its citations, UnsealedCitation (check.rs:390)
        VF->>VF: neither, coverage only (check.rs:391-392)
    end
    VF-->>CL: SealReport, sorted (check.rs:395)
```

**What it shows.** `seal_check` judges each lock entry against its topic and
the model; `verify` adds what only a corpus-wide view sees: canonicality,
pointers with no entry, and citations in topics nobody sealed.

**Why it is this way.** A non-canonical lock is still classified in full, so
one stray byte does not hide a real behaviour change
(`crates/sealmap-corpus/tests/seal.rs:387`). Legacy `path:line` topics are
coverage, not failures, so a corpus migrates one topic at a time
(`docs/DESIGN.md:161`-`163`).

**Debt:** removing a topic's lock entry and its pointer together passes
`verify` whenever the topic cites no `sym:` id, because such a topic is then
indistinguishable from one never sealed (`crates/sealmap-corpus/src/seal/check.rs:391`-`392`);
nothing remembers that a topic was once sealed.

## SM-15.5 Deciding one symbol, and failing closed

```mermaid
flowchart TB
    IN["sealed id with sealed sig and body<br/>check.rs:138"]
    UM["modules of files that did not parse,<br/>found by their tag<br/>check.rs:174-175"]
    P{"in the model?"}
    U1{"is it such a module?<br/>check.rs:146"}
    U2{"is an ancestor such a module?<br/>check.rs:155"}
    CMP["compare sig, then body<br/>check.rs:147, check.rs:150"]
    UNP["Unparsable"]
    RC["Absent, with same-kind ids whose<br/>body equals the sealed body<br/>check.rs:158, check.rs:191-199"]
    IN --> P
    UM -.-> U1
    UM -.-> U2
    P -->|yes| U1
    U1 -->|yes| UNP
    U1 -->|no| CMP
    P -->|no| U2
    U2 -->|yes| UNP
    U2 -->|no| RC
```

**What it shows.** A file the adapter could not parse survives in the model as
its module symbol tagged `parse_error`; any sealed id that is that module, or
sits beneath it and is missing, is unparsable before it can be called absent
or changed. SM-02.5 draws the comparison itself.

**Why it is this way.** The tag is a named constant in the model
(`crates/sealmap-model/src/symbol.rs:360`), so the check needs nothing but the
`Codebase`, not the adapter's diagnostics. An unset fingerprint never matches
as a rename candidate (`crates/sealmap-corpus/src/seal/check.rs:192`).

**Invariant:** unparsable code never reads as holding or absent; pinned on a
broken file holding two sealed methods and the sealed module itself
(`crates/sealmap-corpus/tests/seal.rs:255`).

**Debt:** a module's `body_hash` folds its members in source order
(`crates/sealmap-rust/src/collect.rs:179`-`180`), so reordering methods inside
a sealed module turns the module into Behaviour while every method holds
(`crates/sealmap-corpus/tests/seal.rs:187`); a module is a poor thing to seal.

**Debt:** the generated diagram of `classify` omits this unparsable arm, and
the generated diagram of `sign` omits its success arm, because lowering drops
any arm without a kept call (`crates/sealmap-extract/src/lower.rs:71`,
`crates/sealmap-corpus/src/seal/check.rs:146`,
`crates/sealmap-corpus/src/seal/sign.rs:125`-`126`); read from the generated
corpus alone, both functions look as if they had no such path.

## SM-15.6 Writing a seal

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller, the CLI or a skill
    participant SG as sign<br/>sign.rs:99
    participant TP as topic functions
    participant RS as resolve<br/>check.rs:441
    participant LK as Lock
    CL->>SG: lock, topic file and text, model, lock name, who
    SG->>SG: reviewer and model non-empty, date YYYY-MM-DD (sign.rs:107-111)
    SG->>TP: topic_id, refuse if missing (sign.rs:113)
    SG->>LK: id already sealed for another file? refuse (sign.rs:117-118)
    loop each citation (sign.rs:121)
        SG->>SG: malformed? refuse (sign.rs:122)
        SG->>RS: current hashes (sign.rs:124)
        RS-->>SG: Found with fingerprints, else refuse (sign.rs:125-130)
    end
    SG->>TP: put one sealed: pointer in place (sign.rs:133)
    SG->>LK: drop any entry for this file, set generator (sign.rs:135-136)
    SG->>LK: insert the entry with the topic_hash of the new text (sign.rs:137-140)
    SG-->>CL: topic text with its pointer (sign.rs:146)
```

**What it shows.** A seal is derived, never typed in: every cited id is
resolved now, and any citation `verify` could not confirm stops the seal
before the lock is touched.

**Why it is this way.** The design makes the sign step mechanical so the lock
is never hand-edited (`docs/DESIGN.md:122`-`127`). Replacing by file as well as
by id means a renumbered topic moves its seal rather than leaving the old
entry orphaned (`crates/sealmap-corpus/tests/seal.rs:543`).

**Invariant:** nothing is sealed that `verify` would not then pass: a citation
that is malformed, absent, unparsable or unfingerprinted is refused
(`crates/sealmap-corpus/src/seal/sign.rs:122`-`130`), pinned by
`crates/sealmap-corpus/tests/seal.rs:523`.
