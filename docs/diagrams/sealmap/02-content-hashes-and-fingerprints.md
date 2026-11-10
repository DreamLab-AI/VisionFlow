---
id: SM-02
title: Content hashes and per-symbol fingerprints
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-model/src/hash.rs
  - ../sealmap/crates/sealmap-model/src/symbol.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/check.rs
  - ../sealmap/crates/sealmap-model/src/codebase.rs
  - ../sealmap/crates/sealmap-model/src/source.rs
  - ../sealmap/crates/sealmap-extract/src/fingerprint.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

sealmap keeps two kinds of hash and never confuses them. A `ContentHash` is
BLAKE3 over a whole file's text with line endings normalised, printed
`blake3:<64 hex>`; it is what the 0.1 corpus uses to tell a stale document
from a hand-edited one. A `Fingerprint` is 16 bytes of a keyed BLAKE3 digest,
printed `blake3-16:<32 hex>`, and every symbol carries two of them:
`sig_hash` for its contract and `body_hash` for its implementation
(`crates/sealmap-model/src/symbol.rs:209`, `crates/sealmap-model/src/symbol.rs:212`).

The model crate only stores, compares, prints, parses and merges these values
(`crates/sealmap-model/src/hash.rs:68`-`72`). How the token streams that
feed a fingerprint are chosen per kind of symbol is the extraction core's job
and is drawn in SM-09. After this topic you should know what each value
means, how two `#[cfg]` twins that share an id end up with one fingerprint,
and what the pair lets the seal check decide (built; drawn in SM-15).

The fingerprints arrived in commit `0a304ba` (step 2 of the 0.2 plan,
`docs/DESIGN.md:366`), after rustc's split between a definition's identity
and its query fingerprints (`docs/DESIGN.md:200`).

## For the business

Keeping a reviewed diagram true has a cost, and the expensive part today is
deciding *whether* a change touched what a topic describes. A content hash
per file answers that coarsely: any edit anywhere in a file flags every topic
that cites it. Per-symbol fingerprints answer it per function, and they answer
two questions rather than one: did the function's contract change, or only its
behaviour?

That split was meant to let an adopter route maintenance by cost: a
behaviour change a cheap re-review, a contract change a more careful look.
The values are deliberately blind to formatting and comments, so a
reformatting commit never changes them. The tooling that acts on the split
(seals, `sealmap verify` and `sealmap stale`, SM-15) exists but is frozen as
experimental: per-symbol hashes missed about a quarter of the changes that
made a topic wrong (`README.md:356`). The hashes themselves stay, as the
identity and change signal of the code lens (`docs/DESIGN.md:33`-`39`).

## SM-02.1 The two hash types

```mermaid
classDiagram
    direction LR
    class ContentHash {
        +String blake3 prefix and 64 hex  hash.rs:21
        +of_text(text) normalises newlines  hash.rs:32
        +of_bytes(bytes) raw  hash.rs:38
        +parse(s) Option  hash.rs:45
        +short(n) str  hash.rs:56
    }
    class Fingerprint {
        +16 bytes  hash.rs:92
        +PREFIX blake3-16  hash.rs:96
        +from_blake3(hash) truncates  hash.rs:104
        +is_unset() all zeros  hash.rs:116
        +parse(s) Option  hash.rs:121
        +merge(other) order independent  hash.rs:145
    }
    class SourceFile {
        +hash ContentHash  source.rs:24
    }
    class Symbol {
        +sig_hash Fingerprint  symbol.rs:209
        +body_hash Fingerprint  symbol.rs:212
    }
    SourceFile --> ContentHash
    Symbol --> Fingerprint
```

**What it shows.** Files carry a full-length content hash; symbols carry two
truncated fingerprints. Both print with a self-describing algorithm prefix.

**Why it is this way.** The prefix keeps the algorithm visible in every
model and lock if it ever changes (`crates/sealmap-model/src/hash.rs:8`-`9`).
Truncating to 128 bits keeps an accidental collision out of reach while
halving the text a full digest costs in every model and lock
(`crates/sealmap-model/src/hash.rs:74`-`76`).

**Invariant:** the default fingerprint is all zeros and means "not
fingerprinted" (`crates/sealmap-model/src/hash.rs:116`-`118`); a symbol built
by hand starts there (`crates/sealmap-model/src/symbol.rs:244`-`245`) until an
adapter fills it in.

## SM-02.2 From bytes to printed value

```mermaid
flowchart TB
    subgraph FILE["file text"]
        T["source text"]
        NL["CR LF and lone CR become LF<br/>normalise_newlines, hash.rs:178"]
        B3["blake3 hash, full 32 bytes<br/>hash.rs:39"]
        CH["blake3 colon 64 hex"]
        T --> NL --> B3 --> CH
    end
    subgraph SYM["symbol tokens"]
        TK["token stream from an adapter"]
        KD["keyed hasher, sig or body context<br/>fingerprint.rs:129"]
        TR["first 16 bytes kept<br/>from_blake3, hash.rs:106"]
        FP["blake3-16 colon 32 hex<br/>Display, hash.rs:156"]
        TK --> KD --> TR --> FP
    end
    FILE --> SYM
```

**What it shows.** A file hash normalises newlines and hashes everything; a
symbol fingerprint hashes a token stream under one of two key-derivation
contexts and keeps the first 16 bytes.

**Why it is this way.** Distinct contexts mean a signature and a body with
identical tokens still get different values
(`crates/sealmap-extract/src/fingerprint.rs:40`-`41`). The context strings
embed the algorithm id and a date, so any change to the scheme is a new
algorithm id (`crates/sealmap-extract/src/fingerprint.rs:77`).

**Invariant:** golden values pin the scheme; an encoding change fails
`golden_values` until `ALGORITHM` is bumped
(`crates/sealmap-extract/src/fingerprint.rs:269`).

## SM-02.3 Merging cfg twins

```mermaid
sequenceDiagram
    autonumber
    participant AD as adapter
    participant CB as add_symbol<br/>codebase.rs:98
    participant FP as Fingerprint.merge<br/>hash.rs:145
    AD->>CB: first definition of an id
    CB->>CB: insert as is (codebase.rs:112)
    AD->>CB: second definition, same id, other cfg
    CB->>FP: fold sig_hash (codebase.rs:101)
    FP->>FP: sort the pair, keyed hash of both (hash.rs:146)
    FP-->>CB: merged sig_hash
    CB->>FP: fold body_hash (codebase.rs:102)
    FP-->>CB: merged body_hash
    CB->>CB: differing flows become one branch,<br/>an arm per twin (codebase.rs:103, codebase.rs:244)
    CB->>CB: extend members, add new tags (codebase.rs:104)
    Note over CB: DEBT: scalar fields such as file and span<br/>keep the first twin only, codebase.rs:68
```

**What it shows.** Two definitions with one id, typically
`#[cfg(unix)]` and `#[cfg(windows)]` twins, become one symbol whose
fingerprints are an order-independent fold of both. Their flows are kept
too: when they differ (other than in line numbers) each twin becomes an arm
of one branch labelled `cfg twin at <file>:<line>`, so the second twin's calls
reach the sequence and the call relations
(`crates/sealmap-model/src/codebase.rs:73`-`76`,
`crates/sealmap-model/src/codebase.rs:244`-`259`).

**Why it is this way.** A seal on a cfg-gated function has to notice an edit
to either twin; folding both values with a dedicated key context means an
edit to either changes the result, whichever twin was seen first
(`crates/sealmap-model/src/hash.rs:133`-`137`).

**Debt:** the merged symbol keeps only the first twin's file, span, signature
and doc (`crates/sealmap-model/src/codebase.rs:67`-`71`), so the second
definition has no location in the model except in a twin arm's label, which
exists only when the twins' flows differ
(`crates/sealmap-model/src/codebase.rs:238`-`239`).

## SM-02.4 A fingerprint's life

```mermaid
stateDiagram-v2
    [*] --> Unset: Symbol.new, symbol.rs 244
    Unset --> Set: adapter assigns sig and body
    Set --> Merged: a cfg twin with the same id arrives
    Merged --> Merged: another twin
    Set --> Printed: serialised as blake3-16 text
    Merged --> Printed
    Printed --> Set: parsed back, either case accepted
    Printed --> Refused: wrong prefix or length
```

**What it shows.** A fingerprint starts unset, is set by an adapter, may be
merged with twins, and travels as text in the JSON model; text with the wrong
prefix or length is refused on read.

**Why it is this way.** Serialisation goes through `Display` and `parse`
rather than serde's byte encoding (`crates/sealmap-model/src/hash.rs:164`-`175`),
so the JSON is readable and identical to what a lock would hold.

**Invariant:** a malformed fingerprint string fails deserialisation instead of
reading as zeros (`crates/sealmap-model/src/hash.rs:173`).

## SM-02.5 What the pair decides

```mermaid
flowchart TB
    S["a sealed symbol: id, sig, body<br/>check.rs:138"]
    R{"is the id in the model?"}
    PU{"is it the module of a file<br/>tagged parse_error?<br/>check.rs:146"}
    SG{"sig_hash equal?<br/>check.rs:147"}
    BD{"body_hash equal?<br/>check.rs:150"}
    AU{"is an ancestor module<br/>tagged parse_error?<br/>check.rs:155"}
    UNP["unparsable, fail closed"]
    HOLDS["holds<br/>check.rs:153"]
    BEH["behaviour, body only"]
    CON["contract"]
    ABS["absent: same-kind ids with the sealed<br/>body are rename candidates<br/>check.rs:158"]
    S --> R
    R -->|yes| PU
    PU -->|yes| UNP
    PU -->|no| SG
    SG -->|no| CON
    SG -->|yes| BD
    BD -->|yes| HOLDS
    BD -->|no| BEH
    R -->|no| AU
    AU -->|yes| UNP
    AU -->|no| ABS
```

**What it shows.** The decision the seal check takes from the two
fingerprints: a body-only change is behaviour, a signature change is
contract, and a vanished id whose body reappears in a symbol of the same kind
is a suspected rename. Unparsable code is decided first, so it is never
reported as holding or absent (`crates/sealmap-corpus/src/seal/check.rs:146`,
`crates/sealmap-corpus/src/seal/check.rs:155`); the design's table is
`docs/DESIGN.md:169`-`177`.

**Why it is this way.** The name sits in `sig_hash` and never in `body_hash`,
which is what makes a rename detectable by body
(`crates/sealmap-extract/src/fingerprint.rs:26`-`27`); the README says for users
only that the signature and body fingerprints ignore formatting and comments
(`README.md:365`-`366`).

The crate-surface table records that an optional `flow_hash` was considered
and not built (`docs/DESIGN.md:200`); the model has only the two
(`crates/sealmap-model/src/symbol.rs:209`-`212`), as commit `0a304ba` left it.
