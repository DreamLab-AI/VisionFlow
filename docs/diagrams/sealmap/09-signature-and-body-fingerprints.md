---
id: SM-09
title: Signature and body fingerprints from token streams
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-extract/src/fingerprint.rs
  - ../sealmap/crates/sealmap-rust/src/fingerprint.rs
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-rust/tests/fingerprint.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
  - ../sealmap/crates/sealmap-rust/README.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

SM-02 drew the fingerprint as a value. This topic draws how one is made: a
language adapter feeds a `Fingerprinter` a token stream (identifiers,
punctuation, literals and delimited groups, in source order) and decides which
tokens form the contract and which the implementation
(`crates/sealmap-extract/src/fingerprint.rs:4`-`24`). Whitespace and comments
are never tokens, positions are never hashed, so reformatting, editing
comments or moving an item changes neither value.

The Rust adapter adds a second layer of normalisation on top of syn's tokens
(`crates/sealmap-rust/src/fingerprint.rs:1`-`30`): doc and lint attributes are
dropped, layout-only trailing commas are dropped, and the structural rewrites
rustfmt makes (braces round a single-expression closure or match arm, the
comma after each arm, a leading `|`, `return;`, `use` group order) are undone
on a copy of the syntax tree before it is fed. All of this arrived in commit
`0a304ba`, algorithm id `sm1`, with golden values pinned in both crates.

## For the business

The promise to an adopter is narrow and testable: tidying code costs nothing,
changing it costs exactly one re-look. Running rustfmt, adding a comment,
silencing a lint or moving a function to another file leaves every
fingerprint as it was, so no reviewed diagram is flagged. Changing what a
function does changes its body fingerprint; changing how it is called changes
its signature fingerprint. The extractor's README states the claim that the
fingerprints ignore comments, whitespace, rustfmt rewrites and moves
(`crates/sealmap-rust/README.md:18`-`19`); the 50-column reflow experiment
that once backed it is no longer in the README.

The one gap an adopter should know about is macros: when a macro's arguments
are not ordinary expressions, its body is only normalised token by token, so
a formatter rewrite inside it can still change a fingerprint.

## SM-09.1 The shared fingerprinter

```mermaid
classDiagram
    direction LR
    class Fingerprinter {
        hasher keyed blake3  sealmap-extract/src/fingerprint.rs:145
        +new(kind)  sealmap-extract/src/fingerprint.rs:163
        +token(token)  sealmap-extract/src/fingerprint.rs:185
        +section(label)  sealmap-extract/src/fingerprint.rs:238
        +fingerprint(nested)  sealmap-extract/src/fingerprint.rs:244
        +finish() Fingerprint  sealmap-extract/src/fingerprint.rs:249
    }
    class Token {
        <<enum>>
        Ident Punct Literal Open Close  sealmap-extract/src/fingerprint.rs:105
    }
    class Delim {
        <<enum>>
        Paren Bracket Brace None  sealmap-extract/src/fingerprint.rs:81
    }
    class Kind {
        <<enum>>
        Sig Body, distinct key contexts  sealmap-extract/src/fingerprint.rs:121
    }
    Fingerprinter ..> Token
    Fingerprinter ..> Kind
    Token --> Delim
```

**What it shows.** One streaming hasher per value, fed tokens, section markers
that separate the parts of a declaration, and the fingerprints of nested
symbols.

**Why it is this way.** Each record is a tag byte, a little-endian length and
the payload, so the byte stream decodes to exactly one token sequence: `ab` is
never `a b`, and an identifier is never a literal of the same text
(`crates/sealmap-extract/src/fingerprint.rs:148`-`159`). Sections stop tokens
drifting from one part into the next without changing the value
(`crates/sealmap-extract/src/fingerprint.rs:235`-`240`).

**Invariant:** an invisible group contributes its tokens but no delimiters,
and a literal's line endings are normalised, so macro artefacts and Windows
checkouts fingerprint like their plain equivalents
(`crates/sealmap-extract/src/fingerprint.rs:189`-`206`).

## SM-09.2 Fingerprinting one method

```mermaid
sequenceDiagram
    autonumber
    participant IT as Collector.item<br/>sealmap-rust/src/collect.rs:431
    participant IH as impl_header<br/>sealmap-rust/src/collect.rs:549
    participant CA as callable<br/>sealmap-rust/src/collect.rs:527
    participant FD as feed_canonical<br/>sealmap-rust/src/fingerprint.rs:50
    participant FM as fold_member<br/>sealmap-rust/src/collect.rs:517
    IT->>IH: new sig hasher gets the impl header (sealmap-rust/src/collect.rs:451)
    IH-->>IT: attrs, unsafety, generics, trait, self type, where
    IT->>CA: header so far, method attrs, vis, signature, block (sealmap-rust/src/collect.rs:453)
    CA->>CA: attrs, vis, sig sections into the sig hasher (sealmap-rust/src/collect.rs:534)
    CA->>FD: the block alone into a body hasher (sealmap-rust/src/collect.rs:543)
    FD-->>CA: canonical copy fed
    CA-->>IT: sig_hash, body_hash
    IT->>FM: fold name and both values into the module body (sealmap-rust/src/collect.rs:454)
```

**What it shows.** A method's contract includes the header of the impl it
sits in; its body is the block alone; both are then folded into the enclosing
module's body.

**Why it is this way.** The body never sees the name, so a renamed function
keeps its `body_hash` and a rename detector can match it
(`crates/sealmap-rust/src/collect.rs:524`-`526`); the test
`renaming_changes_the_id_and_keeps_the_body_hash` pins it
(`crates/sealmap-rust/tests/fingerprint.rs:228`).

**Open:** because the impl header is part of every method's contract
(`crates/sealmap-rust/src/collect.rs:450`-`452`), adding a bound to an
impl's `where` clause changes the `sig_hash` of every method in it; the design's
contract class (`docs/DESIGN.md:175`) does not say whether that is meant to
read as a contract change for each of them.

## SM-09.3 What goes into each value

```mermaid
flowchart TB
    subgraph CALL["callables, sealmap-rust/src/collect.rs:527"]
        CS["sig: attrs, vis, signature,<br/>plus impl or trait header"]
        CB["body: the block"]
    end
    subgraph DATA["struct, enum, union, sealmap-rust/src/collect.rs:576"]
        DS["sig: the whole declaration"]
        DB["body: generics and fields or<br/>variants, no name"]
    end
    subgraph TRAIT["traits, sealmap-rust/src/collect.rs:617"]
        TS["sig: header and every member signature"]
        TB["body: every member, defaults included"]
    end
    subgraph VALUE["const, static, sealmap-rust/src/collect.rs:656"]
        VS["sig: everything but the value"]
        VB["body: the value"]
    end
    subgraph MODULE["modules, sealmap-rust/src/collect.rs:93"]
        MS["sig: attrs, vis, name"]
        MB["body: members' values in order,<br/>declarations sorted"]
    end
    CALL --> DATA --> TRAIT --> VALUE --> MODULE
```

**What it shows.** The per-kind rule table from the shared crate
(`crates/sealmap-extract/src/fingerprint.rs:17`-`24`) as the Rust adapter
implements it.

**Why it is this way.** Data-type fields are fed one by one, each with its own
separator, so the trailing comma of a field list never enters the stream
(`crates/sealmap-rust/src/collect.rs:572`-`575`).

**Open:** a module's `body_hash` folds the values of every member
(`crates/sealmap-rust/src/collect.rs:179`-`181`), so it changes on any edit
anywhere in the module; neither the design nor the README says whether a
module is meant to be a sealable symbol.

## SM-09.4 Undoing what rustfmt does

```mermaid
flowchart TB
    N["a block, expression or item"]
    CP["clone it<br/>canonical, sealmap-rust/src/fingerprint.rs:55"]
    V1["closure or arm body that is a block with<br/>one expression: the expression<br/>sealmap-rust/src/fingerprint.rs:80-96"]
    V2["every match arm gets its comma,<br/>or-patterns lose a leading bar<br/>sealmap-rust/src/fingerprint.rs:95-100"]
    V3["final return, break, continue get their<br/>semicolon<br/>sealmap-rust/src/fingerprint.rs:105"]
    V4["use groups sorted<br/>sealmap-rust/src/fingerprint.rs:129"]
    MQ{"macro arguments parse as<br/>comma-separated expressions?<br/>sealmap-rust/src/fingerprint.rs:113"}
    MY["each argument normalised the same way<br/>sealmap-rust/src/fingerprint.rs:116"]
    MN["tokens left as written"]
    FT["feed_trees: drop doc, lint and tool<br/>attributes and layout-only commas<br/>sealmap-rust/src/fingerprint.rs:187"]
    N --> CP --> V1 --> V2 --> V3 --> V4 --> MQ
    MQ -->|yes| MY --> FT
    MQ -->|no| MN --> FT
```

**What it shows.** Structural rewrites are undone on a copy of the syntax tree;
token-level layout is dropped while feeding.

**Why it is this way.** Some rustfmt rewrites change the tree, not just the
spacing, so they cannot be ignored at the token level
(`crates/sealmap-rust/src/fingerprint.rs:20`-`26`). Lint and tool attributes
change diagnostics, not the program, so adding an `#[allow]` is not a contract
change (`crates/sealmap-rust/src/fingerprint.rs:6`-`10`).

**Debt:** macro bodies are only normalised when their arguments parse as
expressions (`crates/sealmap-rust/src/fingerprint.rs:111`-`123`); any other
macro body is hashed token by token
(`crates/sealmap-rust/src/fingerprint.rs:27`-`30`), so a formatter rewrite
inside, for example, a `macro_rules!` arm or a DSL macro changes the
fingerprint, and a `macro_rules!` definition's body is fed raw
(`crates/sealmap-rust/src/collect.rs:423`).

## SM-09.5 Is this comma layout?

```mermaid
flowchart TB
    C["a comma in a token group"]
    NX{"what follows it?<br/>is_trailing, sealmap-rust/src/fingerprint.rs:238"}
    END{"end of the group: is it a paren group<br/>that is not an argument list,<br/>with one top-level comma?<br/>sealmap-rust/src/fingerprint.rs:240"}
    GT["a closing angle bracket: dropped<br/>sealmap-rust/src/fingerprint.rs:241"]
    WH["inside a where clause, before the body<br/>or a semicolon: dropped<br/>sealmap-rust/src/fingerprint.rs:242-243"]
    KEEP["kept: a one-element tuple"]
    DROP["dropped"]
    OTHER["anything else: kept<br/>sealmap-rust/src/fingerprint.rs:244"]
    C --> NX
    NX -->|nothing| END
    NX -->|angle| GT
    NX -->|where| WH
    NX -->|other| OTHER
    END -->|yes| KEEP
    END -->|no| DROP
```

**What it shows.** A comma before a closing brace, bracket or `>` is layout;
before a closing parenthesis it is layout only in an argument or parameter
list, or when the group holds other commas; `(T,)` keeps its comma because it
is a tuple and `(T)` is not.

**Why it is this way.** rustfmt adds trailing commas when it breaks a list over
lines (`crates/sealmap-rust/src/fingerprint.rs:12`-`18`); deciding "argument
list" looks at the token before the group, with a keyword list to tell
`if (a,)` from `f(a,)` (`crates/sealmap-rust/src/fingerprint.rs:177`-`185`).

**Invariant:** the one-tuple distinction is tested both ways
(`crates/sealmap-rust/src/fingerprint.rs:329`-`331`).

## SM-09.6 The semantics the tests pin

```mermaid
flowchart TB
    A["reformatting and comments change nothing<br/>tests/fingerprint.rs:137"]
    B["no symbol is left unfingerprinted<br/>tests/fingerprint.rs:149"]
    C["moving items changes neither hash<br/>tests/fingerprint.rs:157"]
    D["a body edit changes only the body hash<br/>tests/fingerprint.rs:175"]
    E["a signature edit changes the sig hash<br/>tests/fingerprint.rs:200"]
    F["renaming keeps the body hash<br/>tests/fingerprint.rs:228"]
    G["golden values, adapter side<br/>tests/fingerprint.rs:246"]
    H["formatting never changes an id<br/>tests/fingerprint.rs:276"]
    I["golden values, shared side<br/>sealmap-extract/src/fingerprint.rs:269"]
    A --> C --> D --> E --> F
    B --> G --> H --> I
```

**What it shows.** Each claim in the README's change table has a test, and the
encoding itself is pinned by golden values in both crates.

**Why it is this way.** Any change to the encoding or the per-kind table is a
new algorithm id; the golden tests fail until it is bumped
(`crates/sealmap-extract/src/fingerprint.rs:42`-`44`).

**Invariant:** the Windows-line-ending case, including inside a multi-line
string literal, hashes like the Unix one
(`crates/sealmap-rust/tests/fingerprint.rs:142`-`145`).
