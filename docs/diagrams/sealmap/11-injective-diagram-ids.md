---
id: SM-11
title: Injective diagram ids from symbol ids
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-mermaid/src/symbol.rs
  - ../sealmap/crates/sealmap-mermaid/src/escape.rs
  - ../sealmap/crates/sealmap-corpus/src/naming.rs
  - ../sealmap/crates/sealmap-corpus/src/lib.rs
  - ../sealmap/crates/sealmap-corpus/src/overview.rs
  - ../sealmap/crates/sealmap-corpus/tests/contract.rs
  - ../sealmap/crates/sealmap-model/src/sym.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

A Mermaid id may hold only letters, digits and underscores, but a `sym:` id
holds spaces, slashes, hashes, brackets, backticks and arbitrary Unicode.
`Ident::from_symbol` maps one to the other by an encoding that is
**injective by construction**: two different `SymbolId`s never produce the
same diagram id, with no hash suffix and no collision handling
(`crates/sealmap-mermaid/src/symbol.rs:10`-`12`). Where names are plain the
id stays readable: `sym:cargo shop . orders/Orders#place().` becomes
`shop__orders___tOrders___fplace` (`crates/sealmap-mermaid/src/symbol.rs:49`).

This is what lets diagrams drawn separately, in different documents, be
concatenated, merged or diffed by id (`crates/sealmap-corpus/src/lib.rs:64`-`69`).
The encoding replaced, in commit `337301a`, a readable-plus-FNV-1a-suffix
scheme from the hardening commit `8af26b3`, which in turn replaced 0.1's
non-injective `::` → `__` mangling (`docs/DESIGN.md:235`). The corpus still
asserts uniqueness over every generated codebase as a guard. This topic draws
the encoding, why it cannot collide, the proof the tests give, and where ids
are still minted the lossy way.

## For the business

When two different functions share a diagram id, Mermaid silently draws them
as one box. Every arrow into either looks like an arrow into both, and a
reviewer reading the diagram is told something false with nothing on the page
to warn them. In 0.1 that could happen; the self-corpus test that pins the fix
uses exactly such a pair (`crates/sealmap-corpus/tests/contract.rs:165`).

For an adopter this closes a class of silent error rather than reducing its
odds: ids are distinct because of how they are written, not because a check
happened to pass. The readable form also keeps diagrams cheap for a model to
read, since most ids are just the path with separators.

## SM-11.1 Encoding a global id

```mermaid
flowchart TB
    ID["sym:cargo shop . orders/Orders#35;place()."]
    V["borrowed view of the parts<br/>SymbolId.view, sym.rs:634"]
    H["head: the package name<br/>symbol.rs:62"]
    G["manager other than cargo: ___g<br/>symbol.rs:63"]
    R["release version: ___r<br/>symbol.rs:66"]
    N["namespace: __name<br/>symbol.rs:71"]
    T["type ___t, term ___v, method ___f<br/>symbol.rs:72-75"]
    P["type parameter ___p, parameter ___a,<br/>meta ___k, macro ___x<br/>symbol.rs:80-83"]
    D["method disambiguator ___d<br/>symbol.rs:77"]
    OUT["shop__orders___tOrders___fplace"]
    ID --> V --> H --> G --> R --> N --> T --> P --> D --> OUT
```

**What it shows.** The package is the head; each descriptor appends one
component, a module with two underscores and every other kind with three plus a
tag letter; a non-default manager or a release version adds its own tagged
component after the head.

**Why it is this way.** The encoding reads the id through `SymbolId::view`, a
borrowed view that allocates nothing per call site, which was added for this
encoder (`crates/sealmap-model/src/sym.rs:615`-`618`). Path ids start with a
single `_` and unresolved ids with `__u`, so the three id forms never share a
head (`crates/sealmap-mermaid/src/symbol.rs:87`-`102`).

**Invariant:** an encoded id that equals a Mermaid keyword gets `___z`, which
no other rule produces (`crates/sealmap-mermaid/src/symbol.rs:104`-`106`).

## SM-11.2 Plain names and escaped names

```mermaid
flowchart TB
    NM["a name"]
    PL{"plain? letters and digits in runs<br/>joined by single underscores<br/>is_plain, symbol.rs:112"}
    WR["written as is after its tag<br/>symbol.rs:146"]
    ES["tag upper-cased, letters and digits<br/>kept, every other UTF-8 byte as<br/>underscore plus two hex digits<br/>symbol.rs:149, symbol.rs:120"]
    WHY["no name ends with underscore, so<br/>a run of 2 or 3 underscores always<br/>starts the next component<br/>symbol.rs:36-38"]
    NM --> PL
    PL -->|yes| WR --> WHY
    PL -->|no| ES --> WHY
```

**What it shows.** A plain name is copied; any other name is byte-escaped and
flagged by an upper-case tag (`___T`, `___F`, `___N`, `__P` for a head), so a
reader of the id knows which form follows.

**Why it is this way.** The split rule is what makes the encoding injective:
underscores after a name always open the next component, and inside an
escaped name an underscore is always followed by a hex digit, so an id splits
back into exactly one sequence of components
(`crates/sealmap-mermaid/src/symbol.rs:36`-`42`).

**Invariant:** `b__c#` and `b/c#`, and `foo/` and `foo().`, the pairs 0.1's
mangling merged, get different ids (`crates/sealmap-mermaid/src/symbol.rs:54`-`56`).

## SM-11.3 Where diagram ids are minted during generation

```mermaid
sequenceDiagram
    autonumber
    participant GE as generate<br/>sealmap-corpus/src/lib.rs:191
    participant AU as assert_unique_idents<br/>naming.rs:23
    participant IDT as ident<br/>naming.rs:10
    participant FS as Ident.from_symbol<br/>symbol.rs:58
    GE->>AU: the codebase, before any document (sealmap-corpus/src/lib.rs:193)
    loop every symbol id and every relation endpoint
        AU->>IDT: id (naming.rs:27)
        IDT->>FS: encode (naming.rs:11)
        FS-->>AU: diagram id
        AU->>AU: same diagram id seen for another sym id, panic (naming.rs:28)
    end
    Note over GE: documents, sequences and structure all reach<br/>ids through ident, naming.rs:10
```

**What it shows.** Before writing anything, generation encodes every symbol
and every relation endpoint and panics if two distinct ids share a diagram id.

**Why it is this way.** The encoder is injective by construction; the guard
exists so a future change to either grammar cannot silently merge nodes
(`crates/sealmap-corpus/src/naming.rs:14`-`18`), and the design keeps it as an
explicit requirement (`docs/DESIGN.md:134`).

**Debt:** the overview's crate graph mints its node ids with the lossy
`Ident::new` from crate and external root names
(`crates/sealmap-corpus/src/overview.rs:109`,
`crates/sealmap-corpus/src/overview.rs:119`), outside the uniqueness guard,
so two external roots that differ only in characters outside the id charset
share one node.

## SM-11.4 The injectivity proof

```mermaid
flowchart TB
    G["generated sym ids: fixed edge cases,<br/>4096 random global ids, random<br/>path and unresolved ids<br/>symbol.rs:310, symbol.rs:354, symbol.rs:361"]
    E["encode<br/>from_symbol, symbol.rs:58"]
    C["check: charset, no leading digit,<br/>not a keyword<br/>symbol.rs:303-305"]
    DEC["decode, written independently of the<br/>encoding rules<br/>symbol.rs:174"]
    EQ{"decodes back to the same id?<br/>symbol.rs:306"}
    OK["injective on every case"]
    G --> E --> C --> DEC --> EQ
    EQ -->|yes| OK
```

**What it shows.** The tests carry an inverse of the encoding, written from the
rules rather than from the encoder, and require every generated id to decode
back to the symbol it came from.

**Why it is this way.** A function with a left inverse is injective, so
"every id decodes back" is the proof, not merely a sample of non-collisions
(`crates/sealmap-mermaid/src/symbol.rs:171`-`173`).

## SM-11.5 Three generations of diagram id

```mermaid
flowchart TB
    V1["0.1: path ids with :: turned into __<br/>not injective<br/>docs/DESIGN.md:235"]
    V2["hardening, 8af26b3: readable ids,<br/>FNV-1a suffix on collision,<br/>corpus-wide assertion"]
    V3["step 2, 337301a: encoding of the<br/>sym id structure, no suffix<br/>symbol.rs:58"]
    V1 --> V2 --> V3
```

**What it shows.** The id scheme was replaced twice in one day: first patched
with a collision suffix, then rebuilt on the typed `sym:` id.

**Why it is this way.** Once ids had structure, the structure could be encoded
directly, and the suffix and its dead helper were removed (commit `337301a`).
The README describes the current scheme (`README.md:212`-`215`).

The design's hardening table now records the fix as it was built: injective
readable ids derived from `sym:` ids, with no hash suffix
(`docs/DESIGN.md:235`), as the code has it
(`crates/sealmap-mermaid/src/symbol.rs:10`-`12`).
