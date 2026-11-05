---
id: SM-01
title: The sym id grammar
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-model/src/sym.rs
  - ../sealmap/crates/sealmap-model/tests/sym_props.rs
  - ../sealmap/crates/sealmap-extract/src/ids.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

Every symbol sealmap knows about is named by a `SymbolId`, a string in a
SCIP-descriptor-style grammar that carries no file path and no line number
(`crates/sealmap-model/src/sym.rs:18` holds the grammar). This topic is the
contract for that string: its three forms, how it is printed and parsed, why
printing is injective, and how a parent or child id is derived without
allocating. After reading it you should be able to read any id in a
generated model, predict the id a definition will get, and know which
operations are cheap.

The id is the identity half of rustc's identity/freshness split
(`crates/sealmap-model/src/sym.rs:3`-`9`): the content half lives in
fingerprints (SM-02). It was introduced in commit `5973a4e` as step 2 of the
0.2 plan (`docs/DESIGN.md:296`), replacing the 0.1 `::` path ids whose
`::` → `__` mangling was not injective (`docs/DESIGN.md:235`).

The builder that mints ids for real definitions is `sealmap-extract`'s
`ids` module (SM-05); this topic covers the grammar it targets. What this
topic does not cover: the Mermaid id derived from a `SymbolId` (SM-11) and
the JSON schema that carries ids (SM-03).

## For the business

A diagram corpus is only as trustworthy as its citations. Today a citation
written as a file and line breaks the moment someone moves the code, and an
agent harness then pays to re-read and re-stamp topics that did not change in
substance. The `sym:` id is the part of sealmap that removes that cost at the
root: it names a function by where it sits in the program, so moving it
between files changes nothing, while renaming it or moving it to another
module is a visible change.

For a harness owner the property that matters is that the name is exact and
unambiguous. Two different functions can never print the same id, and a
mistyped id is refused rather than quietly matched to something close. That
is what later lets a seal say "this topic was reviewed against exactly these
functions" (built, see SM-15) and be believed in CI without a model
in the loop.

## SM-01.1 The id's shape as types

```mermaid
classDiagram
    direction LR
    class SymbolId {
        +Arc str canonical text  sym.rs:108
        +parse(text) Result  sym.rs:436
        +parent() Option  sym.rs:544
        +child(descriptor) Option  sym.rs:576
        +view() IdView  sym.rs:634
    }
    class Repr {
        <<private enum>>
        Global package descriptors  sym.rs:113
        Path segments  sym.rs:114
        Unresolved name  sym.rs:115
    }
    class Package {
        manager  sym.rs:131
        name  sym.rs:132
        version  sym.rs:133
    }
    class Version {
        <<enum>>
        Current printed dot  sym.rs:140
        Release string  sym.rs:142
    }
    class Descriptor {
        name  sym.rs:157
        suffix  sym.rs:158
    }
    class Suffix {
        <<enum>>
        Namespace Type Term  sym.rs:165
        Method disambiguator  sym.rs:172
        TypeParameter Parameter Meta Macro  sym.rs:179
    }
    SymbolId ..> Repr : built and checked through
    Repr --> Package
    Package --> Version
    Repr --> Descriptor
    Descriptor --> Suffix
```

**What it shows.** A `SymbolId` stores only its canonical text in an
`Arc<str>`; the structured `Repr` exists to build and validate that text and
is reconstructed on demand. A global id is a package (manager, name,
version) plus a list of descriptors, each a name with a kind suffix.

**Why it is this way.** Holding the text means equality, ordering and hashing
are plain string operations, and a parent sorts before its children because
its text is a prefix of theirs (`crates/sealmap-model/src/sym.rs:101`-`104`).
The owner chose SCIP's descriptor suffixes so a global id maps to a SCIP
symbol by swapping the prefix (`crates/sealmap-model/src/sym.rs:11`-`13`),
which is what keeps a later `export --scip` cheap (`docs/DESIGN.md:164`).

**Invariant:** two ids are equal exactly when their texts are, because only
canonical text is ever stored; `parse` refuses anything that does not print
back unchanged (`crates/sealmap-model/src/sym.rs:440`).

**Debt:** the structural accessors `package()` and `descriptors()` re-run the
full parser over the text on every call through `repr()`
(`crates/sealmap-model/src/sym.rs:456`, `crates/sealmap-model/src/sym.rs:475`),
and `repr()` panics through `expect` if the text were ever non-canonical; only
`view()` and the scanner avoid the reparse.

## SM-01.2 Three forms and their suffixes

```mermaid
flowchart TB
    START["a reference or a definition<br/>needs a name"]
    Q1{"are its kinds known<br/>all the way down?"}
    Q2{"is it a method on a receiver<br/>whose type is unknown?"}
    G["global<br/>sym:cargo shop . db/Db#35;insert().<br/>SymbolId.global, sym.rs:405"]
    P["path<br/>sym:extern serde_json::to_string<br/>SymbolId.path, sym.rs:415"]
    U["unresolved<br/>sym:? insert<br/>SymbolId.unresolved, sym.rs:428"]
    SUF["kind suffixes, sym.rs:29-35<br/>slash module, hash type, dot term,<br/>paren-dot method, bang macro,<br/>brackets type parameter or trait impl"]
    ROOT["no descriptors = package root<br/>sym.rs:410"]
    START --> Q1
    Q1 -->|yes| G
    Q1 -->|no| Q2
    Q2 -->|yes| U
    Q2 -->|no| P
    G --> SUF
    G --> ROOT
```

**What it shows.** Everything a language adapter defines gets a global id;
a path into a dependency whose kinds are unknown is a `sym:extern` path id;
a method called on a receiver of unknown type is `sym:? name`
(`crates/sealmap-model/src/sym.rs:50`-`57`).

**Why it is this way.** The version slot is `.` for "the tree being
analysed", so publishing a release does not churn every id
(`crates/sealmap-model/src/sym.rs:51`-`52`). The `[Trait]` descriptor doubles
as the marker of a trait-impl block, so `Db#[Store]put().` names `put` from
`impl Store for Db` and still has `Db#` as its parent
(`crates/sealmap-model/src/sym.rs:176`-`178`).

**Invariant:** `extern` can never be a manager name, so the path form cannot
be confused with a global id (`crates/sealmap-model/src/sym.rs:263`).

**Open:** `Version::Release` is part of the grammar
(`crates/sealmap-model/src/sym.rs:142`), but the only id builder mints
packages at the current tree (`crates/sealmap-extract/src/ids.rs:57`); no
record says when, or by whom, a release-versioned id is meant to be minted.

## SM-01.3 Parsing is the exact inverse of printing

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant PA as parse<br/>sym.rs:436
    participant PI as Parser.id<br/>sym.rs:976
    participant PK as Package.new<br/>sym.rs:221
    participant PR as print<br/>sym.rs:703
    CL->>PA: text
    PA->>PI: scan from byte 0 (sym.rs:437)
    PI->>PI: prefix, then unresolved or extern branch (:980)
    PI->>PK: manager, name, version fields (:999)
    PK-->>PI: refuses bad manager or field (sym.rs:224)
    PI->>PI: descriptors until end of text (:1006)
    PI-->>PA: Repr
    PA->>PR: print the Repr back (sym.rs:440)
    PR-->>PA: canonical text
    Note over PA: INVARIANT: text that does not print back<br/>unchanged is refused, sym.rs:441
    PA-->>CL: SymbolId holding the input text (sym.rs:443)
```

**What it shows.** The parser builds a `Repr`, then prints it and compares
the result with the input. Only text that survives the round trip becomes an
id, so a simple name wrapped in backticks, a doubled separator or a trailing
space is an error rather than an alias.

**Why it is this way.** Ids are compared by text, so two spellings of one
symbol would silently be two symbols. The round-trip guard makes that
impossible by construction; the parser also refuses a simple name that was
needlessly quoted (`crates/sealmap-model/src/sym.rs:1049`-`1051`). The
property tests state the same contract from outside: printing then parsing
gives the id back (`crates/sealmap-model/tests/sym_props.rs:79`) and any
accepted text is canonical (`crates/sealmap-model/tests/sym_props.rs:130`).

**Invariant:** two distinct ids never print the same text, checked over 4,096
generated cases (`crates/sealmap-model/tests/sym_props.rs:85`).

## SM-01.4 Escaping names and package fields

```mermaid
flowchart TB
    N["a descriptor or path name"]
    S{"non-empty and every char<br/>in the simple set?<br/>is_simple, sym.rs:368"}
    BARE["written bare<br/>write_name, sym.rs:373"]
    Q["wrapped in backticks,<br/>backtick written twice<br/>sym.rs:377-384"]
    F["a package name or version"]
    FS["a space is written twice<br/>write_field, sym.rs:389"]
    FC{"empty, edge space,<br/>or control char?<br/>check_field, sym.rs:268"}
    REJ["IdError.Field<br/>sym.rs:200"]
    N --> S
    S -->|yes| BARE
    S -->|no| Q
    F --> FC
    FC -->|yes| REJ
    FC -->|no| FS
```

**What it shows.** Names outside `[A-Za-z0-9_+$-]` are quoted in backticks
with backticks doubled; package fields keep their spaces but write each one
twice, so a single space always separates fields.

**Why it is this way.** SCIP's own space rule is ambiguous at field edges;
forbidding a leading or trailing space closes that ambiguity
(commit `5973a4e`, pinned by the edge-space assertion in
`crates/sealmap-model/tests/sym_props.rs:123`). The scanner relies on it: an
odd-length run of spaces is a separator
(`crates/sealmap-model/src/sym.rs:829`-`832`).

**Invariant:** pairs that the 0.1 `::` → `__` mangling merged, and pairs
SCIP's space escaping would merge, print differently
(`crates/sealmap-model/tests/sym_props.rs:102`).

## SM-01.5 Navigating without reparsing

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant PT as parent<br/>sym.rs:544
    participant SH as Shape.of<br/>sym.rs:814
    participant SD as scan_descriptors<br/>sym.rs:882
    participant CH as child<br/>sym.rs:576
    CL->>PT: parent of method put in the Store impl of Db
    PT->>SH: split text into manager, name, version, descriptors (sym.rs:546)
    PT->>SD: descriptor spans (sym.rs:548)
    PT->>PT: drop last span, then trailing type-parameter spans (:550)
    PT-->>CL: prefix slice of the text, the id of Db (sym.rs:559)
    CL->>CH: child of Db with method get
    CH->>SH: confirm global form (sym.rs:577)
    CH-->>CL: text plus written descriptor (sym.rs:583)
```

**What it shows.** `parent()` and `child()` work on the canonical text: the
parent is a prefix slice of the child's text, and a child is the parent's text
with one descriptor appended. Neither runs the parser.

**Why it is this way.** Corpus generation derives a parent or owner for every
call site, so these sit on the hot path; the borrowed scanner
(`crates/sealmap-model/src/sym.rs:796`-`800`) never allocates unless a name
needs unescaping. Dropping trailing type-parameter descriptors is what makes a
trait-impl method's parent its type rather than the impl block.

**Invariant:** a parent always sorts before its child, because its text is a
strict prefix of the child's (`crates/sealmap-model/tests/sym_props.rs:92`).

## SM-01.6 Where the grammar is documented and claimed

```mermaid
flowchart TB
    D1["DESIGN.md section 4<br/>sym grammar, kind-explicit, no file path<br/>docs/DESIGN.md:130"]
    D2["README ids section<br/>printing injective, canonical-only parse<br/>README.md:182"]
    C1["grammar in rustdoc<br/>crates/sealmap-model/src/sym.rs:17"]
    C2["property tests<br/>crates/sealmap-model/tests/sym_props.rs:75"]
    C3["id builder, one per adapter<br/>crates/sealmap-extract/src/ids.rs:3"]
    D1 --> C1
    D2 --> C1
    C1 --> C2
    C1 --> C3
```

**What it shows.** The design and the README both point at the rustdoc
grammar as the authority, and the property tests and the shared id builder
are the two things that hold the code to it.

**Why it is this way.** The README says the grammar "is documented in
`sealmap_model::sym`" (`README.md:183`) rather than restating it, so there is
one source; the README's example table (`README.md:169`-`175`) is
illustrative.
