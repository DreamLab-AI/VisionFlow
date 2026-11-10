---
id: SM-10
title: Typed Mermaid writers and escaping
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-mermaid/src/lib.rs
  - ../sealmap/crates/sealmap-mermaid/src/escape.rs
  - ../sealmap/crates/sealmap-mermaid/src/writer.rs
  - ../sealmap/crates/sealmap-mermaid/src/sequence.rs
  - ../sealmap/crates/sealmap-mermaid/src/class.rs
  - ../sealmap/crates/sealmap-mermaid/src/er.rs
  - ../sealmap/crates/sealmap-mermaid/src/flowchart.rs
  - ../sealmap/crates/sealmap-mermaid/Cargo.toml
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
  - ../sealmap/crates/sealmap-mermaid/README.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

`sealmap-mermaid` is a safe Mermaid emitter that knows nothing about code
models: four typed builders (`SequenceDiagram`, `ClassDiagram`, `ErDiagram`,
`Flowchart`), one line writer, and two escaping functions that every value
passes through on its way to the output (`crates/sealmap-mermaid/src/lib.rs:3`-`25`).
With `default-features = false` it has no dependencies at all; the default
`model` feature adds the injective `Ident::from_symbol` drawn in SM-11
(`crates/sealmap-mermaid/Cargo.toml:17`-`20`).

The reason for builders over `format!` is Mermaid's grammar: `;` ends a
statement in a sequence message, `end` is a keyword in node ids, braces close
class bodies, `(` turns a class member into a method, angle brackets collide
with generics and `#` starts an entity code
(`crates/sealmap-mermaid/src/lib.rs:17`-`21`). This topic draws the writer
contract, the escaping rules and how each builder renders. It is the reason a
whole generated corpus parses: the mermaid crate README says every label and identifier goes through one
escaping layer so generated diagrams parse, whatever the source text contained
(`crates/sealmap-mermaid/README.md:10`-`11`); the diagram counts the root README once
quoted are no longer in it.

## For the business

Every diagram sealmap generates has to open in an ordinary Mermaid renderer,
or the corpus is a liability rather than a map. Source code is full of
characters that break Mermaid, and a generator that concatenates strings
breaks on them sooner or later. These writers make that structurally hard:
an identifier is cleaned, and every label is escaped with codes Mermaid turns
back into the original characters, so a reader still sees `Vec<String>`.

The writers are also deterministic and minimal: same calls, same bytes, no
styling. That matters for cost, because diagrams are read by models and every
decorative token is paid for. The crate can be adopted on its own, with no
dependencies, by anyone who needs safe Mermaid output.

## SM-10.1 The writers and what they share

```mermaid
classDiagram
    direction LR
    class CodeWriter {
        buf depth  writer.rs:19
        +line(text)  writer.rs:32
        +indented(f)  writer.rs:51
        +finish() one trailing newline  writer.rs:64
    }
    class Ident {
        String, letters digits underscore  escape.rs:58
        +new(raw) lossy  escape.rs:63
    }
    class SequenceDiagram {
        participants body  sequence.rs:205
        +render()  sequence.rs:254
    }
    class SeqBuilder {
        items  sequence.rs:80
        +message +note +block +arms  sequence.rs:86
    }
    class ClassDiagram {
        classes relations  class.rs:182
        +render()  class.rs:232
    }
    class ErDiagram {
        entities rels  er.rs:84
        +render()  er.rs:145
    }
    class Flowchart {
        direction body  flowchart.rs:97
        +render()  flowchart.rs:154
    }
    SequenceDiagram *-- SeqBuilder : derefs to
    SequenceDiagram ..> CodeWriter
    ClassDiagram ..> CodeWriter
    ErDiagram ..> CodeWriter
    Flowchart ..> CodeWriter
    SeqBuilder ..> Ident
    ClassDiagram ..> Ident
    ErDiagram ..> Ident
    Flowchart ..> Ident
```

**What it shows.** Each builder collects typed statements and renders through
one `CodeWriter`; every id is an `Ident`; a `SequenceDiagram` dereferences to
its `SeqBuilder`, so fragments can be built separately and spliced in.

**Why it is this way.** Rendering is a pure function of the builder; statement
order is preserved because in a sequence order is meaning, and nothing is
sorted behind the caller's back (`crates/sealmap-mermaid/src/lib.rs:29`-`32`).
The writer itself is a reduced port of ts-morph's `CodeBlockWriter`
(`crates/sealmap-mermaid/src/writer.rs:1`-`2`).

**Invariant:** output always uses two-space indentation, `\n` endings, no
trailing whitespace and exactly one trailing newline
(`crates/sealmap-mermaid/src/writer.rs:4`-`5`, enforced at
`crates/sealmap-mermaid/src/writer.rs:33` and
`crates/sealmap-mermaid/src/writer.rs:65`-`70`).

## SM-10.2 Escaping a label

```mermaid
flowchart TB
    T["free text from source"]
    E["escape_text<br/>escape.rs:102"]
    H["hash becomes entity 35<br/>escape.rs:107"]
    S["semicolon becomes entity 59<br/>escape.rs:108"]
    P["percent becomes entity 37<br/>escape.rs:109"]
    B["backtick becomes entity 96<br/>escape.rs:110"]
    Q["double quote becomes quot<br/>escape.rs:111"]
    W["whitespace runs squeezed,<br/>ends trimmed<br/>escape.rs:112"]
    TY["escape_type adds angle brackets,<br/>braces, and parens for fields<br/>escape.rs:144"]
    AR["arrow kept as written<br/>escape.rs:145"]
    T --> E
    E --> H
    E --> S
    E --> P
    E --> B
    E --> Q
    E --> W
    E --> TY --> AR
```

**What it shows.** Labels are escaped with Mermaid's own entity codes, which
render back to the original characters; type text in class bodies also escapes
angle brackets, braces and, in fields, parentheses.

**Why it is this way.** Each rule closes one break: a backtick can never end a
Markdown fence and a percent can never start a `%%` comment
(`crates/sealmap-mermaid/src/escape.rs:93`-`95`); a brace would close a class
body and a parenthesis would turn a field into a method
(`crates/sealmap-mermaid/src/escape.rs:131`-`136`). Generics use entity codes
because Mermaid's `~` generic marker cannot nest reliably
(`crates/sealmap-mermaid/src/escape.rs:133`-`134`).

**Invariant:** a label is single-line by construction: newlines and tabs
collapse to one space (`crates/sealmap-mermaid/src/escape.rs:112`-`118`).

## SM-10.3 Building and rendering a sequence

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant SD as SequenceDiagram<br/>sequence.rs:205
    participant SB as SeqBuilder<br/>sequence.rs:80
    participant CW as CodeWriter<br/>writer.rs:19
    CL->>SD: participant id and alias (sequence.rs:232)
    SD->>SD: declared once, first alias wins (sequence.rs:242)
    CL->>SB: message, text escaped on entry (sequence.rs:87)
    CL->>SB: block with a closure building its body (sequence.rs:100)
    SB->>SB: inner builder, label escaped (sequence.rs:103)
    CL->>SD: render (sequence.rs:254)
    SD->>CW: header, title, autonumber, participants (sequence.rs:256)
    SD->>SB: write items recursively (sequence.rs:272)
    SB->>CW: arms opened with loop or alt, next arms else or and, end (sequence.rs:187)
    CW-->>CL: text with one trailing newline (sequence.rs:274)
```

**What it shows.** Text is escaped when it enters the builder, not when it
leaves, so the stored items are already safe; blocks nest by building an inner
`SeqBuilder` in a closure.

**Why it is this way.** Escaping on entry means `extend` can splice a
separately built fragment in without re-escaping it
(`crates/sealmap-mermaid/src/sequence.rs:150`-`153`), which is how the corpus
builds `alt` and `loop` bodies.

**Debt:** a single-arm block kind given extra arms renders them as `else`
rather than refusing (`crates/sealmap-mermaid/src/sequence.rs:61`-`63`), so a
caller's misuse produces a diagram that parses but says something other than
what was built.

## SM-10.4 Class, ER and flowchart rendering rules

```mermaid
flowchart TB
    subgraph CLS["ClassDiagram, class.rs:232"]
        C1["label with generics as entity codes<br/>class.rs:53"]
        C2["relation arrows drawn so the head<br/>sits on the conceptual target<br/>class.rs:244"]
        C3["duplicate classes and relations ignored<br/>class.rs:196"]
    end
    subgraph ER["ErDiagram, er.rs:145"]
        E1["attribute type squeezed to Mermaid's<br/>charset, original kept as a comment<br/>er.rs:112-114"]
        E2["identifying relations solid,<br/>others dotted<br/>er.rs:176"]
    end
    subgraph FC["Flowchart, flowchart.rs:154"]
        F1["every node label quoted<br/>flowchart.rs:40"]
        F2["edge labels quoted<br/>flowchart.rs:163"]
        F3["subgraphs with optional direction<br/>flowchart.rs:167"]
    end
    CLS --> ER --> FC
```

**What it shows.** Each builder has its own guard against its diagram kind's
grammar: class labels move generics to entity codes, ER attribute types are
squeezed into the narrow charset ER accepts with the readable original kept
as a comment, flowchart labels are always quoted.

**Why it is this way.** ER attribute types cannot hold `<` or spaces, so a
lossless form is computed and compared; the comment appears only when the
squeezed form lost something (`crates/sealmap-mermaid/src/er.rs:112`-`114`).

**Debt:** `Class::raw_member` takes a member line with only braces and
newlines removed (`crates/sealmap-mermaid/src/class.rs:96`-`99`), so it is the
one public path to class-diagram output that does not go through
`escape_type`, against the crate's claim that every value goes through `Ident`
or `escape_text` (`crates/sealmap-mermaid/src/lib.rs:22`-`25`).

## SM-10.5 Identifiers from arbitrary text

```mermaid
flowchart TB
    R["raw text"]
    M["every char outside letters, digits<br/>and underscore becomes underscore<br/>escape.rs:64"]
    D{"empty or starts with a digit?<br/>escape.rs:65"}
    U["underscore prefixed<br/>escape.rs:66"]
    K{"a Mermaid keyword, any case?<br/>escape.rs:68"}
    S["underscore suffixed<br/>escape.rs:69"]
    OUT["Ident"]
    R --> M --> D
    D -->|yes| U --> K
    D -->|no| K
    K -->|yes| S --> OUT
    K -->|no| OUT
```

**What it shows.** `Ident::new` sanitises any text into a valid, keyword-safe
id, for ids a caller mints itself.

**Why it is this way.** The keyword list covers words that are reserved in at
least one diagram kind, from `end` and `graph` to `autonumber` and `default`
(`crates/sealmap-mermaid/src/escape.rs:3`-`38`), so one id is safe in every
kind.

**Invariant:** `Ident::new` is documented as lossy; only `Ident::from_symbol`
is injective (`crates/sealmap-mermaid/src/escape.rs:43`-`48`, SM-11).
