---
id: SM-12
title: Generating the 1:1 corpus and its index
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-corpus/src/lib.rs
  - ../sealmap/crates/sealmap-corpus/src/document.rs
  - ../sealmap/crates/sealmap-corpus/src/index.rs
  - ../sealmap/crates/sealmap-corpus/src/corpus_readme.md
  - ../sealmap/crates/sealmap-corpus/tests/contract.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

`sealmap_corpus::generate` projects a `Codebase` into an in-memory map from
output path to text: exactly one Markdown document per source file, plus four
reserved files whose names start with `_` and so never collide with a source
(`crates/sealmap-corpus/src/lib.rs:12`-`58`). It is pure: equal inputs give
byte-identical output (`crates/sealmap-corpus/src/lib.rs:192`-`193`), and every
`CorpusOptions` field is part of that identity, so changing one is reported as
drift by `verify` (`crates/sealmap-corpus/src/lib.rs:133`-`135`).

This topic covers the orchestration, the anatomy of a document, and the
`_index.json` merge map that links every call to the fragment that expands it.
The diagrams inside a document are SM-13; checking and writing a directory is
SM-14. Under the accepted design this 1:1 corpus stopped being a committed,
CI-gated artefact and is an optional view under a gitignored `.sealmap/`
(`docs/DESIGN.md:105`, `docs/DESIGN.md:108`-`112`); the code that produces it was
unchanged by that decision.

## For the business

The generated corpus is the mechanical half of the workflow: it is what an
agent reads when it needs the facts about a file without opening the code, and
what an orchestrating agent stitches together to follow a request across
files. Its value to an adopter is that it costs no model tokens to produce and
is identical every time, so it can be regenerated on demand rather than stored
and maintained.

The caution is a size one: an earlier README recorded that a one-document-per-file
Mermaid corpus is about the same size as the source (1.07 times), and the current
README keeps only the dense-view result of 0.17 to 0.29 times (`README.md:355`), so the corpus is not a
compression and not what people should read. Its job is to be a reliable,
linkable substrate, not the human map.

## SM-12.1 One call to generate

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant GE as generate<br/>sealmap-corpus/src/lib.rs:194
    participant LK as Lookup.new<br/>sealmap-corpus/src/lib.rs:225
    participant DR as document render<br/>document.rs:14
    participant IX as Index<br/>index.rs:101
    CL->>GE: Codebase and CorpusOptions
    GE->>GE: assert every diagram id unique (sealmap-corpus/src/lib.rs:196)
    GE->>IX: new, with stats and generator version (sealmap-corpus/src/lib.rs:197)
    GE->>LK: per-file and per-parent symbol lists (sealmap-corpus/src/lib.rs:198)
    loop each source file in path order
        GE->>DR: file, lookup, options (sealmap-corpus/src/lib.rs:200)
        DR-->>GE: document path, text, DocumentEntry
    end
    GE->>IX: link calls to expanding fragments (sealmap-corpus/src/lib.rs:204)
    GE->>GE: _overview.md, _README.md, _model.json, _index.json (sealmap-corpus/src/lib.rs:206)
    GE-->>CL: Corpus of files and index (sealmap-corpus/src/lib.rs:212)
```

**What it shows.** Generation is a single pass over files in path order, with
the cross-document work (id uniqueness up front, expansion links and the
corpus-level files at the end) around it.

**Why it is this way.** `_model.json` is optional (`emit_model`), the others
are always written (`crates/sealmap-corpus/src/lib.rs:208`-`211`). JSON is
compact by default because it is roughly half the size of pretty output
(`crates/sealmap-corpus/src/lib.rs:156`-`158`).

**Debt:** the index records its generator as `sealmap-corpus` plus the crate
version (`crates/sealmap-corpus/src/index.rs:104`), so a version bump with no
other change rewrites `_index.json`, and a kept copy reports drift under
`generate --check` on every release.

## SM-12.2 Anatomy of a document

```mermaid
flowchart TB
    FM["front matter: sealmap schema 2, source, module<br/>quoted, language, source_hash, lines, fragments<br/>document.rs:101-110"]
    H1["heading: module sym id and source path<br/>document.rs:25"]
    DOC["module doc summary, if any<br/>document.rs:26"]
    PE["parse warning when the module is<br/>tagged parse_error<br/>document.rs:29"]
    ST["structure section: one classDiagram<br/>document.rs:39"]
    SQ["one section per callable with a flow,<br/>in source order, headed by its sym id<br/>document.rs:58-61"]
    META["signature and span line, doc summary<br/>document.rs:63-70"]
    FM --> H1 --> DOC --> PE --> ST --> SQ --> META
```

**What it shows.** A document is front matter, a heading naming the module by
its `sym:` id, an optional structure diagram and one sequence diagram per
callable that makes calls.

**Why it is this way.** Callables are ordered by span rather than id because
source order reads better inside one file (`crates/sealmap-corpus/src/document.rs:57`).
The module id is a JSON-quoted YAML scalar, so ids holding `: ` or ` #` cannot
be misread as structure or comments (`crates/sealmap-corpus/src/document.rs:124`-`128`),
and headings use a longer code fence when an id holds backticks
(`crates/sealmap-corpus/src/document.rs:130`-`134`).

**Invariant:** every generated document opens with `---` front matter whose
first key is the `sealmap: ` marker, and `write` only ever deletes files
whose front matter starts that way
(`crates/sealmap-corpus/src/document.rs:10`-`12`).

## SM-12.3 The index as types

```mermaid
classDiagram
    direction LR
    class Index {
        schema_version generator codebase stats  index.rs:87
        documents  index.rs:97
        +fragment(id) linear search  index.rs:133
    }
    class DocumentEntry {
        source document module  index.rs:70
        source_hash document_hash  index.rs:78
        fragments  index.rs:82
    }
    class FragmentEntry {
        id kind document symbol span  index.rs:36
        sig_hash body_hash  index.rs:49
        participants calls  index.rs:53
        truncated hash  index.rs:59
    }
    class CallRef {
        target confidence line  index.rs:20
        expands Option  index.rs:31
    }
    class FragmentKind {
        <<enum>>
        Structure Sequence  index.rs:11
    }
    Index *-- DocumentEntry
    DocumentEntry *-- FragmentEntry
    FragmentEntry *-- CallRef
    FragmentEntry --> FragmentKind
```

**What it shows.** The index lists every document and fragment with its
participants, its calls in source order, the symbol's fingerprints and a hash
of the fragment's own Mermaid text.

**Why it is this way.** A fragment's id is its symbol's `sym:` id for a
sequence and `structure:<file>` for a structure diagram, unique across the
corpus (`crates/sealmap-corpus/src/index.rs:37`-`38`); the fragment hash exists
so an orchestrator can cache merged results (`crates/sealmap-corpus/src/index.rs:60`).
Fingerprints were added to fragments with schema v2
(`crates/sealmap-corpus/src/lib.rs:117`).

**Invariant:** `_index.json` and `_model.json` carry ids, spans and
fingerprints in schema v2, pinned by
`schema_v2_json_carries_ids_spans_and_fingerprints`
(`crates/sealmap-corpus/tests/contract.rs:190`).

## SM-12.4 Linking calls to the fragments that expand them

```mermaid
sequenceDiagram
    autonumber
    participant GE as generate<br/>sealmap-corpus/src/lib.rs:194
    participant LE as link_expansions<br/>index.rs:112
    participant OR as orchestrating agent
    GE->>LE: after every document is rendered (sealmap-corpus/src/lib.rs:204)
    LE->>LE: collect ids of all sequence fragments (index.rs:113)
    loop every call of every fragment
        LE->>LE: target has a sequence fragment, set expands (index.rs:119)
    end
    LE-->>GE: index with merge links
    OR->>OR: read _index.json, follow expands to inline a callee
    Note over OR: the merge map is how separately rendered<br/>sequences are stitched across files, corpus_readme.md:54
```

**What it shows.** Once all fragments exist, each call whose target has a
sequence of its own gets an `expands` link to it, which is the whole merge map.

**Why it is this way.** Because a fragment id is the callee's `sym:` id, the
link is a lookup, not a guess (`crates/sealmap-corpus/src/lib.rs:50`-`53`);
`index_links_calls_to_expanding_fragments` pins it
(`crates/sealmap-corpus/tests/contract.rs:79`).

## SM-12.5 Per-file lookups built once

```mermaid
flowchart TB
    BEFORE["per document: Codebase.symbols_in_file and<br/>Codebase.children scan every symbol,<br/>O of files times symbols<br/>sealmap-corpus/src/lib.rs:217-218"]
    LK["Lookup.new: one pass over symbols in id<br/>order, by file and by parent<br/>sealmap-corpus/src/lib.rs:230"]
    IF["in_file: same items and order as<br/>symbols_in_file<br/>sealmap-corpus/src/lib.rs:239-241"]
    CH["children: same items and order as<br/>Codebase.children<br/>sealmap-corpus/src/lib.rs:244-246"]
    BEFORE --> LK
    LK --> IF
    LK --> CH
```

**What it shows.** The corpus builds its own per-file and per-parent indexes
once per generation instead of calling the model's linear scans per document.

**Why it is this way.** It was part of the hardening (commit `8af26b3`); the
lists keep the order the linear scans produced, so output did not change
(`crates/sealmap-corpus/src/lib.rs:228`-`229`).

**Invariant:** generation is byte-identical across runs and regardless of the
order sources were inserted (`crates/sealmap-corpus/tests/contract.rs:93`).
