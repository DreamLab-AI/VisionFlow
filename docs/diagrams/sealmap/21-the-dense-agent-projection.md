---
id: SM-21
title: The dense agent projection
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-dense/src/lib.rs
  - ../sealmap/crates/sealmap-dense/src/short.rs
  - ../sealmap/crates/sealmap-dense/src/tree.rs
  - ../sealmap/crates/sealmap-dense/src/skeleton.rs
  - ../sealmap/crates/sealmap-dense/tests/dense.rs
  - ../sealmap/crates/sealmap/src/main.rs
  - ../sealmap/crates/sealmap-rust/src/resolve.rs
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-rust/src/layout.rs
  - ../sealmap/crates/sealmap-rust/tests/resolver_den01.rs
  - ../sealmap/crates/sealmap-corpus/src/pack.rs
  - ../sealmap/crates/sealmap-model/src/sym.rs
  - ../sealmap/crates/sealmap-corpus/src/structure.rs
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

`sealmap-dense` turns a `Codebase` into the text an agent reads instead of
source or Mermaid. It depends only on `sealmap-model`. A `Dense` value is
built once, assigning short names and reading the call graph from the flows
(`crates/sealmap-dense/src/lib.rs:287`). Any number of renders and slices can
then share it. A render is two texts: `dense.txt` holds skeletons and call
trees, and `_index.txt` maps each short name to its `sym:` id and location
(`crates/sealmap-dense/src/lib.rs:144`-`147`). A slice is the bounded excerpt
that `pack` embeds, one per topic (`crates/sealmap-dense/src/lib.rs:347`, SM-17).

Three rules carry the design:

- **Each callable is expanded once per output.** Every later call to it is a
  one-line reference. Every call site appears exactly once, and a test pins
  this at every depth (`crates/sealmap-dense/tests/dense.rs:255`).
- **Short names come from the ids alone.** When names collide, every member of
  the colliding group is qualified together
  (`crates/sealmap-dense/src/short.rs:46`-`55`).
- **A slice over its byte budget is refused, never truncated**
  (`crates/sealmap-dense/src/lib.rs:368`-`373`).

Skeleton lines are grouped by file and sorted by position, with the id as
the tie-break, so the order is total and reads like the source
(`crates/sealmap-dense/src/skeleton.rs:10`-`14`). Each line writes the
signature once, with fields, variants and implemented traits inline
(`crates/sealmap-dense/src/skeleton.rs:40`).

The research behind the crate estimated 0.45× source. As built, `dense.txt`
measures 0.17–0.29× (`docs/DESIGN.md:135`, `docs/DESIGN.md:139`).

On the command line, `sealmap dense [PATH] [-o DIR] [--depth N] [--stats]`
writes both files, by default into `.sealmap/dense`
(`crates/sealmap/src/main.rs:51`, `crates/sealmap/src/main.rs:341`-`348`).
It reads the code with the same `--repo`, `--name` and `--tests` options as
the seal commands, so its `_index.txt` names the ids a seal would. Until the
seal surface merged, the hook was an example binary; the subcommand replaced
it and the example is gone.

This topic also records what reading the projection of sealmap itself
revealed about the Rust adapter (SM-21.6).

## For the business

An agent that loads the source of a codebase to learn its shape pays for
every comment, every formatting choice and every repeated name. The one-file
Mermaid corpus is no cheaper: it costs 0.48–0.88× the source. The dense
projection carries the same symbols, signatures, line spans and resolved
calls at 0.17–0.29× (`README.md:60`). A fixed context budget therefore holds
roughly three times more of the call graph.

Every call is marked exact, inferred or external, so an agent can tell a
proven edge from a guess. The slice gives a review pack a hard size ceiling.
If the request does not fit, the agent gets an error that names the
overflow. It never receives a silently shortened excerpt that looks complete.

## SM-21.1 From a model to two texts and a slice

```mermaid
flowchart TB
    CB["Codebase<br/>symbols, flows, relations"]
    NEW["Dense.new, lib.rs:287"]
    SN["ShortNames.new<br/>short.rs:36"]
    GR["Graph.new: flows, callees, callers<br/>tree.rs:30"]
    TX["text: header, # symbols, # calls<br/>lib.rs:316"]
    IX["index: one line per symbol<br/>lib.rs:307"]
    SL["slice: seeds plus callers and callees<br/>lib.rs:347"]
    OUT["dense.txt and _index.txt<br/>lib.rs:144-147"]
    PK["pack: one slice per topic<br/>pack.rs:480"]
    CB --> NEW
    NEW --> SN
    NEW --> GR
    SN --> TX
    GR --> TX
    SN --> IX
    TX --> OUT
    IX --> OUT
    GR --> SL
    SN --> SL
    SL --> PK
```

**What it shows.** The two expensive steps, naming and the call graph, run
once. Rendering and slicing only read their results.

**Why it is this way.** `pack` takes many slices of one codebase per
request. It builds one `Dense` and slices it once per topic
(`crates/sealmap-corpus/src/pack.rs:438`), and the one-call `slice` function
says to do so (`crates/sealmap-dense/src/lib.rs:473`-`474`).

**Invariant:** the call graph counts a call only when its target is a symbol
of the codebase, and a callable's own recursion is not counted as a caller,
so a self-recursive function can still be an entry point
(`crates/sealmap-dense/src/tree.rs:38`-`42`).

## SM-21.2 One call line and its mark

```mermaid
flowchart TB
    C["a call step<br/>TreeWriter.call, tree.rs:160"]
    T{"target is a symbol<br/>with a flow?<br/>tree.rs:163-164"}
    L["plain line: leaf or<br/>non-internal target"]
    P{"on the current path?<br/>tree.rs:170"}
    CY["ends in ↺<br/>tree.rs:171"]
    E{"expanded already?<br/>tree.rs:172"}
    RF["ends in ^<br/>tree.rs:173"]
    D{"depth at the limit?<br/>tree.rs:174"}
    CU["ends in …, queued for<br/>its own tree<br/>tree.rs:175-178"]
    X["expand: its steps indented<br/>one more space<br/>tree.rs:183-186"]
    C --> T
    T -->|no| L
    T -->|yes| P
    P -->|yes| CY
    P -->|no| E
    E -->|yes| RF
    E -->|no| D
    D -->|yes| CU
    D -->|no| X
```

**What it shows.** Every call line ends in at most one mark. A call is only
expanded if it is not on the current path, not already expanded and not at
the depth limit.

**Why it is this way.** Expanding a callable at every call site grows with
the number of paths through the graph. Expanding it once grows with the
number of call sites. The depth limit hardly matters for size (±3 % between
1 and 10 levels), so its default of 3 is chosen for reading, not for bytes
(`docs/DESIGN.md:139`).

**Invariant:** each callable with a flow is expanded exactly once, whatever
the depth (`crates/sealmap-dense/tests/dense.rs:266`).

## SM-21.3 The order of trees in dense.txt

```mermaid
sequenceDiagram
    autonumber
    participant TX as Dense.text<br/>lib.rs:316
    participant TW as TreeWriter.root<br/>tree.rs:92
    participant TR as TreeWriter.tree<br/>tree.rs:99
    TX->>TX: callables in source order (lib.rs:322)
    loop every callable nothing internal calls (lib.rs:324-327)
        TX->>TW: root, bare header
        TW->>TR: the entry point's tree
        loop every call cut at the depth limit (tree.rs:94)
            TW->>TR: its tree, header ending in … (tree.rs:95)
        end
    end
    loop every callable still not expanded (lib.rs:329-332)
        TX->>TW: root, header ending in ↺
    end
```

**What it shows.** Entry points come first, each followed at once by the
trees of the calls it cut. Callables reachable only through a cycle come
last.

**Why it is this way.** Reading sealmap's own projection showed the need for
the header marks. Without them, a cut continuation looked like an entry point
and suggested dead code that was not there. A bare header now means that
nothing in the codebase calls the function (`crates/sealmap-dense/src/lib.rs:150`).

**Invariant:** a pure cycle with no entry point still gets a tree
(`crates/sealmap-dense/tests/dense.rs:455`).

## SM-21.4 Short names and collisions

```mermaid
flowchart LR
    subgraph ladder["each symbol climbs only when it must, candidate, short.rs:119-124"]
        direction TB
        L0["level 0: owner and name<br/>Db.put, connect, db/"]
        L1["level 1: plus the trait-impl block<br/>Db[Store].put"]
        L2["level 2: plus the module path<br/>db/Db[Store].put"]
        L3["level 3: plus the package<br/>shop:db/Db[Store].put"]
        L4["levels 4 and 5: plus 8 hex digits,<br/>then the whole BLAKE3 of the id<br/>short.rs:133"]
        L0 --> L1 --> L2 --> L3 --> L4
    end
    G{"any name shared?<br/>short.rs:41"}
    B["every member of each group<br/>moves to its next distinct level<br/>short.rs:46-55"]
    U["unique; whitespace removed<br/>short.rs:129"]
    ladder --> G
    G -->|yes| B
    B --> G
    G -->|no| U
```

**What it shows.** A name starts as short as possible and gains one
qualifier only when another symbol shares it. Both symbols move up
together.

**Why it is this way.** Moving every member keeps a name a function of the
set of ids, not of the order in which symbols were found. A test feeds the
same model in reverse and gets the same index
(`crates/sealmap-dense/tests/dense.rs:444`). The levels follow the `sym:`
grammar's own structure, so a short name reads as a fragment of the id it
stands for.

**Invariant:** short names are unique and resolve back to their ids, even
under deliberate collisions: same names across modules, two `From` impls,
two packages, and ids that differ only in a disambiguator
(`crates/sealmap-dense/tests/dense.rs:413`). If BLAKE3 ever collided, the
loop would still number the members (`crates/sealmap-dense/src/short.rs:57`-`64`).

**Tension (short names vs stability):** a new symbol whose name collides
re-qualifies an existing symbol's short name. Short names are stable for one
output, not across commits. Seals and citations must keep using `sym:` ids,
which `_index.txt` maps to (`crates/sealmap-dense/src/lib.rs:307`).

## SM-21.5 A slice, and refusing one

```mermaid
sequenceDiagram
    autonumber
    participant PK as caller, e.g. pack
    participant SL as Dense.slice<br/>lib.rs:347
    participant ST as slice_text<br/>lib.rs:379
    PK->>SL: seed ids, depth, max_bytes
    SL->>SL: split seeds into known and unknown (lib.rs:353-362)
    alt any unknown
        SL-->>PK: UnknownSymbols, in id order (lib.rs:364)
    else all known
        SL->>ST: the seed set
        ST-->>SL: symbols, calls, callers, index sections
        alt longer than max_bytes (lib.rs:368)
            loop each seed (lib.rs:369-371)
                SL->>ST: that seed alone
            end
            SL-->>PK: OverBudget with bytes, budget, per-seed sizes (lib.rs:373)
        else fits
            SL-->>PK: the text, unchanged
        end
    end
```

**What it shows.** A slice either fits whole or is refused. The refusal
carries what a caller needs to split the request into shards: the total size
and the size of each seed's slice alone.

**Why it is this way.** The design requires that `pack` never truncates
silently (`docs/DESIGN.md:161`). The slice's own index section keeps it
self-contained, so a review pack does not need `_index.txt` alongside.
Seeds are a set, so their order and any duplicates make no difference
(`crates/sealmap-dense/tests/dense.rs:240`).

**Invariant:** at exactly the budget, the slice is returned unchanged. One
byte under the budget, it is refused, with the overflow named
(`crates/sealmap-dense/tests/dense.rs:304`).

## SM-21.6 What reading sealmap's own projection revealed

```mermaid
flowchart TB
    R["dense.txt of sealmap<br/>read as an agent would"]
    S["Self::f calls dropped:<br/>SymbolId.global showed no calls"]
    SF["call paths starting Self or self<br/>resolve before the prelude check<br/>resolve.rs:851-854"]
    G["guard calls never walked:<br/>is_shared had no caller"]
    GF["the guard is walked after the<br/>arm's bindings, collect.rs:1228-1229"]
    L["struct-literal receivers lost:<br/>Parser.id under SymbolId.parse"]
    LF["a struct literal names its type<br/>collect.rs:1399"]
    N["a guessed edge vanished when<br/>another crate added root methods"]
    NF["guesses count only crates the caller<br/>can reach, resolve.rs:975-988<br/>layout.rs:75"]
    P["pack's own calls bound to<br/>the CLI's fn pack"]
    PF["a path prefix is a module or type<br/>resolve.rs:391, resolve.rs:666"]
    R --> S --> SF
    R --> G --> GF
    R --> L --> LF
    R --> N --> NF
    R --> P --> PF
```

**What it shows.** Five resolver and walker defects that the Mermaid corpus
hid and the dense trees exposed, each now fixed with a test written first
(`crates/sealmap-rust/tests/resolver_den01.rs:41`). The first four came from
one read of sealmap's own projection. The fifth came from reading the
projection of the `pack` code added afterwards.

**Why it is this way.** In the dense format a missing edge is visible. A
function whose body plainly calls something shows no calls, a private helper
appears as an entry point, or a call is marked `~` where the import makes it
certain. The Mermaid corpus spreads the same facts over one diagram per
function.

**Closed (`d227dd8`):** every `Self::f(..)` call was dropped. `"Self"` is on
the prelude list for type references (`crates/sealmap-rust/src/resolve.rs:60`),
and the call filter checked that list before resolving. A call path starting
`Self` now resolves to the enclosing impl's self type through the resolver's
`Self` arm (`crates/sealmap-rust/src/resolve.rs:678`), exactly as `Type::f(..)`
does. Outside any impl it names nothing and is dropped
(`crates/sealmap-rust/src/resolve.rs:851`-`854`). `self::f(..)` was dropped by
the same check and resolves too. `Self::from_repr` in `SymbolId::global`
(`crates/sealmap-model/src/sym.rs:406`) is now an exact edge. On VisionClaw
this alone adds 477 exact and 20 inferred edges.

**Closed (`d227dd8`):** calls in a match guard were only formatted into the
arm label. The guard is now walked once the arm's bindings are in scope, and
its calls open the arm (`crates/sealmap-rust/src/collect.rs:1228`-`1229`). The
guard calling `is_shared` (`crates/sealmap-corpus/src/structure.rs:194`) gives
it its caller.

**Closed (`d227dd8`):** a method called on a struct literal, such as
`Parser { .. }.id()` in `SymbolId::parse`
(`crates/sealmap-model/src/sym.rs:437`), was lost. The cause, traced:
`recv()` classed a struct literal as an unknown receiver, so the call resolved
to `sym:? id`, which the default external policy drops. A struct literal now
names its own type (`crates/sealmap-rust/src/collect.rs:1399`).

**Closed (`613b9ba`):** the by-name guess for an unknown receiver accepted a
method name only if it was unique in the whole workspace, so `TreeWriter::root`
and `CallerWriter::root` in sealmap-dense (`crates/sealmap-dense/src/tree.rs:92`,
`crates/sealmap-dense/src/tree.rs:244`) removed a correct guessed edge from
`Resolver` to `SymbolId::root` in sealmap-rust, a crate that cannot name
sealmap-dense. Candidates now come only from crates the caller can reach:
itself, its package's library, and the workspace packages its manifest lists
(`crates/sealmap-rust/src/layout.rs:75`, `crates/sealmap-rust/src/layout.rs:247`).
Two or more reachable candidates are a genuine ambiguity, and no edge is
guessed; the call stays `sym:? name` (`crates/sealmap-rust/src/resolve.rs:975`-`989`).
On VisionClaw this drops 62 guesses into crates the caller does not depend on,
such as std's `as_secs()` bound to another crate's `Timestamp::as_secs`.

**Closed (`926144c`):** the CLI's `pack::shard(..)` was drawn as an inferred
call to `sealmap_main::pack::shard`. The local `fn pack` captured the
imported module `pack`, because a leading path segment fell back from the
type namespace to values. A segment with more after it now consults the type
namespace alone (`crates/sealmap-rust/src/resolve.rs:391`,
`crates/sealmap-rust/src/resolve.rs:666`).
