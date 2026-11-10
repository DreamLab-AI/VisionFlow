---
id: SM-05
title: Minting ids for Rust definitions
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-extract/src/ids.rs
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-rust/src/resolve.rs
  - ../sealmap/crates/sealmap-rust/src/lib.rs
  - ../sealmap/crates/sealmap-model/src/symbol.rs
  - ../sealmap/crates/sealmap-rust/tests/resolver_er.rs
  - ../sealmap/README.md
  - ../sealmap/crates/sealmap-rust/README.md
  - ../sealmap/crates/sealmap-corpus/src/corpus_readme.md
verified_commit: 2b351367cab241aa7304d364aa37b52e65bb71f2
---
## For developers

The grammar in SM-01 says what an id may look like; this topic says which id
each Rust definition actually gets. Every id an adapter mints goes through
`sealmap_extract::ids` (`crates/sealmap-extract/src/ids.rs:3`-`4`), so two
adapters can only ever disagree about what they collect, never about how an
id is spelled. The Rust collector decides which items become symbols, and the
resolver decides which type a method belongs to.

The rule that matters most is that a method sits under the type that owns
it, never under its `impl` block (`crates/sealmap-extract/src/ids.rs:20`-`28`):
splitting an `impl` or moving it to another file keeps every method id. The
generated corpus README promises the same thing to users
(`crates/sealmap-corpus/src/corpus_readme.md:34`-`38`). This came
with the grammar in commit `5973a4e`; the undefined-member rule at the end of
this topic came with the dogfood fix in `6c8a9b0`.

## For the business

A seal is a promise about named functions, so the names have to stay put when
the code is merely tidied. This topic is where that is decided for Rust.
Reorganising methods across `impl` blocks or files, the most common
refactor in a growing Rust codebase, changes no name and therefore invalidates
no reviewed diagram. A real rename, or moving a function to another module,
does change the name, which is what a reviewer would expect to look at again.

One limit for adopters to know: items that a macro generates are not seen,
so they get no name, and the README warns that macro-generated code may
remain unresolved (`sealmap/README.md:344`-`346`).

## SM-05.1 The id table

```mermaid
flowchart TB
    PKG["package root<br/>sym:cargo app .<br/>ids.rs:8"]
    MOD["module<br/>sym:cargo app . store/<br/>module_id, ids.rs:68"]
    ITEM["type, fn, const, macro in a scope<br/>item_id, ids.rs:75"]
    METH["inherent method under its type<br/>method_id, ids.rs:82"]
    TRAIT["trait-impl method, Trait descriptor<br/>spelled as written, ids.rs:84"]
    FOREIGN["impl on a type outside the code<br/>module/impl#35;[SelfTy][Trait]m().<br/>impl_method_id, ids.rs:105"]
    EXT["path with unknown kinds<br/>path_id, ids.rs:116"]
    UNR["method on an unknown receiver<br/>unresolved_method_id, ids.rs:122"]
    PKG --> MOD --> ITEM --> METH --> TRAIT
    METH --> FOREIGN
    ITEM -.->|scope not global, ids.rs:76| EXT
    METH -.->|unknown receiver| UNR
```

**What it shows.** Each kind of definition and reference has one builder.
A trait-impl method adds a `[Trait]` descriptor before the method name; an
impl on a type the code does not define is anchored in the module holding
the impl, after rust-analyzer's `impl#[SelfType][Trait]` form
(`crates/sealmap-extract/src/ids.rs:25`-`28`).

**Why it is this way.** The trait is spelled as the impl writes it, generic
arguments kept, so `impl From<A>` and `impl From<B>` give two ids
(`crates/sealmap-extract/src/ids.rs:23`-`25`). A scope that is not a global id
degrades to a path id instead of failing (`crates/sealmap-extract/src/ids.rs:76`).

**Invariant:** building an id never fails: a package name the grammar cannot
hold is repaired to `_`, and an invalid manager falls back to `unknown _ .`
(`crates/sealmap-extract/src/ids.rs:57`-`63`).

## SM-05.2 Which Rust items become symbols

```mermaid
flowchart TB
    IT["an item in a module<br/>Collector.item, collect.rs:191"]
    MOD["inline mod: a module of its own<br/>collect.rs:212"]
    DATA["struct, enum, union<br/>collect.rs:226"]
    TR["trait, with provided methods<br/>collect.rs:275"]
    AL["type alias<br/>collect.rs:360"]
    FN["free function with a flow<br/>collect.rs:377"]
    VAL["const, static<br/>collect.rs:394"]
    MAC{"macro_rules with a name?<br/>collect.rs:410"}
    MDEF["macro symbol, public if macro_export<br/>collect.rs:411"]
    MINV["item-level invocation: folded into<br/>the module body hash only<br/>collect.rs:428"]
    IMPL["impl block: methods kept as RawImpl<br/>collect.rs:431"]
    REST["use, extern crate, foreign blocks:<br/>module body only<br/>collect.rs:484"]
    IT --> MOD
    IT --> DATA
    IT --> TR
    IT --> AL
    IT --> FN
    IT --> VAL
    IT --> MAC
    MAC -->|yes| MDEF
    MAC -->|no| MINV
    IT --> IMPL
    IT --> REST
```

**What it shows.** Modules, data types, traits, aliases, functions, constants,
statics and named macros become symbols; methods arrive through impl blocks
and are given ids in the resolve pass; everything else only feeds the
module's body fingerprint.

**Why it is this way.** An item-level macro invocation such as
`thread_local!` has no name to cite, so the only honest thing to record is
that the module changed when it changes (`crates/sealmap-rust/src/collect.rs:427`-`428`).

**Debt:** the test filter (`self.skip`) is applied to modules, structs,
enums, traits, functions and impls, but not to unions, type aliases,
constants, statics or macros (`crates/sealmap-rust/src/collect.rs:237`,
`crates/sealmap-rust/src/collect.rs:360`,
`crates/sealmap-rust/src/collect.rs:394`); a top-level `#[cfg(test)]` const
is modelled even when tests are excluded.

## SM-05.3 A method's owner is resolved, then the id is built

```mermaid
sequenceDiagram
    autonumber
    participant BU as build<br/>resolve.rs:213
    participant RS as Resolver.resolve<br/>resolve.rs:605
    participant IM as impl_method<br/>resolve.rs:335
    participant ID as impl_method_id<br/>ids.rs:95
    participant MS as method_symbol<br/>resolve.rs:810
    BU->>RS: self type of the impl as written (resolve.rs:299)
    RS-->>BU: owner id, global if internal, or for a type<br/>that is not a path its own text (resolve.rs:597)
    BU->>IM: module, owner, impl, method name (resolve.rs:308)
    IM->>ID: self type text, trait display text (resolve.rs:338)
    alt owner is a global id
        ID-->>IM: method under the owner type (ids.rs:103)
    else owner outside the code
        ID-->>IM: module/impl anchor (ids.rs:110)
    end
    BU->>MS: id, parent is the owner (resolve.rs:309)
    MS-->>BU: Symbol with parent set to the owner (resolve.rs:821)
    BU->>BU: trait impls tag the method impl Trait (resolve.rs:311)
```

**What it shows.** The impl's self type is resolved through imports and
modules first; only then is the method id built, so several inherent impls
of one type, in any files, produce methods under one owner. A self type that
is not a path (a tuple, slice, array, function pointer or trait object) has
nothing to resolve: it becomes the path id of the type as written, and its
methods are anchored under ``impl#[`(u8, u8)`]`` in the impl's module, so
`[u8]` and `&[u8]` stay apart (`crates/sealmap-rust/src/resolve.rs:594`-`598`,
`crates/sealmap-rust/tests/resolver_er.rs:130`).

**Why it is this way.** Inherent impls are registered before trait impls so
that an inherent method wins a name lookup over a trait method of the same
name (`crates/sealmap-rust/src/resolve.rs:556`-`558`).

**Invariant:** a trait-impl method is recorded as public whatever its
written visibility, because trait methods are as visible as the trait
(`crates/sealmap-rust/src/collect.rs:458`).

## SM-05.4 Formatting never reaches an id

```mermaid
flowchart TB
    W["impl Handler of Msg with a trailing comma<br/>as rustfmt writes a broken list"]
    TT["trait_text<br/>collect.rs:494"]
    TK["tokens printed and squeezed<br/>collect.rs:495"]
    TC["trailing comma before a closing<br/>angle or square bracket dropped<br/>collect.rs:496"]
    KEEP["comma before a closing paren kept:<br/>a one-tuple is a different type<br/>collect.rs:493"]
    ID["the same Trait descriptor text<br/>either way"]
    W --> TT --> TK --> TC --> ID
    TC -.-> KEEP
```

**What it shows.** The trait part of a method id is printed from tokens, with
the trailing commas rustfmt adds to a broken generic list removed.

**Why it is this way.** Ids must not change when rustfmt reflows a line; the
extractor README's claim that ids and fingerprints ignore comments,
whitespace, rustfmt rewrites and moves
(`crates/sealmap-rust/README.md:17`-`19`) depends on this rule
(`crates/sealmap-rust/src/collect.rs:489`-`493`).

**Invariant:** `(T,)` and `(T)` stay distinct in a trait's text, because only
commas before `>` and `]` are dropped (`crates/sealmap-rust/src/collect.rs:496`).

## SM-05.5 A member the code does not define stays with its type

```mermaid
flowchart TB
    P["path Fp::default as written"]
    WK["walk the path segment by segment<br/>Resolver.walk, resolve.rs:650"]
    KN{"segment defined on the<br/>current type or module?<br/>resolve.rs:709"}
    LAST{"last segment, current is an<br/>internal type?<br/>resolve.rs:716"}
    VAL["value position: Type method id<br/>resolve.rs:726"]
    TY["type position: Type member id<br/>resolve.rs:727"]
    EP["otherwise: extend as a path id<br/>resolve.rs:730"]
    NEXT["continue from the definition"]
    P --> WK --> KN
    KN -->|yes| NEXT
    KN -->|no| LAST
    LAST -->|yes| VAL
    LAST -->|yes, type ns| TY
    LAST -->|no| EP
```

**What it shows.** `Fingerprint::default()` from a derive has no definition
in the code, but it still belongs to `Fingerprint`, so it becomes
`Fingerprint#default().` rather than a kind-free `sym:extern` path.

**Why it is this way.** Before `6c8a9b0` such calls were cut loose from their
type and drawn on an external lane in the self-corpus; the fix keeps them on
the type's lane (`crates/sealmap-rust/src/resolve.rs:720`-`724`).

**Debt:** such an id names a symbol the model does not contain, so it is not
internal and resolves with `Inferred` confidence inside the workspace or
`External` outside it (`crates/sealmap-rust/src/resolve.rs:614`-`616`); a seal
could cite an id that `Codebase::symbol` never returns.
