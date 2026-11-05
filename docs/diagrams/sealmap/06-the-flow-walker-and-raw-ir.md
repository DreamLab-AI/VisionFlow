---
id: SM-06
title: The flow walker and the raw flow IR
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/crates/sealmap-rust/src/collect.rs
  - ../sealmap/crates/sealmap-rust/src/resolve.rs
  - ../sealmap/crates/sealmap-rust/tests/extract.rs
  - ../sealmap/crates/sealmap-rust/tests/resolver_er.rs
  - ../sealmap/crates/sealmap-extract/src/raw.rs
  - ../sealmap/crates/sealmap-extract/src/labels.rs
  - ../sealmap/crates/sealmap-model/src/flow.rs
  - ../sealmap/README.md
verified_commit: b9a6aebddd2e8379206eea6cfdd3ab84546724b4
---
## For developers

The flow walker is what turns a function body into the ordered skeleton a
sequence diagram is drawn from. It runs in the collect pass, per file and
without any name resolution, so its output is the raw IR in
`sealmap-extract::raw`: calls with an unresolved callee, branches, loops,
optional blocks, parallel arms and early returns, all as owned strings and
vectors (`crates/sealmap-extract/src/raw.rs:1`-`7`). Resolution and lowering
happen later (SM-07, SM-08).

Two things the walker decides shape every diagram. First, **evaluation
order**: receivers and arguments are walked before the call that consumes
them, and closure bodies passed as arguments are placed *after* the call,
because they run during it. Second, **what is known about receivers**: the
walker keeps a small environment from local names to `Recv` values, which is
the only type information resolution will ever have. This topic draws both,
and records where the walker's view of the code diverges from what runs.

The walker was hardened with a depth cap (`crates/sealmap-rust/src/collect.rs:26`-`30`)
and taught destructuring provenance in commit `5aac8a8`.

## For the business

Sequence diagrams are the densest review material sealmap produces, and they
are only useful if their order matches what the code does. The walker gets
the common Rust shapes right: `?` and `.await` on the call they apply to,
`if`/`else` and `match` as alternatives, loops, closures run per element,
spawned tasks as parallel work.

Adopters should know the shapes it gets wrong, because a reviewer reading a
diagram cannot see them. A closure saved in a variable is drawn where it is
written, not where it is called. A macro invocation is never drawn as a call
of its own, only the calls inside its arguments, and only when those
arguments look like ordinary expressions. Both are recorded below as debt.

## SM-06.1 The raw IR

```mermaid
classDiagram
    direction LR
    class RawStep {
        <<enum>>
        Call  raw.rs:31
        Branch arms with labels  raw.rs:35
        Loop label body  raw.rs:37
        Optional label body  raw.rs:39
        Parallel arms  raw.rs:41
        Return label line  raw.rs:43
    }
    class RawCall {
        callee  raw.rs:50
        label  raw.rs:53
        kind  raw.rs:55
        awaited fallible line  raw.rs:57
    }
    class Callee {
        <<enum>>
        Path segments  raw.rs:89
        Method recv name  raw.rs:91
    }
    class Recv {
        <<enum>>
        SelfValue SelfField  raw.rs:103
        Typed paths  raw.rs:108
        Untyped  raw.rs:111
        Returned path  raw.rs:114
        Derived origin  raw.rs:123
        Computed origin  raw.rs:128
        Unknown  raw.rs:130
    }
    RawStep --> RawCall
    RawCall --> Callee
    Callee --> Recv
```

**What it shows.** A raw call knows what was written, not what it means: a
path, or a method name plus what the walker could tell about the receiver.
`Derived` and `Computed` carry their origin, so resolution can still ask
whether a value came from internal code.

**Why it is this way.** No syn node survives the collect pass, so files can be
walked in parallel and resolved afterwards in one deterministic pass
(`crates/sealmap-extract/src/raw.rs:4`-`7`). The IR moved from `sealmap-rust`
to the shared crate unchanged in `a02b381`; the Rust collector imports its
shapers from there (`crates/sealmap-rust/src/collect.rs:24`).

**Invariant:** an empty branch arm is kept in the raw IR, so lowering can
still name the arm that survives resolution
(`crates/sealmap-extract/src/raw.rs:32`-`34`).

## SM-06.2 Walking a method call with a closure argument

```mermaid
sequenceDiagram
    autonumber
    participant EI as expr_inner<br/>collect.rs:1122
    participant AR as arg<br/>collect.rs:1353
    participant RV as recv<br/>collect.rs:1387
    participant PD as place_deferred<br/>sealmap-extract/src/raw.rs:160
    Note over EI: items.iter().for_each(closure calling a)
    EI->>EI: walk the receiver first (collect.rs:1151)
    EI->>AR: each argument (collect.rs:1154)
    AR->>AR: closure body walked with env saved and restored (collect.rs:1356)
    AR-->>EI: body steps held back as deferred (collect.rs:1360)
    EI->>RV: what is the receiver (collect.rs:1156)
    RV-->>EI: Typed, also for a struct literal, Untyped, SelfField or Unknown (collect.rs:1399)
    EI->>EI: push the method call itself (collect.rs:1158)
    EI->>PD: callee name, deferred bodies, LOOPING list (collect.rs:1165)
    PD-->>EI: loop each via for_each, after the call
```

**What it shows.** Receivers and ordinary arguments are walked before the
call, so nested calls appear as earlier steps; closure and async-block
arguments are held back and placed after the call they are passed to.

**Why it is this way.** A closure passed to a call runs during it, not before
(`crates/sealmap-rust/src/collect.rs:1351`-`1352`); drawing its calls ahead of
the call would invert the order a reader sees. `?` and `.await` mark the call
they apply to through `last_call_mut`, which skips any deferred fragment
pushed after it (`crates/sealmap-extract/src/raw.rs:149`-`153`).

**Invariant:** walking stops below 1,024 levels of expression nesting, so
generated code cannot overflow the collector's stack; deeper calls are
omitted (`crates/sealmap-rust/src/collect.rs:1114`).

## SM-06.3 Where a deferred closure body goes

```mermaid
flowchart TB
    D["closure or async body passed to a call"]
    SP{"callee name contains spawn?<br/>deferred_shape, labels.rs:140"}
    LP{"callee in the per-element list?<br/>LOOPING, collect.rs:918"}
    PAR["Parallel arm: spawned task<br/>labels.rs:153"]
    LOOP["Loop: each via callee<br/>labels.rs:154"]
    OPT["Optional: via callee<br/>labels.rs:156"]
    ANON["Optional: closure,<br/>when the callee is not a plain name<br/>labels.rs:155"]
    D --> SP
    SP -->|yes| PAR
    SP -->|no| LP
    LP -->|yes| LOOP
    LP -->|no, named| OPT
    LP -->|no, unnamed| ANON
```

**What it shows.** A body passed to anything with `spawn` in its name becomes
a parallel arm; a body passed to an iterator adaptor that runs it once per
element becomes a loop; anything else may run, so it is optional.

**Why it is this way.** The shape and the label are shared rules in
`sealmap-extract::labels`, so every adapter draws deferred bodies the same way
(`crates/sealmap-extract/src/raw.rs:155`-`159`). The per-element list is the
Rust adapter's own (`crates/sealmap-rust/src/collect.rs:917`-`942`).

**Debt:** the parallel rule is a substring test on the callee's name
(`crates/sealmap-extract/src/labels.rs:140`), so a closure passed to any
method whose name merely contains `spawn` is drawn as a concurrent task.

## SM-06.4 A closure stored with let is drawn where it is defined

```mermaid
flowchart TB
    SRC["let g = closure calling helper<br/>later: g of 1"]
    LOC["Stmt.Local: walk the initialiser<br/>collect.rs:966"]
    CL["Expr.Closure outside an argument:<br/>body walked in place<br/>collect.rs:1274"]
    OPT["Optional fragment labelled closure,<br/>at the definition<br/>collect.rs:1277"]
    BIND["g bound as Computed, type unknown<br/>collect.rs:1014"]
    CALL["g of 1: a bare name that is neither<br/>defined, imported nor a crate<br/>resolve.rs:860"]
    DROP["call dropped as a local closure<br/>resolve.rs:861"]
    SRC --> LOC --> CL --> OPT
    LOC --> BIND
    SRC --> CALL --> DROP
```

**What it shows.** The body of a closure assigned to a local is walked where
the `let` is, and appears there as an optional fragment; the later call
through the variable is recognised as a local closure and dropped.

**Why it is this way.** The walker has no data-flow analysis, and dropping
bare calls to unknown local names keeps closures and function pointers from
appearing as calls to nothing (`crates/sealmap-rust/src/resolve.rs:858`-`859`).
The test that pins the default drops `g(1)` explicitly
(`crates/sealmap-rust/tests/extract.rs:108`-`116`).

**Debt:** a closure stored with `let` is drawn where it is defined, not where
it is called (`crates/sealmap-rust/src/collect.rs:1274`-`1279`): its calls
appear once, at the definition, inside an `opt` labelled `closure`, even when
it is called many times later or never.

## SM-06.5 if and else-if chains

```mermaid
flowchart TB
    IF["if cond, else if cond2, else<br/>Expr.If, collect.rs:1183"]
    C1["first condition's calls go before<br/>the whole branch<br/>collect.rs:1197"]
    C2["later conditions' calls go inside<br/>their own arm, labelled if cond2<br/>collect.rs:1200"]
    EL["else arm with an empty label<br/>collect.rs:1206"]
    PA{"push_arms<br/>sealmap-extract/src/raw.rs:137"}
    NONE["all arms empty: nothing<br/>sealmap-extract/src/raw.rs:138"]
    ONE["one arm: Optional<br/>sealmap-extract/src/raw.rs:143"]
    MANY["Branch, empty arms kept<br/>sealmap-extract/src/raw.rs:145"]
    IF --> C1
    IF --> C2
    IF --> EL
    C1 --> PA
    C2 --> PA
    EL --> PA
    PA --> NONE
    PA --> ONE
    PA --> MANY
```

**What it shows.** The first condition is always evaluated, so its calls run
before the branch; a later `else if` condition only runs when the earlier
ones failed, so its calls sit inside that arm.

**Why it is this way.** This keeps the evaluation order exact without a
control-flow graph; `match` follows the same idea with the scrutinee walked
first and one arm per pattern. The guard is in the arm's label, and its calls
open the arm, since it runs once the pattern has matched
(`crates/sealmap-rust/src/collect.rs:1214`-`1231`). Until `d227dd8` a guard was
only labelled, never walked (SM-21.6). The shape is pinned by
`flow_shapes_follow_control_flow` (`crates/sealmap-rust/tests/extract.rs:143`).

## SM-06.6 Macros in a body

```mermaid
flowchart TB
    M["a macro invocation in a body<br/>mac, collect.rs:1376"]
    E{"body parses as comma-separated<br/>expressions?<br/>collect.rs:1378"}
    B{"body parses as statements?<br/>collect.rs:1382"}
    WA["walk each argument<br/>collect.rs:1380"]
    WS["walk the statements<br/>collect.rs:1383"]
    SK["skipped: no calls seen"]
    NO["no Call for the macro itself<br/>CallKind.Macro, flow.rs:160"]
    M --> E
    E -->|yes| WA
    E -->|no| B
    B -->|yes| WS
    B -->|no| SK
    M -.-> NO
```

**What it shows.** Calls inside `vec![f()]`, `assert!(g())` or
`format!("{}", h())` are seen; a macro whose body is neither an expression
list nor a statement block contributes nothing; the macro invocation itself
is never a call step.

**Why it is this way.** Without expansion, the only thing that can be walked
is a body that happens to parse as Rust (`crates/sealmap-rust/src/collect.rs:1373`-`1375`).

**Debt:** the model defines `CallKind::Macro` for "macro invocations the
adapter chose to keep" (`crates/sealmap-model/src/flow.rs:158`-`160`), but the
Rust walker never emits one: every call it pushes is a function or a method
(`crates/sealmap-rust/src/collect.rs:1141`, `crates/sealmap-rust/src/collect.rs:1162`).

## SM-06.7 What the walker remembers about local names

```mermaid
stateDiagram-v2
    [*] --> Param: typed parameter or self
    [*] --> Bound: let binding
    Bound --> Typed: declared type or a constructor such as Foo.new
    Bound --> Computed: any other initialiser, origin kept
    Bound --> Forgotten: shadowed without a value
    [*] --> Destructured: if let, while let, match arm, for
    Destructured --> Typed: Some, Ok, Err or one-parameter collection peeled
    Destructured --> Derived: anything else, origin kept
    Typed --> Restored: block ends, environment restored
    Computed --> Restored
    Derived --> Restored
    Restored --> [*]
```

**What it shows.** Parameters seed the environment; a `let` records a type
when one is declared or constructed and otherwise records where the value came
from; patterns peel one wrapper where the shape says which type the bound
name gets; every block, arm and loop body restores the environment on exit.

**Why it is this way.** Recording origins rather than guessing types was the
`Path::parent` fix: a part taken from a std value must never be matched to an
internal method of the same name (`crates/sealmap-extract/src/raw.rs:115`-`122`,
and SM-08). Peeling rules are listed in the walker
(`crates/sealmap-rust/src/collect.rs:1026`-`1033`).

**Invariant:** an `if let` or `while let` binding lives only in its own
block. The walker snapshots the environment before the condition and restores
it after the guarded block, so the binding never reaches an `else` branch or
the statements after, where it used to retype a shadowed outer name and mint a
false Exact edge (`crates/sealmap-rust/src/collect.rs:1192`-`1195`,
`crates/sealmap-rust/src/collect.rs:1255`-`1258`,
`crates/sealmap-rust/tests/resolver_er.rs:37`).

**Debt:** a `use` inside a function body is widened to the whole module's
imports (`crates/sealmap-rust/src/collect.rs:155`-`160`), so a local import in
one function can resolve a same-named path in another.
