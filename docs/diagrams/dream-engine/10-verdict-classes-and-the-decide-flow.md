---
id: DE-10
title: Verdict classes and the decide flow
area: dream-engine
governing: [../dream-engine/docs/concepts.md, ../dream-engine/docs/adr/ADR-DE-003-protected-path-guard.md]
adrs: [ADR-DE-003, ADR-DE-009]
sources:
  - ../dream-engine/crates/dream-gate/src/gate.rs
  - ../dream-engine/crates/dream-gate/src/pertest.rs
  - ../dream-engine/crates/dream-core/src/candidate.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-core/src/git.rs
  - ../dream-engine/crates/dream-core/src/replay.rs
  - ../dream-engine/crates/dream-gate/src/guard.rs
  - ../dream-engine/docs/concepts.md
  - ../dream-engine/README.md
  - ../dream-engine/docs/adr/ADR-DE-003-protected-path-guard.md
  - ../dream-engine/docs/adr/ADR-DE-009-candidates-are-formatted-before-the-gate.md
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---
## For developers

The gate is a pure function of a frozen manifest, typed receipts and a strict
parse of the model's verdict (`crates/dream-gate/src/gate.rs:8`). This topic
covers how a veto gets its class, how classes become a verdict, and the order
of the protected-path checks before application, after formatting and inside `decide`. After reading it you can say which verdict a
given failure produces and why a guard refusal can never be diluted into a
harness fault.

Vetoes come in three classes (`crates/dream-gate/src/gate.rs:42`): harness
(evidence could not be gathered), evidence (gathered and against the
candidate) and unproven (nothing to test or nothing readable). The module table
maps them to `BLOCKED-ENV`, `REJECT` and `INCONCLUSIVE`
(`crates/dream-gate/src/gate.rs:17`), which matches the concepts document
(`docs/concepts.md:34`).

Receipt and policy checks include the following controls. The gate validates selected receipt identity fields: which evaluators are required, and each expected command, come from the frozen manifest and the receipts are checked against it
(`crates/dream-gate/src/gate.rs:197`). And the guard now has resource bounds, a
shell-word reader for evaluator commands, and a policy builder that refuses a
command it cannot read (`crates/dream-gate/src/guard.rs:182`,
`crates/dream-gate/src/guard.rs:489`, `crates/dream-gate/src/guard.rs:340`).

## For the business

A nightly engineer that confuses "the test machine was down" with "the change
was bad" teaches the team to distrust it. The class split keeps those apart: a
broken test runner is recorded as an environment problem and does not count
against the repository, while a change that touches the files that decide
whether it passes is rejected outright. A model that claims success but ships
no patch is not trusted either; it is recorded as inconclusive. Receipt command identities are checked against the frozen manifest. The gate still trusts caller-supplied typed outcomes; it does not authenticate execution or independently bind the receipt to the evaluated tree. A patch that is absurdly large, or an evaluator command the
engine cannot read with certainty, stops the night rather than being guessed at.

## DE-10.1 From evaluator outcome to veto class

```mermaid
flowchart TB
    subgraph OUT["EvaluatorOutcome per required evaluator"]
        P["Passed<br/>no veto<br/>gate.rs:123"]
        M["Missing, Silent, Blocked, TimedOut<br/>gate.rs:124-138"]
        F["Failed exit code, ExplicitFail marker<br/>gate.rs:141-146"]
    end
    M --> H["Harness veto<br/>gate.rs:64"]
    F --> E["Evidence veto<br/>gate.rs:71"]
    H --> BE["BLOCKED-ENV"]
    E --> RJ["REJECT"]
    U["Unproven veto<br/>no patch, refused patch, unreadable verdict<br/>gate.rs:78"] --> IC["INCONCLUSIVE"]
```

**What it shows.** When receipt grading runs, each non-passing required evaluator outcome maps to one veto
class in `veto_for` (`crates/dream-gate/src/gate.rs:121`): missing, silent,
blocked and timed-out are harness faults; a non-zero exit and an explicit
failure marker are evidence.
**Why it is this way.** A silent evaluator that exits 0 with no output proves
nothing, so it is a harness veto, not a pass
(`crates/dream-gate/src/gate.rs:128`). An evaluator with no receipt at all is
completed as `Missing` so a skipped evaluator cannot be a silent pass
(`crates/dream-gate/src/gate.rs:153`).
**Invariant:** the gate calls no model and reads no clock; the same inputs
always give the same decision (`crates/dream-gate/src/gate.rs:8`).

## DE-10.2 The decide flow

```mermaid
flowchart TB
    VR["Validate receipt identities against manifest<br/>gate.rs:389"] --> C["Read strict verdict; add unproven veto if unreadable<br/>gate.rs:396"]
    C --> A{"claimed ACCEPT?<br/>gate.rs:405"}
    A -->|yes| S{"candidate state<br/>gate.rs:418"}
    S -->|Applied| G["Recheck patch; violation makes tree untrusted<br/>gate.rs:420"]
    S -->|Protected| E["Evidence veto; tree untrusted"]
    S -->|DidNotApply| H["Harness veto; tree untrusted"]
    S -->|NoPatch, Refused, NotAttempted| U["Unproven veto; tree untrusted"]
    A -->|no| T
    G --> T
    E --> T
    H --> T
    U --> T
    T{"tree trusted? Applied unless recheck refused<br/>gate.rs:461"} -->|yes| R["Grade validated receipts; optional config digest if ACCEPT;<br/>add per-test vetoes<br/>gate.rs:463"]
    T -->|no| V
    R --> V{"claimed ACCEPT?<br/>gate.rs:489"}
    V -->|no| M["Keep parsed model verdict; unreadable is INCONCLUSIVE<br/>gate.rs:493"]
    V -->|yes| P["No veto: ACCEPT; else Harness: BLOCKED-ENV;<br/>else Evidence: REJECT; else INCONCLUSIVE<br/>gate.rs:500"]
```

**What it shows.** `decide` collects vetoes in steps, then picks the verdict
by precedence: harness over evidence over unproven
(`crates/dream-gate/src/gate.rs:500`). A claim that is not ACCEPT keeps the
model's own reading, or `INCONCLUSIVE` if unreadable
(`crates/dream-gate/src/gate.rs:493`).
**Why it is this way.** When the re-check refuses the patch, `tree_trusted`
goes false and step 3 is skipped, so the candidate's receipts are not graded.
Harness faults outrank evidence in general, but a refused patch skips grading,
so a guard veto can never be diluted into `BLOCKED-ENV`
(`crates/dream-gate/src/gate.rs:500`). The digest check runs only after the
guard has said the patch did not touch the config, so drift is the harness and
not the candidate (`crates/dream-gate/src/gate.rs:467`). The receipts graded
in step 3 are the validated set, so a receipt that fails those identity checks becomes `Missing` and yields a harness veto (`crates/dream-gate/src/gate.rs:389`).
**Debt:** candidate config comparison is optional: an absent `candidate_config_digest` skips it (`crates/dream-gate/src/gate.rs:470`).
**Invariant:** acceptance needs a claimed ACCEPT and an empty veto list
(`crates/dream-gate/src/gate.rs:489`).
Resolved since the earlier stamp: the README no longer lists `dream-gate` as
planned; its crate table describes the built gate (`README.md:77`).

## DE-10.3 Guard before apply, recheck inside the gate

```mermaid
sequenceDiagram
    participant H as Host driver
    participant C as candidate run
    participant P as persist build
    participant R as EvaluatorRunner
    participant G as gate decide
    H->>C: patch and baseline policy
    C->>C: binary-delete refusal and guard, candidate.rs line 482
    C->>C: delete stale branch, candidate.rs line 493
    C->>P: prepare and build_branch_worktree, candidate.rs line 494
    P->>P: guard before worktree creation, persist.rs line 139
    P->>P: git apply original bytes on stdin
    opt original strategies fail and normalisation changes text
        P->>P: guard normalised text before applying, persist.rs line 196
    end
    P-->>C: prepared tree and applied patch
    C->>C: optional configured formatter on control host, candidate.rs line 527
    C->>C: optional relocation, candidate.rs line 528
    C->>R: optional build then required evaluators, candidate.rs line 547
    C-->>H: Attempt with prepared hash, patch and receipts
    H->>G: state, receipts, policy and optional config digest
    opt report claims ACCEPT and candidate is Applied
        G->>G: recheck recorded patch text, gate.rs line 420
    end
    G-->>H: GateDecision
```

**What it shows.** The current order includes formatting before relocation, build and evaluation. The first guard precedes branch deletion; the persist-layer guard precedes its worktree creation. Normalised fallback text is checked separately (`crates/dream-core/src/candidate.rs:482`, `crates/dream-core/src/persist.rs:196`).
**Why it is this way.** A protected input must stop before candidate commands run. The gate also checks supplied patch text when acceptance is claimed (`crates/dream-gate/src/gate.rs:418`). The caller must construct policy from the correct frozen baseline; the type alone does not prove that provenance (`crates/dream-gate/src/gate.rs:295`).
**Debt:** the formatter executes on the control host before relocation, and build/evaluator commands execute with runner privileges. Worktree isolation is not an execution sandbox (`crates/dream-core/src/candidate.rs:71`, `crates/dream-core/src/candidate.rs:540`).
**Debt:** the returned tree hash is prepared before build/evaluation; commands can mutate the tree afterwards without a final hash check in `run` (`crates/dream-core/src/candidate.rs:560`).
**Open:** pre-apply non-protected guard failures become `Refused` and therefore unproven; the gate's recheck of an `Applied` patch makes those failures evidence vetoes. The layers intentionally remain described separately (`crates/dream-core/src/candidate.rs:464`, `crates/dream-gate/src/gate.rs:344`).
**Debt:** the worktree path is predictable under the shared temporary directory and an existing directory is removed before use (`crates/dream-core/src/persist.rs:142`).

## DE-10.4 Receipts are checked against the frozen manifest

```mermaid
flowchart TB
    M["frozen manifest<br/>required evaluators and command digests"] --> VR["validated_receipts<br/>gate.rs:197"]
    RC["receipts for the phase"] --> VR
    VR --> OTH["name not required<br/>kept but forced advisory<br/>gate.rs:205"]
    VR --> REQ["each required evaluator<br/>gate.rs:213"]
    REQ --> ONE{"exactly one receipt<br/>gate.rs:216"}
    ONE -->|"no, none or duplicates"| MS["synthesised Missing receipt<br/>gate.rs:228"]
    ONE -->|yes| CHK{"required, digest field equals frozen digest,<br/>and digest of command text equals it<br/>gate.rs:218"}
    CHK -->|no| MS
    CHK -->|yes| OK["receipt accepted as is"]
    MS --> HV["Harness veto naming the evaluator<br/>gate.rs:124"]
    HV --> BE["BLOCKED-ENV"]
    OK --> GR["graded by veto_for<br/>gate.rs:121"]
    VR -.-> ENV["also used by environment_vetoes<br/>on baseline receipts<br/>gate.rs:245"]
```

**What it shows.** `decide` and `environment_vetoes` both build their receipt
set from the manifest, not from the receipts. A receipt that downgrades itself
to advisory, names a different command, carries a digest field that lies about
its own command text, or is duplicated is replaced by a `Missing` receipt,
which `veto_for` turns into a harness veto.
**Why it is this way.** Earlier the required set came from the receipts, so a
forged receipt could take itself out of the veto set. Both the stored digest
field and a digest recomputed from the command text are compared with the
frozen one, because either alone can be forged
(`crates/dream-gate/src/gate.rs:217`). A receipt for a name the manifest does
not require is kept but forced advisory, so it cannot promote itself into the
veto set either (`crates/dream-gate/src/gate.rs:208`). The forged case is a
harness fault and so `BLOCKED-ENV`, not `REJECT`, which keeps a tampered
environment from being counted as evidence against the repository.
**Debt:** passing identity checks leaves the supplied typed outcome unchanged; raw streams are reclassified by replay verification, not this function. Neither path attests real command execution (`crates/dream-gate/src/gate.rs:218`, `crates/dream-core/src/replay.rs:467`).
**Invariant:** the set of required evaluators and each command's identity is
decided by the manifest alone (`crates/dream-gate/src/gate.rs:197`).
**Debt:** `complete_receipts` remains for callers holding only identities and
trusts the receipt's own fields (`crates/dream-gate/src/gate.rs:157`); `decide`
does not use it.

## DE-10.5 Guard resource bounds and evaluator-command paths

```mermaid
flowchart TB
    PT["patch text"] --> EM{"empty?<br/>guard.rs:977"}
    EM -->|yes| EMV["Empty violation"]
    EM -->|no| SZ{"over max_patch_bytes?<br/>guard.rs:980"}
    SZ -->|yes| TL["TooLarge: patch bytes<br/>guard.rs:134"]
    SZ -->|no| LN{"header line over max_header_line_bytes?<br/>guard.rs:1004"}
    LN -->|yes| TL2["TooLarge: header line bytes<br/>guard.rs:1009"]
    LN -->|no| FL{"more than max_files sections?<br/>guard.rs:999"}
    FL -->|yes| TL3["TooLarge: files<br/>guard.rs:1000"]
    FL -->|no| PR["touched paths, then protected check<br/>guard.rs:1168"]
    subgraph FZ["Policy built from the frozen manifest"]
        direction TB
        CMD["each frozen evaluator command<br/>guard.rs:359"] --> SW["shell_words in strict mode<br/>guard.rs:489"]
        SW -->|"unterminated quote or command substitution"| UC["UnsupportedCommand, the night must not run<br/>guard.rs:169"]
        SW -->|"words"| WP["word_path: path-like words and known baseline files<br/>guard.rs:586"]
        WP --> PTH["protected paths in the policy<br/>guard.rs:661"]
    end
    UC --> BEV["host reports BLOCKED-ENV"]
```

**What it shows.** Two things the guard now does before it trusts a patch or a
policy. Total patch size is bounded before line parsing; header line length and file count are bounded during parsing, with defaults of 4 MiB, 2000 files and
8192-byte header lines (`crates/dream-gate/src/guard.rs:192`). And the policy
for a night is built from the frozen evaluator commands read as shell words:
the files they name become protected, and unsupported quoting or command substitution returns an error. This is a shell-word parser, not a complete shell dependency analysis.
**Why it is this way.** The size check runs before the patch is split into lines, and the
unquoted `diff --git` line is split in linear time without allocating
(`crates/dream-gate/src/guard.rs:888`). Any bound
exceeded fails closed (`crates/dream-gate/src/guard.rs:963`). The tokeniser
honours POSIX quoting: unquoted `; | & < > ( ) { } ,` end a word, single quotes
are literal, double quotes honour backslash escapes (`crates/dream-gate/src/guard.rs:489`).
In strict mode an unterminated quote is an error, and so is a command
substitution, `$(` or a backtick, anywhere outside single quotes, including
inside double quotes (`crates/dream-gate/src/guard.rs:536`,
`crates/dream-gate/src/guard.rs:557`). The lenient form,
`evaluator_command_paths`, never fails and over-approximates, and is kept for
callers that are not freezing a night (`crates/dream-gate/src/guard.rs:703`).
`ProtectedPolicy::for_manifest` uses the fallible form on every frozen command
and also refuses a config whose bytes do not hash to the frozen digest
(`crates/dream-gate/src/guard.rs:340`). Files that a command names only
indirectly are the host's job: `PolicyExtras::resolving` resolves absolute
paths under the repository root and bare words that name files in the baseline
tree, errors the same way, and records the result so a replay protects exactly
what the live run did (`crates/dream-core/src/replay.rs:143`,
`crates/dream-core/src/replay.rs:173`).
**Invariant:** a protected-path policy is a pure function of the frozen
manifest, the config bytes and the recorded extras, so equal inputs reconstruct equal policy; a host must supply the same inputs to the live guard and replay (`crates/dream-core/src/replay.rs:173`).
**Tension:** command substitution is refused wherever it is not in single
quotes, so a legitimate evaluator such as `cargo test $(nproc)` stops the night
as `UnsupportedCommand` (`crates/dream-gate/src/guard.rs:557`); the rule is
safe and blunt, and the evaluator has to be rewritten.
**Debt:** the reader cannot see files an evaluator reads indirectly (a
`Makefile` behind `make test`, a `conftest.py` behind `pytest`); those must be
named in the config's `protected` globs, or be baseline-tree files resolved by
the host (`crates/dream-gate/src/guard.rs:703`).
