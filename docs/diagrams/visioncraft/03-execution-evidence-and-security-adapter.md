---
id: VK-03
title: Bounded execution, receipts and the Deepsec adapter
area: visioncraft
governing:
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
  - ../visioncraft/skills/visioncraft/references/deepsec.md
adrs: [visioncraft:ADR-0001, visioncraft:ADR-0002]
sources:
  - ../visioncraft/src/process.rs
  - ../visioncraft/src/cli.rs
  - ../visioncraft/src/evidence.rs
  - ../visioncraft/src/evaluate.rs
  - ../visioncraft/contracts/deepsec.json
  - ../visioncraft/skills/visioncraft/references/quality-gates.md
  - ../visioncraft/src/deepsec.rs
  - ../visioncraft/src/deepsec_evidence.rs
  - ../visioncraft/src/main.rs
  - ../visioncraft/skills/visioncraft/references/deepsec.md
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
  - ../visioncraft/docs/adr/0002-specialist-upstream-boundaries.md
verified_commit: 1067a5f59afad7e2c8780bce6b29967d1130c19d
---

## VK-03.1 One bounded command, start to reap

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller<br/>process.rs:594
    participant RC as run_command_with_env<br/>process.rs:613
    participant CH as child process group
    participant CP as capture threads
    CL->>RC: invocation, root, fresh output dir, cancel flag
    RC->>RC: validate invocation, create stdout.log and stderr.log exclusively (process.rs:640-648)
    RC->>RC: already cancelled returns Interrupted, zero bounds are an error (process.rs:660-670)
    RC->>CH: env_clear, own process group, stdin null, named env only (process.rs:676-689)
    RC->>CP: two capture threads sharing one output bound (process.rs:709-724)
    loop every 10 ms
        RC->>RC: cancel, deadline, capture fault, leader exit (process.rs:728-748)
    end
    RC->>RC: completed leader but live group means CaptureFailed (process.rs:752-756)
    RC->>CH: not completed: SIGINT, 250 ms grace, then SIGKILL to the group (process.rs:783-808)
    RC->>CH: reap the leader exactly once (process.rs:812-816)
    CP-->>RC: join, fault 1 output limit exceeded, 2 capture failed (process.rs:817-833)
    RC-->>CL: CapturedRun with exit code, signal, monotonic elapsed (process.rs:835-839)
    Note over RC: INVARIANT: signals go only to a group the runner still owns, ECHILD never falls through to a numeric kill (process.rs:201-205)
    Note over RC: DEBT: this is a process runner, not a sandbox, children keep the caller's filesystem and network authority (process.rs:1-3)
```

A zombie in the group is not a live descendant, hence the /proc check after signal 0 (process.rs:316). Non-Linux builds return Unsupported before spawning (process.rs:631).

## VK-03.2 Termination to receipt status to exit code

```mermaid
flowchart LR
    T1["Termination: Completed, Unavailable, FailedToStart,<br/>TimedOut, Interrupted, CaptureFailed<br/>process.rs:27-40"]
    PS["process_status maps one to one<br/>cli.rs:670-678"]
    ST["ProcessStatus recorded in the receipt, before any verdict<br/>evidence.rs:14"]
    XE["execution_exit: 78 unavailable, 124 timeout, 130 interrupted, 70 start or capture, else 0 or 1<br/>cli.rs:681-697"]
    OUT["gate-side meaning: Unavailable is Skipped, other incomplete is Error<br/>evaluate.rs:190-195"]
    SIG["a signal-terminated completed process becomes Interrupted<br/>cli.rs:832-835"]
    T1 --> PS --> ST
    ST --> XE
    ST --> OUT
    SIG --> ST
    N1["INVARIANT: process completion is recorded separately from the evaluator's judgement<br/>quality-gates.md:69-73"]
    N2["INVARIANT: the 78, 124, 130, 70 vocabulary is shared with the Deepsec adapter<br/>deepsec.rs:93"]
    ST --- N1
    XE --- N2
```

The runner stops at the first incomplete check (cli.rs:890-892) and keeps earlier receipts. Exit codes are advisory; the authoritative decision is the evaluated gate (see VK-02.5).

## VK-03.3 Deepsec adapter attempt, step by step

```mermaid
sequenceDiagram
    autonumber
    actor U as host or CI
    participant A as Deepsec adapter<br/>deepsec.rs:645
    participant V as persisted-state checks<br/>deepsec_evidence.rs:362
    participant X as external deepsec CLI
    U->>A: visioncraft deepsec with exactly one source mode (deepsec.rs:654-658)
    A->>A: policy validated, disabled returns 78, record SKIPPED not passed (deepsec.rs:686-692)
    A->>A: credential env var must be set unless local auth (deepsec.rs:769-777)
    A->>A: review workspace 0700, locked, report dir must be new (deepsec.rs:778-793)
    A->>V: refuse repository-controlled reviewer state first (deepsec.rs:812)
    A->>A: write config with credential names only (deepsec.rs:826-838)
    A->>X: --version must equal the contract version, else 78 (deepsec.rs:859-864)
    A->>A: select files, record dirty, snapshot inputs-before (deepsec.rs:876-883)
    A->>X: scan (full or scan-only), process with frozen file list (deepsec.rs:909-944)
    X-->>A: exit 0 or 1, exit 1 needs a fresh nonempty comment (deepsec.rs:945-956)
    A->>A: inputs-after must equal inputs-before (deepsec.rs:957-965)
    A->>V: run and per-file records must match the frozen set (deepsec.rs:966-973)
    A->>X: export findings.json at LOW (deepsec.rs:974-989)
    A->>A: parse, count, apply fail_on threshold, PASS or BLOCK (deepsec.rs:990-1015)
    A-->>U: receipt.json plus exit 0 or 1 (deepsec.rs:1034-1044)
    Note over A,X: TENSION: scan-only returns result SCANNED with exit 0, so a naive CI step reads it as success (deepsec.rs:918 and deepsec.rs:1020)
    Note over A: INVARIANT: severity order CRITICAL to LOW is explicit, bug classes included (deepsec.rs:31)
```

Exit 1 from Deepsec can mean findings or failure; only a fresh completed run plus a valid export upgrades it to findings (deepsec.md:38-42). A threshold pass means no counted finding crossed it, not zero findings.

## VK-03.4 Deepsec result classes and process exit codes

```mermaid
flowchart TB
    OC["inner attempt returns Ok(value) or Err(GateError)<br/>deepsec.rs:1018"]
    OK["Ok: result PASS exit 0, BLOCK exit 1<br/>deepsec.rs:1019-1022"]
    ER["Err: 124 TIMEOUT, 130 INTERRUPTED, 78 UNAVAILABLE, else ERROR<br/>deepsec.rs:1023-1026"]
    GC["GateError codes: 70 runtime, 78 configuration, 124 timeout, 130 interruption<br/>deepsec.rs:93"]
    PE["clap parse error: 78 if first argument is deepsec, else 2, help is 0<br/>main.rs:13-21"]
    EX["any other code outside 0..=255 becomes 70<br/>main.rs:33"]
    SO["scan-only success is not a completed review<br/>deepsec.rs:918"]
    RF["receipt always written, including on error<br/>deepsec.rs:1034-1038"]
    OC --> OK
    OC --> ER
    GC --> ER
    OK --> RF
    ER --> RF
    PE --> EX
    OK --- SO
    N1["INVARIANT: unsupported version, disabled policy or missing tool never satisfies a required gate<br/>deepsec.json:7-11"]
    N2["INVARIANT: the doc separates a cheap candidate scan from a required review, matching the SCANNED result<br/>deepsec.md:5-7"]
    ER --- N1
    SO --- N2
```

The legacy exit code 78 for a malformed deepsec invocation keeps old shell callers on the configuration-error path. The receipt records mode, revision, dirty scope and the observed Deepsec version.

## VK-03.5 Evidence on disk and how evaluation re-verifies it

```mermaid
flowchart TB
    OUT["run --out must be a fresh private directory<br/>cli.rs:779"]
    GI[".gitignore containing * written first<br/>cli.rs:781"]
    PJ["plan.json copy<br/>cli.rs:782"]
    CK["check-NNNN/ per attempt<br/>cli.rs:798-800"]
    LG["stdout.log and stderr.log with SHA-256<br/>cli.rs:822-829"]
    RJ["receipt.json<br/>cli.rs:880-881"]
    LR["evaluate loads each receipts directory<br/>cli.rs:976-978"]
    VA["VerifiedArtifacts: bytes hashed in memory<br/>evidence.rs:258"]
    CX["context.json artifacts resolved against its own parent<br/>cli.rs:983-990"]
    GA["evaluate refuses an unverified artifact<br/>evaluate.rs:210-219"]
    OUT --> GI --> PJ --> CK --> LG --> RJ
    RJ --> LR --> VA --> GA
    CX --> VA
    N1["INVARIANT: output directories are never reused; files are created exclusively<br/>process.rs:131-133"]
    N2["INVARIANT: private directories are mode 0700 and refuse symlink parents<br/>process.rs:79-127"]
    N3["EXTERNAL: keep source-bearing evidence out of public commits<br/>deepsec.md:30-34"]
    OUT --- N1
    OUT --- N2
    LG --- N3
```

Evidence is content-addressed, not trusted by path: a mutated log never verifies. Older attempts live in an archive, one selected attempt per check.
