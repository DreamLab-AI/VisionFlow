---
id: VK-02
title: Method lifecycle from expectation to gate
area: visioncraft
governing:
  - ../visioncraft/skills/visioncraft/references/edd.md
  - ../visioncraft/skills/visioncraft/references/quality-gates.md
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
adrs: [visioncraft:ADR-0001]
sources:
  - ../visioncraft/skills/visioncraft/references/edd.md
  - ../visioncraft/skills/visioncraft/references/quality-gates.md
  - ../visioncraft/examples/native-workflow.md
  - ../visioncraft/src/plan.rs
  - ../visioncraft/src/evidence.rs
  - ../visioncraft/src/evaluate.rs
  - ../visioncraft/src/cli.rs
  - ../visioncraft/src/process.rs
  - ../visioncraft/tests/core.rs
  - ../visioncraft/docs/adr/0001-portable-quality-contract.md
verified_commit: 1067a5f59afad7e2c8780bce6b29967d1130c19d
---

## VK-02.1 Seven EDD steps and the typed record each one leaves

```mermaid
flowchart TB
    S1["1 Formulate: one behaviour, boundaries, a counterexample<br/>edd.md:10"]
    S2["2 Implement against expectations, failing tests first<br/>edd.md:12"]
    S3["3 Produce evidence: invocation, output, exact subject<br/>edd.md:14"]
    S4["4 Audit: auditor attempts a counterexample the producer missed<br/>edd.md:16"]
    S5["5 Challenge: resolve ambiguity with the decision-maker<br/>edd.md:18"]
    S6["6 Iterate: retain the earlier result, explain the new claim<br/>edd.md:21"]
    S7["7 Stabilise: regression-critical scenarios become permanent checks<br/>edd.md:23"]
    T1["Expectation record<br/>plan.rs:197"]
    T3["ExecutionReceipt<br/>evidence.rs:46"]
    T4["AuditReceipt<br/>evidence.rs:208"]
    T4b["ManualObservation<br/>evidence.rs:234"]
    T7["Stabilizer: check_id plus test_id<br/>plan.rs:187"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    S6 -. "repeat" .-> S2
    S1 --- T1
    S3 --- T3
    S4 --- T4
    S5 --- T4b
    S7 --- T7
    N1["INVARIANT: an auditor label on the same session creates no independence; record the real model family<br/>edd.md:26-31"]
    N2["DOC-DRIFT: edd.md names a stabilized_by reference, the typed field is stabilizer<br/>edd.md:90 vs plan.rs:212"]
    S4 --- N1
    S7 --- N2
```

The method is a loop; the crate only encodes the record each step leaves, not the loop itself. Steps 5 and 6 have no typed counterpart beyond a manual observation.

## VK-02.2 Plan, run and evaluate through the CLI

```mermaid
sequenceDiagram
    autonumber
    actor U as operator or CI
    participant D as CLI dispatch<br/>cli.rs:1084
    participant R as run operation<br/>cli.rs:742
    participant P as bounded runner<br/>process.rs:594
    participant E as evaluate operation<br/>cli.rs:932
    participant G as gate core<br/>evaluate.rs:541
    U->>D: plan init captures selected source bytes (cli.rs:1086-1088)
    U->>D: plan check validates, executes no command (cli.rs:1089-1104)
    U->>D: run plan --all --out fresh-dir (cli.rs:1106)
    D->>R: dispatch run
    Note over R: reported evaluators refused, exit 78 (cli.rs:754-759)
    Note over R: revision and input bytes must equal the plan, else exit 1 (cli.rs:763-770)
    R->>P: private_directory then run_command per check (cli.rs:779, cli.rs:819-820)
    P-->>R: CapturedRun with termination, exit code, logs
    R->>R: write receipt.json, stop at first non-completed check (cli.rs:880-892)
    R-->>U: gate_evaluated false, receipts listed (cli.rs:894-895)
    U->>D: evaluate or gate evaluate or expectations check (cli.rs:1107-1113)
    D->>E: all three reach one function
    E->>E: observe current revision, inputs, dirty state (cli.rs:937-946)
    E->>E: merge-base ancestry of the planned revision (cli.rs:993-1007)
    E->>G: plan, receipts, context (cli.rs:1026)
    G-->>E: GateReport
    E-->>U: exit 0 passed, 1 blocked or incomplete, 70 error (cli.rs:1072-1076)
    Note over U,E: INVARIANT: run exit code never claims the plan passed, only evaluate does (native-workflow.md:40-45)
```

Three user-facing spellings of evaluation share one code path. Evaluation reads evidence and fixed Git metadata and never executes a command found in a plan or receipt.

## VK-02.3 One check: receipt to outcome

```mermaid
flowchart TB
    RC["receipt for check<br/>evaluate.rs:160"]
    B1{"plan digest, subject, invocation match?<br/>evaluate.rs:171-176"}
    B2{"inputs unchanged during run?<br/>evaluate.rs:177-182"}
    B3{"repository, inputs, ancestry, age current?<br/>evaluate.rs:183-189"}
    B4{"process completed?<br/>evaluate.rs:190-195"}
    B5{"within deadline?<br/>evaluate.rs:206-208"}
    B6{"output bytes verified and within bound?<br/>evaluate.rs:210-222"}
    B7{"dirty snapshot verified, scope covers check?<br/>evaluate.rs:223-240"}
    B8{"exit code in success_codes, or adapter report?<br/>evaluate.rs:243-271"}
    ERR["Error"]
    INC["Inconclusive"]
    SKP["Skipped (executable unavailable)"]
    BLK["Blocked"]
    PAS["Passed"]
    RC --> B1
    B1 -- no --> ERR
    B1 -- yes --> B2
    B2 -- no --> INC
    B2 -- yes --> B3
    B3 -- no --> INC
    B3 -- yes --> B4
    B4 -- unavailable --> SKP
    B4 -- other --> ERR
    B4 -- yes --> B5
    B5 -- no --> ERR
    B5 -- yes --> B6
    B6 -- no --> ERR
    B6 -- yes --> B7
    B7 -- unknown or omitted scope --> INC
    B7 -- ok --> B8
    B8 -- no --> BLK
    B8 -- yes --> PAS
    N1["INVARIANT: a missing receipt is Inconclusive, never a pass<br/>evaluate.rs:617-620"]
    N2["INVARIANT: a changed revision is accepted only if the planned one is a known ancestor<br/>evaluate.rs:140-146"]
    PAS --- N1
    B3 --- N2
```

Policy is applied only after every evidence-integrity check has held. The same ladder of outcomes (CheckOutcome, evaluate.rs:15-26) feeds expectations.

## VK-02.4 One expectation: references, audits and observations

```mermaid
flowchart TB
    EX["expectation report<br/>evaluate.rs:326"]
    C1["each supporting check must have Passed<br/>evaluate.rs:336-338"]
    C2["positive control must have Passed<br/>evaluate.rs:339-341"]
    C3["stabilizer test must appear in observed_tests<br/>evaluate.rs:342-353"]
    C4["declared limitation is recorded as a reason<br/>evaluate.rs:354-356"]
    A1["exactly one selected audit<br/>evaluate.rs:368-372"]
    A2["producer and reviewer actors and attempts distinct<br/>evaluate.rs:390-394"]
    A3["producer ran a supporting check, probe is a different check<br/>evaluate.rs:395-407"]
    A4["probe execution passed, or Blocked with a Contradicted verdict<br/>evaluate.rs:408-423"]
    A5["independent_audit requires different model families<br/>evaluate.rs:424-435"]
    M1["manual: exactly one bound, named observation<br/>evaluate.rs:477-497"]
    V["verdict Supported, Contradicted or Inconclusive<br/>evaluate.rs:439-464"]
    EX --> C1 --> C2 --> C3 --> C4
    C4 --> A1 --> A2 --> A3 --> A4 --> A5 --> V
    C4 --> M1 --> V
    T1["tests: stabiliser needs executed test and distinct control<br/>core.rs:404"]
    T2["tests: families and actor attempts are bound<br/>core.rs:429"]
    T3["tests: a counterexample probe blocks even when its assertion failed<br/>core.rs:467"]
    T4["tests: manual observation is a bound record, not a boolean<br/>core.rs:487"]
    C3 --- T1
    A5 --- T2
    A4 --- T3
    M1 --- T4
```

Audit and manual branches run only when required or when a matching record is supplied. A Contradicted verdict forces Blocked regardless of the supporting checks.

## VK-02.5 Aggregation into a gate decision

```mermaid
flowchart LR
    IN["required checks and required expectations<br/>evaluate.rs:643-655"]
    ER{"any report error or Error outcome?<br/>evaluate.rs:656"}
    BL{"any Blocked?<br/>evaluate.rs:658"}
    NP{"any not Passed?<br/>evaluate.rs:660"}
    GE["GateOutcome::Error<br/>evaluate.rs:38"]
    GB["GateOutcome::Blocked<br/>evaluate.rs:34"]
    GI["GateOutcome::Incomplete<br/>evaluate.rs:36"]
    GP["GateOutcome::Passed<br/>evaluate.rs:32"]
    X70["exit 70<br/>cli.rs:1075"]
    X1["exit 1<br/>cli.rs:1074"]
    X0["exit 0<br/>cli.rs:1073"]
    IN --> ER
    ER -- yes --> GE --> X70
    ER -- no --> BL
    BL -- yes --> GB --> X1
    BL -- no --> NP
    NP -- yes --> GI --> X1
    NP -- no --> GP --> X0
    E1["Receipt faults collected before aggregation: unbound, unplanned, duplicate attempt, multiple attempts<br/>evaluate.rs:568-594"]
    N1["TENSION: Blocked and Incomplete share exit 1, so a shell caller must read --json to tell defect from missing evidence<br/>cli.rs:1074"]
    N2["INVARIANT: optional failures stay visible but never change the required denominator<br/>core.rs:189"]
    IN --- E1
    X1 --- N1
    GP --- N2
```

Error beats Blocked beats Incomplete. Passing requires every required check and expectation to pass; an unrelated optional failure is reported but not counted.

## VK-02.6 What the native runner can and cannot prove

```mermaid
flowchart TB
    RUN["native run builds the receipt<br/>cli.rs:847"]
    OS["observed_scope: None (unknown)<br/>cli.rs:874"]
    OT["observed_tests: empty<br/>cli.rs:875"]
    RP["reported: None<br/>cli.rs:876"]
    ST["stabilizer lookup needs the test id in observed_tests<br/>evaluate.rs:344-347"]
    RS["require_observed_scope policy makes unknown scope Inconclusive<br/>evaluate.rs:235-240"]
    RF["generic run refuses reported evaluators<br/>cli.rs:754-759"]
    POL["policy field require_observed_scope<br/>plan.rs:227-228"]
    RUN --> OS
    RUN --> OT
    RUN --> RP
    OT --> ST
    OS --> RS
    POL --> RS
    RP --> RF
    N1["TENSION: with only the generic runner, a regression-critical expectation or a require_observed_scope plan can never pass; a tool-specific adapter is needed<br/>native-workflow.md:47-52"]
    N2["see VK-05.4"]
    ST --- N1
    RS --- N1
    N1 --- N2
```

This is stated honestly in the example, not hidden: selecting files or naming a test in argv does not prove what a tool inspected or ran. The gap bites the Rust and WebAssembly gates in VK-05.
