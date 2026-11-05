---
id: DE-13
title: Pairwise selection policies and the fault mapping
area: dream-engine
governing: [../dream-engine/docs/adr/ADR-DE-004-pairwise-selection-and-fingerprints.md]
adrs: [ADR-DE-004]
sources:
  - ../dream-engine/crates/pairwise-gate/src/lib.rs
  - ../dream-engine/crates/pairwise-gate/src/model.rs
  - ../dream-engine/crates/pairwise-gate/src/policy.rs
  - ../dream-engine/crates/dream-gate/src/pertest.rs
  - ../dream-engine/crates/dream-core/src/testfmt.rs
  - ../dream-engine/docs/adr/ADR-DE-004-pairwise-selection-and-fingerprints.md
verified_commit: 815f963b1cbd6999e78144ce5475005ae1dd1d8e
---
## For developers

`pairwise-gate` compares a candidate with a baseline on the same ordered list
of tasks and answers `Select`, `Reject` or `Inconclusive`
(`crates/pairwise-gate/src/lib.rs:1`). The rule that matters most is that a
fault is not a loss: an infrastructure failure on either side removes the
pairing from the tally and is reported, so trouble in the testbed is
distinguishable from evidence against the candidate
(`crates/pairwise-gate/src/lib.rs:9`). The engine's gate uses the crate
through `pertest`, which turns per-test results into pairings and maps a
policy `Reject` to an evidence veto and a fault to a harness veto.

## For the business

When a nightly change is judged on a test suite, a flaky test server must not
make a good change look bad, nor a bad change look good. The selection rules
count only comparisons where both sides produced a real result, say so when
too many did not, and apply a no-regression rule to the evidence supplied. That protection depends on complete evidence with unique test identities; duplicate names currently overwrite earlier cases.

## DE-13.1 Tally and the inconclusive guard

```mermaid
flowchart TB
    PR["Pairing: task, baseline Outcome, candidate Outcome<br/>model.rs:117"] --> T{"both sides have a score?<br/>model.rs:195"}
    T -->|no| FA["faults plus 1, excluded from wins and losses<br/>model.rs:205"]
    T -->|"candidate higher"| W["wins plus 1<br/>model.rs:198"]
    T -->|"candidate lower"| L["losses plus 1<br/>model.rs:200"]
    T -->|equal| TI["ties plus 1<br/>model.rs:202"]
    FA --> G{"Guard check<br/>model.rs:283"}
    W --> G
    L --> G
    TI --> G
    G -->|"default: fault share above 0.5, or fewer than 1 comparable"| INC["Inconclusive, before the rule is consulted<br/>policy.rs:39"]
    G -->|ok| RULE["the policy rule decides Select or Reject<br/>policy.rs:40"]
```

**What it shows.** The shared skeleton under `Margin`, `SignTest`,
`WilsonLower` and `NoRegression`: tally, guard, then rule
(`crates/pairwise-gate/src/policy.rs:32`).
**Why it is this way.** Rankers that treat a missing score as negative
infinity would count an outage as a loss; this crate deliberately differs
(`crates/pairwise-gate/src/lib.rs:15`). A fault share exactly at the limit is
tolerated and only a larger share is inconclusive
(`crates/pairwise-gate/src/model.rs:226`).
**Invariant:** a pairing with a fault on either side is never a win or a loss
(`crates/pairwise-gate/src/model.rs:205`).

## DE-13.2 The policies and what each selects on

```mermaid
flowchart TB
    EV["Evidence: primary pairings and optional held-out pairings<br/>model.rs:327"] --> M["Margin: select when wins minus losses exceeds the margin<br/>policy.rs:98"]
    EV --> SG["SignTest: select when the exact one-sided p-value is below alpha<br/>policy.rs:238"]
    EV --> WL["WilsonLower: select when the lower bound of the win rate is above 0.5<br/>policy.rs:305"]
    EV --> NR["NoRegression: reject any pairing that passed on baseline and fails on candidate<br/>policy.rs:364"]
    EV --> FL["Floor: candidate-only, every candidate score at or above the floor<br/>policy.rs:156"]
    EV --> TD["TrainDev: both train and held-out select<br/>policy.rs:435"]
    TD -->|"train policy"| M
    TD -->|"dev policy, usually"| NR
```

**What it shows.** Six policies behind one `Policy::decide` trait method
(`crates/pairwise-gate/src/policy.rs:18`). `TrainDev` composes two of the
others and attaches both sub-decisions.
**Why it is this way.** `NoRegression` is a guard and not a proof of
improvement, so it is paired with an improvement rule through `TrainDev`
(`crates/pairwise-gate/src/policy.rs:329`). In `TrainDev` a thin or fault-heavy
held-out set makes the whole decision `Inconclusive` and never a silent pass
(`crates/pairwise-gate/src/policy.rs:394`).
**Open:** ADR-DE-004 gives 20 cases as the floor for the sign test and the
Wilson bound; the code default is the same figure
(`crates/dream-gate/src/pertest.rs:155`), and it is a field on the options.

## DE-13.3 From per-test results to vetoes

```mermaid
flowchart TB
    NO["no per-test evidence for this evaluator<br/>pertest.rs:187"] --> RO["judged by the receipt alone, no statistics"]
    TO["TestOutcomes: baseline and candidate cases<br/>pertest.rs:122"] --> F1{"any fault, or a baseline test missing on candidate?<br/>pertest.rs:214"}
    F1 -->|yes| HV["harness veto, BLOCKED-ENV, not counted as losses<br/>pertest.rs:232"]
    TO --> F2{"failure fingerprint absent from baseline?<br/>pertest.rs:253"}
    F2 -->|yes| EV1["evidence veto, REJECT<br/>pertest.rs:261"]
    TO --> N{"comparable cases at least min_cases?<br/>pertest.rs:279"}
    N -->|yes| ST["NoRegression plus the optional Floor, Margin, SignTest, WilsonLower<br/>pertest.rs:326"]
    N -->|no| ONE["the evaluator is one case, NoRegression and Floor only<br/>pertest.rs:282"]
    ST --> RJ["policy Reject becomes an evidence veto<br/>pertest.rs:304"]
    ONE --> RJ
    ST -.->|"policy Inconclusive"| DROP["adds no veto at all<br/>pertest.rs:304"]
```

**Debt:** each side is collected into a map keyed only by test name; the last duplicate wins and can hide a fault or regression (`crates/dream-gate/src/pertest.rs:208`).
**What it shows.** Per-test evidence is additive: it can only add vetoes
(`crates/dream-gate/src/pertest.rs:12`). Invalid options are a harness veto when the test map is non-empty; an empty map returns before option validation (`crates/dream-gate/src/pertest.rs:190`).
**Why it is this way.** A statistical test on a handful of cases proves
nothing, so below `min_cases` the evaluator counts as one case
(`crates/dream-gate/src/pertest.rs:20`).
**Tension:** only `Reject` is acted on (`crates/dream-gate/src/pertest.rs:304`).
A policy that comes back `Inconclusive` on per-test evidence, for example
because too many pairings faulted at the crate level, adds no veto here. Faults
are caught separately by the harness veto at step 1, but the two layers are
not obviously equivalent.
**Debt:** evaluators with no per-test format get no statistics. Per-test
evidence exists only when the manifest declares a `test_format` for the
evaluator and a parse of both runs succeeds; otherwise the builder returns
`None`, never fabricated evidence
(`crates/dream-core/src/testfmt.rs:543`).
**Debt:** `TrainDev` and held-out dev evaluators are not wired into the nightly.
The gate's options have no held-out field and the policy builder never
constructs `TrainDev` (`crates/dream-gate/src/pertest.rs:131`,
`crates/dream-gate/src/pertest.rs:314`). ADR-DE-004 states that no consumer
supplies held-out evaluators yet
(`docs/adr/ADR-DE-004-pairwise-selection-and-fingerprints.md:55`).
