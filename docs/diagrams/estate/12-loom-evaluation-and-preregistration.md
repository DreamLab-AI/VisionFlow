---
id: ES-12
title: The Loom evaluation and its preregistration
area: estate
governing:
  - ../loom/docs/design/LOOM-POSITIONING.md
  - ../loom/docs/design/PRD-025-ontology-loom-and-connector-platform.md
adrs: [loom:ADR-135, loom:ADR-136, loom:ADR-137, loom:ADR-138, loom:ADR-139, loom:PRD-025, loom:PRD-026, loom:PRD-027, loom:PRD-028, agentbox:ADR-051, agentbox:ADR-2023, agentbox:ADR-2084, agentbox:ADR-2095]
sources:
  - ../loom/docs/design/PRD-028-does-loom-earn-its-complexity.md
  - ../loom/docs/design/PRD-025-ontology-loom-and-connector-platform.md
  - ../loom/docs/design/PRD-026-loom-consolidation.md
  - ../loom/docs/design/PRD-027-rust-loom-reengineering.md
  - ../loom/docs/design/ADR-135-ontology-loom-node.md
  - ../loom/docs/design/ADR-136-loom-tooling-allocation.md
  - ../loom/docs/design/ADR-137-loom-rust-replatform.md
  - ../loom/docs/design/ADR-138-confidence-surfacing-contract.md
  - ../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md
  - ../loom/docs/design/LOOM-POSITIONING.md
  - ../loom/docs/design/ONTOLOGY-LOOM-PIPELINE.md
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/docs/design/agentbox-ADR-051-loom-client-and-deferred-distillation.md
  - ../project/agentbox/skills/email-search/SKILL.md
  - scripts/diagram-index-gen.cjs
verified_commit: {loom: 07a0e6774, agentbox: b7b1ab81a, visionflow: df22182f3}
---

## For developers

PRD-028 is a preregistration, not a result. It fixes, in advance, what the Loom would have to
show to keep its structured serving path, and the conditions under which it would be judged not
to earn its complexity. Read the records before you read a benchmark number: the only measured
verdicts the estate holds are the three in PRD-025 §3.1, and the later addendum re-centres what
the headline figure means. Nothing in PRD-028 has been run.
The facade mechanics live in AB-24; this topic is about the evidence, not the wire.

## For the business

The Loom is the part of the estate that makes a local model answer accurately about our own
material. This topic records how we have agreed to find out whether that structure is worth its
cost, before we spend on it. The test is written down first, with the pass mark, the stopping
conditions and the honest ways it can fail, so the answer cannot be chosen after the fact.

## ES-12.1 What has been measured, and what the later reading changed
```mermaid
flowchart TB
    BENCH["The one measured benchmark — 37 held-out questions, seed 42,<br/>objective graph-derived gold, paired axes with bootstrap 95%<br/>confidence intervals<br/>../loom/docs/design/PRD-025-ontology-loom-and-connector-platform.md:174-176"]

    subgraph AXES["Mean recall by axis, PRD-025-ontology-loom-and-connector-platform.md:178-183"]
        A0["raw, no grounding — 0.27, :180"]
        A1["static structured scaffold — 0.94, roughly 3.5 times raw<br/>and CI-significant, :181"]
        A2["prose-enriched scaffold — 0.95, about nothing over<br/>structured, :182"]
        A3["agentic tool traversal — 0.65, BELOW static injection, :183"]
    end
    BENCH --> AXES

    F1["FINDING 1 — the static structured scaffold is the dominant win,<br/>so scaffold and index generation is the grounding product<br/>PRD-025-ontology-loom-and-connector-platform.md:187-189"]
    F2["FINDING 2 — prose adds about nothing over structured, so the<br/>prose index is a cheap complement and stays optional<br/>PRD-025-ontology-loom-and-connector-platform.md:190-191"]
    F3["FINDING 3 — feed, do not send traversing. Distillation is<br/>retrieval-fed map-reduce, never tool-driven exploration<br/>PRD-025-ontology-loom-and-connector-platform.md:192-199"]
    A1 --> F1
    A2 --> F2
    A3 --> F3

    ADD["ADDENDUM 2026-08-18 — the 3.5 times figure is faithful DELIVERY<br/>of facts the scaffold already exposes, measured against a<br/>verbatim-copy ceiling, not reasoning over structure. Across ten<br/>models from five providers the gain OVER that ceiling is uniformly<br/>negative. The three findings stand; the interpretation of the<br/>number changes, and the benchmark model is now Qwen3.8-27B.<br/>PRD-025-ontology-loom-and-connector-platform.md:201-210"]
    F1 --> ADD
    F2 --> ADD
    F3 --> ADD

    SPEED["The speed half of the claim is asserted by the consumer, not by<br/>this repository: roughly 3 to 6 times faster than cold parametric<br/>reasoning, agentbox/skills/email-search/SKILL.md:92-93"]
    A1 --> SPEED

    GAP["OPEN — the measured suite covers ONE axis of four. General-question<br/>robustness, a web-search baseline and attribution precision are all<br/>named as needed and not run.<br/>../loom/docs/design/LOOM-POSITIONING.md:68-84"]
    ADD --> GAP
```

## ES-12.2 The preregistration — its question, and the arm it must beat
```mermaid
flowchart TB
    Q["PROPOSED primary question — at matched information access and<br/>context budget, does the Loom improve evidence-supported task<br/>success by a practically meaningful amount over the STRONGEST<br/>flat-text retrieval baseline selected on development data<br/>../loom/docs/design/PRD-028-does-loom-earn-its-complexity.md:29"]

    subgraph ARMS["Five arms, PRD-028-does-loom-earn-its-complexity.md:149-155"]
        AA["A closed book — measures unaided performance, explicitly NOT<br/>architectural advantage, :151"]
        AB["B tuned lexical BM25 — the required simple baseline, :152"]
        AC["C hybrid retrieval with a local reranker — the required<br/>STRONG baseline, :153"]
        AD["D Loom — frozen production retrieval, gating and scaffold, :154"]
        AE["E oracle evidence — a diagnostic reference, not a deployable<br/>competitor, :155"]
    end
    Q --> ARMS

    BSTAR["B-star is the better deployable flat baseline of B and C, chosen<br/>on DEVELOPMENT data with tie-breaking declared by cost, then<br/>FROZEN. The primary contrast is D minus B-star on the sealed test.<br/>PRD-028-does-loom-earn-its-complexity.md:157"]
    AB --> BSTAR
    AC --> BSTAR
    AD --> BSTAR

    PARITY["INVARIANT — information parity. Asserted and materialised inferred<br/>facts are made available to the flat baseline as readable,<br/>provenance-linked records, with NO information exclusive to the<br/>Loom, and both systems retrieve within the same budget.<br/>PRD-028-does-loom-earn-its-complexity.md:100"]
    BSTAR --> PARITY

    LOCK["INVARIANT — source and canonical collections are frozen BEFORE<br/>final question authoring, representations are generated without<br/>the questions or gold answers, and there is no test-driven<br/>ontology repair after lock.<br/>PRD-028-does-loom-earn-its-complexity.md:96"]
    PARITY --> LOCK

    STATUS["PROPOSED — status is proposed requirements and preregistration<br/>specification, no new experiments have been run, and the numerical<br/>thresholds are product decisions rather than constants.<br/>PRD-028-does-loom-earn-its-complexity.md:19"]
    Q --> STATUS
```

## ES-12.3 The endpoint — what counts as success, and what always fails
```mermaid
stateDiagram-v2
    [*] --> Attempted
    Attempted --> Empty: the model returns nothing, times out<br/>or the tool fails
    Empty --> Failed: counted as a failure in the primary<br/>service-level endpoint, PRD-028-does-loom-earn-its-complexity.md:169
    Attempted --> Answered

    Answered --> Checked: score the four conditions
    Checked --> Failed: a required proposition is missing
    Checked --> Failed2: a material false or unsupported<br/>assertion is present
    Checked --> Failed3: the wrong version or scope is selected
    Checked --> Failed4: a material claim carries no citation
    Checked --> Succeeded: all four hold, PRD-028-does-loom-earn-its-complexity.md:177

    Attempted --> Unanswerable
    Unanswerable --> Succeeded: the pre-labelled abstention or<br/>clarification, with nothing invented
    Unanswerable --> Failed5: an answer is invented instead

    Failed --> [*]
    Failed2 --> [*]
    Failed3 --> [*]
    Failed4 --> [*]
    Failed5 --> [*]
    Succeeded --> [*]

    note right of Succeeded
      INVARIANT - the endpoint is BINARY per question,
      with component scores reported separately so a
      citation-formatting failure is not mistaken for a
      factual one. PRD-028-does-loom-earn-its-complexity.md:179
    end note
    note right of Failed2
      INVARIANT - a dump of the corpus, or an enumeration
      of every candidate, FAILS the task-scope rubric even
      when it contains the correct names. PRD-028-does-loom-earn-its-complexity.md:179
    end note
    note left of Empty
      No complete-case deletion. First-attempt and
      retry-policy success are reported separately
      rather than one replacing the other. PRD-028-does-loom-earn-its-complexity.md:169
    end note
```

## ES-12.4 The decision table — every way the preregistration can end
```mermaid
flowchart TB
    EST["PRIMARY ESTIMAND — the mean paired difference in evidence-supported<br/>task success, D minus B-star, with a 95 percent interval from a<br/>paired cluster bootstrap over independent topic families<br/>../loom/docs/design/PRD-028-does-loom-earn-its-complexity.md:216"]
    BAR["PROPOSED meaningful advantage — 5 percentage points absolute<br/>success, PRD-028-does-loom-earn-its-complexity.md:218"]
    EST --> BAR

    O1["lower bound above plus 5 — evidence of a practically meaningful<br/>quality advantage AT THE TESTED OPERATING POINT, :222"]
    O2["lower bound above zero, point estimate at least plus 5, lower<br/>bound at most plus 5 — improvement, magnitude uncertain, :223"]
    O3["interval spans zero AND meaningful improvement — INCONCLUSIVE,<br/>and explicitly not a claim of equivalence or of no value, :224"]
    O4["non-inferiority within a preregistered minus 2 point margin plus a<br/>measured cost or latency reduction — a potential EFFICIENCY case,<br/>separately preregistered and not a post-hoc rescue, :225"]
    O5["material disadvantage, or unacceptable unsupported answering —<br/>do NOT expand this configuration, diagnose the source, :226"]
    BAR --> O1
    BAR --> O2
    BAR --> O3
    BAR --> O4
    BAR --> O5

    EFF["The efficiency route has its own bar — at least 25 percent lower<br/>total cost per successful task at comparable coverage, with<br/>non-inferiority supported and absolute error rates disclosed<br/>PRD-028-does-loom-earn-its-complexity.md:228"]
    O4 --> EFF

    POWER["INVARIANT — 600 questions is an initial planning number, not a<br/>power guarantee, and demonstrating a lower bound above plus 5<br/>requires an assumed true effect greater than plus 5. If the<br/>required sample is unaffordable, the CLAIM narrows before the run<br/>rather than after seeing results.<br/>PRD-028-does-loom-earn-its-complexity.md:230"]
    BAR --> POWER

    OUT["The product outcome is one of four decisions — keep the current<br/>structure, simplify it, specialise it for a demonstrated task<br/>family, or invest in corpus quality instead of serving complexity<br/>PRD-028-does-loom-earn-its-complexity.md:290"]
    O1 --> OUT
    O3 --> OUT
    O5 --> OUT
```

## ES-12.5 The stop conditions, and the claims the next paper may not make
```mermaid
flowchart TB
    PHASES["Seven phases, each with its own gate<br/>../loom/docs/design/PRD-028-does-loom-earn-its-complexity.md:265-273"]
    P0["Phase 0 candidate inventory — gate: a usable non-public corpus<br/>exists and its handling is authorised, :267"]
    P1["Phase 1 corpus pilot — gate: independent gold is achievable and<br/>representation parity passes, :268"]
    P3["Phase 3 scale and lock — gate: the primary contrast and the<br/>margins are FROZEN before sealed-test access, :270"]
    PHASES --> P0 --> P1 --> P3

    STOP["STOP OR REDESIGN if the corpus is largely public material with<br/>renamed titles, if gold is generated solely from the Loom's own<br/>graph, if the baselines lack equivalent facts, if the oracle arm<br/>fails because the questions are ill-defined, or if evidence<br/>cannot be retained<br/>PRD-028-does-loom-earn-its-complexity.md:279"]
    P3 --> STOP

    NOTFAIL["INVARIANT — a strong flat baseline MATCHING the Loom is an<br/>informative result and is explicitly NOT a reason to change the<br/>endpoint. PRD-028-does-loom-earn-its-complexity.md:279"]
    STOP --> NOTFAIL

    FORBID["The next paper may NOT infer general ontology superiority from<br/>beating closed book, may not claim uncontaminated training from<br/>low unaided scores, may not claim absence of reasoning from a<br/>negative gain over copy, and may not claim enterprise<br/>generalisation from a single synthetic organisation<br/>PRD-028-does-loom-earn-its-complexity.md:288"]
    NOTFAIL --> FORBID

    OPENIN["OPEN — the PRD does not assume an eligible private corpus exists.<br/>The candidate corpus, the owner permissions, the empirical token<br/>profile, the checkpoint identities, the labour budget and the<br/>acceptable business error costs all remain to be supplied.<br/>PRD-028-does-loom-earn-its-complexity.md:294"]
    P0 --> OPENIN

    COST["DEBT already visible in the plan — about 200 reviewer-hours of<br/>blinded human review per primary model before adjudication, which<br/>the record itself calls potentially more constraining than GPU<br/>capacity. PRD-028-does-loom-earn-its-complexity.md:277"]
    P3 --> COST
```

## ES-12.6 Why the facade can be told not to ground, and what that costs the evidence
```mermaid
sequenceDiagram
    autonumber
    participant CON as a consumer with a mixed subject
    participant FAC as the facade door<br/>../loom/docs/design/ADR-139-per-request-scaffold-opt-out.md:23
    participant GATE as the lexical confidence gate<br/>../loom/docs/design/ADR-138-confidence-surfacing-contract.md:31
    participant MOD as the model behind the door

    Note over CON,FAC: The incident that produced this record: a repository explainer<br/>drafting a section about a test script sent prose containing the<br/>words node and verification, ADR-139-per-request-scaffold-opt-out.md:13-16
    CON->>FAC: a request about a subject the ontology does NOT cover
    FAC->>GATE: score the packet against the corpus
    GATE-->>FAC: a blockchain class scored far above the verbatim threshold
    FAC-->>CON: the corpus answer, served in milliseconds with zero<br/>completion tokens, ADR-139-per-request-scaffold-opt-out.md:15-16

    Note over CON,FAC: The consumer had NO way to say not this corpus: declining the<br/>verbatim serve still injected the scaffold, and the only<br/>alternative was to bypass the door entirely<br/>ADR-139-per-request-scaffold-opt-out.md:17-19

    CON->>FAC: the same request with the scaffold option set to false
    FAC->>MOD: forward the body unchanged, no retrieval, no injection,<br/>no verbatim serve, the private field stripped<br/>ADR-139-per-request-scaffold-opt-out.md:23-25
    MOD-->>FAC: an ordinary completion
    FAC-->>CON: served mode passthrough, grounding status passthrough,<br/>corpus_backed false, injected tokens zero<br/>ADR-139-per-request-scaffold-opt-out.md:26-29

    Note over FAC: INVARIANT — absence of the key, or any value other than the<br/>boolean false, leaves the scaffold ON. No header and no<br/>environment switch: the choice belongs to the request.<br/>ADR-139-per-request-scaffold-opt-out.md:31-32
    Note over GATE: INVARIANT for the evidence — a passthrough is NOT recorded in<br/>the confidence window and does not mark the generation as served,<br/>so benchmarks can separate the corpus had nothing from the caller<br/>did not ask. ADR-139-per-request-scaffold-opt-out.md:29-30,40-41
    Note over MOD: INVARIANT — the direct model port stays an implementation detail<br/>behind the door, not a documented consumer path<br/>ADR-139-per-request-scaffold-opt-out.md:33-34. see AB-24.6
```

## ES-12.7 The positioning claim the evaluation is built to test
```mermaid
flowchart TB
    CLAIM["THE CLAIM — models score about 0.36 from parametric memory alone<br/>on in-domain graph-derived questions and about 0.94 when given the<br/>curated context, which is offered as proof the knowledge is<br/>genuinely private<br/>../loom/docs/design/LOOM-POSITIONING.md:30-32"]

    CEIL["THE HONEST READING — a curated ontology CONTAINS the answers by<br/>design. Measured as a copy ceiling the model tracks just under it,<br/>and read as a product that is exactly right: the answer is<br/>trustworthy because the source is.<br/>../loom/docs/design/LOOM-POSITIONING.md:36-42"]
    CLAIM --> CEIL

    MULTI["THE BAR IS MULTIVARIATE — a single in-domain recall number is not<br/>the target. Four axes must clear at once: local grounding, general<br/>questions not made jagged, out-of-domain fallback, and cost with<br/>provenance.<br/>../loom/docs/design/LOOM-POSITIONING.md:52-59"]
    CEIL --> MULTI

    RISK["The interference risk is why selective injection exists: injected<br/>context can DISPLACE a model's own correct parametric knowledge<br/>when it is weak or off-topic.<br/>../loom/docs/design/LOOM-POSITIONING.md:61-64"]
    MULTI --> RISK

    PRE["PREREGISTRATION ANSWER — PRD-028 is the design that would test<br/>the first axis properly, against a strong flat baseline at matched<br/>access rather than against closed book.<br/>../loom/docs/design/PRD-028-does-loom-earn-its-complexity.md:27,29"]
    MULTI --> PRE

    CAVEAT["DOC-DRIFT the record flags against itself — privateness alone does<br/>not prove absence from training, and poor closed-book performance<br/>does not prove absence either.<br/>PRD-028-does-loom-earn-its-complexity.md:45"]
    CLAIM --> CAVEAT

    DEBTC["DEBT — this corpus cannot pin the loom repository. The generator's<br/>prefix table has no ../loom entry, so every citation on this page<br/>is checked against the working tree rather than against the<br/>declared revision in the frontmatter.<br/>scripts/diagram-index-gen.cjs:240,259. see ES-90.6"]
    PRE --> DEBTC
```
