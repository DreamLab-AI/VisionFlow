---
id: AB-41
title: Process identity, promotion and Dream adapter repairs
area: agentbox
governing:
  - ../project/agentbox/docs/adr/ADR-2032-daemon-identification-argv-boundaries.md
  - ../project/agentbox/docs/adr/ADR-2109-governance-decisions-apply-through-vault-edit.md
  - ../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md
adrs: [ADR-2032, ADR-2109, ADR-2141]
sources:
  - ../project/agentbox/services/agentbox-ops/src/bin/ruflo-daemon-gc.rs
  - ../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js
  - ../project/agentbox/services/dream-engine/src/candidate_flow.rs
  - ../project/agentbox/services/dream-engine/src/estate.rs
  - ../project/agentbox/services/dream-engine/Cargo.toml
  - ../project/agentbox/services/dream-engine/Cargo.lock
verified_commit: cb08c91a83720a6f5ded8493cb36f9ba4ca53c86
---

## For developers

This scoped successor covers the repaired reaper, explicit promotion digest and Dream host seams. Historical topics retain their declared revisions; unrelated findings are not closed here.

## For the business

The source now protects against signalling a replacement process, mismatching a promoted patch and evaluating an unidentified archive. Image activation and end-to-end governance delivery remain separate acceptance steps.

## AB-41.1 Pin a process before signalling it

```mermaid
flowchart TD
    %% Source: ../project/agentbox/services/agentbox-ops/src/bin/ruflo-daemon-gc.rs:133
    A["Discover daemon"] --> B["Capture argv and start time; derive current workspace"]
    B --> C["Recheck workspace or TTL staleness before action"]
    %% Source: ../project/agentbox/services/agentbox-ops/src/bin/ruflo-daemon-gc.rs:102
    C --> D["Open kernel pidfd"]
    D --> E{"Captured process identity still matches?"}
    %% Source: ../project/agentbox/services/agentbox-ops/src/bin/ruflo-daemon-gc.rs:108
    E -->|Yes| F["Send SIGTERM through pidfd"]
    E -->|No or inaccessible| G["Refuse signalling"]
```

**What it shows:** The reaper no longer treats matching launcher text as proof of the originally discovered process. The owned-child regression exercises refusal and successful pidfd signalling.

**Why it is this way:** ADR-2032 requires stable identity. Failure to open a pidfd fails closed, and delivery is not a claim of confirmed shutdown. The estate-wide signalling inventory remains partial.

## AB-41.2 Explicit promotion binds its typed proposal

```mermaid
flowchart TD
    A["Admitted explicit promote decision"] --> B{"Carries typed patch proposal?"}
    %% Source: ../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js:454
    B -->|Yes| C{"Valid canonical digest equals signed d tag?"}
    C -->|No| D["Refuse before compiler invocation"]
    C -->|Yes| E["Continue existing page and proposal validation"]
    B -->|No| F["Retain legacy promotion handling"]
```

**What it shows:** Explicit promote now applies the digest equality already used by approve, with controls for wrong, absent and empty digests and ordinary promotion success.

**Why it is this way:** ADR-2109 binds a human's decision to the proposed change. This panel does not claim grouped application, deployment or a general signer-roster gate on every non-ontology dispatch.

## AB-41.3 Supply the reusable core with verifiable host inputs

```mermaid
flowchart TD
    %% Source: ../project/agentbox/services/dream-engine/src/candidate_flow.rs:73
    A["Candidate archive on annexe"] --> B["Create Git snapshot and require expected tree hash"]
    B --> C["Published dream-core 0.4.1 evaluates and rechecks final trees"]
    %% Source: ../project/agentbox/services/dream-engine/src/candidate_flow.rs:208
    D["Apply-repair needs source"] --> E["Pass estate denylist into shared excerpt API"]
    E --> F["Redact complete repair prompt before model call"]
```

**What it shows:** The host adopts the registry release instead of copying formatter or source-scanner logic. The full adapter suite, including 11 whole-night tests, runs against registry checksums.

**Why it is this way:** ADR-2141 keeps reusable mechanisms upstream and estate policy in the adapter. A verified final tree does not sandbox evaluator commands, and local tests do not establish a live annexe run.
