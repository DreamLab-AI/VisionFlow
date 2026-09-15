---
id: EXP-AC-001
parent_spec: PRD-augmentation-conditions FR1
linked_adrs: [ADR-2010]
priority: high
regression_critical: true
evidence_category: executable
status: accepted
authored_by: pair
---

## Expectation: The compatibility matrix grades every substrate against C1–C6 with a code citation per cell

`docs/architecture/compatibility-matrix.md` contains a table headed "Augmentation conditions" with exactly six condition rows (C1–C6) and one status column per substrate that publishes or consumes kind 31403 (nostr-rust-forum, agentbox, VisionClaw). Every non-`absent` cell carries a `file:line` citation whose path exists in the sibling checkout. The closeout README's CP-05 row names C2 and C3; CP-07 names C4. `docs/terminology.md` defines "augmentation condition", "task-property triple", "vacuous verification" and "calibration sample".

### In scope
- Table shape and header text
- Citation path existence for every cited cell
- CP-05 / CP-07 references by condition id

### Out of scope (intentionally)
- Whether a `measured` status is *true* at runtime (that is each substrate's EXP)

### Counter-examples (must NOT happen)
- A `measured` or `partial` cell citing a document rather than code or a receipt
- A cell citing a path that does not exist
- Terminology defining a term with a sentence that could apply to any governance system
