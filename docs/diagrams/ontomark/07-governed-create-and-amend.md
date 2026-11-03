---
id: OM-07
title: Governed creation and amendment — proposal, file and receipt identity
area: ontomark
governing:
  - ../project/agentbox/docs/adr/ADR-2109-governance-decisions-apply-through-vault-edit.md
adrs: [ADR-2109, ADR-OM-007]
sources:
  - ../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js
  - ../project/agentbox/management-api/lib/ontology-apply.js
  - ../ontomark/crates/ontomark-compile/src/write/create.rs
  - ../ontomark/crates/ontomark-compile/src/write/apply.rs
  - ../ontomark/crates/ontomark-compile/src/write/proposal.rs
verified_commit: {agentbox: c57d9c0508f4044915f9aecea4b9aa5629cc6a35, ontomark: 4b9d15df48056971321a62b8c4e905f4536c74db}
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](13-certificate-transaction-and-identity-repairs.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## For developers

This topic follows the mutation consumer after a governance decision reaches Agentbox. It distinguishes a proposal's own digest from the decision's case tag, and a proposal's page from the filename selected by creation. The [31 October audit](../../estate-review/2026-10-31-proposal-application.md) includes real-CLI probes and explains which upstream checks were not exercised. Source pins identify the inspected implementation; they do not certify a deployed forum journey.

## For the business

A reviewer needs to know which approved change reached which page. The application can make a valid local change while its receipt names a different page or case. The diagrams show where that disagreement arises and why a successful write alone does not prove that the published and recorded outcomes agree.

## OM-07.1 Outcome selection and case identity

```mermaid
flowchart TD
    A["approve outcome<br/>local-process-manager.js:218"] --> Q["Resolve same case and ontology panel;<br/>proposal digest must equal decision tag<br/>local-process-manager.js:379,397,400"]
    P["promote or demote outcome<br/>local-process-manager.js:226"] --> R["Roster and own-key refusal<br/>local-process-manager.js:232,503"]
    Q --> R
    R --> B["Resolve request; compare request tag and signed IRI<br/>local-process-manager.js:419,444,450"]
    B --> D["Dispatch by outcome and proposal kind<br/>ontology-apply.js:372"]
    D --> C["create proposal: reconstruct staged Markdown<br/>ontology-apply.js:560,578"]
    D --> M["promote amend: stage proposal JSON<br/>ontology-apply.js:692"]
    D --> E["ordinary status edit or demotion<br/>ontology-apply.js:410"]
    B -.-> X["TENSION: explicit promote checks request tag,<br/>but lacks approve's proposal-digest equality check<br/>local-process-manager.js:444,400"]
```

**What it shows.** Approve and explicit promote have different checks before reaching the same amendment writer. A valid proposal digest inside the CLI does not prove that it is the case digest used by the caller.

**Why it is this way.** ADR-2109 added approve-to-amend handling after explicit promotion. The source retains the equality check in that specialised branch. The probe reproduces the discrepancy below signature admission; it does not establish a relay bypass.

## OM-07.2 Creation chooses its filename from the staged title

```mermaid
flowchart TD
    P["Proposal page slug must answer to IRI<br/>ontology-apply.js:570"] --> T["Added diff lines become Markdown;<br/>optional page_sha256 and resource checked<br/>ontology-apply.js:578,586"]
    T --> CLI["ontomark create receives staged file,<br/>docs=1 and verification fields<br/>ontology-apply.js:485"]
    CLI --> NAME["Parse title; title determines target filename<br/>write/create.rs:248,258"]
    NAME --> V["Validate; filename, IRI and slug collisions<br/>write/create.rs:284,289"]
    V --> W["Exclusive creation refuses existing target<br/>write/create.rs:322,333"]
    W --> OUT["CLI outcome names actual title and path<br/>write/create.rs:304"]
    OUT --> R["Adapter receipt uses original proposal page<br/>ontology-apply.js:634"]
    R -.-> GAP["DRIFT: staged title B can write B.md<br/>while adapter receipt names proposal page A<br/>ontology-apply.js:570 vs write/create.rs:258"]
```

**What it shows.** The title and the proposal page travel through different checks. The adapter's optional page hash does not replace a semantic proposal digest check, and the creation CLI receives no proposal object to verify.

**Why it is this way.** The CLI owns validated exclusive creation; the host owns the approved proposal. ADR-2109 describes both as one source of truth, but the isolated A/B fixture shows that the identity fields can disagree at their boundary.

## OM-07.3 Amendment guards and persistence receipts

```mermaid
flowchart TD
    P["Proposal's own semantic fields hash to stated digest<br/>write/proposal.rs:369; write/apply.rs:305"] --> ONE["One amend document and non-renaming patch<br/>write/apply.rs:330"]
    ONE --> BASE["Current resource IRI and exact hunk context<br/>write/apply.rs:229,242"]
    BASE --> VALID["Append verification and refuse introduced errors<br/>write/apply.rs:258,282"]
    VALID --> TEMP["Write synced temporary file;<br/>re-read original before rename<br/>write/apply.rs:395"]
    TEMP --> COMMIT["Commit one path; check ahead count<br/>ontology-apply.js:828,872"]
    COMMIT --> PUSH["Ordinary branch push<br/>ontology-apply.js:879"]
    PUSH --> LEDGER["Attest caller's case and digest;<br/>failure is returned after write<br/>ontology-apply.js:742,900"]
    BASE -.-> H["INVARIANT: staleness is hunk context;<br/>unrelated page edits can survive<br/>write/apply.rs:42"]
    TEMP -.-> RACE["OPEN: compare and rename are separate;<br/>no writer lock at this boundary<br/>write/apply.rs:401,411"]
    PUSH -.-> OTHER["TENSION: writer can add a commit after count;<br/>push then includes it<br/>ontology-apply.js:872,879"]
```

**What it shows.** Proposal integrity, current-text compatibility, file persistence, Git publication and attestation are separate checks and outcomes. The audit reproduces hunk scope and the Git interleaving; the file re-read/rename interval remains a source-level concern.

**Why it is this way.** Ontomark owns local application and Agentbox owns publication and ledger calls. ADR-2109 keeps ledger failure from undoing a completed write. That boundary requires distinct receipts; it does not make the complete sequence atomic.

**Open:** Full signed-decision admission, replay policy and recovery are not certified by these local probes. See the [assessment and evidence](../../estate-review/2026-10-31-proposal-application.md).
