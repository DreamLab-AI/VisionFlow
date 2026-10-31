---
title: Proposal application, identity and layout audit
status: in-progress
date: 2026-10-31
type: explanation
---

# Proposal application, identity and layout audit

Agentbox and Ontomark enforce several useful guards, but their receipts do not always identify the same proposal and page. This pass traces Agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` into Ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41`, and assesses ADR-2109 and ADR-OM-007 with the limits below. Both inspected source sets were clean. The preceding [corpus-consumer audit](2026-10-31-corpus-consumer.md) covers ordinary status edits and the reproduced Git push race.

## Observed boundaries

[Reproducible probe](evidence/2026-10-31/proposal-apply-probes.cjs) and [results](evidence/2026-10-31/proposal-apply-probes.json) use the production adapter and installed Ontomark 0.7.0 against temporary minimal estate vaults. Git calls are deliberately refused by the fixture; attestation is an in-process recorder. The caller probes invoke the internal branches after the roster/admission boundary. They do not establish that a malformed request can pass relay admission, cryptographic verification or the real review UI.

| Case | Observed result | Meaning |
|---|---|---|
| Valid create | Creates A, stamps stable and human verification, records attestation. | The adapter argv works with the installed CLI. |
| Create with incorrect `proposal.digest` | Creates A and records attestation. | This path does not recompute the proposal digest. It checks optional `page_sha256`, which is a different contract, then sends only reconstructed Markdown to the CLI. |
| Create naming page A, whose staged title is B | Writes `knowledge/B.md`; adapter returns `page: A`. | The adapter checks the proposal page's slug against the IRI, but the CLI chooses the filename from the staged title. The returned page and actual file can disagree. The fixture keeps the same resource IRI and computes a valid digest for that staged text. |
| Valid amendment with expired `stale_after` | Applies and records attestation. | Expiry is not enforced by this mutation boundary. Whether the upstream decision remains valid must be established separately. |
| Amendment with incorrect proposal digest | Refuses `DIGEST_MISMATCH`, leaves the page unchanged, no attestation. | The CLI recomputes the proposal's own semantic digest. |
| Valid amendment with a different decision digest | Applies the proposal; returned digest names that proposal, recorded attestation names the decision's other digest. | Internal proposal integrity does not establish agreement with the caller's case identity. |
| Explicit promote branch, matching request/decision `d` tags but different proposal digest | Applies, with the same receipt disagreement. | `_applyOntologyBranch` checks request-tag equality and subject IRI, but not the proposal-digest equality that the approve branch checks. |
| Approve branch, same discrepancy | Refuses before writing or attesting. | `_ontologyApprove` explicitly compares the canonical proposal digest with the decision tag. |
| Amendment with another edit outside its hunk | Applies and preserves the other edit. | This is the documented hunk-based contract, not a whole-page approval precondition. |
| Amendment with another edit inside its hunk | Refuses `STALE`, no attestation. | Exact hunk context protects the edited region. |

A recovery worker or reviewer cannot safely join a receipt to its approved content when page or digest fields disagree, even without a demonstrated signature bypass. Acceptance should prove equality between the reviewed subject, request tag, proposal digest, applied output and attested case on every supported outcome. Creation should compare the parsed title/output identity with the approved page before persistence. This audit records requirements; it does not change product code.

## What the source actually guards

`management-api/lib/ontology-apply.js:560-647` reconstructs creation text from added diff lines, checks the proposal page/IRI and optional page hash, then calls `ontomark create`. Its parser at `:509` does not enforce all unified-diff header/hunk-count semantics. This pass did not probe malformed headers. `crates/ontomark-compile/src/write/create.rs:224-350` derives the target from the staged `title`, checks filename/IRI/slug collisions and validation, then uses exclusive file creation. It does not receive a proposal digest. The adapter's success result uses the original proposal page rather than the CLI's returned page.

Amendment stages the proposal unchanged (`ontology-apply.js:692`) and calls `ontomark apply`. `write/apply.rs:305` requires one document, amend kind, a matching semantic digest and one non-renaming patch. `:208` checks the current resource IRI, applies exact diff hunks to the rendered page, appends the verification and refuses newly introduced validation errors on that page. `write/proposal.rs:369` hashes level, IRI, page, hypothesis and diff; grouped pages and create kind extend the digest. Proposer, generation, expiry and blocker metadata deliberately lie outside it.

Before renaming a synced temporary file, the amendment writer re-reads the target (`write/apply.rs:395`). That catches changes made before the re-read, but the comparison and rename are separate operations. It is not a filesystem compare-and-swap or a lock over other writers. No interleaving at that particular interval was reproduced here. Creation's exclusive open prevents replacement of an existing filename; it does not make a new file invisible to readers until all bytes are written. Crash recovery and directory durability were not tested.

## ADR-2109 section assessment

| Section or decision | Assessment |
|---|---|
| Context | Historical ownership and retired HTTP route explain the design; current corpus verbs belong to Ontomark. Do not read the original “binary does not exist” statement as current. |
| Decision 1 | Apply logic remains in `ontology-apply.js`; orchestration selects outcomes and request identity. Injection points enabled the real-CLI boundary probes. |
| Decision 2 | Array argv remains. `binFor` now splits Ontomark corpus verbs from vault governance transport and the environment includes `ONTOMARK_BIN`. Later appendices supersede the old binary list. Full secret/environment isolation was not reassessed. |
| Decision 3 | Derived key counts and no-op demotion are implemented. The old `knowledge/pages` path is stale; current reads use `knowledge`. The claimed race closure is too broad: [ordinary body-drift probes](2026-10-31-corpus-consumer.md) pass the cardinality guard. |
| Decision 4 | Candidate/page-hint matching uses the basename slug. This does not independently prove equality with the page's frontmatter resource. Amendment performs that additional CLI check; creation has the separate title/receipt mismatch reproduced here. |
| Decision 5 | Explicit promote/demote reads the signed-response IRI; approve derives it from the request after case/panel/digest checks. Complete signature admission and UI correspondence remain outside this pass. |
| Decision 6 | Post-write ledger failures return metadata. “Nothing is attested that was not applied” needs its documented no-op-demotion exception; `editAndAttest` intentionally records that decision. Digest agreement across the amendment receipt and ledger is not enforced on explicit promote. |
| Decision 7 | The orchestration branch catches application errors. A returned not-applied receipt is not evidence of a working retry/reconciliation service. |
| Decisions 8–10 | `ontology-propose.js:214-304` builds vault argv, defaults to content/dry-run and unwraps blockers. Whole elevation selection, signed publication and relation-resolution behaviour remain for a separate trace. These clauses are not fully reverified. |
| Decision 11 | `action` precedes legacy `outcome` in the caller. |
| Decision 12 | Real CLI probes and ten apply integration tests support the basic amendment path. Approve checks proposal/tag digest equality; explicit promote does not. Hunk staleness permits unrelated edits and is not a full-page hash. Grouped proposals remain refused. |
| Decision 13 | The roster and own-key refusal are explicit source checks. The new probes run below this boundary and do not certify it or cryptographic admission. |
| Decision 14 | Commit is limited by file path; push is ordinary branch publication. The [previous real-Git probe](2026-10-31-corpus-consumer.md#commit-and-attestation-boundaries) defeats the “only own commit” claim by inserting another writer after the ahead-count check. |
| Consequences | Mutation ownership is implemented. Older statements about future binaries/tests are dated; the latest acceptance boundary correctly excludes a live forum decision in the running image. |
| Verification and 22 September amendment | Historical suite and end-to-end receipts were read, not rerun wholesale. Our 12 CLI tests and ten boundary cases add narrower, current evidence and counterexamples. |
| 7 October re-verification | Captures amendment/approve/roster/commit additions; the guard and push limitations above qualify the universal claims. |
| 30 October close-out | Records a historical broken CLI dispatch. Current `binFor` and real CLI probes support the source-level correction. |
| 31 October re-verification | Supports the dispatch migration. Installed binary identity does not prove which adapter the running management service loaded. |
| 31 October close-out | The accepted/complete/staged axes describe the recorded migration close-out. They do not eliminate these new identity findings or establish full lifecycle correctness. Owner status was not changed. |

## ADR-OM-007 section assessment

| Section or decision | Assessment |
|---|---|
| Context | Root-relative identity and emitted URL identity are separate. Layout migration is not itself an IRI migration. |
| Decision 1 | `input/page.rs:444-470,584` walks the supplied vault root recursively. There is no special compatibility handling for `pages/`. An old `pages/` directory remains a normal subdirectory, so IDs acquire that prefix. |
| Decision 2 | Exclusion records are present. The built-in furniture list applies to the first path component; hidden components are excluded at any depth. It does not exclude every nested directory named templates. |
| Decision 3 | Emitted-path compatibility is a separate build contract. This pass did not repeat the historical 9,416-page byte comparison. |
| Decision 4 | The local CLI exposes the migrated verbs; create/apply integration tests pass. The 23-query legacy parity receipt is historical, not a current whole-corpus equivalence result. Governance transport still stays in vault. |
| Decision 5 | Deleting retired content is an authoring policy; the loader's hidden-directory exclusion does not enforce deletion. No live content was removed or rewritten. |
| Evidence | Preserve the historical parity counts and generation exceptions as dated receipts. No current independent full-corpus parity run was made. |
| Consequences | The statement that an unflattened vault can load its root and “find no pages” is inaccurate for an ordinary recursive `pages/` subtree. A [temporary layout probe](evidence/2026-10-31/layout-probe.json) found `pages/Legacy`, excluded root `templates/Hidden`, and included `nested/templates/Nested`. It used a plain vault; the shared walker confirms the rule. The shared-pin rollout obligations remain separate from local version checks. |

## Validation and remaining scope

`cargo test --locked --offline -p ontomark --test apply_cli --test create_cli` passed **12 tests** ([log](evidence/2026-10-31/proposal-cli-suite.txt)). The package is named `ontomark`; an initial attempt using the directory name `ontomark-cli` selected no package and was corrected. The ten isolated adapter/CLI cases above include passing controls and refusal controls, with no live corpus, remote Git or HTTP writes.

[OM-07](../diagrams/ontomark/07-governed-create-and-amend.md) now maps the consumer-specific checks. AB-25's old generic write/push description is qualified and links there. Other diagrams' historical revision stamps are not evidence that their present runtime is current. Remaining work includes relay and review-UI admission, elevation/proposal construction, replay/recovery, full filesystem concurrency, ingest-to-proposal activation, and the rest of the estate ADR/diagram queue.

The new OM-07 and qualified AB-25 topics passed strict checks with 49 and 43 revision-resolved citations respectively, no warnings or fallbacks, and all eight diagrams rendered. The [whole-corpus citation run](evidence/2026-10-31/proposal-corpus-citations.txt) covers 159 topics and 1,529 diagrams: 23,717 revision-resolved citations, zero working-tree fallbacks or unresolved citations, and nine explicitly allowlisted generated-file references. Generated indexes and the revision receipt were refreshed. As before, allowlisted build files were copied from the primary checkout; they are not commit-backed evidence.
