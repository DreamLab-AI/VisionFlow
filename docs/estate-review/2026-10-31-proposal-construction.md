---
title: Proposal construction, consequence scope and grouped application
status: in-progress
date: 2026-10-31
type: explanation
---

# Proposal construction, consequence scope and grouped application

An existing page's supposedly immutable `resource` IRI can be changed by a blocker-free signed proposal and the real Ontomark apply command. Separately, `vault propose` signs grouped proposals that Agentbox refuses before invoking a writer. Case identity excludes the assessment context, and consequence counts include version metadata while excluding blank-node-bearing logical changes.

This pass follows [external-source ingest](2026-10-31-external-source-ingest.md) into proposal construction and compares it with the [application](2026-10-31-proposal-application.md) and [signed ingress](2026-10-31-governance-ingress.md) findings. It reads ADR-2116 and ADR-2147 section by section. Sources: VisionClaw `75b72416bac176ca8fde715cac3bd7c77657b3e7`, Agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`, forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`, Ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41`. [Hashes](evidence/2026-10-31/proposal-construction-sources.json) verify inspected files against those revisions. Unrelated dirty work was preserved.

## Executed proposal and consumer cases

The [Rust probe](evidence/2026-10-31/proposal-construction-probe.rs) invokes the freshly built CLI with `--dry-run`, a disposable signing key, a synthetic corpus and local Git commits. [Signed dry-runs](evidence/2026-10-31/proposal-construction-probe.json) preserve proposal bodies and event identities. The [consumer probe](evidence/2026-10-31/proposal-group-consumer.cjs) passes those actual grouped payloads to the production adapter, refusing any CLI, Git or HTTP invocation. [Consumer results](evidence/2026-10-31/proposal-group-consumer.json).

| Fixture | Result | Boundary established |
|---|---|---|
| Add `Alpha is-a Gamma`, omit level | Content-level, signed, no blockers; two new comparable triples including versionIRI | The producer does not automatically label a taxonomy amendment schema-level. |
| Keep that patch, commit `Gamma is-a Beta` elsewhere, propose again | Same case digest and diff; different generation, consequence summary and event ID. New comparable triples rise from two to three. | Case identity excludes assessment context; signed event identity includes the body. |
| Change `Alpha defines-as [requires Gamma]` to `[requires Beta]` | Non-conservative verdict, one added and one lost versionIRI triple; four blank-node-bearing triples not compared | The verdict detects changed metadata without comparing the changed restriction. |
| Amend a page already missing required status | Signed, no blockers; MISSING_REQUIRED and MISSING_STATUS retained as pre-existing | Delta validation works as documented. Postable does not mean the whole proposed corpus is valid. |
| Three-page quality amendment | Signed as content-level grouped amend, no blockers | Grouping does not tighten the level. |
| Add a new Epsilon page to that group | Signed as one four-page create proposal | Mixed create/amend groups are admitted by construction. |
| Pass both groups to the adapter | Both refused on page/IRI identity before external calls | Construction and signing do not establish an available apply path. |
| Change Alpha's resource to a fresh replacement IRI | Signed content amendment, no blockers, subject still names the original IRI; real Ontomark apply succeeds and the file contains the replacement | The documented immutable-resource blocker is missing across this producer/direct-writer path. |

No relay, live corpus, actual reviewer key or HTTP service was used. Proposal construction left the target unchanged; the final direct apply deliberately changed only its disposable fixture. The adapter probe starts after ingress; the earlier ingress audit supplies separate admission evidence, not a deployed grouped journey.

## Immutable resource is not enforced across amendment

The [governing corpus format](../../../project/docs/VAULT-corpus-format.md) Invariant 3 says resource is minted once and a changed resource is a `RESOURCE_MUTATED` blocker. The fixture replaces `urn:ngm:class:alpha` with a fresh `urn:ngm:class:replacement` in an existing Alpha page. Construction returns a signed content amendment with no blockers and the original IRI as its subject. Passing that actual proposal to `ontomark apply --expect docs=1` succeeds and changes the file's resource.

Creation checks IRI collision (`propose.rs:328-335`), but amendments are assessed as two independent corpus states with a delta of findings. No cross-state resource immutability check occurs there. The apply command compares the **current** page resource to the proposal subject (`ontomark-compile/src/write/apply.rs:229-235`), then patches, parses and checks introduced validation errors (`:242-284`). It does not compare the patched resource to the original. A syntactically valid replacement IRI therefore survives both stages.

This establishes the producer/direct-writer contract failure. It does not establish what a deployed reviewer approves or which attestation a live adapter would emit. Acceptance must enforce resource continuity across before/after state, with an explicit governed identity-migration design if replacement is ever intended. Creation collision checks and matching the old subject do not enforce that invariant.

## Grouping crosses an unsupported consumer boundary

The producer reads full proposed pages, drops no-ops, assesses both corpus states once, and gives a group its own `urn:ngm:proposal:<title>` subject (`propose.rs:524-568,641-658,706-738`). Its `page` repeats that subject; `pages` lists the note IDs. Any creation makes the group kind create (`:590-594`). The CLI advertises this as one decision spanning hundreds of pages; its integration test covers construction and JSON framing.

Agentbox expects one page whose slug answers to the signed subject (`ontology-apply.js:570,699`). The actual groups fail there first: a proposal IRI is not a note filename. Dedicated guards immediately afterwards also refuse grouped create/amend (`:574-575,702-703`). This is deliberate consumer refusal, not partial application. The pipeline nevertheless invites review of a scope that cannot reach the promised outcome through this consumer. Ingest's multi-target manifests can reach this boundary.

Acceptance needs one supported scope across preparation, signing and application: either an all-or-nothing grouped apply/recovery path, or identification of unsupported scope before inviting a signature. Silently splitting a group would change the judgement the reviewer approved.

## Level is mostly declared

CLI level defaults to content (`main.rs:214-215`). The builder starts with that level and escalates only **created** pages declaring unknown or provisional keys (`propose.rs:369-377,576-588`). Amendments never enter that escalation loop. The definition-filler change and grouped amendment remain content-level. `nostr::task_properties` maps level directly to stakes: content bounded, demotion significant, schema critical (`nostr.rs:84-94`).

The public Schema description includes taxonomy shape (`vault-core/src/proposal.rs:24-26`), but no general diff-to-level classifier enforces that interpretation here. Consequence count/verdict does not raise the level. This is a policy gap, not evidence that every new parent must be high risk. The ingress audit found downstream floors follow the request's level tag, so they cannot be assumed to correct a lower declared level.

## Case identity and evidence identity differ

`compute_digest_of_kind` hashes level, subject IRI, page, hypothesis, diff, grouped page set and create marker (`proposal.rs:364-389`). Generation, deadline, blockers, pre-existing findings and consequences sit outside that digest. This supports stable patch identity; it does not identify one assessment or approval envelope.

The two-generation fixture demonstrates this without tampering. `nostr::action_request` serialises the complete proposal into event content and `nostr::sign` signs it (`nostr.rs:151,189-198`), producing different event IDs. This is **not** a cryptographic omission from the signed request. Review/replay/receipt consumers must retain the signed request identity when they mean the specific evidence reviewed, and define when changed corpus context requires reassessment. The prior application review establishes hunk compatibility rather than a full generation lock.

## Version metadata enters the consequence comparison

The emitter adds content-derived versionIRI after building the graph (`build/turtle.rs:493-514`). Assessment certifies that graph (`propose.rs:118-122`). The certificate engine compares all non-blank-node-bearing asserted and derived triples (`rdfs-certificate/src/lib.rs:603-659`), including the header. Its consumed crates.io 0.1.0 source is byte-identical to the inspected local source: [dependency identity](evidence/2026-10-31/proposal-certificate-dependency.json).

The definition-filler fixture's sole compared change is versionIRI, yielding +1/−1 and a non-conservative verdict. Four blank-node-bearing triples remain outside comparison. ADR-2147 says a restriction-only filler change reads conservative over named closure; this emitted-graph/CLI path contradicts that example because the version header also changes. Blank-node exclusion works as written; the discrepancy appears when the versioned graph enters the comparison.

“Consequences” includes assertions, annotations and metadata, as the closure comparison specifies. It does not count solely new inferred class relationships. Review should distinguish these categories and expose the excluded logical scope. Removing metadata counts alone would still not establish full EL conservativity. This small probe does not replace the historical full-corpus measurements or independently checked certificates.

## Reviewer presentation

The forum parser retains a flat PatchProposal as a field map (`panel_registry.rs:462-480`); the decision card pretty-prints it in a scrollable proposal block (`pages/governance.rs:739-750`). Consequences have a source-level route into that card as JSON. No browser/deployed card was checked.

ADR-2147 also cites VisionClaw's `OntologyProposalList.tsx` chip/details component. Its formatting tests pass, but its store uses the separate `/api/ontology/proposals` paths and consumes consequence data from status updates (`useOntologyStore.ts:336,414,609`). That component does not prove the forum's presentation or a live bridge. The forum field-map/render code is the relevant evidence here; usability remains unverified.

## ADR-2116 assessment

| Section | Assessment |
|---|---|
| Context | Retirement is supported by source; the named replacement now exists. Historical outline omits current create/amend and grouped limits. |
| Decisions 1–2 | Handler returns 410 guidance without route-specific auth/rate wrapping (`ontology_agent_handler.rs:239-254,381-386`). No current deployed request in this pass. |
| Decisions 3–4 | Handler no longer consumes the mutation service or proposal DTO. Shared mutation facilities still exist elsewhere; this route's retirement does not prove every other write surface is absent. |
| Decision 5 | Current status capabilities omit ontology_propose. |
| Decision 6 | REC-1 retains authenticated/anonymous retirement and other guards. Server tests inspected, not rerun here. |
| Consequences | “Replacement does not exist yet” is historical and superseded. The 410 body still directs callers to `vault edit --expect docs=1,blocks=1`, conflicting with the current Ontomark migration. |
| Verification and September/October re-verifications | Dated build/test/deployment receipts remain historical evidence. Current producer tests and dry-runs prove construction. |
| 30 October close-out | Updates retirement and CLI availability to complete/live. Does not prove application of all advertised scopes; groups are a counterexample. Owner status unchanged. |

## ADR-2147 assessment

| Section | Assessment |
|---|---|
| Context | Independence and visibility remain coherent requirements. Historical performance/parity numbers not remeasured. |
| Decision 1 | Published certificate crate is consumed; source matches the local crate. Workflow downloads a pinned checker and verifies its hash. Unit tests do not imply independent checker execution. |
| Decision 2 | Certify compares named subclasses with Whelk and fails on mismatch (`main.rs:535-579`). Full-corpus/checker execution not repeated. |
| Decision 3 | Both states are certified; delta is advisory. “Only reasoning blocker” is too narrow: relevant-existential cap refusal also blocks, even when present before (`propose.rs:196-201`). Version metadata affects verdicts; case digest does not identify the full assessment. |
| Decision 4 | Shape/evaluator claims are outside this pass's executed scope. Older SHACL receipt is not current verification. |
| Decision 5 | Inspected adapter introduces no upstream lifecycle/service. Sole-write-route claim needs the separate mutation-surface audit. |
| Consequences | Restriction-filler conservative example contradicted by versioned-graph fixture. Forum JSON transport supported by source; tested VisionClaw chip is a different surface. EL/blank-node limits remain material. |
| Verification | Suite counts, corpus timings and external audit remain dated. Current tests establish narrower producer/formatter behaviour; optional checker tests are not counted as independent verification. |
| 30 October re-verification | Extraction and off-by-default rank match source. Workflow still contains checker and uploads; no new CI run/published activation checked. Staged status retained. |

## Reproduction and coverage

Focused validation passed 47 tests: [30 builder tests](evidence/2026-10-31/proposal-builder-suite.txt), [11 proposal-type tests](evidence/2026-10-31/proposal-core-suite.txt), [two certificate-adapter tests](evidence/2026-10-31/proposal-certificate-suite.txt), [one CLI integration test](evidence/2026-10-31/proposal-construction-cli-suite.txt) and [three VisionClaw consequence-formatting tests](evidence/2026-10-31/proposal-consequence-ui-suite.txt). The new probes add seven producer cases, two refused grouped-consumer calls and one successful disposable resource-changing apply. None of these tests executes the independent Lean checker, the deployed forum card or a live approval/apply/attestation journey.

Compile the Rust probe with `rustc --edition=2021`, the checkout's serde_json rlib via `--extern`, and `-L dependency=<project>/target/debug/deps`; pass the freshly built `target/debug/vault` and the Ontomark executable as arguments. Run from an executable filesystem and save stdout. Then run `node proposal-group-consumer.cjs <agentbox> <saved-json>`. The JavaScript probe is minimal glue for the production JavaScript adapter. Writes stay in disposable fixtures; no external application is contacted.

[VC-41](../diagrams/visionclaw/41-proposal-construction-and-consumer-scope.md) maps the current boundary. VC-20.15 corrects the sole-reasoning-blocker claim at its historical pin and links forward. The producer's propose, proposal type and Turtle source files are unchanged between that pin and current HEAD. Estate review, deployed reviewer usability, independent certificate activation and grouped apply/recovery remain open.

Diagram validation: VC-41 and VC-20 passed strict citation checks (63 and 123 references); all 15 diagrams rendered. The [whole-corpus run](evidence/2026-10-31/proposal-construction-citations.txt) passed across 162 topics and 1,538 diagrams, with 23,874 revision-resolved citations, zero fallbacks/unresolved citations and nine explicitly allowlisted generated-file references. The latter are not revision-backed evidence. The verification anchor is fresh for all 38 declared revisions. These checks establish citation resolution and structural validity, not whole-estate semantic correctness or deployment acceptance.
