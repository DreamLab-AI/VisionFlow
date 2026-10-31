---
title: Corpus CLI migration and governance consumer audit
status: in-progress
date: 2026-10-31
type: explanation
---

# Corpus CLI migration and governance consumer audit

Agentbox source was inspected at `9dfc27b16e82e9786535a55e790710493dd51fdf`. Relevant files were clean; an unrelated untracked planning directory was left alone. This pass covers ADR-2136, Nix tool selection and the ordinary promote/demote application path. Creation/amendment dispatch was inspected to distinguish its contract; those complete paths and relay admission remain outside this bounded assessment.

## What runs where

| Boundary | Observed implementation | Limit |
|---|---|---|
| Nix selection | `flake.nix:942` derives `vaultCliActive`; `:956-960` includes both vault and Ontomark and derives the Jev feature choice from System One settings. | Source configuration is not an image build receipt. |
| Ontomark distribution | `lib/ontomark.nix:44` pins 0.7.0. Feature-off fetches a digest-pinned release binary; feature-on builds source `0caae5d4` with its lockfile and `jev`. | The top-of-file rationale that the package always uses the same release bytes is too broad: the feature-on branch compiles a different variant. The adjacent v0.6.1 comment is stale relative to the v0.7.0 code pin. |
| Current session | `ontomark --version` reports 0.7.0; `vault --version` reports 0.1.0. Both resolve into Nix store paths. [Binary paths and SHA-256 hashes](evidence/2026-10-31/corpus-consumer-probes.json) record this observation. | This establishes local executable identity, not the management API process environment, every host's activation, or the feature variant from version text alone. |
| Operation dispatch | `management-api/lib/ontology-apply.js:153-160` routes `propose` and `panel` to vault, other verbs to Ontomark. Corpus writes use edit/create/apply; governance transport remains separate. | Environment overrides can select alternate binaries. The inspected tests deliberately use stubs, so they prove caller contracts rather than Nix-package equivalence. |
| Ordinary promote/demote | The caller checks request-case and signed-subject agreement, then passes outcome/IRI/signer/proposal digest to `applyOntologyDecision`. `editAndAttest` derives changed keys from current status, runs `edit --expect`, then commits and attests. | A signed subject is not a signed page-version check. See the guard distinction below. |
| Persistence and evidence | Commit and push failures are returned as metadata; Loom attestation failure also returns metadata after the write. | `applied`, `committed`, `pushed` and `attested` are separate outcomes. A successful local edit does not establish publication or durable attestation. |

## ADR-2136 section dispositions

| Section | Assessment |
|---|---|
| Context | Historical full-host `vaultSrc` dependency explains the extraction. Current operation ownership is split, with Ontomark handling corpus verbs; do not read the opening paragraph as today's full dispatch description. |
| Decision 1 | Separate Ontomark packaging is implemented and locally visible. Preservation/versioning of the old vault command contract is a migration obligation, not implied by adding a binary. The record remains proposed/partial/inactive; this audit does not change owner adoption. |
| Decision 2 | Caller configuration selects an authored-vault root and rejects invalid explicit roots. The inspected adapter does not make memory or Loom a prerequisite write authority. Universal corpus authority across all consumers remains a wider question. |
| Decision 3 | Shared CLI operations are consumed by the management API. This paragraph's “do not add ... ingestion work” is narrower than the later v0.7.0 CLI's capability surface; capability availability alone does not mean this adapter invokes it. Clarify the intended scope when accepting the decision. |
| Decision 4 | Source/release hash, lockfile and feature choices are recorded in Nix. No fresh Nix build, publisher/Loom output comparison or loaded-management-process receipt was produced here. Local version observation is only one part of the activation ladder. |
| Decision 5 | Separate feature branches are visible. This pass did not audit the full dependency licence graph and does not certify licence closure or offline behaviour of optional model features. |
| Consequences | Reduced corpus-tool coupling is supported by the independent package, but vault remains a separately baked governance tool. Activation is an operational action; proposed ADR status should be reconciled explicitly with the now-observed executable. |
| Verification | The initial “no repin” statement is historical; later appendices qualify it. Existing package/CLI compatibility and generation-equivalence obligations cannot be discharged by the 67 stub-oriented adapter tests run here. |
| Producer ownership continuation — 2026-10-11 | Treat the old package/release numbers as dated lineage. Current runtime observation supersedes its uncertainty about whether this session has any Ontomark binary, but not its broader rollout obligations. |
| Estate alignment and the ADR-2108 mapping — 2026-10-12 | The decision preserves CLI/HTTP ownership and separates governance transport. Its proposed supersession and AB-P6 machine-bundle responsibility require owner acceptance and their own byte-equivalence/consumer evidence; neither is inferred here. |
| The shipped `vault` could not read the migrated corpus — 2026-10-18 | Historical diagnosis and test receipts remain dated evidence. This audit did not rerun the real corpus or legacy validation. |
| Close-out 2026-10-30 | The package was already beside vault. The earlier running 0.6.0 observation must remain tied to its image receipt; current session is 0.7.0. |
| Re-verification — 2026-10-31 | The v0.7.0 source pin matches current Nix selection. The text saying the running image still carries 0.6.0 is superseded for this session by the captured executable identity, not by an assumption that all deployments changed. |

## Guard semantics: blast radius is not reviewed-content identity

The module comment says the edit fails if the subject moved underneath a decision. The ordinary status path is narrower: `readPageStatus` reads the current status, `setArgsFor` chooses changed fields, and `expectFor` counts those fields. The signed digest is passed to the attestation ledger and commit message; this branch does not compare it with current page bytes or pass a page-content precondition to `ontomark edit`.

The Ontomark implementation at `crates/ontomark-compile/src/write/edit.rs:208` clones the page, applies changes and checks document/key counts. A temporary generic-vault probe ran the real 0.7.0 executable twice, once on `reviewed text` and once on `replacement text after review`, using the same `edit A --set status=stable --expect docs=1,blocks=1`. Both exited 0 and retained their different bodies while changing status. [Receipt](evidence/2026-10-31/corpus-consumer-probes.json).

This demonstrates the guard's intended cardinality scope; it is not an end-to-end signed-decision exploit. A policy may deliberately approve an enduring entity rather than a particular version. **Open decision:** specify whether ordinary promotion approves current entity state or the content shown to the reviewer. If it approves a version, bind and verify that version atomically at mutation time. If it approves an entity irrespective of edits, disclose that scope in the review UI and attestation semantics. Do not substitute the proposal's digest for a page hash without defining what each hashes.

Creation and amendment are different paths: amendment stages the approved proposal and calls `ontomark apply`; it must not inherit conclusions about the simpler status edit. Full proposal digest/base-content enforcement remains for a separate consumer trace.

## Commit and attestation boundaries

`commitPage` checks the Git top level, limits the pathspec to one file, records commit errors without undoing the write, and only attempts push when the branch reports one commit ahead of its upstream. These are useful safeguards. They are a sequence of Git calls, not a lock across concurrent writers. A deterministic probe called the production `commitPage` with real Git against temporary repositories and a local bare remote. Immediately after `rev-list --count` returned one, an injected writer committed an unrelated file. The ensuing push sent both commits; the returned receipt reported `pushed: true` and named the earlier governed commit, while the remote branch pointed to the injected commit and contained its unrelated file. [Probe source](evidence/2026-10-31/commit-push-interleaving.cjs) and [result](evidence/2026-10-31/commit-push-interleaving.json).

This establishes an interleaving that defeats the stated “do not push other sessions’ commits” boundary. It does not measure production incidence. Publication should bind to the authorised commit explicitly and handle concurrent branch/remote movement; test the precise published ref and bytes as well as the boolean receipt. Races around page contents or the separate HEAD read were not exercised here.

`attest` has a bounded timeout and returns failure rather than throwing after persistence. This avoids reporting a successful write as if nothing happened. End-to-end recovery must still show that failed attestations are retried or visibly reconciled; this module alone supplies no such retry receipt.

## Validation and documentation changes

`node --test management-api/tests/ontology-apply.test.js` passed **67 tests, zero skipped**. The suite documents its use of stub executables and injected dependencies; no real governance request, relay, external Git remote or Loom call was exercised. The real CLI was used only against temporary local notes; the separate push probe used a temporary local bare remote and cleaned up its repositories. No real corpus file changed.

VF-08.1 now shows the two CLI roles and common package gate. Only the Agentbox revision slice was advanced to the inspected source; other repository revisions remain dated. The old assertion that missing revision mappings silently pass was corrected: the current citation checker warns and strict mode rejects those warnings. The updated topic passed strict citation checking with **22 revision-resolved citations, zero fallbacks/warnings**, and all seven diagrams rendered. These checks do not certify the historical mesh, pod or supporting-repository panels as current.

The full 158-topic citation check also passed: **23,668 revision-resolved citations, zero working-tree fallbacks or unresolved citations, and nine explicitly allowlisted generated-file references**. [Log](evidence/2026-10-31/corpus-citations.txt). The two allowlisted generated files were copied from the primary checkout for this check; they are not commit-backed evidence. Citation resolution does not establish semantic correctness of every diagram.
