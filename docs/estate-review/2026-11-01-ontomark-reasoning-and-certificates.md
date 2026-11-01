---
title: Ontomark reasoning refusals and certificate publication
date: 2026-11-01
status: in-progress
type: explanation
---

# Ontomark reasoning refusals and certificate publication

Ontomark enforces a useful named-subsumption parity check, but its certificate files do not share the bundle's publication transaction. A refused build can leave a new certificate beside an old bundle. A successful build deletes its own certificate when the requested certificate directory sits inside the output directory. The shared reasoning facade names unsupported typed axioms; the compiler's earlier graph extraction can omit shapes without reporting a refusal. These boundaries qualify the broad promises in ADR-OM-004 and ADR-OM-009.

This is a whole-document, section-by-section assessment of those two ADRs at Ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41`. It also assesses OM-04.3 and the four new [OM-10 diagrams](../diagrams/ontomark/10-reasoning-and-certificate-boundaries.md). OM-04.1 receives a narrow certificate-path correction without whole-panel credit. No other emitter panel or cross-repository consumer is newly certified by this pass.

[Source hashes and verification](evidence/2026-11-01/om-cert/verification.json), [nine executable probes](evidence/2026-11-01/om-cert/audit_certificate.rs), [final probe output](evidence/2026-11-01/om-cert/probes-third.log) and [reproduction instructions](evidence/2026-11-01/om-cert/README.md) are retained. Tests ran in a disposable source archive. The Ontomark checkout was clean and unchanged. This report records findings; it does not repair product code.

## Executed evidence and its limits

Baseline execution passed 504 compiler library tests and 11 certificate library tests. The reasoner passed 21 integration tests and three documentation tests. The two independent-checker tests executed the pinned `oo-cert` binary: a genuine derivation was accepted and a forged conclusion refused. All nine additional adversarial tests passed, meaning that the assertions reproduced the behaviours below, including the defects. These are synthetic fixtures, not a fresh full-corpus benchmark or a deployed publisher test.

The tested checker is the upstream v2.0.1 Linux release asset, SHA-256 `68223b4db193783a52bd2810a66ce2ad3a3780f963a5ef84753564c051e3e532`. Its digest was checked before execution. This identifies the tested binary; it is not an independent proof-kernel rebuild or a proof of the compiler-to-native-code toolchain.

| Probe | Observation | What it establishes |
| --- | --- | --- |
| Successful build without `certificate_out` | Bundle succeeds; report has no certificate and no certificate directory exists | Certification remains optional. The ADR title is not an unconditional build guarantee. |
| Valid build followed by an unsatisfiable revision | `UNSATISFIABLE_CLASSES`; old ontology bytes remain; `asserted.tsv` changes | Consistency refusal protects the staged bundle but follows certificate replacement. |
| Certificate directory below output | Build succeeds and returns a certificate directory that no longer exists | Bundle replacement deletes the unstaged certificate as part of retiring the old output tree. |
| Late output-parent failure | Certificate exists; a blocking regular file remains unchanged; build errors | Certificate publication can precede a failure to create the bundle's output parent. |
| Second certificate-file write fails | New assertions coexist with the previous digest | The four certificate writes are not atomic as a set. |
| Raw graph with a universal restriction | Compiler reports zero refusals and consistency; equivalent typed axiom reports `EL_OUTSIDE_PROFILE` | Extraction loses the unsupported shape before the facade sees it. This is a library Graph API case, not a demonstrated normal-vault validation bypass. |
| Malformed equivalent-class definition | Definition omitted, zero refusals, consistent result | A caller cannot infer that all original graph expressions were understood from the compiler's refusal count. |
| Named equivalence beyond RDFS transitivity | Certificate has zero inferred pairs; classifier has two; parity is false | The parity adapter rejects this mismatch. This probe does not execute the CLI exit-code branch. |
| Remove the only derivation and change scope metadata | Checker returns `ok:true`, two assertions, zero derivations | Soundness of supplied steps does not prove completeness. The invocation reads only the two TSV inputs, not `scope.tsv`. This is not a defect in the soundness theorem. |

An initial probe compilation used the wrong typed constructor name and failed. The next run passed five tests but four fixtures failed the earlier `DISJOINT_NOT_SIBLINGS` validation rule. Both logs remain beside the final run. Adding the common parent required by the vocabulary made the unsatisfiability fixture valid; the output-failure fixture was also changed to use a regular file as the output parent. These corrections preserve the positive evidence that normal input validation does run.

## ADR-OM-004: local EL profile and named refusals

The [retained ADR](evidence/2026-11-01/om-cert/source/docs/adr/ADR-OM-004-owl2-el-as-the-local-reasoning-profile.md) was read in full. Its own frontmatter warns that `verified_commit` records the document's last modification and `verified_paths` is empty. Accepted/partial/live is documentary status, not a current implementation attestation.

| Section or obligation | Source-grounded disposition |
| --- | --- |
| Question: is choosing EL a mistake? | The choice is defensible for local incremental authoring. The more precise implementation claim is a supported fragment of EL, exposed through capability and per-run completeness fields. This audit does not establish full OWL 2 EL conformance. |
| Decision 1: tractability | The W3C profile supports polynomial-time reasoning as a design rationale. It does not establish latency for this implementation or every input. The ADR's “25 ms slices” overstates a work budget: `ontomark-wasm/src/engine.rs:35–36,567` sets 20,000 work units and describes a historical native measurement. No wall-clock deadline is enforced there. |
| Decision 1: comparison with other reasoners | ELK's Java distribution and explanation/incremental features are documented upstream. The ADR's dated SNOMED timing is not a fresh comparable benchmark. The broader claim that JVM tools could not run in Obsidian at all is not established by this source inspection; distinguish a current embedding/deployment choice from impossibility. No new inventory of every possible Rust/WASM port was performed. |
| Decision 2: grammar and corpus | The compiler selects named classes, subclass/property relations, supported existentials and definitions from its own emitted shapes. The historical 9,258-note and 13,638-edge census was not repeated. Property-domain/range representation and other unsupported forms remain explicit model/backend gaps; a profile's expressiveness is not the same as the implementation's supported inputs. |
| Decision 3: full reasoning elsewhere | Server-side full-OWL analysis is a reasonable division of responsibility. This pass verifies the local core only; it does not prove deployment or semantic equivalence of every Loom/VisionClaw path or the proposed remote analysis. |
| Risk-closure table: typed facade | `profile.rs:100–114` translates accepted axioms and records refusals. Codes distinguish outside-profile and unimplemented-in-profile constructs. `classify.rs:113–114` requires no refusals and no dropped existentials for completeness. The typed universal-restriction probe confirms the named refusal. |
| Risk-closure table: vocabulary | `vocabulary.rs:523–630` checks configured vocabulary mappings. That protects the ordinary authored-input route. Its `NON_EL_VOCABULARY` category also includes `hasValue` and `oneOf` forms; W3C EL includes value restrictions and singleton nominals. The backend support limit needs distinguishing from the profile limit. This is diagnostic precision, not evidence that the normal loader accepts those forms. |
| Risk-closure table: compiler adapter | `ontomark-compile/src/reason.rs:283–401` first extracts recognised patterns, drops unmatched triples and filters incomplete restrictions. Unreadable definitions are omitted whole and mentioned only with `VAULT_WHELK_TRACE`. The later typed refusal check therefore cannot account for every original graph axiom. The “already closed” claim needs this API boundary. |
| Host-visible incomplete state | `ontomark-wasm/src/reason.rs:221–222,352–370` exposes partial backend capability, refusals and run completeness. It first tries All, then a bounded Relevant fallback on budget exhaustion (`389–395`). Host rendering and lifecycle have separate evidence in the [Obsidian refresh](2026-11-01-obsidian-refresh.md); this core pass does not revalidate every consumer. |
| Consequence: remote full reasoner | Remains a proposed off-by-default second classification. No remote endpoint or merging semantics was exercised here. |
| Consequence: missing predicate forms | The distinction between a model gap and a profile gap is sound. Acceptance needs representative domain/range inputs carried through model, classifier and consumer, with supported/refused state retained. |
| Consequence: future non-EL authoring keys | A named refusal table is a useful requirement. Coverage must include adapters before that table, or a new graph shape can disappear without producing the promised diagnostic. |
| References | The W3C profile was checked directly. The historical literature and measurements remain dated rationale, not newly reproduced evidence. Implementation files and test results supply the current bounded verification. |

Production classification uses Skip without definitions and Relevant when definitions exist. It caps only the kept existentials over transitive properties at 2,000 (`reason.rs:473,570–585`). `checked` refuses the classifier's selected-input refusals. The compiler's `Reasoning::is_consistent` checks unsatisfiable classes and refusals; the shared facade's completeness additionally accounts for dropped existentials. Neither a successful selected-fragment run nor zero adapter refusals is proof that arbitrary input RDF was fully classified.

## ADR-OM-009: derivation certificates

The [retained ADR](evidence/2026-11-01/om-cert/source/docs/adr/ADR-OM-009-every-published-closure-carries-a-derivation-certificate.md) was read in full, including its four decisions, historical verification and three alternatives. Its frontmatter carries the same unarmed source-staleness limitation as ADR-OM-004.

| Section or obligation | Source-grounded disposition |
| --- | --- |
| Context: published closure and consumer trust | The same graph is available to classification, certification and serialisation. The 81,396-subsumption claim belongs to the named historical visionGraph run. Current consumer deployment and corpus totals were not recertified. |
| Decision 1: one shared crate | `rdfs-certificate` accepts N-Triples-spelled terms; the compiler adapter spells its Graph into them. Six RDFS rules produce the certificate. This pass verifies the producer and actual checker execution, not every consumer's current dependency or release packaging. Attribution and permissive licensing are recorded in source; this is not a new legal determination. |
| Decision 1: independent pinned checker in CI | The integration tests return normally when `OO_CERT_BIN` is absent or not a file (`tests/oo_cert.rs:43–50,66–70,85–89`). The inspected `.github/workflows/ci.yml` runs workspace tests without configuring that binary, and the workflow/script search found no setup. Green Cargo results alone therefore do not prove checker execution. The historical run with the variable set and this audit's actual run remain valid evidence to their scope. No recent hosted-CI run was inspected. |
| Decision 2: optional flag and same graph | CLI `main.rs:410,827` passes the optional output; build `mod.rs:685–716` certifies the graph already classified. With no flag, certification is absent. The flag is an explicit exception to the title's “every published closure” wording. |
| Decision 2: publication order | Certificate files are written at `mod.rs:715–716`, consistency is refused at `743–744`, staging starts at `763`, and promotion occurs at `1018`. Same input graph prevents reconstruction drift, but does not bind the surviving certificate files to the surviving bundle after a failure. |
| Decision 3: parity enforcement | `certificate.rs:63–67` compares exact sets, not just counts. The certificate side selects inferred, non-reflexive subclass pairs between IRIs (`lib.rs:515–520`); the classifier side uses its filtered inferred pairs. The negative equivalence control fails parity as intended. This is a named-subsumption boundary, not proof of every EL entailment or byte identity of the publication. |
| Decision 3: future semantic widening | A definition whose consequences exceed the RDFS rule table can trigger the existing mismatch refusal. That is preferable to publishing unsupported proof coverage, but supported authoring and certification requirements must be explained together. Widening the certificate rules still needs a new reviewed contract. |
| Decision 4: consequence delta | The shared crate provides the delta; no new Ontomark proposal verb is implied. The comparison includes asserted and inferred non-blank-node-bearing triples (`lib.rs:630` onward), so metadata changes can count too. The [proposal construction review](2026-10-31-proposal-construction.md) supplies earlier consumer evidence; it is not rerun or promoted to current consumer coverage here. |
| Verification: historical full corpus | The ADR clearly identifies `1ec5774ad+dirty`, 351,421 assertions, 731,141 derivations and dated timings. Preserve that qualification. Current synthetic suites and the genuine/forged checker tests strengthen mechanism evidence but do not renew the historical corpus numbers, performance, whole-workspace checks or goldens. |
| Alternative: separate certify verb | Avoiding a second reconstruction is sensible. “For the bytes that shipped” needs a stronger publication/consumer binding than same-Graph construction, as the failure probes demonstrate. |
| Alternative: vendored or self-written checker | Separate implementation reduces correlated mistakes. Authorship independence alone is not a mathematical guarantee; nor does shared authorship inherently nullify a proof. The inspected warrant is the upstream checker contract plus identified binary and actual tests, within its theorem scope. |
| Alternative: duplicate engines | A common crate reduces implementation drift. It does not ensure every consumer uses the same version or reads matching certificate/bundle generations. Those remain separately testable integration obligations. |

`Certificate::write` sequentially replaces `asserted.tsv`, `derivations.tsv`, `asserted.sha256` and `scope.tsv` (`lib.rs:589–592`). The digest covers the assertion TSV bytes. It is neither a hash of `ontology-inferred.ttl` nor a digest of the derivation file. `scope.tsv` writes `complete=true`, `graph=*` and a sanitised generation label; the checker invocation does not read it. A consumer needs additional checks before treating these sidecars as evidence for the exact bundle it serves.

## Priorities and acceptance criteria

| Priority | Finding and user consequence | Acceptance evidence for a later implementation task |
| --- | --- | --- |
| High | Certificate/bundle generation can diverge on refusal or later output failure | Start from a valid published pair, inject each failure, and prove readers receive either the old matching pair or the new matching pair. Bind the result to a verifiable generation manifest. |
| High | Output overlap deletes a successfully reported certificate | Refuse unsafe path relationships before writing, or include the certificate in the staged publication. Exercise nested, equal and aliased paths and require every reported file to survive success. |
| High | Partial certificate replacement mixes generations | Fail each file-write position and verify that the previous complete certificate remains usable, or that no incomplete result is exposed as current. |
| Medium | Raw graph extraction can conceal unsupported/unreadable input | Specify the supported Graph API input contract and report omitted expressions before classification. Test raw graph and normal authored-input paths separately. |
| Medium | Soundness result can be mistaken for complete publication coverage | Check the expected conclusion set and exact bundle/certificate association, then demonstrate refusal after truncation, substitution or a generation mismatch. Keep proof scope visible to the reader. |
| Medium | Optional checker tests can pass without execution | A required job should obtain and verify the binary, fail when unavailable, and retain evidence that genuine and forged cases invoked it. |
| Medium | Work-unit scheduling is described as a timing guarantee | Measure worker slice latency on representative devices and difficult inputs; expose budget stop and incomplete states. Do not derive a deadline from a native timing comment. |

A useful next demonstration is one authored inference followed through validation, classification, certificate creation, publication and independent reading. Include an invalid edit, a failed write, a truncated proof and an old/new generation mix. An operator should be able to answer which published result was checked, what logic the check covers, and whether all claimed conclusions were supplied.

## Comparison and product utility

OWL 2 EL is a standard tractable profile, with value restrictions and singleton nominals among its permitted expressions. The standard also distinguishes complexity results from practical implementation performance. This supports the local-profile choice while requiring precise disclosure of backend support and latency. [W3C OWL 2 Profiles](https://www.w3.org/TR/owl2-profiles/).

Open Ontologies presents ontology changes as plan/apply/rollback with impact reporting and a certificate another reviewer can check. Its documented `oo-cert` interface consumes assertion and derivation files, with `OOCert.certificate_sound` naming the soundness result. Its broader model/refutation/shape claims are separate contracts, not properties imported merely by adopting one certificate format. This comparison is based on the primary documentation and the specific v2.0.1 binary exercised here, not a fresh audit of the entire upstream project. [Open Ontologies](https://github.com/fabio-rovai/open-ontologies).

ELK provides a Java OWL 2 EL reasoner, concurrent computation, incremental updates and derivation explanations, with standalone and OWL API distributions. It is a useful semantic and performance comparison point for a bounded common fragment. No head-to-head benchmark or superiority result is established here. [ELK](https://github.com/liveontologies/elk-reasoner).

This evidence suggests a product opportunity in an inspectable authored-knowledge workflow: users edit readable notes, see bounded semantic consequences and hand another person reproducible evidence. The differentiator must be the reliable journey across those steps. A certificate sidecar alone is insufficient. A pilot should measure time to explain one changed conclusion, detect an incomplete proof and recover a failed publication without ambiguity. These are proposed utility measures, not measured customer outcomes.

## Diagram and coverage disposition

OM-04.3 now branches correctly on consistency and shows that certificate writes can precede refusal. Its extraction, restriction policy, cap, filtered inference and root-cause logic were assessed against current source. The claim that forced-mode callers exist only inside `reason.rs` tests was corrected: `tests/defined_classes.rs` also exercises that API; no production caller was found. Those runtime bodies match the topic's declared old revision: the retained [diff](evidence/2026-11-01/om-cert/old-pin-runtime-diff.patch) shows only a test-fixture edit in build code and unrelated CLI ingest additions. Its existing revision stamp is retained; sibling panels gain no refreshed assessment.

OM-04.1 no longer claims the certificate directory can never be inside output. That is a focused correction only. OM-10 adds four current-revision panels for extraction/refusal, publication order, checker scope and scheduling/completeness. All eleven panels across the two touched topics rendered with Pele; the six new or changed panels were visually inspected. Strict citation checks resolve 228 citations across these topics with no warning or fallback. The initial prose-line-limit failure in OM-04 was corrected before final validation.

Coverage reconciliation adds two whole-ADR assessments and five panel assessments. The broader estate audit remains open: untouched panels, other ADRs, consumer contracts, deployment evidence and the final cross-repository synthesis still require review.
