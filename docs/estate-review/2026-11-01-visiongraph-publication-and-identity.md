# visionGraph publication and generation identity — 1 November 2026

The pod loader can report a new corpus release as already loaded when VisionClaw's code commit has not changed. The package records the corpus SHA separately, but the loader compares only the build SHA before deciding to skip downloads. Four focused tests reproduce that behaviour and its refusal/update controls using the actual puller and filesystem generation store.

This documentation-only review corrects all ten panels in [VG-03](../diagrams/visiongraph/03-publish-pipeline-and-explorer-build.md), [VG-04](../diagrams/visiongraph/04-downstream-contracts.md) and [VG-05](../diagrams/visiongraph/05-invariants-register.md), and assesses every section of ADR-VG-001 and ADR-VG-002. No product code, ADR status, authored corpus, release or deployment was changed. Findings describe the inspected source and isolated tests, not a newly observed live pod incident.

## Evidence boundary

| Evidence | Scope |
|---|---|
| visionGraph `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6` | Full publisher workflow, two decisions, historical publication contract, dependency pins and domain/final-site gate scripts. |
| VisionClaw `75b72416bac176ca8fde715cac3bd7c77657b3e7` | Configured downstream workflow, packer's identity fields, complete puller and generation-store modules, startup wiring and local graph-inclusion predicate. |
| ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41` | Local source of release tag v0.7.0: typed public predicate, staging, projection/build order and generation-marker construction. No new release-binary checksum verification or compiler build is claimed. |
| New executable checks | Eight Rust tests pass: four existing generation-store tests and four new identity scenarios. Production `ontology_generation.rs` is copied unchanged; `ontology_pull.rs` is copied unchanged up to its original test module. Network access is replaced by an injected map-backed `Fetch`; storage is the real `PublishedStorage` over `solid-pod-rs` 0.5.0-alpha.17. This is not the complete server test suite or a live release pull. |
| Reused publication evidence | Five overlapping source hashes match the earlier [publication-gate audit](2026-11-01-knowledgegraph-publication-gates.md). Its 41 publisher tests, 11 probes and scanner self-test remain dated results, not additional executions in this batch. Domain limits also retain the scope of the [vault/domain review](2026-11-01-visiongraph-vault-and-domain-contracts.md). |
| Reused public observation | The [04:30–04:34 UTC public receipt](2026-11-01-knowledgegraph-inventory-and-live-receipt.md) sampled generation `visionGraph@84c1cee32b6ad03b58bcb31ba33320142c330523`, with a context hash mismatch and six other sampled matches. This batch does not refresh that network observation or assume the public site runs local HEAD. |

[Source manifest](evidence/2026-11-01/vg-publication/sources.json), [Rust execution log](evidence/2026-11-01/vg-publication/pull-tests.log), [identity tests](evidence/2026-11-01/vg-publication/harness/tests/identity.rs), [harness preparation](evidence/2026-11-01/vg-publication/harness/prepare.py), [locked dependencies](evidence/2026-11-01/vg-publication/harness/Cargo.lock) and [source-hash reuse receipt](evidence/2026-11-01/vg-publication/reused-evidence.json) bound the claims.

## Current findings and useful acceptance

| ID | Evidence and implication | Recommended acceptance |
|---|---|---|
| VG-P1 — high, source refresh can be skipped | `project/scripts/ontology/pack-pod-resources.py:116–119` records `sourceSha` from `ONTOLOGY_SOURCE_SHA` and `buildSha` from `GITHUB_SHA`. `ontology_pull.rs:205–230` parses build identity but not source identity; `312–318` returns `UpToDate` when stored build SHA matches and an atomic generation exists. A different corpus under the same VisionClaw commit remains unactivated in the executable fixture. | Use a verified content/generation identity that changes with the corpus and all relevant producer inputs. Keep source and producer identities separately visible. Require a corpus-only update to advance the loaded generation, with unchanged producer code, and retain the checksum-failure control. |
| VG-P2 — early shortcut is weaker than verification | The same-build branch runs before fetching `SHA256SUMS` at line 320. Even a release fixture without that file reports up to date after reading only `index.jsonld`. It does not write unverified content; it leaves the old generation active. | Verify the identity used for the no-change decision and report what was compared. Do not present this shortcut as a fresh verification of all remote bytes. |
| VG-P3 — shared source is not one producer | visionGraph pins ontomark v0.7.0 (`publish.yml:48–51`) and selects all-vault public staging; the downstream workflow still builds its own `vault` crate and selects knowledge (`ontology-publish.yml:164–173`). Its comment still calls that the same binary as the site. | Either consume the shared immutable build as the later design intends or explicitly version/test both projections and producers. A corpus SHA alone cannot prove byte or semantic parity. ADR-VG-007's fuller extraction/profile assessment remains queued. |
| VG-P4 — compiler marker is not a final-site receipt | `Generation` includes commit, dirty state, content/ontology digests, counts, vocabulary version, optional expiry, scope and artefact hashes (`ontomark/.../generation.rs:90–123`). It does not include explicit explorer/dependency/compiler/policy-version fields. The workflow then overlays static contexts, compresses files and adds the UI/notices. The earlier public context mismatch is a concrete boundary failure. | Bind the final served representations and build inputs in a release receipt after all transformations. Preserve compiler and site manifests as distinct stages where useful. Validate the bundle actually loaded by consumers, not just a successful staging copy. |
| VG-P5 — request delivery is not consumer activation | The site dispatches `source_repo` and `source_sha` after deployment (`publish.yml:536–542`). No consumer-completion acknowledgement is checked in that step. Pod, Markdown and Loom routes have separate input and activation contracts. | Expose source generation, distribution generation and loaded generation per consumer, with last successful verification and explicit lag. A website build should not imply that every consumer refreshed. |
| VG-P6 — existing final-site gate gaps remain applicable | Same-revision KG-P evidence covers gzip content skipped by the final text scanner, raw-or-gzip existence checks, equal-count notes substitution, pre-SPA size timing, missing working/assets push filters and pre-compression copy receipts. VG-V evidence qualifies domain-tier and category guarantees. | Retain those findings and their controls; test final stored representations, exact output identities, complete build inputs and final assembled size. These component gaps do not establish a full workflow bypass. |
| VG-P7 — historical wording obscures current obligations | Both decisions leave `verified_paths` empty. Their banners say current operations use `vault`, while the actual site producer is ontomark. The publication contract's body still discusses JSON-LD flags, Python projection and frozen notes, despite its historical banner and the current Quartz notes path. | Preserve dated evidence but attach current producer/consumer receipts and accurate successor references. Do not repeat the old truthy-string/private-ancestor counterexamples as current defects without new evidence; do not treat their historical repair as full acceptance of the successor. |

GitHub documents `repository_dispatch`'s `GITHUB_SHA` as the last commit on the receiving repository's default branch; the payload is separate event data. Thus two corpus events can share a VisionClaw build SHA while naming different source revisions. This supports the configured trigger scenario for VG-P1; it is not evidence that a particular remote run or deployed pod has encountered it. [GitHub event semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#repository_dispatch).

### Executed identity scenarios

| Scenario | Observation |
|---|---|
| Load source A/build 1, then offer source B/build 1 | Initial update succeeds. Second pull returns `UpToDate`, fetches only `index.jsonld` and retains source A. |
| Load source A/build 1, then source B/build 2 | Second pull downloads/verifies and activates source B. |
| Offer source B/build 2 with corrupted Turtle bytes | Digest mismatch refuses activation and source A remains readable. |
| Offer source B/build 1 with no checksum file | Returns `UpToDate`; source A remains active. |

The four original generation-store tests also pass, covering complete reads through activation, injected failures/restart, symlink refusal and corrupt-pointer/ACL separation. They support those tested storage behaviours only. The freshness shortcut occurs before that publication mechanism, so working atomic publication does not cure VG-P1. The harness's first build failed because its dependency list omitted `uuid`; the retained setup log distinguishes that harness error from the subsequent successful run. No network fetch, actual GitHub release or public pod was used.

## ADR-VG-001: publication policy, section by section

| Section | Assessment |
|---|---|
| Frontmatter | Proposed/partial/inactive and the broad contract-change trigger remain coherent with unclosed acceptance. `verified_paths: []` does not arm source staleness checks; the carried commit is not a fresh implementation receipt. |
| Verification note, 31 October | Explicitly records the verification limitation. This review adds source-specific evidence without changing its status. |
| Historical banner | Correctly labels the Python publisher as retired, but its successor name `vault` lags the released site CLI. Retirement alone does not certify policy parity. |
| Context | Distinct inclusion rules remain relevant: the inspected local PageMeta predicate permits `public || owl_class.is_some()`, while the compiler Page predicate requires boolean true. The malformed-fence/string-false/private-ancestor examples describe the historical publisher, not proven current incidents. |
| Proposed decision and acceptance | Named dispositions, typed flags, inferred visibility and the common mixed fixture remain the correct test dimensions. Current source has census, typed flags, projection before closure and credential checks. This pass does not execute the entire proposed cross-consumer fixture or certify every diagnostic/export. |
| Consequences | Correctly separates partial component functionality from the inactive complete contract and denies deployment certification. Retain that distinction. |
| Closeout extension: work packages, owners and dependencies | Names affected maintainers and source identity/publication dependencies. The new pod identity finding is relevant to the consumer contract; source and delivered identity cannot be collapsed. |
| Closeout extension: evidence links and limits | Prior source/synthetic/remote evidence remains separately scoped. This documentation batch does not repair the publisher, corpus or production deployment. |
| Execution qualification: input preflight and projection | The September statement describes the old Python implementation. The successor source inspected here has its own typed Page predicate, publication staging and projection order; these need their own acceptance evidence rather than inheriting a historical green test count. |
| Execution qualification: residual limits and receipt | Known-private filtering is not universal secret detection, and local implementation is not consumer activation. The referenced old receipt remains valid as dated evidence. Current compressed-output scan findings are a separate final-site issue, not proof that the old private-ancestor defect returned. |

## ADR-VG-002: generation identity, section by section

| Section | Assessment |
|---|---|
| Frontmatter and verification note | Proposed/partial/inactive is retained. Empty paths and a carried historical commit leave current source verification incomplete. |
| Historical banner | Correctly prevents treating the old Python source paths as current; `vault` successor wording needs revision-specific qualification. |
| Context | Source repository, distribution checkout, deployed tree and loaded indexes remain separate. The embedded explorer is also distinct from sibling repositories. The new puller test demonstrates why preserving both source and consumer-build identity matters. |
| Proposed decision and acceptance: manifest | A real current compiler marker exists, so saying there is no generation mechanism would be wrong. It still does not satisfy all requested compiler/explorer/dependency/policy fields or bind every final site transformation. |
| Proposed decision and acceptance: transitions | Deletion, equal-count substitution, namespace changes, interrupted activation and rollback must reach the actual publisher and each consumer. This pass adds corpus-only refresh and digest-refusal controls for the pod. It does not close that full transition matrix. |
| Proposed decision and acceptance: browser and historic notes | Independent embedded-browser verification remains needed. `/notes/` now serves current Quartz content; preserving a frozen historical surface must not be confused with claiming that current route still hosts the old app. |
| Consequences | Correctly keeps component implementation distinct from end-to-end acceptance and deployment certification. |
| Closeout extension | Source identity, producer and consumer owners and work packages remain relevant. The new receipt complements the earlier evidence; it does not silently ratify the proposal. |
| Source qualification: old parser/closure counterexample | Keep the dated private-grandparent result and explicit lack of a live leak. The referenced Python paths are retired and this counterexample is not rerun against the current compiler. |
| Source qualification: privacy dependency | Correct: a digest can faithfully identify an inappropriate public bundle. Metadata itself must avoid disclosing private identities. New storage/identity tests do not prove publication privacy. |
| Closeout acceptance paragraph | Complete input diagnostics, policy and namespace behaviour plus exact loaded bytes remain open as a combined requirement. The build-SHA shortcut contradicts using the present pod result as proof of fresh source activation. |
| Execution qualification | Historical projection/rollback repairs do not prove current site receipt completeness or every consumer's freshness. Actual filesystem-generation tests now pass locally, but the new upstream identity mismatch remains. |
| Evidence link | The September receipt stays dated; the present harness and earlier November HTTP sample have different scopes and revisions. |

## Publication-contract section dispositions

| Section | Assessment |
|---|---|
| Historical banner and status | The banner is essential: the body is September closeout evidence. Proposed status does not describe site uptime. Current Quartz notes replace the old notes application. |
| Authority and current behaviour | The authority split is still useful. JSON-LD flag/parser statements require historical qualification; current predicate and build evidence are mapped in VG-05.1. Changing one consumer's policy still cannot prove removal from all consumers. |
| Proposed acceptance requirements | Input disposition, typed flags, intentional private ancestry, complete manifest identity and loaded-consumer reports remain open together. The frozen-notes requirement needs an explicitly historical destination. |
| Evidence and decisions | Correctly links the two proposed records and reopens on producer, format, inference, namespace, consumer or deployment changes. Several such changes have occurred. |
| Source qualification, 7 September | Historical synthetic failure and its no-live-disclosure caveat remain valid. Its pre-repair findings do not override the execution update or current source. |
| Execution update, 7 September | Python projection, title-form Markdown and rename rollback are historical repairs. They are not current workflow instructions or proof of simultaneous-reader atomicity in every successor. |
| Mobile publication qualification | The 375px/502px measurement and candidate fix remain dated. No mobile browser run occurs in this batch; current browser/layout acceptance is not inferred from that old measurement. |

## Diagram disposition

| Panel | Correction |
|---|---|
| VG-03.1 | Updated compiler pin and workflow order; added collision/body gates and distinguished dispatch from activation. |
| VG-03.3 | Updated shared package pins and separated preview data from deployed authoritative machine output. |
| VG-03.5 | Qualified certificate and domain checks; removed the unsupported guarantee that all fifteen supply categories must exist. |
| VG-03.6 | Exposed static overlay, pre-compression copy receipt, pre-SPA size check and final scan/layout limits. |
| VG-04.1 | Added actual server boot/periodic pull, rolling-release checksums and conditional push path; retained the configured dual-producer difference. |
| VG-04.3 | Expanded source/distribution/deployed identities to the loaded-generation decision and reproduced build-SHA freshness gap. |
| VG-05.1 | Replaced broad unsourced invariants with exact current compiler/publication and local inclusion predicates. Layout details remain in VG-01. |
| VG-05.2 | Preserved proposed acceptance and inactive complete-contract status without implying the existing site is inactive. |
| VG-05.3 | Replaced historical-current defect wording with the real current generation marker and final-site binding gap. |
| VG-05.4 | Preserved historical failures and repairs as dated succession, removing obsolete mobile/layout implementation detail from the current policy map. |

## Comparison and user utility

[SLSA build provenance](https://slsa.dev/spec/v1.2/build-provenance) separates output subjects/digests, build definition, resolved dependencies and run details. That provides a useful model for the ADR's missing producer/input identity fields. This review does not assess a SLSA level, verify a signed attestation or claim that provenance alone provides content privacy.

[IPFS's immutability model](https://docs.ipfs.tech/concepts/immutability/) distinguishes content identity from a mutable pointer to that content. The comparable requirement here is that a change in corpus bytes changes the identity used to decide freshness, while a rolling release remains only a discovery pointer. This is a design comparison, not a recommendation to move hosting to IPFS.

My assessment is that the valuable product outcome is a governed knowledge publication whose users can tell which evidence they are reading and whether their application has received an update. A useful acceptance journey is: change a public corpus page without changing application code, publish, observe each consumer move to the intended generation, then roll back and verify the result. Measure convergence time, stale-consumer detection and exact page/query behaviour. Neither a website deployment badge, a hash list nor a derivation certificate alone proves that user journey.

## Verification

The full catalogue passes strict citation checking: 181 topics, 1,604 panels and 24,609 revision-verified citations, with zero warnings, zero working-tree fallbacks and nine existing allowlisted generated-file references. Index/verification freshness checks pass. The ten revised panels render successfully and were visually inspected; their three topic checks verify 89 citations without warnings. These checks establish structure and cited-line resolution, not whole-estate semantic or runtime acceptance. The estate-review index frontmatter was also repaired after earlier continuation entries had been inserted inside it.

Local retained source exports, all ten SVG/Mermaid/PNG render sets and the executed harness are in `/tmp/vf-vg-publication-artifacts`. The [retention manifest](evidence/2026-11-01/vg-publication/retention.json) records their hashes; committed test fixtures, dependency lockfile, source hashes and logs support reproduction beyond that temporary directory.

## Reproduction and remaining scope

Export the two VisionClaw modules from the pinned revision. Run the retained `harness/prepare.py EXPORTED_PROJECT NEW_EXECUTABLE_WORKSPACE`, then `cargo test --locked --offline --manifest-path NEW_EXECUTABLE_WORKSPACE/Cargo.toml`, with `TMPDIR` and `CARGO_TARGET_DIR` on an executable filesystem. The harness pins the pod crate and lockfile, retains the real publication module's four tests, and excludes the original puller test module that depends on the full server. The new four tests use injected bytes and temporary storage only.

Seven of visionGraph's eight operative ADRs now have section assessments across this and the preceding vault/domain and admission reviews. ADR-VG-007 remains. All existing VG-03/04/05 panels are now assessed; VG-06's five shipping-explorer panels remain queued. Public attribution, other intake programmes, current end-to-end browser/consumer acceptance and implementation findings remain open. The wider linked-estate audit is still active.
