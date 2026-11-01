---
title: Ontomark machine bundles, disclosure and generation identity
date: 2026-11-01
status: in-progress
type: explanation
---

# Ontomark machine bundles, disclosure and generation identity

Ontomark implements the machine-scope compiler described by ADR-OM-005. It adds public working notes as plain records, preserves the knowledge ontology and refuses incompatible publication options. The estate-wide transition to one producer remains incomplete in current source: VisionClaw still builds its own bundle and parses pages. The compiler's positive tests therefore establish one component's behaviour, rather than the complete generation loop.

Five adversarial probes qualify that behaviour. A private working-note name survives when an author puts it in a public knowledge title. Changing the profile can change page IRIs without changing the generation ID or `content_digest`. Vocabulary comments are absent from that digest, and caller-supplied commit labels are accepted directly. A real collision in the short working-path hash is correctly refused before output creation.

## Scope and evidence

This is a whole-document assessment of **ADR-OM-005**, including its two amendments and historical real-corpus receipt. It assesses **OM-03.1–2, OM-06.1–2 and OM-11.1–3**. Other panels in the rendered topics receive no new review credit. Cross-repository source checks below assess the producer/consumer boundary only; they do not assess every section of VisionClaw ADR-2137, agentbox ADR-2136 or the Loom ADRs.

| Repository | Inspected revision | Boundary |
| --- | --- | --- |
| Ontomark | `0caae5d4b695cb720942b9b9212017c00a80ad41` | ADR, design, compiler, CLI, publication and generation fields |
| VisionClaw | `ecd78b6cb25d6e702e6c2459c209ccdd2c5c6b4b` | Existing publish workflow, pull URL, local roots and page parser |
| agentbox | `9dfc27b16e82e9786535a55e790710493dd51fdf` | Toolchain pin and migration record |
| Loom | `ba21a689a57dc67dc68ad22e52b058da34a7fb05` | Marker reader and artefact promotion source |
| visionGraph | `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6` | Released producer pin and publish invocation |

The [evidence package](evidence/2026-11-01/om-machine/README.md) retains source exports, whole-file hashes, tests and reproduction instructions. Every inspected source export matches the declared Git revision. Unrelated dirty paths are recorded in its [manifest](evidence/2026-11-01/om-machine/verification.json). Tests ran against a disposable Git archive; product checkouts were not edited. No live corpus build, image deployment, producer-parity window, external embedding service or GPU consumer acceptance was performed.

## Findings and acceptance criteria

| ID | Finding and trigger | Consequence and bounded evidence | Acceptance criterion |
| --- | --- | --- | --- |
| OM-M1 | A public knowledge title contains a private working-note title | Knowledge is projected before the union private-marker redactor is built. The same name is masked in a public working title and retained in knowledge in both scopes. The private body canary remains absent. This is a plain-text name disclosure boundary; it does not demonstrate extraction of an unpublished private body. | Decide whether cross-vault private names must be masked in public metadata. If required, apply one declared policy to both projections and test title, alias, description and body fields while preserving explicitly public identities. |
| OM-M2 | A caller changes `profile.iris.page_namespace` with identical source labels and fixture pages | The generation ID and `content_digest` stay equal while page IRIs and output artefact hashes change. Consumers using source identity alone can conflate distinct builds. Per-file output hashes still distinguish them. | Record the effective profile, producer version and build parameters alongside source identity; demonstrate that a consumer distinguishes profile-only changes using a complete receipt. |
| OM-M3 | A vocabulary-only YAML comment changes the source file | `content_digest` and the complete output tree remain identical. The vocabulary contribution hashes reserialised typed data, contradicting the ADR amendment's literal “every byte” description. | Name the digest's semantic scope accurately, or add raw vocabulary/configuration digests if byte provenance is intended. Test both comment-only and semantic changes. |
| OM-M4 | A library host supplies `Options.source` | `not-a-real-commit` with `dirty=false` is retained as `visionGraph@not-a-real-commit`. This is an exposed host assertion, not a remote attacker bypass or a failure of emitted-file hashing. The CLI passes its source override into the same option. | At the publishing boundary, verify the asserted revision and input tree, and label host assertions separately from observed Git state. Bind the resulting output receipt to the verified inputs. |
| OM-M5 | Two distinct non-ASCII working folders produce the same 32-bit fallback suffix | `雪三地地百/Foo` and `雪五五四三/Foo` both map to `working/x-24f80a23/foo`. The real build returns `PublicationBlocked::Collision` before creating the output directory. Existing “never collide” diagram wording overstated the naming function. | Retain the duplicate gate and a regression fixture. Document the naming scheme as collision-detecting. Any future remint needs the ADR's mapping, visibility diff and rollback evidence. |
| OM-M6 | The estate is described as already running a single shared producer | VisionClaw's workflow still builds `vault`; its pull default still targets its own release; `github_sync_service` still calls `vault_core::parse_page`. The source contradicts complete migration. Released Ontomark pins alone cannot close the retirement gate. | Produce a dated external pod-pack parity receipt, repoint the consumer, reverify ADR-2106 and establish the machine input set before retiring the old job. |
| OM-M7 | ADR metadata and historical dependency statements are read as current implementation evidence | The record says implementation none/inactive, later describes work in progress and retains a successful 12 October build. Current machine implementation passes tests. Its “everything else” permissive dependency row also misses MPL-labelled dependencies present in the default CLI tree. | Refresh status per implemented component and unactivated integration. Link a current dependency inventory and retain historical evidence as dated evidence. This audit makes no legal clearance claim. |

### Why the disclosure finding is narrower than a failed public flag

`project_machine` first invokes `project_vault_with_links` for knowledge (`ir/projection.rs:750–751`). It later combines cleaned records, constructs `union_redactor` (`837–850`) and applies it to surviving plain working records (`869–875`). The returned knowledge corpus is the earlier projection (`890`). `page_api.rs:89` emits its projected title.

This fixture deliberately puts the private note's name into already-public metadata. It does not turn a private page public. Formal cross-vault wikilink cleaning, directional identity refusal and plain-text marker masking are distinct operations. The existing tests support the first two; they do not establish that every name in every public field is masked. Equal public/machine ontology bytes can preserve the same disclosure in both outputs.

### What the identity fields bind

`build/mod.rs:925–942` accepts the explicit source tuple or discovers Git state. `963–983` hashes loaded page/journal hashes plus `serde_yaml::to_string(vocab)`; the profile is absent. Generation artefacts retain their own byte counts and SHA-256 values (`generation.rs:39–45`). Root and data markers scope these receipts differently, and neither hashes itself (`mod.rs:1003–1016`).

These are useful, distinct identities. The source label identifies an asserted source revision; the content digest identifies the compiler's selected input representation; the output manifest identifies emitted files. Treating one as a substitute for the others loses evidence. The [certificate review](2026-11-01-ontomark-reasoning-and-certificates.md) separately records that direct certificate writes occur before consistency and bundle promotion; a successful machine-scope test does not repair that publication boundary.

## ADR-OM-005 section assessment

| Section or obligation | Current assessment |
| --- | --- |
| Metadata, status, scope and verification paragraph | Accepted intent is recorded. `verified_paths` is empty and the ADR explicitly says its commit is its own last modification, not code verification. Implementation none is stale for the machine compiler; inactive remains a reasonable limit on the complete estate transition. Scope's original “changes no code” describes the decision record. |
| Decision 1: VisionClaw consumes the shared build and owns its force projection | Target remains unmet in the inspected current path. `github_sync_service.rs:1667` calls the in-tree parser. Consumer-owned force semantics need an explicit adapter and parity evidence before parser retirement. |
| Decision 2: retire VC-60 under four gates | Gate (i), a dated external producer and byte-equal pack, has no fresh receipt in this assessment. Gate (ii) is not complete: `ontology_pull.rs:42` retains the VisionClaw release default. Gate (iii), ADR-2106 re-verification, is not established here. Gate (iv) has a compiler implementation and fixtures, but no joined consumer acceptance. The workflow at `ontology-publish.yml:168–185` still builds and certifies `vault`, consistent with keeping it until the gates pass. |
| Decision 3: source identity separate from distribution; permitted remints | Implemented fields support distribution-independent labels, but OM-M2–M4 limit what they prove. No remint mapping, visibility diff or rollback exercise is claimed. A matching generation string alone is inadequate consumer parity evidence. |
| Decision 4: retain working pages through an explicit private machine mode | `--scope machine` exists; it includes only public working pages as plain records. It rejects `--vault all`, publish output and disabled redaction. Working contributes to the page API/search view, not the knowledge ontology. “Private bundle” is a distribution intention, not an instruction to include private records or an implemented transport ACL. The ADR's old `knowledge/pages` and `working/pages` defaults have moved to flat roots in current VisionClaw source. |
| Decision 5: owner-authorised port and governance transport separation | `NOTICE` records the grant and exclusions. The 24-commit/29,383-line authorship census is historical evidence at `3c3f4120`; it was not independently repeated here and is not legal certification. Default CLI dependency inspection does not include the AGPL signing crate. |
| Decision 6: dependency direction and released pins | Agentbox `lib/ontomark.nix` and visionGraph `publish.yml` pin v0.7.0 with corresponding digest material; the optional Jev build pins source `0caae5d`. This is configuration evidence. It neither proves the deployed image matches nor establishes every optional feature's dependency boundary. |
| Decision 7: author → validate → publish → consume → propose → review → commit loop | Individual compiler stages exist. The joined loop, consumer identity continuity, human approval and next-generation acceptance remain unverified. The ADR leaves plugin proposal versus local Apply undecided; this review does not close that governance choice. |
| Evidence: provenance subsection | Retain the dated attribution census as historical evidence. Current `NOTICE` is inspected; no new authorship census or rights determination was performed. |
| Evidence: dependency licence table | Default CLI `cargo tree` has no `horned-owl` or `nostr-bbs-core`; `whelk-el/Cargo.toml` makes the former optional. The default tree does contain `im 15.1.0`, `bitmaps 2.1.0` and `sized-chunks 0.6.5` labelled `MPL-2.0+`. `NOTICE` already acknowledges this family. The catch-all table row needs correction; no all-features/release legal conclusion follows from this one tree. |
| Evidence: force-projection table and working-input paragraph | The table describes VisionClaw `3c3f4120` on 12 October. Current parser use confirms the broad migration gap but does not reverify every old field/line reference. Predicate relation types, physics weights, registry mapping and inferred closure still need explicit consumer comparison. The working-root paths are historical; current defaults are `knowledge,working`. |
| Consequence: VC-60 retirement and new Agentbox producer step | A gated disposition is not an accomplished retirement. A scoped search of agentbox scripts/services/config/lib found no `scope machine`, `pack-pod-resources` or `AB-P6` implementation reference; this bounded absence does not rule out an external producer elsewhere. The decisive missing evidence is the dated parity receipt. |
| Consequence: ADR-2106 remains live | Current workflow and pull source agree with retaining the channel. Full assessment of ADR-2106 remains outside this batch. |
| Consequence: agentbox ADR-2108 → ADR-2136 | Current v0.7.0 pin advances the toolchain transition. Read the proposed record separately from deployment state and formal supersession; no wholesale host migration is inferred from the pin. |
| Consequence: replace estate ADR copies with pointers | `docs/adr/ESTATE-ADRS.md` is present and points to owner repositories. Pointer presence avoids treating local copies as authority; it does not prove linked decisions or deployments are current. |
| Consequence: Loom ADR-141/142 amendments | These remain separately owned decisions. Inspected readers use JSON values and preserve the marker; an additional scope field is not rejected by the shown reader. Full reader, deployment and promotion acceptance remain open. |
| Consequence: no code changes; each gate has a deprecation row | Historical statement about the decision itself. Current compiler code exists. `DEPRECATION.md:470` still names the rejected machine `--vault all` combination; the ADR's corrected separate-flag wording is accurate. |
| Deviations and open items | Proposal versus Apply, per-edge property IRI, possible transport relicensing and full activation order remain unresolved by this review. Preserve the historical correction of premature retirement, unequal inputs and the overstated optional dependency rule. |
| OM-00 amendment: exit codes | Typed unsatisfiable refusal and exit 2 are assessed in the preceding certificate/reasoning report. The machine CLI suite freshly verifies its own refusal cases. Do not extend that result to every error path. |
| OM-00 amendment: content digest | Literal all-bytes claim is contradicted by vocabulary reserialisation and the omitted effective profile (OM-M2/M3). Loaded page hashes do contribute. |
| OM-00 amendment: `--with-rvdb` | Build code accepts an embedder and emits records; CLI transport exists. No live embedding/database run was performed here. Scope equality requires matched effective parameters and deterministic external results. |
| OM-00 amendment: inherited VC-60 fault list | Dated findings remain historical. This review confirms that its producer path is retained, but does not re-run every listed VisionClaw defect or claim they are all still present. |
| OM-00 amendment: step 10 is next | Superseded as implementation status: machine compilation and tests now exist. Consumer migration remains separate. |
| Machine design amendment: surface and D1 twins | Scope enum/default and marker omission are implemented. Existing tests cover public/private directional collisions, public/public yielding and private/public silent yielding. These positive cases do not establish every identity normalisation edge. |
| D2 working identity | Path-based IDs and page IRIs exist. Non-ASCII fallback uses a finite 32-bit suffix; the duplicate gate is essential and positively verified by OM-M5. |
| D3 one resolver | Cross-vault resolver serves cleaning and working links; knowledge links are intentionally not re-resolved. Unresolved or yielded working targets can retain the documented slug-verbatim fallback. No complete consumer link-resolution acceptance is claimed. |
| D4 scope-invariant ontology data | Existing fixtures compare all relevant data/OKF outputs with matched inputs and clock. Working expands page/search listings and page-document backlinks. Content-digest equality additionally requires both builds to load the same vaults. The privacy counterexample shows equal data is not a complete disclosure test. |
| Amendment implementation status and design brief | Compiler implementation is present. The brief's proposed generalisation of `project_vault_with_links` to a vault slice differs from the actual dedicated `project_machine` orchestration; the feature is implemented through a different method. Remaining explorer URL handling and full Loom acceptance are not closed by source tolerance alone. |
| First real-corpus machine build | Preserve the 12 October `d9fc5bec4` result: 9,244 classes, 83 working records, 88 yielded twins and 120 seconds. Those are dated counts and timing, not measurements of the current corpus. Historical source fixes and eight held-back residue files were not recensused. |

## Verification and diagram changes

Tests against the current source archive passed **24 existing compiler integration tests**, **four actual CLI tests** and **five audit probes**. These use the small synthetic corpus and real compiler/reasoner code. The final probe run removed two unused imports from the audit-only test file; initial logs are retained. An initial CLI command named the wrong Cargo package; the successful invocation uses package `ontomark`.

OM-03.1 now shows certificate writes before consistency, actual emitter order and digest inputs. OM-03.2 separates the primary ontology root from machine scope's working load. OM-06.1 qualifies output invariance and corrects the stale flag attribution; OM-06.2 describes collision detection rather than uniqueness. Three new [OM-11 panels](../diagrams/ontomark/11-machine-bundle-identity-and-disclosure.md) show disclosure and identity boundaries. The existing topics retain their historical pins: relevant runtime bodies were compared with current source, and the differences are retained as an evidence patch. They are not wholesale re-stamped on the basis of selected-panel review.

All 16 panels in the three changed topics render and pass strict citations: **394 verified citations, zero warnings or fallback**. The seven assessed panels were visually inspected; rendering the other nine does not grant source-review credit. The completion ledger records the broader corpus gate separately.

## Comparisons and user utility

[ROBOT extraction](https://robot.obolibrary.org/extract) offers explicit module strategies: its locality-based extraction retains entailments within the chosen seed/resulting signature, while MIREOT extracts a simpler hierarchy. [ROBOT filtering](https://robot.obolibrary.org/filter) selects axioms with include/exclude and signature/trim controls. These are useful comparison points for explaining a projection's semantic promise. Ontomark's public/machine split addresses authored visibility and consumer records; its current tests do not establish the same semantic-module guarantee as ROBOT's extraction contract.

[SLSA v1.2 build provenance](https://slsa.dev/spec/v1.2/build-provenance) separates the build definition, external parameters and resolved dependencies from run details and output subjects. That model gives the estate an established way to express the distinction exposed by OM-M2–M4. Adopting field names alone would not establish a SLSA level or a trusted builder.

A product opportunity inferred from this evidence is a reviewable knowledge-publication workflow: an author can see what each audience receives, a consumer can identify exactly which build it loaded, and an engineer can explain why a record is present. Prioritise three observable outcomes:

1. A paired public/machine preview displays added, omitted, yielded and masked records before publication, with a private-name canary suite covering metadata and prose.
2. A generation receipt distinguishes source changes, profile-only changes and output changes. A consumer shows the received receipt and rejects an unexpected build definition.
3. A dated migration rehearsal compares the existing VisionClaw view with its bundle projection, including working identity, domain mapping, relation types and force parameters, and demonstrates rollback before retiring the old producer.

These are acceptance proposals, not shipped capabilities or evidence of market leadership. They would make the system easier to trust and operate without requiring users to infer publication guarantees from repository structure.
