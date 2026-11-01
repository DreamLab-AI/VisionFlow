# knowledgeGraph consumers, licence paths and archived extraction

Five panels in KG-01 and KG-06 now distinguish the retained repository, the public publisher, VisionClaw's Markdown ingest and Loom's generation loader. The previous map incorrectly treated the historical ontology bridge as the current agent grounding route. The archived OntoCast adapter retains useful private defaults but four synthetic probes contradict broader claims about evidence preservation, collision safety and output confinement.

This is a source and local-fixture assessment, with documentation changes only. It establishes neither a deployed fault nor acceptance of a current generation. The linked-estate audit remains active.

## Evidence and reproducibility

| Repository | Committed source read |
|---|---|
| knowledgeGraph | `4ed9ac159daf402b4fb252ce559dbeb894a8d91e` |
| visionGraph | `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6` |
| VisionClaw (`project`) | `75b72416bac176ca8fde715cac3bd7c77657b3e7` |
| agentbox | `9dfc27b16e82e9786535a55e790710493dd51fdf` |
| Loom | `ba21a689a57dc67dc68ad22e52b058da34a7fb05` |

Sources were exported with `git show HEAD:<path>`; no working-tree correction was used as committed evidence. The [source manifest](evidence/2026-11-01/kg-consumers/sources.json) records file digests. The archive pipeline was exported with `git archive`, executed only against synthetic fixtures in temporary directories, and used Python 3.12.15, rdflib 7.1.4 and pytest 8.4.2. The repository's old virtual environment had a broken interpreter link; a disposable environment supplied those dependencies.

[Four existing importer tests](evidence/2026-11-01/kg-consumers/existing-tests.txt) pass. [Four additional probes](evidence/2026-11-01/kg-consumers/probes.json) reproduce the behaviours below, including positive controls for private/pending metadata and rejection of an already-existing file. Their assertions describe defects as observed; a passing probe is not acceptance of the adapter. [Harness](evidence/2026-11-01/kg-consumers/probes.py).

Reproduce by exporting `archive/logseq-era-2026-09-22/pipeline` at the knowledgeGraph revision above, installing the named dependencies in a temporary environment, setting `PYTHONPATH` to the exported `archive/logseq-era-2026-09-22` directory and running the harness. Run pytest against that export's `pipeline/tests/test_ontocast_import.py` for the existing controls. No private corpus content is needed.

The [tracked-tree census](evidence/2026-11-01/kg-consumers/tree-census.json) counts 160 explorer files, including 117 under `modern/`, no tracked `explorer/rust-wasm/`, 12 retained `dist/data/` files, 8,138 archived pages and 32 archived pipeline files. These are repository counts at this revision, not the live site's inventory.

## Current routes and documentary gaps

| ID | Finding and evidence | User consequence and acceptance condition |
|---|---|---|
| KG-C1 | The former KG-06 described agentbox ontology-bridge tools as querying VisionClaw's graph. Current agentbox configuration records that bridge's retirement (`agentbox.toml:1168`); `hp-loom-reconcile.sh:133–157` builds a committed visionGraph snapshot through ontomark and verifies its promoted generation. Loom's composition root loads that generation and its own local graph (`lib.rs:72–105`). | An operator needs the served generation, source SHA and consumer state separately. Corrected diagrams remove the mandatory VisionClaw hop. Runtime closeout still needs receipts for each consumer. |
| KG-C2 | VisionClaw's explicit `CORPUS_SOURCE` selects local or GitHub; the default is local only when `VAULT_ROOT` is nonempty (`corpus_source/mod.rs:190–225`). The GitHub adapter lists Markdown (`github.rs:60–74`), and sync filters page identities before building the index (`github_sync_service.rs:290–321`). The earlier static-Turtle sequence was transcribed from historical prose, not established by this path. | Do not diagnose Markdown sync using public Turtle counts. Record effective source and base paths. This assessment does not rule out optional RDF importers elsewhere in VisionClaw. |
| KG-C3 | visionGraph's publisher deploys `www` to knowledgeGraph `gh-pages`, then sends a `corpus-sync` dispatch with the source SHA (`publish.yml:526–542`). Loom's reconciler separately waits for a served-generation match (`hp-loom-reconcile.sh:198–212`). | A successful dispatch proves a request was sent, not that ingest completed. Require destination-specific completion evidence before presenting all consumers as current. |
| KG-C4 | Committed `LICENSING.md:18–26` still names `pipeline/` and `ontology/`; the files now live under the dated archive. `NOTICE` retains the old map, and the archive README only records retirement. The current-path paragraph cited by the former KG-01 belongs to an uncommitted edit, absent from its declared revision. | An integrator cannot obtain an explicit archive-path mapping from the committed table alone. Preserve the three declared grants, reconcile paths in the owning repository, and check the shipped notice inventory. No legal conclusion about ownership, subsistence or enforceability is made here. |
| KG-C5 | The integration document still directs users to `pipeline.ontocast_import` and promotion into the retired source tree. The adapter is now archived, and the current compiler and proposal workflow have changed. | A user following those instructions does not have a current ingest workflow. Mark the document historical and link to a supported, tested ingest contract, without implying that the newer CLI already accepts arbitrary RDF. |

Loom's generation boundary also has an important qualification: `LoadedBundle::activate_or_degraded` rejects an in-flight promotion, verifies a present commit marker, but permits an absent marker as an unattested degraded identity (`bundle.rs:117–128`). The diagram records this rather than converting the startup comment into an unconditional verification guarantee. Adapter loading can degrade independently. These branches were source-reviewed here, not newly exercised against a deployed Loom.

The old workflow is inert at its archived path. Its `contents: read` declaration describes the workflow token's intended permission scope; it is not proof that arbitrary code could never write using some other authority. KG-01 removes that absolute claim and the claim that archived inputs can never be restored.

## Archived importer: adversarial results

| ID | Reproduction and cause | Scope and acceptance condition |
|---|---|---|
| KG-C6 | An `owl:Class` links to a blank node containing a sentinel literal. The candidate preserves the blank-node identifier but omits its outgoing statement. `_entity_subjects` selects URI subjects (`ontocast_import.py:126–137`), and evidence collects only each selected subject's immediate predicate/object pairs (`210–212`). | The document's whole-input losslessness wording is too broad. Preserve the source graph with a digest, or define and test the exact retained subgraph, including blank-node closure and skipped subjects. The probe confirms private and pending fields remain present. |
| KG-C7 | Classes A and B share a label; B is reported as a slug collision and skipped. A third class whose parent is B receives A's local identifier because `_ref` derives the same label slug when B lacks a mapping (`119–123`, `151–156`). | Collision reporting does not prevent semantic aliasing. A reviewer can unknowingly approve the wrong parent. Require explicit unresolved references or a collision-safe identity mapping, then test references to rejected subjects. |
| KG-C8 | One IRI typed as both `owl:Class` and `owl:NamedIndividual` enters the subject list twice. The later local-map assignment supplies the individual identifier to both candidates (`142–156`). Writing creates one file, then raises `FileExistsError` on the second. | Reported candidates need not form a consistent, atomically writable batch. Define a dual-type policy, reject or model it explicitly, and validate the complete destination set before publishing a batch. This is an adapter representation defect, not a claim that RDF forbids the source graph. |
| KG-C9 | A pre-existing dangling symlink at the candidate filename points to another temporary directory. `exists()` returns false and `write_text()` follows it (`247–258`), creating the target outside staging. A repeat write correctly rejects the now-existing target. | Output confinement depends on directory state. Exploitation requires a writable or pre-populated staging path; no live attack was attempted. If reactivated, require exclusive creation and no-follow confinement under a trusted output directory, with symlink and concurrent-creation tests. The current check/write pair also warrants a race test; concurrency was not exercised here. |

These four findings matter if the historical adapter is reused or its guarantees are cited. They are not evidence that the current ontomark ingestion path shares the implementation. The [external-source ingest assessment](2026-10-31-external-source-ingest.md) records that newer proposal boundary and its own open findings.

## Diagram and decision disposition

| Panel | Assessment |
|---|---|
| KG-01.1 | Replaced an undifferentiated live-repository picture with archive, publisher and configured destination boundaries; removed deployment and permission absolutes. |
| KG-01.2 | Retained the three recorded grants, removed ownership/legal inferences and whole-tree identity claims, and recorded the committed archive-path gap. |
| KG-06.1 | Replaced a broad sibling list with the two evidenced distribution routes; removed unsupported universal non-dependency claims. |
| KG-06.2 | Replaced the historical Turtle/bridge sequence with the actual Markdown selection and Loom activation contracts, including degraded activation. |
| KG-06.4 | Preserved private/default-preview behaviour and added the four reproduced limits; removed the unconditional never-overwrite invariant. |

The operative knowledgeGraph decisions retain their earlier section-by-section assessments: [ADR-2001](2026-11-01-knowledgegraph-explorer-contracts.md#knowledgegraph-adr-2001-section-dispositions) and [ADR-2002/2003/2004](2026-11-01-knowledgegraph-publication-gates.md#knowledgegraph-adr-2002-section-dispositions). This continuation adds consumer and documentary evidence; it does not change their status labels or claim a fresh section audit of every related VisionClaw, Loom or agentbox decision. The archived decision inventory and current deployment acceptance remain open, as does the wider estate review.

## Validation

Both changed topics pass strict citation checks: 39 citations at declared revisions, no fallback or unresolved citations, and five successful renders. All five renders were visually inspected. The full catalogue gate passes for 180 topics and 1,598 panels with 24,591 revision-verified citations, zero warnings and nine pre-existing artefact exceptions. Verification freshness covers all 44 declared revisions. Local report/diagram links and `git diff --check` pass. These mechanical checks complement the five-panel semantic review; they do not constitute a fresh semantic review of the whole catalogue. Logs and a local render-retention receipt are in the [evidence directory](evidence/2026-11-01/kg-consumers/retention.json).

## Standards comparison and user utility

The [W3C RDF 1.1 data model](https://www.w3.org/TR/rdf11-concepts/#section-rdf-graph) defines a graph as triples whose nodes can include blank nodes. This supports testing graph preservation rather than counting candidate pages or immediate statements. For a reviewer, a source-graph digest plus a complete retained graph offers a checkable basis for rejecting a bad extraction; the current evidence block alone does not do so.

The [Python `open` contract](https://docs.python.org/3/library/functions.html#open) distinguishes exclusive creation (`x`) from truncating writes (`w`). Exclusive creation is a relevant primitive for the adapter's overwrite promise, but it is not by itself a complete policy for untrusted directory ancestors or batch atomicity. Acceptance should prove both destination confinement and all-or-nothing behaviour under the stated threat model.

The useful product outcome is a review screen that explains which source statements became which candidate entities, flags unresolved identity collisions, and identifies the generation each downstream consumer serves. The immediate recommendation is to correct historical instructions and preserve the distinction between requested refresh and completed refresh; reactivating the old adapter should wait for the four boundary cases above to pass as acceptance tests.
