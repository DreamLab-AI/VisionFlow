---
title: Knowledge production, publication and semantic guarantees
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Knowledge production, publication and semantic guarantees

VisionFlow's knowledge-production layer has substantial working software: a readable authoring format, multiple machine-readable projections, an explorer, publication workflows and tests of the graph format. Its main architectural weakness is that these capabilities are distributed across divergent copies. “The ontology pipeline” does not identify one reproducible implementation or one semantic guarantee.

This chapter covers knowledgeGraph and the newly identified Logseq authoring repository, with the explorer's ownership boundary and Loom's input contract. A complete knowledgeGraph build and both repositories' existing Python test suites ran locally. Synthetic probes exercised publication boundaries. No private page contents were published or reproduced, no external site was queried, and no production incident is asserted.

[Pipeline receipts](evidence/knowledge-snapshot.json), [collector](evidence/knowledge-probes.py), and [lineage/source hashes](evidence/knowledge-boundaries.json) support the findings.

> **Current-source update:** the subsequent [authored-vault trace](authored-vault-transition.md) identifies visionGraph as the current producer and Logseq as retained history. The measurements below remain valid for the named checkouts; they must not be used as the current publisher's census. The same boundary probes were separately repeated against visionGraph.

## Two pipelines, one public account

The local `logseq` path is a symlink to `project4`, whose Git origin identifies `jjohare/logseq`. It is on branch `obsidian` at `4f233f321097`, not a checked-out `main`. knowledgeGraph is a separate repository at `7bbf0aae2b60`. This distinction limits what can be inferred about the publisher currently running remotely.

| Responsibility | knowledgeGraph checkout | Logseq checkout |
|---|---|---|
| Source pages | `ontology/pages` | `mainKnowledgeGraph/pages` |
| Build entry | Seven-stage `pipeline/build.py` | Extended `pipeline/build.py` with closure, scaffold and prose stages |
| Semantic processing | Emits axioms; validates selected page fields | Adds transitive superclass closure and inherited relationship metadata |
| Grounding exports | Fresh build does not emit scaffold/prose indices or inferred Turtle | Emits `scaffold-index.json`, `prose-index.json`, `ontology-inferred.ttl` |
| Workflow role | Build, test, class-count contract, validation, downloadable artefact | Build, validation/conflict/integrity checks, explorer build/smoke, publication |
| Publication destination in workflow | No deployment step | `DreamLab-AI/knowledgeGraph`, branch `gh-pages` |
| Explorer ownership | Embedded source directory | Embedded source directory under `publishing-tools/WasmVOWL` |

Sources: [extracted build](../../../knowledgeGraph/pipeline/build.py), [authoring build](../../../logseq/pipeline/build.py), [extracted CI](../../../knowledgeGraph/.github/workflows/build.yml), and [publisher workflow](../../../logseq/.github/workflows/publish.yml).

**Assessment:** knowledgeGraph is a useful public distribution, but cloning it and running its documented build does not reproduce the current authoring pipeline's full export contract. The estate should name the authoring revision, extraction revision and publication revision separately. A public method must also say which parts can be reproduced from the public distribution alone.

## Authoring model and conversion

The [parser](../../../knowledgeGraph/pipeline/jsonld_parser.py) locates fenced `json-ld` blocks with a regular expression, decodes them as JSON, and selects the first `Page` and first recognised entity block. It supports `OntologyClass`, `Class` and `Individual`, translating v1/v2 fields into shared dataclasses. Subsequent stages consume those objects. This is a specialised authoring-schema parser; it does not perform general JSON-LD context expansion.

Page publication metadata and ontology semantics are separate. The `Page` carries slug, publication flag and outbound links; the entity carries identity, definition, parentage, a closed relationship vocabulary and quality/maturity fields. Body extraction takes the text after the final JSON-LD fence. This design keeps human-readable prose and structured data together, but the conversion's limits should remain explicit: only recognised fields enter the dataclasses, only the selected entity is processed, and additional evidence blocks are not general graph input.

[Turtle generation](../../../knowledgeGraph/pipeline/jsonld_to_turtle.py) makes several deliberate modelling choices. It rewrites recognised URNs into HTTP IRIs, declares schema terms, emits existential restrictions for eligible `requires` and `hasPart` relationships, adds maturity individuals, and gives undeclared associative targets SKOS stub descriptions. Top-level domain disjointness is enabled by default. These choices add semantics beyond a simple serialisation of each source JSON object.

**Assessment:** “lossless” needs to name its boundary. Retaining the original Markdown or an import-evidence block can preserve source material, while the served RDF and graph tiers remain selected and transformed projections. A blanket claim of lossless compilation is too broad without a mapping contract and counterexamples for unsupported fields, multiple entities and discarded projections.

## What a fresh build established

The local knowledgeGraph build completed into a temporary output directory. It reported no errors or warnings and 1,403 informational multi-parent entries. Existing tests reported **13 passed**. Those results support the local pipeline and its tested contracts, not general semantic correctness.

| Measurement | Fresh local result | Meaning |
|---|---:|---|
| Markdown files / parsed pages | 8,138 / 8,138 | Every input file in this snapshot produced a Page object |
| Public pages | 8,138 | All parsed publication flags were true booleans |
| Source entities / source individuals | 8,138 / 0 | Current source entities are classes |
| `stats.json` classes / individuals | 8,138 / 0 | Counts in the graph-tier model |
| `stats.json` pages | 8,134 | The emitter counts distinct page identifiers, not input files |
| Turtle triples | 265,796 | Fresh extracted build, without Logseq's separate inferred export |
| RDF `owl:Class` subjects | 8,140 | Includes schema declarations in addition to source entities |
| RDF `owl:NamedIndividual` subjects | 5 | Generated maturity vocabulary: draft, stub, emerging, established, deprecated |
| Declared / resolvable graph edges | 113,506 / 101,321 | Graph tiers omit edges they cannot resolve to their node set |

The “pure TBox, zero individuals” statement accurately describes the current source entity census; it does **not** describe the entire emitted Turtle file, which includes five named maturity individuals. Similarly, class counts from the viewer and a SPARQL count over the complete RDF need not be equal. These are different denominators, not interchangeable measures of one quantity.

The [graph-tier emitter](../../../knowledgeGraph/pipeline/emit_graph_tiers.py) also makes a presentation trade-off: each binary node has one category, while additional memberships are recorded in `bridges.json`; domain tiers apply caps while `full.bin` is uncapped. This is sensible for rendering, provided consumers do not mistake a display projection for the full ontology.

## Validation is useful but not a semantic reasoner

[validate.py](../../../knowledgeGraph/pipeline/validate.py) checks selected missing fields, duplicate entity IRIs, direct self-parenting, domain values and certain slug relationships. It treats multiple parents as informational. The extracted build calls it and ultimately exits nonzero for errors, although it still writes output before returning the report. Warnings do not fail that command. The public CI additionally enforces a class-count contract and runs tests.

The Logseq [reasoning module](../../../logseq/pipeline/reason.py) computes parent reachability with a cycle-safe breadth-first search and derives non-direct superclass sets. It also inherits relationship metadata from ancestors, capped at eight targets per relation type. It does not call Whelk or ELK in this path, and it does not evaluate the complete set of emitted existential and disjointness axioms. The [conflict checker](../../../logseq/pipeline/conflicts.py) separately detects duplicate concepts, subclass cycles, and selected relation/type contradictions. Its publisher invocation blocks high-severity findings.

**Assessment:** these checks provide real value, but “EL-profile closure” should not be read as proof that the emitted ontology passed a complete OWL 2 EL consistency or satisfiability check. Parent reachability, capped inherited metadata, authoring conflict detection and a reasoner's classification are distinct operations. The README's stronger grounding narrative needs the specific reasoner invocation and artefact receipt that justify it, wherever that invocation runs elsewhere in the estate.

## Boundary probes: failures the current suites do not catch

Both suites passed in this review: **13 knowledgeGraph tests** and **52 Logseq tests**. Inspection showed meaningful coverage, including the independently decoded NGG1 golden fixture, graph caps, authoring conflict checks, closure cases and import staging. Passing these suites does not close the following separately reproduced cases.

### Malformed input can disappear before validation

A directory containing one Markdown file with malformed fenced JSON produced zero parsed pages and a validation report with zero errors in both implementations. `parse_page` skips JSON decoding failures; `parse_corpus` omits files returning `None`. The validator consequently cannot report a page it never receives.

The extracted CI's class-count check is a useful backstop for dropped classes. It does not establish that every rejected file has a diagnostic, and an unchanged total does not prove unchanged membership. On the actual local Logseq corpus, 8,435 Markdown files produced 8,434 Page objects; this count difference is recorded without asserting why that particular file was excluded.

### Publication flags are not type-checked

The synthetic Page with `vc:public` set to the **string** `"false"` passed validation and was emitted into the page API in both repositories. The parser assigns the value directly and consumers test its truthiness. The actual censuses found no non-boolean publication values, so this is a demonstrated input-validation weakness, not evidence that the current corpus contains this malformed flag.

A publication boundary should require an explicit boolean true and reject ambiguous values with a diagnostic. This deserves stronger treatment than an ordinary display-field validation error because it selects what leaves the authoring workspace.

### Private ancestors can affect public exports

The Logseq fixture used a public child, a private parent and a private grandparent. The public child's own declared parent named only the private parent. `compute_closure` processed all three pages without a publication filter, and the downstream public page API and scaffold exposed the otherwise private grandparent's identifier; the page API also exposed its label. The inferred Turtle export contained that ancestry too. The private page itself was correctly absent from the page API.

This establishes that filtering whole pages is insufficient to prevent derived metadata from crossing the public boundary. It does not establish that any real private page has leaked on the deployed site. A fix needs an explicit policy for references to private entities and for inference over private premises, followed by rejection or redaction tests at every output, not just the page-file count.

### Asserted and inferred RDF use different identity forms

The [asserted Turtle converter](../../../logseq/pipeline/jsonld_to_turtle.py) maps `urn:ngm:class:public-child` to `https://narrativegoldmine.com/class/public-child`. The [closure exporter](../../../logseq/pipeline/reason.py) writes its remembered source URN directly. The fixture confirmed those two forms.

An RDF store loading both files will treat the two IRIs as different terms unless a consumer explicitly normalises or relates them. Loom's inspected graph loader loads both files directly. This is a concrete interoperability risk for joins across asserted and inferred statements; a complete consumer query test remains to be run before describing its extent in the runtime.

## Local authoring state is not the released dataset

The Logseq census found 8,434 entities, 8,420 public pages, one validation error (`DUPLICATE_IRI`), 84 warnings (`INVALID_DOMAIN`) and 1,396 multi-parent informational entries. These are aggregate results from the local `obsidian` checkout. Private page names or contents are not included in the review.

The publisher workflow triggers on `main` and uses the build command whose final exit status fails on validation errors. Its later commentary calling validation “non-blocking” does not override the earlier process status. Therefore this local corpus's report is a reason to inspect branch and publication state, not to infer that the public site currently contains the same defect or is currently failing deployment.

## Explorer lineage and extraction imports

The standalone WasmVOWL repository is distinct from both embedded explorer directories. Hashes of `modern/package.json`, `rust-wasm/Cargo.toml` and `modern/src/App.tsx` match between knowledgeGraph's embedded explorer and Logseq's embedded copy, but differ from standalone WasmVOWL. This is a three-file comparison, not proof of whole-tree identity. The source is vendored into the parent repositories rather than established as one shared checkout by those paths.

The [OntoCast importer](../../../knowledgeGraph/pipeline/ontocast_import.py) is a worthwhile separate boundary. It stages extraction results as private draft candidates, requires `--write` to create files, refuses overwrites, and preserves source statements in evidence blocks. Four of the passing knowledgeGraph tests cover staging, overwrite refusal, slug collisions and unsafe labels. That supports the staging mechanism; a recorded pending-review field is not proof of an enforced human promotion workflow. Such enforcement must be traced separately.

## Priorities for this layer

First, make the public/private decision strict and apply it before deriving public knowledge. Second, make malformed-file diagnostics and entity/page identity checks complete enough to explain every input-to-output loss. Third, define one canonical IRI mapping and an explicit semantic-check contract shared by the asserted, inferred and serving projections. Finally, version the extraction and publication relationship so a reader can reproduce the exact artefact that a model or renderer consumed.

These changes would strengthen the estate's central promise: knowledge is curated once and reused with evidence. The current software already supplies much of the machinery; the remaining work is to make the boundaries as precise as the story about them.

## Producer decision reconciliation — 2026-09-04

knowledgeGraph ADR-2001–2004 and its governing baseline now carry CP-01/02/06/08 acceptance conditions. The count tripwire is preserved, with equal-count identity substitution and publication-policy checks added to closeout. CI invokes validation separately; the build function only logs validation errors. ADR-008/012 are explicitly classified as ontology content in the estate inventory, preserving their page identity.

Revalidation: the existing virtual environment passes all 13 pipeline tests; the four-record ADR validator passes and its index is regenerated. System Python lacked pytest, so the checked-in workflow was exercised with the repository environment. [Receipt](evidence/producer-adr-validation.json).
