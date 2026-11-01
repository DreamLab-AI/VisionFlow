# knowledgeGraph decision inventory and public-generation receipt

The knowledgeGraph source-review inventory is now accounted for: five existing topics contained fifteen panels, four operative ADRs have section assessments, and the two archived ADR-named pages are corpus content. This continuation adds one decision-boundary panel and corrects the publication sequence to show the static-context overlay. Implementation findings and browser/runtime acceptance remain open; the estate audit is not complete.

A read-only public HTTP sample also establishes a concrete receipt defect: the served `/ns/v2.jsonld` does not match the root generation manifest. Six other sampled artefacts match their declared uncompressed bytes, including two gzip-delivered pages. This narrows the result to a reproducible publication-manifest mismatch, not a general claim of corrupt deployment.

## Inventory method and disposition

The census reads knowledgeGraph `4ed9ac159daf402b4fb252ce559dbeb894a8d91e`. It enumerates all **8,397 tracked paths**, selects the ADR-named files and ledger-support files, and independently searches Markdown for an ADR heading, ADR identity field or decision-status field. The [path/digest inventory](evidence/2026-11-01/kg-inventory/decision-inventory.json) and [content search](evidence/2026-11-01/kg-inventory/content-discovery.txt) agree: four operative decisions, two archived content pages and a template. The preamble, generated README and generator complete the ten-file inventory. No nested diagram catalogue was found in the tracked knowledgeGraph tree.

| Surface | Disposition and evidence |
|---|---|
| ADR-2001 | Full section supplement in [explorer contracts](2026-11-01-knowledgegraph-explorer-contracts.md#knowledgegraph-adr-2001-section-dispositions). This continuation checks the ledger tooling and content classification that support its consolidation decision. |
| ADR-2002 | Full section supplement in [publication gates](2026-11-01-knowledgegraph-publication-gates.md#knowledgegraph-adr-2002-section-dispositions). Historical pipeline evidence remains scoped to its implementation. |
| ADR-2003 | Full section supplement in [publication gates](2026-11-01-knowledgegraph-publication-gates.md#knowledgegraph-adr-2003-section-dispositions). Count and identity tripwires are not reclassified as current final-site verification. |
| ADR-2004 | Full section supplement in [publication gates](2026-11-01-knowledgegraph-publication-gates.md#knowledgegraph-adr-2004-section-dispositions). Frozen content identity is preserved; the archive relocation tension remains recorded. |
| Archived ADR-008 and ADR-012 | Both explicitly say they are frozen corpus content, not governance. Each carries page metadata, a class block, link-resolution annotation and explanatory prose. Their section assessment is below. |
| `docs/adr/PREAMBLE.md` | Governing-document table and lookup order correctly distinguish normative baseline from reference prose. Legacy-document paragraph identifies absent ADR-NG-001 and upstream LCR records; it still names the retired `ontology/pages/` location. Decision-making instructions describe metadata validation, not substantive evidence validation. |
| `docs/adr/README.md` | Generated index of the same four operative records, all labelled accepted/complete/live. Those fields do not override the dated assessments or establish successor deployment. Preamble path drift is propagated into this generated file. |
| `docs/adr/TEMPLATE.md` | Defines Context, Decision, Consequences and Verification, plus three status axes. Useful authoring structure; the checker does not enforce the body sections or resolve its `verified_commit` field. |
| `scripts/adr-index-gen.js` | Required-field, enum, list and duplicate-ID checks work. Supersession checks warn, and `--check` neither compares the generated index nor validates substantive evidence. Eight probes below bound the claim. |

The absent ADR-NG-001 and LCR documents remain absent or external at this revision. Their cited summaries do not become newly ratified decisions, and this review does not invent missing records.

## Archived content, section by section

The two pages have parallel structure; the findings below apply to each independently. Their full bytes were read at the pinned revision and are hashed in the inventory.

| Section | ADR-008 and ADR-012 assessment |
|---|---|
| Publication property and frozen banner | `public:: true` is a publication property. The banner explicitly denies decision-record authority and points to the actual ledger; no decision-status inference follows from the filenames. |
| Page JSON-LD | Distinct immutable page identifiers, slugs and titles distinguish the two content records. Publication is a boolean true, schema version 2, with generated attribution and timestamps. This is provenance metadata, not proof of a real architectural decision. |
| Class JSON-LD | Each declares `@type: Class`, a generic numbered-ADR definition, a Software Architecture parent and an inferred `requires` relation. Neither identifies an actual project choice, alternative or decision-maker. The generic concept is content, while the numbered labels can misleadingly suggest real decisions. |
| Link-resolution annotation | Resolves the two displayed software links and points to that page's identity. The annotation records a mapping; it does not establish the truth of the concept definition or a governance relationship. |
| Definition | Repeats a generic explanation of a numbered record, with no decision outcome. The frozen classification is consistent with that absence. |
| Semantic Classification | Prose says `owl-role:: Individual`, while the machine block says Class. This is an archived content contradiction. Do not silently reinterpret either as an operative ADR. |
| Relationships | Human prose includes a Software Engineering bridge absent from the machine `relations` object, which only contains `requires`. Link annotation and typed relation projection are different denominators; the page does not establish lossless graph equivalence between them. |
| Content | Explains ADR practice in general. It contains no project-specific Context/Decision/Consequences/Verification body to assess as an architecture decision. |
| Provenance | The prose `sources::` field is empty and records a migration date. Machine provenance names an automated gap-fill rule. That supports synthetic origin, not externally verified factual quality or human approval. |

These content findings are not inherited by today's published pages. The live compressed Markdown sample lacks the archived `owl-role:: Individual` string, so this review does not claim that particular contradiction remains public.

## Validator controls and limits

The [eight-case harness](evidence/2026-11-01/kg-inventory/adr-probes.cjs) runs the exact exported generator in temporary directories. [Results](evidence/2026-11-01/kg-inventory/adr-probes.json): the real four-record pack passes; invalid status, missing owner and duplicate ID controls fail as expected. Four further cases delimit what a green check means:

| ID | Observed behaviour | Consequence and acceptance condition |
|---|---|---|
| KG-I1 | A record with no body, a non-date date and `verified_commit: not-a-revision` passes. | The check is not evidence that a decision has rationale or an executable verification basis. If used as an evidence gate, require the body contract, a resolvable source revision and a separately verified receipt. |
| KG-I2 | Missing supersession targets warn but exit zero; a reciprocal two-record cycle passes without warnings. | A green metadata check does not establish a coherent decision lineage. Decide which external references are allowed, detect local cycles and fail on invalid local lineage. No such cycle was found in the actual four records. |
| KG-I3 | A deliberately stale README passes `--check` and stays untouched. | This command is validation-only by implementation. A freshness gate must generate into a temporary destination and compare, or implement an explicit index comparison. This probe is not evidence that the current README table is stale. |

No production validator was changed. These are bounded tooling gaps and authoring risks, rather than a claim that every field it accepts is invalid or that metadata lint should itself perform a full architecture audit.

## Public HTTP observation and manifest comparison

Read-only observations ran on **1 November 2026, 04:30–04:34 UTC**, without authentication. [HTTP receipt](evidence/2026-11-01/kg-inventory/http-observations.json), [repeat receipt](evidence/2026-11-01/kg-inventory/http-recheck.json), [complete index receipt](evidence/2026-11-01/kg-inventory/http-index-complete.json), [comparisons](evidence/2026-11-01/kg-inventory/manifest-comparisons.json), [page receipts](evidence/2026-11-01/kg-inventory/http-declared-pages.json) and [summary](evidence/2026-11-01/kg-inventory/live-summary.json) record the sampled bytes and identities.

The [root generation marker](https://narrativegoldmine.com/.generation.json) and [data marker](https://narrativegoldmine.com/data/.generation.json) report `visionGraph@84c1cee32b6ad03b58bcb31ba33320142c330523`, generated `2026-10-31T14:46:05.322496+00:00`, clean source, 9,595 pages/classes and the same content and ontology digests. The source commit exists locally and is an ancestor of the audited visionGraph HEAD `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`; the observed site is therefore not assumed to run every current local change. The root manifest lists 38,892 artefacts and the data manifest lists 18. These counts are declarations, not a full remote inventory verification.

The [public census](https://narrativegoldmine.com/api/census.json) reports 9,595 public pages, 14 non-public pages, no rejections and one exclusion. The complete search index contains 9,595 unique IDs. The [graph statistics](https://narrativegoldmine.com/data/graph/stats.json) report 9,595 classes, zero individuals and 9,593 pages; those separate counters are recorded without treating their difference as a proven dropped-page defect. No private names or contents were requested.

**KG-I4 — served context disagrees with the root manifest.** The manifest lists `ns/v2.jsonld` as 6,888 bytes with SHA-256 `452d5ce09759911587032b30a6a4cc5270fe130c92b7f658f37631ff9a97160f`. The [served context](https://narrativegoldmine.com/ns/v2.jsonld) is 3,371 bytes with SHA-256 `7b0d8f4f456677b47b0d928fc0f0185bb57c1d3ff1d46adf59f3312cb7cb4125`. Repeating both requests with a no-cache request directive produced identical bytes and ETags. This is a repeated observation, not a guarantee about all CDN nodes.

The served context matches `visionGraph@84c1cee:static/ns/v2.jsonld` exactly. That revision's publisher copies `static/.` into `site-data/` at line 252, after compilation; the current workflow does so at line 262. The compiler creates the generation markers before promoting its output (`ontomark 0caae5d:crates/ontomark-compile/src/build/mod.rs:990–1018`). This source ordering and byte match explain the observed overwrite. They do not demonstrate malicious tampering or semantic incompatibility of the context.

Six other distinct sampled artefacts match the root manifest: the data marker, census, graph statistics, complete search index and the two decoded Markdown pages. The first index fetch was capped at 8 MB and is explicitly marked truncated; a second, complete 9,367,017-byte fetch supplied the matching hash and parsed count. Truncated bytes were excluded from checksum acceptance.

**Representation qualification.** The manifest names `api/markdown/ADR-008.md` and `ADR-012.md`. Their compressed URLs return 200, and decompression reproduces the listed hashes. Guessed lowercase ID paths return 404 in both raw and gzip form; this is not a finding that the declared pages are missing. The search index itself is served raw, while a guessed `search-index.json.gz` returns 404. Consumers must use the declared naming and representation contract. No browser navigation, rendering, GPU behaviour, exhaustive link crawl or private Loom/VisionClaw acceptance is established by these requests.

The practical acceptance condition is a final-site receipt produced after overlays and compression, identifying stored and decoded representations explicitly. A compiler manifest can remain a useful compiler receipt if consumers are told its scope; it cannot be treated as an unchanged inventory of the deployed website when a listed file differs. This finding strengthens the earlier [publication-gate assessment](2026-11-01-knowledgegraph-publication-gates.md), which was limited to local probes.

## Diagram coverage and comparison

| Topic | Assessed panels and evidence |
|---|---|
| KG-01 | Two panels, [consumer/archive review](2026-11-01-knowledgegraph-consumers-and-archive.md). |
| KG-04 | Five panels, [explorer review](2026-11-01-knowledgegraph-explorer-contracts.md). |
| KG-05 | Two panels, [publication review](2026-11-01-knowledgegraph-publication-gates.md); this continuation adds the manifest-then-overlay ordering to KG-05.3. |
| KG-06 | Three panels, [consumer/archive review](2026-11-01-knowledgegraph-consumers-and-archive.md). |
| KG-07 | Three original panels, [publication review](2026-11-01-knowledgegraph-publication-gates.md), plus new KG-07.4 documenting the decision inventory and validator boundary. |

There are now **five topics and sixteen panels**. KG-02 and KG-03 are not missing current files: [commit `eab90d3`](evidence/2026-11-01/kg-inventory/retired-topics.txt) removed them with the retired producer diagrams on 30 September. The source was archived, so “deleted pipelines” in that commit message should be read as deletion from the active producer paths, not destruction of retained history. This coverage statement closes the knowledgeGraph diagram/decision inventory queue, while leaving its documented implementation findings and acceptance work open.

[MADR](https://adr.github.io/madr/) distinguishes decision rationale from its optional Confirmation section, which records how implementation compliance will be checked. VisionFlow's separate decision/implementation/activation axes are useful additional distinctions, provided they link to evidence of each claim. The user benefit is an audit view that shows what was decided, what source implements it, and which release actually serves it; a green metadata linter or a familiar ADR-shaped title cannot supply all three.


## Verification and retained evidence

Eight validator probes completed with the documented outcomes. Both touched topics pass strict citation checks: 48 citations, no fallback, unresolved citations or exceptions, and six successful renders. The changed sequence and new inventory panel were visually inspected after the final edits. The complete catalogue passes at 180 topics, 1,599 panels and 24,600 revision-verified citations, with zero warnings and nine pre-existing artefact exceptions; freshness covers all 44 declared revisions. Local links and `git diff --check` pass. These are mechanical catalogue checks, not a claim of a fresh semantic review of every estate panel.

The [retention receipt](evidence/2026-11-01/kg-inventory/retention.json) identifies local raw HTTP bodies, source exports and render artefacts. Committed evidence contains the small observation/verification records; the large public manifest and complete index remain under `/tmp/vf-kg-inventory-artifacts`. Reproduction should first rerun the exported validator harness, then independently fetch the listed HTTPS endpoints, preserve status/encoding and complete bytes, decompress the declared page representations and compare SHA-256 and byte length to the marker. A later live generation is a new observation, not a reproduction of these historical bytes.
