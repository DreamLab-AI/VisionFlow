# visionGraph vault and domain contracts

All six VG-01 panels are now assessed against visionGraph `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`. The diagrams now distinguish current asset layout, workflow publication scope and documented defaults, and qualify the domain and Bases gates. ADR-VG-003, ADR-VG-004 and ADR-VG-008 receive section-by-section assessments below. Five other operative ADRs and four other visionGraph topics remain separately queued.

Thirteen scoped probes establish controls and counterexamples. The domain gates accept empty tier files; the supply-chain gate can report success with none of the fifteen expected category IDs; the Bases checker accepts an invalid view type when its name is blank. These are gate-level findings, not demonstrated end-to-end publication bypasses. No product code, corpus page, ADR status or deployment was changed.

## Evidence and scope

[Source digests](evidence/2026-11-01/vg-contract/sources.json) identify committed exports. The [probe harness](evidence/2026-11-01/vg-contract/probes.cjs) executes the exact exported domain gates and shell checker in temporary directories. Its domain fixtures use the real tracked catalogues and research-title register, with synthetic API content and tier bytes. The [results](evidence/2026-11-01/vg-contract/probes.json) record thirteen cases; a passing assertion can mean a defect was reproduced, not that the input satisfies the product contract. The Quartz case executes the exact extracted `shouldPublish` method body, not a complete Quartz render or compiler build.

The [existing Bases run](evidence/2026-11-01/vg-contract/bases-existing.log) passes all ten committed files with yq-go 4.54.1. No Obsidian application session or visual interaction was exercised. The full corpus compiler, reasoner and frontend were not rebuilt in this continuation; prior builds and public observations retain their original scope.

The tracked supply-chain catalogue contains **288 new, 48 rehome and 21 reuse** records, agreeing with the ADR's 357-page breakdown. The space fixture exercises **934 accounted identities** and **936 researched titles**, including reused pages, rather than replacing the real catalogue with a single permissive example. The supply fixture checks all **336 new/rehome identities**. Reused supply pages are outside that gate's accounted loop.

## Findings and acceptance conditions

| ID | Evidence and finding | Consequence and next acceptance condition |
|---|---|---|
| VG-V1 | Both domain scripts use `existsSync` for each domain tier (`check-space-domains.mjs:25`, `check-supply-chain-domain.mjs:36`). The real-catalogue fixtures pass with empty tiers, and space also passes with arbitrary non-NGG1 bytes. Removing a tier fails; changing the space domain ID fails. | A tile's declared presence is not a renderable graph. Decode every required tier, validate its structure and domain identity, and compare its node/edge counts with the declared overview before claiming that these gates verify the public bundle. Other publisher gates may reject these fixtures; no whole-workflow bypass is claimed. |
| VG-V2 | Supply's category loop checks only IDs already present (`39–46`). The fixture provides all 336 catalogue titles/API pages under synthetic IDs, with no category slug IDs, and passes while printing `0 of 15 category hubs are published`. It also accepts declared `memberCount: 1` despite those 336 pages. | Title lookup and nonempty article text do not preserve the stable category identities used by consumers, nor establish a consistent population count. Require the expected category set and join overview, search, API and decoded tiers through stable identities. |
| VG-V3 | If the supply catalogue is absent, the script exits zero after root/domain/tier checks (`50–53`). A root-only fixture passes. | This is a deliberate unfinished-expansion fallback that weakens the release contract when invoked from the wrong directory or with an incomplete checkout. The real repository does contain the catalogue and the workflow runs from its root; this is not a claim that the normal run skips it. Make the release mode require the reviewed catalogue and its digest. |
| VG-V4 | Bases selects invalid types, extracts their `.name`, then tests whether the resulting text is nonempty (`check-bases.sh:15–18`). An invalid type with `name: ""` passes; the same invalid type with a nonempty name fails. | A display label controls whether a validation error is detected. Count invalid records independently of labels and validate the intended name contract separately. |
| VG-V5 | Formula reference discovery covers order, sort and property keys (`19–23`), but not filters or formula expressions. An undeclared formula in `filters` passes; the same reference in `order` fails. | A green checker does not prove that the curator's filters are valid. Define the supported expression surface, parse references across that surface, and retain an application-level fixture proving intended view membership. |
| VG-V6 | Quartz's configured `ExplicitPublish({ key: "public" })` accepts boolean true and the string `"true"` (`explicit.ts:23–25`); false, `"false"`, 1, 0 and null fail in the predicate matrix. The README describes typed true alone. | The downstream predicate is broader than the documented type contract. An upstream typed compiler gate may prevent this value from reaching Quartz, so this is not a demonstrated private-page leak. Test the complete staging-to-render path before describing a strict end-to-end guarantee. |
| VG-V7 | The README calls the compiler pin v0.2.0; agent rules call it v0.5.0; the workflow pins v0.7.0 plus a digest. README also calls `knowledge/index.md` committed, though the tracked tree has no such file and `vault.toml:30–31` calls it a build artefact. The previous diagram showed knowledge assets as a symlink, but the tree and manifest show a real subset. | Maintainers can run the wrong compiler or look for an absent source artefact. Reconcile the instructions against the workflow and tracked layout. The diagram no longer repeats the old sizes or universal no-hardcoded-path claim. |
| VG-V8 | Manifest working role says `publishable=false`; its publish section says knowledge-only, while the workflow explicitly selects `--vault all` and stages public working content (`publish.yml:155–181`). | Role defaults, ontology projection and public notes are different policy surfaces. Document which fields are actually consumed and how CLI overrides interact with them. Do not interpret this documentary tension as proof that private working pages are published. |
| VG-V9 | The three assessed ADRs explicitly say their recorded `verified_commit` identifies the record's last modification, not a code verification, and `verified_paths` is empty. ADR-VG-003 also records a conservative status-enum conversion awaiting owner confirmation. | These hashes and status labels are not implementation receipts. Bind each decision to the actual source paths and evidence before treating staleness checks as armed or interpreting complete/staged as a newly verified result. |

The full pipeline also contains validation, collision, certificate, layout and browser checks. These probes isolate the named gate contracts and do not establish that an arbitrary malformed synthetic bundle would pass every preceding and subsequent step.

## ADR-VG-003: Obsidian-only authoring

| Section | Assessment |
|---|---|
| Metadata | Accepted, partial, staged; review trigger names an alternative parser/writer. Those states preserve a source/activation distinction. `verified_paths: []` leaves the record's source verification incomplete. |
| Status mapping, 31 October | Explicitly maps non-enum historical statuses conservatively and requests owner confirmation. Retain that qualification; do not promote the record to complete/live based on the enum conversion. |
| Verification note, 31 October | Admits the recorded commit is documentary history, not a fresh code check. This assessment supplies evidence for the inspected publisher, manifest and gates only. |
| Decision | YAML frontmatter/Obsidian remains the declared active authoring format. The reference to the Rust `vault` CLI lags the current publisher's pinned `ontomark` command. Retaining historical evidence and keeping unrelated private notebooks out of scope remain coherent requirements. |
| Consequences and activation | The record correctly distinguishes removing source from replacing baked binaries or pinned publishers, and does not authorise rebuild or publication merely because tests pass. The current workflow proves the named producer configuration, not a complete estate-wide legacy-binary removal. |
| Verification instructions within that section | Requests maintained vault tests, transcript-writer integration, corpus validation/body checks, skill lint and explorer compilation. The present gates and Bases probes do not satisfy that full list. Cross-repository runtime acceptance and a current transcript-writer run remain open; the report does not substitute historical execution links for those results. |

## ADR-VG-004: two linked domains

| Section | Assessment |
|---|---|
| Metadata and verification note | Accepted, complete, staged, with the explicit documentary-only commit caveat and no verified paths. Complete is not re-certified by this audit. |
| Decision | The named Space Science and Systems and Earth Observation and Geospatial Sensing roots appear in vocabulary metadata and have stable-ID assertions 6 and 7 in the gate. The programme umbrella is not substituted for either root. |
| Rationale and boundaries | Keeping navigation, membership and `is-a` distinct is sensible for a multi-domain ontology. The manifest disables domain-root disjointness; the vocabulary's provisional `disjoint-with` documentation excludes roots/categories. This is source-policy alignment, not a fresh proof of all emitted OWL semantics or cross-domain page relations. |
| Consequences | Stable IDs and minimum populations are checked, along with every accounted catalogue title. Tier bytes and meaningful bridge semantics are not checked by this script. The sentence describing private, undeployed roots is historical: the retained 1 November public index contains both Class roots with their expected IDs and IRIs. That partial publication observation does not prove full consumer acceptance or answer the competency questions. |
| Downstream consumers, 7 October / historical note, 11 October | The note correctly qualifies the earlier knowledgeGraph viewer paths and package pins. The [retained-viewer assessment](2026-11-01-knowledgegraph-explorer-contracts.md) already establishes that the active publishing copy differs. A current cross-consumer domain/rendering receipt remains required; public root entries alone cannot move this record to live. |

## ADR-VG-008: supply chains and logistics

| Section | Assessment |
|---|---|
| Metadata and verification note | Accepted, complete, staged, again with no code-verification claim behind the recorded revision. Source coverage and activation evidence need explicit attachment. |
| Decision | Vocabulary includes the new root and the gate asserts ID 8. The saved public index identifies it as a Class under the expected IRI. The gate does not independently verify the complete legacy 0–7 mapping. |
| Context | Counts and the former infrastructure alias describe pre-expansion conditions, not the current distribution. They must stay dated rather than become present-day source counts. |
| Naming | The distinct root title appears in the catalogue and published index. This review does not re-run the broader collision policy or certify every retained concept alias. |
| Rules: rehome and bridging | The catalogue breakdown matches 288 new/48 rehome/21 reuse. The gate checks new/rehome API metadata and text, but excludes reused pages and does not verify bridge direction, symmetry, ontology closure or preserved IRIs. The rules are broader than this gate. |
| Rules: homonyms, taxonomy and pure TBox | These are content constraints, not proven by the existence of a domain tier. The isolated gate accepts synthetic article text and only checks selected metadata; factual correctness and all relation policies need separate evidence. |
| Rules: licences and attribution | Refers to ADR-VG-005 and a named generator/attribution entry. This pass does not certify every admitted source's licence, byte pin or attribution. The dedicated source-admission review remains queued. |
| Outcome, 18 October | The 357-page action breakdown is reproduced from the tracked catalogue. Reported research batches, 165 reviewed bridge pairs, 10 competency questions and 55 edge checks remain historical outcome claims here; they were not re-executed. The release gate's success does not validate those claims. |
| Consequences | Appended category positions are declared, but the gate accepts zero of the expected fifteen category IDs and does not decode the tier. The public-root observation establishes partial distribution, not owner approval of every current source change or successful downstream rendering. No push was performed. |

## Diagram disposition

| Panel | Result |
|---|---|
| VG-01.1 | Corrected flattened roots and asset layout, removed unmeasured sizes and the blanket path-authority invariant, and recorded committed-index drift. |
| VG-01.3 | Mapped the actual all-vault staging command and exact Quartz predicate. Separate consumer behaviour remains owned by KG-06/VC-21 rather than copied from the historical publication table. |
| VG-01.4 | Replaced claims that old failures remain unchanged with a historical/current evidence boundary and the still-proposed acceptance records. |
| VG-01.5 | Separated manifest declarations from workflow overrides; corrected the compiler pin and removed claims that declaration alone proves enforcement. |
| VG-01.6 | Preserved the ten-view-file census while narrowing checker guarantees to its implemented fields and reproduced limits. |
| VG-01.7 | Preserved stable domain-ID checks and added missing binary, category-set and catalogue-required checks. |

The [public-root extract](evidence/2026-11-01/vg-contract/domain-census.json) derives from the complete, hash-verified search index retained by the preceding [public-receipt audit](2026-11-01-knowledgegraph-inventory-and-live-receipt.md), observed at generation `visionGraph@84c1cee32b6ad03b58bcb31ba33320142c330523`. It is explicitly that dated sample, not a new network observation or an exhaustive domain audit.

## Comparison and user utility

The [official Obsidian Bases syntax](https://help.obsidian.md/bases/syntax) permits formulas in filters and other formulas, and view types supplied by plugins. This is broader than the script's four-type allowlist and selected-field reference scan. The useful comparison is a versioned local lint profile plus application fixtures, rather than calling this script complete conformance to an evolving upstream schema.

For a curator, acceptance should show that a view includes the intended review queue and that broken filters fail clearly. For an ontology reader, a domain should expose its expected category identities, a decodable graph, preserved entity identity and useful cross-domain answers. Those are observable user outcomes; neither a nonempty description nor a successful existence check establishes them. Keep the lightweight metadata checks, but attach those stronger acceptance cases before presenting a new domain as fully supported.


## Verification and reproduction

The final VG-01 topic passes strict citation checks with 39 revision-verified citations and six successful renders, all visually inspected. The complete catalogue gate passes at 180 topics, 1,599 panels and 24,576 revision-verified citations, with zero warnings, fallback or unresolved citations and nine existing artefact exceptions. Verification freshness covers 45 declared revisions. Local links and `git diff --check` pass. Whole-catalogue mechanical validation does not imply a new semantic review of the other topics.

To reproduce, export the source-manifest files from the pinned visionGraph revision into a temporary root preserving relative paths, then run `node probes.cjs /absolute/export/root`; Node and yq-go must be on PATH. Run `bash scripts/check-bases.sh` from that export to repeat the ten-file control. The harness creates and removes its own fixture directories, never writes the authored vault and never invokes publication. [Retention receipt](evidence/2026-11-01/vg-contract/retention.json) records the local source exports and rendered artefacts under `/tmp/vf-vg-contract-artifacts`.
