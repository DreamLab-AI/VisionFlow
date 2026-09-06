---
title: Current authored vault and producer transition
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Current authored vault and producer transition

The current authored-corpus boundary is **visionGraph**, not the historical Logseq path. This emerged from tracing agentbox's configured input, and it changes the interpretation of the earlier [knowledge-production assessment](knowledge-production.md): the Logseq implementation remains important as lineage, but the current producer needs its own evidence.

The local visionGraph checkout is on `main` at `fabcdbcc9fd67951087a9492fb45780bd379134b`, with origin `jjohare/visionGraph`. Its [README](../../../visionGraph/README.md) describes a history-preserving split from Logseq's `obsidian` branch, renaming the knowledge and working vaults. Older source commit identifiers resolve in the archived Logseq history, because the split rewrote retained commits. The review now includes fourteen repository identities, with different roles for current production, distribution, history and extensions.

[Current-producer receipts](evidence/visiongraph-snapshot.json) record the source hashes, aggregate census, synthetic probes and test output. They do not establish which revision is deployed remotely.

## Actual consumer binding

The checked [agentbox manifest](../../../project/agentbox/agentbox.toml) sets `[vault].root` to `/home/devuser/workspace/visionGraph/knowledge`. Its [entrypoint](../../../project/agentbox/config/entrypoint-unified.sh) derives and exports `VAULT_PAGES` and related paths. This is stronger evidence of the intended local consumer binding than the older public repository map.

visionGraph contains `knowledge`, `working`, transcripts, the publishing pipeline and an embedded explorer. Its [publisher](../../../visionGraph/.github/workflows/publish.yml) reads `knowledge/pages`, runs the pipeline and gates, builds the explorer, and targets knowledgeGraph's `gh-pages` branch. The public knowledgeGraph distribution remains a separate source tree. A release account must distinguish all three: current authoring revision, extracted distribution revision, and deployed artefact revision.

The visionGraph README formerly pointed to a missing vault-format document under VisionFlow. Its link now resolves to the inspected governing document in [VisionClaw](../../../project/docs/VAULT-corpus-format.md).

## Two inclusion policies coexist

The Obsidian vault contract describes frontmatter-based inclusion for VisionClaw: `public: true` or an `owl-class` field can admit a page to its graph. The site publisher's [parser](../../../visionGraph/pipeline/jsonld_parser.py) still reads `vc:public` from the JSON-LD Page fence. This is deliberate compatibility with the existing structured corpus, but it creates two editable representations whose names can look equivalent to an author.

Publication also excludes `_misc` and dot-directories, while namespace directories are traversed recursively. Those rules are independent of the Page publication flag. Therefore a page can belong to the local workbench's corpus without belonging to the public site. The UI and authoring tools should identify which inclusion policy they are changing; a frontmatter switch alone is not proof of site unpublication.

Agentbox's local ontology backend has another projection: it reads top-level Class fences without the publisher's Page-public filter or recursive namespace walk. Its synthetic namespaced class was invisible while a private top-level class was visible. This is a tested consumer mismatch, not a claim that trusted local access to private authoring material is inherently wrong.

## Current corpus and tests

The local census found 8,672 Markdown files recursively under `knowledge/pages`, of which 8,435 were top-level. The publisher parser returned 8,447 entity-bearing pages and marked 8,433 public. The recursive-file total includes directories and files outside the publisher's accepted set, so the difference is not automatically parse loss. Publication flags in the parsed corpus were all booleans.

Validation reported one duplicate-IRI error, 84 invalid-domain warnings and 1,397 informational multi-parent entries. The current test run reported **57 passed, 1 failed**. The failing test, `test_real_corpus_publishes_nothing_from_misc`, had already checked that the publication walk excludes `_misc`; it then required at least 14 Markdown files in that directory but found 8. Thus the observed failure is its corpus-size assertion, not a demonstrated failure of `_misc` exclusion.

The tests run before deployment in the checked workflow. The source revision and test result do not tell us whether a remote run used the same corpus state, but they do show that “all current local publisher tests pass” would be false. No failing test was altered to make the review green.

## Earlier boundary gaps survive the split

The shared [probe collector](evidence/knowledge-probes.py), invoked with `visionGraph`, reproduced the same cases on the current producer:

- A malformed JSON-LD fence can be omitted before validation, leaving a zero-error report over zero parsed pages.
- A string `"false"` publication value is truthy and emits a page; current corpus flags do not contain that malformed value.
- Closure derived from a public child through private ancestors exposes the private grandparent in public API/scaffold/inferred-Turtle outputs, while the private page file itself remains excluded.
- Asserted Turtle uses normalised HTTP class IRIs while inferred Turtle preserves the source URNs.

These are synthetic, local publication-boundary results. They establish that the implementation weaknesses are not confined to archived history. They do not establish that a real private page has leaked to the public site.

## History, authoring and public experience

The split is a sensible simplification of repository responsibility: current content has a clear home, while historical citations retain a resolution target. Moving from flat Logseq filenames to namespace folders also makes the corpus usable as an Obsidian vault. The namespace-aware publisher and its tests are concrete work supporting that transition.

The surrounding estate has not fully caught up. The public repository inventory omits the current source of truth, a governing-document link points to the wrong repository, the direct local consumer misses namespace pages, and publication still depends on a separate JSON-LD flag. The legacy `/notes` surface is explicitly frozen in visionGraph's README and preserved by its deployment workflow; it should not be presented as the live equivalent of the current working vault.

Complete the transition by documenting current versus historical authority, proving the equivalence or intended difference of each inclusion rule, and testing consumers against a shared corpus fixture containing namespaces, private pages, `_misc`, malformed metadata and conflicting identities. That would make a corpus move an observable contract change rather than an assumption hidden in paths.

## Current decision pack

visionGraph now has proposed ADR-VG-001/002 and a [publication contract](../../../visionGraph/docs/PUBLICATION-contract.md) covering inclusion, derived visibility, generation and consumer identity. They preserve the distinction between implemented components and an unaccepted complete contract. [Source revalidation](evidence/visiongraph-closeout.json) matches all 18 earlier receipt hashes before the README link amendment; existing corpus deletions remain untouched. The earlier 57-pass/one-failure result is dated evidence, not a new test run. Current and historical ADR-named ontology pages must remain classified as content.

## Historical decision lineage

The [Logseq decision companion](closeout/logseq-decision-lineage.md) maps all six historical design candidates to continuing responsibilities and acceptance work. It preserves historical declarations and the archived tree. Two rendered ADR-008/012 pages are now correctly classified as ontology content. The IRI gate remains baseline-aware: repair acceptance must include removing obsolete baseline entries and testing recurrence, as well as proving that the resulting generation reaches Loom.

## Runtime path overrides and Notes launch

[Current path receipts](evidence/vault-path-probe.json) pin the entrypoint, local/index consumers, condensation paths and Notes launcher. The [isolated resolver probe](evidence/vault-path-probe.py) invokes the actual extracted function with a stub empty manifest reader: it reports vault disabled and clears VAULT_PAGES, but retains a pre-existing ONTOLOGY_PAGES_DIR. The inspected consumers prefer that legacy override. This is compatibility behaviour, so the declared global disabled state does not guarantee that every corpus consumer is off. No real corpus was read by the probe.

The Notes script supplies a workspace/vault fallback and resolves Rune from PATH or the workspace cargo bin. It launches when a binary is found; the inspected script does not check VAULT_TUI or AGENTBOX_VAULT_ENABLED. A package omitted from a new image can therefore differ from binary discovery on a retained bind mount. This is source evidence, not a live tmux reproduction. A missing vault falls back to the workspace, and failure to create the recovery HOME leaves a degraded launch possible.

Both shell files pass bash syntax checking. That does not prove page editing, external-change merge, crash recovery or current-image activation. Agentbox ADR-2028/2029 now require a precedence/off-state matrix, path relocation with namespaces/private pages, binary-present but mode-none behaviour, and recovery evidence. The earlier corpus and publication tests retain their original scope.

## Converter collision and dry-run boundaries

All 70 unit and 16 integration tests for vault-migrate pass in this checkout. Additional [actual CLI fixtures](evidence/vault-converter-probe.json) use only invented temporary pages. A legacy pages/A___B.md and an existing pages/A/B.md both map to pages/A/B.md. Conversion exits zero; the namespace body's sentinel remains and the folder body's sentinel is absent. Both source files remain unchanged. This demonstrates output collision loss, not alteration of the owner's corpus. [Reproducer](evidence/vault-converter-probe.py) records the exact temporary cases.

The library's claimed-path set prevents starter configuration overwriting planned content, but does not reject duplicate content destinations. Actions are sorted and applied sequentially. Thus deterministic conversion and source preservation do not establish preservation of every distinct input page. Before in-place use, the planner needs an explicit collision policy covering legacy/folder names, journal renames, case/normalisation differences and assets.

The second fixture combines --dry-run with --report: it writes the requested report while creating no vault output. This narrows the ADR's claim that dry-run writes nothing. Report output must either be a documented exception or be refused in strict no-write mode. Check/report and output/source aliasing also require explicit path policy.

CP-01/02/08 requires destination uniqueness before writes, complete input-to-output accounting, failure/recovery evidence and consumer validation of the resulting corpus. Source-safe output mode is a useful safeguard; it does not make an incomplete output safe to promote. ADR-2042 becomes partial for lossless migration and unconditional dry-run claims. No real vault or in-place migration ran.

## Inclusion typing and local fallback

The current shared PageMeta parser and link module pass 56 native tests. Frontmatter public requires a YAML boolean; quoted true does not qualify. Frontmatter takes precedence over leading legacy properties, and body-only markers do not become the publication gate. Explicit public false is not an exclusion override: a non-empty owl-class still enables KG inclusion, as the current policy and tests deliberately specify.

The owl-class parser uses a general scalar renderer, accepting booleans and numbers as non-empty strings. It does not validate a class IRI. Thus malformed typed markers can satisfy the formal-data exception; this is source-derived and requires a parser/ingest negative control, not a claim of observed private-data disclosure. The [receipt](evidence/vault-inclusion-snapshot.json) identifies the inspected parser and consumer sources.

GitHub sync uses the first parsed node's class marker or PageMeta inclusion and drops linked-page stubs from this authored-page path. A separate startup fallback calls FileService::scan_local_files_to_metadata after GitHub-sync failure or empty metadata. That function's inclusion check is commented out and it creates metadata for local markdown regardless of the gate. Startup logs nevertheless call the result public files. This establishes a local metadata-admission difference; downstream graph ownership, visibility and publication must be traced before inferring any user-visible leak.

KG inclusion, node visibility and public-site publication are separate policies. The formal-data exception must be explicit to authors and validated at the intended authority boundary, and every fallback needs a documented contract. CP-01/02/04/08 requires malformed/wrong-type/empty class fixtures, private-plus-class cases, source fallback parity or a justified narrower scope, and a migration receipt before removing legacy tolerance. ADR-2040 is partial for its complete typed/shared-reader guarantee; ADR-2014 retains historical lineage with its old scanner claims qualified. No real vault, sync or publication ran.
