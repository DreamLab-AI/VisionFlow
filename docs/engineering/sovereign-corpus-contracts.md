# Sovereign Corpus — interface contracts for the one-shot (PRD-sovereign-corpus §7)

These are the seams between workstreams. A worker may extend a contract additively; it may not change
a field's meaning without updating this file in the same change. Frozen 2026-09-22; amended 2026-09-22 09:20 (C3: +ontology.json, +data/graph tiers, context path corrected) after the publish research.

## C1. `ontology/vocabulary.yaml` (WS-B writes, WS-C/D/E consume)

```yaml
version: 1
namespace: "urn:ngm:class:"            # resource IRIs are namespace + slug(title)
context: "https://narrativegoldmine.com/context/v1.jsonld"
types:                                  # OKF `type` values permitted in knowledge/
  Class:      { owl: owl:Class, required: [resource, status] }
  Property:   { owl: owl:ObjectProperty, required: [resource, status] }
  Individual: { owl: owl:NamedIndividual, required: [resource, status] }
working_types: [Note, Episode, Transcript, Draft Concept, Journal, Canvas]
relations:                              # list-of-wikilink keys
  is-a:       { owl: rdfs:subClassOf, characteristics: [transitive] }
  requires:   { owl: ngm:requires,  inverse: required-by }
  enables:    { owl: ngm:enables,   inverse: enabled-by }
  has-part:   { owl: ngm:hasPart,   inverse: part-of }
  part-of:    { owl: ngm:partOf,    inverse: has-part }
  # … every predicate actually present in the json-ld Class fences, no more, no fewer
scalars:
  domain:   { type: text }
  maturity: { type: text, enum: [emerging, established, mature, legacy] }   # from the corpus
  quality:  { type: number, min: 0, max: 1 }
okf:                                    # OKF v0.2 fields the build validates
  lifecycle: { status: [draft, stable, deprecated], stale_after: date }
  trust:     { generated: {by: actor, at: datetime}, verified: [ {by: actor, at: datetime} ] }
  actors:    { human: "human:<npub>", process: "process:<name>/<version>", agent: "<producer>/<version>" }
```

Unknown frontmatter keys in `knowledge/` fail `vault validate`; `working/` tolerates unknown keys (OKF §4.1).

### C1 additions (WS-C, 2026-09-22) — required for a lossless migration

Three fence facts cannot be recomputed from a wikilink, so they are migrated
verbatim and declared here. Counts are measured on the live corpus.

```yaml
scalars:
  slug:       { type: text }             # 343/8,446 pages: slug != slugify(title)
  label:      { type: text }             # rdfs:label when it differs from the title
  definition: { type: text }
  links:      { type: text, list: true } # the curated `vc:outboundWikilinks` set.
                                         # Differs from a body scan on ~50% of pages
                                         # and is what EVERY backlink list in the
                                         # build is computed from. Always written by
                                         # `vault migrate`, even when empty: an absent
                                         # key means "scan the body".
```
`resource` (already in C1) is likewise data, not a derivation: 96/8,446 classes
have an IRI whose tail is not the page slug. `page_resource` is reserved for the
historical `urn:visionflow:page:<hash>` identity, which the page API emits as `id`.

```yaml
tail_iris:                               # label -> IRI, for long-tail concepts that
  "IDA*": "urn:ngm:class:ida-star"       # have no page AND whose IRI is not
                                         # namespace + slugify(label). An override;
                                         # normally unnecessary, because `vault
                                         # migrate` writes a long-tail reference as
                                         # `[[<iri-tail>|<label>]]` so the slug — the
                                         # only part any v1 artefact reads — survives.

migration:                               # consumed ONLY by the one-shot; deleted with it
  fence_fields:   { "<json-ld key>": <frontmatter key> }   # `relations.<k>` for nested
  ignore:         [ "@id", "@type", … ]                    # deliberately dropped
  logseq_keys:    { "<key>": <frontmatter key> | null }     # null = drop the line
```
`vault migrate` exits **2** and writes nothing when any fence key or Logseq key
is absent from this map.

**Namespace normalisation (WS-C).** Long-tail references in the legacy
`urn:visionflow:linked:` and `urn:visionflow:owl:class:` namespaces are
normalised onto `namespace` with the slug preserved verbatim. `scaffold-index`
byte-parity is unaffected (every v1 artefact keys the tail on `ref_slug`);
`ontology.ttl` gains `ngm:` where it had `linked:`, which is the PRD's "one clean
namespace, no compatibility shims".

**Known divergence (WS-C).** `api/search-index.json`'s `labels` is now a
superset: Python carried only the fence's `preferred-term`, the migrated corpus
carries every alias.

## C2. `vault` CLI (WS-C implements, WS-G/F/H consume). All subcommands accept `--json`.

```
vault validate  [--vault knowledge|working|all] [--strict]           → exit 0/1, report
vault find      --query <q> [--type T] [--limit N] [--fuzzy]          → [{id,title,type,score}]
vault retrieve  <id>… [--expand is-a=2,requires=1,…] [--max-documents N] → {seeds:[…],expanded:[…]}
vault tree      <id> [--depth N]                                       → nested {id,children}
vault edit      <id> --set k=v… [--unset k]… --expect docs=N,blocks=M  → refused without --expect
vault create    <staged-page> --expect docs=1[,blocks=M] [--set k=v…] [--unset k]… [--dry-run]
                                                                       → writes knowledge/pages/<title>.md only if absent,
                                                                         --set keys in the same write; exit 2 + nothing written on
                                                                         exists / invalid / collision / undeclared --expect
vault propose   <iri> --level content|schema [--hypothesis "…"] [--diff <file>] [--dry-run]
                                                                       → PatchProposal JSON (C4); posts 31402 unless --dry-run
vault gate      [--tier quick|full]                                    → exit 0/1
vault conflicts [--severity high|all]                                  → report, exit 1 on high
vault build     --out <dir> [--vault knowledge] [--stats] [--with-rvdb] → bundle (C3) + quartz static/
                --with-rvdb is opt-in: the rest of the build needs no embedder
vault migrate   --fences-to-properties [--dry-run] [--report <file>]   → one-shot; deleted after use
                exit 2 + nothing written when the `migration:` map does not cover a key
```
Identity of a page = vault-relative path without `.md` (`knowledge/pages/Knowledge Graph` ⇒ id `Knowledge Graph`).

## C3. Build bundle (WS-C writes, WS-H/E consume) — same artefact names Loom already loads

```
<out>/data/ontology.ttl              asserted
<out>/data/ontology-inferred.ttl     Whelk EL++ closure
<out>/data/scaffold-index.json       v1 shape (Loom loom-scaffold/index.rs RawIndex) — byte-parity golden vs current build
<out>/data/prose-index.json
<out>/data/ontology-corpus.rvdb (+ .generation.json sidecar: embedding_model=bge-small-en-v1.5, dimensions=384)
<out>/api/search-index.json, <out>/api/pages/<slug>.json, <out>/api/census.json, <out>/api/validation-report.json
<out>/ns/v2.jsonld                   JSON-LD context (served path is /ns/v2.jsonld; loom-graph-oxigraph cites narrativegoldmine.com/ns/v1#)
<out>/api/schema/context.jsonld      the same document, byte-identical — the old site's pinned URL (publish.yml requires it); also context/v1.jsonld
<out>/data/ontology.json             WebVOWL graph (explorer build input)
<out>/data/graph/{overview.json,domain-*.bin,full.bin,stats.json,bridges.json}   NGG1 graph tiers (explorer physics worker) — port emit_graph_tiers.py
<out>/api/markdown/                   NOT published by default (no consumer; 124 MB); `vault build --with-markdown-mirror` emits it (decided 2026-09-22, Pages 1 GB limit)
<out>/publish/                        Q16: staged Quartz content = knowledge/ ∪ {working/ pages with public: true}; Quartz builds from THIS dir, never from ../knowledge directly. OWL/indexes/scaffold are built from knowledge/ ontology types only.
                                      `--publish-out <dir>` writes the same tree to <dir> instead. Layout = the published site's URL contract
                                      (quartz/scripts/stage-content.sh, quartz.config.ts, /page/<slug> stubs, check-urls.sh):
  publish/pages/<title>.md            public knowledge pages → Quartz slug `pages/<title>`, URLs unchanged from the old site
  publish/working/<subdir>/<title>.md public working pages, subdirectories KEPT (e.g. working/podcast-evidence/<title>)
  publish/index.md                    the generated OKF §8 home page (Quartz slug `index`; carries `public: true` so
                                      ExplicitPublish keeps it)
                                      Held back at any depth whatever the flag: `_misc/`, `misc/` (matches `**/misc/**`
                                      and stage-content.sh, so check-urls' staged == built completeness holds)
<out>/okf/                            OKF bundle export (index.md + concept files) for external exchange
<out>/.generation.json:   { id: "visionGraph@<sha>", commit: <sha>, content_digest, generated_at,
                            class_count, page_count, vocabulary_version, stale_after, artifacts:[{name,sha256,bytes}] }
```

`content_digest` identifies the **source corpus**: SHA-256 over the sorted `(page path, page bytes)` pairs the build read
(`vault::build::generation::content_digest`). It is NOT a digest of the artefacts and cannot be recomputed from them. A
consumer that needs to prove what it serves hashes the listed artefacts itself — Loom's served `identity.content_digest`
is that artefact-set digest (`name:sha256` pairs, sorted, joined by newlines, SHA-256). Artefact `name`s are relative paths
inside the bundle and may include a subdirectory (`graph/full.bin`); `..`, absolute paths and symlinks are refused.
(Clarified 2026-09-23: the two sides had read the undefined field differently, and the reload script refused every real
vault-build bundle.)

## C4. `PatchProposal` JSON (WS-C emits, WS-F consumes)

```json
{ "level": "content|schema", "kind": "amend|create", "iri": "urn:ngm:class:…", "page": "Knowledge Graph",
  "hypothesis": "…", "diff": "<unified diff of frontmatter>", "digest": "sha256:…",
  "blockers": [], "proposer": "process:vault/1.0", "generation": "visionGraph@<sha>",
  "stale_after": "<ISO8601 = now+14d>" }
```
`blockers` non-empty ⇒ the proposal is NOT posted (Whelk inconsistency, SUBCLASS_CYCLE, RELATION_CONTRADICTION, vocabulary violation).

`kind: "create"` — the subject names no existing page and the `--diff` file declares a `title` no page has (in a manifest dir:
an absent id whose file declares `title: <id>`). The diff is against `/dev/null` (`--- /dev/null` / `+++ b/<title>`); `iri`
is the staged page's `resource`. Extra blockers: validation errors on the staged page (assessed inside the real corpus),
IRI_COLLISION, SLUG_COLLISION (slug taken, or the build would re-key an existing slug), FILENAME_COLLISION (case-folded
title), FILENAME_INVALID; conflict/Whelk blockers are deltas with the page added. Level is `content` unless the page declares
a schema-level key (undeclared key or provisional relation) ⇒ `schema`. Digest: an `amend` digest is byte-identical to the
pre-`kind` digest; a `create` digest appends `kind:create`. Apply side on Promote: `vault create <staged-page> --expect docs=1
--set status=stable --set 'verified+={by:human:<npub>,at:<ts>}' --json` (not `vault edit`, which requires an existing page).

## C5. Forum events (WS-F)

- Panel 31400 `ontology-governance`; operator-declared `tp-verifiability`, `tp-reversibility`, `tp-stakes`; Schema ⇒ stakes Critical (tier High floor).
- 31402 ActionRequest: tags `["context_url", <iri>]`, `["d", <digest>]`, `["level", content|schema|demotion]`; content = PatchProposal JSON.
- 31403 ActionResponse (human only): outcome `Promote{iri}` | `Demote{iri}` | `Reject`; rationale ≥ 20 chars.
- Apply path: relay → agentbox `handleGovernanceDecision` → `vault edit <page> --set status=stable --set 'verified+={by:human:<npub>,at:<ts>}' --expect docs=1,blocks=1` (Promote) / `--set status=deprecated` (Demote) → Loom `AttestationLedger.attest({case_id, digest, outcome})`.
- Expiry: proposal `stale_after` passed ⇒ case `expired` receipt, PatchProposal discarded, page untouched, re-surfaced by escalated-on-age cron.

## C6. Baselines captured 2026-09-22 (acceptance §5 measures against these)

VisionClaw graph 13,165 nodes / 153,960 edges; VisionClaw ontology classes 4,167; Loom generation 2026-08-22 8,146 classes; raw vault 8,433 classes / 8,444 files; knowledge/ 8,977 files 138 MB; working/ 2,254 files 966 MB; json-ld fences 8,454 pages; `key::` lines 23,900; `{{embed}}` 37.
