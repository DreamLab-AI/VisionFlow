# Logseq residue — final classification (acceptance criterion 7)

Date: 2026-09-22 · Workstream: ws-residue · Spec: `VisionFlow/docs/PRD-sovereign-corpus.md` (AC-7)

Enumeration: `grep -rIil logseq` over loom, project (VisionClaw + agentbox), VisionFlow and visionGraph,
excluding `.git node_modules target archive dist .quartz-cache public`. In scope for action: everything
except corpus content (`visionGraph/{knowledge,working}/**`, and the generated `quartz/content` and
`quartz/static` projections of it), ADR files, `archive/`, history, `estate-review/`, `estate-closeout/`,
`migration-2026-09-22/`, census docs and CHANGELOGs, which are HISTORY by rule.

Classes: **HISTORY** (past tense, provenance, deliberate rejection, the genuine Logseq
`personal-context-portfolio` graph) · **SHIM** (read-side acceptance of a legacy value) · **STALE**
(docs/comments describing Logseq as current — fixed or marked `<!-- STALE 2026-09-22 -->`) ·
**LIVE** (production code still reading/writing Logseq syntax/paths — fixed with tests).

## Summary (in-scope files)

| repo | before | after (all remaining are HISTORY or SHIM) | LIVE fixed | STALE fixed | SHIM kept | HISTORY left |
|---|---|---|---|---|---|---|
| loom | 16 | 13 | 0 | 5 | 0 | 11 |
| VisionClaw (project, excl. agentbox) | 63 | 54 | 1 (+1 companion script) | 26 | 10 | 26 |
| agentbox | 22 | 18 | 2 | 11 | 2 | 9 |
| VisionFlow (non-archive) | 30 | 30 | 0 | 12 (dated markers / past tense) | 0 | 18 |
| visionGraph (excl. corpus content) | 70 (35 hand-written + 35 generated `quartz/static`) | 28 hand-written (+ generated `quartz/static`, all HISTORY) | 2 | 10 | 1 | 62 rows (incl. generated) |

"After" counts still include fixed files that now name Logseq only as provenance (past tense).

## LIVE fixes

1. `project/.github/workflows/ontology-publish.yml`: CI was running the deleted Python `pipeline.build`. It now runs `vault build`, and `scripts/ontology/pack-pod-resources.py` reads `context/v1.jsonld`. The pack ran locally: 331,045 triples and 8,434 classes. The workflow itself has not been run on GitHub yet.
2. `project/agentbox/services/ontology-tools` `modify`/`enrich`: on a page with frontmatter they wrote a Logseq-format `### OntologyBlock` (`key::` lines). They now refuse such pages and point to `vault edit`. 3 new tests; cargo test 37/37 passes.
3. `project/agentbox/schema/agentbox.toml.schema.json`: `[vault].format = "logseq-legacy"` was accepted. It is now rejected. 2 new jest tests.
4. `visionGraph/publishing-tools/WasmVOWL/modern` `MarkdownRenderer.tsx` + `pageService.ts`: rewrote Logseq syntax and fell back to the Logseq-format markdown mirror. Both now use the new shared `src/lib/vaultMarkdown.ts` (frontmatter, Obsidian wikilinks, embeds). 7 new tests; vitest 54/54 passes.

## Open items for the operator

- `project/config.yml` `tunnel: logseqXR`: the tunnel name is set in Cloudflare, so it has to be renamed there.
- `visionGraph/publishing-tools/notes-mobile.css`: dead code (header marks it unused). Deleting it needs `git rm`.
- `visionGraph/transcripts/*`: 240 files still carry `ingest-status::`. `weekly_ingest.py` still reads them through a SHIM. The data was left as it is.

## loom
| file | class | action |
|---|---|---|
| loom/README.md | STALE | Rewrote maintainer link, repo table row, TBox tagline, ASCII + mermaid corpus node to the visionGraph vault / `vault build` (ADR-141) |
| loom/docs/README.md | STALE | Cross-link list: builder is `vault build` (replaced jjohare/logseq's publish.yml) |
| loom/docs/design/ONTOLOGY-LOOM-PIPELINE.md | STALE→HISTORY | Scope line + §1 now name `vault build`; 2026-08-17 retargeting note put in past tense; remaining mentions are "formerly/previously" provenance |
| loom/docs/design/RUST-ARCHITECTURE.md | STALE | Drop table, cross-links and non-goals name `vault build` / visionGraph as source of truth |
| loom/docs/design/ddd-ontology-loom-context.md | STALE | Honesty table, glossary, context map (mermaid node + edge), §6.x conformist relationship retargeted to the vault build; dropped retired `logseq-publisher-rust` from style referents |
| loom/docs/design/PRD-025-ontology-loom-and-connector-platform.md | HISTORY | Dated plan (2026-08-11); added dated corpus-builder note after H1, body untouched |
| loom/docs/design/PRD-026-loom-consolidation.md | HISTORY | Dated plan; same corpus-builder note |
| loom/docs/design/PRD-027-rust-loom-reengineering.md | HISTORY | Dated plan; same corpus-builder note |
| loom/docs/design/DOC-REENGINEERING-PLAN.md | HISTORY | Dated plan; same corpus-builder note |
| loom/docs/design/ADR-135/137/140/141 | HISTORY | ADR files — untouched by rule |
| loom/docs/research/gain-over-copy-paper/gain-over-copy-paper.tex, assets/fig-ecosystem.tex | HISTORY | Published paper (v9.2 tag) describing the corpus as measured (8,138 Logseq pages); not rewritten |
| loom/legacy/ONTOLOGY-UPLIFT-PLAN.md | HISTORY | Already bannered "Historical — frozen" |
| loom/app/data/{ontology.ttl,ontology-inferred.ttl,scaffold-index.json,prose-index.json} | HISTORY (generated content) | Untracked generation artefacts (generated 2026-08-15; formerly mirrored by `app/mirror.sh`, now emitted by `vault build`). Hits are corpus pages about Logseq the software (e.g. `distributed-logseq-knowledge-network`) — subject matter |

## VisionClaw (project, excluding agentbox)
| file | class | action |
|---|---|---|
| project/.github/workflows/ontology-publish.yml | LIVE | CI still ran the deleted Python `pipeline.build` against the vault. Now builds `vault` (`cargo build --release --locked -p vault`) and runs `vault --repo vault-source build --vault knowledge`. Checkout path `logseq-source`→`vault-source`; dispatch type `logseq-sync`→`corpus-sync` (no sender exists). YAML parses. Comment kept as history |
| project/scripts/ontology/pack-pod-resources.py | LIVE (companion) | Read the old pipeline's `api/schema/context.jsonld`; now reads `vault build`'s `context/v1.jsonld`. Verified against a real vault build bundle: 331,045 triples, 8,434 classes, packed |
| project/data/schema/settings_db.sql | STALE | Seed physics profile `logseq`→`knowledge` (the file is not loaded by any code) |
| project/client/src/features/graph/managers/graphDataManager.ts | SHIM (send side) | `'logseq'` removed from the `setGraphType` union (no callers pass it); now typed `GraphType` |
| project/client/src/features/graph/workers/graph.worker.ts | SHIM (send side) | `'logseq'` removed from the `setGraphType` union |
| project/client/src/features/graph/types/graphTypes.ts | SHIM | Deleted the dead `LEGACY_KNOWLEDGE_GRAPH_TYPE` and `normaliseGraphType` (no production callers); comment now cites ADR-2115 |
| project/client/src/store/settings/settingsHelpers.ts | SHIM (receive) | Migration kept; comment ADR-2041→ADR-2115 |
| project/client/src/store/settingsStore.ts | SHIM (receive) | Kept; comment →ADR-2115 |
| project/client/src/store/settings/physicsSlice.ts | SHIM (receive) | Kept; comments →ADR-2115 |
| project/client/src/api/settings/schemaMappings.ts | SHIM (receive) | `toVisualKey` mapping kept; comments →ADR-2115 |
| project/client/src/features/settings/__tests__/settingsMigration.test.ts | SHIM (test) | Kept the migration tests; dropped the `normaliseGraphType` case; header →ADR-2115 |
| project/client/src/features/graph/managers/__tests__/graphDataManager.test.ts | SHIM (test) | Removed the send-side `setGraphType('logseq')` test |
| project/client/src/store/websocket/__tests__/binaryProtocolAgentAction.test.ts | STALE | Mock `getGraphType` returns `knowledge`; `GraphTypeFlag` mock matches the real enum (`KNOWLEDGE_GRAPH`/`ONTOLOGY`) |
| project/client/src/features/control-center/registry/__tests__/registry.test.ts | SHIM (test) | Normalisation of the frozen fixture kept; comment →ADR-2115 |
| project/client/src/features/control-center/registry/__tests__/legacy-paths.fixture.json | HISTORY | Byte-frozen WP5 record; left as is |
| project/crates/visionclaw-domain/src/types/ontology_tools.rs | STALE | Doc comment "Full Logseq markdown"→"page markdown (YAML frontmatter + body)" |
| project/crates/visionclaw-ontology/src/types/ontology_tools.rs | STALE | Same fix |
| project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs | STALE | Comment now names `vault build` `turtle.rs` as the emitter, with the logseq converter noted as retired |
| project/crates/visionclaw-actors/src/messages/ontology_messages.rs | STALE | Dropped "(Logseq-based)" from the section banner |
| project/crates/visionclaw-contracts/src/github_adapter.rs, bindings/ParsedMarkdown.ts | HISTORY | Past-tense provenance |
| project/sdk/visionflow-contracts/bindings/ParsedMarkdown.ts | STALE | The generated copy still claimed ADR-2040 tolerance; re-synced from `crates/visionclaw-contracts/bindings` |
| project/crates/visionclaw-domain/src/vault/mod.rs, config/graph_type.rs, config/visualisation.rs | HISTORY | Past tense and rejection tests |
| project/crates/vault/src/{migrate,validate,main}.rs, tests/golden_parity.rs, tests/golden/fixture/ontology/vocabulary.yaml | HISTORY | Migration tool, `LOGSEQ_PROPERTY` rejection, residue assertions, test data |
| project/crates/vault/src/repair.rs | STALE→HISTORY | "the corpus is a Logseq export" rewritten to past tense |
| project/crates/vault/README.md | STALE→HISTORY | Same past-tense fix at the code-detection paragraph; the rest describes the migrator and rejection |
| project/crates/vault-core/src/{vocabulary,code,frontmatter,page}.rs | HISTORY | `logseq_keys` migration map (a schema key shared with visionGraph `vocabulary.yaml`), residue escaping, legacy boolean spellings, journal-date note |
| project/src/services/decision_elevation.rs | STALE | "PR'd to `jjohare/logseq`"→ the corpus repo (`GITHUB_OWNER`/`GITHUB_REPO`, visionGraph); `:961` test assertion is HISTORY |
| project/src/services/ontology_query_service.rs | STALE | "Logseq notes"→ vault pages with OKF frontmatter |
| project/src/services/github_sync_service.rs | STALE | "(logseq source)"→"(the visionGraph vault)" |
| project/src/services/ontology_conflict_gate.rs | STALE | Now cites `crates/vault/src/conflicts.rs` (the port of the retired `logseq/pipeline/conflicts.py`) |
| project/src/handlers/socket_flow_handler/position_updates.rs | STALE | "`public:: true` Logseq tag"→ frontmatter `public: true` |
| project/src/services/page_parser.rs, src/actors/elevation_actor.rs, src/handlers/settings_handler/validation.rs | HISTORY | Rejection tests and ADR-2115 comment |
| project/tests/settings_deserialization_test.rs, tests/vault_gate_test.rs | HISTORY | Assert that `logseq` is rejected and that Logseq-format fixtures are private |
| project/tests/ontology_agent_integration_test.rs | STALE | Header bullet now reads "vault markdown with YAML frontmatter"; `:369` is HISTORY |
| project/tests/fixtures/data-model/README.md | STALE | Format note rewritten: the tolerance ended under ADR-2112 and these fixtures are private |
| project/README.md | STALE | Corpus row: the Logseq tolerance "kept" → closed by ADR-2112, local by default (ADR-2115); `:176` is HISTORY |
| project/config.yml | HISTORY | `tunnel: logseqXR` is an external Cloudflare tunnel identifier; renaming it needs a change on the Cloudflare side (owner) |
| project/docs/explanation/ontology-pipeline.md | STALE | §1–3 rewritten: the deleted `LogseqPage`/`parse_logseq_file`/converter/assembler replaced by `page_parser` → `vault_core` → `project_ontology`; module table and description fixed |
| project/docs/BASELINE-architecture.md | STALE | Data-pipeline paragraph now covers the CorpusSource, frontmatter-only format and `page_parser`; the `:25` changelog is HISTORY |
| project/docs/reference/graph-schema.md | STALE | "still accepted"→ body text since ADR-2112 |
| project/docs/reference/agents-catalog.md | STALE | Removed the two rows for the non-existent `logseq-formatted` skill |
| project/docs/how-to/operations/configuration.md | STALE | Fictional `LOGSEQ-*` env block replaced by `CORPUS_SOURCE`/`VAULT_ROOT`/`VAULT_BASE_PATHS`; YAML example uses `knowledge:` (ADR-2115 rejects `logseq`) |
| project/docs/how-to/operations/power-user-bootstrap.md | STALE | `GITHUB_REPO` example `logseq`→`visionGraph` (the `logseq` repo is archived) |
| project/docs/how-to/agent-orchestration.md | STALE | "accepted for one release"→ rejected since ADR-2115 |
| project/docs/how-to/features/ontology-parser.md | STALE | Superseded banner (module deleted) and the tolerance sentence set to past tense |
| project/docs/tutorials/promote-note-to-ontology.md | STALE | "keep working until"→ window closed under ADR-2112 |
| project/docs/VAULT-corpus-format.md | HISTORY | Rejection list, migrator notes, changelog, EXP-V13 |
| project/docs/TODO-unified.md | HISTORY | Records the archived `jjohare/logseq` repo |
| project/docs/security/PRE-DEMO-SECURITY-AUDIT-2026-08-21.md, docs/engineering/vault-migration-coverage-2026-09-22.md, docs/gap-close-evidence/2026-09-02-obsidian-migration-closeout.md | HISTORY | Dated records; left untouched |

## agentbox
| file | class | action |
|---|---|---|
| project/agentbox/services/ontology-tools/src/{modifier,enrichment,markdown}.rs | LIVE | `modify` / `enrich` spliced a Logseq `- ### OntologyBlock` with `key::` lines into vault pages, placing it after the opening `---` of the frontmatter and so corrupting it. Added `markdown::is_vault_page` plus `VAULT_PAGE_WRITE_REFUSAL`. Both write paths now refuse frontmatter pages before any backup or write, and enrich does so without a git rollback. 3 new tests. |
| project/agentbox/services/ontology-tools/src/block.rs | HISTORY | `id` / `collapsed` documented as legacy Logseq read-tolerance, which is accurate. Left. |
| project/agentbox/schema/agentbox.toml.schema.json | LIVE | Removed `"logseq-legacy"` from the `[vault].format` enum (no consumer ever branched on it; the transition window has closed). Description rewritten; condense-scheduler text now says "vault corpus". |
| project/agentbox/tests/config/semantic-rules.test.js | HISTORY (new) | New tests: `obsidian` is accepted and `logseq-legacy` is rejected at `/vault/format` (these are rejection assertions). |
| project/agentbox/agentbox.toml | STALE→fixed | Condense comment now says "vault corpus"; `format` comment updated (value withdrawn). 1 remaining hit is the "former logseq-legacy … withdrawn" note (HISTORY). |
| project/agentbox/setup/agentbox.default.toml | STALE→fixed | Same as agentbox.toml. The remaining hit is HISTORY. |
| project/agentbox/services/agentbox-manifest/tests/golden/live-agentbox.toml | HISTORY | Frozen golden fixture captured from an earlier manifest. Left. |
| project/agentbox/flake.nix | STALE→fixed | Comment changed from "logseq corpus" to "vault corpus". |
| project/agentbox/scripts/ontology-condense-scheduler.mjs | STALE→fixed | Two header comments changed from logseq to vault (code already reads `VAULT_PAGES`). |
| project/agentbox/.env.example | STALE→fixed | `PROJECT_DIR_4` example changed from `…/logseq` to `…/VisionFlow`. Also deleted the dead `ONTOLOGY_ENRICH_KG_ROOT=mainKnowledgeGraph/pages`, which nothing reads. |
| project/agentbox/services/podcast-ingest/src/bulk/domain_probe.rs | STALE→fixed | Operator hint no longer says "promote to mainKnowledgeGraph/pages/"; it now says "promote into knowledge/pages/ with `vault propose`" (the `mainKnowledgeGraph` hit, not a `logseq` one). |
| project/agentbox/skills/ontology-enrich/SKILL.md | STALE→fixed | §1 rewritten: frontmatter `domain:`, `vault validate`, `vault edit … --expect docs=1`, and `vault propose` for bulk normalisation. It no longer uses `source-domain::` grep or `ontology-tools modify`. |
| project/agentbox/skills/ontology-core/SKILL.md | STALE→fixed | Quick path now sends edits to `vault edit`. `ontology-tools` is scoped to the retired OntologyBlock format and refuses vault pages. |
| project/agentbox/skills/playwright/references/legacy-local-scripts/debug-vf.js | STALE→fixed | Reads `visualisation.graphs.knowledge` instead of `.logseq` (the server rejects `logseq` since ADR-2115). |
| project/agentbox/docs/BASELINE-container.md | STALE→fixed (1) / HISTORY (3) | Fixed: the `format` line now says obsidian only and the value was withdrawn on 2026-09-22. Left: the three references to the check-no-logseq-paths guard. |
| project/agentbox/docs/user/configuration.md | STALE→fixed | Comment changed from "Logseq OWL2 DL tools" to "vault OWL2 DL ontology skills". |
| project/agentbox/docs/user/quickstart.md | STALE→fixed | Skill description now refers to the vault corpus and frontmatter conventions. |
| project/agentbox/services/agentbox-mcp/src/web_summary/{mod,types}.rs | SHIM | `logseq` is accepted as an input synonym of `obsidian`. Kept, with its test. |
| project/agentbox/skills/web-summary/SKILL.md | SHIM | Documents `logseq` as a retired alias. Left. |
| project/agentbox/scripts/ci/check-no-logseq-paths.sh | HISTORY | Rejection guard (ADR-2028). Left; it passes. |
| project/agentbox/.github/workflows/invariants.yml | HISTORY | Runs the guard. Left. |
| project/agentbox/docs/GOVERNANCE-capabilities.md | HISTORY | Cites the guard. Left. |
| project/agentbox/skills/system-one/references/data-boundary.md | HISTORY | The personal-context-portfolio graph really is Logseq. Left. |
| project/agentbox/skills/podcast-knowledge-ingest/SKILL.md | HISTORY | "No writer emits `key::` Logseq property lines" is a rejection statement. Left. |
| project/agentbox/tests/config/vault-path-precedence.test.sh | HISTORY | Test name refers to a "stale Logseq-era override". Left. |
| project/agentbox/config/entrypoint-unified.sh | HISTORY | "stale Logseq-era tree" is past tense and describes the guard. Left. |
| project/agentbox/services/ontology-tools/src/markdown.rs | HISTORY (new) | New doc comment dates the migration off Logseq to 2026-09-22. |

## VisionFlow + visionGraph
| file | class | action |
|---|---|---|
| `VisionFlow/docs/PRD-sovereign-corpus.md` | HISTORY | the migration spec itself; leave |
| `VisionFlow/docs/BASELINE-visionflow.md` | HISTORY | past-tense record of the pre-migration state; leave |
| `VisionFlow/docs/architecture/adr-reference-resolutions.json` | HISTORY | record; leave |
| `VisionFlow/docs/architecture/repository-map.md` | HISTORY | lists `../project4` as archived Logseq history; leave |
| `VisionFlow/docs/diagrams/visionclaw/22-data-authority-provenance-erasure.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visionclaw/28-external-services.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visionflow/07-pitch-presentation-reports.md` | HISTORY | past-tense/provenance (archived Logseq repo, roster name, resolved drift); leave |
| `VisionFlow/docs/diagrams/visionclaw/20-ontology-pipeline-oxigraph-whelk.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visionclaw/21-corpus-ingest-and-vault.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visionclaw/09-config-and-env-flags.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visionclaw/34-client-features.md` | HISTORY | records the ADR-2041 key rename; leave |
| `VisionFlow/docs/diagrams/agentbox/25-ontology-tools-and-governed-writes.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/agentbox/01-nix-flake-composition.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/agentbox/22-skills-and-routing.md` | HISTORY | cites the check-no-logseq-paths.sh guard (a rejection); leave |
| `VisionFlow/docs/diagrams/hero/src/batch-generate-phase2.sh` | HISTORY | Logseq named as a prior-art tool (subject matter); leave |
| `VisionFlow/docs/diagrams/knowledgegraph/03-ontology-architecture.md` | HISTORY | describes the frozen knowledgeGraph repository (Logseq-format snapshot) / external precedent; accurate for that repo |
| `VisionFlow/docs/diagrams/visiongraph/06-shipping-explorer-duplication.md` | HISTORY | past-tense/provenance (archived Logseq repo, roster name, resolved drift); leave |
| `VisionFlow/docs/closeout/unified-findings-register.json` | HISTORY | findings record; leave |
| `VisionFlow/docs/diagrams/hero/src/10-prior-art-quadrant.mmd` | HISTORY | Logseq named as a prior-art tool (subject matter); leave |
| `VisionFlow/docs/diagrams/visionclaw/06-settings-round-trip.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/COVERAGE.md` | HISTORY | generated index (scripts/diagram-index-gen.cjs) of heading titles; follows the topic files on regeneration |
| `VisionFlow/docs/diagrams/estate/09-build-deploy-and-ci-estate.md` | STALE | added dated `<!-- STALE 2026-09-22 -->` marker(s) naming the replacement (vault crate / ADR-2115 / `PRIVATE_REPO_GITHUB_PAT` / `obsidian`-only schema); not redrawn |
| `VisionFlow/docs/diagrams/visiongraph/04-downstream-contracts.md` | HISTORY | past-tense/provenance (archived Logseq repo, roster name, resolved drift); leave |
| `VisionFlow/docs/diagrams/estate/90-evidence-and-supporting-repositories.md` | HISTORY | past-tense/provenance (archived Logseq repo, roster name, resolved drift); leave |
| `VisionFlow/docs/diagrams/knowledgegraph/01-repo-composition-and-licensing.md` | HISTORY | describes the frozen knowledgeGraph repository (Logseq-format snapshot) / external precedent; accurate for that repo |
| `VisionFlow/docs/diagrams/knowledgegraph/06-consumers-and-integration.md` | HISTORY | describes the frozen knowledgeGraph repository (Logseq-format snapshot) / external precedent; accurate for that repo |
| `VisionFlow/docs/engineering/sovereign-corpus-contracts.md` | HISTORY | `logseq_keys` migration map consumed only by the one-shot `vault migrate`; leave |
| `VisionFlow/presentation/google-analysis.md` | STALE | "markdown / Logseq" → "markdown (an Obsidian vault since 2026-09; formerly Logseq)" |
| `VisionFlow/scripts/generate-release-manifest.sh` | STALE | roster role for `logseq` rewritten: archived Logseq-era publisher lineage, read-only; header comment (roster name) left |
| `VisionFlow/presentation/report/chapters/13e-ontology-binding.tex` | STALE | lineage sentence to past tense; now maintained as a frontmatter-only Obsidian vault |
| `visionGraph/CLAUDE.md` | HISTORY | rejection rule ("never emit Logseq key:: lines"); leave |
| `visionGraph/README.md` | STALE | "/notes is frozen" section rewritten: retired, `/notes/` redirects to `/` (LegacyRedirects); other hits are rejection rules / split provenance |
| `visionGraph/transcripts/weekly_ingest.py` | SHIM | reader tolerance for the `ingest-status::` marker in transcript files already on disk (240 still carry it); writer emits frontmatter only; keep |
| `visionGraph/transcripts/test_weekly_ingest.py` | HISTORY | test asserts no Logseq property lines are emitted (rejection); leave |
| `visionGraph/docs/source-repair-plan-2026-09-22.md` | HISTORY | dated repair plan; leave |
| `visionGraph/docs/PUBLICATION-contract.md` | HISTORY | archived Logseq history as pre-split citation target; leave |
| `visionGraph/publishing-tools/WasmVOWL/V2-INTEGRATION-GAPS.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/monitor-deployment.mjs` | STALE | dev script wrote screenshots to deleted `/home/devuser/workspace/logseq/docs/`; now `screenshots/` (gitignored) |
| `visionGraph/publishing-tools/notes-mobile.css` | STALE | dead: only consumer pipeline/patch_notes_export.py is deleted; header marked UNUSED (deletion blocked by permission — recommend `git rm`) |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/site/SiteChrome.tsx` | STALE | user-visible copy: source is the `jjohare/visionGraph` Obsidian vault; "Research notes" links to `/` (Quartz) instead of the retired `/notes/` Logseq SPA |
| `visionGraph/publishing-tools/WasmVOWL/README.md` | STALE | Data Pipeline section rewritten: `vault build --out quartz/static` emits data/ontology.json (crates/vault/src/build/webvowl.rs) from knowledge/pages frontmatter |
| `visionGraph/publishing-tools/WasmVOWL/docs/optimization/legacy-feature-comparison.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/docs/UI-MIGRATION-COMPLETE.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/tests/TEST-RESULTS-3D-INTEGRATION.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/tests/MANUAL-3D-INTEGRATION-TEST-REPORT.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/DEPLOYMENT-READY.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/verify-wasm-deployment.mjs` | STALE | dev script wrote screenshots to deleted `/home/devuser/workspace/logseq/docs/`; now `screenshots/` (gitignored) |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/components/PageRenderer/MarkdownRenderer.tsx` | LIVE | replaced `preprocessLogseqMarkdown` (key::/collapsed::/id::/block-ref rewriting) with `normaliseVaultMarkdown` (frontmatter strip + Obsidian wikilinks incl. alias/heading/embed); tests in src/lib/__tests__/vaultMarkdown.test.ts |
| `visionGraph/publishing-tools/WasmVOWL/MERGE_IMPACT_ANALYSIS.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/docs/phase3-task3.2-completion-report.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/tests/performance-reports/phase3-performance-validation-report.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/ontology/vocabulary.yaml` | HISTORY | `logseq_keys` migration map + drop-list of Logseq UI keys (consumed only by `vault migrate`); leave |
| `visionGraph/publishing-tools/WasmVOWL/PHASE_11_INTEGRATION.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/pages/AboutPage.tsx` | STALE | user-visible copy: source is the `jjohare/visionGraph` Obsidian vault; "Research notes" links to `/` (Quartz) instead of the retired `/notes/` Logseq SPA |
| `visionGraph/publishing-tools/WasmVOWL/CHANGES.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/docs/IMPLEMENTATION-SUMMARY.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/site/useStats.ts` | STALE | user-visible copy: source is the `jjohare/visionGraph` Obsidian vault; "Research notes" links to `/` (Quartz) instead of the retired `/notes/` Logseq SPA |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts` | LIVE | replaced `cleanLogseqMarkdown` with `cleanVaultMarkdown` (shared src/lib/vaultMarkdown.ts); removed fallback fetch to the Logseq-era knowledgeGraph gh-pages markdown mirror |
| `visionGraph/publishing-tools/WasmVOWL/modern/inspect-graph.js` | STALE | dev script wrote screenshots to deleted `/home/devuser/workspace/logseq/docs/`; now `screenshots/` (gitignored) |
| `visionGraph/publishing-tools/WasmVOWL/modern/tests/3D-INTEGRATION-SUMMARY.md` | HISTORY | dated phase/report/analysis document; leave |
| `visionGraph/publishing-tools/WasmVOWL/modern/test-local-build.mjs` | STALE | dev script wrote screenshots to deleted `/home/devuser/workspace/logseq/docs/`; now `screenshots/` (gitignored) |
| `visionGraph/quartz/README.md` | HISTORY | redirect table: `/notes/` frozen Logseq SPA → `/`; leave |
| `visionGraph/quartz/static/api/pages/log-seq-spring-thing.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graph-style-guide.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graph-presentation-session-artefact.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/markdown-diagramming-as-code-tool.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/pyodide-rag-corpus-builder-script.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-distillation.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/metaverse-ontology.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/graph-neural-network.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graphing.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/dr-o-hare-writing-for-log-seq.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/scripts/check-urls.sh` | HISTORY | checks the retired-SPA redirect; leave |
| `visionGraph/quartz/static/api/pages/language-modeling.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/quartz/plugins/emitters/legacyRedirects.ts` | HISTORY | redirect stub for the retired Logseq SPA; leave |
| `visionGraph/quartz/static/api/pages/python-sample2.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/applied-ai-research-portfolio.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/pyodide-knowledge-graph-node-enumerator.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/_domain-index.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/practitioner-workflow-optimisation-heuristics.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/visioning-lab-property-crosswalk.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/local-rag-corpus-ingestion-pipeline.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/diagrams-as-code.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/digital-asset-risks.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/domain-expert-contact-index.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/structurizr-dsl.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/search-index.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graph-diagnostic-node.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/python-sample1.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/matplotlib-inline-visualisation-pattern.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graph-kanban-board.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/knowledge-graph-publication-classifier.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/datalog-knowledge-graph-query-language.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/api/pages/distributed-logseq-knowledge-network.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/scripts/check-residue.sh` | HISTORY | residue guard counting Logseq syntax (a rejection); leave |
| `visionGraph/quartz/static/data/ontology-inferred.ttl` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/data/scaffold-index.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/data/ontology.ttl` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/data/prose-index.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/quartz/static/data/ontology.json` | HISTORY | generated content: `vault build --out quartz/static` (publish.yml) from corpus pages that discuss Logseq the software; no edit |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/lib/vaultMarkdown.ts` | HISTORY | new module; doc comment records that Logseq syntax is retired and not interpreted |
| `visionGraph/publishing-tools/WasmVOWL/modern/src/lib/__tests__/vaultMarkdown.test.ts` | HISTORY | new test; asserts retired Logseq syntax is NOT reinterpreted |
| `visionGraph/quartz/static/okf/**` (23 files) + `quartz/static/.generation.json` | HISTORY | generated content from a `vault build --out quartz/static` run during this session (corpus pages about Logseq the software); no edit |
