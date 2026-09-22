# PRD: Sovereign Corpus — one vault, one build, one gate

**Owner:** DreamLab AI (John O'Hare)
**Status:** Accepted 2026-09-22 (owner, Q1–Q14)
**Date:** 2026-09-22
**Version:** 1.0
**Companion records:** VisionFlow ADR-2013 (this PRD as canon), VisionClaw ADR-2112..2116, agentbox ADR-2107..2109, forum ADR-2013, Loom ADR-141 — all created by the one-shot, numbered here so nothing collides.
**Evidence basis:** five research reports stored in RuVector `project-state` under `redesign-2026-09-22-*`; probes in `redesign-2026-09-22-visionclaw-intersection-probe`; Loom ADR-140 Addendum A; arXiv 2609.15779 (EvoOntology); OKF v0.2; IWE (ideas only).

---

## 1. Why now

The estate has one corpus and **four doors that disagree about it**:

| Door | Reads | Classes | Reasoned | Gated | Stamped |
|---|---|---|---|---|---|
| raw vault on disk | `visionGraph/knowledge/pages` | 8,433 | no | no | no |
| agentbox `ontology-bridge` (Node MCP) | the same raw disk, silently, after its VisionClaw URL died | 8,433 | no | no | no |
| Loom served generation | a bundle built 2026-08-22 | 8,146 | yes | yes | yes |
| VisionClaw live ontology | its own parse of the json-ld fences, pulled from GitHub on boot | 4,167 | yes | `public` only | no |

Nobody can say which number is right. The vault spec lists keys (`owl-class`, `source-domain`, `maturity`, `quality`) that occur **zero** times in frontmatter because the ontology actually lives in two `json-ld` fences per page; 23,900 Logseq `key:: value` lines and 37 `{{embed}}`s survive against the vault's own rule; a second, Logseq-format corpus pipeline (`DreamLab-AI/knowledgeGraph`) still claims narrativegoldmine.com; and the flagship README, website and pitch deck describe "a Logseq corpus". The published paper (*The Copy Ceiling*) and the external evidence (EvoOntology's −15.0 for static injection vs +20.0 for tool exposure; OKF's attestation model; IWE's blast-radius guards) all point the same way: **one governed corpus, machine-checkable, agent-drivable, human-signed where it matters.**

Owner's governing principle for this work: *"we have git tracking and should carefully fully migrate to the cleanest outcome."* No compatibility shims, no one-more-release tolerances. Git is the rollback.

## 2. Decisions (owner, 2026-09-22)

| # | Decision | Chosen |
|---|---|---|
| Q1 | Canonical vault | `/home/devuser/workspace/visionGraph` (`knowledge/` + `working/`). Delete `workspace/vault`, `vault-working`; remove the `logseq` symlink and its host bind. |
| Q2 | VisionClaw ingest | Mount the `multi-agent-docker_workspace` named volume read-only; extract a `CorpusSource` trait (`GitHub`, `LocalDirectory`); the full `GitHubSyncService` pipeline runs unchanged over `LocalDirectory`. GitHub pull removed; `GitHubConfig` optional. |
| Q3 | Pipelines | Retire `knowledgeGraph`'s Logseq corpus + pipeline (archive with marker; the repo keeps its published-export role). Rewrite the four "Logseq corpus" passages. Delete `logseq-publisher{,-rust,-npm}`. |
| Q4 | OKF placement | **Frontmatter only.** The two json-ld fences fold into typed Obsidian Properties on all 8,454 pages; the JSON-LD context becomes a build output. |
| Q5 | Relations | Flat predicate keys with wikilink lists (`type`, `resource`, `is-a`, `requires`, `enables`, `has-part`, `part-of`, …) governed by a versioned `ontology/vocabulary.yaml`; unknown keys fail the build. |
| Q6 | Human gate | Content + Schema + **Demotion** require a human-signed forum 31403. Exposure-level (Loom manifest/salience/budget/ranking) self-evolves under a paired-eval gate with auto-revert. Whelk inconsistency and `conflicts` cycles/contradictions are automatic **blockers**, never approvals. |
| Q7 | Signers | Single human signer; Schema floored at tier High. Every PatchProposal carries `stale_after` (14 d); on expiry the *proposal* reverts to draft and is re-surfaced. Quorum deferred. |
| Q8 | Loom identity/build | Generation = `visionGraph@<local sha>` + content digest. On-demand complete-bundle build → verified promotion; enable the supplied 5-minute timer after the first clean promotion. |
| Q9 | Working vault | OKF-conformant with its own types (`Note`, `Episode`, `Transcript`, `Draft Concept`, `Journal`); episodic keys become documented extensions; `Draft Concept` + `status: draft` is what a proposal is generated from. |
| Q10 | Agent access | **Rust CLI-first, no MCP inside the estate.** Humans use Obsidian desktop. Agents use `vault` (Bash) + `loom-client`. `ontology-bridge` deleted; Loom keeps `/mcp` only as the external-host door; `loom-mcp-stdio` dropped. IWE is a source of ideas, never a dependency. |
| Q11 | Crate home | `crates/vault` in the VisionClaw workspace, sharing the `OntologyBlock` parser and Whelk-rs so the corpus is parsed and reasoned by one implementation. `vault build\|gate\|conflicts` replace the Python pipeline and `vault-migrate`. Baked into agentbox via Nix. |
| Q12 | Obsidian tooling | Core-only: Bases + `types.json` + Templater, Canvas, obsidian-git. Committed `.obsidian/` config and a `bases/` folder validated by the build. |
| Q16 | Publish rule | **`public: true` publishes from EITHER vault.** `vault build` stages `knowledge/` plus the public subset of `working/`; Quartz publishes both. Governed ontology bundle = `type: Class\|Property\|Individual` in `knowledge/` only. Podcast evidence = typed `Episode` pages in `working/podcast-evidence/`, **never published (owner, 2026-09-22 14:00: `public: false` on all 188, pipeline defaults new ledgers private; kept in the record, no history rewrite)**; the 188 move there; `_misc` stubs + the About-Me page move to `working/` as `Note`s. Unterminated code fences (454 pages) are a SOURCE defect repaired mechanically (WS-K), never parser leniency. **Coordinator call 2026-09-22 12:30: all 496 non-class files leave `knowledge/`** — 188 evidence, 1 root page, 296 Logseq-form journals (→ `working/journals/`, type Journal), and the 11 `_misc` pages deleted-on-disk-but-unstaged pre-swarm (restored, migrated, relocated to `working/pages/_misc/`; owner to confirm their deletion). Evidence: four independent routes reached the same conclusion (typing pass, residue triage, journal bodies, the crate inferring `Note`). Lesson recorded in the census: a migration map is total only *against a stated extent* — `git ls-tree HEAD` vs `find` disagreeing is a finding. |
| Q17 | Asset publication | **Owner, 2026-09-22 15:00: `.txt` files under assets are never published.** Implemented as an asset-type allowlist (images/video/audio/pdf/fonts) in the publisher, mirrored as a hard rule in the artefact secret gate. |
| Q15 | Site root | Quartz owns `/`; explorer at `/explorer/` with router basename + 404 forwarding; per-page `/page/<slug>` redirect stubs; route redirects; legacy `/notes/` SPA dropped and redirected. |
| Q13 | Publisher | Quartz v4 renders narrativegoldmine.com from `knowledge/` (ExplicitPublish ⇐ `public: true`); `vault build` emits the machine artefacts (`api/*.json`, `ontology.ttl`, OKF bundle, JSON-LD context) into Quartz `static/`. Owner pushes; agents never push. |

## 3. End state

### 3.1 The corpus (`jjohare/visionGraph`) — content only

```
visionGraph/
├── vault.toml                  # vocabulary version, vault roles, build targets, stale defaults
├── ontology/vocabulary.yaml    # key → OWL property, direction, characteristics; versioned; Schema-tier
├── knowledge/                  # the governed OKF bundle (public-gated)
│   ├── .obsidian/              # committed: core plugins, types.json, bases/*.base
│   ├── pages/**.md             # frontmatter-only; no fences, no key:: lines, no {{embed}}
│   └── index.md                # OKF §8 index, generated by `vault build`, committed
├── working/                    # curator's space, OKF-conformant with its own types
│   ├── .obsidian/              # committed: core plugins + Templater templates + Canvas
│   └── pages/**.md
├── quartz/                     # Quartz v4 config + custom emitter hook; content dir → ../knowledge
└── .github/workflows/publish.yml   # owner-triggered: vault build → quartz build → Pages
```

A knowledge page after migration:

```yaml
---
type: Class
title: Knowledge Graph
resource: urn:ngm:class:knowledge-graph
public: true
aliases: [KnowledgeGraph]
domain: spatial-computing
maturity: established
quality: 0.35
is-a: ["[[Content and Assets]]"]
requires: ["[[Ontology]]", "[[Schema Definition]]", "[[Triple Store]]"]
enables: ["[[Reasoning]]", "[[Knowledge Discovery]]", "[[Recommendation System]]"]
part-of: ["[[Semantic Web Infrastructure]]", "[[Knowledge Management System]]"]
status: stable
generated: { by: process:vault-migrate/1.0, at: 2026-09-22T00:00:00Z }
verified: [{ by: human:<npub>, at: 2026-09-22T00:00:00Z }]
sources: [{ id: origin, resource: "[[working/Knowledge graph notes]]" }]   # absorbs elevatedFrom
---
```

`ancestors` (the inferred closure) is **not** stored — it is a build output, as it should be.

### 3.2 The tool (`VisionClaw/crates/vault`) — one door for agents and CI

| Subcommand | Does | Replaces |
|---|---|---|
| `vault validate [--vault knowledge\|working]` | OKF v0.2 conformance + `vocabulary.yaml` + `types.json` agreement + link integrity + `public` gate | `vault-migrate --check`, `pipeline/validate.py`, `iri_integrity.py` |
| `vault find / retrieve / tree` | graph over frontmatter wikilinks, per-edge-type expansion depths, `max_documents` cap, JSON out | `ontology-bridge` reads, `iwe find/retrieve` |
| `vault edit --expect docs=N blocks=M` | guarded mutation; refused without a declared blast radius; names missing guards | (new — IWE idea) |
| `vault propose <iri>` | builds a `PatchProposal` (level, signature, hypothesis, diff, digest), runs Whelk + `conflicts` as blockers, posts a forum 31402 with `context_url`=IRI | `ontology_propose` |
| `vault gate` / `vault conflicts` | the autonomous quality gate and the semantica conflict detector | `pipeline/gate.py`, `conflicts.py` |
| `vault build` | pages → OWL (asserted + Whelk-inferred TTL), OKF export, scaffold/prose/search indexes, page API JSON, JSON-LD context, RVDB sidecar with embedding stamp, generation manifest; writes Quartz `static/` and the Loom bundle | `pipeline/build.py` and friends |
| `vault migrate --fences-to-properties` | the ONE remaining one-shot; deleted after its run | `vault-migrate` (Logseq converter, deleted) |

Library crate `vault-core` holds the parser (shared with VisionClaw's ingest), the vocabulary model, the OKF types and the promotion state machine. Binary baked into agentbox by Nix; `loom-client` (crates.io) is the read client for Loom.

### 3.3 The governance loop — forum-signed, OKF-recorded, ledgered

```
Draft Concept (working, status: draft, unverified)
      │ vault propose  ── Whelk + conflicts: BLOCK on inconsistency/cycle/contradiction
      ▼
machine-confirmed  (verified: process:vault/<ver>)  ── 31402 ActionRequest, tier by panel operator
      │                                                  Schema ⇒ tier High (floor)
      ├─ human 31403 Approve  ──► DecisionOutcome::Promote{iri}  (activated; was dead code)
      │        └─ agentbox handleGovernanceDecision ⇒ vault edit --expect … (verified += human:<npub>, status: stable)
      │                                            ⇒ Loom AttestationLedger.attest(case id)
      ├─ human 31403 Reject   ──► receipt `rejected`, no write
      └─ stale_after passes  ──► proposal → draft, re-surfaced by escalated-on-age cron
stable ── demotion proposed automatically (conflict / stale_after) ── human 31403 Demote{iri} (new variant) ── status: deprecated
```

Exposure-level changes in Loom (manifest, salience, budgets, ranking weights) evolve automatically under ADR-140 D6's paired-eval gate with auto-revert and are ledgered, never signed.

### 3.4 Serving — VisionClaw and Loom read one build

- **VisionClaw**: `CorpusSource::LocalDirectory` over the mounted named volume; same parse (from `vault-core`), same Whelk, same materialiser, same actor reload. `GitHubSource` retained as an unused impl. Class count = the build's class count.
- **Loom**: consumes the bundle `vault build` writes; generation = `visionGraph@<sha>`; reload timer enabled after first clean promotion; `/mcp` kept for external hosts; ADR-140 D4 adopts OKF's vocabulary (Attested Computation ≡ Evidence; `generated`/`verified` tiers; `stale_after` surfaced in `/health` and the manifest).
- **Obsidian desktop** (curator): Bases views over the same frontmatter — review queue, stale, by-domain — no plugin beyond core.

## 4. What is deleted

`workspace/vault`, `workspace/vault-working`, the `logseq` symlink and its host bind (`project4`); `knowledgeGraph/ontology/pages` + `knowledgeGraph/pipeline` (archived with marker); `logseq-publisher`, `-rust`, `-npm`; `visionGraph/pipeline/*.py` and `publishing-tools`; `VisionClaw/crates/vault-migrate`; `agentbox/mcp/servers/ontology-bridge.js` + `ontology-propose.js` and their `.mcp-hub-servers.json` entry; `loom-mcp-stdio`; the ADR-2041 `#[serde(alias = "logseq")]`; every `key:: value` line and `{{embed}}` in both vaults; the `mainKnowledgeGraph/pages` fixtures in `visionclaw-contracts` tests.

## 5. Acceptance (the one-shot is done when all hold)

1. `vault validate` exits 0 on both vaults; zero `key::` lines, zero `{{embed}}`, zero json-ld fences, every knowledge page has `type`, `resource`, `status`.
2. `vault build` produces one generation; **Loom `/health`, VisionClaw `/api/ontology/classes` and `vault build --stats` report the same class count**.
3. VisionClaw boots with no `PRIVATE_REPO_GITHUB_PAT` and ingests from the mounted volume; graph node/edge counts match the pre-change baseline (13,165 / 153,960) within the explainable delta of the fence migration.
4. A `vault propose` on a test IRI produces a 31402 on the local relay; a human 31403 Approve results in `verified` + `status: stable` written back and a ledger entry with the same case id; a Reject writes nothing; a Schema-level proposal is tiered High.
5. Loom serves a generation ≤ 10 minutes after a commit to `knowledge/` once the timer is enabled; `wasDerivedFrom` names the local repo.
6. Quartz builds narrativegoldmine.com from `knowledge/` locally; `static/api/search-index.json`, `static/data/ontology.ttl` and the OKF bundle are present; WasmVOWL's explorer loads against the local build at `/explorer/` (Q15, 2026-09-22). **Every pre-change public URL still resolves**: `/page/<slug>` stubs, `/graph` `/search` `/data` `/about` redirects, `/notes/` → `/`, verified by `quartz/scripts/check-urls.sh`.
7. `grep -ri logseq` across loom, VisionClaw, agentbox, VisionFlow (non-archive), visionGraph returns only ADR/history references.
8. `ontology-augment` and `podcast-knowledge-ingest` skills and the `ontology-curator` agent work via `vault` with no MCP server registered for the corpus.
9. `docs/source-repair-plan-2026-09-22.md` exists with a per-class count, the mechanical fixes are applied with before/after `vault validate` counts, and every semantic fix is a dry-run PatchProposal file ready for `vault propose` (WS-K).
10. Every repo's ADR index validates; `BASELINE-visionflow.md` and `ecosystem-map.md:65` updated in the same change.

## 6. Risks and how the plan carries them

| Risk | Mitigation |
|---|---|
| The 8,454-page rewrite loses information from the fences | `vault migrate` is lossless by construction: it fails on any fence field not covered by the vocabulary; a `--dry-run` diff is committed as evidence before the real run; git is the rollback. |
| VisionClaw's parser rewrite changes the graph | Baseline counts captured now (13,165 / 153,960); acceptance 3 bounds the delta and requires it be explained. |
| Forum wiring is the least-tested part | The promotion loop is exercised end-to-end on a test IRI against the local relay (acceptance 4) before any real proposal is generated. |
| Quartz at 8.6k pages is slow in CI | Build measured locally first; content dir excludes `working/` and assets are referenced not copied. |
| Retiring `knowledgeGraph`'s pipeline breaks a consumer nobody found | The repo keeps serving its last published export unchanged; only the source tree and pipeline are archived. |

## 7. The one-shot — workstreams and ownership

Executed by Opus workers in the mesh (`swarm-1790064371200-1r1qxr`), hooks and memory on, Fable coordinating. Order respects dependencies; A–C are serial, D–H run in parallel once C's `vault-core` API is frozen.

| WS | Repo | Work | ADR slot |
|---|---|---|---|
| **A** Canon | VisionFlow | This PRD accepted; ADR-2013; BASELINE + ecosystem-map updated; the four "Logseq corpus" passages rewritten | VF ADR-2013 |
| **B** Vocabulary & vault contract | visionGraph, VisionClaw docs | `ontology/vocabulary.yaml` derived from the fences' actual predicate set; `vault.toml`; VAULT-corpus-format.md v2 (frontmatter-only, OKF, two-vault types) | VC ADR-2112 (supersedes 2040) |
| **C** `crates/vault` | VisionClaw | `vault-core` (parser shared with ingest, vocabulary, OKF types, state machine) + CLI: validate, find/retrieve/tree, edit --expect, propose, gate, conflicts, build, migrate. Ports gate.py/conflicts.py/build.py logic. Tests incl. golden build parity against the current Python output. | VC ADR-2113 |
| **D** Migration run | visionGraph | `vault migrate --fences-to-properties` dry-run diff committed → real run → `vault validate` green on both vaults → `.obsidian/` config, `types.json`, Bases, Templater templates committed → Python pipeline, publishing-tools deleted | — |
| **E** VisionClaw ingest | VisionClaw | `CorpusSource` trait; `LocalDirectory` impl over `vault-core`; compose mounts the named volume; `GitHubConfig` optional; fixtures updated; ADR-2041 alias dropped; `vault-migrate` deleted | VC ADR-2114, 2115 |
| **F** Governance wiring | forum, agentbox, VisionClaw | `DecisionOutcome::Promote` activated + `Demote` added; ontology-governance panel (31400) with operator tiers, Schema ⇒ High; `handleGovernanceDecision` cases → `vault edit` + Loom ledger; proposal `stale_after` expiry; `vault propose` end-to-end test on the local relay | forum ADR-2013, ab ADR-2109, VC ADR-2116 |
| **G** Agent access | agentbox | `ontology-bridge` + `ontology-propose` deleted; `.mcp-hub-servers.json` entry removed; `ontology-augment`, `podcast-knowledge-ingest`, `ontology-curator` rewritten to `vault`; Nix bakes the binary; `weekly_ingest.py` re-pointed and emits OKF keys | ab ADR-2107, 2108 |
| **H** Loom | loom | Generation identity `visionGraph@<sha>`; consumes `vault build` bundle; ADR-140 D4 → OKF vocabulary; `stale_after` in `/health` + manifest; drop `loom-mcp-stdio`; enable reload timer after first clean promotion | Loom ADR-141 |
| **I** Publish | visionGraph | Quartz v4 config, ExplicitPublish ⇐ `public`, custom emitter step consuming `vault build` output; `publish.yml` rewritten; local build measured | — |
| **K** Source repair plan | visionGraph | After D: `vault validate` + `vault conflicts --severity all` produce the defect census of the SOURCE markdown (cycles, contradictions, duplicate concepts, IRI/filename slug disagreements, dangling wikilinks and asset links, quality/maturity gaps, orphan pages). `docs/source-repair-plan-2026-09-22.md` classifies every defect class with counts into MECHANICAL (lossless, batch `vault edit --expect`, executed in this one-shot with evidence) and SEMANTIC (cycles, duplicates, contradictions — each becomes a `vault propose` for human signature; none applied unsigned). Owner instruction 2026-09-22: "if we find structural problems in the source of the ontology then we should make a plan to fix the source markdown files." | — |
| **J** Estate hygiene | workspace, knowledgeGraph | Delete `vault/`, `vault-working/`, `logseq` symlink + host bind, `logseq-publisher*`; archive `knowledgeGraph` source + pipeline with marker; residue grep to zero | — |

Everything lands **uncommitted in working trees** for owner review, one commit per repo prepared with a message referencing this PRD; nothing is pushed by agents.

## 8. Out of scope (deliberately)

GitHub sync re-enablement (later decision); quorum signing (panel property later); Obsidian Local REST API; **Obsidian Publish — decided against (Q13, 2026-09-22): content would leave local control to a hosted service with no API/JSON export, so the machine artefacts would need a second host anyway; this is the written form of a previously verbal veto**; Datacore/Dataview; VisionClaw consuming Loom's bundle instead of parsing pages (a later convergence step, not the first move); IWE as a dependency.

**Parity note (publish research, 2026-09-22).** `loom/app/ontology-mcp` (Node; four tools; a public mode that read narrativegoldmine.com `/api/*`) was deleted with ADR-140 P0. Its tools map onto Loom `/mcp` (`ontology_search`→`loom.browse`, `class_get`→`loom.resolve`, `neighbours`→`loom.neighbours`, `ask`→`loom.resolve` with expansion). The public-site-reader mode is retired by design (Q10): external hosts reach the corpus through Loom `/mcp`; the published site keeps serving the same `/api/*` JSON for browsers and the explorer.
