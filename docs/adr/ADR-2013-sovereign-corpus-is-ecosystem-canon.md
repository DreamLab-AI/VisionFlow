---
id: ADR-2013
title: One corpus, one build, one gate — the sovereign corpus is ecosystem canon, authored in Obsidian and published as OKF v0.2
date: 2026-09-22
decision_status: accepted
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
verified_commit: fd6162b4bcd47758e647d18ecd74a9e01abbe3e1
owner: jjohare
review_trigger: any proposal to reintroduce a second corpus parser, a second build or an MCP server for the corpus inside the estate; GitHub sync re-enablement for VisionClaw ingest; a vocabulary.yaml major version; the first divergence between Loom's, VisionClaw's and vault build's class counts
repo: visionflow
domain: BASELINE-visionflow.md
lineage: "Canon entry for PRD-sovereign-corpus (docs/PRD-sovereign-corpus.md, accepted 2026-09-22, owner decisions Q1 to Q14). Carries no implementation; it records the ecosystem-level decision that VisionClaw ADR-2112 to ADR-2116, agentbox ADR-2109 to ADR-2108, nostr-rust-forum ADR-2013 and Loom ADR-141 implement. Seams frozen in docs/engineering/sovereign-corpus-contracts.md."
---

# ADR-2013 — One corpus, one build, one gate: the sovereign corpus is ecosystem canon, authored in Obsidian and published as OKF v0.2

## Context

The estate has one corpus and four doors that disagree about it. The raw vault on disk
(`visionGraph/knowledge/pages`) and agentbox's `ontology-bridge` MCP server both read the same
8,433 classes unreasoned, ungated and unstamped; Loom serves 8,146 classes from a bundle built
2026-08-22, reasoned, gated and stamped; VisionClaw parses the json-ld fences itself from a
GitHub pull on boot and reports 4,167 classes, reasoned but gated on `public` only and unstamped.
No one can say which number is right. The vault spec names frontmatter keys (`owl-class`,
`source-domain`, `maturity`, `quality`) that occur zero times, because the ontology lives in two
`json-ld` fences per page; 23,900 Logseq `key:: value` lines and 37 `{{embed}}`s survive against
the vault's own rule; a second Logseq-format pipeline (`DreamLab-AI/knowledgeGraph`) still claims
narrativegoldmine.com; and this repo's README, website and pitch deck all describe "a Logseq
corpus". The owner's governing principle for the remedy: git is the rollback, so migrate fully
to the cleanest outcome rather than carry shims.

## Decision

Canon records the sovereign corpus as the estate's single governed knowledge artefact, on the
terms PRD-sovereign-corpus §2 settles. This record is policy for every repo in the estate; the
carriers named under Consequences hold the implementation.

1. **One vault.** `/home/devuser/workspace/visionGraph` (`knowledge/` + `working/`) is the
   canonical corpus. `workspace/vault`, `workspace/vault-working`, the `logseq` symlink and its
   host bind are deleted. No second copy of the corpus exists anywhere in the estate.
2. **The corpus is authored in Obsidian and published as OKF v0.2 via Quartz.** Ontology lives
   in **frontmatter only** — the two json-ld fences fold into typed Obsidian Properties on all
   8,454 pages and the JSON-LD context becomes a build output. `ancestors` is not stored; the
   inferred closure is a build artefact. Relations are flat predicate keys carrying wikilink
   lists, governed by a versioned `ontology/vocabulary.yaml`; an unknown key fails the build in
   `knowledge/`. Quartz v4 renders narrativegoldmine.com from `knowledge/` under
   ExplicitPublish ⇐ `public: true`. The `knowledgeGraph` repo keeps its published-export role
   and nothing else: its Logseq corpus source tree and pipeline are archived with a marker.
3. **One parser, one reasoner, one build.** `VisionClaw/crates/vault` is the only implementation
   that parses, reasons over or builds the corpus. `vault-core` holds the parser — shared with
   VisionClaw's ingest — the vocabulary model, the OKF types and the promotion state machine;
   the CLI carries `validate`, `find`/`retrieve`/`tree`, `edit --expect`, `propose`, `gate`,
   `conflicts`, `build` and the single-use `migrate`. It replaces `pipeline/*.py`,
   `publishing-tools`, `vault-migrate` and `conflicts.py`. Loom consumes the bundle
   `vault build` writes; VisionClaw ingests the same vault through `CorpusSource::LocalDirectory`
   over a read-only mounted named volume, with GitHub pull removed and `GitHubConfig` optional.
4. **One gate, and it is human where it matters.** Content, Schema and **Demotion** changes
   require a human-signed forum `31403`. Schema is floored at tier High. Exposure-level changes
   in Loom (manifest, salience, budgets, ranking weights) self-evolve under ADR-140 D6's
   paired-eval gate with auto-revert and are ledgered, never signed. Whelk inconsistency,
   subclass cycles and relation contradictions are automatic **blockers** and can never be
   approved around. Every `PatchProposal` carries `stale_after` (14 days); on expiry the
   proposal reverts to draft and is re-surfaced. Quorum is deferred; there is one human signer.
5. **No MCP inside the estate for the corpus.** Agents reach the corpus through the `vault`
   binary over Bash and through `loom-client`; humans use Obsidian desktop with core plugins
   only. `ontology-bridge` and `ontology-propose` are deleted and their `.mcp-hub-servers.json`
   entry removed. Loom keeps `/mcp` solely as the external-host door; `loom-mcp-stdio` is
   dropped. IWE is a source of ideas, never a dependency.
6. **Identity is the local repo.** A Loom generation is `visionGraph@<local sha>` plus a content
   digest; `wasDerivedFrom` names the local repository, not a GitHub URL. Class counts reported
   by Loom `/health`, VisionClaw `/api/ontology/classes` and `vault build --stats` must be
   equal — a divergence is a defect, not a rounding difference.
7. **Canon prose stops saying "Logseq corpus."** Every current-tense claim in this repo's
   README, website, pitch deck, terminology playbook and ecosystem map describes the corpus as
   authored in Obsidian and published as OKF v0.2. Historical statements keep their tense.

## Consequences

Deleted by this decision: `workspace/vault`, `workspace/vault-working`, the `logseq` symlink and
its host bind; `knowledgeGraph/ontology/pages` and `knowledgeGraph/pipeline` (archived with a
marker, not removed); `logseq-publisher`, `-rust` and `-npm`; `visionGraph/pipeline/*.py` and
`publishing-tools`; `VisionClaw/crates/vault-migrate`; agentbox's `ontology-bridge.js` and
`ontology-propose.js` with their hub entry; `loom-mcp-stdio`; the ADR-2041
`#[serde(alias = "logseq")]`; every `key:: value` line and `{{embed}}` in both vaults; and the
`mainKnowledgeGraph/pages` fixtures in the `visionclaw-contracts` tests.

The estate gains a single answer to "how many classes are there", which closes the standing
four-doors finding. It pays for that with a one-shot 8,454-page rewrite whose losslessness rests
on `vault migrate` failing on any fence field the vocabulary does not cover, with a committed
`--dry-run` diff as evidence and git as the rollback. Removing MCP from the corpus path costs
every agent skill a rewrite onto the CLI and makes the Nix-baked binary a boot-path dependency.
VisionClaw's parser rewrite may move the graph; the baseline (13,165 nodes / 153,960 edges) bounds
that delta and any change must be explained rather than absorbed. Forum wiring is the least-tested
seam and is exercised end-to-end on a test IRI against the local relay before any real proposal is
generated. GitHub sync re-enablement, quorum signing, Datacore/Dataview, and VisionClaw consuming
Loom's bundle instead of parsing pages are all deliberately out of scope.

Follow-on carriers: VisionClaw ADR-2112 (vault corpus format, supersedes 2040), ADR-2113
(`crates/vault`), ADR-2114 and ADR-2115 (`CorpusSource` ingest), ADR-2116 (governance apply path);
agentbox ADR-2109 (governance panel), ADR-2107 and ADR-2108 (agent access, Nix baking);
nostr-rust-forum ADR-2013 (`DecisionOutcome::Promote` activated, `Demote` added); Loom ADR-141
(generation identity, OKF vocabulary, reload timer). Seams are frozen in
`docs/engineering/sovereign-corpus-contracts.md`; a worker may extend a contract additively but
may not change a field's meaning without amending that file in the same change.

## Verification

`implementation_status: partial` and `activation_status: staged` are recorded honestly: this
record and the canon prose it governs are written at `fd6162b`, while the workstreams that build
`crates/vault`, run the migration and wire the gate land uncommitted in sibling working trees for
owner review. Nothing in this record promotes any sibling's axis.

Established at `fd6162b` by:

- `grep -rn -i logseq README.md website/ pitch/ docs/ --include=*.md --include=*.html` in this
  repo returning only ADR, history or dated-migration references — every current-tense claim at
  `README.md:112,116`, `website/static/index.html:359,366`,
  `pitch/visionflow-ecosystem.html:274-275` and `docs/terminology.md:62,101-102` rewritten.
- `docs/ecosystem-map.md:65` naming the local Obsidian vault over a mounted volume rather than
  "Logseq/GitHub".
- `docs/BASELINE-visionflow.md` carrying the "one corpus, one build, one gate" invariants and the
  four-doors divergence marked as now being closed, amended in this same change as the change
  process requires.
- `node scripts/adr-index-gen.cjs docs/adr` exiting 0 with this record in the generated index.

Ratification evidence, to be filed as the carriers land, is PRD-sovereign-corpus §5 in full. The
load-bearing item is acceptance 2: **Loom `/health`, VisionClaw `/api/ontology/classes` and
`vault build --stats` reporting the same class count.** Until that equality is demonstrated at a
named commit, this record describes policy rather than a running estate, and no maturity claim
above `partial`/`staged` may cite it.
