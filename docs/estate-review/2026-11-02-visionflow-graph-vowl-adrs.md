---
title: VisionFlow, visionGraph and WasmVOWL ADR assessment
date: 2026-11-02
status: assessed
type: explanation
---

# VisionFlow, visionGraph and WasmVOWL ADR assessment

This pass assesses 23 records section by section, using source tracing at the pinned revisions: VisionFlow ADR-2001 to ADR-2005, ADR-2007, ADR-2009 to ADR-2016 and ADR-2018 to ADR-2021; engineering ADR-004 and ADR-005; visionGraph `knowledge/ADR-008.md` and `ADR-012.md`; and WasmVOWL ADR-001. Of 155 section headings, 79 are implemented by the code they describe (or, for the speculative ADR-005, consistent with its unbuilt status), 22 are contradicted or stale, and 54 cannot be checked from source: runtime, CI, deployment or browser claims, empty sections, or repositories with no wave-2 pin. There are 41 findings. The most consequential are these. ADR-2005's enforced `mcp-ontology-tools` drift axis counts a file (`mcp/servers/ontology-bridge.js`) that no longer exists in agentbox at `c57d9c0508f4`, and stays green only because the counter is pinned to the older `b859b37`. ADR-2011's legacy-panel path still lets an agent's own `risk_tier` stand when no task-property triple is present (`crates/nostr-bbs-core/src/governance.rs:526-527`@c39f41f), contradicting Decision §5. ADR-2013's removal of the GitHub corpus pull is contradicted by VisionClaw, which still selects `GitHubSource` when `VAULT_ROOT` is unset (`src/services/corpus_source/mod.rs:16-21`@86cdd76). Canon still quotes a 99.2%/264-row parity figure where the solid-pod-rs checklist now gives 86.4% over 300 rows (ADR-2018). Classifications: 21 operative decisions; ADR-2004 is superseded and historical; ADR-005 is speculative. visionGraph ADR-008 and ADR-012 are not decision records: they are generated ontology concept pages with empty sections.

## Findings

Each finding names the trigger, consequence, evidence at the pinned revision and what would close it. Only the ADR text was assessed; no product code or ADR was edited.

### ADR-2001

None material. Minor drift: the Context's "no `*.rs` in the repo" is now literally false because audit probe `.rs` files sit under `docs/estate-review/evidence/` (e.g. `docs/estate-review/evidence/2026-10-31/xr-links-probe.rs`@5f8b1da). The claim's intent (no website Rust) still holds. Close by scoping the statement to `website/` in the Baseline, not by editing the dated Context.

### ADR-2002

1. **Single-module claim contradicted.** Trigger: contact form (2026-10-15) added a second `type="module"` entry. Consequence: the Consequences ceiling ("any interactivity beyond one ES module has no build affordance") is already exceeded without a superseding note, and the Verification "one entry" check would now fail. Evidence: `website/static/index.html:1300-1301`@5f8b1da. Close: a dated annex stating the ceiling is "unbundled ES modules" (or counting the entries), with the Baseline updated.
2. **PRD-website still specifies the reversed Rust/WASM design.** Trigger: follow-on debt named in Consequences never closed for the PRD. Consequence: a reader of the PRD gets build instructions contradicting the operative ADR. Evidence: `docs/PRD-website.md:106-132`@5f8b1da. Close: mark the PRD's technology section superseded by ADR-2002 or rewrite it.
3. **Acceptance annex's inventory count is stale.** 13 required vs 17 at `website/assets.manifest.json:6`@5f8b1da. Dated annex, so not wrong at its date; the Baseline or a new annex should carry the current count rather than readers relying on 13.

### ADR-2003

1. **Recorded drift-counter pin is stale.** The annex names agentbox `89301ec7…`; the workflow and allowlist pin `b859b37…`. Evidence: `.github/workflows/deploy.yml:68`, `scripts/drift-counter/allowlist.json:6`@5f8b1da. Consequence: a reader checking the ADR against CI finds a mismatch; no runtime effect because the counter checks the allowlist, not the ADR. Close: a dated annex naming the current pin (or referring to `allowlist.json` instead of a literal SHA).
2. **Recovery procedure still absent.** The closeout's rollback runbook is acknowledged missing and no website rollback document exists at 5f8b1da. Consequence: the published-revision receipt has no consumer. Close: write the runbook and cite it.

### ADR-2004

1. **Superseded record still reads `activation_status: live`.** Frontmatter (`docs/adr/ADR-2004-diagram-baseline-vendored-render-gate.md:5-7`@5f8b1da) pairs `superseded` with `implementation_status: complete`, `activation_status: live`, while the decided render engine (vendored Mermaid over CDP) is absent. Consequence: a status reader may treat the Mermaid path as running. Close: set activation to retired/inactive per the template's superseded convention, or add a dated annex stating only the baseline guard is live and it is now owned by ADR-2020.

### ADR-2005

1. **The mcp-ontology-tools axis measures a component retired in current agentbox.** Trigger: agentbox removed `mcp/servers/ontology-bridge.js` between the counter's pin `b859b37` and the wave-2 pin `c57d9c0508f4` (`git ls-tree c57d9c0508f4 mcp/servers/ontology-bridge.js` is empty; present at `b859b37`, which is an ancestor). Consequence: the enforced axis stays green only because the pin is frozen at an older agentbox; moving the pin to current agentbox makes the source missing, and the policed sites (`allowlist.json` mcp sites, e.g. `docs/architecture/compatibility-matrix.md`, `docs/protocol/mesh-smoke-test.md`) assert a tool count for a component that no longer ships. `allowlist.json:32`@5f8b1da already concedes "the bridge is retired in current Agentbox". Close: retire the axis (or convert its sites to dated historical statements) by a dated ADR annex, then repin to current agentbox.
2. **Annex pin differs from the live pin.** `89301ec7…` in the annex vs `b859b37…` in `allowlist.json:6` and `drift-counter.yml:61`@5f8b1da. Dated annex, but no later annex records the repin. Close: annex the repin with its reason.

### ADR-2007

- **The activation scope is overstated.** Trigger: the Acceptance annex says the gate "runs on every push and pull request". Consequence: a push that changes a record's status only through a non-ADR path never runs the gate. In practice a status change always touches `docs/adr/**`, so the real risk is low. The overstatement is in the wording. Evidence: `.github/workflows/adr-index.yml:9-23@5f8b1da`. To close it, reword the annex to "on pushes to `main` and PRs that touch ADR paths".
- **A citation role is misdescribed.** Trigger: the annex says VF-01's `:24` is "the Decision paragraph". Consequence: a reader following VF-01 lands on the heading, not on the rule text. Evidence: `docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:24,26@5f8b1da`. To close it, whoever owns VF-01 retargets the citation to `:26`; that topic is outside this assignment.

### ADR-2009

None. The record is accurate at `5f8b1da`, and its stated limit, that nothing here says the mesh is correct, holds.

### ADR-2010

- **`verified_commit` is unresolvable.** Trigger: the frontmatter sets `verified_commit: 03db67106e8d5aab6e6a992095107f14df3fc886`. Consequence: the record claims verification at a revision no one can check out. `git cat-file` fails for it in VisionFlow (worktree and primary), VisionClaw, agentbox and nostr-rust-forum. The value was introduced in `ec7f353`. The generator checks only that the field is present (`scripts/adr-index-gen.cjs:28-32@5f8b1da`), so the gate passes anyway. Evidence: `docs/adr/ADR-2010-augmentation-conditions-are-the-canon-audit-lens.md:10@5f8b1da`. To close it, restamp to a reachable VisionFlow commit at which the matrix table exists, and consider having `--check` verify that `verified_commit` resolves as an object.
- **The Verification section names a script that does not exist.** Trigger: "checked by `scripts/check-citations.cjs`". Consequence: a reader cannot run the stated check. Evidence: there is no such path at `5f8b1da`; the real checker is `scripts/check-augmentation-citations.cjs`. To close it, rename the reference.
- **The citation checker is weaker than the decision implies.** Trigger: Decision §1 says a status may be raised "only by a citation to code". Consequence: the gate confirms only that a cited line exists in whatever sibling working tree is checked out. It does not check a pinned revision or what the line says, so a stale or irrelevant citation passes. For example, the matrix still cites `agentbox.toml:822-849` for `authority_class` (`docs/architecture/compatibility-matrix.md:39@5f8b1da`), but that table is now at `agentbox.toml:1416-1444@c57d9c0`. To close it, pin the substrate revisions in the matrix and re-resolve the citations.
- **The website contradicts Consequence 1.** Trigger: unqualified "human-in-the-loop governance" copy. Evidence: `website/static/index.html:8,123@5f8b1da`. To close it, cite the C2 `measured` row or soften the wording.

### ADR-2011

- **A legacy-panel agent can still self-tier below the advertised default.** Trigger: a 31402 carries a `risk_tier` (for example `low`) and neither the panel nor the request carries a task-property triple. Consequence: the effective tier is the agent's own declaration, which can sit below `ESCALATION_DEFAULT_TIER`. This contradicts Decision §5 ("carries no authority on its own") and Context's complaint that "the party with the strongest incentive to under-tier is the only party that tiers". Evidence: `crates/nostr-bbs-core/src/governance.rs:526-527@c39f41f`. Consequences §2 ("legacy panels … behave as today") arguably licenses it, so the record is internally inconsistent. To close it, either make the arm `d.max(advertised_default)`, or amend §5 to scope it to triple-bearing panels.
- **`verified_commit` is unresolvable** (the same object as ADR-2010). Evidence: `docs/adr/ADR-2011-task-properties-set-the-boundary-not-agent-self-tiering.md:10@5f8b1da`. To close it, restamp.
- **Stale citations.** The `agentbox.toml:822-849` citation (lineage, Context) now sits at `:1416-1460@c57d9c0`; the `governance.rs:2893` and `:516` citations are now `:3158` and `:515`. To close it, re-resolve them if the record is revised.

### ADR-2012

- **The retired anchoring sentence is still live.** Trigger: D3 retires the "testnet4 by default and mainnet by explicit operator choice" anchoring note. Consequence: the public site still states it, now attached to taproot commitments, so the public page and canon disagree. Evidence: `website/static/index.html:631@5f8b1da`. To close it, rewrite the sentence to state that parent-network choice is configuration, or record an owner decision to keep it.
- **The registry parity criterion fails.** Trigger: Verification requires the same external rows in VisionFlow's and agentbox's registries. Consequence: the teller kinds 30333 and 3700 are registered in canon but not in agentbox, so agentbox's registry lint would not flag a collision with them. Evidence: `docs/protocol/event-kind-registry.md:33,49@5f8b1da` versus `docs/PROTOCOL-registry.md:165-174@c57d9c0` (no teller rows). To close it, mirror the teller rows into agentbox, or drop the parity criterion.
- **The Context and Verification citations are stale.** Trigger: README was rewritten (163 lines). Consequence: `README.md:134`, `:240-256` and `:248` resolve to nothing, so the Verification criterion "README.md:248 naming sidestr" is unmeetable as written. Evidence: `README.md@5f8b1da` (sidestr-rs appears only at `:31,55`). To close it, retarget the criterion to wherever canon now states the payment-account claim, if anywhere.
- **Two sidestr-rs revisions are given for one claim.** Evidence: "Source qualification" names `2bdee2d…`; "Verification" names `bcbe30b…`. Neither can be checked at a wave-2 pinned revision. To close it, name one revision and the relationship between the two.

### ADR-2013

- **Drift: the "one parser" decision is superseded in practice but not in the record.** Trigger: ADR-2014 moved parsing, reasoning and building to Ontomark. Consequence: ADR-2013 decision 3 and `docs/BASELINE-visionflow.md:333-334`@5f8b1da still name `VisionClaw/crates/vault` as the only implementation (visionGraph `vault.toml:5`@556a015 says the same). In reality there are two derivations: the Ontomark consumers listed above, and VisionClaw's retained `crates/vault`/`vault-core` with its whelk-rs fork (`Cargo.toml:215`@86cdd76). ADR-2013 `superseded_by: []` hides this. Close: amend decision 3 and baseline invariant 8 to name Ontomark, or mark ADR-2013 partly superseded by ADR-2014.
- **Contradiction: GitHub pull is not removed.** Evidence: `src/services/corpus_source/mod.rs:16-21`@86cdd76 ships `GitHubSource` as the selected source when `VAULT_ROOT` is unset. Consequence: a VisionClaw started without the mount silently reverts to the remote pull that the decision removed. Close: delete `GitHubSource`, or amend the decision to make it a declared fallback.
- **Minor drift: `publishing-tools` is retained.** Evidence: visionGraph `README.md:32`@556a015. Close: correct the Consequences list (the Python pipeline is gone; the directory now holds only the WasmVOWL explorer).

### ADR-2014

None beyond the open gates the record itself declares. The surviving duplicates are tracked under ADR-2013 above.

### ADR-2015

- **Drift: the status is stale.** Trigger: Ontomark implemented the typed model. Evidence: `crates/ontomark-model/src/kind.rs:10-30` and `model.rs:65-72,153-160`@4b9d15d, against the ADR front matter `implementation_status: none` and `verified_commit: "not applicable"`@5f8b1da. Consequence: the ADR index under-reports delivered work, and an auditor reading the index would treat the typed model as unbuilt. Close: restamp to `implementation_status: partial` with an Ontomark `verified_commit`, keeping corpus migration as the open item.

### ADR-2016

None found in the checkable sections.

### ADR-2018

- **Contradiction: canon restates a stale parity figure.** Trigger: the 2026-11-01 checklist correction. Evidence: `docs/architecture/compatibility-matrix.md:16`@5f8b1da (99.2%, 264 rows, "unreleased and built by no consumer yet") against `crates/solid-pod-rs/PARITY-CHECKLIST.md:58`@9b6143c (86.4% over 300 rows, upper bound). The release claim is contradicted by agentbox's pin of v0.5.0-alpha.18 (`lib/solid-pod-rs.nix:38`@c57d9c0) and `forge-full` on by default (`flake.nix:1548`). Consequence: the matrix overstates parity by about 13 points and misstates release state, which breaks this ADR's own rule. Close: re-source the Pod row from `scripts/parity-check.sh` at a named solid-pod-rs revision, and update the release clause.

### ADR-2019

- **Open gap (as the record says): the mirror is not started in shipped pods.** Evidence: `agentbox.toml:621`@c57d9c0 (`mirror_follow = []`), `mirror.rs:357-358`@9b6143c. Close: an agentbox change that defaults follow to `self/<repo>` plus a relay, or an amendment to the Decision.
- **Debt: close-out citations are unbound to a revision.** `lib.rs:4965-4981` and `5022-5027` match `76ab7aab`. At 9b6143c the code is `lib.rs:4978-4994` and `5034-5040`. Close: state the revision beside the citation, or rebind.

### ADR-2020

- **Drift: internal status contradiction.** The Consequences follow-on cites `implementation_status: partial`, but the front matter is `complete`@5f8b1da. The close-out resolves the follow-ons. Close: edit the parenthetical.

### ADR-2021

None.

### ADR-004

- **Drift: D6 still states an accepted precedent system that has been retired.** Trigger: agentbox removed the precedent service and bridge, and the template marks the guide retired. Consequence: a reader of the accepted ADR expects auto-applied precedents that do not exist; D6 and the "reduced steering burden" consequence overstate the system. Evidence: agentbox `CHANGELOG.md:743`–`:750`@c57d9c0; VisionFlow `docs/engineering/templates/governance-decision.json:68`–`:73`@5f8b1da. Closes when: an amendment marks D6 Retired or Deferred, with a link to the C4 re-entry condition the template already states.
- **Tension: the CI gate blocks at 50%, while the ADR, script default and matrix speak of an 80% target.** Consequence: pairing coverage can fall to 50% with CI green. Evidence: `.github/workflows/harness-fitness-gates.yml:56`, `:62`–`:65`; `scripts/harness-audit.sh:57`@5f8b1da. Closes when: the workflow's threshold is aligned with 80%, or the ADR states the 50% floor.
- **Drift: the annex attributes the partial-source failure mode to "ADR-005 §Decision 2", but ADR-005 D2 is the mandate inventory.** Consequence: a broken lineage pointer. The pattern actually belongs to the drift counter (ADR-2005). Evidence: `docs/engineering/ADR-004-harness-engineering-framework.md` Acceptance progress, versus `docs/engineering/ADR-005-mandate-at-grant-governance.md` §D2@5f8b1da. Closes when: the reference is corrected.
- **Drift: the compatibility-matrix harness table is the historical 33/40 output, but the ADR reports 31/39 after the fix.** Evidence: `docs/architecture/compatibility-matrix.md:83`, `:97`@5f8b1da. Closes when: the table is regenerated from the current script.

### ADR-005

- **Drift: the Context motivation is stale for two of the three channels it lists as open.** Trigger: agentbox hardened `agent_event_auth` to nip98 and replaced env spreads with allowlists (IR1/IR3). Consequence: the ADR overstates the open attack surface, and it does not record that parts of mandate-at-grant shipped as static hardening outside this design. Evidence: `agentbox.toml:186`; `management-api/adapters/orchestrator/local-process-manager.js:51`, `:594`; `management-api/utils/process-manager.js:216`–`:223`@c57d9c0. Closes when: a dated amendment restates which grants remain ungoverned, with current citations.
- **Drift: stale citation `memory-tools.js:60-68` (now `:203`–`:218`) and a stale example tool `precedent_promote` (removed).** Evidence: agentbox `mcp/servers/lib/memory-tools.js:203`; `CHANGELOG.md:743`@c57d9c0. Closes when: re-cited.
- **Drift: the Acceptance-progress annex credits "this record's §Decision 2" with a partial-source failure mode that this ADR does not contain.** Evidence: §D2 text versus the annex, `docs/engineering/ADR-005-mandate-at-grant-governance.md`@5f8b1da. Closes when: the annex names the actual source of the pattern (ADR-2005 drift counter).

### visionGraph ADR-008, ADR-012

- **Debt: two contentless numbered-ADR concept pages are published as stable, established ontology classes.** Trigger: the enrichment swarm minted classes from bare ADR identifiers. Consequence: `ADR-008` and `ADR-012` enter the sovereign corpus as `public: true`, `status: stable`, `maturity: established`, `type: Class` with `is-a Software Architecture`. Yet they denote no specific record, are ambiguous across the many estate corpora that have an ADR-008/012, and have empty Semantic Classification and Provenance sections. A numbered individual record is also not a subclass of Software Architecture. Evidence: `knowledge/ADR-008.md:2`–`:24`, `:27`, `:34`; `knowledge/ADR-012.md:2`–`:24`, `:27`, `:34`@556a015. Closes when: the pages are retired or merged into a single "Architecture Decision Record" class, or downgraded from stable/established; ontomark's OKF validation should flag empty mandatory sections. Ontomark rules were not checked here.

### WasmVOWL ADR-001

- **Open: the per-tick position write feeds the load effect's dependency, a topology-reset hazard.** Trigger: `updateNodePosition` mutates `nodes` inside an immer `set` on every animation frame, and the load `useEffect` lists `nodes` as a dependency. Consequence: once the Map/immer failure is fixed, every tick would re-run `loadOntology` and `initSimulation`, resetting the layout. This is the "simulation ticks repeatedly reset topology" acceptance item. Evidence: `modern/src/hooks/useWasmSimulation.ts:105`–`:107`, `:120`, `:151`–`:154`; `modern/src/stores/useGraphStore.ts:154`–`:158`@51a1484. Source trace only; not executed. Closes when: positions live outside the topology dependency (a ref or separate store slice), with a browser test that loads a corpus and asserts that `loadOntology` is called once across N ticks.
- **Open: Map drafts through immer without `enableMapSet`.** Evidence: `useGraphStore.ts:6`, `:61`; no `enableMapSet` under `modern/`@51a1484. Consequence: consistent with the ADR's "Map mutation fails". Closes when: `enableMapSet()` is called (or the state moves to plain records) and the frontend suite is re-run.
- **Drift: `verified_commit: 36105cc` predates the pinned revision and the 2026-09-07 annexes.** Evidence: `docs/adr/ADR-001-explorer-consumer-contract-closeout.md:10`@51a1484. Closes when: the stamp is restamped at the revision the annexes were checked against.

## ADR-2001 section assessment

Classification: operative decision (accepted/complete/live), repository governance.

| Section | Disposition |
| --- | --- |
| Context | Claims ADR-001..007 drifted and the site has no compile step. **Implements** (historical premise still true): `website/build.sh:5-7`@5f8b1da says "No compile step, no bundler, no WASM"; no `Cargo.toml`/`*.rs`/`*.wasm` under `website/` at 5f8b1da. The repo-wide "no `*.rs` exists" statement is no longer literally true: audit probe sources such as `docs/estate-review/evidence/2026-10-31/xr-links-probe.rs`@5f8b1da are committed, but none is website code. The engineering ADR-004/005 collision still exists (`docs/engineering/ADR-004-harness-engineering-framework.md`, `ADR-005-mandate-at-grant-governance.md`@5f8b1da). |
| Decision | Baseline plus thin ledger, archive under a tombstone, template, generated index with `repo` enum `visionflow`. **Implements**: `docs/BASELINE-visionflow.md` exists; seven records in `docs/archive/adr/` under a frozen tombstone (`docs/archive/adr/README.md:1-8`@5f8b1da); `docs/adr/TEMPLATE.md` exists; `scripts/adr-index-gen.cjs:46`@5f8b1da `repo: ['visionflow']`; `docs/engineering/` untouched. |
| Consequences | Index generator fails CI on malformed frontmatter; docs/README links point to archive and baseline; engineering collision recorded as known divergence. **Implements**: `.github/workflows/adr-index.yml:44`@5f8b1da runs `--check`, `:53-60` fails on a stale README; `docs/README.md:7,44`@5f8b1da link archive and baseline; `docs/BASELINE-visionflow.md:268-273`@5f8b1da records the dual-numbering hazard. |
| Verification | Historical receipt at `c205575` (git mv, no `docs/ADR-*.md`, index exit 0). **Implements** at 5f8b1da: no `docs/ADR-*.md` remains in `docs/`; index validator exits 0 on an exported copy (21 records valid). The `c205575` run itself is not re-executed. |
| Closeout extension — 2026-09-04 | Validator accepts the operative records; `--check` does not write README; index freshness not established; engineering sequence unresolved. **Implements** for the mechanism: `scripts/adr-index-gen.cjs:11,215-218`@5f8b1da (check mode does not write). Freshness is now enforced by `adr-index.yml:53-60` and the committed README matches a regeneration at 5f8b1da (path echo aside). The engineering sequence and section-level reconciliation of archived records **cannot be checked** as complete from source; they remain open as stated. Hosted CI execution not observed. |

## ADR-2002 section assessment

Classification: operative decision (accepted/complete/live), website build.

| Section | Disposition |
| --- | --- |
| Context | Legacy ADR-001 D1/D3 chose Rust/WASM; none was committed. **Implements** (historical): `docs/archive/adr/ADR-001-website-technology.md`@5f8b1da archived; no Cargo/Rust/WASM under `website/`@5f8b1da. |
| Decision | `build.sh` is the whole build and copy-only: wipe `dist/`, copy `static/*`, emit CNAME, stage images; WebGL2 module `static/js/mesh-webgl.js`. **Implements**: `website/build.sh:17-33`@5f8b1da (`rm -rf dist`, `cp -r static/*`, CNAME at `:27`, `website-assets.mjs stage`/`verify` at `:30,33`); `website/static/js/main.js:1`@5f8b1da imports `mesh-webgl.js`. Staging now runs a Node script, which is still copy-and-verify, not compilation. |
| Consequences | `dist/ == static/ + generated images`; WebGL2 degrades (null without WebGL2, reduced-motion honoured); one ES module ceiling; README, PRD-website and site-verification still assert Rust/WASM as follow-on debt. **Partly implements**: `website/static/js/mesh-webgl.js:11-16`@5f8b1da (`if (!gl) return null`; `prefers-reduced-motion` single frame). **Contradicts** "one ES module": `website/static/index.html:1300-1301`@5f8b1da loads two module entries (`main.js`, `contact.js`). Debt partly closed: `README.md:82`@5f8b1da now says no Rust/WASM; **still open** in `docs/PRD-website.md:106-132`@5f8b1da (wasm-pack, two WASM modules, WASM budgets). |
| Verification | At `cf535f8`: `build.sh:6-7` sentinel, `:10-24` copy steps, one `type="module"` entry. **Historical; citations stale** at 5f8b1da: sentinel is now `build.sh:5-7`, copy steps `:17-36`, and there are two module entries (`index.html:1300-1301`). The `cf535f8` facts are not re-checked at that revision. |
| Closeout extension — 2026-09-04 | Optional image-copy failures were suppressed; closeout requires an asset inventory and browser verification. **Superseded by later implementation**: the `|| true` copy is gone (`website/build.sh:9-14,29-33`@5f8b1da). |
| Acceptance progress — 2026-09-05 | Manifest with 13 required and 4 optional groups; receipt outside `dist/` with tree digest and revisions; test suite of 20 assertions; browser receipts 26/26. **Implements the mechanism** (`scripts/website-assets.mjs:147` source-absent, `:216` published revision, `:231` tree_sha256, `:240` receipt at `website/build-receipt.json`, `:260` `ASSET-INVENTORY-OK`@5f8b1da; `tests/gates/website-assets.test.sh` and `scripts/website-browser-check.mjs` exist; `package.json:18` `test:browser`). **Count drift**: `website/assets.manifest.json:6`@5f8b1da now lists 17 required entries (adds `robots.txt`, `js/nostr-contact.js`, `data/estate-health.json`, and others since 2026-09-06/10-15), not 13. Browser receipt (`docs/estate-closeout/2026-09-05/website-browser-receipt.json`) and test pass counts **cannot be checked** from source; they are dated runtime claims. |

## ADR-2003 section assessment

Classification: operative decision (accepted/complete/live), website deploy.

| Section | Disposition |
| --- | --- |
| Context | Legacy ADR-001 D4 chose a gh-pages branch; artefact/deploy actions shipped instead. **Implements**: no `gh-pages` reference in `.github/`@5f8b1da (git grep empty); no local `gh-pages` branch. |
| Decision | Build on `main`, `upload-pages-artifact@v3`, `deploy-pages@v4`, two-job split, `pages: write` + `id-token: write`, `contents: read`, `concurrency: pages` with `cancel-in-progress: false`, CNAME emitted by `build.sh`. **Implements**: `.github/workflows/deploy.yml:34-37` (push to main), `:39-42` permissions, `:44-46` concurrency, `:183-186` upload `website/dist`, `:188-197` deploy job `needs: build`@5f8b1da; `website/build.sh:27`@5f8b1da writes CNAME. Actions are now pinned by commit SHA (`deploy.yml:184,197`), not by `@v3`/`@v4` tags; that the SHAs correspond to v3/v4 **cannot be checked** from source. |
| Consequences | No `contents: write`, no build output in history, no bot PAT; bound to Pages OIDC; serialised releases; domain in one place `build.sh:16-17`. **Implements**: `deploy.yml:39-42,189-190`@5f8b1da. Citation stale: the CNAME line is now `website/build.sh:27`@5f8b1da. DNS reconciliation absence is consistent with source (nothing references DNS). |
| Verification | At `cf535f8`: `deploy.yml:8-11,13-15,29-32,41-43`. **Historical; citations stale**: the same facts are now at `deploy.yml:39-46,183-186,195-197`@5f8b1da. |
| Closeout extension — 2026-09-04 | Workflow did not call the verify sequence; asks for blocking quality checks, rejection demonstration, published revision record, recovery procedure. **Superseded by later implementation** for gates and revision record (see next row). Recovery procedure: still **not found** in source. |
| Acceptance progress — 2026-09-05 | Six blocking gates before upload; drift counter as a reported/conditionally blocking seventh gate pinned to agentbox `89301ec7…`; published revision written into receipt; local pass results; rollback runbook absent. **Implements** the gates: `deploy.yml:80-129`@5f8b1da (asset inventory with `--published-revision`, build/link sentinels with `|| true` then grep, meta and SD sentinels, diagram baseline), `:131-145` drift counter reported on checkout failure, `:147-174` `published.{target,revision,workflow_run,ref}`, `:176-180` receipt uploaded. **Drift**: the agentbox pin is now `b859b37d2ffb74c08cfbb9aa9ad7e1cc61d6b188` (`deploy.yml:68`@5f8b1da, matching `scripts/drift-counter/allowlist.json:4-7`), not `89301ec7…`. Local gate pass counts and "13/13 required" **cannot be checked** from source and the required count has since become 17. Hosted CI, Pages environment, DNS and serving **cannot be checked**. |

## ADR-2004 section assessment

Classification: superseded/historical (`decision_status: superseded`, `superseded_by: [ADR-2020]`, frontmatter lines 5-9@5f8b1da). The committed-baseline guard survives; the vendored-Mermaid/CDP render path it decided does not.

| Section | Disposition |
| --- | --- |
| Context | Dark-theme exports made labels invisible; legacy ADR-005 D3 prescribed `render-diagrams.sh` with live mmdc. **Implements** (historical): `scripts/render-diagrams.sh` absent at 5f8b1da; `.github/workflows/diagram-render.yml:3-5`@5f8b1da restates the defect. |
| Decision | Committed `rendered/` SVG baseline is authoritative via browserless `check-diagram-text.js`; re-render is secondary via vendored `scripts/diagram-render/vendor/mermaid.min.js` 11.16.0 over raw CDP with `htmlLabels:false`. **Baseline half implements**: `diagram-render.yml:41-43`@5f8b1da runs the guard first; `presentation/report/diagrams/rendered/*.svg` committed. **Render half contradicted** (superseded by ADR-2020): no `scripts/diagram-render/vendor/` at 5f8b1da; `scripts/diagram-render/render.mjs:1-9,18`@5f8b1da renders in-process with Pele from `scripts/vendor/pele/` (via `scripts/lib/pele-static.cjs`), no browser, no CDP. |
| Consequences | No CI dependence on byte-identical browser output; vendored 3.5 MB engine; CI still installs `mermaid-cli` + `puppeteer` for Chrome; baseline regenerated per edit; vendored blob and `MERMAID_CLI_VERSION` kept in lockstep. **Partly implements**: word-diff with byte diff informational (`diagram-render.yml:52-56`@5f8b1da). **Contradicted**: the workflow no longer installs `mermaid-cli`/`puppeteer` and no `MERMAID_CLI_VERSION` exists (git grep empty in `.github`, `scripts/diagram-render/package.json`@5f8b1da); the engine is Pele (`scripts/vendor/pele/pele.js`, `VERSION`). |
| Verification | At `cf535f8`: workflow line ranges, vendored blob with "11.16.0", no `render-diagrams.sh`. **Historical**; only the last fact holds at 5f8b1da. |
| Closeout extension — 2026-09-04 | Browserless probe passes all ten diagrams; no Chrome render or hosted CI. **Implements** for structure: ten `.mmd` sources and ten committed rendered SVGs (`presentation/report/diagrams/`@5f8b1da). Pass result **cannot be checked** from source. |
| Acceptance progress — 2026-09-05 | Three-stage gate run against sidecar Chrome with vendored Mermaid; baseline guard added as a blocking step in `deploy.yml`. **Implements** the deploy gate: `.github/workflows/deploy.yml:128-129`@5f8b1da. The Chrome/Mermaid execution is a dated runtime claim, **cannot be checked**, and describes a toolchain that has since been replaced by Pele. Its "Remaining" item (cross-environment byte parity of puppeteer Chromium) is moot under the current in-process renderer. |
| Estate audit — 2026-09-07 | `diagram-index-gen.cjs` validates structure with warning-only citation checks, names values "declared source revisions", qualifies ADR keys by repository; regression test exists. **Implements**: `scripts/diagram-index-gen.cjs:774` (declared source revisions), `:781` (repository-qualified keys)@5f8b1da; `tests/gates/diagram-index.test.cjs` exists. **Partly stale**: citation checks can now fail the run under `--strict-citations` (`scripts/diagram-index-gen.cjs:24-25`@5f8b1da). |

## ADR-2005 section assessment

Classification: operative decision (accepted/complete/live), canon drift gate.

| Section | Disposition |
| --- | --- |
| Context | Divergent skill and ontology-bridge tool counts; blind grep would false-positive; truths live in agentbox. **Implements** (historical premise): `.github/workflows/drift-counter.yml:3-17`@5f8b1da; exclusions in `scripts/drift-counter/allowlist.json` denominators@5f8b1da. |
| Decision | Counter reads agentbox `skill-count-check.js` and the `ontology-bridge.js` `TOOLS` length, allowlist-anchored, fail-open per axis, `--strict` flips, agentbox checked out beside canon via `DRIFT_AGENTBOX_DIR`. **Implements**: `scripts/drift-counter/drift-counter.mjs:27-29,43,64`@5f8b1da; `allowlist.json:40` (TOOLS source), `:68,89` (unavailable/planned axes)@5f8b1da; `drift-counter.yml:57-62,77-80`@5f8b1da. Both truth files exist at the counter's pin `b859b37`; at agentbox's wave-2 pin `c57d9c0508f4` only `scripts/skill-count-check.js` exists and `mcp/servers/ontology-bridge.js` is **absent**. |
| Consequences | Allowlist is maintenance debt; ontology-classes and roster ship reported-only so status is partial; sibling rename shows UNAVAILABLE; README 7 vs 12 adjacency. **Implements** the axis states (`allowlist.json:68,89`@5f8b1da). The "partial" status is superseded by the 2026-09-05 annex (frontmatter `complete`). The README adjacency is moot: README numeric claims were retired (`allowlist.json:21,32`@5f8b1da). The bridge's removal from agentbox would now surface as an unavailable or failing source only when the pin moves (see finding). |
| Verification | At `cf535f8`: header/body, allowlist sites, `drift-counter.yml:50-55` checkout, `:19-23` UNAVAILABLE note. **Historical**; at 5f8b1da the note is still `drift-counter.yml:19-23`, the checkout now `:57-62`. |
| Closeout extension — 2026-09-04 | Stale archive paths, unpinned sibling, undefined denominators. **Superseded by later implementation** (next row). |
| Acceptance progress — 2026-09-05 | Moved targets repointed; `source_pin` at `89301ec7…` verified with `--allow-pin-drift` override; denominators and resolution blocks; seven figures corrected; ADR-004 excluded; 28-assertion test run before the counter. **Implements**: archived-path sites (`allowlist.json` mcp sites under `docs/archive/adr/ADR-002-…`), pin check (`drift-counter.mjs:65,95-119`), denominators and resolution blocks, `_files_note` excluding ADR Context, test-before-counter (`drift-counter.yml:72-80`)@5f8b1da. **Drift**: pin now `b859b37…` (`allowlist.json:4-7`, `drift-counter.yml:61`@5f8b1da); policed skill-scan files reduced to `docs/ecosystem-map.md`, `docs/PRD-website.md` (`allowlist.json:21,25`) since README and `index.html` dropped numeric claims on 2026-10-21. Test pass counts, "20 policed sites", live PASS **cannot be checked** from source. |

## ADR-2007 section assessment

Classification: operative decision (accepted, narrowed on 2026-10-02 to the evidence rule; the roadmap content is retained as dated history).

| Section | Disposition |
| --- | --- |
| Context | Historical motivation ("component completion labels do not establish cross-repository behaviour"). Cannot be checked from source: it describes a 2026-09-04 situation, not a mechanism. |
| Decision | **Implements** the machine-checkable half. `REQUIRED_FIELDS` requires `repo`, `verified_commit` and the three status axes (`scripts/adr-index-gen.cjs:28-32@5f8b1da`); `evidence_class` is validated against `{source,test,ci,deployment,runtime}` joined by `+` (`scripts/adr-index-gen.cjs:39-40,146-150@5f8b1da`). Whether the named evidence is true stays a review matter, as the record itself says. The "overtaken" roadmap clause is accurately scoped. |
| Consequences | Historical consequences of the original roadmap proposal, overtaken by the Acceptance annex. Cannot be checked from source. |
| Verification | **Implements** (artefact presence): the collector exists (`docs/estate-review/evidence/adr-inventory.py@5f8b1da`). The 2026-09-04 run on a dirty tree cannot be reproduced from source. |
| Closeout extension — 2026-09-04 | Historical; its CP-01–09 and six-journey bar are explicitly overtaken by "Acceptance — 2026-10-02". Cannot be checked. |
| Acceptance progress — 2026-09-05 | **Implements** (presence): `docs/estate-closeout/2026-09-05/` holds `gate-closeout-receipt.json` (11 gates, matching the text), `website-browser-receipt.json`, `release-manifest.local-draft.json`, `logs/` and nine screenshots. The receipts' figures are runtime records and were not re-executed. |
| Estate audit — 2026-09-07 | **Implements** (link targets): the audit, census, sprint and execution files all exist at `5f8b1da`; VisionClaw `docs/TODO-unified.md` exists at `86cdd76`. The policy statements are not mechanisms. |
| Closeout execution — 2026-09-07 | **Implements** (link target `docs/estate-review/2026-09-07-estate-closeout-execution.md@5f8b1da` exists); the runtime and browser receipts it summarises cannot be checked from source. |
| Disposition — 2026-10-02 | Historical triage. It cites `dreamlab-cumbria` planning files, which are not in the wave-2 pinned set, so they cannot be checked at a wave-2 pinned revision. |
| Acceptance — 2026-10-02 | **Partly contradicts.** `REQUIRED_FIELDS` and the status contract (`docs/architecture/adr-status-contract.md@5f8b1da`) are as described. "Runs on every push and pull request" overstates the gate: `adr-index.yml` triggers only on pushes to `main` and on PRs that touch `docs/adr/**`, the generator, the ratchet or the workflow itself (`.github/workflows/adr-index.yml:9-23@5f8b1da`). Also, the VF-01 `:24` citation is described as "the Decision paragraph", but line 24 is the `## Decision` heading; the paragraph is line 26. |
| Close-out 2026-10-31 | **Implements.** The date backstop is `EVIDENCE_CUTOVER = '2026-10-31'` (`scripts/adr-index-gen.cjs:39@5f8b1da`) and still binds nothing (the newest dated record is 2026-10-30). The transition gate is in `scripts/adr-ratchet.sh:80-109@5f8b1da`. Tests are registered (`tests/gates/run-all.sh:28-29@5f8b1da`). Its "six cases" for `adr-ratchet-evidence.test.sh` is now eight, after the wave-2 SIGPIPE cases (`tests/gates/adr-ratchet-evidence.test.sh:40-64@5f8b1da`); the field test has the stated seven (`tests/gates/adr-evidence-field.test.sh:32-38@5f8b1da`). |
| Close-out 2026-10-31 (wave 2): gate in all five ledgers | **Implements** at the pinned revisions, each of which descends from the merge the annex names. The ratchet evidence gate is in every repo: VisionClaw `scripts/adr-ratchet.sh:81-107@86cdd76`, agentbox `scripts/adr-ratchet.sh:81-107@c57d9c0`, forum `scripts/adr-ratchet.sh:81-107@c39f41f`, solid-pod-rs `scripts/adr-ratchet.sh:81-107@9b6143c`. CI wiring: VisionClaw `.github/workflows/docs-ci.yml`, agentbox `.github/workflows/invariants.yml:212-213`, forum and solid-pod-rs `.github/workflows/adr.yml`. Agentbox's `evidence_class` value test lives inside `tests/config/adr-index-check.test.mjs:133-142@c57d9c0`, not in a separate script. The SIGPIPE fix is present: awk drains its input (`scripts/adr-ratchet.sh:52-57@5f8b1da`). Hosted CI runs cannot be checked from source. |

## ADR-2009 section assessment

Classification: operative decision (accepted, complete, live).

| Section | Disposition |
| --- | --- |
| Context | **Implements** (record): the dream ledger shows four `webgl-mesh` nights with exactly the stated verdicts, 08-30 INCONCLUSIVE, 09-02 ACCEPT, 09-06 and 09-07 INCONCLUSIVE (`docs/dream-cycle/LEDGER.md:10,13,17,19@5f8b1da`). `website/static/js/mesh-webgl.js` exists and is not read by any evaluator (see Verification). |
| Decision | **Implements.** `dream.config.json` has four slots, `content-integrity`, `build-pipeline`, `seo-and-meta` and `estate-health`, with no `webgl-mesh` (`dream.config.json:4-27@5f8b1da`). The `browser-checks-out-of-annexe` discipline requiring `HANDOFF (browser):` notes is present (`dream.config.json:70@5f8b1da`). The operator handoff row records the parking (`docs/dream-cycle/LEDGER.md:22@5f8b1da`). |
| Consequences | **Implements** where it is checkable: the `shader-source` and `canvas-fallback` scan surfaces are absent from `dream.config.json@5f8b1da` (no match). No CI mesh-artefact collector exists (no `mesh`/`shader`/`noscript` collection in `scripts/estate-health.mjs` or the workflows at `5f8b1da`), so the reinstatement trigger has correctly not fired. The narrative consequences (nights saved, loss of coverage) cannot be checked. |
| Verification | **Implements** (source half): the slot list matches, and `git grep mesh-webgl` over `scripts/dream-*.sh` and `scripts/estate-health.mjs` returns nothing at `5f8b1da`. The evaluator entrypoints are the five named scripts (`dream.config.json:46-52@5f8b1da`). The evaluator outputs (`BUILD-OK`, the byte counts and so on) are run results at `2a369bf` and cannot be checked from source. |

## ADR-2010 section assessment

Classification: operative decision (accepted, complete, staged).

| Section | Disposition |
| --- | --- |
| Context | Historical premise dated 2026-09-14. The fabricated-rationale surfaces it cites have since been repaired, at forum (`"via governance UI"` is absent from crates; see `.claude/evidence/EXP-AC-002.client.evidence.md:49-53@c39f41f`) and at VisionClaw (`client/src/features/control-center/governance/AcspCaseQueue.tsx:16-24,62-64@86cdd76`). The original line citations describe an earlier tree, so this section cannot be checked as a statement about the current tree. |
| Decision | **Implements.** (1) The matrix carries the Augmentation conditions table with C1–C6 rows, one column per substrate, each graded with citations (`docs/architecture/compatibility-matrix.md:24-36@5f8b1da`). (2) CP-05 names C2/C3 and CP-07 names C4, with the exit bars as stated (`docs/estate-review/closeout/README.md:35,37@5f8b1da`). (3) Workflow-record field ownership is listed (`docs/architecture/compatibility-matrix.md:39@5f8b1da`). (4) No-fabrication: the forum gates the rationale with `check_rationale` and `MIN_HUMAN_RATIONALE_CHARS` (`crates/nostr-bbs-core/src/governance.rs:710,764@c39f41f`) and enforces it at the relay before save (`crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1360-1378@c39f41f`); VisionClaw renders the typed rationale (`AcspCaseQueue.tsx:57-64@86cdd76`). PRD-augmentation-conditions exists. |
| Consequences | **Partly contradicts.** It says "human in the loop" claims on the website must cite a `measured` row or be softened, but `website/static/index.html:8,12,22,39,123@5f8b1da` still asserts "human-in-the-loop governance" with no citation or qualifier. The "Augmentation Ratio provisional" consequence belongs to the VisionClaw KPI page and was not traced. The remaining items are cost and roadmap statements. |
| Verification | **Partly contradicts.** The matrix header grep and the CP-row grep succeed, and `.claude/evidence/EXP-AC-001.evidence.md@5f8b1da` exists. However, `scripts/check-citations.cjs` does not exist anywhere in the tree at `5f8b1da`. The checker that actually exists is `scripts/check-augmentation-citations.cjs@5f8b1da`, registered at `tests/gates/run-all.sh:24@5f8b1da`. That checker resolves against sibling working trees (`--root`) and checks only that each cited line exists, not that it supports the claim (`scripts/check-augmentation-citations.cjs:1-9,18@5f8b1da`). |
| Re-grade at merged mains (2026-09-15) | **Implements** in source: every cell cites code at merged mains, and the NIP-11 escalation block is still `"status": "scaffolded"` (`crates/nostr-bbs-relay-worker/src/nip11.rs:47@c39f41f`), which agrees with `activation_status: staged`. Stale text: it says `verified_commit` "is left `pending`", while the frontmatter carries `03db671…`, which resolves nowhere (see Findings). |

## ADR-2011 section assessment

Classification: operative decision (accepted, complete, staged).

| Section | Disposition |
| --- | --- |
| Context | Historical premise. The relay now enforces what it once only advertised (see Decision), and the cited `agentbox.toml:822-849` has moved to `agentbox.toml:1416-1460@c57d9c0`. Cannot be checked as a statement about the current tree. |
| Decision | **Implements items 1, 3 and 4; item 5 is partly contradicted.** (1) `TaskProperties` exists (`crates/nostr-bbs-core/src/governance.rs:390-397@c39f41f`). (2) `effective_tier` exists (`governance.rs:515-534@c39f41f`), taking a fourth argument, `advertised_default`. Its floors match the ADR: Irreversible or Critical gives High, Opaque gives Medium (`governance.rs:481-489@c39f41f`), and Opaque forbids suppression (`governance.rs:495-499@c39f41f`). (3) The relay folds an unlabelled request to `ESCALATION_DEFAULT_TIER` (`crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:375-380,2649-2654@c39f41f`) and enforces rationale on consequential 31403s (`nip_handlers.rs:1368-1376@c39f41f`). (4) Agentbox derives `zero-tolerance ⇒ irreversible`, anything else ⇒ compensable (`management-api/lib/task-properties.js:111-113@c57d9c0`), with defaults `partial`/`significant` (`task-properties.js:81-82@c57d9c0`; `agentbox.toml:1457-1459@c57d9c0`). **Contradiction:** item 5 says `risk_tier` "carries no authority on its own". When neither panel nor request carries a triple, `(Some(d), None) => d` lets the agent's declaration stand alone, even below the advertised default (`governance.rs:526-527@c39f41f`). |
| Consequences | **Implements.** For legacy panels, an unlabelled request folds to the advertised default rather than to `medium` (`governance.rs:523-525@c39f41f`; test `unlabelled_request_folds_to_advertised_default`, `governance.rs:3203@c39f41f`). The declared-versus-effective telemetry column was not traced. |
| Verification | **Implements** (tests present, not run): `effective_tier_table_holds_for_every_triple_and_tier` (`governance.rs:3158@c39f41f`), `merge_is_tightening_only_over_all_729_pairs` (`governance.rs:3092@c39f41f`), the unlabelled-default tests (`governance.rs:3203,3232@c39f41f`) and agentbox `tests/sovereign/task-properties.test.js:53-54@c57d9c0`. No relay-level integration test named for "a `high` case cannot reach `Decided` via any non-31403 path" was found by symbol search. That half is enforced in handler code, but no test of that name was located. |
| Merged at main, crates published (2026-09-15) | **Implements** in source. `effective_tier` is at `governance.rs:515` (cited `:516`, an off-by-one). The reviewer `Delegate` gate is in `nip_handlers.rs:200-250@c39f41f` and `reversibilityFor` at `task-properties.js:111-113@c57d9c0`. The NIP-11 block is still `scaffolded` (`nip11.rs:47@c39f41f`), which agrees with `staged`. Crate publication cannot be checked from source. The "`verified_commit` left `pending`" text is stale: the same unresolvable `03db671…` as ADR-2010. |
| Executed evidence (branch `feat/augmentation-conditions`, not merged, not deployed) | Historical branch evidence at `aa438f3`/`19463a588`, superseded by the merged annex above. The cited test line `governance.rs:2893` is now `:3158`. Cannot be checked at a wave-2 pinned revision. Its trailing "`implementation_status: partial` / `activation_status: inactive`" lines contradict the frontmatter, but they are scoped to that branch state. |

## ADR-2012 section assessment

Classification: operative decision, proposed (implementation partial, activation inactive).

| Section | Disposition |
| --- | --- |
| Context | Historical premise. Its citations no longer resolve to the claims: `README.md` at `5f8b1da` has 163 lines, so `:134`, `:240-256` and `:248` do not exist, and the payment-account claim is gone from README entirely. `website/static/index.html:597` is no longer the rail sentence. The sidestr-rs paragraph cannot be checked at a wave-2 pinned revision. |
| Decision | **Partly implements, partly contradicted.** D2 is implemented: the site says "No Lightning or NWC rail is built" and that L402/x402 are not payable (`website/static/index.html:631,1125@5f8b1da`), and the PRD carries the dated correction (`docs/PRD-website.md:20-30@5f8b1da`). D4 is implemented: the registry carries 3500, 23500, 23501, 23503, 23510-23514, 33333 (`e` tag), 33500, 33501 (legacy), 33502 (dual-schema), 30333, 3700 and 38420 (`docs/protocol/event-kind-registry.md:32-38,49,64-67,79@5f8b1da`), and the source-of-record rule is at `:8`. D5 is implemented on the site ("one supervised block producer", `index.html:631@5f8b1da`). D7 is implemented: every carrier resolves, agentbox ADR-2096–2103 at `c57d9c0`, solid-pod-rs ADR-2008 at `9b6143c`, VisionClaw ADR-2111 at `86cdd76` and forum ADR-2012 at `c39f41f`, and the anchoring rule is mirrored in registry §2.6 (`event-kind-registry.md:143-152@5f8b1da`). **D3 is contradicted**: the site still pins anchoring as "taproot-anchored commitments, on testnet4 by default and mainnet by explicit operator choice" (`index.html:631@5f8b1da`), the very sentence D3 retires. D1 is in tension (the decision is proposed, so this is not binding): the site still presents a sats Web Ledger, MRC20 tokens and DREAM alongside sidestr (`index.html:631@5f8b1da`). For D6, VisionClaw ADR-124 is now archived (`docs/archive/adr/ADR-124-smart-contract-features-web-contracts.md@86cdd76`); the §7 matrix was not traced. |
| Source qualification — 2026-10-01 | sidestr-rs is not in the wave-2 pinned set, so this section cannot be checked at a wave-2 pinned revision. The record names sidestr-rs `2bdee2d…` here and `bcbe30b…` under Verification, two different revisions for the same claim. |
| Consequences | **Implements** for the withdrawn public claims (as D2 above). The dependency and lockstep consequences are statements of risk and cannot be checked. |
| Verification | **Criteria partly unmet** (expected while the record is proposed). Met: the NWC/Lightning grep returns only withdrawal phrasing (`index.html:631,1125@5f8b1da`); the registry rows are present; the carriers resolve. Unmet or contradicted: (a) "the same external rows as agentbox `docs/PROTOCOL-registry.md`" fails, because agentbox carries the sidestr rows (`docs/PROTOCOL-registry.md:165-174@c57d9c0`) but no teller 30333 or 3700 row; (b) the `README.md:248` target no longer exists; (c) the compatibility matrix has no settlement row (no `sidestr` or `settlement` match in `docs/architecture/compatibility-matrix.md@5f8b1da`). |
| Disposition — 2026-10-02 | **Implements.** The D2 "Done" text matches the site and PRD at `5f8b1da`. The "55x GPU speedup" and "180+ MCP tools" figures are absent from `index.html@5f8b1da`. The D3/D4/D5/D7 revisions are present in the Decision text and the registry. |

## ADR-2013 section assessment

Classification: operative decision (accepted, partial, staged). It is partly overtaken by ADR-2014 without being amended.

| Section | Disposition |
|---|---|
| Context | Historical census (8,433, 8,146 and 4,167 classes; 23,900 `key:: value` lines). **Cannot be checked**: these are counts from a past date and nothing at the pinned revisions reproduces them. The residue it describes is now gone (see Consequences). |
| Decision | **Partly contradicts.** Implemented: (1) one vault, with visionGraph `knowledge/` and `working/` at the root (visionGraph `vault.toml:11-21`@556a015); (2) the ontology is held in frontmatter only. A sampled page carries typed properties, and `git grep` finds 0 files with a `json-ld` fence, a `key:: ` line, an `{{embed` or an `ancestors:` key in `knowledge/` or `working/`@556a015. `ontology/vocabulary.yaml` exists, and Quartz uses `Plugin.ExplicitPublish({ key: "public" })` (`quartz/quartz.config.ts:126`@556a015). (4) Forum `Promote`/`Demote` variants are at `crates/nostr-bbs-core/src/governance.rs:1469,1475`@c39f41f, and `stale_after` is at `crates/vault-core/src/okf.rs:8`@86cdd76. (5) The ontology-bridge MCP server is retired (agentbox `docs/adr/ADR-2108-ontology-bridge-retired-vault-baked-by-nix.md`; `config/entrypoint-unified.sh:2349`@c57d9c0). `ontology-propose` survives only as a non-MCP helper (`management-api/lib/ontology-propose.js:7`@c57d9c0). `loom-mcp-stdio` has no crate, and its deletion is recorded at Loom `docs/design/ADR-141-loom-consumes-the-vault-build.md:72`@ba21a68. Contradicted: (3) "`VisionClaw/crates/vault` is the only implementation". Ontomark is now the producer that VisionClaw (`Cargo.toml:43`, `src/services/corpus_source/local.rs:116`@86cdd76), Loom (`Cargo.toml:56`@ba21a68), agentbox (`flake.nix:956`@c57d9c0) and visionGraph (`README.md:65-70`@556a015) consume. VisionClaw still carries `crates/vault`, `crates/vault-core` and its own whelk-rs fork (`Cargo.toml:215`@86cdd76). Also (3) "GitHub pull removed": `GitHubSource` is retained as the fallback when `VAULT_ROOT` is unset (`src/services/corpus_source/mod.rs:16-21`@86cdd76). (6) Class-count equality is a runtime claim and **cannot be checked** from source. |
| Consequences | **Implements, with one residue.** `pipeline/` is absent from visionGraph@556a015. knowledgeGraph@4ed9ac1 has no `pipeline/` or `ontology/` tree, only `archive/`. VisionClaw has no `crates/vault-migrate`@86cdd76. Logseq syntax is removed from both vaults (counts above). Residue: `publishing-tools/` remains in visionGraph as the WasmVOWL explorer (`README.md:32`@556a015), where the Consequences list it as deleted. The frozen seam file exists (`docs/engineering/sovereign-corpus-contracts.md`@5f8b1da). |
| Verification | **Implements**, for the canon-prose claims. At 5f8b1da no current-tense "Logseq corpus" claim remains in README, pitch or terminology. The website mention is explicitly historical (`website/static/index.html:400`). `docs/ecosystem-map.md:67` names the local Obsidian vault, `docs/BASELINE-visionflow.md:327` carries invariant 8, and `scripts/adr-index-gen.cjs` exists. The load-bearing acceptance 2 (equal class counts) is runtime and **cannot be checked**. |

## ADR-2014 section assessment

Classification: operative decision (accepted, partial, staged).

| Section | Disposition |
|---|---|
| Context | Owner intent and investigation scope. **Cannot be checked** from source. |
| Decision | **Implements** the ownership boundary. The Ontomark workspace holds `ontomark-{cli,compile,ingest,jev,model,ngg1,parse,reason,source,wasm}`, `whelk-el` and `packages/graph`@4b9d15d. Horned-OWL is an optional upstream dependency (`crates/whelk-el/Cargo.toml:25`). Consumers build from released crates: VisionClaw `ontomark-compile` 0.6.3 (`Cargo.toml:43`@86cdd76), Loom `=0.1.1` (`Cargo.toml:56`@ba21a68), agentbox CLI (`flake.nix:956`@c57d9c0). Removal of inventoried duplicates is **not done**: VisionClaw `crates/vault` and `crates/vault-core` remain, with their own Whelk (`Cargo.toml:215`@86cdd76). This is consistent with `partial`. |
| Consequences and tensions | **Implements.** Workspace licence `MIT OR Apache-2.0` (`Cargo.toml:23`@4b9d15d). The NOTICE relicensing grant excludes `vault/src/nostr.rs` and solid-pod/sidestr lineages (`NOTICE`@4b9d15d). The WASM-optional native use is visible in VisionClaw's `default-features = false` dependency. |
| Verification and delivery | `visionGraph/planning/ontology-workspace/README.md` exists@556a015. The acceptance gates (clean builds, parity, installed plugin tests, removal of duplicates) are build and runtime evidence and **cannot be checked** from source. The duplicates are observably still present (above). |
| Implementation continuation — 2026-10-11 | Cites Ontomark `4abbce9`, `671b6be` and `7f5b2bb`, and ontomark-obsidian `ebf8fd6`, none of which is a wave-2 pin. The 32-hash comparison is a run receipt. **Cannot be checked** at a wave-2 pinned revision. The version figures (0.1.x) are historical: the workspace is 0.7.0 (`Cargo.toml:20`@4b9d15d). |
| Estate alignment — 2026-10-12 | **Implements, as recorded decisions with open gates.** (3) A machine bundle exists (`crates/ontomark-cli/src/main.rs:207`@4b9d15d). (5) The NOTICE grant covers the vault ports. Still open, as the section says: (1) VisionClaw's own Whelk derivation (`Cargo.toml:215`@86cdd76); (2) `src/services/ontology_pull.rs` still exists@86cdd76. ADR-OM-005 and `docs/DEPRECATION.md` exist@4b9d15d. |

## ADR-2015 section assessment

Classification: operative decision. Its front matter says `proposed / none / inactive`, but code implements it.

| Section | Disposition |
|---|---|
| Context | Historical defect inventory. **Cannot be checked** against the pre-repair code. Ontomark labels its repairs with these defect numbers (`crates/ontomark-model/src/kind.rs:5-7`, `model.rs:60-62`@4b9d15d), which is consistent with the Context. |
| Proposed decision | **Implemented in Ontomark, contradicting `implementation_status: none`.** Explicit kinds, including the three OWL property kinds, are at `crates/ontomark-model/src/kind.rs:10-30`. Resolution is resolved, ambiguous, unresolved or external (`model.rs:65-72`). Construct dispositions include `PreservedNotReasoned` and `Rejected` (`model.rs:153-160`). Thirteen code references to ADR-2015 are under `crates/`@4b9d15d. The reasoner capability matrix and the public-before-inference ordering were not traced individually. |
| Migration | "No corpus migration has been run under this proposal" and the migration artefact contract. **Cannot be checked** from source: no migration receipt exists at a pinned revision. |
| Consequences and verification | Standards stance and "no corpus migration". **Cannot be checked** from source. |

## ADR-2016 section assessment

Classification: operative decision (accepted, partial, staged). The plugin lives in ontomark-obsidian, which has no wave-2 pin.

| Section | Disposition |
|---|---|
| Context | Owner requirements. **Cannot be checked**. |
| Decision | The plugin's behaviour belongs to ontomark-obsidian: **cannot be checked** at a wave-2 pinned revision. The shared parts it relies on exist in Ontomark (`packages/graph`, `crates/ontomark-wasm`@4b9d15d). |
| Consequences and tensions | Design stance on mobile and third-party patterns. **Cannot be checked**. |
| Verification | Installed-plugin tests. **Cannot be checked** at a wave-2 pinned revision (ontomark-obsidian). |
| Desktop implementation continuation — 2026-10-11 | Cites ontomark-obsidian `ebf8fd6`. **Cannot be checked** at a wave-2 pinned revision. |
| Assisted ingest core continuation — 2026-10-12 | **Implements** the core claims. `plan` and `accept` are at `crates/ontomark-ingest/src/lib.rs:220,233`@4b9d15d. The crate's dependency list has no I/O or network crate (`crates/ontomark-ingest/Cargo.toml`). Spans are given in UTF-16 units (`src/accept.rs:27,45`). OntoCast attribution is at `crates/ontomark-ingest/NOTICE:10`. The engine boundary exposes `ingest_plan`/`ingest_accept` (`crates/ontomark-wasm/src/engine.rs:733,753`), and the engine profile implements `ontomark_model::Profile` (`crates/ontomark-wasm/src/profile.rs:255`). ADR-OM-003 and ADR-OM-004 exist. The named entity-parity test was not located by name, so that claim is not established here. The plugin-side gates **cannot be checked**. |

## ADR-2018 section assessment

Classification: operative decision (accepted, partial, inactive). Bullet 4 is superseded by ADR-2019.

| Section | Disposition |
|---|---|
| Context | Historical snapshot (alpha.15 released; forge unreleased). **Cannot be checked** as of its date, and stale now: the solid-pod-rs workspace is `0.5.0-alpha.18` (`Cargo.toml:15`@9b6143c). |
| Decision | **Contradicted on bullet 1.** Canon quotes "99.2% strict over 264 rows" (`docs/architecture/compatibility-matrix.md:16`@5f8b1da). The checklist the rule names as the source now says "86.4% strict over 300 rows" (`crates/solid-pod-rs/PARITY-CHECKLIST.md:58`@9b6143c). Bullet 3 is implemented: VisionClaw takes caret ranges without forge features (`Cargo.toml:246-249`@86cdd76). Bullet 5 is implemented: VisionClaw links the library with `default-features = false`, and agentbox builds the server (`flake.nix:1538-1558`@c57d9c0). Bullet 4 is superseded (see ADR-2019). |
| Consequences | **Implements (historical).** The two 97.7% diagram nodes are gone. Topics SP-09 and SP-10 now carry 86.4% (`docs/diagrams/solid-pod-rs/09-ci-release-and-consumer-pins.md:281`, `10-failure-modes-and-invariants.md:230`@5f8b1da). The forum comparator line is present (nostr-rust-forum `docs/adr/ADR-2015-member-wallets-on-sidestr-dreamlab-and-dream-tips.md:23`@c39f41f). |
| Verification | Rests on uncommitted solid-pod-rs work at `f382039` and a script run. **Cannot be checked** at a pinned revision. solid-pod-rs ADR-2009 exists (`crates/solid-pod-rs/docs/adr/ADR-2009-forge-plugins-comparator-and-default-off-mirror.md`@9b6143c). |

## ADR-2019 section assessment

Classification: operative decision (accepted, partial, live).

| Section | Disposition |
|---|---|
| Context | **Implements.** The forge features exist (`crates/solid-pod-rs-server/Cargo.toml:165-172`@9b6143c). solid-pod-rs ADR-2010 exists. VisionClaw embeds the library only (`Cargo.toml:246-249`@86cdd76). |
| Decision | **Partly contradicts.** Implemented: `forge-full` is a default in agentbox (`flake.nix:1548`@c57d9c0). `open_edit`, `allow_mainnet` and `private_repos` all default to 0 (`flake.nix:1593-1595`). There is no external follow entry by default, and VisionClaw has no `/forge` route@86cdd76. Contradicted: "the shipped follow list is the operator's own key". The shipped default is `mirror_follow = []` (`agentbox.toml:621`@c57d9c0), and the mirror needs both relays and follow entries (`crates/solid-pod-rs-forge/src/mirror.rs:357-358`@9b6143c). |
| Consequences | **Contradicted in canon.** The matrix Pod row still says forge is "unreleased and built by no consumer yet" with the 264-row figure (`docs/architecture/compatibility-matrix.md:16`@5f8b1da; see the ADR-2018 finding). |
| Verification | "Nothing is released or built by a consumer yet" is historical, at base `4027542` / `8dd5dc1`, and superseded by the close-out. **Cannot be checked** at a pinned revision. |
| Close-out 2026-10-30 | **Implements**, for the source parts. Release `0.5.0-alpha.18` (`Cargo.toml:15`@9b6143c). Agentbox pin `lib/solid-pod-rs.nix:38`@c57d9c0. Feature default `flake.nix:1548`. `mirror_config` returns `None` without follow entries (`crates/solid-pod-rs-server/src/lib.rs:4978-4994` with `mirror.rs:357-358`@9b6143c). The mirror starts only on `Some` (`lib.rs:5034-5040`). The cited lines `4965-4981` and `5022-5027` belong to `76ab7aab` and are shifted at the pinned revision. Log and HTTP observations are runtime and **cannot be checked**. |

## ADR-2020 section assessment

Classification: operative decision (accepted, complete, live). It supersedes ADR-2004.

| Section | Disposition |
|---|---|
| Context | Historical measurements (1,490 of 1,495 blocks). **Cannot be checked** from source. |
| Decision | **Implements.** `scripts/vendor/pele/VERSION` pins 1.12.0 with integrity and sha256. `scripts/lib/pele-static.cjs:29-32` verifies the sha256 before import, `flatten` is at `:38`, and `GATE_OPTIONS = { responsive: false, autoDirection: false }` is at `:21` (all @5f8b1da). The other surfaces agree: agentbox `lib/pele.nix:19`@c57d9c0, dreamlab-ai-website `package.json:74`@0e15728, visionGraph `quartz/quartz/components/scripts/mermaid.inline.ts:143,152`@556a015. |
| Consequences | **Implements, with one internal inconsistency.** The hosted render gate is at `.github/workflows/diagram-index.yml:65`, and there is no browser in the corpus path (`scripts/diagram-index-gen.cjs:683`). Agentbox's `mmdc` is a Pele shim (`lib/pele.nix:66-69`@c57d9c0). The follow-on bullet says it is "tracked by `implementation_status: partial`", while the front matter says `complete`. |
| Verification | Run evidence at `1b0dd42`. **Cannot be checked** from source. The current gate run is the lead's. |
| Close-out 2026-10-31 | **Implements.** `scripts/check-pele-pin.cjs` and `tests/gates/pele-pin.test.sh` exist, and the test is registered at `tests/gates/run-all.sh:30`. The pin surfaces agree at the pinned revisions (above). The checker is absent from `.github/workflows` (no match), which matches the stated boundary. |

## ADR-2021 section assessment

Classification: operative decision, proposed (none, inactive).

| Section | Disposition |
|---|---|
| Context | **Implements** for the licence inputs. AGPL-3.0-only: VisionClaw `Cargo.toml:37`@86cdd76, solid-pod-rs `Cargo.toml:17`@9b6143c, forum `Cargo.toml:64`@c39f41f, agentbox `LICENSE`@c57d9c0. MIT OR Apache-2.0: Loom `Cargo.toml:25`@ba21a68, Ontomark `Cargo.toml:23`@4b9d15d. The 114-day latency is at `docs/business/roadmap-2026-11-to-2028-04.md:33,102`@5f8b1da. sidestr-rs, dream-engine, the Spawn account and the regulatory claims **cannot be checked** at a wave-2 pinned revision. |
| Decision | Proposal. The roadmap exists, with §4 at `roadmap-2026-11-to-2028-04.md:117`. The lineage ADRs exist (agentbox ADR-2085, 2115, 2139, 2153@c57d9c0), and so does the precedent-retirement commit `70d017a3b`. No licence register or gate was found in agentbox, which is consistent with `none`. Otherwise **cannot be checked** (unimplemented). |
| Consequences | Business consequences. **Cannot be checked** from source. |
| Verification | **Implements.** The index row at `docs/adr/README.md:53`@5f8b1da matches proposed/none/inactive, and the cited park register exists (`docs/registers/park-2026-10-30.md`). |

## ADR-004 (engineering) section assessment

Classification: operative decision (Accepted, D3 Deferred), VisionFlow `docs/engineering/ADR-004-harness-engineering-framework.md` at worktree HEAD `5f8b1da`. Source tracing against VisionFlow `5f8b1da`, agentbox `c57d9c0508f4`, VisionClaw `86cdd7635582`, nostr-rust-forum `c39f41f104b1`.

| Section | Disposition |
| --- | --- |
| 2026-07-03 closeout amendment (preamble blockquote, not a heading) | Implements. `handleGovernanceDecision()` exists and mints PROV-O activity/receipt URNs (agentbox `management-api/adapters/orchestrator/local-process-manager.js:142`, `:173`–`:197`@c57d9c0). No agent spec wires `hooks.validate` (no match under `agents/` at c57d9c0); both dependent sensors stay `planned` (`docs/engineering/templates/agentbox-agent-task.json:82`–`:85`, `:109`@5f8b1da). |
| Context | Dated situation statement. The three gaps are consistent with source: no `hooks.validate` exists (above). Counts (110 specs, 115 skills, 139 commands) are dated and not re-counted; the ADR itself says the sourced skill count is now 126. Cannot be checked as current figures. |
| Decision | Heading only; see D1–D8. |
| D1: Adopt the guide/sensor/pairing model | Implements. The compatibility matrix carries a harness coverage table with paired and source-backed columns (`docs/architecture/compatibility-matrix.md:81`–`:97`@5f8b1da). The table is labelled historical (2026-07-03, 33/40) and was not regenerated after the 2026-09-05 audit fix, which reports 31/39. |
| D2: Harness templates are the unit of agent governance | Implements. The schema requires `version`, `topology`, `structure`, `guides`, `sensors`, `pairings` and defines `escalation_rules` (`docs/engineering/schemas/harness-template.schema.json:7`, `:67`@5f8b1da); four templates exist under `docs/engineering/templates/`. "Inspectable via MCP tools" (harness-bridge) cannot be checked at a wave-2 pinned revision. |
| D3: Three-phase agent execution lifecycle | Not implemented, as the section's own Deferred status says. Accurate. |
| D4: Computational sensors are blocking; inferential sensors are advisory | Partly implements. Sensor `type` enum `computational`/`inferential` and `validation_mode` `blocking`/`advisory`/`learning` exist (`harness-template.schema.json:145`, `:189`@5f8b1da). The CI audit blocks only below 50% (`.github/workflows/harness-fitness-gates.yml:56`, `:62`–`:65`@5f8b1da), not at the script's 80% default (`scripts/harness-audit.sh:57`@5f8b1da). Whether the job is a required merge check cannot be checked from source. |
| D5: Cross-substrate templates reference the mesh smoke test | Implements. `governance-decision.json:76`–`:78` names `VisionFlow:docs/protocol/mesh-smoke-test.md`, paired at `:143`, sensor source `scripts/mesh-smoke-preflight.sh` (`:124`@5f8b1da); both files exist. |
| D6: Precedents are feedforward controls generated from feedback | Contradicted. The precedent system was retired in agentbox (CHANGELOG `CHANGELOG.md:743`–`:750`@c57d9c0: precedent service and bridge removed, namespace empty); only the write-protection survives (`mcp/servers/lib/memory-tools.js:203`–`:218`@c57d9c0). The template marks the guide `RETIRED 2026-09-14`, `source_status: planned` (`governance-decision.json:68`–`:73`@5f8b1da). A `Precedent` outcome variant exists in VisionClaw (`src/domain/broker/broker_decision.rs:42`@86cdd76) and the forum (`crates/nostr-bbs-core/src/governance.rs:1478`@c39f41f), but no auto-apply path. The ADR text still states D6 as accepted. |
| D7: Ashby's Law — topology commitment narrows the regulation space | Implements as a schema obligation: every template declares `structure.substrates` (`harness-template.schema.json:28`–`:31`@5f8b1da). The law-based reasoning itself cannot be checked from source. |
| D8: Harness templates are owned by VisionFlow, not substrates | Implements. Schema and templates live only in VisionFlow (`docs/engineering/schemas/`, `docs/engineering/templates/`@5f8b1da). |
| Consequences | Heading only. |
| Positive | Mixed. Visible coverage and CI gating exist (D1, D4). "Reduced steering burden" through precedents is contradicted (D6). "Machine-readable governance via MCP" cannot be checked at a pinned revision. |
| Negative | Janitor automation (FR7.1) and validation overhead cannot be checked from source; validation overhead does not arise because D3 is not built. |
| Neutral | Contradicted in part: no agent template format has a `hooks.validate` field (D3 deferred). The matrix columns were added (D1). |
| Alternatives Considered | Heading only. |
| A1: Informal harness accumulation (status quo) | The rejection is sound, but the quoted figure (33/40, 82.5%) is the pre-fix figure. The same ADR's acceptance annex supersedes it with 31/39 (79.5%). Internal staleness. |
| A2: Centralised harness service | Rejected alternative. Nothing in source builds one. Consistent. |
| A3: LLM-only inferential validation | Rejected alternative. Consistent with the D4 schema split. |
| Closeout extension — 2026-09-05 | Historical record of the probe findings. The defects it names are the ones repaired in the next section. Accurate as history. |
| Acceptance progress — 2026-09-05 | Implements. Source resolution into five states (`scripts/harness-audit.sh:129`–`:196`@5f8b1da), distinct-edge coverage bounded by 100% (`:274`–`:282`), and `--strict-sources` (`:96`). The self-test runs before the audit in CI (`harness-fitness-gates.yml:44`@5f8b1da). Schema validation is still done by hand with jq rather than a JSON Schema validator (`harness-fitness-gates.yml` fixture-parity job), as stated. Two stale points: (1) the precedent guide is now `planned`, not the `descriptor` the annex reports (`governance-decision.json:70`–`:71`@5f8b1da); (2) "ADR-005 §Decision 2 partial-source failure mode" misattributes the pattern, because this repository's ADR-005 D2 is the mandate inventory (see ADR-005 below). The "26 assertions" figure was not recounted. |

## ADR-005 (engineering) section assessment

Classification: speculative decision (no implementation committed), VisionFlow `docs/engineering/ADR-005-mandate-at-grant-governance.md` at `5f8b1da`. Dispositions for D1–D9 assess whether the code contradicts the claim that the mandate model is unbuilt. Context claims are checked against agentbox `c57d9c0508f4`.

| Section | Disposition |
| --- | --- |
| 2026-07-03 closeout amendment (preamble blockquote) | Partly true. `governance-precedents` is write-protected with the quoted error, but at `mcp/servers/lib/memory-tools.js:203`–`:218`@c57d9c0, not `:60`–`:68`. Stale line citation. |
| Context | Partly stale. Still true: the file-channel `createMandate` defaults `expiresAt` to null and issues kind 30078 (`management-api/lib/mandate.js:36`, `:96`–`:108`@c57d9c0); WoT and payment surfaces exist (`management-api/middleware/linked-data/surfaces/s06-wot.js`, `s08-payments.js`@c57d9c0); the schema has no `mandates` section (no match in `harness-template.schema.json`@5f8b1da). Contradicted: `agent_event_auth` now defaults to `"nip98"` (`agentbox.toml:186`@c57d9c0), not `'off'`; `spawnAgent` no longer spreads `process.env` but passes an allowlist (`local-process-manager.js:51`, `:594`–`:606`@c57d9c0); ProcessManager replaces the env spread with an explicit allowlist, tagged "IR3 (mandate-at-grant)" (`management-api/utils/process-manager.js:216`–`:223`@c57d9c0). `signerFromHex` is no longer in `per-user-agent.js` (now `junkiejarvis-agent.js:612`). So "the remaining ungoverned grants are the account, file, and environment-secret channels" is stale for the env-secret channel. |
| Decision | Heading only; see D1–D9. Not implemented, consistent with Speculative status. |
| D1: Mandate-at-grant is a left-shift of the governance model | Not implemented. No 31406 in `agentbox.toml` `allowed_kinds` (`:173`@c57d9c0). The forum's governance parser rejects 31406 as an unknown kind (`crates/nostr-bbs-core/src/governance.rs:2026`, `:2050`–`:2053`@c39f41f). Consistent with status. |
| D2: Three-channel mandate inventory in harness templates | Not implemented: no `mandates` property in `docs/engineering/schemas/harness-template.schema.json`@5f8b1da. Consistent. |
| D3: Capability composition is modelled as a directed graph, not a flat list | Not implemented: no ToolNode or CompositionEdge artefact found in agentbox@c57d9c0 or VisionFlow@5f8b1da. Consistent. |
| D4: Tool classification taxonomy | Not implemented as a taxonomy. One example tool, `precedent_promote`, no longer exists, because the precedent bridge was removed (agentbox `CHANGELOG.md:743`–`:746`@c57d9c0). Example list stale. |
| D5: Mandate-time governance event (kind 31406 MandateGrant) | Not implemented (see D1). The S6/S8 surfaces it would reference exist (Context). Consistent. |
| D6: Mandate routing follows the ADR-122 epistemic pattern | Not implemented: no `governance-mandates` namespace in agentbox@c57d9c0. ADR-122 is a VisionClaw archive ADR; its lane routing was not re-traced. M2's reliance on precedents inherits the retirement noted under ADR-004 D6. |
| D7: Mandate sensors extend the guide/sensor pairing model | Not implemented. It also depends on ADR-004 D3, which is Deferred. Consistent. |
| D8: Blast radius is Ashby-bounded by topology commitment | Not implemented. Cannot be checked beyond the absence of mandates. |
| D9: ActionRequest carries current mandates | Not implemented: no `current_mandates` in agentbox@c57d9c0 outside archive. Consistent. |
| Consequences | Heading only. |
| Positive | Projected benefits of an unbuilt design. Cannot be checked. "Precedent efficiency" depends on a retired precedent system. |
| Negative | Projected costs. Cannot be checked. |
| Neutral | Not implemented: relay kind gates do not accept 31406 (forum `governance.rs:2053`@c39f41f). Consistent with "needs updating". |
| Alternatives Considered | Heading only. |
| A1: Action-time-only governance (status quo) | Status quo accurately described: 31402/31403 flow with `handleGovernanceDecision` (agentbox `local-process-manager.js:142`@c57d9c0). |
| A2: Zero-trust per-call verification | Rejected alternative. Nothing built. Consistent. |
| A3: Static capability configuration (no governance events) | Partly the de facto state: `ALLOWED_TOOLS` and the env allowlist are static configuration without governance events (`process-manager.js:180`, `:216`@c57d9c0). The ADR rejects A3 while the code has adopted its hardening half; record as **Tension**. |
| A4: Embed mandates in DID documents only | Rejected as sole mechanism. Not checked further. |
| Implementation Notes (Speculative) | Phase 1 is not started (no `mandates` schema, no `mandate_blast_radius` tool in agentbox@c57d9c0). Consistent. |
| Closeout extension — 2026-09-05 | Historical. Shared text with ADR-004; accurate as history. |
| Acceptance progress — 2026-09-05 | Accurate that nothing was built: 31406 absent, no `mandates`. It says write-protection was "not re-verified"; this pass verifies it at `memory-tools.js:203`–`:218`@c57d9c0. The claim "§Decision 2 partial-source failure mode" in this record is wrong: D2 here is the three-channel inventory, and the partial-source pattern is not in this ADR. |

## visionGraph ADR-008 section assessment

Classification: not a decision record and not an imported ADR copy. `knowledge/ADR-008.md`@556a015 is a generated ontology concept page (`type: Class`, `resource: urn:ngm:class:adr-008`, generated by `agent:enrichment-swarm/6` via `process:vault-migrate/1.0`, lines 2–13). It defines "an architecture decision record numbered 008" generically, with no project, decision or code claim. visionGraph's own decisions live under `docs/adr/ADR-VG-*`.

| Section | Disposition |
| --- | --- |
| Semantic Classification | Empty section (heading only, `knowledge/ADR-008.md:27`@556a015). Nothing to check. |
| Content | Generic definition of an ADR (`:29`–`:32`). True as general prose. It makes no code claim and names no repository whose eighth ADR it means. |
| Provenance | Empty section (`:34`). The frontmatter `generated` block carries the only provenance. Nothing to check. |

## visionGraph ADR-012 section assessment

Classification: as ADR-008. A generated ontology concept page (`knowledge/ADR-012.md:2`–`:13`@556a015), not a decision record or an imported copy. The content is the same template with "twelfth" substituted.

| Section | Disposition |
| --- | --- |
| Semantic Classification | Empty (`knowledge/ADR-012.md:27`@556a015). |
| Content | Generic ADR definition (`:29`–`:32`). No code claim; ambiguous referent. |
| Provenance | Empty (`:34`). |

Findings: see the shared finding under ADR-008.

## WasmVOWL ADR-001 section assessment

Classification: proposed, partial, inactive decision (`docs/adr/ADR-001-explorer-consumer-contract-closeout.md:5`–`:7`@51a1484), with dated closeout annexes. Its frontmatter `verified_commit` is `36105cc`, which predates the pinned `51a1484`.

| Section | Disposition |
| --- | --- |
| Context | Partly checkable. Still implemented in source: the standalone hook sends `{nodes, edges}` to `loadOntology` (`modern/src/hooks/useWasmSimulation.ts:91`–`:105`@51a1484). Map mutation through immer without the MapSet plugin: the store wraps `nodes: Map` in `immer` (`modern/src/stores/useGraphStore.ts:6`, `:61`, `:154`–`:158`), and no `enableMapSet` appears anywhere under `modern/`. This is consistent with the claimed failure, but confirming it needs runtime. The parser's `class`/`classes` requirement lives in the external `vowl-wasm` release, which cannot be checked at a wave-2 pinned revision. Test counts (47 native; 19 passed, 60 failed) are dated and were not rerun. |
| Proposed decision | Not implemented, consistent with the proposed/inactive status: no versioned consumer schema or release-manifest binding is found in the tree. The owner is still `explorer-maintainer-role-unassigned` (`:8`). |
| Closeout extension — 2026-09-04 | Acceptance obligations are open in source. Topology reset is a live hazard: the load effect depends on `nodes` (`useWasmSimulation.ts:120`), and each tick writes positions into `nodes` through `updateNodePosition` (`:151`–`:154`; `useGraphStore.ts:154`). If Map drafting worked, that would change the dependency and re-run `loadOntology` and `initSimulation` (`:105`–`:107`). Browser, keyboard and performance acceptance cannot be checked from source. |
| Estate execution scope — 2026-09-07 | Implements as described: the modern explorer consumes the published `@dreamlab-ai/vowl-wasm` package (`modern/package.json:19`@51a1484). The publisher explorer is in another repository and cannot be checked here. |
| Remote integration qualification — 2026-09-07 | Implements. Commit `b359aa0` is an ancestor of the pin ("consume the published vowl-wasm 0.1.1 engine, delete the vendored crate"). The v0.1.1 archive is installed (`modern/package.json:19`). No Rust crate remains in the tree. Stable `getNodeIds()` and interleaved positions accessors exist (`useWasmSimulation.ts:23`–`:24`, `:106`). The hook still sends `nodes`/`edges` (`:91`–`:96`). |

## Panel dispositions

None: this assignment covers ADRs only.

## Verification

- **Evidence class:** source tracing at the pinned revisions. No product tests, CI runs, browser sessions or deployments were exercised, and nothing here is promoted to tested, runtime-verified or deployed.
- **Revisions read:**
  - VisionFlow: worktree HEAD `5f8b1da`, read with `git show HEAD:` / `git grep HEAD`.
  - Product repositories, at their wave-2 pins: agentbox `c57d9c0508f4` (plus `b859b37`, the drift counter's own pin and an ancestor of it), VisionClaw `86cdd7635582`, nostr-rust-forum `c39f41f104b1`, solid-pod-rs `9b6143ce55b2`, ontomark `4b9d15df4805`, visionGraph `556a0156b5c3`, WasmVOWL `51a148430190`, loom `ba21a689a57d`, knowledgeGraph `4ed9ac159daf` and dreamlab-ai-website `0e1572801d03`.
  - Repositories without a wave-2 pin (sidestr-rs, ontomark-obsidian, Pele) are recorded as "cannot be checked at a wave-2 pinned revision".
- **Generator commands:**
  - `node scripts/adr-index-gen.cjs <exported copy of docs/adr at 5f8b1da> --check` exited 0 with 21 records valid, and a regeneration matches the committed `docs/adr/README.md` apart from the echoed path argument.
  - The diagram generator was not run, because this assignment edits no topic.
- **Coverage check:** a script compared every `##`/`###`/`####` heading in each assessed record against the first column of its table; no heading is missing.
- **Spot checks:** this reviewer re-read seven findings at the pinned revisions and all held:
  - `ontology-bridge.js` is absent at `c57d9c0` and present at `b859b37`;
  - `governance.rs:526-527`@c39f41f has the `(Some(d), None) => d` arm;
  - `scripts/check-citations.cjs` does not exist;
  - `GitHubSource` is still selectable;
  - two `type="module"` entries are at `website/static/index.html:1300-1301`;
  - 99.2%/264 rows is at `compatibility-matrix.md:16`;
  - the mainnet/testnet4 anchoring sentence is at `index.html:631`.
- **Not verified:** CI and Pages behaviour; the Chrome or Pele render runs; test counts quoted in annexes; any claim resting on sidestr-rs, ontomark-obsidian or Pele source; whether the WasmVOWL topology-reset hazard manifests at runtime (it is inferred from the effect dependencies only).
