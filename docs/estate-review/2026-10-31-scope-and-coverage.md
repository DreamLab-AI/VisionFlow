---
title: Current estate scope and section review queue
status: in-progress
date: 2026-10-31
type: reference
---

# Current estate scope and section review queue

Observed against VisionFlow `9c9aed1`. This supersedes the September scope table for checkout availability, not its historical analysis. The census starts with the 18 entries in `scripts/estate-health/roster.json`, adds the supporting/history entries in `docs/architecture/repository-map.md`, and includes the nested Unmute source explicitly cited by VC-35. Codescape's standalone and consumed checkouts are separate revision observations of one project. Agentbox and Logseq aliases are counted once. Unrelated workspace projects were not scanned.

[Machine-readable evidence](evidence/2026-10-31/scope-adr-diagram-census.json) records full HEADs, working-tree status, ADR file hashes, declared fields and heading line numbers, plus every diagram topic's sources **and** governing files. Discovery uses Git tracked plus nonignored untracked paths, restricted to ADR-like filenames and `adr`/`adrs` directories. Nested Git owners are excluded from parent inventories. Markdown heading detection is lexical and can include examples inside fences; every candidate and heading awaits semantic classification. Counts are neither implemented-decision counts nor review percentages. This is an observation over changing checkouts, not an atomic estate snapshot.

| Checkout | Workspace path | Observed HEAD | ADR-like files |
|---|---|---|---:|
| VisionFlow | `VisionFlow` | `9c9aed1544e8` | 39 |
| VisionClaw | `project` | `75b72416bac1` | 283 |
| agentbox | `project/agentbox` | `9dfc27b16e82` | 239 |
| solid-pod-rs | `solid-pod-rs` | `83cdda68bc58` | 19 |
| nostr-rust-forum | `nostr-rust-forum` | `c39f41f104b1` | 64 |
| dreamlab-ai-website | `dreamlab-ai-website` | `b765a8873ee7` | 47 |
| loom | `loom` | `ba21a689a57d` | 10 |
| knowledgeGraph | `knowledgeGraph` | `4ed9ac159daf` | 9 |
| sidestr-rs | `sidestr-rs` | `d6d0a7ed72e6` | 5 |
| ontomark | `ontomark` | `0caae5d4b695` | 12 |
| ontomark-obsidian | `ontomark-obsidian` | `272259edcda1` | 5 |
| WasmVOWL | `WasmVOWL` | `51a148430190` | 1 |
| visionGraph | `visionGraph` | `556a0156b5c3` | 12 |
| vowl-wasm | `vowl-wasm` | `28af4365a68d` | 2 |
| prose-sanitiser | `prose-sanitiser` | `829f8e33f800` | 0 |
| diagram-ir | `diagram-ir` | `15cba163da70` | 0 |
| dream-engine | `dream-engine` | `815f963b1cbd` | 10 |
| visiontouch-rs | `visiontouch-rs` | `7b95fb20d50a` | 9 |
| dream-machine | `dream-machine` | `b77093e564ef` | 9 |
| Logseq history | `project4` | `4f233f321097` | 10 |
| RuVector | `ruvector` | missing | unavailable |
| RuView | `RuView` | `5728b90949e6` | 336 |
| Codescape standalone | `codescape` | `58108c0357bd` | 4 |
| Codescape consumed | `project/codescape` | `e3b76f63e801` | 4 |
| unmute consumed | `project/voice-stack/unmute` | `e348e567c702` | 0 |

The 25 checkout targets comprise 24 available checkouts and one missing dependency. They contain 1,129 candidate documents and 15,541 lexical headings. The diagram corpus contains 158 topics and 1,526 Mermaid blocks; all declared source/governing paths exist. Existence does not prove a citation or its meaning. ADRs with nonstandard names outside decision directories still require discovery through references; this inventory does not claim to find every architecture decision embedded in prose.

## Scope and build findings

- **Loom dependency is unavailable locally.** `loom/Cargo.toml:66` declares `../ruvector/crates/ruvector-core`. Running `bash loom/scripts/check-ruvector-source.sh` from the workspace returned exit 1: `Missing RuVector sibling; follow docs/development-source.md`. The gate pins `677b2475409c50cb964be8a3b848da2952390535`. This blocks reproducing that consumed native source/build here; it does not establish a production outage. Prior RuVector audit receipts remain historical. Restore the documented dependency at its required revision before repeating the relevant tests.
- **Codescape has two revisions.** `project/codescape` is the clean gitlink observation `e3b76f63e801c6a8460110b93e88a0b420e9b0b4`; the standalone checkout is `58108c0357bd0dbf6588f8b6adf6e044f4c04556`. A standalone fix is not proof that VisionClaw consumes it. Audit integration at the submodule pin and standalone functionality at its own revision.
- **The health roster is narrower than the current map.** It omits Codescape despite the current README/map including it, and still describes dream-engine as an imported former fork while the map distinguishes the extracted engine from archived dream-machine. Treat this as ownership/provenance documentation drift, not permission to assume deployed services. Reconcile roster ownership before claiming complete nightly coverage.
- **Legacy inventory cannot refresh this scope.** `evidence/adr-inventory.py` hardcodes fourteen paths, including the absent RuVector checkout, and writes a fixed September date. Its unchecked `git ls-files` call aborts on a missing repository. Preserve its historical output; a replacement must accept an explicit scope and report unavailable repositories without silently dropping them.

## Whole-corpus render receipt

From the isolated VisionFlow worktree at `9c9aed1`:

```
node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --render
parsed 158 topic files, 1526 mermaid diagrams
rendered 1526/1526 diagrams via pele
```

Exit 0. This proves the generator's structural and render checks for all current topics. `--no-source-paths` is explicit: this is not an all-citation or semantic pass. No topic was restamped from this result.

## Review sequencing

Use the per-record hashes and heading lines to record source-backed dispositions section by section, preserving imported, historical and support classifications. Start with current first-party contracts, then reconcile predecessor obligations and imported consumer paths. For each material claim, record implementation evidence, adversarial counterexamples, tests actually executed, deployment evidence where available, and the owner/acceptance path for remaining work. A queue entry is not a finding and a passing gate is not an adoption decision.

The first newly covered plugin section is [the exclusion and write-recovery review](2026-10-31-obsidian-boundaries.md). The [market comparison](2026-10-31-market-and-audit-gaps.md) remains a bounded product analysis. Completion still requires all material repository paths, ADR dispositions and diagram claims to be assessed.

The [Codescape continuation](2026-10-31-codescape-boundaries.md) classifies its four ADR-like files as one operative decision plus an index, preamble and template. It assesses ADR-0001 section by section and compares standalone and consumed source. This does not close the inherited VisionClaw ADR-2143 obligations or the full Codescape runtime review.
