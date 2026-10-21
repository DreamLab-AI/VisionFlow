# Claims ledger and verification

Completed on 21 October 2026. Source establishes implementation and configuration; it does not establish fresh live acceptance. The starting revisions and README dates remain in [inventory.json](inventory.json); the final revisions and content hashes are in [final-state.json](final-state.json). Every README was read in full and compared with its recorded Git delta, relevant diagram topics and current implementation.

## Source and editorial review

| Repository | Load-bearing source checked | Changes |
|---|---|---|
| dream-engine | `crates/dream-core/src/source.rs`, ADR-DE-008, `Cargo.toml`, CLI help | Host/library split, requested-source budgeting in 0.2.0, workflow sequence, activation and automerge boundaries |
| ontomark | `crates/ontomark-model/src/profile.rs`, `crates/ontomark-compile/src/certificate.rs`, ADR-OM-009, manifests | Generic identity, certificates' scope, independently pinned plugin, qualified generation/freshness claims; historical research moved into details |
| ontomark-obsidian | `src/main.ts`, `src/settings.ts`, `src/ingest-transport.ts`, `manifest.json`, `engine.json`, desktop acceptance | Correct YAML examples, key/header and redirect boundary, pinned versions, diagram and current acceptance limits |
| solid-pod-rs | server `Cargo.toml`, `src/main.rs`, core `config/schema.rs`, `provenance.rs`, ADR-2004, alpha.17 changelog | Fixed config shape, no unsigned-write success promise, opt-in git marks and receipt tiers, implemented forge surface, comparator scope |
| sidestr-rs | workspace/crate manifests, `sidestr-agent/src/main.rs`, ADR-0002/0003 | Offline public-key example, crate/host diagram, eight publishable vs three unpublished crates, historical/live boundaries and teller licence exception |
| knowledgeGraph | archive layout, retired workflow, local and served stats, LICENSING, KG-01 diagram | Rebuilt around distribution role; archived reproduction uses actual paths; corrected local-vs-live counts; historical notices added at stale baseline/licensing text |
| Codescape | `client/src/ui/Repos.tsx`, `input/spaceMotion.ts`, changed tests, server `auth.rs`, deploy contract, evidence index | New repository/scene panel, corrected input axes, complete client build/run path, no implied working require-auth verifier; gallery and config folded |
| visiontouch-rs | governing baseline, Studio source/help, guided spec, AGENTS, current worktree status | Offline first path, architecture diagram, distinct attended hardware path, staged edits and no force acceptance; removed blanket legal permission claims |

| VisionFlow | Website build/workflow, package scripts, baseline and ADR-2019 | Static publication role, linked-repo map, component-specific reuse boundaries |
| VisionClaw | Corpus contract, ADR-2115/2147, env template, implementation and diagram sources | Shared Ontomark ownership, local-input default, independent desktop/Godot paths, scoped certificates and broker gates |
| agentbox | Manifest, runtime contracts, ADR-2141/2146, prepare/start scripts and build delta | Mandatory external memory, default-off isolation/activity integration, candidate preparation versus activation, current scope and new illustrative hero |
| nostr-rust-forum | Core beta.20 manifests, flags, auth/mesh/provenance source and room roadmap | Current permissive shared core, app AGPL, optional DID dependency, staged protocol features and unbuilt room surface |
| dreamlab-ai-website | Three client configs, worker ingress, content and deployment changes | Three clients, independent crate versions, encrypted/locked zones, ingress monitoring and historical screenshots |
| loom | Serving/retrieval routes, strict generation reload, publication CLI and research | Passthrough versus corpus retrieval, six-artifact reload, mirror paths, measured research scope and served-generation receipt |

## Final documentation checks

- All fourteen README local file targets, sibling GitHub file targets and Markdown anchors passed. Code fences and details tags balance; suspicious format/control Unicode and whitespace checks passed. [Machine-readable results](readme-checks.json).
- All fifteen Mermaid diagrams rendered through the estate Mermaid sidecar. Images were visually inspected, including corrected Sidestr dependency syntax. Preview artefacts are in `/tmp/visionflow-readme-render/`.
- Fourteen GFM page previews were checked in Chrome at widths 1280 and 390: one H1 each, no page overflow, no broken embedded images, and functioning disclosure controls. [Twenty-eight results](browser-checks.json). These are local previews, not GitHub-hosted acceptance.
- Existing plugin screenshots, all nine Codescape gallery images, corpus art, site screenshots and sampled VisionClaw video frames were inspected. Historical captures are labelled accordingly. The new Agentbox illustration was inspected alone and in the rendered page; it contains no licence/count claims and is explicitly decorative. [Media production record](media.md).

## Automated verification and limits

- VisionFlow `npm run build`: passed, including seventeen required assets. [Build log](site-build.log).
- `bash scripts/dream-link-check.sh`: passed, fifteen references, eighteen anchors and six CNAME checks. [Log](website-doc-checks.log).
- `bash scripts/dream-meta-tags-scan.sh`: passed. [Log](meta-tags.log).
- Obsidian plugin `npm test`: five files, 46 tests passed. Codescape client `npm test`: eight files, 86 tests passed. DreamLab website config/worker URL/CI subset: three files, 34 tests passed using its Vitest runner. The initial `node --test` invocation was incompatible with Vitest and was corrected. These 166 tests do not establish server, installed-desktop, headset, hardware or live-model acceptance.
- Dream CLI existing binary `--help` confirms the documented offline interface; no fresh source/binary equivalence claim is made.
- Public corpus stats were fetched: dataset 2026-10-20, 9,532 classes, 9,530 pages, zero individuals, nine domains and 68 categories. [Raw receipt](public-corpus-stats-2026-10-21.json). Local committed stats are an older August snapshot. A guessed generation endpoint returned 404 and was excluded from instructions.
- VisionFlow `npm run test:gates`: two of eight suites fail. Harness audit: 25 pass, one fail. Drift counter: 22 pass, six fail. The same failures reproduce with the committed baseline scripts/templates/prose in an isolated archive. They concern unresolved template sources and retired/source-pin assertions, not newly failing README behaviour. [Current gate log](gate-tests.log), [baseline harness](baseline-harness-audit.log), [baseline drift](baseline-drift-counter.log). No product code was changed to conceal them.
- The actual pinned Agentbox drift check passed after retiring numeric assertion locations removed by this refresh. Remaining assertions stay enforced. Its unavailable/planned corpus and roster axes remain explicitly disclosed. [Pinned-source log](pinned-drift.log).

## Concurrent changes and delivery state

Unrelated starting worktree edits were preserved. During review, another process advanced Agentbox through SBOM build-isolation changes and committed the rebuilt overview as `5e5c07779`; those deltas were inspected. The final artwork addition is a subsequent worktree edit. This session issued no commit, push, release or deployment. Other README edits remain in their local checkouts. Final hashes identify the delivered text independently of concurrent commit activity.
