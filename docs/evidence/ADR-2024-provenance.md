# ADR-2024 execution receipt — 15 November 2026

## Source and release

| Component | Revision / release | Evidence |
|---|---|---|
| Ontomark protocol | f9ba200b8051dc3a1f5e26ff2e001d5fc2905426 / v0.9.0 | Published crates and GitHub CLI/WASM assets |
| Ontomark consumer CI | fbe30244c6842879297883b571965f5bc1ce0441 | Pins updated consumer and retains its exclusion fixtures |
| Loom | c016dab | Registry dependencies; producer, attribution and review metadata retained |
| VisionClaw | d8a5e0b71 | Shared review assessment, legacy builder compatibility and v3 projection |
| Agentbox | d98855de7 | Release binary plus Jev source/lock pins; image activation staged |
| visionGraph | 1df8eb49051b6e55f35ea493b3c42e79ffd77bb8 | Release pin, v3 gates and compatible publisher runner |
| Obsidian | 64bdc5bf6309fca94bf0437b52c1484618470ed0 | Engine pin, generated declarations, retained host exclusion fixture |

CLI asset SHA-256:
`bca8adc3e1438f70dc9cdca15661a6fd5532f10f2d5f0776c7099222c34c9f5d`.
Downloaded bytes passed the release checksum and executed as `ontomark 0.9.0`.
The published binary also matched five independent SHA-256 digest vectors
(string, sequence, map and two distinct high-precision floats), leaving each
input file byte-identical.
WASM engine archive SHA-256:
`774b6ca6b7ca6deab147e1213a0294d4b6d5e651d1533753a42306b262ea6213`.
The consumer fetcher verified both digest and source receipt.

Agentbox Jev source NAR SHA-256:
`65edc1b96e1f43222e64df884fdaec2a3d32644da6eb4d216f9a4f5d6b73fa3d`.
`nix-nar-cli 0.6.0 dump-path` reproduced the previous source pin's known NAR
hash as a positive control, then hashed the release source. The lockfile is
copied byte-for-byte from that release. Nix evaluation was unavailable locally.

VisionClaw retains its legacy builder and existing `ontomark-compile 0.6.3`
dependency; the shared review contract comes from registry `ontomark-parse 0.1.4`.
Loom consumes compiler 0.9.0. A final lock/pin audit confirmed these versions,
absence of temporary local Cargo patches, matching CLI hashes across three
publishers and the exact release lockfile in Agentbox. This provenance delivery
does not claim completion of the broader compiler extraction.

## Executed checks

- Ontomark: bounded `cargo test --offline --workspace`: 1,154 passed, 8 existing
  ignored; all-feature suite: 705 passed, 7 ignored; no-default suite: 600 passed,
  2 ignored. Formatting, workspace/all-target Clippy with warnings denied and
  rustdoc with warnings denied passed. Graph JavaScript: 28 passed, 1 skipped.
- Native/WASM review-binding parity passed with matching wasm-bindgen 0.2.127,
  including independently encoded fixed SHA-256 vectors and malformed-scope
  controls. No platform-specific hashing substitute.
- Jev creation feature: 3 controls passed, covering refusal, unavailable advisor
  and non-mutating confident suggestion.
- Loom: 137 tests passed against registry dependencies; full workspace check
  passed. Domain/scaffold Clippy passed. Broader facade Clippy still reports six
  pre-existing warnings in untouched Jev/routes code; it is not a claimed pass.
- VisionClaw: 520 tests passed, 5 existing ignored. Final vault/vault-core
  all-target Clippy passed. All 15 golden compatibility tests passed; only named
  additive metadata is normalised, and new fields have explicit assertions.
- Obsidian: CI-mode production build, 121 unit tests, Node/browser WASM replay,
  bundle boundary and 69 network-disabled offline checks passed. Hosted
  [plugin run 43718397137](https://github.com/DreamLab-AI/ontomark-obsidian/actions/runs/43718397137)
  also passed. The consumer's exclusions differ from native fixture scope, so
  producer CI now retains the consumer fixtures rather than overwriting them.
- visionGraph: 41 publication/explorer/plugin contract tests passed. Agentbox,
  visionGraph, VisionClaw, Ontomark and plugin ADR ledgers passed locally.
- [Producer CI 43718419344](https://github.com/DreamLab-AI/ontomark/actions/runs/43718419344)
  passed all four jobs: ADR ledger, secrets, plugin/native-WASM parity and Rust
  checks/features/docs. Release source and this CI revision differ only in
  consumer-test wiring; published protocol code is unchanged.
- [Loom contract CI 43718403385](https://github.com/DreamLab-AI/loom/actions/runs/43718403385)
  passed against the adopted source.
- VisionFlow ADR, source guard and documentation deployment workflows passed
  at `7d24652c570553ea1fa9541bf790d1eedcee3ef6`.
- [Release workflow 43717471418](https://github.com/DreamLab-AI/ontomark/actions/runs/43717471418)
  passed. Earlier producer CI exposed declaration drift; the updated consumer
  and corrected fixture boundary address that failure without relaxing the gate.

Raw local outputs are retained under
`~/workspace/backups/branch-assessment-20261115/provenance*.log`.
An initial concurrent local Cargo run stalled and was stopped; the bounded
rerun above completed. Neither the stalled run nor ignored tests count as passes.
No independent cross-family audit was run; this receipt establishes implemented
controls, local tests and named hosted checks, not independent audit acceptance.

## Public runtime verification

The corrected [publisher run 43720813096](https://github.com/jjohare/visionGraph/actions/runs/43720813096)
passed all gates, including explorer browser smoke, credential scanning and site
layout. [Pages deployment 43722937353](https://github.com/DreamLab-AI/knowledgeGraph/actions/runs/43722937353)
then passed for `gh-pages` commit `cf29935f59805db79f6d6564ec6d5237226c3969`.
Live HTTP checks on 15 November 2026 verified:

- `/context/v1.jsonld`, `/context/v2.jsonld` and `/ns/v2.jsonld` retain their
  pre-deployment SHA-256 hashes exactly.
- `/context/v3.jsonld` and `/ns/v3.jsonld` are byte-identical and declare the
  scoped current/stale/unbound terms.
- The sampled `/api/v3/pages/1-hive.json` equals the decompressed legacy page
  after removing only `reviewBinding`; its three counters are non-negative.
  This is a one-page live compatibility probe, not a claim that every page was
  independently fetched. Producer fixtures cover current/stale/unbound behaviour.
- `/data/.generation.json` records `ontomark-compile 0.9.0` and source commit
  `1df8eb49051b6e55f35ea493b3c42e79ffd77bb8`.

[Exact live hashes and observations](ADR-2024-live.json) accompany this receipt.
The first probe saw a 404 while the separate Pages deployment was still building;
that observation was retained as pending publication, not treated as a pass.

## Deployment boundary

Source adoption is merged to the repositories' main branches. The first public publication run,
[43718382066](https://github.com/jjohare/visionGraph/actions/runs/43718382066),
passed corpus/build/artefact checks but failed before browser execution because
Playwright 1.56.1 does not support the Ubuntu 26.04 selected by ubuntu-latest.
The publisher now pins Ubuntu 24.04 and installs the browser before Quartz.
The complete rerun is [43720813096](https://github.com/jjohare/visionGraph/actions/runs/43720813096).
Agentbox image rebuilding requires the host environment; the container has no
Nix CLI and the documented host pane is occupied by Codex. Running Loom and
VisionClaw processes have not been restarted by this change. Source, release,
consumer compatibility and runtime activation remain separate claims.
