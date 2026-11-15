# ADR-2024 execution receipt — 15 November 2026

## Source and release

| Component | Revision / release | Evidence |
|---|---|---|
| Ontomark protocol | f9ba200b8051dc3a1f5e26ff2e001d5fc2905426 / v0.9.0 | Published crates and GitHub CLI/WASM assets |
| Ontomark consumer CI | fbe30244c6842879297883b571965f5bc1ce0441 | Pins updated consumer and retains its exclusion fixtures |
| Loom | c016dab | Registry dependencies; producer, attribution and review metadata retained |
| VisionClaw | d8a5e0b71 | Shared review assessment, legacy builder compatibility and v3 projection |
| Agentbox | d98855de7 | Release binary plus Jev source/lock pins; image activation staged |
| visionGraph | fa6856b404eece8ae8d486e0eaf32df46d07820c | Release pin and v3 publication gates |
| Obsidian | 64bdc5bf6309fca94bf0437b52c1484618470ed0 | Engine pin, generated declarations, retained host exclusion fixture |

CLI asset SHA-256:
`bca8adc3e1438f70dc9cdca15661a6fd5532f10f2d5f0776c7099222c34c9f5d`.
Downloaded bytes passed the release checksum and executed as `ontomark 0.9.0`.
WASM engine archive SHA-256:
`774b6ca6b7ca6deab147e1213a0294d4b6d5e651d1533753a42306b262ea6213`.
The consumer fetcher verified both digest and source receipt.

Agentbox Jev source NAR SHA-256:
`65edc1b96e1f43222e64df884fdaec2a3d32644da6eb4d216f9a4f5d6b73fa3d`.
`nix-nar-cli 0.6.0 dump-path` reproduced the previous source pin's known NAR
hash as a positive control, then hashed the release source. The lockfile is
copied byte-for-byte from that release. Nix evaluation was unavailable locally.

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
- [Release workflow 43717471418](https://github.com/DreamLab-AI/ontomark/actions/runs/43717471418)
  passed. Earlier producer CI exposed declaration drift; the updated consumer
  and corrected fixture boundary address that failure without relaxing the gate.

Raw local outputs are retained under
`~/workspace/backups/branch-assessment-20261115/provenance*.log`.
An initial concurrent local Cargo run stalled and was stopped; the bounded
rerun above completed. Neither the stalled run nor ignored tests count as passes.
No independent cross-family audit was run; this receipt establishes implemented
controls, local tests and named hosted checks, not independent audit acceptance.

## Deployment boundary

Source adoption is merged to the repositories' main branches. Public publication
is tracked by [visionGraph run 43718382066](https://github.com/jjohare/visionGraph/actions/runs/43718382066).
Agentbox image rebuilding requires the host environment; the container has no
Nix CLI and the documented host pane is occupied by Codex. Running Loom and
VisionClaw processes have not been restarted by this change. Source, release,
consumer compatibility and runtime activation remain separate claims.
