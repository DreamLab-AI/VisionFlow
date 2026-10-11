# Ecosystem Release Manifests

VisionFlow releases should publish a machine-readable manifest that pins the compatible repository set.

Generate a local manifest with:

```sh
scripts/generate-release-manifest.sh > docs/releases/ecosystem-release.local.json
```

The generated file is intentionally not committed by default. Commit a dated manifest only when cutting a coordinated release.

The current roster contains sixteen repositories, including public
[Ontomark](https://github.com/DreamLab-AI/ontomark) and
[ontomark-obsidian](https://github.com/DreamLab-AI/ontomark-obsidian). The
[inventory](../estate-review/evidence/adr-inventory.json) preserves its original
fourteen-repository 2026-09-05 scan and appends only the two producer observations
dated 2026-10-11. Current observed releases are core native `v0.1.0-rc.1`, graph
`ontomark-graph-v0.1.0` and plugin `0.1.0`; checkout HEADs and release source pins
are separate. See [ADR-2006](../adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md#ontomark-roster-continuation--2026-10-11)
for the governed paths and local count-integrity checks. A roster entry conveys
ownership and checkout state, not maturity or compatibility qualification.

## Release Candidates

| Date | File | Status | Notes |
|---|---|---|---|
| 2026-05-22 | [`candidate-2026-05-22.json`](candidate-2026-05-22.json) | candidate | First ecosystem release candidate. Pins all 6 repos at main branch HEADs. 4 repos have uncommitted local changes (VisionClaw, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website) that must be resolved before promotion to released. |

## Required Fields

| Field | Meaning |
|---|---|
| `manifest_version` | Schema version for the manifest file |
| `generated_at` | UTC generation timestamp |
| `status` | `local-draft`, `candidate`, or `released` |
| `repositories[].name` | Repository identifier |
| `repositories[].path` | Local workspace path used to generate the SHA |
| `repositories[].head` | Full Git commit SHA |
| `repositories[].branch` | Current local branch |
| `repositories[].repository_url` | Optional public producer URL; distinct from an external `upstream` |
| `compatibility` | Human-maintained protocol compatibility assertions |

