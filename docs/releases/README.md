# Ecosystem Release Manifests

VisionFlow releases should publish a machine-readable manifest that pins the compatible repository set.

Generate a local manifest with:

```sh
scripts/generate-release-manifest.sh > docs/releases/ecosystem-release.local.json
```

The generated file is intentionally not committed by default. Commit a dated manifest only when cutting a coordinated release.

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
| `compatibility` | Human-maintained protocol compatibility assertions |

