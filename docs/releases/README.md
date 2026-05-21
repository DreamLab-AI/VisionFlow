# Ecosystem Release Manifests

VisionFlow releases should publish a machine-readable manifest that pins the compatible repository set.

Generate a local manifest with:

```sh
scripts/generate-release-manifest.sh > docs/releases/ecosystem-release.local.json
```

The generated file is intentionally not committed by default. Commit a dated manifest only when cutting a coordinated release.

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

