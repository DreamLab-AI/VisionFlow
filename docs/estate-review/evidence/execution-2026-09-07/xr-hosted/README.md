# Hosted VisionClaw verification at fb72df39a

Observed 2026-09-07 after the visual source, viewport lifecycle correction and Xvfb CI changes were pushed.

| Workflow/job | Hosted receipt | Observed result |
|---|---|---|
| Documentation Quality CI | [run 34126476287](https://github.com/DreamLab-AI/VisionClaw/actions/runs/34126476287) | Success; documentation quality and ADR ledger both green |
| XR GUT under Godot 4.3 GL | [job 101756326729](https://github.com/DreamLab-AI/VisionClaw/actions/runs/34126476293/job/101756326729) | Success; 83 tests, 83 passing, 314 assertions, JUnit uploaded |
| XR Rust crates | [job 101756326670](https://github.com/DreamLab-AI/VisionClaw/actions/runs/34126476293/job/101756326670) | Still installing native dependencies at this observation |
| Quest APK | [job 101756326397](https://github.com/DreamLab-AI/VisionClaw/actions/runs/34126476293/job/101756326397) | Still building arm64 extension at this observation |
| General CI | [run 34126476290](https://github.com/DreamLab-AI/VisionClaw/actions/runs/34126476290) | Format, release dev-auth invariant and Client Vitest green; CPU job ongoing; advisory ESLint failed |

The green GUT job is direct hosted evidence, not an inference from the local run. Its log retains fresh headless-import native-class diagnostics and the two 349,524-byte texture-release diagnostics at test shutdown. The actual GL suite passed all cases without risky tests. A green job does not imply an error-free engine log or headset acceptance.

The older XR run 34119893573 was cancelled by the new push. Its GUT job had spent about 71 minutes in an invocation without resource import, emitting missing GUT font/global-class errors. The new workflow's explicit import followed by actual GL completed successfully. Its Rust job was green. Its Quest arm64 compilation succeeded but release export failed with `Could not find keystore, unable to export.` No release signing identity was fabricated or debug signer substituted. Logs are retained alongside these snapshots.

The JSON snapshots are point-in-time observations. They do not claim that pending jobs or the whole XR run passed. No new CI source fix was needed for the now-green GUT/documentation jobs during this observation.
