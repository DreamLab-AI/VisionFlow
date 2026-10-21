# Estate README refresh, 21 October 2026

## Scope and production record

Rebuild the top-level README for each repository named in the website repository map, and dream-engine, which the page directly names as its self-improvement engine. The published site is https://www.visionflow.info/; the CI source is `website/static/index.html`, published by `.github/workflows/deploy.yml` through `website/build.sh`. The nightly health roster is broader and does not expand this task. Upstream references and showcase projects are excluded. No private authoring corpus is included.

The user authorised the hosted session model on 21 October 2026. Source inspection, drafting and verification use this Codex session. No local drafting model is required. Existing unrelated edits are preserved; the inventory records the starting worktree status. All fourteen README rewrites are complete. The [completion audit](completion-audit.md) and [verification record](verification.md) document the evidence and remaining pre-existing gate failures.

## Starting revisions and README dates

Dates below are Git committer dates for the last commit touching the top-level README, not filesystem modification times. Commit counts cover that commit to the inspected HEAD, including merges and automation; they are discovery signals, not a count of new features. Each history file records both revisions and commit subjects.

| Repository | Local checkout | Last README update (UTC) | Commits since | History |
|---|---|---|---:|---|
| VisionFlow | `VisionFlow` | 2026-10-20T00:50:37Z | 5 | [Revisions and changes](VisionFlow-history.txt) |
| VisionClaw | `project` | 2026-10-20T14:43:25Z | 6 | [Revisions and changes](VisionClaw-history.txt) |
| agentbox | `agentbox` | 2026-10-15T14:11:50Z | 101 | [Revisions and changes](agentbox-history.txt) |
| solid-pod-rs | `solid-pod-rs` | 2026-10-20T00:31:00Z | 0 | [Revisions and changes](solid-pod-rs-history.txt) |
| nostr-rust-forum | `nostr-rust-forum` | 2026-10-20T19:42:03Z | 2 | [Revisions and changes](nostr-rust-forum-history.txt) |
| dreamlab-ai-website | `dreamlab-ai-website` | 2026-09-23T15:48:16Z | 144 | [Revisions and changes](dreamlab-ai-website-history.txt) |
| loom | `loom` | 2026-10-13T12:05:07Z | 5 | [Revisions and changes](loom-history.txt) |
| knowledgeGraph | `knowledgeGraph` | 2026-09-23T15:48:19Z | 0 | [Revisions and changes](knowledgeGraph-history.txt) |
| sidestr-rs | `sidestr-rs` | 2026-10-18T10:54:49Z | 0 | [Revisions and changes](sidestr-rs-history.txt) |
| ontomark | `ontomark` | 2026-10-20T15:29:58Z | 2 | [Revisions and changes](ontomark-history.txt) |
| ontomark-obsidian | `ontomark-obsidian` | 2026-10-17T09:44:02Z | 0 | [Revisions and changes](ontomark-obsidian-history.txt) |
| dream-engine | `dream-engine` | 2026-10-15T21:17:25Z | 15 | [Revisions and changes](dream-engine-history.txt) |
| visiontouch-rs | `visiontouch-rs` | 2026-10-19T10:22:38Z | 1 | [Revisions and changes](visiontouch-rs-history.txt) |
| codescape | `codescape` | 2026-10-19T13:51:35Z | 1 | [Revisions and changes](codescape-history.txt) |

Agentbox's top-level workspace path is a symlink to `project/agentbox`. VisionClaw is `project`, identified by its DreamLab-AI/VisionClaw remote. Codescape has a standalone checkout and a VisionClaw submodule; the standalone checkout is the target here. The full machine-readable inventory is [inventory.json](inventory.json).

## Shared editorial contract

Keep the best existing patterns: the estate's centred identity and navigation; Ontomark's concrete two-note example; Codescape's captured-media entry point and explicit hardware boundaries; Dream Engine's explanation of what a gate does and does not establish. Apply the same reading order without forcing identical content:

1. A precise product description, intended user and practical benefit.
2. A working first path, including prerequisites and configuration actually required by the source.
3. An architecture diagram that distinguishes implementation ownership from runtime integration.
4. Repository-specific capabilities and a concrete workflow, with deep links to code and governing docs.
5. Dated working status: source, automated checks, activation and live acceptance are distinct.
6. Expandable implementation detail, keeping setup and material limitations visible.
7. Ecosystem position, documentation entry points, upstream credit and actual licence boundaries.

Prefer existing estate images whose contents have been viewed. Diagrams stay Mermaid in Markdown. Do not repeat volatile counts as badges or copy research results without their experiment scope. Avoid blanket licences, universal signed-decision claims, presumed deployments, or a single version number for independently released consumers. Retire obsolete present-tense claims where encountered; preserve historical records as history.

## Evidence and completion

Use `docs/diagrams/` as a source map, not proof of current deployment. Each topic's `verified_commit` describes its own snapshot. Follow the cited implementation and compare it with changes since the README update. Source/configuration takes precedence over old marketing prose. Do not infer a live service from a feature flag, release, passing test or ADR title.

For every target: read the whole README; inspect commit diffs and relevant diagrams; trace current setup and feature claims to source; rewrite; check local links and anchors, Markdown structure, Mermaid and media; inspect the final diff. Record which commands were actually run and their limits. The final fourteen-target review is recorded in [completion-audit.md](completion-audit.md).
