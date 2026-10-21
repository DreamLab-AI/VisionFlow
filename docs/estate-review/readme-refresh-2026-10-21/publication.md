# Publication receipts, 21 October 2026

All fourteen README files were pushed to their repositories’ main branches and verified through the GitHub contents API against their local Git blob hashes. Publication used clean worktrees based on remote main, excluding unrelated dirty files and local commits. The forum upgrade branch and Codescape feature commit were left unmerged, and their README status sections explicitly say so.

| Repository | Published documentation commit |
|---|---|
| VisionFlow | [909bd92b14](https://github.com/DreamLab-AI/VisionFlow/commit/909bd92b142d630f0cd62e9a14917154d31007a9) |
| VisionClaw | [d84d0e29ae](https://github.com/DreamLab-AI/VisionClaw/commit/d84d0e29ae1a7fc7f91beb689f4d871f2aca35b2) |
| agentbox | [f9eb13b85a](https://github.com/DreamLab-AI/agentbox/commit/f9eb13b85ab4167612e847136a83048f0e92885d) |
| solid-pod-rs | [7b5c781a4d](https://github.com/DreamLab-AI/solid-pod-rs/commit/7b5c781a4d8b89d59d0b96bca7211ccb984fbf68) |
| nostr-rust-forum | [7eaecc3152](https://github.com/DreamLab-AI/nostr-rust-forum/commit/7eaecc3152942793cbc09fc9a2b243ec2e5f8c4c) |
| dreamlab-ai-website | [af70144a39](https://github.com/DreamLab-AI/dreamlab-ai-website/commit/af70144a3908c79687c4e2f8da6baf961fe58900) |
| loom | [3487ceed5e](https://github.com/DreamLab-AI/loom/commit/3487ceed5e2040eddc3eaa478d383e84094513c6) |
| knowledgeGraph | [1edcfc93de](https://github.com/DreamLab-AI/knowledgeGraph/commit/1edcfc93de90ccff564073ef82cb198c422f75e3) |
| sidestr-rs | [1249e34def](https://github.com/DreamLab-AI/sidestr-rs/commit/1249e34def303fbb4b9f8380de684c0719a9c75f) |
| ontomark | [3e909e52b8](https://github.com/DreamLab-AI/ontomark/commit/3e909e52b8ddd20effa832666677798fda644cfc) |
| ontomark-obsidian | [7af6a92f0e](https://github.com/DreamLab-AI/ontomark-obsidian/commit/7af6a92f0e4166c0e4b80e72f0b3a3ebf14cdebb) |
| dream-engine | [8079c081a5](https://github.com/DreamLab-AI/dream-engine/commit/8079c081a5ee5a5a25a6064efc5f115a50b731c0) |
| visiontouch-rs | [08c6f1bb68](https://github.com/DreamLab-AI/visiontouch-rs/commit/08c6f1bb688a9daf8cfbe89488557a2878cf86cf) |
| codescape | [2593478069](https://github.com/DreamLab-AI/codescape/commit/259347806982a85d0d64c2c7949d9a238acdd01f) |

## Live site

VisionFlow’s [Pages deployment](https://github.com/DreamLab-AI/VisionFlow/actions/runs/39595085897) succeeded for `909bd92b142d630f0cd62e9a14917154d31007a9`. All blocking publication gates, including the pinned-source drift check, passed. A direct fetch of [www.visionflow.info](https://www.visionflow.info/) matched the candidate HTML byte-for-byte: SHA-256 `8499d24ad1f557fa5d97788fed287d3ead5752a75775659bac85641d4f088bc5`.

The subsequent receipt-only commit records these results; it does not change site HTML. The Agentbox illustration is included in its README commit. DreamLab’s website received only its repository README, so no application redeployment was needed there. README publication does not deploy any product runtime or update a host’s submodule pins.

The haptic repository’s required full hardware-free quality gate passed in its isolated publication checkout before commit. Its first attempts encountered the container’s non-executable temporary filesystem and production-only Node installation; moving the checkout and installing its declared development dependencies resolved those environment issues. No gates were skipped.

[Machine-readable remote verification](publication-checks.json). The initial review hashes remain historical snapshots; the publication receipt identifies the final text, including the forum and Codescape status corrections.
