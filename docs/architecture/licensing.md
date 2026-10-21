# Licensing across the estate

Reviewed 21 October 2026 from the local licence files and manifests during the README refresh. This is a navigation map: each component's operative licence, notices, selected features and combined dependency graph govern its use. Proposed relicensing and a GitHub detector badge do not replace those terms.

| Repository or surface | Declared terms | Boundary and source |
|---|---|---|
| VisionFlow | No root licence grant | [Local licence notes](../../LICENSES/README.md) require maintainer confirmation for reuse of documentation/site assets outside the ecosystem |
| VisionClaw application | AGPL-3.0-only | Root `LICENSE` / `Cargo.toml`; the in-tree MPL proposal is not operative; extracted crates have their own terms |
| Agentbox core | AGPL-3.0 | Root licence; bundled tools and extracted services/libraries have distinct notices and dependency boundaries |
| solid-pod-rs | AGPL-3.0-only | Workspace manifests and licence; native, embedded and Worker consumers have different feature graphs |
| nostr-rust-forum | Component-specific | Applications/workers follow AGPL-3.0-only; `nostr-bbs-core` declares MIT OR Apache-2.0 from beta.18, with the default-off `did` feature adding the Solid dependency |
| dreamlab-ai-website | Proprietary branded React site; AGPL forum surfaces | Root package is `UNLICENSED`; `forum-config/` and combined kit surfaces carry their separate source/terms |
| Loom | MIT OR Apache-2.0 | `LICENSE-MIT`, `LICENSE-APACHE`, workspace manifest; corpus terms remain separate |
| knowledgeGraph | ODbL-1.0 data, MIT retained explorer, AGPL-3.0-or-later pipeline/applicable prose | `LICENSING.md` defines path scope, including relocated archive paths |
| sidestr-rs | AGPL family, per crate | Sidestr crates declare AGPL-3.0-only; `webledgers-teller` declares AGPL-3.0-or-later; unpublished adapters are separate from publication status |
| Ontomark | MIT OR Apache-2.0 | Workspace/crate manifests, notices and upstream credits |
| Ontomark for Obsidian | MIT OR Apache-2.0 | Plugin terms plus release third-party notices and pinned engine |
| Dream Engine | MIT OR Apache-2.0 | Independent Rust workspace; its Agentbox host/adapter remains in the consuming product under different terms |
| Codescape | MIT OR Apache-2.0 model/collaboration crates; AGPL-3.0-only application | Per-directory licence; a permissive crate must not depend on an AGPL application crate |
| visiontouch-rs | Private research study, not a general software release grant | Its charter and clean-room/project position govern the study; no public licence is inferred here |

Follow the owning [README](../../README.md#choose-your-starting-point) and its licence links before reusing code, data or media. An identity protocol, a shared wire record or a compatible API does not make all connected repositories share one licence. The broader [repository map](repository-map.md) also includes supporting projects outside this refresh; consult their own terms independently.
