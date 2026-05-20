# Licensing Architecture

**Status:** Docs-only synthesis
**Date:** 2026-05-20

VisionFlow combines repositories with different licence obligations. This page is a map, not legal advice.

| Repository | Licence described in docs | Practical boundary |
|---|---|---|
| VisionFlow | Public docs describe mixed MPL 2.0 + AGPL 3.0 ecosystem | Documentation and static website assets |
| VisionClaw | MPL 2.0 | Knowledge engineering substrate and GPU/XR application code |
| agentbox | AGPL 3.0 | Networked agent runtime and management API |
| solid-pod-rs | AGPL 3.0 | Solid/JSS foundation library and server |
| nostr-rust-forum | AGPL 3.0 per ecosystem docs; workspace crates may carry crate-level terms | Forum kit, Cloudflare Workers, relay/governance UI |
| dreamlab-ai-website | Deployment repo; inherits obligations from consumed kit/components | Branded site, operator config, deployment workflows |

## Boundary Rule

Treat protocol and server components derived from or linked against the JSS/Solid stack as AGPL-sensitive unless the owning repository states otherwise. Treat VisionClaw MPL code as a separate boundary, with explicit review required when importing AGPL libraries into MPL-distributed artifacts.

## Open Licensing Questions

1. Which `nostr-rust-forum` crates are intended for independent publication, and under what crate-level licence?
2. Does VisionClaw link to AGPL code at build time, call it over process/HTTP boundaries, or both?
3. Are generated schemas, fixtures, and protocol docs intended to be reusable outside AGPL repos?
4. What commercial licensing path applies when an enterprise deployment combines all five substrates?

