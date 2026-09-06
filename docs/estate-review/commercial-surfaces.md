---
title: Commercial surfaces and release boundaries
status: in-progress
date: 2026-09-04
type: explanation
---

# Commercial surfaces and release boundaries

The commercial website is an operator overlay: it owns the React marketing journey, branding and deployment configuration, while the forum kit supplies the forum, BBS and workers. Its value depends on visitors reaching a consistent identity, conversation and community experience across those components. Source inspection confirms the boundaries, but does not establish a deployed end-to-end service. The [receipt](evidence/commercial-snapshot.json) identifies inspected source hashes and local tests.

## Build identity and release gates

The [deploy workflow](../../../dreamlab-ai-website/.github/workflows/deploy.yml) builds React at `/`, the forum at `/community/` and the BBS at `/community/bbs/`. It publishes static assets to GitHub Pages and has a separately gated Cloudflare Pages path. [Workers](../../../dreamlab-ai-website/.github/workflows/workers-deploy.yml) deploy independently after their own gate. Old DNS observations in governing documents remain historical; no DNS or deployed configuration was checked here.

Three workflow `KIT_REF` values agree at `a7544687b4d1c09807862d749b27f8c8da307a12`. The four Cargo requirements and compatibility record agree at `1.0.0-beta.9`; the local parity script passes. These requirements lack an exact `=` prefix: the lockfile supplies resolved versions. The parity script compares authored requirements, workflow SHAs and compatibility markers; it does not compare the lockfile's resolved packages or prove package contents match the source clone. The inspected forum checkout has a different HEAD, so tests of its current source cannot automatically certify this pinned consumer.

[CI](../../../dreamlab-ai-website/.github/workflows/ci.yml) runs Vitest and includes pin/admin consistency checks in its `ci-pass` aggregator. However, the deployment workflows depend on the separate [reusable gate](../../../dreamlab-ai-website/.github/workflows/test-and-lint.yml), which builds React and tests the Rust overlay but omits Vitest and makes lint/clippy advisory. Its `always()` summary writes `passed=true` even after a preceding failure; this does not itself make a failed job succeed, but the summary is unreliable evidence. Remote branch protection was not inspected. The [upstream Rust workflow](../../../dreamlab-ai-website/.github/workflows/rust-ci.yml) is manual, and its WASM test command compiles with `--no-run`.

**Closeout:** CP-01/08 needs one release receipt binding resolved libraries, cloned kit, three frontend builds, workers and effective required checks. Inject pin drift and a failed React test and establish that publication of that revision is denied. Record actual deployed revisions and rollback compatibility.

## Configuration and identity

The [authored TOML](../../../dreamlab-ai-website/forum-config/dreamlab.toml) defines four zones, dual-accept cohort labels and Family as the sole encrypted zone. Client keys and zones are hand-mirrored into deployment configuration. CI checks the admin set against relay/search wrangler files; it does not cover all client/zone mirrors or the operator-managed auth-worker secret. These are distinct coverage boundaries, not an absence of consistency checking.

The [identity governing document](../../../dreamlab-ai-website/docs/IDENTITY-zones.md) already records the unsplit admin/governance service key and deferred DID representation. Config flags and authored rosters do not alone prove deployed membership, encryption or authority separation. CP-04/05 needs explicit role separation, a rotation receipt covering every consumer, and grant/revoke tests for both cohort-label vintages against the deployed kit revision.

## Chat transport and product meaning

[AIChatFab](../../../dreamlab-ai-website/src/components/AIChatFab.tsx) creates one ephemeral session identity. Selecting tiers 2/3 calls the extension's `getPublicKey` and displays a greeting, but `sendQuestion` receives only message text and the agent recipient key. Neither the selected tier nor proof of the extension identity is transmitted by this path. The UI's identity/tier labels therefore do not establish authenticated access to private VisionFlow context.

[DmSession](../../../dreamlab-ai-website/src/lib/nostr.ts) subscribes before publishing, distinguishes relay OK from an agent reply, verifies the signed seal and author binding, pins the expected sender, deduplicates wrap IDs and rejects replies older than the session allowance. Open reply listeners retry independently and do not gate connection. The component reports send failure and reply timeout separately and accepts late replies. Any accepted agent reply can resolve the currently pending turn; there is no question identifier checked at this boundary, so a late reply can be associated with a later question. This is a source-level correlation gap, not a reproduced live failure.

All 97 existing Vitest tests pass across seven files, including 26 transport and 12 chat-component tests. They use local mocks and do not establish real agent fan-out, browser delivery, authority or private-context access. CP-03/05 needs an explicit product decision on tier meaning, authority proof if private access is intended, request/reply correlation, and a real approved test journey covering unavailable reply relays, late replies, wrong sender and identity rotation. Browser throttles reset when the panel closes and are not a server-side abuse boundary.

## ADR closeout

All eight operative website ADRs now have scoped closeout extensions. Existing historical evidence and activation declarations are retained; this pass does not re-certify live deployment. Their two governing documents carry matching qualifications. Frozen archive lineage still needs record-by-record disposition under the [estate roadmap](closeout/README.md).
