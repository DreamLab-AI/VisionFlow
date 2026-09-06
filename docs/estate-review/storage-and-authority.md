---
title: Pod storage and authority across tiers
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Pod storage and authority across tiers

Pods give the estate a concrete sovereignty mechanism: a principal can own resources and grant others limited access through policy attached to those resources. The design is more useful than a generic promise of decentralisation because it identifies the decision point, storage boundary and credentials. Its credibility depends on consistent enforcement across the native service, embedded runtime and edge worker, including failures and cached responses.

This pass inspects core pod policy, native authentication/storage/provenance and the forum's edge adapter. It does not yet complete delegated mandate lifecycles, Solid-OIDC, the forge, payments, remote storage federation or disaster recovery. [Receipts](evidence/pod-snapshot.json) identify the sources and **46 passing existing tests**. [Temporary-crate probes](evidence/pod-probes.py) exercise the actual native ACL resolver and replay cache, with [results](evidence/pod-probes.json). No real pod, policy, credential or remote service was changed.

## One library name does not mean one deployed implementation

| Consumer | Current local evidence | Interpretation |
|---|---|---|
| solid-pod-rs checkout | Workspace version `0.5.0-alpha.8` | Current source reviewed here; not proof of deployed version |
| agentbox native service | Nix version `0.5.0-alpha.3`, revision `87b35a1b32f9789e296ebbf7277b9ecc01657c42`, with a local patch | Pinned upstream source plus downstream modification |
| VisionClaw embedded pod | Lockfile resolves `0.4.0-alpha.15`; `solid-pod-embed` is a default feature | Older library/server family remains part of the configured default build |
| forum edge worker | Exact dependency and lockfile `0.5.0-alpha.7`, pure `core` surface | Shared policy logic inside a separately implemented Workers runtime |

These values come from the [native build expression](../../../project/agentbox/lib/solid-pod-rs.nix), [VisionClaw manifest](../../../project/Cargo.toml) and [lockfile](../../../project/Cargo.lock), and the [forum manifest](../../../nostr-rust-forum/Cargo.toml) and [lockfile](../../../nostr-rust-forum/Cargo.lock). A current-source security fix or test result must not be attributed to all three consumers without checking their resolved source and feature set. This review has not yet mapped every patch across those versions.

The native Nix build includes `git` and applies a specific patch to strip a leading `pods/` segment in `git_mark_write`. Its explanatory comment records why: resources served at `/pods/{pod}/...` otherwise select a repository named `pods`. The current upstream helper still reads the first segment directly. The patch is part of the downstream contract, not evidence that a generic upstream build automatically marks this deployment's writes.

## Authentication and policy are separate checks

The current [native server](../../../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs) reconstructs the signed URL, verifies the NIP-98 event, and checks a process-wide replay store before returning a principal. Write paths can supply the request body for payload-hash binding. The [verifier](../../../solid-pod-rs/crates/solid-pod-rs/src/auth/nip98.rs) checks timestamp, method, URL, payload and signature according to enabled features. Tests executed here cover method/payload/skew behaviour and rejection of forged signatures or altered event IDs.

Policy resolution then obtains the effective ACL and evaluates the requested mode. Current native read and write guards both use `effective_acl_target`: accessing an ACL or metadata sidecar requires Control on the resource it governs. That shared mapping prevents read-side policy disclosure and write-side self-escalation from drifting into separate implementations. Ordinary grants remain distinct: Control does not imply Read; Write implies Append. Existing inheritance and mode tests pass.

The [edge ACL adapter](../../../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs) reuses the same upstream sidecar mapping and evaluator. It retains runtime-specific resolution, a stricter 64 KiB document cap, and owner-control preservation for replacements. Structured delegation deliberately grants Read/Write/Append, not Control. Its wrapper currently supplies no request Origin, so native origin-aware policy and the edge wrapper are not interchangeable contracts. Full client/issuer conditions and remote delegation still need consumer-level traces.

## Replay protection has different persistence boundaries

Native [Nip98ReplayCache](../../../solid-pod-rs/crates/solid-pod-rs/src/auth/replay.rs) is a bounded in-process LRU. Our one-entry probe confirms both the useful guard and its documented limit: the same event ID is rejected while retained; inserting another ID evicts it, after which the first ID is accepted again within the TTL. This is an actual helper result with synthetic IDs, not a reproduced signed-request attack.

Process restart, multiple replicas and capacity pressure therefore matter to the effective replay window. The library already documents these limits, so they should be preserved in estate-level claims rather than presented as a newly discovered secret. Cache sizing and token validity need a joint deployment contract.

The [forum authentication wrapper](../../../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/auth.rs) uses a different path: the shared [rate-limit crate's replay adapter](../../../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs) performs an atomic D1 `INSERT OR IGNORE` keyed by event ID. This is stronger evidence of cross-request persistence than a KV read-then-write pattern. It is source evidence only here; neither D1 races nor deployed binding availability were exercised. A rate-limiter's fail-open policy should also not be confused with the separate replay-verification path.

## Missing policy and broken policy currently converge

Both tiers deny when no ACL is resolved. That statement is weaker than saying they deny when policy cannot be read reliably.

The native [storage resolver](../../../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs) walks towards the root. Some parser bound errors are propagated, but other malformed documents and storage read failures can be skipped while seeking a broader policy. The probe installs a permissive root ACL and a malformed resource-specific ACL; the actual resolver falls back to the root and grants the requested Read.

The edge resolver similarly seeks the first parseable R2 sidecar and falls back to legacy whole-pod KV policy if none resolves. It treats oversized/unparseable documents and failed object reads as misses. R2-first precedence fixes the older problem where a stale KV record masked a valid specific grant. It does not distinguish absence from an unavailable or damaged restrictive policy.

**Gap:** policy lookup needs explicit missing, invalid and unavailable outcomes. A missing sidecar can legitimately inherit. A present but unreadable restrictive sidecar should not silently become permission from an ancestor. Deleting a specific grant can also expose inherited rights or legacy fallback, so revocation must be assessed against the resulting effective policy, not merely successful deletion. This pass has not simulated the full delegation/revocation transaction.

## Edge response caching loses the policy distinction

After its WAC check, the [edge resource handler](../../../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs) calls `add_cache_control` on successful responses. That helper accepts only the resource path. It emits `public, max-age=31536000, immutable` for `/media/` paths and `public, max-age=300, must-revalidate` otherwise. It does not ask whether the ACL allowed anonymous access or required an authenticated principal. Ordinary byte responses vary on `Accept`, not on the requesting identity; ACL responses also call the caching helper.

This is a source-established mismatch between authorisation and response metadata. A private successful response is advertised as publicly cacheable. Actual disclosure depends on the caches and delivery configuration, which were not exercised here. Likewise, a revoked grant cannot retract bytes already delivered; response caching policy determines how long stored responses can remain useful after the authority changes.

**Gap:** derive cache policy from effective visibility and representation, then verify private reads, identity changes and revocation through the actual delivery path. Content addressing can establish immutable bytes; it does not establish public access rights.

## Storage integrity is stronger than provenance atomicity

Current native [filesystem storage](../../../solid-pod-rs/crates/solid-pod-rs/src/storage/fs.rs) writes a temporary file, syncs it and renames it into place. Metadata is published before the body and tagged with the future ETag so readers can ignore mismatched metadata. This is concrete protection against torn content/metadata observations. It does not by itself prove power-loss recovery, replicated durability or backup restoration; those require different checks.

The native server calls `git_mark_write` after successful content mutations. The helper skips policy and provenance sidecars, containers, absent data roots and non-git pods. Git marking and optional anchoring then run as follow-up work. A mark failure is logged after the storage write has already succeeded. Thus a successful content response is not an atomic receipt that its provenance commit or external anchor also succeeded.

Agentbox's path patch addresses one cause of missing marks, but cannot make that multi-step operation transactional. The operator needs to distinguish content stored, local provenance committed and external anchor confirmed, and have a repair path for partial success. A restore drill should prove both resource recovery and the ability to explain their provenance afterwards.

Pods are a substantive part of the architecture, with shared policy primitives and targeted regression tests. The next cross-estate proof is a complete journey: grant a scoped principal access, perform an authenticated mutation, record its provenance, revoke access, and verify denial across cache, restart and tier boundaries. That remains open, alongside the wider pod features explicitly scoped above.

## Operative storage decision pack

All seven pod ADRs now carry closeout acceptance conditions. [Current source revalidation](evidence/pod-closeout-snapshot.json) matches the seven previously inspected pod source files; the earlier 46 targeted tests and isolated probes retain their original scope. Newly inspected OIDC discovery and endpoint configuration are source evidence only. The full grant/mutate/provenance/revoke/restart journey is still open. The deliberate WAC choice, default-off git and replay adapter seam remain decisions in their own right, not substitutes for that journey.
