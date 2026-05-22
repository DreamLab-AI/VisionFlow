# Pod Tier Capability Matrix

The VisionFlow ecosystem deploys Solid pods across four deployment tiers, each
with different capability envelopes. This document is the single reference for
what works where.

## Tier Definitions

| Tier | Identifier | Runtime | Storage | Default consumers |
|------|-----------|---------|---------|-------------------|
| **CF Workers** | `nostr-bbs-pod-worker` | Cloudflare Workers (`wasm32-unknown-unknown`) | R2 + Workers KV | `nostr-rust-forum`, `dreamlab-ai-website` |
| **Embedded agentbox** | `solid-pod-rs` library | Tokio (Linux container) | Host filesystem (`/var/lib/solid`) | agentbox sovereign mesh |
| **Native server** | `solid-pod-rs-server` binary | Tokio (standalone) | Filesystem, memory, or S3 | Standalone operators, Cloudflare Tunnel exposure |
| **Git-capable** | `solid-pod-rs-server --features git` | Tokio (standalone) | Filesystem + git repository | Developer-cohort users via admin provisioning |

## Capability Matrix

| Capability | CF Workers | Embedded agentbox | Native server | Git-capable |
|---|:---:|:---:|:---:|:---:|
| **Protocol** | | | | |
| LDP (Basic Containers) | Y | Y | Y | Y |
| Content negotiation (Turtle, JSON-LD, N-Triples) | Y | Y | Y | Y |
| PATCH (N3, SPARQL-Update, JSON Patch) | Y | Y | Y | Y |
| Conditional requests (ETag, If-Match) | Y | Y | Y | Y |
| Container creation via PUT | Y | Y | Y | Y |
| HTTP COPY | Y | Y | Y | Y |
| Glob GET (`/folder/*`) | Y | Y | Y | Y |
| Range requests (RFC 7233) | -- | Y | Y | Y |
| `.meta` sidecars | Y | Y | Y | Y |
| **Access Control** | | | | |
| WAC (full spec) | Y | Y | Y | Y |
| `acl:agent` / `acl:agentClass` / `acl:agentGroup` | Y | Y | Y | Y |
| `acl:default` inheritance | Y | Y | Y | Y |
| `acl:origin` enforcement | -- | Y | Y | Y |
| WAC 2.0 conditions | -- | Y | Y | Y |
| `WAC-Allow` header on 403 | Y | Y | Y | Y |
| PaymentCondition (HTTP 402 Web Ledgers) | -- | -- | Y | Y |
| **Identity and Auth** | | | | |
| NIP-98 (Schnorr/secp256k1) | Y | Y | Y | Y |
| `did:nostr` resolution (Tier 1 + Tier 3) | Y | Y | Y | Y |
| Solid-OIDC + DPoP | -- | Y | Y | Y |
| WebID cross-verification (`alsoKnownAs`) | Y | Y | Y | Y |
| WebAuthn passkeys | -- | -- | Y | Y |
| Schnorr SSO (NIP-07) | -- | -- | Y | Y |
| `did:key` support | -- | -- | Y | Y |
| Embedded identity provider (`solid-pod-rs-idp`) | -- | -- | Y | Y |
| **Notifications** | | | | |
| WebSocketChannel2023 (Solid Notifications 0.2) | -- | Y | Y | Y |
| WebhookChannel2023 (RFC 9421 signed) | -- | Y | Y | Y |
| Legacy `solid-0.1` WebSocket | -- | Y | Y | Y |
| `Updates-Via` header | -- | Y | Y | Y |
| **Federation** | | | | |
| ActivityPub (inbox/outbox, HTTP Signatures) | -- | -- | Y | Y |
| `/.well-known/apps` (JSS #464) | -- | -- | Y | Y |
| NIP-05 pod-resident endpoint | -- | -- | Y | Y |
| **Git** | | | | |
| Git auto-init at provisioning | -- | -- | -- | Y |
| Git HTTP smart transport (clone/push/pull) | -- | -- | -- | Y |
| Pods as app repositories | -- | -- | -- | Y |
| **Admin and Operations** | | | | |
| CORS allowlist | -- | -- | Y | Y |
| PSK admin provisioning (`/_admin/provision`) | -- | -- | Y | Y |
| Cohort-based access control | -- | -- | Y | Y |
| Per-pod storage quota (`.quota.json`) | -- | Y | Y | Y |
| Rate limiting | Y (CF) | -- | Y | Y |
| Webhook signing (RFC 9421 Ed25519) | -- | Y | Y | Y |
| SSRF guard (private IP, DNS rebinding) | -- | Y | Y | Y |
| Path traversal guard | -- | Y | Y | Y |
| Observability (spans, logs, metrics) | CF analytics | Y | Y | Y |
| **Storage Backend** | | | | |
| Cloudflare R2 | Y | -- | -- | -- |
| Workers KV | Y | -- | -- | -- |
| Host filesystem (`fs-backend`) | -- | Y | Y | Y |
| In-memory (`memory-backend`) | -- | Y | Y | Y |
| S3-compatible (`s3-backend`) | -- | -- | Y | Y |
| Custom backend (implement `Storage` trait) | -- | -- | Y | Y |

**Legend:** Y = supported, -- = not available.

## Tier Selection by Substrate

| Substrate | Default tier | Rationale |
|-----------|-------------|-----------|
| `nostr-rust-forum` | CF Workers | Edge deployment; R2/KV/D1/Durable Objects ecosystem; global PoPs |
| `dreamlab-ai-website` | CF Workers | Same CF Workers deployment as the forum |
| `agentbox` (sovereign mesh) | Embedded agentbox | Full Linux runtime; `solid-pod-rs` library at `:8484`; WAC against `did:nostr` |
| Standalone operators | Native server | `solid-pod-rs-server` binary with config.json; full feature surface |
| Developer cohort (DreamLab) | Git-capable | Native server + `--features git`; Cloudflare Tunnel at `pods-native.dreamlab-ai.com` |

## Why the CF Workers Tier is Limited

Cloudflare Workers compile to `wasm32-unknown-unknown`. The runtime has:

- **No `tokio::process`** -- cannot spawn subprocesses (no `git init`, no CGI).
- **No native filesystem** -- storage is R2 (object store) and KV (key-value), not POSIX.
- **No Tokio runtime** -- async is handled by the Workers runtime, not by Tokio.
- **30-second CPU budget** -- hostile to expensive operations like packfile generation.

These constraints are structural, not configuration gaps. ADR-087 documents the
wasm32/Tokio wall; ADR-089 documents the git-specific unavailability.

## Migration Path: CF Workers to Native Tier

Moving a user's pod from the CF Workers tier to the native (or git-capable) tier
is an admin-driven process, not a self-service operation.

### Prerequisites

1. A running `solid-pod-rs-server` instance (with `--features git` if git is needed).
2. A Cloudflare Tunnel or equivalent TLS termination exposing the native server.
3. PSK admin key shared between the auth-worker and the native server.
4. User's cohort added to the `allowlist_cohorts` in `dreamlab.toml`.

### Steps

1. **Provision the native pod.**
   Admin triggers `POST /api/native-pod/provision` via the admin panel.
   The auth-worker forwards to the native server's `/_admin/provision/{pubkey}`
   with the PSK header. The server creates the pod directory, writes a WAC `.acl`
   granting the pubkey owner access, and runs `git init -b main`.

2. **Export data from CF tier.**
   Use the R2 API or `wrangler r2 object get` to download all objects under the
   user's R2 prefix (`{pubkey}/`). Each object maps to an LDP resource.

3. **Import data to native tier.**
   `PUT` each resource to the native pod URL using NIP-98 auth. ACL sidecars
   must be transferred separately. If the native pod is git-enabled, each PUT
   creates a commit.

4. **Update WebID.**
   The user's WebID profile card must update `pod_base_url` from the CF pod URL
   to the native pod URL. Verifiers following the WebID will then resolve to the
   native tier.

5. **Decommission CF pod (optional).**
   Remove or tombstone the R2 prefix. The pod-worker will return 404 for that
   pubkey.

### Limitations

- There is no automated bulk migration tool today. Each resource must be
  individually PUT to the native tier.
- Git history starts fresh at migration time. The CF tier has no version history
  to carry over.
- NIP-05 resolution may need reconfiguration if the user's name was registered
  in the forum's D1/KV store and the native pod now serves its own
  `/.well-known/nostr.json` (see ADR-086).

## Cross-Tier Authentication

Both tiers use NIP-98 Schnorr signatures for authentication. The signature is
self-contained -- it covers the URL, HTTP method, and optional body hash. Each
tier verifies independently from the same secp256k1 pubkey. No token exchange,
no shared session store, and no cross-tier RPC is required (ADR-093 section 2.2).

The same passkey-derived private key signs requests to both tiers. From the
user's perspective, authentication is seamless across tiers.

## References

- [solid-pod-rs README](https://github.com/DreamLab-AI/solid-pod-rs) -- full crate documentation and feature list
- [ADR-087](https://github.com/DreamLab-AI/nostr-rust-forum/blob/main/docs/adr/ADR-087-cf-workers-portable-cores.md) -- CF Workers portable cores (wasm32/Tokio gap)
- [ADR-089](https://github.com/DreamLab-AI/nostr-rust-forum/blob/main/docs/adr/ADR-089-git-pods-cf-workers-limitation.md) -- Git-pods unavailability on CF Workers
- [ADR-093](https://github.com/DreamLab-AI/nostr-rust-forum/blob/main/docs/adr/ADR-093-native-pod-mesh.md) -- Native pod mesh: hybrid two-tier architecture
- [ADR-086](https://github.com/DreamLab-AI/nostr-rust-forum/blob/main/docs/adr/ADR-086-nip05-pod-federation.md) -- NIP-05 pod federation
- [agentbox.toml](https://github.com/DreamLab-AI/agentbox/blob/main/agentbox.toml) -- pod adapter and git configuration
