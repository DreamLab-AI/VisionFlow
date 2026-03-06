# Architecture Flow Document: did:nostr Identity Integration

**ADR-027** | Status: Accepted | Date: 2026-02-21

## Context

JSS v0.0.46 implements NIP-98 HTTP auth returning `did:nostr:<hex>` identities. Upstream v0.0.86 adds Schnorr SSO, passkey auth, DID resolution, and 14+ security fixes. The system needs full did:nostr DID document serving, PodKey-only recommendation, auto-provisioning, and cross-identity normalization.

## Decision

Implement a 7-phase upgrade (Phase 0-6) using file-level upstream sync followed by additive modules. No breaking changes to existing NIP-98 auth flow.

## Architecture Flow

### Authentication Pipeline (Enhanced)

```
HTTP Request
  |
  +-- Authorization: Nostr <base64(kind-27235-event)>
  |     may include ["webid", "did:nostr:<hex>"] tag
  |
  v
src/auth/token.js:getWebIdFromRequestAsync()
  |
  +--> [1] Solid-OIDC (DPoP)  -- existing, unchanged
  +--> [2] NIP-98 Schnorr     -- enhanced with webid tag (Phase 5)
  +--> [3] Bearer token        -- existing, unchanged
  |
  v
src/auth/identity-normalizer.js (Phase 6, NEW)
  |
  +--> did:nostr:<pk> -> extract canonical pubkey
  +--> WebID (HTTP)   -> dereference -> extract nostr:pubkey
  +--> Normalize to canonical form for WAC matching
  |
  v
src/wac/checker.js:checkAccess()
  |
  +--> isAgentAuthorized() -- enhanced with cross-identity matching
  |
  v
Granted / Denied
```

### DID Document Resolution Flow (Phase 2)

```
GET /.well-known/did/nostr/<pubkey>.json
  |
  v
src/did/resolver.js:handleDidResolve()
  |
  +--> Validate pubkey format (64-char hex)
  +--> Construct Tier 1 DID Document (offline, deterministic)
  +--> Lookup pod owner by did:nostr
  |     +--> Found: add alsoKnownAs WebID (Tier 3)
  |     +--> Not found: return Tier 1 only
  +--> Set headers: Nostr-Timestamp, Cache-Control, Content-Type
  |
  v
200 OK: application/did+json
```

### Pod Auto-Provisioning Flow (Phase 4)

```
POST /idp/interaction/:uid/schnorr-login
  |
  v
handleSchnorrLogin() -- verifyNostrAuth()
  |
  +--> Account found? -> proceed with login
  +--> Account NOT found?
        |
        v
      auto-provision.js:provisionPodForNostr()
        |
        +--> Derive pod name (first 8 hex chars of pubkey)
        +--> Create pod directory structure
        +--> Generate WebID profile with nostr:pubkey + owl:sameAs
        +--> Generate Tier 1 DID document at /.well-known/did.json
        +--> Set ACL: owner = did:nostr:<hex>
        +--> Create IdP account linked to did:nostr
        +--> Continue with login flow
```

### Identity Binding Model

```
+-------------------------------------------+
|           Nostr Keypair (secp256k1)        |
|  Private Key -> Signs NIP-98 events       |
|  Public Key  -> did:nostr:<64-hex>        |
+-------------------------------------------+
           |                    |
           v                    v
+-------------------+  +-------------------+
| Solid Pod         |  | DID Document      |
| /profile/card#me  |  | /.well-known/     |
|                   |  | did/nostr/<pk>    |
| owl:sameAs        |  |                   |
|   did:nostr:<pk>  |  | alsoKnownAs       |
| nostr:pubkey      |  |   WebID URI       |
|   <64-hex>        |  |                   |
+-------------------+  +-------------------+
```

## Consequences

- Backwards compatible: existing NIP-98 clients unchanged
- Additive: new modules don't modify existing auth pipeline
- PodKey becomes the canonical client integration point
- Key rotation remains unsupported at DID level (mitigated by NIP-41 plan)
