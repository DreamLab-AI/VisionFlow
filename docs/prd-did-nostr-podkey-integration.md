# PRD: did:nostr Identity Integration with PodKey Authentication

**Status**: Draft
**Date**: 2026-02-21
**Authors**: Research Swarm (5-agent hierarchical-mesh)
**Scope**: End-to-end identity unification — Nostr keypair to Solid Pod

---

## 1. Executive Summary

This PRD defines the upgrade path from the current NIP-98 HTTP authentication to full `did:nostr` identity integration, with PodKey as the sole recommended browser extension, unifying frontend authentication with backend Solid Pod authorization.

### Current State

- JSS v0.0.46 (vendored), 40 versions behind upstream v0.0.86
- NIP-98 auth implemented in `src/auth/nostr.js`, already produces `did:nostr:<hex>` identities
- WAC checker already accepts `did:nostr:` URIs in ACL `acl:agent` values
- Upstream `add-schnorr-sso` branch (merged to gh-pages) adds NIP-07 browser login but references Alby and nos2x alongside PodKey
- PodKey already constructs `did:nostr:<hex>` natively and injects NIP-98 headers transparently

### Target State

- JSS synced to upstream v0.0.86 with all security fixes
- PodKey as sole recommended extension (Alby/nos2x references removed)
- Full `did:nostr` DID document resolution (offline-first, relay-enhanced)
- Bidirectional identity binding: WebID profile <-> did:nostr DID document
- Auto-provisioning: first NIP-98 auth creates pod + WebID + ACLs
- `webid` tag in NIP-98 events for explicit DID assertion per HTTP Schnorr Auth spec

### Why This Matters

The NIP-98 pubkey IS the did:nostr identifier. The upgrade is structurally trivial — the cryptographic foundation already exists. What's missing is the identity lifecycle: provisioning, profile binding, DID document serving, and the removal of OIDC dependency for Nostr-native users.

---

## 2. Problem Statement

### 2.1 Extension Fragmentation

The current SSO login page (`src/idp/views.js` on add-schnorr-sso) recommends three extensions:

```javascript
alert('No Schnorr signer found. Please install a NIP-07 compatible extension like Podkey, nos2x, or Alby.');
```

This creates confusion. Alby is a Lightning wallet that happens to support NIP-07 — it does not understand Solid Pods. nos2x is unmaintained. PodKey is purpose-built for did:nostr + Solid authentication with transparent NIP-98 injection.

### 2.2 Identity Gap

The current system authenticates via NIP-98 but does not:
- Auto-provision pods for new Nostr identities
- Generate WebID profiles with `nostr:pubkey` RDF predicate
- Serve did:nostr DID documents
- Support the `webid` tag from the HTTP Schnorr Auth spec
- Provide bidirectional identity binding (WebID <-> DID)

### 2.3 Version Drift

JSS is 40 versions behind upstream. Critical gaps:
- 14+ security fixes missing (path traversal, DPoP replay, SSRF protection)
- No passkey/WebAuthn support
- No single-user mode, SolidOS UI, range requests, live reload
- Native module dependencies (bcrypt, better-sqlite3) instead of pure JS (bcryptjs, sql.js)

---

## 3. Technical Analysis

### 3.1 Current NIP-98 Auth Flow (What Already Works)

```
HTTP Request
  |
  +-- Authorization: Nostr <base64(kind-27235-event)>
  |
  v
src/auth/token.js:getWebIdFromRequestAsync()
  |
  +--> src/auth/nostr.js:verifyNostrAuth()
  |      |-- Decode base64 -> JSON event
  |      |-- Validate kind == 27235
  |      |-- Check created_at (+-60s)
  |      |-- Verify URL tag matches request
  |      |-- Verify method tag matches
  |      |-- Optional: verify payload SHA-256
  |      |-- Validate pubkey (64-char hex)
  |      |-- nostr-tools.verifyEvent() [Schnorr/BIP-340]
  |      +-- Return {webId: "did:nostr:<pubkey>"}
  |
  v
src/auth/middleware.js:authorize()
  |
  +--> src/wac/checker.js:checkAccess()
  |      +-- acl:agent string comparison against did:nostr:<pubkey>
  |
  v
Granted / Denied
```

Key insight: `pubkeyToDidNostr()` at `src/auth/nostr.js:112-114` already performs the identity transformation:

```javascript
export function pubkeyToDidNostr(pubkey) {
  return `did:nostr:${pubkey.toLowerCase()}`;
}
```

### 3.2 PodKey Architecture (What the Extension Does)

PodKey (v0.0.7, Chrome MV3) operates in three contexts:

| Context | File | Role |
|---------|------|------|
| Service Worker | `src/background.js` | Key management, Schnorr signing, NIP-98 creation |
| Content Script | `src/injected.js` | Bridge between page and extension via CustomEvents |
| Page Context | `src/nostr-provider.js` | `window.nostr` NIP-07 API |
| Page Context | `src/nip98-interceptor.js` | Transparent fetch/XHR interception |

PodKey's differentiator vs Alby:

| Capability | PodKey | Alby |
|-----------|--------|------|
| Transparent NIP-98 injection | Patches fetch/XHR automatically | App must call signEvent manually |
| 401 auto-retry | Built-in | None |
| Solid server auto-detection | Hardcoded domain patterns | None |
| Kind 27235 fast-path | Auto-sign without prompt for trusted origins | Prompts for every event |
| did:nostr display | Native in UI | None |
| Zero-config for Solid apps | Standard fetch() gains auth | Requires Nostr SDK integration |

Crypto stack: `@noble/secp256k1` v3.0.0 + `@noble/hashes` v1.8.0 (BIP-340 Schnorr).

### 3.3 did:nostr DID Document Structure

Per the spec (v0.0.10, W3C Nostr CG draft), three resolution tiers:

**Tier 1 — Minimal (offline, deterministic from pubkey alone, MANDATORY)**:

```json
{
  "@context": ["https://w3id.org/did", "https://w3id.org/nostr/context"],
  "id": "did:nostr:<64-hex-pubkey>",
  "type": "DIDNostr",
  "verificationMethod": [{
    "id": "did:nostr:<pubkey>#key1",
    "type": "Multikey",
    "controller": "did:nostr:<pubkey>",
    "publicKeyMultibase": "fe70102<pubkey>"
  }],
  "authentication": ["#key1"],
  "assertionMethod": ["#key1"]
}
```

**Tier 2 — Enhanced (relay service endpoints)**:
Adds `service` array with `"type": "Relay"` entries pointing to `wss://` relay URIs.

**Tier 3 — Complete (profile + social graph + WebID binding)**:
Adds `profile` (from kind 0), `follows` (from kind 3), and `alsoKnownAs` linking to WebID.

### 3.4 Upstream JSS Delta (v0.0.46 -> v0.0.86)

**Auth-related additions** (all merged to gh-pages):
- Schnorr SSO with NIP-07 browser login (v0.0.78)
- Passkey/WebAuthn via `@simplewebauthn/server` v13 (v0.0.77)
- `did:nostr` resolution module `src/auth/did-nostr.js` (v0.0.58)
- DPoP jti replay attack prevention (v0.0.72)
- WebID-TLS support (v0.0.75)

**Security fixes** (14+ patches):
- Path traversal in git handler (CRITICAL)
- SSRF protection (HIGH)
- Rate limiting and token security (HIGH)
- DPoP replay prevention (HIGH)
- WebSocket ACL checks (HIGH)
- Safe JSON parsing in WAC (HIGH)

**Dependency modernization**:
- `bcrypt` -> `bcryptjs` (pure JS, no node-gyp)
- `better-sqlite3` -> `sql.js` (WASM, no native compilation)

**New features**: single-user mode, SolidOS UI, range requests, live reload, ActivityPub federation, invite-only registration, storage quotas.

**Merge risk**: LOW. `LOCAL_CHANGES.md` confirms no source code modifications to vendored copy. Only `Dockerfile.jss` and `.claude-flow/` are local additions.

---

## 4. Architecture

### 4.1 Target Identity Flow

```
+====================================================================+
|                    did:nostr + Solid Pod Integration                 |
+====================================================================+
|                                                                      |
|  BROWSER                                                             |
|  +----------------------------+                                      |
|  | PodKey Extension (MV3)     |                                      |
|  |                            |                                      |
|  | window.nostr (NIP-07)      |-- Signs kind 27235 events            |
|  | fetch/XHR interceptor      |-- Injects Authorization headers      |
|  | Auto-sign (trusted origins)|-- Zero user friction for Solid ops   |
|  | did:nostr:<hex> display    |-- Shows DID in popup UI              |
|  +----------------------------+                                      |
|             |                                                        |
|             | Authorization: Nostr <base64(event)>                   |
|             | Event includes: ["webid", "did:nostr:<hex>"]           |
|             v                                                        |
|  JSS SERVER                                                          |
|  +------------------------------------------------------------------+|
|  | Authentication Layer                                              ||
|  |   NIP-98 Verifier  (existing, enhanced with webid tag)           ||
|  |   Solid-OIDC       (existing, unchanged)                         ||
|  |   Passkey/WebAuthn (from upstream v0.0.77)                       ||
|  +------------------------------------------------------------------+|
|             |                                                        |
|             v                                                        |
|  +------------------------------------------------------------------+|
|  | Identity Normalizer (NEW)                                         ||
|  |   did:nostr:<pk> -> canonical pubkey                              ||
|  |   WebID (HTTP)   -> dereference -> extract nostr:pubkey           ||
|  |   Both forms     -> match against ACL agents                      ||
|  +------------------------------------------------------------------+|
|             |                                                        |
|             v                                                        |
|  +------------------------------------------------------------------+|
|  | WAC Authorization (existing, enhanced)                            ||
|  |   acl:agent <did:nostr:<hex>> -- direct DID reference             ||
|  |   acl:agent <https://...#me>  -- traditional WebID                ||
|  +------------------------------------------------------------------+|
|             |                                                        |
|             v                                                        |
|  +------------------------------------------------------------------+|
|  | Solid Pod (per user)                                              ||
|  |   /profile/card#me         -- WebID profile (Turtle)              ||
|  |   /.well-known/did.json    -- DID Document (JSON-LD)              ||
|  |   /private/                -- User data                           ||
|  |   /.acl                    -- Access control                      ||
|  +------------------------------------------------------------------+|
|                                                                      |
+======================================================================+
```

### 4.2 Pod Provisioning Flow (New)

```
User clicks "Create Pod" on JSS landing page
  |
  v
PodKey signs NIP-98 challenge (kind 27235)
  |
  v
POST /.pods with Authorization: Nostr <event>
  |
  v
Server verifies Schnorr sig, extracts pubkey
  |
  v
Server creates pod directory structure:
  /alice/
  /alice/profile/card              -- WebID profile with nostr:pubkey + owl:sameAs
  /alice/.well-known/did.json      -- Tier 1 DID Document
  /alice/inbox/
  /alice/public/
  /alice/private/
  /alice/settings/prefs.ttl
  /alice/.acl                      -- Owner: did:nostr:<hex>
  |
  v
Returns pod URL + WebID to client
```

### 4.3 WebID Profile with DID Binding

```turtle
@prefix foaf:  <http://xmlns.com/foaf/0.1/> .
@prefix solid: <http://www.w3.org/ns/solid/terms#> .
@prefix pim:   <http://www.w3.org/ns/pim/space#> .
@prefix nostr: <https://w3id.org/nostr/vocab#> .

<#me>
    a foaf:Person ;
    foaf:name "Alice" ;
    pim:storage </alice/> ;
    pim:preferencesFile </alice/settings/prefs.ttl> ;
    owl:sameAs <did:nostr:124c0fa99407182ece5a24fad9b7f6674902fc422843d3128d38a0afbee0fdd2> ;
    nostr:pubkey "124c0fa99407182ece5a24fad9b7f6674902fc422843d3128d38a0afbee0fdd2" .
```

Bidirectional binding is completed when the user's Nostr kind 0 profile includes:

```json
{
  "name": "Alice",
  "nip05": "alice@pod.example",
  "alsoKnownAs": ["https://pod.example/alice/profile/card#me"]
}
```

### 4.4 NIP-98 Enhanced with webid Tag

Per the HTTP Schnorr Auth companion spec, the signed event gains a `webid` tag:

```json
{
  "kind": 27235,
  "pubkey": "124c0fa99407182ece5a24fad9b7f6674902fc422843d3128d38a0afbee0fdd2",
  "created_at": 1708300000,
  "tags": [
    ["u", "https://pod.example/alice/private/data.ttl"],
    ["method", "GET"],
    ["webid", "did:nostr:124c0fa99407182ece5a24fad9b7f6674902fc422843d3128d38a0afbee0fdd2"]
  ],
  "content": "",
  "sig": "<schnorr-signature>"
}
```

Server verification adds one step: confirm `webid` tag's DID matches the event's `pubkey`.

---

## 5. Implementation Plan

### Phase 0: Upstream Sync (Prerequisite)

**Objective**: Bring JSS from v0.0.46 to v0.0.86.

**Tasks**:
1. File-level sync using existing `scripts/sync-jss-upstream.sh` targeting `jss-upstream/gh-pages`
2. Preserve local additions: `Dockerfile.jss`, `.claude-flow/`, `LOCAL_CHANGES.md`
3. `npm install` (dependency changes: bcryptjs, sql.js, @simplewebauthn/server)
4. Run test suite: `npm test`
5. Verify Docker build: `docker build -f Dockerfile.jss .`
6. Cherry-pick `fix/oidc-cors-all-clients` (3 commits, DPoP CORS fixes)

**Risk**: LOW — no source modifications in local copy.

**Gains**:
- 14+ security fixes
- `src/auth/did-nostr.js` (did:nostr resolution module)
- Schnorr SSO routes (`/idp/interaction/:uid/schnorr-login`)
- Passkey/WebAuthn support
- Pure JS dependencies (no node-gyp)

### Phase 1: PodKey-Only Recommendation

**Objective**: Remove Alby and nos2x references, recommend only PodKey.

**Files to modify** (post-upstream-sync):
- `src/idp/views.js` — Change alert text and login button label
- `README.md` — Update authentication section
- `docs/` — Update any extension references

**Changes**:

```javascript
// BEFORE (add-schnorr-sso branch):
alert('No Schnorr signer found. Please install a NIP-07 compatible extension like Podkey, nos2x, or Alby.');

// AFTER:
alert('No signer found. Please install PodKey: https://github.com/JavaScriptSolidServer/podkey');
```

Login button text:
```javascript
// BEFORE:
"Sign in with Schnorr"

// AFTER:
"Sign in with PodKey"
```

**Risk**: LOW — UI text changes only.

### Phase 2: did:nostr DID Document Serving

**Objective**: Serve DID Documents for pod owners at `/.well-known/did/nostr/<pubkey>.json`.

**New module**: `src/did/resolver.js`

**Responsibilities**:
1. Route handler for `GET /.well-known/did/nostr/<pubkey>.json`
2. Construct Tier 1 (minimal) DID Document from pubkey
3. Enrich to Tier 2 if relay config is available
4. Add `alsoKnownAs` pointing to the user's WebID if pod exists
5. Set response headers: `Nostr-Timestamp`, `Cache-Control`, `Content-Type: application/did+json`

**Implementation approach**:
```javascript
export function resolveDIDNostr(pubkey) {
  // Tier 1: deterministic from pubkey (always available)
  const doc = {
    "@context": ["https://w3id.org/did", "https://w3id.org/nostr/context"],
    id: `did:nostr:${pubkey}`,
    type: "DIDNostr",
    verificationMethod: [{
      id: `did:nostr:${pubkey}#key1`,
      type: "Multikey",
      controller: `did:nostr:${pubkey}`,
      publicKeyMultibase: `fe70102${pubkey}`
    }],
    authentication: ["#key1"],
    assertionMethod: ["#key1"]
  };

  // Tier 3: if pod exists, add alsoKnownAs
  const webId = lookupWebIdByDid(`did:nostr:${pubkey}`);
  if (webId) {
    doc.alsoKnownAs = [webId];
  }

  return doc;
}
```

**Risk**: LOW — additive route, no existing behavior changed.

### Phase 3: Enhanced WebID Profile Generation

**Objective**: Auto-generate WebID profiles with `nostr:pubkey` and `owl:sameAs` predicates.

**File to modify**: `src/webid/profile.js`

**Changes**:
- Add `nostr:pubkey` triple to profile template
- Add `owl:sameAs <did:nostr:<hex>>` triple
- Include `nostr:` prefix in `@prefix` declarations

**Risk**: LOW — extends existing profile generation.

### Phase 4: Auto-Provisioning on First Auth

**Objective**: When a Nostr identity authenticates for the first time, auto-create their pod.

**Current behavior** (add-schnorr-sso branch):
```javascript
if (!account) {
  return reply.code(403).send({
    success: false,
    error: 'No account linked to this identity.'
  });
}
```

**Target behavior**:
```javascript
if (!account) {
  // Auto-provision pod for new Nostr identity
  const podName = derivePodName(pubkey); // hex prefix or NIP-05
  const pod = await createPod(podName, `did:nostr:${pubkey}`);
  const account = await createAccount({
    webId: pod.webId,
    did: `did:nostr:${pubkey}`
  });
  // Continue with login flow
}
```

**Pod naming strategy**:
1. If NIP-05 is available (query relay for kind 0): use `alice` from `alice@example.com`
2. Otherwise: use first 8 chars of hex pubkey: `124c0fa9`

**Risk**: MEDIUM — new provisioning logic, needs rate limiting and abuse prevention.

### Phase 5: webid Tag Support in NIP-98 Verification

**Objective**: Support the `["webid", "did:nostr:<hex>"]` tag from HTTP Schnorr Auth spec.

**File to modify**: `src/auth/nostr.js` (verifyNostrAuth function)

**Changes**:
1. Extract `webid` tag from event tags if present
2. Validate that the DID in the `webid` tag matches `did:nostr:<event.pubkey>`
3. Return the `webid` value as the authenticated identity (instead of constructing it)
4. Fall back to current `pubkeyToDidNostr(pubkey)` if no `webid` tag

**PodKey changes**: Update `src/background.js` `createNip98AuthHeader()` to include `["webid", "did:nostr:<pubkey>"]` tag.

**Risk**: LOW — backwards-compatible (tag is optional).

### Phase 6: Identity Normalizer

**Objective**: Unified identity matching across did:nostr and WebID in WAC checks.

**New module**: `src/auth/identity-normalizer.js`

**Responsibilities**:
1. Given an authenticated identity (did:nostr or WebID), extract the canonical pubkey
2. Given an ACL agent URI (did:nostr or WebID), extract the canonical pubkey
3. Match by pubkey equality, not string equality

This enables:
- A user authenticated via `did:nostr:<pk>` to access resources where ACL references their WebID
- A user authenticated via Solid-OIDC (WebID) to access resources where ACL references their `did:nostr`

**Risk**: MEDIUM — changes authorization matching semantics.

---

## 6. Security Considerations

### 6.1 Key Rotation

did:nostr has no update/revoke mechanism — the DID IS the pubkey. Key compromise requires a new identity.

**Mitigation strategy (NIP-41 + pod-level registry)**:
1. User pre-registers a recovery key via Nostr kind:1776 event
2. On compromise: publish kind:1777 migration event with 60-day contestation
3. Pod provisioning service monitors for kind:1777 events
4. Pod updates owner DID in all ACL files
5. WebID profile `owl:sameAs` updated to new DID
6. Old DID references in external ACLs become stale (inherent limitation)

### 6.2 Replay Protection

NIP-98's 60-second timestamp window is acceptable for reads. For mutating operations (PUT, POST, DELETE):

**Enhancement**: Server issues nonce via `WWW-Authenticate: Nostr realm="pod" nonce="<random>"`. Client includes `["nonce", "<server-nonce>"]` tag in the kind 27235 event. This upgrades from timestamp-only to challenge-response.

### 6.3 Cross-Origin Security

- NIP-07 (`window.nostr`) never exposes private keys to web pages
- The `u` tag in signed events exactly matches the request URL, preventing cross-site token theft
- PodKey's per-origin trust model prevents unauthorized signing
- CORS configuration must accept `Authorization: Nostr` scheme in preflight responses

### 6.4 PodKey-Specific Concerns

| Concern | Current State | Recommendation |
|---------|--------------|----------------|
| Key storage | Plaintext in chrome.storage.local | Add passphrase encryption layer |
| Permission prompts | Auto-approved (MV3 limitation) | Implement chrome.notifications with action buttons |
| Solid server detection | Hardcoded domain list | Dynamic detection via `/.well-known/solid` or `WWW-Authenticate` header |
| Bech32 encoding | Stub (returns prefixed hex) | Implement proper NIP-19 bech32 |
| NIP-04 encryption | Not implemented | Implement for encrypted pod-to-pod messaging |

### 6.5 Hardcoded Key (Existing Issue)

`JavaScriptSolidServer/clock-updater.mjs:11` contains a committed hex private key (`3f188544fb81...`). This is a demo script but should be rotated/removed.

---

## 7. Migration Path

### For Existing Users (NIP-98 without DID)

No breaking changes. The server continues accepting `Authorization: Nostr <event>` without the `webid` tag. Internally, `pubkeyToDidNostr()` constructs the DID as before.

### For New Users

1. Install PodKey browser extension
2. Generate or import keypair (PodKey popup)
3. Navigate to JSS instance
4. PodKey auto-signs NIP-98 challenge
5. Server auto-provisions pod (Phase 4)
6. User accesses their Solid Pod — all subsequent requests auto-authenticated by PodKey

### For Frontend Applications

Zero changes required. PodKey patches `fetch()` and `XMLHttpRequest` at the page level. Any Solid app using standard `fetch()` calls gains NIP-98 authentication transparently.

---

## 8. Testing Strategy

### Unit Tests

- DID Document generation (Tier 1/2/3) from pubkey
- `webid` tag extraction and validation in NIP-98 verifier
- Identity normalizer (did:nostr <-> WebID matching)
- WebID profile generation with `nostr:pubkey` predicate
- Pod auto-provisioning from Nostr identity

### Integration Tests

- Full auth flow: PodKey signs -> JSS verifies -> WAC grants -> resource returned
- Pod provisioning: first auth -> pod created -> subsequent auth -> pod accessed
- DID Document serving: `GET /.well-known/did/nostr/<pubkey>.json` returns valid document
- Cross-identity ACL: did:nostr in ACL, WebID authenticated (and vice versa)

### Existing Test Infrastructure

- `test-nostr-auth.js` — Standalone NIP-98 auth test (adapt for did:nostr)
- `test-git-nostr-auth.js` — Git push/pull with Nostr auth
- `test/wac.test.js:84-99` — WAC parser test with `did:nostr:abc123` agent
- `test/auth.test.js` — Bearer token auth (extend for Nostr)

---

## 9. Dependencies

### Runtime

| Package | Version | Role |
|---------|---------|------|
| `nostr-tools` | ^2.19.4 | Schnorr verification, event handling, NIP-98 |
| `@noble/secp256k1` | ^3.0.0 | BIP-340 Schnorr (PodKey uses directly) |
| `@noble/hashes` | ^1.8.0 | SHA-256 for event IDs and payload hashing |
| `bcryptjs` | (from upstream) | Password hashing (replaces native bcrypt) |
| `sql.js` | (from upstream) | SQLite via WASM (replaces native better-sqlite3) |
| `@simplewebauthn/server` | ^13.2.2 | Passkey/WebAuthn (from upstream) |

### Development

| Package | Role |
|---------|------|
| `did-resolver` | Standard DID resolution interface for testing |

---

## 10. Success Metrics

| Metric | Target |
|--------|--------|
| PodKey as sole extension in all UI/docs | 100% of references |
| Upstream sync to v0.0.86 | All 14+ security fixes applied |
| DID Document resolution | Tier 1 (offline) for all pod owners |
| Auto-provisioning | Pod created on first Nostr auth |
| Auth backwards compatibility | Existing NIP-98 clients work unchanged |
| WAC cross-identity matching | did:nostr and WebID interchangeable in ACLs |

---

## 11. Phase Summary

| Phase | Scope | Risk | Effort | Depends On |
|-------|-------|------|--------|------------|
| 0 | Upstream sync v0.0.46 -> v0.0.86 | LOW | MEDIUM | - |
| 1 | PodKey-only recommendation | LOW | LOW | Phase 0 |
| 2 | DID Document serving | LOW | MEDIUM | Phase 0 |
| 3 | Enhanced WebID profiles | LOW | LOW | Phase 0 |
| 4 | Auto-provisioning on first auth | MEDIUM | MEDIUM | Phase 0, 3 |
| 5 | webid tag in NIP-98 | LOW | LOW | Phase 0 |
| 6 | Identity normalizer | MEDIUM | MEDIUM | Phase 0, 2, 3 |

Phases 1-3 and 5 can execute in parallel after Phase 0 completes.
Phase 4 depends on Phase 3 (profile generation).
Phase 6 depends on Phases 2 and 3 (DID resolution + WebID profiles).

---

## 12. Open Questions

1. **Pod naming**: Should auto-provisioned pods use hex prefix (`124c0fa9/`) or require NIP-05 (`alice/`)? Hex is deterministic but not human-friendly.

2. **Key rotation UX**: When NIP-41 migration occurs, how does the user re-link their pod? Automatic (relay monitoring) or manual (re-auth)?

3. **Multi-pod**: Can one Nostr identity own multiple pods? Current model is 1:1.

4. **Relay configuration**: Should JSS act as a Nostr relay (per the design doc at `docs/design/nostr-relay-integration.md`) to self-host DID resolution?

5. **NIP-05 integration**: Should `/.well-known/nostr.json` be auto-generated for pod owners to enable `alice@pod.example` NIP-05 verification?

6. **Passkey + Nostr**: Can a passkey (WebAuthn credential) be linked to a Nostr identity as a second factor or backup auth method?

---

## Appendix A: Key File Locations (Post-Upstream-Sync)

| File | Role |
|------|------|
| `src/auth/nostr.js` | NIP-98 verification, `pubkeyToDidNostr()` |
| `src/auth/did-nostr.js` | did:nostr resolution (from upstream v0.0.58) |
| `src/auth/token.js` | Auth dispatcher (Solid-OIDC / NIP-98 / Bearer) |
| `src/auth/middleware.js` | Authorization middleware |
| `src/wac/checker.js` | WAC ACL evaluation |
| `src/idp/views.js` | Login page UI (Schnorr SSO button) |
| `src/idp/interactions.js` | SSO handlers (handleSchnorrLogin) |
| `src/idp/accounts.js` | Account storage and lookup |
| `src/webid/profile.js` | WebID profile generation |

## Appendix B: did:nostr Resolution Tiers

| Tier | Resolution | Network | Data |
|------|-----------|---------|------|
| 1 | Minimal (MANDATORY) | None (offline) | Pubkey, verification method |
| 2 | Enhanced | Nostr relays | + relay service endpoints |
| 3 | Complete | Relays + HTTP | + profile, follows, alsoKnownAs |

## Appendix C: References

- [did:nostr Method Specification v0.0.10](https://nostrcg.github.io/did-nostr/)
- [HTTP Authentication Using Schnorr Signatures](https://nostrcg.github.io/http-schnorr-auth/)
- [PodKey Extension](https://github.com/JavaScriptSolidServer/podkey)
- [JavaScriptSolidServer](https://github.com/JavaScriptSolidServer/JavaScriptSolidServer)
- [NIP-98 HTTP Auth](https://nips.nostr.com/98)
- [NIP-07 Browser Extension](https://github.com/nostr-protocol/nips/blob/master/07.md)
- [NIP-19 Bech32 Entities](https://nips.nostr.com/19)
- [NIP-41 Key Migration (Draft)](https://github.com/nostr-protocol/nips/pull/829)
- [BIP-340 Schnorr Signatures](https://bips.xyz/340)
- [Solid-OIDC Specification](https://solid.github.io/solid-oidc/)
- [Web Access Control (WAC)](https://solid.github.io/web-access-control-spec/)
- [Solid WebID Profile](https://solid.github.io/webid-profile/)
- [Nostr-OIDC (Melvin Carvalho)](https://melvincarvalho.github.io/nostr-oidc/)
- [W3C Nostr Community Group](https://www.w3.org/community/nostr/)
- [Solid Specification Issue #217 — DID Support](https://github.com/solid/specification/issues/217)
