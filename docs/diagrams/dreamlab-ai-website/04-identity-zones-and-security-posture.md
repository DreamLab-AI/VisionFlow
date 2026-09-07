---
id: DW-04
title: Identity, zones and security posture
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
adrs: [ADR-2006, ADR-2007, ADR-2008]
sources:
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
  - ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md
  - ../dreamlab-ai-website/docs/security/AUTHENTICATION.md
  - ../dreamlab-ai-website/src/lib/nostr.ts
  - ../dreamlab-ai-website/src/components/AIChatFab.tsx
  - ../dreamlab-ai-website/index.html
verified_commit: 9a3dd8830
---

## DW-04.1 Identity is raw-hex Schnorr, not a DID document
```mermaid
flowchart TB
    KEY["64-hex secp256k1 pubkey<br/>the only identity shape in the shipped surface"] --> GEN["getPublicKey(sk)<br/>src/lib/nostr.ts:98"]
    KEY --> AUTH["finalizeEvent({kind: KIND_NIP42_AUTH})<br/>src/lib/nostr.ts:31, 279-281, 636-638, 785-787"]
    AUTH --> CHALLENGE["NIP-42 relay-challenge AUTH events<br/>kind 22242, NOT NIP-98"]
    NEG["grep -c 'NIP-98' src/lib/nostr.ts -> 0"] -.-> CHALLENGE
    KEY --> METHODS["auth methods offered by the upstream kit:<br/>WebAuthn PRF passkeys, NIP-07 extension, raw private key"]
    METHODS --> KEY
```
- INVARIANT (`IDENTITY-zones.md:141-143`): identity in the shipped surface is the raw 64-hex secp256k1 pubkey; any move to emit or require a Multikey/DID document is a code change and a new ADR, not a docs edit.
- DOC-DRIFT within this repo's OWN docs: `docs/security/AUTHENTICATION.md` (`## NIP-98 HTTP Authentication`, line 200) documents NIP-98 as the upstream kit auth-worker's HTTP-API scheme (passkey/session auth to REST endpoints) — a genuinely different protocol from the NIP-42 relay-challenge flow this repo's own `src/lib/nostr.ts` implements for DM transport. `IDENTITY-zones.md:34-35`'s claim that "the string NIP-98 appears nowhere in nostr.ts" is true and specific to that one file; it does not mean NIP-98 is absent from the deployed system.

## DW-04.2 DID/Multikey convergence — documentation-only
```mermaid
flowchart LR
    LEGACY["legacy ADR-027<br/>did:nostr document form,<br/>Multikey prefix fe70102"] -.->|"two commits, docs + JSON-LD only"| COMMITS["d62ab40, 8d942d7<br/>own messages: 'no identity/key/npub/URN/ACL migration'"]
    COMMITS -.->|cites a binding spec that| MISSING["ADR-125-did-nostr-multikey-convergence.md<br/>DOES NOT EXIST in this repo — kit-owned"]
    LIVE["live code: raw-pubkey Schnorr auth, untouched"] -.->|no code path produces| DIDDOC["a fe70102 Multikey DID document"]
```
- `fe70102` is not a commit hash — it is the Multikey encoding prefix itself: base16-multibase `f` + secp256k1-pub multicodec varint `e701` + compressed-point `02` (`IDENTITY-zones.md:46-49`).
- ADR-027 is archived as "Deferred — kit-owned" (`IDENTITY-zones.md:56`); treat any claim that this repo emits `did:nostr` Multikey documents as false until a code path produces one (`IDENTITY-zones.md:128-130`).

## DW-04.3 Talk-to-AI — client-side Nostr DM, not an HTTP chat endpoint
```mermaid
sequenceDiagram
    autonumber
    participant U as visitor (browser)
    participant FAB as AIChatFab<br/>src/components/AIChatFab.tsx:443 sendQuestion
    participant DM as DmSession<br/>src/lib/nostr.ts:452
    participant W as nip17.wrapEvent<br/>src/lib/nostr.ts:196
    participant R as VITE_RELAY_URL<br/>publishGiftWrap, src/lib/nostr.ts:225
    participant OPEN as open relays<br/>VITE_REPLY_RELAYS: relay.damus.io, relay.primal.net
    U->>FAB: submits question
    FAB->>FAB: encodeQuestion({requestId, question, tier, identityHint: pubkey})<br/>AIChatFab.tsx:428-434
    Note right of FAB: identityHint is EXPLICITLY UNVERIFIED —<br/>no proof of possession, DM rides the ephemeral<br/>session key (AIChatFab.tsx:429-430 comment)
    FAB->>DM: session.sendQuestion(payload, JARVIS_PUBKEY)<br/>AIChatFab.tsx:443
    DM->>W: wrap kind-14 rumor to VITE_JARVIS_PUBKEY<br/>src/lib/nostr.ts:137,196
    W->>R: publish kind-1059 gift wrap<br/>resolves on relay OK-true only, never rejects
    Note right of R: primary relay's whitelist gate rejects<br/>kind-1059 addressed to the ephemeral session key
    OPEN--)FAB: reply readable only from open relays<br/>the agent (junkiejarvis) also publishes to
```
- INVARIANT (`IDENTITY-zones.md:148-149`): the Talk-to-AI reply-relay set must remain a subset of the agent's own publish fan-out, or replies are never seen.
- `AIChatFab.tsx`'s own closeout comment flags what is NOT yet built: "the DM transport carries no protocol-level correlation" for request/reply matching — a late answer to a timed-out question can otherwise read as the answer to whatever was asked most recently (`ChatMessage.lateForQuestion`, `AIChatFab.tsx:9-19`); the estate closeout note (`IDENTITY-zones.md:169`) calls this out explicitly: "Preserve sender verification while adding request/reply correlation and verifying real agent fan-out."

## DW-04.4 NIP-42 AUTH — lazy challenge-response on publish
```mermaid
sequenceDiagram
    autonumber
    participant C as publishGiftWrap<br/>src/lib/nostr.ts:225
    participant WS as relay WebSocket
    C->>WS: ws.onopen -> sendEvent(): ["EVENT", wrap]<br/>src/lib/nostr.ts:266-273
    WS-->>C: ["OK", ..., false, "auth-required"]  (write gate, PRD-010 G4)
    alt opts.authSk provided
        WS-->>C: ["AUTH", challenge]
        C->>C: authAndRetry(): finalizeEvent(kind 22242,<br/>tags [relay, challenge])<br/>src/lib/nostr.ts:277-291
        C->>WS: ["AUTH", authEvent]
        C->>WS: retry ["EVENT", wrap]
    else no authSk
        Note right of C: AUTH frames ignored — legacy allowlist-mode behaviour
    end
    WS-->>C: ["OK", ..., true/false]
    C-->>C: PublishResult { ok, message } — never rejects,<br/>resolves false on transport failure, OK-false, or timeout
```
- The challenge is answered lazily — only after a rejection — "so allowlist-mode relays keep the zero-round-trip fast path" (`src/lib/nostr.ts:260-263` comment).
- `authSk` is minted per submission and discarded, exactly as for the wrap signature — anonymity is preserved even though the publish now requires proof of key possession (`src/lib/nostr.ts:213-216`).

## DW-04.5 Anonymous website ingress — threat register
```mermaid
flowchart TB
    SURFACE["kind-1059 admission:<br/>anonymous ephemeral authors, recipient-gated only<br/>SECURITY_OVERVIEW.md Anonymous Website Ingress"] --> T1["Spam/DoS<br/>mitigation: 10 events/s/IP relay limit only —<br/>no PoW/CAPTCHA/per-recipient throttle;<br/>kind 1059 bypasses content moderation"]
    SURFACE --> T2["PII in DM content<br/>mitigation: NIP-44 E2E encryption;<br/>erasure is operator-side D1 purge only —<br/>self-service NIP-09 deletion is cryptographically<br/>impossible (author key is a discarded throwaway)"]
    SURFACE --> T3["LLM-cost abuse via junkiejarvis<br/>mitigation: serialised sends (one in-flight/session),<br/>client-side throttle, ops kill-switch JUNKIEJARVIS_ENABLED=0"]
```
- Admission rule: the first `["p", ...]` tag pubkey must be whitelisted while the ephemeral author is deliberately unchecked; publishing an EVENT requires no NIP-42 AUTH, but reading kind-1059 DOES require it, with the filter's `#p` force-rewritten to the authed pubkey — a session can only ever read its own inbox (`SECURITY_OVERVIEW.md` Anonymous Website Ingress section).
- Federation posture: single relay today; kind 1059 is already in `dreamlab.toml [mesh].federated_kinds` (see DW-03), so this surface is "federation-ready by construction" though the mesh transport itself is designed, not shipped.

## DW-04.6 Admin identity resolution — three layers
```mermaid
flowchart TB
    L1["1. Static set: ADMIN_PUBKEYS env<br/>mirrors dreamlab.toml [admin].static_pubkeys<br/>deploy-gated, checked first"] --> RESOLVE["admin status =<br/>static ∪ D1<br/>nostr-bbs-auth-worker/src/admin.rs::is_admin"]
    L2["2. D1 path: whitelist.is_admin (relay D1)<br/>then members.is_admin (auth D1)"] --> RESOLVE
    L3["3. Promotion: /api/whitelist/set-admin<br/>last-admin demotion is blocked"] --> RESOLVE
    RESOLVE --> BOOT["first-user-is-admin bootstrap<br/>GET /api/setup-status reports needsSetup<br/>when no is_admin=1 row exists"]
```
- Known gap: the search-worker honours only its own `ADMIN_PUBKEYS` `[vars]` value — D1-promoted admins are not visible to it (`SECURITY_OVERVIEW.md` Admin Identity, "see the forum-flow cartography Gap 2").
- `workers-deploy.yml` blocks the auth-worker deploy if the `ADMIN_PUBKEYS` secret is unset (cross-reference DW-03.7's `validate_required_secrets` gate).

## DW-04.7 Zone visibility and encryption — the security-relevant subset
```mermaid
flowchart LR
    Z1["zone1 welcome<br/>PUBLIC, unencrypted"]
    Z2["zone2 minimoonoir<br/>LOCKED, unencrypted"]
    Z3["zone3 family<br/>LOCKED, ENCRYPTED<br/>only zone with encrypted=true"]
    Z4["zone4 dreamlab<br/>LOCKED, unencrypted"]
    Z1 -->|no cohort check| ANYONE["any visitor"]
    Z2 & Z4 -->|required_cohorts dual-accept| MEMBER["cohort-matched member"]
    Z3 -->|required_cohorts dual-accept + E2E| ENCMEMBER["cohort-matched member,<br/>content unreadable by relay operator"]
```
- INVARIANT (`IDENTITY-zones.md:146-147`): only `zone3` (Family) is encrypted; changing `encrypted` on any zone changes the E2E guarantee and must be recorded — see DW-03.1 for the full zone model this diagram's security view is drawn from.
- The estate closeout note (`IDENTITY-zones.md:169`) flags that zone encryption and cohort configuration "require deployed deny/revoke/recovery evidence before complete-system acceptance" — the config exists and is CI-checked for mirror parity (DW-03.5/03.6), but revocation behaviour has not been separately verified live.

## DW-04.8 Rate/size limits relevant to the identity surface
```mermaid
flowchart TB
    RELAY["Relay: 64KB content (8KB registration),<br/>2000 tags, 1024B/tag, 7-day drift,<br/>10 events/s/IP, 20 conns/IP, 20 subs/socket"]
    AUTHW["Auth worker: display name 1-64 chars,<br/>pubkey exactly 64 hex, challenge TTL 5min,<br/>NIP-98 token max 64KB, drift 60s"]
    POD["Pod worker: 50MB upload, path depth 10,<br/>charset [A-Za-z0-9-_./]"]
    SSRF["Preview-worker SSRF guard: http/https only,<br/>RFC1918/loopback/link-local/169.254.169.254 blocked,<br/>hex/int IP obfuscation blocked, 3-hop redirect cap,<br/>1MB response cap, 5s timeout"]
```
- Source: `docs/security/SECURITY_OVERVIEW.md` Relay/Auth-worker/Pod-worker Limits tables and SSRF Protection section — these limits are enforced in the upstream kit's worker source (cloned at `KIT_REF`, not vendored in this repo), so this diagram documents the operator-visible contract rather than code this repo owns.
