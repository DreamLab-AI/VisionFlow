---
id: DW-04
title: Identity, zones and security posture
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
adrs: [ADR-2006, ADR-2007, ADR-2008, ADR-2011]
sources:
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
  - ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md
  - ../dreamlab-ai-website/docs/security/AUTHENTICATION.md
  - ../dreamlab-ai-website/src/lib/nostr.ts
  - ../dreamlab-ai-website/src/components/AIChatFab.tsx
  - ../dreamlab-ai-website/forum-config/deploy/search-worker.wrangler.toml
  - ../dreamlab-ai-website/forum-config/deploy/relay-worker.wrangler.toml
  - ../dreamlab-ai-website/forum-config/deploy/auth-worker.wrangler.toml
  - ../dreamlab-ai-website/forum-config/deploy/SYSTEM-ONE.md
  - ../dreamlab-ai-website/docs/adr/ADR-2011-website-information-identity.md
  - ../dreamlab-ai-website/.github/workflows/deploy.yml
  - ../dreamlab-ai-website/scripts/lib/website-info.mjs
  - ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs
verified_commit: {dreamlab-ai-website: c3849abb8a2a289fbb013c721dfc0c5544c74b16, nostr-rust-forum: a3812b8e79846b569c1f19054f0774e94fed1248}
---

## DW-04.1 Identity is raw-hex Schnorr, not a DID document
```mermaid
flowchart TB
    KEY["64-hex secp256k1 pubkey<br/>the only identity shape in the shipped surface"] --> GEN["getPublicKey(sk)<br/>src/lib/nostr.ts:98"]
    KEY --> AUTH["finalizeEvent({kind: KIND_NIP42_AUTH})<br/>src/lib/nostr.ts:31, 279-281, 684-686, 849-851"]
    AUTH --> CHALLENGE["NIP-42 relay-challenge AUTH events<br/>kind 22242, NOT NIP-98"]
    NEG["grep -c 'NIP-98' src/lib/nostr.ts -> 0"] -.-> CHALLENGE
    KEY --> METHODS["auth methods offered by the upstream kit:<br/>WebAuthn PRF passkeys, NIP-07 extension, raw private key"]
    METHODS --> KEY
```
- INVARIANT (`IDENTITY-zones.md:175-177`): identity in the shipped surface is the raw 64-hex secp256k1 pubkey; any move to emit or require a Multikey/DID document is a code change and a new ADR, not a docs edit.
- DOC-DRIFT within this repo's OWN docs: `docs/security/AUTHENTICATION.md` (`## NIP-98 HTTP Authentication`, line 200) documents NIP-98 as the upstream kit auth-worker's HTTP-API scheme (passkey/session auth to REST endpoints) — a genuinely different protocol from the NIP-42 relay-challenge flow this repo's own `src/lib/nostr.ts` implements for DM transport. `IDENTITY-zones.md:34-35`'s claim that "the string NIP-98 appears nowhere in nostr.ts" is true and specific to that one file; it does not mean NIP-98 is absent from the deployed system.
- DIVERGENCE: legacy ADR-027's `did:nostr`/Multikey document form is docs-only — `fe70102` is not a commit hash, it is the Multikey encoding prefix itself (base16-multibase `f` + secp256k1-pub multicodec varint `e701` + compressed-point `02`, `IDENTITY-zones.md:46-49`); the two commits that landed it (`d62ab40`, `8d942d7`) touched only docs and JSON-LD, and the binding spec they cite (`ADR-125-did-nostr-multikey-convergence.md`) does not exist in this repo — it is kit-owned. ADR-027 is archived as "Deferred — kit-owned" (`IDENTITY-zones.md:56`); treat any claim that this repo emits `did:nostr` Multikey documents as false until a code path produces one (`IDENTITY-zones.md:162-164`).

## DW-04.3 Talk-to-AI — client-side Nostr DM, not an HTTP chat endpoint
```mermaid
sequenceDiagram
    autonumber
    participant U as visitor (browser)
    participant FAB as AIChatFab<br/>src/components/AIChatFab.tsx:445 sendQuestion
    participant DM as DmSession<br/>src/lib/nostr.ts:486
    participant W as nip17.wrapEvent<br/>src/lib/nostr.ts:196
    participant R as VITE_RELAY_URL<br/>publishGiftWrap, src/lib/nostr.ts:226
    participant OPEN as open relays<br/>VITE_REPLY_RELAYS: relay.damus.io, relay.primal.net
    U->>FAB: submits question
    FAB->>FAB: encodeQuestion({requestId, question, tier, identityHint: pubkey})<br/>AIChatFab.tsx:435-441
    Note right of FAB: identityHint is EXPLICITLY UNVERIFIED —<br/>no proof of possession, DM rides the ephemeral<br/>session key (AIChatFab.tsx:431-432 comment)
    FAB->>DM: session.sendQuestion(payload, JARVIS_PUBKEY)<br/>AIChatFab.tsx:445
    DM->>W: wrap kind-14 rumor to VITE_JARVIS_PUBKEY<br/>src/lib/nostr.ts:137,196
    W->>R: publish kind-1059 gift wrap<br/>resolves on relay OK-true only, never rejects
    Note right of R: primary relay's whitelist gate rejects<br/>kind-1059 addressed to the ephemeral session key
    R-->>DM: OK-true for the question
    DM->>R: publishMonitorCopy: second wrap to VITE_MONITOR_PUBKEY,<br/>best-effort, only after OK-true<br/>src/lib/nostr.ts:352, 729-741
    OPEN--)FAB: reply readable only from open relays<br/>the agent (junkiejarvis) also publishes to
```
- INVARIANT (`IDENTITY-zones.md:182-183`): the Talk-to-AI reply-relay set must remain a subset of the agent's own publish fan-out, or replies are never seen.
- **Drift (corrected):** an earlier reading said request/reply correlation was not yet built. The DM transport still carries no protocol-level correlation (the agent's reply rumor has no `e` tag), but the client now mints its own request id per question and binds replies to it, so a late or out-of-order reply resolves the turn it names or none (`AIChatFab.tsx:51-56`; `requestId` at `AIChatFab.tsx:435-441`); `ChatMessage.lateForQuestion` labels a late answer (`AIChatFab.tsx:9-19`). The estate closeout note still asks to "Preserve sender verification while adding request/reply correlation and verifying real agent fan-out" (`IDENTITY-zones.md:208`); agent-side correlation is outside this repo.

## DW-04.4 NIP-42 AUTH — lazy challenge-response on publish
```mermaid
sequenceDiagram
    autonumber
    participant C as publishGiftWrap<br/>src/lib/nostr.ts:226
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
- `authSk` is minted per submission and discarded, exactly as for the wrap signature — anonymity is preserved even though the publish now requires proof of key possession (`src/lib/nostr.ts:207-213`).

## DW-04.5 Anonymous website ingress — threat register
```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1087
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1135
    A["Write-auth mode gate, then gift-wrap admission<br/>configured ingress or whitelisted recipient"]
    %% Source: ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md:282
    A --> B["Operator docs flag spam risk<br/>IP and structural limits, client validation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2453-2457
    A --> C["Deletion supports author or privileged caller<br/>no gift-wrap recipient-only path here"]
    %% Source: ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md:284
    A --> D["Docs flag model-cost abuse<br/>client throttling and operator kill switch"]
    %% Source: ../dreamlab-ai-website/forum-config/deploy/SYSTEM-ONE.md:3
    %% Source: ../dreamlab-ai-website/forum-config/deploy/SYSTEM-ONE.md:13
    A --> E["Overlay enables advisory TypeSafe calls<br/>rationale local-only restriction is opt-in"]
```
- Admission is mode-dependent: the consumed kit checks session authentication before recipient admission, requiring AUTH in `nip42` mode; configured ingress or a whitelist row can admit the recipient. The operator document's unconditional no-AUTH/whitelist-only wording is stale. The prior panel recorded successful Worker deploy run `41028875471` on 30 October, with credential checks for search and relay but no auth-secret receipt. That observation remains historical; this pass reads configuration and does not redeploy or recheck Worker secrets.
- Federation posture: single relay today; kind 1059 is already in `dreamlab.toml [mesh].federated_kinds` (see DW-03), so this surface is "federation-ready by construction" though the mesh transport itself is designed, not shipped.

## DW-04.6 Admin identity resolution — three layers
```mermaid
flowchart TB
    L1["1. Static set: ADMIN_PUBKEYS env<br/>mirrors dreamlab.toml [admin].static_pubkeys<br/>deploy-gated, checked first<br/>SECURITY_OVERVIEW.md:268"] --> RESOLVE["admin status =<br/>static ∪ D1<br/>nostr-bbs-auth-worker/src/admin.rs::is_admin<br/>SECURITY_OVERVIEW.md:259"]
    L2["2. D1 path: whitelist.is_admin (relay D1)<br/>then members.is_admin (auth D1)<br/>SECURITY_OVERVIEW.md:269"] --> RESOLVE
    L3["3. Promotion: /api/whitelist/set-admin<br/>last-admin demotion is blocked<br/>SECURITY_OVERVIEW.md:270"] --> RESOLVE
    RESOLVE --> BOOT["first-user-is-admin bootstrap<br/>GET /api/setup-status reports needsSetup<br/>when no is_admin=1 row exists<br/>SECURITY_OVERVIEW.md:259"]
```
- Known gap: the search-worker honours only its own `ADMIN_PUBKEYS` `[vars]` value — D1-promoted admins are not visible to it (`SECURITY_OVERVIEW.md:272`, "see the forum-flow cartography Gap 2"); its own separate copy is `search-worker.wrangler.toml:46` (see DW-03.10).
- `workers-deploy.yml` blocks the auth-worker deploy if the `ADMIN_PUBKEYS` secret is unset; the block is that workflow's bash step, not the Rust `validate_required_secrets`, which no workflow calls (see DW-03.8).

## DW-04.8 Rate/size limits relevant to the identity surface
```mermaid
flowchart TB
    RELAY["Relay: 64KB content (8KB registration),<br/>2000 tags, 1024B/tag, 7-day drift,<br/>10 events/s/IP, 20 conns/IP, 20 subs/socket<br/>SECURITY_OVERVIEW.md:201,209"]
    AUTHW["Auth worker: display name 1-64 chars,<br/>pubkey exactly 64 hex, challenge TTL 5min,<br/>NIP-98 token max 64KB, drift 60s<br/>SECURITY_OVERVIEW.md:213,219"]
    POD["Pod worker: 50MB upload, path depth 10,<br/>charset [A-Za-z0-9-_./]<br/>SECURITY_OVERVIEW.md:223,227"]
    SSRF["Preview-worker SSRF guard: http/https only,<br/>RFC1918/loopback/link-local/169.254.169.254 blocked,<br/>hex/int IP obfuscation blocked, 3-hop redirect cap,<br/>1MB response cap, 5s timeout<br/>SECURITY_OVERVIEW.md:233"]
```
- Source: `docs/security/SECURITY_OVERVIEW.md` Relay/Auth-worker/Pod-worker Limits tables (`:197-231`) and SSRF Protection section (`:233`) — these limits are enforced in the upstream kit's worker source (cloned at `KIT_REF`, not vendored in this repo), so this diagram documents the operator-visible contract rather than code this repo owns.

## DW-04.9 The website-info identity — labelled, never privileged (ADR-2011)
```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/.github/workflows/deploy.yml:80
    A["Configured monitor identity"]
    %% Source: ../dreamlab-ai-website/src/lib/nostr.ts:352
    A --> B["Best-effort contact, signup and chat copies"]
    %% Source: ../dreamlab-ai-website/forum-config/dreamlab.toml:407
    A --> C["website-info roster has empty cohorts"]
    %% Source: ../dreamlab-ai-website/forum-config/dreamlab.toml:435
    A --> D["Ingress copy role admits addressed wraps<br/>without requiring a whitelist row"]
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs:97
    A --> E["Setup skips registration only when public<br/>disclosure reports registered true for the key"]
    %% Source: ../dreamlab-ai-website/scripts/lib/website-info.mjs:100
    A --> F["Setup checks reject admin flag<br/>or nonempty cohort array"]
```
- ADR-2011 decides the key is named but unprivileged because the `agent` cohort would receive the zone4 key (`docs/adr/ADR-2011-website-information-identity.md:54-56`); the relay reads only wraps addressed to the viewer, which is why a co-recipient copy exists rather than a second look at the admin's (`src/lib/nostr.ts:342-347`).
- **Open:** the registry's `registered_by` is the agentbox operator key that signs the registration while the roster says `operator-jjohare`; ADR-2011 leaves that gap open until the kit exports config to the registry (`docs/adr/ADR-2011-website-information-identity.md:75-76`).
- **Drift (repaired at source 2026-11-03, adoption not evidenced):** at the previous stamp (`0e15728`) ADR-2011 said the watch reports missing registry/whitelist rows, but configuration-backed ingress and merged disclosure could keep its checks green after row loss, and the provisioner skipped registration whenever the merged response merely listed the key. The website now requires the disclosure's explicit `registered` flag in both the nightly check and the provisioner (`scripts/lib/website-info.mjs:83`, `scripts/lib/website-info.mjs:89`, `scripts/seed/provision-website-info.mjs:97`, `scripts/seed/provision-website-info.mjs:119`), and the forum disclosure marks a registry row `registered: true` and a config-merged ingress entry `registered: false` (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:57`, `../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:88`). ADR-2011's 2026-11-03 repair evidence records that the fix needs the matching forum response deployed, that an older response now fails the registry check, and that no provisioning or live messaging was performed (`docs/adr/ADR-2011-website-information-identity.md:26-37`). DW-09 and the [dated review](../../estate-review/2026-11-01-website-information-and-inbox.md) trace the consumed kit and reproduce the earlier consumer behaviour.
- **Open:** ADR-2011 now points at a 2026-11-02 dated source reconciliation of the external review that supplements the decision without changing its status (`docs/adr/ADR-2011-website-information-identity.md:20-24`); ADR-2011 records no deployment of a forum revision that serves the `registered` field, so the repair is source-complete and adoption-pending.
- The old cryptographic-impossibility wording in DW-04.5 was too broad: current NIP-59 recommends recipient-authorised deletion, whereas the inspected kit deletion handler permits author matching or privileged deletion. The dated review separates protocol capability from this implementation; no deletion was executed.
