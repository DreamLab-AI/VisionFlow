---
id: ES-04
title: did:nostr identity mesh — signing, verification, custody
area: estate
governing:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/docs/SECURITY-profiles.md
adrs: [agentbox:ADR-2153, agentbox:ADR-2042, agentbox:ADR-2098, agentbox:ADR-2101, agentbox:ADR-2033, agentbox:ADR-2002, agentbox:ADR-2009, agentbox:ADR-2010, agentbox:ADR-2011, agentbox:ADR-2013, agentbox:ADR-2026]
sources:
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/management-api/lib/pod-signer.js
  - ../project/agentbox/management-api/lib/agent-identity.js
  - ../project/agentbox/management-api/lib/agent-event-auth.js
  - ../project/agentbox/management-api/lib/action-plane.js
  - ../project/agentbox/management-api/lib/zone-keys.js
  - ../project/agentbox/management-api/adapters/index.js
  - ../project/agentbox/management-api/adapters/pods/_solid-http-base.js
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md
  - ../project/agentbox/services/agentbox-sessions/src/api.rs
  - ../project/agentbox/services/agentbox-sessions/src/boundary.rs
  - ../project/agentbox/config/custody/session-plane-acl.json
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/docs/SECURITY-profiles.md
  - ../project/src/utils/nip98.rs
  - ../project/src/services/nostr_service.rs
  - ../project/src/services/nostr_identity_verifier.rs
  - ../project/src/services/nostr_bridge.rs
  - ../project/src/services/management_api_client.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/client/src/services/nostrAuthService.ts
  - ../project/agentbox/docs/INGRESS-identity.md
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---
## ES-04.1 verification mesh — who signs, who verifies whom
```mermaid
flowchart TB
    subgraph browser["Browser (untrusted)<br/>trust boundary"]
        BX["window.nostr<br/>NIP-07 extension"]
    end
    subgraph vc["VisionClaw process<br/>src/"]
        AE["AuthenticatedUser extractor<br/>src/settings/auth_extractor.rs:89"]
        NS["NostrService::verify_nip98_auth<br/>src/services/nostr_service.rs:648"]
        N98["validate_nip98_token<br/>src/utils/nip98.rs:428"]
        NIV["NostrIdentityVerifier<br/>src/services/nostr_identity_verifier.rs:34"]
        NB["NostrBridge (re-sign)<br/>src/services/nostr_bridge.rs:28"]
        MAC["ManagementApiClient<br/>src/services/management_api_client.rs:27"]
    end
    subgraph agentbox["agentbox host processes"]
        PROXY["nip98-proxy  port 9096<br/>agentbox/config/nip98-proxy/proxy.mjs:19"]
        NBB["NostrBridge.verifyNip98<br/>agentbox/mcp/servers/nostr-bridge.js:500"]
        PS["pod-signer buildPodNip98<br/>agentbox/management-api/lib/pod-signer.js:70"]
        AGID["agent-identity loadOrMint<br/>agentbox/management-api/lib/agent-identity.js:128"]
        AEA["agent-event-auth verifyAgentEventRequest<br/>agentbox/management-api/lib/agent-event-auth.js:135"]
        GW["nostr-gateway SESSION_CLIENT<br/>agentbox/config/nostr-gateway/gateway.cjs:123"]
        MGMT["management-api /mgmt/ upstream<br/>agentbox/config/nip98-proxy/proxy.mjs:304"]
    end
    subgraph sessions["agentbox-sessions (ADR-2153)<br/>unix socket only, no TCP listener"]
        SESS["daemon: socket 0660 in ab-sessions, uid from SO_PEERCRED<br/>checked against the ACL before any request is read<br/>agentbox/services/agentbox-sessions/src/api.rs:74"]
    end
    subgraph solid["EXTERNAL: solid-pod-rs<br/>default-deny pod"]
        POD["Solid pod HTTP endpoint"]
    end
    subgraph nonsign["Non-signing (VisionClaw canon)"]
        VF["VisionFlow canon<br/>read-only KG consumer"]
    end

    BX -- "NIP-07 signEvent()<br/>nostrAuthService.ts:276" --> AE
    AE -- "NIP-98 kind-27235" --> NS --> N98
    NS -- "Bearer + X-Nostr-Pubkey<br/>explicit migration flag; default off" --> AE
    NIV -. "XR presence Schnorr(nonce||ts)<br/>NOT NIP-98" .-> NB
    MAC -- "Authorization: Bearer api_key<br/>management_api_client.rs:305" --> PROXY
    PROXY -- "NIP-98 kind-27235<br/>proxy-verified" --> NBB
    PROXY -- "X-Agentbox-Pubkey<br/>injected, never trusted inbound<br/>proxy.mjs:878" --> MGMT
    GW -- "op create / send as uid 961 (ab-gateway)<br/>gateway.cjs:436" --> SESS
    SESS -- "POST /v1/sessions/boundary<br/>boundary.rs:86" --> MGMT
    PS -- "NIP-98 kind-27235<br/>buildNip98Header" --> POD
    AEA -- "NIP-98 kind-27235<br/>AGENTBOX_AGENT_EVENT_AUTH=nip98" --> NBB
    AGID -. "did:nostr:&lt;hex&gt; mint<br/>consumed by" .-> PS
    NB -. "DIVERGENCE NIP-26 delegation NOT wired —<br/>nostr_bridge.rs re-signs under the BRIDGE key<br/>(service-signed, not delegation), so no request<br/>can be attributed to a user THROUGH an agent" .-> VF

    Note1["EXTERNAL: nostr-rust-forum, dreamlab-ai-website,<br/>code-as-harness sign their own events;<br/>not on disk in this repo — asserted only where cited"]
```

**Drift (resolved by agentbox 6500582, still holds at c57d9c0):** the edge from `AEA` to `NBB` is path 1 only. Path 2 of `verifyAgentEventRequest` accepts the proxy-stamped `X-Agentbox-Pubkey` without a signature (`agentbox/management-api/lib/agent-event-auth.js:192-194`). Before the fix any `MANAGEMENT_API_KEY` holder on the docker network could forge it, because management-api binds `0.0.0.0`. Now the header is honoured only from a loopback TCP peer (`peerIsProxy`, `AGENTBOX_PROXY_PEERS` overrides) in both `agent-event-auth` and `action-plane` `resolveAgentDid` (`agentbox/management-api/lib/agent-event-auth.js:98-106,117`, `agentbox/management-api/lib/action-plane.js:228-232`), so the `did:nostr` carried into agent-events and the task journal cannot be asserted from another container.

**Drift (new at agentbox c57d9c0, ADR-2164):** a third credential now reaches `verifyAgentEventRequest` ahead of both paths: a per-task action token, already checked by the global auth gate, is accepted under every policy and returns `did: null` with the task id, so a task's own agent-action posts carry no `did:nostr` at all (`agentbox/management-api/lib/agent-event-auth.js:143-145`, `agentbox/docs/INGRESS-identity.md:338-344`). The mesh above draws only the two identity-bearing paths.

**Debt:** the stamp's strength is the same-UID boundary of the agentbox container, because any process inside it reaches management-api over loopback (TODO-unified X-1; `agentbox/management-api/lib/agent-event-auth.js:36-39`).

**Drift (resolved at this stamp):** ADR-2153 is live (`decision_status: accepted`, `implementation_status: partial`, `activation_status: live`; `../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:5-7`). Agent of Empires, its port 9095 listener, its bearer token and the proxy hop to it are gone; session control is a 0660 `ab-sessions` unix socket authorised by `SO_PEERCRED` against uids 1000 and 961 (`../project/agentbox/services/agentbox-sessions/src/api.rs:74`, `../project/agentbox/config/custody/session-plane-acl.json:8-11`), and port 9096 forwards named routes only (`../project/agentbox/docs/INGRESS-identity.md:69-71,244-247`). The 2026-10-30 close-out re-checks the running image and leaves the axes unchanged: `command -v aoe` finds nothing and nothing listens on port 9095 (`../project/agentbox/docs/adr/ADR-2153-session-plane-replaces-aoe.md:259`); `partial` stays because the all-seed acceptance gap remains open (`:262`).

**Open (PROPOSED 2026-10-30, not implemented):** VisionClaw's identity-authority record now carries a Proposed room-admission section (ADR-2157, ADR-2160, ADR-2161): a `RoomAdmission` port after the presence handshake, fed by the home relay's signed NIP-29 39002 member list, and a single agentbox-held node federation key that VisionClaw never holds (`docs/IDENTITY-authority-chain.md`, section "Proposed"). Today the handshake alone admits a DID to any room, and the diagrams here describe only that shipped path.

## ES-04.3 validate_nip98_token — the fixed check order, replay claimed LAST
```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant V as validate_nip98_token<br/>src/utils/nip98.rs:428
    participant R as claim_signed_id<br/>src/utils/nip98.rs:227
    participant CACHE as REPLAY_CACHE<br/>Mutex-guarded, src/utils/nip98.rs:221

    C->>V: base64 token, expected url, expected method, body
    alt decode fails
        V-->>C: Err InvalidBase64 / InvalidUtf8 / InvalidJson
    else decoded
        V->>V: kind check
        alt kind is not 27235
            V-->>C: Err InvalidKind — nip98.rs:463
        else kind ok
            V->>V: FRESHNESS — compare created_at to now
            alt age exceeds TOKEN_MAX_AGE_SECONDS 60
                V-->>C: Err TokenExpired — nip98.rs:478
            else created_at is in the future
                V-->>C: Err TokenFromFuture — nip98.rs:483
            else fresh
                V->>V: TAG MATCH — host-checked u tag
                alt url differs
                    V-->>C: Err UrlMismatch{expected, actual} — nip98.rs:511
                else method differs
                    V-->>C: Err MethodMismatch{expected, actual} — nip98.rs:519
                else payload hash present and differs
                    V-->>C: Err PayloadHashMismatch — nip98.rs:537
                else tags match
                    V->>V: SIGNATURE verification
                    alt signature invalid
                        V-->>C: Err InvalidSignature / VerificationFailed
                    else signature valid
                        V->>R: claim_signed_id(event.id, pubkey, Instant::now())
                        Note over R,CACHE: Check-and-insert is ATOMIC under the Mutex —<br/>no TOCTOU (nip98.rs:200-203). Instant is MONOTONIC, so a<br/>backward wall-clock step cannot un-spend an id (nip98.rs:205-208).
                        alt a still-live entry exists
                            R-->>V: Err TokenReplayed — nip98.rs:307
                            V-->>C: rejected
                        else live set is at REPLAY_CACHE_MAX_ENTRIES
                            R-->>V: Err ReplayCacheFull
                            V-->>C: rejected, fail-closed under flood
                        else recorded
                            R-->>V: Ok
                            V-->>C: Ok Nip98ValidationResult
                        end
                    end
                end
            end
        end
    end
    Note over C,CACHE: INVARIANT — the replay CLAIM is the LAST step<br/>(nip98.rs:573). It must never precede signature verification,<br/>or a forged token would burn a legitimate event id.
    Note over V,CACHE: INVARIANT — REPLAY_CACHE_TTL must stay at 2x the freshness<br/>window (nip98.rs:178). A token created at now+60 stays valid<br/>until now+120, so a shorter TTL would let it replay after<br/>its entry prunes.
    Note over R,CACHE: REPLAY_CACHE_PRUNE_THRESHOLD = 4096 triggers an opportunistic<br/>O(n) sweep on insert (nip98.rs:183). INVARIANT — on a FULL cache the code<br/>REFUSES new auth (ReplayCacheFull) and must NOT evict the oldest live<br/>entry — evicting would let a flooder purge a genuine still-valid id and<br/>re-enable the very replay this layer prevents (nip98.rs:189-197).<br/>Bounded memory beats availability.
```

## ES-04.4 agentbox pod-signer — an agent signing a write to a Solid pod
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as management-api caller
    participant B as buildPodNip98<br/>agentbox/management-api/lib/pod-signer.js:70
    participant S as signer (lazy-loaded, cached)
    participant H as buildNip98Header<br/>deps injection, pod-signer.js:92
    participant POD as Solid pod

    CALLER->>B: buildPodNip98(manifest, deps)
    alt sign_requests on (agentbox.toml:590) and no identity port selected
        B-->>CALLER: async (method, url, body) => header<br/>pod-signer.js:124
        CALLER->>H: nip98(method, url, body)
        H->>S: resolve signer (loaded lazily on first use, then cached)
        alt signer available
            S-->>H: signer
            H->>H: buildNip98Header(s, method, url, {body})<br/>pod-signer.js:127
            H-->>CALLER: Authorization header carrying the kind-27235 event
            CALLER->>POD: LDP write with the signed header
            POD-->>CALLER: 2xx
        else signer unavailable or the key will not decrypt
            S-->>H: none
            H-->>CALLER: null header
            Note over CALLER,POD: FAIL-CLOSED (ADR-2064) — with sign_requests on, the adapter<br/>carries requireSigned (adapters/index.js:74) and a null header<br/>throws SigningUnavailable before any byte is sent<br/>(_solid-http-base.js:105). No unsigned fallback exists.
            CALLER--xPOD: no request sent
        end
    else sign_requests off
        B-->>CALLER: null — pods adapter goes out unsigned, the pre-signing baseline<br/>pod-signer.js:73
    end
    Note over CALLER,POD: The signer lifecycle MIRRORS lib/elevation-publisher —<br/>owned here, loaded lazily on first use, cached. No<br/>in-request connect or disconnect. see ES-05.7
    Note over B,S: Under role_isolation, or sign_source = identity-port, the header comes<br/>from the identity port instead and no key is read (pod-signer.js:76-77)
```

**Drift (corrected at agentbox c57d9c0):** this panel used to show a degraded signer falling back to an UNSIGNED write and a boot that could keep the `did:nostr:local` placeholder. Neither holds at the pin, and neither held at 6500582 either (pod-signer.js is byte-identical between the two, and the entrypoint already aborted on a failed mint there): `buildPodNip98` returns `null` only when `sign_requests` is off (`../project/agentbox/management-api/lib/pod-signer.js:73`); when it is on, the adapter is built with `requireSigned` (`../project/agentbox/management-api/adapters/index.js:74`) and a null header throws `SigningUnavailable` (`../project/agentbox/management-api/adapters/pods/_solid-http-base.js:105`); and the entrypoint now aborts boot when the mint fails (`../project/agentbox/config/entrypoint-unified.sh:1429-1438`).

**Drift (source comment):** `agent-identity.js` still says the entrypoint's `did:nostr:local` fallback "runs unconditionally" (`../project/agentbox/management-api/lib/agent-identity.js:41-43`); the entrypoint it describes now exits 1 on a failed or inconsistent mint (`../project/agentbox/config/entrypoint-unified.sh:1429-1438`). The comment is stale, not the code.

## ES-04.5 Key custody — where every signing key actually lives
```mermaid
flowchart TB
    subgraph live["Live signing keys"]
        K1["Bridge identity / unwrap key<br/>AGENTBOX_BRIDGE_SK_FILE, default /run/secrets/nostr.key<br/>legacy environment fallback REMAINS"]
        K2["Shared server publisher identity<br/>relay key list is build-projected"]
        K3["Session-plane socket ACL: no token, no key<br/>authorisation is the peer uid, 1000 devuser and 961 ab-gateway<br/>agentbox/config/custody/session-plane-acl.json:8-11"]
        K4["nip98-proxy session secret<br/>NIP98_PROXY_SESSION_SECRET or per-boot crypto.randomBytes"]
        K5["Session-mirror child key<br/>~/.claude/nostr-mirror/mirror-key.txt<br/>DERIVED, re-derivable, deleted after import"]
    end
    subgraph fixtures["NOT live keys"]
        F1["test-data-idp-accounts/ at the repo root<br/>FIXTURES ONLY — test IdP accounts, never live custody"]
    end
    subgraph boundary["Egress boundary"]
        REL["cloud Nostr relay worker<br/>dreamlab-nostr-relay.solitary-paper-764d.workers.dev<br/>EXTERNAL — this worker is dreamlab-ai-website's<br/>relay-worker, not an agentbox service. see DW-01, DW-04"]
    end

    K5 -->|"NIP-59 gift wrap, kind 1059"| REL

    D1["PARTIALLY RESOLVED (Q15, 2026-10-03) — agentbox.toml:160 now records<br/>the separation: the operator NIP-07 signer signs the 31403 governance<br/>decisions and is NOT visionclaw-server, which signs 31402 under a<br/>different key. The earlier revision of this node cited the roster row<br/>that still marked the governance publisher key-split PENDING (ADR-040 D3)."]
    D2["DIVERGENCE — NIP98_PROXY_SESSION_SECRET defaults to<br/>crypto.randomBytes (proxy.mjs:167), so NIP-07 sessions do<br/>NOT survive a proxy restart. Intentional, but every restart<br/>forces re-authentication."]
    D3["DIVERGENCE — the ACL authorises a UID, not a process: any devuser<br/>(uid 1000) process may call the session socket, so same-user peers<br/>are not isolated. Per-process isolation is future work.<br/>session-plane-acl.json:6, api.rs:32"]
    D5["DIVERGENCE — every custodian, deployed location, rotation<br/>cadence and incident-response window in the agentbox custody<br/>register is UNCONFIRMED (proposed governing surface)."]

    K2 --> D1
    K4 --> D2
    K3 --> D3
    live --> D5
    F1 -.-> live
```

## ES-04.6 Session mirror — derive, gift-wrap, publish, fail open
```mermaid
sequenceDiagram
    autonumber
    participant T as turn end
    participant HK as nostr-live-mirror.cjs<br/>agentbox/config/hooks/nostr-live-mirror.cjs
    participant MK as mirror child key<br/>~/.claude/nostr-mirror/mirror-key.txt
    participant W as NIP-59 gift wrap
    participant REL as cloud relay worker
    participant AM as Amethyst on mobile

    T->>HK: per-turn hook fires (event = SessionStart|UserPromptSubmit|Stop|SessionEnd)
    HK->>HK: EARLY EXIT — egressDecision('live-mirror',{}) then recipientAllowlist(),<br/>checked BEFORE key derivation or stdin read<br/>nostr-live-mirror.cjs:406-415
    alt AGENTBOX_LIVE_MIRROR=0 or no valid recipient allowlist
        HK-->>T: return 0 — ~170ms/event saved vs deriving keys first
    else no recipient pubkey configured
        HK-->>T: silent no-op
    else armed
        HK->>HK: mirroredEvents() gate — default {Stop} only,<br/>widened by AGENTBOX_LIVE_MIRROR_EVENTS<br/>nostr-live-mirror.cjs:279-285
        alt event not in mirroredEvents()
            HK-->>T: bodyForEvent() returns null — skip, nostr-live-mirror.cjs:291-292
        else event mirrored
            HK->>MK: load derived mirror child key
            HK->>HK: build kind-14 DM rumor
            Note over HK: A urn:agentbox:activity reference is placed INSIDE the<br/>already gift-wrap-sealed rumor (mintActivityUrn, nostr-live-mirror.cjs:167),<br/>so the URN never appears in cleartext on the wire.
            HK->>W: seal sender identity, wrap as KIND_GIFT_WRAP 1059<br/>nostr-live-mirror.cjs:62, rumor kind 14 at :63
            W-->>HK: signed gift wrap
            HK->>REL: publish ONE pre-signed gift wrap, await the relay OK<br/>publishWrap nostr-live-mirror.cjs:333
            alt relay accepts
                REL-->>HK: ok
                REL->>AM: readable with the derived mirror child key
            else relay rejects or is unreachable
                REL--xHK: error
                HK-->>T: FAIL-OPEN — the turn is never blocked
            end
        end
    end
    Note over HK,REL: INVARIANT ADR-2026 — the ONLY network egress is the<br/>ENCRYPTED gift wrap. The relay accepts a kind-1059 iff its<br/>FIRST ["p"] recipient is whitelisted (nostr-live-mirror.cjs:19).
    Note over T,AM: Digest sibling — ONE curated kind-30840 summary at SessionEnd<br/>(nostr-live-mirror.cjs:8), via [sovereign_mesh.mobile_bridge].
    Note over HK: DOC-DRIFT agentbox/docs/SECURITY-profiles.md:11 still asserts the live<br/>hook composes UNREDACTED text —<br/>SECURITY-profiles.md:68-69 and the code<br/>(redactForEgress before compose, nostr-live-mirror.cjs:456) say this<br/>closed 2026-09-05 on both paths. Doc line 11 is stale prose, not code.
```

Encrypted forum zones (ADR-2016, INGRESS-identity.md item 10): zone-key grants (rumor kind 21453) and signed NIP-171 epoch tickets (kind 1014, `../project/agentbox/management-api/lib/zone-keys.js:54-56`, `../project/agentbox/docs/INGRESS-identity.md:295-299`), gating and
admin-relay retry live in AB-12.11 and AB-13.5 (`zone-keys.js`, `../project/agentbox/config/nostr-gateway/gateway.cjs:689`), not here.

The 2026-09-07 source closeout makes opaque session issuance, lookup and refresh
conditional on `VISIONCLAW_LEGACY_SESSIONS=1/true`, default off. Browser REST signs
per request; graph WebSocket upgrades now carry a fresh signed NIP-98 event in a
base64url subprotocol and negotiate only a public protocol. Missing/declined signers
fail before connection. These changes do not prove remaining MCP clients or the
actual proxy/reconnect deployment have migrated. See VC-03.17 and the
[execution report](../../estate-review/closeout/2026-09-07-execution-visionclaw.md).

## ES-04.7 PROPOSED key separation — one identity key, two derived families, a second Multikey
```mermaid
flowchart TB
    KID["k_id — the sovereign identity key, minted by the identity binary<br/>agentbox/docs/INGRESS-identity.md:506<br/>PROPOSED rule: it never spends and never seals a block"]

    SPEND["k_spend(chain) = derive_subkey(k_id, sidestr/spend/ chain_id)<br/>used on the wallet path, spends UTXOs on exactly that chain<br/>agentbox/docs/INGRESS-identity.md:507"]
    SIGN["k_sign(chain) = derive_subkey(k_id, sidestr/sign/ chain_id)<br/>federated instance operators only, seals blocks on that chain<br/>agentbox/docs/INGRESS-identity.md:508"]

    KID -->|"HMAC-SHA256 domain-separated child derivation,<br/>a tool the estate already owns<br/>agentbox/docs/INGRESS-identity.md:501-502"| SPEND
    KID -->|"same derivation, different domain string"| SIGN

    BIND["PROPOSED kind 38420 sidestr-account-binding (ADR-2105, band<br/>38400-38499), addressable, d tag is chain id plus did hex,<br/>content is the derived spend pubkey, signed by k_id<br/>agentbox/docs/INGRESS-identity.md:519-521"]
    SPEND --> BIND

    MK["PROPOSED — a SECOND Multikey entry in the DID document for the<br/>per-chain spend key. This AMENDS legacy ADR-033, whose<br/>single-Multikey form is what build_did_document emits today.<br/>agentbox/docs/INGRESS-identity.md:522-524"]
    BIND --> MK

    LIVE["LIVE today — one Multikey, publicKeyMultibase is the<br/>MULTIKEY_PREFIX followed by the x-only hex, and the private key<br/>never leaves the function<br/>agentbox/docs/INGRESS-identity.md:148-152"]
    MK -.->|"amends"| LIVE

    SCOPE["PROPOSED scope note — Invariant 6 is narrowed to IDENTITY<br/>ingress; chain ingress is authenticated by consensus rather than<br/>by the relay allowlist.<br/>agentbox/docs/INGRESS-identity.md:254-255"]
    KID --> SCOPE

    INV["INVARIANT — the whole of this diagram sits in a clearly marked<br/>PROPOSED section and the live compliance surface above it is<br/>unchanged apart from that one scope note.<br/>agentbox/docs/INGRESS-identity.md:18,492-495. see ES-03.11"]
    SCOPE --> INV
```
