---
id: NF-02
title: auth-worker — passkey ceremonies, NIP-98 gating, membership and the REST surface
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2003, ADR-2004]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admins.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/username.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/moderation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/wot.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/welcome.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/schema.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/crypto.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/did.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/pod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/http.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/zone_approval.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/lib.rs
verified_commit: d48a7a546
---

## NF-02.1 Request entry — bootstrap, rate limit, body pre-read, dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant F as fetch<br/>nostr-bbs-auth-worker/src/lib.rs:143
    participant H as handle_request<br/>nostr-bbs-auth-worker/src/lib.rs:157
    participant S as schema::ensure_schema<br/>nostr-bbs-auth-worker/src/schema.rs:19
    participant RL as rate-limit + replay schema<br/>nostr-bbs-auth-worker/src/lib.rs:174
    participant R as route<br/>nostr-bbs-auth-worker/src/lib.rs:355

    C->>F: HTTP request
    F->>H: handle_request (all errors caught by the wrapper)
    H->>S: idempotent D1 bootstrap on every cold start lib.rs:162
    H->>RL: ensure_replay_schema then check_rate_limit 20 req / 60 s per IP lib.rs:174
    alt over budget
        RL-->>C: 429 Too many requests lib.rs:175
    end
    H->>H: Options preflight returns 204 + CORS lib.rs:169
    H->>H: origin = the ACTUAL request URL, not EXPECTED_ORIGIN lib.rs:190
    H->>H: body bytes read BEFORE routing for NIP-98 payload hashing lib.rs:194
    H->>R: route(path, method, body_bytes, origin)
    R-->>H: Response
    H-->>C: with_cors(resp)

    Note over F: INVARIANT: no error ever leaks to the workers-rs framework - a leaked Debug string is not valid JSON. Unhandled errors become a 500 JSON body lib.rs:151
    Note over H: The origin is THREADED through the route functions rather than stored in a thread_local (P2-06) lib.rs:190
```

## NF-02.2 Route table — public, sprint-API and legacy tiers

```mermaid
flowchart TB
    subgraph pub["Public - no auth"]
        DID["GET /.well-known/did/nostr/{pk}.json<br/>lib.rs:233 to did::handle_did_document nostr-bbs-auth-worker/src/did.rs:20"]
        HEALTH["GET /health lib.rs:240"]
        REGO["POST /auth/register/options lib.rs:253"]
        REGV["POST /auth/register/verify lib.rs:258"]
        LOGO["POST /auth/login/options lib.rs:264"]
        LOGV["POST /auth/login/verify lib.rs:269"]
        LOOKUP["POST /auth/lookup lib.rs:274"]
        UCHECK["GET /api/username/check lib.rs:584"]
        URESOLVE["GET /api/username/resolve lib.rs:589"]
        IPREV["GET /api/invites/{code} preview lib.rs:442"]
    end
    subgraph sprint["route_sprint_api - each handler does its own NIP-98 + gate"]
        MOD["mod: ban mute warn unban unmute lib.rs:375 | report lib.rs:383 | actions lib.rs:387 | reports lib.rs:391 | reports/{id}/action lib.rs:396"]
        WOT["wot: status lib.rs:413 | set-referente lib.rs:417 | refresh lib.rs:421 | override add/remove lib.rs:425"]
        INV["invites: create lib.rs:433 | mine lib.rs:437 | revoke + redeem lib.rs:442"]
        WEL["welcome: config lib.rs:471 | configure lib.rs:475 | set-bot-key lib.rs:479 | test lib.rs:483"]
        ADM["admins: list lib.rs:489 | add lib.rs:493 | remove lib.rs:497 | delete-member lib.rs:503"]
        GOV["governance: agents lib.rs:509 register lib.rs:513 provision lib.rs:518 revoke lib.rs:523 | cases lib.rs:528 :532 | decisions lib.rs:538 | roles grant lib.rs:542 revoke lib.rs:546 list lib.rs:550"]
        DEV["devices: list lib.rs:558 | register lib.rs:562 | revoke lib.rs:566"]
        NAME["username: claim lib.rs:593 | release lib.rs:597 | profile/real-name lib.rs:604 :608 | admin/registrations lib.rs:613 :618"]
        NIP1984["GET /api/moderation/reports lib.rs:572"]
        NPOD["POST /api/native-pod/provision lib.rs:578"]
    end
    subgraph legacy["Legacy /api/ tier - central NIP-98 verify"]
        PROF["GET /api/profile lib.rs:333 to pod::handle_profile nostr-bbs-auth-worker/src/pod.rs:9"]
    end

    ENTRY["route lib.rs:355"] --> pub
    ENTRY --> sprint
    ENTRY -->|"falls through when route_sprint_api returns None lib.rs:287"| legacy
    legacy -->|"no match"| NF["404 Not found lib.rs:346"]

    N1["Sprint handlers are dispatched BEFORE the legacy verify branch because they run their own<br/>require_admin / require_authed gates lib.rs:279-287"]
    N2["ANOMALY O11 confirmed live: /api/native-pod/provision lib.rs:578 requires NATIVE_POD_URL and the<br/>NATIVE_POD_ADMIN_KEY secret; both are placeholders in the template nostr-bbs-auth-worker/wrangler.toml:52"]
```

## NF-02.3 Passkey registration — the full verify chain

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser
    participant RO as register_options<br/>nostr-bbs-auth-worker/src/webauthn.rs:723
    participant D1 as D1 challenges / webauthn_credentials
    participant RV as register_verify<br/>nostr-bbs-auth-worker/src/webauthn.rs:814
    participant WOT as wot::is_allowed_by_wot<br/>nostr-bbs-auth-worker/src/wot.rs:431

    C->>RO: POST /auth/register/options
    RO->>RO: 32-byte challenge from getrandom webauthn.rs:733
    RO->>RO: server-controlled PRF salt, 32 bytes webauthn.rs:739
    RO->>D1: batch DELETE expired + INSERT challenge (pubkey column = the challenge) webauthn.rs:756
    RO->>RO: rp_id_required fails CLOSED on a placeholder RP_ID webauthn.rs:773
    RO-->>C: options with residentKey required, userVerification required webauthn.rs:791
    C->>RV: POST /auth/register/verify
    RV->>RV: reject cross-origin ceremony webauthn.rs:878
    RV->>RV: clientData.origin must equal EXPECTED_ORIGIN webauthn.rs:886
    RV->>D1: challenge must exist, be unexpired AND have pubkey = challenge webauthn.rs:906
    RV->>RV: attestation fmt must be none or packed webauthn.rs:933
    RV->>RV: constant-time RP-ID-hash compare webauthn.rs:951
    RV->>RV: flags UP bit0 webauthn.rs:957 and UV bit2 webauthn.rs:960 must be set
    RV->>RV: COSE key must parse as ES256 P-256 webauthn.rs:976
    RV->>D1: reject duplicate pubkey with 409 webauthn.rs:989
    RV->>WOT: registration gate, invite code is the documented bypass webauthn.rs:1001
    RV->>RV: PRF salt DERIVED server-side, client prfSalt ignored webauthn.rs:1028
    RV->>D1: batch INSERT credential + DELETE challenge webauthn.rs:1037
    RV-->>C: verified, pubkey, didNostr webauthn.rs:1054

    Note over RV: INVARIANT P1-1 challenge-bucket binding: register stores the challenge under pubkey = challenge, so a login-bucket challenge cannot be consumed by a registration verify webauthn.rs:898-905
    Note over RV: INVARIANT P0-01: accepting a client-supplied PRF salt would let an attacker derive a different key and bypass the passkey model webauthn.rs:1016-1021
    Note over WOT: is_allowed_by_wot returns Ok(true) when WoT is disabled, so the gate is a no-op on open instances webauthn.rs:996-999
```

## NF-02.4 Passkey login — assertion verification and clone detection

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser
    participant LO as login_options<br/>nostr-bbs-auth-worker/src/webauthn.rs:1080
    participant LV as login_verify<br/>nostr-bbs-auth-worker/src/webauthn.rs:1178
    participant D1 as D1

    C->>LO: POST /auth/login/options
    LO-->>C: indistinguishable shape for known and unknown pubkeys webauthn.rs:1069-1078
    C->>LV: POST /auth/login/verify
    LV->>LV: optional NIP-98 header verified with replay protection webauthn.rs:1200
    LV->>D1: SELECT credential_id, public_key, counter webauthn.rs:1215
    LV->>LV: reject cross-origin webauthn.rs:1261, origin equality webauthn.rs:1266
    LV->>D1: challenge must match pubkey IN (pubkey, __discoverable__) webauthn.rs:1293
    LV->>LV: RP-ID hash webauthn.rs:1330, UP webauthn.rs:1336, UV webauthn.rs:1339
    LV->>LV: sign_count must strictly increase unless 0 webauthn.rs:1349
    LV->>LV: signed_data = authenticatorData || SHA-256(clientDataJSON) webauthn.rs:1387
    LV->>LV: DER ECDSA parse webauthn.rs:1391 then P-256 verify webauthn.rs:1400
    LV->>D1: batch UPDATE counter + DELETE challenge webauthn.rs:1405
    LV-->>C: verified, pubkey, didNostr webauthn.rs:1413

    Note over LV: signCount 0 means the authenticator keeps no counter, so the clone check is skipped - otherwise a non-increasing counter is Credential replay detected webauthn.rs:1348-1350
    Note over LO: Audit C2: a 404 on unknown pubkey was an enumeration oracle - an unregistered pubkey now gets a fresh challenge, empty allowCredentials and a deterministic meaningless prfSalt webauthn.rs:1070-1078
```

## NF-02.5 NIP-98 gate and the single-use replay store

```mermaid
sequenceDiagram
    autonumber
    participant H as Route handler
    participant A as auth::verify_nip98_replay<br/>nostr-bbs-auth-worker/src/auth.rs:16
    participant RLC as nostr_bbs_rate_limit::verify_nip98<br/>nostr-bbs-rate-limit/src/lib.rs:1
    participant D1 as nostr-bbs-auth D1 (binding DB)

    H->>A: header, canonical URL, method, body bytes
    A->>RLC: delegate with REPLAY_DB = "DB" nostr-bbs-auth-worker/src/auth.rs:11
    RLC->>RLC: Schnorr + URL + method + payload-hash + created_at window checks
    RLC->>D1: atomic INSERT OR IGNORE on the replay row
    alt row already present
        D1-->>H: replay rejected
    end
    RLC-->>H: Nip98Token { pubkey, .. }

    Note over A: INVARIANT: every worker binds its replay store to the SAME database (nostr-bbs-auth) so cross-worker replay is detected nostr-bbs-auth-worker/src/auth.rs:9-11 - see NF-08.4
    Note over H: canonical_url composes the ACTUAL origin with the path so a workers.dev token verifies against a custom-domain deployment nostr-bbs-auth-worker/src/lib.rs:312
    Note over RLC: EXTERNAL: agentbox terminates NIP-98 at its own proxy with an independent single-use cache - see AB-10
```

## NF-02.6 The two gates every sprint handler chooses between

```mermaid
flowchart LR
    RA["require_admin<br/>nostr-bbs-auth-worker/src/admin.rs:112"]
    RU["require_authed<br/>nostr-bbs-auth-worker/src/admin.rs:145"]
    ISA["is_admin - static union RELAY_DB union DB<br/>nostr-bbs-auth-worker/src/admin.rs:57"]

    RA --> ISA
    ADMINONLY["Admin-only: mod actions moderation.rs:209 | wot status wot.rs:128 | welcome config welcome.rs:148<br/>| admins add admins.rs:161 | governance register governance_api.rs:266 | provision governance_api.rs:340<br/>| roles grant governance_api.rs:554 | admin registrations username.rs:689"]
    AUTHED["Authed-only: mod report moderation.rs:316 | invite create invites.rs:291 | redeem invites.rs:577<br/>| devices register devices.rs:361 | list devices.rs:452 | revoke devices.rs:488<br/>| username claim username.rs:568 | governance cases governance_api.rs:446 | decisions governance_api.rs:510"]

    RA --> ADMINONLY
    RU --> AUTHED

    N1["INVARIANT device ownership: the owner is ALWAYS the NIP-98 author, never a body field -<br/>no owner leaks through devices.rs:641, gate note nostr-bbs-auth-worker/src/lib.rs:555-557"]
    N2["wot status is admin-gated (wot.rs:128) even though it reads as a self-status endpoint -<br/>a member cannot query their own WoT standing through this API"]
    N3["EXTERNAL: the same admin identity is enforced independently at relay ingress - see NF-03 and NF-08.7"]
```

## NF-02.7 Device-key registry — default-off, dual-worker gate, relay-owned table

```mermaid
stateDiagram-v2
    [*] --> Disabled: DEVICE_KEYS_ENABLED unset/empty/other<br/>nostr-bbs-auth-worker/src/devices.rs:112
    Disabled --> Enabled: exact string "true"<br/>nostr-bbs-auth-worker/src/devices.rs:100
    Enabled --> Registered: POST /api/devices/register<br/>nostr-bbs-auth-worker/src/devices.rs:350
    Registered --> Listed: GET /api/devices<br/>nostr-bbs-auth-worker/src/devices.rs:446
    Registered --> Revoked: POST /api/devices/revoke sets revoked = 1<br/>nostr-bbs-auth-worker/src/devices.rs:477
    Revoked --> [*]
    Disabled --> [*]: handlers return early devices.rs:356

    note right of Registered
        device_keys lives in the RELAY worker's D1 (RELAY_DB binding)
        so the relay DO can read it at NIP-42 AUTH with no cross-worker call
        nostr-bbs-auth-worker/src/devices.rs:88
        table created idempotently on first use devices.rs:123
        owner index devices.rs:132
    end note
    note right of Disabled
        DIVERGENCE: with the gate off a known device to owner mapping is IGNORED
        and revocation has no effect at AUTH - the whole ADR-099/100 device story
        is dormant in a default deployment (both templates ship "false")
    end note
```

## NF-02.8 Membership creation paths — invite redemption and username claim

```mermaid
sequenceDiagram
    autonumber
    participant I as Inviter (authed)
    participant IC as invites::handle_create<br/>nostr-bbs-auth-worker/src/invites.rs:284
    participant N as New user
    participant IR as invites::handle_redeem<br/>nostr-bbs-auth-worker/src/invites.rs:565
    participant UC as username::claim<br/>nostr-bbs-auth-worker/src/username.rs:210
    participant AD as auth D1 members / username_reservations
    participant RD as relay D1 whitelist

    I->>IC: POST /api/invites/create
    IC->>AD: invitations row schema.rs:123
    N->>IR: POST /api/invites/{code}/redeem
    IR->>AD: invitation_redemptions schema.rs:136 + members row schema.rs:116
    N->>UC: POST /api/username/claim
    UC->>AD: username_reservations upsert schema.rs:208
    UC->>RD: whitelist row with the auto-approved cohort set

    Note over UC: INVARIANT pubkey-uniqueness on username_reservations, claimed via upsert-with-DO-NOTHING so the SELECT-then-branch race cannot double-claim invites.rs:217
    Note over IC: consume_for_registration is the registration-time bypass of the WoT gate invites.rs:785
    Note over UC: The granted cohort set is config-driven, not hardcoded - see NF-08.3
```

## NF-02.9 Identity surfaces the auth worker publishes

```mermaid
flowchart LR
    DIDDOC["GET /.well-known/did/nostr/{pk}.json<br/>nostr-bbs-auth-worker/src/did.rs:20"]
    MULTI["Canonical did:nostr Multikey document<br/>asserted nostr-bbs-auth-worker/src/did.rs:73"]
    NOAKA["No alsoKnownAs on the canonical doc<br/>asserted nostr-bbs-auth-worker/src/did.rs:87"]
    NIP05["NIP-05 resolve - local registry then federated pod probe<br/>resolve nostr-bbs-auth-worker/src/username.rs:844<br/>federated URL builder username.rs:829<br/>pubkey extraction username.rs:812"]
    PROFILE["GET /api/profile<br/>nostr-bbs-auth-worker/src/pod.rs:9"]
    CRYPTO["Master-key envelope for stored secrets<br/>encrypt nostr-bbs-auth-worker/src/crypto.rs:77<br/>decrypt crypto.rs:97<br/>nsec helpers crypto.rs:117 crypto.rs:127"]

    DIDDOC --> MULTI --> NOAKA
    NIP05 --> PROFILE

    N1["EXTERNAL: the Multikey encoder of record is solid-pod-rs did_nostr_types - see the solid-pod-rs area (SP-*)<br/>and the estate identity mesh ES-04"]
    N2["EXTERNAL: VisionClaw resolves the same did:nostr identifiers on its own identity spine - see VC-23 and VC-33"]
    N3["DIVERGENCE BASELINE-architecture.md: the CF-Workers pod-federation fallback is degenerate - the<br/>federated resolve returns data D1 already holds, because the pod-resident NIP-05 endpoint is one of the<br/>three wasm32-unreachable Phase-1 surfaces - see NF-04"]
```
