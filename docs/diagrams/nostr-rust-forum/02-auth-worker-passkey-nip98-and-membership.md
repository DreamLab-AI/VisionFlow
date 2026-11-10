---
id: NF-02
title: auth-worker — passkey ceremonies, NIP-98 gating, membership and the REST surface
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2003, ADR-2004, ADR-2010, ADR-2011]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
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
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/rationale_check.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/nip98.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs
verified_commit: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df
---

## NF-02.1 Request entry — bootstrap, rate limit, body pre-read, dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant F as fetch<br/>nostr-bbs-auth-worker/src/lib.rs:144
    participant H as handle_request<br/>nostr-bbs-auth-worker/src/lib.rs:158
    participant S as schema::ensure_schema<br/>nostr-bbs-auth-worker/src/schema.rs:19
    participant RL as check_rate_limit<br/>nostr-bbs-auth-worker/src/lib.rs:175
    participant R as route<br/>nostr-bbs-auth-worker/src/lib.rs:225

    C->>F: HTTP request
    F->>H: handle_request (all errors caught by the wrapper)
    H->>S: idempotent D1 bootstrap at the top of every request, no isolate latch nostr-bbs-auth-worker/src/lib.rs:163
    H->>H: ensure_replay_schema on the DB binding, also every request nostr-bbs-auth-worker/src/lib.rs:164
    H->>H: Options preflight returns 204 + CORS, so preflights are never rate-limited nostr-bbs-auth-worker/src/lib.rs:167
    H->>RL: check_rate_limit 20 req / 60 s per IP nostr-bbs-auth-worker/src/lib.rs:175
    alt over budget
        RL-->>C: 429 Too many requests nostr-bbs-auth-worker/src/lib.rs:176
    end
    H->>H: origin = the ACTUAL request URL, not EXPECTED_ORIGIN nostr-bbs-auth-worker/src/lib.rs:191
    H->>H: body bytes read BEFORE routing for NIP-98 payload hashing nostr-bbs-auth-worker/src/lib.rs:195
    H->>R: route(path, method, body_bytes, origin)
    R-->>H: Response
    H-->>C: with_cors(resp)

    Note over F: INVARIANT: no error ever leaks to the workers-rs framework - a leaked Debug string is not valid JSON. Unhandled errors become a 500 JSON body nostr-bbs-auth-worker/src/lib.rs:152
    Note over H: The origin is THREADED through the route functions rather than stored in a thread_local (P2-06) nostr-bbs-auth-worker/src/lib.rs:191
```

## NF-02.2 Route table — public, sprint-API and legacy tiers

```mermaid
flowchart TB
    subgraph pub["Public - no auth"]
        DID["GET /.well-known/did/nostr/{pk}.json<br/>nostr-bbs-auth-worker/src/lib.rs:234 to did::handle_did_document nostr-bbs-auth-worker/src/did.rs:23<br/>rendering delegates to nostr_bbs_core::did, canonical Multikey doc carries no alsoKnownAs did.rs:8,89,90<br/>the route holds only the D1 identifier, so publicKeyMultibase is the fe70102 even-y form, did-nostr#145 did.rs:18"]
        HEALTH["GET /health nostr-bbs-auth-worker/src/lib.rs:241"]
        REGO["POST /auth/register/options nostr-bbs-auth-worker/src/lib.rs:254"]
        REGV["POST /auth/register/verify nostr-bbs-auth-worker/src/lib.rs:259"]
        LOGO["POST /auth/login/options nostr-bbs-auth-worker/src/lib.rs:265"]
        LOGV["POST /auth/login/verify nostr-bbs-auth-worker/src/lib.rs:270"]
        LOOKUP["POST /auth/lookup nostr-bbs-auth-worker/src/lib.rs:275"]
        UCHECK["GET /api/username/check nostr-bbs-auth-worker/src/lib.rs:622"]
        URESOLVE["GET /api/username/resolve nostr-bbs-auth-worker/src/lib.rs:627<br/>resolve D1 is the trust root, federated pod probe only on miss in Federated mode username.rs:844,829,812"]
        IPREV["GET /api/invites/{code} preview nostr-bbs-auth-worker/src/lib.rs:443"]
    end
    subgraph sprint["route_sprint_api - each handler does its own NIP-98 + gate"]
        MOD["mod: ban mute warn unban unmute nostr-bbs-auth-worker/src/lib.rs:376 | report nostr-bbs-auth-worker/src/lib.rs:384 | actions nostr-bbs-auth-worker/src/lib.rs:388 | reports nostr-bbs-auth-worker/src/lib.rs:392 | reports/{id}/action nostr-bbs-auth-worker/src/lib.rs:397"]
        WOT["wot: status nostr-bbs-auth-worker/src/lib.rs:414 | set-referente nostr-bbs-auth-worker/src/lib.rs:418 | refresh nostr-bbs-auth-worker/src/lib.rs:422 | override add/remove nostr-bbs-auth-worker/src/lib.rs:426"]
        INV["invites: create nostr-bbs-auth-worker/src/lib.rs:434 | mine nostr-bbs-auth-worker/src/lib.rs:438 | revoke + redeem nostr-bbs-auth-worker/src/lib.rs:443"]
        WEL["welcome: config nostr-bbs-auth-worker/src/lib.rs:472 | configure nostr-bbs-auth-worker/src/lib.rs:476 | set-bot-key nostr-bbs-auth-worker/src/lib.rs:480 | test nostr-bbs-auth-worker/src/lib.rs:484<br/>set-bot-key nsec is sealed with ChaCha20-Poly1305 under WELCOME_MASTER_KEY crypto.rs:77,97"]
        ADM["admins: list nostr-bbs-auth-worker/src/lib.rs:490 | add nostr-bbs-auth-worker/src/lib.rs:494 | remove nostr-bbs-auth-worker/src/lib.rs:498 | delete-member nostr-bbs-auth-worker/src/lib.rs:504"]
        GOV["governance: agents nostr-bbs-auth-worker/src/lib.rs:510 register nostr-bbs-auth-worker/src/lib.rs:514 provision nostr-bbs-auth-worker/src/lib.rs:519 revoke nostr-bbs-auth-worker/src/lib.rs:524 | cases nostr-bbs-auth-worker/src/lib.rs:529 :533 | decisions nostr-bbs-auth-worker/src/lib.rs:555 | roles grant nostr-bbs-auth-worker/src/lib.rs:559 revoke nostr-bbs-auth-worker/src/lib.rs:563 list nostr-bbs-auth-worker/src/lib.rs:588 | receipt application nostr-bbs-auth-worker/src/lib.rs:568 | reviewers nostr-bbs-auth-worker/src/lib.rs:584 | OPT-IN POST cases/{id}/rationale-check, routed only while JEV_RATIONALE_CHECK is on nostr-bbs-auth-worker/src/lib.rs:542, handler nostr-bbs-auth-worker/src/rationale_check.rs:257"]
        DEV["devices: list nostr-bbs-auth-worker/src/lib.rs:596 | register nostr-bbs-auth-worker/src/lib.rs:600 | revoke nostr-bbs-auth-worker/src/lib.rs:604"]
        NAME["username: claim nostr-bbs-auth-worker/src/lib.rs:631 | release nostr-bbs-auth-worker/src/lib.rs:635 | profile/real-name nostr-bbs-auth-worker/src/lib.rs:642 :646 | admin/registrations nostr-bbs-auth-worker/src/lib.rs:651, dismiss nostr-bbs-auth-worker/src/lib.rs:656"]
        NIP1984["GET /api/moderation/reports nostr-bbs-auth-worker/src/lib.rs:610"]
        NPOD["POST /api/native-pod/provision nostr-bbs-auth-worker/src/lib.rs:616"]
    end
    subgraph legacy["Legacy /api/ tier - central NIP-98 verify"]
        PROF["GET /api/profile nostr-bbs-auth-worker/src/lib.rs:334 to pod::handle_profile nostr-bbs-auth-worker/src/pod.rs:9"]
    end

    DID ~~~ HEALTH ~~~ REGO ~~~ REGV ~~~ LOGO ~~~ LOGV ~~~ LOOKUP ~~~ UCHECK ~~~ URESOLVE ~~~ IPREV
    MOD ~~~ WOT ~~~ INV ~~~ WEL ~~~ ADM ~~~ GOV ~~~ DEV ~~~ NAME ~~~ NIP1984 ~~~ NPOD

    ENTRY["route nostr-bbs-auth-worker/src/lib.rs:225"] --> pub
    ENTRY --> sprint
    ENTRY -->|"falls through when route_sprint_api returns None nostr-bbs-auth-worker/src/lib.rs:288"| legacy
    legacy -->|"no match"| NF["404 Not found nostr-bbs-auth-worker/src/lib.rs:349"]

    N1["Sprint handlers are dispatched BEFORE the legacy verify branch because they run their own<br/>require_admin / require_authed gates nostr-bbs-auth-worker/src/lib.rs:280-288"]
    N2["ANOMALY O11 (recorded live by an earlier review, not re-run here): /api/native-pod/provision nostr-bbs-auth-worker/src/lib.rs:616 requires NATIVE_POD_URL and the<br/>NATIVE_POD_ADMIN_KEY secret; both are placeholders in the template nostr-bbs-auth-worker/wrangler.toml:66"]
```

**Invariant:** `SESSIONS` (the rate limiter's `rl:{ip}:{bucket}` keys) and `KV` (the admin pubkey cache) bind the same KV namespace id, so the admin cache key must never start with `rl:` (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admins.rs:347`, `../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml:22-25`).

**Open:** the rationale-check route and its flag `JEV_RATIONALE_CHECK` are off in the library but ON in the committed auth template, `JEV_RATIONALE_CHECK = "1"` (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml:53`; flag name `../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/rationale_check.rs:36`); the check is advisory and, since the 2026-10-30 pin, cloud by default: locality is the operator's own `JEV_RATIONALE_CHECK_LOCAL_ONLY` switch (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/rationale_check.rs:38`, read at `../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/rationale_check.rs:303`), default off, so the proxy or TypeSafe with `TYPESAFE_API_KEY` may answer; setting it to `1` makes the route report unavailable until a deployment supplies a sovereign endpoint (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml:54-58`).

## NF-02.3 Passkey registration — the full verify chain

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser
    participant RO as register_options<br/>nostr-bbs-auth-worker/src/webauthn.rs:767
    participant D1 as D1 challenges / webauthn_credentials
    participant RV as register_verify<br/>nostr-bbs-auth-worker/src/webauthn.rs:871
    participant WOT as wot::is_allowed_by_wot<br/>nostr-bbs-auth-worker/src/wot.rs:431

    C->>RO: POST /auth/register/options
    RO->>RO: 32-byte challenge from getrandom webauthn.rs:777
    RO->>RO: server-controlled PRF salt, 32 bytes webauthn.rs:783
    RO->>D1: batch DELETE expired + INSERT challenge (pubkey column = the challenge) + INSERT issued salt webauthn.rs:803 webauthn.rs:814
    RO->>RO: rp_id_required rejects missing or blank RP_ID webauthn.rs:829
    RO-->>C: options with residentKey required, userVerification required webauthn.rs:848
    C->>RV: POST /auth/register/verify
    RV->>RV: mandatory NIP-98 header and matching author webauthn.rs:893 webauthn.rs:906
    RV->>RV: reject cross-origin ceremony webauthn.rs:935
    RV->>RV: clientData.origin must equal EXPECTED_ORIGIN webauthn.rs:943
    RV->>D1: challenge joined to its issued PRF salt must exist, be unexpired AND have pubkey = challenge webauthn.rs:719 webauthn.rs:967
    RV->>RV: attestation fmt must be none or packed webauthn.rs:990
    RV->>RV: constant-time RP-ID-hash compare webauthn.rs:1009
    RV->>RV: flags UP bit0 webauthn.rs:1015 and UV bit2 webauthn.rs:1018 must be set
    RV->>RV: COSE coordinates must form a valid P-256 point webauthn.rs:1034
    RV->>D1: reject duplicate pubkey with 409 webauthn.rs:1047
    RV->>WOT: registration gate, invite code is the documented bypass webauthn.rs:1059
    RV->>RV: PRF salt is the one issued at options time and read with the challenge, never caller-supplied webauthn.rs:1077
    RV->>D1: batch INSERT credential guarded by the issued salt + DELETE challenge + DELETE salt webauthn.rs:1081 webauthn.rs:1100
    RV-->>C: verified, pubkey, didNostr webauthn.rs:1112

    Note over D1,RV: Registration challenge uses pubkey = challenge<br/>Login-bucket challenges do not satisfy this lookup webauthn.rs:955-961
    Note over RV: RESOLVED at nostr-rust-forum 6064622 - the stored salt is the random options salt<br/>Registration and login now share one server-issued salt webauthn.rs:783 webauthn.rs:1077
    Note over RV,WOT: WoT disabled returns Ok(true)<br/>No trust gate on open instances webauthn.rs:1054-1057
```

## NF-02.4 Passkey login — assertion verification and clone detection

```mermaid
sequenceDiagram
    autonumber
    participant C as Browser
    participant LO as login_options<br/>nostr-bbs-auth-worker/src/webauthn.rs:1138
    participant LV as login_verify<br/>nostr-bbs-auth-worker/src/webauthn.rs:1236
    participant D1 as D1

    C->>LO: POST /auth/login/options
    LO-->>C: empty allowCredentials plus selected salt webauthn.rs:1161 webauthn.rs:1186
    C->>LV: POST /auth/login/verify
    LV->>LV: mandatory NIP-98 header verified with replay protection webauthn.rs:1258
    LV->>D1: SELECT credential_id, public_key, counter webauthn.rs:1273
    LV->>LV: reject cross-origin webauthn.rs:1319, origin equality webauthn.rs:1324
    LV->>D1: challenge must match pubkey IN (pubkey, __discoverable__) webauthn.rs:1351
    LV->>LV: RP-ID hash webauthn.rs:1388, UP webauthn.rs:1394, UV webauthn.rs:1397
    LV->>LV: sign_count must strictly increase unless 0 webauthn.rs:1407
    LV->>LV: signed_data = authenticatorData ||<br/>SHA-256(clientDataJSON) webauthn.rs:1445
    LV->>LV: DER ECDSA parse webauthn.rs:1449 then P-256 verify webauthn.rs:1458
    LV->>D1: batch UPDATE counter + DELETE challenge webauthn.rs:1463
    LV-->>C: verified, pubkey, didNostr webauthn.rs:1471

    Note over LV,D1: New signCount zero bypasses the increase check<br/>Otherwise a non-increasing count is rejected webauthn.rs:1406-1408
    Note over LO: Equal response shape does not close the separate public lookup route<br/>or the empty-secret fallback condition webauthn.rs:78 webauthn.rs:1485
```

[The passkey contract review](../../estate-review/2026-11-01-forum-passkey-contracts.md) qualifies NF-02.3/4: at its stamp both clients omitted the required registration header and the issued and stored salts differed; both are repaired at 6064622 (both clients sign register/verify with NIP-98, `../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:182`, `../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs:356`, and the stored salt is the issued one, `../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1077`); lookup remains public, and challenge reads are separate from unconditional consumption. NF-23 records the probes and parser limits. These worker sequences show handler order, not a successfully completed client ceremony.

## NF-02.5 NIP-98 gate and the single-use replay store

```mermaid
sequenceDiagram
    autonumber
    participant H as Route handler
    participant A as auth::verify_nip98_replay<br/>nostr-bbs-auth-worker/src/auth.rs:15
    participant RLC as nostr_bbs_rate_limit::verify_nip98<br/>nostr-bbs-rate-limit/src/replay.rs:148, re-exported nostr-bbs-rate-limit/src/lib.rs:21
    participant D1 as nostr-bbs-auth D1 (binding DB)

    H->>A: header, canonical URL, method, body bytes
    A->>RLC: delegate with REPLAY_DB = "DB" nostr-bbs-auth-worker/src/auth.rs:11
    RLC->>RLC: core verify_token_full - Schnorr, URL, method, payload hash, created_at window - BEFORE any replay write nostr-bbs-core/src/nip98.rs:576
    RLC->>D1: then seen_or_record, an atomic INSERT OR IGNORE on the replay row nostr-bbs-core/src/nip98.rs:585 nostr-bbs-rate-limit/src/replay.rs:13
    alt row already present
        D1-->>H: replay rejected
    end
    RLC-->>H: Nip98Token { pubkey, .. }

    Note over A: INVARIANT: every worker binds its replay store to the SAME database (nostr-bbs-auth) so cross-worker replay is detected nostr-bbs-auth-worker/src/auth.rs:9-11 - the auth template's DB id nostr-bbs-auth-worker/wrangler.toml:11 is the id the relay, pod and search templates give REPLAY_DB - see NF-08.4
    Note over D1: OPEN: RESOLVED at nostr-rust-forum 6064622 - when D1 returns no write metadata the store returns an error, so the replay check fails closed on that path nostr-bbs-rate-limit/src/replay.rs:40-43 nostr-bbs-rate-limit/src/replay.rs:47-55
    Note over H: canonical_url composes the ACTUAL origin with the path so a workers.dev token verifies against a custom-domain deployment nostr-bbs-auth-worker/src/lib.rs:313
    Note over RLC: EXTERNAL: agentbox terminates NIP-98 at its own proxy with an independent single-use cache - see AB-10
```

**Open:** RESOLVED at nostr-rust-forum 6064622 — the D1 replay store no longer answers first-seen when D1 returns no write metadata: absent metadata is an error (`../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs:40-43`) and so is an omitted changed-row count (`../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs:47-55`), which the verifier surfaces as a replay-backend error rather than a pass (`../nostr-rust-forum/crates/nostr-bbs-core/src/nip98.rs:585-588`; the trait contract is `true` on first observation, `../nostr-rust-forum/crates/nostr-bbs-core/src/nip98.rs:94-96`). The full signature, URL, method, payload and freshness check still runs first (`../nostr-rust-forum/crates/nostr-bbs-core/src/nip98.rs:576`).

## NF-02.6 The two gates every sprint handler chooses between

```mermaid
flowchart LR
    RA["require_admin<br/>nostr-bbs-auth-worker/src/admin.rs:92"]
    RU["require_authed<br/>nostr-bbs-auth-worker/src/admin.rs:125"]
    ISA["is_admin - static union RELAY_DB union DB<br/>nostr-bbs-auth-worker/src/admin.rs:53"]

    RA --> ISA
    ADMINONLY["Admin-only: mod actions moderation.rs:197 | wot status wot.rs:128 | welcome config welcome.rs:148<br/>| admins add admins.rs:136 | governance register governance_api.rs:362 | provision governance_api.rs:441<br/>| roles grant governance_api.rs:670 | admin registrations username.rs:689<br/>| reviewer telemetry governance_api.rs:1407"]
    AUTHED["Authed-only: mod report moderation.rs:304 | invite create invites.rs:292 | redeem invites.rs:578<br/>| devices register devices.rs:376 | list devices.rs:467 | revoke devices.rs:503<br/>| username claim username.rs:568 | governance cases governance_api.rs:561 | decisions governance_api.rs:626<br/>| receipt application governance_api.rs:1130 - authed, then authorised by ROLE inside the handler"]

    RA --> ADMINONLY
    RU --> AUTHED

    N1["INVARIANT device ownership: the owner is ALWAYS the NIP-98 author, never a body field -<br/>no owner leaks through devices.rs:656, gate note nostr-bbs-auth-worker/src/lib.rs:593-595"]
    N2["wot status is admin-gated (wot.rs:128) even though it reads as a self-status endpoint -<br/>a member cannot query their own WoT standing through this API"]
    N3["EXTERNAL: the same admin identity is enforced independently at relay ingress - see NF-03 and NF-08.7"]
```

## NF-02.7 Device-key registry — independent gates and delayed revocation

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:100
    A["Auth worker exact true gate<br/>off returns 404 before database work"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:376
    B["Registration authenticates owner via NIP-98"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:398
    C["Require device-signed proof<br/>then exclude known principals"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:427
    D["Conditional upsert of relay D1 registry row<br/>another owner or a revoked row is refused with 409"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:526
    E["Owner-scoped revoke sets revoked to 1"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2069
    F["AUTH stores raw public key<br/>does not consult device registry"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2614
    G["Read/write scope resolves owner later<br/>through a 60-second cache"]
    A --> B --> C --> D --> E
    F --> G
    D -. registry lookup .-> G
```

[ADR-2004 review and NF-20](../../estate-review/2026-11-01-forum-device-keys.md) distinguish exact worker gating from client interoperability and revocation acceptance. The 2026-11-02 forum-workers review ([report](../../estate-review/2026-11-02-forum-workers-auth-relay-pod.md)) re-assessed every NF-02 panel at `c39f41f`.

## NF-02.8 Membership creation paths — invite redemption and username claim

```mermaid
sequenceDiagram
    autonumber
    participant I as Inviter (authed)
    participant IC as invites::handle_create<br/>nostr-bbs-auth-worker/src/invites.rs:285
    participant N as New user
    participant IR as invites::handle_redeem<br/>nostr-bbs-auth-worker/src/invites.rs:566
    participant UC as username::claim<br/>nostr-bbs-auth-worker/src/username.rs:210
    participant AD as auth D1 members / username_reservations
    participant RD as relay D1 whitelist

    I->>IC: POST /api/invites/create
    IC->>AD: invitations row schema.rs:124
    N->>IR: POST /api/invites/{code}/redeem
    IR->>AD: invitation_redemptions invites.rs:662 schema.rs:137 + members row invites.rs:672 schema.rs:117
    IR->>RD: zone invite only - grant_zone_cohorts on the relay whitelist invites.rs:223 invites.rs:238 invites.rs:263
    N->>UC: POST /api/username/claim
    UC->>AD: one atomic INSERT into username_reservations username.rs:237, table schema.rs:209
    UC->>RD: whitelist row with the auto-approved cohort set, ON CONFLICT DO NOTHING keeps admin-set cohorts username.rs:307

    Note over UC: INVARIANT one name per pubkey and one pubkey per name: the claim is a single INSERT that relies on UNIQUE(username) and UNIQUE(pubkey) username.rs:231 schema.rs:212, not an upsert
    Note over IR: INVARIANT ensure-then-merge: the zone grant inserts-or-keeps the whitelist row, then merges cohorts into it, so a concurrent username claim creating the same row cannot drop the grant invites.rs:208-218
    Note over IC: consume_for_registration is the registration-time bypass of the WoT gate invites.rs:786
    Note over UC: The granted cohort set is config-driven, not hardcoded - see NF-08.3
```

## NF-02.10 Application receipts — who may say what became of a decision

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant H as handle_receipt_application<br/>nostr-bbs-auth-worker/src/governance_api.rs:1121
    participant RD as relay D1
    participant PL as plan_application_advance<br/>nostr-bbs-auth-worker/src/governance_api.rs:1055
    participant RV as handle_list_reviewers<br/>nostr-bbs-auth-worker/src/governance_api.rs:1401

    C->>H: POST /api/governance/receipts/{event id}/application governance_api.rs:1128
    H->>H: require_authed - NIP-98 only, no admin gate here governance_api.rs:1130
    H->>H: parse the requested stage, unknown stage refused governance_api.rs:1140
    H->>RD: SELECT stage, case_id FROM governance_receipts governance_api.rs:1153
    RD-->>H: 404 when the receipt does not exist governance_api.rs:1158
    H->>RD: SELECT case state, created_by and the latest decision outcome governance_api.rs:1172
    H->>H: build the caller - is_admin, is_registered_agent, is_case_owner governance_api.rs:1183
    H->>PL: may this caller advance current to requested
    PL-->>H: refusal with its own status, or proceed governance_api.rs:1199
    H->>RD: UPDATE guarded by the CURRENT stage governance_api.rs:1210
    RD-->>C: 409 when another consumer won the race governance_api.rs:1232
    C->>RV: GET /api/governance/reviewers - ADMIN only governance_api.rs:1407

    Note over H: INVARIANT the case owner is matched case-insensitively, so a mixed-case NIP-98 pubkey still resolves to the owner it belongs to governance_api.rs:1181 governance_api.rs:1189
    Note over H: INVARIANT the stage write is a compare-and-swap - two consumers racing the same receipt cannot both win, the loser sees a 409 governance_api.rs:1202-1204
    Note over RV: Reviewer telemetry measures the humans, not only the agents, and is admin-gated where the case and decision reads are member-visible governance_api.rs:1407
```

**Drift:** the compare-and-swap comment names the guard `stage = ?5` (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs:1202`), but in the statement `?5` is the event id and the stage guard is `?6` (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs:1210`). The behaviour matches the invariant; only the comment is stale.
