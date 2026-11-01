---
id: SP-05
title: Identity — NIP-98, replay, did:nostr, did:key, WebID, Solid-OIDC and the IdP
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2003, ADR-2006]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/webid.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/nip98.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/replay.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/replay_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/self_signed.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/lws_cid.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/jwks.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/replay.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/did_nostr_types.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/scheme.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/cross_identity.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2003-solid-oidc-01-defer-lws10.md
  - ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-nostr/src/resolver.rs
  - ../solid-pod-rs/crates/solid-pod-rs-didkey/src/did.rs
  - ../solid-pod-rs/crates/solid-pod-rs-didkey/src/pubkey.rs
  - ../solid-pod-rs/crates/solid-pod-rs-didkey/src/jwt.rs
  - ../solid-pod-rs/crates/solid-pod-rs-didkey/src/verifier.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/provider.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/tokens.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/jwks.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/session.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/registration.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/credentials.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/passkey.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/schnorr.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/discovery.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/axum_binder.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/key_provisioning.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/user_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/account_delete.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/invites.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/auth.rs
  - ../solid-pod-rs/crates/solid-pod-rs-idp/src/password_change.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/rate_limit.rs
verified_commit: 09ec90f26aa573731460292d0e80343914cb43d9
---

## SP-05.1 REST identity extraction and anonymous fallback

```mermaid
flowchart TD
    REQ["inbound request"]
    EX["extract_pubkey_with_body<br/>lib.rs:603"]
    BEAR{"Authorization starts with 'Bearer '?<br/>lib.rs:611"}
    DEV["verify_dev_bearer — HMAC dev token, compiled only with the dev-bearer feature<br/>lib.rs:684"]
    PROD["production build: every Bearer header is anonymous<br/>lib.rs:617"]
    N98["nip98::verify_at over the reconstructed URL<br/>lib.rs:636"]
    REPLAY["NIP98_REPLAY.check_and_record<br/>lib.rs:641"]
    PK["Some(pubkey)"]
    ANON["None — anonymous; WAC decides whether public access is allowed"]
    AGENT["agent_uri: did:nostr prefix, or a WebID URL passed through<br/>lib.rs:657"]

    REQ --> EX --> BEAR
    BEAR -- "yes, dev-bearer build" --> DEV
    BEAR -- "yes, default build" --> PROD --> ANON
    BEAR -- no --> N98 --> REPLAY
    REPLAY -- fresh --> PK
    REPLAY -- replayed --> ANON
    DEV --> PK
    DEV -- "TOKEN_SECRET under 32 bytes, wrong aud, or lifetime over 24 h" --> ANON
    N98 -- verification failed --> ANON
    PK --> AGENT

    N1["The signed URL comes from scheme::request_url: the configured public origin or the plain<br/>Host header, with X-Forwarded-Proto honoured only from a trusted proxy.<br/>lib.rs:631 and scheme.rs:240"]
    N98 -.-> N1
    N4["INVARIANT: the dev bearer needs aud solid-pod-rs-dev-bearer and a lifetime of at most 24 h,<br/>and is off unless the cargo feature is set.<br/>lib.rs:722 and solid-pod-rs-server/Cargo.toml:133"]
    DEV -.-> N4
    N2["INVARIANT: a rejected replay returns None, i.e. anonymous — it never becomes<br/>an error path the caller can distinguish from 'no credential'.<br/>lib.rs:651"]
    REPLAY -.-> N2
    N3["EXTERNAL: agentbox mints a did:nostr per agent at spawn and signs pod writes<br/>with it; the same principal string reaches WAC here. See AB-11 and ES-04."]
    AGENT -.-> N3
```

## SP-05.2 NIP-98 verification, step by step

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant V as verify_at_with_policy<br/>auth/nip98.rs:116
    participant S as verify_schnorr_signature<br/>auth/nip98.rs:342
    participant ID as compute_event_id<br/>auth/nip98.rs:241

    C->>V: Authorization: Nostr <base64 kind-27235 event>
    V->>V: strip the "Nostr " prefix<br/>auth/nip98.rs:30
    V->>V: reject above MAX_EVENT_SIZE, 64 KiB, before AND after base64<br/>auth/nip98.rs:129
    V->>V: kind must equal HTTP_AUTH_KIND 27235<br/>auth/nip98.rs:138
    V->>V: pubkey must be 64 hex chars<br/>auth/nip98.rs:144
    V->>V: created_at within TIMESTAMP_TOLERANCE of now<br/>auth/nip98.rs:147
    V->>V: 'u' tag must match the expected URL per MatchPolicy<br/>auth/nip98.rs:154
    V->>V: 'method' tag must match, or be * under GitLenient<br/>auth/nip98.rs:173
    V->>V: 'payload' tag must equal sha256(raw body) only when supplied body is nonempty<br/>auth/nip98.rs:192
    V->>S: BIP-340 Schnorr verification — UNCONDITIONAL<br/>auth/nip98.rs:208
    S-->>V: ok, and the canonical event id is bound to the signature
    V->>ID: recompute the NIP-01 id from content, never trust event.id<br/>auth/nip98.rs:213
    V-->>C: Nip98Verified { pubkey, url, method, payload_hash, created_at, event_id }<br/>auth/nip98.rs:44
    Note over V: TIMESTAMP_TOLERANCE is 60 s (auth/nip98.rs:28) — a two-sided<br/>window, so the freshness check alone still leaves a replay gap. That gap is<br/>what SP-05.4 is meant to close, but its TTL is not bounded by freshness.
    Note over V: Supplied empty bytes skip payload verification, even with a hash for nonempty bytes.
    Note over V: Server PUT forwards empty bodies here (lib.rs:1735).
```

## SP-05.3 The two match policies

```mermaid
flowchart LR
    MP["MatchPolicy<br/>auth/nip98.rs:89"]
    ST["Strict — trailing-slash-trimmed URL and case-insensitive method<br/>auth/nip98.rs:91"]
    GL["GitLenient — * method plus raw string URL-prefix<br/>auth/nip98.rs:93"]
    REST["Strict verifier entry via verify_at; replay is caller-owned<br/>auth/nip98.rs:97"]
    GIT["BasicNostrExtractor for Git authentication<br/>solid-pod-rs-git/src/auth.rs:144"]

    MP --> ST --> REST
    MP --> GL --> GIT

    N["GitLenient exists because a stock git client has no Nostr keypair mid-push:<br/>one token signed over the repo base URL with method * covers every smart-protocol<br/>sub-request. Schnorr is verified, but prefix matching admits sibling names;
no ReplayStore claim occurs in this extractor.<br/>solid-pod-rs-git/src/auth.rs:154"]
    GL -.-> N
    N2["The git bridge also passes body = None, so a push body is NOT hash-bound.<br/>solid-pod-rs-git/src/auth.rs:148"]
    GIT -.-> N2
    N3["ADR-2006 calls for one NIP-98 verifier; Git delegates to it.
The IdP Schnorr login is a separate challenge protocol and calls k256 directly.
schnorr.rs:397"]
    MP -.-> N3
    N4["Native server rejects push before authentication; Git reads still use this extractor.
lib.rs:4750
lib.rs:4775"]
    GIT -.-> N4
```

## SP-05.4 The single-use replay store (ADR-2006)

```mermaid
flowchart TD
    START["check_and_record
auth/replay.rs:243"] --> LOCK["one async mutex
auth/replay.rs:247"]
    LOCK --> SEEN{"id already stored?"}
    SEEN -->|yes| LIVE{"age less than configured TTL?
auth/replay.rs:255"}
    LIVE -->|yes| REPLAY["Replayed, no expiry refresh"]
    LIVE -->|no| REFRESH["overwrite same slot, accept
auth/replay.rs:260"]
    SEEN -->|no| FULL{"at capacity?
auth/replay.rs:266"}
    FULL -->|no| INSERT["insert and accept"]
    FULL -->|yes| RECLAIM["remove entries at or beyond TTL
auth/replay.rs:200"]
    RECLAIM --> SPACE{"slot available?"}
    SPACE -->|yes| INSERT
    SPACE -->|no| REFUSE["CapacityExhausted, refuse new id
auth/replay.rs:275"]
    REFUSE -.-> KEEP["Never evict an unexpired entry.
Size for rate times retention.
auth/replay.rs:99"]
    LIVE -.-> LIMIT["TTL is not validated against token freshness.
Environment accepts any positive seconds.
Constructor also permits zero.
auth/replay.rs:133
auth/replay.rs:149"]
```

## SP-05.5 The ReplayStore seam and its tier limits

```mermaid
flowchart TD
    TRAIT["ReplayStore: atomic check_and_record
auth/replay_store.rs:90"]
    CACHE["Nip98ReplayCache
process-local reference implementation
auth/replay.rs:243"]
    ERROR["Replayed or CapacityExhausted
non-exhaustive error enum
auth/replay_store.rs:56
auth/replay_store.rs:75"]
    RETAIN["No live eviction and no refresh on rejection
auth/replay_store.rs:108"]
    DEFAULTS["Default: 120 seconds, 10,000 entries
auth/replay.rs:78
auth/replay.rs:82"]
    ENV["TTL and capacity environment overrides
auth/replay.rs:85
auth/replay.rs:86"]
    LIMIT["Restart forgets ids, replicas do not share state.
Reuse is possible while the token remains fresh.
auth/replay_store.rs:115"]
    SEAM["Another tier must implement and consume the trait.
Pinning its crate alone does not integrate a shared store.
auth/replay_store.rs:34"]
    CACHE --> TRAIT
    TRAIT --> ERROR
    TRAIT --> RETAIN
    CACHE --> DEFAULTS
    CACHE --> ENV
    CACHE --> LIMIT
    TRAIT --> SEAM
```

## SP-05.7 Solid-OIDC DPoP proof checks and optional replay admission

```mermaid
flowchart TD
    CALL["caller supplies expected URL, method, time<br/>optional access-token hash and replay cache<br/>oidc/mod.rs:406"]
    CORE["core verifies proof signature and claims<br/>oidc/mod.rs:499"]
    URL["whole URL lower-cased, trailing slash trimmed<br/>query and fragment retained<br/>oidc/mod.rs:657"]
    ATH["ath checked only when caller supplies expected hash<br/>oidc/mod.rs:621"]
    RESULT["DpopVerified includes key thumbprint and jti<br/>oidc/mod.rs:632"]
    CACHE["wrapper claims jti<br/>when cache supplied<br/>oidc/mod.rs:420"]
    EVICT["capacity insertion evicts LRU, even unexpired<br/>oidc/replay.rs:177"]
    CALL --> CORE --> URL --> ATH --> RESULT --> CACHE
    CACHE --> EVICT
```

## SP-05.8 Access-token checks and the separate WebID authority step

```mermaid
flowchart TD
    API["verify_access_token defaults to aud solid<br/>oidc/mod.rs:741<br/>oidc/mod.rs:785"]
    KEY["select compatible JWK and verify signature and issuer<br/>oidc/mod.rs:859<br/>oidc/mod.rs:908<br/>oidc/mod.rs:913"]
    CLAIMS["expiry, audience and cnf.jkt checks<br/>exp equality accepted, nbf not modelled<br/>oidc/mod.rs:917<br/>oidc/mod.rs:921<br/>oidc/mod.rs:928"]
    WEBID["webid then sub, literal http(s) prefix<br/>oidc/mod.rs:1013"]
    PROFILE["separate caller-invoked issuer-profile check<br/>oidc/mod.rs:984"]
    LIMIT["literal issuer property only<br/>no subject or context binding<br/>src/webid.rs:202"]
    DISC["discovery advertises ES256 and RS256<br/>verifiers accept a broader set<br/>oidc/mod.rs:184<br/>oidc/mod.rs:541<br/>oidc/mod.rs:859"]
    API --> KEY --> CLAIMS --> WEBID
    WEBID -. "caller must compose" .-> PROFILE --> LIMIT
    DISC -.-> KEY
```

## SP-05.10 Historical JWKS fetch boundaries and redirect gap

```mermaid
flowchart TD
    CACHE["JWKS cache lookup by canonical issuer<br/>oidc/jwks.rs:401"]
    DISC["miss: fetch discovery, independent of config cache<br/>oidc/jwks.rs:294"]
    POLICY["check issuer destination, build pinned client<br/>oidc/jwks.rs:225<br/>oidc/jwks.rs:239"]
    ISS["check returned<br/>issuer string<br/>oidc/jwks.rs:277"]
    JWK["check jwks_uri separately, build pinned client<br/>oidc/jwks.rs:306<br/>oidc/jwks.rs:322"]
    GAP["GAP at this revision: builder leaves redirects enabled<br/>redirect destination is not checked<br/>oidc/jwks.rs:210"]
    CACHE -->|miss| DISC --> POLICY --> ISS --> JWK
    POLICY -.-> GAP
    JWK -.-> GAP
```

## SP-05.12 did:nostr — document rendering, the parity model and the bidirectional binding

```mermaid
flowchart TD
    PK["NostrPubkey — 32-byte x-only identifier, no parity<br/>solid-pod-rs/src/did_nostr_types.rs:68"]
    URI["did_nostr_uri<br/>solid-pod-rs/src/did_nostr_types.rs:152"]
    WKP["well_known_path<br/>solid-pod-rs/src/did_nostr_types.rs:158"]
    RT["handle_well_known_did_nostr — the served endpoint<br/>solid-pod-rs-server/src/lib.rs:2394"]
    VT["verify_webid_tag<br/>solid-pod-rs/src/did_nostr_types.rs:612"]
    WD["webid_declares_pubkey<br/>solid-pod-rs/src/did_nostr_types.rs:674"]

    subgraph IDONLY["Identifier only — minimal or offline resolution, always 0x02"]
        RD["render_did_document<br/>solid-pod-rs/src/did_nostr_types.rs:230"]
        T1["render_did_document_tier1, back-compat alias<br/>solid-pod-rs/src/did_nostr_types.rs:296"]
        T3["render_did_document_tier3<br/>solid-pod-rs/src/did_nostr_types.rs:311"]
        TC["render_did_document_complete<br/>solid-pod-rs/src/did_nostr_types.rs:335"]
        MB["format_multibase_schnorr — fe70102 then X<br/>solid-pod-rs/src/did_nostr_types.rs:437"]
    end

    subgraph FULL["Controller holds the full key — parity follows the actual y"]
        FPK["NostrPubkey.from_public_key / to_even_public_key<br/>solid-pod-rs/src/did_nostr_types.rs:103, :130"]
        RDP["render_did_document_published<br/>solid-pod-rs/src/did_nostr_types.rs:261"]
        MBF["format_multibase_public_key / format_multibase_sec1<br/>fe70102 or fe70103 then X<br/>solid-pod-rs/src/did_nostr_types.rs:462, :491"]
    end

    subgraph DECODE["Verifiers accept both prefixes"]
        PS["parse_multibase_schnorr — to the x-only identifier<br/>solid-pod-rs/src/did_nostr_types.rs:555"]
        PSEC["parse_multibase_sec1 — to the full point, parity kept<br/>solid-pod-rs/src/did_nostr_types.rs:577"]
    end

    PK --> URI --> RD --> T1 --> T3 --> TC
    RD --> MB
    PK --> WKP --> RT
    URI --> VT --> WD
    PK --> FPK --> RDP --> MBF
    MB --> PS
    MBF --> PS
    MBF --> PSEC

    N["MULTIKEY_PREFIX fe70102 for an even-y key and MULTIKEY_PREFIX_ODD fe70103<br/>(solid-pod-rs/src/did_nostr_types.rs:396 and :404), fixed length 71<br/>(solid-pod-rs/src/did_nostr_types.rs:408)."]
    MBF -.-> N
    N2["EXTERNAL: the same did:nostr identifier is the estate's single spine —<br/>login, WAC principal, provenance author, DID subject and payment account.<br/>See ES-04, AB-11 and VC-23."]
    URI -.-> N2
```
- **What it shows:** since `0.5.0-alpha.10` (2026-10-01, `8b5d0a5` and `0302cde`) the module follows the did:nostr parity model of nostrcg/did-nostr#145 (`solid-pod-rs/src/did_nostr_types.rs:36-49`). The identifier carries no parity. A resolver holding only the identifier emits `0x02`, so `render_did_document` and `format_multibase_schnorr` never change. A controller holding its full key publishes through `render_did_document_published`, which emits `fe70103` when y is odd (`solid-pod-rs/src/did_nostr_types.rs:261-266`, `:462-463`). Decoders accept both prefixes in `split_multikey` (`solid-pod-rs/src/did_nostr_types.rs:512-515`).
- **Why it is this way:** before alpha.10 the encoder always produced `fe70102` and the docs called `0x02` invariant. A key derived by additive tweaking has unpredictable parity, and its controller must be able to publish it truthfully. Point reading matches `basePoint()` and `multikey()` in sidestr/spec PR #28. The module deliberately has no key-arithmetic (tweak) API until that PR settles (`solid-pod-rs/src/did_nostr_types.rs:56-57`).
- **Invariant:** the x-coordinate is the identifier whichever prefix a document carries. `parse_multibase_schnorr` drops the parity byte and returns X (`solid-pod-rs/src/did_nostr_types.rs:555-556`), so a BIP-340 signature verifies against the same key either way.

## SP-05.13 did:nostr resolution — WebID and DID-Doc must agree

```mermaid
sequenceDiagram
    autonumber
    participant A as caller
    participant R as NostrWebIdResolver<br/>solid-pod-rs-nostr/src/resolver.rs:71
    participant SS as SsrfCheck<br/>solid-pod-rs-nostr/src/resolver.rs:36
    participant W as WebID document
    participant D as did:nostr document

    A->>R: resolve_webid_to_nostr(webid)<br/>solid-pod-rs-nostr/src/resolver.rs:132
    R->>SS: DefaultSsrfCheck before any fetch<br/>solid-pod-rs-nostr/src/resolver.rs:54
    R->>W: fetch the profile
    W-->>R: HTML or JSON-LD
    R->>R: extract_json_ld_island<br/>solid-pod-rs-nostr/src/resolver.rs:400
    R->>R: scan_json_for_did_nostr (solid-pod-rs-nostr/src/resolver.rs:302) / scan_text_for_did_nostr, which now binds a 64-hex id only when the run ends there<br/>solid-pod-rs-nostr/src/resolver.rs:382
    R-->>A: NostrPubkey

    A->>R: resolve_nostr_to_webid(pubkey)<br/>solid-pod-rs-nostr/src/resolver.rs:181
    R->>D: fetch the .well-known did:nostr document
    D-->>R: alsoKnownAs / owl:sameAs claims
    R-->>A: the WebID, only when the binding is mutual
    Note over R: parse_did_nostr (solid-pod-rs-nostr/src/resolver.rs:359) is the strict<br/>syntax gate. The SSRF check runs BEFORE the fetch on both directions — a WebID<br/>is an attacker-supplied URL. See SP-06.9.
```

## SP-05.14 did:key — multicodec and self-signed JWT

```mermaid
flowchart LR
    ENC["did::encode / did::decode<br/>solid-pod-rs-didkey/src/did.rs:18"]
    PFX["DID_KEY_PREFIX<br/>solid-pod-rs-didkey/src/did.rs:15"]
    CODEC["multicodec: Ed25519 0xed, secp256k1 0xe7, P-256 0x1200<br/>solid-pod-rs-didkey/src/pubkey.rs:18"]
    ENUM["DidKeyPubkey<br/>solid-pod-rs-didkey/src/pubkey.rs:34"]
    ALG["jws_alg per curve<br/>solid-pod-rs-didkey/src/pubkey.rs:67"]
    FROM["from_multicodec_bytes<br/>solid-pod-rs-didkey/src/pubkey.rs:98"]
    JWT["verify_self_signed_jwt<br/>solid-pod-rs-didkey/src/jwt.rs:68"]
    RK["resolve_key from the header kid or jwk<br/>solid-pod-rs-didkey/src/jwt.rs:176"]
    VS["verify_signature<br/>solid-pod-rs-didkey/src/jwt.rs:263"]
    VER["DidKeyVerifier, DEFAULT_SKEW_SECONDS 60<br/>solid-pod-rs-didkey/src/verifier.rs:15"]

    PFX --> ENC --> CODEC --> ENUM --> ALG
    ENUM --> FROM
    JWT --> RK --> VS
    VER --> JWT

    N["htu_eq (solid-pod-rs-didkey/src/jwt.rs:170) binds the proof to the request URL,<br/>the same binding NIP-98 does with its 'u' tag."]
    JWT -.-> N
```

## SP-05.15 The self-signed / CID verifier registry

```mermaid
classDiagram
    class SelfSignedVerifier {
        <<trait>>
        solid-pod-rs/src/auth/self_signed.rs:104
    }
    class CidVerifier {
        +with(verifier)  solid-pod-rs/src/auth/self_signed.rs:137
        +registered()  solid-pod-rs/src/auth/self_signed.rs:153
    }
    class Nip98Verifier {
        solid-pod-rs/src/auth/nip98.rs:425
    }
    class LwsCidVerifier {
        +with_skew  solid-pod-rs/src/auth/lws_cid.rs:138
    }
    class ProofEnvelope {
        solid-pod-rs/src/auth/self_signed.rs:31
    }
    class VerifiedSubject {
        solid-pod-rs/src/auth/self_signed.rs:62
    }
    SelfSignedVerifier <|.. CidVerifier
    SelfSignedVerifier <|.. Nip98Verifier
    SelfSignedVerifier <|.. LwsCidVerifier
    SelfSignedVerifier ..> ProofEnvelope
    SelfSignedVerifier ..> VerifiedSubject
    note for LwsCidVerifier "validate_alg_jwk_match (solid-pod-rs/src/auth/lws_cid.rs:463) refuses an alg\nthat does not match the JWK's key type — the alg-confusion guard. Per-curve\nverifiers: ES256K (:503), ES256 (:523), EdDSA (:543); time bounds at :341."
```

## SP-05.16 The bundled IdP — authorisation-code flow with PKCE and DPoP

```mermaid
sequenceDiagram
    autonumber
    participant U as User agent
    participant P as Provider<br/>solid-pod-rs-idp/src/provider.rs:84
    participant CS as ClientStore::find<br/>solid-pod-rs-idp/src/registration.rs:211
    participant SS as SessionStore<br/>solid-pod-rs-idp/src/session.rs:95
    participant T as issue_access_token<br/>solid-pod-rs-idp/src/tokens.rs:88
    participant J as Jwks<br/>solid-pod-rs-idp/src/jwks.rs:192

    U->>P: authorize(AuthorizeRequest)<br/>solid-pod-rs-idp/src/provider.rs:152
    P->>CS: resolve the client document
    P->>SS: issue_code with the PKCE challenge<br/>solid-pod-rs-idp/src/session.rs:158
    P-->>U: AuthorizeResponse<br/>solid-pod-rs-idp/src/provider.rs:396
    U->>P: token(TokenRequest)<br/>solid-pod-rs-idp/src/provider.rs:226
    P->>SS: take_code — single use<br/>solid-pod-rs-idp/src/session.rs:185
    P->>P: pkce_s256 verifier check<br/>solid-pod-rs-idp/src/provider.rs:362
    P->>J: active_key, ES256<br/>solid-pod-rs-idp/src/jwks.rs:230
    P->>T: mint the DPoP-bound access token
    T-->>P: AccessToken with a cnf jkt claim<br/>solid-pod-rs-idp/src/tokens.rs:56
    P-->>U: TokenResponse<br/>solid-pod-rs-idp/src/provider.rs:448
    U->>P: userinfo<br/>solid-pod-rs-idp/src/provider.rs:318
    Note over T: ath_hash (solid-pod-rs-idp/src/tokens.rs:126) computes the value a DPoP<br/>proof must echo — the other half of the binding verified in SP-05.7.
```

## SP-05.17 IdP key management and discovery

```mermaid
flowchart TD
    GEN["SigningKey::generate_es256<br/>solid-pod-rs-idp/src/jwks.rs:89"]
    PEM["SigningKey::from_pem<br/>solid-pod-rs-idp/src/jwks.rs:137"]
    ROT["Jwks::rotate<br/>solid-pod-rs-idp/src/jwks.rs:236"]
    PRUNE["prune_expired against the retention window<br/>solid-pod-rs-idp/src/jwks.rs:253"]
    PUB["public_document — the JWKS endpoint body<br/>solid-pod-rs-idp/src/jwks.rs:264"]
    DISC["build_discovery<br/>solid-pod-rs-idp/src/discovery.rs:72"]
    ROUTER["axum router — discovery, jwks, registration,<br/>credentials, password change, account delete<br/>solid-pod-rs-idp/src/axum_binder.rs:92"]

    GEN --> ROT --> PRUNE --> PUB --> ROUTER
    PEM --> ROT
    DISC --> ROUTER

    N["with_retention (solid-pod-rs-idp/src/jwks.rs:224) keeps a rotated key<br/>publishable long enough for tokens signed under it to expire — pruning it<br/>immediately would invalidate live tokens."]
    PRUNE -.-> N
    N2["The axum binder is behind the default-off axum-binder feature (SP-01.9), so a<br/>consumer on a different framework wires Provider into its own router instead."]
    ROUTER -.-> N2
```

## SP-05.18 IdP credential surfaces

```mermaid
flowchart LR
    PWD["login — Argon2-style credential check<br/>solid-pod-rs-idp/src/credentials.rs:104"]
    MINLEN["MIN_PASSWORD_LENGTH 8<br/>solid-pod-rs-idp/src/credentials.rs:31"]
    RL["RATE_LIMIT_ROUTE idp_credentials<br/>solid-pod-rs-idp/src/credentials.rs:36"]
    PK["PasskeyBackend trait plus WebauthnPasskey<br/>solid-pod-rs-idp/src/passkey.rs:123"]
    TODO["PasskeyTodo — the null backend<br/>solid-pod-rs-idp/src/passkey.rs:157"]
    SCH["SchnorrSso trait plus Nip07SchnorrSso<br/>solid-pod-rs-idp/src/schnorr.rs:123"]
    STODO["SchnorrTodo — the null backend<br/>solid-pod-rs-idp/src/schnorr.rs:146"]
    DIG["canonical_digest(token, user_id, pubkey)<br/>solid-pod-rs-idp/src/schnorr.rs:205"]
    US["UserStore trait plus InMemoryUserStore<br/>solid-pod-rs-idp/src/user_store.rs:70"]
    INV["Invite mint_token<br/>solid-pod-rs-idp/src/invites.rs:98"]
    DEL["delete_account behind CONFIRMATION_PHRASE<br/>solid-pod-rs-idp/src/account_delete.rs:55"]

    PWD --> MINLEN
    PWD --> RL
    PK --> TODO
    SCH --> STODO
    SCH --> DIG
    US --> PWD
    US --> INV
    US --> DEL

    N["DIVERGENCE: PasskeyTodo and SchnorrTodo are null backends behind the same<br/>traits. Without the passkey and schnorr-sso features (SP-01.9) those login<br/>routes exist but resolve to the stub implementors."]
    TODO -.-> N
    N2["delete_account requires the literal phrase DELETE MY ACCOUNT<br/>solid-pod-rs-idp/src/account_delete.rs:14"]
    DEL -.-> N2
```

## SP-05.19 Pod key provisioning — deliberately not an HTTP route

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator or library caller
    participant PV as provision_pod_keys<br/>solid-pod-rs-idp/src/key_provisioning.rs:322
    participant B32 as bech32_encode<br/>solid-pod-rs-idp/src/key_provisioning.rs:161
    participant ACL as build_owner_only_acl<br/>solid-pod-rs-idp/src/key_provisioning.rs:186
    participant S as pod Storage

    OP->>PV: provision keys for a pod
    PV->>PV: Schnorr keypair from the plan seed, else Generate::generate<br/>solid-pod-rs-idp/src/key_provisioning.rs:328-330
    PV->>B32: encode the secret key as an nsec
    PV->>S: write the private key document<br/>solid-pod-rs-idp/src/key_provisioning.rs:44
    PV->>ACL: build an owner-only ACL for it
    PV->>S: write the sidecar<br/>solid-pod-rs-idp/src/key_provisioning.rs:47
    PV->>PV: patch_webid_with_nostr_pubkey<br/>solid-pod-rs-idp/src/key_provisioning.rs:253
    PV->>S: update the profile card<br/>solid-pod-rs-idp/src/key_provisioning.rs:51
    PV-->>OP: KeyProvisioningOutcome<br/>solid-pod-rs-idp/src/key_provisioning.rs:67
    Note over PV: INVARIANT: no HTTP route mints keys. Returning a freshly generated nsec over<br/>HTTP is an owner-signoff decision, so this stays a library surface behind the<br/>default-off provision-keys feature — see SP-01.8.
```

## SP-05.21 Password change — the one credential-MUTATION path

```mermaid
sequenceDiagram
    autonumber
    participant C as authenticated user
    participant P as change_password<br/>solid-pod-rs-idp/src/password_change.rs:75
    participant L as RateLimiter<br/>solid-pod-rs/src/security/rate_limit.rs:143
    participant U as UserStore<br/>solid-pod-rs-idp/src/user_store.rs:70
    participant A as Argon2

    C->>P: PasswordChangeRequest — current plus new<br/>solid-pod-rs-idp/src/password_change.rs:56
    P->>L: check RateLimitKey keyed by route and source IP<br/>solid-pod-rs-idp/src/password_change.rs:83
    alt Deny
        L-->>P: retry_after_secs
        P-->>C: PasswordChangeError::RateLimited<br/>solid-pod-rs-idp/src/password_change.rs:91
    end
    P->>P: both fields must be non-empty<br/>solid-pod-rs-idp/src/password_change.rs:95
    P->>P: validate_password_length against MIN_PASSWORD_LENGTH<br/>solid-pod-rs-idp/src/password_change.rs:101
    P->>U: find_by_id<br/>solid-pod-rs-idp/src/password_change.rs:108
    U-->>P: the user, or InvalidRequest 'user not found'<br/>solid-pod-rs-idp/src/password_change.rs:112
    P->>U: verify_password against the CURRENT password<br/>solid-pod-rs-idp/src/password_change.rs:116
    P->>A: hash the NEW password with a fresh SaltString<br/>solid-pod-rs-idp/src/password_change.rs:126
    P->>U: update_password<br/>solid-pod-rs-idp/src/password_change.rs:132
    P-->>C: PasswordChangeResponse<br/>solid-pod-rs-idp/src/password_change.rs:63

    Note over P: INVARIANT: the rate-limit gate runs FIRST, before any validation, lookup or<br/>hash. Argon2 verification is deliberately expensive, so checking credentials<br/>before rate-limiting would turn this endpoint into a CPU-amplification oracle.<br/>Its own route key keeps that budget separate from login<br/>(solid-pod-rs-idp/src/password_change.rs:24 vs credentials.rs:36).
    Note over U: The current password is re-verified even though the caller is already<br/>authenticated — a stolen session must not be enough to change the password.
    Note over P: The server exposes POST /account/password/change (SP-02.9) — this is the IdP<br/>library function behind it.
```

## SP-05.22 Cross-identity — a did:nostr caller presenting as its WebID too

```mermaid
sequenceDiagram
    autonumber
    participant G as enforce_write_ctx / enforce_read_inner<br/>solid-pod-rs-server/src/lib.rs:1021
    participant LI as linked_identities<br/>solid-pod-rs-server/src/cross_identity.rs:114
    participant R as PodCrossIdentity::linked_web_id<br/>solid-pod-rs-server/src/cross_identity.rs:92
    participant L as local profile card<br/>solid-pod-rs-server/src/cross_identity.rs:77
    participant D as remote DidNostrResolver (feature did-nostr)<br/>solid-pod-rs-server/src/cross_identity.rs:100
    participant GI as grant_for_identities<br/>solid-pod-rs-server/src/lib.rs:1102

    G->>LI: agent_uri from the verified NIP-98 pubkey
    LI->>R: pubkey as 64 lowercase hex, else no link
    R->>L: read /profile/card, compare the owner's nostr:pubkey
    L-->>R: WebID card#me when the owner key matches
    opt no local link, operator configured origins
        R->>D: resolve each origin, alsoKnownAs plus back-link
        D-->>R: WebID or nothing
    end
    R-->>LI: Some(webid) only for a bidirectionally verified link
    LI-->>G: linked WebIDs, never the agent itself, http or https only
    G->>GI: primary first, then each linked identity
    Note over GI: The ACL is evaluated once per identity, so acl:agent naming either the<br/>did:nostr or the WebID matches (solid-pod-rs-server/src/lib.rs:1102).
```

**What it shows:** after NIP-98 succeeds, the server asks a resolver whether the pubkey is bound to a WebID and, if so, the caller presents as both.

**Why it is this way:** JSS replaces the did:nostr agent with the WebID; keeping both is a deliberate superset (`solid-pod-rs-server/src/cross_identity.rs:8`), and a one-way link is never accepted (`solid-pod-rs-server/src/cross_identity.rs:6`).

**Invariant:** `linked_web_id` returns a WebID only when the link is bidirectional, either through the pod's own profile card or through a remote origin with a back-link check (`solid-pod-rs-server/src/cross_identity.rs:92`).
