---
title: Solid Pod identity and storage panels
date: 2026-11-02
status: assessed
type: explanation
---

# Solid Pod identity and storage panels

This review covers the 19 panels in SP-05 (identity) and SP-06 (storage, quota, provisioning and git pods) that had no assessment yet. All 19 were traced at Solid Pod `9b6143c`. Results: 3 accurate, 16 corrected, none accurate-with-limits, none inaccurate-unfixed. Two panels credited on 2026-11-01, SP-05.2 and SP-05.10, were also re-traced and corrected. Eight more previously credited panels (SP-05.1, 05.3, 05.4, 05.5, 05.7, 05.8, SP-06.7, SP-06.12) were re-checked at `9b6143c` because the restamp made their claims stale: six are accurate, and SP-05.3 and SP-06.12 are corrected. In total 29 panels are claimed: 9 accurate and 20 corrected. Both topics are restamped from `09ec90f` to `9b6143ce55b242c322c92d8c6bcc0c1a5371b451`, and both pass the strict generator gate.

The most important finding is a defect, not a drawing error. `LwsCidVerifier`'s `iat` window check joins two conditions that can never both be true, so it never rejects a stale `iat`. An LWS-CID JWT without `exp` therefore has no time bound. It is latent today, because nothing in the server wires the CID verifier registry in.

Three corrected panels described server behaviour that does not exist:

- **SP-05.21:** the server's password-change route is a 501 stub, not the IdP function.
- **SP-06.8:** server pod creation never calls the core `provision_pod`, and writes no ACL.
- **SP-06.10:** the server does not use `CorsPolicy`.

Restamping also showed that two panels credited on 2026-11-01 are now out of date, because source changed after they were assessed:

- **SP-05.2:** the empty-body payload bypass was fixed by `8f2df86`.
- **SP-05.10:** the JWKS redirect gap was closed by `fcc624a`.

I corrected both. Their earlier claims no longer match the topic text.

## Findings

**SI-01: the LWS-CID `iat` bound never fires (Debt, latent).**
- **Trigger:** `LwsCidVerifier::verify` calls `validate_time_bounds(&claims, now, self.skew)` (`crates/solid-pod-rs/src/auth/lws_cid.rs:201`), which tests `now.saturating_sub(iat) > skew && iat.saturating_sub(now) > skew` (`crates/solid-pod-rs/src/auth/lws_cid.rs:343`). For any `iat`, one side saturates to 0, so the conjunction is false and no `iat` is ever rejected.
- **Consequence:** a self-signed CID JWT with no `exp` is accepted at any age, provided its signature, `htm` and `htu` match. With `exp`, only expiry and the one-hour lifetime cap apply (`:350-364`). The did:key verifier gets the same test right with `||` (`crates/solid-pod-rs-didkey/src/jwt.rs:144`).
- **Exposure:** no in-tree consumer builds a `CidVerifier`. It is only re-exported (`crates/solid-pod-rs/src/lib.rs:216`), and `lws-cid` is not a server feature.
- **To close:** use `||`, and add a unit test with a stale `iat` and no `exp`. Recorded in SP-05.15.

**SI-02: the server's password change is a stub (SP-05.21 corrected).**
- **Wrong claim:** the panel said `POST /account/password/change` (SP-02.9) runs `change_password`.
- **At `9b6143c`:** that route goes to `handle_password_change` (`crates/solid-pod-rs-server/src/lib.rs:5315`). The handler declares its own `{token, new_password}` body and returns `501` (`:3139-3154`). The IdP function is reachable only through the default-off axum binder's `PUT /idp/credentials` (`crates/solid-pod-rs-idp/src/axum_binder.rs:99`).
- **To close:** wire the server route to the IdP function behind authentication, or keep the 501 and keep the panel as corrected.

**SI-03: server provisioning bypasses the core provisioner (SP-06.8 rewritten).**
- **Public routes:** `/.pods` and `/api/accounts/new` share `provisioning_gate` (`crates/solid-pod-rs-server/src/lib.rs:2730`). `provision_named_pod` then writes container `.meta` markers and a profile card itself (`:2813-2844`). It never calls `provision_pod` (`crates/solid-pod-rs/src/provision.rs:207`), so it writes no type indexes and no ACLs, and runs `GitAutoInit` directly under the `git` feature (`:2846-2851`).
- **Admin key:** compared with `subtle::ConstantTimeEq` (`:2750`). The library's exact-match `check_admin_override` is not used.
- **Admin route:** `POST /_admin/provision/{pubkey}` has its own handler (`:4450`): admin key only, no limiter, an owner ACL naming `did:nostr`, and writes through `tokio::fs`, outside the `FsBackend` capability.
- **Open:** pods from the public routes get no ACL of their own. I did not trace what ancestor ACL then governs them.

**SI-04: two SSRF tiers with different configuration (SP-06.9 corrected).**
- **`SsrfPolicy::resolve_and_check`** (`crates/solid-pod-rs/src/security/ssrf.rs:206`): reads the env toggles, checks the first resolved address and records metrics. It serves OIDC and the server's remote did:nostr cross-identity (`crates/solid-pod-rs-server/src/main.rs:417`).
- **The free `resolve_and_check(host)`** (`:503`): ignores the env toggles, checks every address and short-circuits metadata hostnames. It serves `/proxy` (`crates/solid-pod-rs-server/src/lib.rs:3261`) and the `solid-pod-rs-nostr` default check.
- **Consequence:** an operator allowlist changes one tier and not the other, and `/proxy` blocks never reach the metrics.
- **To close:** state the two-tier contract in the docs, or route `/proxy` through the policy.

**SI-05: the library's CORS policy reflects any Origin with credentials (SP-06.10 corrected).**
- **Library:** `CorsPolicy` with wildcard origins and credentials reflects any Origin with `Allow-Credentials: true` (`crates/solid-pod-rs/src/security/cors.rs:242-247`). Without credentials it still echoes the Origin, and emits `*` only when no Origin header is sent (`:249`).
- **Server:** it never uses `CorsPolicy`. Its `add_cors_headers` emits `*` without credentials when no allowlist is set (`crates/solid-pod-rs-server/src/lib.rs:3607`).
- **Drift:** the doc comments at `:242` and `:3517` still say the server echoes the Origin.

**SI-06: IdP surface drift (SP-05.16, SP-05.17, SP-05.18).**
- **Discovery over-advertises:** it lists `/idp/auth`, `/idp/token`, `/idp/me`, introspection, revocation, end-session, `refresh_token` and `client_credentials` (`crates/solid-pod-rs-idp/src/discovery.rs:82-103`). `Provider::token` accepts only `authorization_code` (`crates/solid-pod-rs-idp/src/provider.rs:227`), and the axum router has five routes (`crates/solid-pod-rs-idp/src/axum_binder.rs:92-102`).
- **Binder login is not DPoP-bound:** the binder calls `login` with `dpop_jkt = None` (`:158`), so it issues `Bearer` tokens.
- **Login skips the length check:** `login` does not check `MIN_PASSWORD_LENGTH`. Only `change_password` does (`crates/solid-pod-rs-idp/src/password_change.rs:101`).
- **Wrong stub note:** the old SP-05.18 note said the passkey and Schnorr login routes resolve to stubs. Those modules do not compile without their features (`crates/solid-pod-rs-idp/src/lib.rs:78-82`), and no router mounts them.
- **Missing step in SP-05.16:** the DPoP proof is verified with the replay cache before the code is taken (`crates/solid-pod-rs-idp/src/provider.rs:241-250`).

**SI-07: credited panels changed by fixes after their assessment.**
- **SP-05.2:** `8f2df86` binds supplied empty bodies (`crates/solid-pod-rs/src/auth/nip98.rs:191`). In source, this closes SP-ID-01 from `2026-11-01-solid-identity-boundaries.md`. No runtime re-check was run.
- **SP-05.10:** `fcc624a` makes the pinned JWKS client refuse redirects (`crates/solid-pod-rs/src/oidc/jwks.rs:214`).
- **SP-06.12:** `1a66c2d` refuses unadvertised object requests (`crates/solid-pod-rs-git/src/service.rs:425-432`). I added this to the panel as a Drift note.
- **SP-05.3 and SP-06.12:** citations re-resolved by exact matching on line content. Meaning unchanged.
- **Ledger effect:** the stored hashes of SP-05.2, SP-05.3, SP-05.10 and SP-06.12 no longer match, so their earlier claims will read as stale. I propose new claims for SP-05.2 and SP-05.10, which I traced in full. SP-05.3 and SP-06.12 need their original reviewers, or the lead, to re-credit them.

**SI-08: the storage atomicity panel was stale (SP-06.4 corrected).**
- **Wrong claim:** the panel cited an open README divergence for body and sidecar writes.
- **At `9b6143c`:** the sidecar carries the future body ETag (`crates/solid-pod-rs/src/storage/fs.rs:254-257`), and `read_meta` discards a sidecar that does not match its body (`:185-193`). A backend-wide `RwLock` serialises readers and writers within one process (`:210`, `:238`). The README now says the audit's findings are remediated.
- **Remaining limit:** the lock is process-local.

**SI-09: the git lifecycle is not one chain (SP-06.11 corrected, Open).**
- **What runs:** in the server, `git_mark_write` needs only `.git` (`crates/solid-pod-rs-server/src/lib.rs:3957`). `apply_write_config` runs only on a `GitHttpService` write (`crates/solid-pod-rs-git/src/service.rs:358-359`), which the server refuses. `write_agent_identity` has no caller.
- **Open, from source only:** pods from the admin route live at `data_root/pods/<pubkey>`. Since `git_mark_write` takes the first path segment as the pod (`crates/solid-pod-rs-server/src/lib.rs:3945-3955`), writes to those pods may never be marked. This needs one request through the real route to confirm or refute.

**SI-10: smaller corrections.**
- **SP-05.12:** the served did:nostr endpoint exists only under the `did-nostr` feature, answers only for the owner's key, and always emits `fe70102` (`crates/solid-pod-rs-server/src/lib.rs:2394-2440`).
- **SP-05.13:** the `solid-pod-rs-nostr` resolver takes `(origin, pubkey)`. It reads only `alsoKnownAs` from the DID document, checks `id`, and uses redirect-none clients. The server does not use it.
- **SP-05.14:** did:key proofs carry no `jti` and are not replay-guarded within the skew window.
- **SP-06.2:** the storage layer does not percent-decode, so the old claim that "a percent-encoded `..` parses to ParentDir" was false. Decoding happens once, in `canonical_pod_path` (`crates/solid-pod-rs-server/src/lib.rs:850`).
- **SP-06.3:** `watch` uses an ambient path, outside the capability.
- **SP-06.6:** quota reserves the positive size delta against `head()`, which is read outside the per-pod lock.
- **Doc drift:** the IdP `Cargo.toml` still says the provisioner bodies are `todo!()` (`crates/solid-pod-rs-idp/Cargo.toml:84`). `ecosystem-integration.md:105` still describes an S3 scaffold that no longer exists.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| SP-05.1 | accurate | Re-checked at `9b6143c`. Bearer branch `lib.rs:611`/`617`, dev bearer needs a 32-byte secret, the `aud` and at most 24 h (`lib.rs:684-730`); `verify_at` `:636`, replay returns `None` `:641-651`; `request_url` takes the configured origin, else `effective_scheme` with proxy trust (`scheme.rs:217-241`). |
| SP-05.2 | corrected | Re-traced after `8f2df86`. The payload step and note now say `Some(body)`, empty included, requires a matching tag, and only `None` skips (`nip98.rs:190-191`). Nine citations re-derived (+4 shift), and server PUT `lib.rs:1735` → `1711`. Drift added. |
| SP-05.3 | corrected | Full re-trace. `MatchPolicy` `nip98.rs:89-101` and the Git extractor `solid-pod-rs-git/src/auth.rs:144-154` (body `None`, prefix match, no replay claim) hold. Push-refusal citations moved `lib.rs:4750/4775` → `4763/4788`, and the N4 note now says push is refused before authentication. |
| SP-05.4 | accurate | Re-checked. `check_and_record` `replay.rs:243-279`: one mutex, no refresh on replay, expired id overwritten in place, reclaim only at capacity, `CapacityExhausted` instead of evicting a live entry. Environment filters `> 0`; `with_config` takes any TTL, including zero (`:133-160`). |
| SP-05.5 | accurate | Re-checked. Trait `replay_store.rs:90`, error variants `:56`/`:75`, retention contract `:108`, restart and replica limits `:115`, seam not integrated `:34`; defaults 120 s / 10,000 and env names `replay.rs:78-86`. |
| SP-05.7 | accurate | Re-checked. `verify_dpop_proof_with_ath` `oidc/mod.rs:406-420` claims `jti` only when a cache is passed; `normalise_htu` lower-cases the whole URL and trims the slash, keeping the query (`:657-659`); `ath` only when expected (`:621`); `oidc/replay.rs` is an `LruCache` that evicts at capacity (`:44`, `:177`). |
| SP-05.8 | accurate | Re-checked. `exp < now` so equality is accepted, and no `nbf` (`oidc/mod.rs:917`); audience `:921`, `cnf.jkt` `:928`; `extract_webid` takes `webid` then `sub` by literal `http(s)://` prefix (`:1013-1025`); issuer binding is a separate call (`:984`); discovery advertises ES256/RS256 (`:184`). |
| SP-05.10 | corrected | Re-traced after `fcc624a`. GAP node now reads closed (`jwks.rs:214`, header `:27`). Citations `401/294/225/239/277/306/322` → `405/298/227/243/281/310/326`. Drift added. |
| SP-05.12 | corrected | Endpoint citation `lib.rs:2394` → `2395`. Added a Tension: feature-gated, owner-only 404, always `fe70102` (`lib.rs:2394-2440`, `interop.rs:341-351`; `did-nostr` enables `did-nostr-types`, `Cargo.toml:216`). Parity model, prefixes and decoders confirmed (`did_nostr_types.rs:36-57, 396-408, 437-439, 462-463, 512-515, 555-556`). |
| SP-05.13 | corrected | Rewritten. `resolve_nostr_to_webid(origin, pk)` (`resolver.rs:181`). DID document gives `id` plus `alsoKnownAs` only, with the id check at `:220` and the back-link at `:242`. Redirect-none clients (`:84-92`, `:107-124`). No server consumer: the server uses `DidNostrResolver` (`main.rs:417`). |
| SP-05.14 | accurate | Every citation holds (`did.rs:15,18`; `pubkey.rs:18,34,67,98`; `jwt.rs:68,170,176,263`; `verifier.rs:15`). Added a Tension: alg gate at `jwt.rs:110-118`, no `jti`, no consumer. |
| SP-05.15 | corrected | `nip98.rs:425` → `427`. The "time bounds at :341" claim now records the dead `iat` check (`lws_cid.rs:343`), with a Debt and an Open for no consumer. |
| SP-05.16 | corrected | Added the session gate (`provider.rs:183-217`) and the DPoP proof verified with the replay cache before `take_code` (`provider.rs:241-256`). `take_code` removes, then checks TTL (`session.rs:185-192`). Added an Open: no in-tree router serves `authorize`, `token` or `userinfo`. |
| SP-05.17 | accurate | `jwks.rs:89,137,224,236,253,264`, `discovery.rs:72` and `axum_binder.rs:92` hold. Added a Drift (discovery over-advertises) and an Open (no rotation caller). |
| SP-05.18 | corrected | Minimum-length edge moved to `change_password` (`password_change.rs:101`). Stub note corrected (`lib.rs:78-82`, router `axum_binder.rs:92-102`). Added a Tension: binder login issues `Bearer` (`axum_binder.rs:158`, `credentials.rs:170-174`). |
| SP-05.19 | accurate | `key_provisioning.rs:44,47,51,67,161,186,253,322,328-330` hold. No server call. Added a Drift for the `todo!()` comment (`Cargo.toml:84`). |
| SP-05.21 | corrected | Server-route note replaced (`lib.rs:3139-3154, 5315`; `axum_binder.rs:99`). Rate-limit-first order confirmed (`password_change.rs:82-92`). |
| SP-05.22 | corrected | Citations `lib.rs:1021` → `997` and `1102` → `1078`. Mechanism confirmed (`cross_identity.rs:92-130`). |
| SP-06.1 | corrected | Note reworded and an open Drift added: `ecosystem-integration.md:105` still describes S3. Trait and backends confirmed (`mod.rs:27,58,73,108,118,127`; `fs.rs:47,53,350`; `memory.rs:45`). |
| SP-06.2 | corrected | Percent-encoding claim replaced. Guard citations `3430/3481/3757` → `3443/3494/3770`. |
| SP-06.3 | corrected | Weak `fs.rs:14` import citation replaced by `fs.rs:215`. Test citation `511` → `510` (the function). Added a Tension for `watch` (`fs.rs:368`). |
| SP-06.4 | corrected | Window and divergence rewritten (`fs.rs:185-193, 210, 238, 254-257`). |
| SP-06.6 | corrected | Delta semantics (`lib.rs:2653-2681`). Citations `+1`. Added an Open for `head()` outside the lock. |
| SP-06.7 | accurate | Re-checked. `SubdomainResolver::resolve` `multitenant.rs:74-151`: port stripped, file-like label and non-matching host fall back with `pod: None`, the label is lower-cased and `..`-scrubbed before the single-label check (`:124-138`), so `al..ice` becomes `alice`. |
| SP-06.8 | corrected | Rewritten from `lib.rs:2730-2851, 4450-4533`. |
| SP-06.9 | corrected | Two tiers drawn (`ssrf.rs:206, 503, 529-535, 557`). Proxy citations `+13`. |
| SP-06.10 | corrected | `*` behaviour (`cors.rs:249`). Server middleware (`lib.rs:3518, 3583-3608`). Tension and Drift added. |
| SP-06.11 | corrected | Transitions split (`lib.rs:3945-3957`; `service.rs:358-359`; `init.rs:100-140`). Added an Open for admin-pod marking. |
| SP-06.12 | corrected | Full re-trace. `handle_git` `lib.rs:4716` onwards: 501 without `data_root`, push refused before auth (`:4767`), extractor `:4788`, Read on the pod-root ACL (`:4792`, `:4797`), service built on `repo_root` (`:4816`); service steps `service.rs:284-390` and `guard.rs:18`/`:68` hold. Added the missing 404 `pod not found` step (`lib.rs:4741`). Drift note on `1a66c2d` added earlier in this audit. |

Panels assessed by earlier reports were re-checked only as restamping required. **SP-05.2** and **SP-05.10** were traced in full and corrected (SI-07); they are claimed here. **SP-05.3** and **SP-06.12** had citations re-resolved, plus a Drift note on SP-06.12. **SP-05.1, SP-05.4, SP-05.5, SP-05.7, SP-05.8 and SP-06.7** cite only bytes unchanged between `09ec90f` and `9b6143c`, so their text and earlier credit stand untouched. That is a byte comparison, not a re-review.

## Verification

- **Evidence class:** source tracing at the pinned revision only. Every cited line was read with `git -C /home/devuser/workspace/solid-pod-rs show 9b6143ce55b2:<path>` or `git grep … 9b6143ce55b2`. No product tests, builds, probes or deployments were run. Nothing here is tested or runtime-verified.
- **Pinned revisions:** solid-pod-rs `9b6143ce55b2`. The topics previously declared `09ec90f`. Between the two, the only cited files that changed are:
  - `crates/solid-pod-rs-server/src/lib.rs`
  - `crates/solid-pod-rs/src/auth/nip98.rs`
  - `crates/solid-pod-rs/src/oidc/jwks.rs`
  - `crates/solid-pod-rs-git/src/service.rs`
  - a comment line in `cross_identity.rs`
  - `ldp.rs`, which neither topic cites
- **Citation re-resolution:** each changed citation was re-derived from its symbol, by matching the old line's exact content in the new revision and checking it by hand where several lines matched. No offsets were applied blindly.
- **Generator, run from the worktree root:**
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 05-identity-nip98-did-oidc.md --check --cite-check --strict-citations --render`: exit 0. 217 citations verified at the declared revision, 0 unresolvable, 18/18 rendered.
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 06-storage-quota-and-git-pods.md --check --cite-check --strict-citations --render`: exit 0. 126 citations verified, 0 unresolvable, 11/11 rendered. The final run had no JEV advisory; an earlier advisory on `fs.rs:511` was fixed by citing the test function at `:510`.
- **Prose citations:** the generator counts above cover the Mermaid citations. Prose citations, including the ones to files outside a topic's `sources:` (`solid-pod-rs-server/src/main.rs:417`, `docs/explanation/ecosystem-integration.md:105`, `solid-pod-rs-idp/Cargo.toml:84`, `solid-pod-rs-idp/src/lib.rs:78-82`), were checked by hand at `9b6143c`.
- **Not verified:**
  - Crash behaviour of the two-rename write (SP-06.4).
  - Concurrent quota drift (SP-06.6).
  - What ACL governs a pod created through `/.pods` (SP-06.8).
  - Whether admin-provisioned pods are git-marked (SP-06.11).
  - The exploitability of SI-01 through any consumer.
  - Whether the server binary is shipped with `did-nostr`, `git` or `axum-binder` enabled.
  - Visual quality of the renders beyond the generator's pass.
