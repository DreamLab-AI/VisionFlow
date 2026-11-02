---
title: Solid forge, extension and archived ADRs
date: 2026-11-02
status: assessed
type: explanation
---

# Solid forge, extension and archived ADRs

I assessed eight Solid Pod ADRs section by section against source at solid-pod-rs `9b6143ce55b2`: the three operative forge and extension records (ADR-2009, ADR-2010 and ADR-2011) and the five archived records (ADR-057 to ADR-061). That is 93 sections. I also classified four support files. No topic was assigned, so there are no panel dispositions. The core of each operative decision holds in source. The forge owns every method under `/forge` ahead of the pod routes. The mirror's guards exist: argv-only git, the clone URL after `--`, http/https-only remote clones, address pinning, the 512 MiB cap with a process-group kill, and signature checks after the cheap filters. `open_edit` and `allow_mainnet` stay off by default. The most important finding contradicts ADR-2010. **The server builds the mirror with the default SSRF policy, not `OperatorRelays`**, so it refuses a loopback relay. Agentbox's shipped default mirror relay is `ws://127.0.0.1:7777`. By source, a consumer on the defaults would refuse its own relay even with follow entries configured. Second, ADR-2009 and ADR-2010 still describe hidden `refs/mirror` as the withholding mechanism. At the pin, store isolation has replaced it: the served store no longer holds candidate refs, and a `hideRefs` setting marks a store as legacy. Both records' 2026-11-01 qualifications are now stale. The archived records are historical. Each section below is marked still true, superseded (naming the ADR-20xx or BASELINE item), or contradicted.

## Findings

### F1. The mirror is not given the operator relay policy that ADR-2010 promises

- **Trigger:** ADR-2010 Decision says a loopback or private-network relay "is admitted only when the operator lists it (`OperatorRelays`)", for both the mirror and announce.
- **Source:**
  - The server wraps only the announce relays in `OperatorRelays` (`crates/solid-pod-rs-server/src/lib.rs:5022-5024`).
  - The server builds the mirror with `Mirror::new` (`lib.rs:5038`), which hard-wires `DefaultSsrfCheck` (`crates/solid-pod-rs-forge/src/mirror.rs:2303-2311`).
  - `relay_session` checks the relay host through that policy before connecting (`mirror.rs:3198-3200`).
  - The policy refuses loopback and private classes (`crates/solid-pod-rs/src/security/ssrf.rs:540-555`).
  - Agentbox at `c57d9c0508f4` defaults `FORGE_MIRROR_RELAYS` to `ws://127.0.0.1:7777` (`flake.nix:1619`, `agentbox.toml:615`).
  - The same wiring is present at `76ab7aa` (`lib.rs:5011`, `:5025`), the revision agentbox pins (`lib/solid-pod-rs.nix:54,58`).
- **Consequence:** with the shipped defaults and a follow entry added, the mirror would log "relay host refused" and back off for ever. ADR-2010's Close-out explains the missing relay session only by the empty follow list. By source there is a second, independent cause.
- **Limit:** source tracing only. I did not run the server.
- **What would close it:**
  - Pass `OperatorRelays::new(&mirror_relays)` through `Mirror::with_parts` (`OperatorRelays` admits listed hosts and defers everything else to `DefaultSsrfCheck`, `crates/solid-pod-rs-forge/src/publish.rs:125-140`), so clone hosts stay on the default policy.
  - Add a server-level test with a listed loopback relay, or correct the ADR text.

### F2. Hidden refs are no longer the withholding mechanism, and both ADRs lag the code

- **Trigger:**
  - ADR-2009 Consequences ("Fixed after review") and ADR-2010 Decision ("hidden `refs/mirror`") name `uploadpack.hideRefs`/`transfer.hideRefs` as the boundary.
  - Both "Review qualification, 2026-11-01" sections, dated `b644e55`, say that complete object withholding is still open.
- **Source:**
  - A fetch lands in a private candidate store under `mirror-work/`, outside `repos/` (`mirror.rs:1179-1183`, `:1817-1838`).
  - Publication pushes approved SHAs only: one `--force-with-lease` per ref, no `+`, and `--atomic` (`mirror.rs:1939-1953`).
  - A store that still has `refs/mirror` or a `hideRefs` naming it is detected as the pre-isolation layout (`mirror.rs:1419-1440`).
  - The Git HTTP service forces all three unadvertised-object options false (`crates/solid-pod-rs-git/src/service.rs:427-429`).
  - Readers are gated by the publication marker (`crates/solid-pod-rs-forge/src/lib.rs:119`, `:752-763`).
- **Consequence:** the ADRs' mechanism text is superseded in practice, and their open-withholding qualification is stale at the pin.
- **What would close it:** an owner amendment to both ADRs naming store isolation, the publication marker and boot migration, citing `9b6143c`.

### F3. The SP-GIT-02 report is confirmed at the pin, with one status nuance

I re-checked `2026-11-02-solid-mirror-publication.md` at the pin. These parts hold:

- the candidate store and the approved-only push (F2);
- the publication marker gate (`lib.rs:119`, `:752`);
- forge writes into a mirror refused with 409 (`refuse_mirror_write`, `lib.rs:728-744`; three callers).

The nuance: an owner's smart-HTTP push into a mirror gets 403 (`ForgeError::Forbidden`, `lib.rs:826-839`; status mapped at `error.rs:68`), not 409.

I did not re-trace the B1/B2 migration internals (journal, fork discovery) beyond confirming that they exist (`migrate.rs:1374`, `migrate_fork.rs:94`).

The consumer gap is real: agentbox pins `76ab7aa`, which predates commits A, B1 and B2.

### F4. ADR-2010's "operator's own key" default is not what ships

- **Trigger:** ADR-2010 Decision says "the shipped default follow list is the operator's own key and nothing external".
- **Source:**
  - The server has no default follow list (`lib.rs:4973-4995`).
  - Agentbox ships `mirror_follow = []` (`agentbox.toml:621`), and `self/<repo>` expands only when it is named (`flake.nix:1603-1618`).
- **Consequence:** the shipped default follows nothing. That is more inert than the ADR says, and consistent with its Close-out. The Decision text is wrong.
- **What would close it:** an ADR text correction.

### F5. ADR-2011's three follow-ons are still open at the pin, and its citations have moved

- **The follow-ons (still open):**
  - The dotfile guard runs as middleware and carves out only `/.well-known/`, `/.pods` and `/pay/` (`lib.rs:3857-3866`, wrapped at `:5218`).
  - `MirrorHandle::stop` is called only in a test (`mirror.rs:3257`, `:5529`). The handle sits in a process-static `OnceLock` registry (`lib.rs:4998-5001`, `:5013`).
  - A `start_forge` failure is logged (`lib.rs:5061`), and every later `/forge` request answers 500 (`lib.rs:5091-5094`).
- **Citation drift:**

  | Cited | Now |
  | --- | --- |
  | `lib.rs:4999-5000` | `:5000-5001` |
  | `:5306-5307` | `:5326-5327` |
  | `:3844-3846` | `:3857-3866` |
  | `:5198` | `:5218` |
  | forge `lib.rs:336` (`ForgeService::handle`) | `:419` |

  `request.rs:23,85` still resolve.
- **Similar drift in the Close-outs of ADR-2009 and ADR-2010:**
  - `start_forge`: `lib.rs:5044` is now `:5057`.
  - `mirror_config`: `4964-4985` is now `4973-4995`.
  - `main.rs:460` and `Cargo.toml:170-172` hold.

### F6. Frontmatter conformance

- **Trigger:** `TEMPLATE.md:14-15` requires `evidence_class` and `acceptance_boundary` once `implementation_status` is `complete`.
- **Gap:** ADR-2009 and ADR-2010 are `complete` and carry neither.
- **Age:** both are `verified_commit: 76ab7aa`, which predates the SP-GIT-02 repair.
- **What would close it:** add both fields and restamp after F2.

### F7. Parity checklist drift (not an assigned file)

- **The headline disagrees with ADR-2009's Close-out:** `PARITY-CHECKLIST.md:52` reads "≤ 86.4% — 242/280". ADR-2009's Close-out quotes `Strict: 245/247 = 99.2%`.
- **Row 242 (line 627) is stale:** it says the server "does not start it at boot (there is no `forge-mirror` server feature…)". `Cargo.toml:170` and `start_forge` contradict this.
- **Hand-off:** this belongs to whoever owns the checklist and to SP-08 (w-solid-e).

### F8. ADR-057 residue: the IdP advertises grants it refuses

- **What is advertised:**
  - The IdP discovery advertises `refresh_token` and `client_credentials` (`crates/solid-pod-rs-idp/src/discovery.rs:99-103`).
  - The library's `discovery_for` advertises `client_credentials` too (`crates/solid-pod-rs/src/oidc/mod.rs:173-177`).
- **What is accepted:** the token endpoint rejects every grant other than `authorization_code` (`crates/solid-pod-rs-idp/src/provider.rs:227-231`). The module doc says refresh tokens are not shipped (`provider.rs:28-31`).
- **No other consumer:** no server code honours `client_credentials` (no match under `crates/solid-pod-rs-server/src`).
- **Consequence:** metadata advertises capabilities the server does not have. This falsifies ADR-057 A.4 ("the pod still honours it") and B.7 ("the new access token inherits the original `jkt`").
- **What would close it:** remove the two grants from both discovery documents, or implement them. Then test discovery against the token endpoint.

### F9. The LWS CID verifier's `iat` window check cannot fire

- **Source:** `validate_time_bounds` joins the past-skew and future-skew comparisons with `&&` (`crates/solid-pod-rs/src/auth/lws_cid.rs:343`). One `saturating_sub` is always 0, so the condition is never true. Compare `||` in the DPoP check (`oidc/mod.rs:611`).
- **Consequence:**
  - A token without `exp` passes the time bounds whatever its `iat`.
  - A token with `exp` is bounded only by `exp + skew >= now` and `exp - iat <= 3600` (`lws_cid.rs:350-362`).
- **Test gap:** no test exercises `iat` outside the window (`lws_cid.rs:808`, `:957`, `:971`; `tests/cid_verifier_sprint11.rs` has no `iat` case).
- **Limit:** source only. The verifier is behind the `lws-cid` feature, and I did not check its callers' freshness or replay policy.
- **What would close it:** change `&&` to `||` and add stale and future `iat` tests.
- **Overlap:** this is relevant to ADR-057's CID context and probably to SP-05 (w-solid-c).

### F10. BASELINE and ADR-060 claims that no longer hold

- **BASELINE divergence 1** (`BASELINE-solid-pod-rs.md:199-207`):
  - It says there is no `authorization_response_iss_parameter_supported`. That is true of the library, but the IdP advertises it (`idp/src/discovery.rs:59`, `:120`).
  - It cites `extract_webid` at `oidc/mod.rs:864-876`. The function is now at `:1013-1025`.
- **ADR-060 Decision 2** says the IdP's `schnorr.rs` delegates to `nip98`. The IdP calls k256 directly (`crates/solid-pod-rs-idp/src/schnorr.rs:236`, `:365`, `:398-399`). This agrees with SP-ID-04 in `2026-11-01-solid-identity-boundaries.md`.

## ADR-2009 section assessment

Classification: **operative decision**. Its default-off clauses are superseded by ADR-2010, as its own Close-out says.

| Section | Disposition |
| --- | --- |
| ADR-2009 — Track the JSS plugins forge as a bounded second parity comparator, read repo-held git-mark trails, and ship the kind-30618 mirror default-off | Title. "Default-off" is superseded by ADR-2010: the server ships `forge-mirror`/`forge-full` (`crates/solid-pod-rs-server/Cargo.toml:170-172`), and `start_forge` runs at boot (`lib.rs:5057`, `main.rs:460`). The rest of the title holds. |
| Context | Historical facts. The JSS commits cannot be checked from this repository. The trail model and the TapTweak walk exist in core (`mrc20::bt_trail_outputs`, used at `crates/solid-pod-rs-forge/src/gitmark.rs:382-388`). The pod lane still refuses push (`lib.rs:4762-4768`). |
| Decision | **Implemented**, with one mechanism superseded. Two comparators: cannot be checked fully (checklist §25 exists; see F7). `gitmark` reuses `bt_trail_outputs`/`bt_trail_addresses` (`gitmark.rs:382-388`). `blocktrails.json` is served with `access-control-allow-origin: *` through `read_blob` with a 512 KiB cap (`forge/src/lib.rs:1116-1151`, `gitmark.rs:44`, `:406-408`). Mirror guards: http/https-only remote clones (`mirror.rs:186-191`), userinfo refused (`:197-212`), `--` before the URL (`:1835`), injected `SsrfCheck` (`:2377-2384`), bounded frames, tags and follows (`:111-119`), branch verdict with key, list and reachability checks (`:769-815`), replay and far-future drop (`:2873-2905`). Chain never consulted (`:752-757`). `/forge` registered before the pod git routes (`lib.rs:5326-5327` vs `:5368-5370`), pinned by `crates/solid-pod-rs-server/tests/forge_prefix_precedence.rs:41`. "Nothing in the server starts it" is superseded (ADR-2010). |
| Consequences | Mixed. The rows 239/240 partial figures are superseded by the Close-out. The `SsrfCheck` blocking loopback is **true** and live (F1). Residual risks (1) to (4) are superseded by ADR-2010's fixes: pinning `mirror.rs:1825`, size cap `migrate.rs:122` with a process-group kill at `migrate.rs:294`, cheap filters before the signature at `mirror.rs:2396-2432`, per-repository queue `:2437-2470`, push refused at `forge/src/lib.rs:826-839`. "Fixed after review" (`hideRefs` on every mirror repository) is **superseded in mechanism** by store isolation (F2). Risk (5), about JSS, cannot be checked from source. |
| Verification | Test counts, clippy, doc and parity-script output are test/CI claims. **Cannot be checked from source.** No run was done here. |
| Close-out 2026-10-30 | `nip34.rs`, `marks_ui.rs` and `trail.rs` exist. `start_forge`/`mirror_config` citations have moved (F5). The parity figure 245/247 disagrees with the checklist at the pin (F7). Release and runtime observations are dated deployment claims and **cannot be checked from source**. Configuration is not the only reason no relay session exists (F1). |
| Review qualification, 2026-11-01 | Correct for `b644e55`. **Stale at the pin**: the unadvertised-object options are forced false (`crates/solid-pod-rs-git/src/service.rs:427-429`), and store isolation now withholds held objects (F2, F3). |

## ADR-2010 section assessment

Classification: **operative decision**, amending ADR-2009 and BASELINE invariant 8.

| Section | Disposition |
| --- | --- |
| ADR-2010 — The forge and the kind-30618 mirror are on in shipped builds; anchoring, open edit and the mainnet stay opt-in | **Implemented** for the server features. "Shipped builds" depends on the consumer: agentbox adds `forge-full` (`agentbox` `flake.nix:1548`). |
| Context | Historical. The forge modules named exist. The `secreview-wave1` findings cannot be checked from source. |
| Decision | **Mostly implemented.** `forge-mirror`/`forge-full` exist (`Cargo.toml:170-172`). Forge crate defaults are empty (`crates/solid-pod-rs-forge/Cargo.toml:55`). The mirror is inert without relays and a valid follow entry (`lib.rs:4973-4995`). `open_edit` defaults false (`forge/src/config.rs:131`) with a boot warning (`forge/src/lib.rs:299-304`). DNS pinning (`mirror.rs:1821-1826`), normalisation (`:197-212`), the size cap with a process-group kill (`migrate.rs:122`, `:294`), signature after cheap filters (`mirror.rs:2396-2432`), a bounded queue (`:2437-2470`), backoff (`:217`, `:2741-2746`) and push refused (`forge/src/lib.rs:826-839`) are all present. **Contradicted:** `OperatorRelays` for mirror relays (F1), and the "operator's own key" default (F4). **Superseded:** "hidden `refs/mirror`" (F2). |
| Consequences | Egress is inert without configuration: **true** (`lib.rs:5036-5042` runs only when `mirror_config` returns `Some`). The VisionClaw `/forge` adapter cannot be checked here. CSP and compare residuals cannot be checked in this pass. Supersession of ADR-2009's default-off clauses holds. |
| Verification | Test and gate claims at `8dd5dc1`. **Cannot be checked from source.** |
| Close-out 2026-10-30 | `Cargo.toml:170-172` holds. `lib.rs:5044` and `4964-4985` have moved (F5). Agentbox `flake.nix:1547` is now `:1548`. Release, pid and binary-string observations are dated runtime claims and **cannot be checked from source**. The stated cause of no session is incomplete (F1). |
| Review qualification, 2026-11-01 | **Stale at the pin**, as in ADR-2009 (F2, F3). Deployed acceptance is still open: agentbox pins `76ab7aa` (`lib/solid-pod-rs.nix:58`). |

## ADR-2011 section assessment

Classification: **operative decision** (proposed).

| Section | Disposition |
| --- | --- |
| ADR-2011 — Extensions are workspace crates composed at compile time; route ownership is the WAC exemption | **Partly implemented**, matching its own `implementation_status: partial`. |
| Context | The forge is a workspace crate behind features (`Cargo.toml:165-172`). The checklist classes rows 221 and 223-229 as not applicable (`PARITY-CHECKLIST.md:570-578`). The JSS `src/plugins.js` and `plugins/NOTES.md` citations are in another repository and **cannot be checked from source**. |
| Decision | Rule 1 is **implemented**: no runtime loader. The only mount is a feature-gated route block (`lib.rs:5324-5327`). Rule 2 is **partly implemented**: every method under `/forge` goes to `handle_forge` (`web::route()`, `lib.rs:5326-5327`) ahead of the pod git and catch-all routes, but the dotfile guard does not defer (`lib.rs:3857-3866`). Rule 3 is **implemented**: no `PodExtension` trait, and the `ForgeRequest`/`ForgeResponse` shape is at `forge/src/request.rs:23`, `:85` and `ForgeService::handle` at `forge/src/lib.rs:419`. Rule 4 is a parity policy and cannot be checked from code. Rule 5 is **not implemented**: `MirrorHandle::stop` is never called outside tests (F5). |
| Consequences | All three follow-ons are **still open** at the pin (F5). Upstream's `src/plugins.js:324-330` cannot be checked here. |
| Verification | Source citations have moved (F5). The runtime PUT 404 and dot-segment 403 results at `e1d16ad` are runtime claims and **cannot be checked from source**. The 403 is consistent with the guard code at the pin. |

## ADR-057 section assessment

Classification: **historical**, superseded by ADR-2003 (holds Solid-OIDC 0.1 and defers LWS10) and BASELINE divergence 1. Unless a row says otherwise, citations are to `crates/solid-pod-rs/src/oidc/mod.rs` at the pin.

| Section | Disposition |
| --- | --- |
| ADR-057 — LWS10 OIDC Delta (Sprint 11 row 150) | The status block's open-items list (lines 3-24) is **still true**: none of the five P1 items landed (B.3, B.5, C.2 and C.1 below; B.1 only in the IdP). Superseded by ADR-2003. |
| 1. Context | Still true as a description of the Solid-OIDC 0.1 surface: registration `:92`, discovery `:156-188`, DPoP with `ath` `:406-459`, alg allow-list `:541-549`, jti cache `oidc/replay.rs:89`, `:312`, introspection `:1051`. |
| 2. Current implementation surface | Still true. Every listed symbol exists. |
| 3. Section A — fields we emit that LWS10 does not require | Still true. |
| A.1 Legacy application_type parameter (registration request) | Still true (`:73-75`; IdP `registration.rs:67`, `:93`). |
| A.2 solid_oidc_supported discovery metadata | Still true (`:151`, `:185`; IdP `discovery.rs:63`, `:121`). |
| A.3 client_secret in "none"-auth responses | Cannot be checked in this pass. The registration response construction was not traced. |
| A.4 grant_types_supported: client_credentials | Advertisement still true (`:173-177`; IdP `discovery.rs:99-103`). "The pod still honours it" is **contradicted** (F8). |
| 4. Section B — fields LWS10 requires that we do not emit | Still true overall. Superseded by ADR-2003. |
| B.1 authorization_response_iss_parameter_supported: true | **Implemented in the IdP only** (`idp/src/discovery.rs:59`, `:120`). Absent from the library's `discovery_for`. |
| B.2 Explicit require_pushed_authorization_requests flag | **Still open**: no PAR endpoint or metadata anywhere (no match for `pushed_authorization` under `crates/`). |
| B.3 dpop_signing_alg_values_supported MUST include EdDSA | **Still open**: `["ES256","RS256"]` (`:184`), while the verifier accepts EdDSA (`:549`). The IdP advertises ES256 only (`discovery.rs:119`). |
| B.4 client_registration_types_supported | **Still open**: no match under `crates/`. |
| B.5 lws_supported replaces solid_oidc_supported | **Still open**: no `lws_supported` field. |
| B.6 Client ID Document MUST be fetchable with DPoP-bound request | **Still open** on the resource-server path: `verify_access_token` does not resolve `client_id` (`:905-940`). The IdP dereferences Client ID Documents for its own registration lookup (`idp/src/registration.rs:211`), which is a different control. |
| B.7 cnf.jkt MUST survive refresh-token rotation | **Contradicted**: the IdP issues no refresh tokens (F8). No `tests/oidc_refresh_jkt_binding.rs` exists. |
| 5. Section C — semantic differences | Mixed; see below. |
| C.1 webid claim location | **Still open**: `CnfClaim` holds only `jkt` (`:683-687`), and `extract_webid` never reads `cnf` (`:1013-1025`). The "now checks `cnf.webid` first" wording was never true. |
| C.2 DPoP proof iat skew | **Still open**: the IdP default is 60 s (`idp/src/provider.rs:77`, used `:241-246`). The server does not call the DPoP verifier at all (no `verify_dpop` under `crates/solid-pod-rs-server/src`). |
| C.3 client_id URL format | Documentation only. Follows B.6 (open). |
| C.4 Token aud semantics | **Implemented differently and stricter**: `AccessTokenPolicy` requires `aud` to be `"solid"` by default or an explicit audience (`:759-787`, `:921`). There is no origin-only switch. Superseded by BASELINE divergence 5c and ADR-2003. |
| C.5 Introspection response DPoP binding | **Still true**: `from_verified` always emits `cnf` (`:1051-1061`). |
| 6. Action items | None of items 1-10 is done, except B.1 in the IdP and C.4 in a different form. |
| 7. Out of scope | Still true: no RAR, FAPI or token-binding code found. |
| 8. Parity verdict | Superseded by ADR-2003: the pod is Solid-OIDC 0.1, not LWS10. |
| 9. References | External references. Cannot be checked from source. |
| 10. Revision history | Historical; accurate as a record. |

## ADR-058 section assessment

Classification: **historical**. Its gaps are superseded by BASELINE divergence 2, ADR-2005 (size cap and iterative `..`) and ADR-2002.

| Section | Disposition |
| --- | --- |
| ADR-058: JSS v0.0.60–v0.0.71 Feature Drift Analysis | The status block "Accepted / Supersedes ADR-057" is historical. ADR-057's OIDC delta is not superseded by it: ADR-2003 carries that. |
| Context | Historical. JSS release facts cannot be checked from this repository. |
| Drift Inventory | Superseded. Most "gap" rows are closed (below). |
| Security Hardening (6 commits — P0/P1) | All six are **closed** at the pin: minimum password length 8 (`crates/solid-pod-rs-idp/src/credentials.rs:31`, `:190`); ACL size cap (`crates/solid-pod-rs/src/wac/mod.rs:28`, env `config/sources.rs:374`); iterative `..` (`multitenant.rs:164`, `:242`); jti cache (`oidc/replay.rs:312`); DNS failure is an error (`security/ssrf.rs:523`); git path guard (`crates/solid-pod-rs-git/src/guard.rs:4`). Whether every DNS-failure caller short-circuits was not traced. |
| ActivityPub Federation (8 commits — P2) | **Closed** where checked: outbox POST wraps a Note in `Create` (`crates/solid-pod-rs-activitypub/src/outbox.rs:135-150`); delivery User-Agent (`delivery.rs:72`, `:93`, `:140`); `cf-visitor` (`crates/solid-pod-rs-server/src/scheme.rs:5`); AP CLI knobs (`main.rs:154-155`); AP conneg (`crates/solid-pod-rs-server/src/activitypub.rs:7`). The `03aec0d` audit item cannot be checked. |
| Identity Provider (2 commits — P1/P2) | **Closed**: `.account` is in `DEFAULT_ALLOWED` (`security/dotfile.rs:24`). Minimum password length as above. |
| Error Pages / Mashlib (4 commits — P3) | **Partly superseded**: a mashlib configuration exists in the server (`lib.rs:170`, `:222-225`). I found no dedicated 401/403 HTML helper. "We don't serve HTML error pages" was not re-traced. |
| Platform (3 commits — explicitly-deferred) | Not applicable; no source claim. |
| Design Docs (informational) | No source claim. |
| Bounded Context Assignment (DDD) | Mostly accurate file map (`security/dotfile.rs`, `wac`, `idp/src/credentials.rs`, `activitypub/src/{outbox,delivery,actor}.rs`). `security/ssrf.rs` exists. |
| Decision | Historical sprint plan. It was carried out for the items checked above. |
| New Parity Rows | Historical. Row numbers were not re-traced against the current checklist. |
| Consequences | Historical projections (row totals, about 98% strict). Superseded by the current checklist headline (F7). |

## ADR-059 section assessment

Classification: **historical**, superseded by ADR-2004 (provenance opt-in), ADR-2006 (D6 replay), ADR-2007/ADR-2008 (D3 explorer and tx path) and BASELINE divergences 3-4.

| Section | Disposition |
| --- | --- |
| ADR-059: Block-trails and git-marks as first-class provenance primitives | The status block (line 3) is **still true**: the composition and `_prov` shipped, the `git` feature is off by default, and there is no enumeration index. |
| Status detail (open items, 2026-07-03) | **Still true.** `MempoolBlockAnchorer` (`crates/solid-pod-rs-server/src/mempool.rs:676`), `bitcoin_tx.rs` (`MempoolBroadcast` at `:502`), `EpochAccumulator` (`handlers/prov.rs:21-34`), `_prov` routes (`handlers/prov.rs:748-749`), single canonical path (`lib.rs:4005`), `git_mark_write` no-op without `git` (`lib.rs:4121`), server default features empty (`Cargo.toml:127`). D2 is still open: no `ProvenanceTrail` type, only README prose. |
| Context | Historical. "`handle_git` never calls the WAC enforcers" is superseded: push is refused for all callers and reads are WAC-gated (`lib.rs:4762-4790`). |
| Decision | Superseded by ADR-2004, ADR-2006, ADR-2007 and ADR-2008, as mapped below. |
| D1 — Two tiers, composed | **Still true with the opt-in caveat** (ADR-2004): `ProvenanceLog` composes the marker with an optional anchorer (`crates/solid-pod-rs/src/provenance.rs:317`, `:382-384`). |
| D2 — Generalise the trail beyond tokens | **Not implemented**: no `ProvenanceTrail` type exists. |
| D3 — Reuse the verified crypto; add only the missing edges | **Superseded by ADR-2008** (rust-bitcoin port). `bitcoin_tx.rs` and `mempool.rs` exist. Their crypto provenance was not re-traced here. |
| D4 — Traits are wasm32-safe; native I/O is feature-gated | **Partly true**: the traits are `?Send` (`provenance.rs:251-253`, `:279`), and the docs say wasm injects a no-op marker (`:12`, `:378-384`). No no-op `GitMarker` implementation exists in this repository (the only `impl GitMarker` is a test mock, `:1495`). The wasm job was not checked. |
| D5 — One Bitcoin tx notarises many commits | **Implemented**: `AnchorPolicy::Epoch` and `EpochAccumulator` (`handlers/prov.rs:21-34`). Not exercised here. |
| D6 — Close the debt the primitives expose | WAC-gate git: **superseded** by refusing all pod push and WAC-gating reads (`lib.rs:4762-4790`). Replay: `check_replay` is implemented on the payment store (`handlers/pay.rs:305`; trait `payments.rs:914`). Call sites on the deposit path were not traced. `PaymentStore` as the sole ledger path: `StoragePaymentStore` exists (`handlers/pay.rs:15-16`), and is further superseded by ADR-2008. |
| D7 — Persistence and surface | **Implemented**: `.prov.ttl` sidecar (`lib.rs:3892`), `_prov` GET/POST (`handlers/prov.rs:748-749`). Emission on `Updates-via` was not traced. |
| Considered options | Historical rationale; no source claim. |
| Consequences | "Every pod write is traceable" is **contradicted** for default builds (ADR-2004). "Anonymous git push closed" is true. The rest is projection. |
| Status → Accepted criteria | Tag `v0.5.0-alpha.0` exists. Green native and wasm builds cannot be checked from source. |

## ADR-060 section assessment

Classification: **historical**. Decision 2 is superseded by ADR-2006, Decision 4 by ADR-2004, and Decision 3 by BASELINE divergence 4.

| Section | Disposition |
| --- | --- |
| ADR-060: Gap-Close sprint — decisions the solid-pod-rs slice forces | The status block "Proposed" is historical. The linked PRD and DDD files exist at the pin. |
| Context | Historical. Recon at `cded64f` cannot be re-run from source. |
| Decision 1 — Close REC-1 PATCH with a receipt and correct the register, rather than re-implement | The fix is **still present**: `handle_patch` (`lib.rs:1884`) and `seed_graph_from_patch_target` (`:2183`), with the test file present (`tests/patch_non_destructive_integration.rs`). The cited lines `1210` and `1467` have rotted (BASELINE divergence 7 also records this). Receipts and canaries cannot be checked from source. |
| Decision 2 — Keep NIP-98 single-sourced in-repo; extract a ReplayStore seam to standalone and record the edge-local exception, rather than build the cross-tier store now | **Superseded by ADR-2006** and implemented: `trait ReplayStore` (`auth/replay_store.rs:90`), with `Nip98ReplayCache` implementing it (`auth/replay.rs:117`, `:235`). Git delegates (`crates/solid-pod-rs-git/src/auth.rs:144-157`). The IdP `schnorr.rs` delegation claim is **contradicted** (F10). |
| Decision 3 — Define the REC-11 provenance-trace contract; do not build the pod-wide index in this sprint; correct ADR-059's stale status | **Still true**: the contract file exists (`docs/reference/provenance-trace-contract.md`). There is no enumeration route, only `_prov/anchor` and `_prov/{sha}` (`handlers/prov.rs:748-749`). ADR-059's status was corrected (ADR-059 lines 19-39). |
| Decision 4 — Keep the git provenance feature off by default in this sprint | **Still true and superseded by ADR-2004** (`Cargo.toml:127`, `lib.rs:4121`). |
| Decision 5 — Regenerate the RES-c diagrams under VisionFlow's RES-b render gate; declare the dependency, do not ship a local render claim | Partly visible in source: 9 `.mmd` and 9 rendered PNGs under `crates/solid-pod-rs/docs/diagrams`, and `scripts/check-diagram-staleness.sh` exists. Freshness and the RES-b gate cannot be checked from source. |
| What This ADR Does Not Govern | Scoping statement; consistent with the code (no enumeration index, no cross-tier store). |
| Consequences | Overview only; the substance is in the three subsections. |
| Positive | Historical. The ADR-059 correction is visible. |
| Tradeoffs | **Still true**: the cross-tier store and the enumeration index are both still open (BASELINE divergences 4-5). |
| Risks | RES-b dependency and `mmdc` toolchain: cannot be checked from source. |
| Governance | Process obligations in another repository; cannot be checked from source. |

## ADR-061 section assessment

Classification: **historical**, superseded by ADR-2007 (one configured explorer URL, public default, no silent fallback) and BASELINE divergence 6. Lines 45, 48 and 51 are `#` comments inside a fenced block, not headings, so they get no rows.

| Section | Disposition |
| --- | --- |
| ADR-061: LAN Bitcoin node (192.168.2.27) as the anchoring + Lightning substrate | The status block (cutover blocked) cannot be checked from source. BASELINE divergence 6 still records it as unverified. |
| Context | **Still true** for the crate: `JSS_PAY_MEMPOOL_URL` (`crates/solid-pod-rs-server/src/mempool.rs:57`), with the default testnet4 noted at `lib.rs:259`. The infrastructure facts cannot be checked from source. |
| Decision | Infrastructure decision; cannot be checked from source. The crate-side REST contract is `MempoolHttpClient` (`mempool.rs:373`). |
| Dual-network posture (2026-08-25 amendment) | Crate side **still true**: no `JSS_PAY_NETWORK` (no match), so the URL selects the network. The LAN URL is not a crate default. Superseded by ADR-2007. |
| Non-goals / rejected | **Still true**: one configured URL, no ordered fallback found. The REST seam is trait-shaped (`MempoolLookup` at `crates/solid-pod-rs/src/mrc20.rs:372`, `MempoolBroadcast` at `bitcoin_tx.rs:502`). Superseded by ADR-2007. |
| Acceptance checklist (gates the cutover — run when .27 is reachable) | Deployment checks; **cannot be checked from source**. The `env-vars.md` it names does not exist at the pin. |
| Consequences | "Anchoring becomes sovereign" is **not true by default** (the public default stays). The unreachability record is historical. |

## Support files

| File | Classification | Accuracy at the pin |
| --- | --- | --- |
| `crates/solid-pod-rs/docs/adr/PREAMBLE.md` | Support (preamble) | Accurate: `scripts/adr-index-gen.js` and `BASELINE-solid-pod-rs.md` exist, and the archive pointer is correct. |
| `crates/solid-pod-rs/docs/adr/README.md` | Support (generated index) | Accurate: "11 record(s)" matches the 11 ADR files, and the status columns match each record's frontmatter. |
| `crates/solid-pod-rs/docs/adr/TEMPLATE.md` | Support (template) | Accurate as a template. ADR-2009 and ADR-2010 do not meet its `evidence_class` rule (F6). |
| `crates/solid-pod-rs/docs/archive/adr/README.md` | Support (archive index) | Accurate: the redirect table's divergence numbers 1-7 exist in the BASELINE (`:199-251`). It names no ADR-20xx successors; the lineage fields of ADR-2003, ADR-2004, ADR-2005, ADR-2006 and ADR-2007 supply them. |

## Verification

- **Evidence class:** source tracing at pinned revisions only. I read every cited line with `git -C <repo> show <sha>:<path>`, or resolved it by `git grep -n <symbol> <sha>`. Nothing was compiled, tested, run or deployed.
- **Pinned revisions:**
  - solid-pod-rs `9b6143ce55b2` (all ADR text and product source);
  - agentbox `c57d9c0508f4` (`flake.nix`, `agentbox.toml`, `lib/solid-pod-rs.nix`);
  - solid-pod-rs `76ab7aa`, read only to confirm that the F1 wiring is the same at the consumer's pin.
- **Generator:** not run. No topic was assigned to this worker, and no topic file was edited.
- **Not verified:**
  - every runtime, test, CI, release and deployment statement in the ADRs (marked "cannot be checked from source");
  - JSS upstream citations;
  - ADR-057 A.3 registration-response contents;
  - DNS-failure short-circuiting at every caller;
  - deposit-path call sites of `check_replay`;
  - `Updates-via` emission of provenance marks;
  - the SP-GIT-02 B1/B2 migration internals beyond existence;
  - whether F1 and F9 show up at runtime.
- **Not edited:** no ADR or product file. ADR wording faults are findings for the owner.
