---
title: Solid Pod OIDC verification and resource admission
date: 2026-11-01
status: source-review-recorded
type: explanation
---

# Solid Pod OIDC verification and resource admission

The library verifies signatures and enforces an audience policy, but the main
pod resource server does not consume that OIDC path. A private-resource probe
accepts its Nostr control, then refuses an otherwise library-verified ES256
token and DPoP proof with HTTP 401 while advertising DPoP in the challenge.
This is an integration gap, not evidence of unauthorised resource access.

The same review reproduced discovery and JWKS redirects reaching a hostname
explicitly denied by the supplied SSRF policy. The producer repair disables
automatic redirects for both fetches, merged in Solid Pod commit `fcc624a3e952e05923d93d7ea038294bcddb25ae`. Direct approved endpoints remain
supported. Published packages, consumers and deployments need separate adoption.

## Decision assessment

All sections of Solid ADR-2003 are assessed against baseline `8f2df86` and the
scoped repair. Its Solid-OIDC 0.1 deferral remains a decision; this review does
not adopt a different protocol profile.

| Section | Disposition |
| --- | --- |
| Context | The older draft and ticket list explain the decision. They are historical context, not current standards evidence. |
| Decision | Discovery still advertises ES256/RS256; the verifier accepts a wider asymmetric set. Profile adoption remains deferred. Enabling a library feature does not connect resource routes. |
| Consequences | Compatibility limitations remain real, but the old proposed `cnf.webid` field list is not a current LWS acceptance checklist. |
| Verification | Source symbols confirm the discovery and extraction behaviour. Old offsets and metadata do not prove an interoperable deployment. |
| Closeout extension | A versioned matrix exists. External-issuer resource admission and deployed compatibility remain unproved; the local route probe now demonstrates the missing dispatch. |
| September acceptance | The 28 compatibility tests pass, but the prose had drifted: current audience policy rejects nonmatching values. Malformed WebID fallback, issuer normalisation, URL matching, nonce omission and public thumbprint comparison retain their distinct meanings. |
| November source refresh | Baseline and matrix are corrected for current audience/key policy, profile-binding and route limits, and the redirect repair. The new section records open acceptance conditions without changing the deferral or deployment status. |

Discovery field names are mechanically compared with the matrix; the rest of
its prose is not automatically checked by a green test suite. The matrix now
states that limitation and distinguishes the reviewed checkout from the
published alpha.18 package.

## Findings and acceptance conditions

| ID | Evidence and consequence | Required acceptance |
| --- | --- | --- |
| SP-OIDC-01 | Both fetch functions followed a 302 from an approved loopback fixture to a denied hostname. Each target received one request and its response was accepted. This proves a policy bypass in the reusable fetch API; external deployment exposure was not tested. | Producer repair refuses redirects; preserve direct-fetch success, then publish, upgrade consumers and verify their network policy. |
| SP-OIDC-02 | Main resource extraction calls NIP-98; production Bearer is refused and DPoP has no dispatch. Its challenge nevertheless advertises both. The valid Nostr control reads private bytes; the ES256 OIDC/DPoP pair passes library checks but receives 401. | Compose the selected OIDC profile into real resource admission, including proof, issuer authority, audience, replay and WAC, and pass an external-client journey. |
| SP-OIDC-03 | `normalise_htu` lower-cases the path and trims trailing slashes. A valid proof for `/Foo` verifies against `/foo` and `/Foo/`, while adding a query causes refusal. | Compare parsed URI components according to the selected DPoP contract, with case-distinct resources and query-bearing requests tested. No resource-route bypass is claimed. |
| SP-OIDC-04 | With capacity one, a valid second proof evicts the first unexpired `jti`; the unchanged first proof then verifies again. The IdP supplies this cache during token exchange, before code consumption. | Retain admission records for the remaining acceptance window and define saturation behaviour. Single-use authorisation codes are a separate defence; this probe does not prove duplicate token issuance. Default-TTL timing remains unexecuted. |
| SP-OIDC-05 | The profile helper accepts a literal compact issuer property despite an unrelated subject and a context assigning that prefix elsewhere. Equivalent Turtle is refused. The cached token verifier does not call this helper, and no external-resource consumer was found in the reviewed crate family. | Bind the exact WebID subject and expanded issuer predicate through the estate's RDF tooling, support the chosen profile representations, and verify the actual consumer composition. |
| SP-OIDC-06 | Real ES256 tokens with `exp == now` or future `nbf` pass. A `https://`-only WebID passes the prefix test; a non-URL WebID silently falls back to `sub`. Wrong audience and expired-by-one-second controls fail. | Correct time and identity-shape policy with deterministic fixtures and explicit compatibility decisions. These are verifier observations, not a demonstrated route takeover. |
| SP-OIDC-07 | Alpha.8 prose claimed arbitrary audiences while current code and test C1 enforce `solid` or an explicit policy. A passing matrix suite had not prevented prose drift. | Corrected in the baseline, matrix, ADR extension and test commentary. Preserve source-version and evidence boundaries on future updates. |

The fetcher still uses independent config/JWKS caches: a JWKS miss fetches
discovery directly, without consulting the config cache. Cached keys are not
refreshed on an unknown `kid` before TTL expiry. Cache growth and key-rotation
behaviour need separate acceptance work; no load or rotation experiment is
claimed here. Caller-supplied HTTP client settings are not preserved when the
fetcher builds its pinned client.

Deepsec returned PASS at its HIGH threshold, with four MEDIUM findings and one
BUG on `jwks.rs`. All five concern pre-existing code: unbounded caches,
uncapped JSON response buffering, proxy routing around the direct-connection
pin, detailed SSRF errors, and bypass of the config cache on a JWKS miss.
Source inspection supports those mechanisms, including the pinned reqwest
0.12.28 implementation. Memory exhaustion, proxy rebinding and client-visible
error disclosure were not reproduced. In particular, a detailed library error
does not prove a remotely reachable disclosure on the unwired resource route.
The findings remain open with those limits. The scanner reviewed only
`jwks.rs`; test execution and manual review cover the new test file changes.

## Executed evidence

Eight native integration targets pass 73 tests, including three new redirect
regressions; 29 OIDC unit tests separately pass. Five bounded observation tests
pass using real ES256 signing through maintained libraries, a disposable pod
and local fixtures. These observations assert current behaviour, including
defects; they are not green security acceptance tests.

Before the repair, four existing fetch tests passed and both redirect
regressions failed because their denied targets received a request. After the
repair, all seven fetch tests pass. Two initial resource probes failed at the
Nostr positive control: the ACL parser discarded the grant written on the
prefix declaration line. The corrected fixture places declarations and grants
on separate lines. Those initial failures remain inconclusive route evidence,
retained separately from the final five passing observations.

Workspace Clippy, formatting and warning-denied core crate documentation pass.
The compatibility matrix's 28 cases pass again after the prose correction;
that rerun is included in the earlier 73-case scope, not additional coverage.
Raw commands, source hashes, probes and review receipts are retained in the
[evidence manifest](evidence/2026-11-01/solid-oidc/manifest.json).

## Standards, alternatives and user utility

RFC 9449 excludes query and fragment components from the proof target URI;
its server nonce mechanism is optional. Nonce absence alone therefore does
not establish a conformance defect. The observed path folding and query
comparison warrant explicit correction. [RFC 9449](https://www.rfc-editor.org/rfc/rfc9449.html)

JWT expiry requires time to precede `exp`, and a present `nbf` restricts when
a token becomes usable. The observed equality and future-`nbf` cases are
bounded verifier gaps. [RFC 7519](https://www.rfc-editor.org/rfc/rfc7519.html)

Solid-OIDC requires issuer authority to be established from the relevant
WebID's profile statements. Its ID-token requirements should not be copied
indiscriminately onto every access-token API. The current LWS authentication
document is a 15 October 2026 Working Draft using subject-controlled-identifier
and issuer-service binding; the old ticket list is not a substitute for reading
it. [Solid-OIDC 0.1](https://solidproject.org/TR/oidc),
[dated LWS draft](https://www.w3.org/TR/2026/WD-lws10-authn-openid-20261015/)

Community Solid Server documents the complete client sign-in and protected
resource journey, and its default configuration imports both OIDC identity and
DPoP resource authentication. That is a useful acceptance comparator; no
comparative runtime benchmark was run. [CSS tutorial](https://github.com/CommunitySolidServer/tutorials/blob/main/getting-started.md),
[CSS default configuration](https://github.com/CommunitySolidServer/CommunitySolidServer/blob/main/config/default.json)

For VisionFlow users, prioritise one reproducible external-client journey:
sign in, read an authorised private resource, refuse an unrelated WebID and a
replayed proof, survive key rotation, and retain the existing Nostr workflow.
These are proposed utility gates. The present evidence supports a Rust/Nostr
storage integration with OIDC components; it does not support claiming a
drop-in, interoperable OIDC pod service.


## Diagram and ledger scope

SP-05.7, SP-05.8 and SP-05.10 are corrected at their historical `09ec90f`
stamp; their OIDC, replay, JWKS and WebID source bytes match the pre-repair
`8f2df86` baseline. New SP-12.1 and SP-12.2 describe the merged repair and
resource integration boundary. Only these five panels receive new assessment
credit. The previous SP-05.1–5 bodies remain unchanged; rebinding their topic
hash adds no credit. Other rendered identity panels remain outside this pass.

The historical topic resolves 208 citations and renders all 18 panels; the new
topic resolves 17 citations and renders both panels. Both gates report zero
warnings, fallbacks or unresolved citations. The five selected panels were
visually inspected. JEV citation samples are advisory, not full semantic
acceptance. The historical whole-corpus receipt is retained without claiming
a new source scan of excluded repositories.
