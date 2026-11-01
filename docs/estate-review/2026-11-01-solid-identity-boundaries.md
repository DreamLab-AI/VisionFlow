---
title: Solid Pod signed-request and replay boundaries
date: 2026-11-01
status: assessed
type: explanation
---

# Solid Pod signed-request and replay boundaries

A real private PUT route accepts an empty body under a token signed for nonempty content, and persists the empty resource. Separately, configurable replay retention can expire while a token remains valid, and Git's string-prefix policy does not enforce repository-path boundaries. These findings remain open in source. This checkpoint assesses every section of ADR-2006 and SP-05.1 through SP-05.5; the remaining identity panels and OIDC decisions still need assessment.

Reviewed source is Solid Pod `e1d16adbfc096527f82ff16a6bce87ffd6d909f8`, workspace version `0.5.0-alpha.18`. Relevant auth, server, Git and IdP source bytes are unchanged from SP-05's declared `09ec90f26aa573731460292d0e80343914cb43d9`. The [source receipt](evidence/2026-11-01/solid-identity/sources.json) records both boundaries. VisionClaw's separately repaired consumer uses a different verifier and remains at its own evidence boundary.

## Findings and acceptance

| Finding | Evidence and consequence | Required acceptance |
| --- | --- | --- |
| SP-ID-01: empty-body substitution | `auth/nip98.rs:187` hashes only nonempty supplied bodies. A valid signature containing the intended nonempty payload hash also verifies with `Some(b"")`. In the actual private PUT route, the empty request succeeds and replaces the original resource with zero bytes. | Verify any supplied payload commitment against the received bytes, including empty bytes. Test legitimate empty writes, nonempty-to-empty substitution and token non-consumption on rejected bodies through the real route. |
| SP-ID-02: retention shorter than freshness | `from_env` accepts any positive TTL; `with_config` also permits zero. With a 5 ms TTL, the same signed token is refused immediately, then accepted after expiry while the verifier clock is still within its 60-second tolerance. Zero TTL admits the same id twice immediately. | Tie production retention to the complete acceptance interval, or retain an explicit credential expiry. Test future-skewed timestamps, inclusive boundary seconds, clock changes and unsafe configuration. Keep a separate controlled-clock seam for short tests. |
| SP-ID-03: Git scope is a string prefix | A token for `/repo` verifies under GitLenient at `/repo-other/git-receive-pack`; a host-only token also matches a longer hostname prefix. An unrelated URL is refused. These are verifier probes, not executed pushes or a WAC bypass. | Compare parsed origins and enforce an explicit repository path boundary. Document the reusable Git credential contract separately from single-use REST requests, and test actual clone/fetch authorisation. |
| SP-ID-04: the consolidation account overstates reuse | The baseline and ADR cite IdP Schnorr login as delegating to the NIP-98 helper. Its implementation calls k256 directly over its own challenge digest. It is a distinct login protocol, not a second kind-27235 verifier. | Correct the architectural account without forcing different protocols into the same verifier. Shared primitives, shared verification policy and shared replay storage are separate claims. |

For SP-ID-01, anonymous and altered nonempty requests are refused and leave the original bytes intact. Removing the body succeeds; resubmitting the same token is then refused because successful authentication consumed it. A fresh correctly bound request succeeds and restores the intended bytes. This demonstrates substitution before first successful use, not a replay-cache bypass. It requires possession of a valid signed request; no live deployment, interception or external account was exercised.

For SP-ID-02, the ADR's claim that an expired cache entry necessarily represents a stale token is not valid for arbitrary configured TTLs. The 120-second default also meets an inclusive integer-second freshness boundary; that edge and wall-clock changes need a dedicated coupled-clock test. This batch does not claim to have reproduced a default-configured live-server replay. Capacity retention itself remains intact: the existing capacity, concurrency, expiry and restart tests pass.

Git's native server path rejects all pushes before authentication when the feature is enabled (`lib.rs:4750`). Reads call `BasicNostrExtractor` and then WAC against the pod root (`lib.rs:4775`). The extractor does not call ReplayStore. Therefore the prefix observation establishes weak request binding in a reusable credential path; it does not establish that arbitrary users can read private repositories. A captured credential remains bounded by the signer's WAC rights, freshness and the deployed route/origin policy. The `git` feature and running deployment were not exercised by these probes.

## ADR-2006 section assessment

| Section | Disposition |
| --- | --- |
| Context | The core and Git share kind-27235 verification. IdP login uses a different challenge protocol; its direct k256 call contradicts the claimed helper delegation but is not duplicate NIP-98 policy. |
| Decision | The runtime-independent ReplayStore trait and process-local implementation exist. Git delegates signature verification but intentionally reuses its credential and skips the store. Server REST claims the global cache after verification. No single cross-tier replay boundary is established. |
| Consequences | Replica and restart limitations are accurate. Adopting a published trait alone would not supply durable shared state. The deployment burden remains open. |
| Verification | The cited method exists and Git calls the shared policy-aware verifier. That source check does not establish equivalent binding or replay semantics across callers. The IdP citation is misleading. |
| Closeout extension: 2026-09-04 | Historical live-entry eviction is repaired. The wider acceptance conditions still require configured-window, restart, replica and deployed-adapter tests. |
| Acceptance progress: 2026-09-05 | One mutex, refusal at live capacity and no refresh on rejection remain implemented. All thirteen replay tests pass again as part of 31 native auth tests. The historical 1,801-test workspace result is not a fresh receipt. Short TTLs invalidate the statement that reclaimed entries cannot still authenticate. |

Baseline invariants 1 and 2 are assessed only for these paths. This is not a whole-baseline assessment. The current server's authentication failure returns an anonymous identity; WAC can still allow publicly accessible operations. The earlier diagram's unconditional statement that anonymous always means denial has been corrected.

## Diagrams and verification

[SP-05](../diagrams/solid-pod-rs/05-identity-nip98-did-oidc.md) retains its source stamp. The five selected panels now distinguish anonymous fallback, nonempty-only payload checking, strict-policy aliases, Git prefix/reuse semantics, unvalidated retention and actual trait adoption. Other panels were rendered and citation-checked for the topic gate; they receive no new semantic assessment credit.

The [evidence manifest](evidence/2026-11-01/solid-identity/manifest.json) preserves the Rust probe, commands, logs and hashes. Five bounded probes pass by asserting the observed behaviours, including the defects; they are not product regression tests proving remediation. Thirty-one existing auth tests pass, including thirteen replay tests. The replay-only command repeats those thirteen and adds no unique-test credit.

An initial HTTP fixture wrote the ACL at a path that did not authorise its request and returned 403. It was inconclusive about body substitution. The corrected fixture uses the canonical parent ACL and proves a legitimate signed write succeeds before testing the defect. Both logs are retained. Tests run against actual crate code with disposable in-memory storage; production source and data remain unchanged. No full workspace lint, OIDC interoperability, live proxy, replica or restart campaign is claimed here.

## Protocol comparison and user utility

[NIP-98](https://raw.githubusercontent.com/nostr-protocol/nips/master/98.md) specifies an exact absolute URL including query and the request method; payload checking is optional in the protocol. This estate promises stronger mutation binding. Solid Pod's Strict policy trims trailing slashes and compares methods case-insensitively, so its name does not prove exact protocol conformance. Those compatibility choices need an explicit contract.

[Git's HTTP protocol](https://git-scm.com/docs/http-protocol) uses repository-relative subrequests, supports HTTP authentication and expects clients to manage their protocol state. That explains why one-shot REST tokens do not drop unchanged into stock Git's multiple requests. It does not require arbitrary string-prefix matching across sibling repositories. A useful product boundary is a clearly documented Git credential scoped to one origin and repository, alongside strictly bound individual REST mutations.

For users, the near-term acceptance journey is straightforward: a signed write changes exactly the intended bytes once, and a Git credential opens only the intended repository. A general-purpose shared store or another verifier does not by itself supply either result. Reuse the existing verifier, fix its boundary policies and test each actual consumer before declaring estate-wide consolidation.
