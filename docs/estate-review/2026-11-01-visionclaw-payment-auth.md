---
title: VisionClaw payment authentication and shared verifier adoption
date: 2026-11-01
status: assessed
type: explanation
---

# VisionClaw payment authentication

VisionClaw `5fa0e1325a2f903c175cfe67bbc6d644fcf38267` repairs payment authentication by adopting the shared NIP-98 verifier, binding all five JSON POST bodies and refusing repeated signed tokens. Six route regressions and 46 shared-verifier tests pass. The fix is merged locally; deployment remains outstanding.

At the reviewed baseline `531df5ef7`, `extract_caller_pubkey` called the stateless Solid verifier with `req.uri()` and no body. Payment handlers therefore bypassed the existing replay cache and body-binding policy. The repair also removes a response-header panic that occurred after a resource debit: mixed-case header names now use fallible parsing instead of `HeaderName::from_static`.

This assessment follows VCS-10 / XR-01 in the [parallel implementation audit](2026-11-01-parallel-implementations.md). It covers the payment route authentication boundary and ADR-2002's adoption claim. The combined issue also includes Solid-proxy URL reconstruction and shared fixture adoption. Those obligations remain separate from the payment repair.

## Contract and user effect

A signed request must authorise the operation the server receives. A repeated token must not charge the caller again. A rejected body or URL must not consume a valid token before the caller can present the intended request. Authentication replay protection is process-local and does not provide business-operation idempotency: a newly signed retry can still repeat a charge.

The [NIP-98 specification](https://raw.githubusercontent.com/nostr-protocol/nips/master/98.md) requires the absolute request URL, including query parameters, and matching HTTP method. Payload verification is optional in that specification. VisionClaw's stricter `BodyBinding::Required` policy is appropriate for payment POSTs because changing an order's amount or price changes the authorised operation. The existing shared verifier retains its URL compatibility rules, including scheme tolerance, relative paths and the `/api` rewrite. Reusing it does not certify exact-URL conformance to the specification.

## Scope limits

`POST /pay/.deposit` is a public 501 stub when enabled, not an authenticated deposit mutation. The imported audit's captured-deposit-token example therefore overstates that route's implementation. The live charge, sell, swap and liquidity paths are the relevant boundaries. `PAY_ENABLED` defaults to false; no running deployment or actual funds were tested.

ADR-2111 still proposes deleting the host payment store and replacing it with the Agentbox wallet boundary. Improving authentication on the current routes does not complete that migration or establish that the host ledger holds no value. The charge endpoint returns a receipt without delivering the named resource. Exchange and ledger writes remain separate operations, and storage read errors still fall back to empty state. Those are distinct payment-integrity and product-completeness obligations.

The Solid proxy reconstructs `X-Forwarded-URI` in two places. The shared `nip98_request_url` uses the request URI instead. Replacing those copies without a proxy-rewrite test would change which URL the signature must cover. This remains open in the combined issue, alongside execution of the shared historical token fixtures with an explicit test clock.

## ADR-2002 section assessment

This is a source assessment of the whole record, including its dated extensions. Historical test counts retain their dates and are not represented as freshly executed tests. The governing security profile is read for the authentication invariant; it does not receive a whole-document assessment here.

| Section | Current disposition |
| --- | --- |
| Context | The freshness-only replay problem is accurate. Payment authentication still bypasses the shared cache at the reviewed baseline. |
| Decision | The shared validator verifies the event before `claim_signed_id`. Admission uses one mutex, monotonic expiry, a 100,000-entry ceiling and a 1,000-event per-signer quota. Expired entries can be pruned; live entries are not evicted to admit more callers. |
| Consequences | The assertion that every call site inherits replay resistance is false for the baseline payment routes. Availability costs and the process-local deployment boundary still apply. |
| Verification | The original 26-test receipt is historical. The code still contains concurrency, capacity, expiry and non-burning tests; the current payment route regression receipt is separate. |
| Closeout extension: 2026-09-04 | Body binding and consumption after downstream failure are relevant acceptance conditions for this repair. Restart, replica, response-loss and wall-clock-step coverage remain open. |
| Acceptance progress: 2026-09-05 | `Required`, `NoBody` and legacy `RequiredWhenTagged` exist. Strict binding is not automatic for every caller. The historical statement that no routes use `Required` must be read at its date. |
| Re-verification: 2026-09-05 | The historical line numbers have moved. Claim ordering, capacity and half-open TTL remain visible in `admit_signed_in` and `claim_in`. The cited 39-test execution is not a fresh run. |
| Landing re-verification: 2026-09-06, 2cf222406 | A historical landing and bind-line correction; it does not establish current payment route adoption. |
| Landing re-verification: 2026-09-06, c9734a524 | A historical documentation refresh; it does not establish runtime activation. |
| Source closeout verification: 2026-09-07 | Per-signer admission still follows signature verification and precedes global admission within the same mutex. The historic WebSocket and embedded-Pod receipts have their own scope. |
| Re-verification: 2026-09-21 | The dev-profile default discussion is historical. No payment repair changes the dev flag or relaxes the authentication boundary. |
| Re-verification: 2026-10-07, b43a2a1e6 | Stacked `/api` authentication reuses the validated identity. `/pay` mounts outside that gate and authenticates once within its own handlers; it must not assume middleware has already spent or validated its token. |
| Re-verification: 2026-10-07, e48099f28 | Missing-credential status and memory-cloud dev bypass changes do not repair payment authentication. This batch does not re-test the memory-cloud bypass. |
| Payment route adoption: 2026-11-01 | Five POST handlers now bind raw JSON bytes; GET handlers use NoBody through the same verifier and replay cache. The appended receipt states process-local scope, token consumption after successful authentication, fresh-token retry limits and the outstanding proxy/fixture work. Six actual-route regressions and 46 verifier tests pass. |

Additional limits remain explicit: the validator uses wall time for freshness and monotonic time for cache expiry; this batch does not inject system-clock jumps or restart a process. The signed timestamp subtraction deserves a separate extreme-integer boundary test. The security profile names 503 for cache exhaustion, while payment authentication currently collapses verifier failures to 401. Correct refusal and correct availability signalling are separate obligations.


## Security review dispositions

Deepsec's configured gate returned PASS at its HIGH security threshold, while reporting eight findings: three MEDIUM, two HIGH_BUG and three BUG. That result does not mean the payment system is sound. Its review covered the authentication change before the subsequent response-header repair. Raw findings and a separate disposition receipt are retained with the evidence.

Existing empty-state fallback, separate exchange/ledger commits, synchronous file lock across awaits, lack of order admission limits and process-local replay boundary remain open. Cross-service replay through forwarded host values needs deployment-specific validation; no cross-service attack was run. The review's blanket zero-amount claim is too broad: the pinned crate rejects zero AMM swaps and zero liquidity inputs. Order creation remains unbounded, fee values are not constrained before arithmetic, and empty currency identifiers need a policy.

Upstream settlement is reviewable in the cached, pinned `solid-pod-rs` 0.5.0-alpha.17 source. `execute_swap` checks both buyer and seller balances in their named currencies. Orders are not escrowed at creation, so an offer can become unfillable, but that does not establish a currency-confusion or unfunded-settlement bug. The audit retains the distinction instead of importing that suspicion as a confirmed finding.

## Product boundary and comparison

The [L402 protocol](https://github.com/lightninglabs/L402/blob/master/protocol-specification.md) combines a payment challenge, Lightning invoice, macaroon and preimage. VisionClaw's current payment handler verifies NIP-98 and debits a local ledger; its advertised Lightning deposit remains a stub. Calling the diagram an HTTP 402 balance/debit gate is accurate; it is not an implemented L402 service.

For users, the next useful milestone is a recoverable path from funding through a paid job to its delivered result. Acceptance needs an actual funding receipt, a logical operation identifier across fresh-token retries, one durable settlement, delivery of the requested resource and recovery after a lost response. ADR-2111's proposed wallet boundary is the consolidation path to assess against those requirements. Another host-owned ledger or a second authentication implementation would add work without supplying that complete user journey.

## Verification and remaining acceptance

The [evidence bundle](evidence/2026-11-01/pay-nip98/manifest.json) retains commands, raw failures, final tests, build inputs, review dispositions and source hashes. Tests use disposable state and synthetic balances. No live funds, network settlement or GPU workload were exercised.

| Check | Result and scope |
| --- | --- |
| Baseline route suite | One pass, four failures. The resource test panicked while constructing its first response; that run did **not** reach the duplicate-charge assertion. Stateless replay acceptance is source-confirmed, not proved by that failed test. |
| Authentication repair before header correction | Five passes, one header-panic failure. This isolated the remaining response defect. |
| `cargo test --locked --offline --test pay_nip98_test` | Six passes with default features. Positive controls prove legitimate requests work; tests cover exact body bytes, unbound and altered bodies, all five POST routes, query/host rejection without token consumption, duplicate refusal, a single persisted resource debit, content type and the 2 MiB limit. |
| `cargo test --locked --offline --lib nip98` | 46 passes. Executed before the later header-only correction; the verifier module was unchanged by that correction. |
| Changed-file `rustfmt --check` | Pass. Whole-workspace formatting still fails on unrelated existing files. |
| `cargo clippy --locked --offline --all-targets -- -D warnings` | Fails with five library and twelve library-test diagnostics. The nine diagnostic paths match baseline bytes. No lint suppression was added; this is not a green full gate. |
| Build prerequisites | Initial missing-schema, feature-off binary and missing-PTX failures retained. The supported default-feature build passed with pinned schema and existing PTX inputs whose origins and hashes are recorded. |
| Diagram gates | VC-55: one rendered panel and 19 revision-resolved citations. VC-04: 23 rendered panels and 207 revision-resolved citations. Both have zero citation warnings or working-tree fallback. Only VC-04.27 and VC-55.1 receive assessment credit and visual inspection. |

[VC-55](../diagrams/visionclaw/55-payment-authentication.md) describes the repair at its merged revision. [VC-04.27](../diagrams/visionclaw/04-handlers-graph-and-state.md) retains its historical stamp and now identifies the old verifier and post-debit response failure. Rendering siblings does not assess them.

VCS-10 / XR-01 remains **partially remediated**. Solid-proxy forwarded-URI compatibility, shared token-fixture execution with an explicit clock, deployment, restart/replica behaviour and durable operation idempotency remain open. The user's springs work was untouched. General lessons from this repair are incorporated in the [review-method update](2026-11-01-review-method-lessons.md).
