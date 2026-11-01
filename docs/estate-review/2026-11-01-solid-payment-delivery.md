---
title: Solid payment authority and explicit provenance delivery
date: 2026-11-01
status: assessed
type: explanation
---

# Solid payment authority and explicit provenance delivery

Solid’s explicit provenance query exposes private metadata and returns sidecars from the wrong revision. Retrying an owner-authorised anchor charges again and advances the trail again. Separately, a canonical payment-state read outage can select stale compatibility mirrors; a subsequent payment mutation then overwrites the authoritative balance and loses pending intents. These are local reproductions against the current native server, not observations of a deployed service.

This batch assesses every section of **ADR-2008** and five existing panels: **SP-07.6/.10/.13/.15/.18**. The crypto port and its golden parity are implemented. The chain-derived ledger is not. The October ADR implementation note has fallen behind the narrower public API, bundled payment commits and consumer changes. The new [parallel-implementation audit](2026-11-01-parallel-implementations.md) adds a separate estate-wide consolidation queue; its source-only findings are not promoted to runtime evidence by this report.

## Evidence and limits

Native source is `solid-pod-rs e1d16adbfc096527f82ff16a6bce87ffd6d909f8`. The [source manifest](evidence/2026-11-01/solid-payments/sources.json) records hashes and revisions, including bounded consumer reads. No product source was changed in this batch. RuView remains excluded and VisionClaw springs were not touched.

The [probe source](evidence/2026-11-01/solid-payments/payment-probe.rs), [runner](evidence/2026-11-01/solid-payments/run-probe.cjs) and [results](evidence/2026-11-01/solid-payments/probe.json) exercise production `build_app` routes. Query and anchor fixtures use disposable FsBackend files, real Git commits and a known test identity. The loopback explorer decodes and records transactions but does **not** validate signatures or consensus. The state-read fixture uses MemoryBackend with an injected canonical-file read error and an explicitly seeded debit/intent snapshot. It exercises actual HTTP balance and sell handlers. No production funds, external explorer, deployed image, OS restart, multi-process writers or chain finality were tested.

Ninety-nine existing tests passed: [24 transaction-builder tests](evidence/2026-11-01/solid-payments/crypto-tests.txt), [50 MRC20 tests](evidence/2026-11-01/solid-payments/mrc20-tests.txt), [10 payment recovery tests](evidence/2026-11-01/solid-payments/recovery-tests.txt) and [15 HTTP/atomicity tests](evidence/2026-11-01/solid-payments/http-tests.txt). The first filtered invocation selected zero tests and is retained as [execution history](evidence/2026-11-01/solid-payments/server-tests-initial.txt), not counted. The corrected invocations ran the named suites. Git was available and the HTTP fixtures executed.

## Reproduced findings

### P1: A canonical read failure can restore a spent balance and erase recovery intents

`StoragePaymentStore::read_state_unchecked` at `handlers/pay.rs:179` falls back whenever `storage.get(state.json)` returns **any** error. The documented contract at `:120` limits fallback to an absent file. Legacy ledger/replay reads also turn all read errors into empty state (`:142`, `:160`). The fallback creates version zero and an empty intent list. `commit_state` at `:198` then writes that selection as the canonical document before updating compatibility mirrors best-effort.

A local probe seeds a canonical balance of 20, version 7 and a pending 80-unit debit intent, with a stale legacy balance of 100. Under an injected canonical read error, authenticated `/pay/.balance` returns **200 and 100**. A subsequent authenticated `/pay/.sell` succeeds and rewrites canonical state to **100, version 1, no pending intents**. The original debit and intent are fixture preconditions, not a crash reproduced in this run. Best-effort mirrors make their divergence a supported storage-failure possibility. A malformed canonical JSON document, by contrast, returns 500: parse errors do not take the fallback.

This breaks the authority boundary even in one process. The mutex cannot make the wrong source authoritative. Fix acceptance: only `NotFound` may enter a clearly defined migration/bootstrap path; all other read failures must return an error and leave both canonical and mirror bytes unchanged. Repeat with backend permission/I/O errors, stale mirrors, pending payouts and refunds. If genuine canonical deletion is recoverable, recovery must reconcile the durable log rather than silently create a new ledger from mirrors.

### P2: The provenance query bypasses private resource access controls

`handle_resolve` at `handlers/prov.rs:351` takes a pod/commit and application state, with no request identity or WAC check. It resolves the repository, chooses a content path and directly reads that resource's sidecar at `:391`. Registration at `:747` precedes the LDP catch-all.

In the filesystem fixture, an owner-only ACL makes anonymous GET of the resource and its sidecar return **401**. Anonymous GET of `/{pod}/_prov/{known_commit}` returns **200**, including resource path, commit metadata and sidecar text. This exposes provenance metadata, not the resource body. A caller needs a known or obtained commit ID; the endpoint does not enumerate the pod. Deployment reachability remains untested.

Fix acceptance: apply the established resource-read authorisation path before returning any commit or sidecar metadata, without inventing a second ACL evaluator. Test owner, authorised reader, stranger and anonymous callers for both current and historical revisions, including deleted resources and revoked ACLs.

### P3: Query results conflate current sidecars with historical commits

This handler's comment promises a sidecar matching the commit, but the body never compares revisions. Its `anchored` field is just `contains("bt:txid")` at `handlers/prov.rs:413`.

After two real writes, querying the first commit returns its Git metadata beside the second write's sidecar. Adding only a Turtle comment containing `bt:txid` changes `anchored` from false to true. The fixture edits its own sidecar directly; this proves the report's weak interpretation, not remote permission to forge sidecars. An owner-authorised anchor of the first commit later overwrites the current sidecar with the old commit's mark (`:555–582`).

Fix acceptance: bind a stored receipt to the full resolved commit and validate its structured anchor data. Preserve historical receipts separately from the resource's current mark. Querying an old revision must never borrow a new revision's anchor. A comment or unrelated predicate must not establish anchoring; confirmation is a separate observed state.

### P4: Logical anchor retries charge and broadcast again

`handle_anchor` resolves the owner, commit and trail, charges at `handlers/prov.rs:512`, then calls the anchorer at `:523`. Neither the body nor the handler carries a logical operation ID or looks up a completed receipt. NIP-98 event replay protection is an authentication control, not operation idempotency.

Two successful requests for the same commit, signed with distinct valid NIP-98 events, each return 200 and `charged_sats: 5`. The fixture records two different transaction IDs and the balance moves **100 → 90**. An old-commit retry also leaves the current resource sidecar describing that old revision. This run tests retries after successful responses; the retained-intent/pending-response retry path is source-reviewed but not freshly reproduced here.

Fix acceptance: persist an operation receipt keyed by owner, trail/network, full commit and an explicit retry identifier or documented deduplication rule. Bind request parameters to it. A timeout/retry must return the existing operation's pending/completed outcome without another charge or trail advance. If deliberate repeated anchoring is supported, make it a separately authorised new operation.

### P5: Accepted anchors can lack their promised receipt; failed refunds have no guaranteed recovery

After broadcast acceptance, a sidecar write failure is logged at `handlers/prov.rs:577`, then the handler returns 200 at `:584`. Replacing the sidecar destination with a directory reproduces **200, charged 5, no sidecar file and no sidecar-error field**. The final balance is 85 after this third request. The transaction is accepted only by the synthetic explorer; no confirmation is claimed.

Before-acceptance failures take the compensating refund path. Existing native tests cover a pre-broadcast refund and retained charges after an ambiguous broadcast. The source's refund-retention guarantee is overstated: if both refund and pending-debt write fail, `refund_or_retain` only logs the second error (`:692`). A scoped crate search found the pending path's writer and test, but no repayment consumer. The charge is not durably linked to an anchor operation before the external call. A crash between debit and intent persistence remains an untested reconciliation gap.

Fix acceptance: expose separate submission, confirmation and receipt-publication states; retain repairable receipt work without refunding a transaction merely because local publication failed. Record charge and operation identity durably before submission. Refund debt needs a resumable, deduplicated repayment path and an observable terminal state, including failure to persist that debt.

## ADR-2008, section by section

| Section or obligation | Current evidence and disposition |
| --- | --- |
| Metadata and lineage | Proposed / partial / staged remains the record's status. Its cited `e62d028` is historical evidence, not current deployment. This assessment does not ratify the decision or give whole-document credit to the Agentbox parent ADRs. |
| Context | The handwritten primitive implementation and TXO stand-in describe the pre-port problem. Current `bitcoin_tx.rs:58` imports rust-bitcoin/libsecp256k1; `mrc20.rs:391` uses the same arithmetic. The claim to own all estate transaction construction is historical and is not established estate-wide by this batch. |
| D1: primitive port | Implemented in the reviewed modules. Serialisation, sighash, signing, point/scalar operations and address encoding delegate to maintained libraries. `xonly_of` remains a wrapper (`bitcoin_tx.rs:130`), contrary to the old diagram's deletion claim. `mrc20.rs:416` handles high hashes by delegating modular addition to libsecp256k1; it is not the old handwritten subtraction implementation. |
| D2: unmodified golden parity | All three JSS golden cases pass. The fixture is byte-identical to `fc56b39`; [receipt](evidence/2026-11-01/solid-payments/crypto-and-consumer-receipts.json). Published BIP-340/341 vectors and MRC20 chained-key vectors also pass. No production-JSS nonce equality or chain-acceptance claim follows. |
| D3: retain anchors, retire token rail | Partial. Chained derivation is implemented and tested; `/pay/.buy` and `.withdraw` remain registered (`handlers/pay.rs:1761`). Their core stores a payout debit and intent **before** broadcast (`:1322–1344`), despite its older function comment saying debit happens afterwards. |
| D4: height-stamped chain view | Absent from the assessed ledger path. `get_balance` is an exact-key lookup of stored entries (`payments.rs:412`); the HTTP handler reads it without a node query (`handlers/pay.rs:404`). No chain height, bounded freshness or node-unreachable error exists on this path. The required 100-DID parity gate therefore cannot be claimed. |
| D5: remove arbitrary authority | Partial progress beyond the implementation note: bare `credit` and `debit` are now crate-private (`payments.rs:565`, `:580`). Public receipt mutations, `charge`/`refund` and authoritative `write_ledger` remain. `credit_by_outpoint` validates shape/replay, not chain facts; the caller must verify them. Passing the old public-function grep does not satisfy the chain-authority requirement. |
| D6: delete stand-in and MRC20 ledger credits | The stand-in is deleted; non-MRC20 bodies return 501, covered by `txo_deposit_is_gone_and_credits_nothing`. The second clause, removal of MRC20 ledger crediting, is still deferred. Current verified deposits credit the token unit and explicitly reject a ticker aliasing sats (`handlers/pay.rs:602`, `:672`). Thus “D6 done” is accurate only for the stand-in deletion. |
| D7: payment atomicity | Partly implemented: canonical state bundles ledger/replay/orders/exchange/intents, guarded locally; a concurrent-writer test passes. P1 demonstrates a remaining authority failure. Multi-process safety and crash reconciliation are not proven. The ADR's old README warning is no longer present; README now records remediation of dated findings. Neither historical wording should substitute for current evidence. |
| D8: consumers together | Not complete: VisionClaw lockfile resolves alpha.17, forum alpha.18. Their manifests and resolved versions are captured; neither consumer was rebuilt. Version equality alone would not prove equivalent features or deployed behaviour. |
| Consequences | Native transaction building remains cfg-gated out of WASM. Size, WASM mrc20 deployment, per-read chain fees/latency and the proposed chain-derived cache are not measured. `payments-and-web-ledger.md` still describes the legacy ledger path and public bare mutations. The order-book/AMM retirement tension is acknowledged in the ADR's later disposition; active native routes remain. No parent decision is silently reinterpreted here. |
| Verification | Golden and primitive-removal evidence is established. The old public credit/debit grep now passes but is insufficient. TXO refusal passes. Chain-view parity/unreachable-node tests and full atomicity closure remain unmet. |
| Disposition: 2026-10-02 | Its “not started” judgement is superseded by the following implementation section. Its consumer pins are historical. The proposed amendment about exchange policy needs coordinated owner/consumer disposition, not treating an earlier paragraph as a deployed contract. |
| Implementation: 2026-10-02 | D1/D2 remain supported, D3 partial, D4 absent. D5/D7 made later partial progress. D6 combines a completed deletion and deferred credit removal. The note's caller inventory is stale: Agentbox now returns 501 itself (`management-api/routes/payments.js:286`); VisionClaw retains its own 501 stub (`src/handlers/pay_handler.rs:471`); forum's adapter now delegates to `pay_ledger`, whose `deposit_txo_from_body` rejects `amount_sats` (`:522`). PAY_ENABLED remains false in the checked Worker config. These are bounded source checks, not fresh consumer runtime tests. |

## Comparisons and user utility

Teller's [pinned pure ledger implementation](https://raw.githubusercontent.com/solidpayorg/teller/7c00cea/lib/teller.mjs) keeps outpoint receipts for credits and IDs for debits/transfers; callers supply verified facts and storage. The native ledger shares this receipt pattern. That similarity does not make its persisted balances a chain-derived view. The useful next step is one shared fixture for receipt identity, units and replay across native and Worker consumers, as EST-08 in the parallel audit recommends.

[Stripe's idempotency contract](https://docs.stripe.com/api/idempotent_requests) makes retry identity explicit and checks reuse against request parameters. This is an API design comparison, not a payment-provider migration proposal. VisionFlow's operator needs an anchor button that can be retried after losing a connection without purchasing the same operation again. Acceptance should include a dropped response, replay with a fresh authentication token, changed parameters under the same operation ID and restart between charge and submission.

[BTCPay Server's invoice states](https://docs.btcpayserver.org/Users/invoices/) distinguish payment processing from settlement and tie fulfilment to settlement policy. The corresponding product improvement here is an independently observable operation: charged, submitted, confirmed, receipt published or refund owed. A successful JSON response and a `bt:txid` substring are insufficient status evidence. For an industrial evidence export, the buyer should receive a revision-bound proof and a clear verification result without gaining access to another customer's private metadata.

These comparisons place this path as a programmable data/provenance service with local payment accounting. They do not establish a production payment processor or the proposed sovereign chain ledger. Prioritise private-query isolation and authoritative-read failure handling, then durable operation receipts, then shared contract fixtures and coordinated consumers. Do not make a new projection another authority beside the existing stores.

## Diagram and reconciliation boundary

Five selected panels were corrected, rendered and visually checked. All nineteen topic diagrams render, but the five remaining unassessed siblings receive no credit. The nine previously assessed panels are byte-identical to the last checkpoint; their whole-file hashes are rebound with no new credit. Declared source revision `09ec90f` is retained only because every declared source file matches current source bytes; [parity receipt](evidence/2026-11-01/solid-payments/revision-parity.json). The old SP-07.6 register-line advisory is corrected by symbol lookup.

Estate audit work remains in progress. These findings and the parallel report expand the remediation queue; they do not close whole-repository security, deployment, market comparison or source coverage.
