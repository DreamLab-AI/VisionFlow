---
title: Solid anchor recovery and endpoint identity
status: source-reviewed-open-findings
date: 2026-11-01
type: explanation
---

# Solid anchor recovery and endpoint identity

Recovery can publish altered trail metadata and discard the pending record before the next extension refuses that trail. A separate partial-save case leaves the public proof missing after recovery clears its intent. Endpoint reporting also has a concrete wiring error: startup selects the default from an unset state field, while request clients honour the environment override.

This batch assesses every section of [ADR-2007](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2007-mempool-single-url-no-fallback.md), relevant baseline endpoint/provenance clauses and four panels: SP-07.7/.11/.12/.20. Source remains Solid `e1d16adbfc096527f82ff16a6bce87ffd6d909f8`. [Sources](evidence/2026-11-01/solid-recovery/sources.json), the [probe](evidence/2026-11-01/solid-recovery/recovery-probe.rs), its [runner](evidence/2026-11-01/solid-recovery/run-probe.cjs) and [results](evidence/2026-11-01/solid-recovery/probe.json) bound the claims. No product code changed.

## R1 — startup can record a different endpoint from request clients

`AppState::new` sets `mempool_url` to `None` (`server/src/lib.rs:417`). Immediately after constructing that state, `main` calls `select_mempool_endpoint(state.mempool_url.as_deref())`, then logs that selection through a process-wide `Once` (`server/src/main.rs:318-328`). This expression does not read `JSS_PAY_MEMPOOL_URL`.

Request construction takes a different route: with no state override, `build_anchorer` uses `MempoolHttpClient::from_env` (`handlers/prov.rs:189-192`). That constructor resolves the environment, but its logging call encounters the same `Once`. The startup record can therefore remain the default public testnet4 selection while later requests use an explicitly configured endpoint.

A child process with a synthetic signet URL reproduces the discrepancy using the exact startup-selection expression over a fresh `AppState` and the actual request client constructor. Startup selection is defaulted testnet4; request selection is explicit signet. This is an expression/constructor probe, not a full server boot or captured production log. Source establishes the `Once` wiring; deployed logs were not inspected.

Acceptance: resolve one selection at startup, store it and construct every client from it. Log and expose that same value. Retain the explicit-versus-defaulted distinction. The ADR's description of per-request construction as merely a structural tidy-up misses this correctness problem.

## R2 — one configured URL does not pin the transport destination or prove transaction presence

`http_client` configures timeouts but no redirect restriction (`mempool.rs:87-94`). A local HTTP fixture at origin A returns 302 to origin B; B returns 200 with a body that is not a transaction. `transaction_exists` returns true. Both fixtures receive the expected request, while `selection().base_url` still names A.

This is two bounded observations: cross-origin redirects are followed, and presence accepts successful status without parsing or binding a transaction response (`mempool.rs:456-474`). Neither is an ordered failover list, so the narrow no-fallback decision remains implemented. They do weaken a stronger reading that requests stay on the recorded endpoint or that a successful response independently establishes the named transaction. The fixture demonstrates GET only; redirected broadcast behaviour was not exercised.

`infer_network` is also a URL heuristic, not a chain query (`mempool.rs:201-259`). The loopback mock is labelled regtest even though it implements no chain. A stored trail's network is compared with the requested network before a new advance; that guard does not compare either with an authenticated chain identity from the explorer.

Acceptance: make redirect policy explicit, distinguish configured and effective destinations when redirects are permitted, and validate the presence response expected from the chosen API. Treat inferred network as unverified until checked against configured chain identity. A chain identity check can detect configuration mistakes; by itself it does not make a dishonest explorer trustworthy.

## R3 — recovery checks transaction identity but not the proposed trail state

`verify_intent_tx` decodes the retained transaction, compares its txid and, when advancing an older stored trail, checks the first input against the stored head (`mempool.rs:1011-1033`). It does not establish that the intent's proposed trail states derive the transaction's outputs. Git-mark recovery then saves that proposed trail when its mark count is greater and clears the intent (`1275-1305`).

Using the production builder, the probe creates a genesis transaction in a synthetic transport, then drops the next broadcast. It changes the second state and its matching TXO URI commit in the persisted intent, leaving the signed raw transaction, txid and input untouched. A fresh filesystem backend and anchorer recover from the same files. Recovery resends the original raw bytes, saves and publishes the changed trail, and clears the intent. The normal verifier reports that the resulting trail is not intact. A later extension with the correct network is refused before any additional broadcast.

| Recovery fixture | Stored marks afterwards | Intent retained | Stored trail intact |
| --- | ---: | --- | --- |
| Original pending record | 2 | no | yes |
| Wrong txid | 1 | yes | yes |
| Altered state and matching URI commit | 2 | no | no |

Rejection of the wrong-txid control confirms that the existing identity guard works. The failure is narrower: proposed state is trusted too early. A deliberately mismatched network on the first recovery-triggering call stops a new advance after recovery, making its persistence effects observable separately. The subsequent correct-network call verifies that normal extension refuses the bad state, rather than spending from it.

This requires access to change a stored intent, or equivalent storage corruption. It does not demonstrate an unauthorised HTTP writer or a consensus-valid false commitment. The fixture decodes and records transactions; it does not implement Bitcoin signature or consensus validation. Production transaction construction, recovery, filesystem storage and verification execute unchanged. A fresh backend instance is not an operating-system restart.

Legacy git-mark intents may lack `rawHex`. That branch cannot run the raw-transaction identity check: a positive presence response can promote the proposed trail, while a negative response drops the intent (`mempool.rs:1275-1296`). This is source evidence; legacy migration was not freshly exercised.

Acceptance: validate the complete proposed trail transition against the retained transaction and existing trail before replacing durable state or publishing it. Keep the pending record when validation fails. Reuse the existing transaction/commitment verification primitives; do not add a second cryptographic implementation in a consumer. The equivalent MRC20 helper calls the same transaction-identity check, but altered MRC20 metadata was not freshly probed.

## R4 — partial public-proof writes are not repaired when the private trail already advanced

`save_gitmark_trail` first writes the private trail and then its public `blocktrails.json` (`trail_store.rs:354-376`). After a broadcast is accepted, a save failure is logged and the anchor is still returned successfully with no block height (`mempool.rs:1366-1387`). The intent remains available for recovery.

To force the public write to fail, the fixture makes its destination a directory. The private trail advances to two marks, the public write fails, and the returned anchor still reports success. After removing the directory and creating a fresh backend/anchorer, recovery sees that the private trail already has the same mark count as the intent. It skips `save_gitmark_trail` and clears the intent. The public proof remains absent. As in R3, the following network mismatch deliberately stops a new advance from incidentally republishing the proof.

Acceptance: track or idempotently repair both durable outputs before clearing the repair record. Preserve the distinction between broadcast acceptance and proof publication; do not refund an accepted transaction merely because publication failed. Existing tests cover failure of the private trail write, which leaves its count behind and does trigger recovery; this public-write case is different.

## R5 — epoch recovery has useful controls but is not an end-to-end durability guarantee

Existing tests pass for unreadable and malformed batches, concurrent pushes, retaining the commit that fills a failed batch, a failed reset with a saved completion marker, bounded backlog, and draining a full backlog. These use in-memory storage and fault/anchor fixtures. They do not establish durable restart, a real chain outcome or multi-process coordination.

Pending commits are preserved before anchoring, and the function tries to save a completion marker if the reset fails (`handlers/prov.rs:301-335`). If both reset and marker fail, it explicitly reports that a later push will anchor the batch again. The process-global path lock does not coordinate separate processes (`trail_store.rs:336-350`).

At four times the epoch threshold, a new commit is omitted while the older batch is still allowed to close (`handlers/prov.rs:283-297`). The passing `a_full_backlog_still_drains_once_anchoring_recovers` test confirms that the closed list equals the old backlog, excluding the triggering `more` commit. This recovers throughput for the retained backlog; it does not recover that omitted commit. No durable queue for omitted commits is present in this function. The [previous receipt finding](2026-11-01-solid-provenance-delivery.md) means the write response does not expose this difference.

`epoch_push_and_maybe_anchor` also discards the returned `BlockTrailAnchor` before returning `ClosedEpoch`; a successful reset removes the pending list. A durable exported batch proof and repair path remain unestablished. The corrected panel distinguishes accumulated, submitted, reset and failed-to-record outcomes; it no longer calls broadcast acceptance confirmed on-chain evidence.

## ADR-2007 section dispositions

| Section | Assessment |
| --- | --- |
| Front matter, lineage and review trigger | Accepted single-base selection is implemented. Historical live/complete axes and `e093e88` do not certify current logging or deployed chain identity. No LAN cutover was attempted. |
| Context | Sovereign explorer intent is a deployment requirement. The historical LAN address/reachability statement was not retested or adopted as current environment authority. |
| Decision | One configured base and public testnet4 default remain; no ordered fallback list or direct bitcoind RPC in the reviewed native client. Redirect following is a separate transport behaviour (R2). |
| Consequences | Public dependency remains possible through default selection. URL-based network labels are heuristics, and matching request/trail strings is not explorer chain attestation. RPC firewall posture and simultaneous-network deployment were not verified. |
| Verification | Current source and thirteen selection tests support default/explicit selection and 404-versus-ambiguous failures. A successful presence status is not parsed. Historical line anchors are superseded by this source manifest. |
| Closeout extension — 4 September | Empty/explicit selection, unavailability, unknown transaction and recovery fixtures are exercised. Startup manifest consistency fails (R1). Operator repair, persistent process restart and real deployment remain open. |
| Acceptance progress: implementation and logging | Selection types and JSON representation exist. `main` resolves from an unset field rather than the environment; one-time logging can preserve the wrong startup choice. Network inference is not observation of a live chain. |
| Acceptance progress: tests and receipts | Thirteen selection tests pass. This batch additionally runs twenty-eight mempool tests and thirteen provenance-handler tests; one live smoke test remains ignored. The historical 1,801-test workspace and Clippy results were not rerun. |
| Acceptance progress: remaining and governed paths | No HTTP manifest/health publication was established. Client construction has a correctness consequence, not just refactoring value. New local probes do not certify production logs, explorer availability, chain confirmation or endpoint cutover. Source ADR and product files are unchanged. |

## Comparison and user utility

[Esplora's API](https://github.com/Blockstream/esplora/blob/master/API.md) documents transaction data, confirmation status, raw transactions, inclusion proofs and block hashes by height. It is a useful reference for separating “HTTP responded”, “transaction found”, “confirmed” and “proof checked”. The estate already uses a compatible REST seam; recovery and operator reporting should retain those distinctions. A height-zero hash check is a possible configuration control, not independent validation of every explorer assertion.

[Reqwest's redirect policy documentation](https://docs.rs/reqwest/latest/reqwest/redirect/struct.Policy.html) exposes no-redirect and custom-policy options. That current documentation is a design reference; this batch's local fixture establishes the behaviour of the estate's locked 0.12.28 build. No claim depends on substituting the latest library for the installed version.

For an inspection handover, expose the configured endpoint, observed destination, verified chain identity, pending intent count and unpublished proof count. Give a repair action an idempotent result: recovered, still ambiguous, invalid retained record or publication pending. Keep those outcomes in the owning Rust crate and consume them in the estate's clients. Measure recovery with a process restart and storage faults at each persistence boundary, and verify exported proof on another machine before presenting it as complete evidence.

## Verification boundary

Fifty-four existing tests pass: [28 mempool](evidence/2026-11-01/solid-recovery/mempool-tests.txt), [13 provenance-handler](evidence/2026-11-01/solid-recovery/prov-tests.txt) and [13 endpoint-selection](evidence/2026-11-01/solid-recovery/selection-tests.txt). One live-network smoke test is ignored. Six new probe groups cover four recovery fixtures, local redirect/presence behaviour and startup/request selection. No live explorer, real transaction, chain confirmation, operating-system restart, production deployment or product remediation occurred. RuView remains excluded; VisionClaw springs remain untouched. ADR-2008 and the remaining payment/provenance surface remain open.
