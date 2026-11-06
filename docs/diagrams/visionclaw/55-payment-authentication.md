---
id: VC-55
title: Payment request binding and single-use authentication
area: visionclaw
governing: [../project/docs/SECURITY-profiles.md]
adrs: [ADR-2002, ADR-2111]
sources:
  - ../project/src/handlers/pay_handler.rs
  - ../project/src/utils/auth.rs
  - ../project/src/utils/nip98.rs
  - ../project/tests/pay_nip98_test.rs
verified_commit: {visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d}
---

## For developers

Payment routes now share the host NIP-98 verifier and URL reconstruction. Five JSON POST handlers bind the transmitted body with Required, while GET handlers use NoBody. Query/body rejection leaves the token unspent, successful authentication spends it even if the operation fails. The tests exercise these boundaries through the actual routes; this is source verification, awaiting deployment.

## For the business

One signed request cannot charge twice within this process. A freshly signed retry can still repeat an operation, and the cache does not span restarts or replicas. The host payment store remains live source pending ADR-2111's proposed wallet migration. Funding, settlement, resource delivery and recovery are separate acceptance gates.

## VC-55.1 Authenticate the received request before payment effects

```mermaid
flowchart TD
    ROUTE["authenticated payment routes<br/>src/handlers/pay_handler.rs:900"]
    BODY["POST: retain JSON bytes, validate content type and shape<br/>2 MiB body limit<br/>src/handlers/pay_handler.rs:429<br/>src/handlers/pay_handler.rs:904"]
    GET["GET: NoBody policy<br/>src/handlers/pay_handler.rs:416"]
    URL["shared forwarded scheme/host plus path and query<br/>src/utils/auth.rs:170"]
    VERIFY["signature, freshness, method, URL and body policy<br/>src/utils/nip98.rs:418"]
    CLAIM["claim signed id under mutex, bounded process-local cache<br/>src/utils/nip98.rs:227<br/>src/utils/nip98.rs:295"]
    REJECT["authentication rejected before payment effects<br/>src/handlers/pay_handler.rs:418"]
    EFFECT["POST operation<br/>sell, swap, pool swap, liquidity or estimate<br/>src/handlers/pay_handler.rs:613<br/>src/handlers/pay_handler.rs:686<br/>src/handlers/pay_handler.rs:776<br/>src/handlers/pay_handler.rs:816<br/>src/handlers/pay_handler.rs:862"]
    READ["balance, offers or pool information<br/>src/handlers/pay_handler.rs:477<br/>src/handlers/pay_handler.rs:640<br/>src/handlers/pay_handler.rs:734"]
    CHARGE["resource fee: durable ledger debit<br/>src/handlers/pay_handler.rs:303"]
    RESPONSE["parse mixed-case header names without panic<br/>src/handlers/pay_handler.rs:571"]
    ROUTE --> BODY
    ROUTE --> GET
    BODY --> URL
    GET --> URL
    URL --> VERIFY
    VERIFY -->|invalid| REJECT
    VERIFY -->|valid| CLAIM
    CLAIM -->|replay or capacity| REJECT
    CLAIM -->|admitted POST| EFFECT
    CLAIM -->|read-only GET| READ
    CLAIM -->|resource GET| CHARGE
    CHARGE --> RESPONSE
```

Six route regressions and 46 shared-verifier tests pass. Separate storage commits, permissive compatibility URL matching and fresh-token operation retries remain open.

## VC-55.2 The replay window is one process's memory, and URL binding is looser than the signed URL

```mermaid
flowchart TD
    AGE["created_at within plus or minus 60 s of the server clock<br/>src/utils/nip98.rs:169"]
    TTL["claimed ids kept 120 s, twice the window<br/>src/utils/nip98.rs:178"]
    STORE["one static Mutex map per process, never shared or persisted<br/>src/utils/nip98.rs:221"]
    CAP["100000 ids overall, 1000 per signer, full cache refuses new auth<br/>src/utils/nip98.rs:198 src/utils/nip98.rs:215 src/utils/nip98.rs:323"]
    MAP["pay handler logs every verifier error and returns None, so a full cache reads as 401<br/>src/handlers/pay_handler.rs:418-423"]
    URL["urls_match: host compared only when both URLs are absolute<br/>src/utils/nip98.rs:697"]
    API["then a path match with or without a leading /api, both directions<br/>src/utils/nip98.rs:711 src/utils/nip98.rs:718"]
    OPEN1["OPEN: a restart or a second replica forgets every claimed id, so a token inside its 60 s window is spendable once more there"]
    DEBT1["DEBT: ReplayCacheFull exists so a caller can answer 503, no payment caller does<br/>src/utils/nip98.rs:187-193"]
    AGE --> TTL --> STORE --> CAP --> MAP
    STORE --- OPEN1
    MAP --- DEBT1
    URL --> API
```

**What it shows.** Single use holds only within one process for 120 s, and a full cache is indistinguishable from a bad signature at the payment boundary.
**Open (risk):** A relative `u` tag skips the host comparison (`src/utils/nip98.rs:697`), and the `/api` strip lets one signature match both `/api/pay/...` and `/pay/...` (`src/utils/nip98.rs:711`).
**Debt (loose end):** The capacity error is collapsed into the generic auth failure at `src/handlers/pay_handler.rs:420`, so the 503 mapping the verifier documents at `src/utils/nip98.rs:193` is unreachable for payments.
