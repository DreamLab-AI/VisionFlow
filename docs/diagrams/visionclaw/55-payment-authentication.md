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
verified_commit: 5fa0e1325a2f903c175cfe67bbc6d644fcf38267
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
