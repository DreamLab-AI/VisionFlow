---
id: AB-40
title: Relay signature rejection and governance dispatch
area: agentbox
governing: [../project/agentbox/docs/GOVERNANCE-capabilities.md]
adrs: [ADR-2009]
sources:
  - ../project/agentbox/mcp/nostr-bridge/relay-consumer.js
  - ../project/agentbox/management-api/lib/authority-consumer.js
  - ../project/agentbox/tests/sovereign/relay-signature-failure.test.cjs
verified_commit: 720e0d89656a4360998b3a716d0f3ee8cac66f93
---

## For developers

This topic records the ABI-05 verification repair. Both the forum-governance and local inbound paths call _verifySig before dispatch. A verifier exception now rejects the event. The existing independent authority consumer already rejects verification exceptions at agentbox/management-api/lib/authority-consumer.js:159; consolidating rosters and ingress remains open.

## For the business

An unavailable signature checker must not turn an unverified decision into permission to act. This repair rejects those decisions. It is merged source, awaiting deployment; it does not certify a running container or complete the wider governance migration.

## AB-40.1 Verification must succeed before a decision has effects

```mermaid
flowchart TD
    FORUM["forum decision kind 31403<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:446"]
    INBOUND["local inbound event<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:546"]
    VERIFY["_verifySig uses injected verifier or nostr-tools<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:743"]
    EXACT{"result is exactly true?<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:748"}
    ERROR["load or verification exception: log once, return false<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:749"]
    REJECT["reject before side effects, increment rejection metric<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:450<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:549"]
    AUTH["forum: signer roster and deduplication<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:454"]
    EFFECT["record event, notify waiter and dispatch orchestrator<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:479<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:487"]
    LOCAL["local: ingress policy, recipient and deduplication<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:555"]
    FORUM --> VERIFY
    INBOUND --> VERIFY
    VERIFY --> EXACT
    VERIFY -- exception --> ERROR --> REJECT
    EXACT -- no --> REJECT
    EXACT -- forum yes --> AUTH --> EFFECT
    EXACT -- local yes --> LOCAL
```

- **What it shows:** explicit signature success precedes other gates and side effects. **Why it is this way:** dependency and verifier failures must reject, including injected test verifiers. **Open:** ABI-05 consumer/roster consolidation and deployment remain unverified; five regression tests include genuine signing and altered content at agentbox/tests/sovereign/relay-signature-failure.test.cjs:43.
