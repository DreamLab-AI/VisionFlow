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
verified_commit: 59b1aca1cc3541b767872ecbc161c89e65d6ebca
---

## For developers

This topic records the ABI-05 verification repair. Both the forum-governance and local inbound paths call _verifySig before dispatch. A verifier exception now rejects the event. The existing independent authority consumer already rejects verification exceptions at agentbox/management-api/lib/authority-consumer.js:165; consolidating rosters and ingress remains open.

## For the business

An unavailable signature checker must not turn an unverified decision into permission to act. This repair rejects those decisions. It is merged source, awaiting deployment; it does not certify a running container or complete the wider governance migration.

## AB-40.1 Verification must succeed before a decision has effects

```mermaid
flowchart TD
    FORUM["forum decision kind 31403<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:453"]
    INBOUND["local inbound event<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:553"]
    VERIFY["_verifySig uses injected verifier or nostr-tools<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:750"]
    EXACT{"result is exactly true?<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:755"}
    ERROR["load or verification exception: log once, return false<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:756"]
    REJECT["reject before side effects, increment rejection metric<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:457<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:556"]
    AUTH["forum: signer roster and deduplication<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:461"]
    EFFECT["record event, notify waiter and dispatch orchestrator<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:486<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:494"]
    LOCAL["local: ingress policy, recipient and deduplication<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:562<br/>allowlist policy now fails closed: an empty allowlist admits nobody<br/>agentbox/mcp/nostr-bridge/relay-consumer.js:775"]
    FORUM --> VERIFY
    INBOUND --> VERIFY
    VERIFY --> EXACT
    VERIFY -- exception --> ERROR --> REJECT
    EXACT -- no --> REJECT
    EXACT -- forum yes --> AUTH --> EFFECT
    EXACT -- local yes --> LOCAL
```

- **What it shows:** explicit signature success precedes other gates and side effects. **Why it is this way:** dependency and verifier failures must reject, including injected test verifiers. **Open:** ABI-05 consumer/roster consolidation and deployment remain unverified; five regression tests include genuine signing and altered content at agentbox/tests/sovereign/relay-signature-failure.test.cjs:43.
