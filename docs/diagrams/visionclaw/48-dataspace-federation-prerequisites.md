---
id: VC-48
title: Proposed data-space federation and current release connection gates
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2161-a-node-publishes-data-spaces-that-peer-nodes-render-as-linked-volumes.md
adrs: [ADR-2021, ADR-2159, ADR-2161, ADR-2162]
sources:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/DATA-authority-erasure.md
  - ../project/docs/adr/ADR-2161-a-node-publishes-data-spaces-that-peer-nodes-render-as-linked-volumes.md
  - ../project/src/uri/mod.rs
  - ../project/src/handlers/socket_flow_handler/http_handler.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/crates/visionclaw-domain/src/utils/visibility_filter.rs
  - ../project/xr-client/rust/src/transport.rs
  - ../project/xr-client/scripts/backend_connection.gd
verified_commit: {visionclaw: 75b72416bac176ca8fde715cac3bd7c77657b3e7}
---

## For developers

The [audit](../../estate-review/2026-10-31-portals-and-federation.md) traces current upgrade and identifier code against ADR-2161. Proposed nodes below are requirements, not implemented services; no live federation session was run.

## For the business

Local portals and concurrent graph connections do not provide cross-node data sharing. Federation still needs authenticated release connections, declared projections, permission checks, replay protection and erasure handling.

## VC-48.1 Current XR authentication occurs after the release upgrade gates

```mermaid
flowchart TD
    DIAL["BackendConnection passes URL and signing secret<br/>backend_connection.gd:316,322"] --> URL["Transport opens socket with URL string<br/>transport.rs:95"]
    URL --> ORIGIN["Server requires Origin in release<br/>http_handler.rs:67,129,131,139"]
    ORIGIN --> AUTH["Verify signed subprotocol or legacy bearer at upgrade<br/>http_handler.rs:146,151,181,183"]
    AUTH --> DENY["Unauthenticated release upgrade refused<br/>http_handler.rs:314,321,322"]
    URL --> AFTER["Only after upgrade: request topology then send authenticate text<br/>transport.rs:99,115,118"]
```

**What it shows.** A secret used for the post-upgrade text message cannot satisfy earlier release Origin and authentication checks. **Why it is this way.** The client and server implement different handshake phases. This source trace does not test a deployed proxy or permissive development mode.

**Tension:** ADR-2161's Origin prerequisite also needs upgrade-time authentication alignment (`transport.rs:95,115`).

## VC-48.2 Existing visibility and identifiers are narrower than a dataspace contract

```mermaid
flowchart TD
    FLAG["Visibility flag defaults ON<br/>explicit false tokens opt out<br/>visibility_filter.rs:26,30,32,38"] --> FILTER["Initial load filters by public or caller owner key<br/>types.rs:258,261,270,276"]
    KINDS["Current typed URN kind set has no Dataspace arm<br/>uri/mod.rs:54,86,94"] --> SHARED["Compile shared agentbox crossing-kind artefact<br/>uri/mod.rs:665"]
    SPEC["PROPOSED: viewer bound to one dataspace with selector and ACL<br/>PROTOCOL-registry.md:352"] --> RECHECK["PROPOSED: re-evaluate and close access-losing connections<br/>PROTOCOL-registry.md:352"]
```

**What it shows.** Eighteen production visibility tests pass; the actual bridge refuses the proposed dataspace kind. Neither result proves the proposed connection mode. **Why it is this way.** Existing owner/public filtering and closed identifier translation predate dataspace selectors, ACLs and revocation.

## VC-48.3 Proposed reference lifetimes distinguish authored data from received copies

```mermaid
flowchart TD
    OWN["PROPOSED: authored link is primary local data<br/>DATA-authority-erasure.md:158"] --> RENEW["Seven-day lease means re-announce, not discard local authorship<br/>DATA-authority-erasure.md:158,162"]
    FOREIGN["PROPOSED: received link or viewer state is disposable<br/>DATA-authority-erasure.md:159"] --> RETRACT["Authoritative retraction removes display and tombstones links<br/>DATA-authority-erasure.md:161,162"]
    RETRACT --> PURGE["Purge by earlier of receipt plus 24 hours or lease expiry<br/>DATA-authority-erasure.md:162"]
    FOREIGN --> EXPIRE["Offline received references expire after seven days<br/>tombstones survive eight days<br/>DATA-authority-erasure.md:162"]
```

**What it shows.** The design has distinct ownership and expiry rules; the current resolver has no 410/tombstone implementation. **Why it is this way.** A received reference is reconstructible, while an authored assertion is primary data. This diagram records the proposal without asserting that its deadlines run today.
