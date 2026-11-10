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
  - ../project/src/services/corpus_history.rs
  - ../project/src/actors/agent_demo_director.rs
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The [audit](../../estate-review/2026-10-31-portals-and-federation.md) traces current upgrade and identifier code against ADR-2161. Proposed nodes below are requirements, not implemented services; no live federation session was run.

## For the business

Local portals and concurrent graph connections do not provide cross-node data sharing. Federation still needs authenticated release connections, declared projections, permission checks, replay protection and erasure handling.

## VC-48.1 The XR client now signs the release upgrade, with a first-message fallback for Codescape

```mermaid
flowchart TD
    DIAL["BackendConnection dials with URL and signing secret<br/>backend_connection.gd:350 backend_connection.gd:356"] --> REQ["graph_upgrade_request: Origin is the target's own scheme and authority<br/>transport.rs:49 transport.rs:63 transport.rs:70"]
    REQ --> CAR["NIP-98 for the http(s) form of the socket URL, carried as nostr.base64url in Sec-WebSocket-Protocol, bare visionclaw when no secret<br/>transport.rs:71-83"]
    CAR --> CONN["connect with the signed request<br/>transport.rs:139 transport.rs:147"]
    CONN --> ORIGIN["Server requires Origin in release<br/>http_handler.rs:67,129,131,139"]
    ORIGIN --> AUTH["Verify signed subprotocol or legacy bearer at upgrade<br/>http_handler.rs:146,151,181,183"]
    AUTH --> DENY["Unauthenticated release upgrade refused<br/>http_handler.rs:314,321,322"]
    CONN --> FB["no subprotocol selected and a secret set: send first-message authenticate, the Codescape contract<br/>transport.rs:165-170"]
```

**What it shows.** The upgrade-phase mismatch the earlier stamp recorded is closed: a configured identity now signs the HTTP upgrade itself, and the post-upgrade text message survives only for a server that selects no subprotocol. **Why it is this way.** One signed carrier for browsers and the headset (`transport.rs:48`); dataspace viewers must still bind one dataspace, which nothing implements.
**Open (risk):** The native client writes its own `Origin` from the target URL (`transport.rs:63`), so the release Origin gate (`http_handler.rs:139`) separates browsers only; the signed carrier is the actual boundary for non-browser peers.

## VC-48.2 One visibility projection, still narrower than a dataspace contract

```mermaid
flowchart TD
    FLAG["Visibility flag defaults ON, absent or unrecognised is ON, explicit false tokens opt out<br/>visibility_filter.rs:29 visibility_filter.rs:34-40"] --> OF["of_node reads public and owner_pubkey, the stated single reading<br/>visibility_filter.rs:97-101"]
    OF --> SUB["visible_subgraph: visible nodes, edges with both ends visible, metadata rows of visible nodes, hidden page names scrubbed, now also wiki links to hidden pages inside definitions<br/>visibility_filter.rs:171 visibility_filter.rs:302 visibility_filter.rs:367"]
    SUB --> WSS["initial_graph_load_projection: visible_subgraph for the caller, then the top-N by quality<br/>types.rs:492 types.rs:509"]
    KINDS["Current typed URN kind set has no Dataspace arm<br/>uri/mod.rs:54,86,94"] --> SHARED["Compile shared agentbox crossing-kind artefact<br/>uri/mod.rs:665"]
    SPEC["PROPOSED: viewer bound to one dataspace with selector and ACL<br/>PROTOCOL-registry.md:392"] --> RECHECK["PROPOSED: re-evaluate at least every 10 s and close access-losing connections<br/>PROTOCOL-registry.md:392"]
    SUB -.->|"PROPOSED: selector and acl intersect after this"| SPEC
```

**What it shows.** Owner/public filtering is now one function shared by REST reads and `/wss`; the identifier bridge still refuses a dataspace kind. **Why it is this way.** ADR-060 Phase 4 unified the gate; the earlier comment-versus-parser drift (OFF vs ON) is gone, `types.rs:184` now says default ON.
**Debt (parallel):** the demo director still re-reads the `public` key itself (`agent_demo_director.rs:500-504`), although `visibility_filter.rs:98-100` calls `of_node` the only reading on every wire path. The two copies agree today. `public_subgraph` was the other copy and now goes through `of_node` (`corpus_history.rs:140`, `corpus_history.rs:144`).

## VC-48.3 Proposed reference lifetimes distinguish authored data from received copies

```mermaid
flowchart TD
    OWN["PROPOSED: authored link is primary local data<br/>DATA-authority-erasure.md:179"] --> RENEW["Seven-day lease means re-announce, not discard local authorship<br/>DATA-authority-erasure.md:179,183"]
    FOREIGN["PROPOSED: received link or viewer state is disposable<br/>DATA-authority-erasure.md:180"] --> RETRACT["Authoritative retraction removes display and tombstones links<br/>DATA-authority-erasure.md:182,183"]
    RETRACT --> PURGE["Purge by earlier of receipt plus 24 hours or lease expiry<br/>DATA-authority-erasure.md:183"]
    FOREIGN --> EXPIRE["Offline received references expire after seven days<br/>tombstones survive eight days<br/>DATA-authority-erasure.md:183"]
```

**What it shows.** The design has distinct ownership and expiry rules; the current resolver has no 410/tombstone implementation. **Why it is this way.** A received reference is reconstructible, while an authored assertion is primary data. This diagram records the proposal without asserting that its deadlines run today.
**Open (risk):** graph shares are a durable copy of the creator's view, private nodes included, and no per-user erasure path reaches them (`DATA-authority-erasure.md:114-115`, `DATA-authority-erasure.md:129-131`). A federated render-only rule that forbids durable copies (`DATA-authority-erasure.md:182`) would need shares carved out or covered.
