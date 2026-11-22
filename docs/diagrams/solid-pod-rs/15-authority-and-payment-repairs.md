---
id: SP-15
title: Resource authority and payment state after the external review
area: solid-pod-rs
governing: []
adrs: [ADR-2004, ADR-2005, ADR-2006, ADR-2008, ADR-2010]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/parser.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/trading.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/replay_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/prov.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/pay.rs
verified_commit: a28a74832dd0a6e47a531e34e4adc46a3946dfd3
---

## For developers

This is the scoped successor to SP-04.10, SP-07.6, SP-07.15 and SP-14.1–2. It records the repaired authority and storage contracts, including the additional payment issues found during review. Other panels retain their historical stamps; this topic does not re-verify them.

## For the business

A public folder grant no longer opens an entire pod's private history. Historical provenance obeys the resource's read policy, and a storage failure cannot silently substitute an older balance. These are source and native-test results; consumers and deployed services need their own adoption receipts.

## SP-15.1 A malformed condition cannot disappear

```mermaid
flowchart TB
    CONDITION["Turtle condition object<br/>parser.rs:443"]
    BROKEN["unbalanced object or invalid parameters<br/>parser.rs:485<br/>parser.rs:490"]
    COST["payment cost must parse<br/>parser.rs:610"]
    DENY["return ACL parse failure<br/>restriction is not omitted"]
    SUBJECT["combine same-subject statements<br/>parser.rs:81"] --> CONDITION
    SCHEME["normalise ACL predicates<br/>parser.rs:232"] --> CONDITION
    SCHEME --> NONIRI["rule predicate with a literal, number, boolean,<br/>blank node or collection object (SP-PAR-265)<br/>parser.rs:240<br/>parser.rs:315"] --> DENY
    CONDITION --> BROKEN --> DENY
    CONDITION --> COST --> DENY
```

**What it shows:** malformed conditions return errors instead of an empty condition list or a zero-cost replacement. Valid typed conditions remain supported. **Why it is this way:** ADR-2005's fail-closed rule applies before the evaluator, including incomplete Turtle condition objects.

**Invariant:** an absent or invalid payment amount cannot default to a free condition (`parser.rs:610`).

## SP-15.2 Whole-history Git export requires the pod owner

```mermaid
flowchart TB
    READ["current root Read check<br/>solid-pod-rs-server/src/lib.rs:5969"]
    OWNER["authenticated principal equals pod key<br/>solid-pod-rs-server/src/lib.rs:5992"]
    DENY["delegated or anonymous history export refused<br/>solid-pod-rs-server/src/lib.rs:5999"]
    SERVICE["service root is data_root<br/>slug extracted once<br/>solid-pod-rs-server/src/lib.rs:6003"]
    READ --> OWNER
    OWNER -->|owner| SERVICE
    OWNER -->|other caller| DENY
```

**What it shows:** current resource permission and whole-history custody are separate checks. The service receives the data root rather than an already-prefixed pod directory. **Why it is this way:** root Read alone did not authorise private children or earlier versions. ADR-2010 now defines this route as an owner export/recovery surface; delegated readers use LDP. Alias pods without a cryptographic owner binding gain no export privilege.

**Invariant:** an anonymous public-root Read grant does not satisfy the owner gate (`solid-pod-rs-server/src/lib.rs:5992`).

## SP-15.3 Provenance inherits resource authority and commit identity

```mermaid
flowchart TB
    AUTH["resolve caller identity<br/>prov.rs:394"]
    WAC["Read check on resolved resource<br/>prov.rs:397"]
    SIDECAR["only one canonical activity<br/>matching requested commit<br/>prov.rs:416"]
    CACHE["normal audience cache policy<br/>prov.rs:442"]
    AUTH --> WAC --> SIDECAR --> CACHE
```

**What it shows:** private-resource provenance is gated before the response. A current sidecar for another commit is omitted from a historical result. **Why it is this way:** opt-in provenance under ADR-2004 never implied public disclosure. The sidecar match accepts the server's canonical generated shape; it does not certify arbitrary authored Turtle as a historical mark.

**Invariant:** provenance resolves the same resource Read contract as ordinary reads (`prov.rs:397`).

## SP-15.4 Authoritative payment state, bounds and custody

```mermaid
flowchart TB
    STATE["canonical state read<br/>only NotFound selects legacy migration<br/>pay.rs:193<br/>pay.rs:198"]
    BALANCE["WAC balance uses PaymentStore<br/>solid-pod-rs-server/src/lib.rs:1091"]
    BOUNDS["fee below 10000 bps<br/>checked token price and swap product<br/>pay.rs:1160<br/>pay.rs:1543<br/>pay.rs:1640<br/>trading.rs:497"]
    INTENT["intent copy clears issuer secret<br/>pay.rs:220"]
    RECOVER["matching custody identity supplies key<br/>pay.rs:290"]
    STATE --> BALANCE
    STATE --> BOUNDS --> INTENT --> RECOVER
```

**What it shows:** failed reads propagate, WAC and payment routes share the canonical ledger, and arithmetic failures are refused. Recovery snapshots carry public state; the issuer key comes from the matching trail store. **Why it is this way:** ADR-2008's canonical state cannot coexist with an authoritative best-effort mirror. The security review also exposed unchecked fee/price arithmetic and unnecessary secret duplication.

**Debt:** old intent copies are scrubbed on a subsequent state commit; this does not erase backups or rotate a previously exported key (`pay.rs:220`).

The shared replay error also has an explicit unavailable-store refusal for durable consumers (`replay_store.rs:80`); this producer declaration does not establish an edge deployment.

## SP-15.5 Payment grants and receipts share one ledger authority

```mermaid
flowchart TB
    DID["authenticated deposit beneficiary"] --> TWEAK["require DID-tweaked target address<br/>pay.rs:620"]
    TWEAK --> CREDIT["proof validation then canonical credit"]
    WAC["granted paid access"] --> CHARGE["shared-lock canonical charge and commit<br/>pay.rs:137<br/>lock_payment_state: in-process then backend lock<br/>solid-pod-rs-server/src/lib.rs:225"]
    CHARGE --> LEDGER["same canonical ledger used by payment routes"]
```

**What it shows:** a public proof cannot credit another principal, and WAC no longer charges a best-effort mirror. **Why it is this way:** the broader review found both receipt ownership and debit authority diverged from the ledger reader. Existing generic-address funds need an explicit migration decision.

**Invariant:** a WAC charge mutates canonical state under the payment lock (`pay.rs:137`).
