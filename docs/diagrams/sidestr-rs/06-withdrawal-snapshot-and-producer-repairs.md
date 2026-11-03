---
id: SR-06
title: Withdrawal multiplicity, code commitments and producer failure scope
area: sidestr-rs
governing: []
adrs: [ADR-0002]
sources:
  - ../sidestr-rs/sidestr-evm/src/state.rs
  - ../sidestr-rs/sidestr-evm/src/world.rs
  - ../sidestr-rs/sidestr-evm/src/rule.rs
  - ../sidestr-rs/sidestr-core/src/error.rs
  - ../sidestr-rs/sidestr-core/src/state.rs
verified_commit: 7f04644411220ba0d3b9c7bdba58bbf56be790fa
---

## For developers

This scoped successor records the repaired SR-03.1 withdrawal and snapshot cases and SR-02.4/SR-03.3 producer eviction boundary. It does not close the remaining Hitch or snapshot-metadata findings.

## For the business

One payment output cannot satisfy two withdrawals, restored contract code must match its commitment, and a bad block clock no longer discards valid queued transactions. Consensus consumers still need coordinated adoption; these source changes do not restart a chain.

## SR-06.1 One output satisfies at most one withdrawal

```mermaid
flowchart TB
    OBLIGATION["next script and amount obligation<br/>sidestr-evm/src/state.rs:515"]
    MATCH["find an unused exact output<br/>sidestr-evm/src/state.rs:519"]
    CONSUME["mark matched output consumed<br/>sidestr-evm/src/state.rs:521"]
    NEXT["continue with next obligation"]
    OBLIGATION --> MATCH --> CONSUME --> NEXT
    NEXT --> OBLIGATION
```

**What it shows:** the validator tracks output multiplicity instead of independently reusing an output for every equal obligation. **Why it is this way:** the earlier predicate allowed duplicate withdrawals to share one payment while the allowance summed both. ADR-0002's optional execution boundary does not excuse an inconsistent payout rule.

**Invariant:** a consumed output cannot satisfy another obligation (`sidestr-evm/src/state.rs:519`).

## SR-06.2 Restore validates executable bytes before state mutation

```mermaid
flowchart TB
    SNAPSHOT["visit every retained world<br/>sidestr-evm/src/state.rs:318"]
    HASH["Keccak bytes must match map key<br/>world.rs:77"]
    PRESENT["non-empty account code must exist<br/>world.rs:84"]
    RESTORE["then install checked snapshot state"]
    SNAPSHOT --> HASH --> PRESENT --> RESTORE
```

**What it shows:** checking an account-trie root alone is insufficient when executable bytes are stored in a separate hash map. **Why it is this way:** every retained world may be selected on rewind. Validation therefore covers the entire snapshot's code maps, not only the selected height.

**Invariant:** code validation precedes snapshot installation (`sidestr-evm/src/state.rs:318`).

## SR-06.3 Header refusal preserves the mempool

```mermaid
flowchart TB
    FAILURE["candidate block failed"]
    HEADER["shared header-rejection classifier<br/>error.rs:104"]
    CORE["core returns without eviction<br/>sidestr-core/src/state.rs:870"]
    EVM["EVM producer uses same classifier<br/>rule.rs:336"]
    FAILURE --> HEADER
    HEADER --> CORE
    HEADER --> EVM
```

**What it shows:** a Bitcoin or Knots header failure is distinct from transaction invalidity. **Why it is this way:** dropping transactions cannot repair a clock or header error. Genuine transaction-rule eviction remains tested and supported.

**Invariant:** header classification is shared by core and the EVM producer (`error.rs:104`).
