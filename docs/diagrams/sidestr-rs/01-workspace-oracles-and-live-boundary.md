---
id: SR-01
title: sidestr-rs workspace, reference-oracle ladder, and live estate boundary
area: sidestr-rs
governing:
  - ../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md
  - ../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
adrs: [ADR-2012, ADR-2096, ADR-2112]
sources:
  - ../sidestr-rs/sidestr-core/src/lib.rs
  - ../sidestr-rs/sidestr-core/src/overlays.rs
  - ../project/agentbox/config/sidechain/run-producer.sh
  - ../project/agentbox/config/sidechain/run-faucet.sh
  - ../project/agentbox/config/sidechain/dreamlab-txbt4/chain.json
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/lib/sidestr-agent.nix
  - ../sidestr-rs/sidestr-core/Cargo.toml
  - ../sidestr-rs/sidestr-header/Cargo.toml
  - ../sidestr-rs/sidestr-wallet/Cargo.toml
  - ../sidestr-rs/sidestr-nostr/Cargo.toml
  - ../sidestr-rs/sidestr-round/Cargo.toml
  - ../sidestr-rs/sidestr-agent/Cargo.toml
  - ../sidestr-rs/sidestr-hitch/Cargo.toml
  - ../sidestr-rs/webledgers-teller/Cargo.toml
  - ../sidestr-rs/sidestr-evm/Cargo.toml
  - ../sidestr-rs/sidestr-reserve/Cargo.toml
  - ../sidestr-rs/sidestr-bridge-liquid/Cargo.toml
  - ../sidestr-rs/sidestr-round/tests/support/interop.rs
  - ../sidestr-rs/sidestr-core/tests/pegtweak.rs
  - ../sidestr-rs/Cargo.toml
  - ../sidestr-rs/README.md
  - ../sidestr-rs/.github/workflows/ci.yml
  - ../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md
  - ../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md
  - docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md
  - ../project/agentbox/config/sidechain/README.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
  - ../project/docs/archive/TODO-unified-2026-09-07-to-2026-11-15.md
verified_commit: {sidestr-rs: 17da8b8e5b2c2decf94551ecb4ea47c732669214, visionflow: 3ae784ed0b547b562e1f94d017996f8ad197f5cb, visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e, agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca}
---

## For developers

`sidestr-rs` now has eleven workspace members. Eight publishable crates carry the reusable chain, wallet, Nostr, federation, agent, Hitch and teller surfaces; EVM and the two reserve crates remain separate, unpublished components. Compatibility is judged against pinned upstream revisions and regenerated fixtures rather than inferred from similar APIs. **Review scope (1 November):** SR-01.1/01.2 follow the [oracle audit](../../estate-review/2026-11-01-sidestr-oracle-contract.md); SR-01.3/01.4 and the core feature label follow the [crate/runtime audit](../../estate-review/2026-11-01-sidestr-crate-and-runtime-boundaries.md).

## For the business

New source expands what the estate can validate and eventually operate, but it does not change what is live. `sidestr:dreamlab` is still a test-only chain produced by the upstream JavaScript engine, and a second sealed research chain, `sidestr:dreamlab-txbt4` beside a BLAKE2b parent, is also produced by that engine. Publication, source completeness and deployment are tracked as separate facts.

## SR-01.1 Workspace ownership

```mermaid
flowchart LR
    subgraph reusable["Eight publishable crates"]
        A["agent"] --> C["core"]
        A --> W["wallet"]
        A --> N["nostr"]
        A --> H["header"]
        A --> CH["hitch"]
        A -. "CLI feature" .-> R["round"]
        R --> C
        R --> H
        R --> N
        W --> C
        W -. "dev only" .-> H
        N --> C
        H -. "core feature" .-> C
        CH --> C
        T["teller"] --> C
        T --> N
    end
    subgraph private["Three private crates"]
        E["evm"]
        L["Liquid adapter"] --> RA["neutral reserve"]
    end
    E --> C
    E -. "dev only" .-> W
```

**What it shows.** Arrows mean direct local dependencies; dotted edges distinguish optional features and test-only dependencies. Eleven members comprise eight `publish=true` and three private crates (`../sidestr-rs/Cargo.toml:3`). Agent's five required local dependencies and optional round are explicit (`../sidestr-rs/sidestr-agent/Cargo.toml:35`); round also imports header (`../sidestr-rs/sidestr-round/Cargo.toml:45`). Wallet-to-header and EVM-to-wallet are dev dependencies (`../sidestr-rs/sidestr-wallet/Cargo.toml:41`, `../sidestr-rs/sidestr-evm/Cargo.toml:42`). Liquid imports neutral reserve, not the reverse (`../sidestr-rs/sidestr-bridge-liquid/Cargo.toml:32`).

**Why it is this way.** The header's optional `core` feature points towards core (`../sidestr-rs/sidestr-header/Cargo.toml:30`); isolated header builds need not import it. EVM and origin adapters stay outside ordinary library dependencies. Publishable is a manifest property, not a fresh crates.io release check. The [dependency receipt](../../estate-review/evidence/2026-11-01/oracle/workspace-edges.json) records every direct local edge and its kind.

## SR-01.2 Two independent CI jobs

```mermaid
flowchart TB
    START["Push or pull request"] --> Q
    START --> P
    Q["Quality job<br/>fmt, clippy, Rust tests, docs,<br/>core std-feature-off and WASM builds<br/>../sidestr-rs/.github/workflows/ci.yml:51-99"]
    P["Oracle job: six pinned checkouts<br/>spec, schema, blaketestnode,<br/>Hitch, Reef and teller<br/>../sidestr-rs/.github/workflows/ci.yml:101"]
    P --> RT["Reference project tests<br/>../sidestr-rs/.github/workflows/ci.yml:169"]
    RT --> FX["Regenerate EVM fixtures<br/>tracked and untracked drift fail<br/>../sidestr-rs/.github/workflows/ci.yml:190"]
    FX --> O["Rust oracle and interop suites<br/>six reference paths supplied<br/>../sidestr-rs/.github/workflows/ci.yml:211"]
    Q --> RESULT["Workflow results<br/>participation still needs evidence"]
    O --> RESULT
```

**What it shows.** Quality and oracle are independent jobs with no `needs` edge (`../sidestr-rs/.github/workflows/ci.yml:33`, `../sidestr-rs/.github/workflows/ci.yml:101`). Core checks build with the `std` feature off; the crate does not declare `#![no_std]` and still imports the standard library (`../sidestr-rs/sidestr-core/src/lib.rs:165-167`, `../sidestr-rs/sidestr-core/src/overlays.rs:7`). Within oracle, checkouts precede reference tests, regenerated EVM fixtures and Rust suites. Drift checks include modified tracked files and untracked fixture output (`../sidestr-rs/.github/workflows/ci.yml:203`). ADR-0001's Decision 1 still says four references; a dated note (20 November) above it records that CI now pins six (`../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md:29`, `../sidestr-rs/docs/adr/ADR-0001-reference-oracle-is-the-compatibility-contract.md:25`, `../sidestr-rs/.github/workflows/ci.yml:17`).

**Why it is this way.** Ordinary Rust tests deliberately allow absent oracles (`../sidestr-rs/.github/workflows/ci.yml:60`). Missing configured files also return early in round interop and pegtweak (`../sidestr-rs/sidestr-round/tests/support/interop.rs:18`, `../sidestr-rs/sidestr-core/tests/pegtweak.rs:406`); a reported pass alone is insufficient parity evidence. [Run 41128639012](https://github.com/DreamLab-AI/sidestr-rs/actions/runs/41128639012) supplies current implementation CI evidence, bounded by the [audit receipt](../../estate-review/evidence/2026-11-01/oracle/ci-receipt.json); it does not prove release enforcement or chain activation.

## SR-01.3 Independent evidence axes

```mermaid
flowchart TB
    STATUS["Ecosystem decision record<br/>docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:115"]
    STATUS --> DEC["Decision: proposed"]
    STATUS --> IMP["Implementation: partial"]
    STATUS --> ACT["Activation: inactive"]
    SOURCE["Crate code and scoped tests"] -. "supports implementation" .-> IMP
    PUB["Publication receipt<br/>separate from source capability"]
    DEPLOY["Named host and runtime receipt"]
    LIFE["Per-capability acceptance receipt"] -. "needed to promote activation" .-> ACT
    PUB -. "does not prove" .-> ACT
    DEPLOY --> LIFE
    BOUND["INVARIANT, ADR-0002 reconciliation 2026-11-02: crate boundaries are an<br/>architectural property, not a consensus-integrity certificate, and an<br/>optional-library defect is not estate-chain exposure without activation evidence<br/>ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md:25, :27"]
    REP["OPEN, ADR-0002 repair 2026-11-03: withdrawal outputs consumed once, snapshot<br/>code hashes checked on restore, header failures no longer evict valid txs.<br/>It rejects formerly accepted blocks, so consumers must adopt together<br/>before activation. agentbox source pins sidestr-agent at 60259e8, carrying those<br/>repairs, sidestr-agent.nix:30, ADR-0002 records no rebuild or activation receipt<br/>ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md:21"]
    SOURCE -.-> BOUND
    REP -. "adoption needed" .-> LIFE
```

**What it shows.** ADR-2012's recorded decision, implementation and activation axes are independent (`docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:115`, `docs/adr/ADR-2012-sidestr-settlement-is-ecosystem-canon.md:130`). Source, package publication, runtime processes and a successful capability lifecycle are different evidence. The matrix does not promote the ecosystem proposal merely because research producers run.

**Why it is this way.** N-9 closes historical source/release work while N-10 asks for activation receipts; both rows now sit on the archived board, since the regenerated master board carries neither (`../project/docs/archive/TODO-unified-2026-09-07-to-2026-11-15.md:102`, `../project/docs/archive/TODO-unified-2026-09-07-to-2026-11-15.md:103`). N-10's claim that Hitch has no host is stale: `sidestr-agent::hitch::host` exists; that still does not establish an accepted deployed channel lifecycle (`../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md:156`). The agentbox pin (60259e8) predates the Hitch follower's genesis pin (`../sidestr-rs/docs/adr/ADR-0002-keep-optional-execution-and-services-behind-crate-boundaries.md:17`, `../project/agentbox/lib/sidestr-agent.nix:30`).

**Invariant:** source evidence may promote implementation status; it cannot by itself promote activation status.

## SR-01.4 Two supervised research chains

```mermaid
flowchart TB
    AB["agentbox supervisor<br/>../project/agentbox/flake.nix:3374"]
    AB --> P1["dreamlab / tbtc4<br/>JavaScript producer port 3450"]
    AB --> P2["dreamlab-txbt4 / txbt4<br/>JavaScript producer port 3451"]
    F1["sidestr-agent faucet"] --> P1
    F2["Forum kit<br/>nostr-bbs-sidestr-admin faucet"] --> P2
    P1 --> M1["dreamlab Git mirror loop"]
    P2 --> M2["Separate txbt4 Git mirror loop"]
    ROUTE["Faucet selected by parent family<br/>../project/agentbox/config/sidechain/run-faucet.sh:59"] -.-> F1
    ROUTE -.-> F2
```

**What it shows.** At 14:15:29 UTC on 1 November, all six named programmes were running; both producer endpoints served documents equal to their sealed checkout documents. The producers execute baked `siding.mjs`; the faucet binaries differ as the runner specifies (`../project/agentbox/config/sidechain/run-producer.sh:167`, `../project/agentbox/config/sidechain/run-faucet.sh:59`). [Runtime receipt](../../estate-review/evidence/2026-11-01/crate-boundary/runtime.json). The second table is enabled and the README now says so (`../project/agentbox/agentbox.toml:2046`, `../project/agentbox/config/sidechain/README.md:188`). **Drift:** that README line names line 2044 of the manifest, two lines above the setting, and the comment over the table still reads "Off until the lead starts it" (`../project/agentbox/agentbox.toml:2042`).

**Why it is this way.** Each chain has its own producer, mirror and faucet; the BLAKE2b family chooses the forum-kit faucet. A responding tip is not evidence of current block progress, mirror freshness, successful grants or Hitch activation. The observed producers have no checkpoint arguments, consistent with default-zero configuration (`../project/agentbox/config/sidechain/run-producer.sh:148`); historical anchoring was not inspected.

**Open:** retain separate runtime and per-capability acceptance receipts; no native Rust producer or channel lifecycle was demonstrated by this observation.
