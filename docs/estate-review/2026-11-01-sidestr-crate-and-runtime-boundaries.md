---
title: SideStr crate boundaries, Hitch chain identity and current runtime
date: 2026-11-01
status: in-progress
type: explanation
---

# SideStr crate and runtime boundaries

Core has no local dependency on Hitch, EVM, wallet or reserve crates, and the local normal-dependency graph is acyclic. A Hitch host also exists in `sidestr-agent`, so “missing host” is no longer an accurate source claim. Its chain follower has a material identity gap: it accepts a different genesis signed by the same key without enforcing the configured genesis hash. The full validator rejects the same input.

This report assesses all sections and both amendments of SideStr ADR-0002, refreshes SR-01.3/01.4 and SR-04.5, and corrects SR-01.2's overbroad `no_std` label. Evidence is tied to [37 source fingerprints](evidence/2026-11-01/crate-boundary/sources.json): SideStr `9c6de86c`, agentbox `9dfc27b1`, VisionClaw `531df5ef7` and VisionFlow `9cc2266`. Only VisionFlow audit material changes. VisionClaw springs and all deployed services remain untouched.

## Findings and acceptance criteria

| Finding | Trigger, evidence and consequence | Required boundary |
| --- | --- | --- |
| **B1: Hitch follows a signer without enforcing the genesis pin** | `ChainFollower::new` keeps family and challenge, but not `doc.genesis_hash` or chain identity (`sidestr-agent/src/hitch/follow.rs:121-140`). `plan` compares index positions/hashes; `apply` checks signatures, index hashes and links (`:188-213,246-295`). Seven native scenarios show that a wrong pin and another same-signer chain's genesis are accepted, including replacement at fork zero. `State::from_genesis` rejects those cases. | Retain and check the expected genesis at initial load and every replacement; refuse a reorganisation through that anchor. Treat index network metadata consistently with the selected document. Add wrong-pin, same-signer foreign-chain and fork-zero regression cases. |
| **B2: disabling the agent CLI does not leave only pure modules** | The host is gated by `cli` and Unix, but `store` is gated only by Unix (`hitch/mod.rs:34-37`). Its no-default-features doctest creates directories and writes, reads and lists a file. This contradicts the host amendment's “only pure parts” description on native Unix. | Either name storage as a deliberate feature-independent native API, or give it an explicit feature and test that feature matrix. This is an API/portability mismatch; the probe does not demonstrate unauthorised filesystem access. |
| **B3: optional clients and admission policy remain inside core/wallet** | Core's `rpc` feature exposes `CoreRpc` over `ureq` (`sidestr-core/Cargo.toml:35-49`, `src/parent.rs:774`); wallet's `client` feature adds delivery HTTP. Core also owns mempool admission via `State::submit`. ADR decision 4's blanket “outside both” wording is stronger than this composition. | Record the feature-gated exceptions and distinguish deterministic validation from admission policy and transport. Do not promise a crate-level network boundary where the actual boundary is a feature. |
| **B4: core's disabled `std` feature is not `no_std`** | `sidestr-core/src/lib.rs:298-307` has no `#![no_std]` attribute; always-built `state` and `overlays` import `std` collections and synchronisation. The CI step labelled “core alone, no std” only disables Cargo features on targets with a standard library. Header, by contrast, has an actual conditional `no_std` attribute at `sidestr-header/src/lib.rs:144`. | Rename the core lane to describe its actual configuration, or implement and test a genuine no-standard-library contract if required. A successful WASM build does not establish embedded/no-standard-library support. SR-01.2 and the preceding oracle report now make this distinction. |
| **B5: source/adoption records lag implemented services** | ADR-0002 still says seven public crates; manifests now allow eight. VisionClaw N-10 says Hitch has no host and the Dell parent carries no estate chain. Agentbox's sidechain README says the second table ships disabled. Current source has a host and both tables enabled; both producers answer locally. | Refresh those owner records without promoting Hitch activation. “Source host implemented”, “process running” and “channel lifecycle accepted” need separate receipts. |

B1 is not a signature bypass. The substituted blocks must satisfy the configured challenge; the synthetic probe deliberately reuses one signer across two chains. The two estate chain documents currently use different signer challenges, so this report does not claim that one of those live chains can simply impersonate the other. A mirror can nevertheless replay a valid alternate history where the signer is shared, and the follower has no genesis anchor to reject it.

Host composition does not supply the missing anchor elsewhere: `Host::load` verifies the account binding against the configured genesis and then constructs `ChainFollower` (`host.rs:647-664`). The watch path fetches an index and delegates directly to `plan`/`apply` (`:2230-2269`). Binding the spend identity to a document does not bind the followed block history to that document. No deployed channel, loss or successful hostile network interaction was tested.

## ADR-0002 section assessment

| Section or obligation | Disposition |
| --- | --- |
| Metadata, status and trigger | Accepted/complete/inactive are the owner's axes. Core has no reverse dependency on Hitch or EVM, but it contains channel script templates under the explicit amendment. “Complete” does not close B1–B4 or establish channel activation. |
| Context | Separating optional execution and host I/O avoids bringing revm and services into every validator. [All-feature metadata](evidence/2026-11-01/crate-boundary/dependency-boundary.json), interpreted over declared local normal edges, confirms the acyclic direction. This is not a whole dependency-security assessment. |
| Decision 1 | Core exports assets, pool and market modules and deterministic rule/state machinery. Presence is established; full economic transition review belongs to SR-02 and is still open. |
| Decision 2 | `OverlayRules::boxed` emits assets, pool, then markets (`overlays.rs:107-116`); `StateOf::from_genesis_with_rules` admits supplied rule implementations. EVM depends on core, with no reverse local edge. Complete EVM semantics remain outside this pass. |
| Decision 3 | Hitch's only direct local normal dependency is core. The kernel returns messages/actions; `sidestr-agent::hitch::host` supplies operational I/O. B1 qualifies its chain watch, while B2 qualifies how storage is exposed. |
| Decision 4 | Wallet construction/signing is a separate crate. Optional HTTP clients and core admission policy make the absolute “outside both” sentence inaccurate (B3). The later amendment already places lock-maturity admission in core. |
| Decision 5 | Eight publishable and three private components now replace the dated seven-plus-EVM inventory. Source release, daemon presence and accepted activation are different facts. Missing a `rules` field excludes explicit optional rule declarations in the observed documents; it does not prove no Hitch channel has ever run. |
| Consequences | Minimal core consumers avoid revm and other local service crates. Feature and platform selection still matter; “minimal” does not mean `no_std` (B4). A host must preserve rule order, durable state and chain identity. |
| Verification | The original commit and release/runtime statements are historical. Current dependency direction is freshly checked. Current two-producer evidence is below; historical channel absence is not treated as a present-day observation. |
| Channel amendment: trigger and decision 1 | Seven Hitch leaf templates are present in `sidestr-core::channel`; this adds consensus data knowledge without a dependency on the channel kernel. The trigger's fork risk explains the amendment, not a newly reproduced live fork. |
| Channel amendment: decision 2 | Template recognition uses rust-bitcoin/secp256k1 for commitments and signatures. A fresh native Bitcoin Core oracle test exercises 436 agreeing cases and six deliberate stricter refusals. It does not prove general Script compatibility or every possible input. |
| Channel amendment: decision 3 | Script checks and block-level relative/absolute maturity remain separate. `State::submit` performs admission checks; the host also calls `premature` before broadcasts and refuses time-based locks it cannot judge. Its focused lock test passes. |
| Channel amendment: evidence, liveness finding and activation | The 436-plus-six script result is freshly reproduced. The dual-engine whole-chain and October live replay claims remain historical; they are not rerun here. The running producer's pin is now `2fd6114`, distinct from the old `fa86dac` example. No immature transaction was submitted to a running producer, so no new liveness acceptance is claimed. Recognising script leaves does not itself activate Hitch. |
| Host amendment: placement and feature gates | `sidestr-agent → sidestr-hitch` is implemented. Host networking/locking uses `cli` and Unix. Storage remains public on Unix without `cli`, contrary to the stated pure-only configuration (B2). |
| Host amendment: envelope and chain watch | A named `sidestr-nostr::hitch` constructor signs kind 23600; content is signed, not encrypted. The follower intentionally trusts transaction validity while checking signatures/links. B1 identifies a separate omitted genesis check, not a demand to duplicate full validation in the watcher. Both header families have existing follower coverage. |
| Host amendment: key separation and activation | Four focused binding tests pass, including the independent vector and rejection of the identity key as spender. The host's binding check names the configured genesis; B1 shows why that alone is insufficient. No current open/pay/close receipt is established by this pass, so activation remains unverified here. |

## Fresh tests and evidence limits

The [channel oracle](evidence/2026-11-01/crate-boundary/channel-oracle.txt) reports one Rust test containing **436 agreeing differential cases and six asserted narrowings**. The separate no-CLI [store doctest](evidence/2026-11-01/crate-boundary/no-cli-store.txt), [three follower tests](evidence/2026-11-01/crate-boundary/follower-existing.txt), [four binding tests](evidence/2026-11-01/crate-boundary/binding.txt) and [one host lock test](evidence/2026-11-01/crate-boundary/host-locks.txt) also pass. That is ten reported native test/doc entries, with no skipped oracle half in these selected tests. It is not 446 separate Rust test functions or a fresh full-workspace run.

[Seven additional scenarios](evidence/2026-11-01/crate-boundary/follower-probe.json) call the production library through [this probe](evidence/2026-11-01/crate-boundary/follower-probe.rs). Positive and negative controls establish matching genesis acceptance, full-validator rejection of mismatched histories, bad-index-hash refusal and wrong-signer refusal. The same-key foreign genesis and fork-zero replacement then demonstrate B1. All material is synthetic; the probe accesses no network or operational state. [The runner](evidence/2026-11-01/crate-boundary/run-probe.cjs) builds with locked/offline dependencies and compiles the probe with warnings denied.

## Runtime observation

At **14:15:29 UTC on 1 November**, [read-only observations](evidence/2026-11-01/crate-boundary/runtime.json) found all six producer/mirror/faucet programmes running. Both producers execute Node 22.23.3 with the baked `siding.mjs produce` entrypoint. The baked reference pins are spec `2fd6114`, schema `b8cbf63`, blaketestnode `de33b34`; they differ from SideStr's CI reference set. The active producer/faucet runner bytes match the agentbox checkout.

| Observed chain | Producer response | Faucet process |
| --- | --- | --- |
| `sidestr:dreamlab`, `tbtc4`, port 3450 | Tip height 5,023; served chain document equals the sealed checkout document | `sidestr-agent` 0.6.0 |
| `sidestr:dreamlab-txbt4`, `txbt4`, port 3451 | Tip height 14,667; served chain document equals the sealed checkout document | Forum kit `nostr-bbs-sidestr-admin`, package beta.16 |

That faucet split is explicit in `run-faucet.sh:59-62`; the previous single-faucet diagram hid it. Both served documents lack explicit optional `rules`. Neither process has checkpoint arguments in the captured command, consistent with disabled/default-zero checkpoint configuration; this pass does not inspect historical parent checkpoint transactions. A responding tip and running mirror do not establish continuing block production, mirror freshness, grant success, parent health or channel lifecycle acceptance.

The aggregate `supervisorctl status` command returned exit 3; the receipt records the selected six `RUNNING` entries rather than treating the command's aggregate exit code as a SideStr failure. No secrets or environment were captured. No service was restarted, funded or reconfigured.

## Comparison and user utility

LDK also separates a protocol library from application choices for storage, wallets, networking and chain monitoring. That is a useful architectural comparison for the Hitch kernel/agent-host split. It does not imply Lightning interoperability: the SideStr README expressly excludes that claim. The next useful operator view should show protocol capability, selected host, chain identity, last validated observation and lifecycle receipt separately. [LDK architecture](https://lightningdevkit.org/introduction/architecture).

For an integrator, the immediate acceptance demonstration is an offline chain-switch refusal: a configured genesis stays fixed when a mirror supplies a different same-signer history. Then exercise open, pay, close, restart and reorganisation in an isolated test setup, preserving receipts before any activation claim. A crate/platform matrix should also distinguish native Unix storage, CLI networking, WASM availability and actual `no_std` support. These are specific missing contracts, not a proposal to replace the existing kernel or duplicate estate tooling.

## Diagram and audit disposition

SR-01.3 now represents publication, adoption and activation as independent evidence axes. SR-01.4 shows two separate producer/mirror/faucet groups and the actual faucet implementations, tied to a dated observation. SR-04.5 names the implemented agent host, its watcher trust boundary and the missing genesis check. Its other four panels retain their earlier evidence boundary. SR-01.2's quality lane now describes core's disabled feature accurately; the [preceding oracle report](2026-11-01-sidestr-oracle-contract.md) is qualified accordingly.

This completes section-level assessment of all three local numbered SideStr ADRs, with open findings preserved. It does not complete SideStr: native overlays, EVM, the remaining Hitch kernel paths, custody/recovery, all activation receipts and the remaining diagrams still need review. Full agentbox and VisionFlow settlement decisions are not credited from this scoped consumer/runtime assessment. The wider estate audit remains active.
