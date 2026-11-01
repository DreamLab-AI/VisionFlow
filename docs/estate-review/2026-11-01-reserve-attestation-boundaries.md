---
title: SideStr reserve statements, confirmation depth and consumer policy
date: 2026-11-01
status: in-progress
type: explanation
---

# SideStr reserve statements, confirmation depth and consumer policy

SideStr has tested code for constructing and signing a private reserve statement. Its Liquid adapter selects confirmed outputs, but does not implement the configured confirmation-depth gate required before signing. Signature verification establishes integrity under the included public key; authority, freshness, replay and supply policy still belong to a future bridge consumer. No activated bridge or loss of funds is demonstrated by this review.

This pass assesses **all sections of local ADR-0003** at sidestr-rs `9c6de86c657a25ec2f2a189baf63db7dedad40eb` and reconciles all three SR-05 panels. Agentbox ADR-2117 is consulted for its confirmation, authority and activation contract at `9dfc27b16e82e9786535a55e790710493dd51fdf`; it receives no whole-document credit. VisionClaw's custody board is read at `531df5ef74453a2ed320bff968c203a9b91faed7`. The [source receipt](evidence/2026-11-01/reserve/sources.json) binds 23 files to committed bytes. Product code, owner statuses and the user's springs work are unchanged.

## Reproduced boundaries and findings

The [Rust probe](evidence/2026-11-01/reserve/reserve-probe.rs), [runner](evidence/2026-11-01/reserve/run-probe.cjs) and [fourteen results](evidence/2026-11-01/reserve/probe.json) use freshly compiled production libraries, synthetic snapshots and fixed test-only keys. These are library observations, not live holdings or mint transactions.

| Finding | Evidence and consequence | Acceptance needed |
| --- | --- | --- |
| **R1: confirmation-depth gate is absent from the adapter/signing composition** | `sidestr-bridge-liquid/src/attest.rs:92-115` keeps the pinned asset with `Some(height)` at or below the reported tip. At tip 100, a credit confirmed at 100 is counted. Changing its confirmation height to 95 produces an identical statement. Neutral signing accepts that statement. Agentbox ADR-2117 decision 4 requires `bridgeConfirmations` before signing; the neutral crate's adapter contract repeats that obligation (`sidestr-reserve/src/lib.rs:26-29`). | Before activating a signer, enforce the configured depth while confirmation-height evidence is still available. Test below/at/above threshold and reorganisation. The existing unsigned CLI does not itself perform premature signing. |
| **R2: checked construction does not make later statements trustworthy** | `attest` sorts ids, rejects duplicates and uses checked arithmetic (`lib.rs:499-525`). `ReserveAttestation` exposes public fields (`:473-485`); `check` validates syntax and duplicate ids (`:533-543`), without recomputing amounts. Changing a 25-unit statement to `u128::MAX` before signing succeeds and verifies. Changing the amount after signing fails. | Treat the signer input as a trusted attester claim. If construction is intended as an invariant, make mutation explicit or retain sufficient evidence to revalidate. Correct the “Built only by attest” wording. This is not a signature forgery. |
| **R3: verification is cryptographic, not bridge authorisation** | `SignedAttestation::verify` takes no expected key, origin, clock or replay state (`lib.rs:688-693`). A second test key's statement verifies against its carried key. Times 0 and `MAX_SAFE_INTEGER`, an empty zero reserve and repeated verification all succeed. Those results fit the function's documented scope. | The consumer must compare the pinned authority and admitted origin, reject stale/future readings under an explicit policy, persist replay state and enforce the reserve/supply rules. No consumer bypass is claimed: the reviewed crates do not implement that consumer. |
| **R4: aggregate evidence is not per-credit value evidence** | Canonical JSON commits the total and credit ids, not each credit's amount (`lib.rs:549-568`). Allocations `(10,20)` and `(20,10)` for the same two ids produce identical bytes. Confirmation heights are also absent. | Decide whether the future mint rule consumes an aggregate balance or individual credit values. A validator cannot derive per-credit amounts or confirmation depth from this statement alone. It needs an explicit trust rule or additional authenticated evidence. |

Controls reject duplicate outpoints, exclude unconfirmed and future-height outputs, verify a normal signature and reject post-signature amount tampering. R2–R4 are API and integration boundaries; they should not be reported as new exploits of a deployed asset. R1 exposes a concrete missing composition gate and overbroad “finality” wording in the local verification record.

The Liquid adapter documents its public Esplora trust boundary: local unblinding supplies asset and amount interpretation, while the server's completeness and freshness remain trusted (`attest.rs:14-25`). `ReserveWallet::snapshot` copies the reported tip and unblinded UTXOs (`wallet.rs:93-117`). Signing does not independently validate that reported chain view. This pass performs no live sync and makes no claim about current network incidents, issuer status or funds.

## ADR-0003 section assessment

| Section | Disposition |
| --- | --- |
| Metadata/status/review trigger | Accepted, complete and inactive are the owner's recorded axes. The format and adapter exist; the “complete” implementation label must not be extended to the absent bridge or depth gate. No second adapter or activation is established here. |
| Context | Origin-neutral format and origin-specific wallet interpretation are separated in source. The statement is project-specific; no interoperability or custody standard is inferred. |
| Decision 1 | Neutral origin/tip/credit types, restricted canonical JSON, tagged digest and BIP-340 signing port are implemented. Fresh tests establish canonical vectors, bounds, duplicate/overflow refusal and signature tamper checks. R2–R4 qualify what the statement proves. |
| Decision 2 | LWK, wallet sync and registry interpretation are in the Liquid crate. `asset.rs:93-124` recomputes the asset commitment and checks the pinned asset, ticker, domain and precision. R1 qualifies “finality”: tests cover confirmed-at-tip filtering, not configurable depth or independent chain validation. No live registry request was made. |
| Decision 3 | Both manifests set `publish=false`. A [default resolved dependency-graph check](evidence/2026-11-01/reserve/dependency-boundary.json) finds no path from any publishable workspace member to either private reserve crate; only the Liquid adapter reaches the neutral reserve crate. This is a local build-graph observation, not a fresh registry publication census or an all-feature proof. |
| Decision 4 | The CLI's `Attest` branch prints unsigned JSON and digest (`main.rs:173-181`). Signing is a library capability. Neither reviewed crate contains the bridge rule, funds a reserve through this path or deploys a chain. Current funding/activation was not inspected; historical “unfunded” wording is not a present-day balance receipt. |
| Decision 5 | SHA-256 comes from `sha2`; Schnorr operations use `secp256k1`. Fresh tests check BIP-340 vectors 0 and 1 and compare the tagged-hash construction against rust-bitcoin. This validates those exercised operations, not every dependency's security. The bespoke statement remains private. |
| Consequences | The dependency split supports a sibling adapter, but replacing an origin also requires a finality/replay policy. Shared statement shape does not make independent reserves or their trust assumptions equivalent. |
| Verification | The dated record remains historical. Its canonicalisation, duplicate, signature and registry assertions have fresh local evidence. “Liquid finality” needs the narrower R1 description. Current public funding, external signing and bridge activation remain unverified. |

## Tests and evidence limits

`cargo test --locked --offline -p sidestr-reserve -p sidestr-bridge-liquid -- --nocapture` completed successfully. Its [raw output](evidence/2026-11-01/reserve/suite.txt) reports **58 passed and one ignored**: 22 Liquid offline cases, one local SOCKS test, 20 attestation cases, eight JCS entries and seven doctests. **One of those eight JCS entries returns early** because `TELLER` is absent, so there are 57 exercised test/doc cases and one reported-pass skip. The ignored test is the explicitly live Liquid sync. Neither omission receives oracle or live acceptance credit. CI's separate oracle job does supply a pinned Teller checkout (`.github/workflows/ci.yml:29-30,141-145,207-215`); that CI job was not rerun here.

A local SOCKS test observes a greeting and a failed sync against a listener that hangs up. It does not prove every deployment route obeys proxy policy. Existing key tests cover creation permissions, overwrite/dangling-symlink refusal, readable-key rejection and redacted diagnostics using temporary test material. No operational reserve key was accessed.

All fourteen additional probe assertions pass. An initial launch from `/tmp` was refused with `EACCES`; the runner now puts its temporary executable in the external Cargo target directory and removes it after running. No assertion failed in that initial attempt because execution did not begin. [Strict clippy](evidence/2026-11-01/reserve/clippy.txt) passes for both crates and all targets. Cargo separately reports an upstream future-compatibility warning for `proc-macro-error2 2.0.1`; that warning is not treated as a clean dependency-security assessment.

## Comparison and user utility

LWK already supplies the watch-only wallet, client sync and separate signing components. Its own example keeps the descriptor-based wallet distinct from the transaction signer. SideStr's useful addition here is a private origin-neutral observation contract over those components, rather than a new wallet stack. Preserve that separation when building an operator view. [LWK architecture and wallet example](https://blockstream.github.io/lwk/book/).

Chainlink's Secure Mint description places reserve data inside a consuming mint decision: the contract checks sufficient unused reserves before creating additional supply. That is the relevant comparison point for SideStr's absent bridge rule. It does not establish that the two systems have equivalent trust, deployment, network support or operational assurance. [Chainlink Secure Mint](https://chain.link/blog/secure-mint).

RFC 8785 defines reproducible JSON bytes, not the truth or freshness of their contents. SideStr intentionally accepts a narrower ASCII/safe-integer subset and verifies it against `serde_jcs`; that supports deterministic encoding, not reserve sufficiency. [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785).

For the owner, the next useful demonstration is an offline evidence-to-decision trace: reported reserve, observation time, authority identity, configured depth, pending supply and an explicit permitted/refused outcome. Replaying a stale, repeated, wrong-authority or insufficient-depth fixture should visibly refuse a hypothetical mint. Keep amounts labelled as observations and show missing evidence as unknown. This is an audit recommendation, not authorisation to fund or activate a bridge.

## Diagram disposition and remaining scope

[SR-05](../diagrams/sidestr-rs/05-reserve-attestation-and-liquid-adapter.md) now shows confirmed-at-tip selection instead of claiming final credits, separates construction from mutable statement/signature checks, and replaces the misleading unpublished-to-unfunded lifecycle with independent publication, CLI and activation facts. All three panels are source-reviewed and re-stamped. Their diagram render/citation receipts are retained beside this report; structural checks are not whole-estate semantic acceptance.

ADR-0001, ADR-0002, the other seventeen SideStr panels, its channel/EVM/producer paths and the full agentbox ADR-2117 still require their own reconciled assessments. The wider estate audit remains active.
