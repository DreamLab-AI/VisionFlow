---
title: SideStr Hitch finality, channel identity and chain evidence
status: source-reviewed-open-findings
date: 2026-11-01
type: explanation
---

# SideStr Hitch finality, channel identity and chain evidence

Two valid protocol channels funded by different outputs of one transaction receive the same short channel ID. The transaction kernel supports distinct outpoints, but the agent indexes channels by that ID. Batch funding therefore needs an explicit protocol/host policy before the five-channel chain fixture can support a product claim.

A receiver can also hold an unverified future update while awaiting the current state's final revocation. Replaying that update performs signature verification and rejects a misbound signature. A reorganisation can reopen a channel when no local close was published. These are narrower behaviours than the previous diagram's unconditional invariants.

This assessment covers SR-04.1, SR-04.2 and SR-04.4 at SideStr `9c6de86c657a25ec2f2a189baf63db7dedad40eb`. The [source manifest](evidence/2026-11-01/hitch-finality/sources.json), [four offline scenarios](evidence/2026-11-01/hitch-finality/probe.json) and [native/oracle receipt](evidence/2026-11-01/hitch-finality/native-oracles.txt) define its evidence boundary. The [previous recovery report](2026-11-01-sidestr-hitch-recovery.md) and [host assessment](2026-11-01-sidestr-crate-and-runtime-boundaries.md) retain their open findings. No product code, operational channels or deployment changed.

## F1 — transaction identity is not protocol channel identity

`ChannelId::from_txid` takes the first sixteen display-order hexadecimal characters of the funding transaction ID (`sidestr-hitch/src/protocol/wire.rs:37-40`). `FunderOpening::propose` uses it without the output index (`opening.rs:210`). F1 completes two independent propose/accept/commit/ready handshakes with the same keys, transaction ID and funding value, but output indices 1 and 2. Both commitments pass the production script verifier and have different transaction IDs. Both peers nevertheless assign the same protocol channel ID to the two channels.

In the host, `on_open` returns immediately when that ID is already in `channels` (`sidestr-agent/src/hitch/host.rs:1327-1331`); its `on_ready` has the same guard (`1481-1484`). This is source evidence of a host constraint, not a live host reproduction. No random prefix collision, wrong-channel execution or fund loss was demonstrated. The chain-consensus fixture constructs `Channel` transactions directly; it does not exercise these protocol maps.

Acceptance: either enforce and clearly expose one channel per funding transaction for the current wire format, or agree an interoperable, versioned identity change and migration. Test two outputs through host registration, restart and message dispatch. Do not silently change IDs and break reference compatibility. For users, a batch-funding screen should promise only channels that the host can subsequently address and recover independently.

## F2 — buffered receipt and final adoption are separate states

`receive_update` shape-checks the message and checks its ID/state number, but stores the next update in `self.buffered` while `awaiting` is set (`machine.rs:795-836`). Proof, next-state and funding-signature checks follow that branch (`845-884`). The receiver has already advanced its state number when it returns the acknowledgement, while `awaiting` withholds finalisation (`891-902`). The proposer finalises on a valid acknowledgement (`910-990`); the receiver finalises on the final reveal (`996-1024`).

F2 holds back that reveal, confirms the receiver cannot propose a new update, and supplies a next-state update carrying the preceding update's signature. It is shape-valid and buffered; the buffer survives snapshot/restore. After the proper reveal, `take_buffered_update` returns it. Replaying it through `receive_update` rejects it and leaves state 1 unchanged. The correct next update then completes state 2. The initial fixture used an incorrectly sized signature and was refused at deserialisation; the retained fixture uses a well-formed, misbound signature so it tests the later boundary.

Buffering preserves receipt without accepting an invalid commitment. This contradicts the documentation's broad claim that no channel state is written before full verification. Acceptance: document the single unverified buffer and require hosts to replay through validation; keep receiver finality, saved state and emitted notifications distinct in the UI. Save-before-send remains a separate host obligation. Collision alternatives and their binding HTLCs remain implemented in `machine.rs:561` and `874-878`; this batch does not prove exhaustive scheduling equivalence.

## F3 — reorganisation restores the previous status conditionally

`un_spend` clears claims, penalties and followed outputs. With a local close and a previously closed-out status, it keeps that status and returns the close for rebroadcast. Otherwise it restores the previous status (`chain.rs:333-357`). F3 observes a peer commitment from an open channel, reverses the spend and confirms the channel is open and accepts a new proposal. A control that first publishes its own close remains closed-out and returns the same close after reversal.

This corrects the diagram's “without reopening” statement; it does not prove the policy unsafe. Host acceptance still needs real block removal/replacement, duplicate observations and persisted recovery. The earlier late-reveal classification finding remains open.

Penalties apply to revocable `to_local` and HTLC outputs (`chain.rs:276-306`), not literally every output. Output watching trusts host lookup results; it retries an unspent claim only when `height - claim.at > CLAIM_REBROADCAST`, whose value is six (`chain.rs:610-619`). The diagram now states that strict threshold. Production chain validation, lookup authenticity and endpoint activation remain separate obligations.

## F4 — revocation controls and the scope of independent evidence

Hitch combines even-parity secrets/points and binds each proof of possession to a channel/role/state context (`lib.rs:740-790`). F4 constructs a real commitment and tests its revocation leaf through `sidestr-core::channel::verify_channel_input`. The owner's state secret and channel key both fail; the counterparty base secret combined with the revealed state secret passes. Changing the proof's role or state context fails. These examples support the intended construction; they are not a general cryptographic proof or a fresh JavaScript parity result.

Running the existing chain suite with `consensus-oracle` invoked `bitcoinconsensus` 0.106.0, Core 26.0 lineage. It carried twelve channel spends and refused twelve candidate blocks. Core's independent script interpreter accepted the twelve carried spends and rejected seven script negatives. It accepted five premature-spend scripts, while SideStr's block rules rejected their timing. Script interpretation cannot establish coin depth or a complete block's validity.

Cargo reports six passing test entries in this selected run. Only four execute substantive checks: three chain tests and one native wire test. The Hitch transaction oracle and Hitch wire oracle return successfully without their reference environment. The chain test also skips its optional siding replay. Their explicit skip messages are retained. Neither the green test count nor the feature name establishes JavaScript participation. The prior thirty-five native tests remain receipts from the unchanged source; they were not rerun in this batch.

## Primary-source comparison and user utility

[BOLT 2](https://github.com/lightning/bolts/blob/master/02-peer-protocol.md#definition-of-channel_id) includes the funding output index in its v1 identity and defines a different basepoint-derived v2 identity. Its [forwarding rules](https://github.com/lightning/bolts/blob/master/02-peer-protocol.md#forwarding-htlcs) require irrevocable incoming commitment before forwarding. These are useful acceptance references for distinct channel identity and finality, not evidence that Hitch speaks Lightning.

[BOLT 3](https://github.com/lightning/bolts/blob/master/03-transactions.md#revocationpubkey-derivation) uses hash-weighted revocation key derivation, unlike Hitch's point sum with proofs of possession. Similar user intent does not establish wire, transaction or security equivalence. The practical product position remains a SideStr-specific payment component. Demonstrate addressable channels, explicit pending/final states and recoverable close handling before promoting it as a general payment network.

## Coverage and reproduction

The [updated topic](../diagrams/sidestr-rs/04-hitch-channel-kernel-and-host-gap.md) completes the current source assessment of its five panels; SR-04.3 and SR-04.5 retain byte-identical bodies and their existing claims. All twenty existing SideStr diagram panels now have recorded assessments. The whole SideStr repository, cross-estate integration and live channel acceptance remain incomplete. No additional whole-ADR assessment is claimed.

Run the [standalone probe](evidence/2026-11-01/hitch-finality/finality-probe.rs) using its [runner](evidence/2026-11-01/hitch-finality/run-probe.cjs), with a SideStr checkout and executable Cargo target directory. The runner builds locked/offline dependencies and compiles with warnings denied. The selected native command was:

```sh
env -u HITCH -u SIDESTR_SIDING -u SCHEMA -u BLAKETESTNODE \
  cargo test --locked --offline -p sidestr-hitch \
  --features consensus-oracle --test chain_consensus --test wire_oracle \
  --test oracle -- --nocapture
```

Removing the reference environment makes the non-participation receipt reproducible. It does not replace a configured reference run. Scoped citation/render checks and visual inspection cover this topic; no fresh whole-corpus semantic acceptance is claimed. RuView remains excluded from the active estate audit.
