// Offline recovery checks against production sidestr-hitch and sidestr-core APIs.
// Deterministic opening fixture adapted from sidestr-hitch/src/protocol/tests.rs.
use bitcoin::consensus::{deserialize, serialize};
use bitcoin::hashes::Hash;
use bitcoin::secp256k1::{Keypair, SecretKey, XOnlyPublicKey};
use bitcoin::{Amount, OutPoint, ScriptBuf, Transaction, Txid, Witness};
use serde_json::{json, Value};
use sidestr_core::{block::secp, channel::verify_channel_input, sighash::SighashRules};
use sidestr_hitch::protocol::*;
use sidestr_hitch::route::{Invoice, InvoiceRecord, RouteAction, Router};

fn public(key: &SecretKey) -> XOnlyPublicKey {
    Keypair::from_secret_key(secp(), key).x_only_public_key().0
}

fn key(byte: u8) -> SecretKey {
    SecretKey::from_slice(&[byte; 32]).unwrap()
}

fn ctx(height: u32) -> Context {
    Context::new(height, 1_800_000_000, [7; 32])
}

const H: u32 = 152_100;

/// Open, accept, commit and ready between A (0x11) and B (0x22), then confirm.
fn open_pair(push: u64) -> (ChannelMachine, ChannelMachine, OpenMessage, AcceptMessage) {
    let channel_a = key(0x11);
    let channel_b = key(0x22);
    let params = OpenParams {
        funding: OutPoint {
            txid: Txid::from_byte_array([0xab; 32]),
            vout: 1,
        },
        funding_value: Amount::from_sat(100_000),
        push: Amount::from_sat(push),
        delay: 6,
        fee: Amount::from_sat(300),
        hub_fee: None,
    };
    let (funder, open) = FunderOpening::propose(
        params,
        OpeningKeys {
            channel: channel_a,
            revocation_base: key(0x30),
            revocation: [key(0x31), key(0x32)],
        },
        public(&channel_b),
        SighashRules::KnotsUnified,
        &[0; 32],
    )
    .unwrap();
    let (receiver, accept) = ReceiverOpening::accept(
        open.clone(),
        public(&channel_a),
        OpeningKeys {
            channel: channel_b,
            revocation_base: key(0x40),
            revocation: [key(0x41), key(0x42)],
        },
        SighashRules::KnotsUnified,
        AcceptPolicy::default(),
        &[0; 32],
    )
    .unwrap();
    let (funder, commit) = funder.accept(accept.clone(), &[0; 32]).unwrap();
    let (mut b, ready) = receiver.commit(commit).unwrap();
    let mut a = funder.ready(ready).unwrap();
    assert_eq!(a.status(), ChannelStatus::Funding);
    a.confirm_funding(H);
    b.confirm_funding(H);
    (a, b, open, accept)
}

fn acknowledge(result: ReceiveUpdate) -> AckMessage {
    match result {
        ReceiveUpdate::Acknowledge(ack) => ack,
        other => panic!("unexpected {other:?}"),
    }
}

/// Pay `amount` from `from` to `to` through all three messages.
fn pay(from: &mut ChannelMachine, to: &mut ChannelMachine, amount: u64, seed: u8) {
    let update = from.pay(amount, None, key(seed), &ctx(H)).unwrap();
    let ack = acknowledge(to.receive_update(update, key(seed + 1), &ctx(H)).unwrap());
    let outcome = from.receive_ack(ack).unwrap();
    assert!(to.receive_revoke(outcome.revoke).unwrap().is_some());
}

fn restored(value: Value) -> Result<ChannelMachine, ProtocolError> {
    ChannelMachine::restore(serde_json::from_value(value).unwrap())
}

fn close_witness() -> Value {
    let (mut a, _, _, _) = open_pair(20_000);
    let original = a.force_close(None, "audit", &ctx(H)).unwrap().tx;
    let prevouts = [a.channel().funding_prevout()];
    assert!(verify_channel_input(&original, 0, &prevouts, SighashRules::KnotsUnified).is_ok());
    let mut snapshot = serde_json::to_value(a.snapshot()).unwrap();
    assert!(restored(snapshot.clone()).is_ok());
    let mut wrong_txid = snapshot.clone();
    wrong_txid["close"]["txid"] = json!(Txid::from_byte_array([0x99; 32]).to_string());
    assert!(restored(wrong_txid).is_err());
    let mut damaged = original.clone();
    damaged.input[0].witness = Witness::new();
    assert_eq!(damaged.compute_txid(), original.compute_txid());
    assert_ne!(damaged.compute_wtxid(), original.compute_wtxid());
    let witness_error =
        verify_channel_input(&damaged, 0, &prevouts, SighashRules::KnotsUnified).unwrap_err();
    snapshot["close"]["tx"] = json!(hex::encode(serialize(&damaged)));
    let mut recovered = restored(snapshot).expect("txid-consistent damaged witness accepted");
    let broadcasts: Vec<_> = recovered
        .tick(&ctx(H + 10), &TickOptions::default())
        .into_iter()
        .filter_map(|a| match a {
            TickAction::Broadcast(b) => Some(b.tx),
            _ => None,
        })
        .collect();
    assert_eq!(broadcasts, vec![damaged]);
    assert!(
        verify_channel_input(&broadcasts[0], 0, &prevouts, SighashRules::KnotsUnified).is_err()
    );
    json!({"id":"H1", "normal_restore":true,"wrong_txid_refused":true,
        "original_spend_valid":true,"damaged_witness_restored":true,"same_txid":true,
        "different_wtxid":true,"damaged_rebroadcasts":broadcasts.len(),"spend_error":format!("{witness_error:?}")})
}

fn missing_revocation() -> Value {
    let (mut a, mut b, _, _) = open_pair(20_000);
    let old_peer_txid = b.signed_commitment(0, &[0; 32]).unwrap().compute_txid();
    pay(&mut a, &mut b, 5_000, 0x51);
    let snapshot = serde_json::to_value(a.snapshot()).unwrap();
    assert!(snapshot["their_revocation"].get("0").is_some());
    let spend = ChainSpend {
        txid: old_peer_txid,
        height: H + 1,
    };
    let destination = ScriptBuf::new();
    let mut control = restored(snapshot.clone()).unwrap();
    let protected = control
        .on_spend(spend, &destination, &ctx(H + 1), |_, _| {
            OutputLookup::Unspent
        })
        .unwrap();
    assert!(matches!(
        protected.kind,
        FundingSpend::RemoteCommitment { revoked: true, .. }
    ));
    assert!(!protected.broadcasts.is_empty());
    let mut wrong_secret = snapshot.clone();
    wrong_secret["their_revocation"]["0"] = json!(hex::encode(key(0x55).secret_bytes()));
    assert!(restored(wrong_secret).is_err());
    let mut missing = snapshot;
    missing["their_revocation"]
        .as_object_mut()
        .unwrap()
        .remove("0");
    let mut recovered = restored(missing.clone()).expect("missing old reveal accepted");
    assert_eq!(recovered.missing_reveals(), vec![0]);
    assert!(matches!(
        recovered.pay(1_000, None, key(0x71), &ctx(H)),
        Err(ProtocolError::MissingReveal(0))
    ));
    let sync = b.sync_message(SyncTag::Synced);
    let mut synced_first = restored(missing).unwrap();
    synced_first.receive_sync(&sync, &ctx(H + 1)).unwrap();
    assert!(synced_first.missing_reveals().is_empty());
    let resynced = synced_first
        .on_spend(spend, &destination, &ctx(H + 1), |_, _| {
            OutputLookup::Unspent
        })
        .unwrap();
    assert!(matches!(
        resynced.kind,
        FundingSpend::RemoteCommitment { revoked: true, .. }
    ));
    assert!(!resynced.broadcasts.is_empty());
    let unprotected = recovered
        .on_spend(spend, &destination, &ctx(H + 1), |_, _| {
            OutputLookup::Unspent
        })
        .unwrap();
    assert!(matches!(
        unprotected.kind,
        FundingSpend::RemoteCommitment { revoked: false, .. }
    ));
    assert!(unprotected.broadcasts.is_empty());
    recovered.receive_sync(&sync, &ctx(H + 2)).unwrap();
    assert!(recovered.has_their_revocation(0));
    let late = recovered
        .on_spend(spend, &destination, &ctx(H + 2), |_, _| {
            OutputLookup::Unspent
        })
        .unwrap();
    assert!(matches!(
        late.kind,
        FundingSpend::RemoteCommitment { revoked: false, .. }
    ));
    assert!(late.broadcasts.is_empty());
    json!({"id":"H2","agreed_state":1,"old_peer_state":0,"wrong_secret_refused":true,
        "control_revoked":true,"control_penalties":protected.broadcasts.len(),
        "missing_secret_restored":true,"new_payment_blocked":true,
        "sync_before_spend_restores_penalty":true,"late_sync_recovers_secret":true,
        "same_spend_still_classified_unrevoked_after_late_sync":true,"recovered_revoked":false,"recovered_broadcasts":unprotected.broadcasts.len()})
}

fn claim_witness() -> Value {
    let (mut a, _, _, _) = open_pair(20_000);
    let close = a.force_close(None, "audit", &ctx(H)).unwrap().tx;
    let outcome = a
        .on_spend(
            ChainSpend {
                txid: close.compute_txid(),
                height: H + 1,
            },
            &ScriptBuf::new(),
            &ctx(H + 10),
            |_, _| OutputLookup::Unspent,
        )
        .unwrap();
    assert_eq!(outcome.broadcasts.len(), 1);
    let original = &outcome.broadcasts[0].tx;
    let prevouts = [close.output[original.input[0].previous_output.vout as usize].clone()];
    assert!(verify_channel_input(original, 0, &prevouts, SighashRules::KnotsUnified).is_ok());
    let mut snapshot = serde_json::to_value(a.snapshot()).unwrap();
    assert_eq!(snapshot["claims"].as_array().unwrap().len(), 1);
    let mut damaged: Transaction =
        deserialize(&hex::decode(snapshot["claims"][0]["tx"].as_str().unwrap()).unwrap()).unwrap();
    damaged.input[0].witness = Witness::new();
    assert_eq!(damaged.compute_txid(), original.compute_txid());
    snapshot["claims"][0]["tx"] = json!(hex::encode(serialize(&damaged)));
    let mut recovered = restored(snapshot).expect("damaged claim witness accepted");
    let broadcasts = recovered.watch_outputs(&ctx(H + 20), |_, _| OutputLookup::Unspent);
    assert_eq!(broadcasts.len(), 1);
    assert_eq!(broadcasts[0].tx, damaged);
    assert!(
        verify_channel_input(&broadcasts[0].tx, 0, &prevouts, SighashRules::KnotsUnified).is_err()
    );
    json!({"id":"H3","original_claim_valid":true,"damaged_claim_restored":true,
        "same_txid":true,"damaged_claim_rebroadcasts":broadcasts.len()})
}

fn invoice_expiry() -> Value {
    let (mut payer, mut payee, _, _) = open_pair(20_000);
    let (preimage, invoice) = Invoice::new(
        NodeId(public(&key(0x22))),
        5_000,
        None,
        vec![],
        0,
        1,
        [0x66; 32],
    )
    .unwrap();
    assert!(invoice.x < ctx(H).now);
    let update = payer
        .add_htlc(5_000, invoice.h, H + 100, None, None, key(0x61), &ctx(H))
        .unwrap();
    let ack = acknowledge(payee.receive_update(update, key(0x62), &ctx(H)).unwrap());
    let revoke = payer.receive_ack(ack).unwrap().revoke;
    let finalised = payee.receive_revoke(revoke).unwrap().unwrap();
    let mut router = Router::new(false, 0);
    let actions = router.on_update(
        &[&payee],
        payee.id(),
        &finalised,
        |_| {
            Some(InvoiceRecord {
                preimage,
                amount: invoice.a,
                paid: false,
            })
        },
        H,
    );
    assert!(matches!(&actions[..], [RouteAction::Settle { .. }]));
    let paid = router.on_update(
        &[&payee],
        payee.id(),
        &finalised,
        |_| {
            Some(InvoiceRecord {
                preimage,
                amount: invoice.a,
                paid: true,
            })
        },
        H,
    );
    assert!(matches!(&paid[..], [RouteAction::Fail { .. }]));
    json!({"id":"H4","invoice_expiry":invoice.x,"now":ctx(H).now,
        "protocol_finalised":true,"unpaid_expired_invoice_settles":true,"paid_control_refused":true,
        "basis":"Router receives height and InvoiceRecord, neither Unix time nor invoice expiry"})
}

fn main() {
    let cases = vec![
        close_witness(),
        missing_revocation(),
        claim_witness(),
        invoice_expiry(),
    ];
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()
    );
}
