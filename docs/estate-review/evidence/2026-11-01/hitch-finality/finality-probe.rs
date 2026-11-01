// Offline checks of production channel identity, finality and reorganisation APIs.
use bitcoin::hashes::Hash;
use bitcoin::secp256k1::{Keypair, SecretKey, XOnlyPublicKey};
use bitcoin::{Amount, OutPoint, ScriptBuf, Txid};
use serde_json::{json, Value};
use sidestr_core::{block::secp, channel::verify_channel_input, sighash::SighashRules};
use sidestr_hitch::protocol::*;
use sidestr_hitch::{pop_sign, pop_verify, revocation_key, sweep_to_local, SweepPath};
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
fn open_pair(push: u64, vout: u32) -> (ChannelMachine, ChannelMachine, OpenMessage, AcceptMessage) {
    let channel_a = key(0x11);
    let channel_b = key(0x22);
    let params = OpenParams {
        funding: OutPoint {
            txid: Txid::from_byte_array([0xab; 32]),
            vout,
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

fn identity() -> Value {
    let (a1, b1, _, _) = open_pair(20_000, 1);
    let (a2, b2, _, _) = open_pair(20_000, 2);
    assert_ne!(a1.channel().funding_outpoint, a2.channel().funding_outpoint);
    assert_eq!(a1.id(), a2.id());
    assert_eq!(b1.id(), b2.id());
    let c1 = a1.signed_commitment(0, &[0; 32]).unwrap();
    let c2 = a2.signed_commitment(0, &[0; 32]).unwrap();
    assert_ne!(c1.compute_txid(), c2.compute_txid());
    for (a, c) in [(&a1, &c1), (&a2, &c2)] {
        assert!(verify_channel_input(
            c,
            0,
            &[a.channel().funding_prevout()],
            SighashRules::KnotsUnified
        )
        .is_ok());
    }
    json!({"id":"F1","funding_outputs":[1,2],"same_protocol_id":a1.id().to_hex(),"distinct_commitments":true,"both_production_verified":true,"host_execution_tested":false})
}

fn finality() -> Value {
    let (mut a, mut b, _, _) = open_pair(20_000, 1);
    let u1 = a.pay(1000, None, key(0x51), &ctx(H)).unwrap();
    let old_sig = serde_json::to_value(&u1).unwrap()["sig"].clone();
    let ack = acknowledge(b.receive_update(u1, key(0x61), &ctx(H)).unwrap());
    assert_eq!(b.state_number(), 1);
    assert!(b.pay(1000, None, key(0x62), &ctx(H)).is_err());
    let out = a.receive_ack(ack).unwrap();
    let u2 = a.pay(1000, None, key(0x52), &ctx(H)).unwrap();
    let mut value = serde_json::to_value(&u2).unwrap();
    value["sig"] = old_sig.clone();
    let bad: UpdateMessage = serde_json::from_value(value).unwrap();
    assert!(bad.well_formed().is_ok());
    assert!(matches!(
        b.receive_update(bad.clone(), key(0x63), &ctx(H)).unwrap(),
        ReceiveUpdate::Buffered
    ));
    assert!(b.take_buffered_update().is_none());
    let held = serde_json::to_value(b.snapshot()).unwrap();
    assert_eq!(held["buffered"]["sig"], old_sig);
    let mut b = ChannelMachine::restore(b.snapshot()).unwrap();
    assert!(b.receive_revoke(out.revoke).unwrap().is_some());
    let replay = b.take_buffered_update().unwrap();
    assert_eq!(replay.sig, bad.sig);
    assert!(matches!(
        b.receive_update(replay, key(0x64), &ctx(H)).unwrap(),
        ReceiveUpdate::Rejected(_)
    ));
    assert_eq!(b.state_number(), 1);
    let ack = acknowledge(b.receive_update(u2, key(0x65), &ctx(H)).unwrap());
    let done = a.receive_ack(ack).unwrap();
    assert!(b.receive_revoke(done.revoke).unwrap().is_some());
    assert_eq!(b.state_number(), 2);
    json!({"id":"F2","receiver_advance_before_final_revoke":1,"receiver_proposal_blocked":true,"shape_valid_bad_signature_buffered_and_restored":true,"bad_buffer_rejected_after_finality":true,"valid_control_final_state":2})
}

fn reorganisation() -> Value {
    let (mut a, b, _, _) = open_pair(20_000, 1);
    let tx = b.signed_commitment(0, &[0; 32]).unwrap();
    let spend = ChainSpend {
        txid: tx.compute_txid(),
        height: H + 1,
    };
    let result = a
        .on_spend(spend, &ScriptBuf::new(), &ctx(H + 1), |_, _| {
            OutputLookup::Unspent
        })
        .unwrap();
    assert!(matches!(
        result.kind,
        FundingSpend::RemoteCommitment { revoked: false, .. }
    ));
    assert_ne!(a.status(), ChannelStatus::Open);
    assert!(a.un_spend().is_none());
    assert_eq!(a.status(), ChannelStatus::Open);
    assert!(a.pay(1000, None, key(0x53), &ctx(H + 2)).is_ok());
    let (mut c, _, _, _) = open_pair(20_000, 1);
    let close = c.force_close(None, "audit", &ctx(H)).unwrap();
    c.on_spend(
        ChainSpend {
            txid: close.tx.compute_txid(),
            height: H + 1,
        },
        &ScriptBuf::new(),
        &ctx(H + 1),
        |_, _| OutputLookup::Unspent,
    )
    .unwrap();
    assert_eq!(c.un_spend().unwrap().tx, close.tx);
    assert_ne!(c.status(), ChannelStatus::Open);
    json!({"id":"F3","remote_close_reorg_returns_open":true,"new_proposal_after_reorg":true,"local_close_reorg_rebroadcasts":true,"local_close_stays_closed_out":true})
}

fn revocation() -> Value {
    let (a, _, _, _) = open_pair(20_000, 1);
    let commitment = a.my_commitment(0).unwrap();
    let prevout = commitment.tx.output[commitment.to_local_vout.unwrap() as usize].clone();
    let combined = revocation_key(&key(0x40), &key(0x31)).unwrap();
    let mut accepted = Vec::new();
    for k in [key(0x31), key(0x11), combined] {
        let tx = sweep_to_local(
            &commitment,
            SweepPath::Revocation,
            ScriptBuf::new(),
            Amount::from_sat(300),
            &k,
            SighashRules::KnotsUnified,
            &[0; 32],
        )
        .unwrap();
        accepted.push(
            verify_channel_input(
                &tx,
                0,
                std::slice::from_ref(&prevout),
                SighashRules::KnotsUnified,
            )
            .is_ok(),
        );
    }
    assert_eq!(accepted, vec![false, false, true]);
    let context = format!("{}/a/0", a.id());
    let proof = pop_sign(&key(0x31), &context, &[0; 32]);
    assert!(pop_verify(&public(&key(0x31)), &proof, &context));
    assert!(!pop_verify(
        &public(&key(0x31)),
        &proof,
        &format!("{}/b/0", a.id())
    ));
    assert!(!pop_verify(
        &public(&key(0x31)),
        &proof,
        &format!("{}/a/1", a.id())
    ));
    json!({"id":"F4","owner_state_key_refused":true,"owner_channel_key_refused":true,"counterparty_combined_key_accepted":true,"pop_role_and_state_bound":true,"independent_reference":false})
}
fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"cases":[identity(),finality(),reorganisation(),revocation()]})
        )
        .unwrap()
    );
}
