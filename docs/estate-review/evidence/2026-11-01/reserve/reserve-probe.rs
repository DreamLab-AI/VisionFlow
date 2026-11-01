//! Offline audit of attestation policy boundaries; every key and snapshot is synthetic.
use lwk_wollet::elements::{BlockHash, OutPoint};
use sidestr_bridge_liquid::{
    attest as liquid_attest, reserve_asset, ChainTip, ReserveSnapshot, ReserveUtxo,
};
use sidestr_reserve::secp256k1::{Keypair, Secp256k1, SecretKey};
use sidestr_reserve::serde_json::{json, Value};
use sidestr_reserve::{
    attest, AttestationSigner, Credit, KeypairSigner, Origin, SignedAttestation, Tip,
    MAX_SAFE_INTEGER,
};
use std::str::FromStr;

fn signer(byte: u8) -> KeypairSigner {
    KeypairSigner::new(Keypair::from_secret_key(
        &Secp256k1::new(),
        &SecretKey::from_slice(&[byte; 32]).unwrap(),
    ))
}
fn snapshot(height: Option<u32>) -> ReserveSnapshot {
    ReserveSnapshot {
        tip: ChainTip {
            height: 100,
            hash: BlockHash::from_str(&"11".repeat(32)).unwrap(),
        },
        utxos: vec![ReserveUtxo {
            outpoint: OutPoint::from_str(&format!("{}:0", "22".repeat(32))).unwrap(),
            asset: reserve_asset(),
            value: 25,
            height,
        }],
        source: "https://fixture.invalid/api".into(),
    }
}
fn main() {
    let mut results: Vec<Value> = Vec::new();
    let mut check = |name: &str, observed: bool, kind: &str| {
        assert!(observed, "{name}");
        results.push(json!({"case":name,"assertion_passed":true,"kind":kind}));
    };
    let one = liquid_attest(&snapshot(Some(100)), &reserve_asset(), 1).unwrap();
    check(
        "output at tip is counted with one confirmation",
        one.amount == 25 && one.credits.len() == 1,
        "policy-gap",
    );
    let six = liquid_attest(&snapshot(Some(95)), &reserve_asset(), 1).unwrap();
    check(
        "one and six confirmation snapshots produce identical statement",
        one == six,
        "policy-gap",
    );
    check(
        "unconfirmed output excluded",
        liquid_attest(&snapshot(None), &reserve_asset(), 1)
            .unwrap()
            .amount
            == 0,
        "control",
    );
    check(
        "future-height output excluded",
        liquid_attest(&snapshot(Some(101)), &reserve_asset(), 1)
            .unwrap()
            .amount
            == 0,
        "control",
    );
    let mut duplicate = snapshot(Some(100));
    duplicate.utxos.push(duplicate.utxos[0]);
    check(
        "duplicate outpoint refused",
        liquid_attest(&duplicate, &reserve_asset(), 1).is_err(),
        "control",
    );
    let a = signer(0x42);
    let b = signer(0x43);
    let signed = SignedAttestation::sign(one.clone(), &a).unwrap();
    check(
        "normal signature verifies",
        signed.verify().is_ok(),
        "control",
    );
    let mut after = signed.clone();
    after.attestation.amount = u128::MAX;
    check(
        "post-signature amount change refused",
        after.verify().is_err(),
        "control",
    );
    let mut before = one.clone();
    before.amount = u128::MAX;
    let inflated = SignedAttestation::sign(before, &a).unwrap();
    check(
        "pre-signature amount mutation is signable and verifies",
        inflated.verify().is_ok() && inflated.attestation.credits == one.credits,
        "trusted-signer-boundary",
    );
    let other = SignedAttestation::sign(one.clone(), &b).unwrap();
    check(
        "different signer verifies against its own carried key",
        other.verify().is_ok() && other.public_key != a.x_only_public_key(),
        "consumer-policy-boundary",
    );
    for (name, time) in [
        ("time zero is signable", 0),
        ("maximum safe future time is signable", MAX_SAFE_INTEGER),
    ] {
        let mut x = one.clone();
        x.time = time;
        check(
            name,
            SignedAttestation::sign(x, &a).unwrap().verify().is_ok(),
            "consumer-policy-boundary",
        );
    }
    check(
        "same signed statement verifies twice without replay state",
        signed.verify().is_ok() && signed.verify().is_ok(),
        "consumer-policy-boundary",
    );
    let empty = liquid_attest(&snapshot(None), &reserve_asset(), 1).unwrap();
    check(
        "zero empty reserve can be signed",
        SignedAttestation::sign(empty, &a).unwrap().verify().is_ok(),
        "consumer-policy-boundary",
    );
    // Amount distribution is discarded; the total and ids, rather than per-credit amounts, are committed.
    let make = |x, y| {
        attest(
            Origin::new("fixture", "asset", 8).unwrap(),
            Tip::new(100, &"11".repeat(32)).unwrap(),
            &[
                Credit::new("a:0", x).unwrap(),
                Credit::new("b:0", y).unwrap(),
            ],
            "fixture",
            1,
        )
        .unwrap()
    };
    check(
        "same ids and total hide different per-credit allocation",
        make(10, 20).canonical_json().unwrap() == make(20, 10).canonical_json().unwrap(),
        "consumer-policy-boundary",
    );
    println!("{}", sidestr_reserve::serde_json::to_string_pretty(&json!({"cases":results,"limits":"Production libraries, synthetic snapshots and fixed test-only keys; no network, reserve, bridge consumer, funds or deployment exercised."})).unwrap());
}
