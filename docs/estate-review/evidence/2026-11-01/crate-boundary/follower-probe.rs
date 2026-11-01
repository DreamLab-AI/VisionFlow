use bitcoin::consensus::encode::serialize;
use bitcoin::secp256k1::SecretKey;
use sidestr_agent::hitch::follow::{Advance, ChainFollower, Plan};
use sidestr_core::block::{challenge_for, pubkey_of};
use sidestr_core::blockfile::{Entry, Index};
use sidestr_core::document::ChainDocument;
use sidestr_core::mirror::encode_record;
use sidestr_core::state::State;

fn doc(name: &str, key: &SecretKey) -> ChainDocument {
    ChainDocument::from_json(
        &serde_json::json!({
            "id":format!("sidestr:{name}"),"name":name,"parent":"tbtc4",
            "challenge":challenge_for(&pubkey_of(key)).to_hex_string(),
            "powLimit":"7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "addressPrefix":"ex","genesisTime":1790000000,"pegs":[]
        })
        .to_string(),
    )
    .unwrap()
}
fn index(name: &str, hash: String, size: usize) -> Index {
    Index {
        network: name.into(),
        from: 0,
        to: 0,
        blocks: vec![Entry {
            height: 0,
            hash,
            offset: 0,
            size: size as u32,
        }],
    }
}
fn main() {
    let key = SecretKey::from_slice(&[7u8; 32]).unwrap();
    let mut a = doc("audit-a", &key);
    let ga = State::genesis_block_for(&a, &key).unwrap();
    a.genesis_hash = Some(ga.block_hash().to_string());
    let ba = serialize(&ga);
    let ra = encode_record(0, &ba);
    let ia = index(&a.id, ga.block_hash().to_string(), ba.len());
    assert!(State::from_genesis(a.clone(), &ga, None).is_ok());
    let mut follower = ChainFollower::new(&a).unwrap();
    assert!(matches!(
        follower.plan(&ia).unwrap(),
        Plan::Fetch { from: 0, .. }
    ));
    follower.apply(&ia, 0, &ra).unwrap();
    assert_eq!(follower.hash_at(0), Some(ga.block_hash()));

    let mut wrong_pin = a.clone();
    wrong_pin.genesis_hash = Some("11".repeat(32));
    assert!(State::from_genesis(wrong_pin.clone(), &ga, None).is_err());
    assert!(ChainFollower::new(&wrong_pin)
        .unwrap()
        .apply(&ia, 0, &ra)
        .is_ok());

    let mut b = doc("audit-b", &key);
    let gb = State::genesis_block_for(&b, &key).unwrap();
    b.genesis_hash = Some(gb.block_hash().to_string());
    let bb = serialize(&gb);
    let rb = encode_record(0, &bb);
    let ib = index(&b.id, gb.block_hash().to_string(), bb.len());
    assert_ne!(ga.block_hash(), gb.block_hash());
    assert!(State::from_genesis(b.clone(), &gb, None).is_ok());
    assert!(State::from_genesis(a.clone(), &gb, None).is_err());
    assert!(ChainFollower::new(&a).unwrap().apply(&ib, 0, &rb).is_ok());
    assert!(matches!(
        follower.plan(&ib).unwrap(),
        Plan::Fetch { from: 0, .. }
    ));
    assert!(matches!(
        follower.apply(&ib, 0, &rb).unwrap(),
        Advance::Reorg { fork: 0, .. }
    ));
    assert_eq!(follower.hash_at(0), Some(gb.block_hash()));

    let wrong_name = index("sidestr:unrelated", ga.block_hash().to_string(), ba.len());
    assert!(ChainFollower::new(&a)
        .unwrap()
        .apply(&wrong_name, 0, &ra)
        .is_ok());
    let bad_hash = index(&a.id, "22".repeat(32), ba.len());
    assert!(ChainFollower::new(&a)
        .unwrap()
        .apply(&bad_hash, 0, &ra)
        .is_err());
    let other = doc("audit-a", &SecretKey::from_slice(&[9u8; 32]).unwrap());
    assert!(ChainFollower::new(&other)
        .unwrap()
        .apply(&ia, 0, &ra)
        .is_err());
    println!(
        "{}",
        serde_json::json!({"cases":[
        "matching pinned genesis accepted by validator and follower",
        "wrong genesis pin rejected by validator but accepted by follower",
        "another same-signer chain genesis rejected by validator but accepted by follower",
        "existing follower replaces genesis at fork zero with same-signer foreign chain",
        "index network label not checked",
        "incorrect index block hash rejected",
        "different signer challenge rejected"
    ],"scope":"synthetic keys and genesis blocks, actual library APIs; no network or operational state"})
    );
}
