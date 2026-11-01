//! Offline production-library audit. Fixture helpers copied from sidestr-evm/tests/chain.rs at 9c6de86.
#![allow(dead_code, unused_imports)]
use alloy_consensus::{SignableTransaction, TxEip1559, TxEnvelope, TxLegacy};
use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::{Address, Bytes, TxKind, B256, U256};
use alloy_signer::SignerSync;
use alloy_signer_local::PrivateKeySigner;
use bitcoin::consensus::encode::serialize;
use bitcoin::secp256k1::{Keypair, Message, SecretKey};
use bitcoin::{
    absolute::LockTime, transaction::Version, Amount, Block, ScriptBuf, Sequence, Transaction,
    TxIn, TxOut, Witness,
};
use sidestr_core::block::{
    build_block, challenge_for, pubkey_of, secp, sign_block, BlockTemplate, HeaderFamily, MARKER,
};
use sidestr_core::mirror::encode_record;
use sidestr_core::records::record_script;
use sidestr_core::sighash::key_path_sighash;
use sidestr_core::state::{NextBlock, State};
use sidestr_core::{ChainDocument, Error as CoreError};
use sidestr_evm::records::{carrier_script, deposit_script, root_script, GWEI, WITHDRAW};
use sidestr_evm::{rules_for, Rules, KNOWN};

const CHAIN_ID: u64 = 21_474;

struct Chain {
    key: SecretKey,
    me: ScriptBuf,
    doc: ChainDocument,
    rules: Rules,
    state: State,
    follower_rules: Rules,
    follower: State,
    blocks: Vec<Block>,
    time: u32,
}

fn doc_for(key: &SecretKey, rules: &str) -> ChainDocument {
    let me = challenge_for(&pubkey_of(key));
    let text = format!(
        r#"{{"id": "sidestr:evmtest", "name": "evmtest", "parent": "tbtc4", "challenge": "{me}", "signer": "{pk}",
        "powLimit": "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "addressPrefix": "evt", "genesisTime": 1790000000, "rules": {rules},
        "evm": {{ "chainId": {CHAIN_ID}, "gasLimit": 30000000 }},
        "pegs": [{{ "txid": "{peg}", "vout": 0, "amount": 5000000000, "script": "{me}" }}]}}"#,
        me = me.to_hex_string(),
        pk = pubkey_of(key),
        peg = "a".repeat(64),
    );
    serde_json::from_str(&text).unwrap()
}

impl Chain {
    fn new() -> Self {
        let key = SecretKey::from_slice(&[7u8; 32]).unwrap();
        let me = challenge_for(&pubkey_of(&key));
        let doc = doc_for(&key, r#"["evm"]"#);
        let rules = rules_for(&doc).unwrap();
        let state = State::with_key_and_rules(doc.clone(), &key, rules.boxed()).unwrap();
        let genesis = State::genesis_block_for(&doc, &key).unwrap();
        let follower_rules = rules_for(&doc).unwrap();
        let follower =
            State::from_genesis_with_rules(doc.clone(), &genesis, None, follower_rules.boxed())
                .unwrap();
        Self {
            key,
            me,
            doc,
            rules,
            state,
            follower_rules,
            follower,
            blocks: vec![genesis],
            time: 1_790_000_000,
        }
    }

    fn evm(&self) -> std::sync::MutexGuard<'_, sidestr_evm::EvmState> {
        self.rules.evm.as_ref().unwrap().state()
    }

    fn follower_evm(&self) -> std::sync::MutexGuard<'_, sidestr_evm::EvmState> {
        self.follower_rules.evm.as_ref().unwrap().state()
    }

    /// Produce the next block and hand it to the validator.
    fn produce(&mut self) -> Vec<(bitcoin::Txid, String)> {
        self.time += 60;
        let next = NextBlock {
            time: self.time,
            claims: vec![],
        };
        let made = self
            .rules
            .produce(&mut self.state, &self.key, &next, None)
            .unwrap();
        let (block, dropped) = (made.block, made.dropped);
        self.follower.add_block(&block, None, None).unwrap();
        self.blocks.push(block);
        assert_eq!(self.follower.tip(), self.state.tip());
        assert_eq!(self.follower_evm().root(), self.evm().root());
        dropped
    }

    /// A signed spend of the largest mature coin not in the mempool, paying
    /// `outputs` then change.
    fn spend(&self, outputs: Vec<TxOut>) -> Transaction {
        let reserved: Vec<_> = self
            .state
            .mempool()
            .flat_map(|t| t.input.iter().map(|i| i.previous_output))
            .collect();
        let coin = self
            .state
            .coins(&self.me)
            .into_iter()
            .filter(|c| {
                !reserved.contains(&c.outpoint)
                    && (!c.coinbase || self.state.height() + 1 - c.height >= 100)
            })
            .max_by_key(|c| c.value)
            .expect("a mature coin");
        let total: u64 = outputs.iter().map(|o| o.value.to_sat()).sum();
        let mut output = outputs;
        output.push(TxOut {
            value: Amount::from_sat(coin.value - total - 5_000),
            script_pubkey: self.me.clone(),
        });
        let mut tx = Transaction {
            version: Version::TWO,
            lock_time: LockTime::ZERO,
            input: vec![TxIn {
                previous_output: coin.outpoint,
                script_sig: ScriptBuf::new(),
                sequence: Sequence(0xffff_fffd),
                witness: Witness::new(),
            }],
            output,
        };
        let prevouts = [TxOut {
            value: Amount::from_sat(coin.value),
            script_pubkey: self.me.clone(),
        }];
        let rules = self.state.family().sighash_rules(self.state.height() + 1);
        let (msg, ht) = key_path_sighash(&tx, 0, &prevouts, rules).unwrap();
        let kp = Keypair::from_secret_key(secp(), &self.key);
        let sig = secp().sign_schnorr_with_aux_rand(&Message::from_digest(msg), &kp, &[0; 32]);
        tx.input[0].witness = Witness::from_slice(&[[sig.serialize().as_slice(), &[ht]].concat()]);
        tx
    }

    /// Through the producer's EVM check first, as `siding submit` does, then its mempool.
    fn submit(&mut self, tx: Transaction) -> Result<(), String> {
        let o = self
            .evm()
            .check_tx(&tx, self.state.height() + 1, self.time + 60);
        if let Some(e) = o.error {
            return Err(e);
        }
        self.state.submit(tx).map(|_| ()).map_err(|e| e.to_string())
    }

    /// A block built by hand on the tip: `txs`, a coinbase of `outputs`, sealed.
    fn hand_built(&self, txs: Vec<Transaction>, outputs: Vec<TxOut>) -> Block {
        let tip = self.state.tip();
        let block = build_block(
            self.state.family(),
            &BlockTemplate {
                height: tip.height + 1,
                prev: tip.hash,
                time: tip.time + 1,
                transactions: txs,
                outputs,
                bits: self.state.bits(),
                marker: MARKER.to_string(),
            },
        );
        sign_block(self.state.family(), &block, &self.me, &self.key, &[0; 32]).unwrap()
    }
}

fn signer(byte: u8) -> PrivateKeySigner {
    PrivateKeySigner::from_bytes(&B256::repeat_byte(byte)).unwrap()
}

fn legacy(
    s: &PrivateKeySigner,
    nonce: u64,
    to: TxKind,
    value: U256,
    gas: u64,
    input: &[u8],
) -> Vec<u8> {
    let tx = TxLegacy {
        chain_id: Some(CHAIN_ID),
        nonce,
        gas_price: u128::from(GWEI),
        gas_limit: gas,
        to,
        value,
        input: Bytes::copy_from_slice(input),
    };
    let sig = s.sign_hash_sync(&tx.signature_hash()).unwrap();
    TxEnvelope::from(tx.into_signed(sig)).encoded_2718()
}

fn carrier(rlp: &[u8]) -> TxOut {
    TxOut {
        value: Amount::ZERO,
        script_pubkey: carrier_script(rlp).unwrap(),
    }
}

fn gwei(n: u64) -> U256 {
    U256::from(n) * U256::from(GWEI)
}

fn funded_chain() -> (Chain, PrivateKeySigner) {
    let mut c = Chain::new();
    for _ in 0..101 {
        c.produce();
    }
    let key = signer(0x42);
    let dep = c.spend(vec![
        TxOut {
            value: Amount::from_sat(1_000_000),
            script_pubkey: c.me.clone(),
        },
        TxOut {
            value: Amount::ZERO,
            script_pubkey: deposit_script(key.address()),
        },
    ]);
    c.submit(dep).unwrap();
    c.produce();
    (c, key)
}
fn out(sats: u64, script: ScriptBuf) -> TxOut {
    TxOut {
        value: Amount::from_sat(sats),
        script_pubkey: script,
    }
}
fn withdrawal_multiplicity() -> serde_json::Value {
    let (mut c, key) = funded_chain();
    let target = challenge_for(&pubkey_of(&SecretKey::from_slice(&[9; 32]).unwrap()));
    let tx = c.spend(
        (0..2)
            .map(|nonce| {
                carrier(&legacy(
                    &key,
                    nonce,
                    TxKind::Call(WITHDRAW),
                    gwei(10_000),
                    30_000,
                    target.as_bytes(),
                ))
            })
            .collect(),
    );
    let seq = c.evm().sequence(
        std::slice::from_ref(&tx),
        c.state.height() + 1,
        c.state.tip().time + 1,
    );
    assert!(seq.dropped.is_empty());
    assert_eq!(seq.withdrawals.len(), 2);
    assert_eq!(seq.withdrawals.iter().map(|w| w.sats).sum::<u64>(), 20_000);
    let missing = c.hand_built(
        vec![tx.clone()],
        vec![out(5_000, c.me.clone()), out(0, root_script(seq.root))],
    );
    assert!(!c
        .follower
        .judge(c.follower.height() + 1, &missing, None)
        .0
        .ok());
    let honest = c.hand_built(
        vec![tx.clone()],
        vec![
            out(5_000, c.me.clone()),
            out(10_000, target.clone()),
            out(10_000, target.clone()),
            out(0, root_script(seq.root)),
        ],
    );
    assert!(c
        .follower
        .judge(c.follower.height() + 1, &honest, None)
        .0
        .ok());
    // One target output is reused for both obligations; the unused allowance pays the signer.
    let underpaid = c.hand_built(
        vec![tx],
        vec![
            out(15_000, c.me.clone()),
            out(10_000, target.clone()),
            out(0, root_script(seq.root)),
        ],
    );
    let v = c
        .follower
        .judge(c.follower.height() + 1, &underpaid, None)
        .0;
    assert!(v.ok(), "{:?}", v.failed());
    c.follower.add_block(&underpaid, None, None).unwrap();
    assert_eq!(
        c.follower
            .coins(&target)
            .iter()
            .map(|x| x.value)
            .sum::<u64>(),
        10_000
    );
    assert_eq!(c.follower_evm().nonce(&key.address()), 2);
    assert_eq!(
        c.follower_evm()
            .receipts()
            .filter(|r| r.to == Some(WITHDRAW) && r.status)
            .count(),
        2
    );
    serde_json::json!({"case":"one_coinbase_output_satisfies_two_withdrawals","obligation_sats":20_000,"recipient_paid_sats":10_000,"signer_paid_sats":15_000,"ordinary_fee_sats":5_000,"missing_payment_rejected":true,"two_separate_payments_accepted":true,"underpaid_block_applied_by_full_follower":true,"successful_withdrawal_receipts":2})
}
fn snapshot_cases() -> Vec<serde_json::Value> {
    let (mut c, key) = funded_chain();
    let init = hex::decode("69602a60005260206000f3600052600a6016f3").unwrap();
    let deployment = legacy(&key, 0, TxKind::Create, U256::ZERO, 100_000, &init);
    c.submit(c.spend(vec![carrier(&deployment)])).unwrap();
    c.produce();
    let hash = alloy_primitives::keccak256(&deployment);
    let (config, snapshot, height, root, contract) = {
        let s = c.evm();
        (
            s.config().clone(),
            s.snapshot(),
            s.height(),
            s.root(),
            s.receipt(&hash).unwrap().contract_address.unwrap(),
        )
    };
    let restored = sidestr_evm::EvmState::from_snapshot(config.clone(), &snapshot, height).unwrap();
    let (ok, before) = restored
        .call(key.address(), contract, Bytes::new(), 100_000)
        .unwrap();
    assert!(ok && before.last() == Some(&42));
    let mut bad_root = snapshot.clone();
    bad_root.blocks.get_mut(&height).unwrap().root = B256::repeat_byte(0x11);
    assert!(sidestr_evm::EvmState::from_snapshot(config.clone(), &bad_root, height).is_err());
    let code_hash = restored
        .world()
        .account(&contract)
        .unwrap()
        .code_hash
        .to_string();
    let mut json = serde_json::to_value(&snapshot).unwrap();
    json["worlds"][height.to_string()]["code"][&code_hash] =
        serde_json::json!("0x602b60005260206000f3");
    let changed: sidestr_evm::EvmSnapshot = serde_json::from_value(json).unwrap();
    let tampered = sidestr_evm::EvmState::from_snapshot(config.clone(), &changed, height).unwrap();
    assert_eq!(tampered.root(), root);
    let (ok, after) = tampered
        .call(key.address(), contract, Bytes::new(), 100_000)
        .unwrap();
    assert!(ok && after.last() == Some(&43));
    let mut metadata = snapshot.clone();
    metadata.receipts[0].status = false;
    metadata.receipts[0].gas_used = 1;
    metadata.times.insert(height, 1);
    let metadata_state =
        sidestr_evm::EvmState::from_snapshot(config.clone(), &metadata, height).unwrap();
    assert!(!metadata_state.receipt(&hash).unwrap().status);
    assert_eq!(metadata_state.receipt(&hash).unwrap().gas_used, 1);
    assert_eq!(metadata_state.snapshot().times[&height], 1);
    assert_eq!(metadata_state.root(), root);
    let rewound = sidestr_evm::EvmState::from_snapshot(config, &snapshot, 0).unwrap();
    assert_eq!(rewound.height(), 0);
    assert!(rewound.balance(&key.address()).is_zero());
    assert!(rewound.receipt(&hash).is_some());
    assert!(rewound.block(height).is_some());
    assert_eq!(rewound.receipts().count(), 1);
    vec![
        serde_json::json!({"case":"snapshot_code_changed_without_root_change","normal_call_result":42,"tampered_call_result":43,"same_root":true,"mismatched_root_control_rejected":true}),
        serde_json::json!({"case":"snapshot_receipt_and_time_not_authenticated","changed_receipt_status":false,"changed_gas_used":1,"changed_time":1,"same_root":true}),
        serde_json::json!({"case":"restore_genesis_retains_future_history","restored_height":0,"balance_at_genesis_zero":true,"future_receipt_height":height,"future_block_and_receipt_still_exposed":true}),
    ]
}
fn cache_lifetime() -> serde_json::Value {
    let mut c = Chain::new();
    let mut hashes = Vec::new();
    for i in 1..=64 {
        let b = c.hand_built(
            vec![],
            vec![out(i, c.me.clone()), out(0, root_script(c.evm().root()))],
        );
        let hash = b.block_hash();
        assert!(!c.follower.judge(1, &b, None).0.ok());
        assert!(c.follower_evm().verdict(&hash).unwrap().ok);
        hashes.push(hash);
    }
    let debug_before = format!("{:?}", *c.follower_evm());
    for hash in &hashes {
        assert!(debug_before.contains(&hash.to_string()));
    }
    c.produce();
    assert_eq!(c.follower_evm().height(), 1);
    assert!(format!("{:?}", *c.follower_evm()).contains("pending: []"));
    for hash in &hashes {
        assert!(c.follower_evm().verdict(hash).is_some());
    }
    serde_json::json!({"case":"candidate_verdicts_outlive_rejection_and_commit","rejected_outer_candidates":64,"pending_cleared_on_commit":true,"all_verdicts_still_queryable_after_commit":true,"heap_bytes_or_network_exposure_measured":false})
}
fn gas_profile() -> serde_json::Value {
    let (c, key) = funded_chain();
    let mut snapshot = c.evm().snapshot();
    snapshot.config.gas_limit = 10_000;
    let state =
        sidestr_evm::EvmState::from_snapshot(snapshot.config.clone(), &snapshot, c.state.height())
            .unwrap();
    let tx = c.spend(vec![carrier(&legacy(
        &key,
        0,
        TxKind::Call(Address::repeat_byte(0x77)),
        U256::from(1),
        30_000,
        &[],
    ))]);
    let result = state.check_tx(&tx, c.state.height() + 1, c.time + 60);
    assert!(result.ok);
    assert_eq!(result.gas_used, 21_000);
    serde_json::json!({"case":"configured_block_gas_limit_is_not_enforced","configured_gas_limit":10_000,"transaction_gas_limit":30_000,"actual_gas_used":21_000,"accepted":true,"scope":"Documented compatibility profile; no stress or denial-of-service test"})
}

fn main() {
    let mut cases = vec![withdrawal_multiplicity()];
    cases.extend(snapshot_cases());
    cases.push(cache_lifetime());
    cases.push(gas_profile());
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"cases":cases})).unwrap()
    );
}
