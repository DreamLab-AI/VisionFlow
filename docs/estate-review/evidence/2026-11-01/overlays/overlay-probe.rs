//! Offline audit probes using the unchanged test helpers from sidestr-evm/tests/pools_markets.rs at 9c6de86.
//! Allow dead helpers/imports retained from that fixture; assertions call production libraries.
#![allow(dead_code, unused_imports)]

use bitcoin::secp256k1::{Keypair, Message, SecretKey};
use bitcoin::{
    absolute::LockTime, transaction::Version, Amount, Block, OutPoint, ScriptBuf, Sequence,
    Transaction, TxIn, TxOut, Txid, Witness,
};
use sidestr_core::block::{
    build_block, challenge_for, pubkey_of, secp, sign_block, BlockTemplate, HeaderFamily, MARKER,
};
use sidestr_core::document::RuleEntry;
use sidestr_core::markets::{no_id_of, MarketStatus};
use sidestr_core::mirror::encode_record;
use sidestr_core::records::record_script;
use sidestr_core::sighash::key_path_sighash;
use sidestr_core::state::{CoinRef, NextBlock, State};
use sidestr_core::{ChainDocument, Error as CoreError};
use sidestr_evm::{rules_for, Rules};

#[derive(Clone)]
struct SpendInput {
    outpoint: OutPoint,
    prevout: TxOut,
    key: Option<SecretKey>,
}

struct TestChain {
    key: SecretKey,
    resolver: SecretKey,
    me: ScriptBuf,
    resolver_script: ScriptBuf,
    doc: ChainDocument,
    rules: Rules,
    state: State,
    blocks: Vec<Block>,
    time: u32,
}

impl TestChain {
    fn new() -> Self {
        let key = SecretKey::from_slice(&[7; 32]).unwrap();
        let resolver = SecretKey::from_slice(&[9; 32]).unwrap();
        let me = challenge_for(&pubkey_of(&key));
        let resolver_script = challenge_for(&pubkey_of(&resolver));
        let doc: ChainDocument = serde_json::from_value(serde_json::json!({
            "id": "sidestr:markets-rust",
            "name": "markets-rust",
            "parent": "tbtc4",
            "challenge": me.to_hex_string(),
            "signer": pubkey_of(&key).to_string(),
            "powLimit": "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "addressPrefix": "pm",
            "genesisTime": 1_790_000_000,
            "rules": ["assets", "pool", { "name": "markets", "from": 0 }],
            "pegs": [{
                "txid": "aa".repeat(32), "vout": 0,
                "amount": 5_000_000_000u64, "script": me.to_hex_string()
            }]
        }))
        .unwrap();
        let rules = rules_for(&doc).unwrap();
        let state = State::with_key_and_rules(doc.clone(), &key, rules.boxed()).unwrap();
        let genesis = State::genesis_block_for(&doc, &key).unwrap();
        Self {
            key,
            resolver,
            me,
            resolver_script,
            doc,
            rules,
            state,
            blocks: vec![genesis],
            time: 1_790_000_000,
        }
    }

    fn produce(&mut self) -> Block {
        self.time += 60;
        let made = self
            .rules
            .produce(
                &mut self.state,
                &self.key,
                &NextBlock {
                    time: self.time,
                    claims: vec![],
                },
                None,
            )
            .unwrap();
        assert!(made.dropped.is_empty(), "{:?}", made.dropped);
        self.blocks.push(made.block.clone());
        made.block
    }

    fn mature(&mut self) {
        while self.state.height() < 101 {
            self.produce();
        }
    }

    fn plain_coin(&self, script: &ScriptBuf) -> CoinRef {
        let assets = self.rules.assets.as_ref().unwrap().view();
        self.state
            .coins(script)
            .into_iter()
            .filter(|coin| {
                (!coin.coinbase || self.state.height() + 1 - coin.height >= 100)
                    && assets.carried(&coin.outpoint).is_none()
            })
            .max_by_key(|coin| coin.value)
            .expect("a plain spendable coin")
    }

    fn owned_input(&self, coin: &CoinRef, script: &ScriptBuf, key: SecretKey) -> SpendInput {
        SpendInput {
            outpoint: coin.outpoint,
            prevout: TxOut {
                value: Amount::from_sat(coin.value),
                script_pubkey: script.clone(),
            },
            key: Some(key),
        }
    }

    fn known_input(
        &self,
        outpoint: OutPoint,
        output: &TxOut,
        key: Option<SecretKey>,
    ) -> SpendInput {
        SpendInput {
            outpoint,
            prevout: output.clone(),
            key,
        }
    }

    fn spend(
        &self,
        inputs: Vec<SpendInput>,
        mut outputs: Vec<TxOut>,
        change_script: &ScriptBuf,
    ) -> Transaction {
        let input_value: u64 = inputs
            .iter()
            .map(|input| input.prevout.value.to_sat())
            .sum();
        let output_value: u64 = outputs.iter().map(|output| output.value.to_sat()).sum();
        let change = input_value
            .checked_sub(output_value + 5_000)
            .expect("inputs cover outputs and fee");
        if change > 0 {
            outputs.push(TxOut {
                value: Amount::from_sat(change),
                script_pubkey: change_script.clone(),
            });
        }
        let mut tx = Transaction {
            version: Version::TWO,
            lock_time: LockTime::ZERO,
            input: inputs
                .iter()
                .map(|input| TxIn {
                    previous_output: input.outpoint,
                    script_sig: ScriptBuf::new(),
                    sequence: Sequence(0xffff_fffd),
                    witness: Witness::new(),
                })
                .collect(),
            output: outputs,
        };
        let prevouts: Vec<TxOut> = inputs.iter().map(|input| input.prevout.clone()).collect();
        let sighash = self.state.family().sighash_rules(self.state.height() + 1);
        for (index, input) in inputs.iter().enumerate() {
            let Some(key) = input.key else { continue };
            let (message, hash_type) = key_path_sighash(&tx, index, &prevouts, sighash).unwrap();
            let keypair = Keypair::from_secret_key(secp(), &key);
            let signature = secp().sign_schnorr_with_aux_rand(
                &Message::from_digest(message),
                &keypair,
                &[0; 32],
            );
            tx.input[index].witness =
                Witness::from_slice(&[[signature.serialize().as_slice(), &[hash_type]].concat()]);
        }
        tx
    }

    fn submit_and_produce(&mut self, tx: Transaction) -> Txid {
        let txid = tx.compute_txid();
        self.state.submit(tx).unwrap();
        self.produce();
        txid
    }

    fn candidate(&self, tx: Transaction) -> Block {
        let tip = self.state.tip();
        let block = build_block(
            self.state.family(),
            &BlockTemplate {
                height: tip.height + 1,
                prev: tip.hash,
                time: tip.time + 1,
                transactions: vec![tx],
                outputs: vec![TxOut {
                    value: Amount::from_sat(5_000),
                    script_pubkey: self.me.clone(),
                }],
                bits: self.state.bits(),
                marker: MARKER.into(),
            },
        );
        sign_block(self.state.family(), &block, &self.me, &self.key, &[0; 32]).unwrap()
    }
}

fn pay(value: u64, script: &ScriptBuf) -> TxOut {
    TxOut {
        value: Amount::from_sat(value),
        script_pubkey: script.clone(),
    }
}

fn op_true(value: u64) -> TxOut {
    TxOut {
        value: Amount::from_sat(value),
        script_pubkey: ScriptBuf::from_bytes(vec![0x51]),
    }
}

fn record(text: &str) -> TxOut {
    TxOut {
        value: Amount::ZERO,
        script_pubkey: record_script(text).unwrap(),
    }
}

fn good_payment(chain: &TestChain) -> Transaction {
    let coin = chain.plain_coin(&chain.me);
    chain.spend(
        vec![chain.owned_input(&coin, &chain.me, chain.key)],
        vec![pay(1000, &chain.me)],
        &chain.me,
    )
}

fn clock_failure(composed: bool) -> serde_json::Value {
    let mut chain = TestChain::new();
    chain.mature();
    let tx = good_payment(&chain);
    let candidate = chain.candidate(tx.clone());
    assert!(chain
        .state
        .judge(chain.state.height() + 1, &candidate, None)
        .0
        .ok());
    chain.state.submit(tx.clone()).unwrap();
    let tip = chain.state.tip();
    let error = if composed {
        chain
            .rules
            .produce(&mut chain.state, &chain.key, &NextBlock::default(), Some(1))
            .unwrap_err()
            .to_string()
    } else {
        chain
            .state
            .produce(&chain.key, &NextBlock::default(), Some(1))
            .unwrap_err()
            .to_string()
    };
    assert_eq!(chain.state.tip(), tip);
    assert_eq!(chain.state.mempool().count(), 0);
    assert!(chain.state.judge(tip.height + 1, &candidate, None).0.ok());
    let replay_error = chain.state.submit(tx).unwrap_err().to_string();
    assert!(
        replay_error.contains("rejected this session: btc:rule-header-time-future"),
        "{replay_error}"
    );
    serde_json::json!({"case":if composed {"composed_producer_clock_failure_evicts_valid_transaction"} else {"core_producer_clock_failure_evicts_valid_transaction"},"candidate_valid_before_and_after":true,"tip_unchanged":true,"mempool_after":0,"production_error":error,"exact_replay_error":replay_error})
}

fn refund_one_side() -> serde_json::Value {
    let mut chain = TestChain::new();
    chain.mature();
    let expiry = chain.state.height() + 5;
    let plain = chain.plain_coin(&chain.me);
    let opening = chain.spend(
        vec![chain.owned_input(&plain, &chain.me, chain.key)],
        vec![
            op_true(1),
            record(&format!(
                "market:self:0:{}:{expiry}:1",
                pubkey_of(&chain.resolver)
            )),
            record("question:Synthetic offline audit?"),
        ],
        &chain.me,
    );
    let market = chain.submit_and_produce(opening);
    let no = no_id_of(&market);
    let plain = chain.plain_coin(&chain.me);
    let split = chain.spend(
        vec![
            chain.owned_input(&plain, &chain.me, chain.key),
            chain.known_input(
                OutPoint {
                    txid: market,
                    vout: 0,
                },
                &op_true(1),
                None,
            ),
        ],
        vec![
            op_true(1001),
            pay(1000, &chain.me),
            pay(1000, &chain.me),
            record(&format!("split:{market}:0")),
            record(&format!("tally:{market}:1=1000")),
            record(&format!("tally:{no}:2=1000")),
        ],
        &chain.me,
    );
    let split_id = chain.submit_and_produce(split);
    while chain.state.height() <= expiry + 1 {
        chain.produce();
    }
    let plain = chain.plain_coin(&chain.me);
    let inputs = vec![
        chain.owned_input(&plain, &chain.me, chain.key),
        chain.known_input(
            OutPoint {
                txid: split_id,
                vout: 0,
            },
            &op_true(1001),
            None,
        ),
        chain.known_input(
            OutPoint {
                txid: split_id,
                vout: 1,
            },
            &pay(1000, &chain.me),
            Some(chain.key),
        ),
    ];
    // 1,000 YES may release at most 500 sats. Reject 501 before trying 500.
    let excessive = chain.spend(
        inputs.clone(),
        vec![op_true(500), record(&format!("redeem:{market}:0"))],
        &chain.me,
    );
    let tip = chain.state.tip();
    let rejected = chain
        .state
        .add_block(&chain.candidate(excessive), None, None)
        .unwrap_err()
        .to_string();
    assert!(rejected.contains("sidestr:rule-markets"));
    assert_eq!(chain.state.tip(), tip);
    assert_eq!(
        chain
            .rules
            .markets
            .as_ref()
            .unwrap()
            .view()
            .get(&market)
            .unwrap()
            .collateral,
        1001
    );
    assert_eq!(
        chain
            .rules
            .assets
            .as_ref()
            .unwrap()
            .view()
            .carried(&OutPoint {
                txid: split_id,
                vout: 1
            })
            .unwrap()[&market],
        1000
    );
    let refund = chain.spend(
        inputs,
        vec![op_true(501), record(&format!("redeem:{market}:0"))],
        &chain.me,
    );
    let refund_id = chain.submit_and_produce(refund);
    let v = chain.rules.markets.as_ref().unwrap().view();
    assert_eq!(v.get(&market).unwrap().status, MarketStatus::Refunding);
    assert_eq!(v.get(&market).unwrap().collateral, 501);
    assert_eq!(
        chain
            .rules
            .assets
            .as_ref()
            .unwrap()
            .view()
            .carried(&OutPoint {
                txid: split_id,
                vout: 2
            })
            .unwrap()[&no],
        1000
    );
    // Refunding is not closed to new split transactions.
    let plain = chain.plain_coin(&chain.me);
    let resplit = chain.spend(
        vec![
            chain.owned_input(&plain, &chain.me, chain.key),
            chain.known_input(
                OutPoint {
                    txid: refund_id,
                    vout: 0,
                },
                &op_true(501),
                None,
            ),
        ],
        vec![
            op_true(601),
            pay(1000, &chain.me),
            pay(1000, &chain.me),
            record(&format!("split:{market}:0")),
            record(&format!("tally:{market}:1=100")),
            record(&format!("tally:{no}:2=100")),
        ],
        &chain.me,
    );
    chain.submit_and_produce(resplit);
    let v = chain.rules.markets.as_ref().unwrap().view();
    assert_eq!(v.get(&market).unwrap().status, MarketStatus::Refunding);
    assert_eq!(v.get(&market).unwrap().collateral, 601);
    serde_json::json!({"case":"one_sided_refund_and_split_while_refunding","yes_burned":1000,"no_burned":0,"refund_sats":500,"excessive_refund_rejected":rejected,"rejected_candidate_preserved_applied_asset_and_market_views":true,"remaining_no":1000,"resplit_sats":100,"final_collateral":601,"final_status":"Refunding"})
}

fn pending_candidates() -> serde_json::Value {
    let mut chain = TestChain::new();
    chain.mature();
    let tip = chain.state.tip();
    let coin = chain.plain_coin(&chain.me);
    let mut hashes = Vec::new();
    for i in 0..64 {
        let tx = chain.spend(
            vec![chain.owned_input(&coin, &chain.me, chain.key)],
            vec![pay(1000 + i, &chain.me)],
            &chain.me,
        );
        let candidate = chain.candidate(tx);
        hashes.push(candidate.block_hash().to_string());
        let error = chain
            .state
            .add_block(&candidate, None, Some(1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("btc:rule-header-time-future"));
    }
    assert_eq!(chain.state.tip(), tip);
    // Inspect derived Debug output of these synthetic rule instances only.
    // This establishes retention, not heap-byte usage or a public attack route.
    for cache in [
        format!("{:?}", chain.rules.assets),
        format!("{:?}", chain.rules.pool),
        format!("{:?}", chain.rules.markets),
    ] {
        for hash in &hashes {
            assert!(cache.contains(hash), "missing candidate {hash}");
        }
    }
    chain.produce();
    for cache in [
        format!("{:?}", chain.rules.assets),
        format!("{:?}", chain.rules.pool),
        format!("{:?}", chain.rules.markets),
    ] {
        for hash in &hashes {
            assert!(
                !cache.contains(hash),
                "candidate retained after application {hash}"
            );
        }
    }
    serde_json::json!({"case":"rejected_candidates_retained_until_application","distinct_rejected_candidates":64,"all_three_rules_retained_every_hash":true,"tip_unchanged_during_rejection":true,"application_cleared_candidate_hashes":true,"heap_bytes_or_network_reachability_measured":false})
}

fn main() {
    let cases = vec![
        clock_failure(false),
        clock_failure(true),
        refund_one_side(),
        pending_candidates(),
    ];
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"cases":cases})).unwrap()
    );
}
