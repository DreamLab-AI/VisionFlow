//! Local recovery/HTTP fixtures. Transport decodes transactions; it is not a consensus node.
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::{json, Value};
use solid_pod_rs::{
    bitcoin_tx::{decode_tx_info, MempoolBroadcast, TxoVoucher},
    blocktrail::verify_blocktrail,
    mrc20::{MempoolLookup, TxInfo, TxOut, Utxo},
    payments::PaymentError,
    provenance::BlockAnchorer,
    storage::{fs::FsBackend, Storage},
};
use solid_pod_rs_server::{
    mempool::{GitmarkAnchorer, MempoolHttpClient, MempoolTxPresence},
    trail_store::{gitmark_blocktrails_path, gitmark_intent_path, load_gitmark_trail},
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[derive(Clone, Default)]
struct Fixture {
    txs: Arc<Mutex<HashMap<String, TxInfo>>>,
    sent: Arc<Mutex<Vec<String>>>,
    drop_next: Arc<AtomicBool>,
}
#[async_trait(?Send)]
impl MempoolLookup for Fixture {
    async fn address_utxos(&self, _: &str) -> Result<Vec<Utxo>, PaymentError> {
        Ok(vec![])
    }
    async fn tx(&self, txid: &str) -> Result<TxInfo, PaymentError> {
        self.txs
            .lock()
            .unwrap()
            .get(txid)
            .cloned()
            .ok_or_else(|| PaymentError::InvalidState("fixture not found".into()))
    }
}
#[async_trait(?Send)]
impl MempoolBroadcast for Fixture {
    async fn broadcast_tx(&self, raw: &str) -> Result<String, PaymentError> {
        self.sent.lock().unwrap().push(raw.into());
        if self.drop_next.swap(false, Ordering::SeqCst) {
            return Err(PaymentError::InvalidState(
                "synthetic transport failure".into(),
            ));
        }
        let info = decode_tx_info(raw)?;
        let id = info.txid.clone();
        self.txs.lock().unwrap().insert(id.clone(), info);
        Ok(id)
    }
}
#[async_trait(?Send)]
impl MempoolTxPresence for Fixture {
    async fn transaction_exists(&self, id: &str) -> Result<bool, PaymentError> {
        Ok(self.txs.lock().unwrap().contains_key(id))
    }
}
async fn recovery_case(case: &str) -> Value {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> =
        Arc::new(FsBackend::new(tmp.path().to_path_buf()).await.unwrap());
    let f = Fixture::default();
    let funding = "11".repeat(32);
    f.txs.lock().unwrap().insert(
        funding.clone(),
        TxInfo {
            txid: funding.clone(),
            vin: vec![],
            vout: vec![TxOut {
                value: 100000,
                scriptpubkey: Some(
                    "512079be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798".into(),
                ),
                scriptpubkey_address: None,
            }],
            confirmed: true,
            block_height: Some(1),
        },
    );
    let voucher = TxoVoucher {
        txid: funding,
        vout: 0,
        amount: 100000,
        privkey: format!("{:064x}", 1),
    };
    let a = GitmarkAnchorer::with_storage(f.clone(), storage.clone());
    let first = "a".repeat(40);
    let second = "b".repeat(40);
    let forged = "d".repeat(40);
    a.genesis("prov", &voucher, &first, "testnet4", "")
        .await
        .unwrap();
    let public = gitmark_blocktrails_path("prov");
    let intent = gitmark_intent_path("prov");
    if case == "public_save_failure" {
        let p = tmp.path().join(public.trim_start_matches('/'));
        std::fs::remove_file(&p).unwrap();
        std::fs::create_dir(&p).unwrap();
        let mark = a.anchor("prov", &second, "testnet4").await.unwrap();
        assert!(mark.blockheight.is_none());
        assert!(storage.exists(&intent).await.unwrap());
        assert_eq!(
            load_gitmark_trail(&storage, "prov")
                .await
                .unwrap()
                .unwrap()
                .trail
                .txo
                .len(),
            2
        );
        std::fs::remove_dir(&p).unwrap();
        let fresh: Arc<dyn Storage> =
            Arc::new(FsBackend::new(tmp.path().to_path_buf()).await.unwrap());
        let restarted = GitmarkAnchorer::with_storage(f.clone(), fresh.clone());
        let err = restarted
            .anchor("prov", &"c".repeat(40), "mainnet")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("network mismatch"));
        assert!(!fresh.exists(&intent).await.unwrap());
        assert!(!p.exists());
        return json!({"case":case,"anchor_returned_success":true,"private_trail_marks":2,"pending_intent_before_recovery":true,"recovery_cleared_intent":true,"public_proof_repaired":false,"next_operation_error":err.to_string(),"limits":"Fresh FsBackend/anchorer over same files, not OS restart; mismatch request isolates recovery from a new advance."});
    }
    f.drop_next.store(true, Ordering::SeqCst);
    let err = a.anchor("prov", &second, "testnet4").await.unwrap_err();
    assert!(err.to_string().contains("anchor pending"));
    let (bytes, _) = storage.get(&intent).await.unwrap();
    let mut pending: Value = serde_json::from_slice(&bytes).unwrap();
    if case == "tampered_trail" {
        let mut stored: solid_pod_rs_server::trail_store::StoredGitmarkTrail =
            serde_json::from_value(pending["stored"].clone()).unwrap();
        assert_eq!(stored.trail.txo[1].commit.as_deref(), Some(second.as_str()));
        stored.trail.txo[1].commit = Some(forged.clone());
        assert_eq!(stored.trail.states[1], json!(second));
        stored.trail.states[1] = json!(forged);
        pending["stored"] = serde_json::to_value(stored).unwrap();
    }
    if case == "wrong_txid" {
        pending["txid"] = json!("f".repeat(64));
    }
    storage
        .put(
            &intent,
            Bytes::from(serde_json::to_vec(&pending).unwrap()),
            "application/json",
        )
        .await
        .unwrap();
    let raw = pending["rawHex"].as_str().unwrap().to_string();
    let fresh: Arc<dyn Storage> = Arc::new(FsBackend::new(tmp.path().to_path_buf()).await.unwrap());
    let restarted = GitmarkAnchorer::with_storage(f.clone(), fresh.clone());
    let result = restarted
        .anchor("prov", &"c".repeat(40), "mainnet")
        .await
        .unwrap_err();
    let saved = load_gitmark_trail(&fresh, "prov").await.unwrap().unwrap();
    let kept = fresh.exists(&intent).await.unwrap();
    let sent = f.sent.lock().unwrap().clone();
    if case == "wrong_txid" {
        assert!(result.to_string().contains("refused"));
        assert_eq!(saved.trail.txo.len(), 1);
        assert!(kept);
        assert_eq!(sent.len(), 2);
    } else {
        assert!(result.to_string().contains("network mismatch"));
        assert_eq!(saved.trail.txo.len(), 2);
        assert!(!kept);
        assert_eq!(sent.last().unwrap(), &raw);
    }
    let report = verify_blocktrail(&saved.trail, &f).await;
    let mut extension_refused = false;
    if case == "tampered_trail" {
        assert_eq!(saved.trail.txo[1].commit.as_deref(), Some(forged.as_str()));
        assert!(!report.is_intact());
        let extension = restarted
            .anchor("prov", &"c".repeat(40), "testnet4")
            .await
            .unwrap_err();
        assert!(extension.to_string().contains("refusing to extend"));
        assert_eq!(f.sent.lock().unwrap().len(), 3);
        extension_refused = true;
        let (public_bytes, _) = fresh.get(&public).await.unwrap();
        let published: solid_pod_rs::blocktrail::Blocktrail =
            serde_json::from_slice(&public_bytes).unwrap();
        assert_eq!(published.txo[1].commit.as_deref(), Some(forged.as_str()));
    } else {
        assert!(report.is_intact(), "{report:?}");
    }
    json!({"case":case,"stored_marks":saved.trail.txo.len(),"pending_intent_retained":kept,"broadcast_attempts":sent.len(),"stored_trail_intact":report.is_intact(),"later_extension_refused":extension_refused,"recovery_error":result.to_string(),"tampered_commit_persisted_and_published":case=="tampered_trail","limits":"Synthetic transport, no consensus or signature validation; real production builders/recovery and FsBackend. Wrong-network next request isolates recovery side effects."})
}
async fn http_cases() -> Value {
    let a = MockServer::start().await;
    let b = MockServer::start().await;
    let id = "ab".repeat(32);
    let route = format!("/api/tx/{id}");
    Mock::given(method("GET"))
        .and(path(&route))
        .respond_with(ResponseTemplate::new(200).set_body_string("not a transaction"))
        .expect(1)
        .mount(&b)
        .await;
    Mock::given(method("GET"))
        .and(path(&route))
        .respond_with(
            ResponseTemplate::new(302).insert_header("Location", format!("{}{route}", b.uri())),
        )
        .expect(1)
        .mount(&a)
        .await;
    let client = MempoolHttpClient::new(a.uri());
    let result = client.transaction_exists(&id).await.unwrap();
    assert!(result);
    assert_eq!(b.received_requests().await.unwrap().len(), 1);
    assert_eq!(client.selection().base_url, a.uri());
    json!({"redirect_followed_to_other_origin":true,"presence_from_non_transaction_200":result,"manifest_remains_original_origin":true,"network_label":client.selection().network.as_str(),"limits":"Two loopback HTTP fixtures, not a live explorer or network attestation."})
}
#[actix_web::main]
async fn main() {
    if std::env::args().nth(1).as_deref() == Some("selection") {
        let state = solid_pod_rs_server::AppState::new(Arc::new(
            solid_pod_rs::storage::memory::MemoryBackend::new(),
        ));
        let startup =
            solid_pod_rs_server::mempool::select_mempool_endpoint(state.mempool_url.as_deref());
        let runtime = MempoolHttpClient::from_env();
        assert_eq!(startup.network.as_str(), "testnet4");
        assert_eq!(runtime.selection().network.as_str(), "signet");
        println!(
            "{}",
            json!({"startup_expression":startup.to_manifest_json(),"request_client":runtime.selection().to_manifest_json(),"limits":"Exact startup-selection expression over fresh AppState and runtime constructor; not a full server boot/log capture. Child-only synthetic environment override; no external requests."})
        );
        return;
    }
    let mut cases = vec![];
    for name in [
        "ordinary_recovery",
        "wrong_txid",
        "tampered_trail",
        "public_save_failure",
    ] {
        cases.push(recovery_case(name).await);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"recovery":cases,"http":http_cases().await})).unwrap()
    );
}
