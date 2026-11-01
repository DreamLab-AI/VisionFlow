//! Audit fixture adapted from native prov_phase5_routes helpers at e1d16ad.
//! Known test key and disposable FS only. No live explorer or production payment.
use std::process::Stdio;
use std::sync::Arc;

use actix_web::{test, web, App, HttpResponse};
use serde_json::{json, Value};
use solid_pod_rs::auth::nip98;
use solid_pod_rs::storage::fs::FsBackend;
use solid_pod_rs_server::trail_store::{save_trail, StoredTrail};
use solid_pod_rs_server::{build_app, AppState};

const NETWORK: &str = "testnet4";
const TICKER: &str = "PROV";
// The pod-owner / trail-issuer key. NIP-98 identity (and the pod name) is its
// x-only pubkey; the trail's issuer pubkey is its compressed (66-hex) form.
const OWNER_SK: &str = "0000000000000000000000000000000000000000000000000000000000000007";

/// Compressed (66-hex) issuer pubkey — the trail's `issuer` / `pubkeyBase`.
fn issuer_pubkey() -> String {
    let sk = k256::SecretKey::from_slice(&hex::decode(OWNER_SK).unwrap()).unwrap();
    hex::encode(sk.public_key().to_sec1_bytes())
}

/// X-only (64-hex) owner pubkey — the NIP-98 identity AND the pod-name segment
/// (pod paths must be 64-hex for `pod_repo_path`).
fn owner_xonly() -> String {
    let sk = k256::schnorr::SigningKey::from_slice(&hex::decode(OWNER_SK).unwrap()).unwrap();
    hex::encode(sk.verifying_key().to_bytes())
}

/// Mint a NIP-98 `Authorization` header for the owner over `method`+`path`.
fn owner_nip98(method: &str, path: &str, body: Option<&[u8]>) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    // Each token MUST be a distinct NIP-98 event: the server's single-use replay
    // guard (`NIP98_REPLAY`, a process-global `LazyLock`) rejects a repeated
    // event_id with 401, and tokens minted in the same second with the same
    // key/url/method are otherwise identical. Anchor `created_at` to a base
    // captured once (30s in the past) + a monotonic per-call counter — strictly
    // unique regardless of wall-clock drift, within the ±60s tolerance window.
    static BASE: OnceLock<u64> = OnceLock::new();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let base = *BASE.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(30)
    });
    let now = base + SEQ.fetch_add(1, Ordering::Relaxed);
    let url = format!("http://localhost:8080{path}");
    let token = nip98::mint_with_payload(&url, method, body, OWNER_SK, now).expect("nip98 mint");
    format!("Nostr {token}")
}

/// ACL granting the owner (and the public, to keep anonymous writes simple)
/// read+write+append over `container`, OPTIONALLY carrying a `ProvenanceAnchor`
/// condition with `mode` (`"always"` ⇒ inline anchor, `"epoch"` ⇒ batched).
fn acl(container: &str, anchor_mode: Option<&str>) -> String {
    let cond = match anchor_mode {
        Some(m) => format!(
            r#", "acl:condition": [{{"@type": "acl:ProvenanceAnchor", "acl:anchorMode": "{m}"}}]"#
        ),
        None => String::new(),
    };
    format!(
        r##"{{
            "@context": {{
                "acl": "http://www.w3.org/ns/auth/acl#",
                "foaf": "http://xmlns.com/foaf/0.1/"
            }},
            "@graph": [{{
                "@id": "#public",
                "acl:agentClass": {{"@id": "foaf:Agent"}},
                "acl:accessTo": {{"@id": "{container}"}},
                "acl:default": {{"@id": "{container}"}},
                "acl:mode": [
                    {{"@id": "acl:Read"}},
                    {{"@id": "acl:Write"}},
                    {{"@id": "acl:Append"}}
                ]{cond}
            }}]
        }}"##
    )
}

/// Genesis chained-key scriptPubKey (`5120<xonly>`) for the single-state PROV
/// trail — what the trail's current UTXO pays.
fn genesis_state() -> solid_pod_rs::mrc20::Mrc20State {
    use solid_pod_rs::mrc20::{Mrc20State, MRC20_PROFILE};
    Mrc20State {
        profile: MRC20_PROFILE.into(),
        prev: "0".repeat(64),
        seq: 0,
        ticker: Some(TICKER.into()),
        name: Some("Prov Trail".into()),
        decimals: Some(0),
        supply: Some(1000),
        balances: Some(std::collections::BTreeMap::from([(issuer_pubkey(), 1000)])),
        ops: vec![],
        anchor: None,
    }
}

fn genesis_spk_hex() -> String {
    use solid_pod_rs::mrc20::{bt_derive_chained_pubkey, jcs};
    let genesis_jcs = jcs(&serde_json::to_value(genesis_state()).unwrap());
    let chained = bt_derive_chained_pubkey(&issuer_pubkey(), &[genesis_jcs]).unwrap();
    format!("5120{}", hex::encode(&chained[1..]))
}

// Synthetic explorer stores decoded transactions, not a consensus or signature oracle.
async fn spawn_fixture() -> (
    String,
    actix_web::dev::ServerHandle,
    Arc<std::sync::Mutex<Vec<Value>>>,
) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let sent = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let records = web::Data::new(sent.clone());
    let server = actix_web::HttpServer::new(move || {
        App::new().app_data(records.clone())
        .route("/api/tx/{txid}",web::get().to(|id:web::Path<String>, data:web::Data<Arc<std::sync::Mutex<Vec<Value>>>>|async move {
            if id.as_str() == "aa".repeat(32) {return HttpResponse::Ok().json(json!({"txid":"aa".repeat(32),"vout":[{"value":50000,"scriptpubkey":genesis_spk_hex()}],"status":{"confirmed":true,"block_height":42000}}));}
            let found=data.lock().unwrap().iter().find(|t| t["txid"].as_str()==Some(id.as_str())).cloned();
            match found {Some(v)=>HttpResponse::Ok().json(v),None=>HttpResponse::NotFound().finish()}
        }))
        .route("/api/address/{addr}/utxo",web::get().to(||async {HttpResponse::Ok().json(json!([]))}))
        .route("/api/tx",web::post().to(|body:bytes::Bytes,data:web::Data<Arc<std::sync::Mutex<Vec<Value>>>>|async move {
            let tx=solid_pod_rs::bitcoin_tx::decode_tx_info(&String::from_utf8_lossy(&body)).unwrap();
            let id=tx.txid.clone();
            let value=json!({"txid":id,"vin":tx.vin.iter().map(|i|json!({"txid":i.txid,"vout":i.vout})).collect::<Vec<_>>(),"vout":tx.vout.iter().map(|o|json!({"value":o.value,"scriptpubkey":o.scriptpubkey})).collect::<Vec<_>>(),"status":{"confirmed":false}});
            data.lock().unwrap().push(value);
            HttpResponse::Ok().body(id)
        }))
    }).listen(listener).unwrap().workers(1).run();
    let handle = server.handle();
    tokio::spawn(server);
    (format!("http://127.0.0.1:{port}"), handle, sent)
}

/// Build an FS-backed, git-initialised pod named after the owner's x-only
/// pubkey, with: the pay-token configured (issuer = owner), a minted PROV
/// trail, the owner's sat balance seeded, the resource container ACL (optionally
/// carrying a ProvenanceAnchor), and `mempool_url` pointing at the fixture.
async fn git_pod_with_trail(
    mempool_url: String,
    container: &str,
    anchor_mode: Option<&str>,
    owner_balance: u64,
) -> (AppState, tempfile::TempDir, String) {
    let pod = owner_xonly();
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().to_path_buf();

    std::fs::create_dir_all(root.join(&pod)).expect("pod dir");
    // Sibling ACL governs the whole pod container; carries the anchor marker.
    std::fs::write(root.join(format!("{pod}.acl")), acl(container, anchor_mode)).expect("acl");

    let init = std::process::Command::new("git")
        .args(["init", "-b", "main"])
        .current_dir(root.join(&pod))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("git init");
    assert!(init.success());

    let fs = FsBackend::new(root.clone()).await.expect("fs backend");
    let mut state = AppState::new(Arc::new(fs));
    state.data_root = Some(root);
    state.mempool_url = Some(mempool_url);
    state.pay_config.token = Some(solid_pod_rs::payments::TokenConfig {
        ticker: TICKER.into(),
        rate: 5, // anchor price defaults to the token rate (5 sats)
        supply: 1000,
        issuer: issuer_pubkey(),
        accepted_issuers: Vec::new(),
    });

    // Seed the owner's ledger balance.
    let did = format!("did:nostr:{}", owner_xonly());
    let mut ledger = solid_pod_rs::payments::WebLedger::new("Pod Credits");
    ledger
        .credit_by_outpoint(&did, "satoshi", &"ee".repeat(32), 0, owner_balance)
        .unwrap();
    state
        .storage
        .put(
            "/.well-known/webledgers/webledgers.json",
            bytes::Bytes::from(serde_json::to_vec(&ledger).unwrap()),
            "application/json",
        )
        .await
        .unwrap();

    // Mint the genesis trail (single state) controlled by the owner key.
    use solid_pod_rs::mrc20::jcs;
    let genesis = genesis_state();
    let genesis_jcs = jcs(&serde_json::to_value(&genesis).unwrap());
    let stored = StoredTrail {
        ticker: TICKER.into(),
        name: "Prov Trail".into(),
        supply: 1000,
        privkey: OWNER_SK.into(),
        pubkey_base: issuer_pubkey(),
        states: vec![genesis],
        state_strings: vec![genesis_jcs],
        current_txid: "aa".repeat(32),
        current_vout: 0,
        current_amount: 50_000,
        network: NETWORK.into(),
        date_created: "2026-06-13T00:00:00Z".into(),
    };
    save_trail(&state.storage, &stored).await.unwrap();

    (state, tmp, pod)
}

fn last_commit_sha(repo: &std::path::Path) -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .expect("git rev-parse");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

async fn query_and_anchor() -> Value {
    let (url, server, sent) = spawn_fixture().await;
    let pod = owner_xonly();
    let (state, tmp, _) = git_pod_with_trail(url, &format!("/{pod}/"), None, 100).await;
    let storage = state.storage.clone();
    let repo = tmp.path().join(&pod);
    let app = test::init_service(build_app(state)).await;
    let resource = format!("/{pod}/private.txt");
    let mut shas = Vec::new();
    for body in ["revision one", "revision two"] {
        let response = test::call_service(
            &app,
            test::TestRequest::put()
                .uri(&resource)
                .insert_header(("Content-Type", "text/plain"))
                .set_payload(body)
                .to_request(),
        )
        .await;
        assert!(response.status().is_success());
        shas.push(last_commit_sha(&repo));
    }
    let latest_ttl = std::fs::read_to_string(repo.join("private.txt.prov.ttl")).unwrap();
    assert!(latest_ttl.contains(&shas[1]));
    assert!(!latest_ttl.contains(&shas[0]) || latest_ttl.contains("parent"));
    // Replace public fixture ACL with an owner-only ACL after creating the history.
    let acl = json!({"@graph":[{"@type":"acl:Authorization","acl:agent":{"@id":format!("did:nostr:{pod}")},"acl:accessTo":{"@id":format!("/{pod}/")},"acl:default":{"@id":format!("/{pod}/")},"acl:mode":[{"@id":"acl:Read"},{"@id":"acl:Write"},{"@id":"acl:Control"}]}]});
    std::fs::write(tmp.path().join(format!("{pod}.acl")), acl.to_string()).unwrap();
    let mut denied = Vec::new();
    for uri in [resource.clone(), format!("{resource}.prov.ttl")] {
        let r = test::call_service(&app, test::TestRequest::get().uri(&uri).to_request()).await;
        denied.push(r.status().as_u16());
    }
    assert_eq!(denied, vec![401, 401]);
    let query = format!("/{pod}/_prov/{}", shas[0]);
    let r = test::call_service(&app, test::TestRequest::get().uri(&query).to_request()).await;
    assert_eq!(r.status().as_u16(), 200);
    let old: Value = test::read_body_json(r).await;
    assert_eq!(old["commit"]["sha"], shas[0]);
    assert_eq!(old["prov_ttl"], latest_ttl);
    assert_eq!(old["anchored"], false);
    // A comment, not an anchor triple, toggles the reported boolean.
    std::fs::write(
        repo.join("private.txt.prov.ttl"),
        format!("{latest_ttl}\n# bt:txid is only a comment\n"),
    )
    .unwrap();
    let r = test::call_service(&app, test::TestRequest::get().uri(&query).to_request()).await;
    let comment: Value = test::read_body_json(r).await;
    assert_eq!(comment["anchored"], true);
    // Authenticated duplicate logical requests get new NIP-98 events, as a real retry must.
    let path = format!("/{pod}/_prov/anchor");
    let body = json!({"commit_sha":shas[0]}).to_string();
    let mut results = Vec::new();
    for _ in 0..2 {
        let r = test::call_service(
            &app,
            test::TestRequest::post()
                .uri(&path)
                .insert_header((
                    "Authorization",
                    owner_nip98("POST", &path, Some(body.as_bytes())),
                ))
                .insert_header(("Content-Type", "application/json"))
                .set_payload(body.clone())
                .to_request(),
        )
        .await;
        let status = r.status().as_u16();
        let value: Value = test::read_body_json(r).await;
        assert_eq!(status, 200, "{value}");
        assert_eq!(value["charged_sats"], 5);
        results.push(value);
    }
    let ledger: WebLedger = serde_json::from_value(
        serde_json::from_slice::<Value>(&storage.get(CANON).await.unwrap().0).unwrap()["ledger"]
            .clone(),
    )
    .unwrap();
    assert_eq!(ledger.get_balance(&format!("did:nostr:{pod}")), 90);
    assert_eq!(sent.lock().unwrap().len(), 2);
    assert_ne!(results[0]["anchor"]["txid"], results[1]["anchor"]["txid"]);
    let after = std::fs::read_to_string(repo.join("private.txt.prov.ttl")).unwrap();
    assert!(after.contains(&shas[0]));
    assert!(!after.contains(&shas[1]));
    // Accepted broadcast plus a failed sidecar write still returns successful JSON.
    std::fs::remove_file(repo.join("private.txt.prov.ttl")).unwrap();
    std::fs::create_dir(repo.join("private.txt.prov.ttl")).unwrap();
    let r = test::call_service(
        &app,
        test::TestRequest::post()
            .uri(&path)
            .insert_header((
                "Authorization",
                owner_nip98("POST", &path, Some(body.as_bytes())),
            ))
            .insert_header(("Content-Type", "application/json"))
            .set_payload(body)
            .to_request(),
    )
    .await;
    assert_eq!(r.status().as_u16(), 200);
    let missing: Value = test::read_body_json(r).await;
    assert_eq!(missing["charged_sats"], 5);
    assert!(missing.get("sidecar_error").is_none());
    assert!(!repo.join("private.txt.prov.ttl").is_file());
    let balance = serde_json::from_value::<WebLedger>(
        serde_json::from_slice::<Value>(&storage.get(CANON).await.unwrap().0).unwrap()["ledger"]
            .clone(),
    )
    .unwrap()
    .get_balance(&format!("did:nostr:{pod}"));
    assert_eq!(balance, 85);
    server.stop(false).await;
    json!({"private_query":{"ordinary_get_statuses":denied,"anonymous_query_status":200,"requested_old_commit_returned_current_sidecar":true,"comment_sets_anchored_true":true},"duplicate_anchor":{"requests":2,"different_broadcast_transactions":2,"initial_balance":100,"balance_after_two":90,"old_anchor_replaced_current_sidecar":true},"sidecar_failure":{"status":200,"charged":5,"balance":balance,"no_sidecar_file":true,"no_sidecar_error_field":true}})
}

use solid_pod_rs::error::PodError;
use solid_pod_rs::payments::WebLedger;
use solid_pod_rs::storage::{ResourceMeta, Storage, StorageEvent};
use std::sync::atomic::{AtomicBool, Ordering};
const CANON: &str = "/.well-known/webledgers/state.json";
const LEGACY: &str = "/.well-known/webledgers/webledgers.json";
struct ReadFault {
    inner: solid_pod_rs::storage::memory::MemoryBackend,
    fail: AtomicBool,
}
#[async_trait::async_trait]
impl Storage for ReadFault {
    async fn get(&self, p: &str) -> Result<(bytes::Bytes, ResourceMeta), PodError> {
        if p == CANON && self.fail.load(Ordering::SeqCst) {
            return Err(PodError::Backend("injected canonical read outage".into()));
        }
        self.inner.get(p).await
    }
    async fn put(&self, p: &str, b: bytes::Bytes, c: &str) -> Result<ResourceMeta, PodError> {
        self.inner.put(p, b, c).await
    }
    async fn delete(&self, p: &str) -> Result<(), PodError> {
        self.inner.delete(p).await
    }
    async fn list(&self, p: &str) -> Result<Vec<String>, PodError> {
        self.inner.list(p).await
    }
    async fn head(&self, p: &str) -> Result<ResourceMeta, PodError> {
        self.inner.head(p).await
    }
    async fn exists(&self, p: &str) -> Result<bool, PodError> {
        self.inner.exists(p).await
    }
    async fn watch(&self, p: &str) -> Result<tokio::sync::mpsc::Receiver<StorageEvent>, PodError> {
        self.inner.watch(p).await
    }
}
async fn state_read_failure() -> Value {
    let fs = Arc::new(ReadFault {
        inner: solid_pod_rs::storage::memory::MemoryBackend::new(),
        fail: AtomicBool::new(false),
    });
    let app = test::init_service(build_app(AppState::new(fs.clone()))).await;
    let did = format!("did:nostr:{}", owner_xonly());
    let mut original = WebLedger::new("audit");
    original
        .credit_by_outpoint(&did, "satoshi", &"bb".repeat(32), 0, 100)
        .unwrap();
    fs.inner
        .put(
            LEGACY,
            bytes::Bytes::from(serde_json::to_vec(&original).unwrap()),
            "application/json",
        )
        .await
        .unwrap();
    // A real route creates the canonical document from the legacy bootstrap.
    let sell = json!({"sell_currency":"tbtc3","sell_amount":10,"buy_currency":"sat","price":100})
        .to_string();
    let request = || {
        test::TestRequest::post()
            .uri("/pay/.sell")
            .insert_header((
                "Authorization",
                owner_nip98("POST", "/pay/.sell", Some(sell.as_bytes())),
            ))
            .insert_header(("Content-Type", "application/json"))
            .set_payload(sell.clone())
            .to_request()
    };
    let r = test::call_service(&app, request()).await;
    assert!(r.status().is_success());
    let mut charged = original.clone();
    let _charge = charged.charge(&did, 80).unwrap();
    let mut authoritative: Value =
        serde_json::from_slice(&fs.inner.get(CANON).await.unwrap().0).unwrap();
    authoritative["ledger"] = serde_json::to_value(&charged).unwrap();
    authoritative["version"] = json!(7);
    authoritative["intents"] = json!([{"txid":"cc".repeat(32),"did":did,"amount":80,"trail":null}]);
    fs.inner
        .put(
            CANON,
            bytes::Bytes::from(authoritative.to_string()),
            "application/json",
        )
        .await
        .unwrap();
    // Mirror remains 100 while authoritative balance is 20. Read outage only.
    fs.fail.store(true, Ordering::SeqCst);
    let r = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/pay/.balance")
            .insert_header(("Authorization", owner_nip98("GET", "/pay/.balance", None)))
            .to_request(),
    )
    .await;
    assert_eq!(r.status().as_u16(), 200);
    let stale: Value = test::read_body_json(r).await;
    assert_eq!(stale["balance"], 100);
    let r = test::call_service(&app, request()).await;
    assert!(r.status().is_success());
    fs.fail.store(false, Ordering::SeqCst);
    let after: Value = serde_json::from_slice(&fs.inner.get(CANON).await.unwrap().0).unwrap();
    assert_eq!(
        serde_json::from_value::<WebLedger>(after["ledger"].clone())
            .unwrap()
            .get_balance(&did),
        100
    );
    assert_eq!(after["intents"], json!([]));
    assert_eq!(after["version"], 1);
    fs.inner
        .put(
            CANON,
            bytes::Bytes::from_static(b"malformed"),
            "application/json",
        )
        .await
        .unwrap();
    let r = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/pay/.balance")
            .insert_header(("Authorization", owner_nip98("GET", "/pay/.balance", None)))
            .to_request(),
    )
    .await;
    assert_eq!(r.status().as_u16(), 500);
    let parse_error: Value = test::read_body_json(r).await;
    assert!(parse_error.to_string().contains("malformed payment state"));
    json!({"canonical_before":20,"stale_mirror":100,"outage_http_balance_status":200,"outage_read_balance":100,"canonical_after_sell":100,"pending_intents_lost":true,"version_reset_from":7,"version_reset_to":1,"malformed_json_status":500,"malformed_json_control":parse_error,"limit":"Injected read-only failure over MemoryBackend; canonical fixture includes pending debit, no OS crash or multiprocess test."})
}
#[actix_web::main]
async fn main() {
    let http = query_and_anchor().await;
    let payment = state_read_failure().await;
    println!("{}",serde_json::to_string_pretty(&json!({"http":http,"state_read_failure":payment,"limits":"Disposable FsBackend/Git and local Actix services, known test identity, synthetic explorer decoding transactions only. No consensus, confirmation, production funds, deployment or OS restart evidence."})).unwrap());
}
