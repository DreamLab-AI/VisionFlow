//! Local in-process HTTP acceptance probes; no network listener or external pod.
use actix_web::{http::Method, test};
use bytes::Bytes;
use serde_json::{json, Value};
use solid_pod_rs::{
    auth::nip98,
    storage::{memory::MemoryBackend, Storage},
};
use solid_pod_rs_server::{build_app, AppState};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, OnceLock,
};

const SK: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const SECRET: &str = "local-audit-private-body";
fn did() -> String {
    let sk = k256::schnorr::SigningKey::from_slice(&hex::decode(SK).unwrap()).unwrap();
    format!("did:nostr:{}", hex::encode(sk.verifying_key().to_bytes()))
}
fn auth(method: &str, path: &str, body: Option<&[u8]>) -> String {
    static BASE: OnceLock<u64> = OnceLock::new();
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let base = *BASE.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 30
    });
    let token = nip98::mint_with_payload(
        &format!("http://localhost:8080{path}"),
        method,
        body,
        SK,
        base + SEQ.fetch_add(1, Ordering::Relaxed),
    )
    .unwrap();
    format!("Nostr {token}")
}
fn owner_acl(target: &str) -> String {
    format!("@prefix acl: <http://www.w3.org/ns/auth/acl#> .\n@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n<#owner> a acl:Authorization ; acl:agent <{}> ; acl:accessTo <{target}> ; acl:mode acl:Read, acl:Write, acl:Control .\n", did())
}
fn read_rule(condition: &str) -> String {
    format!("<#read> a acl:Authorization ; acl:agentClass foaf:Agent ; acl:accessTo </secret.bin> ; acl:mode acl:Read ; acl:condition {condition} .\n")
}
async fn seed() -> Arc<MemoryBackend> {
    let s = Arc::new(MemoryBackend::new());
    s.put(
        "/secret.bin",
        Bytes::from_static(SECRET.as_bytes()),
        "application/octet-stream",
    )
    .await
    .unwrap();
    s.put(
        "/secret.bin.acl",
        Bytes::from(owner_acl("/secret.bin")),
        "text/turtle",
    )
    .await
    .unwrap();
    s
}
async fn request(
    s: Arc<MemoryBackend>,
    method: &str,
    path: &str,
    owner: bool,
    body: Option<&str>,
) -> Value {
    // A new app instance over the same storage per call. This is not an OS restart.
    let app = test::init_service(build_app(AppState::new(s))).await;
    let mut req = test::TestRequest::default()
        .method(Method::from_bytes(method.as_bytes()).unwrap())
        .uri(path);
    if owner {
        req = req.insert_header(("Authorization", auth(method, path, body.map(str::as_bytes))));
    }
    if let Some(b) = body {
        req = req
            .insert_header(("Content-Type", "text/turtle"))
            .set_payload(b.to_owned());
    }
    let response = test::call_service(&app, req.to_request()).await;
    let status = response.status().as_u16();
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned)
    };
    let cc = header("cache-control");
    let vary = header("vary");
    let wac = header("wac-allow");
    let body = test::read_body(response).await;
    json!({"status":status,"cache_control":cc,"vary":vary,"wac_allow":wac,"secret_returned":body.as_ref()==SECRET.as_bytes()})
}
#[actix_web::main]
async fn main() {
    let mut conditions = vec![];
    for (name, condition, expected_put, expected_get) in [
        (
            "valid",
            "[ a acl:ClientCondition ; acl:client <https://client.example/app> ]",
            201,
            401,
        ),
        (
            "unterminated",
            "[ a acl:ClientCondition ; acl:client <https://client.example/app>",
            400,
            401,
        ),
        (
            "missing_type",
            "[ acl:client <https://client.example/app> ]",
            201,
            200,
        ),
        ("empty", "[]", 201, 200),
        ("unknown", "[ a acl:UnsupportedCondition ]", 201, 401),
    ] {
        let s = seed().await;
        let proposed = owner_acl("/secret.bin") + &read_rule(condition);
        let put = request(s.clone(), "PUT", "/secret.bin.acl", true, Some(&proposed)).await;
        let get = request(s, "GET", "/secret.bin", false, None).await;
        assert_eq!(put["status"], expected_put, "{name}: {put}");
        assert_eq!(get["status"], expected_get, "{name}: {get}");
        assert_eq!(get["secret_returned"], expected_get == 200);
        conditions.push(json!({"case":name,"put":put,"anonymous_get":get}));
    }
    let s = seed().await;
    let sidecar = request(s.clone(), "GET", "/secret.bin.acl", true, None).await;
    assert_eq!(sidecar["status"], 200);
    assert_eq!(sidecar["cache_control"], "private, no-store");
    let private = request(s.clone(), "GET", "/secret.bin", true, None).await;
    assert_eq!(private["status"], 200);
    assert_eq!(private["wac_allow"], "user=\"\", public=\"\"");
    assert_eq!(private["cache_control"], "private, no-store");
    assert!(private["vary"].as_str().unwrap().contains("Authorization"));
    let public=owner_acl("/secret.bin")+"<#public> a acl:Authorization ; acl:agentClass foaf:Agent ; acl:accessTo </secret.bin> ; acl:mode acl:Read .\n";
    let denied = request(s.clone(), "PUT", "/secret.bin.acl", false, Some(&public)).await;
    assert_eq!(denied["status"], 401);
    let grant = request(s.clone(), "PUT", "/secret.bin.acl", true, Some(&public)).await;
    assert_eq!(grant["status"], 201);
    let before = request(s.clone(), "GET", "/secret.bin", false, None).await;
    assert_eq!(before["status"], 200);
    assert!(before["cache_control"].is_null());
    assert_eq!(before["wac_allow"], "user=\"\", public=\"\"");
    let remove = request(
        s.clone(),
        "PUT",
        "/secret.bin.acl",
        true,
        Some(&owner_acl("/secret.bin")),
    )
    .await;
    assert_eq!(remove["status"], 201);
    let after = request(s, "GET", "/secret.bin", false, None).await;
    assert_eq!(after["status"], 401);
    assert_eq!(after["secret_returned"], false);
    let s = seed().await;
    let wrong_target = request(
        s.clone(),
        "PUT",
        "/secret.bin.acl",
        true,
        Some(&owner_acl("/elsewhere")),
    )
    .await;
    assert_eq!(wrong_target["status"], 201);
    let locked_out = request(s, "GET", "/secret.bin.acl", true, None).await;
    assert_eq!(locked_out["status"], 403);
    println!("{}",serde_json::to_string_pretty(&json!({"condition_admission":conditions,"private_delivery":{"owner_sidecar":sidecar,"owner_binary":private},"grant_removal":{"unauthorised_put":denied,"grant_put":grant,"before":before,"remove_put":remove,"after_new_app":after},"lockout_guard":{"misdirected_control_put":wrong_target,"owner_sidecar_after":locked_out},"limits":"In-process build_app with synthetic MemoryBackend data and NIP-98 test key. New app per request, no OS restart, intermediary cache, deployment or real private data."})).unwrap());
}
