//! Bounded estate audit probes. Assertions describe observed defects, not desired policy.
use actix_web::{http::StatusCode, test};
use base64::{engine::general_purpose::STANDARD, Engine};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use solid_pod_rs::{
    auth::{
        nip98::{self, MatchPolicy, Nip98Event},
        replay::{Nip98ReplayCache, ReplayStore},
    },
    storage::{memory::MemoryBackend, Storage},
};
use solid_pod_rs_server::{build_app, AppState};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
fn signed(url: &str, method: &str, body: Option<&[u8]>, ts: u64) -> (String, String) {
    let key = k256::schnorr::SigningKey::from_slice(&[0x31; 32]).unwrap();
    let pk = hex::encode(key.verifying_key().to_bytes());
    let mut tags = vec![
        vec!["u".into(), url.into()],
        vec!["method".into(), method.into()],
    ];
    if let Some(b) = body {
        tags.push(vec!["payload".into(), hex::encode(Sha256::digest(b))]);
    }
    let mut ev = Nip98Event {
        id: String::new(),
        pubkey: pk.clone(),
        created_at: ts,
        kind: 27235,
        tags,
        content: String::new(),
        sig: String::new(),
    };
    ev.id = nip98::compute_event_id(&ev);
    ev.sig = hex::encode(
        key.sign_raw(&hex::decode(&ev.id).unwrap(), &[0; 32])
            .unwrap()
            .to_bytes(),
    );
    (
        format!(
            "Nostr {}",
            STANDARD.encode(serde_json::to_vec(&ev).unwrap())
        ),
        pk,
    )
}
#[actix_web::test]
async fn nonempty_signed_body_can_be_removed_at_actual_put_route() {
    let url = "http://pod.example/private/document.txt";
    let intended = b"authorised replacement";
    let (token, pk) = signed(url, "PUT", Some(intended), now());
    let storage = Arc::new(MemoryBackend::new());
    storage.create_container("/private/").await.unwrap();
    let acl = format!(
        r#"@prefix acl: <http://www.w3.org/ns/auth/acl#> .
<#owner> a acl:Authorization; acl:agent <did:nostr:{pk}>; acl:accessTo </private/>; acl:default </private/>; acl:mode acl:Read, acl:Write, acl:Control ."#
    );
    storage
        .put("/private.acl", Bytes::from(acl), "text/turtle")
        .await
        .unwrap();
    storage
        .put(
            "/private/document.txt",
            Bytes::from_static(b"original"),
            "text/plain",
        )
        .await
        .unwrap();
    let app = test::init_service(build_app(AppState::new(storage.clone()))).await;
    let request = |body: &'static [u8], auth: Option<&str>| {
        let mut r = test::TestRequest::put()
            .uri("/private/document.txt")
            .insert_header(("host", "pod.example"))
            .insert_header(("content-type", "text/plain"));
        if let Some(a) = auth {
            r = r.insert_header(("authorization", a));
        }
        r.set_payload(Bytes::from_static(body)).to_request()
    };
    let (control, _) = signed(url, "PUT", Some(b"original"), now() - 2);
    let valid = test::call_service(&app, request(b"original", Some(&control)))
        .await
        .status();
    assert!(
        valid.is_success(),
        "positive control must pass before defect probe: {valid}"
    );
    let anonymous = test::call_service(&app, request(b"", None)).await.status();
    assert!(anonymous == StatusCode::UNAUTHORIZED || anonymous == StatusCode::FORBIDDEN);
    let altered = test::call_service(&app, request(b"different", Some(&token)))
        .await
        .status();
    assert!(altered == StatusCode::UNAUTHORIZED || altered == StatusCode::FORBIDDEN);
    assert_eq!(
        &storage.get("/private/document.txt").await.unwrap().0[..],
        b"original"
    );
    let stripped = test::call_service(&app, request(b"", Some(&token)))
        .await
        .status();
    assert!(
        stripped.is_success(),
        "observed empty-body acceptance: {stripped}"
    );
    assert!(storage
        .get("/private/document.txt")
        .await
        .unwrap()
        .0
        .is_empty());
    let replay = test::call_service(&app, request(intended, Some(&token)))
        .await
        .status();
    assert!(replay == StatusCode::UNAUTHORIZED || replay == StatusCode::FORBIDDEN);
    // A newly signed, correctly bound request succeeds and restores the intended bytes.
    let (fresh, _) = signed(url, "PUT", Some(intended), now() - 1);
    assert!(test::call_service(&app, request(intended, Some(&fresh)))
        .await
        .status()
        .is_success());
    assert_eq!(
        &storage.get("/private/document.txt").await.unwrap().0[..],
        intended
    );
    println!("private PUT: altered nonempty refused; stripped body accepted and persisted empty; same token then refused; fresh correct token succeeds");
}
#[tokio::test]
async fn short_positive_ttl_reopens_a_still_valid_signed_token() {
    let ts = now();
    let url = "http://pod.example/ttl";
    let (t, _) = signed(url, "GET", None, ts);
    let cache = Nip98ReplayCache::with_config(Duration::from_millis(5), 1);
    let v = nip98::verify_at(&t, url, "GET", None, ts).unwrap();
    cache.check_and_record(&v.event_id).await.unwrap();
    assert!(cache.check_and_record(&v.event_id).await.is_err());
    tokio::time::sleep(Duration::from_millis(20)).await;
    let v = nip98::verify_at(&t, url, "GET", None, ts + 1).unwrap();
    cache.check_and_record(&v.event_id).await.unwrap();
    println!("same valid signed token accepted after 5ms cache expiry; verifier clock advanced 1s");
}
#[tokio::test]
async fn zero_ttl_disables_replay_rejection() {
    let cache = Nip98ReplayCache::with_config(Duration::ZERO, 1);
    cache.check_and_record("signed-id").await.unwrap();
    cache.check_and_record("signed-id").await.unwrap();
    println!("public constructor permits zero TTL and immediate second admission");
}
#[tokio::test]
async fn git_prefix_accepts_sibling_name_and_host_prefix() {
    let ts = now();
    for (base, sibling) in [
        (
            "https://pod.example/repo",
            "https://pod.example/repo-other/git-receive-pack",
        ),
        (
            "https://pod.example",
            "https://pod.example.attacker.test/repo",
        ),
    ] {
        let (t, _) = signed(base, "*", None, ts);
        assert!(nip98::verify_at_with_policy(
            &t,
            sibling,
            "POST",
            None,
            ts,
            MatchPolicy::GitLenient
        )
        .is_ok());
        assert!(nip98::verify_at_with_policy(
            &t,
            "https://unrelated.test/repo",
            "POST",
            None,
            ts,
            MatchPolicy::GitLenient
        )
        .is_err());
        println!("GitLenient accepted token base {base} for {sibling}");
    }
}
#[tokio::test]
async fn strict_policy_aliases_and_empty_body_are_not_exact_binding() {
    let ts = now();
    let (t, _) = signed("https://pod.example/item/", "put", Some(b"expected"), ts);
    assert!(nip98::verify_at(&t, "https://pod.example/item", "PUT", Some(b""), ts).is_ok());
    assert!(nip98::verify_at(&t, "https://pod.example/item", "PUT", Some(b"changed"), ts).is_err());
    println!("Strict aliases trailing slash and method case; supplied empty body skips signed payload hash");
}
