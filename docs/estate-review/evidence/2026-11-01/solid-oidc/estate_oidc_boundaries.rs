//! Bounded audit observations, not desired acceptance behaviour.
#![cfg(all(feature = "oidc", feature = "dpop-replay-cache"))]

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use bytes::Bytes;
use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
use p256::{
    ecdsa::SigningKey,
    pkcs8::{EncodePrivateKey, LineEnding},
};
use serde_json::{json, Value};
use solid_pod_rs::oidc::{self, DpopReplayCache, TokenVerifyKey};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const NOW: u64 = 1_700_000_000;
const ISSUER: &str = "https://op.example";
const WEBID: &str = "https://person.example/profile#me";

fn key() -> (EncodingKey, Value) {
    let signing = SigningKey::from_bytes(&[0x31; 32].into()).unwrap();
    let point = signing.verifying_key().to_sec1_point(false);
    let bytes = point.as_bytes();
    let jwk = json!({"kty":"EC","crv":"P-256","alg":"ES256","kid":"audit",
        "x":URL_SAFE_NO_PAD.encode(&bytes[1..33]),"y":URL_SAFE_NO_PAD.encode(&bytes[33..65])});
    let pem = signing.to_pkcs8_pem(LineEnding::LF).unwrap();
    (EncodingKey::from_ec_pem(pem.as_bytes()).unwrap(), jwk)
}

fn jwt(claims: &Value, dpop: bool) -> String {
    let (key, jwk) = key();
    let mut header = Header::new(Algorithm::ES256);
    header.kid = Some("audit".into());
    if dpop {
        header.typ = Some("dpop+jwt".into());
        header.jwk = Some(serde_json::from_value(jwk).unwrap());
    }
    encode(&header, claims, &key).unwrap()
}

fn keyset() -> TokenVerifyKey {
    TokenVerifyKey::Asymmetric(serde_json::from_value(json!({"keys":[key().1]})).unwrap())
}

fn claims(now: u64) -> Value {
    let jwk: oidc::Jwk = serde_json::from_value(key().1).unwrap();
    json!({"iss":ISSUER,"sub":WEBID,"webid":WEBID,"aud":"solid",
        "iat":now,"exp":now+120,"cnf":{"jkt":jwk.thumbprint().unwrap()}})
}

fn proof(url: &str, jti: &str, now: u64, ath: Option<&str>) -> String {
    jwt(
        &json!({"htu":url,"htm":"GET","iat":now,"jti":jti,"ath":ath}),
        true,
    )
}

#[test]
fn observe_access_token_boundaries_with_real_es256() {
    let original = claims(NOW);
    let jkt = original["cnf"]["jkt"].as_str().unwrap();
    assert!(oidc::verify_access_token(&jwt(&original, false), &keyset(), ISSUER, jkt, NOW).is_ok());
    let mut altered = original.clone();
    altered["aud"] = json!("elsewhere");
    assert!(oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW).is_err());
    altered = original.clone();
    altered["exp"] = json!(NOW);
    assert!(oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW).is_ok());
    assert!(
        oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW + 1).is_err()
    );
    altered = original.clone();
    altered["nbf"] = json!(NOW + 3600);
    assert!(oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW).is_ok());
    altered = original.clone();
    altered["webid"] = json!("https://");
    assert_eq!(
        oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW)
            .unwrap()
            .webid,
        "https://"
    );
    altered["webid"] = json!("not-a-url");
    assert_eq!(
        oidc::verify_access_token(&jwt(&altered, false), &keyset(), ISSUER, jkt, NOW)
            .unwrap()
            .webid,
        WEBID
    );
    println!("aud restriction works; exp==now and future nbf accepted; scheme-only webid accepted; malformed webid falls back to sub");
}

#[tokio::test]
async fn observe_dpop_url_matching() {
    let token = proof("https://pod.example/Foo", "url", NOW, None);
    for url in [
        "https://pod.example/Foo",
        "https://pod.example/foo",
        "https://pod.example/Foo/",
    ] {
        assert!(oidc::verify_dpop_proof(&token, url, "GET", NOW, 60, None)
            .await
            .is_ok());
    }
    assert!(
        oidc::verify_dpop_proof(&token, "https://pod.example/Foo?x=1", "GET", NOW, 60, None)
            .await
            .is_err()
    );
    assert!(
        oidc::verify_dpop_proof(&token, "https://pod.example/else", "GET", NOW, 60, None)
            .await
            .is_err()
    );
    println!("case-distinct and slash-distinct paths accepted; request query not excluded from comparison");
}

#[tokio::test]
async fn observe_capacity_eviction_reaccepts_live_proof() {
    let cache = DpopReplayCache::with_config(Duration::from_secs(60), 1);
    let a = proof("https://pod.example/r", "a", NOW, None);
    let b = proof("https://pod.example/r", "b", NOW, None);
    assert!(
        oidc::verify_dpop_proof(&a, "https://pod.example/r", "GET", NOW, 60, Some(&cache))
            .await
            .is_ok()
    );
    assert!(
        oidc::verify_dpop_proof(&a, "https://pod.example/r", "GET", NOW, 60, Some(&cache))
            .await
            .is_err()
    );
    assert!(
        oidc::verify_dpop_proof(&b, "https://pod.example/r", "GET", NOW, 60, Some(&cache))
            .await
            .is_ok()
    );
    assert!(
        oidc::verify_dpop_proof(&a, "https://pod.example/r", "GET", NOW, 60, Some(&cache))
            .await
            .is_ok()
    );
    println!("unchanged valid proof accepted again after one-entry cache evicts its unexpired jti");
}

#[test]
fn observe_profile_binding_ignores_subject_and_context() {
    let html = |value: Value| format!("<script type=\"application/ld+json\">{value}</script>");
    let valid = html(json!({"@id":WEBID,"solid:oidcIssuer":{"@id":ISSUER}}));
    assert!(oidc::check_profile_declares_issuer(valid.as_bytes(), ISSUER).is_ok());
    assert!(
        oidc::check_profile_declares_issuer(valid.as_bytes(), "https://other.example").is_err()
    );
    let wrong = html(json!({"@context":{"solid":"https://unrelated.example/"},
        "@id":"https://someone-else.example/#me","solid:oidcIssuer":{"@id":ISSUER}}));
    assert!(oidc::check_profile_declares_issuer(wrong.as_bytes(), ISSUER).is_ok());
    let turtle = format!("<{WEBID}> <http://www.w3.org/ns/solid/terms#oidcIssuer> <{ISSUER}> .");
    assert!(oidc::check_profile_declares_issuer(turtle.as_bytes(), ISSUER).is_err());
    println!("literal compact predicate accepted without subject/context binding; equivalent Turtle not read");
}

#[actix_web::test]
async fn observe_private_route_advertises_but_does_not_consume_dpop() {
    use actix_web::{http::header, test};
    use solid_pod_rs::{
        auth::nip98,
        storage::{memory::MemoryBackend, Storage},
    };
    use solid_pod_rs_server::{build_app, AppState};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let url = "http://pod.example/private/document.txt";
    let access = jwt(&claims(now), false);
    let ath = URL_SAFE_NO_PAD.encode(<sha2::Sha256 as sha2::Digest>::digest(access.as_bytes()));
    let dpop = proof(url, "route", now, Some(&ath));
    let verified = oidc::verify_dpop_proof_with_ath(&dpop, url, "GET", now, 60, Some(&ath), None)
        .await
        .unwrap();
    let identity =
        oidc::verify_access_token(&access, &keyset(), ISSUER, &verified.jkt, now).unwrap();
    assert_eq!(identity.webid, WEBID);
    let secret = hex::encode([0x37; 32]);
    let nostr = nip98::authorization_header(&nip98::mint(url, "GET", &secret, now).unwrap());
    let public = nip98::verify_at(&nostr, url, "GET", None, now)
        .unwrap()
        .pubkey;
    let storage = Arc::new(MemoryBackend::new());
    storage.create_container("/private/").await.unwrap();
    storage.put("/private.acl",Bytes::from(format!("@prefix acl: <http://www.w3.org/ns/auth/acl#> .\n<#owner> a acl:Authorization; acl:agent <did:nostr:{public}>; acl:accessTo </private/>; acl:default </private/>; acl:mode acl:Read .\n<#webid> a acl:Authorization; acl:agent <{WEBID}>; acl:accessTo </private/>; acl:default </private/>; acl:mode acl:Read .")),"text/turtle").await.unwrap();
    storage
        .put(
            "/private/document.txt",
            Bytes::from_static(b"private"),
            "text/plain",
        )
        .await
        .unwrap();
    let app = test::init_service(build_app(AppState::new(storage))).await;
    let request = |auth: String, proof: Option<&str>| {
        let mut r = test::TestRequest::get()
            .uri("/private/document.txt")
            .insert_header((header::HOST, "pod.example"))
            .insert_header((header::AUTHORIZATION, auth));
        if let Some(p) = proof {
            r = r.insert_header(("dpop", p));
        }
        r.to_request()
    };
    let control = test::call_service(&app, request(nostr, None)).await;
    assert!(
        control.status().is_success(),
        "Nostr positive-control status: {}",
        control.status()
    );
    assert_eq!(
        test::read_body(control).await,
        Bytes::from_static(b"private")
    );
    let rejected = test::call_service(&app, request(format!("DPoP {access}"), Some(&dpop))).await;
    assert!(rejected.status().is_client_error());
    let challenge = rejected
        .headers()
        .get(header::WWW_AUTHENTICATE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(challenge.contains("DPoP"));
    println!("Nostr control reads private bytes; helper-verified ES256 token/proof denied while challenge advertises DPoP: {}",rejected.status());
}
