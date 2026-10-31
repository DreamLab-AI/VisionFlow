//! Disposable audit of the production hub. No service, model or external network is used.
use actix_web::http::header::{HeaderMap, HeaderName, HeaderValue};
use codescape_server::{
    auth,
    origin::OriginPolicy,
    proxy::{parse_peer, ProxyTrust},
    xr_presence::*,
};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const ROOM: &str = "urn:audit:presence";
fn who(seed: &str) -> Identity {
    Identity::resolve(None, seed)
}
fn pose(who: &Identity) -> InboundPose {
    let tf: Vec<u8> = [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    decode_pose(&encode_pose(ROOM, &who.avatar_id, 42, 1, &tf)).unwrap()
}
fn drain(m: &mut Membership) -> Vec<Outbound> {
    let mut out = vec![];
    while let Ok(v) = m.rx.try_recv() {
        out.push(v);
    }
    out
}
fn event(v: &Outbound) -> &str {
    match v {
        Outbound::Binary(_) => "pose",
        Outbound::Close(_) => "close",
        Outbound::Text(_) => "roster",
    }
}
fn local_id(v: Outbound) -> u32 {
    let Outbound::Binary(b) = v else {
        panic!("expected pose")
    };
    decode_sibling(&b).unwrap().2[0].0
}
fn main() {
    let mut cases: Vec<Value> = vec![];
    // Replacement revokes roster membership, but relay checks only the old handle's labels.
    {
        let hub = PresenceHub::new();
        let alice = who("alice");
        let mut observer = hub.join(ROOM, &who("observer"), "observer", None).unwrap();
        let mut old = hub.join(ROOM, &alice, "alice", None).unwrap();
        drain(&mut observer);
        let mut new = hub.join(ROOM, &alice, "alice-new", None).unwrap();
        let announcements = drain(&mut observer);
        assert_eq!(announcements.len(), 2);
        assert!(matches!(old.rx.try_recv().unwrap(), Outbound::Close(_)));
        assert_ne!(old.local_id, new.local_id);
        assert_eq!(old.relay(&pose(&alice)), 2);
        assert_eq!(local_id(observer.rx.try_recv().unwrap()), old.local_id);
        assert_eq!(local_id(new.rx.try_recv().unwrap()), old.local_id);
        cases.push(json!({"case":"replaced_member_still_relays", "classification":"counterexample",
            "old_id":old.local_id,"new_id":new.local_id,"recipients":2,"roster_members":hub.members(ROOM)}));
        drop(old);
        assert_eq!(hub.members(ROOM), 2);
        assert!(observer.rx.try_recv().is_err());
        assert_eq!(new.relay(&pose(&alice)), 1);
        assert_eq!(local_id(observer.rx.try_recv().unwrap()), new.local_id);
        cases.push(
            json!({"case":"old_drop_preserves_replacement", "classification":"passing_control"}),
        );
        let mut wrong = pose(&alice);
        wrong.avatar_id = who("different").avatar_id;
        assert_eq!(new.relay(&wrong), 0);
        wrong = pose(&alice);
        wrong.room_hash = room_wire_hash("other-room");
        assert_eq!(new.relay(&wrong), 0);
        cases.push(
            json!({"case":"other_avatar_and_room_refused", "classification":"passing_control"}),
        );
    }
    // Use fixed synthetic time through the real rate limiter: saturation does not require a burst
    // above the route's 20/s allowance. No wall-clock sleep or live socket is needed.
    {
        let hub = PresenceHub::new();
        let sender_who = who("sender");
        let mut observer = hub.join(ROOM, &who("observer"), "observer", None).unwrap();
        let sender = hub.join(ROOM, &sender_who, "sender", None).unwrap();
        drain(&mut observer);
        let start = Instant::now();
        let mut rate = PoseRate::new(start);
        for i in 0..MEMBER_QUEUE {
            assert!(rate.allow_at(start + Duration::from_millis(i as u64 * 60)));
            assert_eq!(sender.relay(&pose(&sender_who)), 1);
        }
        let newcomer_who = who("newcomer");
        let newcomer = hub.join(ROOM, &newcomer_who, "newcomer", None).unwrap();
        let queued = drain(&mut observer);
        assert_eq!(queued.len(), MEMBER_QUEUE);
        assert!(queued.iter().all(|m| event(m) == "pose"));
        assert_eq!(newcomer.relay(&pose(&newcomer_who)), 2);
        assert_eq!(local_id(observer.rx.try_recv().unwrap()), newcomer.local_id);
        cases.push(json!({"case":"full_queue_loses_join_then_receives_unknown_id", "classification":"counterexample",
            "queue_size":MEMBER_QUEUE,"new_id":newcomer.local_id,"join_announcements_received":0,
            "rate_limited_fill_span_ms":(MEMBER_QUEUE-1)*60}));
        for _ in 0..MEMBER_QUEUE {
            sender.relay(&pose(&sender_who));
        }
        drop(newcomer);
        let queued = drain(&mut observer);
        assert_eq!(queued.len(), MEMBER_QUEUE);
        assert!(queued.iter().all(|m| event(m) == "pose"));
        assert_eq!(hub.members(ROOM), 2);
        cases.push(json!({"case":"full_queue_loses_leave", "classification":"counterexample"}));
    }
    {
        let hub = PresenceHub::new();
        let alice = who("alice");
        let mut old = hub.join(ROOM, &alice, "alice", None).unwrap();
        let sender_who = who("sender");
        let sender = hub.join(ROOM, &sender_who, "sender", None).unwrap();
        drain(&mut old);
        for _ in 0..MEMBER_QUEUE {
            sender.relay(&pose(&sender_who));
        }
        let _new = hub.join(ROOM, &alice, "alice-new", None).unwrap();
        let queued = drain(&mut old);
        assert_eq!(queued.len(), MEMBER_QUEUE);
        assert!(queued.iter().all(|m| event(m) == "pose"));
        assert!(
            old.rx.is_closed(),
            "sender removal eventually terminates the receiver"
        );
        assert_eq!(old.relay(&pose(&alice)), 2);
        cases.push(json!({"case":"full_queue_loses_close_but_receiver_eventually_closes", "classification":"counterexample",
            "stale_relay_recipients":2,"permanent_socket_claim":false}));
    }
    {
        let start = Instant::now();
        let mut rate = PoseRate::new(start);
        for _ in 0..MAX_POSES_PER_SEC {
            assert!(rate.allow_at(start));
        }
        assert!(!rate.allow_at(start));
        assert!(rate.allow_at(start + Duration::from_secs(1)));
        cases.push(json!({"case":"pose_rate_limit_and_reset", "classification":"passing_control"}));
    }
    assert!(auth::build_verifier(true).is_err());
    let open = auth::build_verifier(false).unwrap();
    assert_eq!(open.name(), "open");
    assert_eq!(
        open.verify_http("GET", "http://audit.invalid/wss", None)
            .unwrap()
            .pubkey,
        None
    );
    assert_eq!(
        open.verify_ws_envelope("http://audit.invalid/wss", &json!({}))
            .unwrap()
            .pubkey,
        None
    );
    cases.push(json!({"case":"required_auth_refuses_start_open_mode_has_no_principal", "classification":"passing_control"}));
    let token = auth::IngestToken::new("audit-fixture-token-0001").unwrap();
    assert!(token.check_header(None).is_err());
    assert!(token.check_header(Some("Bearer wrong")).is_err());
    assert!(token
        .check_header(Some("Bearer audit-fixture-token-0001"))
        .is_ok());
    cases.push(json!({"case":"ingest_token_refuses_missing_and_wrong", "classification":"passing_control"}));
    let mut headers = HeaderMap::new();
    for (name, value) in [
        ("host", "internal:4100"),
        ("x-forwarded-host", "codescape.example.invalid"),
        ("x-forwarded-for", "203.0.113.99, 198.51.100.7"),
        ("x-forwarded-proto", "https"),
    ] {
        headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_static(value),
        );
    }
    let peer = Some("192.0.2.10".parse().unwrap());
    let off = ProxyTrust::new(false, "/codescape").unwrap();
    assert_eq!(off.host(&headers, peer), Some("internal:4100"));
    assert_eq!(off.client_ip(&headers, peer), peer);
    let narrowed = ProxyTrust::new(true, "/codescape")
        .unwrap()
        .from_peers(vec![parse_peer("192.0.2.10").unwrap()]);
    assert_eq!(
        narrowed.client_ip(&headers, peer),
        Some("198.51.100.7".parse().unwrap())
    );
    assert_eq!(
        narrowed.host(&headers, peer),
        Some("codescape.example.invalid")
    );
    assert_eq!(
        narrowed.host(&headers, Some("192.0.2.11".parse().unwrap())),
        Some("internal:4100")
    );
    assert!(!narrowed.applies(None));
    cases.push(json!({"case":"proxy_off_and_peer_allowlist_enforced_last_xff_selected", "classification":"passing_control"}));
    assert!(ProxyTrust::new(true, "").unwrap().applies(None));
    cases.push(json!({"case":"enabled_proxy_without_peers_trusts_any_peer", "classification":"documented_configuration_boundary"}));
    let origins = OriginPolicy::new(&[]).unwrap();
    assert!(origins.allows(None, Some("internal:4100")));
    assert!(origins.allows(Some("http://localhost:5173"), Some("internal:4100")));
    assert!(!origins.allows(Some("https://other.invalid"), Some("internal:4100")));
    assert!(!origins.allows(Some("null"), Some("internal:4100")));
    cases.push(json!({"case":"origin_filter_allows_native_clients_and_refuses_foreign_browser_origin", "classification":"passing_control"}));
    println!("{}", serde_json::to_string_pretty(&json!({"scope":"production library, synthetic fixtures; no live socket or model", "cases":cases})).unwrap());
}
