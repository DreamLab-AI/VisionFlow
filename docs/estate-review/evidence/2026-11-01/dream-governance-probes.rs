//! Observation probes; production code is unchanged and all transport is loopback.
use dream_engine::{
    digest, governance as g,
    inbox::{self, InboxItem},
    relay::{self, NostrEvent, UnsignedEvent},
    zone_crypto as z,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio_tungstenite::tungstenite::Message;

fn key(seed: u8) -> relay::SigningKey {
    nostr_bbs_core::keys::signing_key_from_bytes(&[seed; 32]).unwrap()
}
fn item() -> InboxItem {
    InboxItem {
        id: "case-a".into(),
        kind: "question".into(),
        repo: "fixture".into(),
        night_id: "2026-11-01-fixture".into(),
        date: "2026-11-01".into(),
        text: "Should this change be considered?".into(),
        status: "open".into(),
        answer: String::new(),
        last_surfaced: 0,
        published_event_id: "current-request".into(),
        decision_event_id: String::new(),
        withdrawn_event_id: String::new(),
    }
}
fn decision(d: &str, bound: Option<&str>, content: Value, at: u64) -> NostrEvent {
    let k = key(2);
    let mut tags = vec![vec!["d".into(), d.into()]];
    if let Some(b) = bound {
        tags.push(vec!["e".into(), b.into()]);
    }
    relay::sign(
        UnsignedEvent {
            pubkey: relay::pubkey_hex(&k),
            created_at: at,
            kind: 31403,
            tags,
            content: content.to_string(),
        },
        &k,
    )
    .unwrap()
}
fn plan(i: &[InboxItem], events: &[NostrEvent]) -> Vec<g::Applied> {
    g::plan_decisions(i, events, &relay::pubkey_hex(&key(1)), &HashMap::new())
}
#[test]
fn any_other_valid_signer_and_missing_request_binding_are_accepted() {
    let i = item();
    let ev = decision(&g::case_d(&i.id), None, json!({"action":"approve"}), 10);
    assert!(relay::verify(&ev));
    assert_eq!(plan(&[i.clone()], &[ev]).len(), 1);
    let bound = decision(
        &g::case_d(&i.id),
        Some(&i.published_event_id),
        json!({"action":"reject","reasoning":"fixture"}),
        11,
    );
    assert_eq!(plan(&[i], &[bound])[0].resolution.answer, "reject: fixture");
    println!("arbitrary second fixture key accepted with or without request e tag; no role data supplied");
}
#[test]
fn newest_stale_or_delegate_event_suppresses_older_bound_decision() {
    let i = item();
    let d = g::case_d(&i.id);
    let good = decision(
        &d,
        Some(&i.published_event_id),
        json!({"action":"approve"}),
        10,
    );
    assert_eq!(plan(&[i.clone()], &[good.clone()]).len(), 1);
    let stale = decision(&d, Some("old-request"), json!({"action":"reject"}), 20);
    assert!(plan(&[i.clone()], &[good.clone(), stale]).is_empty());
    let delegate = decision(
        &d,
        Some(&i.published_event_id),
        json!({"action":"delegate","delegate_to":"fixture"}),
        30,
    );
    assert!(plan(&[i], &[good, delegate]).is_empty());
}
#[test]
fn later_supersession_does_not_revisit_resolved_local_item() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("inbox.json");
    let i = item();
    inbox::save_to(&p, &[i.clone()]).unwrap();
    assert!(inbox::resolve_in(&p, &i.id, "answered", "approve: first", "first-decision").unwrap());
    let mut ev = decision(
        &g::case_d(&i.id),
        Some(&i.published_event_id),
        json!({"action":"reject","reasoning":"changed"}),
        50,
    );
    let k = key(2);
    let mut tags = ev.tags.clone();
    tags.push(vec![
        "e".into(),
        "first-decision".into(),
        "".into(),
        "supersedes".into(),
    ]);
    ev = relay::sign(
        UnsignedEvent {
            pubkey: ev.pubkey,
            kind: ev.kind,
            created_at: ev.created_at,
            tags,
            content: ev.content,
        },
        &k,
    )
    .unwrap();
    assert!(plan(&inbox::load_from(&p), &[ev]).is_empty());
    assert_eq!(inbox::load_from(&p)[0].answer, "approve: first");
}
#[test]
fn panel_ack_checks_action_and_time_but_not_outcome_or_role() {
    let mut i = item();
    i.kind = "alert".into();
    let ev = decision(
        g::PANEL_D,
        None,
        json!({"action":g::ACK_ALERTS_ACTION,"outcome":"reject"}),
        20,
    );
    let mut times = HashMap::new();
    times.insert(i.id.clone(), 10);
    let out = g::plan_decisions(&[i], &[ev], &relay::pubkey_hex(&key(1)), &times);
    assert_eq!(out[0].resolution.status, "dismissed");
}
#[test]
fn republished_reopened_id_updates_closed_record_instead_of_open_record() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("inbox.json");
    let mut old = item();
    old.status = "answered".into();
    let mut reopened = item();
    reopened.published_event_id.clear();
    inbox::save_to(&p, &[old, reopened]).unwrap();
    assert!(inbox::mark_published_in(&p, "case-a", "new-request").unwrap());
    let rows = inbox::load_from(&p);
    assert_eq!(rows[0].published_event_id, "new-request");
    assert!(rows[1].published_event_id.is_empty());
    let w = g::plan_withdrawals(&rows);
    assert_eq!(w[0].item.published_event_id, "new-request");
    assert!(!w[0].with_coordinate);
    println!("withdrawal now directly targets new-request on the closed row; open row remains unpublished");
}
#[test]
fn stale_snapshot_overwrites_a_durable_resolution() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("inbox.json");
    inbox::save_to(&p, &[item()]).unwrap();
    let snapshot = inbox::load_from(&p);
    inbox::resolve_in(&p, "case-a", "answered", "approved", "d").unwrap();
    inbox::save_to(&p, &snapshot).unwrap();
    assert_eq!(inbox::load_from(&p)[0].status, "open");
    // Deterministic interleaving of public load/save boundaries, not a claimed live race.
}
#[test]
fn malformed_zone_metadata_selects_plaintext_despite_enabled_gate() {
    let zones = z::parse_zones(Some("malformed"));
    let zone = z::section_to_zone("zone4-chat-with-agents", &zones).unwrap_or_default();
    assert_eq!(z::write_plan(&zone, true, &zones, &[]), z::WritePlan::Plain);
    let valid = z::parse_zones(Some(r#"[{"id":"zone4","encrypted":true}]"#));
    assert!(matches!(
        z::write_plan("zone4", true, &valid, &[]),
        z::WritePlan::Refuse(_)
    ));
    let ambiguous = z::parse_zones(Some(
        r#"[{"id":"zone","encrypted":false},{"id":"zone4","encrypted":true}]"#,
    ));
    assert_eq!(
        z::section_to_zone("unmatched-section", &ambiguous).as_deref(),
        Some("zone")
    );
}
#[test]
fn digest_claims_exceed_health_and_ledger_inputs() {
    let date = "2026-11-01";
    let mut ledgers = HashMap::new();
    ledgers.insert(
        "fixture".into(),
        digest::RepoLedger {
            rows: vec![digest::LedgerLine {
                deep: "fixture".into(),
                finding: "a completed evaluator receipt exists".into(),
                verdict: "ACCEPT".into(),
                witness: "fixture-receipt".into(),
            }],
            streak: 0,
        },
    );
    assert!(
        digest::compose(date, None, &ledgers, 1, "https://example.invalid")
            .contains("Nothing was tested")
    );
    let health = digest::NightHealth {
        date: date.into(),
        outcomes: vec![digest::Outcome {
            repo: "fixture".into(),
            verdict: "REJECT".into(),
        }],
        standby: vec![digest::Standby {
            repo: "parked-repo".into(),
            reason: "marker".into(),
            streak: 0,
        }],
        ..Default::default()
    };
    let text = digest::compose(
        date,
        Some(&health),
        &HashMap::new(),
        1,
        "https://example.invalid",
    );
    assert!(text.contains("DISPROVED"));
    assert!(!text.contains("parked-repo"));
    assert!(text.contains("waiting for you on the governance panel"));
}
#[test]
fn case_context_reads_default_ledger_not_configured_alternative() {
    let t = tempfile::tempdir().unwrap();
    let ws = t.path().join("ws");
    let repo = ws.join("fixture");
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join("dream.config.json"),
        r#"{"repo":"o/r","slots":[{"deep":"d","scan":[]}],"ledgerPath":"custom.md"}"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("custom.md"),
        "| 2026-11-01 | deep | finding | witness | https://example.invalid/1 |\n",
    )
    .unwrap();
    let (cfg, _) =
        dream_engine::config::load_dream_config(&repo.join("dream.config.json")).unwrap();
    assert_eq!(cfg.ledger_path, "custom.md");
    let paths = g::CasePaths {
        workspace: ws,
        artefact_dir: t.path().join("artefacts"),
    };
    let ctx = g::case_context(&paths, &item());
    assert!(ctx.finding.is_none() && ctx.pr_url.is_none());
}

async fn relay_stub(
    emit: Option<NostrEvent>,
    auth_ok: bool,
) -> (String, Arc<Mutex<Vec<NostrEvent>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            let sink = sink.clone();
            let emit = emit.clone();
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                ws.send(Message::Text(
                    json!(["AUTH", "fixture-challenge"]).to_string().into(),
                ))
                .await
                .unwrap();
                while let Some(Ok(msg)) = ws.next().await {
                    let Message::Text(text) = msg else { continue };
                    let f: Value = serde_json::from_str(&text).unwrap();
                    match f[0].as_str() {
                        Some("AUTH") => {
                            ws.send(Message::Text(
                                json!(["OK", f[1]["id"], auth_ok, "fixture auth"])
                                    .to_string()
                                    .into(),
                            ))
                            .await
                            .unwrap();
                        }
                        Some("EVENT") => {
                            let ev: NostrEvent = serde_json::from_value(f[1].clone()).unwrap();
                            let id = ev.id.clone();
                            sink.lock().unwrap().push(ev);
                            ws.send(Message::Text(
                                json!(["OK", id, true, ""]).to_string().into(),
                            ))
                            .await
                            .unwrap();
                        }
                        Some("REQ") => {
                            if let Some(ev) = &emit {
                                ws.send(Message::Text(
                                    json!(["EVENT", f[1], ev]).to_string().into(),
                                ))
                                .await
                                .unwrap();
                            }
                            ws.send(Message::Text(
                                json!(["CLOSED", f[1], "auth-required: fixture"])
                                    .to_string()
                                    .into(),
                            ))
                            .await
                            .unwrap();
                        }
                        _ => {}
                    }
                }
            });
        }
    });
    (url, seen)
}
#[tokio::test]
async fn query_accepts_unverified_out_of_filter_events_and_closed_as_success() {
    let mut forged = decision("unrelated", None, json!({"action":"approve"}), 10);
    forged.content = "changed after signing".into();
    assert!(!relay::verify(&forged));
    let (url, _) = relay_stub(Some(forged), false).await;
    let mut s = relay::RelaySession::connect(&url, &key(1)).await.unwrap();
    let got = s
        .query(
            json!({"kinds":[31400],"ids":["not-present"]}),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
    assert_eq!(got.len(), 1);
    assert!(!relay::verify(&got[0]));
    assert_eq!(got[0].kind, 31403);
    s.close().await;
    let (url, _) = relay_stub(None, true).await;
    let mut s = relay::RelaySession::connect(&url, &key(1)).await.unwrap();
    assert!(s
        .query(json!({}), Duration::from_secs(1))
        .await
        .unwrap()
        .is_empty());
    s.close().await;
}
#[tokio::test]
async fn accepted_withdrawal_is_resent_after_local_recording_failure() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("inbox.json");
    let mut closed = item();
    closed.status = "answered".into();
    inbox::save_to(&p, &[closed]).unwrap();
    std::fs::create_dir(p.with_extension("json.tmp")).unwrap();
    let file = t.path().join("fixture-key");
    std::fs::write(&file, hex::encode([1u8; 32])).unwrap();
    let (url, seen) = relay_stub(None, true).await;
    let overrides = [
        ("DREAM_RELAY", url),
        (
            "JUNKIEJARVIS_PRIVKEY_HEX_FILE",
            file.to_string_lossy().into_owned(),
        ),
        ("AGENTBOX_ROLE_ISOLATION", "0".into()),
    ];
    let prior: Vec<_> = overrides
        .iter()
        .map(|(k, _)| (*k, std::env::var_os(k)))
        .collect();
    for (k, v) in &overrides {
        std::env::set_var(k, v);
    }
    let first = g::withdraw(&p, false).await;
    let second = g::withdraw(&p, false).await;
    for (k, v) in prior {
        match v {
            Some(v) => std::env::set_var(k, v),
            None => std::env::remove_var(k),
        }
    }
    assert_eq!((first.withdrawn, second.withdrawn), (1, 1));
    assert_eq!(
        seen.lock().unwrap().iter().filter(|e| e.kind == 5).count(),
        2
    );
    assert!(inbox::load_from(&p)[0].withdrawn_event_id.is_empty());
    println!("accepted deletion sent twice; local recording failure leaves it pending");
}
