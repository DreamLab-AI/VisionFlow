//! Audit probes of unchanged production code; these assert observed defects, not desired behaviour.
use dream_engine::journal::{session_for, Journal, JournalConfig};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn api(replies: Vec<(u16, Value)>) -> (Journal, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let body = loop {
                let mut chunk = [0; 4096];
                let n = socket.read(&mut chunk).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&bytes);
                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text[..end]
                        .lines()
                        .find_map(|l| {
                            let (key, val) = l.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| val.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + len {
                        break bytes[end + 4..end + 4 + len].to_vec();
                    }
                }
            };
            let index = {
                let mut guard = sink.lock().unwrap();
                guard.push(serde_json::from_slice(&body).unwrap());
                guard.len() - 1
            };
            let (status, value) = replies
                .get(index)
                .cloned()
                .unwrap_or((201, json!({"event_id":"recovered"})));
            let body = value.to_string();
            socket.write_all(format!("HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        }
    });
    (
        Journal::new(
            JournalConfig {
                base_url: url,
                api_key: None,
            },
            "audit",
        ),
        seen,
    )
}

#[tokio::test]
async fn failed_completion_is_reported_unpaired_zero() {
    let (j, seen) = api(vec![(201, json!({"event_id":"called"})), (500, json!({}))]).await;
    let call = j.called("fixture.effect", json!({})).await;
    j.completed(call, true, json!({})).await;
    assert_eq!(
        (j.stats().recorded, j.stats().failed, j.stats().unpaired),
        (1, 1, 0)
    );
    assert_eq!(seen.lock().unwrap()[1]["causation"], "called");
    println!("completion failure: {:?}", j.stats());
}

#[tokio::test]
async fn failed_call_can_leave_successful_orphan_completion() {
    let (j, seen) = api(vec![
        (500, json!({})),
        (201, json!({"event_id":"completed"})),
    ])
    .await;
    let call = j.called("fixture.effect", json!({})).await;
    j.completed(call, true, json!({})).await;
    assert_eq!(
        (j.stats().recorded, j.stats().failed, j.stats().unpaired),
        (1, 1, 0)
    );
    assert!(seen.lock().unwrap()[1].get("causation").is_none());
    println!("orphan completion: {:?}", j.stats());
}

#[tokio::test]
async fn empty_acknowledgement_id_counts_as_recorded() {
    let (j, seen) = api(vec![(201, json!({"event_id":"","duplicate":true}))]).await;
    let call = j.called("fixture.effect", json!({})).await;
    j.completed(call, true, json!({})).await;
    assert_eq!(j.stats().recorded, 2);
    assert_eq!(seen.lock().unwrap()[1]["causation"], "");
}

#[tokio::test]
async fn breaker_does_not_probe_recovered_endpoint() {
    let (j, seen) = api(vec![(500, json!({})); 3]).await;
    for _ in 0..5 {
        j.turn_started(json!({})).await;
    }
    assert_eq!((j.stats().failed, j.stats().skipped), (3, 2));
    assert_eq!(seen.lock().unwrap().len(), 3);
}

#[test]
fn session_suffix_can_be_removed_and_names_normalise_to_same_session() {
    assert_eq!(
        session_for(&"a".repeat(160), 1),
        session_for(&"a".repeat(160), 2)
    );
    assert_eq!(session_for("a/b", 1), session_for("a-b", 1));
    assert_eq!(session_for("same-night", 1), session_for("same-night", 1));
    println!("160-character night ID removes the changing time suffix");
}

fn tick(
    state: &str,
    health: Option<&str>,
    initial: &str,
    fail_start: bool,
    at: &str,
) -> (String, String, bool) {
    use std::{fs, os::unix::fs::PermissionsExt, process::Command};
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join(".agentbox");
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("adr-2071-api-down-night"), "2026-11-01\n").unwrap();
    fs::write(dir.join("adr-2071-api-down-night.state"), state).unwrap();
    if let Some(h) = health {
        fs::write(dir.join("dream-last-night.json"), h).unwrap();
    }
    fs::write(temp.path().join("status"), initial).unwrap();
    let stub = temp.path().join("supervisorctl");
    fs::write(
        &stub,
        r#"#!/usr/bin/env bash
case "$1" in
 status) printf 'management-api %s\n' "$(cat "$WORKSPACE/status")" ;;
 stop) echo STOPPED > "$WORKSPACE/status" ;;
 start) [ "$AUDIT_FAIL_START" = 1 ] && exit 1; echo RUNNING > "$WORKSPACE/status" ;;
 *) exit 99 ;;
esac
"#,
    )
    .unwrap();
    fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).unwrap();
    let epoch = Command::new("date")
        .args(["-u", "-d", at, "+%s"])
        .output()
        .unwrap();
    assert!(epoch.status.success());
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/activation/adr-2071-api-down-night.sh");
    let out = Command::new("bash")
        .arg(script)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap())
        .env("WORKSPACE", temp.path())
        .env("SUPERVISORCTL", stub)
        .env("AUDIT_FAIL_START", if fail_start { "1" } else { "0" })
        .env(
            "ADR2071_NOW",
            String::from_utf8(epoch.stdout).unwrap().trim(),
        )
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let state = fs::read_to_string(dir.join("adr-2071-api-down-night.state")).unwrap();
    let status = fs::read_to_string(temp.path().join("status")).unwrap();
    let retired = !dir.join("adr-2071-api-down-night").exists();
    println!("tick: state={state:?}, status={status:?}, retired={retired}");
    (state, status, retired)
}
const STOPPED: &str = "night=2026-11-01\nphase=stopped\nstopped_at=2026-11-01T00:35:00Z\n";

#[test]
fn restart_failure_still_marks_done_and_retires() {
    let (state, status, retired) = tick(STOPPED, None, "STOPPED", true, "2026-11-01 07:35:00 UTC");
    assert!(state.contains("phase=done\n") && state.contains("reason=deadline\n"));
    assert_eq!(status, "STOPPED");
    assert!(retired);
}
#[test]
fn old_done_state_consumes_fresh_marker_without_stopping() {
    let (state, status, retired) = tick(
        "night=2026-10-04\nphase=done\n",
        None,
        "RUNNING",
        false,
        "2026-11-01 00:35:00 UTC",
    );
    assert!(!state.contains("night=2026-11-01"));
    assert_eq!(status, "RUNNING");
    assert!(retired);
}
#[test]
fn same_date_text_restarts_before_window_without_valid_json() {
    let (state, status, retired) = tick(
        STOPPED,
        Some("not JSON: \"date\":\"2026-11-01\""),
        "STOPPED",
        false,
        "2026-11-01 00:45:00 UTC",
    );
    assert!(state.contains("reason=night-record\n"));
    assert_eq!(status.trim(), "RUNNING");
    assert!(retired);
}
#[test]
fn matching_record_masks_already_running_interruption() {
    let (state, _, _) = tick(
        STOPPED,
        Some(r#"{"date":"2026-11-01"}"#),
        "RUNNING",
        false,
        "2026-11-01 03:00:00 UTC",
    );
    assert!(state.contains("reason=night-record\n"));
    assert!(!state.contains("interrupted"));
}

#[test]
fn c3_accepts_asserted_stop_without_restart_or_engine_success() {
    use std::{fs, process::Command};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read_to_string(root.join("scripts/activation/adr-2087-check.sh")).unwrap();
    let start = source.find("if [ -n \"$API_DOWN_NIGHT\" ]; then").unwrap();
    let end = source[start..].find("# ── verdict and receipt").unwrap() + start;
    let script = format!(
        "record() {{ printf '%s %s\\n' \"$1\" \"$2\"; }}\n{}",
        &source[start..end]
    );
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join(".agentbox");
    fs::create_dir(&dir).unwrap();
    let health = dir.join("dream-last-night.json");
    fs::write(&health,r#"{"date":"2026-11-01","outcomes":[{"verdict":"REJECT"}],"journal":[{"failed":1,"skipped":0}]}"#).unwrap();
    fs::write(dir.join("adr-2071-api-down-night.state"),"night=2026-11-01\nphase=done\nreason=deadline\nstopped_at=not-a-timestamp\nengine_exit=99\n").unwrap();
    let out = Command::new("bash")
        .args(["-c", &script])
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap())
        .env("WORKSPACE", temp.path())
        .env("SCRATCH", temp.path())
        .env("NIGHT_RECORD", health)
        .env("API_DOWN_NIGHT", "2026-11-01")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "C3 PASS");
    println!("C3 PASS with invalid stop timestamp, missing restart and engine_exit=99");
}
