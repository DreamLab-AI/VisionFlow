//! Audit probe for the real Codescape command-selection library.
//! Link against a freshly built codescape-server and its Cargo-selected dependencies.
//! Uses synthetic configuration, scripted replies and one loopback HTTP exchange only.
use codescape_collab::{parse_command, Command, ParseError};
use codescape_layout::ptx_source::{content_tag, PtxSource, MANIFEST_FILE, PTX_FILE};
use codescape_server::egress::EgressAllowlist;
use codescape_server::voice::jev::{self, JevClient, PostFuture, Raw, Transport};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Scripted {
    reply: String,
    seen: Mutex<Vec<Value>>,
}
impl Transport for Scripted {
    fn post<'a>(
        &'a self,
        _url: &'a str,
        _bearer: Option<&'a str>,
        body: &'a Value,
        _deadline: Duration,
    ) -> PostFuture<'a> {
        self.seen.lock().unwrap().push(body.clone());
        Box::pin(async move {
            Ok(Raw {
                status: 200,
                body: self.reply.clone(),
            })
        })
    }
}

fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}
fn choice(key: &str, confidence: f64) -> Value {
    json!({"answers":{"command":{"type":"choice","choice":key,"confidence":confidence}}})
}

fn main() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(run());
}

async fn run() {
    let cfg = env(&[
        ("CODESCAPE_JEV", "1"),
        ("CODESCAPE_JEV_ENDPOINT", "https://remote.example.invalid"),
        ("CODESCAPE_JEV_ALLOW_CLOUD", "0"),
        ("CODESCAPE_JEV_LOCAL", "0"),
        ("AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL", "0"),
    ]);
    let lookup = |k: &str| cfg.get(k).cloned();
    let resolved = jev::resolve(&lookup).unwrap();
    assert_eq!(resolved.min_confidence, 1.0);
    let configured_url = jev::egress_url(&lookup).unwrap();
    let allow = EgressAllowlist::from_urls([&configured_url]).unwrap();
    assert!(allow.check(&configured_url).is_ok());
    let proxy_cfg = env(&[
        ("CODESCAPE_JEV", "1"),
        ("AGENTBOX_SYSTEM_ONE_PROXY", "1"),
        (
            "AGENTBOX_SYSTEM_ONE_ENDPOINT",
            "https://remote.example.invalid",
        ),
        ("AGENTBOX_SYSTEM_ONE_MODEL", "fixture"),
    ]);
    assert!(jev::resolve(&|k| proxy_cfg.get(k).cloned()).is_err());
    assert!(jev::resolve(&|_| None).is_err());
    let mut cases = vec![
        json!({"case":"explicit_remote_without_cloud_permission","resolved":resolved.url,"allowlisted":true,"contacted":false}),
        json!({"case":"implicit_proxy_without_local_or_cloud_permission","refused":true}),
        json!({"case":"flag_off","refused":true}),
    ];
    let utterance = "make the picture bigger somehow";
    assert!(matches!(
        parse_command(utterance),
        Err(ParseError::NoMatch { .. })
    ));
    for (name, reply, accepted) in [
        ("default_threshold_099", choice("zoom_in", 0.99), false),
        ("default_threshold_100", choice("zoom_in", 1.0), true),
        ("out_of_range", choice("zoom_in", 1.01), false),
        ("none_at_certainty", choice("none", 1.0), false),
        ("unknown_command", choice("delete_everything", 1.0), false),
    ] {
        let transport = Arc::new(Scripted {
            reply: reply.to_string(),
            seen: Mutex::new(vec![]),
        });
        let client = JevClient::with_transport(resolved.clone(), transport.clone());
        let outcome = jev::after_grammar(parse_command(utterance), Some(&client)).await;
        assert_eq!(outcome == Ok(Command::ZoomIn), accepted);
        if !accepted {
            assert_eq!(outcome, parse_command(utterance));
        }
        assert_eq!(transport.seen.lock().unwrap().len(), 1);
        cases.push(json!({"case":name,"result":format!("{outcome:?}"),"transport_calls":1}));
        let grammar = jev::after_grammar(parse_command("help"), Some(&client)).await;
        assert_eq!(grammar, Ok(Command::Help));
        assert_eq!(transport.seen.lock().unwrap().len(), 1);
    }
    cases.push(json!({"case":"grammar_hit_bypasses_model","checked_with_all_five_replies":true}));

    // Real production HTTP transport against a bounded loopback fixture.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let handle = std::thread::spawn(move || {
        let started = Instant::now();
        let mut socket = loop {
            match listener.accept() {
                Ok((s, _)) => break s,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        started.elapsed() < Duration::from_secs(8),
                        "no loopback request"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => panic!("accept: {e}"),
            }
        };
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let (header_end, length) = loop {
            let mut buf = [0; 4096];
            let n = socket.read(&mut buf).unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&buf[..n]);
            assert!(bytes.len() < 32768);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&bytes[..end]);
                let length: usize = headers
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse().unwrap())
                    })
                    .unwrap();
                if bytes.len() >= end + 4 + length {
                    break (end + 4, length);
                }
            }
        };
        let body: Value = serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
        let reply = choice("zoom_in", 1.0).to_string();
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.len(), reply).unwrap();
        body
    });
    let local_cfg = env(&[
        ("CODESCAPE_JEV", "1"),
        ("CODESCAPE_JEV_ENDPOINT", &endpoint),
    ]);
    let lookup = |k: &str| local_cfg.get(k).cloned();
    let url = jev::egress_url(&lookup).unwrap();
    let allow = EgressAllowlist::from_urls([url]).unwrap();
    let client = JevClient::from_lookup(&lookup, &allow).unwrap();
    let outcome = jev::after_grammar(parse_command(utterance), Some(&client)).await;
    let received = handle.join().unwrap();
    assert_eq!(outcome, Ok(Command::ZoomIn));
    assert_eq!(received["state"]["utterance"], utterance);
    cases.push(json!({"case":"real_loopback_transport_default_threshold","result":format!("{outcome:?}"),"request":received}));
    cases.push(retained_ptx());
    println!("{}", serde_json::to_string_pretty(&json!({"scope":"synthetic configuration, scripted replies, loopback HTTP and extracted startup function; no command execution, live provider or GPU load", "cases":cases})).unwrap());
}

fn retained_ptx() -> Value {
    let repo = std::path::PathBuf::from(std::env::args().nth(1).expect("Codescape source root"));
    let source = std::fs::read_to_string(repo.join("deploy/codescape-entrypoint.sh")).unwrap();
    let start = source.find("build_ptx() {").unwrap();
    let end = source[start..].find("\n}\n\n# The client").unwrap() + start + 2;
    let body = &source[start..end];
    let root = std::env::temp_dir().join(format!("vf-codescape-audit-{}", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    std::fs::create_dir(&root).unwrap();
    let _cleanup = Cleanup(root.clone());
    let cached = root.join("ptx");
    std::fs::create_dir(&cached).unwrap();
    let ptx =
        std::fs::read(repo.join("crates/codescape-layout/tests/fixtures/unified_fresh_212.ptx"))
            .unwrap();
    let tag = content_tag(&ptx);
    std::fs::write(cached.join(PTX_FILE), &ptx).unwrap();
    std::fs::write(cached.join(MANIFEST_FILE), format!("visionclaw_unified source=fixture provenance=compiled isa=9.0 original={tag:016x} rewritten={tag:016x}\n")).unwrap();
    PtxSource::from_dir(&cached).unwrap();
    // Execute the verbatim production function only, with a missing source path.
    // The rest of the entrypoint (builds and server launch) is never executed.
    let script =
        format!("set -euo pipefail\nlog() {{ printf '%s\\n' \"$*\" >&2; }}\n{body}\nbuild_ptx\n");
    let out = std::process::Command::new("/bin/bash")
        .args(["-c", &script])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("CS_TARGET", root.join("target"))
        .env("CODESCAPE_PTX_DIR", &cached)
        .env("CODESCAPE_GPU_SRC", root.join("absent-gpu-source"))
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(std::fs::read(cached.join(PTX_FILE)).unwrap(), ptx);
    PtxSource::from_dir(&cached).unwrap();
    let diagnostic = String::from_utf8(out.stderr).unwrap();
    assert!(diagnostic.contains("will fall back to CPU"));
    json!({"case":"missing_gpu_source_retains_cached_ptx","function":"verbatim build_ptx from supplied source","exit":out.status.code(),"diagnostic":diagnostic.trim(),"cached_pair_still_validates":true,"gpu_loaded":false,"fixture_is_compiler_output":false})
}
