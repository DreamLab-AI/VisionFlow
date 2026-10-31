//! Audit probe: real Ontomark CLI, synthetic corpus, loopback HTTP only.
//! Usage: ingest-boundary-probe /absolute/path/to/ontomark
//! Compile with rustc --edition=2021 and the checkout's serde_json rlib.
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

struct Stub {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
}
impl Stub {
    fn start(content: &str, source: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        let content = content.to_owned();
        // Threads end with this probe process; no subprocess inherits listeners.
        thread::spawn(move || {
            for incoming in listener.incoming() {
                let mut stream: TcpStream = incoming.unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                    .unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buf = [0; 8192];
                    let n = stream.read(&mut buf).unwrap();
                    bytes.extend_from_slice(&buf[..n]);
                    if n == 0 {
                        break;
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    if let Some((head, body)) = text.split_once("\r\n\r\n") {
                        let size = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|s| s.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if body.len() >= size {
                            break;
                        }
                    }
                }
                let request = String::from_utf8(bytes).unwrap();
                let head = request.starts_with("HEAD ");
                log.lock().unwrap().push(request);
                let body = if source {
                    content.clone()
                } else {
                    json!({"choices":[{"message":{"content":content}}]}).to_string()
                };
                let media = if source {
                    "text/plain"
                } else {
                    "application/json"
                };
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: {media}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), if head { "" } else { &body });
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        Self { url, requests }
    }
}

struct Fixture {
    root: PathBuf,
    bin: PathBuf,
}
impl Fixture {
    fn new(bin: &Path) -> Self {
        let root =
            std::env::temp_dir().join(format!("ontomark-ingest-audit-{}", std::process::id()));
        assert!(!root.exists());
        std::fs::create_dir_all(root.join("repo/ontology")).unwrap();
        std::fs::create_dir_all(root.join("repo/knowledge")).unwrap();
        std::fs::create_dir_all(root.join("work")).unwrap();
        std::fs::create_dir_all(root.join("sources")).unwrap();
        std::fs::write(root.join("repo/ontology/vocabulary.yaml"), "version: 1\nnamespace: 'urn:ngm:class:'\nprefixes:\n  vc: 'https://narrativegoldmine.com/ns/v1#'\ntypes:\n  Class: { owl: 'owl:Class', required: [resource, status] }\nrelations:\n  requires: { owl: 'vc:requires' }\n").unwrap();
        for name in ["Quantum Widget", "Gamma"] {
            let slug = name.to_lowercase().replace(' ', "-");
            std::fs::write(root.join(format!("repo/knowledge/{name}.md")), format!("---\ntype: Class\ntitle: {name}\npublic: true\nresource: urn:ngm:class:{slug}\nstatus: stable\n---\nAbout {name}.\n")).unwrap();
        }
        std::fs::write(
            root.join("sources/a.md"),
            "# Widgets\n\nThe Quantum Widget requires the Gamma engine to run.\n",
        )
        .unwrap();
        Self {
            root,
            bin: bin.to_owned(),
        }
    }
    fn run(&self, endpoint: &str, extra: &[&str]) -> Value {
        let mut cmd = Command::new(&self.bin);
        for key in [
            "ONTOMARK_JEV",
            "ONTOMARK_JEV_ALLOW_CLOUD",
            "ONTOMARK_JEV_LOCAL",
            "ONTOMARK_JEV_ENDPOINT",
            "AGENTBOX_SYSTEM_ONE_BACKEND_LOCAL",
            "AGENTBOX_SYSTEM_ONE_PROXY",
            "AGENTBOX_SYSTEM_ONE_ENDPOINT",
            "VAULT_NOSTR_SECRET",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ] {
            cmd.env_remove(key);
        }
        let out = cmd
            .current_dir(self.root.join("work"))
            .arg("--repo")
            .arg(self.root.join("repo"))
            .arg("--json")
            .arg("ingest")
            .arg("--source")
            .arg(self.root.join("sources"))
            .args([
                "--target",
                "Quantum Widget",
                "--endpoint",
                endpoint,
                "--model",
                "audit-stub",
            ])
            .args(extra)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn bundle(&self, summary: &Value) -> PathBuf {
        self.root
            .join("work")
            .join(summary["bundle"].as_str().unwrap())
    }
    fn evidence(&self, summary: &Value) -> Value {
        read(&self.bundle(summary).join("evidence.json"))
    }
    fn ledger(&self) -> Value {
        read(&self.root.join("work/.ontomark/ingest/ledger.json"))
    }
    fn reset(&self) {
        std::fs::remove_dir_all(self.root.join("work")).unwrap();
        std::fs::create_dir(self.root.join("work")).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn compact(v: &Value) -> Value {
    json!({"request_id":v["request_id"], "complete":v["complete"], "budget_exhausted":v["budget_exhausted"], "model_calls":v["model_calls"], "cache_hits":v["cache_hits"], "pages":v["pages"], "source_changes":v["sources"].as_array().unwrap().iter().map(|s|s["change"].clone()).collect::<Vec<_>>()})
}
fn main() {
    let bin = std::fs::canonicalize(std::env::args().nth(1).expect("binary argument")).unwrap();
    let f = Fixture::new(&bin);
    let original = std::fs::read(f.root.join("repo/knowledge/Quantum Widget.md")).unwrap();
    let malformed = Stub::start("not JSON", false);
    let first = f.run(&malformed.url, &[]);
    let first_evidence = f.evidence(&first);
    let second = f.run(&malformed.url, &[]);
    assert_eq!(first["complete"], true);
    assert_eq!(
        first_evidence["targets"][0]["steps"][0]["dropped_chunks"],
        json!([0])
    );
    assert_eq!(second["sources"][0]["change"], "unchanged");
    assert_eq!(second["model_calls"], 1);
    assert_eq!(f.ledger()["items"].as_object().unwrap().len(), 1);
    let malformed_result = json!({"first":compact(&first), "second":compact(&second), "dropped_chunks":first_evidence["targets"][0]["steps"][0]["dropped_chunks"], "ledger_entries":1});

    f.reset();
    let valid = Stub::start(
        r#"{"assertions":[{"key":"requires","value":"Gamma","evidence":"requires the Gamma engine"}]}"#,
        false,
    );
    let empty = Stub::start(r#"{"assertions":[]}"#, false);
    let a = f.run(&valid.url, &[]);
    assert_eq!(a["pages"], json!(["Quantum Widget"]));
    let page_path = f.bundle(&a).join("manifest/Quantum Widget.md");
    assert!(page_path.exists());
    let b = f.run(&empty.url, &[]);
    assert_eq!(a["request_id"], b["request_id"]);
    assert_eq!(b["pages"], json!([]));
    assert!(!page_path.exists());
    let endpoint_result = json!({"valid_endpoint":compact(&a), "empty_endpoint":compact(&b), "same_request_id":true, "previous_manifest_removed":true});

    let cached = f.run(&valid.url, &["--budget-calls", "0"]);
    assert_eq!(cached["model_calls"], 0);
    assert_eq!(cached["cache_hits"], 1);
    assert_eq!(cached["pages"], json!(["Quantum Widget"]));
    std::fs::remove_dir_all(f.root.join("work/.ontomark/ingest/cache")).unwrap();
    let no_cache = f.run(&valid.url, &["--budget-calls", "0"]);
    assert_eq!(cached["request_id"], no_cache["request_id"]);
    assert_eq!(no_cache["budget_exhausted"], true);
    assert_eq!(no_cache["complete"], false);
    assert!(!page_path.exists());
    let budget_result = json!({"cached":compact(&cached), "without_cache":compact(&no_cache), "previous_manifest_removed":true});

    f.reset();
    std::fs::write(f.root.join("sources/b.md"), "Another source.").unwrap();
    let both = f.run(&empty.url, &[]);
    assert_eq!(both["sources"].as_array().unwrap().len(), 2);
    std::fs::remove_file(f.root.join("sources/b.md")).unwrap();
    let remaining = f.run(&empty.url, &[]);
    assert_eq!(remaining["sources"].as_array().unwrap().len(), 1);
    assert_eq!(f.ledger()["items"].as_object().unwrap().len(), 2);
    let removed_result = json!({"reported_source_count":1, "ledger_entries_after_removal":2, "complete":remaining["complete"]});

    f.reset();
    let negative = "The Quantum Widget does not require Gamma.";
    std::fs::write(
        f.root.join("sources/a.md"),
        format!("# Widgets\n\n{negative}\n"),
    )
    .unwrap();
    let negated = Stub::start(
        &json!({"assertions":[{"key":"requires","value":"Gamma","evidence":negative}]}).to_string(),
        false,
    );
    let negation = f.run(&negated.url, &[]);
    let ne = f.evidence(&negation);
    assert_eq!(negation["pages"], json!(["Quantum Widget"]));
    let negation_result =
        json!({"summary":compact(&negation), "accepted":ne["targets"][0]["steps"][0]["accepted"]});

    let web = Stub::start("Quantum Widget and Gamma.", true);
    let planned = f.run(
        &valid.url,
        &["--source", &web.url, "--plan-only", "--budget-calls", "0"],
    );
    assert_eq!(planned["plan_only"], true);
    let verbs: Vec<_> = web
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|s| s.split_whitespace().next().unwrap().to_owned())
        .collect();
    assert_eq!(verbs, ["HEAD", "GET"]);
    assert_eq!(
        std::fs::read(f.root.join("repo/knowledge/Quantum Widget.md")).unwrap(),
        original
    );
    println!("{}", serde_json::to_string_pretty(&json!({"binary":bin, "malformed_reply":malformed_result, "endpoint_identity":endpoint_result, "budget_replacement":budget_result, "removed_source":removed_result, "negated_relation":negation_result, "plan_only_source_requests":verbs, "target_unchanged":true})).unwrap());
}
