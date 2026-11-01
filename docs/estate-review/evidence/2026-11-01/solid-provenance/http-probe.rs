//! Synthetic local HTTP provenance checks; real filesystem/Git, no chain calls.
use actix_web::test;
use serde_json::{json, Value};
use solid_pod_rs::storage::fs::FsBackend;
use solid_pod_rs_server::{build_app, AppState};
use std::{path::Path, process::Command, sync::Arc};

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}
fn acl(mode: Option<&str>) -> String {
    let mut rule = json!({"@type":"acl:Authorization","acl:agentClass":{"@id":"foaf:Agent"},"acl:accessTo":{"@id":"/alice/"},"acl:default":{"@id":"/alice/"},"acl:mode":[{"@id":"acl:Read"},{"@id":"acl:Write"},{"@id":"acl:Append"},{"@id":"acl:Control"}]});
    if let Some(mode) = mode {
        rule["acl:condition"] = json!([{"@type":"acl:ProvenanceAnchor","acl:anchorMode":mode}]);
    }
    json!({"@graph":[rule]}).to_string()
}
async fn run_case(name: &str, compiled_git: bool) -> Value {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let repo = root.join("alice");
    std::fs::create_dir(&repo).unwrap();
    let mode = match name {
        "inline_missing" | "inline_unreadable" => Some("always"),
        "epoch_missing" | "epoch_corrupt_batch" => Some("epoch"),
        _ => None,
    };
    std::fs::write(root.join("alice.acl"), acl(mode)).unwrap();
    if name != "non_git" {
        git(&repo, &["init", "-b", "main"]);
    }
    if name == "mark_failure" {
        std::fs::write(repo.join(".git/index.lock"), "synthetic held lock").unwrap();
    }
    if name == "sidecar_failure" {
        std::fs::create_dir(repo.join("doc.bin.prov.ttl")).unwrap();
    }
    if name == "inline_unreadable" || name == "epoch_corrupt_batch" {
        std::fs::create_dir_all(root.join(".well-known/gitmark")).unwrap();
        std::fs::write(
            root.join(".well-known/gitmark/prov.json"),
            "invalid trail json",
        )
        .unwrap();
    }
    if name == "epoch_corrupt_batch" {
        std::fs::create_dir_all(root.join(".well-known/prov")).unwrap();
        std::fs::write(
            root.join(".well-known/prov/epoch.json"),
            "invalid batch json",
        )
        .unwrap();
    }
    let fs = FsBackend::new(root.to_path_buf()).await.unwrap();
    let mut state = AppState::new(Arc::new(fs));
    state.data_root = Some(root.to_path_buf());
    let app = test::init_service(build_app(state)).await;
    let (path, content_type, body) = if name == "excluded_acl" {
        ("/alice/doc.bin.acl","application/ld+json",json!({"@graph":[{"@type":"acl:Authorization","acl:agentClass":{"@id":"foaf:Agent"},"acl:accessTo":{"@id":"/alice/doc.bin"},"acl:mode":{"@id":"acl:Control"}}]}).to_string())
    } else {
        (
            "/alice/doc.bin",
            "application/octet-stream",
            "synthetic stored bytes".to_owned(),
        )
    };
    let req = test::TestRequest::put()
        .uri(path)
        .insert_header(("Content-Type", content_type))
        .set_payload(body.clone())
        .to_request();
    let response = test::call_service(&app, req).await;
    let status = response.status().as_u16();
    let h = |key: &str| {
        response
            .headers()
            .get(key)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let summary = h("x-provenance").expect("receipt header");
    let commit = h("x-provenance-commit");
    assert_eq!(status, 201, "{name}: {summary}");
    assert_eq!(
        std::fs::read(root.join(path.trim_start_matches('/'))).unwrap(),
        body.as_bytes()
    );
    let sidecar = repo.join("doc.bin.prov.ttl").is_file();
    if !compiled_git {
        assert_eq!(summary, "skipped; skipped=not-configured");
        assert!(commit.is_none());
        assert!(!sidecar);
    } else {
        match name {
            "non_git" => {
                assert_eq!(summary, "skipped; skipped=not-git-backed");
                assert!(commit.is_none());
            }
            "excluded_acl" => {
                assert_eq!(summary, "skipped; skipped=excluded-path");
                assert!(commit.is_none());
                assert!(!repo.join("doc.bin.acl.prov.ttl").exists());
            }
            "mark_failure" => {
                assert!(summary.starts_with("resource-stored; mark-error="));
                assert!(commit.is_none());
                assert!(!sidecar);
            }
            "sidecar_failure" => {
                assert!(summary
                    .starts_with("local-mark-committed; mark-error=PROV-O sidecar write failed:"));
                assert!(commit.is_some());
                assert!(!sidecar);
            }
            "inline_unreadable" => {
                assert!(summary.starts_with("local-mark-committed; anchor-error="));
                assert!(commit.is_some());
                assert!(sidecar);
            }
            _ => {
                assert_eq!(summary, "local-mark-committed");
                assert!(commit.is_some());
                assert!(sidecar);
            }
        }
    }
    if let Some(ref sha) = commit {
        assert_eq!(
            git(&repo, &["show", &format!("{sha}:doc.bin")]),
            "synthetic stored bytes"
        );
    }
    if name == "delete" {
        let before = git(&repo, &["rev-parse", "HEAD"]);
        let req = test::TestRequest::delete()
            .uri("/alice/doc.bin")
            .to_request();
        let response = test::call_service(&app, req).await;
        assert_eq!(response.status().as_u16(), 204);
        assert!(!response.headers().contains_key("x-provenance"));
        assert!(!repo.join("doc.bin").exists());
        assert_eq!(before, git(&repo, &["rev-parse", "HEAD"]));
        assert!(repo.join("doc.bin.prov.ttl").exists());
        return json!({"case":"delete","initial_put_summary":summary,"delete_status":204,"delete_receipt":null,"resource_removed":true,"git_head_unchanged":true,"old_sidecar_remains":true});
    }
    let epoch = root.join(".well-known/prov/epoch.json");
    let epoch_state = if epoch.exists() {
        Some(std::fs::read_to_string(epoch).unwrap())
    } else {
        None
    };
    if name == "epoch_missing" {
        assert!(epoch_state.is_none());
    }
    if name == "epoch_corrupt_batch" {
        assert_eq!(epoch_state.as_deref(), Some("invalid batch json"));
    }
    json!({"case":name,"status":status,"resource_stored":true,"summary":summary,"commit_present":commit.is_some(),"commit_contains_written_bytes":commit.is_some(),"sidecar_file":sidecar,"epoch_state":epoch_state})
}
#[actix_web::main]
async fn main() {
    let compiled_git = std::env::args().nth(1).as_deref() == Some("git");
    let mut cases = vec![];
    for name in if compiled_git {
        vec![
            "ordinary",
            "non_git",
            "excluded_acl",
            "mark_failure",
            "sidecar_failure",
            "inline_missing",
            "inline_unreadable",
            "epoch_missing",
            "epoch_corrupt_batch",
            "delete",
        ]
    } else {
        vec!["default_build"]
    } {
        cases.push(run_case(name, compiled_git).await);
    }
    println!("{}",serde_json::to_string_pretty(&json!({"mode":if compiled_git{"git"}else{"default"},"git_version":String::from_utf8(Command::new("git").arg("--version").output().unwrap().stdout).unwrap().trim(),"cases":cases,"limits":"Local build_app with disposable FsBackend/Git fixtures; no live service, external anchoring, deployment, crash or restart test."})).unwrap());
}
