//! Observation probes for unchanged agentbox dream-engine source; no remote services.
use dream_engine::{dispatch, jev_triage as jt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

struct Scripted {
    replies: Mutex<VecDeque<Result<(u16, String), jt::TransportError>>>,
    seen: Mutex<Vec<Value>>,
}
impl Scripted {
    fn good() -> Self {
        Self {
            replies: Mutex::new(VecDeque::from([
                Ok((
                    200,
                    json!({"model":"fixture", "answers": {
                        "usefulness":{"type":"score","score":3},
                        "problem_supported":{"type":"noul","noul":0.1}
                    }})
                    .to_string(),
                )),
                Ok((
                    200,
                    json!({"answers": {
                        "fit_0":{"type":"score","score":3,"confidence":0.9},
                        "addresses_0":{"type":"noul","noul":1},
                        "violates_0":{"type":"noul","noul":0}
                    }})
                    .to_string(),
                )),
            ])),
            seen: Mutex::new(Vec::new()),
        }
    }
}
impl jt::Transport for Scripted {
    fn post<'a>(&'a self, req: &'a jt::HttpRequest) -> jt::PostFuture<'a> {
        self.seen
            .lock()
            .unwrap()
            .push(serde_json::from_str(&req.body).unwrap());
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected request");
        Box::pin(async { reply })
    }
}
fn cfg() -> jt::Config {
    jt::Config::from_lookup(&|key| match key {
        "AGENTBOX_JEV_DREAM_TRIAGE" | "AGENTBOX_SYSTEM_ONE_PROXY" => Some("1".into()),
        "AGENTBOX_SYSTEM_ONE_ENDPOINT" => Some("http://fixture.invalid/never-contacted".into()),
        "AGENTBOX_JEV_DREAM_TRIAGE_DEFER_BELOW" => Some("0.5".into()),
        _ => None,
    })
}
fn input() -> jt::Input {
    jt::Input {
        report: "Fixture hypothesis: repair a measured failure".into(),
        patch: Some("--- a/a.rs\n+++ b/a.rs\n@@ -1 +1 @@\n-old\n+new\n".into()),
        evaluators: vec!["fixture-check".into()],
        disciplines: vec!["keep-checks".into()],
    }
}
#[tokio::test]
async fn defer_survives_receipt_write_failure() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir(d.path().join(jt::RECORD_FILE)).unwrap();
    let transport = Scripted::good();
    assert!(jt::consider(&cfg(), &transport, d.path(), &input()).await);
    assert!(d.path().join(jt::RECORD_FILE).is_dir());
    assert_eq!(transport.seen.lock().unwrap().len(), 2);
    println!("defer=true although jev-triage.json cannot be written");
}
#[tokio::test]
async fn fallback_leaves_prior_deferred_receipt() {
    let d = tempfile::tempdir().unwrap();
    assert!(jt::consider(&cfg(), &Scripted::good(), d.path(), &input()).await);
    let path = d.path().join(jt::RECORD_FILE);
    let before = std::fs::read(&path).unwrap();
    let transport = Scripted {
        replies: Mutex::new(VecDeque::from([Err(jt::TransportError::Timeout)])),
        seen: Mutex::new(vec![]),
    };
    assert!(!jt::consider(&cfg(), &transport, d.path(), &input()).await);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        serde_json::from_slice::<Value>(&before).unwrap()["deferred"],
        true
    );
    println!("fallback proceeds while prior receipt still says deferred=true");
}
#[tokio::test]
async fn receipt_names_disciplines_not_sent_to_screen() {
    let d = tempfile::tempdir().unwrap();
    let mut i = input();
    i.disciplines = (0..20).map(|n| format!("rule-{n}")).collect();
    let transport = Scripted::good();
    assert!(jt::consider(&cfg(), &transport, d.path(), &i).await);
    let seen = transport.seen.lock().unwrap();
    assert_eq!(
        seen[1]["state"]["disciplines"].as_array().unwrap().len(),
        16
    );
    assert!(!seen[1].to_string().contains("rule-19"));
    let rec: Value =
        serde_json::from_slice(&std::fs::read(d.path().join(jt::RECORD_FILE)).unwrap()).unwrap();
    assert_eq!(
        rec["discipline_violations"]["disciplines"]
            .as_array()
            .unwrap()
            .len(),
        20
    );
    println!("discipline annotation names 20 rules although screen received 16");
}
#[test]
fn deletion_and_rename_can_have_empty_change_surface() {
    let deletion = "diff --git a/guard.rs b/guard.rs\ndeleted file mode 100644\n--- a/guard.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-check\n";
    let rename = "diff --git a/guard.rs b/disabled.rs\nsimilarity index 100%\nrename from guard.rs\nrename to disabled.rs\n";
    assert!(jt::change_surface(deletion).is_empty());
    assert!(jt::change_surface(rename).is_empty());
    println!("deletion and pure rename are absent from triage change_surface");
}
#[test]
fn report_limit_does_not_bound_total_triage_request() {
    let mut i = input();
    i.report = "r".repeat(8000);
    i.evaluators = vec!["e".repeat(20000)];
    let body = jt::build_request(&i, "fixture");
    assert_eq!(
        body["state"]["hypothesis_report"].as_str().unwrap().len(),
        4000
    );
    assert_eq!(
        body["state"]["evaluator_catalogue"][0]
            .as_str()
            .unwrap()
            .len(),
        20000
    );
    assert!(body.to_string().len() > 24000);
    println!("report clipped to 4000 chars; evaluator catalogue retains 20000-char entry");
}
#[tokio::test]
async fn disabled_triage_makes_no_calls() {
    let d = tempfile::tempdir().unwrap();
    let transport = Scripted::good();
    let mut c = cfg();
    c.enabled = false;
    assert!(!jt::consider(&c, &transport, d.path(), &input()).await);
    assert!(transport.seen.lock().unwrap().is_empty());
    assert!(!d.path().join(jt::RECORD_FILE).exists());
}
fn executable(path: &Path, script: &str) {
    std::fs::write(path, script).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
struct PathGuard(Option<OsString>);
impl Drop for PathGuard {
    fn drop(&mut self) {
        match &self.0 {
            Some(v) => std::env::set_var("PATH", v),
            None => std::env::remove_var("PATH"),
        }
    }
}
fn fixture_ssh(dir: &Path) -> PathGuard {
    let old = std::env::var_os("PATH");
    let bash = Command::new("bash")
        .args(["-c", "command -v bash"])
        .output()
        .unwrap();
    let bash = String::from_utf8(bash.stdout).unwrap();
    executable(&dir.join("ssh"), &format!("#!{}\n# Ignore remote argv; evaluate only the fixture's stdin script locally.\nexec {} -s\n", bash.trim(), bash.trim()));
    let mut paths = vec![dir.to_path_buf()];
    paths.extend(std::env::split_paths(old.as_ref().unwrap()));
    std::env::set_var("PATH", std::env::join_paths(paths).unwrap());
    PathGuard(old)
}
#[test]
fn health_probe_writes_then_checks_capacity() {
    let d = tempfile::tempdir().unwrap();
    let _guard = fixture_ssh(d.path());
    executable(&d.path().join("df"), "#!/bin/sh\nprintf 'Filesystem blocks used available capacity mount\nfixture 99999999 0 12582912 0%% /\n'\n");
    let annexe = d.path().join("annexe");
    assert_eq!(
        dispatch::annexe_health("fixture-only", annexe.to_str().unwrap(), 10).unwrap(),
        12
    );
    assert_eq!(
        std::fs::read_dir(&annexe).unwrap().count(),
        0,
        "probe file removed"
    );
    assert!(
        dispatch::annexe_health("fixture-only", annexe.to_str().unwrap(), 13)
            .unwrap_err()
            .contains("below")
    );
    executable(&d.path().join("df"), "#!/bin/sh\nexit 1\n");
    assert!(
        dispatch::annexe_health("fixture-only", annexe.to_str().unwrap(), 0)
            .unwrap_err()
            .contains("no free-space")
    );
    println!("actual probe writes/removes file; 12 GiB accepted at floor10, rejected at13; failed df rejected");
}
#[test]
fn health_refuses_unwritable_target_even_when_capacity_would_pass() {
    let d = tempfile::tempdir().unwrap();
    let _guard = fixture_ssh(d.path());
    let annexe = d.path().join("annexe");
    std::fs::write(&annexe, "regular file").unwrap();
    executable(&d.path().join("df"), "#!/bin/sh\nprintf 'Filesystem blocks used available capacity mount\nfixture 99999999 0 12582912 0%% /\n'\n");
    assert!(dispatch::annexe_health("fixture-only", annexe.to_str().unwrap(), 10).is_err());
}
// Extract the exact sweep template from the pinned source, not a restated implementation.
fn sweep_script(root: &Path) -> String {
    let source = include_str!("../src/engine.rs");
    let line = source
        .lines()
        .find(|l| l.contains("find {} -maxdepth 1 -type d -name '20*'"))
        .unwrap();
    let template = line
        .trim()
        .strip_prefix('"')
        .unwrap()
        .strip_suffix("\",")
        .unwrap();
    let quoted = format!("'{}'", root.display().to_string().replace('\'', "'\\''"));
    template.replacen("{}", &quoted, 1).replace("{{}}", "{}")
}
#[test]
fn retention_sweep_removes_old_parent_with_fresh_active_child() {
    let d = tempfile::tempdir().unwrap();
    let night = d.path().join("2026-10-20-fixture");
    std::fs::create_dir(&night).unwrap();
    std::fs::write(night.join("active-output"), "fresh work").unwrap();
    assert!(Command::new("touch")
        .args(["-d", "6 days ago"])
        .arg(&night)
        .status()
        .unwrap()
        .success());
    let out = Command::new("bash")
        .args(["-c", &sweep_script(d.path())])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(!night.exists());
    println!("exact retention template deletes old parent despite fresh child; no live-job check");
}
#[test]
fn retention_sweep_masks_find_failure() {
    let d = tempfile::tempdir().unwrap();
    let _guard = fixture_ssh(d.path());
    executable(
        &d.path().join("find"),
        "#!/bin/sh\necho fixture-denied >&2\nexit 1\n",
    );
    let out = Command::new("bash")
        .args(["-c", &sweep_script(d.path())])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(out.stderr.is_empty());
    println!("failed find becomes exit0 with empty stderr through exact retention template");
}

struct CapturingLlm(Mutex<Vec<String>>);
impl dream_core::seams::Llm for CapturingLlm {
    fn complete<'a>(
        &'a self,
        req: &'a dream_core::seams::LlmRequest,
    ) -> dream_core::seams::BoxFut<'a, Result<String, dream_core::seams::LlmError>> {
        self.0.lock().unwrap().push(req.prompt.clone());
        Box::pin(async { Ok("NO-PATCH: fixture stops here".into()) })
    }
    fn identity(&self) -> dream_gate::manifest::ModelIdentity {
        dream_gate::manifest::ModelIdentity {
            provider: "fixture".into(),
            model: "capture-only".into(),
            max_tokens: 2048,
            fallback: None,
        }
    }
}
async fn observe_repair_excerpt(path: &str, content: &str, marker: &str) {
    use dream_core::{runner::LocalRunner, source};
    use dream_engine::{
        candidate_flow::{self, Flow},
        journal::Journal,
    };
    use dream_gate::{gate::CandidateState, guard::ProtectedPolicy};
    let d = tempfile::tempdir().unwrap();
    let repo = d.path().join("repo");
    std::fs::create_dir(&repo).unwrap();
    let git = |args: &[&str]| {
        let o = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(args)
            .output()
            .unwrap();
        assert!(
            o.status.success(),
            "git failed: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        String::from_utf8(o.stdout).unwrap()
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["config", "user.name", "Audit fixture"]);
    git(&["config", "user.email", "audit@example.invalid"]);
    std::fs::write(repo.join(path), content).unwrap();
    git(&["add", "--", path]);
    git(&["commit", "-qm", "fixture"]);
    let base = git(&["rev-parse", "HEAD"]).trim().to_string();
    let night = d.path().join("night");
    std::fs::create_dir(&night).unwrap();
    let policy = ProtectedPolicy::default();
    let patch = format!("diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-not-present-in-fixture\n+replacement\n");
    assert!(dream_gate::guard::check(&patch, &policy).is_ok());
    assert!(source::is_denied(path) || source::has_secret_content(content));
    let llm = CapturingLlm(Mutex::new(vec![]));
    let flow = Flow {
        repo: &repo,
        branch: "dream/audit",
        commit_msg: "fixture",
        base: &base,
        policy: &policy,
        build: None,
        required: &[],
        runner: &LocalRunner,
        annexe: None,
        night_dir: &night,
        report: "VERDICT: ACCEPT",
        llm: &llm,
        fallback: None,
    };
    let attempt = candidate_flow::attempt(&flow, &patch, &Journal::disabled("fixture")).await;
    assert!(matches!(attempt.state, CandidateState::DidNotApply { .. }));
    let prompts = llm.0.lock().unwrap();
    assert_eq!(prompts.len(), 1);
    assert!(
        prompts[0].contains(marker),
        "repair did not receive fixture marker"
    );
    assert_eq!(
        git(&["worktree", "list", "--porcelain"])
            .lines()
            .filter(|l| l.starts_with("worktree "))
            .count(),
        1
    );
    assert!(git(&["status", "--porcelain"]).is_empty());
    println!("apply-repair received canary from {path}; normal source policy would withhold it; no network");
}
#[tokio::test]
async fn repair_excerpt_bypasses_key_material_withholding() {
    observe_repair_excerpt(
        "ordinary.txt",
        "-----BEGIN PRIVATE KEY-----\nAUDIT-NOT-A-KEY\n-----END PRIVATE KEY-----\n",
        "AUDIT-NOT-A-KEY",
    )
    .await;
}
#[tokio::test]
async fn repair_excerpt_bypasses_filename_denylist() {
    observe_repair_excerpt(
        "credentials.txt",
        "AUDIT-NOT-A-CREDENTIAL\n",
        "AUDIT-NOT-A-CREDENTIAL",
    )
    .await;
}
