use dream_core::{
    automerge::{self, AutomergeDecision, Refusal},
    persist,
    runner::{EvaluatorRunner, LocalRunner},
    seams::{DraftPr, Forge, GhForge, JsonlSink, MemoryRecord, MemorySink},
};
use dream_gate::{
    config::DreamConfig,
    receipts::{self, EvaluatorOutcome},
    verdict::Verdict,
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().into()
}
struct Fixture {
    root: tempfile::TempDir,
    repo: PathBuf,
    state: PathBuf,
    old_path: Option<OsString>,
    old_state: Option<OsString>,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("repo");
        let remote = root.path().join("remote.git");
        let state = root.path().join("state");
        std::fs::create_dir(&repo).unwrap();
        std::fs::create_dir(&remote).unwrap();
        std::fs::create_dir(&state).unwrap();
        git(&remote, &["init", "--bare", "-q"]);
        git(&repo, &["init", "-q"]);
        git(&repo, &["config", "user.name", "Audit"]);
        git(&repo, &["config", "user.email", "audit@example.invalid"]);
        let hooks = root.path().join("empty-hooks");
        std::fs::create_dir(&hooks).unwrap();
        git(
            &repo,
            &["config", "core.hooksPath", hooks.to_str().unwrap()],
        );
        std::fs::write(repo.join("value.txt"), "fixture\n").unwrap();
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-qm", "fixture"]);
        git(&repo, &["branch", "dream/test"]);
        git(
            &repo,
            &["remote", "add", "origin", remote.to_str().unwrap()],
        );
        std::fs::write(state.join("fixture-marker"), "offline-dream-audit").unwrap();
        std::fs::write(state.join("mode"), mode).unwrap();
        let bin = PathBuf::from(
            std::env::var_os("DREAM_AUDIT_GH_BIN").expect("fixture gh directory required"),
        );
        assert!(bin.join("gh").is_file());
        let old_path = std::env::var_os("PATH");
        let old_state = std::env::var_os("DREAM_AUDIT_GH_STATE");
        let mut paths = vec![bin];
        if let Some(p) = &old_path {
            paths.extend(std::env::split_paths(p));
        }
        std::env::set_var("PATH", std::env::join_paths(paths).unwrap());
        std::env::set_var("DREAM_AUDIT_GH_STATE", &state);
        Self {
            root,
            repo,
            state,
            old_path,
            old_state,
        }
    }
    fn pr(&self) -> DraftPr {
        DraftPr {
            repo_dir: self.repo.clone(),
            repo_slug: "audit/repo".into(),
            branch: "dream/test".into(),
            title: "fixture".into(),
            body: "offline audit".into(),
        }
    }
    fn publish(&self, on: bool, verdict: Verdict) -> persist::Published {
        persist::publish(&GhForge, &cfg(on), &self.pr(), "main", verdict)
    }
    fn log(&self) -> String {
        std::fs::read_to_string(self.state.join("commands.log")).unwrap_or_default()
    }
    fn draft(&self) -> String {
        std::fs::read_to_string(self.state.join("draft")).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        for (key, value) in [
            ("PATH", &self.old_path),
            ("DREAM_AUDIT_GH_STATE", &self.old_state),
        ] {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
fn cfg(on: bool) -> DreamConfig {
    serde_json::from_value(serde_json::json!({"repo":"audit/repo","slots":[{"deep":"audit","scan":["src"]}],"autoMerge":on})).unwrap()
}

#[test]
fn merge_failure_leaves_ready_pr_and_no_undo() {
    let f = Fixture::new("merge-error");
    let out = f.publish(true, Verdict::Accept);
    assert_eq!(out.automerge, AutomergeDecision::Allowed);
    assert!(!out.automerge_enabled);
    assert!(out.automerge_error.unwrap().contains("fixture merge error"));
    assert_eq!(f.draft(), "false");
    assert!(!f.log().contains("--undo"));
    eprintln!("ready-success then merge-failure: fixture draft=false, no compensating command");
}
#[test]
fn ready_failure_stops_before_merge() {
    let f = Fixture::new("ready-error");
    let out = f.publish(true, Verdict::Accept);
    assert!(out.automerge_error.is_some());
    assert_eq!(f.draft(), "true");
    assert!(!f.log().contains("\"merge\""));
}
#[test]
fn success_does_not_pin_head_commit_or_verify_remote_outcome() {
    let f = Fixture::new("ok");
    let out = f.publish(true, Verdict::Accept);
    assert!(out.automerge_enabled);
    let log = f.log();
    assert!(!log.contains("--match-head-commit"));
    assert!(!log.contains("headRefOid"));
    assert_eq!(log.lines().count(), 5);
    assert_eq!(
        git(
            &f.root.path().join("remote.git"),
            &["rev-parse", "dream/test"]
        ),
        git(&f.repo, &["rev-parse", "dream/test"])
    );
    eprintln!("actual command sequence:\n{log}");
}
#[test]
fn empty_success_stdout_is_treated_as_a_pr_url() {
    let f = Fixture::new("empty-url");
    let out = f.publish(true, Verdict::Accept);
    assert_eq!(out.outcome.pr_url.as_deref(), Some(""));
    assert!(out.automerge_enabled);
    assert!(f.log().contains("\"view\", \"\""));
}
#[test]
fn disabled_automerge_only_creates_a_draft() {
    let f = Fixture::new("ok");
    let out = f.publish(false, Verdict::Accept);
    assert_eq!(out.automerge, AutomergeDecision::Disabled);
    assert_eq!(f.draft(), "true");
    assert_eq!(f.log().lines().count(), 1);
}
#[test]
fn wrong_base_missing_checks_and_api_errors_refuse_merge() {
    for mode in ["wrong-base", "no-checks", "api-error"] {
        let f = Fixture::new(mode);
        let out = f.publish(true, Verdict::Accept);
        assert!(matches!(out.automerge, AutomergeDecision::Refused(_)));
        assert_eq!(f.draft(), "true");
        assert!(!f.log().contains("\"ready\""));
    }
}
#[test]
fn create_failure_preserves_successful_local_push_but_discards_stderr() {
    let f = Fixture::new("create-error");
    let out = GhForge.publish_draft(&f.pr());
    assert!(out.pushed);
    assert!(out.pr_url.is_none());
    assert_eq!(
        git(
            &f.root.path().join("remote.git"),
            &["rev-parse", "dream/test"]
        ),
        git(&f.repo, &["rev-parse", "dream/test"])
    );
}
#[test]
fn failed_push_does_not_invoke_gh() {
    let f = Fixture::new("ok");
    git(&f.repo, &["remote", "remove", "origin"]);
    let out = GhForge.publish_draft(&f.pr());
    assert!(!out.pushed && out.pr_url.is_none());
    assert!(f.log().is_empty());
}
#[test]
fn rejected_verdict_does_not_prevent_draft_publication() {
    let f = Fixture::new("ok");
    let out = f.publish(true, Verdict::Reject);
    assert_eq!(
        out.automerge,
        AutomergeDecision::Refused(Refusal::NotAccepted)
    );
    assert!(out.outcome.pushed && out.outcome.pr_url.is_some());
    assert_eq!(f.draft(), "true");
}
#[test]
fn relocated_gate_crate_is_not_detected() {
    let f = Fixture::new("ok");
    let dir = f.repo.join("components/dream-gate");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dream-gate\"\nversion = \"0.0.0\"\n",
    )
    .unwrap();
    std::fs::create_dir(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/lib.rs"), "pub fn gate() {}\n").unwrap();
    git(&f.repo, &["add", "."]);
    git(&f.repo, &["commit", "-qm", "relocated fixture crate"]);
    git(&f.repo, &["branch", "-f", "dream/test", "HEAD"]);
    assert!(!automerge::contains_gate_crates(&f.repo));
    let out = f.publish(true, Verdict::Accept);
    assert_eq!(out.automerge, AutomergeDecision::Allowed);
}
#[test]
fn self_referential_slug_refuses_enable_but_draft_is_already_published() {
    let f = Fixture::new("ok");
    let mut pr = f.pr();
    pr.repo_slug = "DreamLab-AI/dream-engine".into();
    let out = persist::publish(&GhForge, &cfg(true), &pr, "main", Verdict::Accept);
    assert_eq!(
        out.automerge,
        AutomergeDecision::Refused(Refusal::SelfReferential)
    );
    assert!(out.outcome.pushed);
    assert_eq!(f.draft(), "true");
}
#[test]
fn local_timeout_flag_is_false_but_receipt_classifier_recovers_timeout() {
    let d = tempfile::tempdir().unwrap();
    let out = LocalRunner.run(d.path().to_str().unwrap(), "sleep 2", 1);
    assert_eq!(out.exit_code, Some(124));
    assert!(!out.timed_out);
    assert!(matches!(
        receipts::classify(&out, 1),
        EvaluatorOutcome::TimedOut { .. }
    ));
}
#[test]
fn sink_io_error_is_returned_without_journalling_it() {
    let d = tempfile::tempdir().unwrap();
    let sink = JsonlSink::new(d.path());
    assert!(futures::executor::block_on(sink.record(&MemoryRecord::default())).is_err());
    assert!(!d.path().join("run-state.json").exists());
}
