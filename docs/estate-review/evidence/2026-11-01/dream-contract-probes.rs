use dream_core::{candidate, runner::LocalRunner};
use dream_gate::{
    config::Fixer,
    gate::{self, CandidateState, GateContext},
    guard::ProtectedPolicy,
    manifest::{self, EvaluatorIdentity, ExperimentManifest, FormattingOutcome, ModelIdentity},
    outcome::ExecOutcome,
    pertest::{TestCase, TestOutcomes, TestStatus},
    receipts::{self, EvaluatorOutcome, EvaluatorReceipt, Phase},
    verdict::Verdict,
};
use std::{fs, path::Path, process::Command};
fn evaluator(name: &str, cmd: &str) -> EvaluatorIdentity {
    EvaluatorIdentity {
        name: name.into(),
        command: cmd.into(),
        command_digest: manifest::digest(cmd.as_bytes()),
        required: true,
        deeps: vec![],
        timeout_secs: 30,
        test_format: None,
        fixer: None,
    }
}
fn manifest(ids: Vec<EvaluatorIdentity>) -> ExperimentManifest {
    ExperimentManifest {
        schema: 1,
        run_id: "audit-run".into(),
        night_id: "audit-night".into(),
        repo: "audit".into(),
        repo_slug: "audit/local".into(),
        date: "2026-11-01".into(),
        day_int: 20261101,
        deep: "core".into(),
        scan: vec![],
        baseline_revision: "a".repeat(40),
        baseline_tree_hash: "b".repeat(40),
        config_digest: manifest::digest(b"cfg"),
        evaluators: ids,
        model: ModelIdentity {
            provider: "local".into(),
            model: "fixture".into(),
            max_tokens: 1,
            fallback: None,
        },
        engine_version: "audit".into(),
        created_at: "2026-11-01T00:00:00Z".into(),
    }
}
fn receipt(name: &str, cmd: &str) -> EvaluatorReceipt {
    EvaluatorReceipt::from_exec(
        name,
        cmd,
        Phase::Candidate,
        true,
        30,
        ExecOutcome {
            exit_code: Some(0),
            stdout: "ok\n".into(),
            stderr: String::new(),
            duration_ms: 1,
            transport_error: None,
            timed_out: false,
            started_at: "2026-11-01T00:00:00Z".into(),
        },
    )
}
fn patch(path: &str, body: &str) -> String {
    format!("diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{} @@\n{}",body.lines().count(),body.lines().map(|l|format!("+{l}\n")).collect::<String>())
}
fn applied() -> CandidateState {
    CandidateState::Applied {
        tree_hash: "not-a-tree".into(),
        patch: patch("value.rs", "pub fn value() {}\n"),
    }
}
#[test]
fn direct_gate_trusts_typed_outcome_over_raw_failure() {
    let m = manifest(vec![evaluator("check", "echo ok")]);
    let mut r = receipt("check", "echo ok");
    r.exit_code = Some(1);
    r.stdout = "FAIL: fixture\n".into();
    assert_eq!(r.outcome, EvaluatorOutcome::Passed);
    let p = ProtectedPolicy::default();
    assert!(
        gate::decide(
            &m,
            &Ok(Verdict::Accept),
            &applied(),
            &[r],
            &GateContext::new(&p)
        )
        .accepted
    );
}
#[test]
fn direct_gate_has_no_required_check_or_tree_identity_floor() {
    let p = ProtectedPolicy::default();
    let d = gate::decide(
        &manifest(vec![]),
        &Ok(Verdict::Accept),
        &applied(),
        &[],
        &GateContext::new(&p),
    );
    assert!(d.accepted);
    assert!(d.required_outcomes.is_empty());
}
#[test]
fn duplicate_test_names_hide_failure_when_pass_is_last() {
    let m = manifest(vec![evaluator("check", "echo ok")]);
    let p = ProtectedPolicy::default();
    let failed = TestCase::new("same", TestStatus::failed("regression"));
    let passed = TestCase::new("same", TestStatus::Passed);
    let make = |candidate| {
        GateContext::new(&p).with_tests(
            "check",
            TestOutcomes {
                baseline: vec![passed.clone()],
                candidate,
            },
        )
    };
    let r = receipt("check", "echo ok");
    assert!(
        gate::decide(
            &m,
            &Ok(Verdict::Accept),
            &applied(),
            &[r.clone()],
            &make(vec![failed.clone(), passed.clone()])
        )
        .accepted
    );
    assert!(
        !gate::decide(
            &m,
            &Ok(Verdict::Accept),
            &applied(),
            &[r],
            &make(vec![passed.clone(), failed])
        )
        .accepted
    );
}
#[test]
fn receipt_slug_collision_discards_first_receipt() {
    let d = tempfile::tempdir().unwrap();
    receipts::persist(
        d.path(),
        Phase::Candidate,
        &[receipt("a b", "first"), receipt("a-b", "second")],
    )
    .unwrap();
    let rows = receipts::load(d.path(), Phase::Candidate);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].name, "a-b");
    assert_eq!(rows[0].command, "second");
}
#[test]
fn manifest_archive_overwrites_same_run_id_history() {
    let d = tempfile::tempdir().unwrap();
    let mut m = manifest(vec![]);
    manifest::freeze(d.path(), &m).unwrap();
    m.model.model = "second".into();
    manifest::freeze(d.path(), &m).unwrap();
    m.model.model = "third".into();
    manifest::freeze(d.path(), &m).unwrap();
    let archived: ExperimentManifest =
        serde_json::from_slice(&fs::read(d.path().join("manifest-audit-run.json")).unwrap())
            .unwrap();
    assert_eq!(archived.model.model, "second");
    assert_eq!(fs::read_dir(d.path()).unwrap().count(), 2);
}
#[test]
fn malformed_manifest_archive_overwrites_prior_damage() {
    let d = tempfile::tempdir().unwrap();
    let m = manifest(vec![]);
    for text in ["first damaged record", "second damaged record"] {
        fs::write(d.path().join("manifest.json"), text).unwrap();
        manifest::freeze(d.path(), &m).unwrap();
    }
    assert_eq!(
        fs::read_to_string(d.path().join("manifest-unparsable.json")).unwrap(),
        "second damaged record"
    );
}
fn git(repo: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}
fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let repo = d.path().join("repo");
    fs::create_dir(&repo).unwrap();
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["config", "user.name", "audit"]);
    git(&repo, &["config", "user.email", "audit@example.invalid"]);
    fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"audit-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(repo.join("value.txt"), "original\n").unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "fixture"]);
    (d, repo)
}
#[test]
fn formatter_attribute_spacing_reaches_outside_worktree() {
    let (d, repo) = fixture();
    let outside = d.path().join("outside.rs");
    let original = "pub fn outside( )->u8{1}\n";
    fs::write(&outside, original).unwrap();
    let body = format!(
        "#[ path = {:?}]\nmod outside;\npub fn here( ){{}}\n",
        outside.to_str().unwrap()
    );
    let p = patch("src/lib.rs", &body);
    let policy = ProtectedPolicy::default();
    let mut id = evaluator("fmt", "echo ok");
    id.fixer = Some(Fixer::Rustfmt);
    let a = candidate::run(
        &candidate::AttemptInput {
            repo: &repo,
            branch: "dream/audit-format",
            patch: &p,
            commit_msg: "audit",
            base: "HEAD",
            policy: &policy,
            build: None,
            required: &[&id],
        },
        &LocalRunner,
        None,
    );
    let c = a.prepared.as_ref().unwrap();
    let outcome = c.formatting.as_ref().unwrap().outcome.clone();
    let after = fs::read_to_string(&outside).unwrap();
    candidate::discard(&repo, c);
    println!("formatter outcome={outcome:?}; outside before={original:?}; after={after:?}");
    assert_ne!(after, original, "{outcome:?}");
    assert!(
        matches!(
            outcome,
            FormattingOutcome::Applied { .. } | FormattingOutcome::Reverted { .. }
        ),
        "{outcome:?}"
    );
    assert_eq!(
        git(&repo, &["worktree", "list", "--porcelain"])
            .matches("worktree ")
            .count(),
        1
    );
}
#[test]
fn evaluator_can_change_working_tree_after_recorded_hash() {
    let (_d, repo) = fixture();
    let p = patch("new.txt", "candidate\n");
    let policy = ProtectedPolicy::default();
    let id = evaluator("check", "printf 'rewritten\n' > value.txt; echo ok");
    let a = candidate::run(
        &candidate::AttemptInput {
            repo: &repo,
            branch: "dream/audit-mutating",
            patch: &p,
            commit_msg: "audit",
            base: "HEAD",
            policy: &policy,
            build: None,
            required: &[&id],
        },
        &LocalRunner,
        None,
    );
    let c = a.prepared.as_ref().unwrap();
    let value = fs::read_to_string(c.worktree.join("value.txt")).unwrap();
    let status = git(&c.worktree, &["status", "--porcelain"]);
    let m = manifest(vec![id]);
    let ruling = gate::decide(
        &m,
        &Ok(Verdict::Accept),
        &a.state,
        &a.receipts,
        &GateContext::new(&policy),
    );
    candidate::discard(&repo, c);
    assert_eq!(value, "rewritten\n");
    assert!(status.contains("value.txt"));
    assert!(ruling.accepted);
    assert_eq!(
        git(&repo, &["worktree", "list", "--porcelain"])
            .matches("worktree ")
            .count(),
        1
    );
}

#[test]
fn readiness_accepts_untracked_evaluator_script() {
    let (_d, repo) = fixture();
    fs::create_dir(repo.join("scripts")).unwrap();
    fs::write(repo.join("scripts/check.sh"), "echo ok\n").unwrap();
    let cfg=dream_gate::config::DreamConfig::from_json(br#"{"repo":"audit/local","slots":[{"deep":"core","scan":[]}],"evaluatorEntrypoints":{"check":"bash scripts/check.sh"}}"#).unwrap();
    assert!(dream_core::readiness::assess(&cfg, "audit", "core", &repo).admitted());
    assert!(!git(&repo, &["ls-tree", "-r", "--name-only", "HEAD"]).contains("scripts/check.sh"));
}
#[test]
fn readiness_accepts_wrapped_constant_output() {
    let (_d, repo) = fixture();
    let cfg=dream_gate::config::DreamConfig::from_json(br#"{"repo":"audit/local","slots":[{"deep":"core","scan":[]}],"evaluatorEntrypoints":{"check":"bash -c 'echo ok'"}}"#).unwrap();
    assert!(dream_core::readiness::assess(&cfg, "audit", "core", &repo).admitted());
}

#[test]
fn junit_generated_suffix_collides_with_existing_test_name() {
    let baseline = dream_core::testfmt::parse_junit(
        r#"<testsuite><testcase name="x"/><testcase name="x~2"/></testsuite>"#,
    )
    .unwrap();
    let candidate = dream_core::testfmt::parse_junit(r#"<testsuite><testcase name="x"/><testcase name="x"><failure message="regression"/></testcase><testcase name="x~2"/></testsuite>"#).unwrap();
    assert_eq!(candidate[1].name, candidate[2].name);
    assert!(matches!(candidate[1].status, TestStatus::Failed { .. }));
    let policy = ProtectedPolicy::default();
    let context = GateContext::new(&policy).with_tests(
        "check",
        TestOutcomes {
            baseline,
            candidate,
        },
    );
    let result = gate::decide(
        &manifest(vec![evaluator("check", "echo ok")]),
        &Ok(Verdict::Accept),
        &applied(),
        &[receipt("check", "echo ok")],
        &context,
    );
    assert!(result.accepted);
}
