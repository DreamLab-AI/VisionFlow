// Append to pinned agentbox tests/night_e2e.rs in a disposable export.
// Assertions record observed boundaries; they are not desired-behaviour tests.
#[test]
fn audit_receipts_runner_status_boundaries() {
    setup_env();
    let dir = tempfile::tempdir().unwrap();
    let wd = dir.path().to_str().unwrap();
    let local = LocalRunner;
    let remote = dream_engine::runner::SshRunner { host: "receipt-probe.invalid".into() };
    let pipeline = "sh -c 'exit 7' | sh -c 'cat >/dev/null; echo downstream; exit 9'";
    for runner in [&local as &dyn EvaluatorRunner, &remote] {
        let producer = runner.run(wd, "sh -c 'echo producer; exit 7' | tail -1", 5);
        assert_eq!(producer.exit_code, Some(7));
        let out = runner.run(wd, pipeline, 5);
        assert_eq!(out.exit_code, Some(9));
        let masked = runner.run(wd, "false; echo completed", 5);
        assert!(receipts::classify(&masked, 5).is_pass());
        let nested = runner.run(wd, r#"bash -c 'false | echo nested-completed'"#, 5);
        assert!(receipts::classify(&nested, 5).is_pass());
        let reserved = runner.run(wd, "echo immediate; exit 124", 5);
        assert!(!reserved.timed_out);
        assert!(matches!(receipts::classify(&reserved, 5), receipts::EvaluatorOutcome::TimedOut { after_secs: 5 }));
    }
    let missing = dir.path().join("does-not-exist");
    assert!(matches!(receipts::classify(&local.run(missing.to_str().unwrap(), "echo ok", 5), 5), receipts::EvaluatorOutcome::Blocked { .. }));
    assert!(matches!(receipts::classify(&remote.run(missing.to_str().unwrap(), "echo ok", 5), 5), receipts::EvaluatorOutcome::Failed { exit_code: 1 }));
    assert!(matches!(receipts::classify(&local.run(wd, "exit 255", 5), 5), receipts::EvaluatorOutcome::Failed { exit_code: 255 }));
    assert!(matches!(receipts::classify(&remote.run(wd, "exit 255", 5), 5), receipts::EvaluatorOutcome::Blocked { .. }));
    println!("both runners: pipeline 7|9 => 9; false then echo and nested pipeline => PASSED; immediate 124 => TIMED-OUT; missing directory/255 differ across runners");
}

#[test]
fn audit_receipts_outside_sibling_can_overlay_the_target() {
    let rig = Rig::new("audit-overlay", vec![], 0);
    let outside = rig._tmp.path().join("outside").join(&rig.name);
    std::fs::create_dir_all(outside.join("src")).unwrap();
    std::fs::write(outside.join("src/greeting.txt"), "sibling-content\n").unwrap();
    git(&outside, &["init", "-q", "-b", "main"]);
    git(&outside, &["config", "user.name", "audit"]);
    git(&outside, &["config", "user.email", "audit@example.invalid"]);
    git(&outside, &["config", "commit.gpgsign", "false"]);
    git(&outside, &["add", "-A"]);
    git(&outside, &["commit", "-qm", "sibling"]);
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(rig.repo.join("dream.config.json")).unwrap()).unwrap();
    let mut cfg = cfg;
    cfg["annexeInclude"] = serde_json::json!([outside.to_str().unwrap()]);
    let parsed = dream_gate::config::DreamConfig::from_json(&serde_json::to_vec(&cfg).unwrap()).unwrap();
    let target_sub = dream_engine::dispatch::annexe_subpath(&rig.repo, &rig.workspace);
    let sibling_sub = dream_engine::dispatch::annexe_subpath(&outside, &rig.workspace);
    assert_eq!(target_sub, sibling_sub);
    dream_engine::dispatch::clone_repo_and_siblings(&rig.host, &rig.repo, rig.annexe.to_str().unwrap(), &target_sub, &parsed.sibling_repos, &rig.workspace).unwrap();
    let shipped = rig.annexe.join(&target_sub);
    assert_eq!(std::fs::read_to_string(shipped.join("src/greeting.txt")).unwrap(), "sibling-content\n");
    assert!(shipped.join("scripts/check.sh").exists(), "target files remain alongside the sibling overlay");
    println!("validated absolute sibling and target share leaf; annexe contains sibling source plus target evaluator script");
}

#[test]
fn audit_receipts_dispatch_reads_current_head_after_baseline_capture() {
    let rig = Rig::new("audit-head", vec![], 0);
    let frozen = dream_engine::baseline::of(&rig.repo).unwrap();
    std::fs::write(rig.repo.join("src/greeting.txt"), "later-commit\n").unwrap();
    git(&rig.repo, &["add", "src/greeting.txt"]);
    git(&rig.repo, &["commit", "-qm", "advance HEAD after capture"]);
    let later = dream_engine::baseline::of(&rig.repo).unwrap();
    assert_ne!(frozen, later);
    dream_engine::dispatch::clone_to_hp(&rig.repo, &rig.host, rig.annexe.to_str().unwrap(), &rig.name).unwrap();
    assert_eq!(std::fs::read_to_string(rig.annexe.join(&rig.name).join("src/greeting.txt")).unwrap(), "later-commit\n");
    println!("baseline captured commit A; dispatch after commit B ships B because archive resolves HEAD again");
}

#[tokio::test]
async fn audit_receipts_no_patch_accept_still_adds_an_operator_alert() {
    let rig = Rig::new("audit-no-patch", vec!["Finding: no measured change.\nVERDICT: ACCEPT\n".into()], 0);
    let engine = rig.engine(Recording::passthrough());
    assert_eq!(Rig::only_outcome(engine.run_night(0, DATE).await), "INCONCLUSIVE");
    let gate = rig.gate_json();
    assert_eq!(gate["vetoes"].as_array().unwrap().len(), 1);
    assert_eq!(gate["vetoes"][0]["class"], "unproven");
    assert!(dream_engine::inbox::load_items().iter().any(|i| i.repo == rig.name && i.kind == "alert" && i.text.contains("INCONCLUSIVE")));
    assert_eq!(rig.forge.asked.lock().unwrap().len(), 0);
    println!("no-patch ACCEPT => INCONCLUSIVE without Harness veto, but operator inbox receives alert");
}

#[test]
fn audit_receipts_finding_selection_is_syntax_not_step_provenance() {
    use dream_gate::verdict::{sanitise_finding, finding_violations, Verdict};
    let report = format!("> | {DATE} | unrelated-deep | Quoted example chosen first | NONE | NONE | no | REJECT | | abc | |\n\nFinding: actual finding follows.\n");
    assert_eq!(sanitise_finding(&report, Verdict::Accept, DATE), "Quoted example chosen first");
    let report = format!("Given {}", "😀".repeat(90));
    let finding = sanitise_finding(&report, Verdict::Accept, DATE);
    let violations = finding_violations(&finding);
    assert!(violations.contains(&"finding-too-long"));
    assert!(violations.contains(&"finding-hypothesis-leak"));
    println!("first valid same-date quoted table cell wins regardless of deep/verdict; fallback can violate its own UTF-16/hypothesis contract");
}
