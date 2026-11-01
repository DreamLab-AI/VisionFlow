// Append to the pinned, unmodified tests/night_e2e.rs in an isolated export.
#[tokio::test]
async fn audit_controls_target_bypasses_pause_and_standby() {
    let rig = Rig::new("audit-paused", vec![], 0);
    let engine = rig.engine(Recording::passthrough());
    let pause = PathBuf::from(std::env::var("DREAM_PAUSE_FLAG").unwrap());
    std::fs::write(&pause, "paused").unwrap();
    std::fs::write(rig.repo.join(".dream-standby"), "standby").unwrap();
    assert!(engine.run_night(0, DATE).await.is_none());
    let result = engine.run_cycle(Some(&rig.name), 0, DATE, true).await;
    std::fs::remove_file(&pause).unwrap();
    assert!(result.is_ok(), "target dry-run bypasses both controls: {result:?}");
    assert_eq!(rig.llm.calls(), 0);
    println!("paused roster returns None; targeted dry-run still produces a prompt despite standby");
}

#[tokio::test]
async fn audit_controls_finished_night_can_gain_handoff_after_config_change() {
    let rig = Rig::new("audit-readmission", vec![accept_report(GOOD_PATCH)], 0);
    let first = rig.engine(Recording::passthrough());
    assert_eq!(Rig::only_outcome(first.run_night(0, DATE).await), "ACCEPT");
    let before = rig.ledger_text();
    let report_before = std::fs::read_to_string(rig.night_dir().join("report.md")).unwrap();
    let state_before = std::fs::read(rig.night_dir().join("run-state.json")).unwrap();
    let path = rig.repo.join("dream.config.json");
    let mut cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    cfg["evaluatorEntrypoints"] = serde_json::json!({});
    std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
    let again = rig.engine(Recording::passthrough());
    assert_eq!(Rig::only_outcome(again.run_night(0, DATE).await), "HANDOFF");
    assert!(rig.ledger_text().len() > before.len());
    assert!(rig.ledger_text().contains("HANDOFF"));
    let report_after = std::fs::read_to_string(rig.night_dir().join("report.md")).unwrap();
    assert_ne!(report_before, report_after);
    assert!(report_after.contains("VERDICT: HANDOFF"));
    assert_eq!(state_before, std::fs::read(rig.night_dir().join("run-state.json")).unwrap());
    assert_eq!(rig.llm.calls(), 1);
    assert_eq!(rig.forge.asked.lock().unwrap().len(), 1);
    println!("same finished night: changed readiness overwrites report and appends HANDOFF without a second model or forge call");
}
