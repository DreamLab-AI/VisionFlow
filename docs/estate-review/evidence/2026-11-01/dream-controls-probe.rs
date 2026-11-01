//! Isolated observations, not desired-behaviour regression tests.
use dream_engine::{config, dispatch, estate, llm::Provider};
use dream_core::roster;
use std::{fs, os::unix::fs::{symlink, PermissionsExt}, process::Command};

#[test]
fn audit_controls_configuration_transport_discovery_and_roster_boundaries() {
    // This single-test process never calls a model or an estate service.
    for key in ["DREAM_LLM_PROVIDER", "DREAM_LLM_FALLBACK", "ZAI_API_KEY",
                "ZAI_ANTHROPIC_API_KEY", "ZAI_URL", "LOOM_URL"] {
        std::env::remove_var(key);
    }
    let mut rt: config::RuntimeConfig = toml::from_str("llm_provider='loom'").unwrap();
    let primary = estate::llm_config(&rt);
    assert_eq!(primary.provider, Provider::Loom);
    assert!(estate::fallback_llm_config(&rt, &primary).is_none());
    std::env::set_var("ZAI_ANTHROPIC_API_KEY", "fixture-only-not-a-credential");
    assert_eq!(estate::fallback_llm_config(&rt, &primary).unwrap().provider, Provider::Zai);
    std::env::set_var("DREAM_LLM_FALLBACK", "off");
    assert!(estate::fallback_llm_config(&rt, &primary).is_none());
    assert_eq!(Provider::parse("looom"), Provider::Zai);
    println!("provider: Loom + absent key => no fallback; synthetic key => Zai; off disables; typo => Zai");

    std::env::set_var("AUDIT_NESTED_HOST", "${AUDIT_SECOND_HOST}");
    rt.hp_host = "${AUDIT_NESTED_HOST}".into();
    rt.loom_url = "http://${AUDIT_SECOND_HOST}/v1".into();
    rt.hp_annexe_dir = "/srv/${AUDIT_NESTED_HOST}".into();
    rt.resolve_placeholders();
    assert_eq!(rt.hp_host, "${AUDIT_SECOND_HOST}");
    assert_eq!(rt.loom_url, "http://${AUDIT_SECOND_HOST}/v1");
    assert_eq!(rt.hp_annexe_dir, "/srv/${AUDIT_SECOND_HOST}");
    let invalid: config::RuntimeConfig = toml::from_str("window_start=255\nwindow_end=0\nmax_repos_per_night=0\nprune_dry_streak=0").unwrap();
    assert_eq!(invalid.window_start, 255);
    assert_eq!(invalid.max_repos_per_night, 0);
    println!("config: nested and embedded placeholders survive; unusable window and zero caps deserialize");

    let tmp = tempfile::tempdir().unwrap();
    let bin = tmp.path().join("bin");
    fs::create_dir(&bin).unwrap();
    let log = tmp.path().join("scp-args");
    let shim = bin.join("scp");
    fs::write(&shim, "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$AUDIT_SCP_LOG\"\nexit 0\n").unwrap();
    fs::set_permissions(&shim, fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("AUDIT_SCP_LOG", &log);
    std::env::set_var("PATH", format!("{}:{}", bin.display(), std::env::var("PATH").unwrap()));
    let local = tmp.path().join("local");
    assert!(matches!(dispatch::ssh("", "true"), Err(dispatch::DispatchError::NoAnnexeHost)));
    assert!(matches!(dispatch::scp_to(&local, "", "/remote"), Err(dispatch::DispatchError::NoAnnexeHost)));
    assert!(!log.exists());
    dispatch::scp_from("", "/remote", &local).unwrap();
    assert!(fs::read_to_string(&log).unwrap().lines().any(|l| l == ":/remote"));
    println!("transport: empty-host ssh/scp_to blocked; scp_from invokes fake scp with :/remote");

    let ws = tmp.path().join("ws");
    let outside = tmp.path().join("outside");
    fs::create_dir(&ws).unwrap();
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("dream.config.json"), r#"{"repo":"o/r","slots":[{"deep":"x","scan":[]}]}"#).unwrap();
    symlink(&outside, ws.join("linked")).unwrap();
    assert_eq!(config::discover_repos(&ws).len(), 1);
    let js = std::env::var("AUDIT_LEDGER_JS").unwrap();
    let result = Command::new("node").args(["-e", "const m=require(process.argv[1]);console.log(JSON.stringify(m.discoverNominatedRepos(process.argv[2])))", &js, ws.to_str().unwrap()]).output().unwrap();
    assert!(result.status.success(), "node: {}", String::from_utf8_lossy(&result.stderr));
    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), "[]");
    println!("discovery: Rust includes symlinked nomination; JS status discovery excludes it");

    let path = tmp.path().join("roster.json");
    fs::write(&path, r#"{"schema":999,"repos":{}}"#).unwrap();
    assert_eq!(roster::load(&path).schema, 999);
    let mut a = roster::Roster::default();
    a.record("alpha", "2026-11-01", "ACCEPT");
    roster::save(&path, &a).unwrap();
    let eligible = vec!["alpha".into(), "beta".into()];
    assert_eq!(roster::load(&path).select(&eligible, 1), vec!["beta"]);
    fs::write(&path, "broken").unwrap();
    assert_eq!(roster::load(&path).select(&eligible, 1), vec!["alpha"]);
    println!("roster: unknown schema accepted; corrupt state resets rotation to alphabetical order");
}
