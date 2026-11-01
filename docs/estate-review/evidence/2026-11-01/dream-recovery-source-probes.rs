use dream_core::seams::{BoxFut, Llm, LlmError, LlmRequest};
use dream_core::{
    runstate::{self, NightLock, Phase, Resume},
    source::{self, Denylist, FetchedFile, GatherInput, SourceConfig, SourceRequest},
};
use dream_gate::manifest::ModelIdentity;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

struct Planner(&'static str);
impl Llm for Planner {
    fn complete<'a>(&'a self, _: &'a LlmRequest) -> BoxFut<'a, Result<String, LlmError>> {
        Box::pin(async move { Ok(self.0.into()) })
    }
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            provider: "audit-local".into(),
            model: "fixed-reply".into(),
            max_tokens: 2048,
            fallback: None,
        }
    }
}
fn git(dir: &Path, args: &[&str]) {
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
}
fn repo() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    git(d.path(), &["init", "-q"]);
    std::fs::write(d.path().join("a.rs"), "pub fn a() {}\n").unwrap();
    std::fs::write(d.path().join("b.rs"), "pub fn b() {}\n").unwrap();
    std::fs::write(d.path().join("binary.dat"), [0, 1, 2]).unwrap();
    git(d.path(), &["add", "."]);
    git(
        d.path(),
        &[
            "-c",
            "user.name=Audit",
            "-c",
            "user.email=audit@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    );
    d
}
fn gather(
    repo: &Path,
    requests: &[SourceRequest],
    max_files: usize,
    reply: &'static str,
) -> Option<(String, source::SourceRecord)> {
    let cfg = SourceConfig {
        max_files,
        ..SourceConfig::default()
    };
    futures::executor::block_on(source::gather(
        &Planner(reply),
        None,
        &GatherInput::new(
            repo,
            "HEAD",
            "audit",
            "",
            &[],
            "",
            &cfg,
            &Denylist::default(),
        )
        .with_requested(requests),
        &str::to_string,
    ))
}
fn requests(tokens: &[&str]) -> Vec<SourceRequest> {
    tokens
        .iter()
        .map(|s| SourceRequest::parse(s).unwrap())
        .collect()
}

#[test]
fn beyond_eof_request_is_honoured_with_zero_bytes() {
    let d = repo();
    let (text, rec) = gather(d.path(), &requests(&["a.rs:99-100"]), 4, r#"{"files":[]}"#).unwrap();
    assert_eq!(rec.files[0].bytes, 0);
    assert_eq!(rec.requested_honoured, ["a.rs:99-100"]);
    assert!(text.contains("lines 99-1 of 1"));
    eprintln!("EOF request honoured with {} bytes", rec.files[0].bytes);
}
#[test]
fn duplicate_ranges_are_not_all_accounted_for() {
    let d = repo();
    let (_, rec) = gather(
        d.path(),
        &requests(&["a.rs:1", "a.rs:99"]),
        4,
        r#"{"files":[]}"#,
    )
    .unwrap();
    assert_eq!(rec.requested_honoured, ["a.rs:1-1"]);
    assert!(rec.requested_refused.is_empty());
}
#[test]
fn invalid_request_consumes_reserved_slot() {
    let d = repo();
    let (_, rec) = gather(
        d.path(),
        &requests(&["missing.rs", "a.rs"]),
        2,
        r#"{"files":["b.rs"]}"#,
    )
    .unwrap();
    assert!(rec
        .requested_refused
        .contains(&("a.rs".into(), "over-cap".into())));
    assert!(rec.requested_honoured.is_empty());
    assert_eq!(rec.files[0].path, "b.rs");
}
#[test]
fn all_refused_requests_have_no_returned_receipt() {
    let d = repo();
    assert!(gather(d.path(), &requests(&["missing.rs"]), 4, r#"{"files":[]}"#).is_none());
    assert!(gather(d.path(), &requests(&["binary.dat"]), 4, r#"{"files":[]}"#).is_none());
}
#[test]
fn clipping_marker_exceeds_total_and_requested_body_budget() {
    let cfg = SourceConfig {
        budget_bytes: 100,
        file_cap_bytes: 100,
        ..SourceConfig::default()
    };
    let files = [
        FetchedFile {
            path: "a.rs".into(),
            lines: None,
            text: Some("x".repeat(200)),
            requested: true,
        },
        FetchedFile {
            path: "b.rs".into(),
            lines: None,
            text: Some("ok".into()),
            requested: false,
        },
    ];
    let (_, records) = source::render_section("audit", &files, &cfg, &str::to_string);
    assert!(records[0].bytes > cfg.budget_bytes);
    assert_eq!(records[1].status, "over-budget");
    eprintln!(
        "requested body {} bytes, half-budget {}, total {}",
        records[0].bytes,
        cfg.budget_bytes / 2,
        cfg.budget_bytes
    );
}
#[test]
fn normal_budget_is_also_exceeded_by_clipping_markers() {
    let cfg = SourceConfig::default();
    let files: Vec<_> = (0..12)
        .map(|i| FetchedFile {
            path: format!("f{i}.rs"),
            lines: None,
            text: Some("x".repeat(20_000)),
            requested: i < 6,
        })
        .collect();
    let (_, records) = source::render_section("audit", &files, &cfg, &str::to_string);
    let requested: usize = records
        .iter()
        .filter(|r| r.requested)
        .map(|r| r.bytes)
        .sum();
    let total: usize = records.iter().map(|r| r.bytes).sum();
    assert!(requested > cfg.budget_bytes / 2 && total > cfg.budget_bytes);
    eprintln!("default requested body {requested}/30000; total body {total}/60000");
}
#[test]
fn tracked_non_ascii_name_is_missed_by_default_git_quoting() {
    let d = repo();
    std::fs::write(d.path().join("café.rs"), "pub fn cafe() {}\n").unwrap();
    git(d.path(), &["add", "."]);
    git(
        d.path(),
        &[
            "-c",
            "user.name=Audit",
            "-c",
            "user.email=audit@example.invalid",
            "commit",
            "-qm",
            "unicode",
        ],
    );
    git(d.path(), &["config", "core.quotepath", "true"]);
    let files = source::tracked_files(d.path(), "HEAD").unwrap();
    assert!(!files.iter().any(|f| f.path == "café.rs"));
    assert!(source::read_blob(d.path(), "HEAD", "café.rs").is_some());
    assert!(gather(d.path(), &requests(&["café.rs"]), 4, r#"{"files":[]}"#).is_none());
}
#[test]
fn finished_probe_ignores_wrong_night_but_begin_rejects_it() {
    let d = tempfile::tempdir().unwrap();
    runstate::begin(d.path(), "old", "night", "repo", "2026-11-01", 2).unwrap();
    assert!(
        runstate::finished_night(d.path(), "other", "2026-11-01", "night")
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        runstate::begin(d.path(), "new", "night", "other", "2026-11-01", 2),
        Err(runstate::JournalError::WrongNight { .. })
    ));
}
#[test]
fn zero_attempt_budget_still_starts_a_first_run() {
    let d = tempfile::tempdir().unwrap();
    let result = runstate::begin(d.path(), "run", "night", "repo", "2026-11-01", 0).unwrap();
    assert!(result.should_run());
    assert_eq!(result.state().attempts, 1);
}

// Re-executes only this test, signals readiness after the real journal write,
// then the parent kills and reaps the child. No host daemon is involved.
#[test]
fn process_death_releases_lock_and_preserves_restart_rules() {
    const CHILD: &str = "DREAM_AUDIT_RECOVERY_CHILD";
    if let Ok(dir) = std::env::var(CHILD) {
        let dir = Path::new(&dir);
        let _lock = NightLock::acquire(dir).unwrap();
        let mut s = runstate::begin(dir, "original", "night", "repo", "2026-11-01", 2)
            .unwrap()
            .state()
            .clone();
        if std::env::var("DREAM_AUDIT_PERSIST").unwrap() == "yes" {
            runstate::persist(dir, &mut s, "ACCEPT").unwrap();
        } else {
            runstate::advance(dir, &mut s, Phase::ModelCalled).unwrap();
        }
        println!("AUDIT_READY");
        std::io::stdout().flush().unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    for persist in [false, true] {
        let d = tempfile::tempdir().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process_death_releases_lock_and_preserves_restart_rules",
                "--nocapture",
            ])
            .env(CHILD, d.path())
            .env("DREAM_AUDIT_PERSIST", if persist { "yes" } else { "no" })
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = false;
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            if line.unwrap().contains("AUDIT_READY") {
                ready = true;
                break;
            }
        }
        if !ready {
            let _ = child.kill();
            let _ = child.wait();
            panic!("child failed before readiness");
        }
        assert!(matches!(
            NightLock::acquire(d.path()),
            Err(runstate::LockError::Held { .. })
        ));
        child.kill().unwrap();
        let status = child.wait().unwrap();
        assert!(!status.success());
        let _lock = NightLock::acquire_within(d.path(), std::time::Duration::from_secs(2)).unwrap();
        let recovered = runstate::begin(
            d.path(),
            if persist { "moved-head" } else { "original" },
            "night",
            "repo",
            "2026-11-01",
            2,
        )
        .unwrap();
        if persist {
            assert!(matches!(recovered, Resume::AlreadyComplete(_)));
            assert_eq!(recovered.state().verdict.as_deref(), Some("ACCEPT"));
        } else {
            assert!(matches!(recovered, Resume::Resumed(_)));
            assert_eq!(recovered.state().attempts, 2);
            assert_eq!(recovered.state().resumed_from, Some(Phase::ModelCalled));
        }
        eprintln!(
            "process killed after persist={persist}: {:?}",
            recovered.state().phase
        );
    }
}
