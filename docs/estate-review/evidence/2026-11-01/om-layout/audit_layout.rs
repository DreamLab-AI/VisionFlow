//! Audit-only counterexamples for ADR-OM-008; all paths are disposable fixtures.
use ontomark_compile::input::page::{Vault, VaultKind};
use ontomark_compile::profile::Profile;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}
fn note(name: &str) -> String {
    format!("---\ntype: class\ntitle: {name}\npublic: true\n---\n{name} description.\n")
}
fn fixture(layout: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("repo");
    std::fs::create_dir_all(&root).unwrap();
    let profile = temp.path().join("profile.yaml");
    write(
        &profile,
        &format!("id: audit\nversion: '1'\nlayout:\n{layout}"),
    );
    (temp, root, profile)
}
fn cli(root: &Path, profile: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ontomark"))
        .arg("--repo")
        .arg(root)
        .arg("--profile")
        .arg(profile)
        .args(args)
        .output()
        .unwrap()
}
fn success(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn suffix_role_inference_skips_the_knowledge_required_status_rule() {
    let (temp, root, profile) = fixture("  knowledge: nested/drafts\n  working: drafts\n");
    write(&root.join("nested/drafts/Alpha.md"), &note("Alpha"));
    let mut config = std::fs::read_to_string(&profile).unwrap();
    config.push_str("frontmatter:\n  roles:\n    knowledge:\n      required_keys: [status]\n      unknown_keys: allow\n    working:\n      unknown_keys: allow\n");
    write(&profile, &config);
    let validate = cli(&root, &profile, &["--json", "validate"]);
    assert_eq!(validate.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&validate.stdout).contains("MISSING_STATUS"));
    let out = temp.path().join("out");
    let build = cli(
        &root,
        &profile,
        &["--json", "build", "--out", out.to_str().unwrap()],
    );
    success(&build);
    assert!(out.join("publish/working/Alpha.md").is_file());
    assert!(!out.join("publish/pages/Alpha.md").exists());
    println!("AUDIT suffix: validate refuses knowledge missing status; build succeeds and publishes it as working");
}

#[test]
fn root_knowledge_layout_includes_nested_working_class_in_ontology() {
    let (temp, root, profile) = fixture("  knowledge: '.'\n  working: drafts\n");
    write(&root.join("Alpha.md"), &note("Alpha"));
    write(&root.join("drafts/DraftOnly.md"), &note("DraftOnly"));
    let report = success(&cli(
        &root,
        &profile,
        &["--json", "validate", "--vault", "all"],
    ));
    assert_eq!(report["knowledge"]["total_pages"], 2);
    assert_eq!(report["working"]["total_pages"], 1);
    for scope in ["public", "machine"] {
        let out = temp.path().join(scope);
        success(&cli(
            &root,
            &profile,
            &[
                "--json",
                "build",
                "--scope",
                scope,
                "--out",
                out.to_str().unwrap(),
            ],
        ));
        let ttl = std::fs::read_to_string(out.join("data/ontology.ttl")).unwrap();
        assert!(ttl.contains("DraftOnly"));
    }
    println!("AUDIT nested roles: working class is also loaded as knowledge and reaches ontology in both scopes");
}

#[test]
fn equivalent_folder_spellings_are_admitted_as_two_roles() {
    for working in ["./notes", "notes/../notes"] {
        let (_temp, root, profile) =
            fixture(&format!("  knowledge: notes\n  working: {working}\n"));
        write(&root.join("notes/Alpha.md"), &note("Alpha"));
        let p = Profile::from_yaml(&std::fs::read_to_string(&profile).unwrap()).unwrap();
        let k = p
            .layout
            .root_of(&root, VaultKind::Knowledge)
            .unwrap()
            .canonicalize()
            .unwrap();
        let w = p
            .layout
            .root_of(&root, VaultKind::Working)
            .unwrap()
            .canonicalize()
            .unwrap();
        assert_eq!(k, w);
        let report = success(&cli(
            &root,
            &profile,
            &["--json", "validate", "--vault", "all"],
        ));
        assert_eq!(report["knowledge"]["total_pages"], 1);
        assert_eq!(report["working"]["total_pages"], 1);
    }
    println!("AUDIT aliases: lexical inequality admits two roles over the same canonical folder");
}

#[test]
fn explicit_relative_start_misses_an_absolute_ancestor_vocabulary() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("estate");
    let child = root.join("nested");
    write(
        &root.join("ontology/vocabulary.yaml"),
        "[invalid vocabulary\n",
    );
    write(&child.join("Alpha.md"), &note("Alpha"));
    let implicit = Command::new(env!("CARGO_BIN_EXE_ontomark"))
        .current_dir(&child)
        .args(["--json", "validate"])
        .output()
        .unwrap();
    assert_eq!(implicit.status.code(), Some(1));
    let explicit = Command::new(env!("CARGO_BIN_EXE_ontomark"))
        .current_dir(&child)
        .args(["--repo", ".", "--json", "validate"])
        .output()
        .unwrap();
    let report = success(&explicit);
    assert_eq!(report["knowledge"]["total_pages"], 1);
    println!("AUDIT relative start: implicit cwd finds malformed ancestor vocabulary; --repo . falls back to generic");
}

#[test]
fn caller_exclusions_replace_builtins_and_duplicate_journal_loading() {
    let (_temp, root, profile) = fixture("  knowledge: '.'\n  working: null\n  excluded: []\n");
    write(&root.join("Alpha.md"), &note("Alpha"));
    write(
        &root.join("journals/2026_10_17.md"),
        "# Journal\n\nAn observation.\n",
    );
    write(&root.join("templates/Template.md"), "A template.\n");
    let p = Profile::from_yaml(&std::fs::read_to_string(&profile).unwrap()).unwrap();
    let v = Vault::load_excluding(&root, VaultKind::Knowledge, &p.layout.excluded_dirs()).unwrap();
    assert_eq!(v.pages.len(), 3);
    assert_eq!(v.journals.len(), 1);
    assert!(v.pages.iter().any(|p| p.id == "journals/2026_10_17"));
    assert_eq!(
        v.journals[0].path,
        v.pages
            .iter()
            .find(|p| p.id == "journals/2026_10_17")
            .unwrap()
            .path
    );
    println!("AUDIT exclusions: empty override admits templates and puts the same journal in pages and journals");
}

#[test]
fn profile_can_select_a_sibling_outside_the_repository() {
    let (temp, root, profile) = fixture("  knowledge: ../outside\n  working: null\n");
    write(&temp.path().join("outside/Outside.md"), &note("Outside"));
    let report = success(&cli(&root, &profile, &["--json", "validate"]));
    assert_eq!(report["knowledge"]["total_pages"], 1);
    println!("AUDIT boundary: caller-supplied layout reads sibling directory outside declared repository");
}

#[test]
fn arbitrary_required_keys_are_not_a_general_schema_validator() {
    let (_temp, root, profile) = fixture("  knowledge: '.'\n  working: null\n");
    write(&root.join("Alpha.md"), &note("Alpha"));
    let mut config = std::fs::read_to_string(&profile).unwrap();
    config.push_str("frontmatter:\n  roles:\n    knowledge:\n      required_keys: [owner]\n      unknown_keys: allow\n");
    write(&profile, &config);
    let report = success(&cli(&root, &profile, &["--json", "validate"]));
    assert_eq!(report["knowledge"]["errors"], 0);
    println!("AUDIT grammar: required_keys owner is accepted in profile but missing owner is not rejected");
}

#[test]
#[cfg(unix)]
fn symlink_role_aliases_are_admitted_without_following_child_symlinks() {
    let (_temp, root, profile) = fixture("  knowledge: notes\n  working: drafts\n");
    write(&root.join("notes/Alpha.md"), &note("Alpha"));
    std::os::unix::fs::symlink(root.join("notes"), root.join("drafts")).unwrap();
    let report = success(&cli(
        &root,
        &profile,
        &["--json", "validate", "--vault", "all"],
    ));
    assert_eq!(report["knowledge"]["total_pages"], 1);
    assert_eq!(report["working"]["total_pages"], 1);
    println!("AUDIT root symlink: two role roots resolve to same folder and both are loaded");
}

#[test]
fn migration_plan_silently_skips_an_undeclared_working_role() {
    let (_temp, root, profile) = fixture("  knowledge: '.'\n  working: null\n");
    write(&root.join("Alpha.md"), &note("Alpha"));
    let output = cli(
        &root,
        &profile,
        &[
            "--json",
            "migrate",
            "plan",
            "m0001-obsidian-bodies",
            "--vault",
            "working",
        ],
    );
    let report = success(&output);
    assert_eq!(report["vaults"], serde_json::json!(["working"]));
    assert_eq!(report["pages"], serde_json::json!([]));
    println!("AUDIT missing role: migrate plan --vault working returns success and empty plan on flat vault: {report}");
}
