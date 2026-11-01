//! `ontomark build --scope machine` (ADR-OM-005 decision 4): the public
//! knowledge projection plus `working/`'s public pages as plain records, with no
//! OWL contribution, private-link redaction as for any bundle, and no Quartz
//! `publish/` tree.
//!
//! The invariants come from `docs/design/machine-bundle-brief.md`. Each test
//! builds from a scratch copy of the `estate-small` fixture.

#![cfg(feature = "whelk-el")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ontomark_compile::build::{self, Scope};
use ontomark_compile::input::vocabulary::Vocabulary;
use ontomark_compile::ir::projection::PublicationBlocked;
use ontomark_compile::profile::Profile;
use ontomark_compile::whelk::WhelkEl;
use serde_json::Value;

const PRIVATE_CANARY: &str = "PRIVATE-CANARY-NOTE-91c2";

fn profile() -> Profile {
    Profile::from_yaml(include_str!("../../../profiles/estate/profile.yaml"))
        .expect("estate profile")
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/fixtures/estate-small")
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create destination");
    for entry in walkdir::WalkDir::new(from) {
        let entry = entry.expect("walk the fixture");
        let rel = entry.path().strip_prefix(from).expect("relative path");
        let target = to.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&target).expect("create directory");
        } else {
            std::fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

/// A scratch copy of the estate-small fixture: `(scratch, repo)`.
fn fixture_repo() -> (tempfile::TempDir, PathBuf) {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let repo = scratch.path().join("repo");
    copy_tree(&fixture(), &repo);
    (scratch, repo)
}

fn options(repo: &Path, out: &Path, scope: Scope) -> build::Options {
    build::Options {
        vault_root: repo.join("knowledge"),
        out: out.to_path_buf(),
        repo_root: repo.to_path_buf(),
        with_markdown_mirror: false,
        generated_at: Some("2026-09-22T00:00:00+00:00".to_owned()),
        source: Some(("0000000000000000000000000000000000000000".to_owned(), false)),
        with_working_publish: false,
        redact_private_links: None,
        publish_out: None,
        stale_after: None,
        scope,
        certificate_out: None,
    }
}

fn run(repo: &Path, opts: &build::Options) -> anyhow::Result<build::Report> {
    let vocab = Vocabulary::load(repo.join("ontology/vocabulary.yaml")).expect("vocabulary");
    build::run(opts, &profile(), &vocab, &WhelkEl)
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Build `scope` of `repo` into `<scratch>/<scope>`.
fn build_scope(
    scratch: &tempfile::TempDir,
    repo: &Path,
    scope: Scope,
) -> (PathBuf, anyhow::Result<build::Report>) {
    let out = scratch.path().join(scope.as_str());
    let report = run(repo, &options(repo, &out, scope));
    (out, report)
}

/// Every file under `dir`, relative path to bytes.
fn tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .map(|e| e.expect("walk the bundle"))
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            let rel = e
                .path()
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            (rel, std::fs::read(e.path()).unwrap())
        })
        .collect()
}

fn json(out: &Path, name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(out.join(name)).unwrap()).unwrap()
}

/// A working note page: `type: Note`, a title, optional extra frontmatter.
fn note(title: &str, public: bool, extra: &str, body: &str) -> String {
    format!(
        "---\ntype: Note\ntitle: {title}\npublic: {public}\nstatus: stable\n{extra}---\n\n{body}\n"
    )
}

/// A knowledge class page, valid under the fixture's role.
fn class(title: &str, public: bool, slug: &str) -> String {
    format!(
        "---\ntype: Class\ntitle: {title}\nresource: urn:ngm:class:{slug}\npublic: {public}\nstatus: draft\ndomain: blockchain\ngenerated:\n  by: human:npub16kv2h3hdalphuyzj8tkycq8cm9n54ymer23558ds80afkc9e7frqwrawzj\n  at: 2026-10-11T00:00:00Z\n---\n\nSynthetic class {title}.\n"
    )
}

fn built(scratch: &tempfile::TempDir, repo: &Path, scope: Scope) -> (PathBuf, build::Report) {
    let (out, report) = build_scope(scratch, repo, scope);
    (out, report.expect("the build succeeds"))
}

fn blocked(error: &anyhow::Error) -> &PublicationBlocked {
    error
        .downcast_ref::<PublicationBlocked>()
        .unwrap_or_else(|| panic!("a typed PublicationBlocked, got: {error:#}"))
}

#[test]
fn audit_non_ascii_slug_collision_is_real_and_refused() {
    use ontomark_compile::ir::projection::working_slug;
    let a = "雪三地地百/Foo";
    let b = "雪五五四三/Foo";
    assert_ne!(a, b);
    assert_eq!(working_slug(a), "working/x-24f80a23/foo");
    assert_eq!(working_slug(a), working_slug(b));
    let (scratch, repo) = fixture_repo();
    write(
        &repo.join(format!("working/{a}.md")),
        &note(
            "First Unicode",
            true,
            "resource: urn:ngm:class:first-unicode\n",
            "First.",
        ),
    );
    write(
        &repo.join(format!("working/{b}.md")),
        &note(
            "Second Unicode",
            true,
            "resource: urn:ngm:class:second-unicode\n",
            "Second.",
        ),
    );
    let (out, result) = build_scope(&scratch, &repo, Scope::Machine);
    let error = result.unwrap_err();
    assert!(matches!(blocked(&error), PublicationBlocked::Collision(_)));
    assert!(!out.exists());
    println!(
        "AUDIT 32-bit slug collision: {a} and {b}; correctly refused before publication: {error}"
    );
}

#[test]
fn audit_bare_private_working_name_survives_in_knowledge_title() {
    let (scratch, repo) = fixture_repo();
    write(
        &repo.join("knowledge/Knowledge Reference.md"),
        &class(
            "Knowledge mentions Fixture Private Note",
            true,
            "knowledge-reference",
        ),
    );
    write(
        &repo.join("working/Working Reference.md"),
        &note(
            "Working mentions Fixture Private Note",
            true,
            "",
            "Reference.",
        ),
    );
    let (out, _) = built(&scratch, &repo, Scope::Machine);
    let bytes = tree(&out);
    let titles: Vec<String> = bytes
        .iter()
        .filter(|(n, _)| n.starts_with("api/pages/"))
        .filter_map(|(_, b)| serde_json::from_slice::<Value>(b).ok())
        .filter_map(|v| v.get("title").and_then(Value::as_str).map(str::to_owned))
        .collect();
    assert!(
        titles
            .iter()
            .any(|t| t == "Knowledge mentions Fixture Private Note"),
        "{titles:?}"
    );
    assert!(
        titles.iter().any(|t| t == "Working mentions [private]"),
        "{titles:?}"
    );
    assert!(!bytes
        .values()
        .any(|b| String::from_utf8_lossy(b).contains(PRIVATE_CANARY)));
    let (public, _) = built(&scratch, &repo, Scope::Public);
    assert!(tree(&public)
        .values()
        .any(|b| String::from_utf8_lossy(b).contains("Knowledge mentions Fixture Private Note")));
    println!("AUDIT marker boundary: private working title survives in public knowledge title in both scopes; masked in working title; private body canary absent");
}

#[test]
fn audit_profile_change_is_not_part_of_content_digest() {
    let (scratch, repo) = fixture_repo();
    let vocab = Vocabulary::load(repo.join("ontology/vocabulary.yaml")).unwrap();
    let first = scratch.path().join("first");
    let second = scratch.path().join("second");
    let a = build::run(
        &options(&repo, &first, Scope::Machine),
        &profile(),
        &vocab,
        &WhelkEl,
    )
    .unwrap();
    let mut changed = profile();
    changed.iris.page_namespace = "urn:audit:changed-page:".to_owned();
    let b = build::run(
        &options(&repo, &second, Scope::Machine),
        &changed,
        &vocab,
        &WhelkEl,
    )
    .unwrap();
    assert_eq!(a.generation.content_digest, b.generation.content_digest);
    assert_eq!(a.generation.id, b.generation.id);
    let path = "api/pages/working/fixture-public-note.json";
    assert_ne!(json(&first, path)["id"], json(&second, path)["id"]);
    assert_ne!(a.generation.artifacts, b.generation.artifacts);
    println!("AUDIT profile identity: same id/content_digest, changed page IRI; artifact hashes distinguish outputs");
}

#[test]
fn audit_vocabulary_comment_bytes_do_not_change_content_digest() {
    let (scratch, repo) = fixture_repo();
    let (first, a) = built(&scratch, &repo, Scope::Machine);
    let vocab = repo.join("ontology/vocabulary.yaml");
    let original = std::fs::read_to_string(&vocab).unwrap();
    std::fs::write(
        &vocab,
        format!("{original}\n# audit: different input bytes, same parsed vocabulary\n"),
    )
    .unwrap();
    let second = scratch.path().join("after-comment");
    let b = run(&repo, &options(&repo, &second, Scope::Machine)).unwrap();
    assert_eq!(a.generation.content_digest, b.generation.content_digest);
    assert_eq!(tree(&first), tree(&second));
    println!("AUDIT vocabulary digest: a new comment changes source bytes but not content_digest or output bytes");
}

#[test]
fn audit_source_identity_can_be_caller_asserted() {
    let (scratch, repo) = fixture_repo();
    let out = scratch.path().join("asserted");
    let mut opts = options(&repo, &out, Scope::Machine);
    opts.source = Some(("not-a-real-commit".into(), false));
    let r = run(&repo, &opts).unwrap();
    assert_eq!(r.generation.commit, "not-a-real-commit");
    assert!(!r.generation.dirty);
    assert_eq!(r.generation.id, "visionGraph@not-a-real-commit");
    println!("AUDIT source identity override: unvalidated caller assertion retained; manifest hashes still describe emitted files");
}
