//! Real CLI dry-run probes. All pages, Git commits and signing keys are fixtures.
//! Usage: proposal-construction-probe /absolute/path/to/vault /absolute/path/to/ontomark
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
const KEY: &str = "0101010101010101010101010101010101010101010101010101010101010101";
const VOCAB: &str = "version: 1\nnamespace: 'urn:ngm:class:'\nprefixes:\n  vc: 'https://narrativegoldmine.com/ns/v1#'\ntypes:\n  Class: { owl: 'owl:Class', required: [resource, status] }\nrelations:\n  is-a: { owl: 'rdfs:subClassOf' }\n  requires: { owl: 'vc:requires', restriction: true }\n  defines-as: { owl: 'owl:equivalentClass' }\nscalars:\n  quality: { type: number, min: 0, max: 1 }\n";
fn page(name: &str, extra: &str) -> String {
    format!("---\ntype: Class\ntitle: {name}\npublic: true\nresource: urn:ngm:class:{}\nstatus: stable\n{extra}---\nAbout {name}.\n", name.to_lowercase())
}
fn write(path: &Path, content: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}
struct Fixture {
    dir: PathBuf,
    bin: PathBuf,
}
impl Fixture {
    fn new(bin: PathBuf) -> Self {
        let dir = std::env::temp_dir().join(format!("vf-proposal-audit-{}", std::process::id()));
        assert!(!dir.exists());
        let f = Self { dir, bin };
        write(&f.dir.join("repo/ontology/vocabulary.yaml"), VOCAB);
        for name in ["Alpha", "Beta", "Gamma"] {
            f.base(name, "");
        }
        f.git(&["init", "-q"]);
        f.commit("fixture initial state");
        f
    }
    fn base(&self, name: &str, extra: &str) {
        write(
            &self.dir.join(format!("repo/knowledge/{name}.md")),
            &page(name, extra),
        );
    }
    fn git(&self, args: &[&str]) {
        let r = Command::new("git")
            .current_dir(self.dir.join("repo"))
            .args([
                "-c",
                "user.name=Audit fixture",
                "-c",
                "user.email=audit@example.invalid",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    }
    fn commit(&self, msg: &str) {
        self.git(&["add", "knowledge", "ontology"]);
        self.git(&["commit", "-qm", msg]);
    }
    fn run(&self, diff: &Path, extra: &[&str]) -> Value {
        let r = Command::new(&self.bin)
            .current_dir(&self.dir)
            .arg("--repo")
            .arg(self.dir.join("repo"))
            .args([
                "--json",
                "propose",
                "Alpha",
                "--dry-run",
                "--secret",
                KEY,
                "--hypothesis",
                "Synthetic audit fixture",
                "--diff",
            ])
            .arg(diff)
            .args(extra)
            .env_remove("VAULT_NOSTR_SECRET")
            .output()
            .unwrap();
        assert!(
            r.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&r.stdout),
            String::from_utf8_lossy(&r.stderr)
        );
        serde_json::from_slice(&r.stdout).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.dir).unwrap();
    }
}
fn main() {
    let bin = std::fs::canonicalize(std::env::args().nth(1).expect("vault binary")).unwrap();
    let f = Fixture::new(bin);
    let stage = f.dir.join("staged.md");
    write(&stage, &page("Alpha", "is-a:\n- '[[Gamma]]'\n"));
    let single = f.run(&stage, &[]);
    assert_eq!(single["proposal"]["level"], "content");
    assert_eq!(
        single["proposal"]["consequences"]["verdict"],
        "not_conservative_under_rule_table"
    );
    f.base("Gamma", "is-a:\n- '[[Beta]]'\n");
    f.commit("fixture changes untouched parent");
    let context_change = f.run(&stage, &[]);
    assert_eq!(
        single["proposal"]["digest"],
        context_change["proposal"]["digest"]
    );
    assert_eq!(
        single["proposal"]["diff"],
        context_change["proposal"]["diff"]
    );
    assert_ne!(
        single["proposal"]["generation"],
        context_change["proposal"]["generation"]
    );
    assert_ne!(
        single["proposal"]["consequences"],
        context_change["proposal"]["consequences"]
    );
    assert_ne!(single["event"]["id"], context_change["event"]["id"]);

    f.base("Gamma", "");
    f.base("Alpha", "defines-as:\n- requires: '[[Gamma]]'\n");
    f.commit("fixture restriction base");
    write(
        &stage,
        &page("Alpha", "defines-as:\n- requires: '[[Beta]]'\n"),
    );
    let restriction = f.run(&stage, &[]);
    assert_eq!(
        restriction["proposal"]["consequences"]["verdict"],
        "not_conservative_under_rule_table"
    );
    assert_eq!(
        restriction["proposal"]["consequences"]["new_by_predicate"],
        json!({"versionIRI": 1})
    );
    assert_eq!(
        restriction["proposal"]["consequences"]["new_consequences"],
        1
    );
    assert_eq!(
        restriction["proposal"]["consequences"]["lost_consequences"],
        1
    );
    assert!(
        restriction["proposal"]["consequences"]["not_compared_blank_node_bearing"]
            .as_u64()
            .unwrap()
            > 0
    );

    write(
        &f.dir.join("repo/knowledge/Alpha.md"),
        &page("Alpha", "").replace("status: stable\n", ""),
    );
    f.commit("fixture preexisting missing required status");
    write(
        &stage,
        &page("Alpha", "is-a:\n- '[[Gamma]]'\n").replace("status: stable\n", ""),
    );
    let preexisting = f.run(&stage, &[]);
    assert_eq!(preexisting["proposal"]["blockers"], json!([]));
    assert!(!preexisting["proposal"]["preexisting"]
        .as_array()
        .unwrap()
        .is_empty());

    f.base("Alpha", "");
    f.commit("fixture reset");
    write(
        &stage,
        &page("Alpha", "").replace("urn:ngm:class:alpha", "urn:ngm:class:replacement"),
    );
    let resource_change = f.run(&stage, &[]);
    assert_eq!(resource_change["proposal"]["blockers"], json!([]));
    assert_eq!(resource_change["proposal"]["iri"], "urn:ngm:class:alpha");
    let manifest = f.dir.join("manifest");
    for name in ["Alpha", "Beta", "Gamma"] {
        write(
            &manifest.join(format!("{name}.md")),
            &page(name, "quality: 0.5\n"),
        );
    }
    let grouped = f.run(&manifest, &["--title", "Three page judgement"]);
    assert_eq!(grouped["proposal"]["pages"].as_array().unwrap().len(), 3);
    assert_eq!(grouped["proposal"]["level"], "content");
    assert_eq!(grouped["proposal"]["kind"], "amend");
    write(
        &manifest.join("Epsilon.md"),
        &page("Epsilon", "quality: 0.5\n"),
    );
    let mixed = f.run(&manifest, &["--title", "Mixed creation and amendment"]);
    assert_eq!(mixed["proposal"]["kind"], "create");
    assert_eq!(mixed["proposal"]["pages"].as_array().unwrap().len(), 4);

    assert_eq!(
        std::fs::read_to_string(f.dir.join("repo/knowledge/Alpha.md")).unwrap(),
        page("Alpha", "")
    );
    let writer = std::fs::canonicalize(std::env::args().nth(2).expect("ontomark binary")).unwrap();
    let proposal_path = f.dir.join("approved-resource-change.json");
    write(
        &proposal_path,
        &serde_json::to_string(&resource_change["proposal"]).unwrap(),
    );
    let applied = Command::new(&writer)
        .arg("--repo")
        .arg(f.dir.join("repo"))
        .args(["--json", "apply"])
        .arg(&proposal_path)
        .args(["--expect", "docs=1"])
        .output()
        .unwrap();
    assert!(
        applied.status.success(),
        "{} {}",
        String::from_utf8_lossy(&applied.stdout),
        String::from_utf8_lossy(&applied.stderr)
    );
    let apply_result: Value = serde_json::from_slice(&applied.stdout).unwrap();
    let after_apply = std::fs::read_to_string(f.dir.join("repo/knowledge/Alpha.md")).unwrap();
    assert!(after_apply.contains("resource: urn:ngm:class:replacement"));
    println!("{}", serde_json::to_string_pretty(&json!({"binary":f.bin,"writer":writer,"single_default_level":single,"same_patch_new_context":context_change,"restriction_filler_change":restriction,"preexisting_validation":preexisting,"resource_change":resource_change,"resource_apply":{"result":apply_result,"page_after":after_apply},"grouped_amend":grouped,"mixed_grouped_create":mixed,"fixture_target_unchanged_by_propose":true})).unwrap());
}
