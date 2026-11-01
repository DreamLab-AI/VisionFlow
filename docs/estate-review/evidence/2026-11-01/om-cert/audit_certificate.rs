//! Disposable integration probes; actual compiler/certificate APIs, synthetic files only.
use ontomark_compile::build::turtle::{Graph, Subject, Term};
use ontomark_compile::{
    build::{self, Options, Scope},
    certificate,
    input::vocabulary::Vocabulary,
    profile::Profile,
    reason,
    whelk::WhelkEl,
};
use std::{fs, path::Path, process::Command};

const SC: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const OWL: &str = "http://www.w3.org/2002/07/owl#";
fn options(root: &Path) -> Options {
    Options {
        vault_root: root.join("knowledge"),
        out: root.join("www"),
        repo_root: root.to_path_buf(),
        with_markdown_mirror: false,
        with_working_publish: false,
        redact_private_links: None,
        publish_out: None,
        generated_at: Some("2026-11-01T00:00:00Z".into()),
        source: Some(("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(), false)),
        stale_after: None,
        scope: Scope::Public,
        certificate_out: Some(root.join("certificate")),
    }
}
fn fixture(root: &Path, clash: bool) {
    fs::create_dir_all(root.join("knowledge")).unwrap();
    for (id, extra) in [
        ("Parent", ""),
        ("A", "is-a: [\"[[Parent]]\"]\ndisjoint-with: [\"[[B]]\"]\n"),
        ("B", "is-a: [\"[[Parent]]\"]\n"),
        (
            "P",
            if clash {
                "is-a: [\"[[A]]\", \"[[B]]\"]\n"
            } else {
                "is-a: [\"[[A]]\"]\n"
            },
        ),
    ] {
        fs::write(root.join("knowledge").join(format!("{id}.md")), format!("---\ntype: Class\npublic: true\nresource: urn:ngm:class:{}\nstatus: stable\ndomain: infrastructure\n{extra}---\n", id.to_lowercase())).unwrap();
    }
}
fn run(o: &Options) -> anyhow::Result<build::Report> {
    let p = Profile::from_yaml(include_str!("../../../profiles/estate/profile.yaml")).unwrap();
    let v = Vocabulary::from_yaml_str("version: 1\nnamespace: 'urn:ngm:class:'\ntypes:\n  Class: { owl: 'owl:Class', required: [resource] }\nrelations:\n  is-a: { owl: 'rdfs:subClassOf', characteristics: [transitive] }\n  disjoint-with: { owl: 'owl:disjointWith', status: provisional }\nscalars:\n  domain: { type: text, enum: [infrastructure] }\n").unwrap();
    build::run(o, &p, &v, &WhelkEl)
}
#[test]
fn audit_unsatisfiable_build_replaces_certificate_before_refusing() {
    let d = tempfile::tempdir().unwrap();
    fixture(d.path(), false);
    let o = options(d.path());
    run(&o).unwrap();
    let cert = o.certificate_out.as_ref().unwrap();
    let old = fs::read(cert.join("asserted.tsv")).unwrap();
    let bundle = fs::read(o.out.join("data/ontology.ttl")).unwrap();
    fixture(d.path(), true);
    let err = run(&o).unwrap_err();
    assert!(err.to_string().contains("UNSATISFIABLE_CLASSES"), "{err}");
    assert_ne!(fs::read(cert.join("asserted.tsv")).unwrap(), old);
    assert_eq!(fs::read(o.out.join("data/ontology.ttl")).unwrap(), bundle);
    println!("AUDIT unsatisfiable: old bundle preserved, certificate replaced, error={err}");
}
#[test]
fn audit_no_flag_build_has_no_certificate() {
    let d = tempfile::tempdir().unwrap();
    fixture(d.path(), false);
    let mut o = options(d.path());
    o.certificate_out = None;
    let r = run(&o).unwrap();
    assert!(r.certificate.is_none());
    assert!(!d.path().join("certificate").exists());
    println!("AUDIT no flag: successful bundle without certificate");
}
#[test]
fn audit_certificate_under_output_is_accepted_then_lost() {
    let d = tempfile::tempdir().unwrap();
    fixture(d.path(), false);
    let mut o = options(d.path());
    o.certificate_out = Some(o.out.join("certificate"));
    let r = run(&o).unwrap();
    let dir = r.certificate.unwrap().dir;
    println!(
        "AUDIT nested certificate: returned directory={}, exists={}, asserted_exists={}",
        dir.display(),
        dir.exists(),
        dir.join("asserted.tsv").exists()
    );
    assert!(
        !dir.join("asserted.tsv").exists(),
        "hypothesis: output replacement removes unstaged certificate"
    );
}
#[test]
fn audit_late_output_failure_leaves_certificate() {
    let d = tempfile::tempdir().unwrap();
    fixture(d.path(), false);
    let mut o = options(d.path());
    o.out = d.path().join("blocked/www");
    fs::write(d.path().join("blocked"), "occupied").unwrap();
    let err = run(&o).unwrap_err();
    assert!(o
        .certificate_out
        .as_ref()
        .unwrap()
        .join("asserted.tsv")
        .exists());
    assert_eq!(
        fs::read_to_string(d.path().join("blocked")).unwrap(),
        "occupied"
    );
    println!("AUDIT late output failure: certificate exists, output unchanged, error={err}");
}
#[test]
fn audit_graph_adapter_drops_universal_but_facade_refuses_it() {
    use ontomark_reason::axiom::{Axiom, ClassExpression as C, Property};
    let mut g = Graph::new();
    for c in ["urn:A", "urn:B"] {
        g.add_iri(c, TYPE, Term::Iri(format!("{OWL}Class")));
    }
    let b = g.fresh_blank();
    g.add_iri("urn:A", SC, Term::Blank(b.clone()));
    g.add(
        Subject::Blank(b.clone()),
        &format!("{OWL}onProperty"),
        Term::Iri("urn:r".into()),
    );
    g.add(
        Subject::Blank(b),
        &format!("{OWL}allValuesFrom"),
        Term::Iri("urn:B".into()),
    );
    let r = reason::reason(&g, &WhelkEl).unwrap();
    assert!(r.refusals.is_empty());
    assert!(r.is_consistent());
    let checked = ontomark_reason::check(&[Axiom::SubClassOf {
        sub: C::class("urn:A"),
        sup: C::Only {
            property: Property::named("urn:r"),
            filler: Box::new(C::class("urn:B")),
        },
    }]);
    assert_eq!(checked.refusals.len(), 1);
    assert_eq!(checked.refusals[0].code(), "EL_OUTSIDE_PROFILE");
    println!(
        "AUDIT graph adapter: zero refusals versus facade {}",
        checked.refusals[0].code()
    );
}
#[test]
fn audit_malformed_equivalent_definition_is_silent_to_caller() {
    let mut g = Graph::new();
    g.add_iri("urn:A", TYPE, Term::Iri(format!("{OWL}Class")));
    let b = g.fresh_blank();
    g.add_iri("urn:A", &format!("{OWL}equivalentClass"), Term::Blank(b));
    let r = reason::reason(&g, &WhelkEl).unwrap();
    assert_eq!(r.defined_classes, 0);
    assert!(r.refusals.is_empty());
    assert!(r.is_consistent());
    println!("AUDIT malformed definition: omitted, zero refusals");
}
#[test]
fn audit_parity_rejects_definition_entailment_beyond_rdfs() {
    let mut g = Graph::new();
    for c in ["urn:A", "urn:B"] {
        g.add_iri(c, TYPE, Term::Iri(format!("{OWL}Class")));
    }
    g.add_iri(
        "urn:A",
        &format!("{OWL}equivalentClass"),
        Term::Iri("urn:B".into()),
    );
    let r = reason::reason(&g, &WhelkEl).unwrap();
    let (a, b, same) = certificate::parity(&certificate::certify(&g), &r);
    assert!(!same);
    assert!(b > a);
    println!("AUDIT parity positive control: certified={a}, classifier={b}, same={same}");
}
#[test]
fn audit_partial_certificate_write_mixes_files() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path();
    let c = rdfs_certificate::certify([("<urn:A>", "<urn:p>", "<urn:B>")]);
    fs::write(dir.join("asserted.tsv"), "old asserted").unwrap();
    fs::write(dir.join("asserted.sha256"), "old digest").unwrap();
    fs::create_dir(dir.join("derivations.tsv")).unwrap();
    assert!(c.write(dir, "new").is_err());
    assert_eq!(
        fs::read(dir.join("asserted.tsv")).unwrap(),
        c.asserted_bytes()
    );
    assert_eq!(
        fs::read_to_string(dir.join("asserted.sha256")).unwrap(),
        "old digest"
    );
    println!("AUDIT write failure: new assertions coexist with old digest");
}
#[test]
fn audit_checker_accepts_omitted_derivations_and_changed_scope() {
    let bin = std::env::var("OO_CERT_BIN").expect("checker required, no skip");
    let d = tempfile::tempdir().unwrap();
    let sc = rdfs_certificate::nt::iri(SC);
    let c = rdfs_certificate::certify([
        ("<urn:A>", sc.as_str(), "<urn:B>"),
        ("<urn:B>", sc.as_str(), "<urn:C>"),
    ]);
    assert_eq!(c.derivation_count(), 1);
    c.write(d.path(), "generation-one").unwrap();
    let check = || {
        Command::new(&bin)
            .arg(d.path().join("asserted.tsv"))
            .arg(d.path().join("derivations.tsv"))
            .output()
            .unwrap()
    };
    let good = check();
    assert!(good.status.success());
    fs::write(d.path().join("derivations.tsv"), "").unwrap();
    fs::write(d.path().join("scope.tsv"), "unrelated generation").unwrap();
    let truncated = check();
    assert!(truncated.status.success());
    println!(
        "AUDIT omitted conclusions: {}",
        String::from_utf8_lossy(&truncated.stdout)
    );
}
