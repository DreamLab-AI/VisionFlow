//! Producer-to-pure-join audit. The runner copies link_index.rs byte-for-byte from VisionClaw.
#[allow(dead_code)]
#[path = "link_index.rs"]
mod link_index;
use codescape_ingest::annotations::{scan_text, AnnotationTable};
use codescape_model::{CodeNode, GraphSnapshot, Lang, NodeKind, Tier};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

fn snapshot() -> GraphSnapshot {
    GraphSnapshot {
        repo: Arc::from("audit-repo"),
        epoch: 3,
        revision: 23,
        generated_at_ms: 0,
        nodes: vec![CodeNode {
            id: 1,
            key: Arc::from("fixture.rs"),
            kind: NodeKind::File,
            lang: Lang::Rust,
            label: Arc::from("fixture.rs"),
            path: Some(Arc::from("fixture.rs")),
            span: None,
            parent: 0,
            community: 0,
            cluster: 0,
            importance: 0.5,
            tier: Tier::L1,
            size: 3,
        }],
        edges: vec![],
    }
}
fn refs(table: &AnnotationTable) -> Vec<link_index::CodeRef> {
    let frame = codescape_server::xr::initial_graph_load(
        &snapshot(),
        &HashSet::from([1]),
        table.ontology_refs(),
        0,
    );
    frame["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(link_index::code_ref_from_node)
        .collect()
}
fn main() {
    let arg = std::env::args().nth(1).expect("scratch directory");
    let root = Path::new(&arg);
    let file = root.join("fixture.rs");
    let iri = "urn:ngm:class:bm25";
    let targets = HashMap::from([(iri.to_string(), vec![40, 41])]);
    let mut cases = vec![];
    let text = format!("pub const EXAMPLE: &str = r#\"\n// @ontology {iri}\n\"#;\n");
    std::fs::write(&file, &text).unwrap();
    let mut table = AnnotationTable::scan(root, &snapshot());
    let joined = link_index::join(&refs(&table), 2, &targets, 7, |_| 1);
    assert_eq!(table.annotations().len(), 1);
    assert_eq!(joined.records.len(), 2);
    cases.push(json!({"case":"raw_string_example_becomes_annotation_and_links", "classification":"counterexample",
        "fixture":text,"annotation_line":table.annotations()[0].line,"links":joined.records.len()}));
    assert_eq!(joined.records[0].provenance.repo, "");
    assert_eq!(joined.records[0].provenance.revision, None);
    assert_eq!(joined.records[0].provenance.generations, [2, 7]);
    cases.push(json!({"case":"snapshot_repository_and_revision_absent_from_link_provenance", "classification":"contract_gap",
        "snapshot_repo":"audit-repo","snapshot_revision":23,"provenance":joined.records[0].provenance}));
    std::fs::write(&file, format!("// @ontology {iri}\nfn rank() {{}}\n")).unwrap();
    assert!(table.refresh(root, &snapshot(), &["fixture.rs"]));
    assert_eq!(
        link_index::join(&refs(&table), 2, &targets, 7, |_| 1)
            .records
            .len(),
        2
    );
    cases.push(
        json!({"case":"real_comment_and_multiple_targets", "classification":"passing_control"}),
    );
    for text in [
        format!("// see {iri}"),
        format!("call(); // @ontology {iri}"),
        "// @ontology urn:ngm:class:Bad".to_string(),
    ] {
        assert!(scan_text(&text).is_empty());
    }
    cases.push(json!({"case":"bare_inline_and_malformed_annotations_refused", "classification":"passing_control"}));
    let many: String = (0..33)
        .map(|i| format!("// @ontology urn:ngm:class:c{i:02}\n"))
        .collect();
    std::fs::write(&file, many).unwrap();
    table.refresh(root, &snapshot(), &["fixture.rs"]);
    assert_eq!(table.annotations().len(), 33);
    let code = refs(&table);
    assert_eq!(code[0].refs.len(), 32);
    let all_targets: HashMap<String, Vec<u32>> = (0..33)
        .map(|i| (format!("urn:ngm:class:c{i:02}"), vec![100 + i]))
        .collect();
    let limited = link_index::join(&code, 2, &all_targets, 7, |_| 1);
    assert_eq!((limited.records.len(), limited.unresolved), (32, 0));
    cases.push(json!({"case":"producer_cap_is_invisible_to_join_unresolved_count", "classification":"documented_limit_with_observability_gap",
        "bound_annotations":33,"stream_annotations":32,"joined":32,"unresolved":0}));
    std::fs::write(&file, "fn unannotated() {}\n").unwrap();
    assert!(table.refresh(root, &snapshot(), &["fixture.rs"]));
    assert!(table.is_empty());
    assert!(refs(&table).is_empty());
    cases.push(
        json!({"case":"annotation_removal_clears_stream_refs", "classification":"passing_control"}),
    );
    let file_ref = link_index::CodeRef {
        key: "file".into(),
        id: 1,
        path: "fixture.rs".into(),
        is_file: true,
        refs: vec![(iri.into(), 1)],
    };
    let mut symbol_ref = file_ref.clone();
    symbol_ref.key = "symbol".into();
    symbol_ref.id = 2;
    symbol_ref.is_file = false;
    let result = link_index::join(&[file_ref, symbol_ref], 4, &targets, 9, |_| 0);
    assert_eq!(result.records.len(), 2);
    assert!(result
        .records
        .iter()
        .all(|r| r.source.id == 2 && r.target.volume == "knowledge"));
    cases.push(json!({"case":"symbol_wins_rollup_dedup", "classification":"passing_control"}));
    let absent = link_index::join(&code, 2, &HashMap::new(), 7, |_| 1);
    assert_eq!((absent.records.len(), absent.unresolved), (0, 32));
    cases.push(
        json!({"case":"missing_targets_count_unresolved", "classification":"passing_control"}),
    );
    println!("{}",serde_json::to_string_pretty(&json!({"scope":"production Codescape library and verbatim pure XR module; no gdext wrapper or live server", "cases":cases})).unwrap());
}
