use crate::{filter, nip42, sql};
use nostr_bbs_core::{
    event::{verify_event, NostrEvent},
    groups,
};
use rusqlite::{params, Connection};
use serde_json::json;

fn emit(case: &str, result: serde_json::Value) {
    println!("AUDIT_CASE {}", json!({"case":case,"result":result}));
}
fn event(kind: u64, tags: Vec<Vec<String>>) -> NostrEvent {
    NostrEvent {
        id: "a".repeat(64),
        pubkey: "b".repeat(64),
        created_at: 100,
        kind,
        tags,
        content: String::new(),
        sig: "0".repeat(128),
    }
}
#[test]
fn group_kinds_have_no_current_read_gate_but_correspondence_does() {
    for mode in [nip42::AuthMode::Nip42, nip42::AuthMode::Allowlist] {
        for kind in [9, 39000, 39001, 39002, 39003] {
            let e = event(kind, vec![vec!["h".into(), "private-room".into()]]);
            assert!(nip42::protected_read_permitted(
                kind,
                &e.pubkey,
                &[],
                None,
                mode
            ));
            assert!(filter::zone_channel_id(&e).is_none());
        }
        assert!(!nip42::protected_read_permitted(
            4,
            "author",
            &[],
            None,
            mode
        ));
    }
    emit(
        "group_read_predicates_pass_anonymous",
        json!({"kinds":[9,39000,39001,39002,39003],"modes":2,"private_dm_control":"denied","scope":"real pure predicates; full Worker not executed"}),
    );
}
#[test]
fn published_builders_disagree_about_metadata_address() {
    let old =
        groups::create_group_metadata(&[1; 32], "general", "General", "about", "picture").unwrap();
    let new = groups::build_group_metadata(&[1; 32], "general", "General", None, None).unwrap();
    assert!(verify_event(&old) && verify_event(&new));
    assert_eq!(filter::tag_value(&old, "h").as_deref(), Some("general"));
    assert_eq!(filter::d_tag_value(&old), "");
    assert_eq!(filter::tag_value(&new, "h"), None);
    assert_eq!(filter::d_tag_value(&new), "general");
    emit(
        "signed_metadata_builders_disagree",
        json!({"old_h":"general","old_d":"","new_h":null,"new_d":"general","both_signatures_valid":true}),
    );
}
fn db() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch(sql::SCHEMA).unwrap();
    c
}
fn store(c: &mut Connection, id: &str, kind: i64, d: &str, time: i64, rev: u64) {
    let t = c.transaction().unwrap();
    if (10000..20000).contains(&kind) {
        t.execute(sql::DELETE_REPLACEABLE, params!["author", kind, time])
            .unwrap();
    } else {
        t.execute(sql::DELETE_ADDRESSABLE, params!["author", kind, d, time])
            .unwrap();
    }
    let tags = json!([["d", d], ["rev", rev.to_string()]]).to_string();
    t.execute(
        sql::INSERT,
        params![
            id,
            "author",
            time,
            kind,
            tags,
            "payload",
            "fixture-no-signature",
            d,
            1000
        ],
    )
    .unwrap();
    t.commit().unwrap();
}
fn ids(c: &Connection) -> Vec<String> {
    c.prepare("SELECT id FROM events ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}
#[test]
fn real_sql_retains_late_older_events() {
    for kind in [10312, 39000] {
        let mut c = db();
        store(&mut c, "newer", kind, "group", 200, 2);
        store(&mut c, "older", kind, "group", 100, 1);
        assert_eq!(ids(&c), vec!["newer", "older"]);
        let query = sql::READ_UNTAGGED
            .replace("{where_clause}", "WHERE kind = ?1")
            .replace("{limit_placeholder}", "?2");
        let rows: Vec<String> = c
            .prepare(&query)
            .unwrap()
            .query_map(params![kind, 10], |r| r.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(rows, vec!["newer", "older"]);
        emit(
            "late_older_event_retained",
            json!({"kind":kind,"rows":ids(&c)}),
        );
    }
}
#[test]
fn real_sql_retains_both_same_second_ids() {
    for kind in [10312, 39000] {
        let mut c = db();
        store(&mut c, "ff", kind, "group", 100, 1);
        store(&mut c, "00", kind, "group", 100, 1);
        assert_eq!(ids(&c), vec!["00", "ff"]);
        emit(
            "same_second_rows_coexist",
            json!({"kind":kind,"rows":ids(&c)}),
        );
    }
}
#[test]
fn real_sql_later_timestamp_erases_higher_revision() {
    for (kind, d) in [
        (39002, "group"),
        (30312, "group"),
        (30078, "visionclaw:dataspace:one"),
    ] {
        let mut c = db();
        store(&mut c, "higher-rev", kind, d, 100, 9);
        store(&mut c, "lower-rev", kind, d, 200, 8);
        assert_eq!(ids(&c), vec!["lower-rev"]);
        emit(
            "higher_revision_erased",
            json!({"kind":kind,"d":d,"remaining":ids(&c)}),
        );
    }
}
#[test]
fn newer_timestamp_and_address_isolation_controls() {
    let mut c = db();
    store(&mut c, "old-a", 39000, "a", 100, 1);
    store(&mut c, "b", 39000, "b", 100, 1);
    store(&mut c, "new-a", 39000, "a", 200, 2);
    assert_eq!(ids(&c), vec!["b", "new-a"]);
    store(&mut c, "new-a", 39000, "a", 200, 2);
    assert_eq!(ids(&c), vec!["b", "new-a"]);
    emit(
        "newer_and_address_isolation_controls",
        json!({"rows":ids(&c),"duplicate_id":"no extra row"}),
    );
}
#[test]
fn empty_addresses_from_legacy_metadata_collide() {
    let a = groups::create_group_metadata(&[1; 32], "a", "A", "", "").unwrap();
    let b = groups::create_group_metadata(&[1; 32], "b", "B", "", "").unwrap();
    let mut c = db();
    store(&mut c, "group-a", 39000, &filter::d_tag_value(&a), 100, 1);
    store(&mut c, "group-b", 39000, &filter::d_tag_value(&b), 101, 1);
    assert_eq!(ids(&c), vec!["group-b"]);
    emit(
        "legacy_metadata_empty_address_collision",
        json!({"remaining":ids(&c),"scope":"builder addresses plus real SQL; timestamps supplied by fixture"}),
    );
}
#[test]
fn encoded_dot_segments_normalise_outside_authored_prefix() {
    let raw = "/public/%2e%2e/admin";
    assert!(!raw.split('/').any(|s| s == "." || s == ".."));
    let resolved = url::Url::parse("https://home.example")
        .unwrap()
        .join(raw)
        .unwrap();
    assert_eq!(resolved.path(), "/admin");
    emit(
        "content_reference_normalisation",
        json!({"input":raw,"normalised_path":resolved.path(),"scope":"URL library counterexample to raw segment inspection; no production descriptor validator exists"}),
    );
}
