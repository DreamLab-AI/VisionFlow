use dream_core::streak::{advance_failures, record_night};
use dream_gate::gate::{Veto, VetoClass};
use failure_fingerprint::{advance, fingerprint, normalize_cause, Manifest, Observation};

#[test]
fn separator_in_names_aliases_distinct_tuples() {
    let a = Observation::new("alpha\u{1f}beta", "gamma", "failure");
    let b = Observation::new("alpha", "beta\u{1f}gamma", "failure");
    assert_ne!((&a.task, &a.stage), (&b.task, &b.stage));
    assert_eq!(a.fingerprint(), b.fingerprint());
    let (prior, _) = advance(&Manifest::new(0), &[a], 1);
    let (_, class) = advance(&prior, &[b], 2);
    assert_eq!(class.persisting.len(), 1);
    assert!(class.new.is_empty());
}

#[test]
fn harmless_credential_shaped_canary_survives_normalisation() {
    let cause = "password=canary! token=abcd.efgh.ijkl";
    assert_eq!(normalize_cause(cause), cause);
    let (m, _) = advance(
        &Manifest::new(0),
        &[Observation::new("task", "stage", cause)],
        1,
    );
    assert!(m.to_json().contains("password=canary!"));
    assert!(m.to_json().contains("abcd.efgh.ijkl"));
}

#[test]
fn numeric_and_truncated_causes_alias_by_design() {
    assert_eq!(
        fingerprint("t", "s", "HTTP 401"),
        fingerprint("t", "s", "HTTP 500")
    );
    let prefix = "word ".repeat(50);
    assert_eq!(
        fingerprint("t", "s", &format!("{prefix}first")),
        fingerprint("t", "s", &format!("{prefix}second"))
    );
}

#[test]
fn manifest_reader_does_not_validate_record_identity_or_normalisation() {
    let (mut m, _) = advance(&Manifest::new(0), &[Observation::new("t", "s", "boom")], 1);
    m.records[0].cause = "raw /tmp/canary 123456".into();
    m.records[0].task = "different".into();
    let loaded = Manifest::from_json(&m.to_json()).unwrap();
    assert_eq!(loaded, m);
    assert_ne!(
        loaded.records[0].fingerprint,
        fingerprint("different", "s", &loaded.records[0].cause)
    );
}

#[test]
fn repeated_step_still_increments_count() {
    let obs = [Observation::new("t", "s", "boom")];
    let (a, _) = advance(&Manifest::new(0), &obs, 10);
    let (b, c) = advance(&a, &obs, 10);
    assert_eq!(b.records[0].count, 2);
    assert_eq!(b.step, 10);
    assert_eq!(c.persisting.len(), 1);
}

fn vetoes() -> Vec<Veto> {
    vec![Veto {
        class: VetoClass::Harness,
        subject: "tests".into(),
        reason: "timeout".into(),
    }]
}

#[test]
fn duplicate_record_calls_are_counted_as_separate_nights() {
    let d = tempfile::tempdir().unwrap();
    assert!(
        !record_night(d.path(), "repo", "BLOCKED-ENV", &vetoes())
            .unwrap()
            .stuck
    );
    let next = record_night(d.path(), "repo", "BLOCKED-ENV", &vetoes()).unwrap();
    assert!(next.stuck);
    assert_eq!(next.manifest.step, 2);
}

#[test]
fn max_manifest_step_overflows_in_debug_build() {
    let result = std::panic::catch_unwind(|| {
        advance_failures(&Manifest::new(u64::MAX), "BLOCKED-ENV", &vetoes())
    });
    assert!(result.is_err());
}
