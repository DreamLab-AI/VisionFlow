//! Audit-only probes. Fixed test keys and an in-memory relay; no network.
mod common;
use common::*;
use nostr_bbs_core::sealed::seal_original;
use nostr_bbs_core::NostrEvent;
use nostr_bbs_zone_migrate::channels::ChannelSpec;
use nostr_bbs_zone_migrate::engine;
use nostr_bbs_zone_migrate::relay::{DeleteOutcome, Filter, PublishOutcome, Relay, RelayError};
use nostr_bbs_zone_migrate::state::{State, Status};

fn channels() -> Vec<ChannelSpec> {
    vec![ChannelSpec {
        id: CHANNEL.into(),
        zone: "zone3".into(),
    }]
}

fn setup() -> (FakeRelay, NostrEvent, State, Sink) {
    let mut relay = FakeRelay::with_admin(&pk(MIGRATOR));
    let original = post(AUTHOR, CHANNEL, 1_600_000_000, "synthetic history");
    relay.insert(original.clone());
    (
        relay,
        original,
        State::new(RELAY_URL, &pk(MIGRATOR)),
        Sink::default(),
    )
}

fn seal_and_verify<R: Relay>(relay: &mut R, state: &mut State, sink: &mut Sink) {
    let report = sink
        .run(|h| {
            block_on(engine::seal(
                relay,
                &sk(MIGRATOR),
                &channels(),
                &ring(),
                state,
                None,
                h,
            ))
        })
        .unwrap();
    assert!(report.failed.is_empty());
    let report = sink
        .run(|h| block_on(engine::verify(relay, &ring(), state, h)))
        .unwrap();
    assert!(report.failed.is_empty());
}

struct ZeroDelete(FakeRelay);
impl Relay for ZeroDelete {
    async fn query(&mut self, f: &Filter) -> Result<Vec<NostrEvent>, RelayError> {
        self.0.query(f).await
    }
    async fn publish(&mut self, e: &NostrEvent) -> Result<PublishOutcome, RelayError> {
        self.0.publish(e).await
    }
    async fn is_admin(&mut self, p: &str) -> Result<bool, RelayError> {
        self.0.is_admin(p).await
    }
    async fn delete_events(&mut self, _: &[String], _: &str) -> Result<DeleteOutcome, RelayError> {
        // A syntactically valid success body that accounts for no requested id.
        Ok(serde_json::from_str(r#"{"deleted":0}"#).unwrap())
    }
}

#[test]
fn zero_delete_ack_marks_retained_original_purged_and_retry_does_nothing() {
    let (relay, original, mut state, mut sink) = setup();
    let mut relay = ZeroDelete(relay);
    seal_and_verify(&mut relay, &mut state, &mut sink);
    let report = sink
        .run(|h| block_on(engine::purge(&mut relay, &ring(), &mut state, true, h)))
        .unwrap();
    assert_eq!(report.deleted, 0);
    assert_eq!(report.counts.purged, 1);
    assert!(relay.0.has(&original.id));
    assert!(sink
        .lines
        .iter()
        .any(|l| l.contains("warning: relay reported 0 deleted")));
    let retry = sink
        .run(|h| block_on(engine::purge(&mut relay, &ring(), &mut state, true, h)))
        .unwrap();
    assert_eq!(retry.requests, 0);
}

#[test]
fn verify_does_not_recheck_a_purged_envelope_that_has_disappeared() {
    let (mut relay, _, mut state, mut sink) = setup();
    seal_and_verify(&mut relay, &mut state, &mut sink);
    sink.run(|h| block_on(engine::purge(&mut relay, &ring(), &mut state, true, h)))
        .unwrap();
    relay.events.clear();
    let report = sink
        .run(|h| block_on(engine::verify(&mut relay, &ring(), &mut state, h)))
        .unwrap();
    assert_eq!(
        (report.checked, report.verified, report.failed.len()),
        (0, 0, 0)
    );
    assert_eq!(report.counts.purged, 1);
}

#[test]
fn wrong_declared_zone_pubkey_is_adopted_verified_and_original_purged() {
    let (mut relay, original, mut state, mut sink) = setup();
    let envelope = seal_original(&original, "zone3", 2, &pk(ZONE3_E2), &sk(MIGRATOR)).unwrap();
    let mut tags = envelope.tags.clone();
    tags[1][3] = pk(ZONE3_E1); // Valid hex/key, but disagrees with the encryption key.
    let malformed = sign(MIGRATOR, 42, envelope.created_at, tags, &envelope.content);
    relay.insert(malformed.clone());
    let report = sink
        .run(|h| {
            block_on(engine::seal(
                &mut relay,
                &sk(MIGRATOR),
                &channels(),
                &ring(),
                &mut state,
                None,
                h,
            ))
        })
        .unwrap();
    assert_eq!((report.adopted, report.sealed), (1, 0));
    let report = sink
        .run(|h| block_on(engine::verify(&mut relay, &ring(), &mut state, h)))
        .unwrap();
    assert_eq!(report.verified, 1);
    sink.run(|h| block_on(engine::purge(&mut relay, &ring(), &mut state, true, h)))
        .unwrap();
    assert!(!relay.has(&original.id));
    assert_eq!(state.entries[&original.id].status, Status::Purged);
    assert_ne!(
        malformed.tags[1][3],
        ring().get("zone3", 2).unwrap().pubkey()
    );
}

#[test]
fn forged_marker_inflates_plan_already_sealed_but_seal_repairs_it() {
    let (mut relay, original, mut state, mut sink) = setup();
    let mut envelope = seal_original(&original, "zone3", 2, &pk(ZONE3_E2), &sk(MIGRATOR)).unwrap();
    envelope.content = "invalid ciphertext and signature".into();
    relay.insert(envelope);
    let (plan, _) = block_on(engine::scan(&mut relay, &channels(), &ring())).unwrap();
    assert_eq!(plan.totals(), (1, 1, 0));
    let report = sink
        .run(|h| {
            block_on(engine::seal(
                &mut relay,
                &sk(MIGRATOR),
                &channels(),
                &ring(),
                &mut state,
                None,
                h,
            ))
        })
        .unwrap();
    assert_eq!((report.sealed, report.adopted), (1, 0));
}

#[test]
fn dense_second_can_finish_recorded_migration_with_plaintext_left_over() {
    let (mut relay, _, mut state, mut sink) = setup();
    relay.events.clear();
    for i in 0..1001 {
        relay.insert(post(
            AUTHOR,
            CHANNEL,
            1_600_000_000,
            &format!("synthetic {i}"),
        ));
    }
    let report = sink
        .run(|h| {
            block_on(engine::seal(
                &mut relay,
                &sk(MIGRATOR),
                &channels(),
                &ring(),
                &mut state,
                None,
                h,
            ))
        })
        .unwrap();
    assert_eq!(
        report.plan.incomplete_seconds,
        vec![(CHANNEL.into(), 1_600_000_000)]
    );
    assert_eq!(
        (report.sealed, report.remaining, report.failed.len()),
        (1000, 0, 0)
    );
    let report = sink
        .run(|h| block_on(engine::verify(&mut relay, &ring(), &mut state, h)))
        .unwrap();
    assert_eq!(report.verified, 1000);
    let report = sink
        .run(|h| block_on(engine::purge(&mut relay, &ring(), &mut state, true, h)))
        .unwrap();
    assert_eq!(report.counts.purged, 1000);
    assert_eq!(
        relay
            .events
            .iter()
            .filter(|e| !e.tags.iter().any(|t| t[0] == "sealed"))
            .count(),
        1
    );
}
