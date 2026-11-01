use hold_contract_probe::{HoldRegistry, Release};
use std::time::{Duration, Instant};

#[test]
fn a_late_hold_after_disconnect_is_readmitted_until_the_backstop_runs() {
    let mut holds = HoldRegistry::new();
    let now = Instant::now();
    holds.hold(1, 7, [0.0; 3], now);
    assert_eq!(holds.release_holder(1), vec![(7, Release::Rejoined)]);
    holds.hold(1, 7, [1.0; 3], now);
    assert_eq!(holds.holder_of(7), Some(1));
    assert!(holds
        .expire(now + Duration::from_secs(5), Duration::from_secs(5))
        .is_empty());
    assert_eq!(
        holds.expire(now + Duration::from_secs(6), Duration::from_secs(5)),
        vec![(7, Release::Rejoined)]
    );
}

#[test]
fn capacity_is_a_socket_rule_not_a_registry_invariant() {
    let mut holds = HoldRegistry::new();
    for id in 1..=3 {
        holds.hold(1, id, [0.0; 3], Instant::now());
    }
    assert_eq!(holds.held_count(), 3);
}

#[test]
fn stale_release_leaves_gpu_ownership_despite_a_node_only_release_notification() {
    let mut holds = HoldRegistry::new();
    let mut orchestrator_nodes = std::collections::HashSet::new();
    holds.hold(1, 7, [1.0; 3], Instant::now());
    orchestrator_nodes.insert(7);
    holds.hold(2, 7, [2.0; 3], Instant::now());
    orchestrator_nodes.insert(7);
    assert_eq!(holds.release(1, 7), Release::Ignored);
    // This HashSet is an explicit model of the inspected orchestrator handler,
    // not an invocation of the actual actor or an end-to-end test.
    orchestrator_nodes.remove(&7);
    assert!(orchestrator_nodes.is_empty());
    assert_eq!(holds.holder_of(7), Some(2));
}
