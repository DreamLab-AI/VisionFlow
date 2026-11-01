use crate::frame_budget::*;

#[test]
fn call_pressure_and_portal_floor_observations() {
    let graph = GraphDemand {
        nodes: 1_000,
        edges: 1_000,
        draw_calls: 49,
        ..Default::default()
    };
    let plain = allocate(&graph, &MemoryDemand::default());
    let extras = allocate_with_extras(
        &graph,
        &MemoryDemand::default(),
        &ExtrasDemand { tier: 4, frames: 3 },
    );
    assert!(extras.over_budget);
    assert_eq!(extras.draw_calls, 49);
    assert_eq!(extras.extras_tier, 2);
    assert_eq!(extras.extras_allowance, 72);
    let weighted = split_near_caps_weighted(35, &[4_388; 3], &[1.0, 0.15, 0.15], 16);
    assert_eq!(weighted, split_near_caps(35, &[4_388; 3], 16));
    let result = serde_json::json!({
        "call_pressure": {"plain":format!("{plain:?}"), "extras":format!("{extras:?}"),
            "observed_allowance":extras.extras_allowance,"over_budget":extras.over_budget},
        "three_volume_portal_floor": {"cap":35,"demands":[4388,4388,4388],"weights":[1.0,0.15,0.15],"split":weighted},
        "frame_cost": {"eight":extras_cost(2,8).0,"nine":extras_cost(2,9).0}
    });
    std::fs::write(
        std::env::var("XR_BUDGET_AUDIT_OUT").unwrap(),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
}
