//! Private audit runner for the unmodified interaction policy and its native tests.
#[cfg(test)]
#[path = "../source/xr-client/rust/src/interaction.rs"]
pub mod production;

#[cfg(test)]
mod boundary {
    use super::production::{find_target, is_grab_active, HandRay, TargetCandidate};

    #[test]
    fn capture_accepts_both_sides_of_the_gdscript_depth_clamp() {
        let ray = HandRay {
            origin: [0.0; 3],
            direction: [0.0, 0.0, -1.0],
            pinch_strength: 1.0,
            is_tracking: true,
        };
        assert!(is_grab_active(&ray));
        for depth in [0.1, 1.0, 6.0, 10.0, 30.0] {
            let hit = find_target(
                &ray,
                &[TargetCandidate {
                    node_id: 11,
                    position: [0.0, 0.0, -depth],
                }],
            )
            .expect("capture accepts a depth that the GDScript fixture can clamp");
            assert_eq!(hit.distance, depth);
        }
        assert!(find_target(
            &ray,
            &[TargetCandidate {
                node_id: 11,
                position: [0.0, 0.0, -30.1]
            }],
        )
        .is_none());
    }
}
