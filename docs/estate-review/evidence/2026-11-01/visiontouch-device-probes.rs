
#[cfg(test)]
mod estate_audit {
    use super::*;
    #[test]
    fn sdk_accepts_raised_defaults_and_has_no_continuous_dwell_budget() {
        let mut g = ForceGuard::default();
        g.set_limits(ForceLimits { force_n: 2., dac_counts: 300, slew_n_s: 100. }).unwrap();
        g.arm();
        let start = Instant::now();
        for n in 0..=6000 {
            let t = start + Duration::from_millis(n * 2);
            g.request(Demand::Force([0., 2., 0.]), t).unwrap();
            g.tick(t, Some([0; 6]));
        }
        println!("after 12 seconds: {:?}", g.telemetry());
        assert!(g.telemetry().applied_force[1] > CONTINUOUS_FORCE_N);
        assert!(g.telemetry().applied_motor_offsets.iter().all(|x| x.abs() <= 300));
        assert!(!g.telemetry().expired);
        assert!(g.set_limits(ForceLimits::default()).is_err());
        let t = start + Duration::from_millis(12026);
        assert_eq!(g.tick(t, Some([0; 6])), [0; 3]);
        assert!(g.telemetry().expired);
    }
}
