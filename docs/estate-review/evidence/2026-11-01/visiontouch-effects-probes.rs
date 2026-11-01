
#[cfg(test)]
mod estate_audit {
    use super::*;
    #[test]
    fn damping_makes_total_force_affine_in_stiffness() {
        for (kind, p, v) in [
            (Kind::Cube, [0.019, 0., 0.], [-0.01, 0., 0.]),
            (Kind::Plane, [0., -0.001, 0.], [0., -0.01, 0.]),
            (Kind::Balloon, [0.018, 0., 0.], [-0.01, 0., 0.]),
        ] {
            let at = |k| Effect::new(kind, 0.02, Tuning { stiffness_n_m: k, ..Tuning::default() }).step(p, v, 0.002).force;
            let a = at(150.); let b = at(300.);
            println!("{kind:?}: k150={a:?}, k300={b:?}");
            assert!((0..3).any(|i| (b[i] - 2. * a[i]).abs() > 0.01));
        }
    }
    #[test]
    fn spring_body_motion_ignores_stiffness_but_pen_reaction_scales() {
        let at = |k| Effect::new(Kind::Springs, 0.02, Tuning { stiffness_n_m: k, ..Tuning::default() }).step([0.021, 0., 0.], [0.; 3], 0.002);
        let a = at(150.); let b = at(300.);
        assert_eq!(a.bodies, b.bodies);
        assert!(norm(a.force) > 0.);
        for i in 0..3 { assert!((b.force[i] - 2. * a.force[i]).abs() < 1e-12); }
        println!("same bodies; pen reaction k150={:?}, k300={:?}", a.force, b.force);
    }
    #[test]
    fn finite_extreme_inputs_can_produce_nonfinite_effect_output() {
        let frame = Effect::new(Kind::Water, 0.02, Tuning::default()).step([0.; 3], [f64::MAX, 0., 0.], 0.002);
        assert!(frame.force.iter().any(|x| !x.is_finite()));
        println!("finite extreme Water velocity yields {:?}; callers still need guard rejection", frame.force);
    }
}
