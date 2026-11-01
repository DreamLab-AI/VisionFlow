
#[cfg(test)]
mod estate_audit {
    use super::*;
    use crate::sim::{SimHw, SimPlant};
    struct FailedShutdown(SimHw);
    impl Hardware for FailedShutdown {
        fn now(&self) -> Instant { self.0.now() }
        fn sample(&mut self) -> std::io::Result<Sample> { self.0.sample() }
        fn state(&self) -> &EncoderState { self.0.state() }
        fn guard(&self) -> &ForceGuard { self.0.guard() }
        fn register_reset(&mut self) -> Result<Registration, &'static str> { self.0.register_reset() }
        fn clear_registration(&mut self) { self.0.clear_registration() }
        fn set_limits(&mut self, l: ForceLimits) -> Result<(), &'static str> { self.0.set_limits(l) }
        fn enable_feedback(&mut self) -> std::io::Result<()> { self.0.enable_feedback() }
        fn disable_feedback(&mut self) -> std::io::Result<()> { Err(std::io::Error::other("audit: shutdown unconfirmed")) }
        fn feedback_enabled(&self) -> bool { self.0.feedback_enabled() }
        fn request(&mut self, d: Demand, t: Instant) -> Result<(), RequestError> { self.0.request(d, t) }
    }
    #[test]
    fn shutdown_failure_still_reports_output_off() {
        let root = std::env::temp_dir().join(format!("vt-audit-{}", evidence::new_session_id()));
        let mut e = Engine::new(root.clone(), true);
        e.connect(FailedShutdown(SimHw::new(SimPlant::default())));
        e.step(true, 0.).unwrap();
        e.hw.as_mut().unwrap().enable_feedback().unwrap();
        assert!(e.hw.as_ref().unwrap().feedback_enabled());
        e.fault("audit: shutdown unconfirmed");
        assert!(e.status().contains("Output is off; reconnect"));
        assert!(!e.connected());
        println!("fault status despite failed shutdown: {}", e.status());
        if root.exists() { std::fs::remove_dir_all(root).unwrap(); }
    }
    #[test]
    fn disconnect_drops_running_attempt_without_persisting_it() {
        let root = std::env::temp_dir().join(format!("vt-audit-{}", evidence::new_session_id()));
        let mut e = Engine::new(root.clone(), true);
        e.connect(SimHw::new(SimPlant::default()));
        e.step(true, 0.).unwrap();
        e.handle(Command::Start(StepRequest::Liveness { channel: 0 }));
        assert!(e.active().is_some(), "{}", e.status());
        for _ in 0..10 { e.step(true, 0.).unwrap(); }
        assert!(e.running.is_some());
        e.disconnect();
        assert!(e.running.is_none());
        assert!(e.done.is_empty());
        assert!(!root.exists());
        println!("disconnect discarded an active liveness attempt; no evidence directory created");
    }
    #[test]
    fn outputoff_token_is_reusable_after_a_simulator_is_armed() {
        let mut hw = SimHw::new(SimPlant::default());
        let token = OutputOff::confirm(hw.feedback_enabled()).unwrap();
        hw.enable_feedback().unwrap();
        assert!(hw.feedback_enabled());
        assert!(OutputOff::confirm(true).is_err());
        let root = std::env::temp_dir().join(format!("vt-audit-{}", evidence::new_session_id()));
        let p = evidence::create_session_dir(&token, &root, "test", "reused").unwrap();
        assert!(p.exists());
        println!("previous token accepted by evidence writer after simulator armed");
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn evidence_commit_follows_runtime_cwd_not_compiled_sources() {
        if let Ok(expected) = std::env::var("VT_AUDIT_CHILD_HEAD") {
            let actual = evidence::Software::current().git_commit;
            assert_eq!(actual, expected);
            assert_ne!(actual, "7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d");
            println!("archived visiontouch binary reports foreign checkout commit: {actual}");
            return;
        }
        let foreign = std::env::var("VT_AUDIT_FOREIGN_CWD").unwrap();
        let head = std::process::Command::new("git").args(["-C", &foreign, "rev-parse", "HEAD"]).output().unwrap();
        assert!(head.status.success());
        let head = String::from_utf8(head.stdout).unwrap();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "worker::estate_audit::evidence_commit_follows_runtime_cwd_not_compiled_sources", "--nocapture"])
            .current_dir(foreign).env("VT_AUDIT_CHILD_HEAD", head.trim()).status().unwrap();
        assert!(child.success());
    }
    #[test]
    fn one_return_mark_satisfies_repeatability_gate() {
        let mut checks = crate::registration::Checks::default();
        checks.capture(Mark::ReturnToReset, [0.; 3], true).unwrap();
        assert_eq!(checks.return_ok(), Some(true));
        println!("a single return mark satisfies return_ok without repeated physical returns");
    }
}
