    // Appended inside the existing test module in a disposable source archive.
    // Uses the production engine and its existing in-memory storage adapter.
    fn audit_held_prefix() -> Vec<WhitelistTrustRow> {
        let mut rows = Vec::new();
        for name in ["a", "b", "c"] {
            let mut r = idle_row(name, 2, stale_ts());
            r.days_active = 14;
            r.posts_read = 50;
            r.posts_created = 10;
            rows.push(r);
        }
        rows.push(idle_row("z", 1, stale_ts()));
        rows
    }

    #[test]
    fn audit_held_prefix_starves_tail_across_repeated_ticks() {
        let store = FakeStore::new(audit_held_prefix());
        for _ in 0..3 {
            let r = block_on(run_demotion_sweep(&store, &thresholds(), NOW, 2, 3));
            assert_eq!((r.scanned, r.held, r.demoted, r.failed), (3, 3, 0, 0));
            assert!(r.truncated && r.is_balanced());
            assert_eq!(r.resume_pubkey.as_deref(), Some("c"));
            assert_eq!(store.level_of("z"), 1);
        }
        assert!(store.commit_attempts.borrow().is_empty());
    }

    #[test]
    fn audit_larger_budget_reaches_tail_after_same_held_prefix() {
        let store = FakeStore::new(audit_held_prefix());
        let r = block_on(run_demotion_sweep(&store, &thresholds(), NOW, 2, 5));
        assert_eq!((r.scanned, r.held, r.demoted), (4, 3, 1));
        assert!(!r.truncated && r.is_balanced());
        assert_eq!(store.level_of("z"), 0);
    }

    #[test]
    fn audit_failed_prefix_also_starves_tail_across_ticks() {
        let rows = ["a", "b", "c", "z"].into_iter()
            .map(|name| idle_row(name, 1, stale_ts())).collect();
        let store = FakeStore::new(rows)
            .fail_commit_for("a", "persistent failure")
            .fail_commit_for("b", "persistent failure")
            .fail_commit_for("c", "persistent failure");
        for _ in 0..3 {
            let r = block_on(run_demotion_sweep(&store, &thresholds(), NOW, 2, 3));
            assert_eq!((r.scanned, r.failed, r.demoted), (3, 3, 0));
            assert!(r.truncated && r.is_balanced());
            assert_eq!(store.level_of("z"), 1);
        }
        assert!(!store.commit_attempts.borrow().iter().any(|p| p == "z"));
    }

    #[test]
    fn audit_earned_counters_do_not_decay_with_time_alone() {
        let rows = audit_held_prefix();
        for years in [1, 10] {
            let now = NOW + years * 365 * 86400;
            assert!(matches!(decide_demotion(&rows[0], &thresholds(), now),
                DemotionDecision::Hold(crate::trust::HoldReason::ClearsHysteresis)));
        }
    }
