    #[test]
    fn audit_pager_cap_returns_partial_members_without_error() {
        let mut pager = WhitelistPager::new();
        for page in 0..500 {
            assert!(pager.next_query().is_some());
            let users: Vec<_> = (page * 100..(page + 1) * 100)
                .map(|i| serde_json::json!({"pubkey": pk(i), "cohorts": ["members"]}))
                .collect();
            pager.absorb(&serde_json::json!({"users":users,"total":50001}).to_string()).unwrap();
        }
        assert!(pager.next_query().is_none());
        let users = pager.into_users();
        assert_eq!(users.len(), 50000);
        assert_eq!(pending_registrations(&[registration(50000)], &users).len(), 1);
    }

    #[test]
    fn audit_deletion_between_offset_pages_skips_a_live_member() {
        let mut relay = FakeRelay::new();
        for i in 0..201 { relay.add(&pk(i), &["members"]); }
        let mut pager = WhitelistPager::new();
        pager.absorb(&relay.list(&pager.next_query().unwrap())).unwrap();
        relay.rows.retain(|r| r.0 != pk(200));
        while let Some(q) = pager.next_query() { pager.absorb(&relay.list(&q)).unwrap(); }
        let users = pager.into_users();
        assert!(relay.rows.iter().any(|r| r.0 == pk(100)));
        assert!(!users.iter().any(|r| r.pubkey == pk(100)));
        assert_eq!(pending_registrations(&[registration(100)], &users).len(), 1);
    }

    #[test]
    fn audit_case_fold_dedup_keeps_only_first_rows_cohorts() {
        let mut relay = FakeRelay::new();
        let key = pk(0xab);
        relay.add(&key, &["members"]);
        relay.add(&key.to_uppercase(), &["private"]);
        let users = fetch_all(&relay);
        assert_eq!(relay.rows.len(), 2);
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].pubkey, key.to_uppercase());
        assert_eq!(users[0].cohorts, ["private"]);
        assert!(pending_registrations(&[registration(0xab)], &users).is_empty());
    }
