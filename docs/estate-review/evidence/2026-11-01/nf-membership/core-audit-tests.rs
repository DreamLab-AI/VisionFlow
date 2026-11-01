    #[test]
    fn audit_new_row_keeps_duplicate_granted_cohorts() {
        let conn = db();
        grant(&conn, PK, &["home", "home"], 100, "admin");
        assert_eq!(cohorts(&conn, PK).unwrap(), ["home", "home"]);
    }

    #[test]
    fn audit_existing_row_appends_duplicate_new_cohorts() {
        let conn = db();
        grant(&conn, PK, &["home"], 100, "admin");
        grant(&conn, PK, &["zone2", "zone2"], 200, "admin");
        assert_eq!(cohorts(&conn, PK).unwrap(), ["home", "zone2", "zone2"]);
    }

    #[test]
    fn audit_null_in_legacy_array_suppresses_new_grant() {
        let conn = db();
        conn.execute("INSERT INTO whitelist(pubkey,cohorts,added_at) VALUES (?1, '[\"home\",null]', 1)", [PK]).unwrap();
        grant(&conn, PK, &["zone2"], 200, "admin");
        let raw: String = conn.query_row("SELECT cohorts FROM whitelist WHERE pubkey=?1", [PK], |r| r.get(0)).unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&raw).unwrap(), serde_json::json!(["home",null]));
        assert!(serde_json::from_str::<Vec<String>>(&raw).is_err());
    }
