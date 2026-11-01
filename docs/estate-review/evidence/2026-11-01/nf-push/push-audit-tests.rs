#[cfg(test)]
mod audit_push_boundaries {
    use super::*;
    use rusqlite::{params, Connection};
    fn db() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        for s in PUSH_SCHEMA_SQL { c.execute(s, []).unwrap(); }
        c.execute(SQL_UPSERT_SUBSCRIPTION, params!["device", "alice", "https://fcm.googleapis.com/send/test", "oldhash", 100]).unwrap();
        c
    }
    fn bump(c: &Connection) { c.execute(&bump_sql(Category::Dm), ["device"]).unwrap(); }
    fn pending(c: &Connection) -> i64 { c.query_row("SELECT pending_dm FROM push_subscriptions WHERE id='device'", [], |r| r.get(0)).unwrap() }
    fn drain_count(c: &Connection, n: i64) { c.execute(SQL_DRAIN_SUBTRACT, params!["device", n, 0, 0, 0]).unwrap(); }
    fn due(c: &Connection) -> usize {
        let mut q=c.prepare(SQL_PENDING_DUE).unwrap();
        q.query_map(params![880, 100, 30, 1000, 3600], |_| Ok(())).unwrap().count()
    }
    #[test]
    fn fast_drain_then_second_bump_then_mark_leaves_no_due_wakeup() {
        let c=db(); bump(&c);
        c.execute(SQL_CLAIM_PUSH, params!["device",100,95]).unwrap();
        // First push reaches the worker and is drained before send bookkeeping.
        drain_count(&c,1); bump(&c);
        // A second event is counted but throttled while the first is in flight.
        c.execute(SQL_MARK_PUSHED, params!["device",100]).unwrap();
        assert_eq!(pending(&c),1);
        assert_eq!(due(&c),0,"new undrained event is marked as already pushed");
    }
    #[test]
    fn overlapping_drains_double_subtract_a_later_bump() {
        let c=db(); bump(&c);
        let a=pending(&c);let b=pending(&c);assert_eq!((a,b),(1,1));
        drain_count(&c,a); bump(&c); drain_count(&c,b);
        assert_eq!(pending(&c),0,"the later event was included in neither response");
    }
    #[test]
    fn old_authorised_drain_can_subtract_new_owners_counter() {
        let c=db(); bump(&c);
        let (hash,n):(String,i64)=c.query_row(SQL_DRAIN_SELECT,["device"],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
        assert_eq!(hash,"oldhash"); // Authentication of the old request already passed.
        c.execute(SQL_UPSERT_SUBSCRIPTION,params!["device","bob","https://fcm.googleapis.com/send/test","newhash",101]).unwrap();
        bump(&c);drain_count(&c,n);
        assert_eq!(pending(&c),0,"mutation is not guarded by token or registration generation");
    }
    #[test]
    fn refresh_clears_pending_and_new_owner_inherits_dm_budget() {
        let c=db();bump(&c);
        c.execute("UPDATE push_subscriptions SET dm_window_start=100,dm_window_count=30,last_push_at=100 WHERE id='device'",[]).unwrap();
        c.execute(SQL_UPSERT_SUBSCRIPTION,params!["device","bob","https://fcm.googleapis.com/send/test","newhash",101]).unwrap();
        assert_eq!(pending(&c),0);
        assert_eq!(c.execute(SQL_TAKE_DM_BUDGET,params!["device",102,30,3600]).unwrap(),0);
    }
    #[test]
    fn queued_count_remains_due_after_preferences_disable_dm() {
        let c=db();bump(&c);
        c.execute("INSERT INTO push_prefs(pubkey,dms,quiet_start,quiet_end,updated_at) VALUES ('alice',0,0,1439,101)",[]).unwrap();
        assert_eq!(due(&c),1,"trailing due query does not consult current preferences");
    }
    #[test]
    fn late_success_overwrites_a_newer_claim_timestamp() {
        let c=db();bump(&c);
        assert_eq!(c.execute(SQL_CLAIM_PUSH,params!["device",100,95]).unwrap(),1);
        assert_eq!(c.execute(SQL_CLAIM_PUSH,params!["device",106,101]).unwrap(),1);
        c.execute(SQL_MARK_PUSHED,params!["device",100]).unwrap();
        assert_eq!(c.execute(SQL_CLAIM_PUSH,params!["device",107,102]).unwrap(),1,"older completion backdates newer push slot");
    }
}
