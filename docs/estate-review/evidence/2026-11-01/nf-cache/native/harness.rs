
use std::{cell::{Cell,RefCell},collections::HashMap,rc::Rc,future::Future,task::{Context,Poll,Waker}};
#[path="read_cache.rs"] mod read_cache;
use read_cache::TtlCache;
fn run<T>(f:impl Future<Output=T>)->T { let mut f=std::pin::pin!(f); let mut cx=Context::from_waker(Waker::noop()); match f.as_mut().poll(&mut cx) {Poll::Ready(v)=>v,Poll::Pending=>panic!("fixture unexpectedly suspended")} }
mod auth { pub fn js_now_secs()->u64 {100} }
#[derive(Default)] struct Env { zone_calls:Cell<u32>,cohort_calls:Cell<u32>,device_calls:Cell<u32>,zone:RefCell<Option<String>> }
mod trust { use super::Env;
 pub async fn get_channel_zone(_: &str,e:&Env)->Option<String> {e.zone_calls.set(e.zone_calls.get()+1);e.zone.borrow().clone()}
 pub async fn get_viewer_cohorts(_: &str,e:&Env)->(Vec<String>,bool) {e.cohort_calls.set(e.cohort_calls.get()+1);(vec![],false)}
}
#[derive(Clone)] struct NostrEvent {kind:u64,pubkey:String}
#[derive(Clone)] struct NostrFilter;
#[derive(Clone,Default)] struct WebSocket(Rc<RefCell<Vec<u64>>>);
struct Session {authed_pubkey:Option<String>,ws:WebSocket,subscriptions:HashMap<String,Vec<NostrFilter>>}
struct ZoneConfig; impl ZoneConfig {fn load(_: &Env)->Self {Self}}
mod calendar_projection {pub fn is_projected_calendar_kind(_:u64)->bool {false}}
mod nostr_bbs_core {pub fn is_kanban_kind(_:u64)->bool {false}}
fn tag_value(_: &NostrEvent,_:&str)->Option<String>{None}
fn protected_recipients(_: &NostrEvent)->Option<Vec<String>>{None}
fn protected_read_permitted(_:u64,_:&str,_:&Vec<String>,_:Option<&str>,_:())->bool {true}
fn event_matches_filters(_: &NostrEvent,_:&Vec<NostrFilter>)->bool {true}
#[allow(dead_code)] enum ReadDecision {Deliver,DeliverAs(NostrEvent),Withhold}
type DeliveryCandidate=(Option<String>,WebSocket,Vec<(String,Vec<NostrFilter>)>);
struct NostrRelayDO {
 env:Env,channel_zone_cache:TtlCache<String>,cohort_cache:TtlCache<(Vec<String>,bool)>,device_owner_cache:TtlCache<Option<String>>,
 sessions:RefCell<HashMap<String,Session>>,auth_calls:Cell<u32>
}
impl NostrRelayDO {
 fn new()->Self {Self{env:Env::default(),channel_zone_cache:TtlCache::new(60),cohort_cache:TtlCache::new(60),device_owner_cache:TtlCache::new(60),sessions:RefCell::new(HashMap::new()),auth_calls:Cell::new(0)}}
 async fn device_owner(&self,_:&str)->Option<String>{self.env.device_calls.set(self.env.device_calls.get()+1);None}
 fn auth_mode(&self){}
 async fn resolve_viewer_context(&self,_:Option<String>){}
 async fn authorize_event(&self,_:&NostrEvent,_:&(),_:&ZoneConfig)->ReadDecision{self.auth_calls.set(self.auth_calls.get()+1);ReadDecision::Withhold}
 fn send_event(ws:&WebSocket,_:&str,event:&NostrEvent){ws.0.borrow_mut().push(event.kind)}
    pub(crate) async fn cached_channel_zone(&self, channel_id: &str) -> Option<String> {
        let now = auth::js_now_secs();
        if let Some(zone) = self.channel_zone_cache.get(channel_id, now) {
            return Some(zone);
        }
        let zone = trust::get_channel_zone(channel_id, &self.env).await?;
        self.channel_zone_cache
            .insert(channel_id, zone.clone(), now);
        Some(zone)
    }
    pub(crate) async fn cached_viewer_cohorts(&self, pubkey: &str) -> (Vec<String>, bool) {
        let now = auth::js_now_secs();
        if let Some(hit) = self.cohort_cache.get(pubkey, now) {
            return hit;
        }
        let fresh = trust::get_viewer_cohorts(pubkey, &self.env).await;
        self.cohort_cache.insert(pubkey, fresh.clone(), now);
        fresh
    }
    pub(crate) async fn cached_device_owner(&self, pubkey: &str) -> Option<String> {
        let now = auth::js_now_secs();
        if let Some(hit) = self.device_owner_cache.get(pubkey, now) {
            return hit;
        }
        let fresh = self.device_owner(pubkey).await;
        self.device_owner_cache.insert(pubkey, fresh.clone(), now);
        fresh
    }
    pub(crate) async fn broadcast_event(&self, event: &NostrEvent) {
        // NIP-59: kind-1059 (Sealed DMs) must only be delivered to the session
        // whose authenticated pubkey matches the event's `p` tag recipient.
        let kind_1059_recipient: Option<String> = if event.kind == 1059 {
            tag_value(event, "p")
        } else {
            None
        };

        // Only zone-scoped kinds (kind-40 defs, kind-42 content, NIP-52 calendar,
        // kanban boards/cards) require the per-recipient read gate. Everything
        // else broadcasts to every matching subscriber as before, so the common
        // hot path adds no async work.
        let needs_gate = event.kind == 40
            || event.kind == 42
            || calendar_projection::is_projected_calendar_kind(event.kind)
            || nostr_bbs_core::is_kanban_kind(event.kind);

        // Snapshot delivery candidates so we never hold the `sessions` borrow
        // across the async zone/cohort lookups in the read gate below.
        let candidates: Vec<DeliveryCandidate> = {
            let sessions = self.sessions.borrow();
            sessions
                .values()
                .map(|s| {
                    (
                        s.authed_pubkey.clone(),
                        s.ws.clone(),
                        s.subscriptions
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    )
                })
                .collect()
        };

        let zones = if needs_gate {
            Some(ZoneConfig::load(&self.env))
        } else {
            None
        };

        // Protected kinds ({4,13,14,1059,30910-30916}) are decided per event AND per
        // session, here as on REQ and COUNT. The REQ gates look at the kinds a
        // filter names, and a filter that omits `kinds` names none, so it is
        // stored as a live subscription and would otherwise be handed every
        // later moderation event (plaintext reasons and targets) and DM envelope
        // (recipient, timing) with no check at all.
        let protected_recipients = protected_recipients(event);
        let auth_mode = self.auth_mode();

        for (authed_pubkey, ws, subscriptions) in &candidates {
            if let Some(recipients) = &protected_recipients {
                if !protected_read_permitted(
                    event.kind,
                    &event.pubkey,
                    recipients,
                    authed_pubkey.as_deref(),
                    auth_mode,
                ) {
                    continue;
                }
            }

            // For kind-1059 events, skip sessions not authenticated as the
            // intended recipient.
            if let Some(ref recipient) = kind_1059_recipient {
                match authed_pubkey {
                    Some(pk) if pk == recipient => {}
                    _ => continue,
                }
            }

            // Apply the SAME per-recipient read gate REQ uses, so the live-push
            // path cannot leak zone-private channel content or non-public calendar
            // events to a subscriber who could not have read them via an initial
            // REQ. `deliver` is the (possibly calendar-projected) event to send,
            // or None to withhold; for non-zone-scoped events the gate is skipped.
            let deliver: Option<NostrEvent> = if let Some(zones) = &zones {
                if !subscriptions
                    .iter()
                    .any(|(_, filters)| event_matches_filters(event, filters))
                {
                    continue;
                }
                let ctx = self.resolve_viewer_context(authed_pubkey.clone()).await;
                match self.authorize_event(event, &ctx, zones).await {
                    ReadDecision::Deliver => Some(event.clone()),
                    ReadDecision::DeliverAs(out) => Some(out),
                    ReadDecision::Withhold => continue,
                }
            } else {
                None
            };

            for (sub_id, filters) in subscriptions {
                if event_matches_filters(event, filters) {
                    match &deliver {
                        Some(out) => Self::send_event(ws, sub_id, out),
                        None => Self::send_event(ws, sub_id, event),
                    }
                }
            }
        }
    }
}

#[test] fn unbound_channel_cost_scales_with_calls() {let d=NostrRelayDO::new(); for _ in 0..50 {assert_eq!(run(d.cached_channel_zone("c")),None);} assert_eq!(d.env.zone_calls.get(),50);}
#[test] fn bound_channel_warms_once() {let d=NostrRelayDO::new();*d.env.zone.borrow_mut()=Some("private".into());for _ in 0..50 {assert_eq!(run(d.cached_channel_zone("c")),Some("private".into()));}assert_eq!(d.env.zone_calls.get(),1);}
#[test] fn channel_cardinality_costs_more_than_three_misses() {let d=NostrRelayDO::new();*d.env.zone.borrow_mut()=Some("private".into());for i in 0..50 {run(d.cached_channel_zone(&i.to_string()));}assert_eq!(d.env.zone_calls.get(),50);}
#[test] fn negative_viewer_lookups_are_cached() {let d=NostrRelayDO::new();for _ in 0..50 {assert_eq!(run(d.cached_viewer_cohorts("p")),(vec![],false));assert_eq!(run(d.cached_device_owner("p")),None);}assert_eq!((d.env.cohort_calls.get(),d.env.device_calls.get()),(1,1));}
fn delivery(kind:u64)->(u32,Vec<u64>){let d=NostrRelayDO::new();let ws=WebSocket::default();d.sessions.borrow_mut().insert("s".into(),Session{authed_pubkey:Some("nonmember".into()),ws:ws.clone(),subscriptions:HashMap::from([("sub".into(),vec![NostrFilter])])});run(d.broadcast_event(&NostrEvent{kind,pubkey:"author".into()}));let sent=ws.0.borrow().clone();(d.auth_calls.get(),sent)}
#[test] fn kind41_bypasses_denied_viewer_gate() {assert_eq!(delivery(41),(0,vec![41]));}
#[test] fn kind40_consults_denied_viewer_gate() {assert_eq!(delivery(40),(1,vec![]));}
#[test] fn kind42_consults_denied_viewer_gate() {assert_eq!(delivery(42),(1,vec![]));}
