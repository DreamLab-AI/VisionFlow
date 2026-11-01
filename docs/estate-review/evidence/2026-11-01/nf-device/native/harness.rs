
use std::{cell::{Cell,RefCell},collections::{HashMap,HashSet},future::Future,task::{Context,Poll,Waker}};
#[path="read_cache.rs"] mod read_cache;
use read_cache::TtlCache;
fn run<T>(f:impl Future<Output=T>)->T { let mut f=std::pin::pin!(f); match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {Poll::Ready(v)=>v,Poll::Pending=>panic!("unexpected suspension")} }
mod auth {thread_local! {static NOW: std::cell::Cell<u64> = const {std::cell::Cell::new(100)};} pub fn js_now_secs()->u64 {NOW.with(|n|n.get())} pub fn set(n:u64){NOW.with(|v|v.set(n))}}
#[derive(Default)] struct Env {rows:RefCell<HashMap<String,String>>,calls:Cell<u32>,fail:Cell<bool>,admins:RefCell<HashSet<String>>}
struct AdminCache;impl AdminCache {async fn is_admin(&self,pk:&str,env:&Env)->bool {env.admins.borrow().contains(pk)}}
struct ViewerContext {session_pubkey:Option<String>,access_pubkey:Option<String>,is_admin:bool,viewer_cohorts:Vec<String>,viewer_is_admin:bool}
struct ZoneConfig;impl ZoneConfig {fn is_public_read(&self,_:&str)->bool{false}fn cohorts_can_read(&self,_:&str,_:&[String])->bool{false}}
struct NostrRelayDO {env:Env,device_owner_cache:TtlCache<Option<String>>,enabled:Cell<bool>,admin_cache:AdminCache}
impl NostrRelayDO {
 fn new()->Self {auth::set(100);Self{env:Env::default(),device_owner_cache:TtlCache::new(60),enabled:Cell::new(true),admin_cache:AdminCache}}
 fn device_keys_enabled(&self)->bool {self.enabled.get()}
 async fn device_owner(&self,pk:&str)->Option<String>{self.env.calls.set(self.env.calls.get()+1);if self.env.fail.get(){None}else{self.env.rows.borrow().get(pk).cloned()}}
 async fn cached_viewer_cohorts(&self,_:&str)->(Vec<String>,bool){(vec![],false)}
 fn link(&self,d:&str,o:&str){self.env.rows.borrow_mut().insert(d.into(),o.into());}
    pub(crate) async fn cached_device_owner(&self, pubkey: &str) -> Option<String> {
        let now = auth::js_now_secs();
        if let Some(hit) = self.device_owner_cache.get(pubkey, now) {
            return hit;
        }
        let fresh = self.device_owner(pubkey).await;
        self.device_owner_cache.insert(pubkey, fresh.clone(), now);
        fresh
    }
    pub(crate) async fn effective_pubkey(&self, pubkey: &str) -> String {
        if !self.device_keys_enabled() {
            return pubkey.to_string();
        }
        let owner = self.cached_device_owner(pubkey).await;
        effective_principal(pubkey, owner.as_deref(), true)
    }
    pub(crate) async fn resolve_viewer_context(
        &self,
        session_pubkey: Option<String>,
    ) -> ViewerContext {
        let access_pubkey: Option<String> = match &session_pubkey {
            Some(pk) => Some(self.effective_pubkey(pk).await),
            None => None,
        };
        let is_admin = match &access_pubkey {
            Some(pk) => self.admin_cache.is_admin(pk, &self.env).await,
            None => false,
        };
        let (viewer_cohorts, cohort_admin) = match &access_pubkey {
            Some(pk) => self.cached_viewer_cohorts(pk).await,
            None => (Vec::new(), false),
        };
        let viewer_is_admin = is_admin || cohort_admin;
        ViewerContext {
            session_pubkey,
            access_pubkey,
            is_admin,
            viewer_cohorts,
            viewer_is_admin,
        }
    }
}
pub fn effective_principal(pubkey: &str, device_owner: Option<&str>, enabled: bool) -> String {
    if enabled {
        if let Some(owner) = device_owner {
            return owner.to_string();
        }
    }
    pubkey.to_string()
}
pub(crate) fn zone_read_permitted(zones: &ZoneConfig, zone: &str, ctx: &ViewerContext) -> bool {
    zones.is_public_read(zone)
        || ctx.viewer_is_admin
        || zones.cohorts_can_read(zone, &ctx.viewer_cohorts)
}
#[cfg(test)] mod audit {
 use super::*;
 #[test] fn revocation_remains_mapped_until_cached_entry_expires(){let d=NostrRelayDO::new();d.link("device","owner");assert_eq!(run(d.effective_pubkey("device")),"owner");d.env.rows.borrow_mut().remove("device");auth::set(159);assert_eq!(run(d.effective_pubkey("device")),"owner");auth::set(160);assert_eq!(run(d.effective_pubkey("device")),"device");assert_eq!(d.env.calls.get(),2);}
 #[test] fn newly_registered_device_waits_for_negative_entry_expiry(){let d=NostrRelayDO::new();assert_eq!(run(d.effective_pubkey("device")),"device");d.link("device","owner");auth::set(159);assert_eq!(run(d.effective_pubkey("device")),"device");auth::set(160);assert_eq!(run(d.effective_pubkey("device")),"owner");}
 #[test] fn transient_lookup_failure_is_negative_cached(){let d=NostrRelayDO::new();d.link("device","owner");d.env.fail.set(true);assert_eq!(run(d.effective_pubkey("device")),"device");d.env.fail.set(false);auth::set(159);assert_eq!(run(d.effective_pubkey("device")),"device");auth::set(160);assert_eq!(run(d.effective_pubkey("device")),"owner");}
 #[test] fn gate_off_skips_stale_positive_cache_control(){let d=NostrRelayDO::new();d.link("device","owner");assert_eq!(run(d.effective_pubkey("device")),"owner");d.enabled.set(false);assert_eq!(run(d.effective_pubkey("device")),"device");assert_eq!(d.env.calls.get(),1);}
 #[test] fn context_snapshot_outlives_revocation_and_ttl(){let d=NostrRelayDO::new();d.link("device","owner");d.env.admins.borrow_mut().insert("owner".into());let ctx=run(d.resolve_viewer_context(Some("device".into())));d.env.rows.borrow_mut().remove("device");auth::set(500);assert_eq!(run(d.effective_pubkey("device")),"device");assert!(zone_read_permitted(&ZoneConfig,"private",&ctx));assert_eq!(ctx.session_pubkey.as_deref(),Some("device"));}
 #[test] fn owner_admin_is_inherited_for_zone_reads(){let d=NostrRelayDO::new();d.link("device","owner");d.env.admins.borrow_mut().insert("owner".into());let ctx=run(d.resolve_viewer_context(Some("device".into())));assert!(ctx.is_admin && ctx.viewer_is_admin);assert!(ctx.viewer_cohorts.is_empty());assert!(zone_read_permitted(&ZoneConfig,"private",&ctx));}
 #[test] fn device_owner_chain_is_only_one_hop(){let d=NostrRelayDO::new();d.link("child","device");d.link("device","owner");assert_eq!(run(d.effective_pubkey("child")),"device");}
}
