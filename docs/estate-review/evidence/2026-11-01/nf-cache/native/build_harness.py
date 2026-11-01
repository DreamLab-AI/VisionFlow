from pathlib import Path
import hashlib,json
src=Path('/tmp/vf-nf-cache-artifacts/source/crates/nostr-bbs-relay-worker/src')
out=Path(__file__).parent
extracted=[]
def method(file,name):
 text=(src/file).read_text(); start=text.index('    pub(crate) async fn '+name); end=start; depth=0; seen=False
 for end in range(start,len(text)):
  if text[end]=='{': depth+=1; seen=True
  if text[end]=='}':
   depth-=1
   if seen and depth==0:break
 body=text[start:end+1];extracted.append({'path':file,'method':name,'start_line':text[:start].count('\n')+1,'sha256':hashlib.sha256(body.encode()).hexdigest()});return body
cache=(src/'relay_do/read_cache.rs').read_text()
cache+='''
#[cfg(test)]
mod audit_boundaries {
 use super::*;
 #[test] fn expired_entries_are_retained() {
  let c=TtlCache::new(60);
  for i in 0..10_000 { c.insert(&i.to_string(), i, 0); }
  for i in 0..10_000 { assert_eq!(c.get(&i.to_string(), 86_400), None); }
  assert_eq!(c.entries.borrow().len(),10_000);
 }
 #[test] fn idle_pending_reads_have_no_deadline() {
  let l=ActivityLedger::new(); assert_eq!(l.record_read("a",1,100),Some(Flush{reads:1}));
  assert_eq!(l.record_read("a",49,101),None);
  assert_eq!(l.record_read("b",1,86_400),Some(Flush{reads:1}));
  assert_eq!(l.pending_reads.borrow().get("a"),Some(&49));
  assert_eq!(l.record_write("a",86_401),Some(Flush{reads:49}));
 }
 #[test] fn discarded_flush_is_not_retried() {
  let l=ActivityLedger::new(); let unpersisted=l.record_read("a",30,100); assert_eq!(unpersisted,Some(Flush{reads:30}));
  assert_eq!(l.record_read("a",1,101),None);
  assert_eq!(l.record_write("a",400),Some(Flush{reads:1}));
 }
 #[test] fn fifty_reads_can_flush_every_request() {
  let l=ActivityLedger::new();
  for _ in 0..20 { assert_eq!(l.record_read("a",50,100),Some(Flush{reads:50})); }
 }
 #[test] fn_reconstruction_placeholder() {}
}
'''.replace(' #[test] fn_reconstruction_placeholder() {}',''' #[test] fn reconstruction_loses_pending_reads() {
 let l=ActivityLedger::new(); l.record_read("a",1,100); assert_eq!(l.record_read("a",49,101),None);
 drop(l); let replacement=ActivityLedger::new(); assert_eq!(replacement.record_read("a",1,102),Some(Flush{reads:1}));
 }''')
(out/'read_cache.rs').write_text(cache)
common='''
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
'''
for name in ['cached_channel_zone','cached_viewer_cohorts','cached_device_owner']:
 common+=method('relay_do/nip_handlers.rs',name)+'\n'
common+=method('relay_do/broadcast.rs','broadcast_event')+'\n}\n'
common+='''
#[test] fn unbound_channel_cost_scales_with_calls() {let d=NostrRelayDO::new(); for _ in 0..50 {assert_eq!(run(d.cached_channel_zone("c")),None);} assert_eq!(d.env.zone_calls.get(),50);}
#[test] fn bound_channel_warms_once() {let d=NostrRelayDO::new();*d.env.zone.borrow_mut()=Some("private".into());for _ in 0..50 {assert_eq!(run(d.cached_channel_zone("c")),Some("private".into()));}assert_eq!(d.env.zone_calls.get(),1);}
#[test] fn channel_cardinality_costs_more_than_three_misses() {let d=NostrRelayDO::new();*d.env.zone.borrow_mut()=Some("private".into());for i in 0..50 {run(d.cached_channel_zone(&i.to_string()));}assert_eq!(d.env.zone_calls.get(),50);}
#[test] fn negative_viewer_lookups_are_cached() {let d=NostrRelayDO::new();for _ in 0..50 {assert_eq!(run(d.cached_viewer_cohorts("p")),(vec![],false));assert_eq!(run(d.cached_device_owner("p")),None);}assert_eq!((d.env.cohort_calls.get(),d.env.device_calls.get()),(1,1));}
fn delivery(kind:u64)->(u32,Vec<u64>){let d=NostrRelayDO::new();let ws=WebSocket::default();d.sessions.borrow_mut().insert("s".into(),Session{authed_pubkey:Some("nonmember".into()),ws:ws.clone(),subscriptions:HashMap::from([("sub".into(),vec![NostrFilter])])});run(d.broadcast_event(&NostrEvent{kind,pubkey:"author".into()}));let sent=ws.0.borrow().clone();(d.auth_calls.get(),sent)}
#[test] fn kind41_bypasses_denied_viewer_gate() {assert_eq!(delivery(41),(0,vec![41]));}
#[test] fn kind40_consults_denied_viewer_gate() {assert_eq!(delivery(40),(1,vec![]));}
#[test] fn kind42_consults_denied_viewer_gate() {assert_eq!(delivery(42),(1,vec![]));}
'''
(out/'harness.rs').write_text(common)
(out/'extracted-methods.json').write_text(json.dumps(extracted,indent=2))
