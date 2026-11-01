from pathlib import Path
import re,json,hashlib,subprocess
import argparse
parser=argparse.ArgumentParser(description='Prepare native audit fixtures from a clean forum git archive')
parser.add_argument('--kit',required=True);parser.add_argument('--out',required=True)
args=parser.parse_args();kit=Path(args.kit).resolve();ev=Path(args.out).resolve()
assert not (kit/'.git').exists(), 'Use a disposable git archive, not a working checkout'
ev.mkdir(parents=True,exist_ok=True)
client=(kit/'crates/nostr-bbs-forum-client/src/utils/devices.rs').read_text()
exprs=re.findall(r'let body = (serde_json::json!\(\{ "device_pubkey": device_pubkey, "label": label \}\)\.to_string\(\));',client);assert len(exprs)==2
new='\n#[cfg(test)]\nmod audit_client_contract {\n use super::*;\n'
for i,expr in enumerate(exprs):
 new+=f''' #[test]
 fn client_registration_{i}_omits_required_proof() {{
  let device_pubkey = "11".repeat(32); let label = "audit";
  let body = {expr};
  let parsed: RegisterDeviceBody = serde_json::from_str(&body).unwrap();
  assert!(normalize_register(&parsed).is_ok());
  assert!(parsed.device_proof.is_none(), "current producer does not send the proof required by handle_register");
 }}
'''

# Retain the actual client row type, error enum and parser, without its browser
# imports. Exercise them against serialization of the actual server DeviceRow.
def client_item(start):
 begin=client.index(start);op=client.index('{',begin);depth=1;end=op+1
 while depth:
  depth+=(client[end]=='{')-(client[end]=='}');end+=1
 return client[begin:end]
new += "\n" + client_item('#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]') + "\n"
new += client_item('#[derive(Debug, thiserror::Error)]') + "\n"
new += client_item('fn parse_device_list(') + "\n"
new += r'''
 #[test] fn nonempty_server_list_is_rejected_by_client() {
  let row = DeviceRow {device_pubkey: "11".repeat(32),owner_pubkey: "22".repeat(32),label: None,created_at: 100.0,revoked: 0.0};
  let response = serde_json::json!({"devices": vec![row]}).to_string();
  let err = parse_device_list(&response).unwrap_err().to_string();
  assert!(err.contains("floating point"), "{err}");
 }
 #[test] fn empty_server_list_is_accepted_control() {
  let rows: Vec<DeviceRow> = vec![];
  let response = serde_json::json!({"devices": rows}).to_string();
  assert!(parse_device_list(&response).unwrap().is_empty());
 }
 #[test] fn integer_created_at_still_rejects_float_revoked() {
  let row = DeviceRow {device_pubkey: "11".repeat(32),owner_pubkey: "22".repeat(32),label: None,created_at: 100.0,revoked: 0.0};
  let mut response = serde_json::json!({"devices": vec![row]});
  response["devices"][0]["created_at"] = serde_json::json!(100);
  assert!(parse_device_list(&response.to_string()).is_err());
  response["devices"][0]["revoked"] = serde_json::json!(0);
  assert_eq!(parse_device_list(&response.to_string()).unwrap().len(),1);
 }
'''
new+='}\n';(ev/'audit-client-contract.rs').write_text(new)
p=kit/'crates/nostr-bbs-auth-worker/src/devices.rs';assert 'mod audit_client_contract' not in p.read_text(), 'Use a clean source archive';p.write_text(p.read_text()+new)
# Extract unchanged production functions by brace balance. These bodies contain
# no braces in strings/comments; assert source spans and retain their hashes.
src=(kit/'crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs').read_text()
def extract(name):
 m=re.search(r'^    pub\(crate\) async fn '+name+r'\b|^pub(?:\(crate\))? fn '+name+r'\b',src,re.M);assert m,name
 start=m.start();op=src.index('{',start);depth=1;end=op+1
 while depth:
  depth+=(src[end]=='{')-(src[end]=='}');end+=1
 return src[start:end]
methods=['cached_device_owner','effective_pubkey','resolve_viewer_context'];free=['effective_principal','zone_read_permitted']
fixture=r'''
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
'''+ '\n'.join(extract(n) for n in methods)+'\n}\n'+'\n'.join(extract(n) for n in free)+r'''
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
'''
native=ev/'native';native.mkdir(exist_ok=True);(native/'harness.rs').write_text(fixture);(native/'read_cache.rs').write_bytes((kit/'crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs').read_bytes())
(native/'extracted-methods.json').write_text(json.dumps([dict(name=n,sha256=hashlib.sha256(extract(n).encode()).hexdigest(),start_line=src[:src.index(extract(n))].count('\n')+1) for n in methods+free],indent=2)+'\n')
print('Prepared five producer-consumer contract tests and seven cache/authority tests')
