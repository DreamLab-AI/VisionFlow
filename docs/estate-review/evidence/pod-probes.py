#!/usr/bin/env python3
"""Probe actual native pod helpers from a temporary crate, without services."""
import pathlib,tempfile,subprocess,json,os
W=pathlib.Path(__file__).resolve().parents[4]
with tempfile.TemporaryDirectory(prefix='estate-pod-') as tmp:
 p=pathlib.Path(tmp);(p/'src').mkdir()
 (p/'Cargo.toml').write_text('[package]\nname="estate-pod-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nsolid-pod-rs={path='+json.dumps(str(W/'solid-pod-rs/crates/solid-pod-rs'))+',default-features=false,features=["memory-backend","nip98-replay"]}\ntokio={version="1",features=["macros","rt"]}\n')
 (p/'src/main.rs').write_text(r'''
use solid_pod_rs::{storage::{memory::MemoryBackend,Storage},wac::{AclResolver,StorageAclResolver,evaluate_access,AccessMode},auth::replay::{Nip98ReplayCache,ReplayStore}};
use std::{sync::Arc,time::Duration};
#[tokio::main(flavor="current_thread")]
async fn main(){
 let s=Arc::new(MemoryBackend::new());
 let root=r#"{"@graph":[{"acl:agent":{"@id":"did:nostr:alice"},"acl:default":{"@id":"/"},"acl:mode":[{"@id":"acl:Read"}]}]}"#;
 s.put("/.acl",root.as_bytes().to_vec().into(),"application/ld+json").await.unwrap();
 s.put("/secret.acl",b"not valid JSON".to_vec().into(),"application/ld+json").await.unwrap();
 let doc=StorageAclResolver::new(s).find_effective_acl("/secret").await.unwrap();
 println!("malformed_specific_acl_broader_grant={}",evaluate_access(doc.as_ref(),Some("did:nostr:alice"),"/secret",AccessMode::Read,None));
 let cache=Nip98ReplayCache::with_config(Duration::from_secs(240),1);
 cache.check_and_record("a").await.unwrap();
 println!("same_id_rejected={}",cache.check_and_record("a").await.is_err());
 cache.check_and_record("b").await.unwrap();
 println!("evicted_id_accepted_within_ttl={}",cache.check_and_record("a").await.is_ok());
}
''')
 env=os.environ.copy();env['CARGO_TARGET_DIR']=str(W/'solid-pod-rs/target')
 r=subprocess.run(['cargo','run','--quiet','--offline','--manifest-path',str(p/'Cargo.toml')],capture_output=True,text=True,env=env)
 print(json.dumps({'command':'temporary cargo run --offline against local native helpers','exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2))
