#!/usr/bin/env python3
from pathlib import Path
import re,tempfile,subprocess,hashlib,json
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project';p=r/'src/utils/nip98.rs';s=p.read_text()
start=s.index('fn claim_in(');end=s.index('\n}',start)+2;fn=s[start:end]
constants='\n'.join(re.findall(r'^const (?:TOKEN_MAX_AGE_SECONDS|REPLAY_CACHE_TTL|REPLAY_CACHE_PRUNE_THRESHOLD):[^\n]+',s,re.M))
source='use std::collections::HashMap; use std::time::{Instant,Duration};\n#[derive(Debug,PartialEq)] enum Nip98ValidationError {TokenReplayed,ReplayCacheFull}\n'+constants+'\n'+fn+'''
fn main(){let now=Instant::now();let mut map=HashMap::new();assert_eq!(claim_in(&mut map,"a",now,1),Ok(()));assert_eq!(claim_in(&mut map,"a",now,1),Err(Nip98ValidationError::TokenReplayed));assert_eq!(claim_in(&mut map,"b",now,1),Err(Nip98ValidationError::ReplayCacheFull));assert!(map.contains_key("a"));assert_eq!(claim_in(&mut map,"a",now+REPLAY_CACHE_TTL-Duration::from_nanos(1),1),Err(Nip98ValidationError::TokenReplayed));assert_eq!(claim_in(&mut map,"a",now+REPLAY_CACHE_TTL,1),Ok(()));let mut restarted=HashMap::new();assert_eq!(claim_in(&mut restarted,"a",now,1),Ok(()));println!("six claim assertions passed; live entry retained at capacity");}
'''
with tempfile.TemporaryDirectory(prefix='estate-replay-') as tmp:
 d=Path(tmp);(d/'probe.rs').write_text(source);c=subprocess.run(['rustc','--edition=2021',str(d/'probe.rs'),'-o',str(d/'probe')],capture_output=True,text=True);assert c.returncode==0,c.stderr;run=subprocess.run([str(d/'probe')],capture_output=True,text=True);assert run.returncode==0,run.stderr
j={'date':'2026-09-04','scope':'Actual extracted claim_in and constants; synthetic maps/monotonic times and minimal error enum. No signature validator, mutex race, HTTP route or deployment exercised.','stdout':run.stdout.strip(),'compile_exit':c.returncode,'run_exit':run.returncode,'source_sha256':{x:hashlib.sha256((r/x).read_bytes()).hexdigest() for x in ['src/utils/nip98.rs','src/services/nostr_service.rs','src/handlers/solid_proxy_handler.rs']}}
(b/'evidence/replay-cache-probe.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
