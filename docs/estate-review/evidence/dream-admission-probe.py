#!/usr/bin/env python3
from pathlib import Path
import subprocess,tempfile,json,hashlib
root=Path(__file__).resolve().parents[4]/'project/agentbox';p=root/'services/dream-engine/src/config.rs';s=p.read_text();start=s.index('    fn validate(&self)');a=s.index('{',start);i=a+1;depth=1
while depth:
 depth+=(s[i]=='{')-(s[i]=='}');i+=1
method=s[start:i]
code='''use std::collections::HashMap;
#[derive(Debug)] enum ConfigError { Validation(String) }
struct Slot { deep:String }
struct DreamConfig {repo:String,slots:Vec<Slot>,evaluator_entrypoints:HashMap<String,String>}
impl DreamConfig {'''+method+'''}
fn main(){
for (name,cmd,expected) in [("empty",None,true),("inline",Some("echo PASS"),true),("missing_script",Some("bash scripts/does-not-exist.sh"),true),("darwin_without_mode",Some("npx @metaharness/darwin"),false)] {
let mut evals=HashMap::new();if let Some(c)=cmd {evals.insert("unrelated-deep".into(),c.into());}
let cfg=DreamConfig{repo:"fixture/repo".into(),slots:vec![Slot{deep:"selected-deep".into()}],evaluator_entrypoints:evals};
let ok=cfg.validate().is_ok();assert_eq!(ok,expected);println!("{} {}",name,ok);
}}
'''
with tempfile.TemporaryDirectory(prefix='estate-dream-admission-') as td:
 t=Path(td);(t/'probe.rs').write_text(code);subprocess.run(['rustc','--edition=2021',str(t/'probe.rs'),'-o',str(t/'probe')],check=True,capture_output=True);out=subprocess.run([str(t/'probe')],check=True,capture_output=True,text=True).stdout
files=['services/dream-engine/src/config.rs','services/dream-engine/src/engine.rs','services/dream-engine/src/compile.rs']
d={'date':'2026-09-04','scope':'Unchanged validate method extracted into minimal typed harness; no evaluator command, SSH, schedule or model invocation.','cases':out.strip().splitlines(),'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}}
Path(__file__).with_suffix('.json').write_text(json.dumps(d,indent=2)+'\n');print(out)
