#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,json,hashlib
b=Path(__file__).resolve().parents[1];w=b.parents[2];pk='1'*64
inputs={'texts':['','hello','é','e\u0301','{"a":1,"b":2}'],'urns':[f'urn:agentbox:agent:{pk}:fixture',f'urn:agentbox:activity:{pk}:fixture',f'urn:agentbox:thing:{pk}:fixture',f'urn:agentbox:bead:{pk}:sha256-12-0123456789ab',f'urn:agentbox:memory:{pk}:fixture'],'addresses':['sha256-12-0123456789ab','sha256-12-','sha256-12-nothex','sha256-12-0123456789ABCDEF']}
with tempfile.TemporaryDirectory(prefix='estate-uri-probe-') as td:
 p=Path(td);(p/'src').mkdir();(p/'Cargo.toml').write_text('[package]\nname="estate-uri-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nsha2="0.10"\nserde_json="1"\n')
 source='#[path='+json.dumps(str(w/'project/src/uri/mod.rs'))+'] mod uri;\n'+'''fn main(){let v:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap()).unwrap();let hashes:Vec<_>=v["texts"].as_array().unwrap().iter().map(|x|uri::content_address(x.as_str().unwrap())).collect();let crossings:Vec<_>=v["urns"].as_array().unwrap().iter().map(|x|uri::cross_from_agentbox(x.as_str().unwrap()).map(|c|c.visionclaw_id)).collect();let pk="1".repeat(64);let addresses:Vec<_>=v["addresses"].as_array().unwrap().iter().map(|x|{let a=x.as_str().unwrap();serde_json::json!({"address":a,"constructor_accepts":uri::kg_with_address(&pk,a).is_ok(),"parser_accepts":uri::parse(&format!("urn:visionclaw:kg:{pk}:{a}")).is_ok()})}).collect();println!("{}",serde_json::json!({"hashes":hashes,"crossings":crossings,"addresses":addresses}));}'''
 (p/'src/main.rs').write_text(source);(p/'input.json').write_text(json.dumps(inputs))
 run=subprocess.run(['cargo','run','--offline','--quiet','--manifest-path',str(p/'Cargo.toml'),'--',str(p/'input.json')],capture_output=True,text=True);assert run.returncode==0,run.stderr
 rust=json.loads(run.stdout)
 js="const b=require(process.argv[1]);const x=JSON.parse(process.argv[2]);console.log(JSON.stringify({hashes:x.texts.map(b.sha12),crossings:x.urns.map(u=>b.toVisionclaw(u,{onDrop:()=>{}})?.visionclaw_id||null)}));"
 node=subprocess.run(['node','-e',js,str(w/'project/agentbox/management-api/lib/bc20-provenance-bridge.js'),json.dumps(inputs)],capture_output=True,text=True);assert node.returncode==0,node.stderr
 agent=json.loads(node.stdout)
assert rust['hashes']==agent['hashes']
assert rust['crossings'][3] is None and agent['crossings'][3] is not None
assert all(x['constructor_accepts'] and x['parser_accepts'] for x in rust['addresses'])
paths=['project/src/uri/mod.rs','project/agentbox/management-api/lib/bc20-provenance-bridge.js','project/agentbox/management-api/lib/uris.js']
j={'date':'2026-09-04','scope':'actual Rust URI module in temporary offline crate and actual JS bridge; synthetic identifiers; no server or persistence','inputs':inputs,'rust':rust,'agentbox':agent,'source_sha256':{p:hashlib.sha256((w/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/federation-identity-probe.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
