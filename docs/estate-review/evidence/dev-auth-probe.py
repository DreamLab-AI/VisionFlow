#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,json,hashlib,re
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project'
def extract(path,name):
 s=(r/path).read_text();parts=[]
 for m in re.finditer(r'(?:pub )?fn '+name+r'\(',s):
  start=s.rfind('#[cfg(',0,m.start());op=s.index('{',m.end());depth=1;i=op+1
  while depth:
   depth+=(s[i]=='{')-(s[i]=='}');i+=1
  parts.append(s[start:i])
 return '\n'.join(parts)
source=extract('src/utils/auth.rs','dev_full_bypass_active')+'\n'+extract('src/main.rs','enforce_release_env_hygiene')+'\nfn main(){enforce_release_env_hygiene();println!("{}",dev_full_bypass_active());}\n'
rows=[]
with tempfile.TemporaryDirectory(prefix='estate-dev-auth-') as tmp:
 d=Path(tmp);(d/'probe.rs').write_text(source)
 for mode,flags in [('release',['-C','debug-assertions=no']),('release_dev_auth',['-C','debug-assertions=no','--cfg','feature="dev-auth"']),('debug',['-C','debug-assertions=yes'])]:
  c=subprocess.run(['rustc','--edition=2021',str(d/'probe.rs'),'-o',str(d/'probe')]+flags,capture_output=True,text=True);assert c.returncode==0,c.stderr
  for value in [None,'0','1']:
   env={} if value is None else {'VISIONCLAW_DEV_MODE':value}
   run=subprocess.run([str(d/'probe')],env=env,capture_output=True,text=True)
   expected=2 if mode=='release' and value is not None else 0
   assert run.returncode==expected
   if expected==0:assert run.stdout.strip()==str(value=='1' and mode!='release').lower()
   rows.append({'build':mode,'dev_mode':value,'exit_code':run.returncode,'bypass':run.stdout.strip() or None})
paths=['src/utils/auth.rs','src/main.rs','src/handlers/socket_flow_handler/filter_auth.rs','Cargo.toml','Dockerfile.unified','Dockerfile.production']
j={'date':'2026-09-04','scope':'Actual extracted helper and boot-hygiene functions with original cfg attributes; synthetic fresh environments and three rustc feature/debug combinations. No full Cargo/image build, listener, HTTP or headset test.','results':rows,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/dev-auth-probe.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(rows))
