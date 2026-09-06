#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,os,subprocess,tempfile
root=Path(__file__).resolve().parents[4]/'project';source=root/'scripts/rust-backend-wrapper.sh';s=source.read_text()
block=s[s.index('    NEEDS_BUILD=true'):s.index('    if [ "$NEEDS_BUILD" = "true" ]; then')]
rows=[]
with tempfile.TemporaryDirectory(prefix='estate-dev-input-') as td:
 t=Path(td)
 for name,changed in [('rust_change','crates/test/src/lib.rs'),('cuda_change','crates/test/src/kernel.cu'),('crate_manifest_change','crates/test/Cargo.toml')]:
  d=t/name;d.mkdir();paths=['src/main.rs','crates/test/src/lib.rs','crates/test/src/kernel.cu','crates/test/Cargo.toml','Cargo.toml','Cargo.lock','build.rs','binary']
  for p in paths:
   f=d/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture');os.utime(f,(1000,1000))
  os.utime(d/'binary',(2000,2000));os.utime(d/changed,(3000,3000))
  script='set -e\nlog(){ :; }\nRUST_BINARY='+str(d/'binary')+'\n'+block.replace('/app',str(d))+'\nprintf "%s" "$NEEDS_BUILD"\n'
  run=subprocess.run(['bash','-c',script],capture_output=True,text=True,check=True)
  rows.append({'case':name,'changed_input':changed,'needs_build':run.stdout})
assert [r['needs_build'] for r in rows]==['true','false','false']
result={'date':'2026-09-04','scope':'Extracted timestamp decision block with /app replaced by temporary root; no compiler, clean, service launch or Docker operation.','cases':rows,'source_sha256':{'scripts/rust-backend-wrapper.sh':hashlib.sha256(source.read_bytes()).hexdigest()}}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))
