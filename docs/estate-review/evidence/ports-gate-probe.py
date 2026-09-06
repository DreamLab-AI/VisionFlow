#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,shutil,json,hashlib
base=Path(__file__).resolve().parents[1];repo=base.parents[2]/'project/agentbox'
script=repo/'scripts/ci/check-ports-loopback.sh'
flow={'services':{'fixture':{'image':'fixture:no-launch','ports':['0.0.0.0:45678:45678']}}}
cases=[('loopback_block','docker-compose.yml','services:\n  fixture:\n    image: fixture:no-launch\n    ports:\n      - "127.0.0.1:45678:45678"\n',0),('public_block','docker-compose.yml','services:\n  fixture:\n    image: fixture:no-launch\n    ports:\n      - "0.0.0.0:45678:45678"\n',1),('public_json_flow','docker-compose.yml',json.dumps(flow)+'\n',0),('public_service_flow','docker-compose.yml','services:\n  fixture: {image: "fixture:no-launch", ports: ["0.0.0.0:45678:45678"]}\n',0)]
results=[]
for name,filename,body,expected in cases:
 with tempfile.TemporaryDirectory(prefix='estate-ports-') as tmp:
  root=Path(tmp);target=root/'scripts/ci/check-ports-loopback.sh';target.parent.mkdir(parents=True);shutil.copy2(script,target);(root/filename).write_text(body)
  run=subprocess.run(['sh',str(target)],capture_output=True,text=True)
  assert run.returncode==expected,(name,run.returncode,run.stderr)
  results.append({'case':name,'exit_code':run.returncode,'fixture':body,'output':run.stdout.strip(),'errors':run.stderr.strip()})
actual=subprocess.run(['sh',str(script)],capture_output=True,text=True);assert actual.returncode==0
paths=['scripts/ci/check-ports-loopback.sh','.github/workflows/invariants.yml','docker-compose.yml']
receipt={'date':'2026-09-04','scope':'Actual gate copied unchanged into temporary roots; invented compose inputs. JSON flow parsed with Python json; no Docker Compose evaluation, service launch or port binding.', 'current_tree':{'exit_code':actual.returncode,'stdout':actual.stdout.strip()},'results':results,'flow_fixture_parsed':json.loads(cases[2][2]),'source_sha256':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}}
(base/'evidence/ports-gate-probe.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'current_tree':actual.returncode,'fixtures':[{k:v for k,v in x.items() if k in ['case','exit_code']} for x in results]}))
