#!/usr/bin/env python3
from pathlib import Path
import os,subprocess,tempfile,json,hashlib,sys
root=Path(__file__).resolve().parents[4]/'project';script=root/'scripts/canary/d1-beam-check.sh';rows=[]
with tempfile.TemporaryDirectory(prefix='estate-canary-consumer-') as td:
 t=Path(td);fake=t/'curl';fake.write_text('#!'+sys.executable+'\nimport os,sys,json\nwith open(os.environ["ESTATE_CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(os.environ["ESTATE_OBSERVE"] if "POST" in sys.argv else os.environ["ESTATE_ROSTER"])\n');fake.chmod(0o755)
 for name,roster,reply in [('empty','{"count":0}','{"fired":true}'),('count_only','{"count":1}','{"fired":true}'),('false_receipt','{"count":1}','{"fired":false}')]:
  log=t/(name+'.jsonl');env=os.environ.copy();env.update(PATH=str(t)+':'+env['PATH'],VISIONCLAW_TOKEN='',D1_MAX_ATTEMPTS='1',D1_INTERVAL_SECS='0',ESTATE_ROSTER=roster,ESTATE_OBSERVE=reply,ESTATE_CALL_LOG=str(log))
  run=subprocess.run(['bash',str(script),'http://fixture.invalid'],env=env,capture_output=True,text=True)
  rows.append({'case':name,'exit_code':run.returncode,'calls':[json.loads(x) for x in log.read_text().splitlines()]})
assert [r['exit_code'] for r in rows]==[2,0,0]
files=['scripts/canary/d1-beam-check.sh','client/src/services/livenessCanary.ts','client/src/features/bots/components/SwarmObservabilityPanel.tsx']
d={'date':'2026-09-04','scope':'Actual shell checker with fake curl only; no network, event, server, rendered beam or promotion operation. Dashboard source review only.','cases':rows,'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}}
Path(__file__).with_suffix('.json').write_text(json.dumps(d,indent=2)+'\n');print([(r['case'],r['exit_code']) for r in rows])
