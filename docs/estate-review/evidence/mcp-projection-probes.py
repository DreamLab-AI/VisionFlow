#!/usr/bin/env python3
from pathlib import Path
import json,tempfile,subprocess,shutil,hashlib
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project/agentbox';script=r/'scripts/project-mcp-servers.mjs'
managed={'command':'fixture-no-execution','x-agentbox-managed-by':'projector','x-agentbox-gate':'env:FIXTURE_GATE','x-agentbox-requires':[]}
results=[]
with tempfile.TemporaryDirectory(prefix='estate-projector-') as t:
 d=Path(t);reg=d/'registry.json';target=d/'target.json'
 cases=[('gate_on',{'managed':managed},'true'),('gate_off',{'managed':managed},'false'),('deleted_registry_entry',{},'true'),('missing_requires',{'managed':{k:v for k,v in managed.items() if k!='x-agentbox-requires'}},'true'),('malformed_registry',None,'false')]
 for name,defs,gate in cases:
  reg.write_text(json.dumps({'mcpServers':defs}) if defs is not None else '{')
  target.write_text(json.dumps({'mcpServers':{'managed':{'command':'old-fixture'},'bespoke':{'command':'preserve-fixture'}}}))
  run=subprocess.run([shutil.which('node'),str(script)],env={'MCP_REGISTRY':str(reg),'MCP_JSON':str(target),'FIXTURE_GATE':gate},capture_output=True,text=True)
  after=json.loads(target.read_text())['mcpServers'];results.append({'case':name,'exit_code':run.returncode,'managed_present': 'managed' in after,'managed_command':after.get('managed',{}).get('command'),'bespoke_preserved':after.get('bespoke')=={'command':'preserve-fixture'}})
assert [x['managed_present'] for x in results]==[True,False,True,False,True]
assert all(x['exit_code']==0 and x['bespoke_preserved'] for x in results)
paths=['scripts/project-mcp-servers.mjs','skills/mcp.json','config/entrypoint-unified.sh','management-api/lib/system-manifest.js']
j={'date':'2026-09-04','scope':'actual projector with temporary registry/target and fresh environment; no real MCP configuration change or server launch','results':results,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths},'current_registry_managed_entries_without_requires_array':[]}
(b/'evidence/mcp-projection-probes.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
