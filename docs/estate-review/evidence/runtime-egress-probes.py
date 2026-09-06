#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,json,os,hashlib,shutil
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project/agentbox'
with tempfile.TemporaryDirectory(prefix='estate-wrapper-') as tmp:
 root=Path(tmp);binary=root/'bin';binary.mkdir();stub=binary/'claude';stub.write_text('#!/bin/sh\nprintf "STUB_PROVIDER_ONLY\\n"\n');stub.chmod(0o755)
 results=[]
 for slug,good,bad in [('openrouter','https://openrouter.ai/api','https://openrouter.ai.example.invalid/api'),('zai','https://api.z.ai/api','https://api.z.ai.example.invalid/api')]:
  settings=root/'profiles'/slug/'.claude/settings.local.json';settings.parent.mkdir(parents=True)
  for case,url in [('expected',good),('wrong_host_with_expected_substring',bad),('unrelated','https://example.invalid')]:
   settings.write_text(json.dumps({'env':{'ANTHROPIC_BASE_URL':url,'ANTHROPIC_AUTH_TOKEN':'INVENTED_FIXTURE_NOT_A_SECRET'}}))
   run=subprocess.run([shutil.which('bash'),str(r/f'config/harness-wrappers/{slug}.sh')],env={'PATH':str(binary)+':'+os.environ['PATH'],'WORKSPACE':str(root)},capture_output=True,text=True)
   results.append({'wrapper':slug,'case':case,'exit_code':run.returncode,'stub_launched':'STUB_PROVIDER_ONLY' in run.stdout})
 assert sum(x['stub_launched'] for x in results)==4
 node=shutil.which('node');hook=r/'config/hooks/nostr-live-mirror.cjs';sentinel='INVENTED_MIRROR_SENTINEL_20260904'
 mirror=[]
 for case,extra in [('no_identity',{}),('explicit_recipient',{'AGENTBOX_MIRROR_RECIPIENT_PUBKEY':'1'*64}),('off',{'AGENTBOX_MIRROR_RECIPIENT_PUBKEY':'1'*64,'AGENTBOX_LIVE_MIRROR':'0'}),('derived_child',{'AGENTBOX_PRIVKEY_HEX':'0'*63+'1'})]:
  env={'AGENTBOX_MIRROR_DRY_RUN':'1',**extra}
  run=subprocess.run([node,str(hook),'UserPromptSubmit'],input=json.dumps({'session_id':'fixture','prompt':'password='+sentinel}),env=env,capture_output=True,text=True)
  mirror.append({'case':case,'exit_code':run.returncode,'composed_body_contains_sentinel':sentinel in run.stderr})
 assert [x['composed_body_contains_sentinel'] for x in mirror]==[False,True,False,True]
paths=['config/hooks/nostr-live-mirror.cjs','config/harness-wrappers/openrouter.sh','config/harness-wrappers/zai.sh','services/nostr-pod-bridge/src/session_summary.rs','services/nostr-pod-bridge/src/lib.rs']
j={'date':'2026-09-04','scope':'actual wrappers with temporary profiles and stub claude; mirror CLI dry-run with fresh environment and invented inputs; no real secrets, transcript reads, provider calls or sends','wrapper_results':results,'mirror_results':mirror,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/runtime-egress-probes.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
