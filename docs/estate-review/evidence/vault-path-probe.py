#!/usr/bin/env python3
from pathlib import Path
import re,subprocess,json,hashlib
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project/agentbox';source=(r/'config/entrypoint-unified.sh').read_text();function=re.search(r'^_ab_vault_resolve\(\) \{\n.*?^\}',source,re.M|re.S).group(0)
results=[]
for name,legacy in [('no_vault_no_override',''),('no_vault_legacy_override','/fixture/old-pages')]:
 code='_ab_toml_val() { :; }\n'+function+'\n_ab_vault_resolve\nprintf "RESULT:%s|%s|%s\\n" "$AGENTBOX_VAULT_ENABLED" "${VAULT_PAGES:-}" "${ONTOLOGY_PAGES_DIR:-}"\n'
 run=subprocess.run(['bash','-c',code],env={'ONTOLOGY_PAGES_DIR':legacy},capture_output=True,text=True)
 line=next(x for x in run.stdout.splitlines() if x.startswith('RESULT:'));enabled,pages,override=line[7:].split('|');results.append({'case':name,'exit_code':run.returncode,'enabled':enabled,'vault_pages':pages,'legacy_override':override})
assert results[1]['enabled']=='0' and results[1]['legacy_override']=='/fixture/old-pages'
paths=['config/entrypoint-unified.sh','config/tmux-autostart.sh','mcp/servers/lib/ontology-local.js','mcp/servers/lib/ontology-index-build.js','scripts/ontology-condense-scheduler.mjs','scripts/ontology-condense-refresh.sh','lib/rune.nix']
j={'date':'2026-09-04','scope':'actual extracted resolver with stub empty manifest reader and fresh environment; no real vault, terminal or consumer changed','results':results,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/vault-path-probe.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
