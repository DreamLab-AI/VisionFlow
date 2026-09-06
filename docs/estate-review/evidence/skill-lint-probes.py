#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,json,hashlib,shutil
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project/agentbox';script=r/'skills/lint-skills.sh';results=[]
for name,refs,body in [('long_without_references',False,'---\nname: fixture\ndescription: fixture\n---\n'+'ordinary text\n'*300),('long_empty_references',True,'---\nname: fixture\ndescription: fixture\n---\n'+'ordinary text\n'*300),('fields_outside_frontmatter',False,'---\n---\nname: fixture\ndescription: these fields are outside the frontmatter\n')]:
 with tempfile.TemporaryDirectory(prefix='estate-skill-lint-') as t:
  p=Path(t);shutil.copy(script,p/'lint-skills.sh');d=p/'fixture';d.mkdir();(d/'SKILL.md').write_text(body)
  if refs:(d/'references').mkdir()
  run=subprocess.run(['bash',str(p/'lint-skills.sh')],capture_output=True,text=True)
  results.append({'case':name,'exit_code':run.returncode,'stdout':run.stdout})
assert [x['exit_code'] for x in results]==[1,0,0]
paths=['skills/lint-skills.sh','skills/tree-search-coder/SKILL.md','skills/tree-search-coder/references/algorithm.md','flake.nix','.github/workflows/invariants.yml']
j={'date':'2026-09-04','scope':'actual lint script copied to isolated synthetic skill trees; no skill invocation, rebuild, provider or code execution by a model','results':results,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/skill-lint-probes.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
