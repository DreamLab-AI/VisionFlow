#!/usr/bin/env python3
from pathlib import Path
import subprocess,tomllib,hashlib,json
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project/agentbox'
files=subprocess.check_output(['git','-C',str(r),'ls-files','--cached','--others','--exclude-standard','-z','services'],text=True).split('\0')
rows=[]
for rel in sorted(set(x for x in files if x.endswith('/Cargo.toml'))):
 p=r/rel
 if not p.exists():continue
 data=tomllib.loads(p.read_text());pkg=data.get('package')
 if not pkg:continue
 rows.append({'manifest':rel,'name':pkg.get('name'),'declared_license':pkg.get('license'),'publish':pkg.get('publish','unspecified'),'adjacent_license_files':sorted(x.name for x in p.parent.glob('LICENSE*')),'adjacent_readmes':sorted(x.name for x in p.parent.glob('README*')),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
j={'date':'2026-09-04','scope':'Tracked and nonignored local service manifests, adjacent files and document source. No package archive creation, registry lookup/publication or legal compatibility assessment.','packages':rows,'document_sha256':{x:hashlib.sha256((r/x).read_bytes()).hexdigest() for x in ['services/LICENSING-NOTICE.md','docs/developer/licensing.md','docs/developer/ecosystem.md','lib/prose-sanitiser.nix','lib/diagram-ir.nix']}}
(b/'evidence/service-package-inventory.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'packages':len(rows),'missing_adjacent_license_files':sum(not x['adjacent_license_files'] for x in rows),'missing_adjacent_readmes':sum(not x['adjacent_readmes'] for x in rows)}))
