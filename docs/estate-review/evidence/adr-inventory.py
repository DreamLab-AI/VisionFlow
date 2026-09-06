#!/usr/bin/env python3
"""Inventory estate ADR candidates without interpreting declared status as proof."""
import subprocess,pathlib,re,json,hashlib,collections
W=pathlib.Path(__file__).resolve().parents[4]
REPOS=['VisionFlow','project','project/agentbox','solid-pod-rs','nostr-rust-forum','dreamlab-ai-website','loom','knowledgeGraph','project4','visionGraph','WasmVOWL','dream-machine','ruvector','RuView']
rows=[]; summary=[]
for repo in REPOS:
 root=W/repo
 paths=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
 local=[]
 for rel in sorted(set(paths)):
  if repo=='VisionFlow' and rel in ['docs/estate-review/closeout/adr-inventory.md','docs/estate-review/closeout/adr-lineage.md']:continue  # generated self-index
  p=root/rel
  if not p.is_file() or p.suffix.lower() not in ['.md','.mdx','.rst']:continue
  if repo=='project' and rel.startswith('agentbox/'):continue
  parts=pathlib.PurePosixPath(rel).parts
  if not ((p.name.lower().startswith('adr-') or re.search(r'(^|[-_])adr-\d',p.name.lower())) or any(x.lower() in ['adr','adrs'] for x in parts)):continue
  content=p.read_text(errors='replace'); fm={}
  m=re.match(r'^---\s*\n(.*?)\n---',content,re.S)
  if m:
   for line in m[1].splitlines():
    kv=re.match(r'^([a-z_]+):\s*(.*)$',line)
    if kv:fm[kv[1]]=kv[2].strip()
  kind='decision'
  scope_note=None
  if p.name=='adr-architect.md' and '.claude/agents/' in rel:
   kind='support';scope_note='ADR-authoring agent definition with name/type/capabilities/hooks; not an architecture decision.'
  elif repo=='project/agentbox' and rel.startswith('skills/bhil-methodology/') and any(x in parts for x in ['templates','examples']):
   kind='support';scope_note='Reusable BHIL template or worked example; not an adopted estate decision. See catalog-decisions.md.'
  elif p.name.lower() in ['readme.md','index.md','preamble.md','template.md','adr-history-closeout.md'] or 'template' in p.stem.lower():kind='support'
  elif 'cross-link stub, not the canonical document' in content:kind='support'
  elif repo=='project4' and rel.startswith('www/api/markdown/') and 'owl-class:: software-architecture:ADR' in content:kind='ontology-content'
  elif ('knowledge/pages/' in rel or 'KnowledgeGraph/pages/' in rel or 'ontology/pages/' in rel) and '```json-ld' in content:kind='ontology-content'
  elif repo=='project4' or any(x in rel.lower() for x in ['archive/','archived/']):kind='historical-decision'
  title=fm.get('title') or next((x.lstrip('# ').strip() for x in content.splitlines() if x.startswith('# ')),p.stem)
  missing=[x for x in ['decision_status','implementation_status','activation_status','owner','review_trigger','verified_commit'] if not fm.get(x)] if kind=='decision' else []
  item={'repo':repo,'path':rel,'kind':kind,'title':title,'id':fm.get('id',p.stem.split('-')[0]+'-'+p.stem.split('-')[1] if p.stem.startswith('ADR-') else p.stem),'declared':{k:fm.get(k) for k in ['decision_status','implementation_status','activation_status','owner','review_trigger','verified_commit']},'missing_fields':missing,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'closeout':'historical-lineage-review' if kind=='historical-decision' else 'evidence-backed-open' if kind=='decision' and re.search(r'^## Closeout extension — \d{4}-\d{2}-\d{2}\s*$',content,re.M) else 'evidence-review-required' if kind=='decision' else 'scope-classification-review'}
  if scope_note:item['scope_note']=scope_note
  rows.append(item);local.append(item)
 summary.append({'repo':repo,'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'counts':dict(collections.Counter(x['kind'] for x in local))})
base=W/'VisionFlow/docs/estate-review'
(base/'evidence/adr-inventory.json').write_text(json.dumps({'date':'2026-09-05','method':'tracked and nonignored files; name/path candidates; classifications provisional; no declared completion treated as verified','repos':summary,'records':rows},indent=2)+'\n')
lines=['# ADR closeout inventory','', 'Generated from [the collector](../evidence/adr-inventory.py). Classifications are provisional; status fields are declarations, not implementation evidence. Every candidate is listed so that archives, support documents and similarly named ontology pages receive an explicit disposition.','', '| Repository | Decision candidates | Historical | Support | Ontology content |','|---|---:|---:|---:|---:|']
for s in summary:
 c=s['counts'];lines.append('| '+s['repo']+' | '+' | '.join(str(c.get(k,0)) for k in ['decision','historical-decision','support','ontology-content'])+' |')
for s in summary:
 lines+=['','## '+s['repo'],'','| Record | Kind | Declared decision / implementation / activation | Closeout disposition |','|---|---|---|---|']
 for r in rows:
  if r['repo']!=s['repo']:continue
  path='../../../../'+r['repo']+'/'+r['path']; label=r['path'].replace('|','\\|')
  status=' / '.join(str(r['declared'].get(k) or 'unstated') for k in ['decision_status','implementation_status','activation_status'])
  lines.append(f'| [{label}](<{path}>) | {r["kind"]} | {status.replace("|","/")} | {r["closeout"]} |')
(base/'closeout/adr-inventory.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'candidates':len(rows),'counts':dict(collections.Counter(x['kind'] for x in rows)),'repos':len(summary)}))
