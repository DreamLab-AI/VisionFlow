import json, subprocess, tempfile, shutil, sys
from pathlib import Path
source=Path(sys.argv[1]); results=[]
roots=['cat-ai-infrastructure','bc-cryptographic-primitive','bc-network-component','dc-telepresence','infra-data-management','infra-software-engineering']
def run(name,pages,allow='',code=0,kind=None):
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);(p/'knowledge').mkdir();(p/'working').mkdir();(p/'ontology').mkdir();(p/'ontology/collision-allowlist.yaml').write_text(allow)
  for slug in roots:(p/'knowledge'/f'{slug}.md').write_text('---\ntitle: Root\n---\n')
  for file,content in pages.items():(p/file).write_text('---\n'+content+'\n---\n')
  r=subprocess.run([sys.executable,str(source/'collision_lint.py'),'--json'],cwd=p,capture_output=True,text=True);data=json.loads(r.stdout)
  assert r.returncode==code,(name,r.returncode,r.stdout,r.stderr)
  if kind: assert any(f['kind']==kind for f in data['findings']),(name,data)
  results.append({'scenario':name,'exit':r.returncode,**data})
run('knowledge title-variant control',{'knowledge/Tokeniser.md':'title: Tokeniser','knowledge/Tokenizer.md':'title: Tokenizer'},code=1,kind='TITLE_VARIANT')
run('working-only variants are unseen',{'working/Tokeniser.md':'title: Tokeniser','working/Tokenizer.md':'title: Tokenizer'})
run('cross-vault variants are unseen',{'knowledge/Tokeniser.md':'title: Tokeniser','working/Tokenizer.md':'title: Tokenizer'})
run('two unallowed alias owners refuse',{'knowledge/Alpha.md':'aliases:\n- Shared','knowledge/Beta.md':'aliases:\n- Shared'},code=1,kind='ALIAS_COLLISION')
run('one allowed pair suppresses a third alias owner',{'knowledge/Alpha.md':'aliases:\n- Shared','knowledge/Beta.md':'aliases:\n- Shared','knowledge/Gamma.md':'aliases:\n- Shared'},allow='- [Alpha, Beta]\n')
run('category concept missing required relation refuses',{'knowledge/Telepresence.md':'title: Telepresence'},code=1,kind='CATEGORY_ROOT_UNPARENTED')
run('unrelated frontmatter link satisfies category parent guard',{'knowledge/Telepresence.md':'title: Telepresence\nrelated: "[[dc-telepresence]]"'})
print(json.dumps(results,indent=2))
