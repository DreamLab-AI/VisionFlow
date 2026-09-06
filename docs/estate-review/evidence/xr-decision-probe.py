#!/usr/bin/env python3
"""Extract the current hierarchy predicate and its existing test; no actor/GPU."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile
root=Path(__file__).resolve().parents[4]/'project'
p=root/'src/actors/gpu/force_compute_actor.rs';s=p.read_text()
def function(name):
 start=s.index('    fn '+name+'(');brace=s.index('{',start);depth=1;i=brace+1
 while depth:
  depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[start:i]
predicate=function('is_directed_hierarchy_relation');test=function('directed_hierarchy_relation_accepts_only_class_subsumption')
with tempfile.TemporaryDirectory(prefix='estate-xr-decision-') as td:
 t=Path(td);src=t/'probe.rs';exe=t/'probe'
 src.write_text('struct FCA;\nimpl FCA {\n'+predicate+'\n}\n#[test]\n'+test)
 subprocess.run(['rustc','--edition=2021','--test',str(src),'-o',str(exe)],check=True,capture_output=True)
 run=subprocess.run([str(exe),'--nocapture'],capture_output=True,text=True)
 assert run.returncode!=0 and 'hierarchical must NOT' in run.stderr
hud=root/'xr-client/scripts/hud.gd';text=hud.read_text();sites=[]
for m in re.finditer(r'^.*(?:Button|CheckButton)\.new\(\).*$',text,re.M):
 sites.append({'line':text[:m.start()].count('\n')+1,'source':m.group().strip()})
files=['src/actors/gpu/force_compute_actor.rs','xr-client/scripts/hud.gd','xr-client/project.godot','xr-client/scripts/xr_boot.gd']
result={'date':'2026-09-04','scope':'Extracted unchanged predicate and one existing test, compiled with rustc; HUD/config source review. No full actor suite, Godot engine, controller or headset execution.','existing_predicate_test':{'exit_code':run.returncode,'failure':'hierarchical must NOT be treated as directed hierarchy'},'hud_constructor_sites':sites,'explicit_press_mode_lines':[i for i,l in enumerate(text.splitlines(),1) if 'action_mode = BaseButton.ACTION_MODE_BUTTON_PRESS' in l],'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'test_exit':run.returncode,'constructors':len(sites),'explicit_press_sites':len(result['explicit_press_mode_lines'])}))
