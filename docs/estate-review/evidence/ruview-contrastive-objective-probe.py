#!/usr/bin/env python3
"""Compile unchanged local cosine and InfoNCE functions in isolation."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[4]/'RuView'
base=root/'rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src'
s=(base/'embedding.rs').read_text()
def fn(name):
 start=s.index('fn '+name+'('); opening=s.index('{',start);depth=1;i=opening+1
 while depth:
  depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[start:i]
body=fn('cosine_similarity')+'\n'+fn('info_nce_loss')
main='''
fn main() {
 let x=vec![vec![1.0,0.0]]; let y=vec![vec![0.0,1.0]];
 assert_eq!(info_nce_loss(&x,&y,0.07),0.0);
 let collapsed=vec![vec![1.0,0.0];4];
 let loss=info_nce_loss(&collapsed,&collapsed,0.5);
 assert!((loss-4.0f32.ln()).abs()<1e-6);
 let a=vec![vec![1.0,0.0],vec![1.0,0.0]];
 let b=vec![vec![1.0,0.0],vec![0.0,1.0]];
 let ab=info_nce_loss(&a,&b,0.5);let ba=info_nce_loss(&b,&a,0.5);
 assert!((ab-ba).abs()>0.1);
 println!("3 assertions passed: singleton=0; collapsed4={}; AtoB={} BtoA={}",loss,ab,ba);
}
'''
with tempfile.TemporaryDirectory(prefix='ruview-contrastive-review-') as td:
 p=Path(td)/'probe.rs';p.write_text(body+'\n'+main)
 c=subprocess.run(['rustc','--edition=2021',str(p),'-o',str(Path(td)/'probe')],capture_output=True,text=True);assert c.returncode==0,c.stderr
 run=subprocess.run([str(Path(td)/'probe')],capture_output=True,text=True);assert run.returncode==0,run.stderr
files=[base/f for f in ['embedding.rs','trainer.rs','main.rs']]+[root/'docs/adr/ADR-024-contrastive-csi-embedding-model.md']
receipt={'date':'2026-09-05','scope':'Partial ADR024 review: projection, objective and pretraining/fine-tuning boundary only. Three unchanged-function assertions, no full training or dataset evaluation. Remaining ADR requirements are pending.','sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'assertions_passed':3,'stdout':run.stdout}
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(run.stdout.strip())
