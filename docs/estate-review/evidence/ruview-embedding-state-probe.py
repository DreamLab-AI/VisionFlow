#!/usr/bin/env python3
"""Compile unchanged embedding, transformer, SONA and sparse-inference modules as local Rust modules."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[4]/'RuView'
base=root/'rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src'
main='''
fn main() {
 use embedding::{ProjectionHead,EmbeddingConfig,CsiAugmenter};
 use sona::LoraAdapter;
 let cfg=EmbeddingConfig {d_model:2,d_proj:2,temperature:0.07,normalize:false};
 let mut p=ProjectionHead::zeros(cfg.clone());
 let mut adapter=LoraAdapter::new(2,2,1,1.0);
 for row in &mut adapter.a {for v in row {*v=1.0;}}
 for row in &mut adapter.b {for v in row {*v=1.0;}}
 p.lora_1=Some(adapter.clone());p.lora_2=Some(adapter);
 let input=vec![1.0,1.0];let before=p.forward(&input);
 p.merge_lora();let merged=p.forward(&input);
 assert!(merged[0]>before[0]*2.0);
 p.unmerge_lora();assert_eq!(p.forward(&input),before);
 let mut flat=vec![];p.flatten_into(&mut flat);
 let (restored,_)=ProjectionHead::unflatten_from(&flat,&cfg);
 assert!(restored.lora_1.is_none() && restored.forward(&input)!=before);
 let mut aug=CsiAugmenter::new();aug.temporal_jitter=0;aug.noise_std=0.0;
 aug.subcarrier_mask_ratio=0.0;aug.amplitude_scale_range=(1.0,1.0);
 let window=vec![vec![1.0,2.0],vec![3.0,4.0]];
 let (a,b)=aug.augment_pair(&window,42);
 assert_eq!(a,window);
 let scale=b[0][0];assert!(scale<1.0 && scale>0.0);
 assert!((b[1][1]-4.0*scale).abs()<1e-6);
 println!("6 assertions passed; LoRA before={:?} merged={:?}; phase-labelled augmentation scale={}",before,merged,scale);
}
'''
with tempfile.TemporaryDirectory(prefix='ruview-embedding-state-') as td:
 p=Path(td)/'probe.rs'
 modules='\n'.join('#[path='+json.dumps(str(base/(n+'.rs')))+'] mod '+n+';' for n in ['graph_transformer','sona','sparse_inference','embedding'])
 p.write_text('#![allow(dead_code,unused_imports)]\n'+modules+'\n'+main)
 c=subprocess.run(['rustc','--edition=2021',str(p),'-o',str(Path(td)/'probe')],capture_output=True,text=True);assert c.returncode==0,c.stderr
 result=subprocess.run([str(Path(td)/'probe')],capture_output=True,text=True);assert result.returncode==0,result.stderr
files=[base/f for f in ['embedding.rs','graph_transformer.rs','sona.rs','trainer.rs','main.rs','rvf_container.rs','sparse_inference.rs']]
receipt={'date':'2026-09-05','scope':'Six isolated assertions against four unchanged complete Rust source modules; no complete crate suite, training, RVF round trip or physical data execution','sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'assertions_passed':6,'stdout':result.stdout}
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(result.stdout.strip())
