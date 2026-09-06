#!/usr/bin/env python3
"""Probe embedding rank metric and parameter counts using unchanged local modules."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[4]/'RuView'
base=root/'rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src'
main='''
fn main() {
 let miner=embedding::HardNegativeMiner::new(0.25,5);
 let sims=vec![vec![1.0,0.99,0.98,0.97],vec![0.1,1.0,0.1,0.1],vec![0.1,0.1,1.0,0.1],vec![0.1,0.1,0.1,1.0]];
 let warm=miner.mine(&sims,0);assert_eq!(warm.len(),12);
 let selected=miner.mine(&sims,5);assert_eq!(selected.len(),3);
 assert!(selected.iter().all(|(i,_)| *i==0));
 println!("3 assertions passed; warmup={} selected={:?}; all selected negatives belong to anchor0",warm.len(),selected);
}
'''
with tempfile.TemporaryDirectory(prefix='ruview-embedding-mining-') as td:
 p=Path(td)/'probe.rs'
 modules='\n'.join('#[path='+json.dumps(str(base/(n+'.rs')))+'] mod '+n+';' for n in ['graph_transformer','sona','sparse_inference','embedding'])
 p.write_text('#![allow(dead_code,unused_imports)]\n'+modules+'\n'+main)
 c=subprocess.run(['rustc','--edition=2021',str(p),'-o',str(Path(td)/'probe')],capture_output=True,text=True);assert c.returncode==0,c.stderr
 result=subprocess.run([str(Path(td)/'probe')],capture_output=True,text=True);assert result.returncode==0,result.stderr
files=[base/f for f in ['embedding.rs','graph_transformer.rs','sona.rs','trainer.rs','main.rs','rvf_container.rs','sparse_inference.rs']]
receipt={'date':'2026-09-05','scope':'Three isolated hard-negative-miner assertions against four unchanged Rust modules; no full training, suite or model evaluation','sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'assertions_passed':3,'stdout':result.stdout}
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(result.stdout.strip())
