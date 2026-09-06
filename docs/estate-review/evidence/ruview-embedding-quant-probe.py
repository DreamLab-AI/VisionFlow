#!/usr/bin/env python3
"""Probe embedding rank metric and parameter counts using unchanged local modules."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[4]/'RuView'
base=root/'rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src'
main='''
fn main() {
 use embedding::{ProjectionHead,EmbeddingConfig,PoseEncoder,validate_quantized_embeddings};
 use graph_transformer::{CsiToPoseTransformer,TransformerConfig};
 use sparse_inference::Quantizer;
 let data=vec![vec![1.0,0.001],vec![1.0,0.002],vec![0.0,1.0]];
 let q=vec![1.0,0.0];let corr=validate_quantized_embeddings(&data,&q,&Quantizer);
 assert!((corr-0.875).abs()<1e-6);
 let corrected=3.0f32.sqrt()/2.0;
 assert!((corr-corrected).abs()>0.008);
 assert_eq!(validate_quantized_embeddings(&[],&q,&Quantizer),1.0);
 assert_eq!(validate_quantized_embeddings(&data[..1],&q,&Quantizer),1.0);
 let model=CsiToPoseTransformer::new(TransformerConfig {n_subcarriers:56,n_keypoints:17,d_model:64,n_heads:4,n_gnn_layers:2});
 let projection=ProjectionHead::new(EmbeddingConfig::default());
 let pose=PoseEncoder::new(128);
 assert_eq!(projection.param_count(),24832);assert_eq!(pose.param_count(),23168);
 println!("6 assertions passed; reported correlation={} tie-aware correlation={}; backbone={} projection={} pose={} total_without_pose={} total_with_pose={}",corr,corrected,model.param_count(),projection.param_count(),pose.param_count(),model.param_count()+projection.param_count(),model.param_count()+projection.param_count()+pose.param_count());
}
'''
with tempfile.TemporaryDirectory(prefix='ruview-embedding-quant-') as td:
 p=Path(td)/'probe.rs'
 modules='\n'.join('#[path='+json.dumps(str(base/(n+'.rs')))+'] mod '+n+';' for n in ['graph_transformer','sona','sparse_inference','embedding'])
 p.write_text('#![allow(dead_code,unused_imports)]\n'+modules+'\n'+main)
 c=subprocess.run(['rustc','--edition=2021',str(p),'-o',str(Path(td)/'probe')],capture_output=True,text=True);assert c.returncode==0,c.stderr
 result=subprocess.run([str(Path(td)/'probe')],capture_output=True,text=True);assert result.returncode==0,result.stderr
files=[base/f for f in ['embedding.rs','graph_transformer.rs','sona.rs','trainer.rs','main.rs','rvf_container.rs','sparse_inference.rs']]
receipt={'date':'2026-09-05','scope':'Six isolated quantisation/count assertions against four unchanged Rust modules; no quantised-model inference, full suite, hardware, training or benchmark','sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'assertions_passed':6,'stdout':result.stdout}
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(result.stdout.strip())
