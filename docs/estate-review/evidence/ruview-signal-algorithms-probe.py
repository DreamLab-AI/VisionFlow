#!/usr/bin/env python3
"""Inspect six local modules and compile the unchanged Fresnel estimator slice."""
import hashlib,json,re,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[4]/'RuView'
crates=root/'rust-port/wifi-densepose-rs/crates'
base=crates/'wifi-densepose-signal/src'
names=['csi_ratio','hampel','fresnel','spectrogram','subcarrier_selection','bvp']
symbols=['conjugate_multiply','compute_ratio_matrix','hampel_filter','hampel_filter_2d','FresnelBreathingEstimator','compute_spectrogram','compute_multi_subcarrier_spectrogram','select_sensitive_subcarriers','select_by_variance','extract_bvp']
sources={str((base/(n+'.rs')).relative_to(root)):hashlib.sha256((base/(n+'.rs')).read_bytes()).hexdigest() for n in names+['lib']}
hits={s:[] for s in symbols}
for p in crates.rglob('*.rs'):
 text=p.read_text()
 for s in symbols:
  if re.search(r'\b'+s+r'\b',text): hits[s].append(str(p.relative_to(root)))
text=(base/'fresnel.rs').read_text()
body=text[text.index('pub const SPEED_OF_LIGHT'):text.index('/// Estimate TX-body')]
err=text[text.index('#[derive(Debug, thiserror::Error)]'):text.index('#[cfg(test)]\nmod tests')]
err=err.replace('Debug, thiserror::Error','Debug')
err=re.sub(r'    #\[error\([^\n]*\)\]\n','',err)
main='''
fn main() {
 let a=FresnelBreathingEstimator::new(FresnelGeometry::new(1.0,1.0,5e9).unwrap());
 let b=FresnelBreathingEstimator::new(FresnelGeometry::new(20.0,30.0,5e9).unwrap());
 let samples:Vec<f64>=(0..2000).map(|i| 0.1*(2.0*PI*0.25*i as f64/100.0).sin()).collect();
 let x=a.estimate_breathing_rate(&samples,100.0).unwrap();
 let y=b.estimate_breathing_rate(&samples,100.0).unwrap();
 assert_eq!((x.rate_bpm,x.confidence),(y.rate_bpm,y.confidence));
 assert!(FresnelGeometry::new(f64::NAN,1.0,5e9).is_ok());
 assert!(matches!(a.estimate_breathing_rate(&vec![1.0;2000],100.0),Err(FresnelError::NoSignal)));
 println!("3 assertions passed; both geometries rate={} confidence={}",x.rate_bpm,x.confidence);
}
'''
with tempfile.TemporaryDirectory(prefix='ruview-signal-review-') as td:
 p=Path(td)/'probe.rs';p.write_text('use std::f64::consts::PI;\n'+body+err+main)
 compiled=subprocess.run(['rustc','--edition=2021',str(p),'-o',str(Path(td)/'probe')],capture_output=True,text=True)
 assert compiled.returncode==0,compiled.stderr
 result=subprocess.run([str(Path(td)/'probe')],capture_output=True,text=True)
 assert result.returncode==0,result.stderr
receipt={'date':'2026-09-05','scope':'Three isolated unchanged Fresnel algorithm assertions; thiserror derive/attributes removed only; excludes later solver helper, full crate, FFT, captured data and performance validation','sources':sources,'named_api_files_in_local_crate_rust':hits,'module_test_declarations':{n:(base/(n+'.rs')).read_text().count('#[test]') for n in names},'csi_data_exact_mentions':{n:len(re.findall(r'\bCsiData\b',(base/(n+'.rs')).read_text())) for n in names},'probe_stdout':result.stdout,'assertions_passed':3}
Path(__file__).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(result.stdout.strip());print(json.dumps({'test_declarations':receipt['module_test_declarations'],'csi_data_mentions':receipt['csi_data_exact_mentions']}))
