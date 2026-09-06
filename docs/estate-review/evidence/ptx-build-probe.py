#!/usr/bin/env python3
"""Execute unchanged PTX phase extracted from build.rs with invented nvcc output."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[4]/'project'
p=root/'crates/visionclaw-gpu/build.rs';s=p.read_text()
# Stop before native compilation/linking. Everything before this boundary is unchanged.
phase=s.split('    // ── Phase 2: Native linking')[0]+'}\n'
names=re.findall(r'"src/cuda_sources/(\w+)\.cu"',phase)
rows=[]
with tempfile.TemporaryDirectory(prefix='estate-ptx-build-') as td:
 t=Path(td);src=t/'phase.rs';src.write_text(phase);exe=t/'phase'
 subprocess.run(['rustc',str(src),'-o',str(exe)],check=True,capture_output=True)
 for label,mode,payload in [('missing_nvcc','missing','.version 9.1\n.target sm_75\n'),('failed_nvcc_fallback','fail','.version 9.1\n.target sm_75\n'),('invalid_nonempty','success','NOT PTX\n'),('empty','success',''),('already_9_0','success','.version 9.0\n.target sm_75\n'),('future_minor','success','.version 9.10\n.target sm_75\n')]:
  d=t/label;out=d/'out';binpath=d/'bin';fb=d/'src/ptx'
  for x in (out,binpath,fb):x.mkdir(parents=True,exist_ok=True)
  for name in names:(fb/(name+'.ptx')).write_text(payload)
  if mode!='missing':
   nvcc=binpath/'nvcc'
   nvcc.write_text('#!'+sys.executable+'\nimport sys,os\nfrom pathlib import Path\nif os.environ["ESTATE_NVCC_MODE"]=="fail":sys.exit(1)\nPath(sys.argv[sys.argv.index("-o")+1]).write_text(os.environ["ESTATE_PTX_PAYLOAD"])\n')
   nvcc.chmod(0o755)
  env={'PATH':str(binpath),'OUT_DIR':str(out),'CARGO_FEATURE_GPU':'1','CUDA_ARCH':'75','DOCKER_ENV':'1','ESTATE_NVCC_MODE':mode,'ESTATE_PTX_PAYLOAD':payload}
  run=subprocess.run([str(exe)],cwd=d,env=env,capture_output=True,text=True)
  produced=out/'visionclaw_unified.ptx'
  rows.append({'case':label,'exit_code':run.returncode,'fallback_reported':'using pre-compiled PTX' in run.stdout,'downgrade_warnings':run.stdout.count('Downgraded'),'produced_text':produced.read_text() if produced.exists() else None,'failure_lines':[x for x in run.stderr.splitlines() if 'Failed to execute nvcc' in x or 'empty after compilation' in x]})
assert rows[0]['exit_code']!=0 and not rows[0]['fallback_reported']
assert rows[1]['exit_code']==0 and rows[1]['fallback_reported']
assert rows[2]['exit_code']==0 and rows[3]['exit_code']!=0
result={'date':'2026-09-04','scope':'Actual build.rs PTX phase extracted unchanged; native phase omitted. Synthetic nvcc executable/output and temporary bundled files. No real CUDA compilation, PTX validation, driver load or production build.','kernel_count':len(names),'cases':rows,'source_sha256':{'crates/visionclaw-gpu/build.rs':hashlib.sha256(p.read_bytes()).hexdigest()}}
result['runtime_source_sha256']={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in ['crates/visionclaw-gpu/src/ptx_loader.rs','src/utils/unified_gpu_compute/construction.rs']}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
