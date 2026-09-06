#!/usr/bin/env python3
"""Compare extracted current structs using host compilers, without loading CUDA."""
from pathlib import Path
import hashlib,json,re,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[4]/'project'
RUST=ROOT/'src/models/simulation_params.rs'
CUDA=ROOT/'crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu'
def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout.strip()
rust=RUST.read_text(); cuda=CUDA.read_text()
rbody=re.search(r'pub struct SimParams \{(.*?)\n\}',rust,re.S).group(1)
cbody=re.search(r'struct SimParams \{(.*?)\n\};',cuda,re.S).group(1)
rfields=re.findall(r'pub (\w+): (f32|u32|i32),',rbody)
cfields=re.findall(r'^\s*(float|unsigned int|int) (\w+);',cbody,re.M)
assert len(rfields)==len(cfields)==53
assert [(n,{'float':'f32','unsigned int':'u32','int':'i32'}[t]) for t,n in cfields]==rfields
rassert=re.search(r'const _: \(\) = assert!\(std::mem::size_of::<SimParams>\(\) == 212\);',rust).group()
cassert=re.search(r'static_assert\(sizeof\(SimParams\) == 212,.*?\);',cuda).group()
with tempfile.TemporaryDirectory(prefix='estate-simparams-') as td:
    tmp=Path(td)
    rs='#[repr(C)]\npub struct SimParams {'+rbody+'\n}\n'+rassert+'\nfn main(){\n'
    rs+='println!("size {} align {}",std::mem::size_of::<SimParams>(),std::mem::align_of::<SimParams>());\n'
    for n,t in rfields:rs+='println!("'+n+' {}",std::mem::offset_of!(SimParams,'+n+'));\n'
    rs+='}\n';(tmp/'layout.rs').write_text(rs)
    run(['rustc','--edition=2021',str(tmp/'layout.rs'),'-o',str(tmp/'rlayout')])
    rout=run([str(tmp/'rlayout')]).splitlines()
    def cpp(body,name):
        source='#include <cstddef>\n#include <iostream>\nstruct SimParams {'+body+'\n};\n'+cassert+'\nint main(){\n'
        source+='std::cout << "size " << sizeof(SimParams) << " align " << alignof(SimParams) << "\\n";\n'
        for n,t in rfields:source+='std::cout << "'+n+' " << offsetof(SimParams,'+n+') << "\\n";\n'
        source+='}\n';(tmp/(name+'.cpp')).write_text(source)
        run(['c++','-std=c++17',str(tmp/(name+'.cpp')),'-o',str(tmp/name)])
        return run([str(tmp/name)]).splitlines()
    cout=cpp(cbody,'current')
    assert rout==cout
    mutated=cbody.replace('float dt;','float ESTATE_SWAP;').replace('float damping;','float dt;').replace('float ESTATE_SWAP;','float damping;')
    mout=cpp(mutated,'swapped')
    assert mout[0]==rout[0] and mout!=rout
    result={'date':'2026-09-04','scope':'Extracted declarations and original size assertions compiled with host rustc/C++; no CUDA compiler, driver, device copy, shipped PTX or client runtime exercised.','field_count':len(rfields),'current_layouts_equal':True,'host_layout':rout,'same_size_mutation':{'change':'Swap CUDA dt and damping declaration order in temporary fixture only','original_size_assertion_passes':True,'changed_offsets':[{'rust':a,'mutated_cpp':b} for a,b in zip(rout,mout) if a!=b]},'compilers':{'rustc':run(['rustc','--version']),'cpp':run(['c++','--version']).splitlines()[0]},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUST,CUDA]}}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['field_count','current_layouts_equal','same_size_mutation']}))
