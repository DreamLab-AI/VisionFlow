const {execFileSync}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path');
const [repo,target,probe,output]=process.argv.slice(2);
if(!output)throw Error('usage: node run-probe.cjs REPO TARGET PROBE OUTPUT');
const env={...process.env,CARGO_TARGET_DIR:target,CARGO_BUILD_JOBS:'3',CARGO_PROFILE_DEV_DEBUG:'0',CARGO_PROFILE_TEST_DEBUG:'0'};
const built=execFileSync('cargo',['build','--locked','--offline','-p','sidestr-agent','--no-default-features','--lib','--message-format=json'],{cwd:repo,env,encoding:'utf8'});
const artifacts=built.split('\n').filter(Boolean).map(s=>JSON.parse(s)).filter(a=>a.reason==='compiler-artifact');
const deps=path.join(target,'debug/deps');
const args=['--edition=2021','-Dwarnings',probe,'-L','dependency='+deps];
for(const name of ['sidestr_agent','sidestr_core','bitcoin','serde_json']){
 const a=artifacts.find(a=>a.target.name===name&&a.filenames.some(f=>f.endsWith('.rlib')));
 if(!a)throw Error('missing '+name);
 args.push('--extern',name+'='+a.filenames.find(f=>f.endsWith('.rlib')));
}
const binary=path.join(target,'follower-audit-probe-'+process.pid);args.push('-o',binary);
execFileSync('rustc',args,{env,stdio:'inherit'});
try{const result=execFileSync(binary,[],{encoding:'utf8'});JSON.parse(result);fs.writeFileSync(output,result);console.log('Probe passed '+JSON.parse(result).cases.length+' scenarios');}finally{fs.unlinkSync(binary);}
