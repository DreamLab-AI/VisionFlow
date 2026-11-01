const {execFileSync}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path');
const [repo,target,probe,output,mode]=process.argv.slice(2);
if(!output)throw Error('usage: node run-probe.cjs REPO TARGET PROBE OUTPUT');
const env={...process.env,CARGO_TARGET_DIR:target,CARGO_BUILD_JOBS:'3',CARGO_PROFILE_DEV_DEBUG:'0',CARGO_PROFILE_TEST_DEBUG:'0'};
const built=execFileSync('cargo',['test','--locked','--offline','-p','solid-pod-rs-server',...(mode==='git'?['--features','git']:[]),'--test','mempool_selection','--no-run','--message-format=json'],{cwd:repo,env,encoding:'utf8'});
const artifacts=built.split('\n').filter(Boolean).map(s=>JSON.parse(s)).filter(a=>a.reason==='compiler-artifact');
const deps=path.join(target,'debug/deps');
const args=['--edition=2021','-Dwarnings',probe,'-L','dependency='+deps];
for(const name of ['solid_pod_rs','solid_pod_rs_server','serde_json','actix_web','tempfile','async_trait','bytes','wiremock']){
 const a=artifacts.find(a=>a.target.name===name&&a.filenames.some(f=>(f.endsWith('.rlib')||f.endsWith('.so'))));
 if(!a)throw Error('missing '+name);
 args.push('--extern',name+'='+a.filenames.find(f=>(f.endsWith('.rlib')||f.endsWith('.so'))));
}
const binary=path.join(target,'solid-recovery-audit-probe-'+process.pid);args.push('-o',binary);
execFileSync('rustc',args,{env,stdio:'inherit'});
try{const result=execFileSync(binary,[mode],{encoding:'utf8'});const selected=execFileSync(binary,['selection'],{encoding:'utf8',env:{...process.env,JSS_PAY_MEMPOOL_URL:'https://explorer.example.invalid/signet'}});fs.writeFileSync(output,JSON.stringify({...JSON.parse(result),selection:JSON.parse(selected)},null,2)+'\n');console.log('Recovery probe passed: '+mode);}finally{fs.unlinkSync(binary);}
