'use strict';
const path=require('node:path'),fs=require('node:fs'),crypto=require('node:crypto');
const base=path.resolve(__dirname,'..');const repo=path.resolve(base,'../../../project/agentbox');
process.env.OPF_MODE='local';
const privacy=require(path.join(repo,'management-api/middleware/privacy-filter.js'));
const sentinel='INVENTED_DISPATCH_SENTINEL';let calls=0;
global.fetch=async()=>{calls++;return {ok:true,json:async()=>({text:'REDACTED_FIXTURE',replaced:['fixture']})};};
(async()=>{
 const out=[];
 for(const [method,input] of [['store',{key:sentinel,value:sentinel,metadata:{note:sentinel}}],['createEpic',{title:sentinel}],['store',{key:'fixture',value:{note:sentinel}}]]){
  calls=0;let received;
  const fn=privacy.wrapWithPrivacyFilter('beads',method,async (...args)=>{received=args;return 'ok';},{privacy_filter:{policy:{beads:'strict'}}});
  await fn(input);
  out.push({method,opf_calls:calls,received:received[0],value_type:typeof received[0].value});
 }
 if(out[0].opf_calls!==1||out[0].received.metadata.note!==sentinel||out[1].opf_calls!==0||out[2].value_type!=='string')throw Error('unexpected probe result');
 const paths=['management-api/middleware/privacy-filter.js','management-api/observability/metrics.js','management-api/adapters/index.js','management-api/server.js','management-api/middleware/linked-data/encoder.js'];
 const result={date:'2026-09-04',scope:'actual privacy wrapper, synthetic calls and injected fetch; no OPF, real adapter, persistence or network',results:out,source_sha256:Object.fromEntries(paths.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(repo,p))).digest('hex')]))};
 fs.writeFileSync(path.join(base,'evidence/dispatch-privacy-probe.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1});
