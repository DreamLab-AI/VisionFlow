'use strict';
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),cp=require('node:child_process'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const base=path.resolve(__dirname,'..'),repo=path.resolve(base,'../../../project/agentbox');
const temp=fs.mkdtempSync(path.join(os.tmpdir(),'estate-learning-order-'));
(async()=>{
 const validation=[];
 for(const [name,producer,retrieval,routing] of [['off',false,false,false],['retrieval_before_producer',false,true,false],['routing_before_producer',false,false,true],['both_before_producer',false,true,true],['producer_and_consumers',true,true,true]]){
  const file=path.join(temp,'agentbox.toml');
  fs.writeFileSync(file,`[memory_learning]\nenabled = true\nrecord_trajectories = ${producer}\nfeed_retrieval = ${retrieval}\nfeed_routing = ${routing}\n`);
  const r=cp.spawnSync(process.execPath,[path.join(repo,'scripts/agentbox-config-validate.js'),file],{encoding:'utf8'});
  validation.push({case:name,exit_code:r.status,w066:r.stderr.includes('W066'),other_codes:r.stderr.split('\n').filter(x=>x&&!x.startsWith('W066')).map(x=>x.split(' ')[0])});
 }
 assert.deepEqual(validation.map(x=>x.exit_code),[0,0,0,0,0]);
 assert.deepEqual(validation.map(x=>x.w066),[false,true,true,true,false]);
 const {createHybridTools}=require(path.join(repo,'mcp/servers/lib/memory-hybrid.js'));
 const consumer=[];
 for(const enabled of [false,true]){
  process.env.RUVECTOR_MEMORY_LEARNING_ENABLED=String(enabled);
  process.env.RUVECTOR_RECORD_TRAJECTORIES='false';
  process.env.RUVECTOR_FEED_RETRIEVAL='true';
  let calls=0;
  const hybrid=createHybridTools({pool:{query:async()=>({rows:++calls===1?[{key:'fixture',value:'fixture',metadata:{tags:['action:fixture']},score:0.5}]:[{tags:['action:fixture'],wilson:0.8}]})},getPgOk:()=>true,xinfEnsure:async()=>true,getEmbedding:async()=>[0],vecToSql:()=> 'fixture-not-sql',parseVal:x=>x,log:()=>{},memSearch:async()=>{throw Error('unexpected fallback')}});
  const r=await hybrid.memHybridSearch('fixture');
  assert.equal(r.success,true);assert.equal(calls,2);assert.ok(Math.abs(r.results[0].score-0.58)<1e-9);
  consumer.push({master_enabled:enabled,producer_enabled:false,feed_retrieval:true,mock_pool_calls:calls,score:r.results[0].score,effectiveness_bonus:r.results[0].components.effectiveness_bonus});
 }
 const paths=['scripts/agentbox-config-validate.js','schema/agentbox.toml.schema.json','scripts/skill-count-check.js','mcp/servers/lib/ruvector-gates.js','mcp/servers/lib/memory-hybrid.js','config/hooks/trajectory-recorder.cjs'];
 const result={date:'2026-09-04',scope:'actual validator with temporary minimal manifests; actual hybrid factory with injected pool and embedding stubs. No SQL execution, external memory access, hook execution or deployed-route test.',validation,consumer,source_sha256:Object.fromEntries(paths.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(repo,p))).digest('hex')]))};
 fs.writeFileSync(path.join(__dirname,'learning-order-probe.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({validation,consumer}));
})().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>fs.rmSync(temp,{recursive:true,force:true}));
