'use strict';
const fs=require('fs'),os=require('os'),path=require('path'),assert=require('assert'),crypto=require('crypto');
const root=path.resolve(__dirname,'../../../../project/agentbox');
const {ExecutionJournal}=require(path.join(root,'management-api/lib/execution-journal'));
const {LocalJsonlEventsAdapter}=require(path.join(root,'management-api/adapters/events/local-jsonl'));
(async()=>{
 const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'estate-journal-'));
 try{
  const blocked=path.join(tmp,'not-a-directory');fs.writeFileSync(blocked,'fixture');
  const adapter=new LocalJsonlEventsAdapter({eventsDir:blocked});let notifications=0;
  await adapter.subscribe(null,()=>notifications++);
  const journal=new ExecutionJournal({eventsAdapter:adapter});
  const event={session_urn:'fixture-session',event_id:'fixture-event',type:'input.claimed',harness:'fixture',turn:0,payload:{text:'original'}};
  const first=await journal.append(event);const retry=await journal.append(event);
  const trace=await journal.assertModelRequestTraceable({session_urn:'fixture-session',messages:[{content:'different model-visible text',cites:[0]}]});
  assert.equal(first.duplicate,false);assert.equal(retry.duplicate,true);assert.equal(notifications,1);assert.equal(trace.ok,true);assert.equal(fs.readFileSync(blocked,'utf8'),'fixture');
  const files=['management-api/lib/execution-journal.js','management-api/adapters/events/local-jsonl.js'];
  const result={date:'2026-09-04',scope:'Actual journal plus actual local adapter with temporary file blocking log directory creation. No production journal, model request or server.','first_seq':first.envelope.seq,'retry_duplicate':retry.duplicate,'subscriber_notifications':notifications,'changed_content_citing_unpersisted_seq_accepted':trace.ok,'durable_event_written':false,source_sha256:Object.fromEntries(files.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,p))).digest('hex')]))};
  fs.writeFileSync(__filename.replace('.cjs','.json'),JSON.stringify(result,null,2)+'\n');console.log('Five composition assertions pass; failed disk append reproduced');
 }finally{fs.rmSync(tmp,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1});
