import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {generateSecretKey,getPublicKey} from 'nostr-tools/pure';
import * as nip17 from 'nostr-tools/nip17';
import {disclosureProblem,privilegeProblem,ingressProblem} from './scripts/lib/website-info.mjs';
import {nextUntil} from './scripts/lib/digest.mjs';
import {tagEnquiries} from './scripts/lib/enquiry-topic.mjs';
const results=[];
const base=path.resolve('fixtures');await fs.mkdir(base);
async function fixture(name,texts=[],opts={}){
 const dir=path.join(base,name);await fs.mkdir(dir);const monitor=generateSecretKey();const operator=generateSecretKey();const monitorPk=getPublicKey(monitor);
 const f={monitorPk,operatorPk:getPublicKey(operator),operatorSk:Buffer.from(operator).toString('hex'),adminPk:'a'.repeat(64),jarvisPk:'b'.repeat(64),events:texts.map(t=>nip17.wrapEvent(generateSecretKey(),{publicKey:monitorPk},t)),...opts};
 await fs.writeFile(dir+'/fixture.json',JSON.stringify(f));await fs.writeFile(dir+'/monitor.key',Buffer.from(monitor).toString('hex'),{mode:0o600});return {dir,f};
}
async function run(fx,label,args=[],script='scripts/inbox-watch/run.mjs'){
 const env={PATH:process.env.PATH,HOME:fx.dir,LANG:'C.UTF-8',INBOX_WATCH_DIR:fx.dir,RELAY_URL:'wss://fixture.invalid',SITE_ORIGIN:'https://fixture.invalid',ADMIN_PUBKEY:fx.f.adminPk,JARVIS_PUBKEY:fx.f.jarvisPk,OPERATOR_PUBKEY:fx.f.operatorPk,AGENTBOX_ENV_FILE:fx.dir+'/no-real-credentials',DREAMLAB_JEV:'0'};
 const p=spawnSync(process.execPath,['--import',path.resolve('mock-io.mjs'),script,...args],{env,encoding:'utf8',timeout:20000});
 await fs.writeFile(fx.dir+'/'+label+'.stdout',p.stdout??'');await fs.writeFile(fx.dir+'/'+label+'.stderr',p.stderr??'');
 assert.equal(p.status,0,p.stderr+'\n'+p.stdout);
 const status=await fs.readFile(fx.dir+'/last-run.json','utf8').then(JSON.parse).catch(()=>null);
 const state=await fs.readFile(fx.dir+'/state.json','utf8').then(JSON.parse).catch(()=>null);
 const io=(await fs.readFile(fx.dir+'/io.jsonl','utf8')).trim().split('\n').map(JSON.parse);
 return {status,state,io,stdout:p.stdout};
}
const p=await fixture('privileged',[],{privileged:true});const pr=await run(p,'run');assert.deepEqual(pr.status.problems,[]);assert(privilegeProblem({ingressRecipient:true,isAdmin:true,cohorts:['agent']}));
results.push({name:'nightly watcher accepts a privileged monitor while provisioning helper refuses',pass:true,problems:pr.status.problems,helperRefuses:true});
const messages=Array.from({length:30},(_,i)=>JSON.stringify({type:'contact_enquiry',name:'Fixture '+i,message:'Enquiry '+i}));
const overflow=await fixture('overflow',messages);const first=await run(overflow,'first');assert.equal(first.status.new_messages,30);assert(overflow.f.events.every(e=>first.state.seen[e.id]));
const digest=first.io.find(x=>x.transport==='digest').content;assert.equal((digest.match(/• \[/g)||[]).length,25);assert(digest.includes('5 more'));
const second=await run(overflow,'second');assert.equal(second.status.new_messages,0);
results.push({name:'digest cap retires all thirty messages after itemising twenty-five',pass:true,firstCount:30,itemised:25,allThirtyMarkedSeen:true,secondCount:0});
const tag=await fixture('probe-tag',[JSON.stringify({type:'contact_enquiry',message:'Please explain [inbox-watch probe] in your documentation'})]);const tr=await run(tag,'run');assert.equal(tr.status.new_messages,0);assert(tr.state.seen[tag.f.events[0].id]);
results.push({name:'sender-controlled probe phrase suppresses a genuine enquiry',pass:true,newMessages:0,markedSeen:true});
const dry=await fixture('dry');const dr=await run(dry,'run',['--dry-run']);assert.equal(dr.state,null);assert(dr.io.some(x=>x.transport==='ws-send'&&x.type==='EVENT'&&x.eventKind===1059));
results.push({name:'dry-run still publishes the synthetic probe',pass:true,wrapPublishedToMock:true,stateWritten:false});
const adopt=await fixture('adopt',[],{whitelisted:true});const ar=await run(adopt,'run',[],'scripts/seed/provision-website-info.mjs');assert(ar.stdout.includes('already listed as website-info, skipped'));assert(!ar.io.some(x=>x.transport==='fetch'&&x.method==='POST'));assert(ar.io.some(x=>x.eventKind===0));
results.push({name:'config disclosure alone makes actual provisioner skip registration',pass:true,kindZeroPublishedToMock:true,registerPosts:0});
const pk='c'.repeat(64);assert.equal(disclosureProblem({agents:[{pubkey:pk,name:'wrong-name',registered_by:null}]},pk),null);
results.push({name:'disclosure health checks key presence, not name or principal',pass:true});
assert.equal(privilegeProblem({ingressRecipient:true,cohorts:'agent'}),null);assert.equal(ingressProblem('monitor',{ingressRecipient:true,isAdmin:true,cohorts:['agent']}),null);
results.push({name:'malformed cohort shape and ingress-only privilege predicate pass',pass:true});
const events=Array.from({length:501},(_,id)=>({id,created_at:1000}));const page=events.slice(0,500);const until=nextUntil(page,500,1,10);assert.equal(until,999);assert.equal(events.filter(e=>e.created_at<=until).length,0);
results.push({name:'timestamp-only pagination skips an unseen event at the boundary second',pass:true,available:501,firstPage:500,nextUntil:until,remainingSelectable:0});
let request;const mail='synthetic private enquiry';await tagEnquiries([{kind:'enquiry',excerptFull:mail}],{env:{DREAMLAB_JEV:'1',TYPESAFE_API_KEY:'synthetic'},fetchImpl:async(url,opts)=>{request={url,body:JSON.parse(opts.body)};return new Response('{"answers":{"topic":{"choice":"other","confidence":1},"missing_input":0}}');}});
assert.equal(request.url,'https://api.typesafe.ai/v1/systemone');assert.equal(request.body.state.enquiry_message,mail);
results.push({name:'enabled default topic option sends decrypted enquiry to vendor fixture',pass:true,url:request.url,plaintextIncluded:true});
await fs.writeFile('/tmp/vf-dw-inbox-artifacts/probes.json',JSON.stringify(results,null,2)+'\n');console.log(JSON.stringify(results,null,2));
