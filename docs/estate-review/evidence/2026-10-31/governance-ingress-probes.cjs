// Usage: node governance-ingress-probes.cjs /path/to/agentbox [ontomark]
// Synthetic signed events, recording relay bridge, temporary corpus/pod only.
const fs=require('node:fs'), path=require('node:path'), os=require('node:os');
const assert=require('node:assert/strict'), crypto=require('node:crypto');
const {createRequire}=require('node:module'), {spawnSync}=require('node:child_process');
const root=path.resolve(process.argv[2]), req=createRequire(path.join(root,'package.json'));
const {finalizeEvent,verifyEvent,getPublicKey}=req('nostr-tools');
const {RelayConsumer}=req('./mcp/nostr-bridge/relay-consumer');
const {LocalProcessManagerOrchestratorAdapter}=req('./management-api/adapters/orchestrator/local-process-manager');
const binary=process.argv[3]||'ontomark', tmp=fs.mkdtempSync(path.join(os.tmpdir(),'vf-gov-ingress-'));
const originalCwd=process.cwd(); process.chdir(tmp);
const humanKey=new Uint8Array(32).fill(7), agentKey=new Uint8Array(32).fill(8), human=getPublicKey(humanKey);
const now=Math.floor(Date.now()/1000), iri='urn:ngm:class:a', quiet={info(){},warn(){},error(){},debug(){}};
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const sign=(kind,content,tags,key=humanKey,at=now)=>finalizeEvent({kind,content:JSON.stringify(content),tags,created_at:at},key);
function consumer(dir,orch,bridge=null) {return new RelayConsumer({npubs:['audit-local'],podRoot:dir,relayUrls:['ws://127.0.0.1:1'],adapters:{orchestrator:orch},logger:quiet,forumGovernance:{relayUrl:'wss://fixture.invalid',signers:[human],stateFile:path.join(dir,'cursor.json'),bridge}});}
function proposal(kind,diff){const p={level:'content',kind,iri,page:'A',hypothesis:'isolated audit fixture',diff,blockers:[],pages:['A'],proposer:'fixture',generation:'fixture@old',stale_after:'2000-01-01T00:00:00Z'};p.digest='sha256:'+sha([p.level,p.iri,p.page,p.hypothesis,p.diff].join('\n')+'\n'+(kind==='create'?'kind:create\n':''));return p;}
const md=title=>`---\ntype: Class\ntitle: ${title}\npublic: true\nresource: ${iri}\nstatus: draft\n---\noriginal body\n`;
async function signedCase(name,kind,mismatch) {
 const dir=path.join(tmp,name);fs.mkdirSync(path.join(dir,'ontology'),{recursive:true});fs.mkdirSync(path.join(dir,'knowledge'));
 fs.writeFileSync(path.join(dir,'ontology/vocabulary.yaml'),'version: 1\nnamespace: "urn:ngm:class:"\ntypes:\n  Class: { owl: "owl:Class", required: [resource, status] }\nrelations:\n  is-a: { owl: "rdfs:subClassOf" }\n');
 const calls=[],ledger=[];
 const cli=args=>{const r=spawnSync(binary,['--repo',dir,...args],{encoding:'utf8'});if(r.error)throw r.error;calls.push({verb:args[0],exit:r.status});if(r.status!==0){let e=new Error(r.stderr||r.stdout);e.code=r.status;e.stdout=r.stdout;throw e;}return JSON.parse(r.stdout);};
 let p;
 if(kind==='amend') {const stage=path.join(dir,'seed.md');fs.writeFileSync(stage,md('A'));cli(['create',stage,'--expect','docs=1','--json']);fs.unlinkSync(stage);const text=fs.readFileSync(path.join(dir,'knowledge/A.md'),'utf8');const n=text.split('\n').indexOf('original body')+1;p=proposal(kind,`--- a/A\n+++ b/A\n@@ -${n},1 +${n},1 @@\n-original body\n+approved body\n`);}
 else {const lines=md('B').trimEnd().split('\n');p=proposal(kind,`--- /dev/null\n+++ b/A\n@@ -0,0 +1,${lines.length} @@\n`+lines.map(l=>'+'+l+'\n').join(''));}
 const d=mismatch?'f'.repeat(64):p.digest.slice(7);
 const request=sign(31402,p,[['d',d],['panel','ontology-governance'],['level','content'],['context_url',iri]],agentKey);
 const event=sign(31403,{action:'promote',iri,reasoning:'I approve this isolated fixture change.'},[['d',d],['e',request.id]]);
 assert(verifyEvent(request)&&verifyEvent(event));
 const bridge={subscribe(filter,cb){queueMicrotask(()=>cb(request));return 'fixture-sub';},unsubscribe(){}};
 const orch=new LocalProcessManagerOrchestratorAdapter();
 orch._ontologyDeps={repo:dir,tmpRoot:dir,fetchRequest:async()=>null,runVault:async args=>cli(args),runGit:async()=>{throw new Error('Git intercepted');},fetchFn:async (url,init)=>{ledger.push(JSON.parse(init.body));return {ok:true};}};
 let handled;const original=orch.handleGovernanceDecision.bind(orch);orch.handleGovernanceDecision=async (...a)=>(handled=await original(...a));
 const c=consumer(dir,orch,bridge);c._loadForumState();c._onForumGovernance(event,'fixture');await c._forum.chain;
 const files=Object.fromEntries(fs.readdirSync(path.join(dir,'knowledge')).map(f=>[f,fs.readFileSync(path.join(dir,'knowledge',f),'utf8')]));
 assert.equal(handled.ontology.applied,true);
 return {name,signaturesValid:true,decisionTag:d,proposalDigest:p.digest,expiredAt:p.stale_after,ontology:handled.ontology,ledger,files,calls};
}
async function queuedReplay(){
 const dir=path.join(tmp,'queue');fs.mkdirSync(dir);let releaseFirst,releaseSecond;const started=[],completed=[];
 const firstGate=new Promise(r=>releaseFirst=r),secondGate=new Promise(r=>releaseSecond=r);
 const events=[0,1].map(i=>sign(31403,{action:'reject',reason:'fixture'},[['d',String(i)]],humanKey,now+i));
 const c=consumer(dir,{async handleGovernanceDecision(e){const i=events.findIndex(x=>x.id===e.id);started.push(i);await (i===0?firstGate:secondGate);completed.push(i);}});c._loadForumState();
 c._onForumGovernance(events[0],'fixture');c._onForumGovernance(events[1],'fixture');
 await new Promise(r=>setImmediate(r));releaseFirst();await new Promise(r=>setImmediate(r));
 assert.deepEqual(started,[0,1]);assert.deepEqual(completed,[0]);
 const saved=JSON.parse(fs.readFileSync(path.join(dir,'cursor.json'),'utf8'));
 assert(saved.seen[events[1].id]);
 const replayed=[];const restarted=consumer(dir,{async handleGovernanceDecision(e){replayed.push(e.id);}});restarted._loadForumState();restarted._onForumGovernance(events[1],'fixture');await restarted._forum.chain;
 assert.equal(replayed.length,0);assert.equal(restarted._metrics.forum_governance_rejected_duplicate,1);
 const result={method:'Snapshot while second handler is blocked; new consumer restores exactly that persisted state. No process was killed.',completedAtSnapshot:[...completed],secondWasPersistedSeen:true,replayedAfterRestore:replayed.length,rejectedDuplicate:restarted._metrics.forum_governance_rejected_duplicate};
 releaseSecond();await c._forum.chain;return result;
}
(async()=>{try{
 process.env.AGENTBOX_FORUM_GOVERNANCE_SIGNERS=human;process.env.AGENTBOX_NPUB='audit-local';process.env.SOLID_POD_ROOT=tmp;
 // Synthetic identities only; never borrow the session's signing identity.
 for(const k of ['AGENTBOX_PUBKEY','AGENTBOX_PUBKEY_HEX','AGENTBOX_X_ONLY_PUBKEY_HEX'])delete process.env[k];
 const results=[await signedCase('digest-mismatch','amend',true),await signedCase('expired-proposal','amend',false),await signedCase('create-title-mismatch','create',false)];
 assert.notEqual(results[0].ledger[0].digest,results[0].proposalDigest);assert(results[2].files['B.md']);assert.equal(results[2].ontology.page,'A');
 console.log(JSON.stringify({method:'Real signature verification, roster, request-by-id verification, orchestrator and installed CLI. Recording relay bridge, Git refusal and attestation recorder. No live relay, D1 projection or UI.',results,queuedReplay:await queuedReplay()},null,2));
}finally{process.chdir(originalCwd);fs.rmSync(tmp,{recursive:true,force:true});}})().catch(e=>{console.error(e);process.exitCode=1;});
