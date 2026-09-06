'use strict';
// Synthetic-only fixtures, fake relay transport/signatures, temporary corpus.
// No service calls, real signing, or edits to the authored vault.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const root = path.resolve(__dirname, '../../../../project/agentbox');
const {createLocalOntology} = require(path.join(root,'mcp/servers/lib/ontology-local.js'));
const retrieval = require(path.join(root,'mcp/servers/lib/ontology-retrieval.js'));
const propose = require(path.join(root,'mcp/servers/ontology-propose.js'));
const {buildAuthorityConsumer} = require(path.join(root,'management-api/lib/authority-consumer.js'));
const telemetry = {record(){},canary(){}};
(async () => {
 const result = {};
 const temp = fs.mkdtempSync(path.join(os.tmpdir(),'estate-agent-'));
 try {
  const body = (name, pub) => '```json-ld\n'+JSON.stringify({'@type':'Page','@id':'urn:page:'+name,'vc:slug':name,'vc:public':pub})+'\n```\n```json-ld\n'+JSON.stringify({'@type':'Class','@id':'urn:ngm:class:'+name,label:name,definition:'Synthetic fixture',domain:'ai',subClassOf:[],relations:{}})+'\n```\n';
  fs.writeFileSync(path.join(temp,'alpha.md'),body('alpha',true));
  fs.writeFileSync(path.join(temp,'private.md'),body('private',false));
  fs.mkdirSync(path.join(temp,'namespace'));
  fs.writeFileSync(path.join(temp,'namespace','nested.md'),body('nested',true));
  const local = createLocalOntology(temp);
  const remoteGuard = propose.axiomAddDescriptor({axiom_type:'SubClassOf',subject:'alpha',object:'private'},{});
  const before = fs.readFileSync(path.join(temp,'alpha.md'),'utf8');
  const edit = local.axiomAdd({axiom_type:'SubClassOf',subject:'alpha',object:'private'});
  result.local_mode = {remote_direct_load_guarded:remoteGuard.guarded,helper_write_result:edit,markdown_changed:before!==fs.readFileSync(path.join(temp,'alpha.md'),'utf8'),private_class_visible:!!local.classGet({iri:'urn:ngm:class:private'}).iri,namespaced_class_visible:!!local.classGet({iri:'urn:ngm:class:nested'}).iri};
  const seeds = [{iri:'urn:test:alpha',label:'Alpha',domain:'ai',maturity:'established',summary:'a'.repeat(3000)},{iri:'urn:test:beta',label:'Beta',domain:'robotics',maturity:'established',summary:'b'.repeat(3000)}];
  const brain = retrieval.createOntologyRetrieval({seedFn:async()=>seeds,telemetry});
  const first=await brain.ask({query:'test',model_tier:'sonnet',mode:'menu',domain:'ai',max_tokens:1000});
  const second=await brain.ask({query:'test',model_tier:'sonnet',mode:'menu',domain:'robotics',max_tokens:50});
  result.cache_constraints={first:{tokens:first.tokens_used,seeds:first.seed_iris},second:{tokens:second.tokens_used,seeds:second.seed_iris,cache_hit:second.cache_hit},second_requested_domain:'robotics',second_requested_max_tokens:50};
  const degraded = retrieval.createOntologyRetrieval({seedFn:async()=>seeds.slice(0,1),expandFn:async()=>{throw new Error('synthetic expansion outage');},telemetry});
  const r=await degraded.ask({query:'test',model_tier:'sonnet',mode:'expand',depth:1});
  result.expand_failure={degraded:r.degraded,tokens:r.tokens_used,error:r.error||null};
  const approver='c'.repeat(64);
  process.env.AGENTBOX_X_ONLY_PUBKEY_HEX=approver;
  const bridge={handlers:[],subscribe(filter,fn){this.handlers.push(fn);},async publish(evt,signer){return signer.sign(evt);}};
  const consumer=buildAuthorityConsumer({manifest:{sovereign_mesh:{}},bridgeFactory:async()=>bridge,signer:{async sign(evt){return {...evt,id:'synthetic-request',pubkey:approver,sig:'fake-test-only'};}},verifyEvent:()=>true,defaultTimeoutMs:20});
  const request=await consumer.publishActionRequest({kind:31402,created_at:1,tags:[['d','synthetic-panel']],content:'{}'});
  const response={kind:31403,id:'synthetic-response',pubkey:approver,tags:[['e',request.id]],content:JSON.stringify({outcome:'approve'})};
  consumer._handleInboundDecision(response);
  const waited=await consumer.awaitDecision(request,{timeoutMs:20});
  result.decision_before_wait={recorded_decided:consumer.isDecided(request.id),wait_returned_decision:waited!==null,wait_result:waited};
  console.log(JSON.stringify(result,null,2));
 } finally {fs.rmSync(temp,{recursive:true,force:true});}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
