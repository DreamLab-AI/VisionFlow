import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.resolve(process.argv[2]); fs.mkdirSync(out,{recursive:true});
const src=path.join(here,'source');
const ab=path.join(src,'agentbox');
const files=['config/hooks/nostr-live-mirror.cjs','config/nostr-gateway/nostr-send.cjs','config/nostr-gateway/gateway.cjs'];
const texts=files.map(f=>fs.readFileSync(path.join(ab,f),'utf8'));
const root='01'.repeat(32);
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const hex=b=>Buffer.from(b).toString('hex');
// Bounded, asserted source slices, copied verbatim. No entrypoint is imported.
const slice=(t,a,b)=>{const start=t.indexOf(a);assert(start>=0);const end=t.indexOf(b,start+a.length);assert(end>start);return t.slice(start,end).trimEnd();};
const pieces={
 mirrorEnv:slice(texts[0],'function envFirst(', '\n\n/** The operator'),
 mirror:slice(texts[0],'let _childCache;', '\n\n/** Sender identity'),
 sendEnv:slice(texts[1],'const envFirst =', '\n// Same identity'),
 send:slice(texts[1],'function loadIdentity()', '\nconst sk = loadIdentity'),
 gatewayEnv:slice(texts[2],'function envFirst(', '\n// Control identity'),
 gateway:slice(texts[2],'const IDENTITY =', '\nconst pub = tools.getPublicKey'),
};
fs.writeFileSync(path.join(out,'extracted-source.json'),JSON.stringify(Object.fromEntries(Object.entries(pieces).map(([k,v])=>[k,{sha256:hash(v),source:v}])),null,2)+'\n');
function context(env={},key=root,implementation=crypto){const state={key};const ctx=vm.createContext({crypto:implementation,Buffer,Uint8Array,process:{env:{...env},stderr:{write:()=>{}}},operatorKeyHex:()=>state.key,log:()=>{},OP_HEX:key,rawSk:Uint8Array.from(Buffer.from(key,'hex'))});return {ctx,state};}
function mirror(env={},key=root,implementation=crypto){const c=context(env,key,implementation);vm.runInContext(pieces.mirrorEnv+'\n'+pieces.mirror,c.ctx);return {...c,get:()=>vm.runInContext('deriveChildKey()',c.ctx)};}
function send(env={},key=root,implementation=crypto){const c=context(env,key,implementation);return vm.runInContext(pieces.sendEnv+'\n'+pieces.send+'\nloadIdentity()',c.ctx);}
function gateway(env={},key=root){const c=context(env,key);return vm.runInContext(pieces.gatewayEnv+'\n'+pieces.gateway+'\nsk',c.ctx);}
const vectors=JSON.parse(fs.readFileSync(path.join(src,'nostr-rust-forum/crates/nostr-bbs-core/tests/vectors/identity-subkey-vectors.v1.json')));
const expected=id=>vectors.vectors.find(v=>v.id===id).expected_child_secret_hex;
const tests=[];function test(name,f){f();tests.push({name,passed:true});}
test('mirror default matches canonical fixture',()=>assert.equal(hex(mirror().get()),expected('canonical-mirror-v1')));
test('send child matches canonical mirror fixture',()=>assert.equal(hex(send({AGENTBOX_GATEWAY_IDENTITY:'child'})),expected('canonical-mirror-v1')));
test('gateway gateway matches canonical gateway fixture',()=>assert.equal(hex(gateway({AGENTBOX_GATEWAY_IDENTITY:'gateway'})),expected('canonical-gateway-v1')));
test('default send and gateway both select raw root',()=>{assert.equal(hex(send()),root);assert.equal(hex(gateway()),root);});
test('child mode diverges between send and gateway',()=>{assert.equal(hex(gateway({AGENTBOX_GATEWAY_IDENTITY:'child'})),root);assert.notEqual(hex(send({AGENTBOX_GATEWAY_IDENTITY:'child'})),root);});
test('gateway mode diverges between send and gateway',()=>{assert.equal(hex(send({AGENTBOX_GATEWAY_IDENTITY:'gateway'})),root);assert.notEqual(hex(gateway({AGENTBOX_GATEWAY_IDENTITY:'gateway'})),root);});
test('environment tags are trimmed before derivation',()=>assert.equal(hex(mirror({AGENTBOX_MIRROR_KEY_TAG:' agentbox-mirror-v1 '}).get()),expected('canonical-mirror-v1')));
test('warm mirror cache survives root and tag changes',()=>{const m=mirror();const first=hex(m.get());m.ctx.process.env.AGENTBOX_MIRROR_KEY_TAG='agentbox-mirror-v2';m.state.key='02'.repeat(32);assert.equal(hex(m.get()),first);});
test('mirror cached off result survives reenable',()=>{const m=mirror({AGENTBOX_MIRROR_CHILD:'0'});assert.equal(m.get(),null);m.ctx.process.env.AGENTBOX_MIRROR_CHILD='1';assert.equal(m.get(),null);});
test('all-zero root passes mirror lexical validation',()=>assert.equal(Buffer.from(mirror({},'00'.repeat(32)).get()).length,32));
test('all-zero root passes send child lexical validation',()=>assert.equal(Buffer.from(send({AGENTBOX_GATEWAY_IDENTITY:'child'},'00'.repeat(32))).length,32));
test('forced zero HMAC output is not scalar-checked in mirror or send',()=>{const fake={createHmac:()=>({update(){return this;},digest(){return Buffer.alloc(32);}})};assert.equal(hex(mirror({},root,fake).get()),'00'.repeat(32));assert.equal(hex(send({AGENTBOX_GATEWAY_IDENTITY:'child'},root,fake)),'00'.repeat(32));});
// Run the unchanged forum runner against disposable source copies. Mutations are
// adversarial controls, never written to a live checkout.
const runner='scripts/identity-vector-parity.mjs';const fixture='crates/nostr-bbs-core/tests/vectors/identity-subkey-vectors.v1.json';
const scenarios=[];
function scenario(name,mutate,pass){const dir=path.join(out,name);fs.mkdirSync(dir,{recursive:true});const forum=path.join(dir,'forum'),agentbox=path.join(dir,'agentbox');
 for(const f of [runner,fixture]){const to=path.join(forum,f);fs.mkdirSync(path.dirname(to),{recursive:true});fs.copyFileSync(path.join(src,'nostr-rust-forum',f),to);}
 for(const f of files){const to=path.join(agentbox,f);fs.mkdirSync(path.dirname(to),{recursive:true});fs.copyFileSync(path.join(ab,f),to);}
 const extra=mutate({dir,forum,agentbox})||{};const r=spawnSync(process.execPath,[path.join(forum,runner)],{env:{AGENTBOX_ROOT:agentbox},encoding:'utf8'});const result=JSON.parse(r.stdout);assert.equal(r.status,pass?0:1,name);assert.equal(result.ok,pass,name);fs.writeFileSync(path.join(dir,'result.json'),r.stdout);scenarios.push({name,expected_success:pass,exit_code:r.status,summary:result,...extra});}
scenario('baseline',()=>{},true);
scenario('absent-agentbox',({agentbox})=>fs.rmSync(agentbox,{recursive:true}),true);
scenario('expressions-only-in-comments',({agentbox})=>{for(let i=0;i<files.length;i++){const expr=i<2?"crypto.createHmac('sha256', Buffer.from(hex, 'hex')).update(tag).digest()":"crypto.createHmac('sha256', Buffer.from(OP_HEX, 'hex')).update(";fs.writeFileSync(path.join(agentbox,files[i]),'// '+expr+'\n// No executable derivation.\n');}},true);
scenario('changed-gateway-default',({agentbox})=>{const p=path.join(agentbox,files[2]);const t=fs.readFileSync(p,'utf8').replace("envFirst('AGENTBOX_GATEWAY_KEY_TAG') || 'agentbox-gateway-v1'","envFirst('AGENTBOX_GATEWAY_KEY_TAG') || 'different-default-v1'");fs.writeFileSync(p,t);const block=slice(t,'const IDENTITY =','\nconst pub = tools.getPublicKey');const c=context({AGENTBOX_GATEWAY_IDENTITY:'gateway'});const changed=vm.runInContext(pieces.gatewayEnv+'\n'+block+'\nsk',c.ctx);assert.notEqual(hex(changed),expected('canonical-gateway-v1'));return {actual_extracted_selection_changed:true};},true);
scenario('unrelated-hkdf-comment',({agentbox})=>fs.appendFileSync(path.join(agentbox,files[0]),'\n// HKDF is unrelated to this change.\n'),false);
scenario('missing-expression',({agentbox})=>fs.writeFileSync(path.join(agentbox,files[0]),'// No derivation here.\n'),false);
scenario('corrupt-vector',({forum})=>{const p=path.join(forum,fixture);const f=JSON.parse(fs.readFileSync(p));f.vectors[0].expected_child_secret_hex='00'.repeat(32);fs.writeFileSync(p,JSON.stringify(f));},false);
scenario('missing-fixture',({forum})=>fs.unlinkSync(path.join(forum,fixture)),false);
const report={schema:'nf-derivation-audit-v1',node:process.version,selection_tests:tests,runner_scenarios:scenarios,limits:['Extracted selection code with synthetic roots and injected operatorKeyHex/env; no live secret reads, service imports, sockets or messages.','Forced zero HMAC is a mock boundary, not a discovered real preimage. Gateway pre-derivation getPublicKey is outside fixture.','Cache tests are same-module calls, not production process reload tests.','Passing adversarial scenarios demonstrate current validation gaps, not fixes.']};
fs.writeFileSync(path.join(out,'audit-results.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({selection_tests:tests.length,runner_scenarios:scenarios.length,passed:true}));
