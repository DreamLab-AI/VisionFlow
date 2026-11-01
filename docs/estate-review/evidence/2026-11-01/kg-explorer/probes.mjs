import {test} from 'node:test';
import assert from 'node:assert/strict';
import {MessageChannel} from 'node:worker_threads';
import {once} from 'node:events';
import {DoubleBuffer,copyPositions} from './built/workers/positionTransport.mjs';
import {assertScope,decodeScope} from './built/types/scope.mjs';
import {buildOverviewScope} from './built/stores/scopeStore.mjs';

await test('same-object pool return works as existing tests expect',()=>{
 const p=new DoubleBuffer(8),b=p.acquire();assert.equal(p.release(b),true);assert.equal(p.freeCount,2);
});
await test('real transferred returns are rejected and exhaust the two-buffer pool',async()=>{
 const p=new DoubleBuffer(8);const {port1,port2}=new MessageChannel();
 try {
  port2.on('message',b=>port2.postMessage(b,[b]));
  for(let i=0;i<2;i++){
   const original=p.acquire();new Float32Array(original).set([i,42]);
   const pending=once(port1,'message');port1.postMessage(original,[original]);
   const [returned]=await pending;
   assert.equal(original.byteLength,0);assert.notEqual(returned,original);
   assert.deepEqual([...new Float32Array(returned)],[i,42]);
   assert.equal(p.release(returned),false);
  }
  assert.equal(p.freeCount,0);assert.equal(p.acquire(),undefined);
 }finally{port1.close();port2.close();}
});
await test('short position view retains old tail values',()=>{
 const dst=Float32Array.from([1,2,900,901]);copyPositions(Float32Array.from([3,4]),dst);
 assert.deepEqual([...dst],[3,4,900,901]);
});
await test('scope cap rejects ordinary overflow but accepts invalid numeric counts',()=>{
 assert.throws(()=>assertScope({nodeCount:1501,edgeCount:0,tier:'domain',phase:'baked'}),RangeError);
 for(const value of [-1,NaN,0.5])assert.doesNotThrow(()=>assertScope({nodeCount:value,edgeCount:value,tier:'focus',phase:'settling'}));
});
await test('overview builder publishes inconsistent array shapes and endpoints',()=>{
 const {scope}=buildOverviewScope({ids:Uint32Array.of(1),positions:new Float32Array(0),domainOf:new Uint16Array(0),categoryOf:new Uint16Array(0),degreeOf:new Float32Array(0),flagsOf:new Uint8Array(0),labels:[],iris:[],edgeTypes:Uint8Array.of(0),edgePairs:Uint32Array.of(99,100)});
 assert.equal(scope.nodeCount,1);assert.equal(scope.positions.length,0);assert.equal(scope.edgeCount,1);assert.equal(scope.edgePairs[0],99);
});
await test('unknown domain link silently becomes overview',()=>{
 assert.deepEqual(decodeScope({scope:'domain:space'}),{tier:'overview'});
});
let nonce=0;
async function service(){return import(`./built/api/pageService.mjs?fixture=${++nonce}`);}
const originalFetch=globalThis.fetch;
const doc={id:'fixture-page',title:'Fixture Page',definition:'index definition'};
function json(v){return new Response(JSON.stringify(v),{headers:{'content-type':'application/json'}});}
await test('non-404 primary failure also triggers external markdown fallback',async()=>{
 const calls=[];globalThis.fetch=async u=>{calls.push(u);if(u==='/api/search-index.json')return json([doc]);if(u.startsWith('/api/pages/'))return json({id:doc.id,definition:'api definition'});if(u.startsWith('/api/markdown/'))return new Response('',{status:500});if(u.startsWith('https://raw.githubusercontent.com/'))return new Response('remote canary',{headers:{'content-type':'text/plain'}});throw new Error(`Unexpected mocked URL ${u}`);};
 try{const s=await service();const page=await s.fetchPage(doc.id);assert.equal(page.content,'remote canary');assert.equal(calls.filter(x=>x.startsWith('https://')).length,1);}finally{globalThis.fetch=originalFetch;}
});
await test('page API body is discarded when markdown is absent',async()=>{
 globalThis.fetch=async u=>u==='/api/search-index.json'?json([doc]):u.startsWith('/api/pages/')?json({id:doc.id,body:'full body canary',definition:'api definition'}):new Response('',{status:404});
 try{const page=await(await service()).fetchPage(doc.id);assert.equal(page.content,'## Definition\n\napi definition');assert.ok(!page.content.includes('full body canary'));}finally{globalThis.fetch=originalFetch;}
});
await test('failed search index reports known requested slug as content stub without trying page API',async()=>{
 const calls=[];globalThis.fetch=async u=>{calls.push(u);return new Response('',{status:503});};
 try{const page=await(await service()).fetchPage(doc.id);assert.equal(page.properties.status,'stub');assert.match(page.content,/exists in the ontology graph/);assert.deepEqual(calls,['/api/search-index.json']);}finally{globalThis.fetch=originalFetch;}
});
await test('successful page and index caches never observe a new generation in the same module',async()=>{
 let generation=1,calls=0;globalThis.fetch=async u=>{calls++;return u==='/api/search-index.json'?json([doc]):u.startsWith('/api/pages/')?json({id:doc.id,definition:`definition ${generation}`}):new Response(`markdown generation ${generation}`,{headers:{'content-type':'text/plain'}});};
 try{const s=await service();const first=await s.fetchPage(doc.id);const before=calls;generation=2;const second=await s.fetchPage(doc.id);assert.equal(calls,before);assert.equal(first.content,second.content);assert.equal(second.content,'markdown generation 1');}finally{globalThis.fetch=originalFetch;}
});
await test('malformed search document escapes fetch boundary as a TypeError',async()=>{
 globalThis.fetch=async()=>json([{id:'different',title:null}]);
 try{await assert.rejects((await service()).fetchPage(doc.id),TypeError);}finally{globalThis.fetch=originalFetch;}
});
await test('visionGraph counterpart reclaims transferred buffers across repeated round trips',async()=>{
 const {DoubleBuffer:OtherPool}=await import('./built/visiongraph-transport.mjs');
 const p=new OtherPool(8),{port1,port2}=new MessageChannel();
 try{port2.on('message',b=>port2.postMessage(b,[b]));for(let i=0;i<4;i++){const b=p.acquire();assert.ok(b);const pending=once(port1,'message');port1.postMessage(b,[b]);const [returned]=await pending;assert.equal(p.release(returned),true);assert.equal(p.freeCount,2);}}finally{port1.close();port2.close();}
});
