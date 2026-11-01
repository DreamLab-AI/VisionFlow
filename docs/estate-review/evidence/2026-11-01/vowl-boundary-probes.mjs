// Audit-only assertions of observed behaviour; passing does not mean defects are fixed.
// VOWL_AUDIT_ROOT=/path/to/fresh-source-archive node --test this-file.mjs
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
const root=process.env.VOWL_AUDIT_ROOT || '/tmp/vf-vowl-source';
const js=pathToFileURL(`${root}/pkg/vowl_wasm.js`);
const wasm=readFileSync(`${root}/pkg/vowl_wasm_bg.wasm`);
const {initSync,WebVowl,NggExplorer}=await import(js.href);
const output=initSync({module:wasm});
const golden=new Uint8Array(readFileSync(`${root}/tests/fixtures/ngg1-3n2e.bin`));
const ontology=(ids)=>JSON.stringify({class:ids.map(id=>({id,label:id,type:'owl:Class'})),property:[]});
function graph(ids=['a','b']) {const x=new WebVowl();x.loadOntology(ontology(ids));x.initSimulation();return x;}
test('replacement retains obsolete pin IDs and settled alpha',()=>{
 const x=graph();x.pinNode('a',1000,2000);x.runSimulation(1000);assert(x.isFinished());
 x.loadOntology(ontology(['c']));assert.deepEqual(x.getNodeIds(),['c']);
 assert.equal(x.isPinned('a'),true);assert.equal(x.getPinnedCount(),1);assert(x.isFinished());
 const before=x.getNodePositions();x.tick();assert.deepEqual(x.getNodePositions(),before);x.free();
});
test('runSimulation reseeds a pinned node on every call',()=>{
 const x=graph();x.pinNode('a',1000,2000);assert(x.isPinned('a'));x.runSimulation(1);
 assert(x.isPinned('a'));assert.notDeepEqual(Array.from(x.getNodePositions().slice(0,2)),[1000,2000]);x.free();
});
test('hidden nodes still participate in click hit testing',()=>{
 const x=graph(['a']);x.pinNode('a',0,0);x.filterByDegree(99);
 assert.equal(x.checkNodeClick(new Float32Array([0,0,100]),new Float32Array([0,0,-1])),'a');x.free();
});
test('legacy pin accepts NaN whereas CSR pin refuses it',()=>{
 const x=graph();x.pinNode('a',NaN,1);assert(x.isPinned('a'));assert(Number.isNaN(x.getNodePositions()[0]));x.free();
 const c=new NggExplorer();c.loadCsr(golden);assert.equal(c.pinNode(0,NaN,1),false);c.free();
});
test('odd buffer contracts differ: legacy writes pairs, CSR writes individual floats',()=>{
 const x=graph();const a=new Float32Array(3).fill(999);assert.equal(x.writePositionsInto(a),2);assert.equal(a[2],999);x.free();
 const c=new NggExplorer();c.loadCsr(golden);const b=new Float32Array(3).fill(999);c.writePositionsInto(b);assert.deepEqual(Array.from(b),[0,0,1]);c.free();
});
test('CSR infinite reheat produces non-finite positions and does not finish',()=>{
 const c=new NggExplorer();c.loadCsr(golden);c.reheat(Infinity);c.tick();const a=new Float32Array(6);c.writePositionsInto(a);
 assert.equal(c.getAlpha(),Infinity);assert.equal(c.isFinished(),false);assert(a.some(v=>!Number.isFinite(v)));c.free();
});
test('negative alphaMin prevents CSR settling through 10000 ticks',()=>{
 const c=new NggExplorer();c.loadCsr(golden);c.setParam('alphaMin',-1);for(let i=0;i<10000;i++)c.tick();assert.equal(c.isFinished(),false);c.free();
});
test('ordinary memory growth detaches a previously acquired pointer view',()=>{
 const c=new NggExplorer();c.loadCsr(golden);const view=new Float32Array(output.memory.buffer,c.positionsPtr(),c.positionsLen());assert.equal(view.length,6);
 output.memory.grow(1);assert.equal(view.length,0);const renewed=new Float32Array(output.memory.buffer,c.positionsPtr(),c.positionsLen());assert.equal(renewed.length,6);c.free();
});
test('generated copying glue calls malloc once per reusable-buffer call',async()=>{
 const source=readFileSync(js,'utf8');const signature='function passArrayF32ToWasm0(arg, malloc) {';assert(source.includes(signature));
 const instrumented=source.replace(signature,'export let auditMallocCalls = 0;\n'+signature+'\n auditMallocCalls++;');
 const dest=`${root}/pkg/audit-instrumented.mjs`;writeFileSync(dest,instrumented);const m=await import(pathToFileURL(dest));m.initSync({module:wasm});
 const x=new m.WebVowl();x.loadOntology(ontology(['a']));x.initSimulation();const out=new Float32Array(2);const before=m.auditMallocCalls;
 for(let i=0;i<3;i++)x.writePositionsInto(out);assert.equal(m.auditMallocCalls-before,3);x.free();
});
test('coincident legacy Barnes-Hut nodes trap in an isolated child',()=>{
 const script=`import {readFileSync} from 'node:fs';import {initSync,WebVowl} from ${JSON.stringify(js.href)};initSync({module:readFileSync(${JSON.stringify(root+'/pkg/vowl_wasm_bg.wasm')})});const x=new WebVowl();x.loadOntology(${JSON.stringify(ontology(Array.from({length:51},(_,i)=>String(i))))});x.initSimulation();for(let i=0;i<51;i++)x.pinNode(String(i),0,0);x.tick();console.log('completed');`;
 const child=spawnSync(process.execPath,['--input-type=module','-e',script],{encoding:'utf8',timeout:10000,maxBuffer:20000});
 assert.equal(child.error,undefined);assert.notEqual(child.status,0);assert.match(child.stderr,/RuntimeError: memory access out of bounds/);console.log(child.stderr.split('\n').slice(0,6).join('\n'));
});
