// Uses the real TypeScript worker, real fresh WASM and deterministic worker-global shims.
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const root=process.env.VOWL_AUDIT_ROOT || '/tmp/vf-vowl-source';
const consumer=process.env.VOWL_CONSUMER_ROOT || '/home/devuser/workspace/visionGraph/publishing-tools/WasmVOWL/modern';
const {build}=await import(pathToFileURL(`${consumer}/node_modules/esbuild/lib/main.js`));
const bundle=`${root}/audit-worker.mjs`;
await build({entryPoints:[`${consumer}/src/workers/physics.worker.ts`],outfile:bundle,bundle:true,platform:'node',format:'esm',external:['@dreamlab-ai/vowl-wasm']});
const wrapper=`${root}/audit-glue.mjs`;
writeFileSync(wrapper,`import {readFileSync} from 'node:fs';import {initSync} from './pkg/vowl_wasm.js';export {NggExplorer} from './pkg/vowl_wasm.js';export default async function(){return initSync({module:readFileSync(new URL('./pkg/vowl_wasm_bg.wasm',import.meta.url))});}`);
const golden=readFileSync(`${root}/tests/fixtures/ngg1-3n2e.bin`);
async function worker(tag){
 const messages=[],timers=[];
 const ctx={postMessage:(msg,transfer=[])=>messages.push(structuredClone(msg,{transfer})),setTimeout:fn=>timers.push(fn)};
 globalThis.self=ctx;await import(pathToFileURL(bundle).href+`?${tag}`);
 const send=async data=>{ctx.onmessage({data});for(let i=0;i<10;i++)await new Promise(setImmediate);};
 await send({type:'init',wasmUrl:pathToFileURL(wrapper).href,forceTransport:'pingpong'});
 assert.equal(messages[0]?.type,'ready');
 return {messages,timers,send};
}
test('failed scope load relabels old physics and transport with the new generation',async()=>{
 const {messages,timers,send}=await worker('replacement');
 await send({type:'loadScope',generation:1,nodeCount:3,buffer:Uint8Array.from(golden).buffer});assert.equal(messages.at(-1).generation,1);
 await send({type:'loadScope',generation:2,nodeCount:1,buffer:new ArrayBuffer(0)});assert.equal(messages.at(-1).type,'workerError');
 await send({type:'start'});assert.equal(timers.length,1);timers.shift()();
 const frame=messages.at(-1);assert.equal(frame.type,'positions');assert.equal(frame.generation,2);assert.equal(frame.buffer.byteLength,24);
 await send({type:'stop'});
});
test('loadScope trusts a caller count inconsistent with the validated tier',async()=>{
 const {messages,send}=await worker('count');
 await send({type:'loadScope',generation:7,nodeCount:1,buffer:Uint8Array.from(golden).buffer});
 const frame=messages.at(-1);assert.equal(frame.type,'positions');assert.equal(frame.buffer.byteLength,8);assert.equal(frame.generation,7);
});
test('setParam allows infinite reheat through the worker message boundary',async()=>{
 const {messages,timers,send}=await worker('parameter');
 await send({type:'loadScope',generation:8,nodeCount:3,buffer:Uint8Array.from(golden).buffer});
 await send({type:'setParam',key:'reheat',val:Infinity});await send({type:'start'});timers.shift()();
 const frame=messages.at(-1);assert.equal(frame.alpha,Infinity);assert(new Float32Array(frame.buffer).some(v=>!Number.isFinite(v)));await send({type:'stop'});
});
