import { afterEach, expect, test, vi } from 'vitest';
import { gzipSync } from 'node:zlib';
import { assertScope, decodeScope } from '../src/types/scope';
import { buildOverviewScope } from '../src/stores/scopeStore';
const doc = {id:'fixture', title:'Fixture Page', definition:'index definition'};
const json = (value: unknown) => new Response(JSON.stringify(value), {headers:{'content-type':'application/json'}});
afterEach(() => vi.unstubAllGlobals());
async function service() { vi.resetModules(); return import('../src/api/pageService'); }
test('same-module page caches retain prior corpus prose', async () => {
 let generation=1; const calls:string[]=[];
 vi.stubGlobal('fetch', async (u:string) => {calls.push(u); return u==='/api/search-index.json'?json([doc]):u.endsWith('.gz')?new Response('',{status:404}):u.startsWith('/api/pages/')?json({definition:'API'}):new Response(`generation ${generation}`);});
 const s=await service(); expect((await s.fetchPage('fixture')).content).toBe('generation 1'); const before=calls.length;generation=2;
 expect((await s.fetchPage('fixture')).content).toBe('generation 1');expect(calls.length).toBe(before);
});
test('index outage produces a concept stub without trying page data',async()=>{
 const calls:string[]=[];vi.stubGlobal('fetch',async(u:string)=>{calls.push(u);return new Response('',{status:503});});
 const p=await(await service()).fetchPage('fixture');expect(p.properties.status).toBe('stub');expect(p.content).toContain('exists in the ontology');expect(calls).toEqual(['/api/search-index.json']);
});
test('malformed index title rejects outside fetch handling',async()=>{
 vi.stubGlobal('fetch',async()=>json([{id:'different',title:null}]));await expect((await service()).fetchPage('fixture')).rejects.toThrow(TypeError);
});
test('missing mirror omits API body but now links to Quartz without external fallback',async()=>{
 const calls:string[]=[];vi.stubGlobal('fetch',async(u:string)=>{calls.push(u);return u==='/api/search-index.json'?json([doc]):u.endsWith('.gz')?new Response('',{status:404}):u.startsWith('/api/pages/')?json({body:'full body canary',definition:'API'}):new Response('',{status:500});});
 const p=await(await service()).fetchPage('fixture');expect(p.content).toContain('/notes/pages/Fixture-Page');expect(p.content).not.toContain('full body canary');expect(calls.every(u=>u.startsWith('/'))).toBe(true);
});
test('valid gzip representation is read successfully',async()=>{
 const calls:string[]=[];vi.stubGlobal('fetch',async(u:string)=>{calls.push(u);return new Response(gzipSync('compressed canary'));});
 const {fetchArtefact}=await import('../src/api/fetchArtefact');expect(await(await fetchArtefact('/api/markdown/X.md')).text()).toBe('compressed canary');expect(calls).toEqual(['/api/markdown/X.md.gz']);
});
test('gzip magic with a corrupt stream rejects during body consumption and does not retry raw',async()=>{
 const calls:string[]=[];vi.stubGlobal('fetch',async(u:string)=>{calls.push(u);return u.endsWith('.gz')?new Response(Uint8Array.from([31,139,8,0,0,0,0,0,0,3,255])):new Response('raw valid');});
 const {fetchArtefact}=await import('../src/api/fetchArtefact');const response=await fetchArtefact('/api/markdown/X.md');await expect(response.text()).rejects.toThrow();expect(calls).toEqual(['/api/markdown/X.md.gz']);
});
test('scope numeric guard refuses upper cap but accepts negative fractional and NaN counts',()=>{
 expect(()=>assertScope({nodeCount:1501,edgeCount:0,tier:'domain',phase:'baked'})).toThrow(RangeError);
 for(const n of [-1,0.5,NaN])expect(()=>assertScope({nodeCount:n,edgeCount:n,tier:'focus',phase:'settling'})).not.toThrow();
});
test('overview builder accepts invalid array lengths and edge endpoints',()=>{
 const {scope}=buildOverviewScope({ids:Uint32Array.of(1),positions:new Float32Array(0),domainOf:new Uint16Array(0),categoryOf:new Uint16Array(0),degreeOf:new Float32Array(0),flagsOf:new Uint8Array(0),labels:[],iris:[],edgeTypes:Uint8Array.of(0),edgePairs:Uint32Array.of(99,100)});
 expect(scope.nodeCount).toBe(1);expect(scope.positions.length).toBe(0);expect(scope.edgePairs[0]).toBe(99);
});
test('ninth domain resolves while unknown domain falls back to overview',()=>{
 expect(decodeScope({scope:'domain:supply-chains-and-logistics'})).toEqual({tier:'domain',slug:'supply-chains-and-logistics'});expect(decodeScope({scope:'domain:unknown'})).toEqual({tier:'overview'});
});
