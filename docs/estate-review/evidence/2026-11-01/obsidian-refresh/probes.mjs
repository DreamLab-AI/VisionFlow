// Usage: node probes.mjs /absolute/path/to/plugin-source
// Real modules, fake Obsidian elements/vault and delayed API. No network or desktop.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(process.argv[2]);
const require=createRequire(join(root,'package.json'));
const {build}=require('esbuild');
const dir=mkdtempSync(join(tmpdir(),'obsidian-refresh-probes-'));
globalThis.window=globalThis;
try {
 const stub=join(dir,'obsidian.mjs');
 writeFileSync(stub,'export class MarkdownRenderChild {constructor(el){this.containerEl=el}}\nexport class ItemView {}\n');
 const modules={};
 for(const name of ['sparql-block','sparql-rewrite','guard','api']) {
  const outfile=join(dir,name+'.mjs');
  await build({entryPoints:[join(root,'src',name+'.ts')],outfile,bundle:true,platform:'node',format:'esm',alias:{obsidian:stub},logLevel:'silent'});
  modules[name]=await import(pathToFileURL(outfile));
 }
 const {SparqlBlock}=modules['sparql-block'];
 const {GenerationGuard}=modules.guard;
 const {rewriteSparql}=modules['sparql-rewrite'];
 const {OntomarkApi}=modules.api;
 const tick=()=>new Promise(resolve=>setImmediate(resolve));
 const element=()=>({writes:[],addClass(){},empty(){},createDiv(o){this.writes.push(o.text);return this;}});
 const guard=new GenerationGuard();
 const pending=[];
 let id=0;
 const makeApi=()=>({listeners:new Set(),onChange(f){this.listeners.add(f);return()=>this.listeners.delete(f)},query(q,d,l,channel){const request=++id;guard.issued(channel,request);return new Promise((resolve,reject)=>pending.push(()=>{const v=guard.judge(guard.epoch,channel,request,guard.generation);if(v.accept)resolve({form:'ask',boolean:true});else reject(new Error(v.reason));}));}});
 const oldApi=makeApi();
 const plugin={api:oldApi,settings:{queryLimit:50}};
 const one=new SparqlBlock(element(),plugin,'ASK {}','same.md');
 const two=new SparqlBlock(element(),plugin,'ASK {}','same.md');
 one.onload();two.onload();await tick();
 assert.equal(pending.length,2);pending.splice(0).forEach(f=>f());await Promise.all([one.ready,two.ready]);
 assert.equal(one.lastError,'error: superseded');assert.equal(two.lastResult.boolean,true);
 const results=[{case:'same-note simultaneous blocks',first:one.lastError,second:two.lastResult}];
 plugin.api=makeApi();
 results.push({case:'API replaced while blocks loaded',old_subscribers:oldApi.listeners.size,new_subscribers:plugin.api.listeners.size});
 assert.equal(plugin.api.listeners.size,0);one.onunload();two.onunload();
 const el=element();const three=new SparqlBlock(el,plugin,'ASK {}','another.md');three.onload();await tick();three.onunload();
 const before=el.writes.length;pending.splice(0).forEach(f=>f());await three.ready;
 assert.deepEqual(el.writes.slice(before),['true']);results.push({case:'query completes after block unload',writes_after_unload:el.writes.slice(before)});
 const rewrite=rewriteSparql('SELECT ?__THIS__ WHERE { ?__THIS__ ?p ?o }',()=> 'urn:note');
 assert.equal(rewrite.sparql,'SELECT ?<urn:note> WHERE { ?<urn:note> ?p ?o }');results.push({case:'variable containing placeholder',result:rewrite});
 const trailing=rewriteSparql('ASK {}\n# dataset: inferred',()=>null);assert.equal(trailing.dataset,'inferred');results.push({case:'non-leading dataset directive',dataset:trailing.dataset});
 let text='base';
 const api=new OntomarkApi({generation:0,upsert:async()=>{throw new Error('injected index failure')}},{process:async(p,fn)=>{text=fn(text);return text;}});
 let error;
 try {await api.applyPatch({path:'note.md',base_text:'base',new_text:'changed',new_digest:'digest',no_op:false},async()=>true)}catch(e){error=e.message}
 assert.equal(error,'injected index failure');assert.equal(text,'changed');results.push({case:'index fails after vault persistence',persisted_text:text,error});
 console.log(JSON.stringify({method:'actual source modules with bounded host doubles; assertions describe reproduced defects',cases:results},null,2));
}finally{rmSync(dir,{recursive:true,force:true})}
