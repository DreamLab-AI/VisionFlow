#!/usr/bin/env node
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),yaml=require('js-yaml');
const [census,out]=process.argv.slice(2);
if(!census||!out){console.error('usage: estate-adr-graph.cjs <census.json> <output.json>');process.exit(2);}
const workspace=path.resolve(__dirname,'../..');const inventory=JSON.parse(fs.readFileSync(census));
const resolutions=JSON.parse(fs.readFileSync(path.resolve(__dirname,'../docs/architecture/adr-reference-resolutions.json'))).references;
const excluded=inventory.records.filter(r=>r.repo==='RuView'&&r.kind==='operative-candidate').map(r=>({key:r.key,reason:'Excluded from execution by user instruction; no current source read.'}));
const nodes=[],edges=[],unresolved=[],readErrors=[];const candidates=inventory.records.filter(r=>['operative-candidate','historical'].includes(r.kind));
const lookup=new Map();
for(const r of candidates){const key=`${r.repo}:${r.id}`;const hits=lookup.get(key)||[];hits.push(r.key);lookup.set(key,hits);}
for(const r of candidates.filter(r=>r.kind==='operative-candidate'&&r.repo!=='RuView')){
 try{
 const source=fs.readFileSync(path.join(workspace,r.repo,r.path),'utf8');const match=source.match(/^---\r?\n([\s\S]*?)\r?\n---/);const fm=match?yaml.load(match[1])||{}:{};
 nodes.push({key:r.key,id:`${r.repo}:${fm.id||r.id}`,source_sha256:crypto.createHash('sha256').update(source).digest('hex'),status:{decision:fm.decision_status||null,implementation:fm.implementation_status||null,activation:fm.activation_status||null}});
 const refs=[...(Array.isArray(fm.supersedes)?fm.supersedes:[]).map(id=>({kind:'supersedes',id})),...(Array.isArray(fm.superseded_by)?fm.superseded_by:[]).map(id=>({kind:'superseded_by',id})),...[...String(fm.lineage||'').matchAll(/\bADR-[A-Za-z0-9]+(?:-[0-9]+)*\b/g)].map(m=>({kind:'lineage-mention',id:m[0]}))];
 for(const ref of refs.flatMap(ref=>{const range=/^ADR-(\d+)-(\d+)$/.exec(ref.id);return range&&+range[2]>+range[1]&&+range[2]-+range[1]<100?Array.from({length:+range[2]-+range[1]+1},(_,i)=>({...ref,id:'ADR-'+String(+range[1]+i).padStart(range[1].length,'0')})):[ref];})){
 const qualified=ref.id.includes(':')?ref.id:`${r.repo}:${ref.id}`;const hits=lookup.get(qualified)||[];const edge={from:r.key,relation:ref.kind,reference:qualified};
 const resolution=resolutions[r.key+'|'+qualified];
 if(resolution?.target.startsWith('https://')){if(!/^[0-9a-f]{64}$/.test(resolution.sha256||''))throw Error('remote reference requires recorded content hash');edges.push({...edge,to:resolution.target,scope:resolution.scope,resolved_by:'pinned-upstream-receipt',target_sha256:resolution.sha256});}
 else if(resolution){const split=resolution.target.indexOf(':');const target=path.join(workspace,resolution.target.slice(0,split),resolution.target.slice(split+1));const bytes=fs.readFileSync(target);edges.push({...edge,to:resolution.target,scope:resolution.scope,resolved_by:'explicit-file-reference',target_sha256:crypto.createHash('sha256').update(bytes).digest('hex')});}
 else if(hits.length===1)edges.push({...edge,to:hits[0]});else unresolved.push({...edge,reason:hits.length?'ambiguous':'not-in-census',candidates:hits});}
 }catch(error){readErrors.push({key:r.key,error:String(error)});}
}
const report={method:'Repository-qualified declared supersession and frontmatter lineage graph, read from current bytes. A lineage mention is not promoted to supersession or acceptance. Missing/ambiguous historical references require section-level disposition.',excluded,nodes,edges,unresolved,readErrors};fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({nodes:nodes.length,excluded:excluded.length,edges:edges.length,unresolved:unresolved.length,readErrors:readErrors.length}));process.exitCode=readErrors.length||unresolved.length?1:0;
