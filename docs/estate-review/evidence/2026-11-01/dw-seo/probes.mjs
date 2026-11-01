import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import * as C from './scripts/seo-watch/lib/checks.mjs';
import {validatePatch,publish} from './scripts/seo-watch/lib/propose.mjs';
import {runAudit,fetchPage} from './scripts/seo-watch/lib/audit.mjs';
import {decideAdvise} from './scripts/seo-watch/lib/advise.mjs';
import {probePresence} from './scripts/seo-watch/lib/presence.mjs';

const results=[];
async function test(name,fn){const details=await fn();results.push({name,pass:true,details});}
const config=JSON.parse(await fs.readFile('./scripts/seo-watch/config.json','utf8'));
const origin=config.origin;
await test('duplicate robots groups are not combined',()=>{
 const p=C.parseRobots('User-agent: Googlebot\nAllow: /\nUser-agent: Googlebot\nDisallow: /private\n');
 assert.equal(C.robotsAllows(p.groups,'Googlebot','/private'),true);
 return {observed:'allowed',rfc9309Expected:'disallowed'};
});
await test('prefix lookalike origin passes both classifiers',()=>{
 const url=origin+'.audit.invalid/';
 assert(!C.analyseSitemap(`<loc>${url}</loc>`,{origin,minUrls:1}).out.some(x=>x.check==='sitemap-foreign-origin'));
 assert(!C.analyseLlms(`# Site\n> Summary\n[link](${url})`,{origin}).out.some(x=>x.check==='llms-foreign-links'));
 return {url};
});
await test('equal length replacement and missing headings evade render comparison',()=>{
 const raw={title:'Original',h1:'Original',textLength:1000,textHash:'a'};
 const rendered={title:'',h1:'',textLength:1000,textHash:'b'};
 assert.deepEqual(C.compareRender({url:origin,raw,rendered}),[]);
 return {findings:0};
});
await test('Lighthouse gaps and exactly five points count as two nights',()=>{
 const rows=[{date:'2026-01-01',scores:{seo:100}},{date:'2026-06-01',scores:{seo:95}}];
 const out=C.lighthouseRegressions({[origin]:rows},{[origin]:{seo:95}});
 assert(out.some(x=>x.check==='lighthouse-seo-regression'));return out;
});
await test('one token below budget still permits advice',()=>{
 const out=decideAdvise({fingerprint:'b',lastFingerprint:'a',regressions:0,force:false,now:new Date('2026-11-03'),config:config.glm,spentTokens:5999999});
 assert.equal(out.call,true);return out;
});
await test('response body read failure propagates',async()=>{
 await assert.rejects(fetchPage(origin,{fetchImpl:async()=>({status:200,headers:new Headers(),text:async()=>{throw Error('fixture-body-failure')}})}),/fixture-body-failure/);
 await assert.rejects(runAudit({config,state:{},browser:false,fetchImpl:async()=>({status:200,headers:new Headers(),text:async()=>{throw Error('fixture-body-failure')}})}),/fixture-body-failure/);
 return {fetchPageRejected:true,wholeAuditRejected:true};
});
await test('failed baselines retain planned parity coverage with no agent requests',async()=>{
 const calls=[];const audit=await runAudit({config,state:{},browser:false,fetchImpl:async(url,opts)=>{calls.push({url,ua:opts.headers['user-agent']});return new Response('',{status:503});}});
 const agentCalls=calls.filter(c=>config.agents.some(a=>a.ua===c.ua));
 assert.equal(agentCalls.length,0);assert.equal(audit.parity.pages,4);assert.equal(audit.parity.findings,0);
 assert(audit.findings.some(f=>f.check==='page-status'));return {parity:audit.parity,agentRequests:0,highPageStatusPresent:true};
});
await test('lookalike llms origin is actually fetched by audit',async()=>{
 const calls=[];const target=origin+'.audit.invalid/';
 await runAudit({config,state:{},browser:false,fetchImpl:async(url)=>{calls.push(url);return url===origin+'/llms.txt'?new Response(`# Site\n> Summary\n[link](${target})`,{headers:{'content-type':'text/plain'}}):new Response('',{status:503});}});
 assert(calls.includes(target));return {target,fetchWasMocked:true};
});
await test('presence denominator includes failed provider calls',async()=>{
 const r=await probePresence({queries:['fixture'],origin,env:{PERPLEXITY_API_KEY:'synthetic'},fetchImpl:async()=>new Response('{"error":"offline fixture"}',{status:503})});
 assert.equal(r.total,1);assert.equal(r.cited,0);assert(r.results[0].error);return r;
});
const root=path.resolve('local-fixture');await fs.mkdir(root,{recursive:true});
const git=(...args)=>execFileSync('git',['-C',root,...args],{encoding:'utf8'}).trim();
git('init','-q','-b','main');git('config','user.name','Audit fixture');git('config','user.email','fixture@example.invalid');git('config','commit.gpgsign','false');git('config','protocol.allow','never');
await fs.mkdir(path.join(root,'public'));await fs.writeFile(path.join(root,'public/robots.txt'),'User-agent: *\nAllow: /\n');await fs.writeFile(path.join(root,'public/llms.txt'),'# Baseline\n');git('add','.');git('commit','-qm','fixture baseline');
await test('crawler policy change passes validator and git application',async()=>{
 await fs.writeFile(path.join(root,'public/robots.txt'),'User-agent: *\nDisallow: /\n');
 const diff=execFileSync('git',['-C',root,'diff'],{encoding:'utf8'});git('restore','public/robots.txt');
 const v=validatePatch(diff,{allowlist:config.propose.allowlist});assert.equal(v.ok,true);
 execFileSync('git',['-C',root,'apply','--check','-'],{input:diff});execFileSync('git',['-C',root,'apply','-'],{input:diff});
 assert((await fs.readFile(path.join(root,'public/robots.txt'),'utf8')).includes('Disallow: /'));git('restore','public/robots.txt');return {validation:v,applied:true,fullBuildOrPublish:false};
});
await test('publisher commits then rejects its empty staged set',async()=>{
 const shim=path.resolve('shims');await fs.mkdir(shim,{recursive:true});const log=path.resolve('gh-calls.log');
 await fs.writeFile(path.join(shim,'gh'),`#!/usr/bin/env node\nimport fs from 'node:fs';fs.appendFileSync(${JSON.stringify(log)},JSON.stringify(process.argv.slice(2))+'\\n');if(process.argv[2]==='pr'&&process.argv[3]==='list'){console.log('[]')}else{process.exit(91)}\n`,{mode:0o755});
 await fs.writeFile(path.join(root,'public/llms.txt'),'# Updated\n');
 const appliedDiff=execFileSync('git',['-C',root,'diff'],{encoding:'utf8'});const before=git('rev-parse','HEAD');const oldPath=process.env.PATH;
 process.env.PATH=shim+':'+oldPath;let error;
 try{await publish({wt:root,branch:'seo/nightly-2026-11-01',title:'fixture proposal',body:'fixture',files:['public/llms.txt'],appliedDiff,reportRel:'docs/seo/nightly/2026-11-01.md',reportMd:'fixture',label:'seo-watch',runId:'fixture',env:{}});}catch(e){error=e.message;}finally{process.env.PATH=oldPath;}
 assert.match(error,/staged set  differs from/);assert.notEqual(git('rev-parse','HEAD'),before);assert.equal(git('diff','--cached','--name-only'),'');assert.equal(git('remote'),'');
 const committed=git('show','--format=','--name-only','HEAD').split('\n');assert.deepEqual(committed.sort(),['docs/seo/nightly/2026-11-01.md','public/llms.txt']);
 const calls=(await fs.readFile(log,'utf8')).trim().split('\n').map(JSON.parse);assert.equal(calls.length,1);assert.deepEqual(calls[0].slice(0,2),['pr','list']);
 return {error,committed,githubCalls:calls,noRemoteConfigured:true,worktrees:git('worktree','list')};
});
await fs.writeFile('/tmp/vf-dw-seo-artifacts/probes.json',JSON.stringify(results,null,2)+'\n');
console.log(JSON.stringify({passed:results.length,results},null,2));
