import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const base=path.resolve('entrypoint-fixtures');await fs.mkdir(base,{recursive:true});
const shim=path.join(base,'bin');await fs.mkdir(shim);
await fs.writeFile(path.join(shim,'git'),`#!/usr/bin/env node\nimport fs from 'node:fs';const a=process.argv.slice(2);fs.appendFileSync(process.env.FIXTURE_GIT,JSON.stringify(a)+'\\n');if(a.includes('fetch'))process.exit(0);if(a.includes('rev-parse')){console.log('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');process.exit(0)}if(a.includes('show')){console.log('# Fixture source');process.exit(0)}process.exit(92);\n`,{mode:0o755});
const results=[];
async function run(name,mode,date='2026-11-03',stateDir=null){
 const dir=stateDir??path.join(base,name);await fs.mkdir(dir,{recursive:true});
 const calls=path.join(dir,'calls.jsonl');const gitCalls=path.join(dir,'git.jsonl');
 const canary='AUDIT_SYNTHETIC_KEY_NOT_A_REAL_CREDENTIAL';
 const env={PATH:shim+':'+process.env.PATH,HOME:dir,LANG:'C.UTF-8',TZ:'UTC',SEO_WATCH_STATE_DIR:dir,ZAI_URL:'https://fixture.invalid',ZAI_API_KEY:canary,FIXTURE_MODE:mode,FIXTURE_CALLS:calls,FIXTURE_GIT:gitCalls};
 const p=spawnSync(process.execPath,['--import',path.resolve('mock-fetch.mjs'),'scripts/seo-watch/run.mjs','nightly','--no-browser','--no-propose','--date',date],{env,encoding:'utf8',timeout:30000});
 await fs.writeFile(path.join(dir,name+'.stderr'),p.stderr??'');assert.equal(p.status,0,p.stderr);
 const raw=await fs.readFile(path.join(dir,'ledger.jsonl'),'utf8');const last=JSON.parse(raw.trim().split('\n').at(-1));const report=await fs.readFile(path.join(dir,'reports',date+'.md'),'utf8');
 const requests=(await fs.readFile(calls,'utf8')).trim().split('\n').map(JSON.parse).filter(c=>c.url==='https://fixture.invalid/v1/messages');
 return {dir,canary,last,report,stderr:p.stderr,requests,raw};
}
const leak=await run('error','error');assert(leak.last.advise.error.includes(leak.canary));assert(leak.stderr.includes(leak.canary));assert(!leak.report.includes(leak.canary));assert(leak.report.includes('[REDACTED]'));
results.push({name:'provider error canary reaches raw ledger and stderr; report scrubs it',pass:true,ledgerContainsCanary:true,stderrContainsCanary:true,reportContainsCanary:false});
const retry=await run('retry','retry');assert.equal(retry.requests.length,2);assert.deepEqual(retry.last.advise.tokens,{input:7,output:9});
results.push({name:'empty reply retries but ledger counts only second response',pass:true,requests:2,providerFixtureTokens:349,ledgerTokens:16});
const first=await run('weekly-first','success','2026-11-02');const second=await run('weekly-second','success','2026-11-02',first.dir);assert.equal(second.requests.length,2);assert.equal(second.last.advise.reason,'weekly digest');
results.push({name:'same-day weekly rerun makes another request without force',pass:true,requests:2,lastReason:second.last.advise.reason});
const corruptDir=path.join(base,'corrupt');await fs.mkdir(corruptDir);await fs.writeFile(path.join(corruptDir,'ledger.jsonl'),JSON.stringify({date:'2026-11-03',advise:{tokens:{input:6000000,output:0}}})+'\n{torn row\n');
const corrupt=await run('corrupt','success','2026-11-03',corruptDir);assert.equal(corrupt.requests.length,1);assert.equal(corrupt.last.advise.reason,'first run');
results.push({name:'one malformed ledger line discards prior budget history',pass:true,priorTokens:6000000,requests:1,reason:corrupt.last.advise.reason});
await fs.writeFile('/tmp/vf-dw-seo-artifacts/entrypoint-probes.json',JSON.stringify(results,null,2)+'\n');console.log(JSON.stringify(results,null,2));
