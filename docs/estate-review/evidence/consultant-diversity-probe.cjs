'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const root=path.resolve(__dirname,'../../../../project/agentbox');
const d=require(path.join(root,'mcp/consultants/shared/model-diversity.js'));
const cases=[{name:'unknown_producer',result:d.verificationRecord({producerFamily:'unrecognised-fixture-model',verifier:'codex'})},{name:'same_family',result:d.verificationRecord({producerFamily:'openai',verifier:'codex'})},{name:'no_diverse_available',result:d.selectVerifier({producerFamily:'openai',candidates:['codex']})}];
assert.strictEqual(cases[0].result.anti_fox_ok,true);assert.strictEqual(cases[1].result.anti_fox_ok,false);assert.strictEqual(cases[2].result,null);
const files=['mcp/consultants/shared/model-diversity.js','mcp/consultants/shared/consultant-base.js'];
const result={date:'2026-09-04',scope:'Actual pure selection/record helpers only; no consultant invocation, network, provider cost or acceptance decision.',cases,source_sha256:Object.fromEntries(files.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,p))).digest('hex')]))};
fs.writeFileSync(__filename.replace('.cjs','.json'),JSON.stringify(result,null,2)+'\n');console.log('Three helper assertions pass');
