// Run: node proposal-apply-probes.cjs /path/to/agentbox [ontomark executable]
// Production adapter and CLI; only temporary notes. Git and HTTP are intercepted.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { applyOntologyDecision } = require(path.join(path.resolve(process.argv[2]), 'management-api/lib/ontology-apply.js'));
const binary = process.argv[3] || 'ontomark';
const sha = s => crypto.createHash('sha256').update(s).digest('hex');
const iri = 'urn:ngm:class:a';
const vocab = 'version: 1\nnamespace: "urn:ngm:class:"\ntypes:\n  Class: { owl: "owl:Class", required: [resource, status] }\nrelations:\n  is-a: { owl: "rdfs:subClassOf" }\n';
const page = title => `---\ntype: Class\ntitle: ${title}\npublic: true\nresource: ${iri}\nstatus: draft\n---\nfirst body line\nsecond body line\n`;
function proposal(kind, diff) {
  const p = { level: 'content', kind, iri, page: 'A', hypothesis: 'audit fixture', diff,
    blockers: [], pages: ['A'], proposer: 'fixture', generation: 'fixture@old', stale_after: '2000-01-01T00:00:00Z' };
  p.digest = 'sha256:' + sha([p.level,p.iri,p.page,p.hypothesis,p.diff].join('\n') + '\n' + (kind === 'create' ? 'kind:create\n' : ''));
  return p;
}
function createDiff(text) { const lines = text.trimEnd().split('\n'); return `--- /dev/null\n+++ b/A\n@@ -0,0 +1,${lines.length} @@\n` + lines.map(l=>'+'+l+'\n').join(''); }
async function run(name, kind, change = () => {}, route = 'direct') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'vf-proposal-'));
  try {
    fs.mkdirSync(path.join(dir,'ontology')); fs.mkdirSync(path.join(dir,'knowledge'));
    fs.writeFileSync(path.join(dir,'ontology/vocabulary.yaml'),vocab);
    const calls=[], ledger=[], git=[];
    function cli(args) {
      const r=spawnSync(binary,['--repo',dir,...args],{encoding:'utf8'});
      if(r.error) throw r.error;
      calls.push({verb:args[0],exit:r.status,stdout:r.stdout,stderr:r.stderr});
      if(r.status!==0){const e=new Error(r.stderr||r.stdout);e.code=r.status;e.stdout=r.stdout;throw e;}
      return JSON.parse(r.stdout);
    }
    let p;
    if(kind==='amend') {
      const stage=path.join(dir,'seed.md'); fs.writeFileSync(stage,page('A'));
      cli(['create',stage,'--expect','docs=1','--json']); fs.unlinkSync(stage);
      const base=fs.readFileSync(path.join(dir,'knowledge/A.md'),'utf8');
      const index=base.split('\n').indexOf('first body line')+1;
      p=proposal(kind,`--- a/A\n+++ b/A\n@@ -${index},1 +${index},1 @@\n-first body line\n+approved first line\n`);
    } else p=proposal(kind,createDiff(page('A')));
    const decision={outcome:'promote',iri,signerNpub:'fixture',at:'2026-10-31T00:00:00Z',caseId:p.digest.slice(7),digest:p.digest,proposal:p};
    change({p,decision,dir});
    let result,error;
    try { const deps={repo:dir,tmpRoot:dir,
      runVault:async args=>cli(args),runGit:async args=>{git.push(args.slice(2));throw new Error('Git deliberately intercepted');},
      fetchFn:async (url,init)=>{ledger.push(JSON.parse(init.body));return {ok:true};}};
      if(route === 'direct') result=await applyOntologyDecision(decision,deps);
      else {
        const { LocalProcessManagerOrchestratorAdapter } = require(path.join(path.resolve(process.argv[2]),'management-api/adapters/orchestrator/local-process-manager.js'));
        const adapter=new LocalProcessManagerOrchestratorAdapter();
        const event={pubkey:'1'.repeat(64),created_at:1793404800,id:'2'.repeat(64)};
        const ctx={parsed:{iri},outcome:'promote',caseId:decision.caseId,dTag:decision.digest.slice(7),eTag:'3'.repeat(64),decidingPubkey:event.pubkey};
        adapter._ontologyDeps={...deps,fetchRequest:async()=>({tags:[['d',ctx.dTag],['panel','ontology-governance']],content:JSON.stringify(p)})};
        result=route==='approve' ? await adapter._ontologyApprove(event,ctx) : await adapter._applyOntologyBranch(event,ctx);
      }
    } catch(e){error={code:e.code,message:e.message};}
    const files=Object.fromEntries(fs.readdirSync(path.join(dir,'knowledge')).map(f=>[f,fs.readFileSync(path.join(dir,'knowledge',f),'utf8')]));
    return {name,decisionDigest:decision.digest,proposalDigest:p.digest,result,error,calls,ledger,git,files};
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
}
(async()=>{
 const results=[];
 results.push(await run('create baseline','create'));
 results.push(await run('create incorrect proposal digest','create',({p})=>{p.digest='sha256:'+'0'.repeat(64);}));
 results.push(await run('create title differs from proposal page','create',({p,decision})=>{Object.assign(p,proposal('create',createDiff(page('B'))));decision.digest=p.digest;decision.caseId=p.digest.slice(7);}));
 results.push(await run('amend baseline with expired metadata','amend'));
 results.push(await run('amend incorrect proposal digest','amend',({p})=>{p.digest='sha256:'+'0'.repeat(64);}));
 results.push(await run('amend decision digest differs from valid proposal','amend',({decision})=>{decision.digest='sha256:'+'f'.repeat(64);decision.caseId='f'.repeat(64);}));
 results.push(await run('amend unrelated body drift outside hunk','amend',({dir})=>{const f=path.join(dir,'knowledge/A.md');fs.writeFileSync(f,fs.readFileSync(f,'utf8').replace('second body line','other writer second line'));}));
 results.push(await run('amend drift inside hunk','amend',({dir})=>{const f=path.join(dir,'knowledge/A.md');fs.writeFileSync(f,fs.readFileSync(f,'utf8').replace('first body line','other writer first line'));}));
 results.push(await run('promote branch request tag matches decision but proposal digest differs','amend',({decision})=>{decision.digest='sha256:'+'f'.repeat(64);decision.caseId='f'.repeat(64);},'promote'));
 results.push(await run('approve branch refuses the same digest discrepancy','amend',({decision})=>{decision.digest='sha256:'+'f'.repeat(64);decision.caseId='f'.repeat(64);},'approve'));
 assert.equal(results[8].result?.amended,true);
 assert.equal(results[9].result?.refused,true);
 assert.equal(results[9].ledger.length,0);
 assert.equal(results[0].result?.created,true);
 assert.equal(results[3].result?.amended,true);
 assert.equal(results[4].error?.code,'DIGEST_MISMATCH');
 assert.equal(results[7].error?.code,'STALE');
 console.log(JSON.stringify({method:'Real production adapter and installed CLI, temporary minimal estate vaults. Git and attestation intercepted; no signature or relay-admission test.',version:spawnSync(binary,['--version'],{encoding:'utf8'}).stdout.trim(),results},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
