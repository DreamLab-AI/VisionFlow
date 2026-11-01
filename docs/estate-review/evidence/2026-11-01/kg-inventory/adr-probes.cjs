// Execute the exported validator against synthetic decision packs.
// Usage: node adr-probes.cjs /absolute/path/to/exported/knowledgeGraph
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const assert = require('node:assert/strict');
const root = process.argv[2];
const script = path.join(root, 'scripts/adr-index-gen.js');
const base = fs.readFileSync(path.join(root, 'docs/adr/ADR-2001-corpus-consolidation.md'), 'utf8');
const results = [];
function invoke(dir) {
  const r = spawnSync(process.execPath, [script, dir, '--check'], {encoding:'utf8'});
  assert.ifError(r.error);
  return {exit:r.status, stdout:r.stdout.trim(), stderr:r.stderr.trim()};
}
const actual = invoke(path.join(root, 'docs/adr'));
assert.equal(actual.exit, 0); assert.match(actual.stdout, /4 ADR/);
results.push({probe:'committed-pack', ...actual});
function probe(name, docs, expected, check=()=>{}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kg-adr-probe-'));
  try {
    for (const [file,body] of Object.entries(docs)) fs.writeFileSync(path.join(dir,file),body);
    const r = invoke(dir); assert.equal(r.exit, expected, name);check(r,dir);
    results.push({probe:name,...r});
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
}
probe('invalid-enum-rejected', {'a.md':base.replace('decision_status: accepted','decision_status: imaginary')},1);
probe('missing-owner-rejected', {'a.md':base.replace(/^owner:.*\n/m,'')},1);
probe('duplicate-id-rejected', {'a.md':base,'b.md':base},1);
const header = base.slice(0,base.indexOf('\n---',4)+4)+'\n';
probe('empty-body-invalid-date-and-revision-accepted', {'a.md':header.replace('date: 2026-08-31','date: not-a-date').replace(/^verified_commit:.*$/m,'verified_commit: not-a-revision')},0);
probe('missing-supersession-target-warning-only', {'a.md':base.replace('supersedes: []','supersedes: [ADR-9999]')},0,r=>assert.match(r.stderr,/not present/));
const a = base.replace('supersedes: []','supersedes: [ADR-2002]').replace('superseded_by: []','superseded_by: [ADR-2002]');
const b = base.replaceAll('ADR-2001','ADR-2002').replace('supersedes: []','supersedes: [ADR-2001]').replace('superseded_by: []','superseded_by: [ADR-2001]');
probe('reciprocal-cycle-accepted-without-warning', {'a.md':a,'b.md':b},0,r=>assert.equal(r.stderr,''));
probe('stale-readme-accepted-unchanged', {'a.md':base,'README.md':'STALE INDEX\n'},0,(r,d)=>assert.equal(fs.readFileSync(path.join(d,'README.md'),'utf8'),'STALE INDEX\n'));
console.log(JSON.stringify({scope:'validator controls and limits, no product edits',passed:results.length,results},null,2));
