// Exact exported gates, synthetic bundle bytes, real tracked catalogue manifests.
// node probes.cjs /absolute/export/root
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const {spawnSync}=require('node:child_process');const assert=require('node:assert/strict');
const source=path.resolve(process.argv[2]);const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'vg-gates-'));
const results=[];
const write=(f,v)=>{fs.mkdirSync(path.dirname(f),{recursive:true});fs.writeFileSync(f,typeof v==='string'?v:JSON.stringify(v));};
const read=f=>JSON.parse(fs.readFileSync(path.join(source,f),'utf8'));
const supply='supply-chains-and-logistics';
function fixture(kind){
 const cwd=path.join(tmp,kind);const bundle=path.join(cwd,'bundle');
 const rel=kind==='space'?'planning/space-science-and-geo-sensing':'planning/supply-chains-and-logistics';
 const cat=read(rel+'/catalogue.json');write(path.join(cwd,rel,'catalogue.json'),cat);
 const research=kind==='space'?read(rel+'/researched-pages.json'):{batches:{}};
 if(kind==='space')write(path.join(cwd,rel,'researched-pages.json'),research);
 const researched=new Set(Object.values(research.batches).flat());
 const domains=[{id:6,slug:'space-science-and-systems',memberCount:545},{id:7,slug:'earth-observation-and-geospatial-sensing',memberCount:389},{id:8,slug:supply,memberCount:1}];
 for(const x of cat.pages)if(x.domain&&!domains.some(d=>d.slug===x.domain))domains.push({id:domains.length+10,slug:x.domain,memberCount:1});
 const titles=new Set([...cat.pages.map(x=>x.title),...researched,'Space Science and Systems','Earth Observation and Geospatial Sensing','Supply Chains and Logistics']);
 const search=[];let i=0;
 for(const title of titles){
  const id='synthetic-'+i++;search.push({id,title});
  const term=cat.pages.find(x=>x.title===title);
  write(path.join(bundle,'api/pages',id+'.json'),{title,public:true,maturity:'draft',domain:kind==='supply'?supply:term?.domain??'space-science-and-systems',definition:kind==='supply'||researched.has(title)?'Synthetic article.':''});
 }
 write(path.join(bundle,'api/search-index.json'),search);write(path.join(bundle,'data/graph/overview.json'),{domains});
 for(const d of domains)write(path.join(bundle,'data/graph/domain-'+d.slug+'.bin'),'');
 for(const slug of ['star-tracker','spacecraft','global-navigation-satellite-system','synthetic-aperture-radar','earth-observation-processing'])write(path.join(bundle,'api/pages',slug+'.json'),{title:'Synthetic representative',public:true,maturity:'draft',definition:''});
 return {cwd,bundle,rel,cat};
}
function run(name,kind,f,exit){
 const script=path.join(source,'publishing-tools/WasmVOWL/modern/scripts/check-'+(kind==='space'?'space-domains':'supply-chain-domain')+'.mjs');
 const r=spawnSync(process.execPath,[script,f.bundle],{cwd:f.cwd,encoding:'utf8'});assert.ifError(r.error);assert.equal(r.status,exit,name);
 results.push({probe:name,exit:r.status,stdout:r.stdout.trim(),stderr:r.stderr.split('\n').filter(x=>/AssertionError|ENOENT/.test(x)).join('\n')});
}
try{
 const a=fixture('space');run('space-real-catalogue-zero-byte-tiers-pass','space',a,0);
 const tier=path.join(a.bundle,'data/graph/domain-space-science-and-systems.bin');fs.unlinkSync(tier);run('space-missing-tier-rejected','space',a,1);write(tier,'invalid NGG1');
 run('space-invalid-tier-bytes-pass','space',a,0);
 const overview=path.join(a.bundle,'data/graph/overview.json');const ov=JSON.parse(fs.readFileSync(overview));ov.domains[0].id=99;write(overview,ov);run('space-wrong-stable-id-rejected','space',a,1);
 const b=fixture('supply');run('supply-real-catalogue-no-category-slug-ids-and-empty-tier-pass','supply',b,0);
 const ix=path.join(b.bundle,'api/search-index.json');let index=JSON.parse(fs.readFileSync(ix));
 const term=b.cat.pages.find(x=>x.action==='new'||x.action==='rehome');const hit=index.find(x=>x.title===term.title);fs.unlinkSync(path.join(b.bundle,'api/pages',hit.id+'.json'));run('supply-missing-accounted-page-rejected','supply',b,1);
 fs.unlinkSync(path.join(b.cwd,b.rel,'catalogue.json'));write(ix,[{id:'root-only',title:'Supply Chains and Logistics'}]);run('supply-missing-catalogue-root-only-pass','supply',b,0);
 const baseCwd=path.join(tmp,'bases');const valid='views:\n  - type: table\n    name: Control\n';write(path.join(baseCwd,'working/bases/control.base'),valid);
 for(const [name,content,expected] of [
  ['bases-normal-view-pass',valid,0],
  ['bases-named-invalid-type-rejected','views:\n  - type: invalid\n    name: Named\n',1],
  ['bases-blank-name-invalid-type-pass','views:\n  - type: invalid\n    name: ""\n',0],
  ['bases-undeclared-filter-formula-pass','filters:\n  and:\n    - formula.missing\n'+valid,0],
  ['bases-undeclared-order-formula-rejected','views:\n  - type: table\n    name: Control\n    order: [formula.missing]\n',1],
 ]){
  write(path.join(baseCwd,'knowledge/bases/probe.base'),content);
  const r=spawnSync('bash',[path.join(source,'scripts/check-bases.sh')],{cwd:baseCwd,encoding:'utf8'});assert.equal(r.status,expected,name);results.push({probe:name,exit:r.status,stdout:r.stdout.trim(),stderr:r.stderr.trim()});
 }
 const plugin=fs.readFileSync(path.join(source,'quartz/quartz/plugins/filters/explicit.ts'),'utf8');
 const body=plugin.match(/shouldPublish\(_ctx, \[_tree, vfile\]\) \{\n([\s\S]*?)\n    \}/)[1];
 const predicate=new Function('opts','vfile',body);
 const values=[true,'true',false,'false',1,0,null];
 const observed=values.map(value=>predicate({key:'public'},{data:{frontmatter:{public:value}}}));
 assert.deepEqual(observed,[true,true,false,false,false,false,false]);
 results.push({probe:'quartz-exact-publication-predicate',values,observed,note:'Exact extracted method body, not full Quartz render or upstream compiler inclusion'});
 console.log(JSON.stringify({scope:'isolated gate behaviour; not whole-workflow bypass; full tracked catalogues used for first six cases',passed:results.length,results},null,2));
}finally{fs.rmSync(tmp,{recursive:true,force:true});}
