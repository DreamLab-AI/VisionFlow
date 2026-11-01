"""Run scoped publisher gates against synthetic artefacts; never deploy or fetch."""
import gzip,importlib.util,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
VG=Path(os.environ.get('VG_ROOT','/home/devuser/workspace/visionGraph'))
spec=importlib.util.spec_from_file_location('contract',VG/'scripts/tests/test_publish_cli.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
text=(VG/'.github/workflows/publish.yml').read_text();steps={s['name']:s for s in mod.workflow(text)[2]}
REQUIRED=['api/search-index.json','api/census.json','api/schema/context.jsonld','data/ontology.ttl','data/ontology.json','data/ontology-inferred.ttl','data/scaffold-index.json','data/prose-index.json','data/graph/overview.json','data/graph/full.bin','data/graph/stats.json','ns/v2.jsonld','context/v1.jsonld']
def put(root,path,value='fixture'):
 p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value);return p
class Gates(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.runner=self.root/'runner';self.runner.mkdir()
 def tearDown(self):self.tmp.cleanup()
 def runstep(self,name):
  return subprocess.run(['bash','-e','-c',steps[name]['run']],cwd=self.root,env={**os.environ,'RUNNER_TEMP':str(self.runner)},capture_output=True,text=True,timeout=30)
 def hard(self,n=1000,index=None):
  r=self.root/'site-data'
  for f in REQUIRED:put(r,f,'{}')
  entries=[{'id':str(i),'title':str(i)} for i in range(n)] if index is None else index
  put(r,'api/search-index.json',json.dumps(entries));put(r,'data/ontology.ttl','# fixture comment\n'*1000)
  for i in range(1000):put(r,f'api/pages/{i}.json','not JSON')
  put(r,'api/markdown/a.md','body');put(r,'data/graph/domain-fixture.bin','not NGG1')
  return self.runstep('Assert the machine artefacts exist (hard gate)')
 def layout(self):
  r=self.root/'www'
  for f in REQUIRED:put(r,f,'{}')
  put(r,'api/search-index.json',json.dumps([{'id':'one','title':'One'}]))
  put(r,'api/pages/one.json.gz','not gzip');put(r,'api/markdown/One.md.gz','not gzip')
  for p in ['data/graph/domain-fixture.bin','assets/a.js','notes/index.html','notes/404.html','notes/static/contentIndex.json','notes/pages/other.html','.nojekyll']:put(r,p)
  put(r,'index.html','<div id="root"></div>');put(r,'404.html',"sessionStorage.setItem('redirect'\n/notes/404.html")
  put(r,'explorer/index.html','url=/');put(r,'CNAME','narrativegoldmine.com\n');put(self.root,'content/pages/one.md','one')
  return r
 def urls(self,content='content'):
  return subprocess.run(['bash',str(VG/'quartz/scripts/check-urls.sh'),str(self.root/'www'),str(self.root/content)],capture_output=True,text=True,timeout=30)
 def secrets(self,r):
  return subprocess.run(['bash',str(VG/'quartz/scripts/check-secrets.sh'),str(r)],capture_output=True,text=True,timeout=30)
 def test_01_hard_gate_accepts_threshold_sized_but_invalid_bodies(self):self.assertEqual(self.hard().returncode,0)
 def test_02_hard_gate_rejects_999_entries(self):self.assertNotEqual(self.hard(n=999).returncode,0)
 def test_03_nonarray_search_passes_early_gate_but_later_layout_rejects(self):
  p=self.hard(index={});self.assertEqual(p.returncode,0);self.assertRegex(p.stderr,'integer.*expected')
  r=self.layout();put(r,'api/search-index.json','{}');self.assertNotEqual(self.urls().returncode,0)
 def test_04_layout_accepts_invalid_gzip_files(self):self.layout();self.assertEqual(self.urls().returncode,0)
 def test_05_layout_rejects_missing_page_json(self):
  r=self.layout();(r/'api/pages/one.json.gz').unlink();self.assertNotEqual(self.urls().returncode,0)
 def test_06_equal_count_notes_substitution_passes(self):
  r=self.layout();self.assertTrue((self.root/'content/pages/one.md').exists());self.assertFalse((r/'notes/pages/one.html').exists());self.assertEqual(self.urls().returncode,0)
 def test_07_absent_staging_directory_skips_completeness(self):
  self.layout();p=self.urls('absent');self.assertEqual(p.returncode,0);self.assertIn('SKIP',p.stdout)
 def seed_secret(self):
  r=self.root/'www';put(r,'api/pages/one.json','{}');put(r,'data/ontology.json','{}')
  # Deliberately synthetic, identical shape to the detector's own self-test.
  canary='sk-'+'abcdefghij0123456789ABCDEFGHIJ';put(r,'api/markdown/One.md',canary);return r,canary
 def test_08_raw_canary_is_rejected_without_value_in_log(self):
  r,c=self.seed_secret();p=self.secrets(r);self.assertEqual(p.returncode,1);self.assertNotIn(c,p.stdout+p.stderr)
 def test_09_actual_compression_step_hides_canary_from_final_scan(self):
  r,c=self.seed_secret();self.assertEqual(self.runstep('Compress the large machine artefacts at rest').returncode,0)
  self.assertEqual(gzip.decompress((r/'api/markdown/One.md.gz').read_bytes()).decode(),c)
  self.assertEqual(self.secrets(r).returncode,0)
 def test_10_copy_inventory_paths_no_longer_exist_after_compression(self):
  self.hard();self.assertEqual(self.runstep('Place the machine artefacts at the site root').returncode,0)
  self.assertEqual(self.runstep('Compress the large machine artefacts at rest').returncode,0)
  paths=(self.runner/'artefacts-expected.txt').read_text().splitlines();missing=[p for p in paths if not (self.root/'www'/p).is_file()]
  self.assertIn('data/ontology.json',missing);self.assertEqual(len(missing),1002)
 def test_11_size_gate_precedes_additional_site_outputs(self):
  r=self.layout();self.assertEqual(self.runstep('Finalize website').returncode,0)
  with (r/'assets/sparse-fixture.bin').open('wb') as f:f.truncate(1002*1024*1024)
  self.assertGreater(int(subprocess.check_output(['du','-sm','--apparent-size',str(r)],text=True).split()[0]),1000)
  names=list(steps);self.assertLess(names.index('Finalize website'),names.index('Build the explorer SPA'))
  self.assertNotEqual(self.runstep('Finalize website').returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
