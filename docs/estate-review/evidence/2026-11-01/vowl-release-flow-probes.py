"""Run extracted workflow shell with local cargo/gh/npm doubles; no service writes."""
from pathlib import Path
import os,subprocess,tempfile,json,sys,hashlib
repo=Path(sys.argv[1] if len(sys.argv)>1 else '/home/devuser/workspace/vowl-wasm')
source=(repo/'.github/workflows/release.yml').read_text()
def step(name):
    block=source.split('      - name: '+name+'\n',1)[1].split('\n      - ',1)[0]
    script=block.split('        run: |\n',1)[1]
    return '\n'.join(line[10:] if line.startswith('          ') else line for line in script.splitlines()).replace('${{ steps.pack.outputs.tarball }}','fixture.tgz')
stubs=r'''
cargo() { return 0; }
gh() {
 case "$1 $2" in
  'release view') if test -e "$AUDIT_STATE/asset"; then printf '{"assets":[{"name":"vowl-wasm-0.2.2-pkg.tgz"}]}'; else return 1; fi ;;
  'release create') return 0 ;;
  'release upload') touch "$AUDIT_STATE/asset" ;;
  *) return 99 ;;
 esac
}
npm() {
 case "$1" in
  view) test -e "$AUDIT_STATE/registry" ;;
  publish) touch "$AUDIT_STATE/publish-attempt"; if test "$AUDIT_FAIL_PUBLISH" = 1; then return 42; else touch "$AUDIT_STATE/registry"; fi ;;
  *) return 99 ;;
 esac
}
'''
records=[]
for scenario in ['publish_failure_then_retry','existing_registry_different_bytes']:
 with tempfile.TemporaryDirectory(prefix='vf-release-flow-') as tmp:
  root=Path(tmp);env=dict(os.environ,AUDIT_STATE=tmp,AUDIT_FAIL_PUBLISH='1' if scenario.startswith('publish_failure') else '0',VERSION='0.2.2',RELEASE_TAG='v0.2.2',GITHUB_REPOSITORY='audit/local-fixture')
  if scenario.startswith('existing'): (root/'registry').write_text('registry payload differs from newly attached asset')
  runs=[]
  names=['Verify reviewed third-party release inventory','Attach the bundle to the release','Publish to npm (OIDC trusted publishing)']
  if scenario.startswith('publish_failure'):names.append(names[0])
  for name in names:
   r=subprocess.run(['bash','-c',stubs+'\n'+step(name)],cwd=tmp,env=env,capture_output=True,text=True)
   runs.append(dict(step=name,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
  if scenario.startswith('publish_failure'):
   assert [r['exit_code'] for r in runs]==[0,0,42,1],runs
   assert (root/'asset').exists() and not (root/'registry').exists()
  else:
   assert all(r['exit_code']==0 for r in runs),runs
   assert not (root/'publish-attempt').exists()
  records.append(dict(scenario=scenario,runs=runs))
print(json.dumps({'workflow_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'control flow only; cargo, gh and npm are local shell doubles','scenarios':records},indent=2))
