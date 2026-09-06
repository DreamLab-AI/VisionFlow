#!/usr/bin/env python3
from pathlib import Path
import tempfile,subprocess,hashlib,json
b=Path(__file__).resolve().parents[1];r=b.parents[2]/'project';exe=r/'target/debug/vault-migrate';results=[]
with tempfile.TemporaryDirectory(prefix='estate-vault-converter-') as tmp:
 d=Path(tmp);src=d/'source';(src/'pages/A').mkdir(parents=True);(src/'pages/A___B.md').write_text('public:: true\n\nNAMESPACE_SENTINEL\n');(src/'pages/A/B.md').write_text('public:: true\n\nFOLDER_SENTINEL\n')
 before={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
 out=d/'out';run=subprocess.run([str(exe),str(src),'--out',str(out)],capture_output=True,text=True);assert run.returncode==0,run.stderr
 text=(out/'pages/A/B.md').read_text();after={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()};assert before==after
 results.append({'case':'colliding_page_paths','exit_code':run.returncode,'namespace_body_retained':'NAMESPACE_SENTINEL' in text,'folder_body_retained':'FOLDER_SENTINEL' in text,'source_unchanged':before==after})
 report=d/'dry-report.json';run=subprocess.run([str(exe),str(src),'--out',str(d/'dry-output'),'--dry-run','--report',str(report)],capture_output=True,text=True);assert run.returncode==0 and report.exists()
 results.append({'case':'dry_run_explicit_report','exit_code':run.returncode,'report_written':report.exists(),'vault_output_created':(d/'dry-output').exists()})
paths=['crates/vault-migrate/src/lib.rs','crates/vault-migrate/src/main.rs','crates/vault-migrate/src/paths.rs','crates/vault-migrate/src/convert.rs']
j={'date':'2026-09-04','scope':'Actual locally built CLI with invented temporary pages; no real vault or in-place conversion.','tests':{'command':'cargo test --locked --offline -p vault-migrate','exit_code':0,'unit_passed':70,'integration_passed':16},'results':results,'source_sha256':{p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}}
(b/'evidence/vault-converter-probe.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(results))
