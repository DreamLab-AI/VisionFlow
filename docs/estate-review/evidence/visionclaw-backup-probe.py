#!/usr/bin/env python3
"""Run the real SQLite backup script against temporary, non-production data."""
import pathlib,tempfile,sqlite3,subprocess,os,json
w=pathlib.Path(__file__).resolve().parents[4]
with tempfile.TemporaryDirectory(prefix='estate-vc-backup-') as td:
 p=pathlib.Path(td);d=p/'data';d.mkdir()
 c=sqlite3.connect(d/'settings.sqlite3');c.execute('PRAGMA journal_mode=WAL');c.execute('CREATE TABLE evidence(value TEXT)');c.execute('INSERT INTO evidence VALUES (?)',('committed-in-wal',));c.commit()
 env=os.environ.copy();env.update(MODE='host',DATA_DIR=str(d),DBS='settings.sqlite3 missing.sqlite3',KEEP='14');env.pop('BACKUP_ROOT',None)
 r=subprocess.run(['bash',str(w/'project/scripts/backup-sqlite.sh')],cwd=p,env=env,capture_output=True,text=True)
 dest=pathlib.Path(r.stdout.strip())
 if not dest.is_absolute():dest=p/dest
 result={'exit_code':r.returncode,'requested_databases':2,'copied_databases':len(list(dest.glob('*.sqlite3'))),'default_destination_inside_source':dest.is_relative_to(d),'stderr':r.stderr.replace(str(p),'<temporary>')}
 if r.returncode==0:
  b=sqlite3.connect(dest/'settings.sqlite3');result['restored_value']=b.execute('SELECT value FROM evidence').fetchone()[0];result['integrity_check']=b.execute('PRAGMA integrity_check').fetchone()[0];b.close()
 c.close();print(json.dumps(result,indent=2))
