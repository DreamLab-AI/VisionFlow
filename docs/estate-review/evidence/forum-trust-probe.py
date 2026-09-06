#!/usr/bin/env python3
"""Actual source SELECT in isolated SQLite; synthetic demotions, not a D1 run."""
from pathlib import Path
import re, sqlite3, json, hashlib
base=Path(__file__).resolve().parents[1]
repo=base.parents[2]/'nostr-rust-forum'
src=repo/'crates/nostr-bbs-relay-worker/src/cron.rs'
s=src.read_text()
query=re.search(r'"(SELECT pubkey FROM whitelist .*?LIMIT \?4 OFFSET \?5)"',s,re.S).group(1)
query=re.sub(r'\\\s*\n\s*',' ',query)
batch=int(re.search(r'DEMOTION_BATCH_SIZE: u32 = (\d+)',s).group(1))
db=sqlite3.connect(':memory:')
db.execute('CREATE TABLE whitelist (pubkey TEXT, trust_level INTEGER, last_active_at INTEGER, is_admin INTEGER)')
db.executemany('INSERT INTO whitelist VALUES (?,1,?,0)',[(f'fixture-{i:04}',i) for i in range(batch*2)])
offset=0;scanned=0;pages=[]
while True:
 rows=db.execute(query,(1,2,batch*3,batch,offset)).fetchall()
 pages.append(len(rows))
 if not rows:break
 # Model qualifying TL1 -> TL0 updates; no Rust policy execution implied.
 db.executemany('UPDATE whitelist SET trust_level=0 WHERE pubkey=?',rows)
 scanned+=len(rows)
 if len(rows)<batch:break
 offset+=len(rows)
remaining=db.execute('SELECT COUNT(*) FROM whitelist WHERE trust_level=1').fetchone()[0]
result={'date':'2026-09-04','scope':'source-extracted SELECT, local in-memory SQLite; synthetic qualifying updates; no D1, real users or deployed scheduler','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'query':query,'batch_size':batch,'initial_eligible':batch*2,'page_sizes':pages,'scanned':scanned,'remaining_eligible':remaining}
assert scanned==batch and remaining==batch
(base/'evidence/forum-trust-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
