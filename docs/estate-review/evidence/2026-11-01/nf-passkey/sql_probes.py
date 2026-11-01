from pathlib import Path
import sqlite3,json,re,argparse,hashlib
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();here=Path(__file__).resolve().parent
src=(here/'source/crates/nostr-bbs-auth-worker/src/webauthn.rs').read_text();schema=(here/'source/crates/nostr-bbs-auth-worker/src/schema.rs').read_text();queries={}
def sql(name,text,prefix):
 start=text.index('"'+prefix)+1;end=text.index('"',start);raw=text[start:end];q=re.sub(r'\\\n\s*',' ',raw);queries[name]={'sql':q,'source':raw,'sha256':hashlib.sha256(raw.encode()).hexdigest()};return q
ct=sql('challenge_schema',schema,'CREATE TABLE IF NOT EXISTS challenges');wt=sql('credential_schema',schema,'CREATE TABLE IF NOT EXISTS webauthn_credentials')
ins=sql('insert_credential',src,'INSERT INTO webauthn_credentials (pubkey, credential_id');lookup=sql('lookup',src,'SELECT pubkey FROM webauthn_credentials WHERE credential_id')
read=sql('login_challenge',src,'SELECT 1 as ok FROM challenges \\\n             WHERE challenge = ?1 AND created_at > ?2 AND pubkey IN')
update=sql('counter_update',src,'UPDATE webauthn_credentials SET counter');delete=sql('challenge_delete',src,'DELETE FROM challenges WHERE challenge = ?1')
def db():
 c=sqlite3.connect(':memory:');c.execute(ct);c.execute(wt);return c
cases=[]
c=db();c.execute(ins,('owner','cred-A','public-A','salt-A',0,100));c.execute(ins,('owner','cred-B','public-B','salt-B',0,101));assert c.execute('SELECT count(*) FROM webauthn_credentials WHERE pubkey=?',('owner',)).fetchone()[0]==2;cases.append({'case':'bootstrap_schema_allows_duplicate_owner','rows':2})
known=c.execute(lookup,('cred-A',)).fetchone();missing=c.execute(lookup,('missing',)).fetchone();assert known==('owner',) and missing is None;cases.append({'case':'lookup_distinguishes_known_credential','known':known[0],'missing':None})
c=db();c.execute(ins,('owner','cred','public','salt',0,100));c.execute('INSERT INTO challenges VALUES (?,?,?)',('owner','challenge',1000));c.commit()
# Two already-validated requests read before either commits. Crypto and NIP-98 are assumed, not executed.
snapshots=[c.execute(read,('challenge',900,'owner','__discoverable__')).fetchone() for _ in range(2)];assert all(snapshots)
results=[]
for _ in range(2):
 with c:
  u=c.execute(update,(1,'owner')).rowcount;d=c.execute(delete,('challenge',)).rowcount
 results.append({'counter_rows':u,'deleted_challenge_rows':d})
assert results==[{'counter_rows':1,'deleted_challenge_rows':1},{'counter_rows':1,'deleted_challenge_rows':0}];cases.append({'case':'two_prechecked_requests_both_finish_unconditional_batches','results':results})
assert c.execute(read,('challenge',900,'owner','__discoverable__')).fetchone() is None;cases.append({'case':'serial_reuse_has_no_challenge','rejected_by_next_read':True})
args.out.write_text(json.dumps({'cases':cases,'queries':queries,'limits':['Actual extracted SQL on SQLite bootstrap tables; not Cloudflare D1 or HTTP.','Concurrent case is a chosen two-read schedule, not an observed production race. Both callers assumed independently valid NIP-98 tokens and assertions.','Additional deployed constraints and transaction scheduling not established.']},indent=2)+'\n');print('4 SQL scenarios passed')
