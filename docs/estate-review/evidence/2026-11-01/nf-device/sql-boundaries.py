"""Execute source-extracted registry SQL in local SQLite; no Worker or NIP-98 simulation."""
from pathlib import Path
import json,sqlite3,unittest
queries=json.loads(Path(__file__).with_name('device-sql.json').read_text())
def sql(prefix):return next(x['sql'] for x in queries if x['sql'].startswith(prefix))
class Registry(unittest.TestCase):
 def setUp(self):
  self.db=sqlite3.connect(':memory:');self.db.execute(sql('CREATE TABLE'))
 def register(self,device,owner):
  self.db.execute(sql('INSERT'),(device,owner,'synthetic fixture',100))
 def revoke(self,device,owner):
  return self.db.execute(sql('UPDATE'),(device,owner)).rowcount
 def row(self,device):
  return self.db.execute('SELECT owner_pubkey,revoked FROM device_keys WHERE device_pubkey=?',(device,)).fetchone()
 def principal(self,pk):
  return self.db.execute(sql('SELECT'),(pk,)).fetchone() is not None
 def test_owner_scoped_revoke_control(self):
  self.register('d','a');self.assertEqual(self.revoke('d','b'),0);self.assertEqual(self.row('d'),('a',0));self.assertEqual(self.revoke('d','a'),1)
 def test_register_after_revoke_reactivates_same_device(self):
  self.register('d','a');self.revoke('d','a');self.register('d','a');self.assertEqual(self.row('d'),('a',0))
 def test_reassignment_blocks_original_owner_revocation(self):
  self.register('d','a');self.register('d','b');self.assertEqual(self.row('d'),('b',0));self.assertEqual(self.revoke('d','a'),0)
 def test_principal_probe_and_insert_are_not_atomic(self):
  # Neither B nor C is an admin/whitelist principal in this fixture. A's
  # registration of B checks before B's concurrent registration of C commits.
  self.assertFalse(self.principal('b'));self.assertFalse(self.principal('c'))
  self.register('c','b');self.assertTrue(self.principal('b'))
  self.register('b','a');self.assertEqual(self.row('b'),('a',0));self.assertEqual(self.row('c'),('b',0))
if __name__=='__main__':unittest.main(verbosity=2)
